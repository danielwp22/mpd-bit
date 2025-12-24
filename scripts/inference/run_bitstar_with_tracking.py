"""
Run BIT* with interval tracking to monitor optimization progress over time.
Tracks metrics every 1 second and continues until beating the target or timeout.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# IMPORTANT: Import isaacgym FIRST before torch
import isaacgym

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.trajectory.metrics import compute_path_length

from bitstar_minimal_template import MinimalBITStarBaseline, compute_smoothness_finite_diff, compute_jerk_finite_diff


class BITStarWithTracking(MinimalBITStarBaseline):
    """BIT* with interval tracking for anytime optimization analysis."""

    def plan_with_tracking(self, start_state, goal_state, target_path_length=None,
                          tracking_interval=1.0, debug=False):
        """
        Plan with periodic metric tracking.

        Args:
            start_state: Starting configuration
            goal_state: Goal configuration
            target_path_length: Target path length to beat (stops early if achieved)
            tracking_interval: Interval in seconds for metric snapshots
            debug: Print debug information

        Returns:
            Dictionary with:
                - success: bool
                - sol_path: final solution path
                - planning_time: total planning time
                - path_length: final path length
                - smoothness: final smoothness
                - mean_jerk: final mean jerk
                - time_to_first_solution: time to first solution
                - first_solution_cost: first solution cost
                - time_to_target_quality: time to beat target (if provided)
                - interval_metrics: list of dicts with metrics at each interval
        """
        start_time = time.time()

        # Convert to numpy
        start_state = np.array(start_state, dtype=np.float64)
        goal_state = np.array(goal_state, dtype=np.float64)

        # Initialize BIT* structures
        self._initialize(start_state, goal_state)

        # Tracking variables
        iteration = 0
        num_batches = 0  # Track number of batches sampled
        best_cost = float('inf')
        time_to_first_solution = None
        time_to_target_quality = None
        first_solution_cost = None
        interval_metrics = []
        last_tracking_time = start_time

        if debug:
            print(f"\n{'='*80}")
            print(f"BIT* Anytime Optimization with Interval Tracking")
            print(f"{'='*80}")
            if target_path_length:
                print(f"Target path length to beat: {target_path_length:.3f}")
            print(f"Tracking interval: {tracking_interval:.1f} sec")
            print(f"Maximum planning time: {self.allowed_planning_time:.1f} sec")
            print(f"{'='*80}\n")

        # Main BIT* loop
        while time.time() - start_time < self.allowed_planning_time:
            iteration += 1
            current_time = time.time() - start_time

            # Sample new batch if needed
            if not self.edge_queue and not self.vertex_queue:
                num_batches += 1  # Increment batch counter
                if debug and iteration > 1:
                    print(f"[{current_time:6.2f}s] Iteration {iteration}: Sampling new batch #{num_batches} (current best: {best_cost:.3f})")
                self._sample_batch()
                self._update_edge_queue()

            # Expand best vertex or process best edge
            if self.vertex_queue and (not self.edge_queue or self._get_queue_value(self.vertex_queue) <= self._get_queue_value(self.edge_queue)):
                self._expand_vertex()
            elif self.edge_queue:
                success = self._process_edge()
                if success and self.goal_vertex is not None:
                    new_cost = self.goal_vertex.cost
                    if new_cost < best_cost:
                        # Record first solution time
                        if time_to_first_solution is None:
                            time_to_first_solution = time.time() - start_time
                            first_solution_cost = new_cost
                            if debug:
                                print(f"[{time_to_first_solution:6.2f}s] *** FIRST SOLUTION FOUND ***")
                                print(f"              Cost: {new_cost:.3f}")

                        best_cost = new_cost
                        if debug and time_to_first_solution is not None and current_time > time_to_first_solution + 0.1:
                            print(f"[{current_time:6.2f}s] Improved solution: {best_cost:.3f}")

                        # Check if we beat the target path length
                        if target_path_length is not None and time_to_target_quality is None:
                            if best_cost <= target_path_length:
                                time_to_target_quality = time.time() - start_time
                                if debug:
                                    print(f"[{time_to_target_quality:6.2f}s] *** TARGET QUALITY ACHIEVED ***")
                                    print(f"              Cost {best_cost:.3f} <= Target {target_path_length:.3f}")
                                    # Continue optimizing but record we beat it

            # Track metrics at intervals
            if time.time() - last_tracking_time >= tracking_interval:
                last_tracking_time = time.time()
                elapsed = time.time() - start_time

                # Extract current best solution and compute metrics
                if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
                    raw_path = self._extract_path()
                    sol_path = self._interpolate_path(raw_path, self.interpolate_num)

                    # Compute metrics
                    sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
                    path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
                    smoothness = compute_smoothness_finite_diff(sol_path_torch)[0].item()
                    mean_jerk = compute_jerk_finite_diff(sol_path_torch)[0].item()

                    metric_snapshot = {
                        'time': elapsed,
                        'iteration': iteration,
                        'path_length': path_length,
                        'smoothness': smoothness,
                        'mean_jerk': mean_jerk,
                        'num_vertices': len(self.vertices),
                        'num_samples': len(self.samples),
                        'num_batches': num_batches,
                        'has_solution': True,
                    }
                else:
                    metric_snapshot = {
                        'time': elapsed,
                        'iteration': iteration,
                        'path_length': float('inf'),
                        'smoothness': float('inf'),
                        'mean_jerk': float('inf'),
                        'num_vertices': len(self.vertices),
                        'num_samples': len(self.samples),
                        'num_batches': num_batches,
                        'has_solution': False,
                    }

                interval_metrics.append(metric_snapshot)

                if debug:
                    if metric_snapshot['has_solution']:
                        print(f"[{elapsed:6.2f}s] Tracking: length={metric_snapshot['path_length']:.3f}, "
                              f"smoothness={metric_snapshot['smoothness']:.3f}, "
                              f"vertices={metric_snapshot['num_vertices']}, "
                              f"batches={metric_snapshot['num_batches']}")

        planning_time = time.time() - start_time

        # Extract final solution
        if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
            raw_path = self._extract_path()
            sol_path = self._interpolate_path(raw_path, self.interpolate_num)

            # Compute final metrics
            sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
            smoothness = compute_smoothness_finite_diff(sol_path_torch)[0].item()
            mean_jerk = compute_jerk_finite_diff(sol_path_torch)[0].item()

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning SUCCESS")
                print(f"{'='*80}")
                print(f"  Planning time: {planning_time:.3f} sec")
                print(f"  Iterations: {iteration}")
                print(f"  Vertices in tree: {len(self.vertices)}")
                print(f"  Final path length: {path_length:.3f}")
                print(f"  Final smoothness: {smoothness:.3f}")
                print(f"  Final mean jerk: {mean_jerk:.3f}")
                if time_to_first_solution:
                    print(f"  Time to first solution: {time_to_first_solution:.3f} sec")
                    print(f"  First solution cost: {first_solution_cost:.3f}")
                if time_to_target_quality:
                    print(f"  Time to beat target: {time_to_target_quality:.3f} sec")
                print(f"  Tracked {len(interval_metrics)} metric snapshots")
                print(f"{'='*80}\n")

            return {
                'success': True,
                'sol_path': sol_path,
                'planning_time': planning_time,
                'path_length': path_length,
                'smoothness': smoothness,
                'mean_jerk': mean_jerk,
                'num_waypoints': len(sol_path),
                'time_to_first_solution': time_to_first_solution,
                'time_to_target_quality': time_to_target_quality,
                'first_solution_cost': first_solution_cost,
                'interval_metrics': interval_metrics,
                'iterations': iteration,
            }
        else:
            if debug:
                print(f"\n{'='*80}")
                print(f"Planning FAILED after {planning_time:.3f} sec")
                print(f"  Iterations: {iteration}")
                print(f"{'='*80}\n")

            return {
                'success': False,
                'sol_path': None,
                'planning_time': planning_time,
                'path_length': float('inf'),
                'smoothness': float('inf'),
                'mean_jerk': float('inf'),
                'num_waypoints': 0,
                'time_to_first_solution': None,
                'time_to_target_quality': None,
                'first_solution_cost': None,
                'interval_metrics': interval_metrics,
                'iterations': iteration,
            }


def run_bitstar_with_tracking(
    mpd_results_file="logs/2/results_single_plan-000.pt",
    allowed_time=120.0,  # Give it time to optimize
    tracking_interval=1.0,
    output_file="bitstar_tracked_result.pt",
    seed=42
):
    """
    Run BIT* with interval tracking on the same problem as MPD.

    Args:
        mpd_results_file: Path to MPD results
        allowed_time: Time limit for BIT* (default: 120 sec)
        tracking_interval: Tracking interval in seconds (default: 1.0 sec)
        output_file: Output file for results
        seed: Random seed
    """
    fix_random_seed(seed)

    print("\n" + "="*80)
    print("BIT* ANYTIME OPTIMIZATION WITH INTERVAL TRACKING")
    print("="*80 + "\n")

    # Load MPD results
    print("Loading MPD results...")
    if not os.path.exists(mpd_results_file):
        print(f"Error: {mpd_results_file} not found")
        print("Run the diffusion model first: python inference.py")
        return None

    mpd_results = torch.load(mpd_results_file, map_location='cpu')

    # Extract start, goal, and target metrics
    start_state = to_numpy(mpd_results['q_pos_start'])
    goal_state = to_numpy(mpd_results['q_pos_goal'])
    target_path_length = float(mpd_results['metrics']['trajs_best']['path_length'])
    mpd_time = mpd_results.get('t_inference_total', 0)

    print(f"MPD target path length: {target_path_length:.3f}")
    print(f"MPD inference time: {mpd_time:.3f} sec")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Setup environment and robot
    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Initialize BIT* planner with tracking
    planner = BITStarWithTracking(
        robot=robot,
        allowed_planning_time=allowed_time,
        interpolate_num=128,
        device="cuda:0",
        batch_size=100,
    )
    planner.set_obstacles(env)

    # Plan with tracking
    result = planner.plan_with_tracking(
        start_state,
        goal_state,
        target_path_length=target_path_length,
        tracking_interval=tracking_interval,
        debug=True
    )

    # Save results
    if result:
        result['mpd_path_length'] = target_path_length
        result['mpd_time'] = mpd_time
        result['start_state'] = start_state
        result['goal_state'] = goal_state
        result['tracking_interval'] = tracking_interval

        # Ensure output directory exists
        out_dir = os.path.dirname(output_file) or "."
        os.makedirs(out_dir, exist_ok=True)

        # Save obstacle geometry for visualization
        try:
            centers_list = []
            radii_list = []
            objs = []
            if hasattr(env, "obj_all_list"):
                objs = env.obj_all_list
            else:
                objs = getattr(env, "obj_fixed_list", []) + getattr(env, "obj_extra_list", [])
            for obj in objs:
                for field in getattr(obj, "fields", []):
                    if hasattr(field, "centers") and hasattr(field, "radii"):
                        centers_list.append(field.centers.detach().cpu().numpy())
                        radii_list.append(field.radii.detach().cpu().numpy())
            if centers_list and radii_list:
                obs_centers = np.concatenate(centers_list, axis=0)
                obs_radii = np.concatenate(radii_list, axis=0)
                np.savez(os.path.join(out_dir, "obstacles.npz"),
                         centers=obs_centers, radii=obs_radii)
                print(f"Saved obstacles to {os.path.join(out_dir, 'obstacles.npz')}")
            else:
                print("Warning: no obstacle spheres found to save.")
        except Exception as e:
            print(f"Warning: could not save obstacles: {e}")

        torch.save(result, output_file)
        print(f"\nResults saved to: {output_file}")

        # Print summary
        print_summary(result, target_path_length, mpd_time)

    planner.terminate()
    return result


def print_summary(result, target_path_length, mpd_time):
    """Print a summary of the results."""
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)

    if result['success']:
        print(f"\n{'Metric':<40} {'Value':<20}")
        print("-"*80)
        print(f"{'Total planning time':<40} {result['planning_time']:.3f} sec")
        print(f"{'Total iterations':<40} {result['iterations']}")
        print(f"{'Metric snapshots tracked':<40} {len(result['interval_metrics'])}")

        if result['time_to_first_solution']:
            print(f"\n{'First Solution':<40}")
            print("-"*80)
            print(f"{'Time to first solution':<40} {result['time_to_first_solution']:.3f} sec")
            print(f"{'First solution cost':<40} {result['first_solution_cost']:.3f}")

        print(f"\n{'Final Solution':<40}")
        print("-"*80)
        print(f"{'Final path length':<40} {result['path_length']:.3f}")
        print(f"{'Final smoothness':<40} {result['smoothness']:.3f}")
        print(f"{'Final mean jerk':<40} {result['mean_jerk']:.3f}")

        print(f"\n{'Comparison with MPD Target':<40}")
        print("-"*80)
        print(f"{'MPD path length':<40} {target_path_length:.3f}")
        print(f"{'BIT* path length':<40} {result['path_length']:.3f}")

        if result['path_length'] <= target_path_length:
            improvement = (target_path_length - result['path_length']) / target_path_length * 100
            print(f"{'Improvement':<40} {improvement:.1f}% shorter")
            if result['time_to_target_quality']:
                print(f"{'Time to beat MPD':<40} {result['time_to_target_quality']:.3f} sec")
                speedup = mpd_time / result['time_to_target_quality']
                print(f"{'Speedup vs MPD':<40} {speedup:.2f}x")
        else:
            diff = (result['path_length'] - target_path_length) / target_path_length * 100
            print(f"{'Difference':<40} {diff:.1f}% longer")
            print(f"{'Status':<40} Did not beat MPD")

        # Improvement from first to final
        if result['first_solution_cost'] and result['first_solution_cost'] > 0:
            improvement = (result['first_solution_cost'] - result['path_length']) / result['first_solution_cost'] * 100
            print(f"\n{'Anytime Optimization':<40}")
            print("-"*80)
            print(f"{'Path improvement (first → final)':<40} {improvement:.1f}%")
    else:
        print("\nPlanning FAILED - no solution found")

    print("="*80 + "\n")


def plot_metrics_over_time(result, output_plot="bitstar_optimization_plot.png"):
    """Plot metrics over time (optional, requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        if not result['success'] or not result['interval_metrics']:
            print("No metrics to plot")
            return

        metrics = result['interval_metrics']

        # Extract data
        times = [m['time'] for m in metrics if m['has_solution']]
        path_lengths = [m['path_length'] for m in metrics if m['has_solution']]
        smoothness = [m['smoothness'] for m in metrics if m['has_solution']]
        mean_jerks = [m['mean_jerk'] for m in metrics if m['has_solution']]

        if not times:
            print("No valid solutions to plot")
            return

        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        # Plot path length
        axes[0].plot(times, path_lengths, 'b-', linewidth=2, label='BIT* Path Length')
        if 'mpd_path_length' in result:
            axes[0].axhline(result['mpd_path_length'], color='r', linestyle='--', linewidth=2, label='MPD Target')
        axes[0].set_xlabel('Time (sec)')
        axes[0].set_ylabel('Path Length')
        axes[0].set_title('Path Length Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot smoothness
        axes[1].plot(times, smoothness, 'g-', linewidth=2)
        axes[1].set_xlabel('Time (sec)')
        axes[1].set_ylabel('Smoothness')
        axes[1].set_title('Smoothness Over Time')
        axes[1].grid(True, alpha=0.3)

        # Plot mean jerk
        axes[2].plot(times, mean_jerks, 'm-', linewidth=2)
        axes[2].set_xlabel('Time (sec)')
        axes[2].set_ylabel('Mean Jerk')
        axes[2].set_title('Mean Jerk Over Time')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_plot}")
        plt.close()

    except ImportError:
        print("matplotlib not available - skipping plot generation")
    except Exception as e:
        print(f"Error plotting: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BIT* with interval tracking")
    parser.add_argument("--mpd-results", default="logs/2/results_single_plan-000.pt",
                       help="Path to MPD results file")
    parser.add_argument("--time", type=float, default=120.0,
                       help="Time limit for BIT* (seconds)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Tracking interval (seconds)")
    parser.add_argument("--output", default="bitstar_tracked_result.pt",
                       help="Output file for results")
    parser.add_argument("--plot", default="bitstar_optimization_plot.png",
                       help="Output plot file (requires matplotlib)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    result = run_bitstar_with_tracking(
        mpd_results_file=args.mpd_results,
        allowed_time=args.time,
        tracking_interval=args.interval,
        output_file=args.output,
        seed=args.seed
    )

    # Try to generate plot
    if result:
        plot_metrics_over_time(result, output_plot=args.plot)

    print("\nUsage examples:")
    print("  python run_bitstar_with_tracking.py")
    print("  python run_bitstar_with_tracking.py --time 180 --interval 0.5")
    print("  python run_bitstar_with_tracking.py --mpd-results logs/2/results_single_plan-000.pt --output my_result.pt")
