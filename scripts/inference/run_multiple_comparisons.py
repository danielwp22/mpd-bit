"""
Run inference (MPD) and BIT* on multiple start/goal pairs for comprehensive comparison.
Collects all metrics and interval tracking data from multiple runs.

This script:
1. Generates N random start/goal pairs
2. Saves them to a problem set file
3. Runs BIT* on each problem with interval tracking
4. Optionally allows you to run inference separately using the saved problems
5. Aggregates all results with statistics
"""
import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path

# IMPORTANT: Import isaacgym FIRST before torch
import isaacgym

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed

from run_bitstar_with_tracking import BITStarWithTracking


def generate_problem_set(robot, env, n_problems, seed=42, output_dir="multi_run_results"):
    """
    Generate random collision-free start/goal pairs and save them.

    Args:
        robot: Robot instance
        env: Environment instance
        n_problems: Number of problems to generate
        seed: Random seed
        output_dir: Output directory

    Returns:
        List of problem dictionaries
    """
    fix_random_seed(seed)

    print(f"\n{'='*80}")
    print(f"Generating {n_problems} random start/goal pairs")
    print(f"{'='*80}\n")

    problems = []

    for i in range(n_problems):
        print(f"Problem {i+1}/{n_problems}...", end=" ")

        # Generate random start and goal
        start_state = robot.random_q(n_samples=1)[0]
        goal_state = robot.random_q(n_samples=1)[0]

        # Convert to numpy
        start_state_np = to_numpy(start_state)
        goal_state_np = to_numpy(goal_state)

        problem = {
            'problem_idx': i,
            'start_state': start_state_np,
            'goal_state': goal_state_np,
        }

        problems.append(problem)
        print("✓")

    # Save problems
    os.makedirs(output_dir, exist_ok=True)
    problem_file = os.path.join(output_dir, "problem_set.pt")
    torch.save(problems, problem_file)

    print(f"\nGenerated {len(problems)} problems")
    print(f"Saved to: {problem_file}\n")

    # Also save as text for reference
    txt_file = os.path.join(output_dir, "problem_set.txt")
    with open(txt_file, 'w') as f:
        f.write(f"Problem Set ({n_problems} problems)\n")
        f.write(f"Seed: {seed}\n")
        f.write("="*80 + "\n\n")
        for p in problems:
            f.write(f"Problem {p['problem_idx']}:\n")
            f.write(f"  Start: {p['start_state']}\n")
            f.write(f"  Goal:  {p['goal_state']}\n\n")

    print(f"Problem descriptions saved to: {txt_file}\n")

    return problems


def run_bitstar_on_problems(
    problems,
    env,
    robot,
    allowed_time=120.0,
    tracking_interval=1.0,
    output_dir="multi_run_results",
    tensor_args=None,
    target_lengths=None,
):
    """
    Run BIT* on all problems.

    Args:
        problems: List of problem dictionaries
        env: Environment
        robot: Robot
        allowed_time: Planning time limit per problem
        tracking_interval: Tracking interval
        output_dir: Output directory
        tensor_args: Tensor arguments
        target_lengths: Optional list of target path lengths (from MPD)

    Returns:
        List of BIT* results
    """
    print(f"\n{'='*80}")
    print(f"Running BIT* on {len(problems)} problems")
    print(f"{'='*80}\n")

    results = []

    for i, problem in enumerate(problems):
        print(f"\n{'#'*80}")
        print(f"# BIT* on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem['start_state']
        goal_state = problem['goal_state']

        # Get target length if available
        target_length = None
        if target_lengths and i < len(target_lengths):
            target_length = target_lengths[i]

        try:
            # Initialize BIT* planner
            planner = BITStarWithTracking(
                robot=robot,
                allowed_planning_time=allowed_time,
                interpolate_num=128,
                device="cuda:0",  # Pass as string, not tensor_args device
                batch_size=100,
            )
            planner.set_obstacles(env)

            # Plan with tracking
            result = planner.plan_with_tracking(
                start_state,
                goal_state,
                target_path_length=target_length,
                tracking_interval=tracking_interval,
                debug=True
            )

            # Add problem info
            result['problem_idx'] = i
            result['start_state'] = start_state
            result['goal_state'] = goal_state
            if target_length:
                result['target_path_length'] = target_length

            # Save individual result
            result_file = os.path.join(output_dir, f"bitstar_result_{i:03d}.pt")
            torch.save(result, result_file)
            print(f"Saved to: {result_file}")

            planner.terminate()
            results.append(result)

        except Exception as e:
            print(f"ERROR running BIT* on problem {i}: {e}")
            import traceback
            traceback.print_exc()

            result = {
                'success': False,
                'problem_idx': i,
                'start_state': start_state,
                'goal_state': goal_state,
                'error': str(e),
            }
            results.append(result)

    return results


def load_mpd_results(output_dir="multi_run_results", n_problems=None):
    """
    Load MPD inference results if they exist.

    Args:
        output_dir: Directory containing results
        n_problems: Number of problems (optional, will detect from files)

    Returns:
        List of MPD results or None
    """
    # First check for MPD results from run_inference_on_problems.py
    mpd_dir = os.path.join(output_dir, "mpd_results")

    if os.path.exists(mpd_dir):
        result_files = sorted([f for f in os.listdir(mpd_dir) if f.startswith("mpd_result_")])

        if result_files:
            print(f"\nLoading {len(result_files)} MPD results from {mpd_dir}...")

            mpd_results = []
            for filename in result_files:
                filepath = os.path.join(mpd_dir, filename)
                try:
                    result = torch.load(filepath, map_location='cpu')
                    mpd_results.append(result)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

            print(f"Loaded {len(mpd_results)} MPD results\n")
            return mpd_results if mpd_results else None

    # Fallback: Check for old-style inference results in logs directory
    logs_dir = os.path.join(output_dir, "logs")

    if not os.path.exists(logs_dir):
        print(f"No MPD results found in {output_dir}/mpd_results or {logs_dir}")
        return None

    # Look for result files
    result_files = sorted([f for f in os.listdir(logs_dir) if f.startswith("results_single_plan-")])

    if not result_files:
        print(f"No MPD result files found")
        return None

    print(f"\nLoading {len(result_files)} MPD results from {logs_dir}...")

    mpd_results = []
    for i, filename in enumerate(result_files):
        filepath = os.path.join(logs_dir, filename)
        try:
            result_data = torch.load(filepath, map_location='cpu')

            # Extract metrics
            metrics = result_data.get('metrics', {})
            trajs_best = metrics.get('trajs_best', {})

            result = {
                'success': True,
                'problem_idx': i,
                'path_length': float(trajs_best.get('path_length', float('inf'))),
                'smoothness': float(trajs_best.get('smoothness', float('inf'))),
                'inference_time': float(result_data.get('t_inference_total', 0)),
                'start_state': to_numpy(result_data.get('q_pos_start')),
                'goal_state': to_numpy(result_data.get('q_pos_goal')),
            }
            mpd_results.append(result)

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(mpd_results)} MPD results\n")
    return mpd_results if mpd_results else None


def aggregate_results(bitstar_results, mpd_results=None, output_dir="multi_run_results"):
    """
    Aggregate and compute statistics from all results.

    Args:
        bitstar_results: List of BIT* results
        mpd_results: Optional list of MPD results
        output_dir: Output directory

    Returns:
        Dictionary with aggregated statistics
    """
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}\n")

    # Filter successful runs
    bitstar_success = [r for r in bitstar_results if r.get('success', False)]

    # Compute BIT* statistics
    bitstar_stats = {}
    if bitstar_success:
        path_lengths = np.array([r['path_length'] for r in bitstar_success])
        times = np.array([r['planning_time'] for r in bitstar_success])
        smoothness = np.array([r['smoothness'] for r in bitstar_success])
        mean_jerk = np.array([r['mean_jerk'] for r in bitstar_success])
        first_times = np.array([r['time_to_first_solution'] for r in bitstar_success if r.get('time_to_first_solution')])
        first_costs = np.array([r['first_solution_cost'] for r in bitstar_success if r.get('first_solution_cost')])

        bitstar_stats = {
            'n_problems': len(bitstar_results),
            'n_success': len(bitstar_success),
            'success_rate': len(bitstar_success) / len(bitstar_results) if bitstar_results else 0,
            'path_length_mean': float(np.mean(path_lengths)),
            'path_length_std': float(np.std(path_lengths)),
            'path_length_min': float(np.min(path_lengths)),
            'path_length_max': float(np.max(path_lengths)),
            'planning_time_mean': float(np.mean(times)),
            'planning_time_std': float(np.std(times)),
            'smoothness_mean': float(np.mean(smoothness)),
            'smoothness_std': float(np.std(smoothness)),
            'mean_jerk_mean': float(np.mean(mean_jerk)),
            'mean_jerk_std': float(np.std(mean_jerk)),
        }

        if len(first_times) > 0:
            bitstar_stats['time_to_first_solution_mean'] = float(np.mean(first_times))
            bitstar_stats['time_to_first_solution_std'] = float(np.std(first_times))
            bitstar_stats['first_solution_cost_mean'] = float(np.mean(first_costs))
            bitstar_stats['first_solution_cost_std'] = float(np.std(first_costs))

        # Check how many beat the target (if MPD results available)
        if mpd_results:
            beat_target = []
            target_times = []

            for br in bitstar_success:
                idx = br['problem_idx']
                if idx < len(mpd_results) and mpd_results[idx].get('success'):
                    target = mpd_results[idx]['path_length']
                    if br.get('time_to_target_quality') is not None:
                        beat_target.append(br)
                        target_times.append(br['time_to_target_quality'])

            if beat_target:
                bitstar_stats['n_beat_target'] = len(beat_target)
                bitstar_stats['beat_target_rate'] = len(beat_target) / len(bitstar_success)
                bitstar_stats['time_to_beat_target_mean'] = float(np.mean(target_times))
                bitstar_stats['time_to_beat_target_std'] = float(np.std(target_times))

    # Compute MPD statistics if available
    mpd_stats = {}
    if mpd_results:
        mpd_success = [r for r in mpd_results if r.get('success', False)]

        if mpd_success:
            mpd_path_lengths = np.array([r['path_length'] for r in mpd_success])
            mpd_times = np.array([r['inference_time'] for r in mpd_success])
            mpd_smoothness = np.array([r['smoothness'] for r in mpd_success])

            mpd_stats = {
                'n_problems': len(mpd_results),
                'n_success': len(mpd_success),
                'success_rate': len(mpd_success) / len(mpd_results) if mpd_results else 0,
                'path_length_mean': float(np.mean(mpd_path_lengths)),
                'path_length_std': float(np.std(mpd_path_lengths)),
                'path_length_min': float(np.min(mpd_path_lengths)),
                'path_length_max': float(np.max(mpd_path_lengths)),
                'inference_time_mean': float(np.mean(mpd_times)),
                'inference_time_std': float(np.std(mpd_times)),
                'smoothness_mean': float(np.mean(mpd_smoothness)),
                'smoothness_std': float(np.std(mpd_smoothness)),
            }

    # Save aggregated results
    aggregated = {
        'bitstar_results': bitstar_results,
        'mpd_results': mpd_results,
        'bitstar_stats': bitstar_stats,
        'mpd_stats': mpd_stats,
    }

    result_file = os.path.join(output_dir, "aggregated_results.pt")
    torch.save(aggregated, result_file)
    print(f"Aggregated results saved to: {result_file}")

    # Save statistics as YAML
    stats_dict = {
        'bitstar': bitstar_stats,
    }
    if mpd_stats:
        stats_dict['mpd'] = mpd_stats

    yaml_file = os.path.join(output_dir, "statistics.yaml")
    with open(yaml_file, 'w') as f:
        yaml.dump(stats_dict, f, default_flow_style=False)
    print(f"Statistics saved to: {yaml_file}\n")

    return aggregated


def print_summary(aggregated):
    """Print summary of results."""

    bitstar_stats = aggregated.get('bitstar_stats', {})
    mpd_stats = aggregated.get('mpd_stats', {})

    if not bitstar_stats:
        print("\nNo successful BIT* results to display.")
        return

    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}\n")

    if mpd_stats:
        print(f"{'Metric':<45} {'MPD':<20} {'BIT*':<20}")
        print("-"*85)

        # Success rate
        mpd_success_str = f"{mpd_stats['n_success']}/{mpd_stats['n_problems']} ({mpd_stats['success_rate']*100:.1f}%)"
        bitstar_success_str = f"{bitstar_stats['n_success']}/{bitstar_stats['n_problems']} ({bitstar_stats['success_rate']*100:.1f}%)"
        print(f"{'Success rate':<45} {mpd_success_str:<20} {bitstar_success_str:<20}")

        # Path length
        mpd_path = f"{mpd_stats['path_length_mean']:.3f} ± {mpd_stats['path_length_std']:.3f}"
        bitstar_path = f"{bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}"
        print(f"{'Path length (mean ± std)':<45} {mpd_path:<20} {bitstar_path:<20}")

        # Time
        mpd_time = f"{mpd_stats['inference_time_mean']:.3f} ± {mpd_stats['inference_time_std']:.3f}"
        bitstar_time = f"{bitstar_stats['planning_time_mean']:.3f} ± {bitstar_stats['planning_time_std']:.3f}"
        print(f"{'Planning time (sec, mean ± std)':<45} {mpd_time:<20} {bitstar_time:<20}")

        # Smoothness
        mpd_smooth = f"{mpd_stats['smoothness_mean']:.3f} ± {mpd_stats['smoothness_std']:.3f}"
        bitstar_smooth = f"{bitstar_stats['smoothness_mean']:.3f} ± {bitstar_stats['smoothness_std']:.3f}"
        print(f"{'Smoothness (mean ± std)':<45} {mpd_smooth:<20} {bitstar_smooth:<20}")

    else:
        print(f"{'Metric':<45} {'BIT*':<20}")
        print("-"*65)

        # Success rate
        bitstar_success_str = f"{bitstar_stats['n_success']}/{bitstar_stats['n_problems']} ({bitstar_stats['success_rate']*100:.1f}%)"
        print(f"{'Success rate':<45} {bitstar_success_str:<20}")

        # Path length
        bitstar_path = f"{bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}"
        print(f"{'Path length (mean ± std)':<45} {bitstar_path:<20}")

        # Time
        bitstar_time = f"{bitstar_stats['planning_time_mean']:.3f} ± {bitstar_stats['planning_time_std']:.3f}"
        print(f"{'Planning time (sec, mean ± std)':<45} {bitstar_time:<20}")

        # Smoothness
        bitstar_smooth = f"{bitstar_stats['smoothness_mean']:.3f} ± {bitstar_stats['smoothness_std']:.3f}"
        print(f"{'Smoothness (mean ± std)':<45} {bitstar_smooth:<20}")

    # Mean jerk
    bitstar_jerk = f"{bitstar_stats['mean_jerk_mean']:.4f} ± {bitstar_stats['mean_jerk_std']:.4f}"
    if mpd_stats:
        print(f"{'Mean jerk (mean ± std)':<45} {'N/A':<20} {bitstar_jerk:<20}")
    else:
        print(f"{'Mean jerk (mean ± std)':<45} {bitstar_jerk:<20}")

    # BIT* anytime performance
    if 'time_to_first_solution_mean' in bitstar_stats:
        print(f"\n{'BIT* Anytime Performance':<45}")
        print("-"*65)
        first_time = f"{bitstar_stats['time_to_first_solution_mean']:.3f} ± {bitstar_stats['time_to_first_solution_std']:.3f}"
        first_cost = f"{bitstar_stats['first_solution_cost_mean']:.3f} ± {bitstar_stats['first_solution_cost_std']:.3f}"
        print(f"{'Time to first solution (sec)':<45} {first_time:<20}")
        print(f"{'First solution cost':<45} {first_cost:<20}")

        if 'n_beat_target' in bitstar_stats:
            beat_str = f"{bitstar_stats['n_beat_target']}/{bitstar_stats['n_success']} ({bitstar_stats['beat_target_rate']*100:.1f}%)"
            target_time = f"{bitstar_stats['time_to_beat_target_mean']:.3f} ± {bitstar_stats['time_to_beat_target_std']:.3f}"
            print(f"{'Problems that beat MPD target':<45} {beat_str:<20}")
            print(f"{'Time to beat target (sec)':<45} {target_time:<20}")

    print("="*80 + "\n")


def main(
    n_problems=10,
    bitstar_time=600.0,
    tracking_interval=1.0,
    output_dir="multi_run_results",
    seed=42,
    load_mpd=False,
):
    """
    Main function to run multiple comparisons.

    Args:
        n_problems: Number of problems to generate/solve
        bitstar_time: Time limit for each BIT* run
        tracking_interval: Tracking interval
        output_dir: Output directory
        seed: Random seed
        load_mpd: Whether to try loading MPD results
    """
    print("\n" + "="*80)
    print("MULTIPLE PROBLEM COMPARISON")
    print("="*80)
    print(f"Number of problems: {n_problems}")
    print(f"BIT* time limit: {bitstar_time} sec")
    print(f"Tracking interval: {tracking_interval} sec")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print("="*80)

    # Setup environment and robot
    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Check if problems already exist
    problem_file = os.path.join(output_dir, "problem_set.pt")
    if os.path.exists(problem_file):
        print(f"\nLoading existing problems from: {problem_file}")
        problems = torch.load(problem_file)
        print(f"Loaded {len(problems)} problems\n")
    else:
        # Generate new problem set
        problems = generate_problem_set(robot, env, n_problems, seed=seed, output_dir=output_dir)

    # Try to load MPD results if requested
    mpd_results = None
    target_lengths = None
    if load_mpd:
        mpd_results = load_mpd_results(output_dir, n_problems)
        if mpd_results:
            target_lengths = [r['path_length'] for r in mpd_results]

    # Run BIT* on all problems
    bitstar_results = run_bitstar_on_problems(
        problems,
        env,
        robot,
        allowed_time=bitstar_time,
        tracking_interval=tracking_interval,
        output_dir=output_dir,
        tensor_args=tensor_args,
        target_lengths=target_lengths,
    )

    # Aggregate results
    aggregated = aggregate_results(bitstar_results, mpd_results, output_dir)

    # Print summary
    print_summary(aggregated)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Problem set: {problem_file}")
    print(f"BIT* results: {output_dir}/bitstar_result_*.pt")
    print(f"Aggregated results: {output_dir}/aggregated_results.pt")
    print(f"Statistics: {output_dir}/statistics.yaml")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run BIT* on multiple problems with interval tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 problems and run BIT* on each
  python run_multiple_comparisons.py --n-problems 10

  # Use longer time limit per problem
  python run_multiple_comparisons.py --n-problems 5 --bitstar-time 300

  # Load existing MPD results and use as targets
  python run_multiple_comparisons.py --n-problems 10 --load-mpd

  # Continue from existing problem set (will reuse problem_set.pt if it exists)
  python run_multiple_comparisons.py --n-problems 10 --output-dir my_results
        """
    )

    parser.add_argument("--n-problems", type=int, default=10,
                       help="Number of problems to solve (default: 10)")
    parser.add_argument("--bitstar-time", type=float, default=600.0,
                       help="Time limit for each BIT* run in seconds (default: 600)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Tracking interval in seconds (default: 1.0)")
    parser.add_argument("--output-dir", default="multi_run_results",
                       help="Output directory for results (default: multi_run_results)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--load-mpd", action="store_true",
                       help="Try to load existing MPD results from output_dir/logs")

    args = parser.parse_args()

    main(
        n_problems=args.n_problems,
        bitstar_time=args.bitstar_time,
        tracking_interval=args.interval,
        output_dir=args.output_dir,
        seed=args.seed,
        load_mpd=args.load_mpd,
    )
