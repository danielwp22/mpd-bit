"""
BIT* Baseline for Motion Planning
Runs BIT* planner from OMPL and computes metrics comparable to the diffusion model inference.
Ensures fair comparison by using the same start/goal configurations and environment.
"""
import os
import sys

# IMPORTANT: Import isaacgym FIRST to avoid import order issues
import isaacgym

import time
import yaml
import numpy as np
from pathlib import Path

# Add pybullet_ompl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../deps/pybullet_ompl'))

import pybullet as p
from pybullet_utils import bullet_client
from pb_ompl.pb_ompl import PbOMPL, PbOMPLRobot

# Now safe to import torch
import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.trajectory.metrics import compute_path_length


def compute_smoothness_finite_diff(trajs):
    """
    Compute smoothness using finite differences (for position-only trajectories).

    Smoothness = sum of acceleration magnitudes along the trajectory.

    Args:
        trajs: Trajectory tensor of shape (batch, horizon, q_dim) containing only positions

    Returns:
        smoothness: Tensor of shape (batch,) with smoothness values
    """
    assert trajs.ndim == 3  # batch, horizon, q_dim

    # Compute velocities using finite differences
    # vel[t] = (pos[t+1] - pos[t])
    velocities = torch.diff(trajs, dim=1)  # (batch, horizon-1, q_dim)

    # Compute accelerations using finite differences on velocities
    # acc[t] = (vel[t+1] - vel[t])
    accelerations = torch.diff(velocities, dim=1)  # (batch, horizon-2, q_dim)

    # Compute smoothness as sum of acceleration norms
    smoothness = torch.linalg.norm(accelerations, dim=-1).sum(-1)  # (batch,)

    return smoothness


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif obj == float('inf'):
        return 'inf'
    elif obj == float('-inf'):
        return '-inf'
    else:
        return obj


def save_problem_configuration(
    start_state, goal_state, env_config,
    save_path="problem_config.pt"
):
    """Save start/goal and environment configuration for reproducibility."""
    config = {
        'start_state': start_state,
        'goal_state': goal_state,
        'env_config': env_config,
    }
    torch.save(config, save_path)
    print(f"Saved problem configuration to {save_path}")


def load_problem_configuration(load_path="problem_config.pt"):
    """Load start/goal and environment configuration."""
    if not os.path.exists(load_path):
        print(f"Warning: {load_path} not found")
        return None

    config = torch.load(load_path, map_location='cpu')
    print(f"Loaded problem configuration from {load_path}")
    return config


def save_diffusion_problem_config(diffusion_results_file, save_path="problem_config.pt"):
    """Extract and save problem configuration from diffusion model results."""
    results = torch.load(diffusion_results_file, map_location='cpu')

    # Extract start and goal
    start_state = to_numpy(results['q_pos_start'])
    goal_state = to_numpy(results['q_pos_goal'])

    # Environment configuration (if available)
    # For now, we'll need to manually specify or extract from the environment
    env_config = {
        'note': 'Environment configuration needs to be saved separately'
    }

    config = {
        'start_state': start_state,
        'goal_state': goal_state,
        'env_config': env_config,
    }

    torch.save(config, save_path)
    print(f"Extracted problem configuration from {diffusion_results_file}")
    print(f"Saved to {save_path}")
    return config


class BITStarBaseline:
    """BIT* baseline planner for comparison with diffusion models."""

    def __init__(
        self,
        robot,  # torch_robotics robot instance
        robot_urdf_path: str,
        planner_name: str = "BITstar",
        allowed_planning_time: float = 60.0,
        min_distance_robot_env: float = 0.05,
        interpolate_num: int = 128,  # Match inference trajectory resolution
        device: str = "cuda:0",
    ):
        self.planner_name = planner_name
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.torch_robot = robot  # For computing smoothness

        self.device = get_torch_device(device)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # Initialize PyBullet in DIRECT mode (headless)
        self.pybullet_client = bullet_client.BulletClient(p.DIRECT)
        self.pybullet_client.setGravity(0, 0, -9.8)

        # Load robot
        self.robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0])
        # Get end-effector link name from torch robot
        link_name_ee = self.torch_robot.link_name_ee if hasattr(self.torch_robot, 'link_name_ee') else 'ee_link'
        self.robot = PbOMPLRobot(self.pybullet_client, self.robot_id, link_name_ee=link_name_ee)

        # Initialize obstacles list (will be set later)
        self.obstacles = []

        # Setup OMPL interface
        self.pb_ompl_interface = PbOMPL(
            self.pybullet_client,
            self.robot,
            self.obstacles,
            min_distance_robot_env=min_distance_robot_env
        )
        self.pb_ompl_interface.set_planner(planner_name)

        print(f"Initialized {planner_name} planner")

    def set_obstacles(self, obstacles):
        """Set environment obstacles."""
        self.obstacles = obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def plan(self, start_state, goal_state, debug=False):
        """
        Plan a trajectory from start to goal using BIT*.

        Returns:
            results_dict with keys:
                - success: bool
                - sol_path: np.ndarray (interpolate_num, dof)
                - planning_time: float
                - path_length: float
                - smoothness: float
                - num_waypoints_before_interpolation: int
        """
        # Convert to Python list of floats (OMPL needs native Python floats, not numpy)
        start_state = [float(x) for x in start_state]
        goal_state = [float(x) for x in goal_state]

        self.robot.set_state(start_state)

        start_time = time.time()

        results_dict = self.pb_ompl_interface.plan(
            goal_state,
            allowed_time=self.allowed_planning_time,
            interpolate_num=self.interpolate_num,
            simplify_path=True,
            debug=debug,
        )

        planning_time = time.time() - start_time
        results_dict['planning_time'] = planning_time

        if results_dict['success']:
            sol_path = results_dict['sol_path']

            # Convert to torch for metric computation
            sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)  # Add batch dimension

            # Compute path length
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()

            # Compute smoothness using finite differences (BITstar returns position-only trajectories)
            smoothness = compute_smoothness_finite_diff(sol_path_torch)[0].item()

            results_dict['path_length'] = path_length
            results_dict['smoothness'] = smoothness
            results_dict['num_waypoints'] = len(sol_path)

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning SUCCESS")
                print(f"  Planning time: {planning_time:.3f} sec")
                print(f"  Path length: {path_length:.3f}")
                print(f"  Smoothness: {smoothness:.3f}")
                print(f"  Num waypoints: {len(sol_path)}")
                print(f"{'='*80}\n")
        else:
            if debug:
                print(f"\n{'='*80}")
                print(f"Planning FAILED after {planning_time:.3f} sec")
                print(f"{'='*80}\n")
            results_dict['path_length'] = float('inf')
            results_dict['smoothness'] = float('inf')
            results_dict['num_waypoints'] = 0

        return results_dict

    def plan_anytime(self, start_state, goal_state, target_path_length=None, debug=False):
        """
        Plan using BIT*'s anytime optimization capability.
        Tracks when the first solution is found and when a solution as good as
        the target path length is found.

        Args:
            start_state: Starting configuration
            goal_state: Goal configuration
            target_path_length: Target path length to beat (typically from diffusion model)
            debug: Print debug information

        Returns:
            results_dict with keys:
                - success: bool
                - sol_path: np.ndarray (interpolate_num, dof)
                - planning_time_total: float - total time spent
                - planning_time_first_solution: float - time to first solution
                - planning_time_target_quality: float - time to reach target quality (or None)
                - path_length: float - final path length
                - path_length_first: float - path length of first solution
                - smoothness: float
                - smoothness_first: float
                - reached_target_quality: bool
                - num_waypoints: int
                - improvement_ratio: float - final_length / first_length
        """
        # Convert to Python list of floats
        start_state = [float(x) for x in start_state]
        goal_state = [float(x) for x in goal_state]

        self.robot.set_state(start_state)

        # Tracking variables
        start_time = time.time()
        first_solution_time = None
        target_quality_time = None
        first_solution_path_length = None
        first_solution_smoothness = None
        reached_target = False

        # We need to monitor solutions as they improve
        # Use a polling approach: run planner for short intervals and check solution quality
        check_interval = 0.5  # Check every 0.5 seconds
        current_best_length = float('inf')
        best_solution = None

        if debug and target_path_length:
            print(f"\n{'='*80}")
            print(f"Anytime Planning - Target path length: {target_path_length:.3f}")
            print(f"{'='*80}\n")

        # Run planning in intervals
        elapsed = 0.0
        iteration = 0

        while elapsed < self.allowed_planning_time:
            iteration += 1
            time_remaining = self.allowed_planning_time - elapsed

            # Plan for a short interval
            iter_time = min(check_interval, time_remaining)

            # Print batch status before planning
            if debug and iteration > 1:
                print(f"\n[{elapsed:.2f}s] Batch {iteration}: Current best: {current_best_length:.3f}")

            # For first iteration, we need to get the initial solution
            # For subsequent iterations, the planner continues optimizing
            if iteration == 1:
                # First call - this will find initial solution
                results_dict = self.pb_ompl_interface.plan(
                    goal_state,
                    allowed_time=iter_time,
                    interpolate_num=self.interpolate_num,
                    simplify_path=False,  # Don't simplify yet - we want to see progress
                    debug=False,
                )
            else:
                # Continue planning (BIT* is anytime)
                # We need to call plan again with more time
                # OMPL will continue from where it left off
                results_dict = self.pb_ompl_interface.plan(
                    goal_state,
                    allowed_time=elapsed + iter_time,
                    interpolate_num=self.interpolate_num,
                    simplify_path=False,
                    debug=False,
                )

            elapsed = time.time() - start_time

            if results_dict['success']:
                sol_path = results_dict['sol_path']
                sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
                path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()

                # Check if this is the first solution
                if first_solution_time is None:
                    first_solution_time = elapsed
                    first_solution_path_length = path_length
                    smoothness_first = compute_smoothness_finite_diff(sol_path_torch)[0].item()
                    first_solution_smoothness = smoothness_first

                    if debug:
                        print(f"\n{'='*60}")
                        print(f"  [{elapsed:.2f}s] FIRST SOLUTION FOUND!")
                        print(f"  Path length: {path_length:.3f}")
                        print(f"{'='*60}\n")

                # Track best solution
                if path_length < current_best_length:
                    current_best_length = path_length
                    best_solution = sol_path

                    if debug and iteration > 1:
                        print(f"  [{elapsed:.2f}s] >>> Improved solution! Path length: {path_length:.3f}")

                # Check if we've reached target quality
                if target_path_length and path_length <= target_path_length:
                    if not reached_target:
                        target_quality_time = elapsed
                        reached_target = True

                        if debug:
                            print(f"  [{elapsed:.2f}s] *** BEAT TARGET! {path_length:.3f} <= {target_path_length:.3f}")
                            print(f"  Stopping after beating target...")

                        # Early termination if we beat the target
                        break

            # If we have a solution and reached target, stop
            if reached_target:
                break

        # Finalize results
        planning_time_total = time.time() - start_time

        if best_solution is not None:
            # Simplify and interpolate the final best solution
            # Do a final interpolation
            from scipy import interpolate

            # Interpolate to desired resolution
            if len(best_solution) != self.interpolate_num:
                t_original = np.linspace(0, 1, len(best_solution))
                t_new = np.linspace(0, 1, self.interpolate_num)

                interpolator = interpolate.interp1d(
                    t_original, best_solution, axis=0, kind='cubic'
                )
                sol_path_final = interpolator(t_new)
            else:
                sol_path_final = best_solution

            sol_path_torch = to_torch(sol_path_final[None, ...], **self.tensor_args)
            path_length_final = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
            smoothness_final = compute_smoothness_finite_diff(sol_path_torch)[0].item()

            results_dict = {
                'success': True,
                'sol_path': sol_path_final,
                'planning_time_total': planning_time_total,
                'planning_time_first_solution': first_solution_time,
                'planning_time_target_quality': target_quality_time,
                'path_length': path_length_final,
                'path_length_first': first_solution_path_length,
                'smoothness': smoothness_final,
                'smoothness_first': first_solution_smoothness,
                'reached_target_quality': reached_target,
                'num_waypoints': len(sol_path_final),
                'improvement_ratio': path_length_final / first_solution_path_length if first_solution_path_length else 1.0,
            }

            if debug:
                print(f"\n{'='*80}")
                print(f"Anytime Planning COMPLETE")
                print(f"  Total time: {planning_time_total:.3f} sec")
                print(f"  First solution time: {first_solution_time:.3f} sec")
                if target_quality_time:
                    print(f"  Target quality time: {target_quality_time:.3f} sec")
                print(f"  First solution length: {first_solution_path_length:.3f}")
                print(f"  Final path length: {path_length_final:.3f}")
                print(f"  Improvement: {(1 - results_dict['improvement_ratio'])*100:.1f}%")
                if target_path_length:
                    print(f"  Target path length: {target_path_length:.3f}")
                    print(f"  Reached target: {reached_target}")
                print(f"{'='*80}\n")
        else:
            # No solution found
            results_dict = {
                'success': False,
                'planning_time_total': planning_time_total,
                'planning_time_first_solution': None,
                'planning_time_target_quality': None,
                'path_length': float('inf'),
                'path_length_first': float('inf'),
                'smoothness': float('inf'),
                'smoothness_first': float('inf'),
                'reached_target_quality': False,
                'num_waypoints': 0,
                'improvement_ratio': 1.0,
            }

            if debug:
                print(f"\n{'='*80}")
                print(f"Anytime Planning FAILED after {planning_time_total:.3f} sec")
                print(f"{'='*80}\n")

        return results_dict

    def evaluate_multiple_problems(
        self,
        start_states,
        goal_states,
        results_dir="logs_bitstar",
        debug=False
    ):
        """
        Evaluate planner on multiple start-goal pairs.

        Args:
            start_states: list of start configurations
            goal_states: list of goal configurations
            results_dir: directory to save results
            debug: whether to print debug info

        Returns:
            statistics: dict with aggregated metrics
        """
        os.makedirs(results_dir, exist_ok=True)

        n_problems = len(start_states)
        results_all = []

        success_count = 0
        planning_times = []
        path_lengths = []
        smoothness_values = []

        for i, (start, goal) in enumerate(zip(start_states, goal_states)):
            print(f"\n{'='*80}")
            print(f"Problem {i+1}/{n_problems}")
            print(f"{'='*80}")

            result = self.plan(start, goal, debug=debug)
            results_all.append(result)

            if result['success']:
                success_count += 1
                planning_times.append(result['planning_time'])
                path_lengths.append(result['path_length'])
                smoothness_values.append(result['smoothness'])

                # Save individual trajectory
                save_path = os.path.join(results_dir, f"trajectory_{i:03d}.npy")
                np.save(save_path, result['sol_path'])

        # Compute statistics
        success_rate = success_count / n_problems

        statistics = {
            'planner': self.planner_name,
            'n_problems': n_problems,
            'success_count': success_count,
            'success_rate': success_rate,
            'planning_time_mean': np.mean(planning_times) if planning_times else float('inf'),
            'planning_time_std': np.std(planning_times) if planning_times else 0.0,
            'planning_time_min': np.min(planning_times) if planning_times else float('inf'),
            'planning_time_max': np.max(planning_times) if planning_times else 0.0,
            'path_length_mean': np.mean(path_lengths) if path_lengths else float('inf'),
            'path_length_std': np.std(path_lengths) if path_lengths else 0.0,
            'path_length_min': np.min(path_lengths) if path_lengths else float('inf'),
            'path_length_max': np.max(path_lengths) if path_lengths else 0.0,
            'smoothness_mean': np.mean(smoothness_values) if smoothness_values else float('inf'),
            'smoothness_std': np.std(smoothness_values) if smoothness_values else 0.0,
            'smoothness_min': np.min(smoothness_values) if smoothness_values else float('inf'),
            'smoothness_max': np.max(smoothness_values) if smoothness_values else 0.0,
        }

        # Save statistics
        with open(os.path.join(results_dir, "statistics.yaml"), 'w') as f:
            yaml.dump(convert_to_serializable(statistics), f, default_flow_style=False)

        # Print summary
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY - {self.planner_name}")
        print(f"{'='*80}")
        print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{n_problems})")
        if planning_times:
            print(f"Planning time: {statistics['planning_time_mean']:.3f} ± {statistics['planning_time_std']:.3f} sec")
            print(f"Path length: {statistics['path_length_mean']:.3f} ± {statistics['path_length_std']:.3f}")
            print(f"Smoothness: {statistics['smoothness_mean']:.3f} ± {statistics['smoothness_std']:.3f}")
        print(f"{'='*80}\n")

        return statistics

    def terminate(self):
        """Cleanup PyBullet connection."""
        self.pybullet_client.disconnect()


# Example usage functions for different environments
def run_panda_spheres3d(
    n_problems=10,
    planner_name="BITstar",
    allowed_time=60.0,
    seed=42,
    use_same_problems_as_diffusion=False,
    diffusion_results_file=None,
    save_config_path=None,
    use_anytime=False,
):
    """Run BIT* on Panda Spheres3D environment."""
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda
    from torch_robotics.tasks.tasks import PlanningTask
    from torch_robotics.torch_utils.seed import fix_random_seed

    fix_random_seed(seed)

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Create environment and robot
    # Don't precompute SDF - OMPL does its own collision checking
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=False,
        sdf_cell_size=0.05,  # Larger cell size to save memory if needed
        tensor_args=tensor_args
    )
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Get target path length from diffusion if using anytime mode
    target_path_length = None
    if use_anytime and use_same_problems_as_diffusion and diffusion_results_file:
        diff_results = torch.load(diffusion_results_file, map_location='cpu')
        if 'metrics' in diff_results and 'trajs_best' in diff_results['metrics']:
            target_path_length = float(diff_results['metrics']['trajs_best']['path_length'])
            print(f"\nTarget path length from diffusion: {target_path_length:.3f}")

    # Sample or load start/goal pairs
    if use_same_problems_as_diffusion and diffusion_results_file:
        print(f"\nLoading start/goal from diffusion results: {diffusion_results_file}")
        diff_results = torch.load(diffusion_results_file, map_location='cpu')

        start_state = to_numpy(diff_results['q_pos_start'])
        goal_state = to_numpy(diff_results['q_pos_goal'])

        start_states = [start_state]
        goal_states = [goal_state]

        print(f"Start state: {start_state}")
        print(f"Goal state: {goal_state}")

        # Save configuration for reproducibility
        if save_config_path:
            env_config = {
                'env_type': 'EnvSpheres3D',
                'seed': seed,
                'note': 'Environment uses same random seed as diffusion model'
            }
            save_problem_configuration(start_state, goal_state, env_config, save_config_path)
    else:
        print("Sampling new random start and goal configurations...")
        start_states = []
        goal_states = []
        for i in range(n_problems):
            # Sample random joint configurations
            q_start = robot.random_q(n_samples=1)[0]  # Get single sample from batch
            q_goal = robot.random_q(n_samples=1)[0]   # Get single sample from batch
            start_states.append(to_numpy(q_start))
            goal_states.append(to_numpy(q_goal))
            print(f"  Sampled problem {i+1}/{n_problems}")
            print(f"    Note: Not checking for collisions - OMPL will handle that")

        # Save first problem configuration for reproducibility
        if save_config_path and len(start_states) > 0:
            env_config = {
                'env_type': 'EnvSpheres3D',
                'seed': seed,
                'n_problems': n_problems,
                'note': 'Random samples, not guaranteed collision-free'
            }
            save_problem_configuration(
                start_states[0], goal_states[0], env_config, save_config_path
            )

    # Get robot URDF path
    robot_urdf_path = robot.robot_urdf_file

    # Initialize baseline planner (pass torch_robotics robot for smoothness computation)
    baseline = BITStarBaseline(
        robot=robot,
        robot_urdf_path=robot_urdf_path,
        planner_name=planner_name,
        allowed_planning_time=allowed_time,
        min_distance_robot_env=0.05,
        interpolate_num=128,
    )

    # Evaluate using anytime mode or regular mode
    results_dir = f"logs_{planner_name.lower()}_panda_spheres3d"

    if use_anytime:
        statistics = run_anytime_evaluation(
            baseline,
            start_states,
            goal_states,
            target_path_length=target_path_length,
            results_dir=results_dir,
        )
    else:
        statistics = baseline.evaluate_multiple_problems(
            start_states,
            goal_states,
            results_dir=results_dir,
            debug=True
        )

    baseline.terminate()

    return statistics


def run_anytime_evaluation(baseline, start_states, goal_states, target_path_length=None, results_dir="logs_bitstar_anytime"):
    """Run anytime evaluation on multiple problems."""
    os.makedirs(results_dir, exist_ok=True)

    n_problems = len(start_states)
    results_all = []

    success_count = 0
    planning_times_total = []
    planning_times_first = []
    planning_times_target = []
    path_lengths = []
    path_lengths_first = []
    smoothness_values = []
    reached_target_count = 0

    for i, (start, goal) in enumerate(zip(start_states, goal_states)):
        print(f"\n{'='*80}")
        print(f"Problem {i+1}/{n_problems}")
        print(f"{'='*80}")

        result = baseline.plan_anytime(
            start, goal,
            target_path_length=target_path_length,
            debug=True
        )
        results_all.append(result)

        if result['success']:
            success_count += 1
            planning_times_total.append(result['planning_time_total'])
            planning_times_first.append(result['planning_time_first_solution'])
            path_lengths.append(result['path_length'])
            path_lengths_first.append(result['path_length_first'])
            smoothness_values.append(result['smoothness'])

            if result['reached_target_quality']:
                reached_target_count += 1
                planning_times_target.append(result['planning_time_target_quality'])

            # Save individual trajectory
            save_path = os.path.join(results_dir, f"trajectory_{i:03d}.npy")
            np.save(save_path, result['sol_path'])

    # Compute statistics
    success_rate = success_count / n_problems

    statistics = {
        'planner': baseline.planner_name,
        'mode': 'anytime',
        'n_problems': n_problems,
        'success_count': success_count,
        'success_rate': success_rate,
        'target_path_length': target_path_length if target_path_length else None,
        'reached_target_count': reached_target_count,
        'reached_target_rate': reached_target_count / n_problems if target_path_length else None,

        # Total planning time
        'planning_time_total_mean': np.mean(planning_times_total) if planning_times_total else float('inf'),
        'planning_time_total_std': np.std(planning_times_total) if planning_times_total else 0.0,

        # First solution time
        'planning_time_first_mean': np.mean(planning_times_first) if planning_times_first else float('inf'),
        'planning_time_first_std': np.std(planning_times_first) if planning_times_first else 0.0,

        # Time to reach target quality
        'planning_time_target_mean': np.mean(planning_times_target) if planning_times_target else float('inf'),
        'planning_time_target_std': np.std(planning_times_target) if planning_times_target else 0.0,

        # Final path lengths
        'path_length_mean': np.mean(path_lengths) if path_lengths else float('inf'),
        'path_length_std': np.std(path_lengths) if path_lengths else 0.0,

        # First solution path lengths
        'path_length_first_mean': np.mean(path_lengths_first) if path_lengths_first else float('inf'),
        'path_length_first_std': np.std(path_lengths_first) if path_lengths_first else 0.0,

        # Smoothness
        'smoothness_mean': np.mean(smoothness_values) if smoothness_values else float('inf'),
        'smoothness_std': np.std(smoothness_values) if smoothness_values else 0.0,
    }

    # Save statistics
    with open(os.path.join(results_dir, "statistics.yaml"), 'w') as f:
        yaml.dump(convert_to_serializable(statistics), f, default_flow_style=False)

    # Print summary
    print(f"\n{'='*80}")
    print(f"ANYTIME EVALUATION SUMMARY - {baseline.planner_name}")
    print(f"{'='*80}")
    print(f"Success rate: {success_rate*100:.1f}% ({success_count}/{n_problems})")

    if planning_times_first:
        print(f"\nFirst solution:")
        print(f"  Time: {statistics['planning_time_first_mean']:.3f} ± {statistics['planning_time_first_std']:.3f} sec")
        print(f"  Path length: {statistics['path_length_first_mean']:.3f} ± {statistics['path_length_first_std']:.3f}")

    if planning_times_total:
        print(f"\nFinal solution (after optimization):")
        print(f"  Time: {statistics['planning_time_total_mean']:.3f} ± {statistics['planning_time_total_std']:.3f} sec")
        print(f"  Path length: {statistics['path_length_mean']:.3f} ± {statistics['path_length_std']:.3f}")
        print(f"  Smoothness: {statistics['smoothness_mean']:.3f} ± {statistics['smoothness_std']:.3f}")

    if target_path_length:
        print(f"\nTarget quality (path length ≤ {target_path_length:.3f}):")
        print(f"  Reached target: {reached_target_count}/{n_problems} ({reached_target_count/n_problems*100:.1f}%)")
        if planning_times_target:
            print(f"  Time to target: {statistics['planning_time_target_mean']:.3f} ± {statistics['planning_time_target_std']:.3f} sec")

    print(f"{'='*80}\n")

    return statistics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BIT* baseline planner")
    parser.add_argument("--n-problems", type=int, default=1,
                       help="Number of start-goal pairs to test")
    parser.add_argument("--planner", default="BITstar",
                       choices=["BITstar", "ABITstar", "AITstar", "RRTstar", "RRTConnect"],
                       help="OMPL planner to use")
    parser.add_argument("--time", type=float, default=60.0,
                       help="Max planning time per problem (seconds)")
    parser.add_argument("--seed", type=int, default=2,
                       help="Random seed (use same as diffusion model for fair comparison)")
    parser.add_argument("--use-diffusion-problem", action="store_true",
                       help="Use same start/goal as diffusion model")
    parser.add_argument("--diffusion-results", default="logs/2/results_single_plan-000.pt",
                       help="Path to diffusion model results file")
    parser.add_argument("--save-config", default="problem_config.pt",
                       help="Path to save problem configuration")
    parser.add_argument("--anytime", action="store_true",
                       help="Use anytime mode: track when first solution is found and when it reaches diffusion quality")

    args = parser.parse_args()

    print("\n" + "="*80)
    if args.anytime:
        print(f"Running {args.planner} Baseline (ANYTIME MODE) on Panda Spheres3D Environment")
    else:
        print(f"Running {args.planner} Baseline on Panda Spheres3D Environment")
    print("="*80 + "\n")

    statistics = run_panda_spheres3d(
        n_problems=args.n_problems,
        planner_name=args.planner,
        allowed_time=args.time,
        seed=args.seed,
        use_same_problems_as_diffusion=args.use_diffusion_problem,
        diffusion_results_file=args.diffusion_results if args.use_diffusion_problem else None,
        save_config_path=args.save_config if args.use_diffusion_problem else None,
        use_anytime=args.anytime,
    )

    print(f"\nDone! Results saved to logs_{args.planner.lower()}_panda_spheres3d/")

    print("\nUsage examples:")
    print("  # Run on new random problems:")
    print("  python bitstar_baseline.py --n-problems 5 --planner BITstar")
    print("\n  # Use SAME problem as diffusion model (fair comparison):")
    print("  python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt")
    print("\n  # Use ANYTIME mode to track when BIT* reaches diffusion quality:")
    print("  python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --anytime")