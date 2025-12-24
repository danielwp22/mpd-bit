"""
Run MPD inference on a saved problem set.
This allows you to run inference on the same start/goal pairs used for BIT*.
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
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness

# Import inference components
from mpd.models import UNET_DIM_MULTS
from mpd.utils.loaders import get_planning_task_and_dataset, get_model, load_params_from_yaml
from experiment_launcher import single_experiment_yaml

from dotmap import DotMap


def load_mpd_model(cfg_path='./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml', device='cuda:0'):
    """
    Load the MPD diffusion model.

    Args:
        cfg_path: Path to config file
        device: Device to load model on

    Returns:
        model, dataset, args
    """
    print(f"Loading MPD model from config: {cfg_path}")

    # Load configuration
    args_inference = DotMap(load_params_from_yaml(cfg_path))

    # Set model directory based on selection
    if args_inference.model_selection == "bspline":
        args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    elif args_inference.model_selection == "waypoints":
        args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
    else:
        raise NotImplementedError(f"Unknown model selection: {args_inference.model_selection}")

    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    # Load training arguments and override with inference arguments
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        reload_data=False,
        load_indices=False,
        tensor_args={'device': device, 'dtype': torch.float32},
    )

    # Get planning task and dataset
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    # Get the full dataset from the subset
    dataset = train_subset.dataset

    # Get model - load from checkpoint
    model_path = os.path.join(
        args_inference.model_dir, "checkpoints",
        f'{"ema_" if args_train.get("use_ema", False) else ""}model_current.pth'
    )
    print(f"Loading model from: {model_path}")
    model = get_model(
        checkpoint_path=model_path,
        freeze_loaded_model=True,
        tensor_args={'device': device, 'dtype': torch.float32},
    )

    print(f"Model loaded successfully")

    return model, dataset, planning_task, args_train


def run_inference_on_problem(model, dataset, planning_task, args, env, robot,
                             start_state, goal_state, problem_idx,
                             n_samples=64, n_steps=100,
                             output_dir="multi_run_results/mpd_results",
                             tensor_args=None):
    """
    Run MPD inference on a single problem.

    Args:
        model: MPD model
        dataset: Dataset
        args: Configuration
        env: Environment
        robot: Robot
        start_state: Start configuration (numpy)
        goal_state: Goal configuration (numpy)
        problem_idx: Problem index
        n_samples: Number of trajectory samples to generate
        n_steps: Number of diffusion steps
        output_dir: Output directory
        tensor_args: Tensor arguments

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Running MPD Inference on Problem {problem_idx}")
    print(f"{'='*80}")

    # Convert to torch
    start_state_torch = to_torch(start_state, **tensor_args).unsqueeze(0)  # (1, dof)
    goal_state_torch = to_torch(goal_state, **tensor_args).unsqueeze(0)    # (1, dof)

    inference_start_time = time.time()

    try:
        # Generate trajectories
        print(f"Generating {n_samples} trajectory samples with {n_steps} diffusion steps...")

        # Sample from the model
        with torch.no_grad():
            # Create data sample with proper hard conditions and context
            # Use dataset's method to ensure correct format
            input_data_sample = dataset.create_data_sample_normalized(
                start_state_torch.squeeze(0),  # Remove batch dimension for dataset method
                goal_state_torch.squeeze(0),
            )

            # Extract hard conditions and build context
            hard_conds = input_data_sample["hard_conds"]
            context_d = dataset.build_context(input_data_sample)

            # Add batch dimension to all context tensors (model expects batched inputs)
            for key in context_d:
                if torch.is_tensor(context_d[key]):
                    context_d[key] = context_d[key].unsqueeze(0)

            # Sample trajectories using run_inference
            samples = model.run_inference(
                context_d=context_d,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=dataset.n_learnable_control_points,
                return_chain=False,
            )

            # samples shape: (n_samples, n_support_points, dof)

        inference_time = time.time() - inference_start_time

        # Convert control points to full trajectories for proper metrics
        q_traj_d = planning_task.parametric_trajectory.get_q_trajectory(
            samples,
            start_state_torch.squeeze(0),
            goal_state_torch.squeeze(0),
            get_type=("pos", "vel", "acc"),
            get_time_representation=True
        )
        q_trajs_pos = q_traj_d["pos"]
        q_trajs_vel = q_traj_d["vel"]
        q_trajs_acc = q_traj_d["acc"]

        # Compute metrics on full trajectories
        path_lengths = compute_path_length(q_trajs_pos, robot)
        smoothness = compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)

        # Find best trajectory
        best_idx = torch.argmin(path_lengths)
        best_path_length = path_lengths[best_idx].item()
        best_smoothness = smoothness[best_idx].item()
        best_traj = q_trajs_pos[best_idx].cpu().numpy()  # Use full trajectory, not control points

        # Compute statistics over all samples
        path_lengths_np = path_lengths.cpu().numpy()
        smoothness_np = smoothness.cpu().numpy()

        # Count collision-free trajectories (using full position trajectories)
        collision_free = []
        for i in range(len(q_trajs_pos)):
            traj = q_trajs_pos[i]
            # Check collisions
            is_free = check_trajectory_collision_free(traj, robot, env)
            collision_free.append(is_free)

        n_collision_free = sum(collision_free)
        success_rate = n_collision_free / len(samples)

        result = {
            'success': True,
            'problem_idx': problem_idx,
            'start_state': start_state,
            'goal_state': goal_state,
            'inference_time': inference_time,
            'n_samples': n_samples,
            'n_diffusion_steps': n_steps,
            # Best trajectory metrics
            'path_length': best_path_length,
            'smoothness': best_smoothness,
            'best_trajectory': best_traj,
            # Statistics over all samples
            'all_path_lengths': path_lengths_np,
            'all_smoothness': smoothness_np,
            'path_length_mean': float(np.mean(path_lengths_np)),
            'path_length_std': float(np.std(path_lengths_np)),
            'path_length_min': float(np.min(path_lengths_np)),
            'path_length_max': float(np.max(path_lengths_np)),
            'smoothness_mean': float(np.mean(smoothness_np)),
            'smoothness_std': float(np.std(smoothness_np)),
            # Collision statistics
            'n_collision_free': n_collision_free,
            'success_rate': success_rate,
        }

        print(f"MPD Results:")
        print(f"  Inference time: {inference_time:.3f} sec")
        print(f"  Best path length: {best_path_length:.3f}")
        print(f"  Best smoothness: {best_smoothness:.3f}")
        print(f"  Path length (mean ± std): {result['path_length_mean']:.3f} ± {result['path_length_std']:.3f}")
        print(f"  Collision-free samples: {n_collision_free}/{n_samples} ({success_rate*100:.1f}%)")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

        inference_time = time.time() - inference_start_time
        result = {
            'success': False,
            'problem_idx': problem_idx,
            'start_state': start_state,
            'goal_state': goal_state,
            'inference_time': inference_time,
            'error': str(e),
        }

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"mpd_result_{problem_idx:03d}.pt")
    torch.save(result, result_file)
    print(f"Saved to: {result_file}")

    return result


def check_trajectory_collision_free(trajectory, robot, env, n_interpolate=10):
    """
    Check if a trajectory is collision-free.

    Args:
        trajectory: Trajectory tensor (horizon, dof)
        robot: Robot instance
        env: Environment instance
        n_interpolate: Number of interpolation points between waypoints

    Returns:
        bool: True if collision-free
    """
    # Interpolate trajectory for finer collision checking
    horizon = len(trajectory)
    trajectory_np = to_numpy(trajectory)

    # Check each segment
    for i in range(horizon - 1):
        q1 = trajectory_np[i]
        q2 = trajectory_np[i + 1]

        # Interpolate between waypoints
        for alpha in np.linspace(0, 1, n_interpolate):
            q = q1 * (1 - alpha) + q2 * alpha
            q_torch = to_torch(q, device=trajectory.device, dtype=trajectory.dtype).unsqueeze(0)

            # Get collision sphere positions
            x_pos = robot.fk_map_collision(q_torch)  # (1, num_spheres, 3)

            # Check environment collision
            radii = robot.link_collision_spheres_radii
            x_pos_flat = x_pos.reshape(-1, 3)  # (num_spheres, 3)
            sdf_vals = env.compute_sdf(x_pos_flat)  # (num_spheres,)

            # Check if any sphere is in collision
            if (sdf_vals < radii).any():
                return False

    return True


def aggregate_mpd_results(mpd_results, output_dir="multi_run_results"):
    """
    Aggregate MPD results and compute statistics.

    Args:
        mpd_results: List of MPD result dictionaries
        output_dir: Output directory

    Returns:
        Aggregated statistics
    """
    print(f"\n{'='*80}")
    print("AGGREGATING MPD RESULTS")
    print(f"{'='*80}\n")

    # Filter successful runs
    mpd_success = [r for r in mpd_results if r.get('success', False)]

    if not mpd_success:
        print("No successful MPD results")
        return {}

    # Compute statistics
    path_lengths = np.array([r['path_length'] for r in mpd_success])
    times = np.array([r['inference_time'] for r in mpd_success])
    smoothness = np.array([r['smoothness'] for r in mpd_success])
    success_rates = np.array([r['success_rate'] for r in mpd_success])

    mpd_stats = {
        'n_problems': len(mpd_results),
        'n_success': len(mpd_success),
        'success_rate': len(mpd_success) / len(mpd_results) if mpd_results else 0,
        'path_length_mean': float(np.mean(path_lengths)),
        'path_length_std': float(np.std(path_lengths)),
        'path_length_min': float(np.min(path_lengths)),
        'path_length_max': float(np.max(path_lengths)),
        'inference_time_mean': float(np.mean(times)),
        'inference_time_std': float(np.std(times)),
        'smoothness_mean': float(np.mean(smoothness)),
        'smoothness_std': float(np.std(smoothness)),
        'sample_success_rate_mean': float(np.mean(success_rates)),
        'sample_success_rate_std': float(np.std(success_rates)),
    }

    # Save aggregated results
    aggregated = {
        'mpd_results': mpd_results,
        'mpd_stats': mpd_stats,
    }

    result_file = os.path.join(output_dir, "mpd_aggregated_results.pt")
    torch.save(aggregated, result_file)
    print(f"Aggregated results saved to: {result_file}")

    # Save statistics as YAML
    yaml_file = os.path.join(output_dir, "mpd_statistics.yaml")
    with open(yaml_file, 'w') as f:
        yaml.dump({'mpd': mpd_stats}, f, default_flow_style=False)
    print(f"Statistics saved to: {yaml_file}\n")

    return aggregated


def print_mpd_summary(aggregated):
    """Print summary of MPD results."""

    mpd_stats = aggregated.get('mpd_stats', {})

    if not mpd_stats:
        print("\nNo MPD statistics to display.")
        return

    print(f"\n{'='*80}")
    print("MPD AGGREGATE STATISTICS")
    print(f"{'='*80}\n")

    print(f"{'Metric':<45} {'MPD':<20}")
    print("-"*65)

    # Success rate
    mpd_success_str = f"{mpd_stats['n_success']}/{mpd_stats['n_problems']} ({mpd_stats['success_rate']*100:.1f}%)"
    print(f"{'Success rate (found solutions)':<45} {mpd_success_str:<20}")

    # Sample success rate
    sample_success = f"{mpd_stats['sample_success_rate_mean']*100:.1f}% ± {mpd_stats['sample_success_rate_std']*100:.1f}%"
    print(f"{'Sample success rate (collision-free)':<45} {sample_success:<20}")

    # Path length
    mpd_path = f"{mpd_stats['path_length_mean']:.3f} ± {mpd_stats['path_length_std']:.3f}"
    print(f"{'Path length (mean ± std)':<45} {mpd_path:<20}")

    # Time
    mpd_time = f"{mpd_stats['inference_time_mean']:.3f} ± {mpd_stats['inference_time_std']:.3f}"
    print(f"{'Inference time (sec, mean ± std)':<45} {mpd_time:<20}")

    # Smoothness
    mpd_smooth = f"{mpd_stats['smoothness_mean']:.3f} ± {mpd_stats['smoothness_std']:.3f}"
    print(f"{'Smoothness (mean ± std)':<45} {mpd_smooth:<20}")

    print("="*80 + "\n")


def main(
    problem_set_file="multi_run_results/problem_set.pt",
    output_dir="multi_run_results/mpd_results",
    cfg_path='./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml',
    n_samples=64,
    n_steps=100,
    device='cuda:0',
    seed=42,
):
    """
    Run MPD inference on all problems in a problem set.

    Args:
        problem_set_file: Path to problem set file
        output_dir: Output directory for results
        cfg_path: Path to MPD config file
        n_samples: Number of trajectory samples per problem
        n_steps: Number of diffusion steps
        device: Device to use
        seed: Random seed
    """
    print("\n" + "="*80)
    print("MPD INFERENCE ON PROBLEM SET")
    print("="*80)
    print(f"Problem set: {problem_set_file}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per problem: {n_samples}")
    print(f"Diffusion steps: {n_steps}")
    print("="*80)

    fix_random_seed(seed)

    # Load problem set
    if not os.path.exists(problem_set_file):
        print(f"\nError: Problem set file not found: {problem_set_file}")
        print("Run run_multiple_comparisons.py first to generate a problem set.")
        return

    print(f"\nLoading problem set from: {problem_set_file}")
    problems = torch.load(problem_set_file)
    print(f"Loaded {len(problems)} problems\n")

    # Setup
    tensor_args = {"device": get_torch_device(device), "dtype": torch.float32}

    # Load environment and robot
    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Load MPD model
    model, dataset, planning_task, args = load_mpd_model(cfg_path=cfg_path, device=device)

    # Run inference on all problems
    mpd_results = []

    for i, problem in enumerate(problems):
        print(f"\n{'#'*80}")
        print(f"# MPD Inference on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem['start_state']
        goal_state = problem['goal_state']

        result = run_inference_on_problem(
            model, dataset, planning_task, args, env, robot,
            start_state, goal_state,
            problem_idx=i,
            n_samples=n_samples,
            n_steps=n_steps,
            output_dir=output_dir,
            tensor_args=tensor_args,
        )

        mpd_results.append(result)

    # Aggregate results
    aggregated = aggregate_mpd_results(mpd_results, output_dir=output_dir)

    # Print summary
    print_mpd_summary(aggregated)

    print(f"\n{'='*80}")
    print("MPD RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Problem set: {problem_set_file}")
    print(f"MPD results: {output_dir}/mpd_result_*.pt")
    print(f"Aggregated results: {output_dir}/mpd_aggregated_results.pt")
    print(f"Statistics: {output_dir}/mpd_statistics.yaml")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MPD inference on a saved problem set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MPD on the default problem set
  python run_inference_on_problems.py

  # Run on a specific problem set
  python run_inference_on_problems.py --problem-set my_results/problem_set.pt

  # Use more samples for better results
  python run_inference_on_problems.py --n-samples 128

  # Use custom output directory
  python run_inference_on_problems.py --output-dir my_results/mpd_results
        """
    )

    parser.add_argument("--problem-set", default="multi_run_results/problem_set.pt",
                       help="Path to problem set file (default: multi_run_results/problem_set.pt)")
    parser.add_argument("--output-dir", default="multi_run_results/mpd_results",
                       help="Output directory for MPD results (default: multi_run_results/mpd_results)")
    parser.add_argument("--cfg", default="./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml",
                       help="Path to MPD config file")
    parser.add_argument("--n-samples", type=int, default=64,
                       help="Number of trajectory samples per problem (default: 64)")
    parser.add_argument("--n-steps", type=int, default=100,
                       help="Number of diffusion steps (default: 100)")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    main(
        problem_set_file=args.problem_set,
        output_dir=args.output_dir,
        cfg_path=args.cfg,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        device=args.device,
        seed=args.seed,
    )
