"""
Complete comparison workflow: Run both BIT* and MPD on multiple problems.

This script:
1. Generates N random start/goal pairs (or loads existing)
2. Runs BIT* on each problem for 600s with per-second tracking
3. Runs MPD on each problem with NO visualization
4. Collects comprehensive metrics for both algorithms
5. Stores all data in analyzable format

BIT* metrics (per second):
- Path length
- Smoothness
- Tree size (num vertices)
- Number of batches
- Success status
- Time when better/equal than MPD
- Time when first solution found

MPD metrics (per problem):
- Collision frequency
- Mean successful path length
- Best successful path length
- Smoothness
- Inference time
- Success rate

All data is saved for later analysis.
"""
import os
import sys
import time
import yaml
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

# IMPORTANT: Import isaacgym FIRST before torch
import isaacgym

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness

# Import BIT* components
from run_bitstar_with_tracking import BITStarWithTracking

# Import MPD components
from mpd.models import UNET_DIM_MULTS
from mpd.utils.loaders import get_planning_task_and_dataset, get_model, load_params_from_yaml
from dotmap import DotMap


def generate_problem_set(robot, env, n_problems, seed=42, output_dir="multi_run_results"):
    """
    Generate random collision-free start/goal pairs and save them.
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
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
    allowed_time=600.0,
    tracking_interval=1.0,
    output_dir="multi_run_results",
    tensor_args=None,
):
    """
    Run BIT* on all problems with detailed per-second tracking.
    """
    print(f"\n{'='*80}")
    print(f"Running BIT* on {len(problems)} problems")
    print(f"Time per problem: {allowed_time}s")
    print(f"Tracking interval: {tracking_interval}s")
    print(f"{'='*80}\n")

    results = []

    for i, problem in enumerate(problems):
        print(f"\n{'#'*80}")
        print(f"# BIT* on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem['start_state']
        goal_state = problem['goal_state']

        try:
            # Initialize BIT* planner
            planner = BITStarWithTracking(
                robot=robot,
                allowed_planning_time=allowed_time,
                interpolate_num=128,
                device="cuda:0",
                batch_size=100,
            )
            planner.set_obstacles(env)

            # Plan with tracking (no target initially)
            result = planner.plan_with_tracking(
                start_state,
                goal_state,
                target_path_length=None,  # Will be set later after MPD runs
                tracking_interval=tracking_interval,
                debug=True
            )

            # Add problem info
            result['problem_idx'] = i
            result['start_state'] = start_state
            result['goal_state'] = goal_state

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
                'planning_time': allowed_time,
                'interval_metrics': [],
            }
            results.append(result)

    return results


def load_mpd_model(cfg_path='./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml', device='cuda:0'):
    """Load the MPD diffusion model."""
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


def check_trajectory_collision_free(trajectory, robot, env, n_interpolate=10):
    """Check if a trajectory is collision-free."""
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


def run_mpd_on_problems(
    problems,
    env,
    robot,
    model,
    dataset,
    planning_task,
    args,
    n_samples=64,
    n_steps=100,
    output_dir="multi_run_results",
    tensor_args=None,
):
    """
    Run MPD inference on all problems (NO visualization).
    """
    print(f"\n{'='*80}")
    print(f"Running MPD on {len(problems)} problems")
    print(f"Samples per problem: {n_samples}")
    print(f"Diffusion steps: {n_steps}")
    print(f"Visualization: DISABLED")
    print(f"{'='*80}\n")

    mpd_output_dir = os.path.join(output_dir, "mpd_results")
    os.makedirs(mpd_output_dir, exist_ok=True)

    # Build sampling kwargs from the config so we match the fast inference path
    diffusion_method = getattr(args, "diffusion_sampling_method", "ddim")
    diffusion_cfg = args.get(diffusion_method, {}) if hasattr(args, "get") else {}
    diffusion_cfg = dict(diffusion_cfg)  # shallow copy; avoid mutating args

    if diffusion_method == "ddim":
        # Respect the requested number of diffusion steps
        diffusion_cfg["ddim_sampling_timesteps"] = n_steps
        frac = diffusion_cfg.get("t_start_guide_steps_fraction", 0.0)
        diffusion_cfg["t_start_guide"] = int(np.ceil(frac * diffusion_cfg["ddim_sampling_timesteps"]))
    elif diffusion_method == "ddpm":
        # ddpm uses the model's n_diffusion_steps; keep fraction consistent
        frac = diffusion_cfg.get("t_start_guide_steps_fraction", 0.0)
        diffusion_cfg["t_start_guide"] = int(np.ceil(frac * model.n_diffusion_steps))

    sampling_kwargs = {"method": diffusion_method, **diffusion_cfg}
    print(f"Using diffusion sampling: {diffusion_method} with {sampling_kwargs.get('ddim_sampling_timesteps', model.n_diffusion_steps if hasattr(model, 'n_diffusion_steps') else 'n/a')} steps")

    results = []

    for i, problem in enumerate(problems):
        print(f"\n{'#'*80}")
        print(f"# MPD on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem['start_state']
        goal_state = problem['goal_state']

        # Convert to torch
        start_state_torch = to_torch(start_state, **tensor_args).unsqueeze(0)  # (1, dof)
        goal_state_torch = to_torch(goal_state, **tensor_args).unsqueeze(0)    # (1, dof)

        inference_start_time = time.time()

        try:
            # Generate trajectories (NO VISUALIZATION)
            print(f"Generating {n_samples} trajectory samples...")

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

                # NOTE: Do NOT add batch dimension manually - run_inference handles batching internally
                # using einops.repeat to replicate context for n_samples

                # Sample trajectories using run_inference
                samples = model.run_inference(
                context_d=context_d,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=dataset.n_learnable_control_points,
                return_chain=False,
                **sampling_kwargs,
            )

            inference_time = time.time() - inference_start_time

            # Convert control points to full trajectories (pos, vel, acc)
            # This is necessary for proper smoothness computation
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
            smoothness_values = compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)

            # Find best trajectory
            best_idx = torch.argmin(path_lengths)
            best_path_length = path_lengths[best_idx].item()
            best_smoothness = smoothness_values[best_idx].item()
            best_traj = q_trajs_pos[best_idx].cpu().numpy()  # Use full trajectory, not control points

            # Compute statistics over all samples
            path_lengths_np = path_lengths.cpu().numpy()
            smoothness_np = smoothness_values.cpu().numpy()

            # Count collision-free trajectories (check using full position trajectories)
            collision_free_mask = []
            collision_free_lengths = []
            for j in range(len(q_trajs_pos)):
                traj = q_trajs_pos[j]
                is_free = check_trajectory_collision_free(traj, robot, env)
                collision_free_mask.append(is_free)
                if is_free:
                    collision_free_lengths.append(path_lengths_np[j])

            n_collision_free = sum(collision_free_mask)
            collision_rate = 1.0 - (n_collision_free / len(samples))

            # Statistics for collision-free paths only
            if collision_free_lengths:
                mean_collision_free_length = float(np.mean(collision_free_lengths))
                best_collision_free_length = float(np.min(collision_free_lengths))
            else:
                mean_collision_free_length = float('inf')
                best_collision_free_length = float('inf')

            result = {
                'success': True,
                'problem_idx': i,
                'start_state': start_state,
                'goal_state': goal_state,
                'inference_time': inference_time,
                'n_samples': n_samples,
                'n_diffusion_steps': n_steps,
                # Best trajectory metrics (may include collisions)
                'path_length': best_path_length,
                'smoothness': best_smoothness,
                'best_trajectory': best_traj,
                # Collision statistics
                'n_collision_free': n_collision_free,
                'n_total_samples': len(samples),
                'collision_rate': collision_rate,
                'collision_free_mask': collision_free_mask,
                # Collision-free trajectory statistics
                'mean_collision_free_path_length': mean_collision_free_length,
                'best_collision_free_path_length': best_collision_free_length,
                # All samples statistics
                'all_path_lengths': path_lengths_np,
                'all_smoothness': smoothness_np,
                'path_length_mean': float(np.mean(path_lengths_np)),
                'path_length_std': float(np.std(path_lengths_np)),
                'smoothness_mean': float(np.mean(smoothness_np)),
                'smoothness_std': float(np.std(smoothness_np)),
            }

            print(f"MPD Results:")
            print(f"  Inference time: {inference_time:.3f} sec")
            print(f"  Best path length: {best_path_length:.3f}")
            print(f"  Best smoothness: {best_smoothness:.3f}")
            print(f"  Collision-free samples: {n_collision_free}/{n_samples} ({(1-collision_rate)*100:.1f}%)")
            print(f"  Mean collision-free path length: {mean_collision_free_length:.3f}")
            print(f"  Best collision-free path length: {best_collision_free_length:.3f}")

        except Exception as e:
            print(f"Error during MPD inference on problem {i}: {e}")
            import traceback
            traceback.print_exc()

            inference_time = time.time() - inference_start_time
            result = {
                'success': False,
                'problem_idx': i,
                'start_state': start_state,
                'goal_state': goal_state,
                'inference_time': inference_time,
                'error': str(e),
            }

        # Save individual result
        result_file = os.path.join(mpd_output_dir, f"mpd_result_{i:03d}.pt")
        torch.save(result, result_file)
        print(f"Saved to: {result_file}")

        results.append(result)

    return results


def compute_comparison_metrics(bitstar_results, mpd_results):
    """
    Compute comparison metrics between BIT* and MPD.

    For each BIT* result, determines:
    - Time when BIT* solution was better or equal to MPD
    - Time when BIT* found first solution
    """
    comparison_data = []

    for br in bitstar_results:
        idx = br['problem_idx']

        if idx >= len(mpd_results):
            continue

        mr = mpd_results[idx]

        if not br.get('success', False) or not mr.get('success', False):
            comparison_data.append({
                'problem_idx': idx,
                'bitstar_success': br.get('success', False),
                'mpd_success': mr.get('success', False),
                'time_to_first_solution': None,
                'time_to_match_mpd': None,
                'bitstar_beats_mpd': False,
            })
            continue

        # Get MPD target (best collision-free path)
        mpd_target = mr.get('best_collision_free_path_length', mr['path_length'])

        # Check interval metrics to find when BIT* matched/beat MPD
        time_to_match_mpd = None
        time_to_first_solution = br.get('time_to_first_solution')

        for metric in br.get('interval_metrics', []):
            if metric.get('has_solution', False):
                if time_to_match_mpd is None and metric['path_length'] <= mpd_target:
                    time_to_match_mpd = metric['time']
                    break

        bitstar_final = br['path_length']
        bitstar_beats_mpd = (bitstar_final <= mpd_target)

        comparison_data.append({
            'problem_idx': idx,
            'bitstar_success': True,
            'mpd_success': True,
            'mpd_target_length': mpd_target,
            'bitstar_final_length': bitstar_final,
            'time_to_first_solution': time_to_first_solution,
            'time_to_match_mpd': time_to_match_mpd if time_to_match_mpd else "N/A",
            'bitstar_beats_mpd': bitstar_beats_mpd,
        })

    return comparison_data


def aggregate_all_results(bitstar_results, mpd_results, comparison_data, output_dir="multi_run_results"):
    """
    Aggregate all results and compute comprehensive statistics.
    """
    print(f"\n{'='*80}")
    print("AGGREGATING ALL RESULTS")
    print(f"{'='*80}\n")

    # BIT* statistics
    bitstar_success = [r for r in bitstar_results if r.get('success', False)]

    bitstar_stats = {}
    if bitstar_success:
        path_lengths = np.array([r['path_length'] for r in bitstar_success])
        times = np.array([r['planning_time'] for r in bitstar_success])
        smoothness = np.array([r['smoothness'] for r in bitstar_success])
        mean_jerk = np.array([r['mean_jerk'] for r in bitstar_success])
        first_times = np.array([r['time_to_first_solution'] for r in bitstar_success if r.get('time_to_first_solution')])

        # Tree size statistics (from final interval metric)
        tree_sizes = []
        for r in bitstar_success:
            if r.get('interval_metrics'):
                last_metric = r['interval_metrics'][-1]
                tree_sizes.append(last_metric.get('num_vertices', 0))

        bitstar_stats = {
            'n_problems': len(bitstar_results),
            'n_success': len(bitstar_success),
            'success_rate': len(bitstar_success) / len(bitstar_results) if bitstar_results else 0,
            'path_length_mean': float(np.mean(path_lengths)),
            'path_length_std': float(np.std(path_lengths)),
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

        if tree_sizes:
            bitstar_stats['tree_size_mean'] = float(np.mean(tree_sizes))
            bitstar_stats['tree_size_std'] = float(np.std(tree_sizes))

    # MPD statistics
    mpd_success = [r for r in mpd_results if r.get('success', False)]

    mpd_stats = {}
    if mpd_success:
        collision_rates = np.array([r['collision_rate'] for r in mpd_success])
        mean_cf_lengths = np.array([r['mean_collision_free_path_length'] for r in mpd_success if r['mean_collision_free_path_length'] != float('inf')])
        best_cf_lengths = np.array([r['best_collision_free_path_length'] for r in mpd_success if r['best_collision_free_path_length'] != float('inf')])
        smoothness = np.array([r['smoothness'] for r in mpd_success])
        times = np.array([r['inference_time'] for r in mpd_success])

        mpd_stats = {
            'n_problems': len(mpd_results),
            'n_success': len(mpd_success),
            'success_rate': len(mpd_success) / len(mpd_results) if mpd_results else 0,
            'collision_rate_mean': float(np.mean(collision_rates)),
            'collision_rate_std': float(np.std(collision_rates)),
            'mean_collision_free_path_length_mean': float(np.mean(mean_cf_lengths)) if len(mean_cf_lengths) > 0 else float('inf'),
            'mean_collision_free_path_length_std': float(np.std(mean_cf_lengths)) if len(mean_cf_lengths) > 0 else 0,
            'best_collision_free_path_length_mean': float(np.mean(best_cf_lengths)) if len(best_cf_lengths) > 0 else float('inf'),
            'best_collision_free_path_length_std': float(np.std(best_cf_lengths)) if len(best_cf_lengths) > 0 else 0,
            'smoothness_mean': float(np.mean(smoothness)),
            'smoothness_std': float(np.std(smoothness)),
            'inference_time_mean': float(np.mean(times)),
            'inference_time_std': float(np.std(times)),
        }

    # Comparison statistics
    n_bitstar_beats = sum(1 for c in comparison_data if c.get('bitstar_beats_mpd', False))
    match_times = [c['time_to_match_mpd'] for c in comparison_data if isinstance(c.get('time_to_match_mpd'), (int, float))]

    comparison_stats = {
        'n_both_success': sum(1 for c in comparison_data if c.get('bitstar_success') and c.get('mpd_success')),
        'n_bitstar_beats_mpd': n_bitstar_beats,
        'bitstar_beats_mpd_rate': n_bitstar_beats / len(comparison_data) if comparison_data else 0,
    }

    if match_times:
        comparison_stats['time_to_match_mpd_mean'] = float(np.mean(match_times))
        comparison_stats['time_to_match_mpd_std'] = float(np.std(match_times))

    # Save aggregated results
    aggregated = {
        'bitstar_results': bitstar_results,
        'mpd_results': mpd_results,
        'comparison_data': comparison_data,
        'bitstar_stats': bitstar_stats,
        'mpd_stats': mpd_stats,
        'comparison_stats': comparison_stats,
        'timestamp': datetime.now().isoformat(),
    }

    result_file = os.path.join(output_dir, "complete_aggregated_results.pt")
    torch.save(aggregated, result_file)
    print(f"Aggregated results saved to: {result_file}")

    # Save statistics as YAML
    stats_dict = {
        'bitstar': bitstar_stats,
        'mpd': mpd_stats,
        'comparison': comparison_stats,
    }

    yaml_file = os.path.join(output_dir, "complete_statistics.yaml")
    with open(yaml_file, 'w') as f:
        yaml.dump(stats_dict, f, default_flow_style=False)
    print(f"Statistics saved to: {yaml_file}")

    # Save comparison data as JSON for easy reading
    json_file = os.path.join(output_dir, "comparison_data.json")
    with open(json_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Comparison data saved to: {json_file}\n")

    return aggregated


def export_to_csv(bitstar_results, mpd_results, comparison_data, output_dir="multi_run_results"):
    """
    Export all data to comprehensive CSV files.

    Creates three CSV files:
    1. bitstar_timeseries.csv: Second-by-second BIT* metrics for each problem
    2. mpd_results.csv: MPD metrics for each problem
    3. comparison_summary.csv: Comparison metrics for each problem
    """
    print(f"\n{'='*80}")
    print("EXPORTING DATA TO CSV FILES")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Export BIT* time-series data
    bitstar_csv_path = os.path.join(output_dir, "bitstar_timeseries.csv")
    print(f"Exporting BIT* time-series data to: {bitstar_csv_path}")

    with open(bitstar_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'time',
            'iteration',
            'path_length',
            'smoothness',
            'mean_jerk',
            'num_vertices',
            'num_samples',
            'num_batches',
            'has_solution',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        for result in bitstar_results:
            problem_idx = result['problem_idx']
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            # Write each interval metric as a row
            for metric in result.get('interval_metrics', []):
                row = {
                    'problem_idx': problem_idx,
                    'time': metric['time'],
                    'iteration': metric['iteration'],
                    'path_length': metric['path_length'],
                    'smoothness': metric['smoothness'],
                    'mean_jerk': metric['mean_jerk'],
                    'num_vertices': metric['num_vertices'],
                    'num_samples': metric['num_samples'],
                    'num_batches': metric['num_batches'],
                    'has_solution': metric['has_solution'],
                    'start_state': start_state_str,
                    'goal_state': goal_state_str,
                }
                writer.writerow(row)
                total_rows += 1

        print(f"  Wrote {total_rows} rows (time-series data points)")

    # 2. Export MPD results
    mpd_csv_path = os.path.join(output_dir, "mpd_results.csv")
    print(f"Exporting MPD results to: {mpd_csv_path}")

    with open(mpd_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'success',
            'inference_time',
            'n_samples',
            'n_diffusion_steps',
            'best_path_length',
            'best_smoothness',
            'n_collision_free',
            'n_total_samples',
            'collision_rate',
            'mean_collision_free_path_length',
            'best_collision_free_path_length',
            'path_length_mean',
            'path_length_std',
            'smoothness_mean',
            'smoothness_std',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in mpd_results:
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            row = {
                'problem_idx': result['problem_idx'],
                'success': result.get('success', False),
                'inference_time': result.get('inference_time', 0),
                'n_samples': result.get('n_samples', 0),
                'n_diffusion_steps': result.get('n_diffusion_steps', 0),
                'best_path_length': result.get('path_length', float('inf')),
                'best_smoothness': result.get('smoothness', float('inf')),
                'n_collision_free': result.get('n_collision_free', 0),
                'n_total_samples': result.get('n_total_samples', 0),
                'collision_rate': result.get('collision_rate', 1.0),
                'mean_collision_free_path_length': result.get('mean_collision_free_path_length', float('inf')),
                'best_collision_free_path_length': result.get('best_collision_free_path_length', float('inf')),
                'path_length_mean': result.get('path_length_mean', float('inf')),
                'path_length_std': result.get('path_length_std', 0),
                'smoothness_mean': result.get('smoothness_mean', float('inf')),
                'smoothness_std': result.get('smoothness_std', 0),
                'start_state': start_state_str,
                'goal_state': goal_state_str,
            }
            writer.writerow(row)

        print(f"  Wrote {len(mpd_results)} rows (one per problem)")

    # 3. Export comparison summary
    comparison_csv_path = os.path.join(output_dir, "comparison_summary.csv")
    print(f"Exporting comparison summary to: {comparison_csv_path}")

    with open(comparison_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'bitstar_success',
            'mpd_success',
            'mpd_target_length',
            'bitstar_final_length',
            'time_to_first_solution',
            'time_to_match_mpd',
            'bitstar_beats_mpd',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for comp in comparison_data:
            row = {
                'problem_idx': comp['problem_idx'],
                'bitstar_success': comp.get('bitstar_success', False),
                'mpd_success': comp.get('mpd_success', False),
                'mpd_target_length': comp.get('mpd_target_length', float('inf')),
                'bitstar_final_length': comp.get('bitstar_final_length', float('inf')),
                'time_to_first_solution': comp.get('time_to_first_solution', 'N/A'),
                'time_to_match_mpd': comp.get('time_to_match_mpd', 'N/A'),
                'bitstar_beats_mpd': comp.get('bitstar_beats_mpd', False),
            }
            writer.writerow(row)

        print(f"  Wrote {len(comparison_data)} rows (one per problem)")

    # 4. Export detailed MPD sample-level data (all 64 samples per problem)
    mpd_samples_csv_path = os.path.join(output_dir, "mpd_all_samples.csv")
    print(f"Exporting MPD all-samples data to: {mpd_samples_csv_path}")

    with open(mpd_samples_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'sample_idx',
            'path_length',
            'smoothness',
            'is_collision_free',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_sample_rows = 0
        for result in mpd_results:
            if not result.get('success', False):
                continue

            problem_idx = result['problem_idx']
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            all_path_lengths = result.get('all_path_lengths', [])
            all_smoothness = result.get('all_smoothness', [])
            collision_free_mask = result.get('collision_free_mask', [])

            # Write each sample as a row
            for sample_idx in range(len(all_path_lengths)):
                row = {
                    'problem_idx': problem_idx,
                    'sample_idx': sample_idx,
                    'path_length': all_path_lengths[sample_idx],
                    'smoothness': all_smoothness[sample_idx],
                    'is_collision_free': collision_free_mask[sample_idx] if sample_idx < len(collision_free_mask) else False,
                    'start_state': start_state_str,
                    'goal_state': goal_state_str,
                }
                writer.writerow(row)
                total_sample_rows += 1

        print(f"  Wrote {total_sample_rows} rows (all samples from all problems)")

    print(f"\n{'='*80}")
    print("CSV EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"BIT* time-series: {bitstar_csv_path}")
    print(f"MPD per-problem:  {mpd_csv_path}")
    print(f"Comparison:       {comparison_csv_path}")
    print(f"MPD all samples:  {mpd_samples_csv_path}")
    print(f"{'='*80}\n")

    return {
        'bitstar_timeseries': bitstar_csv_path,
        'mpd_results': mpd_csv_path,
        'comparison_summary': comparison_csv_path,
        'mpd_all_samples': mpd_samples_csv_path,
    }


def print_summary(aggregated):
    """Print comprehensive summary of all results."""

    bitstar_stats = aggregated.get('bitstar_stats', {})
    mpd_stats = aggregated.get('mpd_stats', {})
    comparison_stats = aggregated.get('comparison_stats', {})

    print(f"\n{'='*80}")
    print("COMPLETE COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Metric':<50} {'MPD':<20} {'BIT*':<20}")
    print("-"*90)

    # Success rate
    if mpd_stats and bitstar_stats:
        mpd_success_str = f"{mpd_stats['n_success']}/{mpd_stats['n_problems']} ({mpd_stats['success_rate']*100:.1f}%)"
        bitstar_success_str = f"{bitstar_stats['n_success']}/{bitstar_stats['n_problems']} ({bitstar_stats['success_rate']*100:.1f}%)"
        print(f"{'Success rate':<50} {mpd_success_str:<20} {bitstar_success_str:<20}")

        # Path length
        mpd_best = f"{mpd_stats['best_collision_free_path_length_mean']:.3f} ± {mpd_stats['best_collision_free_path_length_std']:.3f}"
        bitstar_path = f"{bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}"
        print(f"{'Best path length (mean ± std)':<50} {mpd_best:<20} {bitstar_path:<20}")

        # Time
        mpd_time = f"{mpd_stats['inference_time_mean']:.3f} ± {mpd_stats['inference_time_std']:.3f}"
        bitstar_time = f"{bitstar_stats['planning_time_mean']:.3f} ± {bitstar_stats['planning_time_std']:.3f}"
        print(f"{'Planning time (sec, mean ± std)':<50} {mpd_time:<20} {bitstar_time:<20}")

        # Smoothness
        mpd_smooth = f"{mpd_stats['smoothness_mean']:.3f} ± {mpd_stats['smoothness_std']:.3f}"
        bitstar_smooth = f"{bitstar_stats['smoothness_mean']:.3f} ± {bitstar_stats['smoothness_std']:.3f}"
        print(f"{'Smoothness (mean ± std)':<50} {mpd_smooth:<20} {bitstar_smooth:<20}")

        # MPD-specific metrics
        print(f"\n{'MPD-Specific Metrics':<50}")
        print("-"*90)
        collision_rate = f"{mpd_stats['collision_rate_mean']*100:.1f}% ± {mpd_stats['collision_rate_std']*100:.1f}%"
        print(f"{'Collision rate':<50} {collision_rate:<20}")

        mean_cf = f"{mpd_stats['mean_collision_free_path_length_mean']:.3f} ± {mpd_stats['mean_collision_free_path_length_std']:.3f}"
        print(f"{'Mean collision-free path length':<50} {mean_cf:<20}")

        # BIT*-specific metrics
        print(f"\n{'BIT*-Specific Metrics':<50}")
        print("-"*90)

        if 'time_to_first_solution_mean' in bitstar_stats:
            first_time = f"{bitstar_stats['time_to_first_solution_mean']:.3f} ± {bitstar_stats['time_to_first_solution_std']:.3f}"
            print(f"{'Time to first solution (sec)':<50} {first_time:<20}")

        if 'tree_size_mean' in bitstar_stats:
            tree_size = f"{bitstar_stats['tree_size_mean']:.0f} ± {bitstar_stats['tree_size_std']:.0f}"
            print(f"{'Tree size (vertices)':<50} {tree_size:<20}")

        jerk = f"{bitstar_stats['mean_jerk_mean']:.4f} ± {bitstar_stats['mean_jerk_std']:.4f}"
        print(f"{'Mean jerk':<50} {jerk:<20}")

        # Comparison metrics
        if comparison_stats:
            print(f"\n{'Comparison Metrics':<50}")
            print("-"*90)

            both_success = f"{comparison_stats['n_both_success']}"
            print(f"{'Problems where both succeeded':<50} {both_success:<20}")

            beats = f"{comparison_stats['n_bitstar_beats_mpd']} ({comparison_stats['bitstar_beats_mpd_rate']*100:.1f}%)"
            print(f"{'Problems where BIT* beats/matches MPD':<50} {beats:<20}")

            if 'time_to_match_mpd_mean' in comparison_stats:
                match_time = f"{comparison_stats['time_to_match_mpd_mean']:.3f} ± {comparison_stats['time_to_match_mpd_std']:.3f}"
                print(f"{'Time for BIT* to match MPD (sec)':<50} {match_time:<20}")

    print("="*80 + "\n")


def main(
    n_problems=10,
    bitstar_time=600.0,
    tracking_interval=1.0,
    mpd_n_samples=64,
    mpd_n_steps=100,
    output_dir="multi_run_results",
    cfg_path='./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml',
    device='cuda:0',
    seed=42,
):
    """
    Main function to run complete comparison workflow.
    """
    print("\n" + "="*80)
    print("COMPLETE BIT* vs MPD COMPARISON")
    print("="*80)
    print(f"Number of problems: {n_problems}")
    print(f"BIT* time limit: {bitstar_time} sec")
    print(f"BIT* tracking interval: {tracking_interval} sec")
    print(f"MPD samples per problem: {mpd_n_samples}")
    print(f"MPD diffusion steps: {mpd_n_steps}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print("="*80)

    fix_random_seed(seed)

    # Setup environment and robot
    device = get_torch_device(device)
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

    # Step 1: Run BIT* on all problems
    print("\n" + "="*80)
    print("STEP 1/3: Running BIT* on all problems")
    print("="*80)
    bitstar_results = run_bitstar_on_problems(
        problems,
        env,
        robot,
        allowed_time=bitstar_time,
        tracking_interval=tracking_interval,
        output_dir=output_dir,
        tensor_args=tensor_args,
    )

    # Step 2: Run MPD on all problems
    print("\n" + "="*80)
    print("STEP 2/3: Running MPD on all problems")
    print("="*80)

    # Load MPD model
    model, dataset, planning_task, args = load_mpd_model(cfg_path=cfg_path, device=device)

    mpd_results = run_mpd_on_problems(
        problems,
        env,
        robot,
        model,
        dataset,
        planning_task,
        args,
        n_samples=mpd_n_samples,
        n_steps=mpd_n_steps,
        output_dir=output_dir,
        tensor_args=tensor_args,
    )

    # Step 3: Compute comparison metrics and aggregate
    print("\n" + "="*80)
    print("STEP 3/4: Computing comparison metrics and aggregating results")
    print("="*80)

    comparison_data = compute_comparison_metrics(bitstar_results, mpd_results)
    aggregated = aggregate_all_results(bitstar_results, mpd_results, comparison_data, output_dir)

    # Step 4: Export to CSV
    print("\n" + "="*80)
    print("STEP 4/4: Exporting data to CSV files")
    print("="*80)

    csv_files = export_to_csv(bitstar_results, mpd_results, comparison_data, output_dir)

    # Print summary
    print_summary(aggregated)

    print(f"\n{'='*80}")
    print("COMPLETE RESULTS SAVED")
    print(f"{'='*80}")
    print(f"Problem set: {problem_file}")
    print(f"BIT* results: {output_dir}/bitstar_result_*.pt")
    print(f"MPD results: {output_dir}/mpd_results/mpd_result_*.pt")
    print(f"Aggregated results: {output_dir}/complete_aggregated_results.pt")
    print(f"Statistics: {output_dir}/complete_statistics.yaml")
    print(f"Comparison data: {output_dir}/comparison_data.json")
    print(f"\nCSV Files:")
    print(f"  BIT* time-series: {csv_files['bitstar_timeseries']}")
    print(f"  MPD per-problem:  {csv_files['mpd_results']}")
    print(f"  Comparison:       {csv_files['comparison_summary']}")
    print(f"  MPD all samples:  {csv_files['mpd_all_samples']}")
    print(f"{'='*80}\n")

    print("You can now analyze the results using:")
    print("  - Python: torch.load('multi_run_results/complete_aggregated_results.pt')")
    print("  - YAML: cat multi_run_results/complete_statistics.yaml")
    print("  - JSON: cat multi_run_results/comparison_data.json")
    print("  - CSV: Open in Excel, pandas, or any spreadsheet software")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete BIT* vs MPD comparison workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete comparison on 10 problems
  python run_complete_comparison.py --n-problems 10

  # Run with custom settings
  python run_complete_comparison.py --n-problems 5 --bitstar-time 300 --mpd-samples 128

  # Continue from existing problem set
  python run_complete_comparison.py --n-problems 10 --output-dir my_results
        """
    )

    parser.add_argument("--n-problems", type=int, default=10,
                       help="Number of problems to solve (default: 10)")
    parser.add_argument("--bitstar-time", type=float, default=600.0,
                       help="Time limit for each BIT* run in seconds (default: 600)")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="BIT* tracking interval in seconds (default: 1.0)")
    parser.add_argument("--mpd-samples", type=int, default=64,
                       help="Number of MPD trajectory samples per problem (default: 64)")
    parser.add_argument("--mpd-steps", type=int, default=100,
                       help="Number of MPD diffusion steps (default: 100)")
    parser.add_argument("--output-dir", default="multi_run_results",
                       help="Output directory for results (default: multi_run_results)")
    parser.add_argument("--cfg", default="./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml",
                       help="Path to MPD config file")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    main(
        n_problems=args.n_problems,
        bitstar_time=args.bitstar_time,
        tracking_interval=args.interval,
        mpd_n_samples=args.mpd_samples,
        mpd_n_steps=args.mpd_steps,
        output_dir=args.output_dir,
        cfg_path=args.cfg,
        device=args.device,
        seed=args.seed,
    )
