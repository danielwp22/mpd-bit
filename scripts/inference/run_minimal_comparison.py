"""
Minimal comparison workflow: run the minimal BIT* implementation and fast MPD inference
on the same problem set and save comparable metrics.
"""
import os
import time
import json
import csv
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Import isaacgym first
import isaacgym

import torch
from dotmap import DotMap
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness

from bitstar_minimal_template import MinimalBITStarBaseline
from mpd.utils.loaders import get_planning_task_and_dataset, get_model, load_params_from_yaml


def generate_problem_set(robot, env, n_problems, seed=42, output_dir="minimal_results"):
    """Generate random collision-free start/goal pairs and save them."""
    fix_random_seed(seed)

    print(f"\n{'='*80}")
    print(f"Generating {n_problems} random start/goal pairs")
    print(f"{'='*80}\n")

    problems = []

    for i in range(n_problems):
        print(f"Problem {i+1}/{n_problems}...", end=" ")
        start_state = robot.random_q(n_samples=1)[0]
        goal_state = robot.random_q(n_samples=1)[0]

        problems.append({
            "problem_idx": i,
            "start_state": to_numpy(start_state),
            "goal_state": to_numpy(goal_state),
        })
        print("✓")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(problems, os.path.join(output_dir, "problem_set.pt"))
    print(f"\nSaved problems to: {output_dir}/problem_set.pt\n")
    return problems


def load_mpd_model(cfg_path="./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml", device="cuda:0"):
    """Load MPD model and dataset."""
    print(f"Loading MPD model from config: {cfg_path}")
    args_inference = DotMap(load_params_from_yaml(cfg_path))

    if args_inference.model_selection == "bspline":
        args_inference.model_dir = args_inference.model_dir_ddpm_bspline
    elif args_inference.model_selection == "waypoints":
        args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
    else:
        raise NotImplementedError(f"Unknown model selection: {args_inference.model_selection}")

    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))
    args_train.update(
        **args_inference,
        reload_data=False,
        load_indices=False,
        tensor_args={"device": device, "dtype": torch.float32},
    )

    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)
    dataset = train_subset.dataset

    model_path = os.path.join(
        args_inference.model_dir,
        "checkpoints",
        f'{"ema_" if args_train.get("use_ema", False) else ""}model_current.pth',
    )
    print(f"Loading model from: {model_path}")
    model = get_model(
        checkpoint_path=model_path,
        freeze_loaded_model=True,
        tensor_args={"device": device, "dtype": torch.float32},
    )
    print("Model loaded.")

    return model, dataset, planning_task, args_train


def build_sampling_kwargs(args, model, n_steps):
    """Build sampling kwargs so we follow the fast inference path (DDIM by default)."""
    diffusion_method = getattr(args, "diffusion_sampling_method", "ddim")
    diffusion_cfg = args.get(diffusion_method, {}) if hasattr(args, "get") else {}
    diffusion_cfg = dict(diffusion_cfg)

    if diffusion_method == "ddim":
        diffusion_cfg["ddim_sampling_timesteps"] = n_steps
        frac = diffusion_cfg.get("t_start_guide_steps_fraction", 0.0)
        diffusion_cfg["t_start_guide"] = int(np.ceil(frac * diffusion_cfg["ddim_sampling_timesteps"]))
    elif diffusion_method == "ddpm":
        frac = diffusion_cfg.get("t_start_guide_steps_fraction", 0.0)
        diffusion_cfg["t_start_guide"] = int(np.ceil(frac * model.n_diffusion_steps))

    sampling_kwargs = {"method": diffusion_method, **diffusion_cfg}
    print(f"Using diffusion sampling: {diffusion_method} with {sampling_kwargs.get('ddim_sampling_timesteps', getattr(model, 'n_diffusion_steps', 'n/a'))} steps")
    return sampling_kwargs


def check_trajectory_collision_free(trajectory, robot, env, n_interpolate=10):
    """Check if a trajectory is collision-free via dense interpolation."""
    horizon = len(trajectory)
    trajectory_np = to_numpy(trajectory)

    for i in range(horizon - 1):
        q1 = trajectory_np[i]
        q2 = trajectory_np[i + 1]
        for alpha in np.linspace(0, 1, n_interpolate):
            q = q1 * (1 - alpha) + q2 * alpha
            q_torch = to_torch(q, device=trajectory.device, dtype=trajectory.dtype).unsqueeze(0)
            x_pos = robot.fk_map_collision(q_torch)
            radii = robot.link_collision_spheres_radii
            sdf_vals = env.compute_sdf(x_pos.reshape(-1, 3))
            if (sdf_vals < radii).any():
                return False
    return True


def run_mpd_fast_on_problems(
    problems,
    env,
    robot,
    model,
    dataset,
    planning_task,
    args,
    n_samples=64,
    n_steps=15,
    output_dir="minimal_results",
    tensor_args=None,
):
    """Run MPD inference with fast sampling on all problems."""
    print(f"\n{'='*80}")
    print(f"Running MPD (fast) on {len(problems)} problems")
    print(f"Samples per problem: {n_samples}")
    print(f"Diffusion steps: {n_steps}")
    print(f"{'='*80}\n")

    mpd_output_dir = os.path.join(output_dir, "mpd_results")
    os.makedirs(mpd_output_dir, exist_ok=True)

    sampling_kwargs = build_sampling_kwargs(args, model, n_steps)
    results = []

    for i, problem in enumerate(problems):
        problem_start_wall = time.time()
        print(f"\n{'#'*80}")
        print(f"# MPD on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem["start_state"]
        goal_state = problem["goal_state"]

        start_state_torch = to_torch(start_state, **tensor_args).unsqueeze(0)
        goal_state_torch = to_torch(goal_state, **tensor_args).unsqueeze(0)

        inference_start_time = time.time()

        try:
            with torch.no_grad():
                print(f"[MPD] Building data sample...", flush=True)
                input_data_sample = dataset.create_data_sample_normalized(
                    start_state_torch.squeeze(0),
                    goal_state_torch.squeeze(0),
                )
                print(f"[MPD] Data sample ready. Building context...", flush=True)
                hard_conds = input_data_sample["hard_conds"]
                context_d = dataset.build_context(input_data_sample)
                print(f"[MPD] Context ready. Running diffusion sampling with {n_samples} samples...", flush=True)
                samples = model.run_inference(
                    context_d=context_d,
                    hard_conds=hard_conds,
                    n_samples=n_samples,
                    horizon=dataset.n_learnable_control_points,
                    return_chain=False,
                    **sampling_kwargs,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[MPD] Sampling done in {(time.time() - inference_start_time):.3f}s. Converting to trajectories...", flush=True)

            inference_time = time.time() - inference_start_time

            q_traj_d = planning_task.parametric_trajectory.get_q_trajectory(
                samples,
                start_state_torch.squeeze(0),
                goal_state_torch.squeeze(0),
                get_type=("pos", "vel", "acc"),
                get_time_representation=True,
            )
            q_trajs_pos = q_traj_d["pos"]
            q_trajs_vel = q_traj_d["vel"]
            q_trajs_acc = q_traj_d["acc"]

            path_lengths = compute_path_length(q_trajs_pos, robot)
            smoothness_values = compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)

            best_idx = torch.argmin(path_lengths)
            best_path_length = path_lengths[best_idx].item()
            best_smoothness = smoothness_values[best_idx].item()
            best_traj = q_trajs_pos[best_idx].cpu().numpy()

            path_lengths_np = path_lengths.cpu().numpy()
            smoothness_np = smoothness_values.cpu().numpy()

            collision_free_mask = []
            collision_free_lengths = []
            for j in range(len(q_trajs_pos)):
                traj = q_trajs_pos[j]
                is_free = check_trajectory_collision_free(traj, robot, env)
                collision_free_mask.append(is_free)
                if is_free:
                    collision_free_lengths.append(path_lengths_np[j])
                if (j + 1) % 8 == 0 or j == len(q_trajs_pos) - 1:
                    elapsed = time.time() - inference_start_time
                    print(f"[MPD] Collision check {j+1}/{len(q_trajs_pos)} (elapsed {elapsed:.2f}s)", flush=True)

            n_collision_free = sum(collision_free_mask)
            collision_rate = 1.0 - (n_collision_free / len(samples))
            mean_collision_free_length = float(np.mean(collision_free_lengths)) if collision_free_lengths else float("inf")
            best_collision_free_length = float(np.min(collision_free_lengths)) if collision_free_lengths else float("inf")

            result = {
                "success": True,
                "problem_idx": i,
                "start_state": start_state,
                "goal_state": goal_state,
                "inference_time": inference_time,
                "n_samples": n_samples,
                "n_diffusion_steps": n_steps,
                "path_length": best_path_length,
                "smoothness": best_smoothness,
                "best_trajectory": best_traj,
                "n_collision_free": n_collision_free,
                "n_total_samples": len(samples),
                "collision_rate": collision_rate,
                "collision_free_mask": collision_free_mask,
                "mean_collision_free_path_length": mean_collision_free_length,
                "best_collision_free_path_length": best_collision_free_length,
                "all_path_lengths": path_lengths_np,
                "all_smoothness": smoothness_np,
                "path_length_mean": float(np.mean(path_lengths_np)),
                "path_length_std": float(np.std(path_lengths_np)),
                "smoothness_mean": float(np.mean(smoothness_np)),
                "smoothness_std": float(np.std(smoothness_np)),
            }

            print(f"MPD Results: time={inference_time:.3f}s, best length={best_path_length:.3f}, collision-free {n_collision_free}/{n_samples}")

        except Exception as e:
            print(f"Error during MPD inference on problem {i}: {e}")
            import traceback
            traceback.print_exc()

            inference_time = time.time() - inference_start_time
            result = {
                "success": False,
                "problem_idx": i,
                "start_state": start_state,
                "goal_state": goal_state,
                "inference_time": inference_time,
                "error": str(e),
            }

        torch.save(result, os.path.join(mpd_output_dir, f"mpd_result_{i:03d}.pt"))
        results.append(result)
        print(f"[MPD] Problem {i+1} total wall time: {time.time() - problem_start_wall:.2f}s", flush=True)

    return results


def run_bitstar_minimal_on_problems(
    problems,
    env,
    robot,
    allowed_time=60.0,
    output_dir="minimal_results",
    tensor_args=None,
    target_path_lengths=None,
):
    """Run the minimal BIT* implementation on all problems."""
    print(f"\n{'='*80}")
    print(f"Running minimal BIT* on {len(problems)} problems (time per problem: {allowed_time}s)")
    print(f"{'='*80}\n")

    results = []
    for i, problem in enumerate(problems):
        print(f"\n{'#'*80}")
        print(f"# BIT* on Problem {i+1}/{len(problems)}")
        print(f"{'#'*80}")

        start_state = problem["start_state"]
        goal_state = problem["goal_state"]

        planner = MinimalBITStarBaseline(
            robot=robot,
            allowed_planning_time=allowed_time,
            interpolate_num=128,
            device=str(tensor_args["device"]),
            batch_size=100,
        )
        planner.set_obstacles(env)

        target_len = None
        if target_path_lengths and i < len(target_path_lengths):
            target_len = target_path_lengths[i]

        result = planner.plan(
            start_state,
            goal_state,
            debug=True,
            target_path_length=target_len,
        )

        result.update({
            "problem_idx": i,
            "start_state": start_state,
            "goal_state": goal_state,
            "planning_time": result.get("planning_time", allowed_time),
        })

        torch.save(result, os.path.join(output_dir, f"bitstar_minimal_result_{i:03d}.pt"))
        print(f"Saved BIT* result to: {output_dir}/bitstar_minimal_result_{i:03d}.pt")
        results.append(result)

    return results


def compute_comparison_metrics(bitstar_results, mpd_results):
    """Compare BIT* and MPD outcomes."""
    comparison_data = []
    for br in bitstar_results:
        idx = br["problem_idx"]
        if idx >= len(mpd_results):
            continue
        mr = mpd_results[idx]

        mpd_target = mr.get("best_collision_free_path_length", mr.get("path_length", float("inf")))

        if not br.get("success", False) or not mr.get("success", False):
            comparison_data.append({
                "problem_idx": idx,
                "bitstar_success": br.get("success", False),
                "mpd_success": mr.get("success", False),
                "mpd_target_length": mpd_target,
                "bitstar_final_length": br.get("path_length", float("inf")),
                "bitstar_beats_mpd": False,
            })
            continue

        bitstar_final = br["path_length"]
        comparison_data.append({
            "problem_idx": idx,
            "bitstar_success": True,
            "mpd_success": True,
            "mpd_target_length": mpd_target,
            "bitstar_final_length": bitstar_final,
            "time_to_first_solution": br.get("time_to_first_solution"),
            "time_to_target_quality": br.get("time_to_target_quality"),
            "bitstar_beats_mpd": bitstar_final <= mpd_target,
        })
    return comparison_data


def aggregate_all_results(bitstar_results, mpd_results, comparison_data, output_dir="minimal_results"):
    """Aggregate statistics for both planners."""
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}\n")

    bitstar_success = [r for r in bitstar_results if r.get("success", False)]
    mpd_success = [r for r in mpd_results if r.get("success", False)]

    bitstar_stats = {}
    if bitstar_success:
        path_lengths = np.array([r["path_length"] for r in bitstar_success])
        smoothness = np.array([r["smoothness"] for r in bitstar_success])
        mean_jerk = np.array([r["mean_jerk"] for r in bitstar_success])
        times = np.array([r["planning_time"] for r in bitstar_success])
        first_times = np.array([r["time_to_first_solution"] for r in bitstar_success if r.get("time_to_first_solution")])

        bitstar_stats = {
            "n_problems": len(bitstar_results),
            "n_success": len(bitstar_success),
            "success_rate": len(bitstar_success) / len(bitstar_results),
            "path_length_mean": float(np.mean(path_lengths)),
            "path_length_std": float(np.std(path_lengths)),
            "smoothness_mean": float(np.mean(smoothness)),
            "smoothness_std": float(np.std(smoothness)),
            "mean_jerk_mean": float(np.mean(mean_jerk)),
            "mean_jerk_std": float(np.std(mean_jerk)),
            "planning_time_mean": float(np.mean(times)),
            "planning_time_std": float(np.std(times)),
        }
        if len(first_times) > 0:
            bitstar_stats["time_to_first_solution_mean"] = float(np.mean(first_times))
            bitstar_stats["time_to_first_solution_std"] = float(np.std(first_times))

    mpd_stats = {}
    if mpd_success:
        collision_rates = np.array([r["collision_rate"] for r in mpd_success])
        smoothness = np.array([r["smoothness"] for r in mpd_success])
        times = np.array([r["inference_time"] for r in mpd_success])
        best_cf = np.array([r["best_collision_free_path_length"] for r in mpd_success])
        mean_cf = np.array([r["mean_collision_free_path_length"] for r in mpd_success])

        mpd_stats = {
            "n_problems": len(mpd_results),
            "n_success": len(mpd_success),
            "success_rate": len(mpd_success) / len(mpd_results),
            "collision_rate_mean": float(np.mean(collision_rates)),
            "collision_rate_std": float(np.std(collision_rates)),
            "smoothness_mean": float(np.mean(smoothness)),
            "smoothness_std": float(np.std(smoothness)),
            "inference_time_mean": float(np.mean(times)),
            "inference_time_std": float(np.std(times)),
            "best_collision_free_path_length_mean": float(np.mean(best_cf)),
            "best_collision_free_path_length_std": float(np.std(best_cf)),
            "mean_collision_free_path_length_mean": float(np.mean(mean_cf)),
            "mean_collision_free_path_length_std": float(np.std(mean_cf)),
        }

    n_bitstar_beats = sum(1 for c in comparison_data if c.get("bitstar_beats_mpd"))
    comparison_stats = {
        "n_both_success": sum(1 for c in comparison_data if c.get("bitstar_success") and c.get("mpd_success")),
        "n_bitstar_beats_mpd": n_bitstar_beats,
        "bitstar_beats_mpd_rate": n_bitstar_beats / len(comparison_data) if comparison_data else 0,
    }

    aggregated = {
        "bitstar_results": bitstar_results,
        "mpd_results": mpd_results,
        "comparison_data": comparison_data,
        "bitstar_stats": bitstar_stats,
        "mpd_stats": mpd_stats,
        "comparison_stats": comparison_stats,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(aggregated, os.path.join(output_dir, "minimal_aggregated_results.pt"))
    with open(os.path.join(output_dir, "minimal_statistics.yaml"), "w") as f:
        yaml.dump({"bitstar": bitstar_stats, "mpd": mpd_stats, "comparison": comparison_stats}, f)
    with open(os.path.join(output_dir, "minimal_comparison_data.json"), "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    return aggregated


def export_to_csv(bitstar_results, mpd_results, comparison_data, output_dir="minimal_results"):
    """Export per-problem results and comparisons to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    bitstar_csv = os.path.join(output_dir, "bitstar_results.csv")
    with open(bitstar_csv, "w", newline="") as csvfile:
        fieldnames = [
            "problem_idx",
            "success",
            "planning_time",
            "path_length",
            "smoothness",
            "mean_jerk",
            "time_to_first_solution",
            "time_to_target_quality",
            "first_solution_cost",
            "start_state",
            "goal_state",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in bitstar_results:
            writer.writerow({
                "problem_idx": r["problem_idx"],
                "success": r.get("success", False),
                "planning_time": r.get("planning_time", 0),
                "path_length": r.get("path_length", float("inf")),
                "smoothness": r.get("smoothness", float("inf")),
                "mean_jerk": r.get("mean_jerk", float("inf")),
                "time_to_first_solution": r.get("time_to_first_solution"),
                "time_to_target_quality": r.get("time_to_target_quality"),
                "first_solution_cost": r.get("first_solution_cost"),
                "start_state": ",".join(map(str, r["start_state"])),
                "goal_state": ",".join(map(str, r["goal_state"])),
            })

    # BIT* time-series data (interval metrics)
    bitstar_timeseries_csv = os.path.join(output_dir, "bitstar_timeseries.csv")
    with open(bitstar_timeseries_csv, "w", newline="") as csvfile:
        fieldnames = [
            "problem_idx",
            "time",
            "iteration",
            "path_length",
            "smoothness",
            "mean_jerk",
            "num_vertices",
            "num_samples",
            "num_batches",
            "has_solution",
            "start_state",
            "goal_state",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in bitstar_results:
            start_state_str = ",".join(map(str, r["start_state"]))
            goal_state_str = ",".join(map(str, r["goal_state"]))
            for m in r.get("interval_metrics", []):
                writer.writerow({
                    "problem_idx": r["problem_idx"],
                    "time": m.get("time", ""),
                    "iteration": m.get("iteration", ""),
                    "path_length": m.get("path_length", ""),
                    "smoothness": m.get("smoothness", ""),
                    "mean_jerk": m.get("mean_jerk", ""),
                    "num_vertices": m.get("num_vertices", ""),
                    "num_samples": m.get("num_samples", ""),
                    "num_batches": m.get("num_batches", ""),
                    "has_solution": m.get("has_solution", False),
                    "start_state": start_state_str,
                    "goal_state": goal_state_str,
                })

    mpd_csv = os.path.join(output_dir, "mpd_results.csv")
    with open(mpd_csv, "w", newline="") as csvfile:
        fieldnames = [
            "problem_idx",
            "success",
            "inference_time",
            "n_samples",
            "n_diffusion_steps",
            "best_path_length",
            "best_smoothness",
            "n_collision_free",
            "n_total_samples",
            "collision_rate",
            "mean_collision_free_path_length",
            "best_collision_free_path_length",
            "path_length_mean",
            "path_length_std",
            "smoothness_mean",
            "smoothness_std",
            "start_state",
            "goal_state",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in mpd_results:
            writer.writerow({
                "problem_idx": r["problem_idx"],
                "success": r.get("success", False),
                "inference_time": r.get("inference_time", 0),
                "n_samples": r.get("n_samples", 0),
                "n_diffusion_steps": r.get("n_diffusion_steps", 0),
                "best_path_length": r.get("path_length", float("inf")),
                "best_smoothness": r.get("smoothness", float("inf")),
                "n_collision_free": r.get("n_collision_free", 0),
                "n_total_samples": r.get("n_total_samples", 0),
                "collision_rate": r.get("collision_rate", 1.0),
                "mean_collision_free_path_length": r.get("mean_collision_free_path_length", float("inf")),
                "best_collision_free_path_length": r.get("best_collision_free_path_length", float("inf")),
                "path_length_mean": r.get("path_length_mean", float("inf")),
                "path_length_std": r.get("path_length_std", 0),
                "smoothness_mean": r.get("smoothness_mean", float("inf")),
                "smoothness_std": r.get("smoothness_std", 0),
                "start_state": ",".join(map(str, r["start_state"])),
                "goal_state": ",".join(map(str, r["goal_state"])),
            })

    comparison_csv = os.path.join(output_dir, "comparison_summary.csv")
    with open(comparison_csv, "w", newline="") as csvfile:
        fieldnames = [
            "problem_idx",
            "bitstar_success",
            "mpd_success",
            "mpd_target_length",
            "bitstar_final_length",
            "time_to_first_solution",
            "time_to_target_quality",
            "bitstar_beats_mpd",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for c in comparison_data:
            writer.writerow({
                "problem_idx": c["problem_idx"],
                "bitstar_success": c.get("bitstar_success", False),
                "mpd_success": c.get("mpd_success", False),
                "mpd_target_length": c.get("mpd_target_length", float("inf")),
                "bitstar_final_length": c.get("bitstar_final_length", float("inf")),
                "time_to_first_solution": c.get("time_to_first_solution"),
                "time_to_target_quality": c.get("time_to_target_quality"),
                "bitstar_beats_mpd": c.get("bitstar_beats_mpd", False),
            })

    mpd_samples_csv = os.path.join(output_dir, "mpd_all_samples.csv")
    with open(mpd_samples_csv, "w", newline="") as csvfile:
        fieldnames = [
            "problem_idx",
            "sample_idx",
            "path_length",
            "smoothness",
            "is_collision_free",
            "start_state",
            "goal_state",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in mpd_results:
            if not r.get("success", False):
                continue
            for sample_idx, (pl, sm, cf) in enumerate(zip(
                r.get("all_path_lengths", []),
                r.get("all_smoothness", []),
                r.get("collision_free_mask", []),
            )):
                writer.writerow({
                    "problem_idx": r["problem_idx"],
                    "sample_idx": sample_idx,
                    "path_length": pl,
                    "smoothness": sm,
                    "is_collision_free": cf,
                    "start_state": ",".join(map(str, r["start_state"])),
                    "goal_state": ",".join(map(str, r["goal_state"])),
                })

    print(f"\nCSV files saved to {output_dir}:")
    print(f"  BIT*: {bitstar_csv}")
    print(f"  BIT* time-series: {bitstar_timeseries_csv}")
    print(f"  MPD:  {mpd_csv}")
    print(f"  Comparison: {comparison_csv}")
    print(f"  MPD samples: {mpd_samples_csv}")

    return {
        "bitstar_results": bitstar_csv,
        "bitstar_timeseries": bitstar_timeseries_csv,
        "mpd_results": mpd_csv,
        "comparison_summary": comparison_csv,
        "mpd_all_samples": mpd_samples_csv,
    }


def main(
    n_problems=5,
    bitstar_time=120.0,
    mpd_n_samples=32,
    mpd_n_steps=15,
    output_dir="minimal_results",
    cfg_path="./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml",
    device="cuda:0",
    seed=42,
):
    print("\n" + "="*80)
    print("MINIMAL BIT* vs FAST MPD")
    print("="*80)
    print(f"Problems: {n_problems}")
    print(f"BIT* time limit: {bitstar_time} sec")
    print(f"MPD samples: {mpd_n_samples}")
    print(f"MPD diffusion steps: {mpd_n_steps}")
    print(f"Output dir: {output_dir}")
    print("="*80)

    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    problem_file = os.path.join(output_dir, "problem_set.pt")
    if os.path.exists(problem_file):
        print(f"Loading existing problems from: {problem_file}")
        problems = torch.load(problem_file)
    else:
        problems = generate_problem_set(robot, env, n_problems, seed=seed, output_dir=output_dir)

    model, dataset, planning_task, args = load_mpd_model(cfg_path=cfg_path, device=device)

    # Run MPD first to set target lengths for BIT* anytime comparison
    mpd_results = run_mpd_fast_on_problems(
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

    target_lengths = [
        r.get("best_collision_free_path_length", r.get("path_length", float("inf")))
        if r.get("success", False) else None
        for r in mpd_results
    ]

    bitstar_results = run_bitstar_minimal_on_problems(
        problems,
        env,
        robot,
        allowed_time=bitstar_time,
        output_dir=output_dir,
        tensor_args=tensor_args,
        target_path_lengths=target_lengths,
    )

    comparison_data = compute_comparison_metrics(bitstar_results, mpd_results)
    aggregated = aggregate_all_results(bitstar_results, mpd_results, comparison_data, output_dir)
    csv_files = export_to_csv(bitstar_results, mpd_results, comparison_data, output_dir)

    print("\n" + "="*80)
    print("DONE. Key outputs:")
    print("="*80)
    print(f"Problems: {problem_file}")
    print(f"BIT*: {output_dir}/bitstar_minimal_result_*.pt")
    print(f"MPD: {output_dir}/mpd_results/mpd_result_*.pt")
    print(f"Aggregated: {output_dir}/minimal_aggregated_results.pt")
    print(f"Stats: {output_dir}/minimal_statistics.yaml")
    print(f"Comparison: {output_dir}/minimal_comparison_data.json")
    print(f"CSVs: {csv_files}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal BIT* vs fast MPD comparison")
    parser.add_argument("--n-problems", type=int, default=5, help="Number of problems")
    parser.add_argument("--bitstar-time", type=float, default=120.0, help="BIT* time per problem (sec)")
    parser.add_argument("--mpd-samples", type=int, default=32, help="MPD samples per problem")
    parser.add_argument("--mpd-steps", type=int, default=15, help="MPD diffusion steps (DDIM)")
    parser.add_argument("--output-dir", default="minimal_results", help="Output directory")
    parser.add_argument("--cfg", default="./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml", help="MPD config path")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(
        n_problems=args.n_problems,
        bitstar_time=args.bitstar_time,
        mpd_n_samples=args.mpd_samples,
        mpd_n_steps=args.mpd_steps,
        output_dir=args.output_dir,
        cfg_path=args.cfg,
        device=args.device,
        seed=args.seed,
    )
