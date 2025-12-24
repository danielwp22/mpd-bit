"""
Run BIT* against MPD on the same problem and compare timing.
Shows when BIT* finds first solution and when it beats MPD quality.
"""
import os
import sys
from pathlib import Path

# IMPORTANT: Import isaacgym FIRST before torch
import isaacgym

import torch
import numpy as np
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed

# Import BIT* implementation
from bitstar_minimal_template import MinimalBITStarBaseline


def load_mpd_results(results_file="logs/2/results_single_plan-000.pt"):
    """Load MPD diffusion model results."""
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        print("Run the diffusion model first: python inference.py")
        return None

    results = torch.load(results_file, map_location='cpu')
    return results


def run_comparison(mpd_results_file="logs/2/results_single_plan-000.pt",
                   allowed_time=30.0, seed=42):
    """
    Run BIT* on the same problem as MPD and compare timing.

    Args:
        mpd_results_file: Path to MPD results
        allowed_time: Time limit for BIT*
        seed: Random seed
    """
    fix_random_seed(seed)

    print("\n" + "="*80)
    print("BIT* vs MPD COMPARISON")
    print("="*80 + "\n")

    # Load MPD results
    print("Loading MPD results...")
    mpd_results = load_mpd_results(mpd_results_file)
    if mpd_results is None:
        return

    # Extract start, goal, and metrics
    start_state = to_numpy(mpd_results['q_pos_start'])
    goal_state = to_numpy(mpd_results['q_pos_goal'])
    mpd_path_length = float(mpd_results['metrics']['trajs_best']['path_length'])
    mpd_smoothness = float(mpd_results['metrics']['trajs_best'].get('smoothness', 0))
    mpd_time = mpd_results.get('t_inference_total', 0)

    print(f"MPD path length: {mpd_path_length:.3f}")
    print(f"MPD smoothness: {mpd_smoothness:.3f}")
    print(f"MPD inference time: {mpd_time:.3f} sec")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Setup environment and robot
    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Initialize BIT* planner
    print(f"\n{'='*80}")
    print("Running BIT* planner...")
    print(f"{'='*80}\n")

    baseline = MinimalBITStarBaseline(
        robot=robot,
        allowed_planning_time=allowed_time,
        interpolate_num=128,
        device="cuda:0",
        batch_size=100,
    )
    baseline.set_obstacles(env)

    # Plan with BIT* - pass target path length for comparison
    result = baseline.plan(
        start_state,
        goal_state,
        debug=True,
        target_path_length=mpd_path_length
    )

    # Print comparison
    print("\n" + "="*80)
    print("TIMING COMPARISON")
    print("="*80)

    if result['success']:
        print(f"\n{'Metric':<40} {'MPD':<20} {'BIT*':<20}")
        print("-"*80)
        print(f"{'Path length':<40} {mpd_path_length:.3f}{'':<16} {result['path_length']:.3f}")
        print(f"{'Smoothness (sum of accel norms)':<40} {mpd_smoothness:.3f}{'':<16} {result['smoothness']:.3f}")
        print(f"{'Mean jerk (avg jerk magnitude)':<40} {'N/A':<20} {result.get('mean_jerk', 'N/A')}")
        print(f"{'Total time (sec)':<40} {mpd_time:.3f}{'':<16} {result['planning_time']:.3f}")

        if result['time_to_first_solution'] is not None:
            print(f"\n{'BIT* Anytime Performance':<40}")
            print("-"*80)
            print(f"{'Time to FIRST solution':<40} {'':<20} {result['time_to_first_solution']:.3f} sec")
            print(f"{'First solution cost':<40} {'':<20} {result['first_solution_cost']:.3f}")

            if result['time_to_target_quality'] is not None:
                print(f"{'Time to BEAT MPD quality':<40} {'':<20} {result['time_to_target_quality']:.3f} sec")
                speedup = mpd_time / result['time_to_target_quality']
                print(f"{'Speedup vs MPD':<40} {'':<20} {speedup:.2f}x")
            else:
                print(f"{'Time to BEAT MPD quality':<40} {'':<20} Did not reach")

            # Path improvement
            improvement = (result['first_solution_cost'] - result['path_length']) / result['first_solution_cost'] * 100
            print(f"{'Path improvement (first → final)':<40} {'':<20} {improvement:.1f}%")

        # Quality comparison
        if result['path_length'] < mpd_path_length:
            improvement = (mpd_path_length - result['path_length']) / mpd_path_length * 100
            print(f"\n✓ BIT* found {improvement:.1f}% shorter path than MPD")
        elif result['path_length'] > mpd_path_length:
            diff = (result['path_length'] - mpd_path_length) / mpd_path_length * 100
            print(f"\n✗ MPD found {diff:.1f}% shorter path than BIT*")
        else:
            print(f"\n= Both methods found similar path lengths")

    else:
        print("\nBIT* failed to find a solution.")
        print(f"MPD path length: {mpd_path_length:.3f}")
        print(f"MPD time: {mpd_time:.3f} sec")

    print("="*80 + "\n")

    baseline.terminate()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run BIT* vs MPD comparison")
    parser.add_argument("--mpd-results", default="logs/2/results_single_plan-000.pt",
                       help="Path to MPD results file")
    parser.add_argument("--time", type=float, default=30.0,
                       help="Time limit for BIT* (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    run_comparison(
        mpd_results_file=args.mpd_results,
        allowed_time=args.time,
        seed=args.seed
    )

    print("\nUsage examples:")
    print("  python run_bitstar_vs_mpd.py")
    print("  python run_bitstar_vs_mpd.py --mpd-results logs/2/results_single_plan-000.pt --time 60")
