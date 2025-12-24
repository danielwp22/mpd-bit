"""
Quick test script to verify MPD inference works after the fix.
Tests MPD on a single problem from the existing problem set.
"""
import os
import sys
import time
import numpy as np

# IMPORTANT: Import isaacgym FIRST before torch
import isaacgym

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness

# Import MPD components (reuse functions from run_complete_comparison.py)
from run_complete_comparison import load_mpd_model, check_trajectory_collision_free

def test_mpd_single_problem():
    """Test MPD on a single problem."""
    print("\n" + "="*80)
    print("TESTING MPD INFERENCE ON SINGLE PROBLEM")
    print("="*80 + "\n")

    # Setup
    device = get_torch_device('cuda:0')
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Load existing problem set
    problem_file = "multi_run_results/problem_set.pt"
    if not os.path.exists(problem_file):
        print(f"ERROR: Problem file not found: {problem_file}")
        print("Please run the full script first to generate problems, or create one manually.")
        return

    print(f"Loading problems from: {problem_file}")
    problems = torch.load(problem_file)
    print(f"Loaded {len(problems)} problems\n")

    # Load MPD model
    cfg_path = './cfgs/config_EnvSpheres3D-RobotPanda_00.yaml'
    model, dataset, planning_task, args = load_mpd_model(cfg_path=cfg_path, device=device)

    # Test on problem 2 (the one that was failing)
    test_idx = min(2, len(problems) - 1)
    problem = problems[test_idx]

    print(f"\n{'='*80}")
    print(f"Testing MPD on Problem {test_idx}")
    print(f"{'='*80}\n")

    start_state = problem['start_state']
    goal_state = problem['goal_state']

    # Convert to torch
    start_state_torch = to_torch(start_state, **tensor_args).unsqueeze(0)  # (1, dof)
    goal_state_torch = to_torch(goal_state, **tensor_args).unsqueeze(0)    # (1, dof)

    n_samples = 64

    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")
    print(f"Generating {n_samples} trajectory samples...\n")

    inference_start_time = time.time()

    try:
        with torch.no_grad():
            # Create data sample with proper hard conditions and context
            input_data_sample = dataset.create_data_sample_normalized(
                start_state_torch.squeeze(0),  # Remove batch dimension for dataset method
                goal_state_torch.squeeze(0),
            )

            # Extract hard conditions and build context
            hard_conds = input_data_sample["hard_conds"]
            context_d = dataset.build_context(input_data_sample)

            # Debug: print shapes
            print("Context tensor shapes:")
            for key, val in context_d.items():
                if torch.is_tensor(val):
                    print(f"  {key}: {val.shape}")
            print()

            # Sample trajectories using run_inference
            samples = model.run_inference(
                context_d=context_d,
                hard_conds=hard_conds,
                n_samples=n_samples,
                horizon=dataset.n_learnable_control_points,
                return_chain=False,
            )

        inference_time = time.time() - inference_start_time

        print(f"✓ SUCCESS! MPD inference completed")
        print(f"  Inference time: {inference_time:.3f} sec")
        print(f"  Generated samples shape: {samples.shape}")

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
        smoothness_values = compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)

        best_idx = torch.argmin(path_lengths)
        best_path_length = path_lengths[best_idx].item()
        best_smoothness = smoothness_values[best_idx].item()

        print(f"  Best path length: {best_path_length:.3f}")
        print(f"  Best smoothness: {best_smoothness:.3f}")

        # Count collision-free trajectories (using full position trajectories)
        collision_free_count = 0
        for j in range(len(q_trajs_pos)):
            traj = q_trajs_pos[j]
            if check_trajectory_collision_free(traj, robot, env):
                collision_free_count += 1

        collision_rate = 1.0 - (collision_free_count / len(samples))
        print(f"  Collision-free samples: {collision_free_count}/{n_samples} ({(1-collision_rate)*100:.1f}%)")

        print("\n" + "="*80)
        print("TEST PASSED! MPD is working correctly.")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ ERROR during MPD inference: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "="*80)
        print("TEST FAILED! MPD inference error.")
        print("="*80 + "\n")

        return False

if __name__ == "__main__":
    success = test_mpd_single_problem()
    sys.exit(0 if success else 1)
