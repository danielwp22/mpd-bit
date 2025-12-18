"""
Minimal BIT* Template Solution
This is the completed version - use this to check your implementation!
"""
import os
import sys
import time
import numpy as np

# IMPORTANT: Import isaacgym FIRST to avoid import order issues
import isaacgym

# Add pybullet_ompl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../deps/pybullet_ompl'))

import pybullet as p
from pybullet_utils import bullet_client
from pb_ompl.pb_ompl import PbOMPL, PbOMPLRobot

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness


class MinimalBITStarBaseline:
    """Minimal BIT* baseline planner (SOLUTION)."""

    def __init__(
        self,
        robot,
        robot_urdf_path: str,
        planner_name: str = "BITstar",
        allowed_planning_time: float = 60.0,
        interpolate_num: int = 128,
        device: str = "cuda:0",
    ):
        self.planner_name = planner_name
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.torch_robot = robot

        self.device = get_torch_device(device)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # SOLUTION 1: Initialize PyBullet client
        self.pybullet_client = bullet_client.BulletClient(p.DIRECT)
        self.pybullet_client.setGravity(0, 0, -9.8)

        # SOLUTION 2: Load robot URDF
        self.robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0])

        # SOLUTION 3: Create PbOMPLRobot wrapper
        link_name_ee = robot.link_name_ee if hasattr(robot, 'link_name_ee') else 'ee_link'
        self.robot = PbOMPLRobot(self.pybullet_client, self.robot_id, link_name_ee=link_name_ee)

        self.obstacles = []

        # SOLUTION 4: Setup OMPL interface
        self.pb_ompl_interface = PbOMPL(
            self.pybullet_client,
            self.robot,
            self.obstacles,
            min_distance_robot_env=0.05
        )
        self.pb_ompl_interface.set_planner(planner_name)

        print(f"Initialized {planner_name} planner")

    def set_obstacles(self, obstacles):
        """Set environment obstacles."""
        # SOLUTION 5: Store obstacles and update OMPL interface
        self.obstacles = obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def plan(self, start_state, goal_state, debug=False):
        """Plan a trajectory from start to goal."""
        # SOLUTION 6: Convert states to Python float list
        start_state = [float(x) for x in start_state]
        goal_state = [float(x) for x in goal_state]

        # SOLUTION 7: Set robot to start state
        self.robot.set_state(start_state)

        # SOLUTION 8: Call the planner and measure time
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

        # SOLUTION 9-12: Compute metrics if planning succeeded
        if results_dict['success']:
            sol_path = results_dict['sol_path']

            # SOLUTION 10: Convert to torch tensor
            sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)

            # SOLUTION 11: Compute path length
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()

            # SOLUTION 12: Compute smoothness
            smoothness = compute_smoothness(sol_path_torch, self.torch_robot)[0].item()

            results_dict['path_length'] = path_length
            results_dict['smoothness'] = smoothness
            results_dict['num_waypoints'] = len(sol_path)

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning SUCCESS")
                print(f"  Planning time: {planning_time:.3f} sec")
                print(f"  Path length: {path_length:.3f}")
                print(f"  Smoothness: {smoothness:.3f}")
                print(f"{'='*80}\n")
        else:
            results_dict['path_length'] = float('inf')
            results_dict['smoothness'] = float('inf')
            results_dict['num_waypoints'] = 0

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning FAILED after {planning_time:.3f} sec")
                print(f"{'='*80}\n")

        return results_dict

    def terminate(self):
        """Cleanup PyBullet connection."""
        # SOLUTION 13: Disconnect PyBullet client
        self.pybullet_client.disconnect()


def run_minimal_example():
    """Run a minimal example with Panda robot in Spheres3D environment."""
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda
    from torch_robotics.torch_utils.seed import fix_random_seed

    print("\n" + "="*80)
    print("Minimal BIT* Solution - Testing Implementation")
    print("="*80 + "\n")

    fix_random_seed(42)

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # SOLUTION 14: Create environment
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=False,
        sdf_cell_size=0.05,
        tensor_args=tensor_args
    )

    # SOLUTION 15: Create robot
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # SOLUTION 16: Sample random start and goal
    q_start = robot.sample_q_pos()
    q_goal = robot.sample_q_pos()
    start_state = to_numpy(q_start)
    goal_state = to_numpy(q_goal)

    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # SOLUTION 17: Get robot URDF path
    robot_urdf_path = robot.robot_urdf_file

    # SOLUTION 18: Initialize baseline planner
    baseline = MinimalBITStarBaseline(
        robot=robot,
        robot_urdf_path=robot_urdf_path,
        planner_name="BITstar",
        allowed_planning_time=60.0,
        interpolate_num=128,
    )

    # SOLUTION 19: Plan from start to goal
    result = baseline.plan(start_state, goal_state, debug=True)

    # SOLUTION 20: Print results
    if result['success']:
        print("\n" + "="*80)
        print("IMPLEMENTATION WORKS!")
        print("="*80)
        print(f"Path length: {result['path_length']:.3f}")
        print(f"Smoothness: {result['smoothness']:.3f}")
        print(f"Planning time: {result['planning_time']:.3f} sec")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("Planning failed")
        print("="*80 + "\n")

    # SOLUTION 21: Cleanup
    baseline.terminate()

    return result


def load_diffusion_problem_and_plan():
    """Load a problem from diffusion model results and plan."""
    print("\n" + "="*80)
    print("Loading Problem from Diffusion Model Results")
    print("="*80 + "\n")

    diffusion_results_file = "logs/2/results_single_plan-000.pt"

    # SOLUTION 22: Check if diffusion results exist
    if not os.path.exists(diffusion_results_file):
        print(f"Error: {diffusion_results_file} not found")
        print("Run the diffusion model first: python inference.py")
        return None

    # SOLUTION 23: Load diffusion results
    diff_results = torch.load(diffusion_results_file, map_location='cpu')

    # SOLUTION 24: Extract start and goal states
    start_state = to_numpy(diff_results['q_pos_start'])
    goal_state = to_numpy(diff_results['q_pos_goal'])

    # SOLUTION 25: Extract diffusion path length
    diff_path_length = float(diff_results['metrics']['trajs_best']['path_length'])

    print(f"Diffusion model path length: {diff_path_length:.3f}")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # SOLUTION 26: Setup environment and robot
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=False,
        sdf_cell_size=0.05,
        tensor_args=tensor_args
    )
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    baseline = MinimalBITStarBaseline(
        robot=robot,
        robot_urdf_path=robot.robot_urdf_file,
        planner_name="BITstar",
        allowed_planning_time=60.0,
        interpolate_num=128,
    )

    # SOLUTION 27: Plan with BIT*
    result = baseline.plan(start_state, goal_state, debug=True)

    # SOLUTION 28: Compare with diffusion
    if result['success']:
        print("\n" + "="*80)
        print("COMPARISON: BIT* vs Diffusion Model")
        print("="*80)
        print(f"Diffusion path length: {diff_path_length:.3f}")
        print(f"BIT* path length:      {result['path_length']:.3f}")

        if result['path_length'] < diff_path_length:
            improvement = (diff_path_length - result['path_length']) / diff_path_length * 100
            print(f"\n🎉 BIT* found a {improvement:.1f}% shorter path!")
        elif result['path_length'] > diff_path_length:
            diff = (result['path_length'] - diff_path_length) / diff_path_length * 100
            print(f"\nDiffusion found a {diff:.1f}% shorter path")
        else:
            print(f"\nBoth methods found similar path lengths!")

        print("="*80 + "\n")

    # SOLUTION 29: Cleanup
    baseline.terminate()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal BIT* Solution")
    parser.add_argument("--mode", default="minimal",
                       choices=["minimal", "compare"],
                       help="Run minimal example or compare with diffusion")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MINIMAL BIT* SOLUTION")
    print("="*80 + "\n")

    if args.mode == "minimal":
        result = run_minimal_example()
    else:
        result = load_diffusion_problem_and_plan()
