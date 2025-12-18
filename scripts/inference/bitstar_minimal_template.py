"""
Minimal BIT* Template for Motion Planning
A simplified template for implementing BIT* baseline comparison with diffusion model.

TODO: Fill in the sections marked with TODO to complete the implementation.
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
    """Minimal BIT* baseline planner template."""

    def __init__(
        self,
        robot,  # torch_robotics robot instance
        robot_urdf_path: str,
        planner_name: str = "BITstar",
        allowed_planning_time: float = 60.0,
        interpolate_num: int = 128,
        device: str = "cuda:0",
    ):
        """
        Initialize the BIT* baseline planner.

        TODO: Understand what each parameter does and how to use them.

        Args:
            robot: torch_robotics robot instance (for computing metrics)
            robot_urdf_path: Path to robot URDF file
            planner_name: OMPL planner name (BITstar, RRTConnect, etc.)
            allowed_planning_time: Max time for planning in seconds
            interpolate_num: Number of waypoints in final trajectory
            device: Device for torch computations
        """
        self.planner_name = planner_name
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.torch_robot = robot

        self.device = get_torch_device(device)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # TODO 1: Initialize PyBullet client
        # Hint: Use bullet_client.BulletClient(p.DIRECT) for headless mode
        # Hint: Set gravity with pybullet_client.setGravity(0, 0, -9.8)
        self.pybullet_client = None  # TODO: Replace with actual initialization

        # TODO 2: Load robot URDF into PyBullet
        # Hint: Use p.loadURDF(robot_urdf_path, [0, 0, 0])
        self.robot_id = None  # TODO: Replace with loaded robot ID

        # TODO 3: Create PbOMPLRobot wrapper
        # Hint: link_name_ee = robot.link_name_ee if hasattr(robot, 'link_name_ee') else 'ee_link'
        # Hint: self.robot = PbOMPLRobot(self.pybullet_client, self.robot_id, link_name_ee=link_name_ee)
        self.robot = None  # TODO: Replace with PbOMPLRobot instance

        # Initialize obstacles list (will be set later)
        self.obstacles = []

        # TODO 4: Setup OMPL interface
        # Hint: Create PbOMPL instance with client, robot, obstacles
        # Hint: Call set_planner(planner_name) to select the planner
        self.pb_ompl_interface = None  # TODO: Replace with PbOMPL instance

        print(f"Initialized {planner_name} planner (template)")

    def set_obstacles(self, obstacles):
        """Set environment obstacles."""
        # TODO 5: Store obstacles and update OMPL interface
        # Hint: self.obstacles = obstacles
        # Hint: self.pb_ompl_interface.set_obstacles(self.obstacles)
        pass  # TODO: Implement

    def plan(self, start_state, goal_state, debug=False):
        """
        Plan a trajectory from start to goal using the configured planner.

        TODO: Implement the core planning function.

        Args:
            start_state: Starting configuration (numpy array or list)
            goal_state: Goal configuration (numpy array or list)
            debug: Print debug information

        Returns:
            results_dict with keys:
                - success: bool
                - sol_path: np.ndarray (interpolate_num, dof)
                - planning_time: float
                - path_length: float
                - smoothness: float
        """
        # TODO 6: Convert states to Python float list
        # Hint: OMPL needs native Python floats, not numpy types
        # Hint: start_state = [float(x) for x in start_state]
        start_state = None  # TODO: Convert start_state
        goal_state = None   # TODO: Convert goal_state

        # TODO 7: Set robot to start state
        # Hint: self.robot.set_state(start_state)

        # TODO 8: Call the planner and measure time
        # Hint: Use time.time() to measure start and end
        # Hint: Call self.pb_ompl_interface.plan(goal_state, allowed_time=..., interpolate_num=..., simplify_path=True)
        start_time = time.time()

        results_dict = {}  # TODO: Replace with actual planning call

        planning_time = time.time() - start_time
        results_dict['planning_time'] = planning_time

        # TODO 9: Compute metrics if planning succeeded
        if results_dict.get('success', False):
            sol_path = results_dict['sol_path']

            # TODO 10: Convert solution path to torch tensor
            # Hint: Add batch dimension with [None, ...]
            # Hint: sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
            sol_path_torch = None  # TODO: Convert to torch

            # TODO 11: Compute path length metric
            # Hint: Use compute_path_length(sol_path_torch, self.torch_robot)
            # Hint: Extract scalar value with [0].item()
            path_length = 0.0  # TODO: Compute path length

            # TODO 12: Compute smoothness metric
            # Hint: Use compute_smoothness(sol_path_torch, self.torch_robot)
            smoothness = 0.0  # TODO: Compute smoothness

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
            # Planning failed
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
        # TODO 13: Disconnect PyBullet client
        # Hint: self.pybullet_client.disconnect()
        pass  # TODO: Implement


def run_minimal_example():
    """
    Run a minimal example with Panda robot in Spheres3D environment.

    TODO: Complete this example function to test your implementation.
    """
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda
    from torch_robotics.torch_utils.seed import fix_random_seed

    print("\n" + "="*80)
    print("Minimal BIT* Template - Testing Your Implementation")
    print("="*80 + "\n")

    # Fix random seed for reproducibility
    fix_random_seed(42)

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # TODO 14: Create environment
    # Hint: env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    env = None  # TODO: Create environment

    # TODO 15: Create robot
    # Hint: robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)
    robot = None  # TODO: Create robot

    # TODO 16: Sample random start and goal configurations
    # Hint: q_start = robot.sample_q_pos()
    # Hint: q_goal = robot.sample_q_pos()
    # Hint: Convert to numpy with to_numpy()
    start_state = None  # TODO: Sample and convert start
    goal_state = None   # TODO: Sample and convert goal

    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # TODO 17: Get robot URDF path
    # Hint: robot_urdf_path = robot.robot_urdf_file
    robot_urdf_path = None  # TODO: Get URDF path

    # TODO 18: Initialize your baseline planner
    # Hint: Pass robot, robot_urdf_path, planner_name, etc.
    baseline = None  # TODO: Create MinimalBITStarBaseline instance

    # TODO 19: Plan from start to goal
    # Hint: result = baseline.plan(start_state, goal_state, debug=True)
    result = None  # TODO: Call plan method

    # TODO 20: Print results
    if result and result.get('success'):
        print("\n" + "="*80)
        print("YOUR IMPLEMENTATION WORKS!")
        print("="*80)
        print(f"Path length: {result['path_length']:.3f}")
        print(f"Smoothness: {result['smoothness']:.3f}")
        print(f"Planning time: {result['planning_time']:.3f} sec")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("Planning failed - check your implementation")
        print("="*80 + "\n")

    # TODO 21: Cleanup
    # Hint: baseline.terminate()

    return result


def load_diffusion_problem_and_plan():
    """
    Load a problem from diffusion model results and plan with your BIT* implementation.

    TODO: Complete this function to compare with diffusion model.
    """
    print("\n" + "="*80)
    print("Loading Problem from Diffusion Model Results")
    print("="*80 + "\n")

    diffusion_results_file = "logs/2/results_single_plan-000.pt"

    # TODO 22: Check if diffusion results exist
    if not os.path.exists(diffusion_results_file):
        print(f"Error: {diffusion_results_file} not found")
        print("Run the diffusion model first: python inference.py")
        return None

    # TODO 23: Load diffusion results
    # Hint: diff_results = torch.load(diffusion_results_file, map_location='cpu')
    diff_results = None  # TODO: Load results

    # TODO 24: Extract start and goal states
    # Hint: start_state = to_numpy(diff_results['q_pos_start'])
    # Hint: goal_state = to_numpy(diff_results['q_pos_goal'])
    start_state = None  # TODO: Extract start
    goal_state = None   # TODO: Extract goal

    # TODO 25: Extract diffusion path length for comparison
    # Hint: diff_path_length = float(diff_results['metrics']['trajs_best']['path_length'])
    diff_path_length = None  # TODO: Extract path length

    print(f"Diffusion model path length: {diff_path_length:.3f}")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # TODO 26: Setup environment and robot (same as run_minimal_example)
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = None      # TODO: Create environment
    robot = None    # TODO: Create robot
    baseline = None # TODO: Create baseline planner

    # TODO 27: Plan with your BIT* implementation
    result = None  # TODO: Call plan method

    # TODO 28: Compare with diffusion
    if result and result.get('success'):
        print("\n" + "="*80)
        print("COMPARISON: Your BIT* vs Diffusion Model")
        print("="*80)
        print(f"Diffusion path length: {diff_path_length:.3f}")
        print(f"BIT* path length:      {result['path_length']:.3f}")

        if result['path_length'] < diff_path_length:
            improvement = (diff_path_length - result['path_length']) / diff_path_length * 100
            print(f"\n🎉 Your BIT* found a {improvement:.1f}% shorter path!")
        elif result['path_length'] > diff_path_length:
            diff = (result['path_length'] - diff_path_length) / diff_path_length * 100
            print(f"\nDiffusion found a {diff:.1f}% shorter path")
        else:
            print(f"\nBoth methods found similar path lengths!")

        print("="*80 + "\n")

    # TODO 29: Cleanup
    # Hint: baseline.terminate()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Minimal BIT* Template")
    parser.add_argument("--mode", default="minimal",
                       choices=["minimal", "compare"],
                       help="Run minimal example or compare with diffusion")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MINIMAL BIT* TEMPLATE - IMPLEMENTATION EXERCISE")
    print("="*80)
    print("\nThis template helps you implement BIT* baseline from scratch.")
    print("Follow the TODO comments in the code to complete the implementation.\n")
    print("Steps:")
    print("  1. Fill in the TODOs in MinimalBITStarBaseline.__init__")
    print("  2. Fill in the TODOs in MinimalBITStarBaseline.plan")
    print("  3. Fill in the TODOs in run_minimal_example")
    print("  4. Test with: python bitstar_minimal_template.py --mode minimal")
    print("  5. Once working, compare with diffusion: python bitstar_minimal_template.py --mode compare")
    print("="*80 + "\n")

    if args.mode == "minimal":
        result = run_minimal_example()
    else:
        result = load_diffusion_problem_and_plan()

    if result is None:
        print("\n⚠️  Implementation incomplete - fill in the TODOs!")
        print("See the comments in the code for hints.\n")
