"""
GPU-Based BIT* Baseline for Motion Planning

Pure Python implementation of Batch Informed Trees (BIT*) using GPU for computations.
Uses the same SDF-based collision checking as the diffusion model for fair comparison.

Based on: "Batch Informed Trees (BIT*): Sampling-based optimal planning via the
heuristically guided search of implicit random geometric graphs" by Gammell et al.
"""
import os
import sys
import time
import yaml
import numpy as np
from heapq import heappush, heappop

# IMPORTANT: Import isaacgym FIRST to avoid import order issues
import isaacgym

import torch
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy
from torch_robotics.trajectory.metrics import compute_path_length


def compute_smoothness_finite_diff(trajs):
    """
    Compute smoothness using finite differences (for position-only trajectories).

    Args:
        trajs: Trajectory tensor of shape (batch, horizon, q_dim)

    Returns:
        smoothness: Tensor of shape (batch,)
    """
    assert trajs.ndim == 3
    velocities = torch.diff(trajs, dim=1)
    accelerations = torch.diff(velocities, dim=1)
    smoothness = torch.linalg.norm(accelerations, dim=-1).sum(-1)
    return smoothness


def convert_to_serializable(obj):
    """Convert numpy/torch types to Python native types for YAML serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
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


class Vertex:
    """Represents a state/configuration in the search tree."""

    def __init__(self, state, vertex_id):
        self.state = state  # torch tensor for GPU operations
        self.id = vertex_id
        self.parent = None
        self.children = []
        self.cost = float('inf')

    def __lt__(self, other):
        return self.id < other.id


class Edge:
    """Represents a potential connection between two vertices."""

    def __init__(self, v1, v2, cost):
        self.v1 = v1
        self.v2 = v2
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class BITStarGPU:
    """GPU-based BIT* algorithm implementation."""

    def __init__(
        self,
        robot,
        task,  # PlanningTask with environment
        allowed_planning_time: float = 60.0,
        interpolate_num: int = 128,
        device: str = "cuda:0",
        batch_size: int = 100,
        max_edge_length: float = None,
        goal_region_radius: float = 0.05,
    ):
        """
        Initialize BIT* planner.

        Args:
            robot: torch_robotics robot instance
            task: PlanningTask with environment for collision checking
            allowed_planning_time: Max planning time in seconds
            interpolate_num: Number of waypoints in final trajectory
            device: Device for torch computations
            batch_size: Number of samples per batch
            max_edge_length: Maximum edge length (None = auto)
            goal_region_radius: Radius to consider goal reached
        """
        self.torch_robot = robot
        self.task = task
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.batch_size = batch_size
        self.goal_region_radius = goal_region_radius

        self.device = get_torch_device(device)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # Configuration space bounds
        self.q_min = robot.q_min
        self.q_max = robot.q_max
        self.dof = len(self.q_min)

        # Compute max edge length
        if max_edge_length is None:
            self.max_edge_length = torch.linalg.norm(self.q_max - self.q_min).item() * 0.15
        else:
            self.max_edge_length = max_edge_length

        # BIT* data structures
        self.vertices = []
        self.vertex_id_counter = 0
        self.samples = []
        self.edge_queue = []
        self.vertex_queue = []

        self.start_vertex = None
        self.goal_vertex = None

        print(f"Initialized BIT* GPU (batch_size={batch_size}, max_edge={self.max_edge_length:.3f})")

    def plan(self, start_state, goal_state, debug=False):
        """
        Plan trajectory from start to goal using BIT*.

        Returns:
            results_dict with success, sol_path, planning_time, path_length, smoothness
        """
        start_time = time.time()

        # Convert to torch tensors
        start_state = to_torch(start_state, **self.tensor_args)
        goal_state = to_torch(goal_state, **self.tensor_args)

        # Initialize
        self._initialize(start_state, goal_state)

        # Main BIT* loop
        iteration = 0
        best_cost = float('inf')

        while time.time() - start_time < self.allowed_planning_time:
            iteration += 1

            # Sample new batch if queues empty
            if not self.edge_queue and not self.vertex_queue:
                if debug and iteration > 1:
                    print(f"Iteration {iteration}: Sampling new batch (current best: {best_cost:.3f})")
                self._sample_batch()
                self._update_edge_queue()

            # Expand vertex or process edge
            if self.vertex_queue and (not self.edge_queue or
                self._get_queue_value(self.vertex_queue) <= self._get_queue_value(self.edge_queue)):
                self._expand_vertex()
            elif self.edge_queue:
                success = self._process_edge()
                if success and self.goal_vertex is not None:
                    new_cost = self.goal_vertex.cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        if debug:
                            print(f"  Found solution with cost: {best_cost:.3f}")

            # Early termination
            if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
                if time.time() - start_time > min(2.0, self.allowed_planning_time * 0.3):
                    break

        planning_time = time.time() - start_time

        # Extract solution
        if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
            raw_path = self._extract_path()
            sol_path = self._interpolate_path(raw_path, self.interpolate_num)

            # Compute metrics
            sol_path_torch = sol_path[None, ...]  # Add batch dimension
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
            smoothness = compute_smoothness_finite_diff(sol_path_torch)[0].item()

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning SUCCESS")
                print(f"  Planning time: {planning_time:.3f} sec")
                print(f"  Iterations: {iteration}")
                print(f"  Vertices: {len(self.vertices)}")
                print(f"  Path length: {path_length:.3f}")
                print(f"  Smoothness: {smoothness:.3f}")
                print(f"{'='*80}\n")

            return {
                'success': True,
                'sol_path': to_numpy(sol_path),
                'planning_time': planning_time,
                'path_length': path_length,
                'smoothness': smoothness,
                'num_waypoints': len(sol_path),
            }
        else:
            if debug:
                print(f"\n{'='*80}")
                print(f"Planning FAILED after {planning_time:.3f} sec")
                print(f"  Iterations: {iteration}")
                print(f"  Vertices: {len(self.vertices)}")
                print(f"{'='*80}\n")

            return {
                'success': False,
                'sol_path': None,
                'planning_time': planning_time,
                'path_length': float('inf'),
                'smoothness': float('inf'),
                'num_waypoints': 0,
            }

    def _initialize(self, start_state, goal_state):
        """Initialize BIT* data structures."""
        self.vertices = []
        self.vertex_id_counter = 0
        self.samples = []
        self.edge_queue = []
        self.vertex_queue = []

        # Create start vertex
        self.start_vertex = Vertex(start_state, self.vertex_id_counter)
        self.vertex_id_counter += 1
        self.start_vertex.cost = 0.0
        self.vertices.append(self.start_vertex)

        # Store goal
        self.goal_state = goal_state
        self.goal_vertex = None

        # Add goal to samples
        goal_sample = Vertex(goal_state, self.vertex_id_counter)
        self.vertex_id_counter += 1
        self.samples.append(goal_sample)

    def _sample_batch(self):
        """Sample batch of collision-free configurations."""
        for _ in range(self.batch_size):
            # Random sample in configuration space
            q = torch.rand(self.dof, **self.tensor_args)
            q = self.q_min + q * (self.q_max - self.q_min)

            # Check collision
            if self._is_collision_free(q):
                v = Vertex(q, self.vertex_id_counter)
                self.vertex_id_counter += 1
                self.samples.append(v)

    def _is_collision_free(self, q):
        """Check if configuration is collision-free."""
        # Use task's collision checking (SDF-based)
        in_collision = self.task.compute_collision(q.unsqueeze(0)).squeeze() > 0.5
        return not in_collision.item()

    def _update_edge_queue(self):
        """Update edge queue with potential connections."""
        for v_tree in self.vertices:
            for v_sample in self.samples:
                dist = torch.linalg.norm(v_tree.state - v_sample.state).item()
                if dist <= self.max_edge_length:
                    edge_cost = v_tree.cost + dist + self._heuristic(v_sample.state)
                    heappush(self.edge_queue, (edge_cost, Edge(v_tree, v_sample, dist)))

    def _expand_vertex(self):
        """Expand the best vertex."""
        if not self.vertex_queue:
            return

        _, v = heappop(self.vertex_queue)

        # Add edges to nearby samples
        for v_sample in self.samples:
            dist = torch.linalg.norm(v.state - v_sample.state).item()
            if dist <= self.max_edge_length:
                edge_cost = v.cost + dist + self._heuristic(v_sample.state)
                heappush(self.edge_queue, (edge_cost, Edge(v, v_sample, dist)))

    def _process_edge(self):
        """Process the best edge."""
        if not self.edge_queue:
            return False

        _, edge = heappop(self.edge_queue)
        v1, v2, cost = edge.v1, edge.v2, edge.cost

        # Check if edge is useful
        estimated_cost = v1.cost + cost + self._heuristic(v2.state)
        if self.goal_vertex is not None and estimated_cost >= self.goal_vertex.cost:
            return False

        # Check if v2 can be improved
        new_cost = v1.cost + cost
        if new_cost >= v2.cost:
            return False

        # Check edge collision
        if not self._is_edge_collision_free(v1.state, v2.state):
            return False

        # Add v2 to tree
        if v2 in self.samples:
            self.samples.remove(v2)
            self.vertices.append(v2)

            # Check if goal
            if torch.linalg.norm(v2.state - self.goal_state).item() < self.goal_region_radius:
                self.goal_vertex = v2

        # Update parent
        if v2.parent is not None:
            v2.parent.children.remove(v2)

        v2.parent = v1
        v2.cost = new_cost
        v1.children.append(v2)

        # Add to vertex queue
        queue_value = v2.cost + self._heuristic(v2.state)
        heappush(self.vertex_queue, (queue_value, v2))

        return True

    def _heuristic(self, q):
        """Heuristic cost-to-go."""
        return torch.linalg.norm(q - self.goal_state).item()

    def _is_edge_collision_free(self, q1, q2, resolution=10):
        """Check edge collision."""
        for alpha in torch.linspace(0, 1, resolution, **self.tensor_args):
            q = q1 * (1 - alpha) + q2 * alpha
            if not self._is_collision_free(q):
                return False
        return True

    def _extract_path(self):
        """Extract path from start to goal."""
        path = []
        v = self.goal_vertex
        while v is not None:
            path.append(v.state)
            v = v.parent
        path.reverse()
        return torch.stack(path)

    def _interpolate_path(self, path, num_waypoints):
        """Interpolate path to num_waypoints."""
        if len(path) == 0:
            return path
        if len(path) == num_waypoints:
            return path

        # Compute cumulative distances
        distances = torch.zeros(len(path), **self.tensor_args)
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + torch.linalg.norm(path[i] - path[i-1])

        total_distance = distances[-1]
        if total_distance < 1e-6:
            return path[0].repeat(num_waypoints, 1)

        # Interpolate uniformly
        interpolated_path = []
        for i in range(num_waypoints):
            target_dist = total_distance * i / (num_waypoints - 1)

            # Find segment
            idx = torch.searchsorted(distances, target_dist).item()
            if idx == 0:
                interpolated_path.append(path[0])
            elif idx >= len(path):
                interpolated_path.append(path[-1])
            else:
                # Linear interpolation
                alpha = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                q = path[idx-1] * (1 - alpha) + path[idx] * alpha
                interpolated_path.append(q)

        return torch.stack(interpolated_path)

    def _get_queue_value(self, queue):
        """Get best queue value."""
        return queue[0][0] if queue else float('inf')


def run_bitstar_gpu_on_diffusion_problem(
    diffusion_results_file,
    max_time=60.0,
    batch_size=100,
    interpolate_num=128,
    device="cuda:0",
    seed=2,
):
    """
    Run GPU-based BIT* on the same problem as diffusion model.
    """
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda
    from torch_robotics.tasks.tasks import PlanningTask
    from torch_robotics.torch_utils.seed import fix_random_seed

    fix_random_seed(seed)

    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # Load diffusion results
    print(f"\nLoading diffusion results from: {diffusion_results_file}")
    diff_results = torch.load(diffusion_results_file, map_location='cpu')

    start_state = diff_results['q_pos_start']
    goal_state = diff_results['q_pos_goal']

    print(f"Start state: {to_numpy(start_state)}")
    print(f"Goal state: {to_numpy(goal_state)}")

    # Create environment and robot (same as diffusion)
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.005,
        tensor_args=tensor_args
    )
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Create planning task (provides collision checking)
    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], **tensor_args),
        obstacle_cutoff_margin=0.03,
        tensor_args=tensor_args
    )

    # Initialize BIT* GPU
    print(f"\nInitializing BIT* GPU planner...")
    print(f"  Max time: {max_time}s")
    print(f"  Batch size: {batch_size}")

    planner = BITStarGPU(
        robot=robot,
        task=task,
        allowed_planning_time=max_time,
        interpolate_num=interpolate_num,
        device=device,
        batch_size=batch_size,
    )

    # Plan
    print(f"\nRunning BIT* GPU planner...")
    results = planner.plan(start_state, goal_state, debug=True)

    return results


def save_results(results_dict, results_dir="logs_bitstar_gpu_panda_spheres3d"):
    """Save results."""
    os.makedirs(results_dir, exist_ok=True)

    statistics = {
        'planner': 'BITstar-GPU',
        'n_problems': 1,
        'success_count': 1 if results_dict['success'] else 0,
        'success_rate': 1.0 if results_dict['success'] else 0.0,
        'planning_time_mean': results_dict['planning_time'],
        'planning_time_std': 0.0,
        'path_length_mean': results_dict['path_length'],
        'path_length_std': 0.0,
        'smoothness_mean': results_dict['smoothness'],
        'smoothness_std': 0.0,
    }

    stats_file = os.path.join(results_dir, "statistics.yaml")
    with open(stats_file, 'w') as f:
        yaml.dump(convert_to_serializable(statistics), f, default_flow_style=False)
    print(f"\nSaved statistics to {stats_file}")

    if results_dict['success']:
        traj_file = os.path.join(results_dir, "trajectory_000.npy")
        np.save(traj_file, results_dict['sol_path'])
        print(f"Saved trajectory to {traj_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GPU-based BIT* baseline")
    parser.add_argument("--use-diffusion-problem", action="store_true",
                       help="Use same start/goal as diffusion model")
    parser.add_argument("--diffusion-results", default="logs/2/results_single_plan-000.pt",
                       help="Path to diffusion model results")
    parser.add_argument("--time", type=float, default=60.0,
                       help="Max planning time (seconds)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for sampling")
    parser.add_argument("--device", default="cuda:0",
                       help="Device (cuda:0 or cpu)")
    parser.add_argument("--seed", type=int, default=2,
                       help="Random seed")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("GPU-Based BIT* Baseline")
    print("Uses PyTorch + SDF collision checking (same as diffusion model)")
    print("="*80 + "\n")

    if args.use-diffusion-problem:
        results = run_bitstar_gpu_on_diffusion_problem(
            diffusion_results_file=args.diffusion_results,
            max_time=args.time,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        )

        save_results(results)

        print(f"\nDone! Results saved to logs_bitstar_gpu_panda_spheres3d/")
        print("\nTo compare with diffusion model:")
        print("  python compare_results.py --baselines bitstar_gpu")
    else:
        print("Error: Must specify --use-diffusion-problem")
        print("\nUsage:")
        print("  python bitstar_gpu_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt")
