"""
GPU-Accelerated BIT* Baseline for Motion Planning

Pure Python implementation of Batch Informed Trees (BIT*) with GPU acceleration.
Uses the same SDF-based collision checking as the diffusion model for fair comparison.

Key Features:
- GPU-accelerated batch collision checking (100 samples checked in parallel)
- GPU-accelerated edge validation (10 interpolated points checked in parallel)
- SDF-based collision detection (identical to diffusion model)
- Pure PyTorch implementation (no OMPL dependency)

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
    """
    GPU-accelerated BIT* algorithm implementation with batch collision checking.

    This implementation uses:
    - GPU-accelerated batch sampling (batch_size samples checked at once)
    - GPU-accelerated batch edge checking (10 interpolated points checked at once)
    - SDF-based collision detection (same as diffusion model for fair comparison)
    """

    def __init__(
        self,
        robot,
        env,  # Environment for collision checking
        allowed_planning_time: float = 60.0,
        interpolate_num: int = 128,
        device: str = "cuda:0",
        batch_size: int = 100,
        max_edge_length: float = None,
        goal_region_radius: float = 0.05,
    ):
        """
        Initialize GPU-accelerated BIT* planner.

        Args:
            robot: torch_robotics robot instance
            env: Environment for SDF-based collision checking
            allowed_planning_time: Max planning time in seconds
            interpolate_num: Number of waypoints in final trajectory
            device: Device for torch computations (cuda:0 or cpu)
            batch_size: Number of samples to check in parallel per batch
            max_edge_length: Maximum edge length (None = auto-compute as 15% of workspace diagonal)
            goal_region_radius: Radius to consider goal reached
        """
        self.torch_robot = robot
        self.env = env
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.batch_size = batch_size
        self.goal_region_radius = goal_region_radius

        # Device might already be a torch.device object or a string
        if isinstance(device, str):
            self.device = get_torch_device(device)
        else:
            self.device = device
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # Configuration space bounds
        self.q_min = robot.q_pos_min
        self.q_max = robot.q_pos_max
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

        print(f"Initialized GPU-accelerated BIT* (batch_size={batch_size}, max_edge={self.max_edge_length:.3f})")
        print(f"  Using batch collision checking: {batch_size} samples checked in parallel")

    def plan(self, start_state, goal_state, target_path_length=None, debug=False):
        """
        Plan trajectory from start to goal using BIT*.

        Args:
            start_state: Starting configuration
            goal_state: Goal configuration
            target_path_length: Target path length to beat (e.g., from diffusion baseline)
                              If provided, planning continues until this is beaten or timeout
            debug: Print debug information

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
        first_solution_time = None
        first_solution_cost = None

        if target_path_length and debug:
            print(f"  Target path length to beat: {target_path_length:.3f}")

        while time.time() - start_time < self.allowed_planning_time:
            iteration += 1

            # Sample new batch if queues empty
            if not self.edge_queue and not self.vertex_queue:
                if debug and iteration > 1:
                    print(f"\n[{time.time() - start_time:.2f}s] Batch {iteration}: Current best: {best_cost:.3f}")
                    print(f"  Tree vertices: {len(self.vertices)}, Samples: {len(self.samples)}")

                self._sample_batch(debug=debug)

                # Prune samples if we have a solution
                if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
                    pruned = self._prune_samples()
                    if debug and pruned > 0:
                        print(f"  Pruned {pruned} samples that can't improve solution")

                self._update_edge_queue()
                if debug:
                    print(f"  Edge queue size: {len(self.edge_queue)}, Vertex queue size: {len(self.vertex_queue)}")

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

                        # Track first solution
                        if first_solution_time is None:
                            first_solution_time = time.time() - start_time
                            first_solution_cost = new_cost
                            if debug:
                                print(f"\n{'='*60}")
                                print(f"  [{first_solution_time:.2f}s] FIRST SOLUTION FOUND!")
                                print(f"  Path length: {best_cost:.3f}")
                                print(f"{'='*60}\n")

                            # Prune samples immediately after first solution
                            pruned = self._prune_samples()
                            if debug and pruned > 0:
                                print(f"  Pruned {pruned} samples after finding first solution")
                        else:
                            if debug:
                                print(f"  [{time.time() - start_time:.2f}s] >>> Improved solution! Path length: {best_cost:.3f}")

                        # Check if we beat the target
                        if target_path_length is not None and best_cost <= target_path_length:
                            if debug:
                                print(f"  [{time.time() - start_time:.2f}s] *** BEAT TARGET! {best_cost:.3f} <= {target_path_length:.3f}")
                            # Continue optimizing for a bit more to see if we can do even better
                            # But stop after 10% more of allowed time
                            if time.time() - start_time > self.allowed_planning_time * 0.1:
                                if debug:
                                    print(f"  Stopping after beating target...")
                                break

            # Early termination only if no target specified
            if target_path_length is None:
                if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
                    if time.time() - start_time > min(2.0, self.allowed_planning_time * 0.3):
                        if debug:
                            print(f"  Early termination (no target specified)")
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
                'first_solution_time': first_solution_time,
                'first_solution_cost': first_solution_cost,
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
                'first_solution_time': None,
                'first_solution_cost': None,
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

    def _sample_batch(self, debug=False):
        """Sample batch of collision-free configurations (GPU-accelerated)."""
        # Generate batch of random samples on GPU
        q_batch = torch.rand(self.batch_size, self.dof, **self.tensor_args)
        q_batch = self.q_min + q_batch * (self.q_max - self.q_min)

        # Batch collision check on GPU
        collision_free_mask = self._batch_collision_check(q_batch)

        # Count collision-free samples
        num_collision_free = collision_free_mask.sum().item()

        # Add collision-free samples to the list
        for i in range(self.batch_size):
            if collision_free_mask[i]:
                v = Vertex(q_batch[i], self.vertex_id_counter)
                self.vertex_id_counter += 1
                self.samples.append(v)

        if debug:
            print(f"  Sampled {num_collision_free}/{self.batch_size} collision-free configs")

        return num_collision_free

    def _prune_samples(self):
        """Prune samples that cannot improve current solution."""
        if self.goal_vertex is None or self.goal_vertex.cost == float('inf'):
            return 0

        best_cost = self.goal_vertex.cost
        original_count = len(self.samples)

        # Keep only samples that could potentially improve the solution
        self.samples = [
            s for s in self.samples
            if self._heuristic(s.state) < best_cost  # Optimistic estimate
        ]

        pruned = original_count - len(self.samples)
        return pruned

    def _batch_collision_check(self, q_batch):
        """
        Batch collision checking on GPU using SDF-based detection.

        Args:
            q_batch: (batch_size, dof) tensor of configurations

        Returns:
            collision_free: (batch_size,) boolean tensor
        """
        batch_size = q_batch.shape[0]

        # Get robot collision sphere positions for all configurations
        # Shape: (batch_size, num_spheres, 3)
        x_pos_batch = self.torch_robot.fk_map_collision(q_batch, pos_only=True)

        # Check collision for each configuration
        collision_free = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        # For each configuration
        for i in range(batch_size):
            x_pos = x_pos_batch[i]  # (num_spheres, 3)

            # Compute SDF for all spheres at once
            # Reshape for batch computation: (num_spheres, 3)
            sdf_values = self.env.compute_sdf(x_pos)  # (num_spheres,)

            # Configuration is in collision if any sphere has negative SDF
            if (sdf_values < 0.0).any():
                collision_free[i] = False

        return collision_free

    def _is_collision_free(self, q):
        """
        Check if configuration is collision-free using SDF-based collision detection.
        This uses the same method as the diffusion model.
        """
        # Use batch checking for single configuration
        result = self._batch_collision_check(q.unsqueeze(0))
        return result[0].item()

    def _update_edge_queue(self):
        """Update edge queue with potential connections (optimized with batch GPU operations)."""
        if len(self.vertices) == 0 or len(self.samples) == 0:
            return

        # Stack all vertex and sample states for batch distance computation
        vertex_states = torch.stack([v.state for v in self.vertices])  # (n_vertices, dof)
        sample_states = torch.stack([s.state for s in self.samples])  # (n_samples, dof)

        # Compute all pairwise distances on GPU: (n_vertices, n_samples)
        # Use broadcasting: (n_vertices, 1, dof) - (1, n_samples, dof)
        dists = torch.norm(vertex_states.unsqueeze(1) - sample_states.unsqueeze(0), dim=2)

        # Find pairs within max_edge_length
        valid_pairs = (dists <= self.max_edge_length).nonzero(as_tuple=False)

        # Add edges for valid pairs
        for v_idx, s_idx in valid_pairs:
            v_tree = self.vertices[v_idx]
            v_sample = self.samples[s_idx]
            dist = dists[v_idx, s_idx].item()
            edge_cost = v_tree.cost + dist + self._heuristic(v_sample.state)
            heappush(self.edge_queue, (edge_cost, Edge(v_tree, v_sample, dist)))

    def _expand_vertex(self):
        """Expand the best vertex (optimized with batch GPU operations)."""
        if not self.vertex_queue:
            return

        _, v = heappop(self.vertex_queue)

        if len(self.samples) == 0:
            return

        # Stack all sample states for batch distance computation
        sample_states = torch.stack([s.state for s in self.samples])  # (n_samples, dof)

        # Compute distances from v to all samples on GPU
        dists = torch.norm(sample_states - v.state.unsqueeze(0), dim=1)  # (n_samples,)

        # Find samples within max_edge_length
        valid_indices = (dists <= self.max_edge_length).nonzero(as_tuple=False).squeeze(-1)

        # Add edges for valid samples
        for idx in valid_indices:
            idx = idx.item()
            v_sample = self.samples[idx]
            dist = dists[idx].item()
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

        # Track if this is a new vertex being added to tree
        is_new_vertex = False

        # Add v2 to tree if it's still a sample
        if v2 in self.samples:
            self.samples.remove(v2)
            self.vertices.append(v2)
            is_new_vertex = True

        # Update parent
        if v2.parent is not None:
            v2.parent.children.remove(v2)

        v2.parent = v1
        v2.cost = new_cost
        v1.children.append(v2)

        # Check if this vertex is in the goal region (check for both new and improved vertices)
        if torch.linalg.norm(v2.state - self.goal_state).item() < self.goal_region_radius:
            if self.goal_vertex is None or v2.cost < self.goal_vertex.cost:
                self.goal_vertex = v2

        # Add to vertex queue for expansion
        queue_value = v2.cost + self._heuristic(v2.state)
        heappush(self.vertex_queue, (queue_value, v2))

        return True

    def _heuristic(self, q):
        """Heuristic cost-to-go."""
        return torch.linalg.norm(q - self.goal_state).item()

    def _is_edge_collision_free(self, q1, q2, resolution=10):
        """Check edge collision (GPU-accelerated batch checking)."""
        # Generate all interpolated points along the edge
        alphas = torch.linspace(0, 1, resolution, **self.tensor_args)
        # Broadcast and compute all interpolated configurations at once
        # q1 and q2 are (dof,), alphas is (resolution,)
        # Result: (resolution, dof)
        q_interp = q1[None, :] * (1 - alphas[:, None]) + q2[None, :] * alphas[:, None]

        # Batch collision check all interpolated points at once
        collision_free_mask = self._batch_collision_check(q_interp)

        # Edge is collision-free only if all interpolated points are collision-free
        return collision_free_mask.all().item()

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

    # Extract diffusion model's path length as target to beat
    target_path_length = None
    if 'metrics' in diff_results and 'trajs_best' in diff_results['metrics']:
        target_path_length = float(diff_results['metrics']['trajs_best']['path_length'])
        print(f"\nDiffusion model path length: {target_path_length:.3f}")
        print(f"BIT* will attempt to beat this baseline...")

    # Create environment and robot (same collision checking as diffusion, but no SDF precomputation)
    # Note: BIT* only needs point-wise collision checking, not gradients,
    # so we don't precompute the SDF to save GPU memory
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=False,
        sdf_cell_size=0.05,  # Larger cell size since we compute on-the-fly
        tensor_args=tensor_args
    )
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Initialize BIT* GPU
    print(f"\nInitializing BIT* GPU planner...")
    print(f"  Max time: {max_time}s")
    print(f"  Batch size: {batch_size}")

    planner = BITStarGPU(
        robot=robot,
        env=env,
        allowed_planning_time=max_time,
        interpolate_num=interpolate_num,
        device=device,
        batch_size=batch_size,
    )

    # Plan (with target path length to beat)
    print(f"\nRunning BIT* GPU planner...")
    results = planner.plan(start_state, goal_state, target_path_length=target_path_length, debug=True)

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

    # Add anytime statistics if available
    if results_dict.get('first_solution_time') is not None:
        statistics['planning_time_first_mean'] = results_dict['first_solution_time']
        statistics['planning_time_first_std'] = 0.0
        statistics['path_length_first_mean'] = results_dict['first_solution_cost']
        statistics['path_length_first_std'] = 0.0
        statistics['mode'] = 'anytime'

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

    if args.use_diffusion_problem:
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
