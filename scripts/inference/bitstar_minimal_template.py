"""
BIT* Algorithm Implementation for Motion Planning
Pure algorithmic implementation of Batch Informed Trees (BIT*) without using OMPL.

This implements the BIT* algorithm from:
"Batch Informed Trees (BIT*): Sampling-based optimal planning via the heuristically
guided search of implicit random geometric graphs" by Gammell et al.
"""
import os
import sys
import time
import numpy as np
from heapq import heappush, heappop
from collections import defaultdict

# IMPORTANT: Import isaacgym FIRST to avoid import order issues
import isaacgym

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


def compute_jerk_finite_diff(trajs):
    """
    Compute mean jerk using finite differences (for position-only trajectories).

    Jerk = rate of change of acceleration (third derivative of position).
    Mean jerk = average magnitude of jerk over the trajectory.

    Args:
        trajs: Trajectory tensor of shape (batch, horizon, q_dim) containing only positions

    Returns:
        mean_jerk: Tensor of shape (batch,) with mean jerk values
    """
    assert trajs.ndim == 3  # batch, horizon, q_dim

    # Compute velocities: vel[t] = pos[t+1] - pos[t]
    velocities = torch.diff(trajs, dim=1)  # (batch, horizon-1, q_dim)

    # Compute accelerations: acc[t] = vel[t+1] - vel[t]
    accelerations = torch.diff(velocities, dim=1)  # (batch, horizon-2, q_dim)

    # Compute jerk: jerk[t] = acc[t+1] - acc[t]
    jerks = torch.diff(accelerations, dim=1)  # (batch, horizon-3, q_dim)

    # Compute mean jerk as average of jerk magnitudes
    jerk_norms = torch.linalg.norm(jerks, dim=-1)  # (batch, horizon-3)
    mean_jerk = jerk_norms.mean(-1)  # (batch,)

    return mean_jerk


class Vertex:
    """Represents a state/configuration in the search tree."""

    def __init__(self, state, vertex_id):
        self.state = np.array(state, dtype=np.float64)
        self.id = vertex_id
        self.parent = None
        self.children = []
        self.cost = float('inf')  # Cost-to-come from start

    def __lt__(self, other):
        return self.id < other.id  # For heap tie-breaking


class Edge:
    """Represents a potential connection between two vertices."""

    def __init__(self, v1, v2, cost):
        self.v1 = v1
        self.v2 = v2
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class MinimalBITStarBaseline:
    """BIT* algorithm implementation."""

    def __init__(
        self,
        robot,  # torch_robotics robot instance
        robot_urdf_path: str = None,  # Not used in pure algorithmic implementation
        planner_name: str = "BITstar",
        allowed_planning_time: float = 60.0,
        interpolate_num: int = 128,
        device: str = "cuda:0",
        batch_size: int = 100,
        rewire_factor: float = 1.1,
        max_edge_length: float = None,
    ):
        """
        Initialize the BIT* planner.

        Args:
            robot: torch_robotics robot instance (for collision checking and metrics)
            robot_urdf_path: Not used (kept for compatibility)
            planner_name: Planner name (kept for compatibility)
            allowed_planning_time: Max time for planning in seconds
            interpolate_num: Number of waypoints in final trajectory
            device: Device for torch computations
            batch_size: Number of samples per batch
            rewire_factor: Radius factor for edge connections
            max_edge_length: Maximum edge length (None = auto-compute)
        """
        self.torch_robot = robot
        self.allowed_planning_time = allowed_planning_time
        self.interpolate_num = interpolate_num
        self.batch_size = batch_size
        self.rewire_factor = rewire_factor

        self.device = get_torch_device(device)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}

        # Get configuration space bounds
        self.q_min = to_numpy(robot.q_pos_min)
        self.q_max = to_numpy(robot.q_pos_max)
        self.dof = len(self.q_min)

        # Compute max edge length if not provided
        if max_edge_length is None:
            self.max_edge_length = np.linalg.norm(self.q_max - self.q_min) * 0.2  # Increased for better connectivity
        else:
            self.max_edge_length = max_edge_length

        # BIT* data structures
        self.vertices = []
        self.vertex_id_counter = 0
        self.samples = []  # Unconnected samples
        self.edge_queue = []  # Priority queue of edges to process
        self.vertex_queue = []  # Priority queue of vertices to expand

        # Tree structure
        self.start_vertex = None
        self.goal_vertex = None
        self.goal_region_radius = 0.2  # Radius to consider goal reached

        # Environment (will be set via set_obstacles)
        self.env = None

        print(f"Initialized BIT* algorithm (batch_size={batch_size})")

    def set_obstacles(self, env):
        """Set environment for collision checking."""
        self.env = env

    def plan(self, start_state, goal_state, debug=False, target_path_length=None):
        """
        Plan a trajectory from start to goal using BIT* algorithm.

        Args:
            start_state: Starting configuration (numpy array or list)
            goal_state: Goal configuration (numpy array or list)
            debug: Print debug information
            target_path_length: Target path length to beat (for anytime comparison)

        Returns:
            results_dict with keys:
                - success: bool
                - sol_path: np.ndarray (interpolate_num, dof)
                - planning_time: float
                - path_length: float
                - smoothness: float
                - time_to_first_solution: float or None
                - time_to_target_quality: float or None
                - first_solution_cost: float or None
        """
        start_time = time.time()
        tracking_interval = 1.0  # seconds
        next_track_time = tracking_interval
        interval_metrics = []

        # Convert to numpy
        start_state = np.array(start_state, dtype=np.float64)
        goal_state = np.array(goal_state, dtype=np.float64)

        # Initialize BIT* structures
        self._initialize(start_state, goal_state)

        # Main BIT* loop
        iteration = 0
        best_cost = float('inf')
        time_to_first_solution = None
        time_to_target_quality = None
        first_solution_cost = None

        def record_interval(iteration_idx):
            """Collect lightweight interval metrics."""
            elapsed = time.time() - start_time
            has_solution = self.goal_vertex is not None and self.goal_vertex.cost < float('inf')
            path_length_val = float('inf')
            smoothness_val = float('inf')
            mean_jerk_val = float('inf')

            if has_solution:
                # Cost is sum of edge lengths; use as path length fallback
                path_length_val = float(self.goal_vertex.cost)
                try:
                    raw_path = self._extract_path()
                    sol_path = self._interpolate_path(raw_path, self.interpolate_num)
                    sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
                    path_length_val = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
                    smoothness_val = compute_smoothness_finite_diff(sol_path_torch)[0].item()
                    mean_jerk_val = compute_jerk_finite_diff(sol_path_torch)[0].item()
                except Exception:
                    pass

            interval_metrics.append({
                'time': elapsed,
                'iteration': iteration_idx,
                'has_solution': has_solution,
                'path_length': path_length_val,
                'smoothness': smoothness_val,
                'mean_jerk': mean_jerk_val,
                'num_vertices': len(self.vertices),
                'num_samples': len(self.samples),
                'num_batches': getattr(self, "batch_counter", 0),
            })

        while time.time() - start_time < self.allowed_planning_time:
            iteration += 1

            # Sample new batch if needed
            if not self.edge_queue and not self.vertex_queue:
                if debug and iteration > 1:
                    print(f"Iteration {iteration}: Sampling new batch (current best: {best_cost:.3f})")
                self._sample_batch()
                self._update_edge_queue()

            # Expand best vertex or process best edge
            if self.vertex_queue and (not self.edge_queue or self._get_queue_value(self.vertex_queue) <= self._get_queue_value(self.edge_queue)):
                self._expand_vertex()
            elif self.edge_queue:
                success = self._process_edge()
                if success and self.goal_vertex is not None:
                    new_cost = self.goal_vertex.cost
                    if new_cost < best_cost:
                        # Record first solution time
                        if time_to_first_solution is None:
                            time_to_first_solution = time.time() - start_time
                            first_solution_cost = new_cost
                            if debug:
                                print(f"  Found FIRST solution at {time_to_first_solution:.3f}s with cost: {new_cost:.3f}")

                        best_cost = new_cost
                        if debug and time_to_first_solution is not None and time.time() - start_time > time_to_first_solution + 0.1:
                            print(f"  Found improved solution with cost: {best_cost:.3f}")

                        # Check if we beat the target path length
                        if target_path_length is not None and time_to_target_quality is None:
                            if best_cost <= target_path_length:
                                time_to_target_quality = time.time() - start_time
                                if debug:
                                    print(f"  *** BEAT TARGET at {time_to_target_quality:.3f}s! Cost {best_cost:.3f} <= {target_path_length:.3f}")

            # Record interval metrics at the requested cadence
            if time.time() - start_time >= next_track_time:
                record_interval(iteration)
                next_track_time += tracking_interval

            # Don't do early termination - let it optimize for the full time
            # This enables anytime behavior

        planning_time = time.time() - start_time

        # Extract solution
        if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
            raw_path = self._extract_path()
            sol_path = self._interpolate_path(raw_path, self.interpolate_num)

            # Compute metrics
            sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
            # Use finite difference smoothness for position-only trajectories
            smoothness = compute_smoothness_finite_diff(sol_path_torch)[0].item()
            # Compute mean jerk (rate of change of acceleration)
            mean_jerk = compute_jerk_finite_diff(sol_path_torch)[0].item()

            if debug:
                print(f"\n{'='*80}")
                print(f"Planning SUCCESS")
                print(f"  Planning time: {planning_time:.3f} sec")
                print(f"  Iterations: {iteration}")
                print(f"  Vertices in tree: {len(self.vertices)}")
                print(f"  Samples remaining: {len(self.samples)}")
                print(f"  Total configurations: {self.vertex_id_counter}")
                print(f"  Path length: {path_length:.3f}")
                print(f"  Smoothness: {smoothness:.3f}")
                print(f"  Mean jerk: {mean_jerk:.3f}")
                print(f"{'='*80}\n")

            return {
                'success': True,
                'sol_path': sol_path,
                'planning_time': planning_time,
                'path_length': path_length,
                'smoothness': smoothness,
                'mean_jerk': mean_jerk,
                'num_waypoints': len(sol_path),
                'time_to_first_solution': time_to_first_solution,
                'time_to_target_quality': time_to_target_quality,
                'first_solution_cost': first_solution_cost,
                'interval_metrics': interval_metrics,
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
                'time_to_first_solution': None,
                'time_to_target_quality': None,
                'first_solution_cost': None,
                'interval_metrics': interval_metrics,
            }

    def _initialize(self, start_state, goal_state):
        """Initialize BIT* data structures."""
        self.vertices = []
        self.vertex_id_counter = 0
        self.samples = []
        self.edge_queue = []
        self.vertex_queue = []
        self.batch_counter = 0

        # Create start vertex
        self.start_vertex = Vertex(start_state, self.vertex_id_counter)
        self.vertex_id_counter += 1
        self.start_vertex.cost = 0.0
        self.vertices.append(self.start_vertex)

        # Store goal state (will connect when sampled)
        self.goal_state = goal_state
        self.goal_vertex = None

        # Add goal to samples
        goal_sample = Vertex(goal_state, self.vertex_id_counter)
        self.vertex_id_counter += 1
        self.samples.append(goal_sample)

    def _sample_batch(self):
        """Sample a batch of random configurations using GPU batching."""
        # Sample many configurations at once
        samples_needed = self.batch_size
        max_attempts = self.batch_size * 3  # Try 3x the batch size

        # Generate batch of random configurations
        q_batch = np.random.uniform(
            self.q_min,
            self.q_max,
            size=(max_attempts, self.dof)
        )

        # Check collisions in batch
        collision_free_mask = self._is_collision_free_batch(q_batch)

        # Get collision-free samples
        q_free = q_batch[collision_free_mask]

        # Add up to batch_size samples
        num_to_add = min(len(q_free), samples_needed)
        for i in range(num_to_add):
            v = Vertex(q_free[i], self.vertex_id_counter)
            self.vertex_id_counter += 1
            self.samples.append(v)
        self.batch_counter += 1

    def _update_edge_queue(self):
        """Update edge queue with potential connections."""
        # Get current best solution cost for pruning
        best_solution_cost = self.goal_vertex.cost if self.goal_vertex is not None else float('inf')

        # Add edges from tree vertices to samples
        for v_tree in self.vertices:
            for v_sample in self.samples:
                dist = self._distance(v_tree.state, v_sample.state)
                if dist <= self.max_edge_length:
                    # Estimated cost through this edge
                    edge_cost = v_tree.cost + dist + self._heuristic(v_sample.state)
                    # Prune edges that can't improve the solution
                    if edge_cost < best_solution_cost:
                        heappush(self.edge_queue, (edge_cost, Edge(v_tree, v_sample, dist)))

        # Also add edges between tree vertices for rewiring (anytime optimization)
        if self.goal_vertex is not None:
            for v1 in self.vertices:
                for v2 in self.vertices:
                    if v1 is v2 or v1 is v2.parent:
                        continue
                    dist = self._distance(v1.state, v2.state)
                    if dist <= self.max_edge_length:
                        new_cost = v1.cost + dist
                        # Only consider if this would improve v2's cost
                        if new_cost < v2.cost:
                            edge_cost = new_cost + self._heuristic(v2.state)
                            if edge_cost < best_solution_cost:
                                heappush(self.edge_queue, (edge_cost, Edge(v1, v2, dist)))

    def _expand_vertex(self):
        """Expand the best vertex from the vertex queue."""
        if not self.vertex_queue:
            return

        _, v = heappop(self.vertex_queue)

        # Get current best solution cost for pruning
        best_solution_cost = self.goal_vertex.cost if self.goal_vertex is not None else float('inf')

        # Add edges to nearby samples
        for v_sample in self.samples:
            dist = self._distance(v.state, v_sample.state)
            if dist <= self.max_edge_length:
                edge_cost = v.cost + dist + self._heuristic(v_sample.state)
                # Prune edges that can't improve the solution
                if edge_cost < best_solution_cost:
                    heappush(self.edge_queue, (edge_cost, Edge(v, v_sample, dist)))

    def _process_edge(self):
        """Process the best edge from the edge queue."""
        if not self.edge_queue:
            return False

        _, edge = heappop(self.edge_queue)
        v1, v2, cost = edge.v1, edge.v2, edge.cost

        # Check if this edge is still useful
        estimated_cost = v1.cost + cost + self._heuristic(v2.state)
        if self.goal_vertex is not None and estimated_cost >= self.goal_vertex.cost:
            return False

        # Check if v2 can be improved
        new_cost = v1.cost + cost
        if new_cost >= v2.cost:
            return False

        # Check collision on edge
        if not self._is_edge_collision_free(v1.state, v2.state):
            return False

        # Add v2 to tree if it's a sample
        if v2 in self.samples:
            self.samples.remove(v2)
            self.vertices.append(v2)

            # Check if this is the goal
            if np.linalg.norm(v2.state - self.goal_state) < self.goal_region_radius:
                self.goal_vertex = v2

        # Update parent and cost (rewire)
        if v2.parent is not None:
            v2.parent.children.remove(v2)

        v2.parent = v1
        v2.cost = new_cost
        v1.children.append(v2)

        # Rewire children of v2 (propagate cost updates)
        self._rewire_children(v2)

        # Add to vertex queue for expansion
        queue_value = v2.cost + self._heuristic(v2.state)
        heappush(self.vertex_queue, (queue_value, v2))

        return True

    def _rewire_children(self, v):
        """Propagate cost updates to all children of a rewired vertex."""
        for child in v.children:
            edge_cost = self._distance(v.state, child.state)
            new_cost = v.cost + edge_cost
            if new_cost < child.cost:
                child.cost = new_cost
                # Recursively update children
                self._rewire_children(child)

    def _distance(self, q1, q2):
        """Compute distance between two configurations."""
        return np.linalg.norm(q1 - q2)

    def _heuristic(self, q):
        """Heuristic estimate of cost-to-go to goal."""
        return np.linalg.norm(q - self.goal_state)

    def _is_collision_free_batch(self, q_batch):
        """
        Check if a batch of configurations are collision-free.
        Returns a boolean numpy array indicating which configs are collision-free.
        """
        if self.env is None:
            return np.ones(len(q_batch), dtype=bool)

        q_torch = to_torch(q_batch, **self.tensor_args)

        # Get collision sphere positions via forward kinematics
        # x_pos shape: (batch, num_collision_spheres, 3)
        x_pos = self.torch_robot.fk_map_collision(q_torch)

        # Check environment collision for all spheres
        batch_size, num_spheres, _ = x_pos.shape
        radii = self.torch_robot.link_collision_spheres_radii

        # Reshape for batch SDF query: (batch * num_spheres, 3)
        x_pos_flat = x_pos.reshape(-1, 3)

        # Get SDF values for all sphere positions
        sdf_vals = self.env.compute_sdf(x_pos_flat)  # (batch * num_spheres,)

        # Reshape back: (batch, num_spheres)
        sdf_vals = sdf_vals.reshape(batch_size, num_spheres)

        # Check if distance is less than radius for each sphere
        # radii shape: (num_spheres,) -> broadcast to (batch, num_spheres)
        in_collision = sdf_vals < radii.unsqueeze(0)

        # A configuration is collision-free if ALL its spheres are collision-free
        # any(dim=1) gives True if ANY sphere is in collision
        collision_free = ~in_collision.any(dim=1)

        return to_numpy(collision_free).astype(bool)

    def _is_collision_free(self, q):
        """Check if a single configuration is collision-free."""
        return self._is_collision_free_batch(q[None, :])[0]

    def _is_edge_collision_free(self, q1, q2, resolution=5):
        """Check if an edge is collision-free using GPU batching."""
        # Create interpolated configurations
        alphas = np.linspace(0, 1, resolution)
        q_interp = np.array([q1 * (1 - alpha) + q2 * alpha for alpha in alphas])

        # Check all in batch
        collision_free_mask = self._is_collision_free_batch(q_interp)

        # All must be collision-free
        return collision_free_mask.all()

    def _extract_path(self):
        """Extract path from start to goal."""
        path = []
        v = self.goal_vertex
        while v is not None:
            path.append(v.state)
            v = v.parent
        return np.array(path[::-1])

    def _interpolate_path(self, path, num_waypoints):
        """Interpolate path to have exactly num_waypoints."""
        if len(path) == 0:
            return path

        if len(path) == num_waypoints:
            return path

        # Compute cumulative distances
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])

        total_distance = distances[-1]
        if total_distance < 1e-6:
            return np.tile(path[0], (num_waypoints, 1))

        # Interpolate uniformly
        interpolated_path = []
        for i in range(num_waypoints):
            target_dist = total_distance * i / (num_waypoints - 1)

            # Find segment
            idx = np.searchsorted(distances, target_dist)
            if idx == 0:
                interpolated_path.append(path[0])
            elif idx >= len(path):
                interpolated_path.append(path[-1])
            else:
                # Linear interpolation within segment
                alpha = (target_dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                q = path[idx-1] * (1 - alpha) + path[idx] * alpha
                interpolated_path.append(q)

        return np.array(interpolated_path)

    def _get_queue_value(self, queue):
        """Get the best value from a queue without popping."""
        if not queue:
            return float('inf')
        return queue[0][0]

    def terminate(self):
        """Cleanup (no-op for pure algorithmic implementation)."""
        pass


def run_minimal_example():
    """
    Run a minimal example with Panda robot in Spheres3D environment.
    """
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda
    from torch_robotics.torch_utils.seed import fix_random_seed

    print("\n" + "="*80)
    print("BIT* Algorithm - Testing Implementation")
    print("="*80 + "\n")

    # Fix random seed for reproducibility
    fix_random_seed(42)

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    # Create environment
    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)

    # Create robot
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Sample random start and goal configurations
    q_start = robot.random_q(n_samples=1)[0]
    q_goal = robot.random_q(n_samples=1)[0]
    start_state = to_numpy(q_start)
    goal_state = to_numpy(q_goal)

    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Initialize BIT* planner
    baseline = MinimalBITStarBaseline(
        robot=robot,
        allowed_planning_time=30.0,
        interpolate_num=128,
        device="cuda:0",
        batch_size=100,
    )
    baseline.set_obstacles(env)

    # Plan from start to goal
    result = baseline.plan(start_state, goal_state, debug=True)

    # Print results
    if result and result.get('success'):
        print("\n" + "="*80)
        print("BIT* IMPLEMENTATION WORKS!")
        print("="*80)
        print(f"Path length: {result['path_length']:.3f}")
        print(f"Smoothness: {result['smoothness']:.3f}")
        print(f"Planning time: {result['planning_time']:.3f} sec")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("Planning failed - may need more time or different parameters")
        print("="*80 + "\n")

    # Cleanup
    baseline.terminate()

    return result


def load_diffusion_problem_and_plan():
    """
    Load a problem from diffusion model results and plan with BIT* implementation.
    """
    print("\n" + "="*80)
    print("Loading Problem from Diffusion Model Results")
    print("="*80 + "\n")

    diffusion_results_file = "logs/2/results_single_plan-000.pt"

    # Check if diffusion results exist
    if not os.path.exists(diffusion_results_file):
        print(f"Error: {diffusion_results_file} not found")
        print("Run the diffusion model first: python inference.py")
        return None

    # Load diffusion results
    diff_results = torch.load(diffusion_results_file, map_location='cpu')

    # Extract start and goal states
    start_state = to_numpy(diff_results['q_pos_start'])
    goal_state = to_numpy(diff_results['q_pos_goal'])

    # Extract diffusion path length for comparison
    diff_path_length = float(diff_results['metrics']['trajs_best']['path_length'])

    print(f"Diffusion model path length: {diff_path_length:.3f}")
    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Setup environment and robot
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots.robot_panda import RobotPanda

    device = get_torch_device("cuda:0")
    tensor_args = {"device": device, "dtype": torch.float32}

    env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args=tensor_args)
    robot = RobotPanda(use_object_collision_spheres=True, tensor_args=tensor_args)

    # Initialize BIT* planner
    baseline = MinimalBITStarBaseline(
        robot=robot,
        allowed_planning_time=60.0,
        interpolate_num=128,
        device="cuda:0",
        batch_size=100,
    )
    baseline.set_obstacles(env)

    # Plan with BIT* implementation
    result = baseline.plan(start_state, goal_state, debug=True)

    # Compare with diffusion
    if result and result.get('success'):
        print("\n" + "="*80)
        print("COMPARISON: BIT* vs Diffusion Model")
        print("="*80)
        print(f"Diffusion path length: {diff_path_length:.3f}")
        print(f"BIT* path length:      {result['path_length']:.3f}")

        if result['path_length'] < diff_path_length:
            improvement = (diff_path_length - result['path_length']) / diff_path_length * 100
            print(f"\nBIT* found a {improvement:.1f}% shorter path!")
        elif result['path_length'] > diff_path_length:
            diff = (result['path_length'] - diff_path_length) / diff_path_length * 100
            print(f"\nDiffusion found a {diff:.1f}% shorter path")
        else:
            print(f"\nBoth methods found similar path lengths!")

        print("="*80 + "\n")

    # Cleanup
    baseline.terminate()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIT* Algorithm Implementation")
    parser.add_argument("--mode", default="minimal",
                       choices=["minimal", "compare"],
                       help="Run minimal example or compare with diffusion")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BIT* ALGORITHM IMPLEMENTATION")
    print("="*80)
    print("\nPure algorithmic implementation of Batch Informed Trees (BIT*).")
    print("This implementation does not use OMPL - it's a from-scratch algorithm.\n")
    print("Modes:")
    print("  minimal  - Test BIT* on a random problem")
    print("  compare  - Compare BIT* with diffusion model results")
    print("="*80 + "\n")

    if args.mode == "minimal":
        result = run_minimal_example()
    else:
        result = load_diffusion_problem_and_plan()

    if result is None:
        print("\nExecution completed with no result.")
