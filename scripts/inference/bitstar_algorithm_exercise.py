"""
BIT* Algorithm Implementation Exercise
Educational template for implementing Batch Informed Trees (BIT*) from scratch.

This is an exercise to learn the BIT* algorithm by implementing it yourself.
Follow the TODOs to complete the implementation.

BIT* Reference:
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
from torch_robotics.trajectory.metrics import compute_path_length, compute_smoothness


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


class BITStarExercise:
    """BIT* algorithm implementation exercise."""

    def __init__(
        self,
        robot,  # torch_robotics robot instance
        robot_urdf_path: str = None,
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
        self.q_min = to_numpy(robot.q_min)
        self.q_max = to_numpy(robot.q_max)
        self.dof = len(self.q_min)

        # Compute max edge length if not provided
        if max_edge_length is None:
            self.max_edge_length = np.linalg.norm(self.q_max - self.q_min) * 0.1
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
        self.goal_region_radius = 0.1  # Radius to consider goal reached

        # Environment (will be set via set_obstacles)
        self.env = None

        print(f"Initialized BIT* algorithm exercise (batch_size={batch_size})")

    def set_obstacles(self, env):
        """Set environment for collision checking."""
        self.env = env

    def plan(self, start_state, goal_state, debug=False):
        """
        Plan a trajectory from start to goal using BIT* algorithm.

        TODO: Implement the main planning loop.

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
        start_time = time.time()

        # Convert to numpy
        start_state = np.array(start_state, dtype=np.float64)
        goal_state = np.array(goal_state, dtype=np.float64)

        # Initialize BIT* structures
        self._initialize(start_state, goal_state)

        # TODO 1: Implement the main BIT* loop
        # Hints:
        # - Loop while time remaining
        # - Sample new batch if queues are empty
        # - Decide whether to expand vertex or process edge based on queue values
        # - Track best cost found
        # - Consider early termination once a good solution is found

        iteration = 0
        best_cost = float('inf')

        # TODO: Your main planning loop here
        # while time.time() - start_time < self.allowed_planning_time:
        #     iteration += 1
        #
        #     # Check if we need to sample a new batch
        #     # if queues are empty:
        #     #     sample new batch
        #     #     update edge queue
        #
        #     # Decide what to process next:
        #     # if should expand vertex:
        #     #     expand vertex
        #     # elif should process edge:
        #     #     process edge
        #     #     if goal found and improved:
        #     #         update best cost
        #
        #     # Consider early termination

        planning_time = time.time() - start_time

        # Extract solution
        if self.goal_vertex is not None and self.goal_vertex.cost < float('inf'):
            raw_path = self._extract_path()
            sol_path = self._interpolate_path(raw_path, self.interpolate_num)

            # Compute metrics
            sol_path_torch = to_torch(sol_path[None, ...], **self.tensor_args)
            path_length = compute_path_length(sol_path_torch, self.torch_robot)[0].item()
            smoothness = compute_smoothness(sol_path_torch, self.torch_robot)[0].item()

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
                'sol_path': sol_path,
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

        # Store goal state (will connect when sampled)
        self.goal_state = goal_state
        self.goal_vertex = None

        # Add goal to samples
        goal_sample = Vertex(goal_state, self.vertex_id_counter)
        self.vertex_id_counter += 1
        self.samples.append(goal_sample)

    def _sample_batch(self):
        """
        Sample a batch of random configurations.

        TODO: Implement batch sampling.

        Hints:
        - Sample self.batch_size random configurations
        - Use np.random.uniform with self.q_min and self.q_max
        - Check each sample for collision using self._is_collision_free()
        - Create Vertex objects with unique IDs
        - Add valid samples to self.samples list
        """
        # TODO: Implement sampling
        # for _ in range(self.batch_size):
        #     q = sample random configuration
        #     if collision-free:
        #         create vertex
        #         add to samples
        pass

    def _update_edge_queue(self):
        """
        Update edge queue with potential connections.

        TODO: Implement edge queue updating.

        Hints:
        - For each vertex in self.vertices (the tree)
        - For each sample in self.samples (unconnected samples)
        - Compute distance between them
        - If distance <= self.max_edge_length:
        -   Compute estimated cost: v.cost + distance + heuristic(sample)
        -   Create Edge object
        -   Push to self.edge_queue using heappush with (cost, edge)
        """
        # TODO: Implement edge queue update
        # for v_tree in self.vertices:
        #     for v_sample in self.samples:
        #         dist = distance(v_tree, v_sample)
        #         if dist <= max_edge_length:
        #             edge_cost = estimate cost through this edge
        #             heappush(self.edge_queue, (edge_cost, Edge(...)))
        pass

    def _expand_vertex(self):
        """
        Expand the best vertex from the vertex queue.

        TODO: Implement vertex expansion.

        Hints:
        - Pop best vertex from self.vertex_queue using heappop
        - For each sample in self.samples:
        -   Compute distance
        -   If within max_edge_length:
        -     Compute estimated edge cost
        -     Push edge to self.edge_queue
        """
        # TODO: Implement vertex expansion
        # if vertex_queue not empty:
        #     _, v = heappop(self.vertex_queue)
        #
        #     for v_sample in self.samples:
        #         dist = distance(v, v_sample)
        #         if dist <= max_edge_length:
        #             create and push edge
        pass

    def _process_edge(self):
        """
        Process the best edge from the edge queue.

        TODO: Implement edge processing with tree rewiring.

        Hints:
        - Pop best edge from self.edge_queue
        - Check if edge is still useful (pruning)
        - Check if connecting improves v2's cost
        - Check collision on the edge
        - If v2 is in samples, move it to vertices
        - Update v2's parent and cost
        - Add v2 to vertex_queue for expansion
        - Check if v2 is the goal

        Returns:
            bool: True if edge was successfully processed
        """
        # TODO: Implement edge processing
        # if edge_queue not empty:
        #     _, edge = heappop(self.edge_queue)
        #     v1, v2, cost = edge.v1, edge.v2, edge.cost
        #
        #     # Pruning checks:
        #     # - Is estimated cost still better than current best?
        #     # - Does this edge improve v2's cost?
        #
        #     # Collision check
        #     if not self._is_edge_collision_free(v1.state, v2.state):
        #         return False
        #
        #     # Move v2 from samples to vertices if needed
        #     # Update parent/child relationships
        #     # Update v2.cost
        #     # Push v2 to vertex_queue
        #     # Check if v2 is near goal
        #
        #     return True
        return False

    def _distance(self, q1, q2):
        """Compute distance between two configurations."""
        return np.linalg.norm(q1 - q2)

    def _heuristic(self, q):
        """
        Heuristic estimate of cost-to-go to goal.

        TODO: Implement heuristic function.

        Hints:
        - Return Euclidean distance to goal
        - Use self.goal_state
        """
        # TODO: Implement heuristic
        # return distance from q to self.goal_state
        return 0.0

    def _is_collision_free(self, q):
        """Check if a configuration is collision-free."""
        if self.env is None:
            return True

        q_torch = to_torch(q, **self.tensor_args)
        return not self.torch_robot.check_self_collision(q_torch) and \
               not self.env.check_collision(q_torch)

    def _is_edge_collision_free(self, q1, q2, resolution=10):
        """
        Check if an edge is collision-free.

        TODO: Implement edge collision checking.

        Hints:
        - Interpolate between q1 and q2
        - Check collision at multiple points along the edge
        - Use np.linspace for interpolation
        - Call self._is_collision_free() for each point
        """
        # TODO: Implement edge collision checking
        # for alpha in np.linspace(0, 1, resolution):
        #     q = interpolate between q1 and q2
        #     if collision:
        #         return False
        # return True
        return True

    def _extract_path(self):
        """
        Extract path from start to goal.

        TODO: Implement path extraction.

        Hints:
        - Start from self.goal_vertex
        - Follow parent pointers back to start
        - Collect states along the way
        - Reverse the path (start to goal)
        - Return as numpy array
        """
        # TODO: Implement path extraction
        # path = []
        # v = self.goal_vertex
        # while v is not None:
        #     path.append(v.state)
        #     v = v.parent
        # return np.array(path[::-1])
        return np.array([])

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
    print("BIT* Algorithm Exercise - Testing Your Implementation")
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
    q_start = robot.sample_q_pos()
    q_goal = robot.sample_q_pos()
    start_state = to_numpy(q_start)
    goal_state = to_numpy(q_goal)

    print(f"Start state: {start_state}")
    print(f"Goal state: {goal_state}")

    # Initialize BIT* planner
    baseline = BITStarExercise(
        robot=robot,
        allowed_planning_time=30.0,
        interpolate_num=128,
        device=device,
        batch_size=100,
    )
    baseline.set_obstacles(env)

    # Plan from start to goal
    result = baseline.plan(start_state, goal_state, debug=True)

    # Print results
    if result and result.get('success'):
        print("\n" + "="*80)
        print("YOUR BIT* IMPLEMENTATION WORKS!")
        print("="*80)
        print(f"Path length: {result['path_length']:.3f}")
        print(f"Smoothness: {result['smoothness']:.3f}")
        print(f"Planning time: {result['planning_time']:.3f} sec")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("Planning failed - check your implementation")
        print("="*80 + "\n")

    # Cleanup
    baseline.terminate()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIT* Algorithm Exercise")
    parser.add_argument("--mode", default="minimal",
                       choices=["minimal"],
                       help="Run minimal example to test implementation")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("BIT* ALGORITHM IMPLEMENTATION EXERCISE")
    print("="*80)
    print("\nComplete the TODOs to implement the BIT* algorithm:")
    print("  1. _sample_batch() - Sample random configurations")
    print("  2. _update_edge_queue() - Build edge queue from tree to samples")
    print("  3. _expand_vertex() - Expand a vertex by adding edges to samples")
    print("  4. _process_edge() - Process edge and rewire tree")
    print("  5. _heuristic() - Compute heuristic cost-to-go")
    print("  6. _is_edge_collision_free() - Check edge for collisions")
    print("  7. _extract_path() - Extract solution path from tree")
    print("  8. plan() - Main BIT* loop")
    print("\nReference the completed version in bitstar_minimal_template.py if needed.")
    print("="*80 + "\n")

    result = run_minimal_example()

    if result is None or not result.get('success'):
        print("\n" + "="*80)
        print("Implementation incomplete - keep working on the TODOs!")
        print("="*80 + "\n")
