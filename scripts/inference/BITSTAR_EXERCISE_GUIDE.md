# BIT* Algorithm Implementation Exercise

This guide will help you implement the BIT* (Batch Informed Trees) algorithm from scratch.

## Overview

BIT* is a sampling-based optimal motion planning algorithm that:
1. Samples configurations in batches
2. Builds an implicit random geometric graph
3. Searches for optimal paths using best-first search with A* heuristics
4. Rewires the tree when better paths are found

## Files

- `bitstar_algorithm_exercise.py` - Template with TODOs for you to implement
- `bitstar_minimal_template.py` - Complete reference implementation
- This guide - Explanations and hints

## The BIT* Algorithm

### High-Level Algorithm

```
1. Initialize with start vertex in tree
2. Add goal to samples
3. WHILE time remaining:
   a. IF queues empty:
      - Sample new batch of configurations
      - Add potential edges to edge queue
   b. IF best vertex better than best edge:
      - Expand vertex (add edges to nearby samples)
   c. ELSE:
      - Process best edge (try to connect and rewire)
4. Extract and return path
```

### Key Data Structures

- **vertices**: Configurations connected to the tree
- **samples**: Unconnected configurations
- **edge_queue**: Priority queue of edges sorted by estimated cost
- **vertex_queue**: Priority queue of vertices sorted by estimated cost

### Important Concepts

**Cost-to-come (g)**: Actual cost from start to a vertex
**Cost-to-go (h)**: Heuristic estimate from vertex to goal
**Estimated total cost (f)**: g + h (used for prioritization)

## Implementation Steps

### Step 1: Implement `_heuristic(q)`

**Purpose**: Estimate remaining cost from configuration `q` to goal.

**Hints**:
- Use Euclidean distance: `np.linalg.norm(q - self.goal_state)`
- This guides the search toward the goal

**Example**:
```python
def _heuristic(self, q):
    return np.linalg.norm(q - self.goal_state)
```

---

### Step 2: Implement `_sample_batch()`

**Purpose**: Sample random collision-free configurations.

**Algorithm**:
```
FOR i in range(batch_size):
    q = sample random configuration uniformly
    IF q is collision-free:
        Create Vertex(q, unique_id)
        Add to self.samples
```

**Hints**:
- Use `np.random.uniform(self.q_min, self.q_max)` to sample
- Use `self._is_collision_free(q)` to check collisions
- Use `self.vertex_id_counter` for unique IDs (increment after each use)

**Example**:
```python
def _sample_batch(self):
    for _ in range(self.batch_size):
        q = np.random.uniform(self.q_min, self.q_max)
        if self._is_collision_free(q):
            v = Vertex(q, self.vertex_id_counter)
            self.vertex_id_counter += 1
            self.samples.append(v)
```

---

### Step 3: Implement `_is_edge_collision_free(q1, q2)`

**Purpose**: Check if straight-line path between two configurations is collision-free.

**Algorithm**:
```
FOR alpha in [0, 0.1, 0.2, ..., 1.0]:
    q = (1 - alpha) * q1 + alpha * q2
    IF q has collision:
        RETURN False
RETURN True
```

**Hints**:
- Use `np.linspace(0, 1, resolution)` to get interpolation parameters
- Interpolate: `q = q1 * (1 - alpha) + q2 * alpha`
- Check each point with `self._is_collision_free(q)`

**Example**:
```python
def _is_edge_collision_free(self, q1, q2, resolution=10):
    for alpha in np.linspace(0, 1, resolution):
        q = q1 * (1 - alpha) + q2 * alpha
        if not self._is_collision_free(q):
            return False
    return True
```

---

### Step 4: Implement `_update_edge_queue()`

**Purpose**: Add potential edges from tree vertices to samples.

**Algorithm**:
```
FOR each vertex v in tree:
    FOR each sample s in samples:
        dist = distance(v, s)
        IF dist <= max_edge_length:
            estimated_cost = v.cost + dist + heuristic(s)
            edge = Edge(v, s, dist)
            heappush(edge_queue, (estimated_cost, edge))
```

**Hints**:
- Use `self._distance(v.state, s.state)`
- Cost estimate = current cost + edge cost + heuristic
- Use `heappush(self.edge_queue, (cost, edge))`

**Example**:
```python
def _update_edge_queue(self):
    for v_tree in self.vertices:
        for v_sample in self.samples:
            dist = self._distance(v_tree.state, v_sample.state)
            if dist <= self.max_edge_length:
                edge_cost = v_tree.cost + dist + self._heuristic(v_sample.state)
                heappush(self.edge_queue, (edge_cost, Edge(v_tree, v_sample, dist)))
```

---

### Step 5: Implement `_expand_vertex()`

**Purpose**: Expand a vertex by adding edges to nearby samples.

**Algorithm**:
```
IF vertex_queue not empty:
    _, v = heappop(vertex_queue)
    FOR each sample s in samples:
        dist = distance(v, s)
        IF dist <= max_edge_length:
            estimated_cost = v.cost + dist + heuristic(s)
            edge = Edge(v, s, dist)
            heappush(edge_queue, (estimated_cost, edge))
```

**Hints**:
- Use `heappop(self.vertex_queue)` to get best vertex
- Similar to `_update_edge_queue` but only for one vertex

**Example**:
```python
def _expand_vertex(self):
    if not self.vertex_queue:
        return

    _, v = heappop(self.vertex_queue)

    for v_sample in self.samples:
        dist = self._distance(v.state, v_sample.state)
        if dist <= self.max_edge_length:
            edge_cost = v.cost + dist + self._heuristic(v_sample.state)
            heappush(self.edge_queue, (edge_cost, Edge(v, v_sample, dist)))
```

---

### Step 6: Implement `_process_edge()`

**Purpose**: Process edge, potentially adding sample to tree or rewiring.

**Algorithm**:
```
IF edge_queue not empty:
    _, edge = heappop(edge_queue)
    v1, v2, cost = edge.v1, edge.v2, edge.cost

    # Pruning
    estimated_cost = v1.cost + cost + heuristic(v2)
    IF goal_found AND estimated_cost >= goal_cost:
        RETURN False

    new_cost = v1.cost + cost
    IF new_cost >= v2.cost:
        RETURN False

    # Collision check
    IF NOT edge_collision_free(v1, v2):
        RETURN False

    # Move sample to tree if needed
    IF v2 in samples:
        Remove v2 from samples
        Add v2 to vertices
        IF v2 near goal:
            goal_vertex = v2

    # Rewire
    IF v2 has parent:
        Remove v2 from old parent's children
    v2.parent = v1
    v2.cost = new_cost
    Add v2 to v1's children

    # Add to vertex queue for expansion
    queue_value = v2.cost + heuristic(v2)
    heappush(vertex_queue, (queue_value, v2))

    RETURN True
```

**Hints**:
- This is the most complex function!
- Pruning avoids processing useless edges
- Rewiring updates parent/child relationships
- Check if sample is near goal: `np.linalg.norm(v2.state - self.goal_state) < self.goal_region_radius`

---

### Step 7: Implement `_extract_path()`

**Purpose**: Extract solution path by following parent pointers.

**Algorithm**:
```
path = []
v = goal_vertex
WHILE v is not None:
    path.append(v.state)
    v = v.parent
RETURN reverse(path)
```

**Example**:
```python
def _extract_path(self):
    path = []
    v = self.goal_vertex
    while v is not None:
        path.append(v.state)
        v = v.parent
    return np.array(path[::-1])
```

---

### Step 8: Implement Main Loop in `plan()`

**Purpose**: Main BIT* planning loop.

**Algorithm**:
```
WHILE time_remaining:
    iteration += 1

    # Sample new batch if needed
    IF edge_queue empty AND vertex_queue empty:
        sample_batch()
        update_edge_queue()

    # Process queues
    IF vertex_queue not empty AND
       (edge_queue empty OR best_vertex < best_edge):
        expand_vertex()
    ELIF edge_queue not empty:
        success = process_edge()
        IF success AND goal_found:
            new_cost = goal_vertex.cost
            IF new_cost < best_cost:
                best_cost = new_cost
                print improvement

    # Early termination
    IF goal_found AND enough_time_elapsed:
        BREAK
```

**Hints**:
- Use `self._get_queue_value(queue)` to peek at best value
- Compare vertex queue value vs edge queue value
- Allow some time after finding solution to improve it

---

## Testing Your Implementation

### Run the exercise:
```bash
python scripts/inference/bitstar_algorithm_exercise.py
```

### Expected output when working:
```
Planning SUCCESS
  Planning time: X.XXX sec
  Iterations: XXX
  Vertices: XXX
  Path length: X.XXX
  Smoothness: X.XXX

YOUR BIT* IMPLEMENTATION WORKS!
```

### Debugging Tips

1. **No samples being added**: Check `_sample_batch()` and collision checking
2. **No edges in queue**: Check `_update_edge_queue()` and max_edge_length
3. **No progress**: Check `_process_edge()` - ensure edges are being processed
4. **Crash in path extraction**: Check that goal_vertex is set correctly in `_process_edge()`
5. **Poor paths**: Increase batch_size or allowed_planning_time

### Common Issues

- **Forgetting to increment vertex_id_counter** in `_sample_batch()`
- **Not checking both self-collision and environment collision**
- **Not removing sample from self.samples** when moving to self.vertices
- **Not updating parent/child relationships** correctly
- **Comparing wrong queue values** (vertex vs edge queue)

## Advanced Challenges

Once you have a working implementation, try:

1. **Informed sampling**: Sample in an ellipse between start and goal
2. **Dynamic batch size**: Increase batch size over time
3. **Adaptive edge length**: Adjust max_edge_length based on tree density
4. **Path smoothing**: Add post-processing to smooth the final path
5. **Multi-query**: Keep the tree for multiple queries

## References

- Original paper: Gammell et al., "Batch Informed Trees (BIT*): Sampling-based optimal planning via the heuristically guided search of implicit random geometric graphs", ICRA 2015
- Reference implementation: `bitstar_minimal_template.py`

## Need Help?

If you get stuck:
1. Check the hints in the TODO comments
2. Review this guide's algorithm pseudocode
3. Look at the reference implementation in `bitstar_minimal_template.py`
4. Print debug information (vertex count, edge count, queue sizes)

Good luck implementing BIT*!
