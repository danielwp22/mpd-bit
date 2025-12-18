# BIT* Implementation Template Exercise

This exercise helps you understand how to implement BIT* baseline comparison with the diffusion model by filling in a template.

## Files

1. **bitstar_minimal_template.py** - Template with TODOs for you to fill in
2. **bitstar_minimal_template_solution.py** - Complete solution (check your work!)
3. **bitstar_baseline.py** - Full-featured implementation with anytime mode

## Learning Objectives

By completing this exercise, you'll learn:
- How to set up PyBullet for motion planning
- How to interface with OMPL planners (BIT*, RRT*, etc.)
- How to compute trajectory metrics (path length, smoothness)
- How to load and compare with diffusion model results
- How motion planning baselines work under the hood

## Step-by-Step Guide

### Step 1: Understand the Structure

Open `bitstar_minimal_template.py` and read through the code. Notice:
- The `MinimalBITStarBaseline` class structure
- The TODOs numbered from 1 to 29
- Helpful hints after each TODO comment

### Step 2: Complete the Initialization (TODOs 1-4)

Fill in the `__init__` method to:
1. Initialize PyBullet in DIRECT (headless) mode
2. Load the robot URDF file
3. Create a PbOMPLRobot wrapper
4. Setup the OMPL planning interface

**Hints:**
```python
# TODO 1
self.pybullet_client = bullet_client.BulletClient(p.DIRECT)
self.pybullet_client.setGravity(0, 0, -9.8)

# TODO 2
self.robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0])

# TODO 3
link_name_ee = self.torch_robot.link_name_ee if hasattr(self.torch_robot, 'link_name_ee') else 'ee_link'
self.robot = PbOMPLRobot(self.pybullet_client, self.robot_id, link_name_ee=link_name_ee)

# TODO 4
self.pb_ompl_interface = PbOMPL(
    self.pybullet_client,
    self.robot,
    self.obstacles,
    min_distance_robot_env=0.05
)
self.pb_ompl_interface.set_planner(planner_name)
```

### Step 3: Implement set_obstacles (TODO 5)

This is straightforward - just store the obstacles and pass them to OMPL:

```python
def set_obstacles(self, obstacles):
    self.obstacles = obstacles
    self.pb_ompl_interface.set_obstacles(self.obstacles)
```

### Step 4: Complete the plan Method (TODOs 6-12)

This is the core planning function. Fill in:
- State conversion (OMPL needs Python floats, not numpy)
- Setting the robot to start state
- Calling the OMPL planner
- Computing metrics (path length, smoothness)

**Key concepts:**
- OMPL expects native Python types, not numpy arrays
- Metrics are computed using torch_robotics functions
- Need to add batch dimension to trajectory for metric computation

### Step 5: Implement Cleanup (TODO 13)

Simply disconnect PyBullet:
```python
def terminate(self):
    self.pybullet_client.disconnect()
```

### Step 6: Complete the Example Functions (TODOs 14-29)

Fill in:
- `run_minimal_example()` - Creates environment, samples random start/goal, and plans
- `load_diffusion_problem_and_plan()` - Loads a problem from diffusion results and compares

### Step 7: Test Your Implementation

Run the minimal example:
```bash
python bitstar_minimal_template.py --mode minimal
```

You should see:
```
================================================================================
YOUR IMPLEMENTATION WORKS!
================================================================================
Path length: 7.234
Smoothness: 125.432
Planning time: 3.456 sec
================================================================================
```

### Step 8: Compare with Diffusion Model

First, run the diffusion model:
```bash
python inference.py
```

Then compare with your BIT* implementation:
```bash
python bitstar_minimal_template.py --mode compare
```

You should see:
```
================================================================================
COMPARISON: Your BIT* vs Diffusion Model
================================================================================
Diffusion path length: 7.500
BIT* path length:      7.234

🎉 Your BIT* found a 3.5% shorter path!
================================================================================
```

### Step 9: Check Your Work

Compare your implementation with the solution:
```bash
# Run the solution
python bitstar_minimal_template_solution.py --mode minimal

# Compare outputs - they should be similar (but not identical due to randomness)
```

## Common Issues and Solutions

### Issue 1: "ImportError: No module named pybullet"
**Solution:** Make sure you've installed dependencies:
```bash
pip install pybullet
```

### Issue 2: "Planning always fails"
**Possible causes:**
- Didn't set robot to start state (TODO 7)
- Wrong state format (need Python floats, not numpy)
- URDF path is incorrect

**Debug:**
- Set `debug=True` when calling `plan()`
- Check that `self.robot_id` is not None
- Verify `robot_urdf_path` exists

### Issue 3: "Metrics are all 0.0"
**Possible causes:**
- Didn't convert trajectory to torch tensor correctly
- Forgot batch dimension in `sol_path[None, ...]`
- Wrong tensor_args device

**Debug:**
- Print `sol_path.shape` - should be (128, 7) for Panda
- Print `sol_path_torch.shape` - should be (1, 128, 7)

### Issue 4: "PyBullet GUI opens"
**Solution:** Make sure you're using `p.DIRECT` mode, not `p.GUI`:
```python
self.pybullet_client = bullet_client.BulletClient(p.DIRECT)  # Not p.GUI!
```

## What's Different from bitstar_baseline.py?

The minimal template focuses on core concepts. The full version adds:
- Anytime planning mode (tracks optimization over time)
- Multi-problem evaluation with statistics
- Command-line arguments for flexibility
- More robust error handling
- Support for multiple environments
- Detailed logging and metrics

## Next Steps

After completing the template:

1. **Experiment with different planners:**
   - Change `planner_name` to "RRTConnect", "RRTstar", "AITstar"
   - Compare their performance

2. **Add anytime tracking:**
   - Modify `plan()` to track solution improvements
   - Record when path length improves
   - See `bitstar_baseline.py:plan_anytime()` for reference

3. **Batch evaluation:**
   - Plan on multiple start-goal pairs
   - Compute mean and std of metrics
   - See `bitstar_baseline.py:evaluate_multiple_problems()` for reference

4. **Try different environments:**
   - EnvSimple2D with RobotPointMass2D
   - EnvWarehouse with RobotPanda
   - Modify the template to support these

## Understanding the Code Flow

```
MinimalBITStarBaseline.__init__()
    ↓
    Initialize PyBullet (headless)
    ↓
    Load robot URDF
    ↓
    Create OMPL interface
    ↓
plan(start, goal)
    ↓
    Convert states to Python floats
    ↓
    Set robot to start state
    ↓
    Call OMPL planner
    ↓
    Compute metrics (path length, smoothness)
    ↓
    Return results dict
    ↓
terminate()
    ↓
    Cleanup PyBullet
```

## Key Concepts

### Why PyBullet?
- Provides physics simulation and collision checking
- OMPL uses it to check if configurations are collision-free
- Faster than torch-based SDF collision checking for sampling-based planners

### Why OMPL?
- Industry-standard motion planning library
- Implements many state-of-the-art algorithms (RRT*, BIT*, PRM, etc.)
- Well-tested and optimized

### Why torch_robotics?
- Provides robot models compatible with diffusion training
- Computes metrics consistently between diffusion and baselines
- Handles forward kinematics, velocities, accelerations

### State Representation
- **Configuration space (q)**: Joint angles [q1, q2, ..., q7] for 7-DOF Panda
- **Trajectory**: Sequence of configurations over time (128 waypoints)
- **Path length**: Sum of distances between consecutive waypoints
- **Smoothness**: Sum of acceleration magnitudes (lower = smoother)

## Success Criteria

You've successfully completed the exercise when:
- [ ] All TODOs are filled in (no `None` or `pass` statements remain)
- [ ] `--mode minimal` runs without errors
- [ ] Planning succeeds and shows valid metrics
- [ ] `--mode compare` loads diffusion results and compares
- [ ] Your understanding of the code flow is clear

## Additional Resources

- **OMPL Documentation**: http://ompl.kavrakilab.org/
- **PyBullet Quickstart**: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/
- **Full BIT* implementation**: See `bitstar_baseline.py`
- **Diffusion model code**: See `inference.py`

## Questions to Test Your Understanding

1. Why do we need to convert numpy arrays to Python floats for OMPL?
2. What does the `interpolate_num` parameter control?
3. Why do we add a batch dimension `[None, ...]` when computing metrics?
4. What's the difference between path length and smoothness?
5. How does BIT* differ from RRT*?
6. Why is diffusion model inference faster than BIT* planning?
7. When would you use BIT* instead of the diffusion model?

## Congratulations!

Once you complete this exercise, you'll have a solid understanding of how motion planning baselines work and how to compare them fairly with learning-based approaches like diffusion models!

Happy coding! 🚀
