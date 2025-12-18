# Motion Planning Baselines and Metrics Guide

## Overview
This guide explains how to:
1. Extract metrics from the diffusion model inference
2. Run BIT* baseline for comparison
3. Compare results between methods

---

## 1. Diffusion Model Metrics (inference.py)

### Available Metrics

The `inference.py` script **already computes** comprehensive metrics including:

#### Success Metrics
- `success`: Binary (0 or 1) - whether a collision-free trajectory was found
- `fraction_valid`: Success rate across all sampled trajectories (0.0 to 1.0)
- `success_no_joint_limits_vel_acc`: Success ignoring joint limits/velocity/acceleration

#### Trajectory Quality Metrics
- `path_length`: Total path length in configuration space (sum of distances between waypoints)
- `smoothness`: Sum of acceleration norms along trajectory (lower is smoother)
- `collision_intensity`: Average penetration depth into obstacles (lower is better, 0 is collision-free)

#### Goal Achievement Metrics
- `ee_pose_goal_error_position_norm`: End-effector position error at goal (meters)
- `ee_pose_goal_error_orientation_norm`: End-effector orientation error at goal (radians)

#### Diversity Metrics (for multiple trajectories)
- `diversity`: Vendi score measuring trajectory diversity

### Where Metrics Are Saved

Metrics are saved in two locations:

1. **Console output** - Printed at the end of each planning run
2. **Results file** - Saved as `logs/<seed>/results_single_plan-<idx>.pt`

### Extracting Metrics from Results File

```python
import torch

# Load results
results = torch.load('logs/2/results_single_plan-000.pt')

# Access metrics
metrics = results['metrics']

# Best trajectory metrics
path_length = metrics['trajs_best']['path_length']
smoothness = metrics['trajs_best']['smoothness']

# Overall statistics
success = metrics['trajs_all']['success']
collision_intensity = metrics['trajs_all']['collision_intensity']
fraction_valid = metrics['trajs_all']['fraction_valid']

print(f"Success: {success}")
print(f"Path length: {path_length:.3f}")
print(f"Collision intensity: {collision_intensity:.4f}")
print(f"Valid trajectories: {fraction_valid*100:.1f}%")
```

### Example Output
```
metrics:
{'trajs_all': {'collision_intensity': array(0.005, dtype=float32),
               'fraction_valid': 0.87,
               'success': 1},
 'trajs_best': {'path_length': array(6.995, dtype=float32),
                'smoothness': array(57.415, dtype=float32)},
 'trajs_valid': {'path_length_mean': array(8.249, dtype=float32),
                 'path_length_std': array(1.073, dtype=float32)}}
```

---

## 2. Running BIT* Baseline

### Basic Usage

```bash
cd /home/ubuntu/Projects/MotionPlanningDiffusion/mpd-splines-public/scripts/inference
python bitstar_baseline.py
```

### Customization

Edit `bitstar_baseline.py` to change:

```python
statistics = run_panda_spheres3d(
    n_problems=10,           # Number of start-goal pairs to test
    planner_name="BITstar",  # Options: BITstar, ABITstar, AITstar, RRTstar, etc.
    allowed_time=60.0,       # Max planning time per problem (seconds)
    seed=42,                 # Random seed for reproducibility
)
```

### Available OMPL Planners

The script supports any OMPL planner. Change `planner_name` to:
- `"BITstar"` - Batch Informed Trees*
- `"ABITstar"` - Advanced BIT*
- `"AITstar"` - Adaptively Informed Trees*
- `"RRTstar"` - RRT*
- `"RRTConnect"` - RRT-Connect (fast, not optimal)
- `"PRM"` - Probabilistic Roadmap
- `"PRMstar"` - PRM*
- `"FMT"` - Fast Marching Tree*

### Output

Results are saved to `logs_<planner>_panda_spheres3d/`:
- `statistics.yaml` - Summary statistics
- `trajectory_000.npy`, `trajectory_001.npy`, ... - Individual planned trajectories

### Statistics Output Example

```yaml
planner: BITstar
n_problems: 10
success_count: 8
success_rate: 0.8
planning_time_mean: 12.456
planning_time_std: 3.21
path_length_mean: 7.234
path_length_std: 1.12
```

---

## 3. Comparing Diffusion Model vs BIT*

### Side-by-Side Comparison Script

Create `compare_results.py`:

```python
import torch
import yaml
import numpy as np

# Load diffusion model results
diff_results = torch.load('logs/2/results_single_plan-000.pt')
diff_metrics = diff_results['metrics']

# Load BIT* results
with open('logs_bitstar_panda_spheres3d/statistics.yaml', 'r') as f:
    bitstar_stats = yaml.safe_load(f)

# Compare
print("="*80)
print("COMPARISON: Diffusion Model vs BIT*")
print("="*80)

print(f"\nSuccess Rate:")
print(f"  Diffusion:  {diff_metrics['trajs_all']['fraction_valid']*100:.1f}%")
print(f"  BIT*:       {bitstar_stats['success_rate']*100:.1f}%")

print(f"\nPath Length:")
print(f"  Diffusion:  {diff_metrics['trajs_best']['path_length']:.3f}")
print(f"  BIT*:       {bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}")

print(f"\nPlanning Time:")
print(f"  Diffusion:  {diff_results['t_inference_total']:.3f} sec")
print(f"  BIT*:       {bitstar_stats['planning_time_mean']:.3f} ± {bitstar_stats['planning_time_std']:.3f} sec")

print(f"\nSmoothness:")
print(f"  Diffusion:  {diff_metrics['trajs_best']['smoothness']:.3f}")
print(f"  BIT*:       N/A (not computed)")
```

### Key Differences to Consider

1. **Sampling vs Optimization**
   - Diffusion: Samples N=100 trajectories, picks best
   - BIT*: Optimizes single trajectory

2. **Time Comparison**
   - Diffusion: Includes generation time + (optional) refinement
   - BIT*: Pure planning time

3. **Metrics Available**
   - Diffusion: path_length, smoothness, diversity, collision_intensity
   - BIT*: path_length, planning_time, success_rate

4. **Trajectory Resolution**
   - Both use 128 waypoints by default (configurable)

---

## 4. Advanced Usage

### Testing on Different Environments

Modify `bitstar_baseline.py` to use other environments:

```python
# For 2D Point Mass
from torch_robotics.environments import EnvSimple2D
env = EnvSimple2D(...)
robot = RobotPointMass(...)

# For Panda Warehouse
from torch_robotics.environments import EnvWarehouse
env = EnvWarehouse(...)
robot = RobotPanda(...)
```

### Batch Evaluation

Run multiple planners in sequence:

```python
for planner in ["BITstar", "RRTConnect", "RRTstar"]:
    print(f"\nEvaluating {planner}...")
    run_panda_spheres3d(
        n_problems=20,
        planner_name=planner,
        allowed_time=60.0,
    )
```

### Using Same Start/Goal as Diffusion Model

To ensure fair comparison, load the same start/goal from inference results:

```python
# Load from diffusion results
diff_results = torch.load('logs/2/results_single_plan-000.pt')
start_state = diff_results['q_pos_start'].cpu().numpy()
goal_state = diff_results['q_pos_goal'].cpu().numpy()

# Use in baseline
baseline = BITStarBaseline(...)
result = baseline.plan(start_state, goal_state)
```

---

## 5. Quick Reference

### Run Diffusion Model (Panda Spheres3D)
```bash
cd scripts/inference
# Edit inference.py: set cfg_inference_path to config_EnvSpheres3D-RobotPanda_00.yaml
python inference.py
# Results in: logs/2/
```

### Run BIT* Baseline
```bash
cd scripts/inference
python bitstar_baseline.py
# Results in: logs_bitstar_panda_spheres3d/
```

### Compare Results
```bash
python compare_results.py
```

---

## Notes

- **Collision Checking**: Diffusion model uses SDF-based collision checking, BIT* uses OMPL's built-in collision checking
- **Optimality**: BIT* is asymptotically optimal, diffusion model is not guaranteed optimal but can generate diverse solutions
- **Speed**: Diffusion model is typically faster for inference after training, BIT* requires no training
- **Smoothness**: Diffusion model trajectories are inherently smooth (B-splines), BIT* paths may need post-processing
