# Fair Comparison Guide: Diffusion Model vs BIT* (and other OMPL planners)

This guide ensures you compare the diffusion model and BIT* on **exactly the same problem instances** with **the same environment configuration**.

---

## Quick Start: Run BIT* on Same Problem as Diffusion Model

```bash
cd /home/ubuntu/Projects/MotionPlanningDiffusion/mpd-splines-public/scripts/inference

# 1. Run diffusion model and save results
python inference.py
# Results saved to: logs/2/results_single_plan-000.pt

# 2. Run BIT* on the SAME start/goal
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt

# 3. Compare the results
python compare_results.py
```

---

## Anytime Comparison: Tracking BIT*'s Optimization Progress

BIT* is an **anytime** algorithm, meaning it finds an initial solution quickly, then continues to improve it over time. To fairly compare with the diffusion model, you can track:

1. **Time to first solution** - How long BIT* takes to find any valid path
2. **Time to match diffusion quality** - How long BIT* takes to find a path as good as the diffusion model's
3. **Final solution quality** - How much BIT* improves given the full time budget

### Quick Start: Anytime Comparison

```bash
# 1. Run diffusion model
python inference.py
# Suppose it finds a path with length 7.5 in 2.3 seconds

# 2. Run BIT* in anytime mode with the same problem
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --anytime --time 60.0

# Output will show:
#   [0.52s] First solution found! Path length: 9.2
#   [3.41s] Improved solution! Path length: 8.1
#   [7.89s] Reached target quality! Path length: 7.4 <= 7.5
#   (stops early since it beat the target)

# 3. Compare results
python compare_results.py
```

### What Gets Tracked in Anytime Mode

The anytime mode records:
- **First solution time**: When BIT* first finds any collision-free path
- **First solution path length**: Quality of that initial path
- **Target quality time**: When BIT* finds a path ≤ diffusion model's path length
- **Final solution**: Best path found within the time budget
- **Improvement ratio**: How much BIT* improved from first to final solution

### Example Anytime Comparison Output

```
================================================================================
COMPARISON: Diffusion Model vs BITSTAR
(Baseline in ANYTIME mode)
================================================================================

Metric                         Diffusion            BITSTAR
--------------------------------------------------------------------------------
Success (single trajectory)    1                    1/1
Success rate (%)               87.0%                100.0%

Path Length (best/mean)        7.500                7.412 ± 0.000
  First solution path length   N/A                  9.234 ± 0.000

Planning Time (sec)            Diffusion            BITSTAR
  Total time                   2.300                7.892 ± 0.000
  Time to first solution       2.300                0.521 ± 0.000

Target Quality Comparison
  Target path length           7.500
  Reached target (%)           N/A                  100.0%
  Time to reach target         2.300 sec            7.892 ± 0.000 sec
================================================================================
```

### Key Insights from Anytime Comparison

1. **Speed vs Quality Tradeoff**:
   - Diffusion: Fixed time (e.g., 2.3s) → Fixed quality (e.g., 7.5)
   - BIT*: Variable time → Improving quality
     - 0.5s → Path length 9.2 (fast but lower quality)
     - 7.9s → Path length 7.4 (slower but beats diffusion)

2. **When Diffusion Wins**:
   - If you need a solution in < first solution time (e.g., < 0.5s)
   - If diffusion quality is good enough and faster than target time

3. **When BIT* Wins**:
   - If you can afford more time than diffusion (anytime optimization)
   - If you need the best possible solution (let it run longer)

---

## Why Fair Comparison Matters

When comparing motion planning algorithms, you need to ensure:

1. **Same Start and Goal States**: Both algorithms solve the exact same problem
2. **Same Environment**: Identical obstacle configurations
3. **Same Robot Configuration**: Same kinematic parameters, joint limits, etc.
4. **Same Success Criteria**: Consistent collision checking and goal tolerance
5. **Same Trajectory Resolution**: Same number of waypoints (default: 128)

Without this, you're comparing apples to oranges.

---

## Method 1: Run BIT* on Diffusion Model's Problems (Recommended)

This is the **easiest and most reliable** method.

### Step 1: Run Diffusion Model

```bash
python inference.py
```

This generates a problem instance and saves it to `logs/2/results_single_plan-000.pt`, which contains:
- Start state (`q_pos_start`)
- Goal state (`q_pos_goal`)
- Environment configuration
- Diffusion model solution

### Step 2: Run BIT* on the Same Problem

```bash
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt
```

This:
1. Loads the exact start/goal from the diffusion results
2. Recreates the same environment
3. Runs BIT* on that problem
4. Saves results to `logs_bitstar_panda_spheres3d/`

### Step 3: Compare Results

```bash
python compare_results.py
```

Output example:
```
================================================================================
COMPARISON: Diffusion Model vs BITSTAR
================================================================================

Metric                         Diffusion            BITSTAR
--------------------------------------------------------------------------------
Success (single trajectory)    1                    1/1
Success rate (%)               87.0%                100.0%

Path Length (best/mean)        6.995                7.123 ± 0.000

Planning Time (sec)            2.456                12.345 ± 0.000
  Generator time               1.234
  Guidance time                0.987

Smoothness (lower=better)      57.415               N/A
Collision Intensity            0.0050               N/A
================================================================================
```

---

## Method 2: Run Multiple Baseline Planners

To compare against multiple OMPL planners:

```bash
# Run diffusion model once
python inference.py

# Run multiple baselines on the same problem
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --planner BITstar
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --planner RRTConnect
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --planner RRTstar

# Compare all at once
python compare_results.py --baselines bitstar rrtconnect rrtstar --summary
```

Summary table output:
```
====================================================================================================
SUMMARY COMPARISON TABLE
====================================================================================================
Method               Success %    Path Length     Time (sec)   Smoothness
----------------------------------------------------------------------------------------------------
Diffusion            87.0         6.995           2.456        57.415
BITSTAR              100.0        7.123±0.00      12.345       N/A
RRTCONNECT           100.0        8.456±0.00      1.234        N/A
RRTSTAR              100.0        7.234±0.00      18.567       N/A
====================================================================================================
```

---

## Method 3: Batch Evaluation on N Problems

To get statistically significant results, run on multiple problem instances:

```python
# In bitstar_baseline.py, modify:

def run_fair_comparison(n_problems=50):
    """Run diffusion and BIT* on N identical problems."""

    # Load environment and robot (same config as diffusion)
    from torch_robotics.environments import EnvSpheres3D
    from torch_robotics.robots import RobotPanda

    env = EnvSpheres3D()
    robot = RobotPanda()

    # Sample N start-goal pairs
    starts, goals = sample_n_problems(env, robot, n=n_problems)

    # Run diffusion on all problems
    for i, (start, goal) in enumerate(zip(starts, goals)):
        run_diffusion_inference(start, goal, output=f"logs/2/results_single_plan-{i:03d}.pt")

    # Run BIT* on all problems
    for i in range(n_problems):
        run_bitstar_from_diffusion(f"logs/2/results_single_plan-{i:03d}.pt")

    # Aggregate and compare
    aggregate_results("logs/2/", "logs_bitstar_panda_spheres3d/")
```

Then analyze:
```bash
python compare_results.py --aggregate --n-problems 50
```

---

## Understanding the Metrics

### Success Rate
- **Diffusion**: Fraction of sampled trajectories that are collision-free (`fraction_valid`)
- **BIT***: Binary success (0 or 1) for finding a single collision-free path

**Why different?**
- Diffusion generates N=100 trajectories, reports % that are valid
- BIT* returns one trajectory, either success or failure

### Path Length
- **Diffusion**: Length of the best (shortest) collision-free trajectory
- **BIT***: Length of the single returned trajectory

**Unit**: Sum of Euclidean distances between waypoints in configuration space

### Planning Time
- **Diffusion**: Total inference time (generation + guidance)
- **BIT***: Time to find a solution (or timeout)

**Note**: Diffusion time includes GPU forward passes; BIT* is CPU-only

### Smoothness
- **Diffusion**: Sum of acceleration magnitudes (lower is smoother)
- **BIT***: Not computed by default (add custom metric if needed)

**Formula**: `Σ ||q̈(t)||` where `q̈` is configuration acceleration

### Collision Intensity
- **Diffusion**: Average SDF penetration depth into obstacles
- **BIT***: Not applicable (OMPL uses binary collision checking)

---

## Environment Configuration Matching

### Critical: Use Identical Environments

When running baselines, ensure you replicate the EXACT environment from the diffusion model:

```python
# In bitstar_baseline.py

# For Panda + Spheres3D
from torch_robotics.environments import EnvSpheres3D
env = EnvSpheres3D(
    precompute_sdf_obj_fixed=True,
    sdf_cell_size=0.005,  # Match diffusion config!
    tensor_args={'device': 'cpu', 'dtype': torch.float32}
)

# For Panda + Warehouse
from torch_robotics.environments import EnvWarehouse
env = EnvWarehouse(
    # Use exact same parameters as config file!
)
```

**How to check**: Compare `scripts/inference/cfgs/config_EnvSpheres3D-RobotPanda_00.yaml` with your baseline environment initialization.

---

## Common Pitfalls

### ❌ Don't Do This
```python
# Running on different problems
diffusion_result = run_diffusion(start=random_start_1, goal=random_goal_1)
bitstar_result = run_bitstar(start=random_start_2, goal=random_goal_2)
```

### ✅ Do This
```python
# Load same start/goal
diffusion_result = torch.load('logs/2/results_single_plan-000.pt')
start = diffusion_result['q_pos_start'].cpu().numpy()
goal = diffusion_result['q_pos_goal'].cpu().numpy()

# Use in baseline
bitstar_result = run_bitstar(start=start, goal=goal)
```

### ❌ Don't Compare
- Diffusion on one environment vs BIT* on a different environment
- Diffusion with 128 waypoints vs BIT* with 50 waypoints
- Diffusion success rate (%) vs BIT* binary success (this is apples-to-oranges)

### ✅ Do Compare
- Path length of best trajectory from each method
- Planning time (but note GPU vs CPU difference)
- Success on the same set of N problem instances
- Smoothness if you compute it for BIT* trajectories

---

## Advanced: Adding Custom Metrics to BIT*

To compute smoothness for BIT* trajectories:

```python
# In bitstar_baseline.py

def compute_smoothness(trajectory):
    """Compute smoothness metric for a trajectory."""
    # trajectory shape: (n_waypoints, n_dof)

    # Compute velocities (finite differences)
    velocities = np.diff(trajectory, axis=0)

    # Compute accelerations
    accelerations = np.diff(velocities, axis=0)

    # Smoothness = sum of acceleration magnitudes
    smoothness = np.sum(np.linalg.norm(accelerations, axis=1))

    return smoothness

# Add to statistics
result = baseline.plan(start, goal)
if result['success']:
    trajectory = result['trajectory']
    smoothness = compute_smoothness(trajectory)
    statistics['smoothness_values'].append(smoothness)
```

---

## Example: Complete Fair Comparison Workflow

### Standard Comparison

```bash
# 1. Run diffusion model inference
cd /home/ubuntu/Projects/MotionPlanningDiffusion/mpd-splines-public/scripts/inference
python inference.py
# Output: logs/2/results_single_plan-000.pt

# 2. View diffusion metrics
python view_metrics.py --results-file logs/2/results_single_plan-000.pt

# 3. Run BIT* on the same problem
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt

# 4. Run RRT-Connect for comparison
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --planner RRTConnect

# 5. Compare all methods
python compare_results.py --baselines bitstar rrtconnect

# 6. Generate summary table
python compare_results.py --baselines bitstar rrtconnect --summary
```

### Anytime Comparison Workflow

```bash
# 1. Run diffusion model inference
python inference.py
# Output: logs/2/results_single_plan-000.pt
# Suppose diffusion finds path length 7.5 in 2.3 seconds

# 2. Run BIT* in ANYTIME mode on the same problem
python bitstar_baseline.py \
    --use-diffusion-problem \
    --diffusion-results logs/2/results_single_plan-000.pt \
    --anytime \
    --time 60.0

# This will:
# - Load the diffusion path length (7.5) as the target
# - Track when BIT* finds its first solution
# - Track when BIT* reaches quality ≤ 7.5
# - Stop early if it beats the target before 60 seconds
# - Save detailed timing statistics

# 3. View the anytime comparison
python compare_results.py

# Output will show:
# - Time to first solution vs diffusion time
# - Whether BIT* reached target quality
# - How long it took to reach target quality
# - Final solution quality after full optimization
```

---

## Interpreting Results

### When Diffusion Wins
- **Faster planning time**: Amortized over training, inference is typically faster
- **Smoother trajectories**: B-spline representation inherently smooth
- **Diverse solutions**: Can generate multiple high-quality trajectories

### When BIT* Wins
- **Higher success rate**: Deterministic search often more reliable
- **Shorter paths**: Asymptotically optimal guarantees
- **No training required**: Works out-of-the-box

### Fair Interpretation
Both methods have different strengths. Key factors:
1. **Training time**: Diffusion requires hours of training; BIT* requires none
2. **Inference time**: Diffusion is faster per query after training
3. **Optimality**: BIT* has theoretical guarantees; diffusion does not
4. **Smoothness**: Diffusion trajectories are smoother by design
5. **Diversity**: Diffusion can generate multiple solutions; BIT* finds one

---

## Citation Tip

When reporting comparisons in papers:

```latex
We evaluated our diffusion-based planner against BIT* \cite{bitstar}
on 50 identical problem instances in the Panda + Spheres3D environment.
The diffusion model achieved a 87\% success rate with mean path length
7.12 ± 0.95 in 2.5s inference time, while BIT* achieved 94\% success
with mean path length 7.45 ± 1.12 in 12.3s planning time. While BIT*
found solutions more reliably, the diffusion model produced smoother
trajectories (smoothness: 57.4 vs. 142.3) and was 4.9× faster.
```

**Key**: Always mention you tested on identical problem instances!

---

## Troubleshooting

### "BIT* succeeds but diffusion fails"
- Check if the problem is solvable (BIT* success confirms it is)
- Diffusion may need more sampling steps or better guidance
- Try increasing `n_samples` in config

### "Diffusion succeeds but BIT* fails"
- Check timeout: BIT* may need more time
- Verify environment matching: SDF vs geometric collision checking differs
- BIT* may be stuck in local optima

### "Path lengths are very different"
- Ensure same number of waypoints (both should use 128)
- Check if one method shortcuts through narrow passages
- Verify collision checking is consistent

---

## Summary Checklist

- [ ] Run diffusion model, save results with start/goal
- [ ] Run BIT* using `--use-diffusion-problem --diffusion-results` flags with same problem
- [ ] Verify environment configurations match
- [ ] Compare on same metrics (path length, time, success)
- [ ] Report sample size (N problems) for statistical validity
- [ ] Mention training time for diffusion vs zero training for BIT*
- [ ] Acknowledge complementary strengths of both methods

---

For more details, see:
- `README_BASELINES.md` - Overview of metrics and baseline running
- `bitstar_baseline.py` - BIT* implementation with fair comparison mode
- `compare_results.py` - Automated comparison script
- `view_metrics.py` - View diffusion model metrics
