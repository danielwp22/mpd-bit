# Fair Comparison Guide: Diffusion Model vs BIT*

This guide ensures you compare the diffusion model and BIT* on **exactly the same problem instances** with **the same environment configuration** and **the same collision checking method**.

---

## Which BIT* Implementation to Use?

We provide **two BIT* implementations** for comparison:

### 1. **GPU-Based BIT*** (Recommended - Fairest Comparison)
- **Script**: `bitstar_gpu_baseline.py`
- **Collision Checking**: SDF-based (same as diffusion model) ✅
- **Platform**: Pure Python + PyTorch (GPU)
- **Why Fair**: Uses identical environment representation and collision detection
- **Use when**: You want the most apples-to-apples comparison

### 2. **OMPL-Based BIT*** (Traditional Baseline)
- **Script**: `bitstar_baseline.py`
- **Collision Checking**: Geometric (different from diffusion) ⚠️
- **Platform**: C++ OMPL library via PyBullet
- **Why Different**: Different collision detection may give different success rates
- **Use when**: You want to compare against established motion planning libraries

**Recommendation**: Use the **GPU-based version** for fair comparisons in papers/research, as it eliminates the confounding variable of different collision checking methods.

---

## Quick Start: Run BIT* GPU on Same Problem as Diffusion Model

```bash
cd /home/ubuntu/Projects/MotionPlanningDiffusion/mpd-splines-public/scripts/inference

# 1. Run diffusion model and save results
python inference.py
# Results saved to: logs/2/results_single_plan-000.pt

# 2. Run BIT* GPU on the SAME start/goal (RECOMMENDED)
python bitstar_gpu_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt

# 3. Compare the results
python compare_results.py --baselines bitstar_gpu
```

**Alternative**: Use OMPL-based BIT*:
```bash
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt
python compare_results.py --baselines bitstar
```

---

## Why Fair Comparison Matters

When comparing motion planning algorithms, you need to ensure:

1. **Same Start and Goal States**: Both algorithms solve the exact same problem ✅
2. **Same Environment**: Identical obstacle configurations ✅
3. **Same Collision Checking**: **This is critical!** ⚠️
   - The GPU-based BIT* uses **SDF-based collision checking** (same as diffusion)
   - The OMPL-based BIT* uses **geometric collision checking** (different!)
   - Different collision checking can lead to different success rates and path qualities
4. **Same Trajectory Resolution**: Same number of waypoints (default: 128) ✅

**Key Insight**: The GPU-based BIT* (`bitstar_gpu_baseline.py`) eliminates the collision checking confound, making it the fairest comparison.

---

## Method 1: Run BIT* GPU on Diffusion Model's Problems (Recommended)

This is the **easiest and fairest** method.

### Step 1: Run Diffusion Model

```bash
python inference.py
```

This generates a problem instance and saves it to `logs/2/results_single_plan-000.pt`, which contains:
- Start state (`q_pos_start`)
- Goal state (`q_pos_goal`)
- Environment configuration
- Diffusion model solution

### Step 2: Run BIT* GPU on the Same Problem

```bash
python bitstar_gpu_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt
```

This:
1. Loads the exact start/goal from the diffusion results
2. Recreates the same environment with SDF collision checking
3. Runs BIT* on that problem
4. Saves results to `logs_bitstar_gpu_panda_spheres3d/`

### Step 3: Compare Results

```bash
python compare_results.py --baselines bitstar_gpu
```

Output example:

```
================================================================================
COMPARISON: Diffusion Model vs BITSTAR-GPU
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
- **BIT* GPU**: Time to find a solution (or timeout)
- **BIT* OMPL**: Time to find a solution (or timeout)

**Note**:
- Diffusion uses GPU for neural network inference
- BIT* GPU uses GPU for collision checking and computations
- BIT* OMPL is CPU-only (C++ implementation)

### Smoothness
- **Diffusion**: Sum of acceleration magnitudes from B-spline representation (lower is smoother)
- **BIT* GPU**: Sum of acceleration magnitudes computed via finite differences ✅
- **BIT* OMPL**: Sum of acceleration magnitudes computed via finite differences ✅

**Formula**: `Σ ||q̈(t)||` where `q̈` is configuration acceleration

### Collision Intensity
- **Diffusion**: Average SDF penetration depth into obstacles
- **BIT* GPU**: Can be computed (uses SDF) but not computed by default
- **BIT* OMPL**: Not applicable (uses geometric/binary collision checking)

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
- **Diffusion (SDF collision) vs BIT* OMPL (geometric collision)** - Different collision checking!
- Diffusion on one environment vs BIT* on a different environment
- Diffusion with 128 waypoints vs BIT* with 50 waypoints
- Diffusion success rate (%) vs BIT* binary success (this is apples-to-oranges)

### ✅ Do Compare
- **Diffusion vs BIT* GPU** - Same SDF collision checking! ✅
- Path length of best trajectory from each method
- Planning time (both use GPU)
- Success on the same set of N problem instances
- Smoothness (both methods compute this automatically)

---

## Example: Complete Fair Comparison Workflow

### Standard Comparison (Recommended)

```bash
# 1. Run diffusion model inference
cd /home/ubuntu/Projects/MotionPlanningDiffusion/mpd-splines-public/scripts/inference
python inference.py
# Output: logs/2/results_single_plan-000.pt

# 2. View diffusion metrics
python view_metrics.py --results-file logs/2/results_single_plan-000.pt

# 3. Run BIT* GPU on the same problem (FAIREST comparison - same SDF collision checking)
python bitstar_gpu_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt

# 4. Compare the results
python compare_results.py --baselines bitstar_gpu

# 5. Generate summary table
python compare_results.py --baselines bitstar_gpu --summary
```

### Using OMPL Baselines (Optional)

If you want to compare against traditional OMPL planners (note: different collision checking):

```bash
# Run OMPL BIT*
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt

# Run OMPL RRT-Connect
python bitstar_baseline.py --use-diffusion-problem --diffusion-results logs/2/results_single_plan-000.pt --planner RRTConnect

# Compare all
python compare_results.py --baselines bitstar_gpu bitstar rrtconnect
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
To ensure a fair comparison, we implemented BIT* using the same SDF-based
collision checking as our diffusion model. The diffusion model achieved
an 87\% success rate with mean path length 7.12 ± 0.95 in 2.5s inference
time, while BIT* achieved 94\% success with mean path length 7.45 ± 1.12
in 12.3s planning time. While BIT* found solutions more reliably, the
diffusion model produced smoother trajectories (smoothness: 57.4 vs. 142.3)
and was 4.9× faster.
```

**Key Points to Mention**:
- Tested on identical problem instances ✅
- **Used same collision checking method** ✅
- Reported both success rates and path quality ✅
- Acknowledged different strengths of each method ✅

---

## Troubleshooting

### "BIT* succeeds but diffusion fails"
- Check if the problem is solvable (BIT* success confirms it is)
- Diffusion may need more sampling steps or better guidance
- Try increasing `n_samples` in config

### "Diffusion succeeds but BIT* fails"
- Check timeout: BIT* may need more time (increase `--time`)
- Try larger batch size (increase `--batch-size`)
- BIT* may be stuck in local optima - this is expected for some problems

### "Path lengths are very different"
- Ensure same number of waypoints (both should use 128)
- Check if one method shortcuts through narrow passages
- **If comparing Diffusion vs BIT* OMPL**: Different collision checking can cause this!
- **Solution**: Use BIT* GPU for fair comparison with identical collision checking

### "Success rates are very different"
- **Most likely cause**: Different collision checking methods!
- BIT* OMPL uses geometric collision detection (different from diffusion's SDF)
- **Solution**: Use `bitstar_gpu_baseline.py` for same collision checking

---

## Summary Checklist

**For Fair Comparisons (Recommended):**
- [ ] Run diffusion model, save results with start/goal
- [ ] Run **BIT* GPU** using `--use-diffusion-problem --diffusion-results` flags
- [ ] Verify **same collision checking method** (SDF-based) ✅
- [ ] Compare on same metrics (path length, time, success, smoothness)
- [ ] Report sample size (N problems) for statistical validity
- [ ] Mention training time for diffusion vs zero training for BIT*
- [ ] **Explicitly state that you used the same collision checking in your paper/report**

**Optional: Compare against OMPL baselines**
- [ ] Run OMPL BIT* using `bitstar_baseline.py` (note: different collision checking)
- [ ] Acknowledge that OMPL uses geometric collision checking vs SDF

---

## Files Reference

- `bitstar_gpu_baseline.py` - **GPU-based BIT* with SDF collision (RECOMMENDED)**
- `bitstar_baseline.py` - OMPL-based BIT* with geometric collision
- `compare_results.py` - Automated comparison script
- `view_metrics.py` - View diffusion model metrics
- `FAIR_COMPARISON_GUIDE.md` - This guide
