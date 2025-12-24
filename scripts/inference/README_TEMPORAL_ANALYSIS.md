# Temporal Convergence Analysis

This guide explains how to analyze second-by-second convergence of BIT* across multiple problems using the `analyze_temporal_convergence.py` script.

## Overview

The temporal convergence analysis computes **aggregate statistics at each time step** across all problems in a comparison run. This shows how BIT* converges over time on average, rather than just looking at final results.

## Quick Start

```bash
# Run complete comparison first (if you haven't already)
python run_complete_comparison.py --n-problems 10

# Analyze temporal convergence
python analyze_temporal_convergence.py

# With custom settings
python analyze_temporal_convergence.py --time-step 0.5 --output-dir temporal_analysis
```

## What This Script Computes

### Second-by-Second Aggregate Metrics

For each time step (default: every 1 second), the script computes:

**Success Metrics:**
- **Success rate**: % of problems that have found a solution at this time
- **Success count**: Number of problems with solutions

**Path Quality Metrics (for problems with solutions):**
- **Path length**: Mean, std, min, max across all problems
- **Smoothness**: Mean, std (if available)
- **Mean jerk**: Mean, std (if available)

**Tree Growth Metrics:**
- **Tree size**: Mean, std number of vertices
- **Number of samples**: Total samples generated
- **Number of batches**: Total batches processed

### Output Files

The script generates:

1. **`temporal_statistics.npz`** - NumPy archive with all temporal data
2. **`temporal_statistics.csv`** - CSV file for easy viewing in Excel/spreadsheet
3. **Multiple plots** (see below)

### Generated Plots

1. **`temporal_convergence_overview.png`**
   - 4-panel overview showing:
     - Success rate over time
     - Path length convergence (with MPD target if provided)
     - Tree size growth
     - Number of batches

2. **`path_length_convergence_detailed.png`**
   - Detailed path length plot showing:
     - Mean path length
     - ±1 standard deviation
     - Min/max envelope
     - MPD target line (if provided)
     - Time when BIT* matches MPD (if applicable)

3. **`smoothness_jerk_convergence.png`**
   - Smoothness and mean jerk over time
   - Shows trajectory quality improvement

4. **`sampling_efficiency.png`**
   - Total samples over time
     - Success rate vs tree size scatter plot

## Usage Examples

### Basic Usage

```bash
# Analyze default results directory
python analyze_temporal_convergence.py
```

This will:
- Load all `bitstar_result_*.pt` files from `multi_run_results/`
- Compute second-by-second statistics with 1.0s intervals
- Auto-load MPD target from `complete_aggregated_results.pt` (if available)
- Save plots and statistics to `multi_run_results/`

### Custom Results Directory

```bash
# Analyze specific experiment
python analyze_temporal_convergence.py \
    --results-dir my_experiment/results \
    --output-dir my_experiment/temporal_analysis
```

### Finer Time Resolution

```bash
# Analyze with 0.5 second intervals
python analyze_temporal_convergence.py --time-step 0.5
```

This gives more detailed convergence curves but requires that BIT* was run with `--interval 0.5` or smaller.

### With MPD Comparison Target

```bash
# Manually specify MPD target
python analyze_temporal_convergence.py --mpd-target 5.823

# Auto-load from aggregated results
python analyze_temporal_convergence.py \
    --aggregated multi_run_results/complete_aggregated_results.pt
```

### Limit Time Range

```bash
# Only analyze first 300 seconds
python analyze_temporal_convergence.py --max-time 300
```

## Understanding the Results

### CSV Output

The `temporal_statistics.csv` file contains one row per time point:

```csv
time,success_rate,success_count,path_length_mean,path_length_std,...
0.0,0.0000,0,nan,nan,...
1.0,0.3000,3,8.234,1.234,...
2.0,0.5000,5,7.823,1.123,...
...
```

You can open this in Excel or use pandas:

```python
import pandas as pd
df = pd.read_csv('multi_run_results/temporal_statistics.csv')
print(df[['time', 'success_rate', 'path_length_mean']].head(10))
```

### NumPy Archive

The `temporal_statistics.npz` file can be loaded in Python:

```python
import numpy as np

# Load statistics
stats = np.load('multi_run_results/temporal_statistics.npz')

# Access data
time_points = stats['time_points']
success_rate = stats['success_rate']
path_length_mean = stats['path_length_mean']
path_length_std = stats['path_length_std']

# Plot custom analysis
import matplotlib.pyplot as plt
plt.plot(time_points, path_length_mean)
plt.fill_between(time_points,
                 path_length_mean - path_length_std,
                 path_length_mean + path_length_std,
                 alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Path Length')
plt.title('Custom Analysis')
plt.show()
```

### Key Insights from Temporal Analysis

The temporal analysis helps answer questions like:

1. **How quickly does BIT* find initial solutions?**
   - Look at the success rate curve
   - Check when success_rate first becomes > 0

2. **How does solution quality improve over time?**
   - Look at path_length_mean curve
   - Compare slope in early vs late stages

3. **When does BIT* match MPD quality?**
   - Look for intersection of BIT* mean and MPD target line
   - Check the detailed path length plot

4. **Is BIT* still improving near timeout?**
   - Look at slope of path_length_mean near max_time
   - If still decreasing significantly, might benefit from more time

5. **How does tree size relate to solution quality?**
   - Look at tree growth plot
   - Compare with path length convergence

6. **Are all problems equally difficult?**
   - Look at path_length_std (standard deviation)
   - High std means high variability across problems

## Advanced Analysis Examples

### Compare Multiple Experiments

```python
import numpy as np
import matplotlib.pyplot as plt

# Load different experiments
exp1 = np.load('exp1/temporal_statistics.npz')
exp2 = np.load('exp2/temporal_statistics.npz')

# Plot comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(exp1['time_points'], exp1['success_rate']*100, label='Exp 1')
plt.plot(exp2['time_points'], exp2['success_rate']*100, label='Exp 2')
plt.xlabel('Time (s)')
plt.ylabel('Success Rate (%)')
plt.legend()
plt.title('Success Rate Comparison')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(exp1['time_points'], exp1['path_length_mean'], label='Exp 1')
plt.plot(exp2['time_points'], exp2['path_length_mean'], label='Exp 2')
plt.xlabel('Time (s)')
plt.ylabel('Path Length')
plt.legend()
plt.title('Path Length Comparison')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_comparison.png')
```

### Compute Time to Reach Target Quality

```python
import numpy as np

stats = np.load('multi_run_results/temporal_statistics.npz')

target_quality = 6.0  # desired path length
times = stats['time_points']
path_lengths = stats['path_length_mean']

# Find when mean path length drops below target
idx = np.where(path_lengths <= target_quality)[0]
if len(idx) > 0:
    time_to_target = times[idx[0]]
    print(f"Time to reach {target_quality:.3f}: {time_to_target:.1f}s")
else:
    print(f"Target {target_quality:.3f} not reached")
```

### Analyze Early vs Late Convergence

```python
import numpy as np

stats = np.load('multi_run_results/temporal_statistics.npz')

times = stats['time_points']
lengths = stats['path_length_mean']

# Remove NaN values
valid = ~np.isnan(lengths)
times_valid = times[valid]
lengths_valid = lengths[valid]

# Split into early and late phases
split_time = 60.0  # 1 minute
early_mask = times_valid <= split_time
late_mask = times_valid > split_time

# Compute improvement rates
if early_mask.any() and late_mask.any():
    early_improvement = (lengths_valid[early_mask][0] -
                        lengths_valid[early_mask][-1])
    late_improvement = (lengths_valid[late_mask][0] -
                       lengths_valid[late_mask][-1])

    print(f"Early improvement (0-{split_time}s): {early_improvement:.3f}")
    print(f"Late improvement ({split_time}s+): {late_improvement:.3f}")

    early_rate = early_improvement / split_time
    late_rate = late_improvement / (times_valid[-1] - split_time)

    print(f"Early improvement rate: {early_rate:.4f} per second")
    print(f"Late improvement rate: {late_rate:.4f} per second")
```

## Integration with Complete Comparison

The temporal analysis is designed to work seamlessly with `run_complete_comparison.py`:

```bash
# Step 1: Run complete comparison
python run_complete_comparison.py \
    --n-problems 10 \
    --bitstar-time 600 \
    --interval 1.0

# Step 2: Analyze overall results
python analyze_comparison_results.py

# Step 3: Analyze temporal convergence
python analyze_temporal_convergence.py

# Step 4: View all results
ls multi_run_results/*.png
cat multi_run_results/temporal_statistics.csv | head -20
```

## Command-Line Options

```
--results-dir DIR      Directory with bitstar_result_*.pt files (default: multi_run_results)
--output-dir DIR       Output directory for plots/stats (default: same as results-dir)
--time-step FLOAT      Time step for binning in seconds (default: 1.0)
--max-time FLOAT       Maximum time to analyze (default: auto-detect)
--mpd-target FLOAT     Target path length from MPD for comparison (default: None)
--aggregated PATH      Path to complete_aggregated_results.pt for auto MPD target
```

## Troubleshooting

### Issue: No data in plots

**Cause**: BIT* results don't have `interval_metrics`

**Solution**: Make sure you ran BIT* with tracking enabled. The `run_complete_comparison.py` script does this automatically.

### Issue: All smoothness values are 0.0 or NaN

**Cause**: The smoothness metric requires acceleration data, which may not be available for all trajectory representations.

**Solution**: This is a known issue with certain trajectory types. The smoothness values for MPD B-spline trajectories may be 0.0 because the acceleration is computed from a state vector that only contains positions. Focus on path length and jerk metrics instead.

### Issue: Plots look jagged/noisy

**Cause**: Not enough problems or high variability

**Solution**:
1. Run more problems (increase `--n-problems`)
2. Use larger time step (increase `--time-step`)
3. Use smoothing in post-processing

### Issue: Memory error when loading results

**Cause**: Too many problems or very long runs

**Solution**:
1. Process in batches
2. Reduce `--max-time` to analyze only first part
3. Increase `--time-step` to reduce data points

## Notes on Smoothness Metric

**UPDATE:** The smoothness bug has been fixed! See `SMOOTHNESS_FIX.md` for details.

**Previous Issue (FIXED):** Earlier versions of `run_complete_comparison.py` had a bug where MPD smoothness was always 0.0. This was because:

1. The script directly computed smoothness on B-spline control points
2. Without converting them to full trajectories with acceleration
3. The `get_acceleration` method returned empty tensors, resulting in smoothness = 0.0

**Current Status (after fix):** The script now:
1. ✅ Converts control points to full trajectories (pos, vel, acc)
2. ✅ Computes smoothness with proper acceleration data
3. ✅ Returns correct smoothness values (e.g., 78.728 ± 25.483)

**To verify you have the fixed version:**
```bash
grep "get_q_trajectory" run_complete_comparison.py
# Should show trajectory conversion code
```

**If you have old results with 0.0 smoothness:**
- Delete them: `rm -rf multi_run_results/`
- Regenerate: `python run_complete_comparison.py --n-problems 10`

## Performance Tips

1. **Time step**: Use 1.0s for most cases, 0.5s for detailed analysis
2. **Number of problems**: More problems (20+) give smoother aggregate curves
3. **BIT* interval**: Must be ≤ analysis time_step for accurate results
4. **Output format**: Use CSV for easy viewing, NPZ for Python analysis
5. **Plotting**: Can disable with `--no-plots` if matplotlib not available (requires code modification)

## See Also

- `run_complete_comparison.py` - Run BIT* and MPD comparison
- `analyze_comparison_results.py` - Analyze final aggregate statistics
- `COMPLETE_COMPARISON_GUIDE.md` - Complete workflow guide
