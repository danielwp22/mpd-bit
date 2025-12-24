# Complete BIT* vs MPD Comparison Guide

This guide explains how to run a comprehensive comparison between BIT* and MPD using a single script that handles everything.

## Quick Start

Run both BIT* and MPD on multiple problems with one command:

```bash
# Run on 10 problems (BIT* runs 600s each, MPD with no visualization)
python run_complete_comparison.py --n-problems 10

# Run on 5 problems with custom settings
python run_complete_comparison.py --n-problems 5 --bitstar-time 300 --mpd-samples 128
```

## What This Script Does

The `run_complete_comparison.py` script provides a **complete end-to-end workflow**:

1. **Generates** N random start/goal pairs (or loads existing ones)
2. **Runs BIT*** on each problem for 600 seconds with per-second tracking
3. **Runs MPD** on each problem with **NO visualization** (for speed)
4. **Computes comparison metrics** between both algorithms
5. **Saves all data** in multiple formats for analysis

## Data Collected

### BIT* Metrics (Per Second)

For each second of BIT* execution, the following metrics are tracked:

- **Path length**: Current best path length
- **Smoothness**: Trajectory smoothness metric
- **Mean jerk**: Mean jerk of trajectory
- **Tree size**: Number of vertices in the search tree
- **Number of samples**: Total samples generated
- **Has solution**: Whether a solution exists at this time

### BIT* Aggregate Metrics (Per Problem)

- **Success**: Whether BIT* found a solution
- **Final path length**: Best path length achieved
- **Final smoothness**: Smoothness of final path
- **Final mean jerk**: Jerk of final path
- **Planning time**: Total time spent
- **Time to first solution**: When first solution was found
- **First solution cost**: Cost of first solution
- **Time to match MPD**: When BIT* matched or beat MPD's path length (if applicable)
- **Iterations**: Total number of iterations
- **Tree size**: Final number of vertices

### MPD Metrics (Per Problem)

- **Inference time**: Time to generate all samples
- **Best path length**: Shortest path among all samples
- **Best smoothness**: Smoothness of best path
- **Collision rate**: Percentage of trajectories in collision
- **Number of collision-free samples**: Count of valid trajectories
- **Mean collision-free path length**: Average length of valid paths
- **Best collision-free path length**: Shortest valid path
- **Path length statistics**: Mean, std, min, max across all samples
- **Smoothness statistics**: Mean, std across all samples

### Comparison Metrics

For each problem where both algorithms succeeded:

- **BIT* beats MPD**: Boolean indicating if BIT* achieved better or equal path length
- **Time to match MPD**: How long BIT* took to match MPD's performance
- **Time to first solution**: When BIT* found its first solution
- **Success comparison**: Which algorithm(s) succeeded

### Aggregate Statistics

Across all problems, the script computes:

- **Success rates**: For both algorithms
- **Mean ± std** for all metrics
- **Comparison rates**: How often BIT* beats MPD
- **Average time to match**: Mean time for BIT* to reach MPD quality

## Output Files

All files are saved in the output directory (default: `multi_run_results/`):

### Problem Set
- **`problem_set.pt`** - The start/goal pairs (reusable across runs)
- **`problem_set.txt`** - Human-readable problem list

### Individual Results
- **`bitstar_result_000.pt`** - BIT* result for problem 0 (with per-second tracking data)
- **`bitstar_result_001.pt`** - BIT* result for problem 1
- **`mpd_results/mpd_result_000.pt`** - MPD result for problem 0
- **`mpd_results/mpd_result_001.pt`** - MPD result for problem 1

### Aggregated Results
- **`complete_aggregated_results.pt`** - All results combined (Python dict)
- **`complete_statistics.yaml`** - Summary statistics (human-readable)
- **`comparison_data.json`** - Per-problem comparison data (JSON)

## Usage Examples

### Basic Usage

```bash
# Run complete comparison on 10 problems
python run_complete_comparison.py --n-problems 10
```

This will:
- Generate 10 random start/goal pairs
- Run BIT* on each for 600 seconds
- Run MPD on each with 64 samples
- Save all results and statistics

### Custom Settings

```bash
# Fewer problems, shorter BIT* time, more MPD samples
python run_complete_comparison.py \
    --n-problems 5 \
    --bitstar-time 300 \
    --mpd-samples 128

# High-detail BIT* tracking (0.5s intervals)
python run_complete_comparison.py \
    --n-problems 10 \
    --interval 0.5

# Custom output directory
python run_complete_comparison.py \
    --n-problems 10 \
    --output-dir my_experiment
```

### Continue from Existing Problems

If `problem_set.pt` already exists, it will be reused:

```bash
# First run generates problems
python run_complete_comparison.py --n-problems 10

# Second run uses same problems (useful for parameter tuning)
python run_complete_comparison.py --n-problems 10 --bitstar-time 300
```

## Analyzing Results

### Python Analysis

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load aggregated results
results = torch.load('multi_run_results/complete_aggregated_results.pt')

# Access statistics
bitstar_stats = results['bitstar_stats']
mpd_stats = results['mpd_stats']
comparison_stats = results['comparison_stats']

print(f"BIT* success rate: {bitstar_stats['success_rate']*100:.1f}%")
print(f"MPD success rate: {mpd_stats['success_rate']*100:.1f}%")
print(f"BIT* beats MPD: {comparison_stats['bitstar_beats_mpd_rate']*100:.1f}%")

# Access individual problem results
bitstar_results = results['bitstar_results']
mpd_results = results['mpd_results']

# Example: Plot BIT* optimization over time for problem 0
problem_0 = bitstar_results[0]
if problem_0['success']:
    times = [m['time'] for m in problem_0['interval_metrics'] if m['has_solution']]
    lengths = [m['path_length'] for m in problem_0['interval_metrics'] if m['has_solution']]

    plt.figure(figsize=(10, 6))
    plt.plot(times, lengths, 'b-', linewidth=2, label='BIT*')

    # Add MPD target line
    mpd_target = mpd_results[0]['best_collision_free_path_length']
    plt.axhline(mpd_target, color='r', linestyle='--', linewidth=2, label=f'MPD ({mpd_target:.3f})')

    plt.xlabel('Time (s)')
    plt.ylabel('Path Length')
    plt.title('Problem 0: BIT* Anytime Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('problem_0_optimization.png')
```

### Analyzing Per-Second BIT* Data

```python
# Analyze how BIT* tree grows over time
problem = bitstar_results[0]

times = [m['time'] for m in problem['interval_metrics']]
tree_sizes = [m['num_vertices'] for m in problem['interval_metrics']]
path_lengths = [m['path_length'] if m['has_solution'] else None
                for m in problem['interval_metrics']]

# Plot tree growth
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times, tree_sizes, 'g-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Tree Size (vertices)')
plt.title('BIT* Tree Growth Over Time')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
valid_times = [t for t, pl in zip(times, path_lengths) if pl is not None]
valid_lengths = [pl for pl in path_lengths if pl is not None]
plt.plot(valid_times, valid_lengths, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Path Length')
plt.title('BIT* Solution Quality Over Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bitstar_detailed_analysis.png')
```

### Analyzing MPD Collision Statistics

```python
# Analyze MPD collision rates across all problems
mpd_results = results['mpd_results']

collision_rates = [r['collision_rate'] for r in mpd_results if r['success']]
mean_cf_lengths = [r['mean_collision_free_path_length']
                   for r in mpd_results
                   if r['success'] and r['mean_collision_free_path_length'] != float('inf')]

print(f"Average collision rate: {np.mean(collision_rates)*100:.1f}%")
print(f"Average collision-free path length: {np.mean(mean_cf_lengths):.3f}")

# Plot distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(collision_rates, bins=10, edgecolor='black')
plt.xlabel('Collision Rate')
plt.ylabel('Frequency')
plt.title('Distribution of MPD Collision Rates')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(mean_cf_lengths, bins=10, edgecolor='black')
plt.xlabel('Mean Collision-Free Path Length')
plt.ylabel('Frequency')
plt.title('Distribution of MPD Path Lengths')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mpd_collision_analysis.png')
```

### Reading YAML Statistics

```bash
# View statistics in terminal
cat multi_run_results/complete_statistics.yaml
```

Example output:
```yaml
bitstar:
  n_problems: 10
  n_success: 10
  success_rate: 1.0
  path_length_mean: 7.234
  path_length_std: 1.123
  planning_time_mean: 345.678
  time_to_first_solution_mean: 3.456
  tree_size_mean: 2453.2

mpd:
  n_problems: 10
  n_success: 10
  success_rate: 1.0
  collision_rate_mean: 0.127
  mean_collision_free_path_length_mean: 6.234
  best_collision_free_path_length_mean: 5.823
  inference_time_mean: 2.123

comparison:
  n_both_success: 10
  n_bitstar_beats_mpd: 8
  bitstar_beats_mpd_rate: 0.8
  time_to_match_mpd_mean: 123.456
```

### Reading JSON Comparison Data

```bash
# View comparison data
cat multi_run_results/comparison_data.json
```

Example output:
```json
[
  {
    "problem_idx": 0,
    "bitstar_success": true,
    "mpd_success": true,
    "mpd_target_length": 5.823,
    "bitstar_final_length": 5.654,
    "time_to_first_solution": 3.2,
    "time_to_match_mpd": 45.6,
    "bitstar_beats_mpd": true
  },
  {
    "problem_idx": 1,
    "bitstar_success": true,
    "mpd_success": true,
    "mpd_target_length": 6.123,
    "bitstar_final_length": 6.345,
    "time_to_first_solution": 2.8,
    "time_to_match_mpd": "N/A",
    "bitstar_beats_mpd": false
  }
]
```

## Command-Line Options

```
--n-problems N         Number of problems to solve (default: 10)
--bitstar-time T       Time limit for each BIT* run in seconds (default: 600)
--interval I           BIT* tracking interval in seconds (default: 1.0)
--mpd-samples N        Number of MPD trajectory samples per problem (default: 64)
--mpd-steps N          Number of MPD diffusion steps (default: 100)
--output-dir DIR       Output directory for results (default: multi_run_results)
--cfg PATH             Path to MPD config file
--device DEVICE        Device to use (default: cuda:0)
--seed SEED            Random seed (default: 42)
```

## Performance Tips

1. **Start Small**: Test with 2-3 problems first to verify everything works
2. **BIT* Time**: 600 seconds gives BIT* enough time to find good solutions
3. **Tracking Interval**: Use 1.0s for most cases, 0.5s for detailed analysis
4. **MPD Samples**: 64 samples provides good statistics, 128 for better accuracy
5. **Visualization**: Disabled by default for speed (MPD runs much faster)
6. **GPU Memory**: If running out of memory, reduce `--mpd-samples`

## Troubleshooting

### Problem: Out of GPU memory during MPD

**Solution**: Reduce `--mpd-samples` to 32 or 16

### Problem: BIT* failing on many problems

**Solution**: Increase `--bitstar-time` to give more time

### Problem: Want to regenerate problems

**Solution**: Delete `problem_set.pt` or use a different `--output-dir`

### Problem: Script crashes midway

**Solution**: Individual results are saved as they complete, so you can:
1. Check which problems completed
2. Load partial results from the individual `*_result_*.pt` files
3. Rerun with a different seed or output directory

## Differences from Previous Workflow

The old `MULTI_RUN_GUIDE.md` workflow required:
1. Run `run_multiple_comparisons.py` (BIT* only)
2. Run `run_inference_on_problems.py` (MPD only)
3. Run `run_multiple_comparisons.py --load-mpd` (combine)

The new `run_complete_comparison.py` does **all three steps in one command** and collects **more detailed metrics**.

## Advanced Analysis Examples

### Compare Success Rates by Difficulty

```python
# Load results
results = torch.load('multi_run_results/complete_aggregated_results.pt')

# Classify problems by difficulty (based on BIT* time to solution)
bitstar_results = results['bitstar_results']
mpd_results = results['mpd_results']

easy_problems = []
hard_problems = []

for br, mr in zip(bitstar_results, mpd_results):
    if br['success'] and br.get('time_to_first_solution'):
        if br['time_to_first_solution'] < 10.0:
            easy_problems.append((br, mr))
        else:
            hard_problems.append((br, mr))

print(f"Easy problems: {len(easy_problems)}")
print(f"Hard problems: {len(hard_problems)}")

# Compare MPD performance on easy vs hard
easy_mpd_lengths = [mr['best_collision_free_path_length'] for br, mr in easy_problems]
hard_mpd_lengths = [mr['best_collision_free_path_length'] for br, mr in hard_problems]

print(f"MPD on easy problems: {np.mean(easy_mpd_lengths):.3f} ± {np.std(easy_mpd_lengths):.3f}")
print(f"MPD on hard problems: {np.mean(hard_mpd_lengths):.3f} ± {np.std(hard_mpd_lengths):.3f}")
```

### Visualize BIT* vs MPD Across All Problems

```python
import matplotlib.pyplot as plt

# Load results
results = torch.load('multi_run_results/complete_aggregated_results.pt')
comparison_data = results['comparison_data']

# Extract data for successful problems
problem_indices = []
bitstar_lengths = []
mpd_lengths = []

for c in comparison_data:
    if c.get('bitstar_success') and c.get('mpd_success'):
        problem_indices.append(c['problem_idx'])
        bitstar_lengths.append(c['bitstar_final_length'])
        mpd_lengths.append(c['mpd_target_length'])

# Plot
plt.figure(figsize=(12, 6))
x = np.arange(len(problem_indices))
width = 0.35

plt.bar(x - width/2, bitstar_lengths, width, label='BIT*', alpha=0.8)
plt.bar(x + width/2, mpd_lengths, width, label='MPD', alpha=0.8)

plt.xlabel('Problem Index')
plt.ylabel('Path Length')
plt.title('BIT* vs MPD Path Length Comparison')
plt.xticks(x, problem_indices)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitstar_vs_mpd_comparison.png')
```

## Next Steps

After running the comparison:

1. **Analyze per-second BIT* data** to understand optimization behavior
2. **Compare collision rates** to evaluate MPD sample quality
3. **Identify challenging problems** where both struggled
4. **Tune parameters** based on statistics
5. **Generate plots** for papers/presentations
