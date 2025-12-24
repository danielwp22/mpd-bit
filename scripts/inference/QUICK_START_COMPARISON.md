# Quick Start: BIT* vs MPD Comparison

Complete workflow to compare BIT* and MPD on the same problems.

## 📋 3-Step Process

### Step 1: Run BIT* on Multiple Problems
```bash
# Generate 10 problems and run BIT* (600 seconds each)
python run_multiple_comparisons.py --n-problems 10
```

**Time:** ~100 minutes (10 problems × 10 minutes each)

**Outputs:**
- `multi_run_results/problem_set.pt` - Problem definitions (reusable!)
- `multi_run_results/bitstar_result_*.pt` - BIT* results with interval tracking
- `multi_run_results/aggregated_results.pt` - Aggregated statistics
- `multi_run_results/statistics.yaml` - Summary statistics

---

### Step 2: Run MPD on Same Problems
```bash
# Run MPD inference on the same problem set
python run_inference_on_problems.py
```

**Time:** ~5-10 minutes (10 problems × 30-60 seconds each)

**Outputs:**
- `multi_run_results/mpd_results/mpd_result_*.pt` - MPD results
- `multi_run_results/mpd_results/mpd_aggregated_results.pt` - Aggregated MPD statistics
- `multi_run_results/mpd_results/mpd_statistics.yaml` - MPD summary

---

### Step 3: Compare Results
```bash
# Generate combined comparison
python run_multiple_comparisons.py --n-problems 10 --load-mpd
```

**Outputs:**
Side-by-side comparison of MPD vs BIT* showing:
- Success rates
- Path lengths (mean ± std)
- Planning/inference times
- Smoothness metrics
- BIT* anytime performance (time to first solution, time to beat MPD)

---

## 📊 What You Get

### For Each Problem (0-9):

**BIT* Results** (`bitstar_result_000.pt`):
- Final path length, smoothness, mean jerk
- Time to first solution
- **Interval metrics** (every 1 second):
  - Path length over time
  - Smoothness over time
  - Mean jerk over time
  - Tree size
- Complete solution trajectory

**MPD Results** (`mpd_results/mpd_result_000.pt`):
- Best path length from 64 samples
- Best smoothness
- Inference time (~2-3 seconds)
- Statistics over all 64 samples
- Collision-free success rate
- Complete best trajectory

### Aggregated Across All Problems:

**Statistics** (mean ± std):
- Success rates
- Path lengths
- Planning times
- Smoothness
- Mean jerk (BIT* only)
- Time to first solution (BIT* only)
- Time to beat MPD target (BIT* only)

---

## 💡 Customization

### More Problems
```bash
python run_multiple_comparisons.py --n-problems 20
python run_inference_on_problems.py
```

### Different Time Limits
```bash
# 5 minutes per problem
python run_multiple_comparisons.py --n-problems 10 --bitstar-time 300

# 15 minutes per problem
python run_multiple_comparisons.py --n-problems 10 --bitstar-time 900
```

### More MPD Samples
```bash
# 128 trajectory samples instead of 64
python run_inference_on_problems.py --n-samples 128
```

### Finer Tracking Intervals
```bash
# Track every 0.5 seconds instead of 1 second
python run_multiple_comparisons.py --n-problems 10 --interval 0.5
```

### Custom Output Directory
```bash
python run_multiple_comparisons.py --n-problems 10 --output-dir experiment_1
python run_inference_on_problems.py --problem-set experiment_1/problem_set.pt --output-dir experiment_1/mpd_results
```

---

## 📈 Example Analysis

### Plot BIT* Optimization vs MPD Target

```python
import torch
import matplotlib.pyplot as plt

# Load results for problem 0
bitstar = torch.load('multi_run_results/bitstar_result_000.pt')
mpd = torch.load('multi_run_results/mpd_results/mpd_result_000.pt')

# Extract BIT* optimization over time
times = [m['time'] for m in bitstar['interval_metrics'] if m['has_solution']]
lengths = [m['path_length'] for m in bitstar['interval_metrics'] if m['has_solution']]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(times, lengths, 'b-', linewidth=2, label='BIT* (anytime)')
plt.axhline(mpd['path_length'], color='r', linestyle='--', linewidth=2,
            label=f'MPD Target ({mpd["path_length"]:.3f})')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Path Length', fontsize=12)
plt.title('Problem 0: BIT* Optimization vs MPD Target', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problem_0_comparison.png', dpi=150)
print(f"Plot saved to: problem_0_comparison.png")
```

### Compare Success Rates

```python
import torch
import yaml

# Load aggregated statistics
with open('multi_run_results/statistics.yaml', 'r') as f:
    bitstar_stats = yaml.safe_load(f)['bitstar']

with open('multi_run_results/mpd_results/mpd_statistics.yaml', 'r') as f:
    mpd_stats = yaml.safe_load(f)['mpd']

print(f"BIT* Success Rate: {bitstar_stats['success_rate']*100:.1f}%")
print(f"MPD Success Rate: {mpd_stats['success_rate']*100:.1f}%")
print(f"\nBIT* Avg Path Length: {bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}")
print(f"MPD Avg Path Length: {mpd_stats['path_length_mean']:.3f} ± {mpd_stats['path_length_std']:.3f}")
```

---

## 🎯 Key Features

✅ **Fair Comparison**: Exact same start/goal pairs for both methods
✅ **Interval Tracking**: See how BIT* improves over time
✅ **Anytime Analysis**: Track when BIT* finds first solution and beats MPD
✅ **Reusable Problems**: Save problem set and reuse for different experiments
✅ **Complete Metrics**: Path length, smoothness, jerk, success rates, timing
✅ **Individual + Aggregate**: Results for each problem + statistics across all problems

---

## 📁 Directory Structure After Running

```
multi_run_results/
├── problem_set.pt              # Problem definitions (reusable)
├── problem_set.txt             # Human-readable problem list
├── bitstar_result_000.pt       # BIT* results for problem 0
├── bitstar_result_001.pt       # BIT* results for problem 1
├── ...
├── aggregated_results.pt       # All BIT* results combined
├── statistics.yaml             # BIT* summary stats
└── mpd_results/
    ├── mpd_result_000.pt       # MPD results for problem 0
    ├── mpd_result_001.pt       # MPD results for problem 1
    ├── ...
    ├── mpd_aggregated_results.pt  # All MPD results combined
    └── mpd_statistics.yaml     # MPD summary stats
```

---

## ⚡ Quick Test (2-3 Problems)

Test the workflow with just 2-3 problems to verify everything works:

```bash
# Step 1: Generate 3 problems and run BIT* (30 seconds each for quick test)
python run_multiple_comparisons.py --n-problems 3 --bitstar-time 30

# Step 2: Run MPD on same problems
python run_inference_on_problems.py

# Step 3: Compare
python run_multiple_comparisons.py --n-problems 3 --load-mpd
```

**Time:** ~2-3 minutes total

Once verified, scale up to your desired number of problems!
