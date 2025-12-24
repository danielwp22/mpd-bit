# BIT* vs MPD Comparison Tools

This directory contains tools for comparing the BIT* baseline planner with the MPD (Motion Planning Diffusion) model.

## Overview

Two comparison approaches are available:

1. **run_bitstar_vs_mpd.py** - Direct head-to-head comparison on the same problem
2. **compare_results.py** - Compare aggregate statistics from multiple runs

## Quick Start

### 1. Direct Comparison (Recommended for Anytime Analysis)

Run BIT* on the same problem as MPD and see detailed timing:

```bash
# Run comparison with 30 second time limit
python run_bitstar_vs_mpd.py --time 30

# Use a different MPD result file
python run_bitstar_vs_mpd.py --mpd-results logs/2/results_single_plan-001.pt --time 60
```

**Output shows:**
- ✅ **Time to FIRST solution** - When BIT* finds any valid path
- ✅ **Time to BEAT MPD quality** - When BIT* finds a path shorter than MPD's
- Path improvement over time
- Final quality comparison

**Example output:**
```
BIT* Anytime Performance
--------------------------------------------------------------------------------
Time to FIRST solution                                        5.438 sec
First solution cost                                           18.110
Time to BEAT MPD quality                                      Did not reach
Path improvement (first → final)                              47.5%

✗ MPD found 36.0% shorter path than BIT*
```

### 2. Aggregate Statistics Comparison

Compare multiple runs using pre-computed statistics:

```bash
# Compare with baseline statistics
python compare_results.py

# Compare multiple baselines
python compare_results.py --baselines bitstar rrtconnect rrtstar

# Show compact summary table
python compare_results.py --summary
```

## Detailed Timing Information

The BIT* implementation now tracks:

### Timing Metrics
- `time_to_first_solution`: Time when any valid solution is found
- `time_to_target_quality`: Time when path length beats the target (MPD)
- `first_solution_cost`: Path length of the first solution
- `path_length`: Final optimized path length
- `planning_time`: Total planning time

### Example Use Case

To see if BIT* can beat MPD faster with more optimization time:

```bash
# Run with different time limits
python run_bitstar_vs_mpd.py --time 30
python run_bitstar_vs_mpd.py --time 60
python run_bitstar_vs_mpd.py --time 120
```

## Understanding the Results

### BIT* Anytime Behavior

BIT* exhibits anytime optimization:
1. **Finds initial solution quickly** (typically 3-10 seconds)
2. **Continuously improves** the path quality over time
3. **May or may not beat MPD** depending on time available

### Key Insights

- **MPD**: Fast (2-3 sec) but single-shot, no improvement over time
- **BIT***: Slower initial solution but can improve with more time
- **Comparison depends on**:
  - Problem difficulty
  - Time available
  - Quality requirements

## Files

- `bitstar_minimal_template.py` - GPU-batched BIT* implementation (no OMPL)
- `run_bitstar_vs_mpd.py` - Direct comparison script
- `compare_results.py` - Aggregate statistics comparison
- `README_comparison.md` - This file

## Notes

- BIT* uses GPU batching for collision checking (no OMPL dependency)
- Smoothness is computed using finite differences for position-only trajectories
- Goal region radius is set to 0.2 for fair comparison
- Anytime optimization runs for the full time limit (no early termination)
