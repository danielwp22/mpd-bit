# Changelog: Smoothness Fix

This document tracks all changes made to fix the MPD smoothness computation bug.

## Date
2024-12-22

## Issue
MPD smoothness was incorrectly computed as 0.0 in comparison scripts, while the regular inference pipeline correctly computed ~78.728 ± 25.483.

## Root Cause
Scripts were computing smoothness directly on B-spline control points instead of converting them to full trajectories (pos, vel, acc) first.

## Files Modified

### 1. `run_complete_comparison.py` ✅ FIXED
**Changes:**
- Modified `load_mpd_model()` to return `planning_task` (line 241)
  - Old: `return model, dataset, args_train`
  - New: `return model, dataset, planning_task, args_train`

- Modified `run_mpd_on_problems()` signature (line 281)
  - Added `planning_task` parameter

- Added trajectory conversion before metrics (lines 347-358)
  ```python
  # Convert control points to full trajectories
  q_traj_d = planning_task.parametric_trajectory.get_q_trajectory(
      samples, start_state, goal_state,
      get_type=("pos", "vel", "acc"),
      get_time_representation=True
  )
  ```

- Updated metrics computation (lines 361-362)
  - Old: `compute_smoothness(samples, robot)`
  - New: `compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)`

- Updated collision checking to use full trajectories (line 377)
- Updated best trajectory storage (line 368)
- Updated function calls in `main()` (line 783)

**Status:** ✅ Complete

---

### 2. `run_inference_on_problems.py` ✅ FIXED
**Changes:**
- Modified `load_mpd_model()` to return `planning_task` (line 85)
  - Old: `return model, dataset, args_train`
  - New: `return model, dataset, planning_task, args_train`

- Modified `run_inference_on_problem()` signature (line 88)
  - Added `planning_task` parameter

- Added trajectory conversion before metrics (lines 158-168)
  ```python
  # Convert control points to full trajectories
  q_traj_d = planning_task.parametric_trajectory.get_q_trajectory(...)
  q_trajs_pos = q_traj_d["pos"]
  q_trajs_vel = q_traj_d["vel"]
  q_trajs_acc = q_traj_d["acc"]
  ```

- Updated metrics computation (lines 171-172)
  - Old: `compute_smoothness(samples, robot)`
  - New: `compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)`

- Updated collision checking to use full trajectories (line 186)
- Updated best trajectory storage (line 178)
- Updated function calls in `main()` (line 446, 460)

**Status:** ✅ Complete

---

### 3. `test_mpd_only.py` ✅ FIXED
**Changes:**
- Updated `load_mpd_model()` call (line 48)
  - Old: `model, dataset, args = load_mpd_model(...)`
  - New: `model, dataset, planning_task, args = load_mpd_model(...)`

- Added trajectory conversion before metrics (lines 107-117)
  ```python
  # Convert control points to full trajectories for proper metrics
  q_traj_d = planning_task.parametric_trajectory.get_q_trajectory(...)
  ```

- Updated metrics computation (lines 120-121)
  - Old: `compute_smoothness(samples, robot)`
  - New: `compute_smoothness(q_trajs_pos, robot, trajs_acc=q_trajs_acc)`

- Updated collision checking loop (line 132)
  - Old: `for j in range(len(samples))`
  - New: `for j in range(len(q_trajs_pos))`

**Status:** ✅ Complete

---

### 4. `analyze_temporal_convergence.py` ✅ NEW
**Description:** New script for second-by-second convergence analysis

**Features:**
- Computes aggregate statistics at each time step across all problems
- Generates temporal plots showing convergence over time
- Outputs CSV and NPZ files with temporal data
- Automatically loads MPD target for comparison

**Status:** ✅ Complete

---

### 5. `README_TEMPORAL_ANALYSIS.md` ✅ NEW
**Description:** Complete guide for temporal analysis workflow

**Sections:**
- Quick start
- Metrics computed
- Usage examples
- Understanding results
- Advanced analysis
- Smoothness fix notes

**Status:** ✅ Complete

---

### 6. `SMOOTHNESS_FIX.md` ✅ NEW
**Description:** Detailed technical explanation of the bug and fix

**Sections:**
- The bug
- Root cause
- The fix
- Technical details
- Verification
- Impact on existing results

**Status:** ✅ Complete

---

### 7. `CHANGELOG_SMOOTHNESS_FIX.md` ✅ NEW
**Description:** This file - tracks all changes

**Status:** ✅ Complete

---

## Breaking Changes

### API Changes
All scripts that call `load_mpd_model()` must now handle 4 return values:
```python
# OLD (breaks with this update)
model, dataset, args = load_mpd_model(...)

# NEW (required)
model, dataset, planning_task, args = load_mpd_model(...)
```

### Affected Scripts
All scripts that use `load_mpd_model()` have been updated:
- ✅ `run_complete_comparison.py`
- ✅ `run_inference_on_problems.py`
- ✅ `test_mpd_only.py`

### Data Compatibility
**Important:** Results generated before this fix have incorrect MPD smoothness (0.0).

**Action Required:**
1. Delete old results: `rm -rf multi_run_results/`
2. Regenerate: `python run_complete_comparison.py --n-problems 10`
3. Re-analyze: `python analyze_comparison_results.py`

**Other metrics are still valid:**
- ✅ Path length - not affected
- ✅ BIT* smoothness - not affected
- ✅ Collision rates - not affected (actually improved with full trajectory checking)
- ❌ MPD smoothness - was 0.0, needs regeneration

---

## Verification

### Quick Test
```bash
# Run test script
python test_mpd_only.py

# Should show non-zero smoothness like:
#   Best smoothness: 78.XXX (not 0.0!)
```

### Full Verification
```bash
# Run small comparison
python run_complete_comparison.py --n-problems 2

# Check MPD smoothness
python -c "
import torch
r = torch.load('multi_run_results/mpd_results/mpd_result_000.pt')
print(f'Smoothness: {r[\"smoothness\"]:.3f}')
print(f'Expected: ~70-90 (NOT 0.0)')
"
```

---

## Technical Notes

### Why the Bug Existed
1. B-spline control points are just positions (shape: `[n_samples, n_control_points, 7]`)
2. `compute_smoothness()` tries to extract acceleration from state vectors
3. `robot.get_acceleration(x)` returns `x[..., 2*q_dim:3*q_dim]` (indices 14-21 for Panda)
4. Control points only have 7 dimensions, so indices 14-21 → empty tensor
5. Norm of empty tensor = 0.0

### Why It Works Now
1. Convert control points to full trajectories first
2. Parametric trajectory object computes pos, vel, acc via B-spline derivatives
3. Pass acceleration explicitly to `compute_smoothness(..., trajs_acc=q_trajs_acc)`
4. Acceleration is now computed correctly from B-spline derivatives

### Reference Implementation
The fix follows the same approach as the main inference pipeline:
- `mpd/inference/inference.py` lines 570-575 (compute_trajectories_from_control_points)
- `mpd/inference/inference.py` line 632 (smoothness with explicit acceleration)

---

## Future Work

### Potential Improvements
1. ⚠️ Consider refactoring `load_mpd_model()` to return a config object instead of multiple values
2. 💡 Add smoothness validation test to catch regressions
3. 📝 Update docstrings to clarify trajectory vs control point distinction

### Related Issues
- None currently

---

## Credits
- **Bug Report:** User observation that inference.py shows ~78.7 but comparison shows 0.0
- **Root Cause Analysis:** Traced to control point vs full trajectory distinction
- **Fix:** Added trajectory conversion following main inference pipeline
- **Date:** 2024-12-22
