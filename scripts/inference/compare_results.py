"""
Compare results between Diffusion Model and baseline planners (BIT*, RRT-Connect, etc.)
"""
import os
import torch
import yaml
import numpy as np
from pathlib import Path


def load_diffusion_results(results_dir="logs/2", idx=0):
    """Load diffusion model results."""
    results_file = os.path.join(results_dir, f"results_single_plan-{idx:03d}.pt")

    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return None

    results = torch.load(results_file, map_location='cpu')
    return results


def load_baseline_results(planner_name="bitstar", environment="panda_spheres3d"):
    """Load baseline planner results."""
    results_dir = f"logs_{planner_name.lower()}_{environment}"
    stats_file = os.path.join(results_dir, "statistics.yaml")

    if not os.path.exists(stats_file):
        print(f"Warning: {stats_file} not found")
        return None

    with open(stats_file, 'r') as f:
        stats = yaml.safe_load(f)

    return stats


def print_comparison(diff_results, baseline_stats, baseline_name="BIT*"):
    """Print side-by-side comparison."""

    print("\n" + "="*80)
    print(f"COMPARISON: Diffusion Model vs {baseline_name}")
    if baseline_stats and baseline_stats.get('mode') == 'anytime':
        print("(Baseline in ANYTIME mode)")
    print("="*80)

    if diff_results is None:
        print("\nDiffusion results not available!")
        return

    if baseline_stats is None:
        print(f"\n{baseline_name} results not available!")
        return

    diff_metrics = diff_results['metrics']
    is_anytime = baseline_stats.get('mode') == 'anytime'

    # Success Rate
    print(f"\n{'Metric':<30} {'Diffusion':<20} {baseline_name:<20}")
    print("-"*80)

    diff_success = diff_metrics['trajs_all']['success']
    diff_fraction_valid = diff_metrics['trajs_all']['fraction_valid']
    baseline_success_rate = baseline_stats.get('success_rate', 0)

    print(f"{'Success (single trajectory)':<30} {diff_success:<20} {baseline_stats.get('success_count', 0)}/{baseline_stats.get('n_problems', 0)}")
    print(f"{'Success rate (%)':<30} {diff_fraction_valid*100:.1f}%{'':<16} {baseline_success_rate*100:.1f}%")

    # Path Length
    diff_path_length = float(diff_metrics['trajs_best']['path_length'])
    baseline_path_length = baseline_stats.get('path_length_mean', float('inf'))
    baseline_path_length_std = baseline_stats.get('path_length_std', 0)

    # Convert 'inf' string to float if needed
    if isinstance(baseline_path_length, str) and baseline_path_length == 'inf':
        baseline_path_length = float('inf')
    if isinstance(baseline_path_length_std, str) and baseline_path_length_std == 'inf':
        baseline_path_length_std = float('inf')

    baseline_path_length = float(baseline_path_length)
    baseline_path_length_std = float(baseline_path_length_std)

    # Format inf values nicely
    if baseline_path_length == float('inf'):
        baseline_path_str = "inf"
    else:
        baseline_path_str = f"{baseline_path_length:.3f} ± {baseline_path_length_std:.3f}"

    print(f"\n{'Path Length (best/mean)':<30} {diff_path_length:.3f}{'':<16} {baseline_path_str}")

    if 'path_length_mean' in diff_metrics.get('trajs_valid', {}):
        diff_path_length_mean = float(diff_metrics['trajs_valid']['path_length_mean'])
        diff_path_length_std = float(diff_metrics['trajs_valid']['path_length_std'])
        print(f"{'  (over valid trajectories)':<30} {diff_path_length_mean:.3f} ± {diff_path_length_std:.3f}")

    # Anytime mode: show first solution path length
    if is_anytime and 'path_length_first_mean' in baseline_stats:
        baseline_first_length = baseline_stats.get('path_length_first_mean', float('inf'))
        baseline_first_length_std = baseline_stats.get('path_length_first_std', 0)

        # Convert 'inf' string to float if needed
        if isinstance(baseline_first_length, str) and baseline_first_length == 'inf':
            baseline_first_length = float('inf')
        if isinstance(baseline_first_length_std, str) and baseline_first_length_std == 'inf':
            baseline_first_length_std = float('inf')

        baseline_first_length = float(baseline_first_length)
        baseline_first_length_std = float(baseline_first_length_std)

        if baseline_first_length == float('inf'):
            print(f"{'  First solution path length':<30} {'N/A':<20} inf")
        else:
            print(f"{'  First solution path length':<30} {'N/A':<20} {baseline_first_length:.3f} ± {baseline_first_length_std:.3f}")

    # Planning/Inference Time
    diff_time = diff_results.get('t_inference_total', 0)

    if is_anytime:
        # Show multiple timing metrics for anytime mode
        baseline_time_first = baseline_stats.get('planning_time_first_mean', 0)
        baseline_time_first_std = baseline_stats.get('planning_time_first_std', 0)
        baseline_time_total = baseline_stats.get('planning_time_total_mean', 0)
        baseline_time_total_std = baseline_stats.get('planning_time_total_std', 0)

        print(f"\n{'Planning Time (sec)':<30} {'Diffusion':<20} {baseline_name:<20}")
        print(f"{'  Total time':<30} {diff_time:.3f}{'':<16} {baseline_time_total:.3f} ± {baseline_time_total_std:.3f}")
        print(f"{'  Time to first solution':<30} {diff_time:.3f}{'':<16} {baseline_time_first:.3f} ± {baseline_time_first_std:.3f}")

        # Show time to reach target quality if available
        if baseline_stats.get('target_path_length') is not None:
            target_length = baseline_stats['target_path_length']
            reached_target_rate = baseline_stats.get('reached_target_rate', 0)

            print(f"\n{'Target Quality Comparison':<30}")
            print(f"  {'Target path length':<28} {target_length:.3f}")
            print(f"  {'Reached target (%)':<28} N/A{'':<16} {reached_target_rate*100:.1f}%")

            if baseline_stats.get('planning_time_target_mean', float('inf')) != float('inf'):
                baseline_time_target = baseline_stats['planning_time_target_mean']
                baseline_time_target_std = baseline_stats.get('planning_time_target_std', 0)
                print(f"  {'Time to reach target':<28} {diff_time:.3f} sec{'':<10} {baseline_time_target:.3f} ± {baseline_time_target_std:.3f} sec")
    else:
        # Regular mode timing
        baseline_time = baseline_stats.get('planning_time_mean', 0)
        baseline_time_std = baseline_stats.get('planning_time_std', 0)
        print(f"\n{'Planning Time (sec)':<30} {diff_time:.3f}{'':<16} {baseline_time:.3f} ± {baseline_time_std:.3f}")

    # Breakdown of diffusion time
    if 't_generator' in diff_results:
        print(f"  {'Generator time':<28} {diff_results['t_generator']:.3f}")
    if 't_guide' in diff_results:
        print(f"  {'Guidance time':<28} {diff_results['t_guide']:.3f}")

    # Smoothness
    if 'smoothness' in diff_metrics.get('trajs_best', {}):
        diff_smoothness = float(diff_metrics['trajs_best']['smoothness'])
        baseline_smoothness = baseline_stats.get('smoothness_mean', None)
        baseline_smoothness_std = baseline_stats.get('smoothness_std', 0)

        # Convert 'inf' string to float if needed
        if isinstance(baseline_smoothness, str) and baseline_smoothness == 'inf':
            baseline_smoothness = float('inf')
        if isinstance(baseline_smoothness_std, str) and baseline_smoothness_std == 'inf':
            baseline_smoothness_std = float('inf')

        if baseline_smoothness is not None:
            # Try to convert to float
            try:
                baseline_smoothness = float(baseline_smoothness)
                baseline_smoothness_std = float(baseline_smoothness_std)
                if baseline_smoothness != float('inf'):
                    baseline_str = f"{baseline_smoothness:.3f} ± {baseline_smoothness_std:.3f}"
                    print(f"\n{'Smoothness (lower=better)':<30} {diff_smoothness:.3f}{'':<16} {baseline_str}")
                else:
                    print(f"\n{'Smoothness (lower=better)':<30} {diff_smoothness:.3f}{'':<16} N/A")
            except (ValueError, TypeError):
                print(f"\n{'Smoothness (lower=better)':<30} {diff_smoothness:.3f}{'':<16} N/A")
        else:
            print(f"\n{'Smoothness (lower=better)':<30} {diff_smoothness:.3f}{'':<16} N/A")

    # Mean Jerk
    baseline_mean_jerk = baseline_stats.get('mean_jerk_mean', None)
    baseline_mean_jerk_std = baseline_stats.get('mean_jerk_std', 0)

    if baseline_mean_jerk is not None:
        try:
            baseline_mean_jerk = float(baseline_mean_jerk)
            baseline_mean_jerk_std = float(baseline_mean_jerk_std)
            if baseline_mean_jerk != float('inf'):
                baseline_jerk_str = f"{baseline_mean_jerk:.3f} ± {baseline_mean_jerk_std:.3f}"
                print(f"{'Mean Jerk (lower=better)':<30} {'N/A':<20} {baseline_jerk_str}")
            else:
                print(f"{'Mean Jerk (lower=better)':<30} {'N/A':<20} N/A")
        except (ValueError, TypeError):
            print(f"{'Mean Jerk (lower=better)':<30} {'N/A':<20} N/A")

    # Collision Intensity (diffusion only)
    if 'collision_intensity' in diff_metrics.get('trajs_all', {}):
        diff_collision = float(diff_metrics['trajs_all']['collision_intensity'])
        print(f"{'Collision Intensity':<30} {diff_collision:.4f}{'':<16} N/A")

    # Diversity (diffusion only)
    if 'diversity' in diff_metrics.get('trajs_valid', {}):
        diff_diversity = float(diff_metrics['trajs_valid']['diversity'])
        print(f"{'Trajectory Diversity':<30} {diff_diversity:.3f}{'':<16} N/A (single traj)")

    # Goal Error (diffusion only)
    if 'ee_pose_goal_error_position_norm' in diff_metrics.get('trajs_best', {}):
        diff_pos_error = float(diff_metrics['trajs_best']['ee_pose_goal_error_position_norm'])
        diff_ori_error = float(diff_metrics['trajs_best']['ee_pose_goal_error_orientation_norm'])
        print(f"\n{'End-Effector Goal Error':<30}")
        print(f"  {'Position (m)':<28} {diff_pos_error:.4f}")
        print(f"  {'Orientation (rad)':<28} {diff_ori_error:.4f}")

    # IsaacGym statistics (if available)
    if 'isaacgym_statistics' in diff_results:
        ig_stats = diff_results['isaacgym_statistics']
        if ig_stats and len(ig_stats) > 0:
            print(f"\n{'IsaacGym Validation':<30}")
            print(f"  {'Free trajectories':<28} {ig_stats.get('n_trajectories_free', 0)}/{ig_stats.get('n_trajectories_free', 0) + ig_stats.get('n_trajectories_collision', 0)}")
            print(f"  {'Fraction free':<28} {ig_stats.get('n_trajectories_free_fraction', 0)*100:.1f}%")

    print("="*80 + "\n")


def compare_multiple_baselines(diff_results_dir="logs/2", diff_idx=0,
                               baseline_planners=["bitstar", "rrtconnect", "rrtstar"],
                               environment="panda_spheres3d"):
    """Compare diffusion model against multiple baseline planners."""

    diff_results = load_diffusion_results(diff_results_dir, diff_idx)

    for planner in baseline_planners:
        baseline_stats = load_baseline_results(planner, environment)
        if baseline_stats:
            print_comparison(diff_results, baseline_stats, planner.upper())


def load_bitstar_single_run_results(bitstar_results_file="bitstar_result.pt"):
    """Load results from a single BIT* run (for detailed timing comparison)."""
    if not os.path.exists(bitstar_results_file):
        return None

    results = torch.load(bitstar_results_file, map_location='cpu')
    return results


def print_anytime_comparison(diff_results, bitstar_result):
    """Print detailed anytime comparison showing when BIT* finds solutions."""

    print("\n" + "="*80)
    print("ANYTIME OPTIMIZATION COMPARISON: MPD vs BIT*")
    print("="*80)

    if diff_results is None or bitstar_result is None:
        print("\nResults not available for comparison!")
        return

    mpd_metrics = diff_results['metrics']
    mpd_path_length = float(mpd_metrics['trajs_best']['path_length'])
    mpd_time = diff_results.get('t_inference_total', 0)

    bitstar_path_length = bitstar_result.get('path_length', float('inf'))
    bitstar_time = bitstar_result.get('planning_time', 0)
    bitstar_first_time = bitstar_result.get('time_to_first_solution')
    bitstar_first_cost = bitstar_result.get('first_solution_cost')
    bitstar_target_time = bitstar_result.get('time_to_target_quality')

    print(f"\n{'Metric':<45} {'MPD':<20} {'BIT*':<20}")
    print("-"*80)
    print(f"{'Final path length':<45} {mpd_path_length:.3f}{'':<16} {bitstar_path_length:.3f}")
    print(f"{'Total time (sec)':<45} {mpd_time:.3f}{'':<16} {bitstar_time:.3f}")

    if bitstar_first_time is not None:
        print(f"\n{'BIT* Anytime Behavior':<45}")
        print("-"*80)
        print(f"{'Time to FIRST solution (sec)':<45} {'':<20} {bitstar_first_time:.3f}")
        print(f"{'First solution path length':<45} {'':<20} {bitstar_first_cost:.3f}")

        if bitstar_target_time is not None:
            print(f"{'Time to BEAT MPD quality (sec)':<45} {'':<20} {bitstar_target_time:.3f}")
            speedup = mpd_time / bitstar_target_time if bitstar_target_time > 0 else float('inf')
            print(f"{'Speedup vs MPD':<45} {'':<20} {speedup:.2f}x")
        else:
            print(f"{'Time to BEAT MPD quality':<45} {'':<20} Did not reach")

        # Improvement from first to final
        if bitstar_first_cost and bitstar_first_cost > 0:
            improvement = (bitstar_first_cost - bitstar_path_length) / bitstar_first_cost * 100
            print(f"{'BIT* path improvement (first → final)':<45} {'':<20} {improvement:.1f}%")

    # Quality comparison
    print(f"\n{'Quality Comparison':<45}")
    print("-"*80)
    if bitstar_path_length < mpd_path_length:
        improvement = (mpd_path_length - bitstar_path_length) / mpd_path_length * 100
        print(f"{'BIT* vs MPD':<45} {'':<20} {improvement:.1f}% shorter")
    elif bitstar_path_length > mpd_path_length:
        diff = (bitstar_path_length - mpd_path_length) / mpd_path_length * 100
        print(f"{'MPD vs BIT*':<45} {'':<20} {diff:.1f}% shorter")
    else:
        print(f"{'Result':<45} {'':<20} Similar quality")

    print("="*80 + "\n")


def print_summary_table(diff_results_dir="logs/2", diff_idx=0,
                       baseline_planners=["bitstar", "rrtconnect"],
                       environment="panda_spheres3d"):
    """Print a compact summary table for easy comparison."""

    diff_results = load_diffusion_results(diff_results_dir, diff_idx)

    print("\n" + "="*120)
    print("SUMMARY COMPARISON TABLE")
    print("="*120)
    print(f"{'Method':<20} {'Success %':<12} {'Path Length':<15} {'Time (sec)':<12} {'Smoothness':<12} {'Mean Jerk':<12}")
    print("-"*120)

    # Diffusion row
    if diff_results:
        diff_metrics = diff_results['metrics']
        success = diff_metrics['trajs_all']['fraction_valid'] * 100
        path_length = float(diff_metrics['trajs_best']['path_length'])
        time_total = diff_results.get('t_inference_total', 0)
        smoothness = float(diff_metrics['trajs_best'].get('smoothness', 0))

        print(f"{'Diffusion':<20} {success:<12.1f} {path_length:<15.3f} {time_total:<12.3f} {smoothness:<12.3f} {'N/A':<12}")

    # Baseline rows
    for planner in baseline_planners:
        baseline_stats = load_baseline_results(planner, environment)
        if baseline_stats:
            success = baseline_stats.get('success_rate', 0) * 100
            path_length = baseline_stats.get('path_length_mean', 0)
            path_std = baseline_stats.get('path_length_std', 0)
            time_mean = baseline_stats.get('planning_time_mean', 0)
            smoothness_mean = baseline_stats.get('smoothness_mean', None)
            smoothness_std = baseline_stats.get('smoothness_std', 0)
            mean_jerk_mean = baseline_stats.get('mean_jerk_mean', None)
            mean_jerk_std = baseline_stats.get('mean_jerk_std', 0)

            path_str = f"{path_length:.3f}±{path_std:.2f}"
            if smoothness_mean is not None and smoothness_mean != float('inf'):
                smoothness_str = f"{smoothness_mean:.1f}±{smoothness_std:.1f}"
            else:
                smoothness_str = "N/A"
            if mean_jerk_mean is not None and mean_jerk_mean != float('inf'):
                mean_jerk_str = f"{mean_jerk_mean:.3f}±{mean_jerk_std:.3f}"
            else:
                mean_jerk_str = "N/A"
            print(f"{planner.upper():<20} {success:<12.1f} {path_str:<15} {time_mean:<12.3f} {smoothness_str:<12} {mean_jerk_str:<12}")

    print("="*120 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare diffusion model with baseline planners")
    parser.add_argument("--diffusion-dir", default="logs/2", help="Diffusion results directory")
    parser.add_argument("--diffusion-idx", type=int, default=0, help="Diffusion result index")
    parser.add_argument("--baselines", nargs="+", default=["bitstar"],
                       help="Baseline planners to compare (e.g., bitstar rrtconnect)")
    parser.add_argument("--environment", default="panda_spheres3d",
                       help="Environment name")
    parser.add_argument("--summary", action="store_true",
                       help="Print compact summary table")
    parser.add_argument("--anytime", action="store_true",
                       help="Show anytime comparison with detailed BIT* timing")
    parser.add_argument("--bitstar-result", default="bitstar_result.pt",
                       help="Path to BIT* single run result file (for anytime comparison)")

    args = parser.parse_args()

    if args.anytime:
        # Show detailed anytime comparison
        diff_results = load_diffusion_results(args.diffusion_dir, args.diffusion_idx)
        bitstar_result = load_bitstar_single_run_results(args.bitstar_result)
        print_anytime_comparison(diff_results, bitstar_result)
    elif args.summary:
        print_summary_table(
            diff_results_dir=args.diffusion_dir,
            diff_idx=args.diffusion_idx,
            baseline_planners=args.baselines,
            environment=args.environment
        )
    else:
        compare_multiple_baselines(
            diff_results_dir=args.diffusion_dir,
            diff_idx=args.diffusion_idx,
            baseline_planners=args.baselines,
            environment=args.environment
        )

    # Example usage message
    print("\nUsage examples:")
    print("  python compare_results.py")
    print("  python compare_results.py --baselines bitstar rrtconnect rrtstar")
    print("  python compare_results.py --summary")
    print("  python compare_results.py --anytime --bitstar-result bitstar_result.pt")
    print("  python compare_results.py --diffusion-dir logs/2 --diffusion-idx 0")
