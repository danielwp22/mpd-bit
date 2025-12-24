"""
Analyze second-by-second convergence of BIT* across multiple problems.

This script:
1. Loads all BIT* results from a complete comparison run
2. Computes second-by-second averages across all problems
3. Generates plots showing convergence behavior over time
4. Saves aggregate temporal statistics

Metrics computed at each time step:
- Average path length (across problems with solutions)
- Average tree size
- Average smoothness
- Success rate (% of problems with a solution)
- Standard deviations for all metrics
"""
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def load_bitstar_results(results_dir="multi_run_results"):
    """Load all individual BIT* results."""
    print(f"\nLoading BIT* results from: {results_dir}")

    bitstar_results = []
    idx = 0
    while True:
        result_file = os.path.join(results_dir, f"bitstar_result_{idx:03d}.pt")
        if not os.path.exists(result_file):
            break

        result = torch.load(result_file, map_location='cpu')
        bitstar_results.append(result)
        idx += 1

    print(f"Loaded {len(bitstar_results)} BIT* results\n")
    return bitstar_results


def compute_temporal_statistics(bitstar_results, time_step=1.0, max_time=None):
    """
    Compute statistics at each time step across all problems.

    Args:
        bitstar_results: List of BIT* result dictionaries
        time_step: Time step for binning (seconds)
        max_time: Maximum time to analyze (None = auto-detect)

    Returns:
        Dictionary with temporal statistics
    """
    print("Computing temporal statistics...")

    # Determine time range
    if max_time is None:
        max_time = 0.0
        for result in bitstar_results:
            if result.get('success') and result.get('interval_metrics'):
                result_max = max(m['time'] for m in result['interval_metrics'])
                max_time = max(max_time, result_max)

    # Create time bins
    time_points = np.arange(0, max_time + time_step, time_step)
    n_times = len(time_points)
    n_problems = len(bitstar_results)

    print(f"Time range: 0 to {max_time:.1f} seconds")
    print(f"Time step: {time_step:.1f} seconds")
    print(f"Number of time points: {n_times}")
    print(f"Number of problems: {n_problems}\n")

    # Initialize storage for each time point
    temporal_data = {
        'time_points': time_points,
        'path_lengths': defaultdict(list),  # {time_idx: [lengths]}
        'smoothness': defaultdict(list),
        'mean_jerk': defaultdict(list),
        'tree_sizes': defaultdict(list),
        'num_samples': defaultdict(list),
        'num_batches': defaultdict(list),
        'has_solution': defaultdict(list),  # {time_idx: [True/False]}
    }

    # Process each problem
    for prob_idx, result in enumerate(bitstar_results):
        if not result.get('success'):
            continue

        interval_metrics = result.get('interval_metrics', [])
        if not interval_metrics:
            continue

        # Group metrics by time bin
        for metric in interval_metrics:
            t = metric['time']
            # Find closest time point
            time_idx = int(round(t / time_step))
            if time_idx >= n_times:
                continue

            # Record whether this problem has a solution at this time
            has_sol = metric.get('has_solution', False)
            temporal_data['has_solution'][time_idx].append(has_sol)

            # Only record metrics if solution exists
            if has_sol:
                temporal_data['path_lengths'][time_idx].append(metric['path_length'])
                temporal_data['smoothness'][time_idx].append(metric.get('smoothness', 0.0))
                temporal_data['mean_jerk'][time_idx].append(metric.get('mean_jerk', 0.0))

            # Always record tree stats (solution or not)
            temporal_data['tree_sizes'][time_idx].append(metric.get('num_vertices', 0))
            temporal_data['num_samples'][time_idx].append(metric.get('num_samples', 0))
            temporal_data['num_batches'][time_idx].append(metric.get('num_batches', 0))

    # Compute statistics at each time point
    stats = {
        'time_points': time_points,
        'n_problems': n_problems,
        'time_step': time_step,
        'max_time': max_time,
    }

    # Success rate at each time
    success_rate = np.zeros(n_times)
    success_count = np.zeros(n_times)
    for time_idx in range(n_times):
        if temporal_data['has_solution'][time_idx]:
            success_count[time_idx] = sum(temporal_data['has_solution'][time_idx])
            success_rate[time_idx] = success_count[time_idx] / n_problems

    stats['success_rate'] = success_rate
    stats['success_count'] = success_count

    # Path length statistics (only for problems with solutions)
    path_length_mean = np.full(n_times, np.nan)
    path_length_std = np.full(n_times, np.nan)
    path_length_min = np.full(n_times, np.nan)
    path_length_max = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['path_lengths'][time_idx]:
            lengths = np.array(temporal_data['path_lengths'][time_idx])
            path_length_mean[time_idx] = np.mean(lengths)
            path_length_std[time_idx] = np.std(lengths)
            path_length_min[time_idx] = np.min(lengths)
            path_length_max[time_idx] = np.max(lengths)

    stats['path_length_mean'] = path_length_mean
    stats['path_length_std'] = path_length_std
    stats['path_length_min'] = path_length_min
    stats['path_length_max'] = path_length_max

    # Smoothness statistics
    smoothness_mean = np.full(n_times, np.nan)
    smoothness_std = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['smoothness'][time_idx]:
            vals = np.array(temporal_data['smoothness'][time_idx])
            smoothness_mean[time_idx] = np.mean(vals)
            smoothness_std[time_idx] = np.std(vals)

    stats['smoothness_mean'] = smoothness_mean
    stats['smoothness_std'] = smoothness_std

    # Mean jerk statistics
    mean_jerk_mean = np.full(n_times, np.nan)
    mean_jerk_std = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['mean_jerk'][time_idx]:
            vals = np.array(temporal_data['mean_jerk'][time_idx])
            mean_jerk_mean[time_idx] = np.mean(vals)
            mean_jerk_std[time_idx] = np.std(vals)

    stats['mean_jerk_mean'] = mean_jerk_mean
    stats['mean_jerk_std'] = mean_jerk_std

    # Tree size statistics
    tree_size_mean = np.full(n_times, np.nan)
    tree_size_std = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['tree_sizes'][time_idx]:
            vals = np.array(temporal_data['tree_sizes'][time_idx])
            tree_size_mean[time_idx] = np.mean(vals)
            tree_size_std[time_idx] = np.std(vals)

    stats['tree_size_mean'] = tree_size_mean
    stats['tree_size_std'] = tree_size_std

    # Number of samples statistics
    num_samples_mean = np.full(n_times, np.nan)
    num_samples_std = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['num_samples'][time_idx]:
            vals = np.array(temporal_data['num_samples'][time_idx])
            num_samples_mean[time_idx] = np.mean(vals)
            num_samples_std[time_idx] = np.std(vals)

    stats['num_samples_mean'] = num_samples_mean
    stats['num_samples_std'] = num_samples_std

    # Number of batches statistics
    num_batches_mean = np.full(n_times, np.nan)
    num_batches_std = np.full(n_times, np.nan)

    for time_idx in range(n_times):
        if temporal_data['num_batches'][time_idx]:
            vals = np.array(temporal_data['num_batches'][time_idx])
            num_batches_mean[time_idx] = np.mean(vals)
            num_batches_std[time_idx] = np.std(vals)

    stats['num_batches_mean'] = num_batches_mean
    stats['num_batches_std'] = num_batches_std

    print("Temporal statistics computed successfully\n")
    return stats


def print_temporal_summary(stats):
    """Print summary of temporal statistics."""
    print("\n" + "="*80)
    print("TEMPORAL CONVERGENCE SUMMARY")
    print("="*80 + "\n")

    time_points = stats['time_points']
    success_rate = stats['success_rate']
    path_length_mean = stats['path_length_mean']

    # Find key milestones
    first_solution_idx = np.where(success_rate > 0)[0]
    if len(first_solution_idx) > 0:
        first_solution_time = time_points[first_solution_idx[0]]
        print(f"First solution found at: {first_solution_time:.1f}s")

    all_solved_idx = np.where(success_rate >= 1.0)[0]
    if len(all_solved_idx) > 0:
        all_solved_time = time_points[all_solved_idx[0]]
        print(f"All problems solved by: {all_solved_time:.1f}s")
    else:
        print(f"Not all problems solved within {stats['max_time']:.1f}s")

    # Final statistics
    final_idx = -1
    while final_idx >= -len(time_points) and np.isnan(path_length_mean[final_idx]):
        final_idx -= 1

    if final_idx >= -len(time_points):
        print(f"\nFinal statistics (at t={time_points[final_idx]:.1f}s):")
        print(f"  Success rate: {success_rate[final_idx]*100:.1f}%")
        print(f"  Average path length: {path_length_mean[final_idx]:.3f} ± {stats['path_length_std'][final_idx]:.3f}")
        print(f"  Average tree size: {stats['tree_size_mean'][final_idx]:.0f} ± {stats['tree_size_std'][final_idx]:.0f}")

    print("="*80 + "\n")


def plot_convergence_overview(stats, output_dir, mpd_target=None):
    """Plot overview of BIT* convergence over time."""
    if not HAS_MATPLOTLIB:
        return

    time_points = stats['time_points']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Success rate over time
    ax = axes[0, 0]
    ax.plot(time_points, stats['success_rate'] * 100, 'b-', linewidth=2)
    ax.fill_between(time_points, 0, stats['success_rate'] * 100, alpha=0.3)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # 2. Path length over time
    ax = axes[0, 1]
    valid_mask = ~np.isnan(stats['path_length_mean'])
    times_valid = time_points[valid_mask]
    mean_valid = stats['path_length_mean'][valid_mask]
    std_valid = stats['path_length_std'][valid_mask]

    ax.plot(times_valid, mean_valid, 'b-', linewidth=2, label='BIT* Mean')
    ax.fill_between(times_valid, mean_valid - std_valid, mean_valid + std_valid,
                    alpha=0.3, label='BIT* ±1 std')

    # Add MPD target if provided
    if mpd_target is not None:
        ax.axhline(mpd_target, color='r', linestyle='--', linewidth=2,
                  label=f'MPD Target ({mpd_target:.3f})')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Path Length', fontsize=12)
    ax.set_title('Path Length Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. Tree size growth
    ax = axes[1, 0]
    valid_mask = ~np.isnan(stats['tree_size_mean'])
    times_valid = time_points[valid_mask]
    mean_valid = stats['tree_size_mean'][valid_mask]
    std_valid = stats['tree_size_std'][valid_mask]

    ax.plot(times_valid, mean_valid, 'g-', linewidth=2, label='Mean')
    ax.fill_between(times_valid, mean_valid - std_valid, mean_valid + std_valid,
                    alpha=0.3, color='g', label='±1 std')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Tree Size (vertices)', fontsize=12)
    ax.set_title('BIT* Tree Growth Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Number of batches
    ax = axes[1, 1]
    valid_mask = ~np.isnan(stats['num_batches_mean'])
    times_valid = time_points[valid_mask]
    mean_valid = stats['num_batches_mean'][valid_mask]

    ax.plot(times_valid, mean_valid, 'purple', linewidth=2)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Number of Batches', fontsize=12)
    ax.set_title('BIT* Batches Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'temporal_convergence_overview.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_path_length_detail(stats, output_dir, mpd_target=None):
    """Plot detailed path length convergence with min/max."""
    if not HAS_MATPLOTLIB:
        return

    time_points = stats['time_points']
    valid_mask = ~np.isnan(stats['path_length_mean'])
    times_valid = time_points[valid_mask]
    mean_valid = stats['path_length_mean'][valid_mask]
    std_valid = stats['path_length_std'][valid_mask]
    min_valid = stats['path_length_min'][valid_mask]
    max_valid = stats['path_length_max'][valid_mask]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot mean with shaded std
    ax.plot(times_valid, mean_valid, 'b-', linewidth=2.5, label='Mean', zorder=3)
    ax.fill_between(times_valid, mean_valid - std_valid, mean_valid + std_valid,
                    alpha=0.3, color='blue', label='±1 std', zorder=2)

    # Plot min/max envelope
    ax.plot(times_valid, min_valid, 'g--', linewidth=1.5, alpha=0.7, label='Min', zorder=2)
    ax.plot(times_valid, max_valid, 'r--', linewidth=1.5, alpha=0.7, label='Max', zorder=2)
    ax.fill_between(times_valid, min_valid, max_valid, alpha=0.1, color='gray', zorder=1)

    # Add MPD target if provided
    if mpd_target is not None:
        ax.axhline(mpd_target, color='darkred', linestyle='--', linewidth=2.5,
                  label=f'MPD Target ({mpd_target:.3f})', zorder=4)

        # Find when mean crosses MPD target
        cross_idx = np.where(mean_valid <= mpd_target)[0]
        if len(cross_idx) > 0:
            cross_time = times_valid[cross_idx[0]]
            ax.axvline(cross_time, color='orange', linestyle=':', linewidth=2,
                      alpha=0.7, label=f'Matched MPD ({cross_time:.1f}s)', zorder=3)

    ax.set_xlabel('Time (seconds)', fontsize=13)
    ax.set_ylabel('Path Length', fontsize=13)
    ax.set_title('BIT* Path Length Convergence (Detailed)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'path_length_convergence_detailed.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_smoothness_convergence(stats, output_dir):
    """Plot smoothness and jerk convergence."""
    if not HAS_MATPLOTLIB:
        return

    time_points = stats['time_points']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Smoothness
    ax = axes[0]
    valid_mask = ~np.isnan(stats['smoothness_mean'])
    if valid_mask.any():
        times_valid = time_points[valid_mask]
        mean_valid = stats['smoothness_mean'][valid_mask]
        std_valid = stats['smoothness_std'][valid_mask]

        ax.plot(times_valid, mean_valid, 'b-', linewidth=2, label='Mean')
        ax.fill_between(times_valid, mean_valid - std_valid, mean_valid + std_valid,
                        alpha=0.3, label='±1 std')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Smoothness', fontsize=12)
        ax.set_title('Smoothness Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No smoothness data available',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)

    # Mean jerk
    ax = axes[1]
    valid_mask = ~np.isnan(stats['mean_jerk_mean'])
    if valid_mask.any():
        times_valid = time_points[valid_mask]
        mean_valid = stats['mean_jerk_mean'][valid_mask]
        std_valid = stats['mean_jerk_std'][valid_mask]

        ax.plot(times_valid, mean_valid, 'r-', linewidth=2, label='Mean')
        ax.fill_between(times_valid, mean_valid - std_valid, mean_valid + std_valid,
                        alpha=0.3, color='r', label='±1 std')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean Jerk', fontsize=12)
        ax.set_title('Mean Jerk Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No jerk data available',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'smoothness_jerk_convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_sampling_efficiency(stats, output_dir):
    """Plot sampling and tree growth efficiency."""
    if not HAS_MATPLOTLIB:
        return

    time_points = stats['time_points']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Samples over time
    ax = axes[0]
    valid_mask = ~np.isnan(stats['num_samples_mean'])
    if valid_mask.any():
        times_valid = time_points[valid_mask]
        mean_valid = stats['num_samples_mean'][valid_mask]

        ax.plot(times_valid, mean_valid, 'b-', linewidth=2)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Total Samples Generated', fontsize=12)
        ax.set_title('Sampling Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Success rate vs tree size
    ax = axes[1]
    valid_success = ~np.isnan(stats['tree_size_mean'])
    if valid_success.any():
        tree_sizes = stats['tree_size_mean'][valid_success]
        success_rates = stats['success_rate'][valid_success] * 100
        times_valid = time_points[valid_success]

        # Color by time
        scatter = ax.scatter(tree_sizes, success_rates, c=times_valid,
                           cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Time (s)')
        ax.set_xlabel('Average Tree Size (vertices)', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate vs Tree Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'sampling_efficiency.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def save_temporal_statistics(stats, output_dir):
    """Save temporal statistics to file."""
    # Save as numpy
    output_file = os.path.join(output_dir, 'temporal_statistics.npz')
    np.savez(output_file, **stats)
    print(f"Saved temporal statistics to: {output_file}")

    # Save as CSV for easy viewing
    csv_file = os.path.join(output_dir, 'temporal_statistics.csv')
    with open(csv_file, 'w') as f:
        # Header
        f.write("time,success_rate,success_count,path_length_mean,path_length_std,")
        f.write("path_length_min,path_length_max,smoothness_mean,smoothness_std,")
        f.write("mean_jerk_mean,mean_jerk_std,tree_size_mean,tree_size_std,")
        f.write("num_samples_mean,num_batches_mean\n")

        # Data rows
        for i, t in enumerate(stats['time_points']):
            f.write(f"{t:.1f},")
            f.write(f"{stats['success_rate'][i]:.4f},")
            f.write(f"{stats['success_count'][i]:.0f},")
            f.write(f"{stats['path_length_mean'][i]:.6f},")
            f.write(f"{stats['path_length_std'][i]:.6f},")
            f.write(f"{stats['path_length_min'][i]:.6f},")
            f.write(f"{stats['path_length_max'][i]:.6f},")
            f.write(f"{stats['smoothness_mean'][i]:.6f},")
            f.write(f"{stats['smoothness_std'][i]:.6f},")
            f.write(f"{stats['mean_jerk_mean'][i]:.6f},")
            f.write(f"{stats['mean_jerk_std'][i]:.6f},")
            f.write(f"{stats['tree_size_mean'][i]:.6f},")
            f.write(f"{stats['tree_size_std'][i]:.6f},")
            f.write(f"{stats['num_samples_mean'][i]:.6f},")
            f.write(f"{stats['num_batches_mean'][i]:.6f}\n")

    print(f"Saved temporal statistics to: {csv_file}")


def main(results_dir, output_dir=None, time_step=1.0, max_time=None,
         mpd_target=None, aggregated_file=None):
    """
    Main analysis function.

    Args:
        results_dir: Directory containing bitstar_result_*.pt files
        output_dir: Output directory for plots and statistics
        time_step: Time step for binning (seconds)
        max_time: Maximum time to analyze
        mpd_target: Target path length from MPD (for comparison)
        aggregated_file: Path to complete_aggregated_results.pt (to extract MPD target)
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Set output directory
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("TEMPORAL CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Time step: {time_step}s")
    print("="*80)

    # Load BIT* results
    bitstar_results = load_bitstar_results(results_dir)

    if not bitstar_results:
        print("No BIT* results found!")
        return

    # Try to load MPD target from aggregated results
    if mpd_target is None and aggregated_file is not None:
        if os.path.exists(aggregated_file):
            print(f"\nLoading MPD target from: {aggregated_file}")
            aggregated = torch.load(aggregated_file, map_location='cpu')
            mpd_stats = aggregated.get('mpd_stats', {})
            if 'best_collision_free_path_length_mean' in mpd_stats:
                mpd_target = mpd_stats['best_collision_free_path_length_mean']
                print(f"MPD target path length: {mpd_target:.3f}\n")

    # Compute temporal statistics
    stats = compute_temporal_statistics(bitstar_results, time_step, max_time)

    # Print summary
    print_temporal_summary(stats)

    # Save statistics
    save_temporal_statistics(stats, output_dir)

    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        print("-" * 80)
        plot_convergence_overview(stats, output_dir, mpd_target)
        plot_path_length_detail(stats, output_dir, mpd_target)
        plot_smoothness_convergence(stats, output_dir)
        plot_sampling_efficiency(stats, output_dir)
        print("-" * 80)
        print(f"\nAll plots saved to: {output_dir}")
    else:
        print("\nSkipping plots (matplotlib not available)")

    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze second-by-second convergence of BIT* across problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with default settings
  python analyze_temporal_convergence.py

  # Analyze specific results directory
  python analyze_temporal_convergence.py --results-dir my_results

  # Use finer time resolution
  python analyze_temporal_convergence.py --time-step 0.5

  # Include MPD comparison target
  python analyze_temporal_convergence.py --mpd-target 5.823

  # Auto-load MPD target from aggregated results
  python analyze_temporal_convergence.py --aggregated multi_run_results/complete_aggregated_results.pt
        """
    )

    parser.add_argument("--results-dir", default="multi_run_results",
                       help="Directory containing bitstar_result_*.pt files (default: multi_run_results)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for plots and statistics (default: same as results-dir)")
    parser.add_argument("--time-step", type=float, default=1.0,
                       help="Time step for binning in seconds (default: 1.0)")
    parser.add_argument("--max-time", type=float, default=None,
                       help="Maximum time to analyze in seconds (default: auto-detect)")
    parser.add_argument("--mpd-target", type=float, default=None,
                       help="Target path length from MPD for comparison (default: None)")
    parser.add_argument("--aggregated", default="multi_run_results/complete_aggregated_results.pt",
                       help="Path to complete_aggregated_results.pt to auto-load MPD target")

    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        time_step=args.time_step,
        max_time=args.max_time,
        mpd_target=args.mpd_target,
        aggregated_file=args.aggregated,
    )
