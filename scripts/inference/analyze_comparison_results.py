"""
Helper script to analyze and visualize results from run_complete_comparison.py

This script loads the aggregated results and generates useful plots and statistics.
"""
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def print_statistics(results):
    """Print comprehensive statistics from results."""

    bitstar_stats = results.get('bitstar_stats', {})
    mpd_stats = results.get('mpd_stats', {})
    comparison_stats = results.get('comparison_stats', {})

    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS")
    print("="*80 + "\n")

    # Overall success rates
    print("Success Rates:")
    print("-" * 80)
    if bitstar_stats:
        print(f"  BIT*: {bitstar_stats['n_success']}/{bitstar_stats['n_problems']} ({bitstar_stats['success_rate']*100:.1f}%)")
    if mpd_stats:
        print(f"  MPD:  {mpd_stats['n_success']}/{mpd_stats['n_problems']} ({mpd_stats['success_rate']*100:.1f}%)")
    if comparison_stats:
        print(f"  Both succeeded: {comparison_stats['n_both_success']}")

    # Path quality
    print("\nPath Quality:")
    print("-" * 80)
    if mpd_stats:
        print(f"  MPD best collision-free: {mpd_stats['best_collision_free_path_length_mean']:.3f} ± {mpd_stats['best_collision_free_path_length_std']:.3f}")
    if bitstar_stats:
        print(f"  BIT* final:              {bitstar_stats['path_length_mean']:.3f} ± {bitstar_stats['path_length_std']:.3f}")
    if comparison_stats:
        print(f"  BIT* beats/matches MPD:  {comparison_stats['n_bitstar_beats_mpd']}/{comparison_stats['n_both_success']} ({comparison_stats['bitstar_beats_mpd_rate']*100:.1f}%)")

    # Planning time
    print("\nPlanning Time:")
    print("-" * 80)
    if mpd_stats:
        print(f"  MPD inference:           {mpd_stats['inference_time_mean']:.3f} ± {mpd_stats['inference_time_std']:.3f} sec")
    if bitstar_stats:
        print(f"  BIT* total:              {bitstar_stats['planning_time_mean']:.3f} ± {bitstar_stats['planning_time_std']:.3f} sec")
        if 'time_to_first_solution_mean' in bitstar_stats:
            print(f"  BIT* to first solution:  {bitstar_stats['time_to_first_solution_mean']:.3f} ± {bitstar_stats['time_to_first_solution_std']:.3f} sec")
    if comparison_stats and 'time_to_match_mpd_mean' in comparison_stats:
        print(f"  BIT* to match MPD:       {comparison_stats['time_to_match_mpd_mean']:.3f} ± {comparison_stats['time_to_match_mpd_std']:.3f} sec")

    # Smoothness
    print("\nSmoothness:")
    print("-" * 80)
    if mpd_stats:
        print(f"  MPD:   {mpd_stats['smoothness_mean']:.4f} ± {mpd_stats['smoothness_std']:.4f}")
    if bitstar_stats:
        print(f"  BIT*:  {bitstar_stats['smoothness_mean']:.4f} ± {bitstar_stats['smoothness_std']:.4f}")

    # MPD collision statistics
    if mpd_stats:
        print("\nMPD Collision Statistics:")
        print("-" * 80)
        print(f"  Collision rate:          {mpd_stats['collision_rate_mean']*100:.1f}% ± {mpd_stats['collision_rate_std']*100:.1f}%")
        print(f"  Mean CF path length:     {mpd_stats['mean_collision_free_path_length_mean']:.3f} ± {mpd_stats['mean_collision_free_path_length_std']:.3f}")

    # BIT* specific
    if bitstar_stats:
        print("\nBIT* Specific Metrics:")
        print("-" * 80)
        print(f"  Mean jerk:               {bitstar_stats['mean_jerk_mean']:.4f} ± {bitstar_stats['mean_jerk_std']:.4f}")
        if 'tree_size_mean' in bitstar_stats:
            print(f"  Final tree size:         {bitstar_stats['tree_size_mean']:.0f} ± {bitstar_stats['tree_size_std']:.0f} vertices")

    print("="*80 + "\n")


def plot_path_length_comparison(results, output_dir):
    """Plot path length comparison for all problems."""
    if not HAS_MATPLOTLIB:
        return

    comparison_data = results.get('comparison_data', [])

    # Extract data
    problem_indices = []
    bitstar_lengths = []
    mpd_lengths = []

    for c in comparison_data:
        if c.get('bitstar_success') and c.get('mpd_success'):
            problem_indices.append(c['problem_idx'])
            bitstar_lengths.append(c['bitstar_final_length'])
            mpd_lengths.append(c['mpd_target_length'])

    if not problem_indices:
        print("No successful problems to plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(problem_indices))
    width = 0.35

    bars1 = ax.bar(x - width/2, mpd_lengths, width, label='MPD', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x + width/2, bitstar_lengths, width, label='BIT*', alpha=0.8, color='#3498db')

    ax.set_xlabel('Problem Index', fontsize=12)
    ax.set_ylabel('Path Length', fontsize=12)
    ax.set_title('Path Length Comparison: BIT* vs MPD', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_indices)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'path_length_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_bitstar_optimization_examples(results, output_dir, n_examples=3):
    """Plot BIT* optimization over time for example problems."""
    if not HAS_MATPLOTLIB:
        return

    bitstar_results = results.get('bitstar_results', [])
    mpd_results = results.get('mpd_results', [])

    # Find successful problems
    successful_problems = []
    for i, br in enumerate(bitstar_results):
        if br.get('success') and i < len(mpd_results) and mpd_results[i].get('success'):
            successful_problems.append(i)

    if not successful_problems:
        print("No successful problems to plot optimization")
        return

    # Plot first n_examples
    n_to_plot = min(n_examples, len(successful_problems))

    fig, axes = plt.subplots(n_to_plot, 1, figsize=(12, 5*n_to_plot))
    if n_to_plot == 1:
        axes = [axes]

    for idx, prob_idx in enumerate(successful_problems[:n_to_plot]):
        br = bitstar_results[prob_idx]
        mr = mpd_results[prob_idx]

        # Extract BIT* optimization trajectory
        times = [m['time'] for m in br['interval_metrics'] if m.get('has_solution')]
        lengths = [m['path_length'] for m in br['interval_metrics'] if m.get('has_solution')]

        if not times:
            continue

        ax = axes[idx]

        # Plot BIT* optimization
        ax.plot(times, lengths, 'b-', linewidth=2, label='BIT* Anytime Optimization')

        # Add MPD target line
        mpd_target = mr.get('best_collision_free_path_length', mr['path_length'])
        ax.axhline(mpd_target, color='r', linestyle='--', linewidth=2, label=f'MPD Target ({mpd_target:.3f})')

        # Mark time to first solution
        if br.get('time_to_first_solution'):
            ax.axvline(br['time_to_first_solution'], color='g', linestyle=':', alpha=0.7, linewidth=1.5,
                      label=f'First Solution ({br["time_to_first_solution"]:.1f}s)')

        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Path Length', fontsize=11)
        ax.set_title(f'Problem {prob_idx}: BIT* Optimization Progress', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'bitstar_optimization_examples.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_mpd_collision_analysis(results, output_dir):
    """Plot MPD collision rate analysis."""
    if not HAS_MATPLOTLIB:
        return

    mpd_results = results.get('mpd_results', [])
    mpd_success = [r for r in mpd_results if r.get('success')]

    if not mpd_success:
        print("No successful MPD results to plot")
        return

    collision_rates = [r['collision_rate'] * 100 for r in mpd_success]
    mean_cf_lengths = [r['mean_collision_free_path_length'] for r in mpd_success
                       if r['mean_collision_free_path_length'] != float('inf')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collision rate distribution
    axes[0].hist(collision_rates, bins=10, edgecolor='black', color='#e74c3c', alpha=0.7)
    axes[0].axvline(np.mean(collision_rates), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(collision_rates):.1f}%')
    axes[0].set_xlabel('Collision Rate (%)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('MPD Collision Rate Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Path length distribution
    if mean_cf_lengths:
        axes[1].hist(mean_cf_lengths, bins=10, edgecolor='black', color='#2ecc71', alpha=0.7)
        axes[1].axvline(np.mean(mean_cf_lengths), color='darkgreen', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(mean_cf_lengths):.3f}')
        axes[1].set_xlabel('Mean Collision-Free Path Length', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('MPD Path Length Distribution', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'mpd_collision_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_time_analysis(results, output_dir):
    """Plot time-related analysis."""
    if not HAS_MATPLOTLIB:
        return

    comparison_data = results.get('comparison_data', [])

    # Extract timing data
    first_solution_times = []
    match_mpd_times = []

    for c in comparison_data:
        if c.get('bitstar_success'):
            if c.get('time_to_first_solution'):
                first_solution_times.append(c['time_to_first_solution'])

            if isinstance(c.get('time_to_match_mpd'), (int, float)):
                match_mpd_times.append(c['time_to_match_mpd'])

    if not first_solution_times:
        print("No timing data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time to first solution
    axes[0].hist(first_solution_times, bins=10, edgecolor='black', color='#3498db', alpha=0.7)
    axes[0].axvline(np.mean(first_solution_times), color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(first_solution_times):.1f}s')
    axes[0].set_xlabel('Time to First Solution (seconds)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('BIT* Time to First Solution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Time to match MPD
    if match_mpd_times:
        axes[1].hist(match_mpd_times, bins=10, edgecolor='black', color='#9b59b6', alpha=0.7)
        axes[1].axvline(np.mean(match_mpd_times), color='purple', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(match_mpd_times):.1f}s')
        axes[1].set_xlabel('Time to Match/Beat MPD (seconds)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('BIT* Time to Match MPD Quality', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1].text(0.5, 0.5, 'BIT* did not match MPD\non any problem',
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title('BIT* Time to Match MPD Quality', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'time_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main(results_file, output_dir=None):
    """Main analysis function."""

    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        return

    # Load results
    print(f"Loading results from: {results_file}")
    results = torch.load(results_file, map_location='cpu')

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Print statistics
    print_statistics(results)

    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        print("-" * 80)
        plot_path_length_comparison(results, output_dir)
        plot_bitstar_optimization_examples(results, output_dir, n_examples=3)
        plot_mpd_collision_analysis(results, output_dir)
        plot_time_analysis(results, output_dir)
        print("-" * 80)
        print("\nAll plots saved to:", output_dir)
    else:
        print("\nSkipping plots (matplotlib not available)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze results from run_complete_comparison.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results with default output
  python analyze_comparison_results.py

  # Analyze specific results file
  python analyze_comparison_results.py --results my_results/complete_aggregated_results.pt

  # Save plots to custom directory
  python analyze_comparison_results.py --output-dir my_plots
        """
    )

    parser.add_argument("--results", default="multi_run_results/complete_aggregated_results.pt",
                       help="Path to aggregated results file (default: multi_run_results/complete_aggregated_results.pt)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for plots (default: same as results file)")

    args = parser.parse_args()

    main(args.results, args.output_dir)
