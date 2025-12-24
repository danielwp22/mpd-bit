"""
Export existing comparison results to CSV files.

Use this script to export data from an existing run to CSV format
without having to re-run the entire comparison.

Usage:
    python export_results_to_csv.py --input-dir multi_run_results
    python export_results_to_csv.py --input-dir my_custom_results --output-name my_export
"""
import os
import sys
import csv
import argparse
import torch
import numpy as np


def export_to_csv(bitstar_results, mpd_results, comparison_data, output_dir, output_prefix=""):
    """
    Export all data to comprehensive CSV files.

    Creates four CSV files:
    1. bitstar_timeseries.csv: Second-by-second BIT* metrics for each problem
    2. mpd_results.csv: MPD metrics for each problem
    3. comparison_summary.csv: Comparison metrics for each problem
    4. mpd_all_samples.csv: All MPD samples with individual metrics
    """
    print(f"\n{'='*80}")
    print("EXPORTING DATA TO CSV FILES")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Add prefix to filenames if provided
    prefix = f"{output_prefix}_" if output_prefix else ""

    # 1. Export BIT* time-series data
    bitstar_csv_path = os.path.join(output_dir, f"{prefix}bitstar_timeseries.csv")
    print(f"Exporting BIT* time-series data to: {bitstar_csv_path}")

    with open(bitstar_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'time',
            'iteration',
            'path_length',
            'smoothness',
            'mean_jerk',
            'num_vertices',
            'num_samples',
            'num_batches',
            'has_solution',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        for result in bitstar_results:
            problem_idx = result['problem_idx']
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            # Write each interval metric as a row
            for metric in result.get('interval_metrics', []):
                row = {
                    'problem_idx': problem_idx,
                    'time': metric['time'],
                    'iteration': metric['iteration'],
                    'path_length': metric['path_length'],
                    'smoothness': metric['smoothness'],
                    'mean_jerk': metric['mean_jerk'],
                    'num_vertices': metric['num_vertices'],
                    'num_samples': metric['num_samples'],
                    'num_batches': metric['num_batches'],
                    'has_solution': metric['has_solution'],
                    'start_state': start_state_str,
                    'goal_state': goal_state_str,
                }
                writer.writerow(row)
                total_rows += 1

        print(f"  Wrote {total_rows} rows (time-series data points)")

    # 2. Export MPD results
    mpd_csv_path = os.path.join(output_dir, f"{prefix}mpd_results.csv")
    print(f"Exporting MPD results to: {mpd_csv_path}")

    with open(mpd_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'success',
            'inference_time',
            'n_samples',
            'n_diffusion_steps',
            'best_path_length',
            'best_smoothness',
            'n_collision_free',
            'n_total_samples',
            'collision_rate',
            'mean_collision_free_path_length',
            'best_collision_free_path_length',
            'path_length_mean',
            'path_length_std',
            'smoothness_mean',
            'smoothness_std',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in mpd_results:
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            row = {
                'problem_idx': result['problem_idx'],
                'success': result.get('success', False),
                'inference_time': result.get('inference_time', 0),
                'n_samples': result.get('n_samples', 0),
                'n_diffusion_steps': result.get('n_diffusion_steps', 0),
                'best_path_length': result.get('path_length', float('inf')),
                'best_smoothness': result.get('smoothness', float('inf')),
                'n_collision_free': result.get('n_collision_free', 0),
                'n_total_samples': result.get('n_total_samples', 0),
                'collision_rate': result.get('collision_rate', 1.0),
                'mean_collision_free_path_length': result.get('mean_collision_free_path_length', float('inf')),
                'best_collision_free_path_length': result.get('best_collision_free_path_length', float('inf')),
                'path_length_mean': result.get('path_length_mean', float('inf')),
                'path_length_std': result.get('path_length_std', 0),
                'smoothness_mean': result.get('smoothness_mean', float('inf')),
                'smoothness_std': result.get('smoothness_std', 0),
                'start_state': start_state_str,
                'goal_state': goal_state_str,
            }
            writer.writerow(row)

        print(f"  Wrote {len(mpd_results)} rows (one per problem)")

    # 3. Export comparison summary
    comparison_csv_path = os.path.join(output_dir, f"{prefix}comparison_summary.csv")
    print(f"Exporting comparison summary to: {comparison_csv_path}")

    with open(comparison_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'bitstar_success',
            'mpd_success',
            'mpd_target_length',
            'bitstar_final_length',
            'time_to_first_solution',
            'time_to_match_mpd',
            'bitstar_beats_mpd',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for comp in comparison_data:
            row = {
                'problem_idx': comp['problem_idx'],
                'bitstar_success': comp.get('bitstar_success', False),
                'mpd_success': comp.get('mpd_success', False),
                'mpd_target_length': comp.get('mpd_target_length', float('inf')),
                'bitstar_final_length': comp.get('bitstar_final_length', float('inf')),
                'time_to_first_solution': comp.get('time_to_first_solution', 'N/A'),
                'time_to_match_mpd': comp.get('time_to_match_mpd', 'N/A'),
                'bitstar_beats_mpd': comp.get('bitstar_beats_mpd', False),
            }
            writer.writerow(row)

        print(f"  Wrote {len(comparison_data)} rows (one per problem)")

    # 4. Export detailed MPD sample-level data (all 64 samples per problem)
    mpd_samples_csv_path = os.path.join(output_dir, f"{prefix}mpd_all_samples.csv")
    print(f"Exporting MPD all-samples data to: {mpd_samples_csv_path}")

    with open(mpd_samples_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'problem_idx',
            'sample_idx',
            'path_length',
            'smoothness',
            'is_collision_free',
            'start_state',
            'goal_state',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_sample_rows = 0
        for result in mpd_results:
            if not result.get('success', False):
                continue

            problem_idx = result['problem_idx']
            start_state_str = ','.join(map(str, result['start_state']))
            goal_state_str = ','.join(map(str, result['goal_state']))

            all_path_lengths = result.get('all_path_lengths', [])
            all_smoothness = result.get('all_smoothness', [])
            collision_free_mask = result.get('collision_free_mask', [])

            # Write each sample as a row
            for sample_idx in range(len(all_path_lengths)):
                row = {
                    'problem_idx': problem_idx,
                    'sample_idx': sample_idx,
                    'path_length': all_path_lengths[sample_idx],
                    'smoothness': all_smoothness[sample_idx],
                    'is_collision_free': collision_free_mask[sample_idx] if sample_idx < len(collision_free_mask) else False,
                    'start_state': start_state_str,
                    'goal_state': goal_state_str,
                }
                writer.writerow(row)
                total_sample_rows += 1

        print(f"  Wrote {total_sample_rows} rows (all samples from all problems)")

    print(f"\n{'='*80}")
    print("CSV EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"BIT* time-series: {bitstar_csv_path}")
    print(f"MPD per-problem:  {mpd_csv_path}")
    print(f"Comparison:       {comparison_csv_path}")
    print(f"MPD all samples:  {mpd_samples_csv_path}")
    print(f"{'='*80}\n")

    return {
        'bitstar_timeseries': bitstar_csv_path,
        'mpd_results': mpd_csv_path,
        'comparison_summary': comparison_csv_path,
        'mpd_all_samples': mpd_samples_csv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export existing comparison results to CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from default directory
  python export_results_to_csv.py

  # Export from custom directory
  python export_results_to_csv.py --input-dir my_results

  # Export with custom output name prefix
  python export_results_to_csv.py --input-dir my_results --output-name experiment1
        """
    )

    parser.add_argument(
        "--input-dir",
        default="multi_run_results",
        help="Directory containing the aggregated results file (default: multi_run_results)"
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional prefix for output CSV filenames (default: none)"
    )

    args = parser.parse_args()

    # Load aggregated results
    results_file = os.path.join(args.input_dir, "complete_aggregated_results.pt")

    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        print(f"\nMake sure you've run the comparison first using:")
        print(f"  python run_complete_comparison.py --output-dir {args.input_dir}")
        sys.exit(1)

    print(f"\nLoading results from: {results_file}")
    aggregated = torch.load(results_file)

    bitstar_results = aggregated['bitstar_results']
    mpd_results = aggregated['mpd_results']
    comparison_data = aggregated['comparison_data']

    print(f"Loaded data:")
    print(f"  BIT* results: {len(bitstar_results)} problems")
    print(f"  MPD results: {len(mpd_results)} problems")
    print(f"  Comparison data: {len(comparison_data)} problems")

    # Export to CSV
    csv_files = export_to_csv(
        bitstar_results,
        mpd_results,
        comparison_data,
        output_dir=args.input_dir,
        output_prefix=args.output_name
    )

    print("Done! You can now open the CSV files in Excel, pandas, or any spreadsheet software.")


if __name__ == "__main__":
    main()
