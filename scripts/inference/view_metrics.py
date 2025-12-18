"""
View metrics from diffusion model inference results.
"""
import torch
import os
from pprint import pprint


def view_metrics(results_file="logs/2/results_single_plan-000.pt"):
    """Display all available metrics from inference results."""

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return

    print(f"\nLoading: {results_file}")
    results = torch.load(results_file, map_location='cpu')

    print("\n" + "="*80)
    print("AVAILABLE DATA IN RESULTS FILE")
    print("="*80)
    print("\nTop-level keys:")
    for key in results.keys():
        print(f"  - {key}")

    print("\n" + "="*80)
    print("METRICS")
    print("="*80)

    metrics = results['metrics']

    # All trajectories metrics
    print("\n--- Metrics over ALL sampled trajectories ---")
    for key, value in metrics['trajs_all'].items():
        print(f"  {key:<45} {value}")

    # Best trajectory metrics
    print("\n--- Metrics for BEST trajectory ---")
    for key, value in metrics['trajs_best'].items():
        print(f"  {key:<45} {value}")

    # Valid trajectories metrics
    if 'trajs_valid' in metrics:
        print("\n--- Metrics over VALID (collision-free) trajectories ---")
        for key, value in metrics['trajs_valid'].items():
            print(f"  {key:<45} {value}")

    print("\n" + "="*80)
    print("TIMING")
    print("="*80)
    print(f"  Total inference time:          {results.get('t_inference_total', 0):.3f} sec")
    print(f"  Generator time:                {results.get('t_generator', 0):.3f} sec")
    print(f"  Guidance time:                 {results.get('t_guide', 0):.3f} sec")

    print("\n" + "="*80)
    print("START AND GOAL STATES")
    print("="*80)
    if 'q_pos_start' in results:
        print(f"  Start state: {results['q_pos_start']}")
    if 'q_pos_goal' in results:
        print(f"  Goal state:  {results['q_pos_goal']}")
    if 'ee_pose_goal' in results:
        print(f"  End-effector goal pose shape: {results['ee_pose_goal'].shape}")

    print("\n" + "="*80)
    print("ISAACGYM STATISTICS")
    print("="*80)
    if 'isaacgym_statistics' in results and results['isaacgym_statistics']:
        pprint(dict(results['isaacgym_statistics']))
    else:
        print("  Not available")

    print("\n" + "="*80)
    print("TRAJECTORY DATA")
    print("="*80)
    print(f"  Best trajectory shape:         {results.get('q_trajs_pos_best', torch.tensor([])).shape}")
    print(f"  Valid trajectories shape:      {results.get('q_trajs_pos_valid', torch.tensor([])).shape}")

    print("\n" + "="*80)
    print("KEY METRICS SUMMARY")
    print("="*80)
    print(f"  Success:                       {metrics['trajs_all']['success']}")
    print(f"  Fraction valid:                {metrics['trajs_all']['fraction_valid']*100:.1f}%")
    print(f"  Path length (best):            {metrics['trajs_best']['path_length']:.3f}")
    print(f"  Smoothness (best):             {metrics['trajs_best']['smoothness']:.3f}")
    print(f"  Collision intensity:           {metrics['trajs_all']['collision_intensity']:.4f}")
    if 'diversity' in metrics.get('trajs_valid', {}):
        print(f"  Trajectory diversity:          {metrics['trajs_valid']['diversity']:.3f}")

    print("="*80 + "\n")


def list_available_results(results_dir="logs/2"):
    """List all available result files in a directory."""

    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found")
        return []

    result_files = sorted([f for f in os.listdir(results_dir) if f.startswith('results_single_plan')])

    print(f"\nAvailable result files in {results_dir}:")
    for f in result_files:
        print(f"  - {f}")

    return result_files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View metrics from inference results")
    parser.add_argument("--results-file", default="logs/2/results_single_plan-000.pt",
                       help="Path to results file")
    parser.add_argument("--list-dir", default=None,
                       help="List all result files in a directory")

    args = parser.parse_args()

    if args.list_dir:
        list_available_results(args.list_dir)
    else:
        view_metrics(args.results_file)

    print("\nUsage examples:")
    print("  python view_metrics.py")
    print("  python view_metrics.py --results-file logs/2/results_single_plan-000.pt")
    print("  python view_metrics.py --list-dir logs/2")
