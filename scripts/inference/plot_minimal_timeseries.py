"""
Plot BIT* vs MPD aggregated metrics from the CSV outputs of run_minimal_comparison.py.

Produces two plots:
- average BIT* path length over time (excluding inf) with a constant line for the
  average MPD best collision-free path length on problems where both succeeded.
- average BIT* smoothness over time with a constant line for the average MPD
  best smoothness on the same common set.
"""
import argparse
import os
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_data(output_dir):
    def _require(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}. "
                f"Run run_minimal_comparison.py with --output-dir {output_dir} to generate it."
            )
        return path

    bitstar_ts = pd.read_csv(_require(os.path.join(output_dir, "bitstar_timeseries.csv")))
    bitstar_results = pd.read_csv(_require(os.path.join(output_dir, "bitstar_results.csv")))
    mpd_results = pd.read_csv(_require(os.path.join(output_dir, "mpd_results.csv")))
    return bitstar_ts, bitstar_results, mpd_results


def filter_common_problems(bitstar_results, mpd_results):
    """
    Select problems where both planners succeeded with finite, collision-free paths.
    """
    bs_ok = bitstar_results[
        (bitstar_results["success"] == True)
        & np.isfinite(bitstar_results["path_length"])
        & np.isfinite(bitstar_results["smoothness"])
    ]
    mpd_ok = mpd_results[
        (mpd_results["success"] == True)
        & (mpd_results["n_collision_free"] > 0)
        & np.isfinite(mpd_results["best_collision_free_path_length"])
        & np.isfinite(mpd_results["best_smoothness"])
    ]
    common_ids = set(bs_ok["problem_idx"].astype(int)).intersection(
        set(mpd_ok["problem_idx"].astype(int))
    )
    return bs_ok[bs_ok["problem_idx"].isin(common_ids)], mpd_ok[mpd_ok["problem_idx"].isin(common_ids)], mpd_ok


def aggregate_bitstar_timeseries(bitstar_ts, common_ids, time_bin):
    """
    Bin timeseries by integer seconds (or provided bin) and average path length and smoothness.
    Excludes infinite values.
    """
    ts = bitstar_ts[bitstar_ts["problem_idx"].isin(common_ids)].copy()
    if ts.empty:
        return pd.DataFrame(columns=["time_bin", "path_length_mean", "smoothness_mean"])

    # Bin by time
    ts["time_bin"] = (ts["time"] / time_bin).apply(math.floor) * time_bin

    # Exclude non-finite metrics
    ts = ts[np.isfinite(ts["path_length"]) & np.isfinite(ts["smoothness"])]

    grouped = ts.groupby("time_bin").agg(
        path_length_mean=("path_length", "mean"),
        smoothness_mean=("smoothness", "mean"),
        count=("path_length", "count"),
    ).reset_index()
    return grouped


def aggregate_bitstar_best_so_far(bitstar_ts, common_ids, time_bin):
    """
    Compute average best-so-far path length per time bin across problems (monotonic non-increasing).
    """
    ts = bitstar_ts[bitstar_ts["problem_idx"].isin(common_ids)].copy()
    if ts.empty:
        return pd.DataFrame(columns=["time_bin", "path_length_best_mean"])

    ts["time_bin"] = (ts["time"] / time_bin).apply(math.floor) * time_bin
    ts = ts[np.isfinite(ts["path_length"])]

    all_bins = sorted(ts["time_bin"].unique())
    best_records = []
    for pid, df_pid in ts.groupby("problem_idx"):
        # Collapse duplicate bins per problem by keeping the best value in the bin
        df_pid = df_pid.groupby("time_bin", as_index=False)["path_length"].min().sort_values("time_bin")
        df_pid["best_so_far"] = df_pid["path_length"].cummin()
        # reindex to all bins and forward-fill to keep monotonic, constant denominator
        df_pid = df_pid[["time_bin", "best_so_far"]].set_index("time_bin").reindex(all_bins)
        df_pid = df_pid.ffill()
        df_pid["problem_idx"] = pid
        best_records.append(df_pid.reset_index().rename(columns={"index": "time_bin"}))

    if not best_records:
        return pd.DataFrame(columns=["time_bin", "path_length_best_mean"])

    all_best = pd.concat(best_records, ignore_index=True)
    grouped = all_best.groupby("time_bin").agg(path_length_best_mean=("best_so_far", "mean")).reset_index()
    return grouped


def aggregate_bitstar_success_over_time(bitstar_results, time_bin, max_time=None):
    """
    Compute cumulative success rate over time using time_to_first_solution from results.
    """
    total_problems = len(bitstar_results)
    if total_problems == 0:
        return pd.DataFrame(columns=["time_bin", "success_rate"])

    successes = bitstar_results[bitstar_results["success"] == True]
    times = successes["time_to_first_solution"]
    times = times[np.isfinite(times)]
    if times.empty:
        return pd.DataFrame(columns=["time_bin", "success_rate"])

    if max_time is None or not np.isfinite(max_time):
        max_time = times.max()
    if not np.isfinite(max_time) or max_time <= 0:
        max_time = time_bin

    bins = np.arange(0, max_time + time_bin, time_bin)
    records = []
    for b in bins:
        solved_count = (times <= b).sum()
        rate = solved_count / total_problems
        records.append({"time_bin": b, "success_rate": rate})

    return pd.DataFrame(records)


def aggregate_bitstar_best_so_far_smoothness(bitstar_ts, common_ids, time_bin):
    """
    Compute average best-so-far smoothness per time bin across problems (monotonic non-increasing).
    """
    ts = bitstar_ts[bitstar_ts["problem_idx"].isin(common_ids)].copy()
    if ts.empty:
        return pd.DataFrame(columns=["time_bin", "smoothness_best_mean"])

    ts["time_bin"] = (ts["time"] / time_bin).apply(math.floor) * time_bin
    ts = ts[np.isfinite(ts["smoothness"])]

    all_bins = sorted(ts["time_bin"].unique())
    best_records = []
    for pid, df_pid in ts.groupby("problem_idx"):
        df_pid = df_pid.groupby("time_bin", as_index=False)["smoothness"].min().sort_values("time_bin")
        df_pid["best_so_far"] = df_pid["smoothness"].cummin()
        df_pid = df_pid[["time_bin", "best_so_far"]].set_index("time_bin").reindex(all_bins)
        df_pid = df_pid.ffill()
        df_pid["problem_idx"] = pid
        best_records.append(df_pid.reset_index().rename(columns={"index": "time_bin"}))

    if not best_records:
        return pd.DataFrame(columns=["time_bin", "smoothness_best_mean"])

    all_best = pd.concat(best_records, ignore_index=True)
    grouped = all_best.groupby("time_bin").agg(smoothness_best_mean=("best_so_far", "mean")).reset_index()
    return grouped


def plot_bitstar_only(time_series_df, metric_key, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    if not time_series_df.empty:
        plt.plot(time_series_df["time_bin"], time_series_df[metric_key],
                 label="BIT* (avg over successful problems)", color="tab:blue", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_bitstar_vs_mpd(time_series_df, metric_key, mpd_value, ylabel, out_path, monotonic=False):
    plt.figure(figsize=(8, 5))
    series = time_series_df.copy()
    if not series.empty:
        plt.plot(series["time_bin"], series[metric_key],
                 label="BIT* (avg over successful problems)", color="tab:blue", linewidth=2)
    plt.axhline(mpd_value, color="tab:orange", linestyle="--", linewidth=2, label="MPD (avg, common problems)")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_success_vs_mpd(success_df, mpd_rate, out_path):
    plt.figure(figsize=(8, 5))
    if not success_df.empty:
        plt.plot(success_df["time_bin"], success_df["success_rate"], color="tab:green", linewidth=2, label="BIT* success rate")
    plt.axhline(mpd_rate, color="tab:orange", linestyle="--", linewidth=2, label="MPD success rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Success Rate")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot BIT* vs MPD time-series metrics.")
    parser.add_argument("--output-dir", default="minimal_results", help="Directory containing CSVs from run_minimal_comparison.py")
    parser.add_argument("--time-bin", type=float, default=1.0, help="Time bin size in seconds for averaging BIT* series")
    args = parser.parse_args()

    try:
        bitstar_ts, bitstar_results, mpd_results = load_data(args.output_dir)
    except FileNotFoundError as e:
        print(e)
        return
    bs_common, mpd_common, mpd_ok_all = filter_common_problems(bitstar_results, mpd_results)

    common_ids = set(bs_common["problem_idx"].astype(int))
    if not common_ids:
        print("No common successful problems with finite paths between BIT* and MPD.")
        return

    ts_agg = aggregate_bitstar_timeseries(bitstar_ts, common_ids, args.time_bin)
    ts_best = aggregate_bitstar_best_so_far(bitstar_ts, common_ids, args.time_bin)
    ts_best_smooth = aggregate_bitstar_best_so_far_smoothness(bitstar_ts, common_ids, args.time_bin)
    # BIT* success over all attempted problems (cumulative)
    max_time = bitstar_ts["time"].max() if not bitstar_ts.empty else bitstar_results["time_to_first_solution"].max()
    success_agg = aggregate_bitstar_success_over_time(bitstar_results, time_bin=args.time_bin, max_time=max_time)

    mpd_path_mean = mpd_common["best_collision_free_path_length"].mean()
    mpd_smooth_mean = mpd_common["best_smoothness"].mean()
    mpd_cf_mask = (mpd_results["n_collision_free"] > 0) & np.isfinite(mpd_results["best_collision_free_path_length"])
    mpd_cf_rate = mpd_cf_mask.mean()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_bitstar_vs_mpd(
        ts_agg,
        metric_key="path_length_mean",
        mpd_value=mpd_path_mean,
        ylabel="Path Length",
        out_path=os.path.join(args.output_dir, "bitstar_vs_mpd_path_length.png"),
    )

    # Smoothness: plot BIT* only (no MPD line)
    if not ts_best_smooth.empty:
        plot_bitstar_only(
            ts_best_smooth,
            metric_key="smoothness_best_mean",
            ylabel="Smoothness",
            out_path=os.path.join(args.output_dir, "bitstar_smoothness.png"),
        )

    # Best-so-far path length curve (monotonic)
    if not ts_best.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(ts_best["time_bin"], ts_best["path_length_best_mean"],
                 label="BIT* best-so-far (avg)", color="tab:purple", linewidth=2)
        plt.axhline(mpd_path_mean, color="tab:orange", linestyle="--", linewidth=2, label="MPD (avg, common problems)")
        plt.xlabel("Time (s)")
        plt.ylabel("Path Length")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_best = os.path.join(args.output_dir, "bitstar_best_so_far_path_length.png")
        plt.savefig(out_best)
        plt.close()
        print(f"Saved plot: {out_best}")

    # Success rate over time
    if not success_agg.empty:
        out_sr = os.path.join(args.output_dir, "bitstar_vs_mpd_success_rate.png")
        plot_success_vs_mpd(success_agg, mpd_cf_rate, out_sr)

    # Box plot: BIT* initial vs BIT* final vs MPD best, on common successful problems
    # Approximate BIT* initial solution as the first finite interval metric per problem.
    first_finite = []
    first_finite_smooth = []
    for pid in common_ids:
        ts_pid = bitstar_ts[
            (bitstar_ts["problem_idx"] == pid)
            & np.isfinite(bitstar_ts["path_length"])
            & np.isfinite(bitstar_ts["smoothness"])
        ]
        if not ts_pid.empty:
            ts_sorted = ts_pid.sort_values("time")
            first_finite.append(ts_sorted.iloc[0]["path_length"])
            first_finite_smooth.append(ts_sorted.iloc[0]["smoothness"])

    common_list = list(common_ids)
    bitstar_final_df = bs_common.set_index("problem_idx").loc[common_list]
    bitstar_final = bitstar_final_df["path_length"].tolist()
    bitstar_final_smooth = bitstar_final_df["smoothness"].tolist()
    mpd_best = mpd_common.set_index("problem_idx").loc[common_list]["best_collision_free_path_length"].tolist()

    data = {
        "BIT* initial": first_finite,
        "BIT* final": bitstar_final,
        "MPD best": mpd_best,
    }
    df_box = (
        pd.DataFrame(
            [(label, v) for label, vals in data.items() for v in vals if np.isfinite(v)],
            columns=["method", "path_length"],
        )
    )
    if not df_box.empty:
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            data=df_box,
            x="method",
            y="path_length",
            medianprops={"color": "black", "linewidth": 2.5},
        )
        sns.stripplot(
            data=df_box,
            x="method",
            y="path_length",
            color="0.3",
            size=3,
            alpha=0.6,
            marker="o",
            dodge=False,
        )
        plt.ylabel("Path Length")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        out_box = os.path.join(args.output_dir, "bitstar_mpd_path_length_boxplot.png")
        plt.savefig(out_box)
        plt.close()
        print(f"Saved plot: {out_box}")
    else:
        print("No finite path lengths available for box plot on common problems.")

    # Summary text file
    summary_lines = []
    # Success rates
    bs_success_rate = bitstar_results["success"].mean()
    mpd_cf_mask = (mpd_results["n_collision_free"] > 0) & np.isfinite(mpd_results["best_collision_free_path_length"])
    mpd_cf_rate = mpd_cf_mask.mean()
    summary_lines.append(f"BIT* success rate: {bs_success_rate:.3f}")
    summary_lines.append(f"MPD collision-free success rate: {mpd_cf_rate:.3f}")
    if not success_agg.empty:
        crossover = success_agg[success_agg["success_rate"] >= mpd_cf_rate]
        if not crossover.empty:
            first_cross = crossover.iloc[0]["time_bin"]
            summary_lines.append(f"BIT* success rate crosses MPD at t={first_cross:.1f}s")
        else:
            summary_lines.append("BIT* success rate never crosses MPD within recorded time")

    # Time to first solution (BIT*) and inference time (MPD)
    bs_times = bitstar_results.loc[np.isfinite(bitstar_results["time_to_first_solution"]), "time_to_first_solution"]
    if not bs_times.empty:
        summary_lines.append(f"BIT* time to first solution: mean={bs_times.mean():.3f}s, std={bs_times.std():.3f}s")
    else:
        summary_lines.append("BIT* time to first solution: no data")
    mpd_times = mpd_results.loc[mpd_results["success"] == True, "inference_time"]
    if not mpd_times.empty:
        summary_lines.append(f"MPD inference time: mean={mpd_times.mean():.3f}s, std={mpd_times.std():.3f}s")
    else:
        summary_lines.append("MPD inference time: no data")

    # MPD smoothness stats (best smoothness)
    mpd_smooth = mpd_results.loc[mpd_cf_mask, "best_smoothness"]
    if not mpd_smooth.empty:
        summary_lines.append(f"MPD best smoothness: mean={mpd_smooth.mean():.3f}, std={mpd_smooth.std():.3f}")
    else:
        summary_lines.append("MPD best smoothness: no data")

    # BIT* smoothness stats: initial and final (common successes)
    if first_finite_smooth:
        summary_lines.append(f"BIT* initial smoothness: mean={np.mean(first_finite_smooth):.3f}, std={np.std(first_finite_smooth):.3f}")
    else:
        summary_lines.append("BIT* initial smoothness: no data")
    if bitstar_final_smooth:
        summary_lines.append(f"BIT* final smoothness: mean={np.mean(bitstar_final_smooth):.3f}, std={np.std(bitstar_final_smooth):.3f}")
    else:
        summary_lines.append("BIT* final smoothness: no data")

    # Percentage where BIT* beats MPD on common successes
    if not bs_common.empty and not mpd_common.empty:
        comp = pd.read_csv(os.path.join(args.output_dir, "comparison_summary.csv"))
        comp = comp[comp["bitstar_success"] == True]
        comp = comp[comp["mpd_success"] == True]
        beats = comp["bitstar_beats_mpd"].mean() if not comp.empty else 0.0
        summary_lines.append(f"BIT* beats/matches MPD on common problems: {beats*100:.1f}%")
    else:
        summary_lines.append("BIT* beats/matches MPD on common problems: no common successes")

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
