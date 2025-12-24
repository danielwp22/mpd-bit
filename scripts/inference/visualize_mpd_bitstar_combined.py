"""
Visualize MPD and BIT* trajectories for the same Panda Spheres3D problem.

Plots MPD best (and optional all valid) plus BIT* trajectory in one 3D figure.
Attempts to plot obstacles from a saved npz or by instantiating EnvSpheres3D.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.robots.robot_panda import RobotPanda


def extract_obstacles_from_env(device):
    """Instantiate EnvSpheres3D and extract sphere centers/radii if available."""
    try:
        env = EnvSpheres3D(precompute_sdf_obj_fixed=False, sdf_cell_size=0.05, tensor_args={"device": device, "dtype": torch.float32})
        centers_list, radii_list = [], []
        objs = getattr(env, "obj_all_list", []) or getattr(env, "obj_fixed_list", []) + getattr(env, "obj_extra_list", [])
        for obj in objs:
            for field in getattr(obj, "fields", []):
                centers = getattr(field, "centers", None)
                radii = getattr(field, "radii", None)
                if centers is not None and radii is not None:
                    centers_list.append(centers.detach().cpu().numpy())
                    radii_list.append(radii.detach().cpu().numpy())
        if centers_list and radii_list:
            return np.concatenate(centers_list, axis=0), np.concatenate(radii_list, axis=0).flatten()
    except Exception as e:
        print(f"Warning: could not extract obstacles from EnvSpheres3D: {e}")
    return None, None


def load_obstacles(obstacles_path, device):
    """Load obstacles from npz if provided, else try to build from env."""
    if obstacles_path and os.path.exists(obstacles_path):
        data = np.load(obstacles_path)
        centers = data.get("centers", None)
        radii = data.get("radii", None)
        if centers is not None and radii is not None:
            return np.asarray(centers), np.asarray(radii).flatten()
        print(f"Warning: obstacles file {obstacles_path} missing centers or radii; falling back to env.")
    return extract_obstacles_from_env(device)


def save_single_plot(ee_paths, colors, labels, start_point, goal_point, centers, radii, out_path, linewidths=None, alphas=None):
    """Save a single 3D plot with optional multiple paths and obstacles."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot in two passes so the primary path stays on top
    # Pass 1: all but the first path
    for idx, (ee, color, label) in enumerate(zip(ee_paths, colors, labels)):
        if idx == 0:
            continue
        lw = linewidths[idx] if linewidths is not None and idx < len(linewidths) else 2.5
        alpha = alphas[idx] if alphas is not None and idx < len(alphas) else 0.8
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, linewidth=lw, label=label, alpha=alpha, zorder=1)

    # Pass 2: primary path on top
    lw0 = linewidths[0] if linewidths is not None else 2.5
    alpha0 = alphas[0] if alphas is not None else 0.9
    ax.plot(ee_paths[0][:, 0], ee_paths[0][:, 1], ee_paths[0][:, 2],
            color=colors[0], linewidth=lw0, label=labels[0], alpha=alpha0, zorder=3)

    if start_point is not None:
        ax.scatter(start_point[0], start_point[1], start_point[2], color="green", s=50, label="start")
    if goal_point is not None:
        ax.scatter(goal_point[0], goal_point[1], goal_point[2], color="red", s=50, label="goal")

    if centers is not None and radii is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=(radii * 300) ** 2, color="gray", alpha=0.35, marker="o", label="obstacles")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.zaxis.labelpad = 18
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.12, top=0.95)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MPD + BIT* Panda EE paths with obstacles.")
    parser.add_argument(
        "--mpd-results",
        default="logs/2/results_single_plan-000.pt",
        help="Path to MPD results_single_plan-XXX.pt",
    )
    parser.add_argument(
        "--bitstar-trajectory",
        default="logs_bitstar_panda_spheres3d/trajectory_000.npy",
        help="Path to BIT* joint trajectory (.npy)",
    )
    parser.add_argument(
        "--bitstar-result",
        default="logs_bitstar_panda_spheres3d/bitstar_tracked_result.pt",
        help="Optional BIT* result file to read start/goal + solution path.",
    )
    parser.add_argument(
        "--out",
        default="minimal_results_50_merged/mpd_bitstar_problem_000_paths.png",
        help="Output image file",
    )
    parser.add_argument(
        "--obstacles",
        default="logs_bitstar_panda_spheres3d/obstacles.npz",
        help="Optional obstacles npz (centers, radii); if missing, will instantiate EnvSpheres3D.",
    )
    parser.add_argument(
        "--plot-all-mpd",
        action="store_true",
        help="Plot all valid MPD trajectories (not just best).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--mpd-out",
        default="minimal_results_50_merged/mpd_problem_000_paths.png",
        help="Output image for MPD-only visualization.",
    )
    parser.add_argument(
        "--bitstar-out",
        default="minimal_results_50_merged/bitstar_problem_000_path.png",
        help="Output image for BIT*-only visualization.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.mpd_results):
        raise FileNotFoundError(f"MPD results file not found: {args.mpd_results}")
    if not os.path.exists(args.bitstar_trajectory) and not os.path.exists(args.bitstar_result):
        raise FileNotFoundError(f"BIT* trajectory file not found: {args.bitstar_trajectory} and result file not found: {args.bitstar_result}")

    # Load MPD results
    res = torch.load(args.mpd_results, map_location=device)
    q_best = res.get("q_trajs_pos_best", None)
    q_valid = res.get("q_trajs_pos_valid", None)
    if q_best is None:
        raise ValueError("No best trajectory (q_trajs_pos_best) found in MPD results file.")
    mpd_start = res.get("q_pos_start", None)
    mpd_goal = res.get("q_pos_goal", None)

    # Load BIT* trajectory (prefer result file if it contains a solution path)
    traj_bs = None
    bs_start = bs_goal = None
    if args.bitstar_result and os.path.exists(args.bitstar_result):
        bs_res = torch.load(args.bitstar_result, map_location=device)
        bs_start = bs_res.get("start_state", None)
        bs_goal = bs_res.get("goal_state", None)
        sol = bs_res.get("sol_path", None)
        if sol is not None:
            traj_bs = np.array(sol, dtype=np.float32)
    if traj_bs is None:
        if not os.path.exists(args.bitstar_trajectory):
            raise FileNotFoundError(f"BIT* trajectory file not found: {args.bitstar_trajectory}")
        traj_bs = np.load(args.bitstar_trajectory)

    if traj_bs.ndim != 2 or traj_bs.shape[1] != 7:
        raise ValueError(f"Expected BIT* trajectory shape (N, 7); got {traj_bs.shape}")

    robot = RobotPanda(device=device)

    # MPD EE positions
    q_best_t = q_best.to(device)
    with torch.no_grad():
        ee_best = robot.get_EE_pose(q_best_t, flatten_pos_quat=True)
    ee_best_pos = ee_best[:, :3].detach().cpu().numpy()

    ee_valid_pos = None
    if args.plot_all_mpd and q_valid is not None and q_valid.numel() > 0:
        q_valid_t = q_valid.to(device)  # [B, T, 7]
        B, T, D = q_valid_t.shape
        q_flat = q_valid_t.reshape(-1, D)
        with torch.no_grad():
            ee_flat = robot.get_EE_pose(q_flat, flatten_pos_quat=True)
        ee_flat_pos = ee_flat[:, :3].detach().cpu().numpy()
        ee_valid_pos = ee_flat_pos.reshape(B, T, 3)

    # BIT* EE positions
    q_bs_t = torch.as_tensor(traj_bs, dtype=torch.float32, device=device)
    with torch.no_grad():
        ee_bs = robot.get_EE_pose(q_bs_t, flatten_pos_quat=True)
    ee_bs_pos = ee_bs[:, :3].detach().cpu().numpy()

    # Check start/goal alignment
    if mpd_start is not None and bs_start is not None:
        def to_np(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.array(x)

        mpd_start_np = to_np(mpd_start).reshape(-1)
        mpd_goal_np = to_np(mpd_goal).reshape(-1)
        bs_start_np = to_np(bs_start).reshape(-1)
        bs_goal_np = to_np(bs_goal).reshape(-1)
        tol = 1e-3
        if mpd_start_np.shape == bs_start_np.shape:
            if np.max(np.abs(mpd_start_np - bs_start_np)) > tol or np.max(np.abs(mpd_goal_np - bs_goal_np)) > tol:
                print("Warning: MPD and BIT* start/goal differ; trajectories may be from different problems.")
        else:
            print("Warning: cannot compare start/goal due to shape mismatch.")

    # Obstacles
    centers, radii = load_obstacles(args.obstacles, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all MPD valid trajectories
    if ee_valid_pos is not None:
        for traj in ee_valid_pos:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:gray", alpha=0.25, linewidth=1)

    # Plot MPD best
    ax.plot(ee_best_pos[:, 0], ee_best_pos[:, 1], ee_best_pos[:, 2],
            color="tab:blue", linewidth=2.5, label="MPD best")
    ax.scatter(ee_best_pos[0, 0], ee_best_pos[0, 1], ee_best_pos[0, 2],
               color="green", s=50, label="start")
    ax.scatter(ee_best_pos[-1, 0], ee_best_pos[-1, 1], ee_best_pos[-1, 2],
               color="red", s=50, label="goal")

    # Plot BIT* trajectory
    ax.plot(ee_bs_pos[:, 0], ee_bs_pos[:, 1], ee_bs_pos[:, 2],
            color="tab:purple", linewidth=2.5, label="BIT*")

    # Plot obstacles
    if centers is not None and radii is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=(radii * 300) ** 2, color="gray", alpha=0.35, marker="o", label="obstacles")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.zaxis.labelpad = 18
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.12, top=0.95)
    plt.savefig(args.out, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"Saved MPD + BIT* path visualization to {args.out}")

    # MPD-only plot
    mpd_paths = [ee_best_pos]
    if ee_valid_pos is not None:
        mpd_paths.extend([traj for traj in ee_valid_pos])
    mpd_colors = ["tab:blue"] + (["tab:gray"] * (len(mpd_paths) - 1))
    mpd_labels = ["MPD best"] + ([None] * (len(mpd_paths) - 1))  # unlabeled thin lines
    mpd_linewidths = [2.5] + ([1.0] * (len(mpd_paths) - 1))
    mpd_alphas = [0.9] + ([0.3] * (len(mpd_paths) - 1))

    save_single_plot(
        ee_paths=mpd_paths,
        colors=mpd_colors,
        labels=mpd_labels,
        linewidths=mpd_linewidths,
        alphas=mpd_alphas,
        start_point=ee_best_pos[0],
        goal_point=ee_best_pos[-1],
        centers=centers,
        radii=radii,
        out_path=args.mpd_out,
    )

    # BIT*-only plot
    save_single_plot(
        ee_paths=[ee_bs_pos],
        colors=["tab:purple"],
        labels=["BIT*"],
        start_point=ee_bs_pos[0],
        goal_point=ee_bs_pos[-1],
        centers=centers,
        radii=radii,
        out_path=args.bitstar_out,
    )


if __name__ == "__main__":
    main()
