"""
Visualize MPD Panda trajectories (best + all valid) in the Spheres3D environment.

Loads a saved MPD inference result (.pt), computes end-effector paths, and plots
obstacles as spheres when available.
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


def main():
    parser = argparse.ArgumentParser(description="Plot MPD Panda EE paths (best + all valid) with obstacles.")
    parser.add_argument(
        "--results",
        default="logs/2/results_single_plan-000.pt",
        help="Path to MPD results_single_plan-XXX.pt",
    )
    parser.add_argument(
        "--out",
        default="minimal_results_50_merged/mpd_problem_000_paths.png",
        help="Output image file",
    )
    parser.add_argument(
        "--obstacles",
        default=None,
        help="Optional obstacles npz (centers, radii); if not provided, env will be instantiated.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot all valid trajectories (not just best).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results file not found: {args.results}")

    res = torch.load(args.results, map_location=device)
    q_best = res.get("q_trajs_pos_best", None)
    q_valid = res.get("q_trajs_pos_valid", None)

    if q_best is None:
        raise ValueError("No best trajectory (q_trajs_pos_best) found in results file.")

    robot = RobotPanda(device=device)

    # Best path EE positions
    q_best_t = q_best.to(device)
    with torch.no_grad():
        ee_best = robot.get_EE_pose(q_best_t, flatten_pos_quat=True)
    ee_best_pos = ee_best[:, :3].detach().cpu().numpy()

    # All valid paths EE positions
    ee_valid_pos = None
    if args.plot_all and q_valid is not None and q_valid.numel() > 0:
        q_valid_t = q_valid.to(device)  # [B, T, 7]
        B, T, D = q_valid_t.shape
        q_flat = q_valid_t.reshape(-1, D)
        with torch.no_grad():
            ee_flat = robot.get_EE_pose(q_flat, flatten_pos_quat=True)
        ee_flat_pos = ee_flat[:, :3].detach().cpu().numpy()
        ee_valid_pos = ee_flat_pos.reshape(B, T, 3)

    # Obstacles
    centers, radii = load_obstacles(args.obstacles, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all valid trajectories (light)
    if ee_valid_pos is not None:
        for traj in ee_valid_pos:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="tab:gray", alpha=0.3, linewidth=1)

    # Plot best trajectory
    ax.plot(ee_best_pos[:, 0], ee_best_pos[:, 1], ee_best_pos[:, 2],
            color="tab:blue", linewidth=2.5, label="MPD best")
    ax.scatter(ee_best_pos[0, 0], ee_best_pos[0, 1], ee_best_pos[0, 2],
               color="green", s=50, label="start")
    ax.scatter(ee_best_pos[-1, 0], ee_best_pos[-1, 1], ee_best_pos[-1, 2],
               color="red", s=50, label="goal")

    # Plot obstacles
    if centers is not None and radii is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=(radii * 300) ** 2, color="gray", alpha=0.35, marker="o", label="obstacles")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.zaxis.labelpad = 18  # avoid clipping
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Looser margins to keep labels visible
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.12, top=0.95)
    plt.savefig(args.out, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"Saved MPD path visualization to {args.out}")


if __name__ == "__main__":
    main()
