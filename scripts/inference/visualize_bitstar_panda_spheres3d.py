"""
Quick visualization of a BIT* trajectory for Panda in the Spheres3D task.

Loads a saved trajectory (joint positions) and plots the end-effector path in 3D.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_robotics.robots.robot_panda import RobotPanda


def main():
    parser = argparse.ArgumentParser(description="Plot EE path for a BIT* Panda trajectory.")
    parser.add_argument(
        "--trajectory",
        default="logs_bitstar_panda_spheres3d/trajectory_000.npy",
        help="Path to BIT* joint trajectory (.npy)",
    )
    parser.add_argument(
        "--out",
        default="minimal_results_50_merged/bitstar_problem_000_path.png",
        help="Output image file",
    )
    parser.add_argument(
        "--obstacles",
        default="logs_bitstar_panda_spheres3d/obstacles.npz",
        help="Optional obstacles file (npz with centers, radii)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trajectory):
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")

    traj = np.load(args.trajectory)
    if traj.ndim != 2 or traj.shape[1] != 7:
        raise ValueError(f"Expected trajectory shape (N, 7); got {traj.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = RobotPanda(device=device)

    q = torch.as_tensor(traj, dtype=torch.float32, device=device)
    with torch.no_grad():
        ee_pose = robot.get_EE_pose(q, flatten_pos_quat=True)
    ee_pos = ee_pose[:, :3].detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], color="tab:blue", linewidth=2, label="EE path")
    ax.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], color="green", s=40, label="start")
    ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], color="red", s=40, label="goal")

    # Optional obstacles (spheres)
    if args.obstacles and os.path.exists(args.obstacles):
        data = np.load(args.obstacles)
        centers = data.get("centers", None)
        radii = data.get("radii", None)
        if centers is not None and radii is not None:
            centers = np.asarray(centers)
            radii = np.asarray(radii).flatten()
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                       s=(radii * 300) ** 2,  # scale for visibility
                       color="gray", alpha=0.4, marker="o", label="obstacles")
        else:
            print(f"Warning: obstacles file {args.obstacles} missing centers or radii")
    elif args.obstacles:
        print(f"Obstacles file not found: {args.obstacles}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.zaxis.labelpad = 18  # avoid clipping
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.14, right=0.94, bottom=0.12, top=0.95)
    plt.savefig(args.out, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"Saved end-effector path visualization to {args.out}")


if __name__ == "__main__":
    main()
