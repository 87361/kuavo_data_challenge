#!/usr/bin/env python3
# Usage:
#   python scripts/gui/lerobot_editor/trajectory_visualizer.py
#   python scripts/gui/lerobot_editor/trajectory_visualizer.py --episode 12 --hand right --output /tmp/episode_12_right.png
#   python scripts/gui/lerobot_editor/trajectory_visualizer.py --dataset /path/to/lerobot --urdf /path/to/biped_s45.urdf --source state
#   python scripts/gui/lerobot_editor/trajectory_visualizer.py --source action --hand both --show
#
# This script loads one LeRobot v2.1 episode, computes Kuavo left/right
# end-effector XYZ positions with the lightweight URDF FK reader, and renders a
# publication-style 3D trajectory plot. The default dataset/URDF paths match the
# local Kuavo challenge data and assets discovered during repo setup.

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

try:
    from .dataset import LeRobotV21Dataset, load_v21_dataset
    from .urdf_fk import SimpleArmFk
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset import LeRobotV21Dataset, load_v21_dataset
    from urdf_fk import SimpleArmFk


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = Path("/mnt/data/kuavo_tianchi/lerobot_v21_task1_full/task1_zhuomian/lerobot")
DEFAULT_URDF = REPO_ROOT / "third_party/kuavo-ros-opensource-kdc/src/kuavo_assets/models/biped_s45/urdf/biped_s45.urdf"
DEFAULT_OUTPUT = Path("/tmp/kuavo_eef_trajectory_episode_000000.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one Kuavo LeRobot episode as a 3D EEF trajectory.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="LeRobot v2.1 dataset root")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Kuavo URDF used for FK")
    parser.add_argument("--episode", type=int, default=0, help="episode index to plot")
    parser.add_argument(
        "--source",
        choices=["state", "action"],
        default="state",
        help="column to visualize: observation.state or action",
    )
    parser.add_argument(
        "--hand",
        choices=["auto", "left", "right", "both"],
        default="auto",
        help="which end-effector trajectory to plot; auto selects the longer path",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output PNG path")
    parser.add_argument("--title", default=None, help="figure title; defaults to dataset/episode summary")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI")
    parser.add_argument("--show", action="store_true", help="open an interactive matplotlib window")
    return parser.parse_args()


def load_episode_positions(
    dataset: LeRobotV21Dataset,
    episode_index: int,
    fk: SimpleArmFk,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise ValueError(f"episode {episode_index} is out of range; dataset has {len(dataset.episodes)} episodes")

    column = "observation.state" if source == "state" else "action"
    df = dataset.read_episode_dataframe(episode_index)
    if column not in df:
        raise ValueError(f"episode dataframe does not contain {column!r}")

    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    for value in df[column].tolist():
        left, right = fk.state_positions(np.asarray(value, dtype=np.float64))
        left_points.append(left)
        right_points.append(right)
    return np.asarray(left_points), np.asarray(right_points)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def selected_hands(hand: str, left: np.ndarray, right: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    colors = {
        "left": "viridis",
        "right": "turbo",
    }
    if hand == "left":
        return [("left", left, colors["left"])]
    if hand == "right":
        return [("right", right, colors["right"])]
    if hand == "both":
        return [("left", left, colors["left"]), ("right", right, colors["right"])]
    return [("left", left, colors["left"])] if path_length(left) >= path_length(right) else [("right", right, colors["right"])]


def add_colored_path(ax: matplotlib.axes.Axes, points: np.ndarray, cmap_name: str, label: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if len(points) < 2:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=28, label=label)
        return

    values = np.linspace(0.0, 1.0, len(points))
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = Line3DCollection(segments, cmap=cmap_name, linewidth=2.4, alpha=0.96)
    collection.set_array(values[:-1])
    ax.add_collection3d(collection)

    start_color = matplotlib.colormaps[cmap_name](0.05)
    end_color = matplotlib.colormaps[cmap_name](0.95)
    ax.scatter(*points[0], s=42, color=start_color, edgecolor="black", linewidth=0.4, label=f"{label} start")
    ax.scatter(*points[-1], s=52, marker="X", color=end_color, edgecolor="black", linewidth=0.4, label=f"{label} end")
    add_direction_arrows(ax, points, end_color)


def add_direction_arrows(ax: matplotlib.axes.Axes, points: np.ndarray, color: tuple[float, float, float, float]) -> None:
    if len(points) < 8:
        return
    diffs = np.diff(points, axis=0)
    magnitudes = np.linalg.norm(diffs, axis=1)
    useful = np.flatnonzero(magnitudes > max(float(np.nanmax(magnitudes)) * 0.05, 1e-5))
    if len(useful) == 0:
        return
    chosen = useful[np.linspace(0, len(useful) - 1, min(7, len(useful))).astype(int)]
    scale = max(float(np.nanmax(magnitudes)) * 2.2, 0.015)
    for idx in chosen:
        direction = diffs[idx]
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            continue
        vec = direction / norm * scale
        ax.quiver(
            points[idx, 0],
            points[idx, 1],
            points[idx, 2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            linewidth=0.9,
            arrow_length_ratio=0.42,
            alpha=0.78,
        )


def set_axes_equal(ax: matplotlib.axes.Axes, point_groups: list[np.ndarray]) -> None:
    points = np.concatenate([group for group in point_groups if len(group)], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.05)
    pad = radius * 0.18
    radius += pad
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_trajectory(
    left: np.ndarray,
    right: np.ndarray,
    hand: str,
    title: str,
    output: Path,
    dpi: int,
    show: bool,
) -> None:
    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    selected = selected_hands(hand, left, right)
    fig = plt.figure(figsize=(8.2, 6.0), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    for label, points, cmap in selected:
        add_colored_path(ax, points, cmap, label)

    set_axes_equal(ax, [points for _, points, _ in selected])
    ax.view_init(elev=23, azim=-50)
    try:
        ax.set_box_aspect((1, 1, 0.82))
    except Exception:
        pass

    ax.set_xlabel(r"$x_e(m)$", labelpad=8)
    ax.set_ylabel(r"$y_e(m)$", labelpad=8)
    ax.set_zlabel(r"$z_e(m)$", labelpad=8)
    ax.grid(True, alpha=0.42)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="z", which="major", pad=-2, labelsize=8)
    for tick_label in [*ax.xaxis.get_ticklabels(), *ax.yaxis.get_ticklabels(), *ax.zaxis.get_ticklabels()]:
        tick_label.set_clip_on(False)
    ax.xaxis.label.set_clip_on(False)
    ax.yaxis.label.set_clip_on(False)
    ax.zaxis.label.set_clip_on(False)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    fig.subplots_adjust(left=0.04, right=0.84, top=0.96, bottom=0.18)
    title_size = 17 if len(title) <= 54 else 14
    fig.text(0.5, 0.06, title, color="white", fontsize=title_size, fontweight="bold", ha="center", va="center")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.14)
    print(f"Wrote {output}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    urdf_path = Path(args.urdf).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    dataset = load_v21_dataset(dataset_path)
    fk = SimpleArmFk(urdf_path)
    left, right = load_episode_positions(dataset, args.episode, fk, args.source)
    source_label = "observation.state" if args.source == "state" else "action"
    title = args.title or f"Kuavo EEF Trajectory · episode {args.episode} · {source_label}"

    print(
        "Episode",
        args.episode,
        f"frames={len(left)}",
        f"left_length={path_length(left):.4f}m",
        f"right_length={path_length(right):.4f}m",
    )
    draw_trajectory(left, right, args.hand, title, output_path, args.dpi, args.show)


if __name__ == "__main__":
    main()
