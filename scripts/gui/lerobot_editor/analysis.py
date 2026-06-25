from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    from .dataset import LeRobotV21Dataset
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset import LeRobotV21Dataset


ProgressCallback = Callable[[dict[str, Any]], None]

CLUSTER_COLORS = [
    "#57a6ff",
    "#68d391",
    "#f4bf63",
    "#ef6b73",
    "#b58cff",
    "#67d7e5",
    "#d9e368",
    "#f5987a",
    "#82cfff",
    "#8be28b",
    "#ffc857",
    "#ff7f9a",
    "#c6a7ff",
    "#6ee7b7",
    "#f7e463",
    "#ffae70",
    "#5bc0eb",
    "#9ee493",
    "#ffd166",
    "#f28482",
    "#b8b8ff",
    "#64dfdf",
    "#e9ff70",
    "#ff9f80",
]


def _state_matrix(values: list[Any]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except ValueError:
        matrix = np.vstack([np.asarray(value, dtype=np.float32).reshape(-1) for value in values])
    if matrix.ndim == 1:
        matrix = matrix.reshape(len(values), -1)
    return np.nan_to_num(matrix.reshape(matrix.shape[0], -1), copy=False)


def select_joint_coverage_indexes(dim: int) -> tuple[list[int], list[int]]:
    if dim >= 16:
        feature_indexes = [*range(0, 6), *range(8, 14)]
        auxiliary_indexes = [idx for idx in [6, 7, 14, 15] if idx < dim]
    elif dim >= 14:
        feature_indexes = [*range(0, 6), *range(7, 13)]
        auxiliary_indexes = [idx for idx in [6, 13] if idx < dim]
    elif dim >= 12:
        feature_indexes = [*range(0, 12)]
        auxiliary_indexes = [idx for idx in range(12, dim)]
    else:
        raise ValueError(f"observation.state must contain at least 12 arm joint values, got {dim}")
    return feature_indexes, auxiliary_indexes


def _auto_cluster_count(n_windows: int) -> int:
    if n_windows <= 1:
        return 1
    target = int(round(math.sqrt(n_windows * 2)))
    return min(n_windows, max(2, min(24, max(8, target))))


def resolve_cluster_count(cluster_count: int | str, n_windows: int) -> int:
    if n_windows <= 0:
        raise ValueError("no analysis windows were produced")
    if isinstance(cluster_count, str):
        if cluster_count.strip().lower() != "auto":
            raise ValueError("cluster_count must be an integer or 'auto'")
        return _auto_cluster_count(n_windows)
    return max(1, min(int(cluster_count), n_windows))


def _initial_centers(features: np.ndarray, k: int) -> np.ndarray:
    mean = features.mean(axis=0)
    first = int(np.argmin(np.sum((features - mean) ** 2, axis=1)))
    centers = [features[first].copy()]
    while len(centers) < k:
        stacked = np.vstack(centers)
        dists = np.sum((features[:, None, :] - stacked[None, :, :]) ** 2, axis=2)
        nearest = np.min(dists, axis=1)
        centers.append(features[int(np.argmax(nearest))].copy())
    return np.vstack(centers)


def _assign(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dists = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).astype(np.int32)


def _kmeans(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    scaled = (features - feature_mean) / feature_std

    centers_scaled = _initial_centers(scaled, k)
    labels = np.zeros((scaled.shape[0],), dtype=np.int32)
    for _ in range(60):
        labels = _assign(scaled, centers_scaled)
        next_centers = centers_scaled.copy()
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                next_centers[cluster_id] = scaled[mask].mean(axis=0)
        if np.max(np.abs(next_centers - centers_scaled)) < 1e-5:
            centers_scaled = next_centers
            break
        centers_scaled = next_centers

    labels = _assign(scaled, centers_scaled)
    centers = centers_scaled * feature_std + feature_mean
    return labels, centers, centers_scaled, feature_mean, feature_std


def _names_for_indexes(names: list[str], indexes: list[int]) -> list[str]:
    return [names[idx] if idx < len(names) and names[idx] else f"j{idx}" for idx in indexes]


def _rounded_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(round(float(value), 6)) for value in values]


def compute_coverage_analysis(
    dataset: LeRobotV21Dataset,
    *,
    window_seconds: float = 1.0,
    cluster_count: int | str = "auto",
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    fps = float(dataset.fps or 10)
    if not math.isfinite(fps) or fps <= 0:
        fps = 10.0
    if not math.isfinite(float(window_seconds)) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    window_frames = max(1, int(round(float(window_seconds) * fps)))

    windows: list[dict[str, Any]] = []
    window_features: list[np.ndarray] = []
    feature_indexes: list[int] | None = None
    auxiliary_indexes: list[int] | None = None
    frame_features_by_episode: dict[int, np.ndarray] = {}
    frame_aux_by_episode: dict[int, np.ndarray] = {}

    total_episodes = max(1, len(dataset.episodes))
    for episode_pos, episode in enumerate(dataset.episodes):
        episode_index = int(episode["episode_index"])
        if progress is not None:
            progress(
                {
                    "status": "running",
                    "message": f"reading episode {episode_pos + 1}/{total_episodes}",
                    "progress": 0.05 + 0.5 * (episode_pos / total_episodes),
                }
            )
        df = dataset.read_episode_dataframe(episode_index)
        if "observation.state" not in df:
            raise ValueError("dataset is missing observation.state")
        states = _state_matrix(df["observation.state"].tolist())
        if states.shape[0] == 0:
            continue
        if feature_indexes is None or auxiliary_indexes is None:
            feature_indexes, auxiliary_indexes = select_joint_coverage_indexes(states.shape[1])
        max_required = max(feature_indexes + auxiliary_indexes, default=-1)
        if states.shape[1] <= max_required:
            raise ValueError(
                f"episode {episode_index} observation.state has {states.shape[1]} values, expected index {max_required}"
            )

        selected = states[:, feature_indexes].astype(np.float32, copy=False)
        auxiliary = states[:, auxiliary_indexes].astype(np.float32, copy=False) if auxiliary_indexes else np.empty((len(states), 0))
        frame_features_by_episode[episode_index] = selected
        frame_aux_by_episode[episode_index] = auxiliary

        for start in range(0, len(states), window_frames):
            end = min(len(states), start + window_frames)
            feature_mean = selected[start:end].mean(axis=0)
            aux_mean = auxiliary[start:end].mean(axis=0) if auxiliary.shape[1] else np.empty((0,), dtype=np.float32)
            window_index = len(windows)
            windows.append(
                {
                    "index": window_index,
                    "episode_index": episode_index,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "start_time": round(start / fps, 6),
                    "end_time": round(end / fps, 6),
                    "feature_mean": _rounded_list(feature_mean),
                    "auxiliary_mean": _rounded_list(aux_mean),
                }
            )
            window_features.append(feature_mean)

    if not windows or feature_indexes is None or auxiliary_indexes is None:
        raise ValueError("no observation.state windows were available for analysis")

    feature_matrix = np.vstack(window_features).astype(np.float32, copy=False)
    k = resolve_cluster_count(cluster_count, feature_matrix.shape[0])
    if progress is not None:
        progress({"status": "running", "message": f"clustering {len(windows)} windows", "progress": 0.62})
    window_labels, centers, centers_scaled, feature_mean, feature_std = _kmeans(feature_matrix, k)

    frame_counts = np.zeros((k,), dtype=np.int64)
    window_counts = np.bincount(window_labels, minlength=k).astype(np.int64)
    for episode_index, frame_features in frame_features_by_episode.items():
        scaled = (frame_features - feature_mean) / feature_std
        frame_labels = _assign(scaled, centers_scaled)
        frame_counts += np.bincount(frame_labels, minlength=k)
        for window in [item for item in windows if item["episode_index"] == episode_index]:
            start = int(window["start_frame"])
            end = int(window["end_frame"])
            counts = np.bincount(frame_labels[start:end], minlength=k).astype(np.float64)
            total = max(1.0, float(counts.sum()))
            proportions = counts / total
            dominant = int(np.argmax(proportions))
            window["dominant_cluster"] = dominant
            window["window_cluster"] = int(window_labels[window["index"]])
            window["cluster_proportions"] = _rounded_list(proportions)

    clusters = []
    for cluster_id in range(k):
        clusters.append(
            {
                "id": cluster_id,
                "color": CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                "window_count": int(window_counts[cluster_id]),
                "frame_count": int(frame_counts[cluster_id]),
                "center": _rounded_list(centers[cluster_id]),
            }
        )

    if progress is not None:
        progress({"status": "running", "message": "building result", "progress": 0.92})

    return {
        "dataset": {
            "path": str(dataset.root),
            "fps": fps,
            "total_episodes": len(dataset.episodes),
            "total_frames": int(dataset.info.get("total_frames") or sum(int(row.get("length", 0)) for row in dataset.episodes)),
        },
        "window_seconds": float(window_seconds),
        "window_frames": int(window_frames),
        "requested_cluster_count": cluster_count,
        "cluster_count": int(k),
        "feature_indexes": feature_indexes,
        "feature_names": _names_for_indexes(dataset.state_names, feature_indexes),
        "auxiliary_indexes": auxiliary_indexes,
        "auxiliary_names": _names_for_indexes(dataset.state_names, auxiliary_indexes),
        "clusters": clusters,
        "windows": windows,
    }
