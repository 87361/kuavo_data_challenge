from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


@dataclass(frozen=True)
class DatasetSummary:
    path: str
    name: str
    version: str
    editable: bool
    total_episodes: int
    total_frames: int
    fps: int | float
    tasks: list[str]
    video_keys: list[str]


@dataclass
class LeRobotV21Dataset:
    root: Path
    info: dict[str, Any]
    episodes: list[dict[str, Any]]
    tasks: list[dict[str, Any]]

    @property
    def fps(self) -> int | float:
        return self.info.get("fps", 10)

    @property
    def chunks_size(self) -> int:
        return int(self.info.get("chunks_size") or 1000)

    @property
    def video_keys(self) -> list[str]:
        return video_keys_from_info(self.info)

    @property
    def state_names(self) -> list[str]:
        feature = self.info.get("features", {}).get("observation.state", {})
        names = feature.get("names") or {}
        return list(names.get("state_names") or [])

    @property
    def action_names(self) -> list[str]:
        feature = self.info.get("features", {}).get("action", {})
        names = feature.get("names") or {}
        return list(names.get("action_names") or [])

    def episode_length(self, episode_index: int) -> int:
        return int(self.episodes[episode_index]["length"])

    def episode_chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def parquet_path(self, episode_index: int) -> Path:
        pattern = self.info.get("data_path") or LEGACY_DATA_PATH
        return self.root / pattern.format(
            episode_chunk=self.episode_chunk(episode_index),
            episode_index=episode_index,
        )

    def video_path(self, episode_index: int, video_key: str) -> Path:
        pattern = self.info.get("video_path") or LEGACY_VIDEO_PATH
        return self.root / pattern.format(
            episode_chunk=self.episode_chunk(episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )

    def read_episode_dataframe(self, episode_index: int) -> pd.DataFrame:
        return pq.read_table(self.parquet_path(episode_index)).to_pandas()


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [key for key, feature in features.items() if feature.get("dtype") == "video"]


def tasks_from_root(root: Path) -> list[dict[str, Any]]:
    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        return read_jsonl(tasks_path)
    return []


def task_names(tasks: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in tasks:
        value = row.get("task") or row.get("tasks")
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, list):
            names.extend(str(item) for item in value)
    return names


def dataset_summary(root: Path) -> DatasetSummary | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = read_json(info_path)
    except Exception:
        return None

    version = str(info.get("codebase_version") or "unknown")
    tasks = tasks_from_root(root)
    editable = (
        version == "v2.1"
        and (root / "meta" / "episodes.jsonl").exists()
        and (root / "meta" / "episodes_stats.jsonl").exists()
    )
    return DatasetSummary(
        path=str(root),
        name=root.parent.name if root.name == "lerobot" else root.name,
        version=version,
        editable=editable,
        total_episodes=int(info.get("total_episodes") or 0),
        total_frames=int(info.get("total_frames") or 0),
        fps=info.get("fps") or 0,
        tasks=task_names(tasks),
        video_keys=video_keys_from_info(info),
    )


def find_lerobot_datasets(data_root: Path) -> list[DatasetSummary]:
    summaries: list[DatasetSummary] = []
    seen: set[Path] = set()
    for info_path in sorted(data_root.glob("**/meta/info.json")):
        root = info_path.parent.parent.resolve()
        try:
            rel_parts = root.relative_to(data_root.resolve()).parts
        except ValueError:
            rel_parts = root.parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if root in seen:
            continue
        seen.add(root)
        summary = dataset_summary(root)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda item: (not item.editable, item.path))
    return summaries


def load_v21_dataset(root: str | Path) -> LeRobotV21Dataset:
    dataset_root = Path(root).expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{dataset_root} is {info.get('codebase_version')}, only v2.1 is editable")
    episodes = sorted(
        read_jsonl(dataset_root / "meta" / "episodes.jsonl"),
        key=lambda row: int(row["episode_index"]),
    )
    tasks = tasks_from_root(dataset_root)
    return LeRobotV21Dataset(root=dataset_root, info=info, episodes=episodes, tasks=tasks)


def dataframe_to_curve_payload(df: pd.DataFrame, key: str) -> list[list[float]]:
    if key not in df:
        return []
    payload: list[list[float]] = []
    for value in df[key].tolist():
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        payload.append([float(item) for item in arr])
    return payload


@lru_cache(maxsize=512)
def decode_frame_jpeg(video_path: str, frame_index: int, max_width: int = 720, quality: int = 85) -> bytes:
    path = Path(video_path)
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_index:
                image = frame.to_ndarray(format="rgb24")
                if max_width > 0 and image.shape[1] > max_width:
                    scale = max_width / image.shape[1]
                    image = cv2.resize(
                        image,
                        (max_width, int(image.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
                )
                if not ok:
                    raise RuntimeError(f"failed to encode frame {frame_index} from {path}")
                return encoded.tobytes()
    raise IndexError(f"frame {frame_index} not found in {path}")
