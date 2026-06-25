from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .dataset import LeRobotV21Dataset, load_v21_dataset, read_jsonl, write_json, write_jsonl
    from .edits import deleted_ranges, kept_ranges, normalize_episode_edit, transition_frame_count
    from .urdf_fk import SimpleArmFk
except ImportError:  # pragma: no cover - direct script execution fallback
    from dataset import LeRobotV21Dataset, load_v21_dataset, read_jsonl, write_json, write_jsonl
    from edits import deleted_ranges, kept_ranges, normalize_episode_edit, transition_frame_count
    from urdf_fk import SimpleArmFk


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class ExportJob:
    status: str = "idle"
    message: str = ""
    progress: float = 0.0
    output_path: str | None = None
    error: str | None = None
    manifest: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "progress": self.progress,
                "output_path": self.output_path,
                "error": self.error,
                "manifest": self.manifest,
            }


def choose_video_codec(requested: str | None = None) -> str:
    candidates = [requested] if requested else ["libsvtav1", "libx264", "mpeg4"]
    for name in candidates:
        if not name:
            continue
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError("no supported PyAV video encoder found")


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _numeric_stats(values: np.ndarray) -> dict[str, list[Any]]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": np.min(arr, axis=0).tolist(),
        "max": np.max(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def compute_episode_stats(
    df: pd.DataFrame,
    info: dict[str, Any],
    source_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_stats = source_stats or {}
    stats: dict[str, Any] = {}
    features = info.get("features", {})
    for key, feature in features.items():
        dtype = feature.get("dtype")
        if key not in df.columns:
            if key in source_stats:
                stats[key] = source_stats[key]
            continue
        if dtype == "video":
            stats[key] = source_stats.get(key, {"count": [int(len(df))]})
            continue
        if dtype in {"float32", "float64"} and feature.get("shape") not in ([1], (1,), None):
            values = np.stack([_as_vector(item) for item in df[key].tolist()])
        else:
            values = np.asarray(df[key].tolist())
        stats[key] = _numeric_stats(values)
    return _to_jsonable(stats)


def interpolate_rows(
    left_row: pd.Series,
    right_row: pd.Series,
    frame_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left_state = _as_vector(left_row["observation.state"])
    right_state = _as_vector(right_row["observation.state"])
    left_action = _as_vector(left_row["action"])
    right_action = _as_vector(right_row["action"])
    for idx in range(frame_count):
        alpha = float(idx + 1) / float(frame_count + 1)
        row = left_row.to_dict()
        row["observation.state"] = ((1.0 - alpha) * left_state + alpha * right_state).astype(np.float32)
        row["action"] = ((1.0 - alpha) * left_action + alpha * right_action).astype(np.float32)
        row["_synthetic_transition"] = True
        rows.append(row)
    return rows


def build_output_dataframe(
    source_df: pd.DataFrame,
    edit: dict[str, Any],
    fps: int | float,
    episode_index: int,
    global_start_index: int,
    fk: SimpleArmFk | None,
    transition_step_m: float,
    min_transition_frames: int,
    max_transition_frames: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    length = len(source_df)
    normalized = normalize_episode_edit(edit, length)
    ranges = kept_ranges(length, normalized["cuts"], normalized["deleted_segments"])
    if not ranges:
        raise ValueError(f"episode {episode_index} keeps no frames")

    rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for range_idx, (start, end) in enumerate(ranges):
        if range_idx > 0:
            prev_end = ranges[range_idx - 1][1]
            if prev_end < start:
                left_row = source_df.iloc[prev_end - 1]
                right_row = source_df.iloc[start]
                distance = None
                if fk is not None:
                    distance = fk.max_eef_distance(left_row["observation.state"], right_row["observation.state"])
                    frames = transition_frame_count(
                        distance,
                        step_m=transition_step_m,
                        min_frames=min_transition_frames,
                        max_frames=max_transition_frames,
                    )
                else:
                    frames = min_transition_frames
                rows.extend(interpolate_rows(left_row, right_row, frames))
                transitions.append(
                    {
                        "from_source_frame": int(prev_end - 1),
                        "to_source_frame": int(start),
                        "deleted_range": [int(prev_end), int(start)],
                        "eef_distance_m": float(distance) if distance is not None else None,
                        "transition_frames": int(frames),
                    }
                )
        for _, row in source_df.iloc[start:end].iterrows():
            item = row.to_dict()
            item["_synthetic_transition"] = False
            rows.append(item)

    output_df = pd.DataFrame(rows)
    output_df = output_df.drop(columns=["_synthetic_transition"], errors="ignore")
    for frame_index in range(len(output_df)):
        output_df.at[frame_index, "frame_index"] = int(frame_index)
        output_df.at[frame_index, "timestamp"] = float(frame_index) / float(fps)
        output_df.at[frame_index, "episode_index"] = int(episode_index)
        output_df.at[frame_index, "index"] = int(global_start_index + frame_index)
    return output_df, transitions


def _write_dataframe_like_source(df: pd.DataFrame, source_parquet: Path, output_parquet: Path) -> None:
    source_schema = pq.read_schema(source_parquet)
    columns: dict[str, list[Any]] = {}
    for name in source_schema.names:
        values = df[name].tolist()
        if name in {"observation.state", "action"}:
            values = [_as_vector(item).astype(np.float32).tolist() for item in values]
        columns[name] = values
    try:
        table = pa.Table.from_pydict(columns, schema=source_schema)
    except Exception:
        table = pa.Table.from_pydict(columns)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")


class StreamingVideoReader:
    def __init__(self, input_path: Path):
        self.container = av.open(str(input_path))
        self.stream = self.container.streams.video[0]
        self.iterator = enumerate(self.container.decode(self.stream))
        self.pending: tuple[int, np.ndarray] | None = None

    @property
    def width(self) -> int:
        return int(self.stream.codec_context.width)

    @property
    def height(self) -> int:
        return int(self.stream.codec_context.height)

    def close(self) -> None:
        self.container.close()

    def next_until(self, target_index: int) -> tuple[int, np.ndarray]:
        if self.pending is not None and self.pending[0] == target_index:
            pending = self.pending
            self.pending = None
            return pending
        for idx, frame in self.iterator:
            if idx < target_index:
                continue
            image = frame.to_ndarray(format="rgb24")
            return idx, image
        raise IndexError(f"frame {target_index} not found")

    def stash(self, index: int, image: np.ndarray) -> None:
        self.pending = (index, image)


def encode_edited_video(
    input_path: Path,
    output_path: Path,
    ranges: list[tuple[int, int]],
    transitions: list[dict[str, Any]],
    fps: int | float,
    codec: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transition_by_next = {int(item["to_source_frame"]): int(item["transition_frames"]) for item in transitions}
    reader = StreamingVideoReader(input_path)
    try:
        with av.open(str(output_path), mode="w") as out:
            stream = out.add_stream(codec, rate=int(round(float(fps))))
            stream.width = reader.width
            stream.height = reader.height
            stream.pix_fmt = "yuv420p"
            if codec == "libsvtav1":
                stream.options = {"preset": "10", "crf": "35"}

            last_kept: np.ndarray | None = None
            for range_idx, (start, end) in enumerate(ranges):
                if range_idx > 0 and start in transition_by_next and last_kept is not None:
                    idx, next_image = reader.next_until(start)
                    if idx != start:
                        raise IndexError(f"expected frame {start}, got {idx}")
                    for frame_idx in range(transition_by_next[start]):
                        alpha = float(frame_idx + 1) / float(transition_by_next[start] + 1)
                        blend = cv2.addWeighted(last_kept, 1.0 - alpha, next_image, alpha, 0.0)
                        frame = av.VideoFrame.from_ndarray(blend, format="rgb24")
                        for packet in stream.encode(frame):
                            out.mux(packet)
                    reader.stash(start, next_image)

                current = start
                while current < end:
                    idx, image = reader.next_until(current)
                    if idx >= end:
                        reader.stash(idx, image)
                        break
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        out.mux(packet)
                    last_kept = image
                    current = idx + 1

            for packet in stream.encode():
                out.mux(packet)
    finally:
        reader.close()


def copy_or_encode_videos(
    dataset: LeRobotV21Dataset,
    output_root: Path,
    episode_index: int,
    edit: dict[str, Any],
    transitions: list[dict[str, Any]],
    codec: str,
) -> None:
    normalized = normalize_episode_edit(edit, dataset.episode_length(episode_index))
    ranges = kept_ranges(
        dataset.episode_length(episode_index),
        normalized["cuts"],
        normalized["deleted_segments"],
    )
    unchanged = not normalized["cuts"] and not normalized["deleted_segments"]
    for video_key in dataset.video_keys:
        source = dataset.video_path(episode_index, video_key)
        target = (
            output_root
            / "videos"
            / f"chunk-{dataset.episode_chunk(episode_index):03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if unchanged:
            shutil.copy2(source, target)
        else:
            encode_edited_video(source, target, ranges, transitions, dataset.fps, codec)


def export_v21_dataset(
    source_root: str | Path,
    output_root: str | Path,
    edits: dict[str, Any],
    urdf_path: str | Path | None,
    transition_step_m: float = 0.025,
    min_transition_frames: int = 2,
    max_transition_frames: int = 20,
    video_codec: str | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    dataset = load_v21_dataset(source_root)
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    fk = SimpleArmFk(urdf_path) if urdf_path is not None else None
    codec = choose_video_codec(video_codec)

    source_stats_rows = {
        int(row["episode_index"]): row.get("stats", {})
        for row in read_jsonl(dataset.root / "meta" / "episodes_stats.jsonl")
    }
    total = len(dataset.episodes)
    global_frame = 0
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    manifest_episodes: dict[str, Any] = {}

    for episode_index, episode in enumerate(dataset.episodes):
        if progress:
            progress(
                {
                    "message": f"exporting episode {episode_index + 1}/{total}",
                    "progress": episode_index / max(1, total),
                }
            )
        length = int(episode["length"])
        edit = normalize_episode_edit(edits.get(str(episode_index)), length)
        source_df = dataset.read_episode_dataframe(episode_index)
        output_df, transitions = build_output_dataframe(
            source_df=source_df,
            edit=edit,
            fps=dataset.fps,
            episode_index=episode_index,
            global_start_index=global_frame,
            fk=fk,
            transition_step_m=transition_step_m,
            min_transition_frames=min_transition_frames,
            max_transition_frames=max_transition_frames,
        )

        output_parquet = (
            output
            / "data"
            / f"chunk-{dataset.episode_chunk(episode_index):03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        _write_dataframe_like_source(output_df, dataset.parquet_path(episode_index), output_parquet)
        copy_or_encode_videos(dataset, output, episode_index, edit, transitions, codec)

        new_length = int(len(output_df))
        output_episodes.append(
            {
                "episode_index": int(episode_index),
                "tasks": episode.get("tasks", []),
                "length": new_length,
            }
        )
        output_stats.append(
            {
                "episode_index": int(episode_index),
                "stats": compute_episode_stats(output_df, dataset.info, source_stats_rows.get(episode_index)),
            }
        )
        manifest_episodes[str(episode_index)] = {
            "source_length": length,
            "output_length": new_length,
            "cuts": edit["cuts"],
            "deleted_segments": edit["deleted_segments"],
            "deleted_ranges": deleted_ranges(length, edit["cuts"], edit["deleted_segments"]),
            "kept_ranges": kept_ranges(length, edit["cuts"], edit["deleted_segments"]),
            "transitions": transitions,
        }
        global_frame += new_length

    (output / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset.root / "meta" / "tasks.jsonl", output / "meta" / "tasks.jsonl")
    write_jsonl(output / "meta" / "episodes.jsonl", output_episodes)
    write_jsonl(output / "meta" / "episodes_stats.jsonl", output_stats)

    info = dict(dataset.info)
    info["total_episodes"] = len(output_episodes)
    info["total_frames"] = global_frame
    info["total_videos"] = len(output_episodes) * len(dataset.video_keys)
    info["total_chunks"] = (len(output_episodes) + dataset.chunks_size - 1) // dataset.chunks_size
    info["splits"] = {"train": f"0:{len(output_episodes)}"}
    for video_key in dataset.video_keys:
        feature = info.get("features", {}).get(video_key, {})
        feature.setdefault("info", {})["video.codec"] = codec
    write_json(output / "meta" / "info.json", info)

    manifest = {
        "source_dataset": str(dataset.root),
        "output_dataset": str(output),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "format": "lerobot_v2.1",
        "fps": dataset.fps,
        "urdf_path": str(Path(urdf_path).expanduser().resolve()) if urdf_path is not None else None,
        "transition": {
            "mode": "adaptive_eef_distance" if fk is not None else "fixed_frame_count",
            "step_m": transition_step_m,
            "min_frames": min_transition_frames,
            "max_frames": max_transition_frames,
            "video_crossfade": "endpoint_alpha_blend",
            "state_action": "linear_interpolation",
        },
        "video_codec": codec,
        "episodes": manifest_episodes,
    }
    (output / "edit_manifest.json").write_text(
        json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress({"message": "export complete", "progress": 1.0})
    return manifest
