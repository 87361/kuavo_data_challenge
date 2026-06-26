from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .analysis import compute_coverage_analysis
    from .dataset import (
        LeRobotV21Dataset,
        dataframe_to_curve_payload,
        decode_frame_jpeg,
        find_lerobot_datasets,
        load_v21_dataset,
    )
    from .edits import build_segments, normalize_episode_edit
    from .exporter import ExportJob, export_v21_dataset
except ImportError:  # pragma: no cover - direct script execution fallback
    from analysis import compute_coverage_analysis
    from dataset import (
        LeRobotV21Dataset,
        dataframe_to_curve_payload,
        decode_frame_jpeg,
        find_lerobot_datasets,
        load_v21_dataset,
    )
    from edits import build_segments, normalize_episode_edit
    from exporter import ExportJob, export_v21_dataset


STATIC_DIR = Path(__file__).resolve().parent / "static"


class OpenRequest(BaseModel):
    path: str


class CutsRequest(BaseModel):
    episode_index: int
    cuts: list[int] = []
    deleted_segments: list[int] = []


class ExportRequest(BaseModel):
    output_path: str
    video_codec: str | None = None
    urdf_path: str | None = None


class ProgressSaveRequest(BaseModel):
    last_export_path: str | None = None


class EpisodeProgressRequest(BaseModel):
    completed: bool


class EpisodeAnnotationRequest(BaseModel):
    completed: bool | None = None
    rating: int | None = None
    notes: list[str] | None = None


class TrajectoryPreviewRequest(BaseModel):
    urdf_path: str | None = None
    video_key: str | None = None
    source: str = "state"
    hand: str = "auto"


class CoverageAnalysisRequest(BaseModel):
    window_seconds: float = 1.0
    cluster_count: int | str = "auto"


class AppLogBuffer:
    def __init__(self, limit: int = 500) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._items: deque[dict[str, Any]] = deque(maxlen=limit)

    def append(self, level: str, source: str, message: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            item = {
                "seq": self._seq,
                "time": _utc_now(),
                "level": level,
                "source": source,
                "message": str(message),
                **fields,
            }
            self._items.append(item)
            return dict(item)

    def read_after(self, after: int = 0) -> dict[str, Any]:
        with self._lock:
            logs = [dict(item) for item in self._items if int(item["seq"]) > after]
            return {"logs": logs, "next_seq": self._seq}


class AnalysisJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"
        self.message = ""
        self.progress = 0.0
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.params: dict[str, Any] | None = None

    def reset(self) -> None:
        self.update(status="idle", message="", progress=0.0, error=None, result=None, params=None)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "message": self.message,
                "progress": self.progress,
                "error": self.error,
                "params": self.params,
                "result": self.result,
            }


class TrajectoryJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "idle"
        self.message = ""
        self.progress = 0.0
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.params: dict[str, Any] | None = None

    def reset(self) -> None:
        self.update(status="idle", message="", progress=0.0, error=None, result=None, params=None)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            payload = {
                "status": self.status,
                "message": self.message,
                "progress": self.progress,
                "error": self.error,
                "params": self.params,
                "result": self.result,
            }
            if self.result:
                payload.update(self.result)
            return payload


class AppState:
    def __init__(self) -> None:
        self.data_root = Path("/mnt/data/kuavo_tianchi")
        self.urdf_path: Path | None = None
        self.transition_step_m = 0.025
        self.min_transition_frames = 2
        self.max_transition_frames = 60
        self.canonical_source_path: Path | None = None
        self.active_workspace_path: Path | None = None
        self.dataset: LeRobotV21Dataset | None = None
        self.edits: dict[str, Any] = {}
        self.episode_annotations: dict[str, dict[str, Any]] = {}
        self.note_labels: list[str] = []
        self.completed_episodes: set[int] = set()
        self.progress_saved_at: str | None = None
        self.last_export_path: str | None = None
        self.export_job = ExportJob()
        self.analysis_job = AnalysisJob()
        self.trajectory_job = TrajectoryJob()
        self.app_logs = AppLogBuffer()

    def require_dataset(self) -> LeRobotV21Dataset:
        if self.dataset is None:
            raise HTTPException(status_code=400, detail="No dataset is open")
        return self.dataset


state = AppState()
app = FastAPI(title="LeRobot v2.1 Editor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _progress_path(dataset: LeRobotV21Dataset) -> Path:
    return dataset.root / ".lerobot_editor" / "progress.json"


def _progress_path_for_root(root: Path) -> Path:
    return root / ".lerobot_editor" / "progress.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(level: str, source: str, message: str, **fields: Any) -> None:
    state.app_logs.append(level, source, message, **fields)


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path(root: Path) -> Path:
    return root / "edit_manifest.json"


def _editor_exports_root() -> Path:
    return state.data_root / "lerobot_edits"


def _source_slug(root: Path) -> str:
    resolved = root.expanduser().resolve()
    try:
        base = resolved.parent if resolved.name == "lerobot" else resolved
        rel = base.relative_to(state.data_root)
        text = "_".join(rel.parts)
    except ValueError:
        parts = resolved.parts[-3:-1] if resolved.name == "lerobot" and len(resolved.parts) >= 3 else resolved.parts[-2:]
        text = "_".join(parts)
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "lerobot"


def _default_workspace_path(root: Path) -> Path:
    return _editor_exports_root() / _source_slug(root) / "lerobot"


def _resolve_manifest_root_source(manifest: dict[str, Any], manifest_root: Path, seen: set[Path] | None = None) -> Path | None:
    root_value = manifest.get("root_source_dataset")
    if root_value:
        return Path(root_value).expanduser().resolve()
    source_value = manifest.get("source_dataset")
    if not source_value:
        return None
    source = Path(source_value).expanduser().resolve()
    seen = seen or set()
    if source in seen:
        return source
    seen.add(source)
    nested_manifest = _manifest_path(source)
    if nested_manifest.exists():
        try:
            nested = _read_json_file(nested_manifest)
        except Exception:
            return source
        return _resolve_manifest_root_source(nested, source, seen) or source
    return source


def _resolve_canonical_source(root: str | Path) -> Path:
    current = Path(root).expanduser().resolve()
    seen: set[Path] = set()
    while True:
        if current in seen:
            return current
        seen.add(current)
        manifest_path = _manifest_path(current)
        if not manifest_path.exists():
            return current
        try:
            manifest = _read_json_file(manifest_path)
        except Exception:
            return current
        next_value = manifest.get("root_source_dataset") or manifest.get("source_dataset")
        if not next_value:
            return current
        current = Path(next_value).expanduser().resolve()


def _find_active_workspace(canonical_source: Path) -> Path | None:
    exports_root = _editor_exports_root()
    if not exports_root.exists():
        return None
    best: tuple[str, float, Path] | None = None
    for manifest_path in exports_root.glob("**/edit_manifest.json"):
        try:
            manifest = _read_json_file(manifest_path)
        except Exception:
            continue
        root_source = _resolve_manifest_root_source(manifest, manifest_path.parent)
        if root_source != canonical_source:
            continue
        generated = str(manifest.get("generated_at") or "")
        mtime = manifest_path.stat().st_mtime
        candidate = (generated, mtime, manifest_path.parent)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else None


def _workspace_source_path(workspace: Path, canonical_source: Path) -> Path:
    manifest_path = _manifest_path(workspace)
    if not manifest_path.exists():
        return canonical_source
    try:
        manifest = _read_json_file(manifest_path)
    except Exception:
        return canonical_source
    source_value = manifest.get("source_dataset") or manifest.get("root_source_dataset")
    if not source_value:
        return canonical_source
    return Path(source_value).expanduser().resolve()


def _read_progress_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return _read_json_file(path)
    except Exception:
        return None


def _progress_matches_dataset(payload: dict[str, Any], canonical_source: Path, active_dataset: Path) -> bool:
    source = payload.get("root_source_dataset") or payload.get("source_dataset")
    if source is None:
        return True
    try:
        source_path = Path(source).expanduser().resolve()
    except Exception:
        return False
    return source_path in {canonical_source, active_dataset}


def _normalize_progress_edits(dataset: LeRobotV21Dataset, edits: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, edit in (edits or {}).items():
        try:
            episode_index = int(key)
        except (TypeError, ValueError):
            continue
        if episode_index < 0 or episode_index >= len(dataset.episodes):
            continue
        value = normalize_episode_edit(edit, dataset.episode_length(episode_index))
        if value["cuts"] or value["deleted_segments"]:
            normalized[str(episode_index)] = value
    return normalized


def _normalize_completed_episodes(dataset: LeRobotV21Dataset, episodes: Any) -> set[int]:
    completed: set[int] = set()
    for item in episodes or []:
        try:
            episode_index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= episode_index < len(dataset.episodes):
            completed.add(episode_index)
    return completed


def _normalize_notes(notes: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in notes or []:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _normalize_note_labels(labels: Any) -> list[str]:
    return _normalize_notes(labels)


def _normalize_annotation(dataset: LeRobotV21Dataset, value: dict[str, Any] | None, fallback_saved_at: str | None = None) -> dict[str, Any]:
    value = value or {}
    rating = value.get("rating")
    try:
        rating_value = int(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating_value = None
    if rating_value is not None and not 1 <= rating_value <= 10:
        rating_value = None
    return {
        "completed": bool(value.get("completed", False)),
        "rating": rating_value,
        "notes": _normalize_notes(value.get("notes")),
        "updated_at": str(value.get("updated_at") or fallback_saved_at or _utc_now()),
    }


def _annotation_is_empty(value: dict[str, Any]) -> bool:
    return not value.get("completed") and value.get("rating") is None and not value.get("notes")


def _normalize_episode_annotations(
    dataset: LeRobotV21Dataset,
    annotations: Any,
    completed_episodes: Any = None,
    fallback_saved_at: str | None = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in (annotations or {}).items():
        try:
            episode_index = int(key)
        except (TypeError, ValueError):
            continue
        if episode_index < 0 or episode_index >= len(dataset.episodes):
            continue
        annotation = _normalize_annotation(dataset, value, fallback_saved_at)
        if not _annotation_is_empty(annotation):
            normalized[str(episode_index)] = annotation

    for episode_index in _normalize_completed_episodes(dataset, completed_episodes):
        key = str(episode_index)
        current = normalized.get(key) or {
            "completed": False,
            "rating": None,
            "notes": [],
            "updated_at": fallback_saved_at or _utc_now(),
        }
        current["completed"] = True
        current["updated_at"] = current.get("updated_at") or fallback_saved_at or _utc_now()
        normalized[key] = current
    return normalized


def _merge_episode_annotations(
    current: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = dict(current)
    for key, annotation in incoming.items():
        existing = merged.get(key)
        if existing is None or str(annotation.get("updated_at") or "") >= str(existing.get("updated_at") or ""):
            merged[key] = annotation
    return merged


def _sync_completed_from_annotations() -> None:
    state.completed_episodes = {
        int(key)
        for key, annotation in state.episode_annotations.items()
        if annotation.get("completed")
    }


def _merge_note_labels(*groups: Any) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for label in _normalize_note_labels(group):
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def _apply_progress_payload(dataset: LeRobotV21Dataset, payload: dict[str, Any]) -> None:
    saved_at = payload.get("saved_at")
    edits = _normalize_progress_edits(dataset, payload.get("edits"))
    state.edits.update(edits)
    annotations = _normalize_episode_annotations(
        dataset,
        payload.get("episode_annotations"),
        payload.get("completed_episodes"),
        saved_at,
    )
    state.episode_annotations = _merge_episode_annotations(state.episode_annotations, annotations)
    state.note_labels = _merge_note_labels(state.note_labels, payload.get("note_labels"))
    if saved_at and (state.progress_saved_at is None or str(saved_at) > str(state.progress_saved_at)):
        state.progress_saved_at = str(saved_at)
    last_export = payload.get("last_export_path") or payload.get("active_workspace_path")
    if last_export:
        state.last_export_path = str(last_export)


def _apply_manifest_progress(dataset: LeRobotV21Dataset, manifest: dict[str, Any]) -> None:
    manifest_edits: dict[str, Any] = {}
    for key, entry in (manifest.get("episodes") or {}).items():
        manifest_edits[key] = {
            "cuts": entry.get("cuts", []),
            "deleted_segments": entry.get("deleted_segments", []),
        }
    state.edits.update(_normalize_progress_edits(dataset, manifest_edits))
    state.episode_annotations = _merge_episode_annotations(
        state.episode_annotations,
        _normalize_episode_annotations(
            dataset,
            manifest.get("episode_annotations"),
            manifest.get("completed_episodes"),
            manifest.get("generated_at"),
        ),
    )
    state.note_labels = _merge_note_labels(state.note_labels, manifest.get("note_labels"))


def _current_annotation(episode_index: int) -> dict[str, Any]:
    return state.episode_annotations.get(
        str(episode_index),
        {"completed": False, "rating": None, "notes": [], "updated_at": None},
    )


def _progress_payload(dataset: LeRobotV21Dataset) -> dict[str, Any]:
    pending_export_episodes = _pending_export_episodes(dataset)
    _sync_completed_from_annotations()
    canonical = state.canonical_source_path or dataset.root
    active_workspace = state.active_workspace_path or (
        Path(state.last_export_path).expanduser().resolve() if state.last_export_path else None
    )
    return {
        "source_dataset": str(canonical),
        "active_dataset": str(dataset.root),
        "root_source_dataset": str(canonical),
        "saved_at": state.progress_saved_at,
        "edits": state.edits,
        "episode_annotations": state.episode_annotations,
        "note_labels": state.note_labels,
        "completed_episodes": sorted(state.completed_episodes),
        "completed_count": len(state.completed_episodes),
        "edited_count": len(state.edits),
        "pending_export_episodes": pending_export_episodes,
        "pending_export_count": len(pending_export_episodes),
        "total_episodes": len(dataset.episodes),
        "active_workspace_path": str(active_workspace) if active_workspace else None,
        "default_export_path": str(_default_workspace_path(canonical)),
        "last_export_path": str(active_workspace) if active_workspace else state.last_export_path,
    }


def _pending_export_episodes(dataset: LeRobotV21Dataset) -> list[int]:
    export_path = state.active_workspace_path or (Path(state.last_export_path).expanduser().resolve() if state.last_export_path else None)
    if export_path is None:
        return sorted(int(key) for key in state.edits)
    manifest_path = export_path / "edit_manifest.json"
    if not manifest_path.exists():
        return sorted(int(key) for key in state.edits)
    try:
        manifest = _read_json_file(manifest_path)
    except Exception:
        return sorted(int(key) for key in state.edits)
    root_source = _resolve_manifest_root_source(manifest, export_path)
    if root_source != (state.canonical_source_path or dataset.root):
        return sorted(int(key) for key in state.edits)

    pending: list[int] = []
    exported = manifest.get("episodes") or {}
    for episode_index in range(len(dataset.episodes)):
        current = normalize_episode_edit(state.edits.get(str(episode_index)), dataset.episode_length(episode_index))
        entry = exported.get(str(episode_index)) or {}
        exported_edit = {
            "cuts": entry.get("cuts", []),
            "deleted_segments": entry.get("deleted_segments", []),
        }
        if current != exported_edit:
            pending.append(episode_index)
    return pending


def _load_progress(dataset: LeRobotV21Dataset) -> None:
    state.edits = {}
    state.episode_annotations = {}
    state.note_labels = []
    state.completed_episodes = set()
    state.progress_saved_at = None
    canonical = state.canonical_source_path or dataset.root
    active_dataset = dataset.root
    if state.active_workspace_path is not None:
        manifest_path = _manifest_path(state.active_workspace_path)
        if manifest_path.exists():
            try:
                _apply_manifest_progress(dataset, _read_json_file(manifest_path))
            except Exception:
                pass
        state.last_export_path = str(state.active_workspace_path)
    else:
        state.last_export_path = str(_default_workspace_path(canonical))

    source_payload = _read_progress_payload(_progress_path_for_root(canonical))
    if source_payload and _progress_matches_dataset(source_payload, canonical, active_dataset):
        _apply_progress_payload(dataset, source_payload)

    if state.active_workspace_path is not None:
        workspace_payload = _read_progress_payload(_progress_path_for_root(state.active_workspace_path))
        if workspace_payload and _progress_matches_dataset(workspace_payload, canonical, active_dataset):
            _apply_progress_payload(dataset, workspace_payload)

    _sync_completed_from_annotations()


def _save_progress(dataset: LeRobotV21Dataset) -> dict[str, Any]:
    state.progress_saved_at = _utc_now()
    payload = _progress_payload(dataset)
    paths = [_progress_path_for_root(state.canonical_source_path or dataset.root)]
    if state.active_workspace_path is not None and state.active_workspace_path.exists():
        paths.append(_progress_path_for_root(state.active_workspace_path))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/logs")
def api_logs(after: int = Query(0, ge=0)) -> dict[str, Any]:
    return state.app_logs.read_after(after)


@app.get("/api/datasets")
def api_datasets() -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in find_lerobot_datasets(state.data_root):
        canonical = _resolve_canonical_source(item.path)
        item_path = Path(item.path).expanduser().resolve()
        if canonical != item_path or canonical in seen:
            continue
        seen.add(canonical)
        active_workspace = _find_active_workspace(canonical)
        payload = item.__dict__.copy()
        payload["canonical_source_path"] = str(canonical)
        payload["active_workspace_path"] = str(active_workspace) if active_workspace else None
        payload["default_export_path"] = str(_default_workspace_path(canonical))
        datasets.append(payload)
    return {"datasets": datasets}


@app.post("/api/open")
def api_open(req: OpenRequest) -> dict[str, Any]:
    try:
        canonical = _resolve_canonical_source(req.path)
        active_workspace = _find_active_workspace(canonical)
        active_dataset_path = _workspace_source_path(active_workspace, canonical) if active_workspace else canonical
        dataset = load_v21_dataset(active_dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.canonical_source_path = canonical
    state.active_workspace_path = active_workspace
    state.dataset = dataset
    _load_progress(dataset)
    state.analysis_job.reset()
    state.trajectory_job.reset()
    _log("info", "dataset", f"Opened dataset {canonical}")
    return {
        "path": str(canonical),
        "active_dataset": str(dataset.root),
        "active_workspace_path": str(active_workspace) if active_workspace else None,
        "default_export_path": str(_default_workspace_path(canonical)),
        "fps": dataset.fps,
        "total_episodes": len(dataset.episodes),
        "total_frames": int(dataset.info.get("total_frames") or 0),
        "video_keys": dataset.video_keys,
        "state_names": dataset.state_names,
        "action_names": dataset.action_names,
        "urdf_path": str(state.urdf_path) if state.urdf_path is not None else None,
        "episodes": [
            {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "tasks": row.get("tasks", []),
            }
            for row in dataset.episodes
        ],
        "progress": _progress_payload(dataset),
    }


@app.get("/api/episode/{episode_index}")
def api_episode(episode_index: int) -> dict[str, Any]:
    dataset = state.require_dataset()
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    df = dataset.read_episode_dataframe(episode_index)
    length = len(df)
    edit = normalize_episode_edit(state.edits.get(str(episode_index)), length)
    return {
        "episode_index": episode_index,
        "length": length,
        "fps": dataset.fps,
        "video_keys": dataset.video_keys,
        "state_names": dataset.state_names,
        "action_names": dataset.action_names,
        "curves": {
            "observation.state": dataframe_to_curve_payload(df, "observation.state"),
            "action": dataframe_to_curve_payload(df, "action"),
        },
        "annotation": _current_annotation(episode_index),
        "cuts": edit["cuts"],
        "deleted_segments": edit["deleted_segments"],
        "segments": [seg.as_dict() for seg in build_segments(length, edit["cuts"], edit["deleted_segments"])],
    }


@app.get("/api/frame")
def api_frame(
    episode_index: int = Query(..., ge=0),
    frame_index: int = Query(..., ge=0),
    video_key: str = Query(...),
    max_width: int = Query(720, ge=100, le=1920),
) -> Response:
    dataset = state.require_dataset()
    if video_key not in dataset.video_keys:
        raise HTTPException(status_code=404, detail=f"unknown video key: {video_key}")
    if frame_index >= dataset.episode_length(episode_index):
        raise HTTPException(status_code=404, detail="frame not found")
    try:
        jpeg = decode_frame_jpeg(str(dataset.video_path(episode_index, video_key)), frame_index, max_width)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/video")
def api_video(
    episode_index: int = Query(..., ge=0),
    video_key: str = Query(...),
) -> FileResponse:
    dataset = state.require_dataset()
    if video_key not in dataset.video_keys:
        raise HTTPException(status_code=404, detail=f"unknown video key: {video_key}")
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    path = dataset.video_path(episode_index, video_key)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/cuts")
def api_cuts(req: CutsRequest) -> dict[str, Any]:
    dataset = state.require_dataset()
    if req.episode_index < 0 or req.episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    try:
        edit = normalize_episode_edit(
            {"cuts": req.cuts, "deleted_segments": req.deleted_segments},
            dataset.episode_length(req.episode_index),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if edit["cuts"] or edit["deleted_segments"]:
        state.edits[str(req.episode_index)] = edit
    else:
        state.edits.pop(str(req.episode_index), None)
    return {
        "episode_index": req.episode_index,
        "cuts": edit["cuts"],
        "deleted_segments": edit["deleted_segments"],
        "segments": [
            seg.as_dict()
            for seg in build_segments(dataset.episode_length(req.episode_index), edit["cuts"], edit["deleted_segments"])
        ],
        "all_edits": state.edits,
    }


@app.get("/api/progress")
def api_progress() -> dict[str, Any]:
    dataset = state.require_dataset()
    return _progress_payload(dataset)


@app.post("/api/progress/save")
def api_progress_save(req: ProgressSaveRequest) -> dict[str, Any]:
    dataset = state.require_dataset()
    if req.last_export_path is not None:
        state.last_export_path = req.last_export_path
        state.active_workspace_path = Path(req.last_export_path).expanduser().resolve()
    return _save_progress(dataset)


@app.post("/api/progress/episode/{episode_index}")
def api_episode_progress(episode_index: int, req: EpisodeProgressRequest) -> dict[str, Any]:
    dataset = state.require_dataset()
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    annotation = dict(_current_annotation(episode_index))
    annotation["completed"] = bool(req.completed)
    annotation["updated_at"] = _utc_now()
    if _annotation_is_empty(annotation):
        state.episode_annotations.pop(str(episode_index), None)
    else:
        state.episode_annotations[str(episode_index)] = _normalize_annotation(dataset, annotation)
    _sync_completed_from_annotations()
    return _progress_payload(dataset)


@app.post("/api/annotations/episode/{episode_index}")
def api_episode_annotation(episode_index: int, req: EpisodeAnnotationRequest) -> dict[str, Any]:
    dataset = state.require_dataset()
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    fields = getattr(req, "model_fields_set", None)
    if fields is None:
        fields = getattr(req, "__fields_set__", set())
    annotation = dict(_current_annotation(episode_index))
    if "completed" in fields:
        annotation["completed"] = bool(req.completed)
    if "rating" in fields:
        if req.rating is None:
            annotation["rating"] = None
        elif 1 <= int(req.rating) <= 10:
            annotation["rating"] = int(req.rating)
        else:
            raise HTTPException(status_code=400, detail="rating must be between 1 and 10")
    if "notes" in fields:
        annotation["notes"] = _normalize_notes(req.notes)
        state.note_labels = _merge_note_labels(state.note_labels, annotation["notes"])
    annotation["updated_at"] = _utc_now()
    annotation = _normalize_annotation(dataset, annotation)
    if _annotation_is_empty(annotation):
        state.episode_annotations.pop(str(episode_index), None)
    else:
        state.episode_annotations[str(episode_index)] = annotation
    _sync_completed_from_annotations()
    return _progress_payload(dataset)


def _run_export(req: ExportRequest) -> None:
    try:
        dataset = state.require_dataset()
        state.export_job.update(
            status="running",
            message="starting export",
            progress=0.0,
            output_path=req.output_path,
            error=None,
            manifest=None,
        )
        _log("info", "export", f"Starting export to {req.output_path}")

        def progress(payload: dict[str, Any]) -> None:
            state.export_job.update(**payload)

        requested_urdf = req.urdf_path.strip() if req.urdf_path else ""
        urdf_path = Path(requested_urdf).expanduser().resolve() if requested_urdf else state.urdf_path
        if urdf_path is not None:
            state.urdf_path = urdf_path

        manifest = export_v21_dataset(
            source_root=dataset.root,
            output_root=req.output_path,
            edits=state.edits,
            urdf_path=urdf_path,
            root_source_dataset=state.canonical_source_path or dataset.root,
            episode_annotations=state.episode_annotations,
            note_labels=state.note_labels,
            transition_step_m=state.transition_step_m,
            min_transition_frames=state.min_transition_frames,
            max_transition_frames=state.max_transition_frames,
            video_codec=req.video_codec,
            progress=progress,
        )
        state.active_workspace_path = Path(req.output_path).expanduser().resolve()
        state.last_export_path = str(state.active_workspace_path)
        _save_progress(dataset)
        state.export_job.update(status="complete", message="export complete", progress=1.0, manifest=manifest)
        _log("info", "export", "Export complete", output_path=req.output_path)
    except Exception as exc:
        state.export_job.update(status="failed", message="export failed", error=str(exc))
        _log("error", "export", f"Export failed: {exc}")


@app.post("/api/export")
def api_export(req: ExportRequest) -> dict[str, Any]:
    state.require_dataset()
    if state.export_job.status == "running":
        raise HTTPException(status_code=409, detail="export already running")
    thread = threading.Thread(target=_run_export, args=(req,), daemon=True)
    thread.start()
    return state.export_job.as_dict()


@app.get("/api/export/status")
def api_export_status() -> dict[str, Any]:
    return state.export_job.as_dict()


def _trajectory_cache_dir() -> Path:
    root = state.active_workspace_path or state.canonical_source_path or state.require_dataset().root
    return root / ".lerobot_editor" / "trajectory_previews"


def _trajectory_cache_name(
    dataset: LeRobotV21Dataset,
    episode_index: int,
    urdf_path: Path,
    video_key: str,
    source: str,
    hand: str,
) -> str:
    edit = normalize_episode_edit(state.edits.get(str(episode_index)), dataset.episode_length(episode_index))
    fingerprint = {
        "dataset": str(dataset.root),
        "episode": episode_index,
        "urdf": str(urdf_path),
        "urdf_mtime": urdf_path.stat().st_mtime_ns if urdf_path.exists() else None,
        "video_key": video_key,
        "source": source,
        "hand": hand,
        "edit": edit,
    }
    digest = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"episode_{episode_index:06d}_{digest}.mp4"


@app.post("/api/trajectory/episode/{episode_index}")
def api_trajectory_preview(episode_index: int, req: TrajectoryPreviewRequest) -> dict[str, Any]:
    context = _prepare_trajectory_preview(episode_index, req)
    if context is None:
        return state.trajectory_job.as_dict()

    output_path = context["output_path"]
    params = context["params"]
    if output_path.exists():
        result = _trajectory_result(output_path, cached=True)
        state.trajectory_job.update(
            status="complete",
            message="cached preview ready",
            progress=1.0,
            error=None,
            result=result,
            params=params,
        )
        _log("info", "trajectory", f"Using cached trajectory preview for episode {episode_index}")
        return state.trajectory_job.as_dict()

    current = state.trajectory_job.as_dict()
    if current["status"] == "running":
        return current

    state.trajectory_job.update(
        status="running",
        message="queued trajectory preview",
        progress=0.0,
        error=None,
        result=None,
        params=params,
    )
    _log("info", "trajectory", f"Queued trajectory preview for episode {episode_index}")
    thread = threading.Thread(target=_run_trajectory_preview, args=(context,), daemon=True)
    thread.start()
    return state.trajectory_job.as_dict()


@app.get("/api/trajectory/status")
def api_trajectory_status() -> dict[str, Any]:
    return state.trajectory_job.as_dict()


def _prepare_trajectory_preview(episode_index: int, req: TrajectoryPreviewRequest) -> dict[str, Any] | None:
    dataset = state.require_dataset()
    if episode_index < 0 or episode_index >= len(dataset.episodes):
        raise HTTPException(status_code=404, detail="episode not found")
    if req.source not in {"state", "action"}:
        raise HTTPException(status_code=400, detail="source must be state or action")
    if req.hand not in {"auto", "left", "right", "both"}:
        raise HTTPException(status_code=400, detail="hand must be auto, left, right, or both")

    requested_urdf = req.urdf_path.strip() if req.urdf_path else ""
    urdf_path = Path(requested_urdf).expanduser().resolve() if requested_urdf else state.urdf_path
    if urdf_path is None or not urdf_path.exists():
        message = "Set a valid URDF path to generate the trajectory preview"
        state.trajectory_job.update(
            status="failed",
            message=message,
            progress=1.0,
            error=message,
            result=None,
            params={"episode_index": episode_index, "source": req.source, "hand": req.hand},
        )
        _log("warning", "trajectory", message)
        return None
    state.urdf_path = urdf_path

    video_key = req.video_key or (
        "observation.images.head_cam_h"
        if "observation.images.head_cam_h" in dataset.video_keys
        else (dataset.video_keys[0] if dataset.video_keys else "")
    )
    if video_key not in dataset.video_keys:
        raise HTTPException(status_code=404, detail=f"unknown video key: {video_key}")

    cache_dir = _trajectory_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / _trajectory_cache_name(dataset, episode_index, urdf_path, video_key, req.source, req.hand)
    params = {
        "episode_index": episode_index,
        "urdf_path": str(urdf_path),
        "video_key": video_key,
        "source": req.source,
        "hand": req.hand,
        "output_path": str(output_path),
    }
    return {
        "dataset": dataset,
        "episode_index": episode_index,
        "urdf_path": urdf_path,
        "video_key": video_key,
        "source": req.source,
        "hand": req.hand,
        "output_path": output_path,
        "params": params,
    }


def _trajectory_result(output_path: Path, cached: bool) -> dict[str, Any]:
    return {
        "url": f"/api/trajectory/preview/{output_path.name}",
        "cached": bool(cached),
        "path": str(output_path),
    }


def _run_trajectory_preview(context: dict[str, Any]) -> None:
    try:
        try:
            from .trajectory_video import render_video
            from .trajectory_visualizer import load_episode_positions
            from .urdf_fk import SimpleArmFk
        except ImportError:  # pragma: no cover - direct script execution fallback
            from trajectory_video import render_video
            from trajectory_visualizer import load_episode_positions
            from urdf_fk import SimpleArmFk

        dataset = context["dataset"]
        episode_index = int(context["episode_index"])
        urdf_path = context["urdf_path"]
        video_key = str(context["video_key"])
        output_path = context["output_path"]
        source = str(context["source"])
        hand = str(context["hand"])

        state.trajectory_job.update(status="running", message="loading trajectory", progress=0.05)
        _log("info", "trajectory", f"Rendering trajectory preview for episode {episode_index}")
        fk = SimpleArmFk(urdf_path)
        left, right = load_episode_positions(dataset, episode_index, fk, source)
        state.trajectory_job.update(status="running", message="rendering frames", progress=0.2)

        def progress(payload: dict[str, Any]) -> None:
            state.trajectory_job.update(**payload)

        args = argparse.Namespace(
            episode=episode_index,
            video_key=video_key,
            source=source,
            hand=hand,
            output=str(output_path),
            title=None,
            width=1280,
            height=720,
            dpi=120,
            stride=1,
            fps=None,
            max_frames=None,
            camera_left=False,
            codec="mp4v",
        )
        render_video(dataset, args, left, right, progress=progress)
        result = _trajectory_result(output_path, cached=False)
        state.trajectory_job.update(
            status="complete",
            message="trajectory preview ready",
            progress=1.0,
            error=None,
            result=result,
        )
        _log("info", "trajectory", f"Trajectory preview ready for episode {episode_index}")
    except Exception as exc:
        state.trajectory_job.update(status="failed", message="trajectory preview failed", progress=1.0, error=str(exc))
        _log("error", "trajectory", f"Trajectory preview failed: {exc}")


@app.get("/api/trajectory/preview/{filename}")
def api_trajectory_preview_file(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = _trajectory_cache_dir() / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="trajectory preview not found")
    return FileResponse(path, media_type="video/mp4", filename=filename, content_disposition_type="inline")


def _run_coverage_analysis(req: CoverageAnalysisRequest) -> None:
    try:
        dataset = state.require_dataset()
        params = {"window_seconds": req.window_seconds, "cluster_count": req.cluster_count}
        state.analysis_job.update(
            status="running",
            message="starting analysis",
            progress=0.0,
            error=None,
            result=None,
            params=params,
        )

        def progress(payload: dict[str, Any]) -> None:
            state.analysis_job.update(**payload)

        result = compute_coverage_analysis(
            dataset,
            window_seconds=req.window_seconds,
            cluster_count=req.cluster_count,
            progress=progress,
        )
        state.analysis_job.update(status="complete", message="analysis complete", progress=1.0, result=result)
    except Exception as exc:
        state.analysis_job.update(status="failed", message="analysis failed", error=str(exc), progress=1.0)


@app.post("/api/analysis/coverage")
def api_analysis_coverage(req: CoverageAnalysisRequest) -> dict[str, Any]:
    state.require_dataset()
    if state.analysis_job.status == "running":
        raise HTTPException(status_code=409, detail="coverage analysis already running")
    if req.window_seconds <= 0:
        raise HTTPException(status_code=400, detail="window_seconds must be positive")
    if isinstance(req.cluster_count, int) and req.cluster_count <= 0:
        raise HTTPException(status_code=400, detail="cluster_count must be positive")
    if isinstance(req.cluster_count, str) and req.cluster_count.strip().lower() != "auto":
        raise HTTPException(status_code=400, detail="cluster_count must be an integer or 'auto'")
    state.analysis_job.update(
        status="running",
        message="queued analysis",
        progress=0.0,
        error=None,
        result=None,
        params={"window_seconds": req.window_seconds, "cluster_count": req.cluster_count},
    )
    thread = threading.Thread(target=_run_coverage_analysis, args=(req,), daemon=True)
    thread.start()
    return state.analysis_job.as_dict()


@app.get("/api/analysis/coverage/status")
def api_analysis_coverage_status() -> dict[str, Any]:
    return state.analysis_job.as_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local LeRobot v2.1 visual editor")
    parser.add_argument("--data-root", default="/mnt/data/kuavo_tianchi")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--urdf", default=os.environ.get("KUAVO_URDF_PATH"))
    parser.add_argument("--transition-step-m", type=float, default=0.025)
    parser.add_argument("--min-transition-frames", type=int, default=2)
    parser.add_argument("--max-transition-frames", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state.data_root = Path(args.data_root).expanduser().resolve()
    state.urdf_path = Path(args.urdf).expanduser().resolve() if args.urdf else None
    state.transition_step_m = args.transition_step_m
    state.min_transition_frames = args.min_transition_frames
    state.max_transition_frames = args.max_transition_frames
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
