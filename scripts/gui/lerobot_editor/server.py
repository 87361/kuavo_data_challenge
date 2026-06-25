from __future__ import annotations

import argparse
import mimetypes
import os
import threading
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


class CoverageAnalysisRequest(BaseModel):
    window_seconds: float = 1.0
    cluster_count: int | str = "auto"


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


class AppState:
    def __init__(self) -> None:
        self.data_root = Path("/mnt/data/kuavo_tianchi")
        self.urdf_path: Path | None = None
        self.transition_step_m = 0.025
        self.min_transition_frames = 2
        self.max_transition_frames = 20
        self.dataset: LeRobotV21Dataset | None = None
        self.edits: dict[str, Any] = {}
        self.export_job = ExportJob()
        self.analysis_job = AnalysisJob()

    def require_dataset(self) -> LeRobotV21Dataset:
        if self.dataset is None:
            raise HTTPException(status_code=400, detail="No dataset is open")
        return self.dataset


state = AppState()
app = FastAPI(title="LeRobot v2.1 Editor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/datasets")
def api_datasets() -> dict[str, Any]:
    return {"datasets": [item.__dict__ for item in find_lerobot_datasets(state.data_root)]}


@app.post("/api/open")
def api_open(req: OpenRequest) -> dict[str, Any]:
    try:
        dataset = load_v21_dataset(req.path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.dataset = dataset
    state.edits = {}
    state.analysis_job.reset()
    return {
        "path": str(dataset.root),
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
            transition_step_m=state.transition_step_m,
            min_transition_frames=state.min_transition_frames,
            max_transition_frames=state.max_transition_frames,
            video_codec=req.video_codec,
            progress=progress,
        )
        state.export_job.update(status="complete", message="export complete", progress=1.0, manifest=manifest)
    except Exception as exc:
        state.export_job.update(status="failed", message="export failed", error=str(exc))


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
    parser.add_argument("--max-transition-frames", type=int, default=20)
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
