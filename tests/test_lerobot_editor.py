from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.gui.lerobot_editor.server as editor_server
from scripts.gui.lerobot_editor.analysis import compute_coverage_analysis
from scripts.gui.lerobot_editor.dataset import find_lerobot_datasets, load_v21_dataset
from scripts.gui.lerobot_editor.edits import build_segments, kept_ranges, transition_frame_count
from scripts.gui.lerobot_editor.exporter import export_v21_dataset
from scripts.gui.lerobot_editor.server import app, state
from scripts.gui.lerobot_editor.urdf_fk import SimpleArmFk


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_test_urdf(path: Path) -> Path:
    def chain(prefix: str, y: float) -> str:
        parts = [
            f'<joint name="zarm_{prefix}1_joint" type="revolute"><origin xyz="0 {y} 0" rpy="0 0 0"/><parent link="base_link"/><child link="zarm_{prefix}1_link"/><axis xyz="0 1 0"/></joint>'
        ]
        for idx in range(2, 8):
            parent = f"zarm_{prefix}{idx - 1}_link"
            child = f"zarm_{prefix}{idx}_link"
            axis = "1 0 0" if idx in {2, 6} else ("0 0 1" if idx in {3, 5} else "0 1 0")
            parts.append(
                f'<joint name="zarm_{prefix}{idx}_joint" type="revolute"><origin xyz="0 0 -0.1" rpy="0 0 0"/><parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/></joint>'
            )
        parts.append(
            f'<joint name="zarm_{prefix}7_end_effector_joint" type="fixed"><origin xyz="0 0 -0.2" rpy="0 0 0"/><parent link="zarm_{prefix}7_link"/><child link="zarm_{prefix}7_end_effector"/><axis xyz="0 0 0"/></joint>'
        )
        return "\n".join(parts)

    links = ["base_link"]
    for side in ["l", "r"]:
        links.extend([f"zarm_{side}{idx}_link" for idx in range(1, 8)])
        links.append(f"zarm_{side}7_end_effector")
    path.write_text(
        "<robot name='test'>"
        + "".join(f"<link name='{link}'/>" for link in links)
        + chain("l", 0.3)
        + chain("r", -0.3)
        + "</robot>",
        encoding="utf-8",
    )
    return path


def write_video(path: Path, frames: int = 5, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = size[0]
        stream.height = size[1]
        stream.pix_fmt = "yuv420p"
        for idx in range(frames):
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            image[..., 0] = idx * 30
            image[..., 1] = 120
            image[..., 2] = 220 - idx * 20
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def make_v21_dataset(root: Path, episode_count: int = 1, frames_per_episode: int = 5) -> Path:
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "total_episodes": episode_count,
        "total_frames": episode_count * frames_per_episode,
        "total_videos": episode_count,
        "chunks_size": 1000,
        "total_chunks": 1,
        "total_tasks": 1,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": {"state_names": [f"j{i}" for i in range(16)]},
            },
            "action": {
                "dtype": "float32",
                "shape": [16],
                "names": {"action_names": [f"j{i}" for i in range(16)]},
            },
            "observation.images.head_cam_h": {
                "dtype": "video",
                "shape": [3, 48, 64],
                "names": ["channels", "height", "width"],
                "info": {"video.fps": 10, "video.codec": "mpeg4"},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(root / "meta" / "info.json", info)
    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test"}])
    write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": idx, "tasks": ["test"], "length": frames_per_episode} for idx in range(episode_count)],
    )
    write_jsonl(root / "meta" / "episodes_stats.jsonl", [{"episode_index": idx, "stats": {}} for idx in range(episode_count)])

    import pyarrow as pa

    for episode_index in range(episode_count):
        rows = []
        for idx in range(frames_per_episode):
            vec = np.zeros(16, dtype=np.float32)
            vec[0] = (episode_index * 10 + idx) * 0.1
            vec[8] = (episode_index * 10 + idx) * 0.1
            rows.append(
                {
                    "observation.state": vec,
                    "action": vec.copy(),
                    "timestamp": idx / 10,
                    "frame_index": idx,
                    "episode_index": episode_index,
                    "index": episode_index * frames_per_episode + idx,
                    "task_index": 0,
                }
            )
        parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pydict(
            {
                "observation.state": [row["observation.state"].tolist() for row in rows],
                "action": [row["action"].tolist() for row in rows],
                "timestamp": [row["timestamp"] for row in rows],
                "frame_index": [row["frame_index"] for row in rows],
                "episode_index": [row["episode_index"] for row in rows],
                "index": [row["index"] for row in rows],
                "task_index": [row["task_index"] for row in rows],
            }
        )
        pq.write_table(table, parquet_path, compression="snappy")
        write_video(
            root / "videos" / "chunk-000" / "observation.images.head_cam_h" / f"episode_{episode_index:06d}.mp4",
            frames=frames_per_episode,
        )
    return root


def test_segments_and_ranges() -> None:
    segments = build_segments(10, [3, 7], [1])
    assert [(seg.start, seg.end, seg.deleted) for seg in segments] == [
        (0, 3, False),
        (3, 7, True),
        (7, 10, False),
    ]
    assert kept_ranges(10, [3, 7], [1]) == [(0, 3), (7, 10)]
    assert transition_frame_count(0.0) == 2
    assert transition_frame_count(0.08, step_m=0.025) == 4
    assert transition_frame_count(2.0, step_m=0.025) == 20


def test_urdf_fk_positions(tmp_path: Path) -> None:
    fk = SimpleArmFk(make_test_urdf(tmp_path / "test.urdf"))
    left0, right0 = fk.state_positions(np.zeros(16, dtype=np.float32))
    moved = np.zeros(16, dtype=np.float32)
    moved[0] = 0.5
    moved[8] = -0.5
    left1, right1 = fk.state_positions(moved)
    assert left0.shape == (3,)
    assert right0.shape == (3,)
    assert np.linalg.norm(left1 - left0) > 0
    assert np.linalg.norm(right1 - right0) > 0


def test_find_and_export_v21_dataset(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    urdf = make_test_urdf(tmp_path / "test.urdf")
    found = find_lerobot_datasets(tmp_path)
    assert len(found) == 1
    assert found[0].editable is True

    output = tmp_path / "out" / "task" / "lerobot"
    manifest = export_v21_dataset(
        source_root=source,
        output_root=output,
        edits={"0": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=urdf,
        transition_step_m=100.0,
        min_transition_frames=2,
        max_transition_frames=2,
        video_codec="mpeg4",
    )
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_frames"] == 5
    assert manifest["episodes"]["0"]["output_length"] == 5
    table = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.num_rows == 5
    with av.open(str(output / "videos" / "chunk-000" / "observation.images.head_cam_h" / "episode_000000.mp4")) as container:
        assert sum(1 for _ in container.decode(container.streams.video[0])) == 5
    assert (output / "edit_manifest.json").exists()


def test_export_without_urdf_uses_fixed_transition_frames(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    output = tmp_path / "out" / "task" / "lerobot"

    manifest = export_v21_dataset(
        source_root=source,
        output_root=output,
        edits={"0": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=None,
        min_transition_frames=2,
        max_transition_frames=20,
        video_codec="mpeg4",
    )

    transition = manifest["episodes"]["0"]["transitions"][0]
    assert manifest["urdf_path"] is None
    assert manifest["transition"]["mode"] == "fixed_frame_count"
    assert transition["eef_distance_m"] is None
    assert transition["transition_frames"] == 2
    assert (output / "meta" / "info.json").exists()


def test_incremental_export_updates_only_later_dirty_episode(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    output = tmp_path / "out" / "task" / "lerobot"

    first = export_v21_dataset(source, output, edits={}, urdf_path=None, video_codec="mpeg4")
    assert first["regenerated_episodes"] == [0, 1]

    video0 = output / "videos" / "chunk-000" / "observation.images.head_cam_h" / "episode_000000.mp4"
    video1 = output / "videos" / "chunk-000" / "observation.images.head_cam_h" / "episode_000001.mp4"
    parquet0 = output / "data" / "chunk-000" / "episode_000000.parquet"
    before = {
        "video0": video0.stat().st_mtime_ns,
        "video1": video1.stat().st_mtime_ns,
        "parquet0": parquet0.stat().st_mtime_ns,
    }
    time.sleep(0.02)

    second = export_v21_dataset(
        source,
        output,
        edits={"1": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=None,
        video_codec="mpeg4",
    )

    assert second["regenerated_episodes"] == [1]
    assert second["reindexed_episodes"] == []
    assert video0.stat().st_mtime_ns == before["video0"]
    assert parquet0.stat().st_mtime_ns == before["parquet0"]
    assert video1.stat().st_mtime_ns > before["video1"]


def test_incremental_export_reindexes_downstream_without_reencoding_video(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    output = tmp_path / "out" / "task" / "lerobot"

    export_v21_dataset(
        source,
        output,
        edits={},
        urdf_path=None,
        min_transition_frames=1,
        video_codec="mpeg4",
    )
    video1 = output / "videos" / "chunk-000" / "observation.images.head_cam_h" / "episode_000001.mp4"
    parquet1 = output / "data" / "chunk-000" / "episode_000001.parquet"
    video1_mtime = video1.stat().st_mtime_ns
    parquet1_mtime = parquet1.stat().st_mtime_ns
    time.sleep(0.02)

    manifest = export_v21_dataset(
        source,
        output,
        edits={"0": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=None,
        min_transition_frames=1,
        video_codec="mpeg4",
    )

    assert manifest["regenerated_episodes"] == [0]
    assert manifest["reindexed_episodes"] == [1]
    assert video1.stat().st_mtime_ns == video1_mtime
    assert parquet1.stat().st_mtime_ns > parquet1_mtime
    table = pq.read_table(parquet1).to_pandas()
    assert table["index"].tolist() == [4, 5, 6, 7, 8]
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_frames"] == 9


def test_incremental_export_rejects_non_editor_output_directory(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    output = tmp_path / "not_an_export" / "lerobot"
    output.mkdir(parents=True)
    (output / "notes.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not an editor export workspace"):
        export_v21_dataset(source, output, edits={}, urdf_path=None, video_codec="mpeg4")


def test_export_endpoint_accepts_urdf_path_from_request(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    urdf = make_test_urdf(tmp_path / "test.urdf")
    output = tmp_path / "api_out" / "task" / "lerobot"
    state.dataset = load_v21_dataset(source)
    state.canonical_source_path = source
    state.active_workspace_path = None
    state.urdf_path = None
    state.edits = {"0": {"cuts": [2, 4], "deleted_segments": [1]}}
    state.episode_annotations = {}
    state.note_labels = []
    state.export_job.status = "idle"
    client = TestClient(app)

    response = client.post(
        "/api/export",
        json={"output_path": str(output), "urdf_path": str(urdf), "video_codec": "mpeg4"},
    )

    assert response.status_code == 200
    for _ in range(40):
        status = client.get("/api/export/status").json()
        if status["status"] == "complete":
            break
        time.sleep(0.05)
    assert status["status"] == "complete"
    assert status["manifest"]["urdf_path"] == str(urdf.resolve())
    assert status["manifest"]["transition"]["mode"] == "adaptive_eef_average_speed"


def test_progress_save_load_and_restore_edits(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    state.dataset = None
    state.canonical_source_path = None
    state.active_workspace_path = None
    state.edits = {}
    state.episode_annotations = {}
    state.note_labels = []
    state.completed_episodes = set()
    state.progress_saved_at = None
    state.last_export_path = None
    client = TestClient(app)

    response = client.post("/api/open", json={"path": str(source)})
    assert response.status_code == 200
    assert response.json()["progress"]["completed_count"] == 0

    cut_response = client.post(
        "/api/cuts",
        json={"episode_index": 1, "cuts": [2, 4], "deleted_segments": [1]},
    )
    assert cut_response.status_code == 200
    complete_response = client.post("/api/progress/episode/1", json={"completed": True})
    assert complete_response.status_code == 200
    save_response = client.post("/api/progress/save", json={})
    assert save_response.status_code == 200
    saved = save_response.json()
    assert saved["completed_episodes"] == [1]
    assert saved["edited_count"] == 1
    assert (source / ".lerobot_editor" / "progress.json").exists()

    reopen = client.post("/api/open", json={"path": str(source)}).json()
    assert reopen["progress"]["completed_episodes"] == [1]
    assert reopen["progress"]["edits"]["1"]["cuts"] == [2, 4]
    episode = client.get("/api/episode/1").json()
    assert episode["cuts"] == [2, 4]
    assert episode["deleted_segments"] == [1]


def test_legacy_completed_progress_migrates_to_episode_annotations(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    write_json(
        source / ".lerobot_editor" / "progress.json",
        {
            "source_dataset": str(source),
            "saved_at": "2026-06-25T00:00:00+00:00",
            "completed_episodes": [1],
            "edits": {},
        },
    )
    state.dataset = None
    state.canonical_source_path = None
    state.active_workspace_path = None
    state.edits = {}
    state.episode_annotations = {}
    state.note_labels = []
    client = TestClient(app)

    response = client.post("/api/open", json={"path": str(source)})

    progress = response.json()["progress"]
    assert progress["completed_episodes"] == [1]
    assert progress["episode_annotations"]["1"]["completed"] is True


def test_annotation_rating_notes_save_and_reopen(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    state.dataset = None
    state.canonical_source_path = None
    state.active_workspace_path = None
    state.edits = {}
    state.episode_annotations = {}
    state.note_labels = []
    client = TestClient(app)

    assert client.post("/api/open", json={"path": str(source)}).status_code == 200
    response = client.post(
        "/api/annotations/episode/1",
        json={"completed": True, "rating": 10, "notes": ["good", "good", ""]},
    )
    assert response.status_code == 200
    saved = client.post("/api/progress/save", json={}).json()
    assert saved["completed_episodes"] == [1]
    assert saved["episode_annotations"]["1"]["rating"] == 10
    assert saved["episode_annotations"]["1"]["notes"] == ["good"]
    assert saved["note_labels"] == ["good"]
    assert saved["annotation_stats"]["completed_count"] == 1
    assert saved["annotation_stats"]["ratings"] == [{"rating": 10, "count": 1}]
    assert saved["annotation_stats"]["labels"] == [{"label": "good", "count": 1}]

    reopened = client.post("/api/open", json={"path": str(source)}).json()
    annotation = reopened["progress"]["episode_annotations"]["1"]
    assert annotation["completed"] is True
    assert annotation["rating"] == 10
    assert annotation["notes"] == ["good"]
    episode = client.get("/api/episode/1").json()
    assert episode["annotation"]["rating"] == 10


def test_episode_payload_marks_gripper_transitions(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", frames_per_episode=6)
    info_path = source / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"]["names"]["state_names"][7] = "left_claw"
    info["features"]["observation.state"]["names"]["state_names"][15] = "right_claw"
    write_json(info_path, info)

    import pyarrow as pa

    parquet_path = source / "data" / "chunk-000" / "episode_000000.parquet"
    df = pq.read_table(parquet_path).to_pandas()
    states = []
    for idx, value in enumerate(df["observation.state"].tolist()):
        vec = np.asarray(value, dtype=np.float32)
        vec[7] = 0.0 if idx < 2 else (1.0 if idx < 4 else 0.0)
        vec[15] = 0.0
        states.append(vec.tolist())
    df["observation.state"] = states
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    state.dataset = load_v21_dataset(source)
    state.edits = {}
    state.episode_annotations = {}
    client = TestClient(app)

    episode = client.get("/api/episode/0").json()
    assert episode["gripper"]["dimensions"] == [
        {"index": 7, "name": "left_claw", "side": "left"},
        {"index": 15, "name": "right_claw", "side": "right"},
    ]
    assert [
        (item["name"], item["direction"], item["start_frame"], item["end_frame"])
        for item in episode["gripper"]["transitions"]
    ] == [
        ("left_claw", "closing", 1, 2),
        ("left_claw", "opening", 3, 4),
    ]


def test_open_source_dataset_adopts_latest_legacy_workspace(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", episode_count=2)
    first = tmp_path / "lerobot_edits" / "task_edited_day1" / "lerobot"
    second = tmp_path / "lerobot_edits" / "task_edited_day2" / "lerobot"
    export_v21_dataset(
        source,
        first,
        edits={"0": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=None,
        video_codec="mpeg4",
    )
    time.sleep(0.02)
    export_v21_dataset(
        first,
        second,
        edits={"1": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=None,
        root_source_dataset=source,
        video_codec="mpeg4",
    )
    state.data_root = tmp_path
    client = TestClient(app)

    datasets = client.get("/api/datasets").json()["datasets"]
    assert [item["path"] for item in datasets] == [str(source)]
    assert datasets[0]["active_workspace_path"] == str(second.resolve())

    opened = client.post("/api/open", json={"path": str(source)}).json()
    assert opened["path"] == str(source.resolve())
    assert opened["active_dataset"] == str(second.resolve())
    assert opened["active_workspace_path"] == str(second.resolve())
    assert opened["show_deleted_segments"] is False
    assert opened["progress"]["edits"] == {}
    assert opened["progress"]["last_export_path"] == str(second.resolve())

    show_deleted = client.post(
        "/api/open",
        json={"path": str(source), "show_deleted_segments": True},
    ).json()
    assert show_deleted["active_dataset"] == str(first.resolve())
    assert show_deleted["show_deleted_segments"] is True
    assert show_deleted["progress"]["edits"]["1"] == {"cuts": [2, 4], "deleted_segments": [1]}


def test_annotation_only_export_does_not_rewrite_video(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    output = tmp_path / "out" / "task" / "lerobot"
    export_v21_dataset(source, output, edits={}, urdf_path=None, video_codec="mpeg4")
    video = output / "videos" / "chunk-000" / "observation.images.head_cam_h" / "episode_000000.mp4"
    before = video.stat().st_mtime_ns
    time.sleep(0.02)

    manifest = export_v21_dataset(
        source,
        output,
        edits={},
        urdf_path=None,
        episode_annotations={"0": {"completed": True, "rating": 8, "notes": ["ok"]}},
        note_labels=["ok"],
        video_codec="mpeg4",
    )

    assert video.stat().st_mtime_ns == before
    assert manifest["episode_annotations"]["0"]["rating"] == 8
    assert manifest["note_labels"] == ["ok"]


def test_middle_delete_uses_non_linear_transition(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot", frames_per_episode=6)
    urdf = make_test_urdf(tmp_path / "test.urdf")
    output = tmp_path / "out" / "task" / "lerobot"

    manifest = export_v21_dataset(
        source,
        output,
        edits={"0": {"cuts": [2, 4], "deleted_segments": [1]}},
        urdf_path=urdf,
        min_transition_frames=1,
        max_transition_frames=10,
        video_codec="mpeg4",
    )

    transition_frames = manifest["episodes"]["0"]["transitions"][0]["transition_frames"]
    assert transition_frames >= 2
    assert manifest["episodes"]["0"]["transitions"][0]["transition_method"] == "cubic_hermite"
    df = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet").to_pandas()
    values = np.asarray([row[0] for row in df["observation.state"].tolist()], dtype=np.float32)
    synthetic = values[2 : 2 + transition_frames]
    linear = np.linspace(values[1], values[2 + transition_frames], transition_frames + 2, dtype=np.float32)[1:-1]
    assert not np.allclose(synthetic, linear)


def test_video_endpoint_serves_episode_video(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    state.dataset = load_v21_dataset(source)
    client = TestClient(app)

    response = client.get(
        "/api/video",
        params={"episode_index": 0, "video_key": "observation.images.head_cam_h"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/")
    assert response.content

    range_response = client.get(
        "/api/video",
        params={"episode_index": 0, "video_key": "observation.images.head_cam_h"},
        headers={"Range": "bytes=0-99"},
    )
    assert range_response.status_code == 206
    assert range_response.headers["accept-ranges"] == "bytes"
    assert range_response.headers["content-range"].startswith("bytes 0-99/")
    assert len(range_response.content) == 100


def test_coverage_analysis_clusters_joint_windows(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    dataset = load_v21_dataset(source)

    auto = compute_coverage_analysis(dataset, window_seconds=0.2, cluster_count="auto")
    assert auto["feature_indexes"] == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    assert auto["auxiliary_indexes"] == [6, 7, 14, 15]
    assert auto["cluster_count"] == len(auto["windows"])
    assert all("cluster_proportions" in window for window in auto["windows"])

    manual = compute_coverage_analysis(dataset, window_seconds=0.2, cluster_count=2)
    assert manual["cluster_count"] == 2
    assert len(manual["clusters"]) == 2
    assert sum(cluster["window_count"] for cluster in manual["clusters"]) == len(manual["windows"])


def test_coverage_analysis_endpoint(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    state.dataset = load_v21_dataset(source)
    state.analysis_job.reset()
    client = TestClient(app)

    response = client.post(
        "/api/analysis/coverage",
        json={"window_seconds": 0.2, "cluster_count": 2},
    )

    assert response.status_code == 200
    assert response.json()["status"] in {"running", "complete"}
    for _ in range(40):
        status = client.get("/api/analysis/coverage/status").json()
        if status["status"] == "complete":
            break
        time.sleep(0.05)
    assert status["status"] == "complete"
    assert status["result"]["cluster_count"] == 2


def test_app_logs_endpoint_returns_entries_after_cursor() -> None:
    client = TestClient(app)
    cursor = state.app_logs.read_after(0)["next_seq"]

    state.app_logs.append("info", "test", "hello log")

    payload = client.get("/api/logs", params={"after": cursor}).json()
    assert payload["logs"][-1]["message"] == "hello log"
    assert payload["logs"][-1]["source"] == "test"
    assert payload["next_seq"] >= payload["logs"][-1]["seq"]


def test_trajectory_preview_missing_urdf_reports_failed_job(tmp_path: Path) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    state.dataset = load_v21_dataset(source)
    state.canonical_source_path = source
    state.active_workspace_path = None
    state.urdf_path = None
    state.trajectory_job.reset()
    client = TestClient(app)

    response = client.post(
        "/api/trajectory/episode/0",
        json={"urdf_path": str(tmp_path / "missing.urdf"), "video_key": "observation.images.head_cam_h"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert "valid URDF" in payload["error"]
    status = client.get("/api/trajectory/status").json()
    assert status["status"] == "failed"
    assert "valid URDF" in status["message"]


def test_trajectory_preview_endpoint_runs_background_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_v21_dataset(tmp_path / "dataset" / "task" / "lerobot")
    urdf = make_test_urdf(tmp_path / "test.urdf")
    state.dataset = load_v21_dataset(source)
    state.canonical_source_path = source
    state.active_workspace_path = None
    state.urdf_path = urdf
    state.trajectory_job.reset()

    def fake_run(context: dict) -> None:
        output_path = Path(context["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state.trajectory_job.update(status="running", message="fake render", progress=0.5)
        time.sleep(0.01)
        output_path.write_bytes(b"fake mp4")
        state.trajectory_job.update(
            status="complete",
            message="trajectory preview ready",
            progress=1.0,
            error=None,
            result=editor_server._trajectory_result(output_path, cached=False),
        )

    monkeypatch.setattr(editor_server, "_run_trajectory_preview", fake_run)
    client = TestClient(app)

    response = client.post(
        "/api/trajectory/episode/0",
        json={"urdf_path": str(urdf), "video_key": "observation.images.head_cam_h"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    for _ in range(40):
        status = client.get("/api/trajectory/status").json()
        if status["status"] == "complete":
            break
        time.sleep(0.05)
    assert status["status"] == "complete"
    assert status["url"].startswith("/api/trajectory/preview/")
    assert status["cached"] is False
