#!/usr/bin/env python3
"""Prepare Kuavo Tianchi real_suzhou_3.0 datasets.

The script is intentionally resumable:

- Manifest and batch layout are written once under the state directory.
- Each batch gets a v2.1 conversion marker only after output validation.
- The final v2.1 dataset is built in a work directory, then moved into place.
- The v3.0 dataset is produced from the final v2.1 dataset in a work directory.

Raw bags are downloaded to the local machine first. By default they are deleted
after their batch has been converted to save disk space; pass --keep-raw to keep
the complete raw dataset in --raw-root.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DATASET_ID = "lejurobot/LET-Tianchi-Dataset"
TASK_CONFIGS = {
    "task1": {
        "dataset_subdir": "real_suzhou_3.0/task1_zhuomian",
        "task_name": "task1_zhuomian",
        "description": "Task1 Desktop Parts Pick And Place",
    },
    "task2": {
        "dataset_subdir": "real_suzhou_3.0/task2_chengzhong",
        "task_name": "task2_chengzhong",
        "description": "Task2 Industrial Parts Weighing",
    },
    "task3": {
        "dataset_subdir": "real_suzhou_3.0/task3_dajian",
        "task_name": "task3_dajian",
        "description": "Task3 Large Automotive Part Loading",
    },
}
REMOTE_DATA = "/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_remove_tree(path: Path, allowed_root: Path) -> None:
    if not path.exists():
        return
    if not is_relative_to(path, allowed_root):
        raise RuntimeError(f"Refusing to remove path outside {allowed_root}: {path}")
    shutil.rmtree(path)


def safe_unlink(path: Path, allowed_root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    check_path = path.parent if path.is_symlink() else path
    if not is_relative_to(check_path, allowed_root):
        raise RuntimeError(f"Refusing to remove path outside {allowed_root}: {path}")
    path.unlink()


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    retries: int = 1,
    retry_sleep_s: int = 10,
) -> None:
    for attempt in range(1, retries + 1):
        print(f"+ {format_cmd(cmd)}", flush=True)
        try:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(f"Command failed, retrying in {retry_sleep_s}s ({attempt}/{retries})", flush=True)
            time.sleep(retry_sleep_s)


def python_cmd(conda_env: str) -> list[str]:
    if conda_env in {"", "current", "none"}:
        return [sys.executable]
    return ["conda", "run", "-n", conda_env, "python"]


def task_config(args: argparse.Namespace) -> dict[str, str]:
    return TASK_CONFIGS[args.task]


def dataset_subdir(args: argparse.Namespace) -> str:
    return task_config(args)["dataset_subdir"]


def task_name(args: argparse.Namespace) -> str:
    return task_config(args)["task_name"]


def task_description(args: argparse.Namespace) -> str:
    return task_config(args)["description"]


def output_tag(args: argparse.Namespace) -> str:
    if args.output_tag:
        return args.output_tag
    if args.limit:
        return f"{args.task}_{args.limit}"
    if args.expected_episodes == 1000:
        return f"{args.task}_full"
    return f"{args.task}_{args.expected_episodes}"


def state_dir(args: argparse.Namespace) -> Path:
    return args.state_dir or (args.data_root / f".{output_tag(args)}_state")


def manifest_path(args: argparse.Namespace) -> Path:
    return state_dir(args) / "manifest.jsonl"


def batches_path(args: argparse.Namespace) -> Path:
    return state_dir(args) / "batches.json"


def batch_dir(args: argparse.Namespace, batch_index: int) -> Path:
    return state_dir(args) / "batches" / f"batch_{batch_index:04d}"


def batch_raw_subset_dir(args: argparse.Namespace, batch_index: int) -> Path:
    return batch_dir(args, batch_index) / "raw_subset" / dataset_subdir(args)


def batch_v21_root(args: argparse.Namespace, batch_index: int) -> Path:
    return batch_dir(args, batch_index) / "lerobot_v21" / task_name(args) / "lerobot"


def batch_v21_parent(args: argparse.Namespace, batch_index: int) -> Path:
    return batch_v21_root(args, batch_index).parent


def v21_parent(args: argparse.Namespace) -> Path:
    return args.data_root / f"lerobot_v21_{output_tag(args)}"


def v21_root(args: argparse.Namespace) -> Path:
    return v21_parent(args) / task_name(args) / "lerobot"


def v30_parent(args: argparse.Namespace) -> Path:
    return args.data_root / f"lerobot_v30_{output_tag(args)}"


def v30_root(args: argparse.Namespace) -> Path:
    return v30_parent(args) / task_name(args) / "lerobot"


def raw_file_path(args: argparse.Namespace, dataset_path: str) -> Path:
    return args.raw_root / dataset_path


def fetch_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise RuntimeError("modelscope is required. Install with: python -m pip install -U modelscope") from exc

    api = HubApi()
    page = 1
    page_size = 1000
    rows: list[dict[str, Any]] = []
    while True:
        files = api.get_dataset_files(
            DATASET_ID,
            root_path=dataset_subdir(args),
            recursive=True,
            page_number=page,
            page_size=page_size,
        )
        for item in files:
            path = item.get("Path", "")
            if item.get("Type") == "blob" and path.endswith(".bag"):
                rows.append(
                    {
                        "path": path,
                        "name": item.get("Name") or Path(path).name,
                        "size": int(item.get("Size") or 0),
                        "sha256": item.get("Sha256") or "",
                    }
                )
        if len(files) < page_size:
            break
        page += 1

    rows.sort(key=lambda row: row["path"])
    write_jsonl(manifest_path(args), rows)
    atomic_write_json(
        state_dir(args) / "manifest.meta.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_subdir": dataset_subdir(args),
            "total_bags": len(rows),
            "total_bytes": sum(row["size"] for row in rows),
            "generated_at": int(time.time()),
        },
    )
    return rows


def load_or_fetch_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = manifest_path(args)
    if path.exists() and not args.refresh_manifest:
        return read_jsonl(path)
    return fetch_manifest(args)


def build_batches(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_bytes = int(args.batch_max_gib * (1024**3))
    max_items = args.batch_size or len(manifest)
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_size = 0

    for row in manifest:
        size = int(row["size"])
        would_exceed_size = current and current_size + size > max_bytes
        would_exceed_count = current and len(current) >= max_items
        if would_exceed_size or would_exceed_count:
            batches.append(
                {
                    "index": len(batches),
                    "files": current,
                    "total_bytes": current_size,
                }
            )
            current = []
            current_size = 0
        current.append(row)
        current_size += size

    if current:
        batches.append({"index": len(batches), "files": current, "total_bytes": current_size})

    atomic_write_json(
        batches_path(args),
        {
            "dataset_id": DATASET_ID,
            "dataset_subdir": dataset_subdir(args),
            "batch_max_gib": args.batch_max_gib,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "total_bags": len(manifest),
            "total_bytes": sum(row["size"] for row in manifest),
            "batches": batches,
        },
    )
    return batches


def load_or_build_batches(args: argparse.Namespace, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = batches_path(args)
    if path.exists() and not args.rebuild_batches:
        data = read_json(path)
        if (
            data.get("dataset_subdir") == dataset_subdir(args)
            and data.get("batch_max_gib") == args.batch_max_gib
            and data.get("batch_size") == args.batch_size
            and data.get("limit") == args.limit
            and data.get("total_bags") == len(manifest)
        ):
            return data["batches"]
    return build_batches(args, manifest)


def print_plan(manifest: list[dict[str, Any]], batches: list[dict[str, Any]], args: argparse.Namespace) -> None:
    total_gib = sum(row["size"] for row in manifest) / (1024**3)
    print(f"dataset: {DATASET_ID}/{dataset_subdir(args)}")
    print(f"bags: {len(manifest)}")
    print(f"raw size: {total_gib:.2f} GiB")
    print(f"data root: {args.data_root}")
    print(f"raw root: {args.raw_root}")
    print(f"state dir: {state_dir(args)}")
    print(f"batches: {len(batches)}")
    for batch in batches[:8]:
        print(
            f"  batch {batch['index']:04d}: "
            f"{len(batch['files'])} bags, {batch['total_bytes'] / (1024**3):.2f} GiB"
        )
    if len(batches) > 8:
        print(f"  ... {len(batches) - 8} more batches")


def raw_batch_complete(args: argparse.Namespace, batch: dict[str, Any]) -> bool:
    for row in batch["files"]:
        path = raw_file_path(args, row["path"])
        if not path.exists() or path.stat().st_size != int(row["size"]):
            return False
    return True


def download_batch(args: argparse.Namespace, batch: dict[str, Any]) -> None:
    if raw_batch_complete(args, batch):
        print(f"batch {batch['index']:04d}: raw already complete")
        return

    args.raw_root.mkdir(parents=True, exist_ok=True)
    files = [row["path"] for row in batch["files"]]
    chunk_size = args.download_file_args
    for start in range(0, len(files), chunk_size):
        part = files[start : start + chunk_size]
        cmd = [
            "modelscope",
            "download",
            "--dataset",
            DATASET_ID,
            "--local_dir",
            str(args.raw_root),
            "--max-workers",
            str(args.max_workers),
            *part,
        ]
        run_cmd(cmd)

    if not raw_batch_complete(args, batch):
        missing = [
            row["path"]
            for row in batch["files"]
            if not raw_file_path(args, row["path"]).exists()
            or raw_file_path(args, row["path"]).stat().st_size != int(row["size"])
        ]
        raise RuntimeError(f"batch {batch['index']:04d}: raw files incomplete: {missing[:5]}")


def link_batch_raw_subset(args: argparse.Namespace, batch: dict[str, Any]) -> Path:
    subset = batch_raw_subset_dir(args, batch["index"])
    subset.mkdir(parents=True, exist_ok=True)
    for row in batch["files"]:
        src = raw_file_path(args, row["path"]).resolve()
        dst = subset / Path(row["path"]).name
        if dst.is_symlink() and dst.resolve() == src:
            continue
        safe_unlink(dst, state_dir(args))
        dst.symlink_to(src)
    return subset


def v21_dataset_valid(root: Path, expected_episodes: int | None = None) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        info = read_json(info_path)
    except Exception:
        return False
    if info.get("codebase_version") != "v2.1":
        return False
    if expected_episodes is not None and int(info.get("total_episodes", -1)) != expected_episodes:
        return False
    return True


def v30_dataset_valid(root: Path, expected_episodes: int | None = None) -> bool:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    try:
        info = read_json(info_path)
    except Exception:
        return False
    if info.get("codebase_version") != "v3.0":
        return False
    if expected_episodes is not None and int(info.get("total_episodes", -1)) != expected_episodes:
        return False
    return True


def convert_batch_v21(args: argparse.Namespace, batch: dict[str, Any]) -> None:
    done_marker = batch_dir(args, batch["index"]) / "v21.done.json"
    expected = len(batch["files"])
    if done_marker.exists() and v21_dataset_valid(batch_v21_root(args, batch["index"]), expected):
        print(f"batch {batch['index']:04d}: v2.1 already converted")
        return

    if not raw_batch_complete(args, batch):
        download_batch(args, batch)
    raw_subset = link_batch_raw_subset(args, batch)

    safe_remove_tree(batch_v21_parent(args, batch["index"]), state_dir(args))
    run_dir = batch_dir(args, batch["index"]) / "run_v21"
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_root = args.repo_root
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    env["HYDRA_FULL_ERROR"] = "1"
    cmd = [
        *python_cmd(args.v21_env),
        str(repo_root / "kuavo_data" / "CvtRosbag2Lerobot.py"),
        f"rosbag.rosbag_dir={raw_subset}",
        f"rosbag.lerobot_dir={batch_v21_parent(args, batch['index'])}",
        f"rosbag.chunk_size={args.chunk_size}",
        "dataset.eef_type=leju_claw",
        "dataset.which_arm=both",
        "dataset.use_depth=false",
        "dataset.resize.width=848",
        "dataset.resize.height=480",
        "dataset.sample_drop=10",
        f"dataset.task_description={task_description(args)}",
    ]
    run_cmd(cmd, cwd=run_dir, env=env)

    if not v21_dataset_valid(batch_v21_root(args, batch["index"]), expected):
        info_path = batch_v21_root(args, batch["index"]) / "meta" / "info.json"
        got = read_json(info_path).get("total_episodes") if info_path.exists() else "missing"
        raise RuntimeError(f"batch {batch['index']:04d}: expected {expected} episodes, got {got}")

    atomic_write_json(
        done_marker,
        {
            "batch_index": batch["index"],
            "total_bags": expected,
            "total_bytes": batch["total_bytes"],
            "converted_at": int(time.time()),
        },
    )
    print(f"batch {batch['index']:04d}: v2.1 conversion done")

    if not args.keep_raw:
        remove_batch_raw(args, batch)


def remove_batch_raw(args: argparse.Namespace, batch: dict[str, Any]) -> None:
    safe_remove_tree(batch_dir(args, batch["index"]) / "raw_subset", state_dir(args))
    for row in batch["files"]:
        path = raw_file_path(args, row["path"])
        if path.exists():
            safe_unlink(path, args.raw_root)


def replace_column(table: Any, name: str, values: list[int]) -> Any:
    import pyarrow as pa

    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    field = table.schema.field(idx)
    return table.set_column(idx, field, pa.array(values, type=field.type))


def update_episode_stats(stats_row: dict[str, Any], new_ep: int, start_index: int, length: int) -> dict[str, Any]:
    row = json.loads(json.dumps(stats_row))
    row["episode_index"] = new_ep
    stats = row.get("stats", {})
    if "episode_index" in stats:
        count = stats["episode_index"].get("count", [length])
        stats["episode_index"].update({"min": [new_ep], "max": [new_ep], "mean": [float(new_ep)], "std": [0.0], "count": count})
    if "index" in stats:
        end_index = start_index + length - 1
        count = stats["index"].get("count", [length])
        std = stats["index"].get("std", [0.0])
        stats["index"].update(
            {
                "min": [start_index],
                "max": [end_index],
                "mean": [(start_index + end_index) / 2.0],
                "std": std,
                "count": count,
            }
        )
    return row


def v21_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]


def merge_v21(args: argparse.Namespace, batches: list[dict[str, Any]], total_episodes: int) -> None:
    if v21_dataset_valid(v21_root(args), total_episodes):
        print("full v2.1 dataset already exists")
        return
    if v21_parent(args).exists():
        raise RuntimeError(f"Target exists but is not a complete v2.1 dataset: {v21_parent(args)}")

    for batch in batches:
        if not v21_dataset_valid(batch_v21_root(args, batch["index"]), len(batch["files"])):
            raise RuntimeError(f"batch {batch['index']:04d} is not converted yet")

    work_parent = state_dir(args) / "merge_v21_work" / f"lerobot_v21_{output_tag(args)}"
    safe_remove_tree(work_parent.parent, state_dir(args))
    root = work_parent / task_name(args) / "lerobot"
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    first_info = read_json(batch_v21_root(args, batches[0]["index"]) / "meta" / "info.json")
    chunks_size = int(first_info.get("chunks_size") or 1000)
    video_keys = v21_video_keys(first_info)
    global_ep = 0
    global_frame = 0

    episodes_out = root / "meta" / "episodes.jsonl"
    stats_out = root / "meta" / "episodes_stats.jsonl"
    with episodes_out.open("w") as episodes_f, stats_out.open("w") as stats_f:
        for batch in batches:
            src_root = batch_v21_root(args, batch["index"])
            episodes = sorted(read_jsonl(src_root / "meta" / "episodes.jsonl"), key=lambda row: row["episode_index"])
            stats_by_ep = {
                row["episode_index"]: row for row in read_jsonl(src_root / "meta" / "episodes_stats.jsonl")
            }
            for episode in episodes:
                old_ep = int(episode["episode_index"])
                length = int(episode["length"])
                old_chunk = old_ep // chunks_size
                new_chunk = global_ep // chunks_size

                src_parquet = src_root / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_ep:06d}.parquet"
                dst_parquet = root / "data" / f"chunk-{new_chunk:03d}" / f"episode_{global_ep:06d}.parquet"
                dst_parquet.parent.mkdir(parents=True, exist_ok=True)
                table = pq.read_table(src_parquet)
                row_count = table.num_rows
                if row_count != length:
                    raise RuntimeError(f"{src_parquet} has {row_count} rows but metadata length is {length}")
                table = replace_column(table, "episode_index", [global_ep] * row_count)
                table = replace_column(table, "index", list(range(global_frame, global_frame + row_count)))
                pq.write_table(table, dst_parquet, compression="snappy")

                for video_key in video_keys:
                    src_video = (
                        src_root
                        / "videos"
                        / f"chunk-{old_chunk:03d}"
                        / video_key
                        / f"episode_{old_ep:06d}.mp4"
                    )
                    dst_video = (
                        root
                        / "videos"
                        / f"chunk-{new_chunk:03d}"
                        / video_key
                        / f"episode_{global_ep:06d}.mp4"
                    )
                    dst_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_video, dst_video)

                episode_out = dict(episode)
                episode_out["episode_index"] = global_ep
                episodes_f.write(json.dumps(episode_out, ensure_ascii=False, separators=(",", ":")) + "\n")

                stats_row = update_episode_stats(stats_by_ep[old_ep], global_ep, global_frame, row_count)
                stats_f.write(json.dumps(stats_row, ensure_ascii=False, separators=(",", ":")) + "\n")

                global_ep += 1
                global_frame += row_count

    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task_description(args)}])
    info = dict(first_info)
    info["total_episodes"] = global_ep
    info["total_frames"] = global_frame
    info["total_tasks"] = 1
    info["total_videos"] = global_ep * len(video_keys)
    info["total_chunks"] = (global_ep + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{global_ep}"}
    atomic_write_json(root / "meta" / "info.json", info)

    if not v21_dataset_valid(root, total_episodes):
        raise RuntimeError("merged v2.1 dataset failed validation")
    shutil.move(str(work_parent), str(v21_parent(args)))
    print(f"full v2.1 dataset ready: {v21_root(args)}")


def convert_full_v30(args: argparse.Namespace, total_episodes: int) -> None:
    if v30_dataset_valid(v30_root(args), total_episodes):
        print("full v3.0 dataset already exists")
        return
    if v30_parent(args).exists():
        raise RuntimeError(f"Target exists but is not a complete v3.0 dataset: {v30_parent(args)}")
    if not v21_dataset_valid(v21_root(args), total_episodes):
        raise RuntimeError("full v2.1 dataset is missing; cannot build v3.0")

    work_parent = state_dir(args) / "v30_build" / f"lerobot_v30_{output_tag(args)}"
    safe_remove_tree(work_parent.parent, state_dir(args))
    work_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(v21_parent(args) / task_name(args), work_parent / task_name(args))

    repo_root = args.repo_root
    env = os.environ.copy()
    lerobot_src = repo_root / "third_party" / "lerobot" / "src"
    env["PYTHONPATH"] = f"{lerobot_src}:{repo_root}:{env.get('PYTHONPATH', '')}"
    env["HYDRA_FULL_ERROR"] = "1"
    cmd = [
        *python_cmd(args.v30_env),
        str(repo_root / "third_party" / "lerobot" / "src" / "lerobot" / "datasets" / "v30" / "convert_dataset_v21_to_v30.py"),
        "--repo-id",
        f"{task_name(args)}/lerobot",
        "--root",
        str(work_parent),
        "--push-to-hub=false",
        "--force-conversion",
    ]
    run_cmd(cmd, cwd=repo_root, env=env)

    old_copy = work_parent / task_name(args) / "lerobot_old"
    safe_remove_tree(old_copy, work_parent)
    if not v30_dataset_valid(work_parent / task_name(args) / "lerobot", total_episodes):
        raise RuntimeError("v3.0 conversion failed validation")
    shutil.move(str(work_parent), str(v30_parent(args)))
    print(f"full v3.0 dataset ready: {v30_root(args)}")


def print_dataset_info(label: str, root: Path) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print(f"{label}: missing ({root})")
        return
    info = read_json(info_path)
    print(
        f"{label}: version={info.get('codebase_version')} "
        f"episodes={info.get('total_episodes')} frames={info.get('total_frames')} fps={info.get('fps')}"
    )


def parse_remote(remote: str) -> tuple[str, str]:
    if ":" not in remote:
        raise ValueError("remote must look like host:/absolute/path")
    host, path = remote.split(":", 1)
    if not host or not path.startswith("/"):
        raise ValueError("remote must look like host:/absolute/path")
    return host, path.rstrip("/")


def upload_dir(local: Path, remote: str, retries: int) -> None:
    host, remote_path = parse_remote(remote)
    run_cmd(["ssh", host, "mkdir", "-p", remote_path], retries=retries)
    run_cmd(
        [
            "rsync",
            "-a",
            "--partial",
            "--info=progress2",
            f"{local}/",
            f"{host}:{remote_path}/",
        ],
        retries=retries,
    )


def upload_outputs(args: argparse.Namespace) -> None:
    total_episodes = args.expected_episodes
    if not v21_dataset_valid(v21_root(args), total_episodes):
        raise RuntimeError("v2.1 output is missing or incomplete")
    if not v30_dataset_valid(v30_root(args), total_episodes):
        raise RuntimeError("v3.0 output is missing or incomplete")

    remote_base = args.remote.rstrip("/")
    upload_dir(v21_parent(args), f"{remote_base}/lerobot_v21_{output_tag(args)}", args.ssh_retries)
    upload_dir(v30_parent(args), f"{remote_base}/lerobot_v30_{output_tag(args)}", args.ssh_retries)
    if args.upload_raw:
        upload_dir(args.raw_root / dataset_subdir(args), f"{remote_base}/raw/{dataset_subdir(args)}", args.ssh_retries)


def run_pipeline(args: argparse.Namespace) -> None:
    manifest = load_or_fetch_manifest(args)
    if args.limit:
        manifest = manifest[: args.limit]
    batches = load_or_build_batches(args, manifest)
    print_plan(manifest, batches, args)
    if len(manifest) != args.expected_episodes:
        raise RuntimeError(f"Expected {args.expected_episodes} bags, manifest has {len(manifest)}")

    if args.stop_after == "plan":
        return

    for batch in batches:
        if args.max_batches is not None and batch["index"] >= args.max_batches:
            break
        batch_done = (
            (batch_dir(args, batch["index"]) / "v21.done.json").exists()
            and v21_dataset_valid(batch_v21_root(args, batch["index"]), len(batch["files"]))
        )
        if args.stop_after == "download" or args.keep_raw or not batch_done:
            download_batch(args, batch)
        if args.stop_after == "download":
            continue
        convert_batch_v21(args, batch)

    if args.max_batches is not None:
        print("--max-batches was set; skipping final merge/conversion")
        return
    if args.stop_after in {"download", "convert"}:
        return

    merge_v21(args, batches, args.expected_episodes)
    if args.stop_after == "merge-v21":
        return

    if not args.skip_v30:
        convert_full_v30(args, args.expected_episodes)
    if args.stop_after == "convert-v30":
        return

    if args.upload:
        upload_outputs(args)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/kuavo_tianchi"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="task1")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N bags from the sorted manifest")
    parser.add_argument("--output-tag", default=None, help="Suffix used in lerobot_v21/v30 output directory names")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--rebuild-batches", action="store_true")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_root = args.data_root.expanduser().resolve()
    args.raw_root = (args.raw_root or (args.data_root / "raw")).expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.expected_episodes is None:
        args.expected_episodes = args.limit or 1000
    if args.state_dir is not None:
        args.state_dir = args.state_dir.expanduser().resolve()
    args.data_root.mkdir(parents=True, exist_ok=True)
    state_dir(args).mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Fetch/list the manifest and show the batch layout")
    add_common_args(plan)
    plan.add_argument("--batch-max-gib", type=float, default=80.0)
    plan.add_argument("--batch-size", type=int, default=0)

    run = subparsers.add_parser("run", help="Download, convert, merge, optionally upload")
    add_common_args(run)
    run.add_argument("--batch-max-gib", type=float, default=80.0)
    run.add_argument("--batch-size", type=int, default=0)
    run.add_argument("--download-file-args", type=int, default=80, help="Number of file paths per modelscope CLI call")
    run.add_argument("--max-workers", type=int, default=8)
    run.add_argument("--v21-env", default="kdc")
    run.add_argument("--v30-env", default="kdc_icra")
    run.add_argument("--chunk-size", type=int, default=100)
    run.add_argument("--keep-raw", action="store_true", help="Keep raw bags in --raw-root after conversion")
    run.add_argument("--skip-v30", action="store_true")
    run.add_argument(
        "--stop-after",
        choices=["plan", "download", "convert", "merge-v21", "convert-v30", "upload"],
        default="upload",
    )
    run.add_argument("--max-batches", type=int, default=None, help="Debug helper; skips final merge/conversion")
    run.add_argument("--upload", action="store_true")
    run.add_argument("--remote", default=f"pi1022:{REMOTE_DATA}")
    run.add_argument("--upload-raw", action="store_true")
    run.add_argument("--ssh-retries", type=int, default=3)

    verify = subparsers.add_parser("verify", help="Print local v2.1/v3.0 dataset metadata")
    add_common_args(verify)

    upload = subparsers.add_parser("upload", help="Upload existing v2.1/v3.0 outputs")
    add_common_args(upload)
    upload.add_argument("--remote", default=f"pi1022:{REMOTE_DATA}")
    upload.add_argument("--upload-raw", action="store_true")
    upload.add_argument("--ssh-retries", type=int, default=3)

    args = normalize_args(parser.parse_args())
    if args.command == "plan":
        manifest = load_or_fetch_manifest(args)
        if args.limit:
            manifest = manifest[: args.limit]
        batches = load_or_build_batches(args, manifest)
        print_plan(manifest, batches, args)
    elif args.command == "run":
        run_pipeline(args)
    elif args.command == "verify":
        print_dataset_info("v2.1", v21_root(args))
        print_dataset_info("v3.0", v30_root(args))
    elif args.command == "upload":
        upload_outputs(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
