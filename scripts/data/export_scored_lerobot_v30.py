#!/usr/bin/env python3
"""Export scored LeRobot v2.1 editor episodes as a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
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


def format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"+ {format_cmd(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def source_data_path(root: Path, info: dict[str, Any], episode_index: int, chunks_size: int) -> Path:
    pattern = info.get("data_path") or LEGACY_DATA_PATH
    return root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def source_video_path(root: Path, info: dict[str, Any], episode_index: int, chunks_size: int, video_key: str) -> Path:
    pattern = info.get("video_path") or LEGACY_VIDEO_PATH
    return root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def output_data_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    return root / LEGACY_DATA_PATH.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def output_video_path(root: Path, episode_index: int, chunks_size: int, video_key: str) -> Path:
    return root / LEGACY_VIDEO_PATH.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )


def replace_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
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
        stats["episode_index"].update(
            {
                "min": [new_ep],
                "max": [new_ep],
                "mean": [float(new_ep)],
                "std": [0.0],
                "count": count,
            }
        )
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


def load_annotations(source_root: Path, preferred: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    candidates: list[Path] = []
    manifest = source_root / "edit_manifest.json"
    progress = source_root / ".lerobot_editor" / "progress.json"
    if preferred in {"auto", "manifest"}:
        candidates.append(manifest)
    if preferred in {"auto", "progress"}:
        candidates.append(progress)

    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path)
        annotations = payload.get("episode_annotations") or {}
        if annotations:
            return annotations, payload, path
    expected = "edit_manifest.json or .lerobot_editor/progress.json" if preferred == "auto" else preferred
    raise FileNotFoundError(f"No episode_annotations found in {expected} under {source_root}")


def rating_is_selected(annotation: dict[str, Any], min_rating: int, max_rating: int, include_incomplete: bool) -> bool:
    rating = annotation.get("rating")
    if not isinstance(rating, int):
        return False
    if rating < min_rating or rating > max_rating:
        return False
    return include_incomplete or annotation.get("completed") is True


def select_episodes(
    annotations: dict[str, Any],
    episode_count: int,
    min_rating: int,
    max_rating: int,
    include_incomplete: bool,
) -> list[int]:
    selected: list[int] = []
    for key, annotation in annotations.items():
        try:
            episode_index = int(key)
        except ValueError:
            continue
        if episode_index < 0 or episode_index >= episode_count:
            continue
        if rating_is_selected(annotation, min_rating, max_rating, include_incomplete):
            selected.append(episode_index)
    return sorted(set(selected))


def rating_counts(annotations: dict[str, Any], *, completed_only: bool) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for annotation in annotations.values():
        if completed_only and annotation.get("completed") is not True:
            continue
        rating = annotation.get("rating")
        if isinstance(rating, int):
            counts[rating] += 1
    return {str(rating): counts[rating] for rating in sorted(counts)}


def derive_repo_id(output_root: Path) -> str:
    if output_root.name == "lerobot":
        return f"{output_root.parent.name}/lerobot"
    return f"{output_root.name}/lerobot"


def build_v21_subset(
    source_root: Path,
    subset_root: Path,
    selected_episodes: list[int],
    annotations: dict[str, Any],
    annotation_source: Path,
    annotation_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    info = read_json(source_root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{source_root} is {info.get('codebase_version')}, expected v2.1")

    source_episodes = {
        int(row["episode_index"]): row
        for row in read_jsonl(source_root / "meta" / "episodes.jsonl")
    }
    source_stats = {
        int(row["episode_index"]): row
        for row in read_jsonl(source_root / "meta" / "episodes_stats.jsonl")
    }
    chunks_size = int(info.get("chunks_size") or 1000)
    video_keys = video_keys_from_info(info)

    episodes_out: list[dict[str, Any]] = []
    stats_out: list[dict[str, Any]] = []
    episode_map: list[dict[str, Any]] = []
    global_frame = 0

    for new_ep, old_ep in enumerate(selected_episodes):
        if old_ep not in source_episodes:
            raise KeyError(f"episode {old_ep} missing from source metadata")
        src_data = source_data_path(source_root, info, old_ep, chunks_size)
        dst_data = output_data_path(subset_root, new_ep, chunks_size)
        if not src_data.exists():
            raise FileNotFoundError(src_data)

        table = pq.read_table(src_data)
        row_count = table.num_rows
        source_length = int(source_episodes[old_ep]["length"])
        if row_count != source_length:
            raise RuntimeError(f"{src_data} has {row_count} rows but metadata length is {source_length}")

        table = replace_column(table, "episode_index", [new_ep] * row_count)
        table = replace_column(table, "index", list(range(global_frame, global_frame + row_count)))
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_data, compression="snappy")

        for video_key in video_keys:
            src_video = source_video_path(source_root, info, old_ep, chunks_size, video_key)
            dst_video = output_video_path(subset_root, new_ep, chunks_size, video_key)
            if not src_video.exists():
                raise FileNotFoundError(src_video)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

        old_episode = source_episodes[old_ep]
        episodes_out.append(
            {
                "episode_index": new_ep,
                "tasks": old_episode.get("tasks", []),
                "length": row_count,
            }
        )
        if old_ep not in source_stats:
            raise KeyError(f"episode {old_ep} missing from source stats")
        stats_out.append(update_episode_stats(source_stats[old_ep], new_ep, global_frame, row_count))

        annotation = annotations.get(str(old_ep), {})
        episode_map.append(
            {
                "new_episode_index": new_ep,
                "source_episode_index": old_ep,
                "length": row_count,
                "rating": annotation.get("rating"),
                "completed": annotation.get("completed"),
                "notes": annotation.get("notes") or [],
            }
        )
        global_frame += row_count
        if (new_ep + 1) % 25 == 0 or new_ep + 1 == len(selected_episodes):
            print(f"v2.1 subset: {new_ep + 1}/{len(selected_episodes)} episodes", flush=True)

    (subset_root / "meta").mkdir(parents=True, exist_ok=True)
    tasks_src = source_root / "meta" / "tasks.jsonl"
    if tasks_src.exists():
        shutil.copy2(tasks_src, subset_root / "meta" / "tasks.jsonl")
    else:
        write_jsonl(subset_root / "meta" / "tasks.jsonl", [])
    write_jsonl(subset_root / "meta" / "episodes.jsonl", episodes_out)
    write_jsonl(subset_root / "meta" / "episodes_stats.jsonl", stats_out)

    subset_info = dict(info)
    subset_info["total_episodes"] = len(selected_episodes)
    subset_info["total_frames"] = global_frame
    subset_info["total_videos"] = len(selected_episodes) * len(video_keys)
    subset_info["total_chunks"] = (len(selected_episodes) + chunks_size - 1) // chunks_size
    subset_info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    subset_info["data_path"] = LEGACY_DATA_PATH
    subset_info["video_path"] = LEGACY_VIDEO_PATH if video_keys else None
    atomic_write_json(subset_root / "meta" / "info.json", subset_info)

    source_manifest = {
        "source_dataset": str(source_root),
        "annotation_source": str(annotation_source),
        "annotation_payload_source_dataset": annotation_payload.get("source_dataset"),
        "root_source_dataset": annotation_payload.get("root_source_dataset"),
        "format": "lerobot_v2.1_scored_subset",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": {
            "min_rating": args.min_rating,
            "max_rating": args.max_rating,
            "include_incomplete": args.include_incomplete,
            "selected_episodes": len(selected_episodes),
            "completed_rating_counts": rating_counts(annotations, completed_only=True),
            "all_rating_counts": rating_counts(annotations, completed_only=False),
        },
        "episode_map": episode_map,
        "note_labels": annotation_payload.get("note_labels") or [],
    }
    atomic_write_json(subset_root / "scored_export_manifest.json", source_manifest)
    return source_manifest


def validate_v30(root: Path, expected_episodes: int) -> dict[str, Any]:
    info = read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(f"{root} is {info.get('codebase_version')}, expected v3.0")
    if int(info.get("total_episodes", -1)) != expected_episodes:
        raise RuntimeError(f"{root} has {info.get('total_episodes')} episodes, expected {expected_episodes}")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter scored LeRobot v2.1 editor episodes and convert the subset to LeRobot v3.0."
    )
    parser.add_argument("--source-root", required=True, help="Source LeRobot v2.1 dataset root.")
    parser.add_argument("--output-root", required=True, help="Final LeRobot v3.0 dataset root.")
    parser.add_argument("--min-rating", type=int, default=8, help="Minimum score to export, inclusive.")
    parser.add_argument("--max-rating", type=int, default=10, help="Maximum score to export, inclusive.")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include episodes that have a score but are not marked completed.",
    )
    parser.add_argument(
        "--annotation-source",
        choices=["auto", "manifest", "progress"],
        default="auto",
        help="Where to read episode_annotations from.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Temporary repo id for the v2.1->v3.0 converter. Defaults to '<output parent>/lerobot'.",
    )
    parser.add_argument("--work-dir", default=None, help="Scratch directory. Defaults next to --output-root.")
    parser.add_argument("--repo-root", default=str(repo_root_from_script()), help="Repository root.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for v3.0 conversion.")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output-root if it already exists.")
    parser.add_argument("--keep-work", action="store_true", help="Keep the scratch v2.1/v3.0 conversion tree.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.min_rating <= args.max_rating <= 10:
        raise ValueError("--min-rating/--max-rating must be within 1..10 and min <= max")

    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    repo_id = args.repo_id or derive_repo_id(output_root)
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_root.parent / f".{output_root.parent.name}_scored_v30_build"
    )

    info = read_json(source_root / "meta" / "info.json")
    total_episodes = int(info.get("total_episodes") or 0)
    annotations, annotation_payload, annotation_path = load_annotations(source_root, args.annotation_source)
    selected = select_episodes(
        annotations,
        total_episodes,
        args.min_rating,
        args.max_rating,
        args.include_incomplete,
    )
    print(f"annotation source: {annotation_path}")
    print(f"completed rating counts: {rating_counts(annotations, completed_only=True)}")
    print(f"all rating counts: {rating_counts(annotations, completed_only=False)}")
    print(
        f"selected episodes: {len(selected)} "
        f"(rating {args.min_rating}-{args.max_rating}, include_incomplete={args.include_incomplete})"
    )
    if not selected:
        raise RuntimeError("No episodes matched the scoring filter")
    if args.dry_run:
        print(f"first selected episodes: {selected[:20]}")
        print(f"last selected episodes: {selected[-20:]}")
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        safe_remove_tree(output_root, output_root.parent)
    safe_remove_tree(work_dir, work_dir.parent)

    subset_root = work_dir / "v21" / repo_id
    subset_root.parent.mkdir(parents=True, exist_ok=True)
    print(f"building v2.1 scored subset: {subset_root}")
    manifest = build_v21_subset(
        source_root=source_root,
        subset_root=subset_root,
        selected_episodes=selected,
        annotations=annotations,
        annotation_source=annotation_path,
        annotation_payload=annotation_payload,
        args=args,
    )

    converter = repo_root / "third_party" / "lerobot" / "src" / "lerobot" / "datasets" / "v30" / "convert_dataset_v21_to_v30.py"
    if not converter.exists():
        raise FileNotFoundError(converter)
    env = os.environ.copy()
    lerobot_src = repo_root / "third_party" / "lerobot" / "src"
    env["PYTHONPATH"] = f"{lerobot_src}:{repo_root}:{env.get('PYTHONPATH', '')}"
    env["HYDRA_FULL_ERROR"] = "1"
    cmd = [
        args.python,
        str(converter),
        "--repo-id",
        repo_id,
        "--root",
        str(work_dir / "v21"),
        "--push-to-hub=false",
        "--force-conversion",
    ]
    run_cmd(cmd, cwd=repo_root, env=env)

    converted_root = work_dir / "v21" / repo_id
    validate_v30(converted_root, len(selected))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(converted_root), str(output_root))

    final_info = validate_v30(output_root, len(selected))
    manifest["output_dataset"] = str(output_root)
    manifest["output_format"] = "lerobot_v3.0"
    manifest["total_frames"] = final_info.get("total_frames")
    atomic_write_json(output_root / "scored_export_manifest.json", manifest)

    if not args.keep_work:
        safe_remove_tree(work_dir, work_dir.parent)

    print(f"v3.0 scored dataset ready: {output_root}")
    print(f"episodes={final_info.get('total_episodes')} frames={final_info.get('total_frames')} fps={final_info.get('fps')}")


if __name__ == "__main__":
    main()
