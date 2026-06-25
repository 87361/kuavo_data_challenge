from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    deleted: bool = False

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "deleted": self.deleted,
        }


def normalize_cuts(cuts: list[int] | None, length: int) -> list[int]:
    if length <= 1:
        return []
    return sorted({int(cut) for cut in (cuts or []) if 0 < int(cut) < length})


def build_segments(
    length: int,
    cuts: list[int] | None = None,
    deleted_segments: list[int] | None = None,
) -> list[Segment]:
    deleted = {int(item) for item in (deleted_segments or [])}
    boundaries = [0, *normalize_cuts(cuts, length), int(length)]
    return [
        Segment(index=i, start=boundaries[i], end=boundaries[i + 1], deleted=i in deleted)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]


def kept_ranges(
    length: int,
    cuts: list[int] | None = None,
    deleted_segments: list[int] | None = None,
) -> list[tuple[int, int]]:
    return [(seg.start, seg.end) for seg in build_segments(length, cuts, deleted_segments) if not seg.deleted]


def deleted_ranges(
    length: int,
    cuts: list[int] | None = None,
    deleted_segments: list[int] | None = None,
) -> list[tuple[int, int]]:
    return [(seg.start, seg.end) for seg in build_segments(length, cuts, deleted_segments) if seg.deleted]


def normalize_episode_edit(edit: dict[str, Any] | None, length: int) -> dict[str, list[int]]:
    edit = edit or {}
    cuts = normalize_cuts([int(item) for item in edit.get("cuts", [])], length)
    segments = build_segments(length, cuts)
    max_seg = len(segments) - 1
    deleted = sorted({int(item) for item in edit.get("deleted_segments", []) if 0 <= int(item) <= max_seg})
    if len(deleted) >= len(segments) and segments:
        raise ValueError("an episode must keep at least one segment")
    return {"cuts": cuts, "deleted_segments": deleted}


def transition_frame_count(
    distance_m: float,
    step_m: float = 0.025,
    min_frames: int = 2,
    max_frames: int = 20,
) -> int:
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    raw = int(math.ceil(max(0.0, float(distance_m)) / step_m))
    return max(int(min_frames), min(int(max_frames), raw))


def has_deleted_middle_segment(length: int, edit: dict[str, Any] | None) -> bool:
    normalized = normalize_episode_edit(edit, length)
    segments = build_segments(length, normalized["cuts"], normalized["deleted_segments"])
    for idx, segment in enumerate(segments):
        if segment.deleted and idx > 0 and idx < len(segments) - 1:
            return True
    return False

