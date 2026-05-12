"""Clip sampling helpers for center-frame VOD experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def clip_offsets(length: int, *, causal: bool = False) -> list[int]:
    if length < 1:
        raise ValueError("length must be >= 1.")
    if length == 1:
        return [0]
    if causal:
        return list(range(-(length - 1), 1))
    if length % 2 == 0:
        raise ValueError("offline center-frame clips require an odd length.")
    half = length // 2
    return list(range(-half, half + 1))


def clip_target_index(length: int, *, causal: bool = False, target: str = "center") -> int:
    if length < 1:
        raise ValueError("length must be >= 1.")
    if target == "current" or causal:
        return length - 1
    if target != "center":
        raise ValueError(f"Unsupported clip target: {target!r}")
    if length % 2 == 0:
        raise ValueError("center target requires an odd clip length.")
    return length // 2


def sample_clip_indices(
    frame_index: int,
    num_frames: int,
    *,
    length: int = 5,
    causal: bool = False,
    boundary: str = "clamp",
) -> list[int]:
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1.")

    indices = [frame_index + offset for offset in clip_offsets(length, causal=causal)]
    if boundary == "clamp":
        return [min(max(index, 0), num_frames - 1) for index in indices]
    if boundary == "drop":
        if any(index < 0 or index >= num_frames for index in indices):
            return []
        return indices
    raise ValueError(f"Unsupported boundary mode: {boundary}")


def group_records_by_video(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("video_id", ""))].append(record)
    for video_records in grouped.values():
        video_records.sort(key=lambda item: int(item.get("frame_index", 0)))
    return dict(grouped)


def sample_record_clip(
    video_records: list[dict[str, Any]],
    target_position: int,
    *,
    length: int = 5,
    causal: bool = False,
    boundary: str = "clamp",
) -> list[dict[str, Any]]:
    indices = sample_clip_indices(
        target_position,
        len(video_records),
        length=length,
        causal=causal,
        boundary=boundary,
    )
    return [video_records[index] for index in indices]
