"""Shared helpers for derived detection dataset exports."""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from src.shared.config import PROJECT_ROOT


IMAGE_MODES = {"none", "symlink", "hardlink", "copy"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def resolve_project_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def category_mapping(records: Iterable[dict[str, Any]]) -> tuple[dict[int, int], list[str], list[int]]:
    category_names: dict[int, str] = {}
    for record in records:
        for obj in record.get("objects", []):
            category_id = int(obj["category_id"])
            category_names.setdefault(category_id, str(obj.get("label", category_id)))
    source_category_ids = sorted(category_names)
    return (
        {category_id: index for index, category_id in enumerate(source_category_ids)},
        [category_names[category_id] for category_id in source_category_ids],
        source_category_ids,
    )


def clip_value(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def clamped_xyxy(obj: dict[str, Any], width: float, height: float) -> list[float] | None:
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = [float(value) for value in obj["bbox_xyxy"]]
    x1 = clip_value(x1, 0.0, width)
    x2 = clip_value(x2, 0.0, width)
    y1 = clip_value(y1, 0.0, height)
    y2 = clip_value(y2, 0.0, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def clamped_xywh(obj: dict[str, Any], width: float, height: float) -> list[float] | None:
    box = clamped_xyxy(obj, width, height)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def record_stem(record: dict[str, Any], source_image: Path) -> Path:
    video_id = str(record["video_id"])
    frame_index = int(record.get("frame_index", 0))
    return Path(video_id) / f"{frame_index:06d}_{source_image.stem}"


def relative_image_path(record: dict[str, Any], source_image: Path) -> Path:
    return record_stem(record, source_image).with_suffix(source_image.suffix)


def materialize_image(source: Path, target: Path, image_mode: str) -> str:
    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {sorted(IMAGE_MODES)}, got {image_mode!r}.")
    if image_mode == "none":
        return "skipped"
    if not source.exists():
        return "missing_source"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    if image_mode == "symlink":
        os.symlink(source, target)
        return "symlinked"
    if image_mode == "hardlink":
        os.link(source, target)
        return "hardlinked"
    if image_mode == "copy":
        shutil.copy2(source, target)
        return "copied"
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def ordered_counter_dict(counter: Any) -> dict[str, int]:
    return dict(OrderedDict(sorted(counter.items(), key=lambda item: str(item[0]))))
