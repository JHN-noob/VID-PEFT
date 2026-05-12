"""Export converted YouTube-VIS manifests to Ultralytics YOLO detection layout."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.shared.config import PROJECT_ROOT, write_json
from src.shared.detection_export import (
    IMAGE_MODES,
    category_mapping,
    clamped_xyxy,
    materialize_image,
    read_jsonl,
    record_stem,
    resolve_project_path,
)


def _category_mapping(records: Iterable[dict[str, Any]]) -> tuple[dict[int, int], list[str]]:
    category_to_index, class_names, _ = category_mapping(records)
    return category_to_index, class_names


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _to_yolo_line(obj: dict[str, Any], width: float, height: float, category_to_index: dict[int, int]) -> str | None:
    box = clamped_xyxy(obj, width, height)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    class_index = category_to_index[int(obj["category_id"])]
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    return (
        f"{class_index} "
        f"{_clip(x_center):.6f} {_clip(y_center):.6f} "
        f"{_clip(box_w / width):.6f} {_clip(box_h / height):.6f}"
    )


def _write_yolo_data_yaml(path: Path, dataset_root: Path, class_names: list[str]) -> None:
    def quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    lines = [
        f"path: {quote(str(dataset_root))}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {index}: {quote(name)}" for index, name in enumerate(class_names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_split(
    records: list[dict[str, Any]],
    *,
    yolo_split: str,
    dataset_root: Path,
    category_to_index: dict[int, int],
    image_mode: str,
) -> dict[str, Any]:
    image_dir = dataset_root / "images" / yolo_split
    label_dir = dataset_root / "labels" / yolo_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    materialized = Counter()
    object_count = 0
    skipped_boxes = 0
    for record in records:
        source_image = resolve_project_path(record["image_path"])
        stem = record_stem(record, source_image)
        target_image = image_dir / stem.with_suffix(source_image.suffix)
        target_label = label_dir / stem.with_suffix(".txt")
        target_label.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        width = float(record.get("width") or 0)
        height = float(record.get("height") or 0)
        for obj in record.get("objects", []):
            line = _to_yolo_line(obj, width, height, category_to_index)
            if line is None:
                skipped_boxes += 1
                continue
            lines.append(line)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        object_count += len(lines)
        materialized[materialize_image(source_image, target_image, image_mode)] += 1

    return {
        "frames": len(records),
        "objects": object_count,
        "skipped_boxes": skipped_boxes,
        "image_mode_counts": dict(materialized),
    }


def export_yolo_detection_dataset(
    train_manifest: str | Path = PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_train.jsonl",
    val_manifest: str | Path = PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_dev.jsonl",
    *,
    output_root: str | Path = PROJECT_ROOT / "outputs" / "yolo_datasets" / "youtube_vis_pilot",
    image_mode: str = "none",
    expected_num_classes: int = 40,
) -> dict[str, Any]:
    """Export train/val JSONL manifests to a YOLO dataset directory.

    image_mode="none" validates labels without creating image references.
    Use image_mode="symlink" or "hardlink" for an actual local smoke run.
    """

    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {sorted(IMAGE_MODES)}, got {image_mode!r}.")

    train_records = read_jsonl(train_manifest)
    val_records = read_jsonl(val_manifest)
    category_to_index, class_names = _category_mapping([*train_records, *val_records])
    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(class_names)}. "
            "Use a train split with full label coverage for the first smoke test."
        )

    dataset_root = Path(output_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_summary = _export_split(
        train_records,
        yolo_split="train",
        dataset_root=dataset_root,
        category_to_index=category_to_index,
        image_mode=image_mode,
    )
    val_summary = _export_split(
        val_records,
        yolo_split="val",
        dataset_root=dataset_root,
        category_to_index=category_to_index,
        image_mode=image_mode,
    )

    data_yaml = dataset_root / "data.yaml"
    _write_yolo_data_yaml(data_yaml, dataset_root, class_names)
    summary = {
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "image_mode": image_mode,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train": train_summary,
        "val": val_summary,
        "notes": {
            "source_data_modified": False,
            "copy_mode_warning": "image_mode='copy' duplicates frames under outputs and should stay local only.",
        },
    }
    write_json(dataset_root / "export_summary.json", summary)
    return summary


def export_default_yolo_pilot_dataset(
    *,
    image_mode: str = "none",
    output_root: str | Path = PROJECT_ROOT / "outputs" / "yolo_datasets" / "youtube_vis_pilot",
) -> dict[str, Any]:
    """Export the current default pilot split for YOLO smoke tests."""

    return export_yolo_detection_dataset(
        PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_train.jsonl",
        PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_dev.jsonl",
        output_root=output_root,
        image_mode=image_mode,
        expected_num_classes=40,
    )
