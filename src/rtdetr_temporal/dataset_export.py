"""Export converted YouTube-VIS manifests to RT-DETR COCO detection layout."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.shared.config import PROJECT_ROOT, write_json
from src.shared.detection_export import (
    IMAGE_MODES,
    category_mapping,
    clamped_xywh,
    materialize_image,
    read_jsonl,
    relative_image_path,
    resolve_project_path,
)


def _categories(class_names: list[str], source_category_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            "name": class_names[index],
            "source_category_id": source_category_ids[index],
        }
        for index in range(len(class_names))
    ]


def _export_split(
    records: list[dict[str, Any]],
    *,
    split_name: str,
    dataset_root: Path,
    category_to_index: dict[int, int],
    class_names: list[str],
    source_category_ids: list[int],
    image_mode: str,
) -> dict[str, Any]:
    image_dir = dataset_root / "images" / split_name
    ann_dir = dataset_root / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    materialized = Counter()
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    skipped_boxes = 0
    annotation_id = 1

    for image_id, record in enumerate(records, start=1):
        width = int(record.get("width") or 0)
        height = int(record.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Missing image size for record {record.get('video_id')}:{record.get('frame_index')}")

        source_image = resolve_project_path(record["image_path"])
        rel_image = relative_image_path(record, source_image)
        target_image = image_dir / rel_image
        materialized[materialize_image(source_image, target_image, image_mode)] += 1

        images.append(
            {
                "id": image_id,
                "file_name": rel_image.as_posix(),
                "width": width,
                "height": height,
                "video_id": str(record.get("video_id")),
                "frame_index": int(record.get("frame_index", 0)),
            }
        )

        for obj in record.get("objects", []):
            bbox = clamped_xywh(obj, float(width), float(height))
            if bbox is None:
                skipped_boxes += 1
                continue
            box_w = bbox[2]
            box_h = bbox[3]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_index[int(obj["category_id"])],
                    "bbox": [round(float(value), 4) for value in bbox],
                    "area": round(float(box_w * box_h), 4),
                    "iscrowd": int(obj.get("iscrowd", 0)),
                    "track_id": str(obj.get("track_id", "")),
                }
            )
            annotation_id += 1

    ann_path = ann_dir / f"instances_{split_name}.json"
    write_json(
        ann_path,
        {
            "images": images,
            "annotations": annotations,
            "categories": _categories(class_names, source_category_ids),
        },
    )
    return {
        "split": split_name,
        "img_folder": str(image_dir),
        "ann_file": str(ann_path),
        "frames": len(images),
        "objects": len(annotations),
        "skipped_boxes": skipped_boxes,
        "image_mode_counts": dict(materialized),
    }


def export_rtdetr_coco_dataset(
    train_manifest: str | Path = PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_train.jsonl",
    val_manifest: str | Path = PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_dev.jsonl",
    *,
    output_root: str | Path = PROJECT_ROOT / "outputs" / "rtdetr_datasets" / "youtube_vis_pilot",
    image_mode: str = "none",
    expected_num_classes: int = 40,
) -> dict[str, Any]:
    """Export train/val JSONL manifests to RT-DETR-compatible COCO JSON.

    COCO category ids are intentionally zero-based because the official
    RT-DETR loader uses category_id directly when remap_mscoco_category=False.
    """

    if image_mode not in IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {sorted(IMAGE_MODES)}, got {image_mode!r}.")

    train_records = read_jsonl(train_manifest)
    val_records = read_jsonl(val_manifest)
    category_to_index, class_names, source_category_ids = category_mapping([*train_records, *val_records])
    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(class_names)}. "
            "Use a train split with full label coverage for the first smoke test."
        )

    dataset_root = Path(output_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    train_summary = _export_split(
        train_records,
        split_name="train",
        dataset_root=dataset_root,
        category_to_index=category_to_index,
        class_names=class_names,
        source_category_ids=source_category_ids,
        image_mode=image_mode,
    )
    val_summary = _export_split(
        val_records,
        split_name="val",
        dataset_root=dataset_root,
        category_to_index=category_to_index,
        class_names=class_names,
        source_category_ids=source_category_ids,
        image_mode=image_mode,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "image_mode": image_mode,
        "num_classes": len(class_names),
        "category_id_base": 0,
        "class_names": class_names,
        "source_category_ids": source_category_ids,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train": train_summary,
        "val": val_summary,
        "notes": {
            "source_data_modified": False,
            "remap_mscoco_category": False,
            "copy_mode_warning": "image_mode='copy' duplicates frames under outputs and should stay local only.",
        },
    }
    write_json(dataset_root / "export_summary.json", summary)
    return summary


def export_default_rtdetr_pilot_dataset(
    *,
    image_mode: str = "none",
    output_root: str | Path = PROJECT_ROOT / "outputs" / "rtdetr_datasets" / "youtube_vis_pilot",
) -> dict[str, Any]:
    """Export the current default pilot split for RT-DETR smoke tests."""

    return export_rtdetr_coco_dataset(
        PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_train.jsonl",
        PROJECT_ROOT / "outputs" / "splits" / "youtube_vis_pilot_dev.jsonl",
        output_root=output_root,
        image_mode=image_mode,
        expected_num_classes=40,
    )
