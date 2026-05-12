"""YouTube-VIS manifest conversion using annotation-provided bounding boxes."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT, read_json


def bbox_xywh_to_xyxy(bbox: Iterable[float]) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def _annotation_json_paths(config: dict[str, Any]) -> list[tuple[str, Path]]:
    paths = config.get("paths", {})
    annotation_files = paths.get("annotation_files")
    if annotation_files:
        return [(split, Path(path)) for split, path in annotation_files.items()]

    root = Path(paths.get("annotations_root", PROJECT_ROOT / "data" / "YouTubeVIS" / "annotations"))
    candidates = [
        ("train", root / "train.json"),
        ("val", root / "valid.json"),
        ("val", root / "val.json"),
    ]
    return [(split, path) for split, path in candidates if path.exists()]


def _video_lookup(data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(video["id"]): video for video in data.get("videos", [])}


def _category_lookup(data: dict[str, Any]) -> dict[int, str]:
    return {int(category["id"]): str(category.get("name", category["id"])) for category in data.get("categories", [])}


def _frame_objects(data: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    categories = _category_lookup(data)
    by_frame: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        video_id = int(ann["video_id"])
        category_id = int(ann.get("category_id", -1))
        label = categories.get(category_id, str(category_id))
        bboxes = ann.get("bboxes")
        if bboxes is None and "bbox" in ann:
            bboxes = ann["bbox"]
        if bboxes is None:
            continue

        # YouTube-VIS commonly stores one bbox per frame. If a single bbox is
        # provided, treat it as a one-frame annotation.
        if bboxes and isinstance(bboxes[0], (int, float)):
            bboxes = [bboxes]

        for frame_index, bbox in enumerate(bboxes):
            if bbox in (None, [], [None]):
                continue
            bbox_xyxy = bbox_xywh_to_xyxy(bbox)
            by_frame[(video_id, frame_index)].append(
                {
                    "label": label,
                    "category_id": category_id,
                    "track_id": str(ann.get("id")),
                    "bbox_xywh": [float(value) for value in bbox],
                    "bbox_xyxy": bbox_xyxy,
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )
    return by_frame


def convert_youtube_vis_json(
    annotation_json: str | Path,
    *,
    split: str,
    videos_root: str | Path,
    output_file,
) -> int:
    data = read_json(annotation_json)
    videos = _video_lookup(data)
    objects_by_frame = _frame_objects(data)
    count = 0
    videos_root = Path(videos_root)

    for video_id, video in sorted(videos.items()):
        file_names = video.get("file_names", [])
        width = video.get("width")
        height = video.get("height")
        for frame_index, file_name in enumerate(file_names):
            record = {
                "dataset": "youtube_vis",
                "split": split,
                "video_id": str(video_id),
                "frame_index": frame_index,
                "image_path": str(videos_root / file_name),
                "annotation_path": str(annotation_json),
                "width": width,
                "height": height,
                "objects": objects_by_frame.get((video_id, frame_index), []),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_youtube_vis_manifest(
    config_or_path: str | Path | dict,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Convert YouTube-VIS COCO-style annotations to a detection manifest."""

    config = config_or_path if isinstance(config_or_path, dict) else read_json(config_or_path)
    paths = config.get("paths", {})
    videos_root = Path(paths.get("videos_root", PROJECT_ROOT / "data" / "YouTubeVIS" / "train"))
    videos_roots = {split: Path(path) for split, path in paths.get("videos_roots", {}).items()}
    manifest_path = Path(output_path or paths.get("manifest_path", PROJECT_ROOT / "outputs" / "manifests" / "youtube_vis.jsonl"))
    annotation_paths = _annotation_json_paths(config)
    if not annotation_paths:
        raise FileNotFoundError("No YouTube-VIS annotation JSON files were found.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for split, annotation_json in annotation_paths:
            split_videos_root = videos_roots.get(split, videos_root)
            convert_youtube_vis_json(annotation_json, split=split, videos_root=split_videos_root, output_file=f)
    return manifest_path
