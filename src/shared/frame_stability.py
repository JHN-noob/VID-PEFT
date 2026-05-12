"""Frame-level stability metrics for detection prediction JSONL files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


PredictionRecord = dict[str, Any]
Detection = dict[str, Any]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def _mean(values: Iterable[float]) -> float | None:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)


def _bbox_xyxy(det: Detection) -> tuple[float, float, float, float] | None:
    if "bbox_xyxy" in det:
        values = det["bbox_xyxy"]
        if len(values) != 4:
            return None
        x1, y1, x2, y2 = (float(value) for value in values)
        return x1, y1, x2, y2
    if "bbox_xywh" in det:
        values = det["bbox_xywh"]
        if len(values) != 4:
            return None
        x, y, w, h = (float(value) for value in values)
        return x, y, x + w, y + h
    if "bbox" in det:
        values = det["bbox"]
        if len(values) != 4:
            return None
        fmt = str(det.get("bbox_format", "xywh")).lower()
        if fmt == "xyxy":
            x1, y1, x2, y2 = (float(value) for value in values)
            return x1, y1, x2, y2
        x, y, w, h = (float(value) for value in values)
        return x, y, x + w, y + h
    return None


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(left) + _area(right) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _normalize_detection(det: Detection, *, score_threshold: float) -> Detection | None:
    score = float(det.get("score", det.get("confidence", 1.0)))
    if score < score_threshold:
        return None
    box = _bbox_xyxy(det)
    if box is None or _area(box) <= 0:
        return None
    category_id = det.get("category_id", det.get("class_id", det.get("label", "unknown")))
    return {
        "bbox_xyxy": box,
        "score": score,
        "category_id": str(category_id),
    }


def _frame_detections(
    record: PredictionRecord,
    *,
    score_threshold: float,
    max_detections: int,
) -> list[Detection]:
    detections = [
        normalized
        for det in record.get("detections", [])
        if (normalized := _normalize_detection(det, score_threshold=score_threshold)) is not None
    ]
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections[:max_detections]


def _match_detections(
    previous: list[Detection],
    current: list[Detection],
    *,
    iou_threshold: float,
) -> list[tuple[Detection, Detection, float]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left in enumerate(previous):
        for right_index, right in enumerate(current):
            if left["category_id"] != right["category_id"]:
                continue
            iou = _iou(left["bbox_xyxy"], right["bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, left_index, right_index))

    matches: list[tuple[Detection, Detection, float]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for iou, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((previous[left_index], current[right_index], iou))
    return matches


def _record_key(record: PredictionRecord) -> tuple[str, int]:
    return str(record.get("video_id", "")), int(record.get("frame_index", 0))


def evaluate_frame_stability(
    predictions_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.1,
    max_detections: int = 100,
) -> dict[str, Any]:
    """Evaluate frame-to-frame detection stability from a prediction JSONL file.

    Expected JSONL schema per line:
    {
      "video_id": "...",
      "frame_index": 0,
      "width": 1280,
      "height": 720,
      "detections": [
        {"category_id": 3, "score": 0.91, "bbox_xyxy": [x1, y1, x2, y2]}
      ]
    }
    """

    predictions_path = Path(predictions_path)
    records = read_jsonl(predictions_path)
    grouped: dict[str, list[PredictionRecord]] = {}
    for record in records:
        grouped.setdefault(str(record.get("video_id", "")), []).append(record)
    for video_records in grouped.values():
        video_records.sort(key=lambda item: int(item.get("frame_index", 0)))

    frame_counts: list[int] = []
    count_deltas: list[float] = []
    unmatched_rates: list[float] = []
    matched_ious: list[float] = []
    center_shifts: list[float] = []
    area_log_ratios: list[float] = []
    score_deltas: list[float] = []
    transitions = 0
    comparable_transitions = 0
    matched_pairs = 0

    for video_id, video_records in grouped.items():
        previous_detections: list[Detection] | None = None
        previous_record: PredictionRecord | None = None
        for record in video_records:
            detections = _frame_detections(
                record,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            frame_counts.append(len(detections))
            if previous_detections is None or previous_record is None:
                previous_detections = detections
                previous_record = record
                continue

            transitions += 1
            count_deltas.append(abs(len(detections) - len(previous_detections)))
            denom = len(detections) + len(previous_detections)
            matches = _match_detections(previous_detections, detections, iou_threshold=iou_threshold)
            matched_pairs += len(matches)
            if denom > 0:
                comparable_transitions += 1
                unmatched_rates.append((denom - 2 * len(matches)) / denom)

            width = float(record.get("width") or previous_record.get("width") or 1.0)
            height = float(record.get("height") or previous_record.get("height") or 1.0)
            image_diag = max(1.0, math.hypot(width, height))
            for left, right, iou in matches:
                left_box = left["bbox_xyxy"]
                right_box = right["bbox_xyxy"]
                left_center = _center(left_box)
                right_center = _center(right_box)
                matched_ious.append(iou)
                center_shifts.append(
                    math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
                    / image_diag
                )
                left_area = max(_area(left_box), 1e-6)
                right_area = max(_area(right_box), 1e-6)
                area_log_ratios.append(abs(math.log(right_area / left_area)))
                score_deltas.append(abs(float(right["score"]) - float(left["score"])))

            previous_detections = detections
            previous_record = record

    report = {
        "prediction_path": str(predictions_path),
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "max_detections": int(max_detections),
        "videos": len(grouped),
        "frames": len(records),
        "transitions": transitions,
        "comparable_transitions": comparable_transitions,
        "matched_pairs": matched_pairs,
        "detections_total": int(sum(frame_counts)),
        "detections_per_frame_mean": _mean(frame_counts),
        "count_delta_abs_mean": _mean(count_deltas),
        "unmatched_rate_mean": _mean(unmatched_rates),
        "matched_iou_mean": _mean(matched_ious),
        "center_shift_norm_mean": _mean(center_shifts),
        "area_log_ratio_abs_mean": _mean(area_log_ratios),
        "score_delta_abs_mean": _mean(score_deltas),
    }

    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        stem = predictions_path.stem
        write_json(output_root / f"{stem}_stability.json", report)
        _write_summary_csv(output_root / f"{stem}_stability.csv", [report])
    return report


def evaluate_frame_stability_many(
    prediction_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.1,
    max_detections: int = 100,
) -> list[dict[str, Any]]:
    reports = [
        evaluate_frame_stability(
            path,
            output_dir=output_dir,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        for path in prediction_paths
    ]
    _write_summary_csv(Path(output_dir) / "frame_stability_summary.csv", reports)
    write_json(Path(output_dir) / "frame_stability_summary.json", {"runs": reports})
    return reports


def _write_summary_csv(path: str | Path, reports: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_path",
        "videos",
        "frames",
        "transitions",
        "matched_pairs",
        "detections_per_frame_mean",
        "count_delta_abs_mean",
        "unmatched_rate_mean",
        "matched_iou_mean",
        "center_shift_norm_mean",
        "area_log_ratio_abs_mean",
        "score_delta_abs_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report.get(key) for key in fieldnames})
    return path
