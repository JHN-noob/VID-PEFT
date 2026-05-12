"""Export YOLO frame predictions for frame-stability analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from src.shared.clip_sampler import clip_target_index, group_records_by_video, sample_record_clip
from src.shared.config import project_relative_path
from src.shared.main_experiment import build_frame_stability_specs
from src.shared.prediction_export import (
    find_checkpoint,
    image_path_from_record,
    load_manifest_records,
    load_run_config,
    resolve_path,
    select_specs,
    write_json,
)

from .backend import build_yolo_model, prepare_ultralytics_repo


def _result_to_detections(result: Any, *, score_threshold: float) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    detections: list[dict[str, Any]] = []
    for box, score, cls in zip(xyxy, scores, classes):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        detections.append(
            {
                "category_id": int(cls),
                "score": score_value,
                "bbox_xyxy": [round(float(value), 4) for value in box],
            }
        )
    return detections


def _target_result(results: list[Any], *, clip_length: int, target_index: int) -> Any:
    if len(results) == 1:
        return results[0]
    if len(results) == clip_length:
        return results[target_index]
    raise RuntimeError(
        f"Unexpected YOLO prediction result count: got {len(results)}, "
        f"expected 1 or clip_length={clip_length}."
    )


def _load_bgr_image(path: str | Path) -> Any:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image for YOLO prediction: {path}")
    return image


def _prediction_batch_size(preds: Any) -> int | None:
    item = preds
    if isinstance(item, (list, tuple)) and item:
        item = item[0]
    shape = getattr(item, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[0])


def _make_direct_predictor(
    *,
    checkpoint: Path,
    repo: Path,
    imgsz: int,
    score_threshold: float,
    device: str | int | None,
) -> Any:
    """Build a YOLO predictor used without the high-level result loop.

    Clip-aware YOLO checkpoints may reduce a T-frame input to one target-frame
    output inside the Detect pre-hook. Ultralytics' high-level predictor still
    iterates over all T input paths and can index past the single result. This
    direct predictor reuses Ultralytics preprocessing/postprocessing while
    constructing Results only for the actual output batch.
    """

    prepare_ultralytics_repo(repo)
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils.checks import check_imgsz

    overrides: dict[str, Any] = {
        "model": str(checkpoint),
        "task": "detect",
        "imgsz": int(imgsz),
        "conf": float(score_threshold),
        "iou": 0.7,
        "max_det": 300,
        "save": False,
        "save_txt": False,
        "show": False,
        "verbose": False,
        "batch": 1,
    }
    if device is not None:
        overrides["device"] = device

    yolo = build_yolo_model(str(checkpoint), repo_path=repo)
    predictor = DetectionPredictor(overrides=overrides)
    predictor.setup_model(model=yolo.model, verbose=False)
    predictor.imgsz = check_imgsz(predictor.args.imgsz, stride=predictor.model.stride, min_dim=2)
    return predictor


def _predict_target_frame(
    predictor: Any,
    *,
    image_paths: list[str],
    target_index: int,
) -> Any:
    import torch

    clip_images = [_load_bgr_image(path) for path in image_paths]
    # Ultralytics postprocess scales boxes with an in-place tensor update.
    # torch.inference_mode() creates inference tensors that reject that update,
    # so use no_grad() here instead.
    with torch.no_grad():
        im = predictor.preprocess(clip_images)
        if not predictor.done_warmup:
            predictor.model.warmup(
                imgsz=(1, predictor.model.channels, *predictor.imgsz)
            )
            predictor.done_warmup = True
        preds = predictor.inference(im)

    pred_batch = _prediction_batch_size(preds)
    if pred_batch == 1:
        result_paths = [image_paths[target_index]]
        result_images = [clip_images[target_index]]
        result_target_index = 0
        result_clip_length = 1
    elif pred_batch == len(image_paths):
        result_paths = image_paths
        result_images = clip_images
        result_target_index = target_index
        result_clip_length = len(image_paths)
    else:
        raise RuntimeError(
            f"Unexpected YOLO raw prediction batch size: got {pred_batch}, "
            f"expected 1 or input clip length {len(image_paths)}."
        )

    predictor.batch = (result_paths, result_images, [""] * len(result_paths))
    results = predictor.postprocess(preds, im, result_images)
    return _target_result(results, clip_length=result_clip_length, target_index=result_target_index)


def export_yolo_frame_stability_predictions(
    specs: Iterable[dict[str, Any]] | None = None,
    *,
    manifest_path: str | Path = "outputs/splits/youtube_vis_pilot_dev.jsonl",
    checkpoint_name: str = "best.pt",
    score_threshold: float = 0.001,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_frames: int | None = None,
    max_videos: int | None = None,
    overwrite: bool = False,
    missing_only: bool = False,
) -> list[dict[str, Any]]:
    """Export YOLO predictions to the shared frame-stability JSONL schema.

    For clip runs, each target frame is inferred with its sampled clip. The
    temporal adapter wrapper selects the target-frame feature internally, so the
    JSONL record remains one line per target frame.
    """

    if specs is None:
        specs = build_frame_stability_specs(model_families=("yolo_temporal",))
    selected_specs = select_specs(specs, model_family="yolo_temporal", missing_only=missing_only)

    records = load_manifest_records(manifest_path, max_videos=max_videos, max_frames=max_frames)
    grouped = group_records_by_video(records)
    repo = prepare_ultralytics_repo()
    reports: list[dict[str, Any]] = []

    for spec in selected_specs:
        prediction_path = resolve_path(spec["prediction_jsonl"])
        if prediction_path.exists() and not overwrite:
            reports.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "status": "skipped_exists",
                    "prediction_jsonl": project_relative_path(prediction_path),
                }
            )
            continue

        run_dir = resolve_path(spec["run_dir"])
        checkpoint = find_checkpoint(run_dir, candidates=(f"weights/{checkpoint_name}", "weights/best.pt", "best.pt"))
        run_config = load_run_config(run_dir)
        train_args = dict(run_config.get("yolo", {}).get("train_args", {}) or {})
        predict_imgsz = int(imgsz or train_args.get("imgsz", 640))

        clip = dict(spec.get("clip", {}) or {})
        clip_length = int(clip.get("length", 1))
        causal = bool(clip.get("causal", False))
        target = str(clip.get("target", "center"))
        target_index = clip_target_index(clip_length, causal=causal, target=target)

        predictor = _make_direct_predictor(
            checkpoint=checkpoint,
            repo=repo,
            imgsz=predict_imgsz,
            score_threshold=score_threshold,
            device=device,
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        written = 0

        with prediction_path.open("w", encoding="utf-8") as f:
            for video_id in sorted(grouped, key=str):
                video_records = grouped[video_id]
                for target_position, target_record in enumerate(video_records):
                    clip_records = sample_record_clip(
                        video_records,
                        target_position,
                        length=clip_length,
                        causal=causal,
                        boundary="clamp",
                    )
                    image_paths = [str(image_path_from_record(record)) for record in clip_records]
                    target_result = _predict_target_frame(
                        predictor,
                        image_paths=image_paths,
                        target_index=target_index,
                    )
                    output_record = {
                        "video_id": str(target_record.get("video_id", "")),
                        "frame_index": int(target_record.get("frame_index", target_position)),
                        "width": int(target_record.get("width", 0)),
                        "height": int(target_record.get("height", 0)),
                        "detections": _result_to_detections(target_result, score_threshold=score_threshold),
                    }
                    f.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1

        report = {
            "experiment_id": spec["experiment_id"],
            "status": "completed",
            "model_family": "yolo_temporal",
            "checkpoint": project_relative_path(checkpoint),
            "prediction_jsonl": project_relative_path(prediction_path),
            "frames": written,
            "score_threshold": float(score_threshold),
            "elapsed_sec": round(time.time() - start, 3),
        }
        write_json(prediction_path.with_suffix(".meta.json"), report)
        reports.append(report)

    return reports
