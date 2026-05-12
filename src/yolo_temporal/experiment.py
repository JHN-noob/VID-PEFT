"""Experiment entrypoint for YOLO temporal adapter runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.config import load_experiment_config, project_relative_path, save_config_snapshot
from src.shared.preflight import check_environment
from src.shared.reporting import save_metrics

from .backend import build_yolo_model, prepare_ultralytics_repo
from .trainer import (
    make_head_only_detection_trainer,
    make_p4_temporal_detection_trainer,
    make_policy_recording_detection_trainer,
)
from src.shared.clip_sampler import clip_target_index


TEMPORAL_TUNING_MODES = {"spatial_temporal_peft"}
HEAD_WARMUP_TUNING_MODES = {"head_only"}
SPATIAL_ONLY_FULL_TUNING_MODES = {"spatial_only_full_ft"}


def run_yolo_temporal_experiment(config_or_path: str | Path | dict) -> dict[str, Any]:
    """Prepare or run a YOLO temporal experiment.

    Pilot configs default to execution.dry_run=true. Real training requires
    the notebook kernel to satisfy the official Ultralytics dependencies.
    """

    config = load_experiment_config(config_or_path)
    if config.model_family != "yolo_temporal":
        raise ValueError(f"Expected model_family='yolo_temporal', got {config.model_family!r}")

    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)
    preflight = check_environment(config.to_dict(), print_report=True)

    repo = prepare_ultralytics_repo(config.resolved_path("paths", "ultralytics_repo", default="third_party/ultralytics"))
    dry_run = bool(config.get("execution", "dry_run", default=True))
    weights = str(config.get("yolo", "weights", default="yolov8m.pt"))
    weight_path = Path(weights)
    if not weight_path.is_absolute():
        weight_path = config.project_root / weight_path

    metrics = {
        "status": "dry_run" if dry_run else "started",
        "experiment_id": config.experiment_id,
        "model_family": config.model_family,
        "tuning_mode": config.tuning_mode,
        "budget": config.budget,
        "adapter_dim": config.budget_settings.get("adapter_dim"),
        "seed": config.seed,
        "clip_length": config.clip_length,
        "causal": config.causal,
        "yolo_weights": project_relative_path(weight_path),
        "ultralytics_repo": project_relative_path(repo),
        "preflight": [result.__dict__ for result in preflight],
    }
    save_metrics(output_dir, metrics)

    if dry_run:
        return metrics

    if not weight_path.exists():
        raise FileNotFoundError(
            "YOLO COCO-pretrained weights must exist locally before training. "
            f"Configured weights: {weights!r}"
        )

    model = build_yolo_model(str(weight_path), repo_path=repo)
    train_args = dict(config.get("yolo", "train_args", default={}) or {})
    train_args.setdefault("project", str(output_dir.parent))
    train_args.setdefault("name", output_dir.name)
    train_args.setdefault("seed", config.seed)
    train_args.setdefault("exist_ok", True)
    trainer = None
    if config.tuning_mode in HEAD_WARMUP_TUNING_MODES:
        trainer = make_head_only_detection_trainer(tuning_mode=config.tuning_mode)
        train_args.setdefault("freeze", 22)
    elif config.tuning_mode in SPATIAL_ONLY_FULL_TUNING_MODES:
        trainer = make_policy_recording_detection_trainer(tuning_mode=config.tuning_mode)
        train_args.pop("freeze", None)
    elif config.tuning_mode in TEMPORAL_TUNING_MODES:
        clip_loader_enabled = bool(config.get("yolo", "clip_loader", "enabled", default=False))
        if config.clip_length > 1 and not clip_loader_enabled:
            metrics["status"] = "blocked"
            metrics["blocked_reason"] = (
                "YOLO clip_length>1 training requires a clip-aware dataloader. "
                "Use yolo.clip_loader.enabled=true."
            )
            save_metrics(output_dir, metrics)
            raise RuntimeError(metrics["blocked_reason"])
        if clip_loader_enabled and config.clip_length > 1:
            for key, value in {
                "mosaic": 0.0,
                "mixup": 0.0,
                "cutmix": 0.0,
                "copy_paste": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "close_mosaic": 0,
            }.items():
                train_args.setdefault(key, value)
        adapter_cfg = dict(config.get("yolo", "adapter", default={}) or {})
        clip_target = str(config.get("clip", "target", default="center"))
        target_index = clip_target_index(config.clip_length, causal=config.causal, target=clip_target)
        placement = str(
            adapter_cfg.get(
                "placement",
                "detect_pre_forward" if clip_loader_enabled and config.clip_length > 1 else "layer_local_forward_hook",
            )
        )
        trainer = make_p4_temporal_detection_trainer(
            tuning_mode=config.tuning_mode,
            layer_index=int(adapter_cfg.get("layer_index", 18)),
            channels=adapter_cfg.get("channels"),
            adapter_dim=int(adapter_cfg.get("adapter_dim", config.budget_settings.get("adapter_dim", 32))),
            kernel_size=int(adapter_cfg.get("kernel_size", 3)),
            clip_length=config.clip_length,
            clip_loader_enabled=clip_loader_enabled,
            causal=config.causal,
            target=clip_target,
            target_index=target_index,
            boundary=str(config.get("yolo", "clip_loader", "boundary", default="clamp")),
            residual_scale=float(adapter_cfg.get("residual_scale", 1.0)),
            placement=placement,
        )
        train_args.setdefault("freeze", 23)
    else:
        metrics["status"] = "blocked"
        metrics["blocked_reason"] = (
            f"Unsupported YOLO tuning_mode for the active pipeline: {config.tuning_mode!r}. "
            "Validated YOLO modes are head_only, spatial_only_full_ft, and spatial_temporal_peft."
        )
        save_metrics(output_dir, metrics)
        raise RuntimeError(metrics["blocked_reason"])

    result = model.train(trainer=trainer, **train_args)
    metrics["status"] = "completed"
    metrics["result"] = str(result)
    policy_report_path = output_dir / "policy_report.json"
    if policy_report_path.exists():
        import json

        with policy_report_path.open("r", encoding="utf-8") as f:
            metrics["policy_report"] = json.load(f)
    save_metrics(output_dir, metrics)
    return metrics
