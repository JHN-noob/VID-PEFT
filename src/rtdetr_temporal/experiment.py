"""Experiment entrypoint for RT-DETRv2-S temporal runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from src.shared.config import load_experiment_config, project_relative_path, project_relative_text, save_config_snapshot
from src.shared.preflight import check_environment
from src.shared.reporting import save_metrics

from .backend import build_train_command, prepare_rtdetr_repo, resolve_local_checkpoint, run_official_train
from .config_builder import write_rtdetr_youtube_vis_config
from .dataset_export import export_rtdetr_coco_dataset
from .diagnostics import check_rtdetr_runtime
from src.shared.clip_sampler import clip_target_index


RTDETR_TRAIN_REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "yaml",
    "scipy",
    "pycocotools",
    "faster_coco_eval",
    "tensorboard",
)
TEMPORAL_TUNING_MODES = {"spatial_temporal_peft"}
LORA_TUNING_MODES = {"spatial_temporal_peft"}


def _missing_modules(names: tuple[str, ...]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def run_rtdetr_temporal_experiment(config_or_path: str | Path | dict) -> dict[str, Any]:
    """Prepare or run an RT-DETRv2-S YouTube-VIS detection experiment."""

    config = load_experiment_config(config_or_path)
    if config.model_family != "rtdetr_temporal":
        raise ValueError(f"Expected model_family='rtdetr_temporal', got {config.model_family!r}")

    output_dir = config.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir)
    preflight = check_environment(config.to_dict(), print_report=True)

    repo = prepare_rtdetr_repo(config.resolved_path("paths", "rtdetr_repo", default="third_party/RT-DETR"))
    dry_run = bool(config.get("execution", "dry_run", default=True))
    rtdetr_cfg = dict(config.get("rtdetr", default={}) or {})
    export_cfg = dict(rtdetr_cfg.get("dataset_export", {}) or {})
    train_cfg = dict(rtdetr_cfg.get("train_args", {}) or {})

    dataset_export = export_rtdetr_coco_dataset(
        export_cfg.get("train_manifest", output_dir.parents[1] / "splits" / "youtube_vis_pilot_train.jsonl"),
        export_cfg.get("val_manifest", output_dir.parents[1] / "splits" / "youtube_vis_pilot_dev.jsonl"),
        output_root=export_cfg.get("output_root", "outputs/rtdetr_datasets/youtube_vis_pilot"),
        image_mode=export_cfg.get("image_mode", "none"),
        expected_num_classes=int(config.get("dataset", "num_classes", default=40)),
    )
    clip_loader_enabled = bool(config.get("rtdetr", "clip_loader", "enabled", default=False))
    clip_target = str(config.get("clip", "target", default="center"))
    target_index = clip_target_index(config.clip_length, causal=config.causal, target=clip_target)

    generated_config = write_rtdetr_youtube_vis_config(
        dataset_export=dataset_export,
        output_dir=output_dir,
        config_path=rtdetr_cfg.get("generated_config", output_dir / "rtdetrv2_s_youtube_vis.yml"),
        repo_path=repo,
        num_classes=int(config.get("dataset", "num_classes", default=40)),
        epochs=int(train_cfg.get("epochs", 1)),
        total_batch_size=int(train_cfg.get("total_batch_size", 4)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        print_freq=int(train_cfg.get("print_freq", 20)),
        use_amp=bool(train_cfg.get("use_amp", True)),
        use_ema=bool(train_cfg.get("use_ema", False)),
        sync_bn=bool(train_cfg.get("sync_bn", False)),
        clip_loader_enabled=clip_loader_enabled,
        clip_length=config.clip_length,
        clip_causal=config.causal,
        clip_target=clip_target,
        clip_boundary=str(config.get("rtdetr", "clip_loader", "boundary", default="clamp")),
        optimizer_lr=train_cfg.get("optimizer_lr"),
        optimizer_backbone_lr=train_cfg.get("optimizer_backbone_lr"),
        optimizer_weight_decay=train_cfg.get("optimizer_weight_decay"),
    )

    checkpoint = rtdetr_cfg.get("checkpoint")
    local_checkpoint = resolve_local_checkpoint(checkpoint)
    diagnostics = check_rtdetr_runtime(repo, checkpoint=checkpoint)
    metrics = {
        "status": "dry_run" if dry_run else "started",
        "experiment_id": config.experiment_id,
        "model_family": config.model_family,
        "model_variant": rtdetr_cfg.get("variant", "rtdetrv2-s-r18vd"),
        "tuning_mode": config.tuning_mode,
        "budget": config.budget,
        "budget_settings": config.budget_settings,
        "lora_rank": config.get("rtdetr", "lora", "rank", default=config.budget_settings.get("lora_rank"))
        if config.tuning_mode in LORA_TUNING_MODES
        else 0,
        "seed": config.seed,
        "clip_length": config.clip_length,
        "causal": config.causal,
        "rtdetr_repo": project_relative_path(repo),
        "generated_config": project_relative_path(generated_config),
        "dataset_export": dataset_export,
        "checkpoint": project_relative_path(local_checkpoint) if local_checkpoint is not None else str(checkpoint),
        "preflight": [result.__dict__ for result in preflight],
        "diagnostics": diagnostics,
    }
    save_metrics(output_dir, metrics)

    if dry_run:
        return metrics

    uses_temporal_adapter = config.tuning_mode in TEMPORAL_TUNING_MODES or (
        clip_loader_enabled and config.clip_length > 1
    )
    if uses_temporal_adapter and config.clip_length > 1 and not clip_loader_enabled:
        metrics["status"] = "blocked"
        metrics["blocked_reason"] = (
            "RT-DETR clip_length>1 training requires a clip-aware dataloader. "
            "Use rtdetr.clip_loader.enabled=true."
        )
        save_metrics(output_dir, metrics)
        raise RuntimeError(metrics["blocked_reason"])

    missing_modules = _missing_modules(RTDETR_TRAIN_REQUIRED_MODULES)
    if missing_modules:
        metrics["status"] = "blocked"
        metrics["missing_modules"] = missing_modules
        save_metrics(output_dir, metrics)
        raise RuntimeError(
            "RT-DETR training requires missing Python modules in the active notebook kernel: "
            + ", ".join(missing_modules)
            + ". Install them in the notebook environment before running smoke training."
        )

    if local_checkpoint is None or not local_checkpoint.exists():
        raise FileNotFoundError(
            "RT-DETRv2-S COCO checkpoint must exist locally before training. "
            f"Configured checkpoint: {checkpoint!r}"
        )

    command = build_train_command(
        generated_config=generated_config,
        output_dir=output_dir,
        tuning_checkpoint=local_checkpoint,
        repo_path=repo,
        tuning_mode=config.tuning_mode,
        clip_length=config.clip_length,
        temporal_adapter=uses_temporal_adapter,
        temporal_adapter_index=int(config.get("rtdetr", "adapter", "feature_index", default=1)),
        temporal_adapter_channels=int(config.get("rtdetr", "adapter", "channels", default=256)),
        temporal_adapter_dim=int(config.get("rtdetr", "adapter", "adapter_dim", default=config.budget_settings.get("adapter_dim", 32))),
        temporal_kernel_size=int(config.get("rtdetr", "adapter", "kernel_size", default=3)),
        temporal_residual_scale=float(config.get("rtdetr", "adapter", "residual_scale", default=1.0)),
        clip_loader_enabled=clip_loader_enabled,
        target_index=target_index,
        lora_rank=int(config.get("rtdetr", "lora", "rank", default=config.budget_settings.get("lora_rank", 0)))
        if config.tuning_mode in LORA_TUNING_MODES
        else 0,
        lora_alpha=config.get("rtdetr", "lora", "alpha", default=None),
        lora_targets=config.get("rtdetr", "lora", "targets", default=None),
        seed=config.seed,
        device=train_cfg.get("device"),
        use_amp=bool(train_cfg.get("use_amp", True)),
    )
    result = run_official_train(command, repo_path=repo, log_dir=output_dir, check=False)
    metrics["returncode"] = int(result.returncode)
    metrics["command"] = [project_relative_text(item) for item in command]
    if result.returncode != 0:
        metrics["status"] = "failed"
        metrics["stdout_log"] = project_relative_path(output_dir / "official_train_stdout.log")
        metrics["stderr_log"] = project_relative_path(output_dir / "official_train_stderr.log")
        stderr_tail = "\n".join((result.stderr or "").splitlines()[-40:])
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-20:])
        if stderr_tail:
            metrics["stderr_tail"] = stderr_tail
        if stdout_tail:
            metrics["stdout_tail"] = stdout_tail
        save_metrics(output_dir, metrics)
        detail = [
            f"RT-DETR official train failed with return code {result.returncode}.",
            f"Logs: {project_relative_path(output_dir)}",
        ]
        if stderr_tail:
            detail.append("stderr tail:\n" + stderr_tail)
        if stdout_tail:
            detail.append("stdout tail:\n" + stdout_tail)
        raise RuntimeError("\n\n".join(detail))

    metrics["status"] = "completed"
    policy_report_path = output_dir / "policy_report.json"
    if policy_report_path.exists():
        with policy_report_path.open("r", encoding="utf-8") as f:
            metrics["policy_report"] = json.load(f)
    save_metrics(output_dir, metrics)
    return metrics
