"""Generate RT-DETRv2-S YAML configs without modifying the official repo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.shared.config import PROJECT_ROOT

from .backend import official_rtdetrv2_s_config, prepare_rtdetr_repo


def _quote_path(path: str | Path) -> str:
    return "'" + Path(path).as_posix().replace("'", "''") + "'"


def _quote_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _yaml_float(value: float) -> str:
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _relative_include(config_path: Path, include_path: Path) -> str:
    rel = include_path.resolve().relative_to(PROJECT_ROOT.resolve())
    base = config_path.parent.resolve().relative_to(PROJECT_ROOT.resolve())
    relative = Path(*([".."] * len(base.parts))) / rel
    return relative.as_posix()


def _runtime_relative_path(path: str | Path, *, runtime_cwd: Path) -> str:
    raw = Path(path)
    absolute = raw if raw.is_absolute() else PROJECT_ROOT / raw
    return Path(os.path.relpath(absolute.resolve(), runtime_cwd.resolve())).as_posix()


def write_rtdetr_youtube_vis_config(
    *,
    dataset_export: dict[str, Any],
    output_dir: str | Path,
    config_path: str | Path,
    repo_path: str | Path | None = None,
    num_classes: int = 40,
    epochs: int = 1,
    total_batch_size: int = 4,
    num_workers: int = 0,
    print_freq: int = 20,
    use_amp: bool = True,
    use_ema: bool = False,
    sync_bn: bool = False,
    clip_loader_enabled: bool = False,
    clip_length: int = 1,
    clip_causal: bool = False,
    clip_target: str = "center",
    clip_boundary: str = "clamp",
    optimizer_lr: float | None = None,
    optimizer_backbone_lr: float | None = None,
    optimizer_weight_decay: float | None = None,
) -> Path:
    """Write a project-local RT-DETR config for YouTube-VIS bbox training."""

    repo = prepare_rtdetr_repo(repo_path)
    runtime_cwd = repo / "rtdetrv2_pytorch"
    base_config = official_rtdetrv2_s_config(repo)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    include_path = _relative_include(config_path, base_config)
    stop_epoch = max(0, int(epochs) - 1)

    train = dataset_export["train"]
    val = dataset_export["val"]
    dataset_type = "VideoClipCocoDetection" if clip_loader_enabled and int(clip_length) > 1 else "CocoDetection"
    collate_type = "ClipBatchImageCollateFunction" if clip_loader_enabled and int(clip_length) > 1 else "BatchImageCollateFunction"
    lines = [
        "__include__:",
        f"  - {_quote_text(include_path)}",
        "",
        f"output_dir: {_quote_path(_runtime_relative_path(output_dir, runtime_cwd=runtime_cwd))}",
        f"num_classes: {int(num_classes)}",
        "remap_mscoco_category: False",
        f"epoches: {int(epochs)}",
        f"print_freq: {int(print_freq)}",
        f"use_amp: {str(bool(use_amp))}",
        f"use_ema: {str(bool(use_ema))}",
        f"sync_bn: {str(bool(sync_bn))}",
        "",
        "PResNet:",
        "  depth: 18",
        "  freeze_at: -1",
        "  freeze_norm: False",
        "  pretrained: False",
        "",
        "train_dataloader:",
        "  dataset:",
        f"    type: {dataset_type}",
        f"    img_folder: {_quote_path(_runtime_relative_path(train['img_folder'], runtime_cwd=runtime_cwd))}",
        f"    ann_file: {_quote_path(_runtime_relative_path(train['ann_file'], runtime_cwd=runtime_cwd))}",
        "    return_masks: False",
    ]
    if dataset_type == "VideoClipCocoDetection":
        lines.extend(
            [
                f"    clip_length: {int(clip_length)}",
                f"    clip_causal: {str(bool(clip_causal))}",
                f"    clip_target: {_quote_text(str(clip_target))}",
                f"    clip_boundary: {_quote_text(str(clip_boundary))}",
            ]
        )
    lines.extend(
        [
            "    transforms:",
            "      policy:",
            f"        epoch: {stop_epoch}",
            "  collate_fn:",
            f"    type: {collate_type}",
            "    scales: ~",
            f"    stop_epoch: {stop_epoch}",
            "  shuffle: True",
            f"  total_batch_size: {int(total_batch_size)}",
            f"  num_workers: {int(num_workers)}",
            "",
            "val_dataloader:",
            "  dataset:",
            f"    type: {dataset_type}",
            f"    img_folder: {_quote_path(_runtime_relative_path(val['img_folder'], runtime_cwd=runtime_cwd))}",
            f"    ann_file: {_quote_path(_runtime_relative_path(val['ann_file'], runtime_cwd=runtime_cwd))}",
            "    return_masks: False",
        ]
    )
    if dataset_type == "VideoClipCocoDetection":
        lines.extend(
            [
                f"    clip_length: {int(clip_length)}",
                f"    clip_causal: {str(bool(clip_causal))}",
                f"    clip_target: {_quote_text(str(clip_target))}",
                f"    clip_boundary: {_quote_text(str(clip_boundary))}",
            ]
        )
    lines.extend(
        [
            "  shuffle: False",
            f"  total_batch_size: {int(total_batch_size)}",
            f"  num_workers: {int(num_workers)}",
            "  collate_fn:",
            f"    type: {collate_type}",
            "",
        ]
    )
    if optimizer_lr is not None:
        weight_decay = 0.0001 if optimizer_weight_decay is None else float(optimizer_weight_decay)
        lines.extend(
            [
                "optimizer:",
                "  type: AdamW",
                "  params:",
            ]
        )
        if optimizer_backbone_lr is not None:
            lines.extend(
                [
                    "    -",
                    "      params: '^(?=.*backbone)(?!.*norm).*$'",
                    f"      lr: {_yaml_float(float(optimizer_backbone_lr))}",
                ]
            )
        lines.extend(
            [
                "    -",
                "      params: '^(?=.*(?:norm|bn)).*$'",
                "      weight_decay: 0.",
                f"  lr: {_yaml_float(float(optimizer_lr))}",
                "  betas: [0.9, 0.999]",
                f"  weight_decay: {_yaml_float(weight_decay)}",
                "",
            ]
        )
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path
