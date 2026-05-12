"""Run official RT-DETR training with project-local trainability policies.

This file intentionally avoids importing the project ``src`` package after the
official RT-DETR repository is placed on ``sys.path``. The official repository
also uses a top-level package named ``src``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


HEAD_KEYWORDS = ("enc_score_head", "dec_score_head", "denoising_class_embed")
TEMPORAL_KEYWORDS = ("temporal", "adapter")
LORA_KEYWORDS = ("lora_a", "lora_b")
SPATIAL_KEYWORDS = LORA_KEYWORDS


class LoRALinear(nn.Module):
    """LoRA wrapper for official RT-DETR linear layers."""

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float | None = None):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1.")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        weight = self.lora_a.weight
        lora_in = x.to(dtype=weight.dtype) if x.dtype != weight.dtype else x
        delta = self.lora_b(self.lora_a(lora_in)) * self.scaling
        return base_out + delta.to(dtype=base_out.dtype)


class TemporalConvAdapter(nn.Module):
    """Bottleneck temporal adapter for [B,T,C,H,W] feature clips."""

    def __init__(
        self,
        channels: int,
        adapter_dim: int = 32,
        kernel_size: int = 3,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Conv3d(channels, adapter_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(adapter_dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                adapter_dim,
                adapter_dim,
                kernel_size=(kernel_size, 1, 1),
                padding=(padding, 0, 0),
                groups=adapter_dim,
                bias=False,
            ),
            nn.BatchNorm3d(adapter_dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(adapter_dim, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("TemporalConvAdapter expects [B, T, C, H, W].")
        weight = next(self.net.parameters(), None)
        if weight is not None and weight.device != x.device:
            self.to(device=x.device)
            weight = next(self.net.parameters(), None)
        compute_dtype = weight.dtype if weight is not None else x.dtype
        y = x.permute(0, 2, 1, 3, 4).contiguous()
        if y.dtype != compute_dtype:
            y = y.to(dtype=compute_dtype)
        y = self.net(y)
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        return x + y * self.residual_scale


class HybridEncoderTemporalAdapterWrapper(nn.Module):
    """Wrap HybridEncoder and adapt one output level, usually P4."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        feature_index: int = 1,
        channels: int = 256,
        adapter_dim: int = 32,
        kernel_size: int = 3,
        clip_length: int = 1,
        target_index: int = 0,
        residual_scale: float = 1.0,
        select_target_only: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_index = int(feature_index)
        self.clip_length = int(clip_length)
        self.target_index = int(target_index)
        self.select_target_only = bool(select_target_only)
        self.temporal_adapter_p4 = TemporalConvAdapter(
            channels=channels,
            adapter_dim=adapter_dim,
            kernel_size=kernel_size,
            residual_scale=residual_scale,
        )
        self.adapter_info = {
            "feature_level": "P4" if feature_index == 1 else f"index_{feature_index}",
            "feature_index": int(feature_index),
            "channels": int(channels),
            "adapter_dim": int(adapter_dim),
            "kernel_size": int(kernel_size),
            "clip_length": int(clip_length),
            "target_index": int(target_index),
            "residual_scale": float(residual_scale),
            "placement": "HybridEncoder output",
        }

    def forward(self, feats):  # type: ignore[no-untyped-def]
        outs = list(self.encoder(feats))
        feature = outs[self.feature_index]
        if feature.shape[0] == 1 and self.clip_length > 1:
            return outs
        clip = _reshape_flat_feature_to_clip(feature, self.clip_length)
        adapted = self.temporal_adapter_p4(clip)
        if self.select_target_only:
            outs[self.feature_index] = adapted[:, self.target_index, :, :, :].contiguous()
            for index, output in enumerate(outs):
                if index == self.feature_index:
                    continue
                output_clip = _reshape_flat_feature_to_clip(output, self.clip_length)
                outs[index] = output_clip[:, self.target_index, :, :, :].contiguous()
        else:
            batch_time, channels, height, width = feature.shape
            outs[self.feature_index] = adapted.reshape(batch_time, channels, height, width)
        return outs


def _reshape_flat_feature_to_clip(feature: torch.Tensor, clip_length: int) -> torch.Tensor:
    if feature.ndim != 4:
        raise ValueError("Expected feature tensor shaped [B*T, C, H, W].")
    batch_time, channels, height, width = feature.shape
    if batch_time % clip_length != 0:
        raise RuntimeError(f"Feature batch {batch_time} is not divisible by clip_length={clip_length}.")
    batch = batch_time // clip_length
    return feature.reshape(batch, clip_length, channels, height, width)


def _set_all_requires_grad(model, value: bool) -> None:  # type: ignore[no-untyped-def]
    for param in model.parameters():
        param.requires_grad = value


def _matches(name: str, keywords: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _enable_by_keywords(model, keywords: Iterable[str]) -> list[str]:  # type: ignore[no-untyped-def]
    trainable: list[str] = []
    for name, param in model.named_parameters():
        if _matches(name, keywords):
            param.requires_grad = True
            trainable.append(name)
    return trainable


def _parameter_summary(model) -> dict[str, float]:  # type: ignore[no-untyped-def]
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def apply_trainability_policy(model, tuning_mode: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Apply the same policy semantics used by the project wrapper."""

    if tuning_mode == "spatial_only_full_ft":
        _set_all_requires_grad(model, True)
        trainable = [name for name, _ in model.named_parameters()]
    else:
        _set_all_requires_grad(model, False)
        if tuning_mode == "head_only":
            trainable = _enable_by_keywords(model, HEAD_KEYWORDS)
        elif tuning_mode == "spatial_temporal_peft":
            trainable = _enable_by_keywords(model, TEMPORAL_KEYWORDS + SPATIAL_KEYWORDS)
        else:
            raise ValueError(f"Unsupported RT-DETR tuning mode: {tuning_mode}")

    return {
        "tuning_mode": tuning_mode,
        "trainable_names": trainable,
        "parameter_summary": _parameter_summary(model),
    }


def _write_policy_report(output_dir: str | Path, report: dict[str, object]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _de_parallel(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _target_matches(name: str, targets: tuple[str, ...]) -> bool:
    return any(target in name for target in targets)


def _replace_lora_linears(
    module: nn.Module,
    *,
    prefix: str = "",
    targets: tuple[str, ...],
    rank: int,
    alpha: float | None,
    device: torch.device,
    replaced: list[dict[str, object]],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and _target_matches(full_name, targets):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha).to(device)
            setattr(module, child_name, wrapped)
            replaced.append(
                {
                    "name": full_name,
                    "in_features": int(child.in_features),
                    "out_features": int(child.out_features),
                    "rank": int(rank),
                    "parameters": int(rank * (child.in_features + child.out_features)),
                }
            )
        else:
            _replace_lora_linears(
                child,
                prefix=full_name,
                targets=targets,
                rank=rank,
                alpha=alpha,
                device=device,
                replaced=replaced,
            )


def inject_lora(model: nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    module = _de_parallel(model)
    targets = tuple(item.strip() for item in args.lora_targets.split(",") if item.strip())
    if not targets:
        raise ValueError("At least one LoRA target substring is required.")
    replaced: list[dict[str, object]] = []
    _replace_lora_linears(
        module,
        targets=targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        device=device,
        replaced=replaced,
    )
    if not replaced:
        raise RuntimeError(f"No RT-DETR Linear layers matched LoRA targets: {targets}")
    return {
        "rank": int(args.lora_rank),
        "alpha": float(args.lora_alpha if args.lora_alpha is not None else args.lora_rank),
        "targets": list(targets),
        "layers": replaced,
        "parameters": int(sum(item["parameters"] for item in replaced)),
    }


def attach_temporal_adapter(model: nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    module = _de_parallel(model)
    if not hasattr(module, "encoder"):
        raise AttributeError("RT-DETR model has no encoder attribute.")
    wrapper = HybridEncoderTemporalAdapterWrapper(
        module.encoder,
        feature_index=args.temporal_adapter_index,
        channels=args.temporal_adapter_channels,
        adapter_dim=args.temporal_adapter_dim,
        kernel_size=args.temporal_kernel_size,
        clip_length=args.clip_length,
        target_index=args.target_index,
        residual_scale=args.temporal_residual_scale,
        select_target_only=args.clip_loader_enabled and args.clip_length > 1,
    ).to(device)
    module.encoder = wrapper
    return dict(wrapper.adapter_info)


def clip_target_index(length: int, *, causal: bool = False, target: str = "center") -> int:
    if length < 1:
        raise ValueError("clip length must be >= 1.")
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
    length: int,
    causal: bool,
    boundary: str,
) -> list[int]:
    target = "current" if causal else "center"
    target_pos = clip_target_index(length, causal=causal, target=target)
    offsets = [index - target_pos for index in range(length)]
    indices = [frame_index + offset for offset in offsets]
    if boundary == "clamp":
        return [min(max(index, 0), num_frames - 1) for index in indices]
    if boundary == "drop":
        if any(index < 0 or index >= num_frames for index in indices):
            return []
        return indices
    raise ValueError(f"Unsupported clip boundary: {boundary!r}")


def register_clip_dataset_types(register):  # type: ignore[no-untyped-def]
    from src.data.dataset.coco_dataset import CocoDetection
    from src.data.dataloader import BaseCollateFunction

    @register()
    class VideoClipCocoDetection(CocoDetection):  # type: ignore[no-redef]
        def __init__(
            self,
            img_folder,
            ann_file,
            transforms,
            return_masks=False,
            remap_mscoco_category=False,
            clip_length=1,
            clip_causal=False,
            clip_target="center",
            clip_boundary="clamp",
        ):
            super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
            self.clip_length = int(clip_length)
            self.clip_causal = bool(clip_causal)
            self.clip_target = str(clip_target)
            self.clip_boundary = str(clip_boundary)
            self.clip_target_index = clip_target_index(
                self.clip_length,
                causal=self.clip_causal,
                target=self.clip_target,
            )
            grouped = {}
            for base_index, image_id in enumerate(self.ids):
                info = self.coco.imgs[image_id]
                video_id = str(info.get("video_id", ""))
                frame_index = int(info.get("frame_index", base_index))
                grouped.setdefault(video_id, []).append((frame_index, base_index))
            self._video_indices = {}
            self._base_to_video_pos = {}
            for video_id, pairs in grouped.items():
                ordered = [base_idx for _, base_idx in sorted(pairs)]
                self._video_indices[video_id] = ordered
                for position, base_idx in enumerate(ordered):
                    self._base_to_video_pos[base_idx] = (video_id, position)
            self._target_base_indices = list(range(len(self.ids)))

        def __len__(self):
            return len(self._target_base_indices)

        def _clip_base_indices(self, base_index):
            video_id, position = self._base_to_video_pos[base_index]
            video_indices = self._video_indices[video_id]
            positions = sample_clip_indices(
                position,
                len(video_indices),
                length=self.clip_length,
                causal=self.clip_causal,
                boundary=self.clip_boundary,
            )
            if not positions:
                raise IndexError(f"No valid clip for index={base_index}.")
            return [video_indices[pos] for pos in positions]

        def __getitem__(self, idx):
            target_base_index = self._target_base_indices[idx]
            clip_indices = self._clip_base_indices(target_base_index)
            clip_items = [super(VideoClipCocoDetection, self).__getitem__(base_idx) for base_idx in clip_indices]
            return {
                "clip": clip_items,
                "target": clip_items[self.clip_target_index],
                "clip_indices": clip_indices,
                "target_base_index": target_base_index,
            }

        def load_item(self, idx):
            return super().load_item(self._target_base_indices[idx])

    @register()
    class ClipBatchImageCollateFunction(BaseCollateFunction):  # type: ignore[no-redef]
        def __init__(self, scales=None, stop_epoch=None):
            super().__init__()
            self.scales = scales
            self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

        def __call__(self, items):
            flat_clip_items = [frame for item in items for frame in item["clip"]]
            target_items = [item["target"] for item in items]
            images = torch.cat([x[0][None] for x in flat_clip_items], dim=0)
            targets = [x[1] for x in target_items]

            if self.scales is not None and self.epoch < self.stop_epoch:
                size = self.scales[0] if isinstance(self.scales, (list, tuple)) else self.scales
                if isinstance(size, int):
                    size = [size, size]
                images = F.interpolate(images, size=size)

            return images, targets

    # RT-DETR's workspace stores the defining module and later calls
    # getattr(module, class_name). These classes are defined inside this
    # registration helper, so expose them on the runner module explicitly.
    globals()["VideoClipCocoDetection"] = VideoClipCocoDetection
    globals()["ClipBatchImageCollateFunction"] = ClipBatchImageCollateFunction


def main(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    if not (repo / "tools" / "train.py").exists():
        raise FileNotFoundError(f"RT-DETR repo root is invalid: {repo}")

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig, yaml_utils, register  # type: ignore[import-not-found]
    from src.misc import dist_utils  # type: ignore[import-not-found]
    from src.solver import TASKS  # type: ignore[import-not-found]

    register_clip_dataset_types(register)

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), "Only support from_scratch or resume or tuning at one time"

    custom_keys = {
        "update",
        "repo",
        "tuning_mode",
        "temporal_adapter",
        "temporal_adapter_index",
        "temporal_adapter_channels",
        "temporal_adapter_dim",
        "temporal_kernel_size",
        "temporal_residual_scale",
        "clip_length",
        "clip_loader_enabled",
        "target_index",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
    }
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in vars(args).items() if k not in custom_keys and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    original_setup = solver._setup

    def setup_with_policy() -> None:
        original_setup()
        adapter_info = None
        lora_info = None
        if args.lora_rank > 0:
            lora_info = inject_lora(solver.model, args, solver.device)
        if args.temporal_adapter:
            if args.clip_length > 1 and not args.clip_loader_enabled:
                raise RuntimeError(
                    "RT-DETR temporal adapter training with clip_length>1 requires a clip-aware dataloader. "
                    "Use clip_length=1 for adapter attachment smoke until the clip loader is enabled."
                )
            adapter_info = attach_temporal_adapter(solver.model, args, solver.device)
        report = apply_trainability_policy(solver.model, args.tuning_mode)
        if adapter_info is not None:
            report["adapter"] = adapter_info
        if lora_info is not None:
            report["lora"] = lora_info
        _write_policy_report(cfg.output_dir, report)
        print("VID-PEFT trainability policy:", json.dumps(report["parameter_summary"], sort_keys=True))

    solver._setup = setup_with_policy

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-t", "--tuning", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--tuning-mode", type=str, required=True)
    parser.add_argument("--temporal-adapter", action="store_true")
    parser.add_argument("--temporal-adapter-index", type=int, default=1)
    parser.add_argument("--temporal-adapter-channels", type=int, default=256)
    parser.add_argument("--temporal-adapter-dim", type=int, default=32)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--temporal-residual-scale", type=float, default=1.0)
    parser.add_argument("--clip-length", type=int, default=1)
    parser.add_argument("--clip-loader-enabled", action="store_true")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="encoder.0.layers.0.linear1,encoder.0.layers.0.linear2",
    )
    main(parser.parse_args())
