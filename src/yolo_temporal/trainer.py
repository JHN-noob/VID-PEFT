"""Ultralytics trainer factories for VID-PEFT adapter smoke runs."""

from __future__ import annotations

import json
from typing import Any

from src.shared.metrics import parameter_summary

from .adapters import attach_yolo_p4_temporal_adapter
from .clip_dataset import YOLOClipDataset


def _base_model(model):  # type: ignore[no-untyped-def]
    return model.module if hasattr(model, "module") else model


def _enable_layer_adapter(model, *, layer_index: int = 18) -> None:  # type: ignore[no-untyped-def]
    base = _base_model(model)
    layers = getattr(base, "model", None)
    if layers is None:
        return
    for layer in layers:
        adapter = getattr(layer, "temporal_adapter_p4", None)
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = True


def _write_policy_report(trainer, *, tuning_mode: str, adapter: dict[str, Any] | None = None) -> None:  # type: ignore[no-untyped-def]
    report: dict[str, Any] = {
        "tuning_mode": tuning_mode,
        "adapter": adapter or {},
        "parameter_summary": parameter_summary(trainer.model),
        "trainable_names": [
            name for name, param in trainer.model.named_parameters() if param.requires_grad
        ],
    }
    trainer.save_dir.mkdir(parents=True, exist_ok=True)
    (trainer.save_dir / "policy_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_policy_recording_detection_trainer(*, tuning_mode: str = "head_only"):
    """Return a DetectionTrainer subclass that records trainability."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    class PolicyRecordingDetectionTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            _write_policy_report(self, tuning_mode=tuning_mode)

    return PolicyRecordingDetectionTrainer


def make_head_only_detection_trainer(*, tuning_mode: str = "head_only"):
    """Return a DetectionTrainer subclass that records head-only trainability."""

    return make_policy_recording_detection_trainer(tuning_mode=tuning_mode)


def make_p4_temporal_detection_trainer(
    *,
    tuning_mode: str = "spatial_temporal_peft",
    layer_index: int = 18,
    channels: int | None = None,
    adapter_dim: int = 32,
    kernel_size: int = 3,
    clip_length: int = 1,
    clip_loader_enabled: bool = False,
    causal: bool = False,
    target: str = "center",
    target_index: int = 0,
    boundary: str = "clamp",
    residual_scale: float = 1.0,
    placement: str = "layer_local_forward_hook",
):
    """Return a DetectionTrainer subclass that attaches a top-level P4 adapter."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    class P4TemporalDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            dataset = super().build_dataset(img_path=img_path, mode=mode, batch=batch)
            if clip_loader_enabled and clip_length > 1:
                return YOLOClipDataset(
                    dataset,
                    clip_length=clip_length,
                    causal=causal,
                    target=target,
                    boundary=boundary,
                )
            return dataset

        def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            attach_yolo_p4_temporal_adapter(
                model,
                layer_index=layer_index,
                channels=channels,
                adapter_dim=adapter_dim,
                kernel_size=kernel_size,
                clip_length=clip_length,
                target_index=target_index,
                residual_scale=residual_scale,
                placement=placement,
            )
            return model

        def build_optimizer(self, model, *args, **kwargs):  # type: ignore[no-untyped-def]
            _enable_layer_adapter(model, layer_index=layer_index)
            return super().build_optimizer(model, *args, **kwargs)

        def _setup_train(self) -> None:
            if clip_length > 1:
                if not clip_loader_enabled:
                    raise RuntimeError(
                        "YOLO temporal adapter training with clip_length>1 requires yolo.clip_loader.enabled=true."
                    )
                if placement != "detect_pre_forward":
                    raise RuntimeError(
                        "YOLO target-frame clip training requires adapter placement='detect_pre_forward'."
                    )
            super()._setup_train()
            _enable_layer_adapter(self.model, layer_index=layer_index)
            _write_policy_report(
                self,
                tuning_mode=tuning_mode,
                adapter=getattr(_base_model(self.model), "_vid_peft_temporal_adapter_info", {}),
            )

    return P4TemporalDetectionTrainer
