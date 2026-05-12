"""Temporal adapter modules for YOLO feature tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@dataclass(frozen=True)
class AdapterConfig:
    channels: int
    adapter_dim: int = 32
    kernel_size: int = 3
    residual_scale: float = 1.0


if nn is not None:

    class TemporalConvAdapter(nn.Module):
        """Bottleneck 3D temporal adapter for tensors shaped [B, T, C, H, W]."""

        def __init__(
            self,
            channels: int,
            adapter_dim: int = 32,
            kernel_size: int = 3,
            residual_scale: float = 1.0,
        ):
            super().__init__()
            if channels < 1 or adapter_dim < 1:
                raise ValueError("channels and adapter_dim must be >= 1.")
            padding = kernel_size // 2
            self.residual_scale = residual_scale
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

        def forward(self, x):  # type: ignore[no-untyped-def]
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


    class MultiScaleTemporalAdapter(nn.Module):
        """Named multi-scale temporal adapters, e.g. P3/P4/P5."""

        def __init__(self, configs: dict[str, AdapterConfig]):
            super().__init__()
            self.adapters = nn.ModuleDict(
                {
                    name: TemporalConvAdapter(
                        channels=cfg.channels,
                        adapter_dim=cfg.adapter_dim,
                        kernel_size=cfg.kernel_size,
                        residual_scale=cfg.residual_scale,
                    )
                    for name, cfg in configs.items()
                }
            )

        def forward(self, features: dict[str, object]) -> dict[str, object]:
            return {
                name: self.adapters[name](value) if name in self.adapters else value
                for name, value in features.items()
            }


    class P4TemporalAdapterHook:
        """Pickle-safe forward hook that applies a layer-local temporal adapter."""

        def __init__(self, *, adapter_attr: str = "temporal_adapter_p4", clip_length: int = 1):
            self.adapter_attr = adapter_attr
            self.clip_length = int(clip_length)

        def __call__(self, module, _inputs, output):  # type: ignore[no-untyped-def]
            adapter = getattr(module, self.adapter_attr)
            return apply_temporal_adapter_to_flat_feature(
                output,
                adapter,
                clip_length=self.clip_length,
            )


    class P4TemporalDetectPreHook:
        """Pickle-safe Detect pre-hook for target-frame clip training."""

        def __init__(
            self,
            *,
            adapter_attr: str = "temporal_adapter_p4",
            clip_length: int = 1,
            target_index: int = 0,
            feature_index: int = 1,
        ):
            self.adapter_attr = adapter_attr
            self.clip_length = int(clip_length)
            self.target_index = int(target_index)
            self.feature_index = int(feature_index)

        def __call__(self, module, inputs):  # type: ignore[no-untyped-def]
            features = inputs[0]
            if not isinstance(features, (list, tuple)):
                raise ValueError("YOLO Detect pre-hook expects a list of feature maps.")
            adapter = getattr(module, self.adapter_attr)
            selected = []
            for index, feature in enumerate(features):
                if index == self.feature_index:
                    selected.append(
                        apply_temporal_adapter_to_flat_feature(
                            feature,
                            adapter,
                            clip_length=self.clip_length,
                            target_index=self.target_index,
                        )
                    )
                else:
                    selected.append(
                        select_target_from_flat_feature(
                            feature,
                            clip_length=self.clip_length,
                            target_index=self.target_index,
                        )
                    )
            return (selected,)


    def apply_temporal_adapter_to_flat_feature(
        feature,
        adapter: TemporalConvAdapter,
        *,
        clip_length: int = 1,
        target_index: int | None = None,
    ):
        """Apply a [B,T,C,H,W] adapter to a flattened [B*T,C,H,W] feature map."""

        if feature.ndim != 4:
            raise ValueError("Expected a feature tensor shaped [B*T, C, H, W].")
        if clip_length < 1:
            raise ValueError("clip_length must be >= 1.")
        batch_time, channels, height, width = feature.shape
        if batch_time % clip_length != 0:
            if batch_time == 1 and clip_length > 1:
                return feature
            raise ValueError(
                f"Feature batch {batch_time} is not divisible by clip_length={clip_length}."
            )
        batch = batch_time // clip_length
        clip = feature.reshape(batch, clip_length, channels, height, width)
        adapted = adapter(clip)
        if target_index is not None:
            return adapted[:, int(target_index), :, :, :].contiguous()
        return adapted.reshape(batch_time, channels, height, width)


    def select_target_from_flat_feature(
        feature,
        *,
        clip_length: int = 1,
        target_index: int = 0,
    ):
        """Select target-frame features from a flattened clip feature tensor."""

        if feature.ndim != 4:
            raise ValueError("Expected a feature tensor shaped [B*T, C, H, W].")
        batch_time, channels, height, width = feature.shape
        if batch_time % clip_length != 0:
            if batch_time == 1 and clip_length > 1:
                return feature
            raise ValueError(
                f"Feature batch {batch_time} is not divisible by clip_length={clip_length}."
            )
        batch = batch_time // clip_length
        clip = feature.reshape(batch, clip_length, channels, height, width)
        return clip[:, int(target_index), :, :, :].contiguous()


    def infer_yolo_layer_channels(layer: Any) -> int:
        """Best-effort channel inference for common Ultralytics modules."""

        for path in (("cv2", "conv"), ("conv",), ("cv1", "conv")):
            current = layer
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    break
            if current is not None and hasattr(current, "out_channels"):
                return int(current.out_channels)
        raise ValueError(f"Could not infer output channels from layer {layer!r}.")


    def attach_yolo_p4_temporal_adapter(
        model,
        *,
        layer_index: int = 18,
        detect_layer_index: int | None = None,
        channels: int | None = None,
        adapter_dim: int = 32,
        kernel_size: int = 3,
        clip_length: int = 1,
        target_index: int = 0,
        residual_scale: float = 1.0,
        placement: str = "layer_local_forward_hook",
    ) -> dict[str, Any]:
        """Attach a top-level P4 temporal adapter hook to an Ultralytics model.

        The adapter is registered as a real submodule on the P4 target layer so
        Ultralytics checkpoints can pickle the model. The custom trainer
        re-enables this adapter after Ultralytics applies ``freeze=22``.
        """

        layers = getattr(model, "model", None)
        if layers is None or layer_index >= len(layers):
            raise ValueError(f"YOLO model does not have layer index {layer_index}.")
        p4_target = layers[layer_index]
        channels = int(channels or infer_yolo_layer_channels(p4_target))

        for layer in layers:
            for hook_id, hook in list(getattr(layer, "_forward_hooks", {}).items()):
                if isinstance(hook, P4TemporalAdapterHook):
                    del layer._forward_hooks[hook_id]
            for hook_id, hook in list(getattr(layer, "_forward_pre_hooks", {}).items()):
                if isinstance(hook, P4TemporalDetectPreHook):
                    del layer._forward_pre_hooks[hook_id]

        adapter = TemporalConvAdapter(
            channels=channels,
            adapter_dim=adapter_dim,
            kernel_size=kernel_size,
            residual_scale=residual_scale,
        )
        if placement == "layer_local_forward_hook":
            p4_target.temporal_adapter_p4 = adapter
            p4_target.register_forward_hook(P4TemporalAdapterHook(clip_length=clip_length))
        elif placement == "detect_pre_forward":
            detect_layer_index = int(detect_layer_index if detect_layer_index is not None else len(layers) - 1)
            detect = layers[detect_layer_index]
            detect.temporal_adapter_p4 = adapter
            detect.register_forward_pre_hook(
                P4TemporalDetectPreHook(
                    clip_length=clip_length,
                    target_index=target_index,
                    feature_index=1,
                )
            )
        else:
            raise ValueError(f"Unsupported YOLO temporal adapter placement: {placement!r}")
        model._vid_peft_temporal_adapter_info = {
            "feature_level": "P4",
            "layer_index": int(layer_index),
            "channels": channels,
            "adapter_dim": int(adapter_dim),
            "kernel_size": int(kernel_size),
            "clip_length": int(clip_length),
            "target_index": int(target_index),
            "residual_scale": float(residual_scale),
            "placement": placement,
        }
        return dict(model._vid_peft_temporal_adapter_info)

else:

    class TemporalConvAdapter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to instantiate TemporalConvAdapter.")

    class MultiScaleTemporalAdapter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to instantiate MultiScaleTemporalAdapter.")

    class P4TemporalAdapterHook:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to instantiate P4TemporalAdapterHook.")

    class P4TemporalDetectPreHook:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required to instantiate P4TemporalDetectPreHook.")

    def apply_temporal_adapter_to_flat_feature(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required to apply temporal adapters.")

    def select_target_from_flat_feature(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required to select temporal target features.")

    def infer_yolo_layer_channels(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required to inspect YOLO layers.")

    def attach_yolo_p4_temporal_adapter(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("torch is required to attach temporal adapters.")
