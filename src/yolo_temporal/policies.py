"""Trainability policies for YOLO temporal adapter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.shared.metrics import parameter_summary


HEAD_KEYWORDS = ("detect", "head", "cv2", "cv3", "dfl")
TEMPORAL_KEYWORDS = ("temporal", "adapter")
SPATIAL_KEYWORDS = ("backbone", "neck", "bottleneck", "c2f", "conv")


@dataclass(frozen=True)
class PolicyReport:
    tuning_mode: str
    trainable_names: list[str]
    parameter_summary: dict[str, float]


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


def apply_yolo_tuning_policy(model, tuning_mode: str) -> PolicyReport:
    if tuning_mode == "spatial_only_full_ft":
        _set_all_requires_grad(model, True)
        trainable = [name for name, _ in model.named_parameters()]
        return PolicyReport(tuning_mode, trainable, parameter_summary(model))

    _set_all_requires_grad(model, False)
    if tuning_mode == "head_only":
        trainable = _enable_by_keywords(model, HEAD_KEYWORDS)
    elif tuning_mode == "spatial_temporal_peft":
        trainable = _enable_by_keywords(model, TEMPORAL_KEYWORDS + SPATIAL_KEYWORDS)
    else:
        raise ValueError(f"Unsupported YOLO tuning mode: {tuning_mode}")
    return PolicyReport(tuning_mode, trainable, parameter_summary(model))
