"""Lightweight metrics helpers for experiment bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any, Callable, Iterable


def count_parameters(model: Any) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return sum(int(param.numel()) for param in model.parameters())


def count_trainable_parameters(model: Any) -> int:
    if not hasattr(model, "parameters"):
        return 0
    return sum(int(param.numel()) for param in model.parameters() if getattr(param, "requires_grad", False))


def parameter_summary(model: Any) -> dict[str, float]:
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    ratio = trainable / total if total else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": ratio,
    }


def trainable_alignment(
    left_trainable: int,
    right_trainable: int,
    *,
    tolerance: float = 0.10,
) -> dict[str, float | bool]:
    """Compare two trainable-parameter counts against a relative tolerance."""

    larger = max(int(left_trainable), int(right_trainable))
    smaller = min(int(left_trainable), int(right_trainable))
    diff = larger - smaller
    relative_diff = diff / larger if larger else 0.0
    return {
        "left_trainable": int(left_trainable),
        "right_trainable": int(right_trainable),
        "absolute_diff": diff,
        "relative_diff": relative_diff,
        "tolerance": tolerance,
        "aligned": relative_diff <= tolerance,
    }


def get_vram_usage_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


@dataclass(frozen=True)
class LatencyResult:
    latency_ms_mean: float
    latency_ms_std: float
    fps: float
    repeats: int


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 20,
    sync: Callable[[], Any] | None = None,
) -> LatencyResult:
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        durations.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(durations) if durations else 0.0
    std_ms = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return LatencyResult(mean_ms, std_ms, fps, repeats)


def format_epoch_log(epoch: int, train_loss: float, val_loss: float, val_acc: float, lr: float) -> str:
    return (
        f"epoch={epoch} "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f} "
        f"val_acc={val_acc:.6f} "
        f"lr={lr:.8f}"
    )


def mean_std(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }
