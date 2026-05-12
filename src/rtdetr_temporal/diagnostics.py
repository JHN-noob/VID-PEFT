"""Notebook-friendly diagnostics for the official RT-DETRv2 repository."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from src.shared.config import PROJECT_ROOT, project_relative_path

from .backend import (
    RTDETRV2_S_CHECKPOINT_URL,
    find_rtdetr_repo,
    official_rtdetrv2_s_config,
    resolve_local_checkpoint,
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }


def check_rtdetr_runtime(
    repo_path: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Return RT-DETR readiness details without installing or building anything."""

    blockers: list[str] = []
    try:
        repo = Path(repo_path) if repo_path is not None else find_rtdetr_repo()
        license_ok = (repo.parent / "LICENSE").exists()
        config_path = official_rtdetrv2_s_config(repo)
        repo_ok = True
    except Exception as exc:
        repo = None
        license_ok = False
        config_path = None
        repo_ok = False
        blockers.append(f"rtdetr_repo_unavailable: {type(exc).__name__}: {exc}")

    modules = {
        name: _module_available(name)
        for name in ("torch", "torchvision", "yaml", "scipy", "pycocotools", "faster_coco_eval", "tensorboard")
    }
    missing = [name for name, ok in modules.items() if not ok]
    if missing:
        blockers.append("missing_python_packages: " + ", ".join(missing))

    checkpoint_path = resolve_local_checkpoint(checkpoint, project_root=PROJECT_ROOT)
    checkpoint_ok = checkpoint_path.exists() if checkpoint_path is not None else False
    if checkpoint is not None and not checkpoint_ok:
        blockers.append(f"missing_rtdetr_checkpoint: {project_relative_path(checkpoint_path)}")

    return {
        "blockers": blockers,
        "repo": {
            "ok": repo_ok,
            "path": project_relative_path(repo) if repo is not None else None,
            "license_ok": license_ok,
            "base_config": project_relative_path(config_path) if config_path is not None else None,
        },
        "checkpoint": {
            "configured": str(checkpoint) if checkpoint is not None else None,
            "local_path": project_relative_path(checkpoint_path) if checkpoint_path is not None else None,
            "exists": checkpoint_ok,
            "official_url": RTDETRV2_S_CHECKPOINT_URL,
        },
        "modules": modules,
        "torch": _torch_status(),
    }
