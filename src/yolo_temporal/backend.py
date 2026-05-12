"""Wrapper utilities for the official Ultralytics repository."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from src.shared.config import PROJECT_ROOT


def find_ultralytics_repo(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    repo = root / "third_party" / "ultralytics"
    if (repo / "LICENSE").exists() and (repo / "ultralytics" / "__init__.py").exists():
        return repo
    raise FileNotFoundError("Ultralytics repo not found under third_party/ultralytics.")


def prepare_ultralytics_repo(repo_path: str | Path | None = None) -> Path:
    repo = Path(repo_path) if repo_path is not None else find_ultralytics_repo()
    if not (repo / "LICENSE").exists():
        raise FileNotFoundError(f"Ultralytics LICENSE not found: {repo / 'LICENSE'}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def load_yolo_class(repo_path: str | Path | None = None):
    prepare_ultralytics_repo(repo_path)
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "Could not import YOLO from third_party/ultralytics. "
            "Check torch and Ultralytics runtime dependencies in the notebook kernel."
        ) from exc
    return YOLO


def build_yolo_model(weights_or_yaml: str = "yolov8m.pt", repo_path: str | Path | None = None) -> Any:
    YOLO = load_yolo_class(repo_path)
    return YOLO(weights_or_yaml)
