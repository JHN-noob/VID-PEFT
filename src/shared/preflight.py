"""Preflight checks for paths, licenses, and optional training dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import platform
import sys
from typing import Iterable

from .config import PROJECT_ROOT, ExperimentConfig, load_experiment_config, project_relative_path, read_json


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_path(name: str, path: Path, required: bool = True) -> CheckResult:
    return CheckResult(name=name, ok=path.exists(), detail=project_relative_path(path), required=required)


def _format_result(result: CheckResult) -> str:
    status = "OK" if result.ok else ("MISS" if result.required else "WARN")
    return f"[{status}] {result.name}: {result.detail}"


def _default_paths(config: ExperimentConfig | dict | None, project_root: Path) -> dict[str, Path]:
    if config is None:
        return {
            "dataset_root": project_root / "data" / "VIS2021",
            "rtdetr_repo": project_root / "third_party" / "RT-DETR",
            "ultralytics_repo": project_root / "third_party" / "ultralytics",
        }

    if isinstance(config, dict):
        paths = config.get("paths", {})
        dataset_root = paths.get("youtube_vis_root") or paths.get("dataset_root") or "data/VIS2021"
        rtdetr_repo = paths.get("rtdetr_repo", "third_party/RT-DETR")
        ultralytics_repo = paths.get("ultralytics_repo", "third_party/ultralytics")
        return {
            "dataset_root": _resolve(project_root, dataset_root),
            "rtdetr_repo": _resolve(project_root, rtdetr_repo),
            "ultralytics_repo": _resolve(project_root, ultralytics_repo),
        }

    return {
        "dataset_root": config.dataset_root() or project_root / "data" / "VIS2021",
        "rtdetr_repo": config.resolved_path("paths", "rtdetr_repo", default="third_party/RT-DETR")
        or project_root / "third_party" / "RT-DETR",
        "ultralytics_repo": config.resolved_path("paths", "ultralytics_repo", default="third_party/ultralytics")
        or project_root / "third_party" / "ultralytics",
    }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_preflight_config(config_or_path: str | Path | dict, project_root: Path) -> ExperimentConfig | dict:
    if isinstance(config_or_path, dict):
        data = config_or_path
    else:
        data = read_json(config_or_path)
    if isinstance(data, dict) and "model_family" in data:
        return load_experiment_config(data, project_root=project_root)
    return data


def check_environment(
    config_or_path: str | Path | dict | None = None,
    *,
    project_root: str | Path | None = None,
    print_report: bool = True,
    raise_on_missing: bool = False,
) -> list[CheckResult]:
    """Check local prerequisites without installing or modifying anything."""

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    config = _load_preflight_config(config_or_path, root) if config_or_path is not None else None
    paths = _default_paths(config, root)

    rtdetr_repo = paths["rtdetr_repo"]
    ultralytics_repo = paths["ultralytics_repo"]
    model_family = config.model_family if isinstance(config, ExperimentConfig) else None
    raw_config = config.to_dict() if isinstance(config, ExperimentConfig) else (config if isinstance(config, dict) else {})

    extra_results: list[CheckResult] = []
    if model_family == "yolo_temporal":
        weight = ((raw_config.get("yolo") or {}).get("weights") or "yolov8m.pt")
        extra_results.append(_check_path("YOLO local weights", _resolve(root, weight), required=True))
        extra_results.append(
            CheckResult("polars module", _module_available("polars"), "required by official Ultralytics", required=True)
        )
    elif model_family == "rtdetr_temporal":
        checkpoint = ((raw_config.get("rtdetr") or {}).get("checkpoint"))
        if checkpoint:
            extra_results.append(_check_path("RT-DETR local checkpoint", _resolve(root, checkpoint), required=True))

    results = [
        CheckResult("python", True, f"{platform.python_version()} ({Path(sys.executable).name})"),
        _check_path("project root", root),
        _check_path("dataset root", paths["dataset_root"], required=False),
        _check_path("RT-DETR repo", rtdetr_repo),
        _check_path("RT-DETR LICENSE", rtdetr_repo / "LICENSE"),
        _check_path("RT-DETR PyTorch train.py", rtdetr_repo / "rtdetrv2_pytorch" / "tools" / "train.py"),
        _check_path("Ultralytics repo", ultralytics_repo),
        _check_path("Ultralytics LICENSE", ultralytics_repo / "LICENSE"),
        CheckResult("numpy module", _module_available("numpy"), "required for array utilities", required=False),
        CheckResult("torch module", _module_available("torch"), "required for training", required=False),
        CheckResult("cv2 module", _module_available("cv2"), "optional for visualization", required=False),
        CheckResult("yaml module", _module_available("yaml"), "required by official RT-DETR YAML configs", required=False),
        CheckResult("scipy module", _module_available("scipy"), "required by RT-DETR matcher", required=False),
        CheckResult("pycocotools module", _module_available("pycocotools"), "required by COCO-style evaluation", required=False),
        CheckResult("faster_coco_eval module", _module_available("faster_coco_eval"), "required by RT-DETR COCO loader", required=False),
        CheckResult("tensorboard module", _module_available("tensorboard"), "required by RT-DETR training logger", required=False),
    ] + extra_results

    if print_report:
        for result in results:
            print(_format_result(result))

    missing_required = [result for result in results if result.required and not result.ok]
    if raise_on_missing and missing_required:
        names = ", ".join(result.name for result in missing_required)
        raise FileNotFoundError(f"Missing required preflight items: {names}")

    return results


def missing_required(results: Iterable[CheckResult]) -> list[CheckResult]:
    return [result for result in results if result.required and not result.ok]
