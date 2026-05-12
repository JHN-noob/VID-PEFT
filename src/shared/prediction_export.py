"""Common helpers for frame-stability prediction exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.shared.config import PROJECT_ROOT
from src.shared.detection_export import read_jsonl, resolve_project_path
from src.shared.clip_sampler import group_records_by_video


def resolve_path(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a project-relative path without touching the filesystem."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """Load ``config_snapshot.json`` from an experiment run directory."""

    path = resolve_path(run_dir) / "config_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run config snapshot: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_checkpoint(
    run_dir: str | Path,
    *,
    candidates: Iterable[str],
) -> Path:
    """Return the first existing checkpoint under ``run_dir``."""

    root = resolve_path(run_dir)
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    names = ", ".join(candidates)
    raise FileNotFoundError(f"No checkpoint found under {root}; tried: {names}")


def select_specs(
    specs: Iterable[dict[str, Any]],
    *,
    model_family: str | None = None,
    missing_only: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Filter frame-stability specs for one model family or missing outputs."""

    selected: list[dict[str, Any]] = []
    for spec in specs:
        if model_family is not None and spec.get("model_family") != model_family:
            continue
        prediction_path = resolve_path(spec["prediction_jsonl"], project_root=project_root)
        if missing_only and prediction_path.exists():
            continue
        selected.append(spec)
    return selected


def load_manifest_records(
    manifest_path: str | Path,
    *,
    max_videos: int | None = None,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Load manifest records with optional deterministic truncation."""

    records = read_jsonl(resolve_path(manifest_path))
    grouped = group_records_by_video(records)
    selected: list[dict[str, Any]] = []
    for video_index, video_id in enumerate(sorted(grouped, key=str)):
        if max_videos is not None and video_index >= max_videos:
            break
        for record in grouped[video_id]:
            selected.append(record)
            if max_frames is not None and len(selected) >= max_frames:
                return selected
    return selected


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write records to UTF-8 JSONL."""

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """Write a compact JSON report."""

    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def image_path_from_record(record: dict[str, Any]) -> Path:
    """Resolve the source image path from a YouTube-VIS manifest record."""

    return resolve_project_path(record["image_path"])
