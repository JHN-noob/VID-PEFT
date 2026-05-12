"""Aggregate VID-PEFT seed-repeat results from local run artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.shared.config import PROJECT_ROOT, make_experiment_id, project_relative_path, write_json
from src.shared.main_experiment import (
    MODEL_FAMILIES,
    build_head_warmup_configs,
    build_main_experiment_configs,
)
from src.shared.metrics import mean_std


YOLO_PRIMARY = "metrics/mAP50-95(B)"
YOLO_SECONDARY = "metrics/mAP50(B)"
RTDETR_PRIMARY_INDEX = 0
RTDETR_SECONDARY_INDEX = 1


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_config(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "config_snapshot.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip_label(config: dict[str, Any]) -> str:
    clip = dict(config.get("clip", {}) or {})
    length = int(clip.get("length", 1))
    return f"T{length}_{'causal' if clip.get('causal') else 'offline'}"


def _run_dir_from_config(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / "outputs" / "runs" / make_experiment_id(config)


def _parse_yolo_results(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"YOLO results.csv has no rows: {path}")
    for row in rows:
        row["_primary"] = float(row[YOLO_PRIMARY])
        row["_secondary"] = float(row[YOLO_SECONDARY])
    best = max(rows, key=lambda row: row["_primary"])
    final = rows[-1]
    return {
        "primary_metric": "mAP50-95",
        "secondary_metric": "mAP50",
        "best_epoch": int(float(best["epoch"])),
        "best_primary": float(best["_primary"]),
        "best_secondary": float(best["_secondary"]),
        "final_epoch": int(float(final["epoch"])),
        "final_primary": float(final["_primary"]),
        "final_secondary": float(final["_secondary"]),
        "train_time_sec": float(final.get("time", 0.0) or 0.0),
    }


def _parse_rtdetr_results(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "log.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"RT-DETR log.txt has no rows: {path}")
    best = max(rows, key=lambda row: float(row["test_coco_eval_bbox"][RTDETR_PRIMARY_INDEX]))
    final = rows[-1]
    return {
        "primary_metric": "AP",
        "secondary_metric": "AP50",
        "best_epoch": int(best["epoch"]),
        "best_primary": float(best["test_coco_eval_bbox"][RTDETR_PRIMARY_INDEX]),
        "best_secondary": float(best["test_coco_eval_bbox"][RTDETR_SECONDARY_INDEX]),
        "final_epoch": int(final["epoch"]),
        "final_primary": float(final["test_coco_eval_bbox"][RTDETR_PRIMARY_INDEX]),
        "final_secondary": float(final["test_coco_eval_bbox"][RTDETR_SECONDARY_INDEX]),
        "train_time_sec": None,
    }


def summarize_run(run_dir: str | Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse one completed run directory into a normalized summary row."""

    run_dir = _resolve(run_dir)
    config = config or _read_config(run_dir)
    if config is None:
        raise FileNotFoundError(f"Missing config_snapshot.json under {run_dir}")
    model_family = str(config["model_family"])
    if model_family == "yolo_temporal":
        metrics = _parse_yolo_results(run_dir)
    elif model_family == "rtdetr_temporal":
        metrics = _parse_rtdetr_results(run_dir)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")
    return {
        "experiment_id": run_dir.name,
        "run_dir": project_relative_path(run_dir),
        "model_family": model_family,
        "tuning_mode": str(config["tuning_mode"]),
        "budget": str(config["budget"]),
        "seed": int(config["seed"]),
        "clip": _clip_label(config),
        **metrics,
    }


def _target_configs_for_seed_summary(seeds: Iterable[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    configs.extend(build_head_warmup_configs(seeds=seeds, dry_run=False))
    for seed in seeds:
        configs.extend(build_main_experiment_configs(seed=int(seed), dry_run=False))
    return configs


def collect_seed_repeat_rows(
    *,
    seeds: Iterable[int] = (0, 1, 2),
    require_all: bool = False,
) -> list[dict[str, Any]]:
    """Collect normalized rows for head-only and main-comparison seed repeats."""

    rows: list[dict[str, Any]] = []
    for config in _target_configs_for_seed_summary(seeds):
        run_dir = _run_dir_from_config(config)
        try:
            rows.append(summarize_run(run_dir, config=config))
        except (FileNotFoundError, ValueError) as exc:
            if require_all:
                raise
            rows.append(
                {
                    "experiment_id": make_experiment_id(config),
                    "run_dir": project_relative_path(run_dir),
                    "model_family": str(config["model_family"]),
                    "tuning_mode": str(config["tuning_mode"]),
                    "budget": str(config["budget"]),
                    "seed": int(config["seed"]),
                    "clip": _clip_label(config),
                    "status": "missing",
                    "missing_reason": str(exc),
                }
            )
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model_family"]),
        str(row["tuning_mode"]),
        str(row["budget"]),
        str(row["clip"]),
    )


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    stats = mean_std(values)
    return {"mean": stats["mean"], "std": stats["std"], "n": len(values)}


def summarize_seed_repeats(
    *,
    seeds: Iterable[int] = (0, 1, 2),
    output_dir: str | Path = "outputs/summaries/seed_repeats",
    require_all: bool = False,
) -> dict[str, Any]:
    """Write per-run and mean/std summaries for repeated seeds."""

    rows = collect_seed_repeat_rows(seeds=seeds, require_all=require_all)
    completed = [row for row in rows if row.get("status") != "missing"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in completed:
        grouped.setdefault(_group_key(row), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (model_family, tuning_mode, budget, clip), items in sorted(grouped.items()):
        summary_rows.append(
            {
                "model_family": model_family,
                "tuning_mode": tuning_mode,
                "budget": budget,
                "clip": clip,
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda item: item["seed"])),
                "primary_metric": items[0]["primary_metric"],
                "secondary_metric": items[0]["secondary_metric"],
                "best_primary": _metric_summary([float(item["best_primary"]) for item in items]),
                "final_primary": _metric_summary([float(item["final_primary"]) for item in items]),
                "best_secondary": _metric_summary([float(item["best_secondary"]) for item in items]),
                "final_secondary": _metric_summary([float(item["final_secondary"]) for item in items]),
                "best_epoch": _metric_summary([float(item["best_epoch"]) for item in items]),
            }
        )

    output_root = _resolve(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    per_run_csv = _write_rows_csv(output_root / "seed_repeat_runs.csv", rows)
    summary_csv = _write_summary_csv(output_root / "seed_repeat_summary.csv", summary_rows)
    summary_json = write_json(
        output_root / "seed_repeat_summary.json",
        {
            "seeds": list(seeds),
            "runs": rows,
            "summary": summary_rows,
            "notes": {
                "missing_rows": sum(1 for row in rows if row.get("status") == "missing"),
                "head_only_source": "seed-specific head warmup run",
                "budget_clip_sweeps": "not included by default",
            },
        },
    )
    return {
        "runs": rows,
        "summary": summary_rows,
        "per_run_csv": str(per_run_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "experiment_id",
        "model_family",
        "tuning_mode",
        "budget",
        "clip",
        "seed",
        "status",
        "primary_metric",
        "secondary_metric",
        "best_epoch",
        "best_primary",
        "best_secondary",
        "final_epoch",
        "final_primary",
        "final_secondary",
        "train_time_sec",
        "run_dir",
        "missing_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _format_stat(value: dict[str, float | int]) -> str:
    return f"{float(value['mean']):.6f} +/- {float(value['std']):.6f} (n={int(value['n'])})"


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "model_family",
        "tuning_mode",
        "budget",
        "clip",
        "seeds",
        "primary_metric",
        "best_primary",
        "final_primary",
        "secondary_metric",
        "best_secondary",
        "final_secondary",
        "best_epoch",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_family": row["model_family"],
                    "tuning_mode": row["tuning_mode"],
                    "budget": row["budget"],
                    "clip": row["clip"],
                    "seeds": row["seeds"],
                    "primary_metric": row["primary_metric"],
                    "best_primary": _format_stat(row["best_primary"]),
                    "final_primary": _format_stat(row["final_primary"]),
                    "secondary_metric": row["secondary_metric"],
                    "best_secondary": _format_stat(row["best_secondary"]),
                    "final_secondary": _format_stat(row["final_secondary"]),
                    "best_epoch": _format_stat(row["best_epoch"]),
                }
            )
    return path
