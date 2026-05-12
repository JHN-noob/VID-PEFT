"""Result persistence and seed aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import project_relative_path
from .metrics import mean_std


def save_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, item: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def collect_run_metrics(runs_root: str | Path) -> list[dict[str, Any]]:
    runs_root = Path(runs_root)
    metrics: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/metrics.json")):
        item = load_metrics(metrics_path)
        item["_metrics_path"] = project_relative_path(metrics_path)
        metrics.append(item)
    return metrics


def summarize_seed_results(records: Iterable[dict[str, Any]], metric_keys: Iterable[str]) -> dict[str, Any]:
    records = list(records)
    summary: dict[str, Any] = {"num_runs": len(records), "metrics": {}}
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record and record[key] is not None]
        summary["metrics"][key] = mean_std(values)
    return summary


def summarize_results(runs_root: str | Path, metric_keys: Iterable[str] = ("mAP", "AP50", "AP75")) -> dict[str, Any]:
    return summarize_seed_results(collect_run_metrics(runs_root), metric_keys)
