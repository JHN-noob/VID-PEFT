"""Publication-style tables and figures for VID-PEFT results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from src.shared.config import PROJECT_ROOT, make_experiment_id, project_relative_path
from src.shared.main_experiment import (
    build_budget_sweep_configs,
    build_clip_sweep_configs,
    build_frame_stability_specs,
    build_head_warmup_configs,
    build_main_experiment_configs,
    make_main_experiment_config,
)
from src.shared.prediction_export import load_run_config, resolve_path
from src.shared.result_aggregation import summarize_run


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "summaries" / "publication"
MODEL_LABELS = {
    "yolo_temporal": "YOLOv8m",
    "rtdetr_temporal": "RT-DETRv2-S",
}
MODEL_COLORS = {
    "YOLOv8m": (37, 99, 235),
    "RT-DETRv2-S": (220, 38, 38),
}
YOLO_PRIMARY_COLUMN = "metrics/mAP50-95(B)"
RTDETR_COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]
IMPORTANT_DETAIL_KEYS = [
    "best_yolo_metrics_map50_95_b",
    "final_yolo_metrics_map50_95_b",
    "best_yolo_metrics_map50_b",
    "final_yolo_metrics_map50_b",
    "best_yolo_metrics_precision_b",
    "final_yolo_metrics_precision_b",
    "best_yolo_metrics_recall_b",
    "final_yolo_metrics_recall_b",
    "best_rtdetr_coco_ap",
    "final_rtdetr_coco_ap",
    "best_rtdetr_coco_ap50",
    "final_rtdetr_coco_ap50",
    "best_rtdetr_coco_ap75",
    "final_rtdetr_coco_ap75",
    "best_rtdetr_coco_ap_small",
    "final_rtdetr_coco_ap_small",
    "best_rtdetr_coco_ap_medium",
    "final_rtdetr_coco_ap_medium",
    "best_rtdetr_coco_ap_large",
    "final_rtdetr_coco_ap_large",
]
BUDGET_ORDER = {"small": 0, "medium": 1, "large": 2}
CONDITION_ORDER = {
    "Head-only": 0,
    "Spatial full FT": 1,
    "Spatial+Temporal PEFT": 2,
}
CLIP_ORDER = {
    "T1_offline": 0,
    "T3_offline": 1,
    "T5_offline": 2,
    "T7_offline": 3,
    "T5_causal": 4,
}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.mean(items)) if items else 0.0


def _std(values: Iterable[float]) -> float:
    items = list(values)
    return float(statistics.pstdev(items)) if len(items) > 1 else 0.0


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _format_mean_std(values: list[float], digits: int = 4) -> str:
    if not values:
        return ""
    return f"{_mean(values):.{digits}f} +/- {_std(values):.{digits}f} (n={len(values)})"


def _format_int_mean_std(values: list[float]) -> str:
    if not values:
        return ""
    mean_value = _mean(values)
    std_value = _std(values)
    if std_value == 0:
        return f"{int(round(mean_value))} (n={len(values)})"
    return f"{mean_value:.1f} +/- {std_value:.1f} (n={len(values)})"


def _seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds_int = int(round(float(seconds)))
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60
    secs = seconds_int % 60
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def _training_time_from_stdout(run_dir: Path) -> float | None:
    path = run_dir / "official_train_stdout.log"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"Training time\s+([0-9:]+)", text)
    if not matches:
        return None
    parts = [int(part) for part in matches[-1].split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0])


def _policy_params(run_dir: Path) -> dict[str, float | None]:
    path = run_dir / "policy_report.json"
    if not path.exists():
        return {"total_parameters": None, "trainable_parameters": None, "trainable_ratio": None}
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    summary = report.get("parameter_summary", {})
    return {
        "total_parameters": summary.get("total_parameters"),
        "trainable_parameters": summary.get("trainable_parameters"),
        "trainable_ratio": summary.get("trainable_ratio"),
    }


def _metric_key(text: str) -> str:
    key = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return key or "metric"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _read_yolo_detail_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "results.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def primary(row: dict[str, str]) -> float:
        return float(row.get(YOLO_PRIMARY_COLUMN, 0.0) or 0.0)

    best = max(rows, key=primary)
    final = rows[-1]
    metrics: dict[str, float] = {}
    for prefix, row in (("best", best), ("final", final)):
        for column, value in row.items():
            number = _to_float(value)
            if number is None:
                continue
            metrics[f"{prefix}_yolo_{_metric_key(column)}"] = number
    return metrics


def _read_rtdetr_detail_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "log.txt"
    if not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}

    best = max(rows, key=lambda row: float(row["test_coco_eval_bbox"][0]))
    final = rows[-1]
    metrics: dict[str, float] = {}
    for prefix, row in (("best", best), ("final", final)):
        for key, value in row.items():
            if key == "test_coco_eval_bbox":
                for index, metric_name in enumerate(RTDETR_COCO_METRIC_NAMES):
                    if index < len(value):
                        metrics[f"{prefix}_rtdetr_coco_{_metric_key(metric_name)}"] = float(value[index])
                continue
            number = _to_float(value)
            if number is not None:
                metrics[f"{prefix}_rtdetr_{_metric_key(key)}"] = number
    return metrics


def _detail_metrics(run_dir: Path, model_family: str) -> dict[str, float]:
    if model_family == "yolo_temporal":
        return _read_yolo_detail_metrics(run_dir)
    if model_family == "rtdetr_temporal":
        return _read_rtdetr_detail_metrics(run_dir)
    return {}


def _run_dir(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / "outputs" / "runs" / make_experiment_id(config)


def _run_record(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = _run_dir(config)
    summary = summarize_run(run_dir, config=config)
    params = _policy_params(run_dir)
    train_time = summary.get("train_time_sec")
    if train_time is None:
        train_time = _training_time_from_stdout(run_dir)
    return {
        **summary,
        **params,
        "train_time_sec": train_time,
        **_detail_metrics(run_dir, str(summary["model_family"])),
    }


def _condition_label(tuning_mode: str) -> str:
    if tuning_mode == "head_only":
        return "Head-only"
    if tuning_mode == "spatial_only_full_ft":
        return "Spatial full FT"
    if tuning_mode == "spatial_temporal_peft":
        return "Spatial+Temporal PEFT"
    return tuning_mode


def _is_t5_offline_anchor(config: dict[str, Any]) -> bool:
    clip = dict(config.get("clip", {}) or {})
    return (
        str(config.get("tuning_mode")) == "spatial_temporal_peft"
        and int(clip.get("length", 1)) == 5
        and not bool(clip.get("causal", False))
    )


def _aggregate_records(records: list[dict[str, Any]], *, group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record[key] for key in group_keys), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        first = items[0]
        row = {group_key: value for group_key, value in zip(group_keys, key)}
        trainable_values = [
            float(item["trainable_parameters"])
            for item in items
            if item.get("trainable_parameters") is not None
        ]
        time_values = [
            float(item["train_time_sec"])
            for item in items
            if item.get("train_time_sec") is not None
        ]
        ratio_values = [
            float(item["trainable_ratio"])
            for item in items
            if item.get("trainable_ratio") is not None
        ]
        row.update(
            {
                "model": MODEL_LABELS.get(first["model_family"], first["model_family"]),
                "condition": _condition_label(first["tuning_mode"]),
                "metric": first["primary_metric"],
                "best": _format_mean_std([float(item["best_primary"]) for item in items]),
                "final": _format_mean_std([float(item["final_primary"]) for item in items]),
                "secondary_metric": first["secondary_metric"],
                "best_secondary": _format_mean_std([float(item["best_secondary"]) for item in items]),
                "final_secondary": _format_mean_std([float(item["final_secondary"]) for item in items]),
                "train_time": _format_mean_std(time_values, digits=1),
                "train_time_hms_mean": _seconds_to_hms(_mean(time_values)) if time_values else "",
                "trainable_params": _format_int_mean_std(trainable_values),
                "trainable_ratio": _format_mean_std(ratio_values, digits=6),
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda x: x["seed"])),
                "n": len(items),
            }
        )
        for metric_key in _detail_metric_keys(items):
            values = [
                float(item[metric_key])
                for item in items
                if item.get(metric_key) is not None
            ]
            if values:
                row[metric_key] = _format_mean_std(values, digits=6)
        rows.append(row)
    return rows


def _detail_metric_keys(records: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for record in records:
        for key in IMPORTANT_DETAIL_KEYS:
            if _to_float(record.get(key)) is not None and key not in keys:
                keys.append(key)
    return keys


def _detail_metric_row(record: dict[str, Any]) -> dict[str, str]:
    row: dict[str, str] = {}
    for key in _detail_metric_keys([record]):
        value = _to_float(record.get(key))
        if value is not None:
            row[key] = _format_float(value, digits=6)
    return row


def _main_records(seeds: Iterable[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    configs.extend(build_head_warmup_configs(seeds=seeds, dry_run=False))
    for seed in seeds:
        configs.extend(
            build_main_experiment_configs(
                seed=int(seed),
                dry_run=False,
                head_warmup_seed=int(seed),
            )
        )
    return [_run_record(config) for config in configs]


def table1_main(seeds: Iterable[int] = (0, 1, 2)) -> list[dict[str, Any]]:
    records = _main_records(seeds)
    rows = _aggregate_records(records, group_keys=["model_family", "tuning_mode", "budget", "clip"])
    rows.sort(key=lambda row: (row["model"], row["condition"]))
    return rows


def _budget_configs() -> list[dict[str, Any]]:
    configs = build_budget_sweep_configs(seed=0, include_medium=False, dry_run=False)
    configs.extend(
        [
            make_main_experiment_config(
                model_family,
                "spatial_temporal_peft",
                seed=0,
                budget="medium",
                epochs=5,
                dry_run=False,
            )
            for model_family in ("yolo_temporal", "rtdetr_temporal")
        ]
    )
    return configs


def table2_budget() -> list[dict[str, Any]]:
    records = [_run_record(config) for config in _budget_configs()]
    rows = _aggregate_records(records, group_keys=["model_family", "budget", "clip"])
    rows.sort(key=lambda row: (row["model"], BUDGET_ORDER.get(row["budget"], 99)))
    return rows


def _clip_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for config in build_clip_sweep_configs(seed=0, include_anchor=True, dry_run=False):
        if _is_t5_offline_anchor(config):
            config = make_main_experiment_config(
                config["model_family"],
                "spatial_temporal_peft",
                seed=0,
                budget="medium",
                epochs=5,
                dry_run=False,
            )
        configs.append(config)
    return configs


def table3_clip() -> list[dict[str, Any]]:
    records = [_run_record(config) for config in _clip_configs()]
    rows = _aggregate_records(records, group_keys=["model_family", "clip"])
    rows.sort(key=lambda row: (row["model"], CLIP_ORDER.get(row["clip"], 99)))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def table4_stability() -> list[dict[str, Any]]:
    specs = {
        Path(spec["prediction_jsonl"]).stem: spec
        for spec in build_frame_stability_specs(seed=0)
    }
    rows: list[dict[str, Any]] = []
    for row in _read_csv("outputs/frame_stability/reports/frame_stability_summary.csv"):
        stem = Path(row["prediction_path"]).stem
        spec = specs.get(stem)
        if spec is None:
            continue
        run_dir = resolve_path(spec["run_dir"])
        config = load_run_config(run_dir)
        run = _run_record(config)
        stability_row = {
                "model": MODEL_LABELS.get(run["model_family"], run["model_family"]),
                "clip": run["clip"],
                "metric": run["primary_metric"],
                "best": _format_mean_std([float(run["best_primary"])]),
                "final": _format_mean_std([float(run["final_primary"])]),
                "secondary_metric": run["secondary_metric"],
                "best_secondary": _format_mean_std([float(run["best_secondary"])]),
                "final_secondary": _format_mean_std([float(run["final_secondary"])]),
                "matched_iou_mean": _format_float(float(row["matched_iou_mean"])),
                "unmatched_rate_mean": _format_float(float(row["unmatched_rate_mean"])),
                "center_shift_norm_mean": _format_float(float(row["center_shift_norm_mean"]), digits=5),
                "area_log_ratio_abs_mean": _format_float(float(row["area_log_ratio_abs_mean"])),
                "score_delta_abs_mean": _format_float(float(row["score_delta_abs_mean"])),
                "detections_per_frame": _format_float(float(row["detections_per_frame_mean"])),
                "train_time": _format_mean_std([float(run["train_time_sec"])], digits=1)
                if run.get("train_time_sec") is not None
                else "",
                "train_time_hms_mean": _seconds_to_hms(run.get("train_time_sec")),
                "trainable_params": _format_int_mean_std([float(run["trainable_parameters"])])
                if run.get("trainable_parameters") is not None
                else "",
                "trainable_ratio": _format_mean_std([float(run["trainable_ratio"])], digits=6)
                if run.get("trainable_ratio") is not None
                else "",
                "seeds": str(run["seed"]),
                "n": 1,
            }
        stability_row.update(_detail_metric_row(run))
        rows.append(stability_row)
    rows.sort(key=lambda item: (item["model"], CLIP_ORDER.get(item["clip"], 99)))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _write_markdown(path: Path, tables: dict[str, list[dict[str, Any]]]) -> Path:
    sections = [
        "# VID-PEFT Publication Tables",
        "Tables are aggregate rows built from completed local run artifacts. Table 1 combines seed 0/1/2 as mean +/- std. Tables 2-4 are currently seed0-only ablations and are formatted with n=1.",
        "## Table 1. Main Results Across Seeds",
        _markdown_table(
            tables["table1_main"],
            [
                "model",
                "condition",
                "metric",
                "best",
                "final",
                "train_time_hms_mean",
                "trainable_params",
                "seeds",
            ],
        ),
        "## Table 2. Budget Sweep",
        _markdown_table(
            tables["table2_budget"],
            [
                "model",
                "budget",
                "metric",
                "best",
                "final",
                "train_time_hms_mean",
                "trainable_params",
                "seeds",
            ],
        ),
        "## Table 3. Clip Sweep",
        _markdown_table(
            tables["table3_clip"],
            [
                "model",
                "clip",
                "metric",
                "best",
                "final",
                "train_time_hms_mean",
                "trainable_params",
                "seeds",
            ],
        ),
        "## Table 4. Frame Stability",
        _markdown_table(
            tables["table4_stability"],
            [
                "model",
                "clip",
                "matched_iou_mean",
                "unmatched_rate_mean",
                "center_shift_norm_mean",
                "train_time_hms_mean",
                "trainable_params",
                "seeds",
            ],
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def _font(size: int = 18) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _value_from_stat(text: str) -> float:
    return float(str(text).split("+/-")[0].strip())


def _chart_canvas(title: str, width: int = 1280, height: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), title, fill=(20, 20, 20), font=_font(28))
    return image, draw


def _draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, y_max: float) -> None:
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left - 5, y, left, y), fill=(30, 30, 30), width=1)
        draw.text((left - 70, y - 8), f"{value:.2f}", fill=(60, 60, 60), font=_font(14))
        draw.line((left, y, right, y), fill=(230, 230, 230), width=1)


def _save_bar_figure(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    label_key: str,
    series: list[tuple[str, str, tuple[int, int, int]]],
) -> Path:
    image, draw = _chart_canvas(title)
    box = (90, 100, 1220, 610)
    values = [_value_from_stat(row[key]) for row in rows for _, key, _ in series]
    y_max = max(values) * 1.15 if values else 1.0
    _draw_axes(draw, box, y_max=y_max)
    left, top, right, bottom = box
    group_width = (right - left) / max(len(rows), 1)
    bar_width = min(38, group_width / (len(series) + 1.5))

    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        start = center - (len(series) * bar_width) / 2
        for series_index, (_, key, color) in enumerate(series):
            value = _value_from_stat(row[key])
            x0 = start + series_index * bar_width
            x1 = x0 + bar_width * 0.82
            y0 = bottom - (bottom - top) * value / y_max
            draw.rectangle((x0, y0, x1, bottom), fill=color)
        label = str(row[label_key])
        if row.get("model", "").startswith("RT"):
            label = "R-" + label
        elif row.get("model", "").startswith("YOLO"):
            label = "Y-" + label
        draw.text((center - 45, bottom + 14), label[:14], fill=(50, 50, 50), font=_font(13))

    legend_x = 900
    for index, (name, _, color) in enumerate(series):
        y = 35 + index * 26
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=color)
        draw.text((legend_x + 28, y - 2), name, fill=(40, 40, 40), font=_font(17))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _save_main_grouped_bar_figure(path: Path, rows: list[dict[str, Any]]) -> Path:
    image, draw = _chart_canvas("Figure 1. Main Best Detection Performance")
    box = (90, 100, 1220, 610)
    values = [_value_from_stat(row["best"]) for row in rows]
    y_max = max(values) * 1.15 if values else 1.0
    _draw_axes(draw, box, y_max=y_max)
    left, top, right, bottom = box

    conditions = sorted(
        {str(row["condition"]) for row in rows},
        key=lambda value: CONDITION_ORDER.get(value, 99),
    )
    row_map = {(row["model"], row["condition"]): row for row in rows}
    series = [
        ("YOLOv8m", "YOLOv8m", MODEL_COLORS["YOLOv8m"]),
        ("RT-DETRv2-S", "RT-DETRv2-S", MODEL_COLORS["RT-DETRv2-S"]),
    ]
    group_width = (right - left) / max(len(conditions), 1)
    bar_width = min(58, group_width / 3.8)

    for condition_index, condition in enumerate(conditions):
        center = left + group_width * (condition_index + 0.5)
        start = center - (len(series) * bar_width) / 2
        for series_index, (_, model, color) in enumerate(series):
            row = row_map.get((model, condition))
            if row is None:
                continue
            value = _value_from_stat(row["best"])
            x0 = start + series_index * bar_width
            x1 = x0 + bar_width * 0.84
            y0 = bottom - (bottom - top) * value / y_max
            draw.rectangle((x0, y0, x1, bottom), fill=color)
            draw.text((x0 - 1, y0 - 20), f"{value:.3f}", fill=color, font=_font(13))
        label = {
            "Head-only": "Head-only",
            "Spatial full FT": "Spatial FT",
            "Spatial+Temporal PEFT": "S+T PEFT",
        }.get(condition, condition)
        draw.text((center - 42, bottom + 14), label, fill=(50, 50, 50), font=_font(15))

    legend_x = 815
    for index, (name, _, color) in enumerate(series):
        y = 30 + index * 25
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=color)
        draw.text((legend_x + 28, y - 2), name, fill=(40, 40, 40), font=_font(16))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _save_line_figure(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str = "best",
    order: dict[str, int] | None = None,
    zoom_y: bool = False,
) -> Path:
    image, draw = _chart_canvas(title)
    box = (90, 100, 1220, 610)
    values = [_value_from_stat(row[y_key]) for row in rows]
    if zoom_y:
        raw_min = min(values) if values else 0.0
        raw_max = max(values) if values else 1.0
        raw_span = raw_max - raw_min
        padding = max(raw_span * 0.18, 0.004)
        y_min = max(0.0, raw_min - padding)
        y_max = raw_max + padding
    else:
        y_min = min(values) * 0.95 if values else 0.0
        y_max = max(values) * 1.05 if values else 1.0
    if math.isclose(y_min, y_max):
        y_min = 0.0
        y_max = max(1.0, y_max)
    left, top, right, bottom = box
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((left - 5, y, left, y), fill=(30, 30, 30), width=1)
        draw.text((left - 82, y - 8), f"{value:.3f}", fill=(60, 60, 60), font=_font(14))
        draw.line((left, y, right, y), fill=(230, 230, 230), width=1)

    x_values = sorted({str(row[x_key]) for row in rows}, key=lambda value: (order or {}).get(value, 99))
    x_pos = {
        value: left + (right - left) * idx / max(len(x_values) - 1, 1)
        for idx, value in enumerate(x_values)
    }
    for value, x in x_pos.items():
        draw.text((x - 35, bottom + 14), value, fill=(50, 50, 50), font=_font(15))

    for model in sorted({row["model"] for row in rows}):
        points = []
        for row in sorted([item for item in rows if item["model"] == model], key=lambda item: x_pos[str(item[x_key])]):
            value = _value_from_stat(row[y_key])
            x = x_pos[str(row[x_key])]
            y = bottom - (bottom - top) * (value - y_min) / (y_max - y_min)
            points.append((x, y, value))
        color = MODEL_COLORS.get(model, (80, 80, 80))
        if len(points) > 1:
            draw.line([(x, y) for x, y, _ in points], fill=color, width=4)
        for x, y, value in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            label_y = y - 22 if model == "YOLOv8m" else y + 8
            draw.text((x + 7, label_y), f"{value:.3f}", fill=color, font=_font(13))

    legend_x = 920
    for idx, (model, color) in enumerate(MODEL_COLORS.items()):
        y = 35 + idx * 26
        draw.line((legend_x, y + 9, legend_x + 25, y + 9), fill=color, width=4)
        draw.text((legend_x + 35, y - 2), model, fill=(40, 40, 40), font=_font(17))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _save_stability_figure(path: Path, rows: list[dict[str, Any]]) -> Path:
    image, draw = _chart_canvas("Figure 4. Frame Stability: Matched IoU by Clip")
    left, top, right, bottom = (90, 100, 1220, 585)
    clips = sorted({row["clip"] for row in rows}, key=lambda value: CLIP_ORDER.get(value, 99))
    models = ["YOLOv8m", "RT-DETRv2-S"]
    clip_labels = {
        "T1_offline": "T1",
        "T3_offline": "T3",
        "T5_offline": "T5",
        "T7_offline": "T7",
        "T5_causal": "C5",
    }
    values = [float(row["matched_iou_mean"]) for row in rows]
    if values:
        raw_min = min(values)
        raw_max = max(values)
        span = max(raw_max - raw_min, 0.02)
        y_min = max(0.0, raw_min - max(span * 0.35, 0.025))
        y_max = min(1.0, raw_max + max(span * 0.20, 0.025))
        if math.isclose(y_min, y_max):
            y_min = max(0.0, y_min - 0.05)
            y_max = min(1.0, y_max + 0.05)
    else:
        y_min, y_max = 0.0, 1.0

    def y_pos(value: float) -> float:
        return bottom - (bottom - top) * (value - y_min) / (y_max - y_min)

    draw.line((left, bottom, right, bottom), fill=(30, 30, 30), width=2)
    draw.line((left, top, left, bottom), fill=(30, 30, 30), width=2)
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        draw.line((left - 5, y, left, y), fill=(30, 30, 30), width=1)
        draw.text((left - 72, y - 8), f"{value:.2f}", fill=(60, 60, 60), font=_font(13))
        draw.line((left, y, right, y), fill=(230, 230, 230), width=1)

    group_width = (right - left) / max(len(models), 1)
    bar_width = min(42, group_width / (len(clips) + 2.4))
    for model_index, model in enumerate(models):
        center = left + group_width * (model_index + 0.5)
        start = center - (len(clips) * bar_width) / 2
        for clip_index, clip in enumerate(clips):
            match = next((row for row in rows if row["clip"] == clip and row["model"] == model), None)
            if match is None:
                continue
            value = float(match["matched_iou_mean"])
            x0 = start + clip_index * bar_width
            x1 = x0 + bar_width * 0.85
            y0 = y_pos(value)
            y1 = bottom
            draw.rectangle((x0, y0, x1, y1), fill=MODEL_COLORS.get(model, (80, 80, 80)))
            draw.text((x0 - 2, y0 - 18), f"{value:.3f}", fill=MODEL_COLORS.get(model, (60, 60, 60)), font=_font(12))
            draw.text((x0 + 4, bottom + 12), clip_labels.get(clip, clip), fill=(70, 70, 70), font=_font(12))
        draw.text((center - 55, bottom + 34), model, fill=(50, 50, 50), font=_font(16))

    draw.text((90, 645), "Y-axis: matched_iou_mean, zoomed scale (higher is more stable)", fill=(40, 40, 40), font=_font(16))
    legend_x = 910
    for idx, (model, color) in enumerate(MODEL_COLORS.items()):
        y = 35 + idx * 26
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=color)
        draw.text((legend_x + 28, y - 2), model, fill=(40, 40, 40), font=_font(17))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def write_publication_outputs(output_dir: str | Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Write Table 1-4 CSV/Markdown and Figure 1-4 PNG outputs."""

    output_dir = resolve_path(output_dir)
    tables = {
        "table1_main": table1_main(),
        "table2_budget": table2_budget(),
        "table3_clip": table3_clip(),
        "table4_stability": table4_stability(),
    }
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    paths = {
        "table1_csv": project_relative_path(_write_csv(table_dir / "table1_main_seed_combined.csv", tables["table1_main"])),
        "table2_csv": project_relative_path(_write_csv(table_dir / "table2_budget_seed_combined.csv", tables["table2_budget"])),
        "table3_csv": project_relative_path(_write_csv(table_dir / "table3_clip_seed_combined.csv", tables["table3_clip"])),
        "table4_csv": project_relative_path(_write_csv(table_dir / "table4_stability_seed_combined.csv", tables["table4_stability"])),
        "tables_md": project_relative_path(_write_markdown(table_dir / "publication_tables.md", tables)),
    }
    with (table_dir / "publication_tables.json").open("w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
        f.write("\n")
    paths["tables_json"] = project_relative_path(table_dir / "publication_tables.json")

    paths["figure1_png"] = str(
        project_relative_path(_save_main_grouped_bar_figure(
            figure_dir / "figure1_main_best_final.png",
            tables["table1_main"],
        ))
    )
    paths["figure2_png"] = str(
        project_relative_path(_save_line_figure(
            figure_dir / "figure2_budget_sweep.png",
            tables["table2_budget"],
            title="Figure 2. Budget Sweep: Best Detection Performance",
            x_key="budget",
            order=BUDGET_ORDER,
            zoom_y=True,
        ))
    )
    paths["figure3_png"] = str(
        project_relative_path(_save_line_figure(
            figure_dir / "figure3_clip_sweep.png",
            tables["table3_clip"],
            title="Figure 3. Clip Sweep: Best Detection Performance",
            x_key="clip",
            order=CLIP_ORDER,
            zoom_y=True,
        ))
    )
    paths["figure4_png"] = str(
        project_relative_path(_save_stability_figure(
            figure_dir / "figure4_frame_stability.png",
            tables["table4_stability"],
        ))
    )

    return {
        "output_dir": project_relative_path(output_dir),
        "paths": paths,
        "counts": {key: len(value) for key, value in tables.items()},
        "notes": {
            "table1": "seed 0/1/2 combined mean +/- std",
            "table2_4": "currently seed0-only ablations, formatted as aggregate rows with n=1",
            "figures": "PNG generated with PIL; no matplotlib/pandas dependency required",
        },
    }
