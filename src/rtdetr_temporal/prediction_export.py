"""Subprocess wrapper for RT-DETR frame-stability prediction export."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from src.shared.clip_sampler import clip_target_index
from src.shared.config import project_relative_path, project_relative_text
from src.shared.main_experiment import build_frame_stability_specs
from src.shared.prediction_export import find_checkpoint, load_run_config, resolve_path, select_specs, write_json

from .backend import prepare_rtdetr_repo


def _build_predict_command(
    *,
    run_config: dict[str, Any],
    run_dir: Path,
    checkpoint: Path,
    prediction_path: Path,
    repo: Path,
    score_threshold: float,
    max_frames: int | None,
    max_videos: int | None,
    device: str | None,
) -> list[str]:
    rtdetr = dict(run_config.get("rtdetr", {}) or {})
    adapter = dict(rtdetr.get("adapter", {}) or {})
    lora = dict(rtdetr.get("lora", {}) or {})
    clip = dict(run_config.get("clip", {}) or {})
    train_args = dict(rtdetr.get("train_args", {}) or {})

    clip_length = int(clip.get("length", 1))
    causal = bool(clip.get("causal", False))
    target = str(clip.get("target", "center"))
    target_index = clip_target_index(clip_length, causal=causal, target=target)
    generated_config = resolve_path(rtdetr.get("generated_config", run_dir / "rtdetrv2_s_youtube_vis.yml"))
    if not generated_config.exists():
        raise FileNotFoundError(f"Missing generated RT-DETR YAML: {generated_config}")

    command = [
        sys.executable,
        str((Path(__file__).resolve().parent / "official_predict_with_policy.py").resolve()),
        "-c",
        str(generated_config.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-jsonl",
        str(prediction_path.resolve()),
        "--meta-json",
        str(prediction_path.with_suffix(".meta.json").resolve()),
        "--repo",
        str(repo.resolve()),
        "--tuning-mode",
        str(run_config.get("tuning_mode", "spatial_temporal_peft")),
        "--clip-length",
        str(clip_length),
        "--target-index",
        str(target_index),
        "--score-threshold",
        str(float(score_threshold)),
        "--output-dir",
        str(run_dir.resolve()),
        "--seed",
        str(int(run_config.get("seed", 0))),
    ]
    if device:
        command.extend(["--device", str(device)])
    elif train_args.get("device"):
        command.extend(["--device", str(train_args["device"])])

    clip_loader = dict(rtdetr.get("clip_loader", {}) or {})
    if bool(clip_loader.get("enabled", False)):
        command.append("--clip-loader-enabled")

    if adapter:
        command.append("--temporal-adapter")
        command.extend(["--temporal-adapter-index", str(int(adapter.get("feature_index", 1)))])
        command.extend(["--temporal-adapter-channels", str(int(adapter.get("channels", 256)))])
        command.extend(["--temporal-adapter-dim", str(int(adapter.get("adapter_dim", 32)))])
        command.extend(["--temporal-kernel-size", str(int(adapter.get("kernel_size", 3)))])
        command.extend(["--temporal-residual-scale", str(float(adapter.get("residual_scale", 1.0)))])

    if lora:
        rank = int(lora.get("rank", 0))
        if rank > 0:
            command.extend(["--lora-rank", str(rank)])
            command.extend(["--lora-alpha", str(float(lora.get("alpha", rank)))])
            targets = lora.get("targets")
            if targets:
                target_text = targets if isinstance(targets, str) else ",".join(str(item) for item in targets)
                command.extend(["--lora-targets", target_text])

    if max_frames is not None:
        command.extend(["--max-frames", str(int(max_frames))])
    if max_videos is not None:
        command.extend(["--max-videos", str(int(max_videos))])
    return command


def export_rtdetr_frame_stability_predictions(
    specs: Iterable[dict[str, Any]] | None = None,
    *,
    checkpoint_name: str = "best.pth",
    score_threshold: float = 0.001,
    max_frames: int | None = None,
    max_videos: int | None = None,
    device: str | None = None,
    overwrite: bool = False,
    missing_only: bool = False,
) -> list[dict[str, Any]]:
    """Export RT-DETR predictions to the shared frame-stability JSONL schema."""

    if specs is None:
        specs = build_frame_stability_specs(model_families=("rtdetr_temporal",))
    selected_specs = select_specs(specs, model_family="rtdetr_temporal", missing_only=missing_only)
    repo = prepare_rtdetr_repo()
    reports: list[dict[str, Any]] = []

    for spec in selected_specs:
        prediction_path = resolve_path(spec["prediction_jsonl"])
        if prediction_path.exists() and not overwrite:
            reports.append(
                {
                    "experiment_id": spec["experiment_id"],
                    "status": "skipped_exists",
                    "prediction_jsonl": project_relative_path(prediction_path),
                }
            )
            continue

        run_dir = resolve_path(spec["run_dir"])
        run_config = load_run_config(run_dir)
        checkpoint = find_checkpoint(run_dir, candidates=(checkpoint_name, "best.pth", "last.pth"))
        command = _build_predict_command(
            run_config=run_config,
            run_dir=run_dir,
            checkpoint=checkpoint,
            prediction_path=prediction_path,
            repo=repo,
            score_threshold=score_threshold,
            max_frames=max_frames,
            max_videos=max_videos,
            device=device,
        )

        log_dir = resolve_path("outputs/frame_stability/logs") / spec["experiment_id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        result = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (log_dir / "official_predict_command.json").write_text(
            json.dumps(
                {
                    "cwd": project_relative_path(repo),
                    "command": [project_relative_text(item) for item in command],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (log_dir / "official_predict_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (log_dir / "official_predict_stderr.log").write_text(result.stderr or "", encoding="utf-8")

        report = {
            "experiment_id": spec["experiment_id"],
            "model_family": "rtdetr_temporal",
            "returncode": int(result.returncode),
            "checkpoint": project_relative_path(checkpoint),
            "prediction_jsonl": project_relative_path(prediction_path),
            "log_dir": project_relative_path(log_dir),
            "elapsed_sec": round(time.time() - start, 3),
        }
        if result.returncode != 0:
            report["status"] = "failed"
            report["stderr_tail"] = "\n".join((result.stderr or "").splitlines()[-40:])
            write_json(prediction_path.with_suffix(".meta.json"), report)
            raise RuntimeError(
                "RT-DETR prediction export failed with return code "
                f"{result.returncode}. Logs: {project_relative_path(log_dir)}\n\n"
                + report.get("stderr_tail", "")
            )

        report["status"] = "completed"
        if prediction_path.with_suffix(".meta.json").exists():
            with prediction_path.with_suffix(".meta.json").open("r", encoding="utf-8") as f:
                report["meta"] = json.load(f)
        write_json(prediction_path.with_suffix(".wrapper_meta.json"), report)
        reports.append(report)

    return reports
