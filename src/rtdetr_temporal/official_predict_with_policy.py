"""Export RT-DETR predictions with the project trainability wrappers enabled.

This runner is executed as a subprocess because the official RT-DETR repository
uses a top-level package named ``src``. Keeping this file isolated prevents that
package from colliding with the project ``src`` package used by notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch


RUNNER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNNER_DIR.parents[1]
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

from official_train_with_policy import (  # noqa: E402
    apply_trainability_policy,
    attach_temporal_adapter,
    inject_lora,
    register_clip_dataset_types,
)


def _project_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _to_device_targets(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def _dataset_image_meta(dataset: Any) -> dict[int, dict[str, Any]]:
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    coco = getattr(dataset, "coco", None)
    if coco is None and hasattr(dataset, "dataset"):
        coco = getattr(dataset.dataset, "coco", None)
    if coco is None:
        return {}
    return {int(image_id): dict(info) for image_id, info in coco.imgs.items()}


def _rtdetr_detections(result: dict[str, torch.Tensor], *, score_threshold: float) -> list[dict[str, Any]]:
    labels = result["labels"].detach().cpu().tolist()
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    detections: list[dict[str, Any]] = []
    for label, box, score in zip(labels, boxes, scores):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        detections.append(
            {
                "category_id": int(label),
                "score": score_value,
                "bbox_xyxy": [round(float(value), 4) for value in box],
            }
        )
    return detections


def _target_image_id(target: dict[str, torch.Tensor]) -> int:
    value = target["image_id"]
    return int(value.item() if hasattr(value, "item") else value)


def main(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    if not (repo / "tools" / "train.py").exists():
        raise FileNotFoundError(f"RT-DETR repo root is invalid: {repo}")

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig, yaml_utils, register  # type: ignore[import-not-found]
    from src.misc import dist_utils  # type: ignore[import-not-found]
    from src.solver import TASKS  # type: ignore[import-not-found]

    register_clip_dataset_types(register)
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    custom_keys = {
        "update",
        "repo",
        "checkpoint",
        "output_jsonl",
        "meta_json",
        "tuning_mode",
        "temporal_adapter",
        "temporal_adapter_index",
        "temporal_adapter_channels",
        "temporal_adapter_dim",
        "temporal_kernel_size",
        "temporal_residual_scale",
        "clip_length",
        "clip_loader_enabled",
        "target_index",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
        "score_threshold",
        "max_frames",
        "max_videos",
    }
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in vars(args).items() if k not in custom_keys and v is not None})
    update_dict["resume"] = str(Path(args.checkpoint).resolve())

    cfg = YAMLConfig(args.config, **update_dict)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    original_setup = solver._setup

    def setup_with_policy() -> None:
        original_setup()
        if args.lora_rank > 0:
            inject_lora(solver.model, args, solver.device)
        if args.temporal_adapter:
            attach_temporal_adapter(solver.model, args, solver.device)
        apply_trainability_policy(solver.model, args.tuning_mode)

    solver._setup = setup_with_policy
    solver.eval()

    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    solver.postprocessor.eval()

    image_meta = _dataset_image_meta(solver.val_dataloader.dataset)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    meta_json = Path(args.meta_json) if args.meta_json else output_jsonl.with_suffix(".meta.json")
    meta_json.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    written = 0
    seen_videos: set[str] = set()
    stop = False

    with output_jsonl.open("w", encoding="utf-8") as f:
        with torch.no_grad():
            for samples, targets in solver.val_dataloader:
                samples = samples.to(solver.device)
                targets = _to_device_targets(targets, solver.device)
                outputs = module(samples)
                orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
                results = solver.postprocessor(outputs, orig_target_sizes)

                for target, result in zip(targets, results):
                    image_id = _target_image_id(target)
                    meta = image_meta.get(image_id, {})
                    video_id = str(meta.get("video_id", image_id))
                    if args.max_videos is not None and video_id not in seen_videos and len(seen_videos) >= args.max_videos:
                        continue
                    seen_videos.add(video_id)

                    record = {
                        "video_id": video_id,
                        "frame_index": int(meta.get("frame_index", image_id - 1)),
                        "width": int(meta.get("width", int(target["orig_size"][1].item()))),
                        "height": int(meta.get("height", int(target["orig_size"][0].item()))),
                        "detections": _rtdetr_detections(result, score_threshold=args.score_threshold),
                    }
                    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    if args.max_frames is not None and written >= args.max_frames:
                        stop = True
                        break
                if stop:
                    break

    report = {
        "status": "completed",
        "model_family": "rtdetr_temporal",
        "checkpoint": _project_relative(args.checkpoint),
        "config": _project_relative(args.config),
        "prediction_jsonl": _project_relative(output_jsonl),
        "frames": written,
        "videos": len(seen_videos),
        "score_threshold": float(args.score_threshold),
        "elapsed_sec": round(time.time() - start, 3),
    }
    meta_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--meta-json", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("-u", "--update", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--tuning-mode", type=str, required=True)
    parser.add_argument("--temporal-adapter", action="store_true")
    parser.add_argument("--temporal-adapter-index", type=int, default=1)
    parser.add_argument("--temporal-adapter-channels", type=int, default=256)
    parser.add_argument("--temporal-adapter-dim", type=int, default=32)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--temporal-residual-scale", type=float, default=1.0)
    parser.add_argument("--clip-length", type=int, default=1)
    parser.add_argument("--clip-loader-enabled", action="store_true")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="encoder.0.layers.0.linear1,encoder.0.layers.0.linear2",
    )
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-videos", type=int)
    main(parser.parse_args())
