"""Main-experiment config helpers for the active VID-PEFT pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .config import (
    PROJECT_ROOT,
    budget_settings,
    make_experiment_id,
    read_json,
    validate_experiment_config,
    write_json,
)


MODEL_FAMILIES = ("yolo_temporal", "rtdetr_temporal")
MAIN_TUNING_MODES = ("spatial_only_full_ft", "spatial_temporal_peft")
BUDGET_SWEEP_BUDGETS = ("small", "medium", "large")
SEED_REPEAT_SEEDS = (1, 2)
MAIN_CLIP = {"length": 5, "causal": False, "target": "center"}
SPATIAL_ONLY_CLIP = {"length": 1, "causal": False, "target": "center"}
CLIP_SWEEP_CLIPS = (
    {"name": "single_frame", "length": 1, "causal": False, "target": "center"},
    {"name": "T3", "length": 3, "causal": False, "target": "center"},
    {"name": "T5", "length": 5, "causal": False, "target": "center"},
    {"name": "T7", "length": 7, "causal": False, "target": "center"},
    {"name": "causal_T5", "length": 5, "causal": True, "target": "current"},
)
HEAD_ONLY_REFERENCE_RUNS = {
    "yolo_temporal": "outputs/runs/youtube_vis_yolo_temporal_head_only_medium_t1_offline_seed0_head_warmup_e5",
    "rtdetr_temporal": "outputs/runs/youtube_vis_rtdetr_temporal_head_only_medium_t1_offline_seed0_head_warmup_e5",
}

STABLE_MAIN_RUN_TAG = "main_stable_e{epochs}"
STABLE_MAIN_HPARAMS = {
    ("yolo_temporal", "spatial_only_full_ft"): {"lr0": 0.00005},
    ("yolo_temporal", "spatial_temporal_peft"): {"lr0": 0.0001, "residual_scale": 0.1},
    ("rtdetr_temporal", "spatial_only_full_ft"): {
        "optimizer_lr": 0.00001,
        "optimizer_backbone_lr": 0.000001,
    },
    ("rtdetr_temporal", "spatial_temporal_peft"): {
        "optimizer_lr": 0.00001,
        "residual_scale": 0.1,
    },
}

_HEAD_WARMUP_CONFIGS = {
    "yolo_temporal": "configs/yolo_head_warmup_seed0.json",
    "rtdetr_temporal": "configs/rtdetr_head_warmup_seed0.json",
}
_BASE_CONFIGS = {
    ("yolo_temporal", "spatial_temporal_peft"): "configs/yolo_spatial_temporal_peft_T5_medium_seed0.json",
    ("yolo_temporal", "spatial_only_full_ft"): "configs/yolo_spatial_only_full_ft_T1_medium_seed0.json",
    ("rtdetr_temporal", "spatial_temporal_peft"): "configs/rtdetr_spatial_temporal_peft_T5_medium_seed0.json",
    ("rtdetr_temporal", "spatial_only_full_ft"): "configs/rtdetr_spatial_only_full_ft_T1_medium_seed0.json",
}


def _copy_head_warmup_config(model_family: str) -> dict[str, Any]:
    try:
        path = _HEAD_WARMUP_CONFIGS[model_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported head warmup model_family: {model_family}") from exc
    return deepcopy(read_json(PROJECT_ROOT / path))


def _copy_base_config(model_family: str, tuning_mode: str) -> dict[str, Any]:
    try:
        path = _BASE_CONFIGS[(model_family, tuning_mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported main experiment condition: {model_family}/{tuning_mode}") from exc
    return deepcopy(read_json(PROJECT_ROOT / path))


def _head_warmup_run_tag(epochs: int) -> str:
    return f"head_warmup_e{int(epochs)}"


def head_warmup_run_dir(
    model_family: str,
    *,
    seed: int,
    epochs: int = 5,
) -> Path:
    """Return the expected head-warmup output directory for a model/seed."""

    data = _copy_head_warmup_config(model_family)
    data["seed"] = int(seed)
    data["run_tag"] = _head_warmup_run_tag(epochs)
    data.setdefault("execution", {})["epochs"] = int(epochs)
    return PROJECT_ROOT / "outputs" / "runs" / make_experiment_id(data)


def head_warmup_checkpoint_path(
    model_family: str,
    *,
    seed: int,
    epochs: int = 5,
) -> str:
    """Return the project-relative head-warmup checkpoint path."""

    run_dir = head_warmup_run_dir(model_family, seed=seed, epochs=epochs)
    rel_dir = run_dir.resolve().relative_to(PROJECT_ROOT.resolve())
    if model_family == "yolo_temporal":
        return (rel_dir / "weights" / "best.pt").as_posix()
    if model_family == "rtdetr_temporal":
        return (rel_dir / "best.pth").as_posix()
    raise ValueError(f"Unsupported model_family: {model_family}")


def _apply_head_warmup_checkpoint(
    data: dict[str, Any],
    *,
    seed: int,
    epochs: int = 5,
) -> None:
    model_family = str(data["model_family"])
    checkpoint = head_warmup_checkpoint_path(model_family, seed=seed, epochs=epochs)
    if model_family == "yolo_temporal":
        yolo = data.setdefault("yolo", {})
        yolo["weights"] = checkpoint
        yolo["head_warmup_checkpoint"] = checkpoint
    elif model_family == "rtdetr_temporal":
        rtdetr = data.setdefault("rtdetr", {})
        rtdetr["checkpoint"] = checkpoint
        rtdetr["head_warmup_checkpoint"] = checkpoint
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")


def _set_train_epochs(data: dict[str, Any], *, epochs: int, batch_size: int) -> None:
    execution = data.setdefault("execution", {})
    execution["epochs"] = int(epochs)

    model_family = data["model_family"]
    if model_family == "yolo_temporal":
        train_args = data.setdefault("yolo", {}).setdefault("train_args", {})
        train_args["epochs"] = int(epochs)
        train_args["batch"] = int(batch_size)
    elif model_family == "rtdetr_temporal":
        train_args = data.setdefault("rtdetr", {}).setdefault("train_args", {})
        train_args["epochs"] = int(epochs)
        train_args["total_batch_size"] = int(batch_size)
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")


def _apply_budget(data: dict[str, Any], budget: str) -> None:
    data["budget"] = budget
    settings = budget_settings(str(data["model_family"]), budget)
    model_family = str(data["model_family"])
    if model_family == "yolo_temporal":
        adapter = data.setdefault("yolo", {}).setdefault("adapter", {})
        adapter["adapter_dim"] = settings["adapter_dim"]
    elif model_family == "rtdetr_temporal":
        rtdetr = data.setdefault("rtdetr", {})
        adapter = rtdetr.setdefault("adapter", {})
        adapter["adapter_dim"] = settings["adapter_dim"]
        lora = rtdetr.setdefault("lora", {})
        lora["rank"] = settings["lora_rank"]
        lora["alpha"] = float(settings["lora_rank"])


def _set_rtdetr_generated_config(data: dict[str, Any]) -> None:
    if data["model_family"] != "rtdetr_temporal":
        return
    data.setdefault("rtdetr", {})["generated_config"] = (
        f"outputs/rtdetr_configs/{make_experiment_id(data)}.yml"
    )


def build_head_warmup_configs(
    *,
    seeds: Iterable[int] = SEED_REPEAT_SEEDS,
    epochs: int = 5,
    yolo_batch_size: int = 4,
    rtdetr_batch_size: int = 4,
    dry_run: bool = False,
    model_families: Iterable[str] = MODEL_FAMILIES,
) -> list[dict[str, Any]]:
    """Build seed-specific 40-class head warmup configs."""

    configs: list[dict[str, Any]] = []
    for seed in seeds:
        for model_family in model_families:
            data = _copy_head_warmup_config(model_family)
            data["seed"] = int(seed)
            data["run_tag"] = _head_warmup_run_tag(epochs)
            data.setdefault("execution", {})["dry_run"] = bool(dry_run)
            _set_train_epochs(
                data,
                epochs=epochs,
                batch_size=yolo_batch_size if model_family == "yolo_temporal" else rtdetr_batch_size,
            )
            if model_family == "rtdetr_temporal":
                _set_rtdetr_generated_config(data)
            validate_experiment_config(data)
            configs.append(data)
    return configs


def _configure_spatial_only_full_ft(data: dict[str, Any]) -> None:
    data["clip"] = deepcopy(SPATIAL_ONLY_CLIP)
    data["tuning_mode"] = "spatial_only_full_ft"
    head = data.setdefault("head_replacement", {})
    head["enabled"] = True
    head["train"] = True
    head["phase"] = "spatial_only_full_ft_after_warmup"
    if data["model_family"] == "yolo_temporal":
        yolo = data.setdefault("yolo", {})
        yolo.pop("adapter", None)
        yolo.setdefault("clip_loader", {})["enabled"] = False
    elif data["model_family"] == "rtdetr_temporal":
        rtdetr = data.setdefault("rtdetr", {})
        rtdetr.pop("adapter", None)
        rtdetr.pop("lora", None)
        rtdetr.setdefault("clip_loader", {})["enabled"] = False
        rtdetr.setdefault("dataset_export", {})["image_mode"] = "hardlink"
        _set_rtdetr_generated_config(data)


def make_main_experiment_config(
    model_family: str,
    tuning_mode: str,
    *,
    seed: int = 0,
    budget: str = "medium",
    epochs: int = 5,
    batch_size: int = 1,
    dry_run: bool = False,
    run_tag: str | None = None,
    head_warmup_seed: int | None = None,
    head_warmup_epochs: int = 5,
) -> dict[str, Any]:
    """Build one runnable main-experiment config from a validated base config.

    Head-only is intentionally not generated here. It is a prerequisite warmup
    baseline, and the validated e5 warmup runs are referenced separately.
    """

    if tuning_mode not in MAIN_TUNING_MODES:
        raise ValueError(f"Unsupported main tuning mode: {tuning_mode}")

    data = _copy_base_config(model_family, tuning_mode)
    data["model_family"] = model_family
    data["tuning_mode"] = tuning_mode
    data["seed"] = int(seed)
    data["clip"] = deepcopy(SPATIAL_ONLY_CLIP if tuning_mode == "spatial_only_full_ft" else MAIN_CLIP)
    data["run_tag"] = run_tag or STABLE_MAIN_RUN_TAG.format(epochs=epochs)
    data.setdefault("execution", {})["dry_run"] = bool(dry_run)
    _set_train_epochs(data, epochs=epochs, batch_size=batch_size)
    _apply_budget(data, budget)

    if tuning_mode == "spatial_only_full_ft":
        _configure_spatial_only_full_ft(data)
    else:
        head = data.setdefault("head_replacement", {})
        head["enabled"] = True
        head["train"] = False
        head["phase"] = "frozen_after_warmup"

    if model_family == "rtdetr_temporal":
        export_cfg = data.setdefault("rtdetr", {}).setdefault("dataset_export", {})
        export_cfg["image_mode"] = "hardlink"
        _set_rtdetr_generated_config(data)
    _apply_head_warmup_checkpoint(
        data,
        seed=int(seed if head_warmup_seed is None else head_warmup_seed),
        epochs=head_warmup_epochs,
    )
    _apply_stable_main_hparams(data)
    validate_experiment_config(data)
    return data


def build_main_experiment_configs(
    *,
    seed: int = 0,
    budget: str = "medium",
    epochs: int = 5,
    batch_size: int = 1,
    dry_run: bool = False,
    run_tag: str | None = None,
    model_families: Iterable[str] = MODEL_FAMILIES,
    tuning_modes: Iterable[str] = MAIN_TUNING_MODES,
    head_warmup_seed: int | None = None,
    head_warmup_epochs: int = 5,
) -> list[dict[str, Any]]:
    """Build the active main-comparison configs, excluding completed head warmup."""

    return [
        make_main_experiment_config(
            model_family,
            tuning_mode,
            seed=seed,
            budget=budget,
            epochs=epochs,
            batch_size=batch_size,
            dry_run=dry_run,
            run_tag=run_tag,
            head_warmup_seed=head_warmup_seed,
            head_warmup_epochs=head_warmup_epochs,
        )
        for model_family in model_families
        for tuning_mode in tuning_modes
    ]


def build_budget_sweep_configs(
    *,
    seed: int = 0,
    epochs: int = 5,
    batch_size: int = 1,
    dry_run: bool = False,
    include_medium: bool = False,
    model_families: Iterable[str] = MODEL_FAMILIES,
    head_warmup_seed: int | None = None,
    head_warmup_epochs: int = 5,
) -> list[dict[str, Any]]:
    """Build the active budget sweep.

    The medium budget is already covered by the completed main-stable run. By
    default this helper returns only small/large to avoid repeating the costly
    medium condition. Set ``include_medium=True`` for a clean 3-budget rerun.
    """

    budgets = BUDGET_SWEEP_BUDGETS if include_medium else ("small", "large")
    configs: list[dict[str, Any]] = []
    for model_family in model_families:
        for budget in budgets:
            configs.append(
                make_main_experiment_config(
                    model_family,
                    "spatial_temporal_peft",
                    seed=seed,
                    budget=budget,
                    epochs=epochs,
                    batch_size=batch_size,
                    dry_run=dry_run,
                    run_tag=f"budget_sweep_e{epochs}",
                    head_warmup_seed=head_warmup_seed,
                    head_warmup_epochs=head_warmup_epochs,
                )
            )
    return configs


def build_clip_sweep_configs(
    *,
    seed: int = 0,
    epochs: int = 5,
    batch_size: int = 1,
    dry_run: bool = False,
    include_anchor: bool = False,
    model_families: Iterable[str] = MODEL_FAMILIES,
    head_warmup_seed: int | None = None,
    head_warmup_epochs: int = 5,
) -> list[dict[str, Any]]:
    """Build the active clip sweep for medium-budget spatial-temporal PEFT.

    Offline T=5 medium is already covered by the completed main-stable run. By
    default this helper skips that anchor and returns only the additional clip
    conditions. Set ``include_anchor=True`` for a full clip sweep rerun.
    """

    configs: list[dict[str, Any]] = []
    for model_family in model_families:
        for clip in CLIP_SWEEP_CLIPS:
            if not include_anchor and clip["length"] == 5 and not clip["causal"]:
                continue
            data = make_main_experiment_config(
                model_family,
                "spatial_temporal_peft",
                seed=seed,
                budget="medium",
                epochs=epochs,
                batch_size=batch_size,
                dry_run=dry_run,
                run_tag=f"clip_sweep_{clip['name']}_e{epochs}",
                head_warmup_seed=head_warmup_seed,
                head_warmup_epochs=head_warmup_epochs,
            )
            data["clip"] = {key: value for key, value in clip.items() if key != "name"}
            if model_family == "rtdetr_temporal":
                _set_rtdetr_generated_config(data)
            validate_experiment_config(data)
            configs.append(data)
    return configs


def build_frame_stability_specs(
    *,
    seed: int = 0,
    epochs: int = 5,
    output_root: str | Path = "outputs/frame_stability",
    model_families: Iterable[str] = MODEL_FAMILIES,
    head_warmup_seed: int | None = None,
    head_warmup_epochs: int = 5,
) -> list[dict[str, Any]]:
    """Return prediction/report paths for clip-wise frame stability analysis."""

    output_root = Path(output_root)
    specs: list[dict[str, Any]] = []
    configs = build_clip_sweep_configs(
        seed=seed,
        epochs=epochs,
        dry_run=False,
        include_anchor=True,
        model_families=model_families,
        head_warmup_seed=head_warmup_seed,
        head_warmup_epochs=head_warmup_epochs,
    )
    for config in configs:
        if (
            config["tuning_mode"] == "spatial_temporal_peft"
            and int(config["clip"]["length"]) == 5
            and not bool(config["clip"].get("causal", False))
        ):
            config = make_main_experiment_config(
                config["model_family"],
                "spatial_temporal_peft",
                seed=seed,
                budget="medium",
                epochs=epochs,
                batch_size=1,
                dry_run=False,
                run_tag=STABLE_MAIN_RUN_TAG.format(epochs=epochs),
                head_warmup_seed=head_warmup_seed,
                head_warmup_epochs=head_warmup_epochs,
            )
        experiment_id = make_experiment_id(config)
        specs.append(
            {
                "experiment_id": experiment_id,
                "model_family": config["model_family"],
                "tuning_mode": config["tuning_mode"],
                "budget": config["budget"],
                "clip": deepcopy(config["clip"]),
                "run_dir": f"outputs/runs/{experiment_id}",
                "prediction_jsonl": str(output_root / "predictions" / f"{experiment_id}.jsonl"),
                "report_json": str(output_root / "reports" / f"{experiment_id}_stability.json"),
                "report_csv": str(output_root / "reports" / f"{experiment_id}_stability.csv"),
            }
        )
    return specs


def build_seed_repeat_main_configs(
    *,
    seeds: Iterable[int] = SEED_REPEAT_SEEDS,
    budget: str = "medium",
    epochs: int = 5,
    batch_size: int = 1,
    dry_run: bool = False,
    model_families: Iterable[str] = MODEL_FAMILIES,
    tuning_modes: Iterable[str] = MAIN_TUNING_MODES,
    head_warmup_epochs: int = 5,
) -> list[dict[str, Any]]:
    """Build main-comparison configs for repeated seeds.

    Each seed uses its own head-warmup checkpoint by default, so run
    ``build_head_warmup_configs`` first for the same seed list.
    """

    configs: list[dict[str, Any]] = []
    for seed in seeds:
        configs.extend(
            build_main_experiment_configs(
                seed=int(seed),
                budget=budget,
                epochs=epochs,
                batch_size=batch_size,
                dry_run=dry_run,
                model_families=model_families,
                tuning_modes=tuning_modes,
                head_warmup_seed=int(seed),
                head_warmup_epochs=head_warmup_epochs,
            )
        )
    return configs


def build_seed_repeat_plan(
    *,
    seeds: Iterable[int] = SEED_REPEAT_SEEDS,
    epochs: int = 5,
    warmup_epochs: int = 5,
    dry_run: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Build the recommended seed-repeat plan.

    The default plan repeats only the final main comparison. Budget and clip
    sweeps remain seed0 analyses unless explicitly expanded later.
    """

    return {
        "head_warmup": build_head_warmup_configs(
            seeds=seeds,
            epochs=warmup_epochs,
            dry_run=dry_run,
        ),
        "main": build_seed_repeat_main_configs(
            seeds=seeds,
            epochs=epochs,
            dry_run=dry_run,
            head_warmup_epochs=warmup_epochs,
        ),
    }


def write_frame_stability_specs(
    specs: Iterable[dict[str, Any]],
    output_path: str | Path = "outputs/frame_stability/frame_stability_specs.json",
) -> Path:
    """Persist frame-stability prediction/report path specs."""

    return write_json(output_path, {"specs": list(specs)})


def summarize_config_plan(configs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return notebook-friendly rows for checking a plan before running it."""

    rows: list[dict[str, Any]] = []
    for config in configs:
        model_section = config.get("yolo") if config["model_family"] == "yolo_temporal" else config.get("rtdetr")
        model_section = model_section or {}
        train_args = model_section.get("train_args", {})
        adapter = model_section.get("adapter", {})
        rows.append(
            {
                "experiment_id": make_experiment_id(config),
                "model_family": config["model_family"],
                "tuning_mode": config["tuning_mode"],
                "budget": config["budget"],
                "seed": config["seed"],
                "clip_length": config["clip"]["length"],
                "dry_run": config.get("execution", {}).get("dry_run", True),
                "epochs": config.get("execution", {}).get("epochs"),
                "optimizer": train_args.get("optimizer", "default"),
                "lr": train_args.get("lr0", train_args.get("optimizer_lr")),
                "backbone_lr": train_args.get("optimizer_backbone_lr"),
                "residual_scale": adapter.get("residual_scale"),
            }
        )
    return rows


def _apply_yolo_hparams(
    data: dict[str, Any],
    *,
    lr0: float,
    epochs: int,
    residual_scale: float | None = None,
) -> None:
    train_args = data.setdefault("yolo", {}).setdefault("train_args", {})
    batch = int(train_args.get("batch", 1))
    train_args.update(
        {
            "optimizer": "AdamW",
            "lr0": float(lr0),
            "lrf": 0.1,
            "warmup_epochs": 1.0,
            "warmup_bias_lr": float(lr0),
            "epochs": int(epochs),
            "batch": batch,
        }
    )
    data.setdefault("execution", {})["epochs"] = int(epochs)
    if residual_scale is not None:
        data.setdefault("yolo", {}).setdefault("adapter", {})["residual_scale"] = float(residual_scale)


def _apply_rtdetr_hparams(
    data: dict[str, Any],
    *,
    optimizer_lr: float,
    epochs: int,
    optimizer_backbone_lr: float | None = None,
    residual_scale: float | None = None,
) -> None:
    train_args = data.setdefault("rtdetr", {}).setdefault("train_args", {})
    total_batch_size = int(train_args.get("total_batch_size", 1))
    train_args.update(
        {
            "optimizer_lr": float(optimizer_lr),
            "optimizer_weight_decay": 0.0001,
            "epochs": int(epochs),
            "total_batch_size": total_batch_size,
        }
    )
    if optimizer_backbone_lr is not None:
        train_args["optimizer_backbone_lr"] = float(optimizer_backbone_lr)
    data.setdefault("execution", {})["epochs"] = int(epochs)
    if residual_scale is not None:
        data.setdefault("rtdetr", {}).setdefault("adapter", {})["residual_scale"] = float(residual_scale)
    _set_rtdetr_generated_config(data)


def _apply_stable_main_hparams(data: dict[str, Any]) -> None:
    """Apply the fixed seed0 main-comparison hyperparameters.

    These values are locked from the seed0 stabilization runs and reused by the
    active main, budget, and clip helpers.
    """

    epochs = int(data.setdefault("execution", {}).get("epochs", 5))
    model_family = str(data["model_family"])
    tuning_mode = str(data["tuning_mode"])
    hparams = STABLE_MAIN_HPARAMS.get((model_family, tuning_mode))

    if not hparams:
        return

    if model_family == "yolo_temporal":
        _apply_yolo_hparams(data, epochs=epochs, **hparams)
    elif model_family == "rtdetr_temporal" and tuning_mode == "spatial_only_full_ft":
        _apply_rtdetr_hparams(
            data,
            epochs=epochs,
            **hparams,
        )
    elif model_family == "rtdetr_temporal":
        _apply_rtdetr_hparams(data, epochs=epochs, **hparams)


def write_main_experiment_configs(
    configs: Iterable[dict[str, Any]],
    output_dir: str | Path = "outputs/main_configs",
) -> list[Path]:
    """Persist generated configs under outputs for reproducible notebook runs."""

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    paths: list[Path] = []
    for config in configs:
        path = output_root / f"{make_experiment_id(config)}.json"
        paths.append(write_json(path, config))
    return paths


def run_main_experiment_configs(configs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run generated configs sequentially from a notebook cell."""

    results: list[dict[str, Any]] = []
    for config in configs:
        if config["model_family"] == "yolo_temporal":
            from src.yolo_temporal.experiment import run_yolo_temporal_experiment

            results.append(run_yolo_temporal_experiment(config))
        elif config["model_family"] == "rtdetr_temporal":
            from src.rtdetr_temporal.experiment import run_rtdetr_temporal_experiment

            results.append(run_rtdetr_temporal_experiment(config))
        else:
            raise ValueError(f"Unsupported model_family: {config['model_family']}")
    return results
