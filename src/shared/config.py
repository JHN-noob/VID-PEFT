"""JSON configuration helpers for VID-PEFT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_MODEL_FAMILIES = {"rtdetr_temporal", "yolo_temporal"}
SUPPORTED_TUNING_MODES = {
    "head_only",
    "spatial_only_full_ft",
    "spatial_temporal_peft",
}

RTDETR_BUDGETS = {
    # Calibrated against YOLO adapter budgets with frozen heads:
    # YOLO d=16/32/64 vs RT-DETR LoRA r=4/8/16 + temporal d=4/8/16.
    "small": {"lora_rank": 4, "adapter_dim": 4},
    "medium": {"lora_rank": 8, "adapter_dim": 8},
    "large": {"lora_rank": 16, "adapter_dim": 16},
}

YOLO_BUDGETS = {
    "small": {"adapter_dim": 16},
    "medium": {"adapter_dim": 32},
    "large": {"adapter_dim": 64},
}


class ConfigError(ValueError):
    """Raised when an experiment config is incomplete or invalid."""


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object, got {type(value).__name__}.")
    return value


def _sanitize_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_") or "unset"


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return _as_dict(json.load(f))


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def project_relative_path(path: str | Path) -> str:
    """Return a portable project-relative path when possible."""

    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def project_relative_text(value: Any) -> str:
    """Replace this project's absolute root prefix in arbitrary text."""

    text = str(value).replace("\\", "/")
    executable = Path(sys.executable).resolve().as_posix()
    if text == executable:
        return Path(sys.executable).name
    root = PROJECT_ROOT.resolve().as_posix()
    prefix = root + "/"
    if text == root:
        return "."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def make_experiment_id(data: Mapping[str, Any]) -> str:
    clip = _as_dict(data.get("clip", {}))
    causal = "causal" if bool(clip.get("causal", False)) else "offline"
    dataset = data.get("dataset", "dataset")
    if isinstance(dataset, Mapping):
        dataset = dataset.get("name", "dataset")
    parts = [
        dataset,
        data.get("model_family", "model"),
        data.get("tuning_mode", "tuning"),
        data.get("budget", "budget"),
        f"T{clip.get('length', 'x')}",
        causal,
        f"seed{data.get('seed', 'x')}",
    ]
    if data.get("run_tag"):
        parts.append(data["run_tag"])
    return "_".join(_sanitize_part(part) for part in parts)


def budget_settings(model_family: str, budget: str) -> dict[str, int]:
    if model_family == "rtdetr_temporal":
        table = RTDETR_BUDGETS
    elif model_family == "yolo_temporal":
        table = YOLO_BUDGETS
    else:
        raise ConfigError(f"Unsupported model_family: {model_family}")

    if budget not in table:
        raise ConfigError(f"Unsupported budget for {model_family}: {budget}")
    return dict(table[budget])


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved view over a VID-PEFT JSON config."""

    data: Mapping[str, Any]
    path: Path | None = None
    project_root: Path = PROJECT_ROOT

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current

    @property
    def dataset_name(self) -> str:
        return str(self.get("dataset", "name", default=self.get("dataset", default="unknown")))

    @property
    def model_family(self) -> str:
        return str(self.get("model_family"))

    @property
    def tuning_mode(self) -> str:
        return str(self.get("tuning_mode"))

    @property
    def budget(self) -> str:
        return str(self.get("budget"))

    @property
    def seed(self) -> int:
        return int(self.get("seed", default=0))

    @property
    def clip_length(self) -> int:
        return int(self.get("clip", "length", default=1))

    @property
    def causal(self) -> bool:
        return bool(self.get("clip", "causal", default=False))

    @property
    def experiment_id(self) -> str:
        return make_experiment_id(self.data)

    @property
    def budget_settings(self) -> dict[str, int]:
        return budget_settings(self.model_family, self.budget)

    def resolved_path(self, *keys: str, default: str | Path | None = None) -> Path | None:
        value = self.get(*keys, default=default)
        if value is None:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def output_dir(self) -> Path:
        outputs_root = self.resolved_path("paths", "outputs_root", default="outputs")
        assert outputs_root is not None
        return outputs_root / "runs" / self.experiment_id

    def dataset_root(self) -> Path | None:
        return (
            self.resolved_path("paths", "youtube_vis_root")
            or self.resolved_path("paths", "dataset_root")
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def validate_experiment_config(data: Mapping[str, Any]) -> None:
    missing = [key for key in ("model_family", "tuning_mode", "budget", "clip", "seed", "paths") if key not in data]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    model_family = str(data["model_family"])
    tuning_mode = str(data["tuning_mode"])
    budget = str(data["budget"])
    clip = _as_dict(data["clip"])

    if model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ConfigError(f"Unsupported model_family: {model_family}")
    if tuning_mode not in SUPPORTED_TUNING_MODES:
        raise ConfigError(f"Unsupported tuning_mode: {tuning_mode}")
    if int(clip.get("length", 0)) < 1:
        raise ConfigError("clip.length must be >= 1.")
    budget_settings(model_family, budget)


def load_experiment_config(
    config_or_path: str | Path | Mapping[str, Any],
    project_root: str | Path | None = None,
) -> ExperimentConfig:
    if isinstance(config_or_path, Mapping):
        data = dict(config_or_path)
        path = None
    else:
        path = Path(config_or_path)
        data = read_json(path)

    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    validate_experiment_config(data)
    return ExperimentConfig(data=data, path=path, project_root=root)


def save_config_snapshot(config: ExperimentConfig, output_dir: str | Path | None = None) -> Path:
    target_dir = Path(output_dir) if output_dir is not None else config.output_dir()
    snapshot = dict(config.to_dict())
    snapshot["_experiment_id"] = config.experiment_id
    if config.path is not None:
        snapshot["_source_config"] = project_relative_path(config.path)
    return write_json(target_dir / "config_snapshot.json", snapshot)
