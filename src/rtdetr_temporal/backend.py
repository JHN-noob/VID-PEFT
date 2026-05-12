"""Wrapper utilities for the official RT-DETR repository."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
from typing import Any

from src.shared.config import PROJECT_ROOT, project_relative_path, project_relative_text


RTDETRV2_S_CHECKPOINT_URL = (
    "https://github.com/lyuwenyu/storage/releases/download/v0.2/"
    "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
)


def find_rtdetr_repo(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    repo = root / "third_party" / "RT-DETR"
    pytorch_root = repo / "rtdetrv2_pytorch"
    if (repo / "LICENSE").exists() and (pytorch_root / "tools" / "train.py").exists():
        return pytorch_root
    raise FileNotFoundError("RT-DETR repo not found under third_party/RT-DETR/rtdetrv2_pytorch.")


def prepare_rtdetr_repo(repo_path: str | Path | None = None) -> Path:
    repo = Path(repo_path) if repo_path is not None else find_rtdetr_repo()
    if repo.name != "rtdetrv2_pytorch":
        repo = repo / "rtdetrv2_pytorch"
    parent_license = repo.parent / "LICENSE"
    if not parent_license.exists():
        raise FileNotFoundError(f"RT-DETR LICENSE not found: {parent_license}")
    if not (repo / "tools" / "train.py").exists():
        raise FileNotFoundError(f"RT-DETR train.py not found: {repo / 'tools' / 'train.py'}")
    return repo


def official_rtdetrv2_s_config(repo_path: str | Path | None = None) -> Path:
    repo = prepare_rtdetr_repo(repo_path)
    return repo / "configs" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco.yml"


def resolve_local_checkpoint(path: str | Path | None, *, project_root: Path = PROJECT_ROOT) -> Path | None:
    if path is None:
        return None
    text = str(path)
    if text.startswith(("http://", "https://")):
        return None
    checkpoint = Path(path)
    return checkpoint if checkpoint.is_absolute() else project_root / checkpoint


def build_train_command(
    *,
    generated_config: str | Path,
    output_dir: str | Path,
    tuning_checkpoint: str | Path,
    repo_path: str | Path,
    tuning_mode: str,
    clip_length: int = 1,
    temporal_adapter: bool = False,
    temporal_adapter_index: int = 1,
    temporal_adapter_channels: int = 256,
    temporal_adapter_dim: int = 32,
    temporal_kernel_size: int = 3,
    temporal_residual_scale: float = 1.0,
    clip_loader_enabled: bool = False,
    target_index: int = 0,
    lora_rank: int = 0,
    lora_alpha: float | None = None,
    lora_targets: str | None = None,
    seed: int = 0,
    device: str | None = None,
    use_amp: bool = True,
) -> list[str]:
    runner = PROJECT_ROOT / "src" / "rtdetr_temporal" / "official_train_with_policy.py"
    command = [
        sys.executable,
        str(runner.resolve()),
        "-c",
        str(Path(generated_config).resolve()),
        "-t",
        str(Path(tuning_checkpoint).resolve()),
        "--output-dir",
        str(Path(output_dir).resolve()),
        "--seed",
        str(seed),
        "--repo",
        str(Path(repo_path).resolve()),
        "--tuning-mode",
        str(tuning_mode),
        "--clip-length",
        str(int(clip_length)),
    ]
    if temporal_adapter:
        command.append("--temporal-adapter")
        command.extend(["--temporal-adapter-index", str(int(temporal_adapter_index))])
        command.extend(["--temporal-adapter-channels", str(int(temporal_adapter_channels))])
        command.extend(["--temporal-adapter-dim", str(int(temporal_adapter_dim))])
        command.extend(["--temporal-kernel-size", str(int(temporal_kernel_size))])
        command.extend(["--temporal-residual-scale", str(float(temporal_residual_scale))])
    if clip_loader_enabled:
        command.append("--clip-loader-enabled")
    command.extend(["--target-index", str(int(target_index))])
    if lora_rank > 0:
        command.extend(["--lora-rank", str(int(lora_rank))])
        if lora_alpha is not None:
            command.extend(["--lora-alpha", str(float(lora_alpha))])
        if lora_targets:
            command.extend(["--lora-targets", str(lora_targets)])
    if device:
        command.extend(["--device", device])
    if use_amp:
        command.append("--use-amp")
    return command


def run_official_train(
    command: list[str],
    *,
    repo_path: str | Path | None = None,
    log_dir: str | Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[Any]:
    repo = prepare_rtdetr_repo(repo_path)
    result = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if log_dir is not None:
        logs = Path(log_dir)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "official_train_command.json").write_text(
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
        (logs / "official_train_stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (logs / "official_train_stderr.log").write_text(result.stderr or "", encoding="utf-8")

    if check and result.returncode != 0:
        stderr_tail = "\n".join((result.stderr or "").splitlines()[-40:])
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-20:])
        detail = [
            f"RT-DETR official train failed with return code {result.returncode}.",
            "Command: " + " ".join(project_relative_text(item) for item in command),
        ]
        if log_dir is not None:
            detail.append(f"Logs: {project_relative_path(log_dir)}")
        if stderr_tail:
            detail.append("stderr tail:\n" + stderr_tail)
        if stdout_tail:
            detail.append("stdout tail:\n" + stdout_tail)
        raise RuntimeError("\n\n".join(detail))

    return result
