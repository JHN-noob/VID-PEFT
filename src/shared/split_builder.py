"""Video-level split helpers for YouTube-VIS manifest files."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .config import PROJECT_ROOT, write_json


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _video_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("split", "unknown")), str(record["video_id"])


def _target_count(total: int, ratio: float, minimum: int) -> int:
    if ratio <= 0 or total <= 0:
        return 0
    return min(total, max(minimum, round(total * ratio)))


def _label_counter(records: Iterable[dict[str, Any]]) -> Counter[str]:
    labels: Counter[str] = Counter()
    for record in records:
        for obj in record.get("objects", []):
            labels[str(obj.get("label", obj.get("category_id", "unknown")))] += 1
    return labels


def _labels_for_records(records: Iterable[dict[str, Any]]) -> set[str]:
    return set(_label_counter(records))


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    videos = {_video_key(record) for record in records}
    labels = _label_counter(records)
    return {
        "frames": len(records),
        "videos": len(videos),
        "objects": int(sum(labels.values())),
        "labels": len(labels),
        "split_counts": dict(Counter(str(record.get("split", "unknown")) for record in records)),
    }


def _complete_label_coverage(
    selected_keys: set[tuple[str, str]],
    candidate_keys: list[tuple[str, str]],
    records_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    target_labels: set[str],
) -> set[tuple[str, str]]:
    covered_labels: set[str] = set()
    for key in selected_keys:
        covered_labels.update(_labels_for_records(records_by_key[key]))

    missing_labels = set(target_labels) - covered_labels
    if not missing_labels:
        return selected_keys

    remaining_keys = [key for key in candidate_keys if key not in selected_keys]
    while missing_labels:
        best_key = None
        best_new_labels: set[str] = set()
        best_object_count = -1
        for key in remaining_keys:
            labels = _labels_for_records(records_by_key[key])
            new_labels = labels & missing_labels
            if not new_labels:
                continue
            object_count = sum(len(record.get("objects", [])) for record in records_by_key[key])
            if (
                len(new_labels) > len(best_new_labels)
                or (len(new_labels) == len(best_new_labels) and object_count > best_object_count)
            ):
                best_key = key
                best_new_labels = new_labels
                best_object_count = object_count
        if best_key is None:
            return selected_keys
        selected_keys.add(best_key)
        remaining_keys.remove(best_key)
        missing_labels -= best_new_labels
    return selected_keys


def build_video_level_splits(
    manifest_path: str | Path = PROJECT_ROOT / "outputs" / "manifests" / "youtube_vis.jsonl",
    *,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "splits",
    dataset_name: str = "youtube_vis",
    seed: int = 0,
    pilot_train_ratio: float = 0.05,
    pilot_dev_ratio: float = 0.01,
    min_pilot_train_videos: int = 1,
    min_pilot_dev_videos: int = 1,
    ensure_pilot_train_label_coverage: bool = True,
) -> dict[str, Any]:
    """Create reproducible video-level pilot/main split files.

    Pilot splits are sampled from the train videos only. The main train split
    intentionally includes all train videos again, including pilot videos.
    """

    records = _read_jsonl(manifest_path)
    train_keys = sorted({_video_key(record) for record in records if record.get("split") == "train"})
    val_keys = sorted({_video_key(record) for record in records if record.get("split") == "val"})
    if not train_keys:
        raise ValueError("No train videos found in manifest.")
    if not val_keys:
        raise ValueError("No val videos found in manifest.")

    records_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        records_by_key.setdefault(_video_key(record), []).append(record)
    train_key_set = set(train_keys)
    val_key_set = set(val_keys)

    rng = random.Random(seed)
    shuffled_train_keys = list(train_keys)
    rng.shuffle(shuffled_train_keys)

    pilot_train_count = _target_count(len(train_keys), pilot_train_ratio, min_pilot_train_videos)
    pilot_train_keys = set(shuffled_train_keys[:pilot_train_count])
    if ensure_pilot_train_label_coverage:
        full_train_labels = _labels_for_records(
            record for record in records if _video_key(record) in train_key_set
        )
        pilot_train_keys = _complete_label_coverage(
            pilot_train_keys,
            shuffled_train_keys,
            records_by_key,
            full_train_labels,
        )

    dev_candidates = [key for key in shuffled_train_keys if key not in pilot_train_keys]
    remaining_count = len(dev_candidates)
    pilot_dev_count = min(
        remaining_count,
        _target_count(len(train_keys), pilot_dev_ratio, min_pilot_dev_videos),
    )

    pilot_dev_keys = set(dev_candidates[:pilot_dev_count])

    split_records = {
        f"{dataset_name}_pilot_train": [record for record in records if _video_key(record) in pilot_train_keys],
        f"{dataset_name}_pilot_dev": [record for record in records if _video_key(record) in pilot_dev_keys],
        f"{dataset_name}_train": [record for record in records if _video_key(record) in train_key_set],
        f"{dataset_name}_val": [record for record in records if _video_key(record) in val_key_set],
    }

    output_dir = Path(output_dir)
    files: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, subset_records in split_records.items():
        path = output_dir / f"{name}.jsonl"
        _write_jsonl(path, subset_records)
        files[name] = str(path)
        summaries[name] = _summarize(subset_records)

    pilot_overlap = pilot_train_keys & pilot_dev_keys
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "source_manifest": str(manifest_path),
        "seed": seed,
        "policy": {
            "unit": "video",
            "pilot_source": "train",
            "main_train_includes_pilot": True,
            "val_usage": "final_evaluation_only",
            "pilot_train_ratio": pilot_train_ratio,
            "pilot_dev_ratio": pilot_dev_ratio,
            "ensure_pilot_train_label_coverage": ensure_pilot_train_label_coverage,
        },
        "source": _summarize(records),
        "files": files,
        "splits": summaries,
        "checks": {
            "train_videos": len(train_keys),
            "val_videos": len(val_keys),
            "pilot_train_videos": len(pilot_train_keys),
            "pilot_dev_videos": len(pilot_dev_keys),
            "pilot_train_dev_video_overlap": len(pilot_overlap),
        },
    }
    write_json(output_dir / f"{dataset_name}_split_summary.json", summary)
    return summary


def build_default_youtube_vis_splits(
    manifest_path: str | Path = PROJECT_ROOT / "outputs" / "manifests" / "youtube_vis.jsonl",
    *,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "splits",
    seed: int = 0,
) -> dict[str, Any]:
    """Build the current default YouTube-VIS pilot/main splits."""

    return build_video_level_splits(
        manifest_path,
        output_dir=output_dir,
        dataset_name="youtube_vis",
        seed=seed,
        pilot_train_ratio=0.05,
        pilot_dev_ratio=0.01,
    )
