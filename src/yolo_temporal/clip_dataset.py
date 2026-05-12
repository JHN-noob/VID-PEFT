"""Clip-aware dataset wrapper for Ultralytics YOLO datasets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from src.shared.clip_sampler import clip_target_index, sample_clip_indices


_FRAME_PREFIX = re.compile(r"^(\d+)_")


def _parse_video_frame(image_path: str | Path) -> tuple[str, int]:
    path = Path(image_path)
    video_id = path.parent.name
    match = _FRAME_PREFIX.match(path.stem)
    if not match:
        raise ValueError(f"Could not parse frame index from YOLO image path: {image_path}")
    return video_id, int(match.group(1))


class YOLOClipDataset:
    """Wrap a YOLODataset so each dataloader item is a target-frame clip.

    The collate function flattens clip images to ``[B*T, C, H, W]`` while
    keeping labels only for the target frames. The model-side Detect pre-hook
    then selects target-frame features before the YOLO loss sees predictions.
    """

    def __init__(
        self,
        base_dataset,
        *,
        clip_length: int,
        causal: bool = False,
        target: str = "center",
        boundary: str = "clamp",
    ) -> None:
        self.base_dataset = base_dataset
        self.clip_length = int(clip_length)
        self.causal = bool(causal)
        self.target = str(target)
        self.boundary = str(boundary)
        self.target_index = clip_target_index(self.clip_length, causal=self.causal, target=self.target)

        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for base_index, image_path in enumerate(getattr(base_dataset, "im_files", [])):
            video_id, frame_index = _parse_video_frame(image_path)
            grouped[video_id].append((frame_index, base_index))

        self._video_indices: dict[str, list[int]] = {}
        self._base_to_video_pos: dict[int, tuple[str, int]] = {}
        for video_id, pairs in grouped.items():
            ordered = [base_index for _, base_index in sorted(pairs)]
            self._video_indices[video_id] = ordered
            for position, base_index in enumerate(ordered):
                self._base_to_video_pos[base_index] = (video_id, position)

        self._target_base_indices = list(range(len(base_dataset)))

    def __len__(self) -> int:
        return len(self._target_base_indices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    @property
    def labels(self):  # type: ignore[no-untyped-def]
        return self.base_dataset.labels

    @property
    def mosaic(self):  # type: ignore[no-untyped-def]
        return getattr(self.base_dataset, "mosaic", False)

    @mosaic.setter
    def mosaic(self, value):  # type: ignore[no-untyped-def]
        if hasattr(self.base_dataset, "mosaic"):
            self.base_dataset.mosaic = value

    def close_mosaic(self, hyp) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self.base_dataset, "close_mosaic"):
            self.base_dataset.close_mosaic(hyp)

    def _clip_base_indices(self, base_index: int) -> list[int]:
        video_id, position = self._base_to_video_pos[base_index]
        video_indices = self._video_indices[video_id]
        positions = sample_clip_indices(
            position,
            len(video_indices),
            length=self.clip_length,
            causal=self.causal,
            boundary=self.boundary,
        )
        if not positions:
            raise IndexError(
                f"Target base index {base_index} has no valid clip under boundary={self.boundary!r}."
            )
        return [video_indices[position] for position in positions]

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_base_index = self._target_base_indices[index]
        clip_indices = self._clip_base_indices(target_base_index)
        clip_items = [self.base_dataset[base_index] for base_index in clip_indices]
        return {
            "clip": clip_items,
            "target": clip_items[self.target_index],
            "clip_indices": clip_indices,
            "target_base_index": target_base_index,
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        flat_clip_items = [item for sample in batch for item in sample["clip"]]
        target_items = [sample["target"] for sample in batch]
        collated = self.base_dataset.__class__.collate_fn(target_items)
        collated["img"] = self.base_dataset.__class__.collate_fn(flat_clip_items)["img"]
        collated["clip_length"] = self.clip_length
        collated["clip_target_index"] = self.target_index
        return collated
