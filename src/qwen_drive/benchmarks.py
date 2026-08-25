# Copyright 2026 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reading the released benchmark scene files.

One JSON object per line describes one planning query. The layout is shared by the
NAVSIM, Waymo Open Dataset end-to-end and NVIDIA PhysicalAI splits:

``messages[0].content``
    The user turn, as a list of ``{"text": ...}`` and ``{"image": ..., "resized_width":
    ..., "resized_height": ...}`` items, grouped by camera view then by timestamp.
``trajectory``
    Ego history and future at 10 Hz, the ego status and the navigation command.
``meta_info``
    Identifiers and camera order.

Image paths are relative to the split's frame root, for example
``val/<context>/FRONT/134.jpg``. Supply the frames either as a directory tree
(``image_root``) or as Parquet shards built by ``scripts/pack_images.py``
(``image_archive``); see ``docs/data.md`` for how to obtain each dataset's frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image

from .images import ImageArchive
from .scene import CAMERA_VIEWS, CameraFrame, DrivingScene

__all__ = ["BenchmarkSample", "read_scene_file"]

Resolver = Callable[[str], "str | Path | Image.Image"]


def _make_resolver(
    image_root: str | Path | None,
    image_archive: ImageArchive | None,
    image_resolver: Resolver | None = None,
) -> Resolver:
    """Return a function mapping a frame path to something loadable.

    An explicit ``image_resolver`` takes precedence: any callable mapping a frame
    path to something loadable, for storage neither a directory nor an archive.
    """
    # Resolvers return a *lazy* image source (a path or a zero-argument
    # callable): frames are only fetched and decoded when a worker preprocesses
    # the sample, so materializing a whole split stays cheap.
    if image_resolver is not None:
        if hasattr(image_resolver, "reference"):
            return image_resolver.reference
        return lambda path: (lambda bound=path: image_resolver(bound))
    if image_archive is not None:
        return lambda path: (lambda bound=path: image_archive.read(bound))
    root = Path(image_root or ".")
    return lambda path: root / path


def _history(trajectory: dict, key: str, count: int) -> np.ndarray:
    """Read a history series, preferring the 1.5 s window when the file provides one."""
    values = trajectory.get(f"{key}_1p5s_10hz") or trajectory[f"{key}_10hz"]
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] < count:
        padding = np.repeat(array[:1], count - array.shape[0], axis=0)
        array = np.concatenate([padding, array], axis=0)
    return array[-count:]


@dataclass
class BenchmarkSample:
    """One scene together with the ground truth needed to score it."""

    scene: DrivingScene
    token: str
    scene_token: str
    future_trajectory: np.ndarray | None
    future_valid: np.ndarray | None
    preference_trajectories: list[np.ndarray]
    preference_scores: list[float]
    initial_speed: float


def _to_scene(record: dict, resolve: Resolver, num_history: int) -> DrivingScene:
    content = record["messages"][0]["content"]
    images = [item for item in content if "image" in item]
    per_view, remainder = divmod(len(images), len(CAMERA_VIEWS))
    if remainder or per_view == 0:
        raise ValueError(
            f"expected a whole number of frames per camera view, got {len(images)} images"
        )
    views = {}
    for index, view in enumerate(CAMERA_VIEWS):
        frames = []
        for item in images[index * per_view : (index + 1) * per_view]:
            width, height = item.get("resized_width"), item.get("resized_height")
            frames.append(
                CameraFrame(
                    image=resolve(item["image"]),
                    target_size=(int(width), int(height)) if width and height else None,
                )
            )
        views[view] = frames

    trajectory = record["trajectory"]
    ego = trajectory["ego_status"]
    instruction = [item["text"] for item in content if "text" in item][-1]
    return DrivingScene(
        views=views,
        history=_history(trajectory, "hist_traj", num_history),
        history_velocity=_history(trajectory, "hist_vel", num_history),
        history_acceleration=_history(trajectory, "hist_acc", num_history),
        ego_velocity=ego["ego_velocity"],
        ego_acceleration=ego["ego_acceleration"],
        driving_command=ego["driving_command"],
        nav_command=int(trajectory["nav_command"]),
        instruction_text=instruction,
        token=str(record["meta_info"].get("token", "")),
        metadata=record["meta_info"],
    )


def read_scene_file(
    path: str | Path,
    image_root: str | Path | None = None,
    image_archive: ImageArchive | None = None,
    image_resolver: Resolver | None = None,
    num_history_points: int = 16,
    limit: int | None = None,
) -> Iterator[BenchmarkSample]:
    """Stream samples from a benchmark scene file."""
    resolve = _make_resolver(image_root, image_archive, image_resolver)
    with open(path) as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            record = json.loads(line)
            trajectory = record["trajectory"]
            future = trajectory.get("future_traj_10hz")
            valid = trajectory.get("future_valid_mask_10hz")
            preferences = trajectory.get("preference_trajectories") or []
            ego = trajectory["ego_status"]["ego_velocity"]
            yield BenchmarkSample(
                scene=_to_scene(record, resolve, num_history_points),
                token=str(record["meta_info"].get("token", "")),
                scene_token=str(record["meta_info"].get("scene_token") or ""),
                future_trajectory=None if future is None else np.asarray(future, dtype=np.float32),
                future_valid=None if valid is None else np.asarray(valid, dtype=np.float32),
                preference_trajectories=[
                    np.asarray(entry["pref_traj"], dtype=np.float64) for entry in preferences
                ],
                preference_scores=[float(entry["preference_score"]) for entry in preferences],
                initial_speed=float(np.hypot(ego[0], ego[1])),
            )
