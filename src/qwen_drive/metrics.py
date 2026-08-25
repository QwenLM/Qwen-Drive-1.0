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

"""Open-loop metrics for the released benchmarks.

Everything here works on numpy arrays of shape ``[..., num_poses, 3]`` and needs nothing
beyond the scene files, so the NVIDIA PhysicalAI and Waymo displacement metrics can be
reproduced offline. The Waymo rater feedback score and the NAVSIM pseudo-closed-loop
score are the two exceptions and are documented where they appear.
"""

from __future__ import annotations

import numpy as np

from .trajectory import displacement_errors, heading_mae, resample_to_4hz

__all__ = [
    "open_loop_metrics",
    "waymo_displacement_metrics",
    "rater_feedback_score",
    "navsim_displacement_metrics",
]


def open_loop_metrics(
    candidates: np.ndarray, ground_truth: np.ndarray, trajectory_hz: float = 10.0
) -> dict[str, float]:
    """Displacement metrics over several horizons for a set of candidate trajectories.

    ``candidates`` is ``[num_samples, num_poses, 3]`` and ``ground_truth`` is
    ``[num_poses, 3]``. For each horizon both the mean over candidates and the best
    candidate are reported. The latter (``min*``) is an oracle upper bound, since
    picking it needs the ground truth.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    num_poses = min(candidates.shape[1], ground_truth.shape[0])
    distance = np.linalg.norm(
        candidates[:, :num_poses, :2] - ground_truth[np.newaxis, :num_poses, :2], axis=-1
    )

    rate = int(round(trajectory_hz))
    horizons = sorted({min(seconds * rate, num_poses) for seconds in (1, 2, 3, 4)} | {num_poses})
    metrics: dict[str, float] = {}
    for horizon in horizons:
        if horizon <= 0:
            continue
        label = f"{horizon / rate:g}s"
        per_candidate_ade = distance[:, :horizon].mean(axis=-1)
        per_candidate_fde = distance[:, horizon - 1]
        metrics[f"ADE_{label}"] = float(per_candidate_ade.mean())
        metrics[f"minADE_{label}"] = float(per_candidate_ade.min())
        metrics[f"FDE_{label}"] = float(per_candidate_fde.mean())
        metrics[f"minFDE_{label}"] = float(per_candidate_fde.min())
    return metrics


def waymo_displacement_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray, official_4hz_grid: bool = False
) -> dict[str, float]:
    """Waymo end-to-end displacement metrics on both the 10 Hz and 4 Hz grids.

    ``ADE@Xs`` is the mean distance up to X seconds and ``FDE@Xs`` the distance at X
    seconds. The benchmark reports the 4 Hz numbers. The 10 Hz ones are the model's
    native output and are kept for diagnosis.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    num_poses = min(predictions.shape[1], ground_truth.shape[1])
    predictions, ground_truth = predictions[:, :num_poses], ground_truth[:, :num_poses]

    metrics: dict[str, float] = {}
    distance, ade, fde = displacement_errors(predictions, ground_truth)
    metrics["ADE_10hz"] = ade
    metrics["FDE_10hz"] = fde
    metrics["HeadingMAE_10hz"] = heading_mae(predictions, ground_truth)
    for seconds, index in ((1, 9), (3, 29), (5, 49)):
        if distance.shape[1] > index:
            metrics[f"ADE@{seconds}s_10hz"] = float(distance[:, : index + 1].mean())
            metrics[f"FDE@{seconds}s_10hz"] = float(distance[:, index].mean())

    predictions = np.stack([resample_to_4hz(t, 20, official_grid=official_4hz_grid) for t in predictions])
    ground_truth = np.stack([resample_to_4hz(t, 20, official_grid=official_4hz_grid) for t in ground_truth])
    distance, ade, fde = displacement_errors(predictions, ground_truth)
    metrics["ADE"] = ade
    metrics["FDE"] = fde
    metrics["HeadingMAE"] = heading_mae(predictions, ground_truth)
    for seconds, index in ((1, 3), (3, 11), (5, 19)):
        if distance.shape[1] > index:
            metrics[f"ADE@{seconds}s"] = float(distance[:, : index + 1].mean())
            metrics[f"FDE@{seconds}s"] = float(distance[:, index].mean())
    return metrics


def rater_feedback_score(
    predictions: np.ndarray,
    preference_trajectories: list[list[np.ndarray]],
    preference_scores: list[np.ndarray],
    initial_speeds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Waymo's rater feedback score for 4 Hz, 5 s predictions.

    Needs the official implementation from the Waymo Open Dataset, which is not a
    dependency of this package. Install it, or set ``WAYMO_OPEN_DATASET_SRC`` to a
    checkout's ``src`` directory before calling this.
    """
    import os
    import sys

    source = os.environ.get("WAYMO_OPEN_DATASET_SRC")
    if source and source not in sys.path:
        sys.path.insert(0, source)
    try:
        from waymo_open_dataset.metrics.python.rater_feedback_utils import (
            get_rater_feedback_score,
        )
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "the rater feedback score needs waymo-open-dataset, install it or set "
            "WAYMO_OPEN_DATASET_SRC to a checkout's src directory"
        ) from error

    return get_rater_feedback_score(
        inference_trajectories=np.asarray(predictions, dtype=np.float64)[:, np.newaxis, :, :2],
        inference_probs=np.ones((len(predictions), 1), dtype=np.float64),
        rater_specified_trajectories=preference_trajectories,
        rater_feedback_labels=preference_scores,
        init_speed=np.asarray(initial_speeds, dtype=np.float64),
        frequency=4,
        length_seconds=5,
    )


def navsim_displacement_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray, num_poses: int = 40
) -> dict[str, float]:
    """Displacement errors over NAVSIM's 4 s scoring window."""
    predictions = np.asarray(predictions, dtype=np.float64)[:, :num_poses]
    ground_truth = np.asarray(ground_truth, dtype=np.float64)[:, :num_poses]
    _, ade, fde = displacement_errors(predictions, ground_truth)
    return {"ADE_4s": ade, "FDE_4s": fde}
