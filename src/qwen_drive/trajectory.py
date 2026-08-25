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

"""Trajectory normalization, resampling and error metrics.

Qwen-Drive predicts 50 waypoints at 10 Hz (5 s). Each benchmark expects a different
grid, so the resamplers here are the bridge between the model output and the metrics:

============  ==========================================  ==================
Benchmark     Grid                                        Helper
============  ==========================================  ==================
NAVSIM v1.1   8 poses at 2 Hz (t = 0.5 .. 4.0 s), or the  :func:`resample_10hz_to_2hz`
              leading 40 poses used directly at 10 Hz
Waymo E2E     20 poses at 4 Hz (t = 0.25 .. 5.0 s)        :func:`resample_to_4hz`
PhysicalAI    50 poses at 10 Hz, used as predicted        --
============  ==========================================  ==================
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "normalize_trajectory",
    "denormalize_trajectory",
    "normalize_history",
    "resample_10hz_to_2hz",
    "resample_uniform",
    "resample_to_4hz",
    "displacement_errors",
    "heading_mae",
    "wrap_angle",
]


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians into ``[-pi, pi)``."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _wrap_heading(trajectory: torch.Tensor) -> torch.Tensor:
    heading = torch.remainder(trajectory[..., 2:3] + torch.pi, 2 * torch.pi) - torch.pi
    return torch.cat([trajectory[..., :2], heading], dim=-1)


def normalize_trajectory(trajectory: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map metres/radians into the network's normalized units."""
    return _wrap_heading(trajectory) / scale.view(1, 1, -1)


def denormalize_trajectory(trajectory: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map normalized units back to metres/radians."""
    return _wrap_heading(trajectory * scale.view(1, 1, -1))


def normalize_history(history: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Re-reference history poses to the oldest one and normalize.

    The oldest pose becomes the origin, so history and future both progress in the
    driving direction. That origin row carries no information and is dropped, leaving
    ``num_history_points - 1`` poses.
    """
    history = _wrap_heading(history - history[:, 0:1, :])
    return normalize_trajectory(history[:, 1:, :], scale)


def resample_10hz_to_2hz(
    trajectory: np.ndarray, horizon_s: float = 4.0, source_hz: int = 10, target_hz: int = 2
) -> np.ndarray:
    """Pick the 2 Hz poses out of a 10 Hz trajectory.

    Inverse of NAVSIM's interpolation grid ``index = (k + 1) * source_hz / target_hz - 1``,
    i.e. indices ``[4, 9, ..., 39]`` for a 4 s horizon.
    """
    trajectory = np.asarray(trajectory)
    step = source_hz // target_hz
    count = int(round(horizon_s * target_hz))
    indices = [min(trajectory.shape[0] - 1, (k + 1) * step - 1) for k in range(count)]
    return trajectory[indices]


def _interpolate(trajectory: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    out = np.zeros((target_t.shape[0], trajectory.shape[1]), dtype=np.float64)
    out[:, 0] = np.interp(target_t, source_t, trajectory[:, 0])
    out[:, 1] = np.interp(target_t, source_t, trajectory[:, 1])
    if trajectory.shape[1] > 2:
        # Interpolate the heading through its sine and cosine so wrap-around is safe.
        out[:, 2] = np.arctan2(
            np.interp(target_t, source_t, np.sin(trajectory[:, 2])),
            np.interp(target_t, source_t, np.cos(trajectory[:, 2])),
        )
    return out


def resample_uniform(trajectory: np.ndarray, num_poses: int) -> np.ndarray:
    """Resample onto ``num_poses`` samples of the normalized index, keeping both endpoints."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.shape[0] == num_poses:
        return trajectory
    return _interpolate(
        trajectory,
        np.linspace(0.0, 1.0, trajectory.shape[0]),
        np.linspace(0.0, 1.0, num_poses),
    )


def resample_to_4hz(
    trajectory: np.ndarray, num_poses: int = 20, horizon_s: float = 5.0, official_grid: bool = False
) -> np.ndarray:
    """Resample a 10 Hz trajectory onto the Waymo E2E 4 Hz grid.

    With ``official_grid`` the target timestamps are the benchmark's
    ``t = 0.25, 0.5, ..., 5.0`` s interpolated from the prediction's
    ``t = 0.1, 0.2, ..., 5.0`` s. Otherwise both grids are normalized index ramps,
    which preserves the endpoints but shifts the intermediate poses slightly. That is
    the convention the reported numbers were produced with.
    """
    if not official_grid:
        return resample_uniform(trajectory, num_poses)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    source = np.arange(trajectory.shape[0]) + 1
    return _interpolate(
        trajectory,
        source * (horizon_s / trajectory.shape[0]),
        (np.arange(num_poses) + 1) * (horizon_s / num_poses),
    )


def displacement_errors(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Return per-step xy distances plus the average and final displacement errors."""
    distance = np.linalg.norm(prediction[..., :2] - ground_truth[..., :2], axis=-1)
    return distance, float(distance.mean()), float(distance[..., -1].mean())


def heading_mae(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean absolute heading error in radians, wrapped to ``[-pi, pi)``."""
    return float(np.abs(wrap_angle(prediction[..., 2] - ground_truth[..., 2])).mean())
