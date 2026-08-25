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

"""Plotting predicted trajectories in the ego frame."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from .scene import CAMERA_VIEWS

__all__ = ["plot_trajectories", "plot_scene_summary"]


def _draw_trajectories(
    axis,
    trajectories: np.ndarray,
    history: np.ndarray | None,
    ground_truth: np.ndarray | None,
    lateral_span: float,
) -> None:
    """Draw history, ground truth and predictions on ``axis`` in the ego frame.

    Vehicle conventions: lateral offset on the horizontal axis with left positive,
    longitudinal distance on the vertical axis, driving direction pointing up.
    """
    if history is not None:
        history = np.asarray(history)
        axis.plot(history[:, 1], history[:, 0], color="0.6", linewidth=2, label="history")
    if ground_truth is not None:
        ground_truth = np.asarray(ground_truth)
        axis.plot(
            ground_truth[:, 1], ground_truth[:, 0], color="black", linewidth=2, label="ground truth"
        )
    for index, trajectory in enumerate(trajectories):
        axis.plot(
            trajectory[:, 1],
            trajectory[:, 0],
            linewidth=1.5,
            alpha=0.9 if index == 0 else 0.4,
            label="prediction" if index == 0 else None,
        )
    axis.scatter([0], [0], marker="s", color="red", zorder=5, label="ego")

    axis.set_xlabel("lateral y [m]  (left positive)")
    axis.set_ylabel("longitudinal x [m]")
    axis.set_aspect("equal")
    lateral = np.concatenate(
        [trajectories[:, :, 1].ravel()]
        + ([history[:, 1]] if history is not None else [])
        + ([ground_truth[:, 1]] if ground_truth is not None else [])
        + [np.zeros(1)]
    )
    centre = 0.5 * (lateral.min() + lateral.max())
    half = max(0.5 * lateral_span, 0.5 * (lateral.max() - lateral.min()) * 1.1)
    axis.set_xlim(centre - half, centre + half)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=8)


def plot_trajectories(
    trajectories: np.ndarray,
    history: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    title: str = "",
    output: str | Path | None = None,
    lateral_span: float = 20.0,
):
    """Draw trajectories in the ego frame, with the driving direction pointing up."""
    import matplotlib

    if output is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories = np.atleast_3d(np.asarray(trajectories))
    if trajectories.ndim == 2:
        trajectories = trajectories[None]

    figure, axis = plt.subplots(figsize=(5, 8))
    _draw_trajectories(axis, trajectories, history, ground_truth, lateral_span)
    if title:
        axis.set_title(title, fontsize=10)
    figure.tight_layout()

    if output is not None:
        figure.savefig(output, dpi=140)
        plt.close(figure)
        return None
    return figure


def _view_label(view: str) -> str:
    return view.strip("<>").replace(" VIEW", "").title()


def plot_scene_summary(
    scene,
    trajectories: np.ndarray,
    history: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    reasoning: str | None = None,
    title: str = "",
    output: str | Path | None = None,
    lateral_span: float = 20.0,
):
    """A one-figure summary of a planning result.

    The camera ring fills a three-row grid on the left (front, front-left, front-right,
    one row each, oldest frame to current left to right), the ego-frame trajectory plot
    sits on the right, and the generated reasoning, when given, is printed underneath.
    """
    import matplotlib

    if output is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectories = np.atleast_3d(np.asarray(trajectories))
    if trajectories.ndim == 2:
        trajectories = trajectories[None]

    frames = {view: [frame.load() for frame in scene.views[view]] for view in CAMERA_VIEWS}
    num_frames = max(len(frames[view]) for view in CAMERA_VIEWS)

    figure = plt.figure(figsize=(3.1 * num_frames + 5.0, 7.6))
    outer = figure.add_gridspec(1, 2, width_ratios=[num_frames * 0.62, 1.0], wspace=0.08)
    montage = outer[0].subgridspec(len(CAMERA_VIEWS), num_frames, wspace=0.03, hspace=0.06)

    for row, view in enumerate(CAMERA_VIEWS):
        for column in range(num_frames):
            ax = figure.add_subplot(montage[row, column])
            ax.set_xticks([])
            ax.set_yticks([])
            if column < len(frames[view]):
                image = frames[view][column]
                ax.imshow(image)
                ax.set_box_aspect(image.height / image.width)
            if column == 0:
                ax.set_ylabel(_view_label(view), fontsize=10, rotation=90, labelpad=6)
            if row == 0:
                label = f"frame {column}" + ("  (current)" if column == num_frames - 1 else "")
                ax.set_title(label, fontsize=9, color="#2f3437")

    axis = figure.add_subplot(outer[1])
    _draw_trajectories(axis, trajectories, history, ground_truth, lateral_span)
    if title:
        axis.set_title(title, fontsize=10)

    bottom = 0.06
    if reasoning:
        wrapped = textwrap.fill(f"reasoning: {reasoning}", width=110)
        figure.text(0.5, 0.015, wrapped, ha="center", va="bottom", fontsize=10, color="#2f3437")
        bottom = 0.03 + 0.028 * (wrapped.count("\n") + 1)
    figure.subplots_adjust(left=0.035, right=0.99, top=0.94, bottom=bottom)

    if output is not None:
        figure.savefig(output, dpi=150)
        plt.close(figure)
        return None
    return figure
