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

"""Render one perception frame as a single summary figure.

The camera ring wraps around the BEV panel, each camera placed in the ring cell
that matches its azimuth in the ego frame, so the montage reads like the rig
seen from above::

    front-left    front      front-right
    rear-left     BEV        rear-right
                  rear

Underneath, the occupancy prediction and ground truth are drawn as 3D voxels and
the map pair as BEV rasters. Camera panels carry the predicted boxes, the BEV
panel adds the ground truth in green, and ego-forward points up in every
top-down panel.

Every top-down panel is drawn in the ego frame, which is where the occupancy
grid, the map raster and the packed ground-truth boxes live. Predicted boxes and
lidar points arrive in the lidar frame and are converted through ``lidar2ego``,
a plain 90 degree yaw on nuScenes and the identity on nuPlan.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import patches

from . import geometry
from .configuration_perception import (
    DET_BOX_COLORS,
    DET_CLASS_NAMES,
    GT_BOX_COLOR,
    MAP_CLASS_NAMES,
    MAP_PALETTE,
    OCC_BACKGROUND_LABEL,
    OCC_EMPTY_LABEL,
    OCC_PALETTE,
    OCC_ROAD_LABEL,
)

__all__ = ["render_frame"]

_MAP_PALETTE = np.array(MAP_PALETTE, dtype=np.uint8)
_OCC_PALETTE = np.array(OCC_PALETTE, dtype=np.uint8)

BEV_RADIUS = 50.0
CAMERA_SIZE = (896, 512)
CAMERA_ASPECT = CAMERA_SIZE[0] / CAMERA_SIZE[1]
# Ring geometry, in units of the length below. The side columns are wide enough
# for the top and bottom camera rows to fill their cells, and the centre column is
# as wide as the middle row is tall so the square BEV fills the middle cell.
UNIT_INCHES = 3.6
SIDE_WIDTH = 1.22
CENTER_WIDTH = 1.0
FRAME_COLOR = "#b0b7bf"
TITLE_COLOR = "#2f3437"

# Ring cells of a 3x3 grid keyed by the azimuth they cover, front slots first so a
# rig with fewer than eight cameras keeps the forward and side views. +Y is left in
# the ego frame, so positive azimuths sit in the left column.
_RING_SLOTS = (
    (0.0, (0, 1)),
    (45.0, (0, 0)),
    (-45.0, (0, 2)),
    (90.0, (1, 0)),
    (-90.0, (1, 2)),
    (180.0, (2, 1)),
    (135.0, (2, 0)),
    (-135.0, (2, 2)),
)


def _class_name(label) -> str:
    index = int(label)
    return DET_CLASS_NAMES[index] if 0 <= index < len(DET_CLASS_NAMES) else "generic_object"


def _class_rgb(label) -> tuple[int, int, int]:
    return DET_BOX_COLORS.get(_class_name(label), (255, 138, 0))


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def _camera_title(name: str) -> str:
    return name.replace("CAM_", "").replace("_", " ").title()


def _boxes_to_ego(boxes, lidar2ego) -> np.ndarray:
    if not len(boxes):
        return np.asarray(boxes)
    converted = geometry.lidar_to_ego_boxes(
        torch.as_tensor(np.asarray(boxes), dtype=torch.float32), torch.as_tensor(lidar2ego, dtype=torch.float32)
    )
    return converted.numpy()


def _points_to_ego(points, lidar2ego) -> np.ndarray:
    lidar2ego = np.asarray(lidar2ego, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) @ lidar2ego[:3, :3].T + lidar2ego[:3, 3]


def _ring_layout(frame) -> dict[str, tuple[int, int]]:
    """Assign every camera to the ring cell matching its viewing direction."""
    # A camera looks along its own +Z, so the third column of sensor2lidar is its
    # viewing direction, which lidar2ego then brings into the ego frame.
    rotation = np.asarray(frame.lidar2ego, dtype=np.float64)[:3, :3]
    cameras = []
    for index, cam in enumerate(frame.cam_order):
        forward = rotation @ np.asarray(frame.sensor2lidar_rotation[index], dtype=np.float64)[:, 2]
        cameras.append((float(np.degrees(np.arctan2(forward[1], forward[0]))), cam))

    # The forward camera anchors the ring and the rest follow clockwise from it.
    # Matching each camera to its nearest slot instead would break the rear pair,
    # which sits halfway between two slots and is not mirrored, and ranking raw
    # azimuths would break on the rear camera, whose angle flips between -180 and
    # +180 from frame to frame.
    anchor = min(cameras, key=lambda item: abs(item[0]))
    clockwise = sorted(
        (camera for camera in cameras if camera[1] != anchor[1]),
        key=lambda item: (anchor[0] - item[0]) % 360.0,
    )
    slots = dict(_RING_SLOTS[: len(cameras)])
    cells = sorted((azimuth for azimuth in slots if azimuth != 0.0), key=lambda a: -a % 360.0)
    layout = {anchor[1]: slots[0.0]}
    layout.update({cam: slots[azimuth] for (_, cam), azimuth in zip(clockwise, cells)})
    return layout


def _row_heights(layout: dict[str, tuple[int, int]]) -> list[float]:
    """Height of each ring row, just enough for the panels it holds.

    A rig with six cameras leaves the rear corners empty, and there is no reason
    for that row to be as tall as a full-width camera.
    """
    columns = (SIDE_WIDTH, CENTER_WIDTH, SIDE_WIDTH)
    heights = []
    for row in range(3):
        widths = [columns[column] for _, (r, column) in layout.items() if r == row]
        height = max((width / CAMERA_ASPECT for width in widths), default=0.0)
        if row == 1:  # the square BEV sits here
            height = max(height, CENTER_WIDTH)
        heights.append(max(height, 0.2))
    return heights


def _draw_box_edges(image, uv, color, thickness):
    corners = np.rint(uv).astype(int)
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(corners[i + 4]), tuple(corners[(i + 1) % 4 + 4]), color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(corners[i]), tuple(corners[i + 4]), color, thickness, cv2.LINE_AA)


def _camera_image(frame, index: int, cam: str, boxes, labels) -> np.ndarray:
    """The camera frame at model resolution with the predicted boxes drawn on."""
    width, height = CAMERA_SIZE
    image = np.asarray(frame.image(cam).resize((width, height)))
    canvas = image.copy()
    if len(boxes):
        uv, valid = geometry.project_to_image(
            geometry.box_corners(boxes), frame.img_metas()["lidar2img"][index], width, height
        )
        for i in np.where(valid)[0]:
            _draw_box_edges(canvas, uv[i], _class_rgb(labels[i]), 2)
    return canvas


def _panel_image(ax, image: np.ndarray, title: str = "") -> None:
    ax.imshow(image, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=9.5, color=TITLE_COLOR, pad=3.5)


def _panel_bev(ax, points, pred_boxes, pred_labels, gt_boxes) -> None:
    """Lidar sweep coloured by height with the boxes of both sources on top.

    Everything is expected in the ego frame.
    """
    ax.set_xlim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_ylim(-BEV_RADIUS, BEV_RADIUS)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(0.8)

    for radius in (20.0, 40.0):
        ax.add_patch(
            patches.Circle((0, 0), radius, fill=False, edgecolor="#dfe3e8", linewidth=0.7, zorder=1)
        )

    if points is not None and len(points):
        points = points[np.abs(points[:, :2]).max(axis=1) < BEV_RADIUS]
        if len(points) > 80000:
            points = points[:: len(points) // 80000]
        height = (np.clip(points[:, 2], -3.0, 5.0) + 3.0) / 8.0
        # Screen axes: right is -Y, up is +X, so ego-forward points up.
        ax.scatter(
            -points[:, 1],
            points[:, 0],
            s=1.1,
            c=height,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    def draw(boxes, color_of, linewidth, zorder):
        corners = geometry.box_corners(boxes)
        for index in range(len(boxes)):
            footprint = corners[index][:4, :2]
            xy = np.stack([-footprint[:, 1], footprint[:, 0]], axis=1)
            color = color_of(index)
            ax.add_patch(
                patches.Polygon(xy, closed=True, fill=False, edgecolor=color, linewidth=linewidth, zorder=zorder)
            )
            front = corners[index][:2, :2].mean(axis=0)
            center = footprint.mean(axis=0)
            ax.plot(
                [-center[1], -front[1]],
                [center[0], front[0]],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                solid_capstyle="round",
            )

    if len(gt_boxes):
        draw(gt_boxes, lambda _: _hex(GT_BOX_COLOR), 1.7, 4)
    if len(pred_boxes):
        draw(pred_boxes, lambda i: _hex(_class_rgb(pred_labels[i])), 1.2, 5)

    ax.add_patch(
        patches.FancyArrow(0, 0, 0, 4.2, width=0.7, head_width=2.2, head_length=2.6, fc="#111111", ec="none", zorder=6)
    )


def _occupancy_window(*grids) -> tuple[float, float, float, float]:
    """The occupied XY window shared by several grids, with a small margin.

    Labelled voxels rarely reach the edge of the 200x200 grid, and cropping to
    what is actually filled makes the drawing much larger. Both panels of a pair
    use the same window so they stay comparable.
    """
    bounds = []
    for grid in grids:
        x, y = np.where((np.asarray(grid).reshape(200, 200, 16) != OCC_EMPTY_LABEL).any(axis=2))
        if len(x):
            bounds.append((x.min(), x.max(), y.min(), y.max()))
    if not bounds:
        return 0.0, 199.0, 0.0, 199.0
    bounds = np.array(bounds)
    x0, x1 = bounds[:, 0].min() - 4, bounds[:, 1].max() + 4
    y0, y1 = bounds[:, 2].min() - 4, bounds[:, 3].max() + 4
    return float(max(x0, 0)), float(min(x1, 199)), float(max(y0, 0)), float(min(y1, 199))


def _panel_occupancy(ax, grid: np.ndarray, window) -> None:
    """Semantic voxels seen from behind and above, ego-forward up-screen.

    Background voxels that hang over the road (canopies, gantries, footbridges)
    are left out: seen from above they would hide the road surface, which is the
    part of the grid worth looking at. Both panels of a pair are treated the
    same way, so they stay comparable.
    """
    semantics = np.asarray(grid).reshape(200, 200, 16).astype(np.int32)
    heights = np.arange(semantics.shape[2])
    road_top = np.where(semantics == OCC_ROAD_LABEL, heights, -1).max(axis=2)
    over_road = (road_top >= 0)[..., None] & (heights > road_top[..., None])
    hides_road = (semantics == OCC_BACKGROUND_LABEL) & over_road
    x, y, z = np.where((semantics != OCC_EMPTY_LABEL) & ~hides_road)
    labels = np.clip(semantics[x, y, z], 0, len(_OCC_PALETTE) - 1)
    ax.scatter(
        x,
        y,
        z,
        c=_OCC_PALETTE[labels] / 255.0,
        s=2.4,
        marker="s",
        edgecolors=(0.12, 0.16, 0.15, 0.55),
        linewidths=0.035,
        alpha=0.96,
        rasterized=True,
    )
    x0, x1, y0, y1 = window
    ax.view_init(elev=48, azim=205)
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(0, 16))
    # The height axis is exaggerated, otherwise 16 voxels next to 200 would be a
    # flat sheet, and ``zoom`` fills the wide margins a 3D projection leaves
    # inside its own box.
    ax.set_box_aspect((x1 - x0, y1 - y0, 16 * 2.75), zoom=1.4)
    ax.set_axis_off()


def _map_rgb(grid: np.ndarray) -> np.ndarray:
    """The (Y, X) canvas as an image with ego-forward up and ego-left left."""
    labels = np.clip(np.asarray(grid, dtype=np.int64), 0, len(_MAP_PALETTE) - 1)
    return _MAP_PALETTE[labels].transpose(1, 0, 2)[::-1, ::-1]


def _legend(fig) -> None:
    boxes = [
        patches.Patch(color=_hex(_class_rgb(index)), label=name.replace("_", " "))
        for index, name in enumerate(DET_CLASS_NAMES)
    ]
    boxes.append(patches.Patch(facecolor="none", edgecolor=_hex(GT_BOX_COLOR), label="ground-truth box"))
    rasters = [
        patches.Patch(color=_hex(OCC_PALETTE[7]), label="occupancy: driveable"),
        patches.Patch(color=_hex(OCC_PALETTE[8]), label="occupancy: background"),
    ]
    rasters += [
        patches.Patch(color=_hex(MAP_PALETTE[index]), label=f"map: {MAP_CLASS_NAMES[index].replace('_', ' ')}")
        for index in range(1, len(MAP_CLASS_NAMES))
    ]
    rasters += [patches.Patch(visible=False, label="")] * (len(boxes) - len(rasters))
    # A legend fills its columns top to bottom, so interleaving the two groups
    # puts the box classes on the first line and the rasters on the second.
    handles = [handle for pair in zip(boxes, rasters) for handle in pair]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=len(boxes),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.3,
        handletextpad=0.5,
    )


def _grow(ax, scale_x: float, scale_y: float) -> None:
    """Grow an axes about its centre.

    The voxel panels are stretched past their grid cell so they fill the band.
    Only vertically: growing them sideways would overlap the neighbouring panel,
    and a feature at the edge of one would look like it belongs to the other.
    """
    box = ax.get_position()
    center_x, center_y = box.x0 + box.width / 2, box.y0 + box.height / 2
    width, height = box.width * scale_x, box.height * scale_y
    ax.set_position((center_x - width / 2, center_y - height / 2, width, height))


def render_frame(frame, result, score_threshold: float = 0.25, dpi: int = 170) -> np.ndarray:
    """Assemble the summary image (BGR) of one frame.

    The camera ring surrounds the BEV panel in a 3x3 grid ordered by camera
    azimuth. The occupancy and map pairs sit in the row below, the class legend
    at the bottom.
    """
    keep = result["scores"] >= score_threshold
    pred_boxes, pred_labels = result["boxes"][keep], result["labels"][keep]

    layout = _ring_layout(frame)
    rows = _row_heights(layout)
    width = (2.0 * SIDE_WIDTH + CENTER_WIDTH) * UNIT_INCHES
    ring_height = sum(rows) * UNIT_INCHES
    bottom_height = 1.12 * UNIT_INCHES
    legend_height = 0.62
    height = ring_height + bottom_height + legend_height

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[ring_height, bottom_height],
        left=0.006,
        right=0.994,
        top=0.975,
        bottom=legend_height / height,
        hspace=0.02,
    )

    ring = outer[0].subgridspec(
        3,
        3,
        width_ratios=[SIDE_WIDTH, CENTER_WIDTH, SIDE_WIDTH],
        height_ratios=rows,
        wspace=0.02,
        hspace=0.06,
    )
    for index, cam in enumerate(frame.cam_order):
        row, column = layout[cam]
        ax = fig.add_subplot(ring[row, column])
        _panel_image(ax, _camera_image(frame, index, cam, pred_boxes, pred_labels), _camera_title(cam))
    _panel_bev(
        fig.add_subplot(ring[1, 1]),
        _points_to_ego(frame.lidar, frame.lidar2ego) if frame.lidar is not None else None,
        _boxes_to_ego(pred_boxes, frame.lidar2ego),
        pred_labels,
        frame.gt["boxes"],
    )

    # A title strip keeps the labels clear of the panels, which fill the rest of
    # the band, and of the 3D axes grown past their own cell.
    bottom = outer[1].subgridspec(
        2, 4, width_ratios=[1.3, 1.3, 0.5, 0.5], height_ratios=[0.09, 1.0], wspace=0.05, hspace=0.0
    )
    occ_pred = fig.add_subplot(bottom[1, 0], projection="3d")
    occ_gt = fig.add_subplot(bottom[1, 1], projection="3d")
    window = _occupancy_window(result["occ"], frame.gt["occ"])
    _panel_occupancy(occ_pred, result["occ"], window)
    _panel_occupancy(occ_gt, frame.gt["occ"], window)
    _panel_image(fig.add_subplot(bottom[1, 2]), _map_rgb(result["map"]))
    _panel_image(fig.add_subplot(bottom[1, 3]), _map_rgb(frame.gt["map"]))

    titles = ("occupancy prediction", "occupancy ground truth", "map prediction", "map ground truth")
    for column, title in enumerate(titles):
        cell = bottom[1, column].get_position(fig)
        fig.text(
            cell.x0 + cell.width / 2,
            cell.y1 + 0.004,
            title,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=TITLE_COLOR,
        )
    for ax in (occ_pred, occ_gt):
        _grow(ax, 1.0, 1.35)

    fig.text(0.006, 0.996, f"{frame.token}   {frame.dataset_type}", fontsize=9, color=TITLE_COLOR, va="top")
    _legend(fig)

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
