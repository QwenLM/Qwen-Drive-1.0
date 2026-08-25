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

"""Geometry helpers: calibration, ego/lidar box transforms and camera projection.

Frames and conventions:

* ``lidar``: the sensor frame the 3D boxes are reported in.
* ``ego``: the vehicle frame the BEV grid and the occupancy volume use, with
  X forward, Y left, Z up.
* ``image``: pixel coordinates of the resized (896x512) camera image.

Boxes are ``[x, y, z, w, l, h, yaw, vx, vy]`` with the center at (x, y, z) and
yaw around +Z.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "build_lidar2img",
    "build_lidar2ego",
    "ego_to_lidar_boxes",
    "lidar_to_ego_boxes",
    "box_corners",
    "project_to_image",
]


def build_lidar2img(cam_intrinsic, sensor2lidar_rotation, sensor2lidar_translation) -> np.ndarray:
    """Compose the 4x4 lidar-to-image matrix of one camera (pre-resize)."""
    lidar2cam_r = np.linalg.inv(np.asarray(sensor2lidar_rotation, dtype=np.float32))
    lidar2cam_t = np.asarray(sensor2lidar_translation, dtype=np.float32) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def apply_image_scale(lidar2img: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
    """Fold the image resize into the projection matrix (top 3 rows only)."""
    out = np.asarray(lidar2img, dtype=np.float32).copy()
    scale = np.array([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]], dtype=np.float32)
    out[:3, :] = scale @ out[:3, :]
    return out


def _quaternion_rotation(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def build_lidar2ego(translation, rotation) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quaternion_rotation(rotation)
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def _transform_boxes(boxes: torch.Tensor, rt: torch.Tensor) -> torch.Tensor:
    """Rotate/translate box centers, add the planar yaw, rotate the velocity."""
    if boxes.numel() == 0:
        return boxes.clone()
    centers = boxes[..., :3]
    centers_h = torch.cat([centers, torch.ones_like(centers[..., :1])], dim=-1)
    transformed = boxes.clone()
    transformed[..., :3] = torch.matmul(centers_h, rt.transpose(-1, -2))[..., :3]
    transformed[..., 6] = transformed[..., 6] + torch.atan2(rt[..., 1, 0], rt[..., 0, 0])
    if transformed.size(-1) >= 9:
        rot2 = rt[..., :2, :2]
        transformed[..., 7:9] = torch.matmul(transformed[..., 7:9], rot2.transpose(-1, -2))
    return transformed


def ego_to_lidar_boxes(boxes: torch.Tensor, lidar2ego: torch.Tensor) -> torch.Tensor:
    ego2lidar = torch.linalg.inv(lidar2ego.to(dtype=torch.float32))
    return _transform_boxes(boxes, ego2lidar.to(boxes.dtype))


def lidar_to_ego_boxes(boxes: torch.Tensor, lidar2ego: torch.Tensor) -> torch.Tensor:
    return _transform_boxes(boxes, lidar2ego.to(dtype=torch.float32).to(boxes.dtype))


def box_corners(boxes: np.ndarray) -> np.ndarray:
    """``(N, 9)`` boxes (z at the box bottom) -> ``(N, 8, 3)`` corners.

    Box layout is ``[x, y, z, w, l, h, yaw]`` with yaw around +Z: following
    mmdet3d's LiDARInstance3DBoxes, ``w`` is the extent along the heading
    direction (local +x) and ``l`` the lateral extent (local +y). Corners are
    the bottom face counter-clockwise from (+x, +y), then the same order on
    the top face.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    centers = boxes[:, None, :3]
    w, l, h = boxes[:, 3:4], boxes[:, 4:5], boxes[:, 5:6]
    yaw = boxes[:, 6]
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    local_xy = np.array(
        [[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64
    ) * np.concatenate([w / 2, l / 2], axis=-1)[:, None, :]  # (N, 4, 2)
    local_x = local_xy[..., 0] * cos - local_xy[..., 1] * sin
    local_y = local_xy[..., 0] * sin + local_xy[..., 1] * cos
    z_bottom = np.zeros((len(boxes), 4))
    z_top = np.repeat(h, 4, axis=1)
    xy = np.concatenate([local_x[..., None], local_y[..., None]], axis=-1)
    corners = np.concatenate(
        [np.concatenate([xy, z_bottom[..., None]], axis=-1), np.concatenate([xy, z_top[..., None]], axis=-1)],
        axis=1,
    )
    return centers + corners


def project_to_image(corners: np.ndarray, lidar2img: np.ndarray, width: int, height: int):
    """Project ``(N, 8, 3)`` lidar corners into one camera.

    A box is dropped when any corner is behind the camera or when no corner
    lands inside the image. Returns ``(N, 8, 2)`` pixel coordinates plus a
    validity mask.
    """
    corners = np.asarray(corners, dtype=np.float64)
    corners_h = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=-1)
    proj = np.einsum("ij,nkj->nki", np.asarray(lidar2img, dtype=np.float64), corners_h)
    depths = proj[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[..., :2] / depths[..., None]
    valid = (depths > 1e-3).all(axis=1)
    in_frame = (
        (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    ).any(axis=1)
    return uv, valid & in_frame
