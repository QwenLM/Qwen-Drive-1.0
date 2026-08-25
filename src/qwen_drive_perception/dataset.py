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

"""Loading self-contained demo frames and turning them into model inputs.

A frame directory packs everything one sample needs::

    <token>/frame.json     ordered prompt content (view tag / image pairs), camera
                           order and dataset type (``nuscenes`` or ``nuplan``)
    <token>/images/*.jpg   original camera frames
    <token>/calib.npz      cam_intrinsic, sensor2lidar_*, lidar2ego
    <token>/gt.npz         occupancy, map raster and 3D boxes (already label-mapped)
    <token>/lidar.npy      LiDAR points for visualization

The prompt replays the exact training-time ChatML layout: the user turn lists
each camera as a view tag followed by its image, then the instruction; the
assistant turn is opened but left empty because the perception heads only read
the prompt's hidden states.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from . import geometry

__all__ = ["PerceptionFrame", "PerceptionProcessor"]


def smart_resize(height, width, factor=32, min_pixels=4096, max_pixels=1638400):
    """Snap to a whole number of merge blocks (like the Qwen-VL processors)."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt(height * width / max_pixels)
        h_bar = int(np.floor(height / beta / factor) * factor)
        w_bar = int(np.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = int(np.ceil(height * beta / factor) * factor)
        w_bar = int(np.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


class PerceptionFrame:
    """One packed demo frame."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frame = json.loads((self.path / "frame.json").read_text())
        self.token = self.path.name
        self.dataset_type = self.frame["dataset_type"]
        self.cam_order = self.frame["cam_order"]
        self.content = self.frame["content"]
        calib = np.load(self.path / "calib.npz")
        self.cam_intrinsic = calib["cam_intrinsic"]
        self.sensor2lidar_rotation = calib["sensor2lidar_rotation"]
        self.sensor2lidar_translation = calib["sensor2lidar_translation"]
        self.lidar2ego = calib["lidar2ego"]
        gt = np.load(self.path / "gt.npz")
        self.gt = {k: gt[k] for k in gt.files}
        self.lidar = np.load(self.path / "lidar.npy") if (self.path / "lidar.npy").exists() else None

    def image(self, cam: str) -> Image.Image:
        return Image.open(self.path / "images" / f"{cam}.jpg").convert("RGB")

    def img_metas(self, image_size=(896, 512), image_grid=(32, 56)) -> dict:
        """Metadata the heads consume: calibration, resize scales, dataset type."""
        target_w, target_h = image_size
        lidar2img = np.stack(
            [
                geometry.apply_image_scale(
                    geometry.build_lidar2img(K, R, t), target_w / w, target_h / h
                )
                for K, R, t, (w, h) in zip(
                    self.cam_intrinsic,
                    self.sensor2lidar_rotation,
                    self.sensor2lidar_translation,
                    [self.image(cam).size for cam in self.cam_order],
                )
            ]
        )
        return {
            "sample_token": self.token,
            "dataset_type": self.dataset_type,
            "cam_order": self.cam_order,
            "lidar2img": lidar2img.astype(np.float32),
            "lidar2ego": np.repeat(self.lidar2ego[None], len(self.cam_order), axis=0).astype(np.float32),
            "img_shape": [(target_h, target_w)] * len(self.cam_order),
            "box_coord_system": "ego",
        }


class PerceptionProcessor:
    """Turn a :class:`PerceptionFrame` into (inputs, img_metas) for :meth:`infer`."""

    def __init__(self, tokenizer, image_size=(896, 512), patch_size=16, merge_size=2, temporal_patch_size=2):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.factor = patch_size * merge_size
        self.image_token_id = getattr(tokenizer, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _patchify(self, image: Image.Image, target_size) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize to ``target_size`` and flatten into block-ordered patches."""
        target_w, target_h = target_size
        image = TF.resize(image, [target_h, target_w], interpolation=InterpolationMode.BICUBIC)
        width, height = image.size
        grid_height, grid_width = smart_resize(height, width, self.factor)
        image = TF.resize(image, [grid_height, grid_width], interpolation=InterpolationMode.BICUBIC)
        pixels = TF.pil_to_tensor(image).float().div_(255.0).sub_(0.5).div_(0.5)

        rows = grid_height // self.patch_size
        cols = grid_width // self.patch_size
        merge = self.merge_size
        pixels = pixels.unsqueeze(1).expand(-1, self.temporal_patch_size, -1, -1)
        patches = pixels.reshape(
            pixels.shape[0],
            1,
            self.temporal_patch_size,
            rows // merge,
            merge,
            self.patch_size,
            cols // merge,
            merge,
            self.patch_size,
        )
        patches = patches.permute(1, 3, 6, 4, 7, 0, 2, 5, 8).reshape(rows * cols, -1)
        return patches, (rows, cols)

    def __call__(self, frame: PerceptionFrame, device: str = "cpu") -> tuple[dict, dict]:
        patch_list, grids, token_counts = [], [], []
        for item in frame.content:
            if "image" not in item:
                continue
            patches, (rows, cols) = self._patchify(frame.image(item["image"]), self.image_size)
            patch_list.append(patches)
            grids.append((1, rows, cols))
            token_counts.append(rows * cols // self.merge_size**2)

        body: list[int] = []
        count_index = 0
        for item in frame.content:
            if "text" in item:
                body += self._encode(item["text"])
            else:
                body += (
                    [self.vision_start_id]
                    + [self.image_token_id] * token_counts[count_index]
                    + [self.vision_end_id]
                )
                count_index += 1

        prompt = (
            [self.im_start_id]
            + self._encode("user")
            + self.newline_ids
            + body
            + [self.im_end_id]
            + self.newline_ids
            + [self.im_start_id]
            + self._encode("assistant")
            + self.newline_ids
        )
        inputs = {
            "input_ids": torch.tensor([prompt], dtype=torch.long, device=device),
            "pixel_values": torch.cat(patch_list, dim=0).to(device),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long, device=device),
        }
        return inputs, frame.img_metas(image_size=self.image_size)
