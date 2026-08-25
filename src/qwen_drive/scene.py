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

"""Driving scene inputs: camera frames, ego history, prompt and image patches."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .configuration_qwen_drive import QwenDriveConfig

__all__ = [
    "CAMERA_VIEWS",
    "NAV_COMMANDS",
    "REASONING_REQUEST",
    "CameraFrame",
    "DrivingScene",
    "QwenDriveProcessor",
    "smart_resize",
]

# Camera views in the order the model was trained to read them. Frames are grouped by
# view (all four timestamps of one camera, then the next camera).
CAMERA_VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")

# Navigation command vocabulary, indexed by the integer command.
NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")

# Labels of the four camera timestamps, oldest first.
HISTORY_FRAME_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")

# Appended to the user turn to ask for a reasoning trace before planning.
REASONING_REQUEST = (
    "\n\nGive a one-sentence brief reasoning of the ego's future driving decision ONLY."
)

_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn\uff1a\n"
)


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Snap a resolution to a multiple of ``factor`` inside a pixel budget."""
    bar_h = _round_by_factor(height, factor)
    bar_w = _round_by_factor(width, factor)
    pixels = bar_h * bar_w
    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        bar_h = math.floor(height / beta / factor) * factor
        bar_w = math.floor(width / beta / factor) * factor
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        bar_h = math.ceil(height * beta / factor) * factor
        bar_w = math.ceil(width * beta / factor) * factor
    return bar_h, bar_w


@dataclass
class CameraFrame:
    """One camera image.

    ``target_size`` is an optional ``(width, height)`` the image is resized to before it
    is snapped to the patch grid. The released benchmark metadata carries these sizes,
    which is what keeps history frames at ~320p and the current frame at ~720p. Leaving
    it unset falls back to the pixel budgets in the model config.
    """

    image: str | Path | Image.Image | Callable[[], Image.Image]
    target_size: tuple[int, int] | None = None

    def load(self) -> Image.Image:
        if isinstance(self.image, Image.Image):
            return self.image.convert("RGB")
        if callable(self.image):
            return self.image().convert("RGB")
        return Image.open(self.image).convert("RGB")


@dataclass
class DrivingScene:
    """Everything the model needs for one planning query.

    All trajectory quantities live in the ego frame of the current timestamp: ``x``
    forward, ``y`` left, heading positive for a left turn. History runs oldest to
    newest at 10 Hz and its last pose is the current one, ``(0, 0, 0)``.
    """

    views: Mapping[str, Sequence[CameraFrame]]
    history: np.ndarray
    history_velocity: np.ndarray
    history_acceleration: np.ndarray
    ego_velocity: Sequence[float]
    ego_acceleration: Sequence[float]
    driving_command: Sequence[float]
    nav_command: int
    instruction_text: str | None = None
    token: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # History keeps its incoming precision: casting to float32 here would
        # round the 4th decimal printed in the instruction text one digit off
        # the training data. The expert casts to float32 where it matters.
        self.history = np.asarray(self.history)
        self.history_velocity = np.asarray(self.history_velocity)
        self.history_acceleration = np.asarray(self.history_acceleration)
        missing = [view for view in CAMERA_VIEWS if view not in self.views]
        if missing:
            raise ValueError(f"scene {self.token!r} is missing camera views {missing}")

    @property
    def ego_status(self) -> np.ndarray:
        """Velocity, acceleration and the one-hot driving command, in that order."""
        return np.concatenate(
            [
                np.asarray(self.ego_velocity, dtype=np.float32),
                np.asarray(self.ego_acceleration, dtype=np.float32),
                np.asarray(self.driving_command, dtype=np.float32),
            ]
        )

    @property
    def num_camera_frames(self) -> int:
        return len(self.views[CAMERA_VIEWS[0]])

    def frames_in_order(self) -> list[CameraFrame]:
        """All frames, grouped by view then by timestamp."""
        return [frame for view in CAMERA_VIEWS for frame in self.views[view]]

    def instruction(self) -> str:
        """The text block that follows the images in the user turn.

        Scenes read from a benchmark file carry this verbatim, so the evaluated prompt is
        reproduced exactly. For scenes built from scratch it is synthesized from the
        history poses and the navigation command.
        """
        if self.instruction_text is not None:
            return self.instruction_text
        num_frames = self.num_camera_frames
        labels = HISTORY_FRAME_LABELS[-num_frames:]
        stride = max(1, (len(self.history) - 1) // max(1, num_frames - 1))
        lines = "".join(
            " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *self.history[index * stride])
            for index, label in enumerate(labels)
        )
        header = _INSTRUCTION_HEADER.format(
            num_frames=num_frames,
            interval=1.5 / max(1, num_frames - 1),
            first_label=labels[0],
            last_frame=num_frames - 1,
        )
        command = NAV_COMMANDS[int(self.nav_command)]
        return f"{header}{lines}2. Active navigation command: [{command}]"


class QwenDriveProcessor:
    """Turns a :class:`DrivingScene` into model inputs.

    Images go through two resizes: the optional per-frame ``target_size``, then a snap to
    a whole number of ``patch_size * merge_size`` blocks. Pixels are scaled to ``[0, 1]``
    and normalized with mean and standard deviation ``0.5``.
    """

    def __init__(self, tokenizer, config: QwenDriveConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.patch_size = config.image_patch_size
        self.merge_size = config.image_spatial_merge_size
        self.temporal_patch_size = config.image_temporal_patch_size
        self.factor = self.patch_size * self.merge_size
        self.min_pixels = 4 * self.factor**2
        self.grid_pixel_limit = 12800 * self.factor**2
        vlm = config.vlm_config
        self.image_token_id = vlm.image_token_id
        self.vision_start_id = vlm.vision_start_token_id
        self.vision_end_id = vlm.vision_end_token_id
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    # ---- images ----

    def _patchify(self, frame: CameraFrame, budget: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize one frame onto the patch grid and flatten it into patches.

        ``budget`` caps the pixel count only for frames without an explicit
        ``target_size``. A frame that carries one has already been sized deliberately, so
        it is snapped to the grid without being shrunk further.
        """
        image = frame.load()
        max_pixels = budget
        if frame.target_size is not None:
            width, height = frame.target_size
            image = TF.resize(image, [height, width], interpolation=InterpolationMode.BICUBIC)
            max_pixels = self.grid_pixel_limit
        width, height = image.size
        grid_height, grid_width = smart_resize(
            height, width, self.factor, self.min_pixels, max_pixels
        )
        image = TF.resize(
            image, [grid_height, grid_width], interpolation=InterpolationMode.BICUBIC
        )
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

    def encode_images(self, scene: DrivingScene) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Return flattened patches, ``image_grid_thw`` and the token count per image."""
        frames = scene.frames_in_order()
        per_view = scene.num_camera_frames
        patch_list, grids, token_counts = [], [], []
        for index, frame in enumerate(frames):
            is_current = (index % per_view) == per_view - 1
            budget = (
                self.config.current_image_pixels if is_current else self.config.history_image_pixels
            )
            patches, (rows, cols) = self._patchify(frame, budget)
            patch_list.append(patches)
            grids.append((1, rows, cols))
            token_counts.append(rows * cols // self.merge_size**2)
        return (
            torch.cat(patch_list, dim=0),
            torch.tensor(grids, dtype=torch.long),
            token_counts,
        )

    # ---- text ----

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_input_ids(
        self, scene: DrivingScene, token_counts: Sequence[int], with_reasoning: bool
    ) -> list[int]:
        """Compose the ChatML prompt.

        With ``with_reasoning`` the prompt stops after the assistant header so the model
        can write its reasoning. Otherwise an empty assistant turn is closed immediately,
        which is the format the model was trained to plan from.
        """
        body: list[int] = []
        per_view = scene.num_camera_frames
        for view_index, view in enumerate(CAMERA_VIEWS):
            body += self._encode(view)
            for frame_index in range(per_view):
                body += self._encode(f"frame: {frame_index}")
                count = token_counts[view_index * per_view + frame_index]
                body += [self.vision_start_id] + [self.image_token_id] * count + [self.vision_end_id]
        instruction = scene.instruction() + (REASONING_REQUEST if with_reasoning else "")
        body += self._encode(instruction)

        assistant_header = (
            [self.im_start_id] + self._encode("assistant") + self.newline_ids
        )
        prompt = (
            [self.im_start_id]
            + self._encode("user")
            + self.newline_ids
            + body
            + [self.im_end_id]
            + self.newline_ids
            + assistant_header
        )
        if not with_reasoning:
            prompt += [self.im_end_id] + self.newline_ids
        return prompt

    # ---- everything at once ----

    def encode_vqa(
        self,
        images: Sequence[str | Path | Image.Image | CameraFrame],
        question: str,
        system: str | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Encode a free-form question about a list of images."""
        frames = [
            image if isinstance(image, CameraFrame) else CameraFrame(image) for image in images
        ]
        patch_list, grids, token_counts = [], [], []
        for frame in frames:
            patches, (rows, cols) = self._patchify(frame, self.config.current_image_pixels)
            patch_list.append(patches)
            grids.append((1, rows, cols))
            token_counts.append(rows * cols // self.merge_size**2)

        body: list[int] = []
        for count in token_counts:
            body += [self.vision_start_id] + [self.image_token_id] * count + [self.vision_end_id]
        body += self._encode(question)

        prompt: list[int] = []
        if system:
            prompt += (
                [self.im_start_id]
                + self._encode("system")
                + self.newline_ids
                + self._encode(system)
                + [self.im_end_id]
                + self.newline_ids
            )
        prompt += (
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
        return {
            "input_ids": torch.tensor([prompt], dtype=torch.long, device=device),
            "pixel_values": torch.cat(patch_list, dim=0).to(device),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long, device=device),
        }

    def __call__(
        self, scene: DrivingScene, with_reasoning: bool = False, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        pixel_values, image_grid_thw, token_counts = self.encode_images(scene)
        input_ids = self.build_input_ids(scene, token_counts, with_reasoning)
        history = torch.from_numpy(scene.history).unsqueeze(0)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "pixel_values": pixel_values.to(device),
            "image_grid_thw": image_grid_thw.to(device),
            "history": history.to(device),
            "history_velocity": torch.from_numpy(scene.history_velocity).unsqueeze(0).to(device),
            "history_acceleration": (
                torch.from_numpy(scene.history_acceleration).unsqueeze(0).to(device)
            ),
            "ego_status": torch.from_numpy(scene.ego_status).unsqueeze(0).to(device),
            "nav_command": torch.tensor([scene.nav_command], dtype=torch.long, device=device),
        }
