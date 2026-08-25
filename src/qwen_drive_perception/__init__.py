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

"""Qwen-Drive perception: BEV 3D detection, occupancy and map segmentation.

The perception head consumes two feature taps of the Qwen3.5 VLM and predicts
3D boxes, an occupancy grid and a BEV map segmentation in a single frame:

* ``img_llm_feats``: last decoder-layer hidden states at the image token
  positions, reshaped back to ``[num_cams, H/2, W/2, llm_dim]``.
* ``img_vit_feats``: ViT patch features from before the spatial merge, reshaped
  to ``[num_cams, H, W, vit_dim]``.

The network below (SimpleFPN adaptors, UVTR view transform, BEVFormer-style
encoder and decoder, occupancy refiner, map encoder and detection head) is the
single configuration the released weights were trained with. Alternatives that
existed during training have been removed.
"""

from .configuration_perception import QwenDrivePerceptionConfig
from .modeling_perception import QwenDrivePerception

__all__ = ["QwenDrivePerception", "QwenDrivePerceptionConfig"]
