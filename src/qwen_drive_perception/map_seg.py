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

"""BEV map segmentation decoder (``MapSegEncode``).

A resnet18-style trunk whose BatchNorm layers were replaced in-place by
32-group GroupNorm during training. The attribute names (``bn1``, ``bn2``)
survived the swap and are kept for checkpoint compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MapSegEncode"]


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class _GNBasicBlock(nn.Module):
    """resnet18 BasicBlock with GroupNorm under the original ``bn*`` names."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = _gn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = _gn(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class MapSegEncode(nn.Module):
    """Decode the cropped ego-BEV feature into joint6 map logits (200x400)."""

    def __init__(self, inC: int, outC: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = _gn(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.up1 = _Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    @staticmethod
    def _make_layer(inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                _gn(planes),
            )
        layers = [_GNBasicBlock(inplanes, planes, stride=stride, downsample=downsample)]
        layers += [_GNBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """``x`` is a ``[B, inC, 200, 400]`` ego-BEV feature, returns ``[B, outC, 200, 400]`` logits."""
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)
        x = self.up1(x2, x1)
        return self.up2(x)
