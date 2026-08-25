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

"""3D U-Net that refines the occupancy feature volume."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OccVoxelUNetRefiner"]


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(out_channels)
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out)) + identity
        return self.relu(out)


class ResidualStage3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1), num_blocks: int = 2):
        super().__init__()
        blocks = [BasicBlock3D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock3D(out_channels, out_channels, stride=(1, 1, 1)))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class OccVoxelUNetRefiner(nn.Module):
    """Residual 3D U-Net that downsamples only in XY while preserving Z."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c0 = max(64, out_channels * 2)
        c1 = c0 * 2
        c2 = c0 * 4
        c3 = c0 * 6

        self.input_proj = ResidualStage3D(in_channels, c0, num_blocks=2)
        self.enc1 = ResidualStage3D(c0, c1, stride=(1, 2, 2), num_blocks=2)
        self.enc2 = ResidualStage3D(c1, c2, stride=(1, 2, 2), num_blocks=2)
        self.enc3 = ResidualStage3D(c2, c3, stride=(1, 2, 2), num_blocks=2)
        self.bottleneck = ResidualStage3D(c3, c3, num_blocks=2)

        self.dec2 = ResidualStage3D(c3 + c2, c2, num_blocks=2)
        self.dec1 = ResidualStage3D(c2 + c1, c1, num_blocks=2)
        self.dec0 = ResidualStage3D(c1 + c0, c0, num_blocks=2)

        self.out_block = ResidualStage3D(c0, c0, num_blocks=2)
        self.out_proj = nn.Sequential(
            nn.Conv3d(c0, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upsample_to(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        skip0 = self.input_proj(x)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        x = self.enc3(skip2)
        x = self.bottleneck(x)

        x = self._upsample_to(x, skip2)
        x = self.dec2(torch.cat([x, skip2], dim=1))

        x = self._upsample_to(x, skip1)
        x = self.dec1(torch.cat([x, skip1], dim=1))

        x = self._upsample_to(x, skip0)
        x = self.dec0(torch.cat([x, skip0], dim=1))

        return self.out_proj(self.out_block(x))
