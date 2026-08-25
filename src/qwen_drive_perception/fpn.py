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

"""SimpleFPN: ViTDet-style single-input feature pyramid (``adaptor`` / ``vit_neck``)."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SimpleFPN"]


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over NCHW, as in the ViTDet implementation."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleFPN(nn.Module):
    """Build a multi-scale pyramid from a single feature map.

    ``scale_factors=(4.0, 2.0, 1.0, 0.5)`` produces the four levels the BEV
    encoder's spatial cross-attention samples; ``(1.0,)`` is used for the ViT
    neck that feeds the depth net.
    """

    def __init__(self, dim: int, out_channels: int = 256, scale_factors=(4.0, 2.0, 1.0, 0.5)):
        super().__init__()
        self.scale_factors = scale_factors
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            layers = []
            out_dim = dim
            if scale == 4.0:
                layers += [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers.append(nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2))
                out_dim = dim // 2
            elif scale == 1.0:
                pass
            elif scale == 0.5:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported")
            layers += [
                nn.Conv2d(out_dim, out_channels, kernel_size=1, bias=False),
                LayerNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_channels),
            ]
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        """``x`` is ``[B, C, H, W]``, returns a tuple of multi-scale feature maps."""
        return tuple(stage(x) for stage in self.stages)
