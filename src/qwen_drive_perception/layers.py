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

"""The handful of building blocks the perception head used to import from
mmcv / mmdet / mmdet3d, reimplemented as plain ``torch.nn`` modules.

Attribute names are reproduced exactly (``gn1`` vs ``bn1`` for a GroupNorm
inside a resnet BasicBlock, ``conv``/``bn``/``activate`` inside a ConvModule,
``layers.0.0`` inside an FFN), so the released checkpoint loads without any
key remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_norm_layer",
    "ConvModule",
    "BasicBlock",
    "FFN",
    "MultiheadAttention",
    "LearnedPositionalEncoding",
    "inverse_sigmoid",
    "multi_scale_deformable_attn_pytorch",
]

_NORM_ABBR = {"GN": "gn", "LN": "ln", "BN": "bn", "BN1d": "bn", "BN2d": "bn", "BN3d": "bn", "SyncBN": "bn"}


def build_norm_layer(cfg: dict, num_features: int, postfix=""):
    """Return ``(name, layer)`` with the mmcv abbreviation, e.g. ``('gn1', GroupNorm)``."""
    kind = cfg["type"]
    abbr = _NORM_ABBR[kind]
    if kind == "GN":
        layer = nn.GroupNorm(cfg["num_groups"], num_features, eps=cfg.get("eps", 1e-5))
    elif kind == "LN":
        layer = nn.LayerNorm(num_features, eps=cfg.get("eps", 1e-5))
    elif kind == "BN3d":
        layer = nn.BatchNorm3d(num_features, eps=cfg.get("eps", 1e-5))
    else:
        layer = nn.BatchNorm2d(num_features, eps=cfg.get("eps", 1e-5))
    return abbr + str(postfix), layer


def _build_activation(cfg: dict | None):
    if cfg is None:
        return nn.Identity()
    return nn.ReLU(inplace=cfg.get("inplace", False))


class ConvModule(nn.Module):
    """Conv -> Norm -> Act with mmcv attribute names.

    Only the shapes used by the perception head are supported: ``Conv2d`` or
    ``Conv3d`` (``conv_cfg=dict(type='Conv3d')``) followed by a norm layer and
    a ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
    ):
        super().__init__()
        is_3d = conv_cfg is not None and conv_cfg.get("type") == "Conv3d"
        conv_cls = nn.Conv3d if is_3d else nn.Conv2d
        self.conv = conv_cls(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # mmcv registers the norm under its bare abbreviation, so a BatchNorm3d
        # lands at ``self.bn`` and a GroupNorm at ``self.gn``.
        self.norm_name = None
        if norm_cfg is not None:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        self.activate = _build_activation(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_name is not None:
            x = getattr(self, self.norm_name)(x)
        return self.activate(x)


class BasicBlock(nn.Module):
    """mmdet resnet BasicBlock with configurable norm (GroupNorm here)."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride=1, downsample=None, norm_cfg: dict | None = None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN")
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.add_module(norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.add_module(norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.norm1_name = norm1_name
        self.norm2_name = norm2_name

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class FFN(nn.Module):
    """mmcv FFN: two linear layers with the parameter names ``layers.0.0`` and ``layers.1``."""

    def __init__(self, embed_dims: int = 256, feedforward_channels: int = 512, ffn_drop: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True), nn.Dropout(ffn_drop)),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_drop),
        )

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + out


class MultiheadAttention(nn.Module):
    """mmcv MultiheadAttention wrapper around ``torch.nn.MultiheadAttention``."""

    def __init__(self, embed_dims: int = 256, num_heads: int = 8, dropout: float = 0.0, batch_first: bool = False):
        super().__init__()
        self.embed_dims = embed_dims
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        out = self.attn(query=query, key=key, value=value)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + out


class LearnedPositionalEncoding(nn.Module):
    """mmcv learned positional encoding for BEV queries."""

    def __init__(self, num_feats: int, row_num_embed: int, col_num_embed: int):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """``mask`` is ``[bs, 1, h, w]``, returns ``[bs, 2*num_feats, h, w]``."""
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.size(0), 1, 1, 1)
        )
        return pos


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure-torch fallback for the multi-scale deformable attention CUDA kernel."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, int(H_), int(W_))
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (
        torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
    ).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()
