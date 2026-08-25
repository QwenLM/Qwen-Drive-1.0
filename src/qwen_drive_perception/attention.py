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

"""Deformable-attention modules of the BEV encoder and the detection decoder.

Multi-scale deformable attention follows:

    Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object
    Detection", ICLR 2021. https://arxiv.org/abs/2010.04159

    Li et al., "BEVFormer: Learning Bird's-Eye-View Representation from
    Multi-Camera Images via Spatiotemporal Transformers", ECCV 2022.
    https://arxiv.org/abs/2203.17270

The operator under ``ops/`` is our own bfloat16 implementation: values, sampling
locations and output stay in bf16 storage while the bilinear interpolation and
the weighted accumulation run in fp32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import multi_scale_deformable_attn_pytorch

__all__ = [
    "TemporalSelfAttention",
    "SpatialCrossAttention",
    "MSDeformableAttention3D",
    "CustomMSDeformableAttention",
]


def multi_scale_deformable_attn_cuda(
    value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
):
    """Dispatch to our bf16 CUDA kernel, or to the torch fallback off-GPU."""
    if not value.is_cuda:
        return multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
    if value.dtype != torch.bfloat16:
        raise TypeError(
            "the ms_deform_attn kernel requires CUDA bfloat16 tensors, "
            f"got {value.dtype}; run the model in bfloat16"
        )
    from .ops import ms_deform_attn_bf16_forward

    return ms_deform_attn_bf16_forward(
        value.contiguous(),
        value_spatial_shapes.contiguous(),
        value_level_start_index.contiguous(),
        sampling_locations.contiguous(),
        attention_weights.contiguous(),
        int(im2col_step),
    )


class TemporalSelfAttention(nn.Module):
    """BEV self-attention. Single-frame inference degenerates the BEV queue to
    ``[query, query]``, i.e. plain deformable self-attention over BEV."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4, num_bev_queue=2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = True
        self.sampling_offsets = nn.Linear(embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
            .contiguous()
        )
        sampling_offsets = (
            sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
            None, None, None, :, None, :
        ]

        output = multi_scale_deformable_attn_cuda(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64
        )
        # (bs*2, num_query, embed_dims) -> mean over the BEV queue -> (bs, num_query, embed_dims)
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)
        output = self.output_proj(output.permute(2, 0, 1))
        return self.dropout(output) + identity


class MSDeformableAttention3D(nn.Module):
    """Deformable attention over multi-scale multi-camera image features, used
    inside the spatial cross-attention without an output projection."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = True
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = None

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # Each BEV query carries num_Z_anchors reference points per image, and the
        # sampling points are folded into the num_points axis.
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        bs, num_query, num_Z_anchors, xy = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy
        )
        sampling_locations = reference_points + sampling_offsets
        sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy)

        return multi_scale_deformable_attn_cuda(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64
        )


class SpatialCrossAttention(nn.Module):
    """Camera-aware cross-attention: each BEV query only attends to cameras it
    projects into (vectorized top-k + scatter-add rebatch)."""

    def __init__(self, embed_dims=256, pc_range=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.dropout = nn.Dropout(0.1)
        self.deformable_attention = MSDeformableAttention3D(embed_dims=embed_dims, num_heads=8, num_levels=4, num_points=8)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        num_cams = reference_points_cam.size(0)
        D = reference_points_cam.size(3)

        mask_any = bev_mask[:, 0].any(dim=-1)
        valid_counts = mask_any.to(torch.int32).sum(dim=-1)
        max_len = int(valid_counts.max().item())
        rebatch_indices = mask_any.to(torch.int32).topk(max_len, dim=-1, largest=True).indices

        query_index = rebatch_indices[None, :, :, None].expand(bs, num_cams, max_len, self.embed_dims)
        queries_rebatch = query[:, None, :, :].expand(bs, num_cams, num_query, self.embed_dims).gather(2, query_index)

        reference_points_index = rebatch_indices[None, :, :, None, None].expand(bs, num_cams, max_len, D, 2)
        reference_points_rebatch = reference_points_cam.permute(1, 0, 2, 3, 4).gather(2, reference_points_index)

        key_num_cams, l, key_bs, _ = key.shape
        key = key.permute(2, 0, 1, 3).reshape(bs * num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * num_cams, l, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * num_cams, max_len, D, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, num_cams, max_len, self.embed_dims)

        valid_query_mask = torch.arange(max_len, device=query.device)[None, :] < valid_counts[:, None]
        queries = queries * valid_query_mask[None, :, :, None].to(queries.dtype)
        scatter_index = rebatch_indices.reshape(1, num_cams * max_len, 1).expand(bs, -1, self.embed_dims)
        slots.scatter_add_(1, scatter_index, queries.reshape(bs, num_cams * max_len, self.embed_dims))

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0).to(queries.dtype)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


class CustomMSDeformableAttention(nn.Module):
    """Deformable cross-attention used by the detection decoder."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = False
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
            None, None, None, :, None, :
        ]

        output = multi_scale_deformable_attn_cuda(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=64
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
