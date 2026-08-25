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

"""CUDA operators of the perception head, JIT-compiled on first use.

Both kernels are our own implementations, written for this model:

* ``ms_deform_attn_bf16``: multi-scale deformable attention in bfloat16. Values,
  sampling locations and output stay in bf16 storage while the bilinear
  interpolation and the weighted accumulation run in fp32.
* ``voxel_pool``: depth-aware voxel pooling of the view transform, fusing the
  per-pixel depth distribution and the feature scatter into a single pass.

Compilation needs a CUDA toolkit (``CUDA_HOME``/``nvcc``) matching the torch
build and takes a couple of minutes the first time. The result is cached under
``~/.cache/torch_extensions``.
"""

import os

import torch
from torch.utils.cpp_extension import load

__all__ = ["ms_deform_attn_bf16_forward", "voxel_pool_depth"]


def _load(name: str, sources, cuda_flags):
    return load(
        name=name,
        sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), s) for s in sources],
        extra_cuda_cflags=cuda_flags,
        verbose=os.getenv("QWEN_DRIVE_PERCEPTION_OPS_VERBOSE", "0") == "1",
    )


_MS_DEFORM_ATTN_BF16_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]


def _load_ms_deform_attn_bf16():
    return _load(
        "qwen_drive_ms_deform_attn_bf16",
        ["ms_deform_attn_bf16/src/ms_deform_attn_bf16.cpp", "ms_deform_attn_bf16/src/ms_deform_attn_bf16_cuda.cu"],
        _MS_DEFORM_ATTN_BF16_FLAGS,
    )


def _load_voxel_pool():
    return _load(
        "qwen_drive_voxel_pool",
        ["voxel_pool/src/voxel_pool.cpp", "voxel_pool/src/voxel_pool_cuda.cu"],
        ["--use_fast_math"],
    )


def ms_deform_attn_bf16_forward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
    """Multi-scale deformable attention forward (bf16 values, fp32 math)."""
    return _load_ms_deform_attn_bf16().forward(
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights, int(im2col_step)
    )


class _VoxelPoolDepthCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img_feats, img_depth, coords, point_indices, ranks, B, N_sweep, N_cam, X, Y, Z, D, H, W):
        coords = coords.int().contiguous()
        point_indices = point_indices.int().contiguous()
        ranks = ranks.contiguous()
        if ranks.numel() == 0:
            sort_indices = torch.empty((0,), device=ranks.device, dtype=torch.int32)
            interval_starts = torch.empty((0,), device=ranks.device, dtype=torch.int32)
            interval_lengths = torch.empty((0,), device=ranks.device, dtype=torch.int32)
        else:
            sort_indices_long = ranks.argsort()
            sorted_ranks = ranks[sort_indices_long]
            kept = torch.ones(sorted_ranks.shape[0], device=sorted_ranks.device, dtype=torch.bool)
            kept[1:] = sorted_ranks[1:] != sorted_ranks[:-1]
            interval_starts = torch.where(kept)[0].int()
            interval_lengths = torch.zeros_like(interval_starts)
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
            interval_lengths[-1] = sorted_ranks.shape[0] - interval_starts[-1]
            sort_indices = sort_indices_long.int()

        out = _load_voxel_pool().voxel_pool_depth_forward_all(
            img_feats,
            img_depth,
            coords,
            point_indices,
            sort_indices,
            interval_lengths,
            interval_starts,
            B,
            N_sweep,
            N_cam,
            X,
            Y,
            Z,
            D,
            H,
            W,
        )
        out_dtype = torch.promote_types(img_feats.dtype, img_depth.dtype)
        return out.to(out_dtype)


def voxel_pool_depth(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z):
    """Fuse image features with depth distributions into a voxel volume.

    ``voxel_coords``/``mask``: ``[B, N, 1, D, H, W, 4]`` / ``[B, N, 1, D, H, W]``
    where N is the camera count.
    """
    assert img_feats.dim() == 5
    assert img_depth.dim() == 4
    assert voxel_coords.dim() == 7
    assert mask.shape == voxel_coords.shape[:-1]

    img_feats = img_feats.contiguous()
    img_depth = img_depth.contiguous()
    voxel_coords = voxel_coords.contiguous()
    mask = mask.contiguous()

    _, n_images, feat_channels, H, W = img_feats.shape
    _, N_sweep, N_cam, D, H_coords, W_coords = mask.shape
    assert H == H_coords and W == W_coords
    assert n_images == N_sweep * N_cam
    assert img_depth.shape[0] == B * N_sweep * N_cam
    assert img_depth.shape[1:] == (D, H, W)

    flat_mask = mask.reshape(-1)
    flat_coords = voxel_coords.reshape(-1, 4)
    point_indices_all = torch.arange(B * N_sweep * N_cam * D * H * W, device=mask.device, dtype=torch.int32)
    point_indices = point_indices_all[flat_mask]
    coords = flat_coords[flat_mask].int()
    camera_indices = (point_indices // (D * H * W)) % N_cam
    ranks = (
        coords[:, 0] * N_cam * X * Y * Z
        + camera_indices * X * Y * Z
        + coords[:, 1] * Y * Z
        + coords[:, 2] * Z
        + coords[:, 3]
    )
    return _VoxelPoolDepthCuda.apply(
        img_feats, img_depth, coords, point_indices, ranks, B, N_sweep, N_cam, X, Y, Z, D, H, W
    )
