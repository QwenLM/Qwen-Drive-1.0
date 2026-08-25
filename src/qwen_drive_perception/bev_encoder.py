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

"""BEV encoder and detection decoder transformer stacks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CustomMSDeformableAttention, SpatialCrossAttention, TemporalSelfAttention
from .layers import FFN, MultiheadAttention

__all__ = ["BEVFormerEncoder", "BEVFormerLayer", "DetectionTransformerDecoder", "DetrTransformerDecoderLayer"]

OPERATION_ORDER = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")


class BaseTransformerLayer(nn.Module):
    """Generic transformer layer with mmcv's attribute layout
    (``attentions``, ``ffns``, ``norms``) and post-norm execution order."""

    def __init__(self, attn_modules, ffn: FFN, batch_first: bool):
        super().__init__()
        self.operation_order = OPERATION_ORDER
        self.num_attn = len(attn_modules)
        self.batch_first = batch_first
        self.pre_norm = False
        self.attentions = nn.ModuleList(attn_modules)
        self.embed_dims = attn_modules[0].embed_dims
        self.ffns = nn.ModuleList([ffn])
        self.norms = nn.ModuleList([nn.LayerNorm(self.embed_dims) for _ in range(3)])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        norm_index = attn_index = ffn_index = 0
        identity = query
        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class BEVFormerLayer(BaseTransformerLayer):
    """Encoder layer: temporal self-attention over BEV + spatial cross-attention
    into the camera features, both deformable, plus FFN."""

    def __init__(self, pc_range, embed_dims: int = 256, ffn_channels: int = 512):
        super().__init__(
            attn_modules=[
                TemporalSelfAttention(embed_dims=embed_dims, num_heads=8, num_levels=1),
                SpatialCrossAttention(pc_range=pc_range, embed_dims=embed_dims),
            ],
            ffn=FFN(embed_dims=embed_dims, feedforward_channels=ffn_channels, ffn_drop=0.1),
            batch_first=True,
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        norm_index = attn_index = ffn_index = 0
        identity = query
        bev_spatial_shapes = query.new_tensor([[bev_h, bev_w]], dtype=torch.long)
        bev_level_start_index = bev_spatial_shapes.new_zeros((1,))

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    reference_points=ref_2d,
                    spatial_shapes=bev_spatial_shapes,
                    level_start_index=bev_level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


def _extract_lidar2ego(img_metas, ref_tensor):
    lidar2ego_list = []
    for img_meta in img_metas:
        lidar2ego = img_meta["lidar2ego"]
        if isinstance(lidar2ego, torch.Tensor):
            lidar2ego_mat = lidar2ego.float()
        else:
            lidar2ego_mat = ref_tensor.new_tensor(lidar2ego).float()
        if lidar2ego_mat.dim() == 3:
            lidar2ego_mat = lidar2ego_mat[0]
        lidar2ego_list.append(lidar2ego_mat)
    return torch.stack(lidar2ego_list, dim=0)


class BEVFormerEncoder(nn.Module):
    """Six BEVFormer layers over ego-anchored BEV queries."""

    def __init__(self, num_layers: int, pc_range, num_points_in_pillar: int = 4):
        super().__init__()
        self.pc_range = list(pc_range)
        self.num_points_in_pillar = num_points_in_pillar
        self.reference_points_coord_system = "ego"
        self.layers = nn.ModuleList(
            BEVFormerLayer(pc_range=self.pc_range) for _ in range(num_layers)
        )

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, device="cuda", dtype=torch.float):
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            return ref_3d[None].repeat(bs, 1, 1, 1)
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing="ij",
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        return ref_2d.repeat(bs, 1, 1).unsqueeze(2)

    def point_sampling(self, reference_points, pc_range, img_metas):
        """Project ego-frame 3D reference points into every camera, in fp32."""
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        if torch.is_tensor(lidar2img[0]):
            lidar2img = torch.stack(lidar2img, dim=0)
        else:
            lidar2img = reference_points.new_tensor(torch.asarray(lidar2img))
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4, 1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4)

        # Reference points are defined in ego coordinates: compose
        # lidar2img @ inv(lidar2ego) before projecting.
        ego2lidar = torch.linalg.inv(_extract_lidar2ego(img_metas, reference_points))
        ego2lidar = ego2lidar.view(1, B, 1, 1, 4, 4)
        lidar2img = torch.matmul(lidar2img.to(torch.float32), ego2lidar.to(torch.float32))
        reference_points_cam = torch.matmul(lidar2img, reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
        )
        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]
        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query,
        key,
        value,
        bev_h,
        bev_w,
        bev_pos,
        spatial_shapes,
        level_start_index,
        **kwargs,
    ):
        """``bev_query`` is ``(num_query, bs, embed_dims)``, returns the final BEV embedding."""
        img_metas = kwargs["img_metas"]

        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype
        )
        reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)

        # Single-frame: no shift and no history. prev_bev stays None, so each
        # layer's temporal self-attention builds its BEV queue from the current
        # query (torch.stack([query, query])), exactly as the trained path does.
        shift = bev_query.new_tensor([0, 0]).unsqueeze(0)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)

        for layer in self.layers:
            bev_query = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=None,
                **kwargs,
            )
        return bev_query


class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Decoder layer: standard self-attention over object queries + deformable
    cross-attention into the BEV embedding. ``batch_first`` is False, matching
    the mmcv layer sequence the weights were trained with."""

    def __init__(self, embed_dims: int = 256, ffn_channels: int = 512):
        super().__init__(
            attn_modules=[
                MultiheadAttention(embed_dims=embed_dims, num_heads=8, dropout=0.1, batch_first=False),
                CustomMSDeformableAttention(embed_dims=embed_dims, num_heads=8, num_levels=1, num_points=4),
            ],
            ffn=FFN(embed_dims=embed_dims, feedforward_channels=ffn_channels, ffn_drop=0.1),
            batch_first=False,
        )


class DetectionTransformerDecoder(nn.Module):
    """Six query-refining decoder layers with iterative reference-point updates."""

    def __init__(self, num_layers: int, embed_dims: int = 256, ffn_channels: int = 512):
        super().__init__()
        self.layers = nn.ModuleList(
            DetrTransformerDecoderLayer(embed_dims=embed_dims, ffn_channels=ffn_channels)
            for _ in range(num_layers)
        )
        self.return_intermediate = True

    def forward(
        self,
        query,
        value,
        query_pos,
        reference_points,
        reg_branches=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        from .layers import inverse_sigmoid

        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)
