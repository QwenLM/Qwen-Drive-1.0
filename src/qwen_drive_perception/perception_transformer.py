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

"""The perception transformer: BEV encoder + detection decoder + occupancy branch.

Coordinate conventions (all frozen from the released training configuration):

* The BEV encoder works in the **ego** frame over ``det_pc_range`` (200x200
  grid, 10.4 cm cells, z in [-5.0, 5.4]).
* The occupancy branch crops/resamples that ego volume into the dataset's
  occ range (nuScenes [-40,40]x[-1,5.4], nuPlan [-50,50]x[-4,4]) on a
  200x200x16 grid with axes **X forward, Y left, Z up**.
* Detection boxes are regressed in ego coordinates and converted back to the
  lidar frame in :meth:`BEVFormerHead.get_bboxes`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_encoder import BEVFormerEncoder, DetectionTransformerDecoder
from .layers import ConvModule
from .occ_refiner import OccVoxelUNetRefiner

__all__ = ["PerceptionTransformer"]


class PerceptionTransformer(nn.Module):
    """BEVFormer-style transformer with the occupancy and UVTR fusion branches."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        det_pc_range=None,
        occ_pc_range=None,
        nuplan_occ_pc_range=None,
        det_voxel_size=None,
        occ_voxel_size=None,
        nuplan_occ_voxel_size=None,
        occ_pillar_h: int = 16,
        occ_dim: int = 32,
        occ_num_classes: int = 10,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.det_pc_range = list(det_pc_range)
        self.occ_pc_range = list(occ_pc_range)
        self.nuplan_occ_pc_range = list(nuplan_occ_pc_range)
        self.det_voxel_size = list(det_voxel_size)
        self.occ_voxel_size = list(occ_voxel_size)
        self.nuplan_occ_voxel_size = list(nuplan_occ_voxel_size)
        self.occ_pillar_h = occ_pillar_h
        self.occ_dim = occ_dim
        self.occ_num_classes = occ_num_classes
        self.num_feature_levels = 4
        self.fp16_enabled = False

        self.encoder = BEVFormerEncoder(num_layers=num_encoder_layers, pc_range=self.det_pc_range)
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.decoder = DetectionTransformerDecoder(num_layers=num_decoder_layers)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.middle_dims = self.embed_dims // self.occ_pillar_h
        self.occ_decoder = OccVoxelUNetRefiner(self.middle_dims, self.occ_dim)
        self.occ_pred_head = nn.Sequential(
            nn.Linear(self.occ_dim, self.occ_dim * 2),
            nn.Softplus(),
            nn.Linear(self.occ_dim * 2, self.occ_num_classes),
        )
        self.uvtr_occ_proj = nn.Conv3d(self.embed_dims, self.middle_dims, kernel_size=1, stride=1, padding=0)
        self.uvtr_occ_fuse = ConvModule(
            self.middle_dims * 2,
            self.middle_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(type="BN3d"),
            act_cfg=dict(type="ReLU", inplace=True),
        )

    # ---------------------------------------------------------------- BEV

    def get_bev_features(self, mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=None, **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            _, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )
        return bev_embed

    # ------------------------------------------------------------- occupancy

    def _axis_center_coords(self, target_min, target_max, source_min, source_max, source_size, target_size, device, dtype):
        source_size = max(int(source_size), 1)
        target_size = max(int(target_size), 1)
        source_step = (source_max - source_min) / max(float(source_size), 1e-6)
        source_center0 = source_min + 0.5 * source_step

        target_step = (target_max - target_min) / max(float(target_size), 1e-6)
        target_centers = target_min + (torch.arange(target_size, device=device, dtype=dtype) + 0.5) * target_step

        source_index = (target_centers - source_center0) / max(source_step, 1e-6)
        if source_size > 1:
            source_norm = 2.0 * source_index / float(source_size - 1) - 1.0
        else:
            source_norm = torch.zeros_like(source_index)
        return source_norm.clamp(-1.0, 1.0)

    def _get_occ_target_spec(self, img_metas):
        dataset_type = img_metas[0].get("dataset_type") if img_metas else None
        if dataset_type == "nuplan":
            return self.nuplan_occ_pc_range, self.nuplan_occ_voxel_size
        return self.occ_pc_range, self.occ_voxel_size

    def _normalized_occ_grid(self, bev_h, bev_w, device, dtype, occ_pc_range=None, occ_voxel_size=None):
        sx0, sy0, sz0, sx1, sy1, sz1 = self.det_pc_range
        occ_pc_range = self.occ_pc_range if occ_pc_range is None else occ_pc_range
        occ_voxel_size = self.occ_voxel_size if occ_voxel_size is None else occ_voxel_size
        tx0, ty0, tz0, tx1, ty1, tz1 = occ_pc_range

        target_w = max(int(round((tx1 - tx0) / max(float(occ_voxel_size[0]), 1e-6))), 1)
        target_h = max(int(round((ty1 - ty0) / max(float(occ_voxel_size[1]), 1e-6))), 1)
        target_d = self.occ_pillar_h

        x = self._axis_center_coords(tx0, tx1, sx0, sx1, bev_w, target_w, device, dtype)
        y = self._axis_center_coords(ty0, ty1, sy0, sy1, bev_h, target_h, device, dtype)
        z = self._axis_center_coords(tz0, tz1, sz0, sz1, self.occ_pillar_h, target_d, device, dtype)

        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1)
        return grid.unsqueeze(0), target_h, target_w

    def _adapt_volume_for_occ(self, volume_feat, bev_h, bev_w, occ_pc_range, occ_voxel_size):
        bs = volume_feat.shape[0]
        grid, _, _ = self._normalized_occ_grid(
            bev_h, bev_w, volume_feat.device, volume_feat.dtype, occ_pc_range=occ_pc_range, occ_voxel_size=occ_voxel_size
        )
        grid = grid.expand(bs, -1, -1, -1, -1)
        return F.grid_sample(volume_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _adapt_bev_for_occ(self, bev_feat, bev_h, bev_w, occ_pc_range, occ_voxel_size):
        """Reshape the ego BEV embedding into a 3D volume and crop it to the occ range."""
        bs = bev_feat.shape[0]
        bev_feat_3d = bev_feat.view(bs, -1, self.occ_pillar_h, bev_h, bev_w)
        return self._adapt_volume_for_occ(bev_feat_3d, bev_h, bev_w, occ_pc_range, occ_voxel_size)

    def _fuse_uvtr_occ_feat(self, bev_feat, uvtr_occ_feat):
        uvtr_occ_feat = self.uvtr_occ_proj(uvtr_occ_feat.to(dtype=bev_feat.dtype))
        return bev_feat + self.uvtr_occ_fuse(torch.cat([bev_feat, uvtr_occ_feat], dim=1))

    # ------------------------------------------------------------- forward

    def forward(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        uvtr_occ_feat=None,
        **kwargs,
    ):
        # Single-frame: prev_bev is accepted for interface parity and ignored.
        bev_embed = self.get_bev_features(
            mlvl_feats, bev_queries, bev_h, bev_w, bev_pos=bev_pos, **kwargs
        )

        bs = mlvl_feats[0].size(0)
        img_metas = kwargs.get("img_metas", [])
        active_occ_pc_range, active_occ_voxel_size = self._get_occ_target_spec(img_metas)

        # The encoder output is already in ego coordinates. Keep it 2D for the
        # map branch and as-is for the detection decoder.
        bev_feat_ego = bev_embed.reshape(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
        bev_embed_for_decoder = bev_embed

        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed_dec = bev_embed_for_decoder.permute(1, 0, 2)
        bev_spatial_shapes = query.new_tensor([[bev_h, bev_w]], dtype=torch.long)
        bev_level_start_index = bev_spatial_shapes.new_zeros((1,))

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed_dec,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=bev_spatial_shapes,
            level_start_index=bev_level_start_index,
            **kwargs,
        )
        inter_references_out = inter_references

        # Occupancy: crop the ego volume to the dataset's occ range and fuse
        # the UVTR voxel features.
        bev_feat = self._adapt_bev_for_occ(
            bev_feat_ego, bev_h, bev_w, occ_pc_range=active_occ_pc_range, occ_voxel_size=active_occ_voxel_size
        )
        if uvtr_occ_feat is not None:
            uvtr_occ_feat = self._adapt_volume_for_occ(
                uvtr_occ_feat, bev_h, bev_w, occ_pc_range=active_occ_pc_range, occ_voxel_size=active_occ_voxel_size
            )
            bev_feat = self._fuse_uvtr_occ_feat(bev_feat, uvtr_occ_feat)

        # [B, occ_dim, Z, Y, X] -> [B, X, Y, Z, occ_dim], matching the GT voxel order.
        occ_feat = self.occ_decoder(bev_feat).permute(0, 4, 3, 2, 1)
        occ_pred = self.occ_pred_head(occ_feat)

        return (
            bev_embed_dec,
            inter_states,
            init_reference_out,
            inter_references_out,
            occ_pred,
            bev_feat_ego,
        )
