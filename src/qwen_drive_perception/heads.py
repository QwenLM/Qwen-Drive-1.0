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

"""Detection / occupancy / map heads and the NMS-free box decoder.

Box format throughout: ``[x, y, z, w, l, h, yaw, vx, vy]`` with (x, y, z) at
the box center, sizes in metres, yaw around +Z. Detection outputs are in the
**lidar** frame. The head regresses in ego coordinates and converts back.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from . import geometry
from .layers import LearnedPositionalEncoding
from .map_seg import MapSegEncode
from .perception_transformer import PerceptionTransformer

__all__ = ["BEVFormerHead", "NMSFreeCoder", "denormalize_bbox"]


def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()
    vx = normalized_bboxes[..., 8:9]
    vy = normalized_bboxes[..., 9:10]
    return torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)


class NMSFreeCoder:
    """Top-k box decoder without NMS, filtered by a post-center range."""

    def __init__(self, pc_range, post_center_range, max_num: int = 300, num_classes: int = 7):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.num_classes = num_classes

    def decode_single(self, cls_scores, bbox_preds):
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(self.max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)

        post_center_range = torch.tensor(self.post_center_range, device=scores.device)
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(1)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(1)

        return {
            "bboxes": final_box_preds[mask],
            "scores": scores[mask],
            "labels": labels[mask],
        }

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        return [
            self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(all_cls_scores.size(0))
        ]


def _calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)
    return bev_resolution, bev_start_position, bev_dimension


class BevFeatureSlicer(nn.Module):
    """Crop the detection BEV (ego frame) to the map-segmentation window.

    The sampling grid is fixed, so the only state is the ``bev_start_position``
    buffer; the released checkpoint carries its bf16-rounded values, which are
    loaded verbatim rather than recomputed.
    """

    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        bev_resolution, bev_start_position, _ = _calculate_birds_eye_view_parameters(
            grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
        )
        self.register_buffer("bev_start_position", bev_start_position)

        map_res_x = map_grid_conf["xbound"][2]
        map_res_y = map_grid_conf["ybound"][2]
        map_start_x = map_grid_conf["xbound"][0] + map_res_x / 2.0
        map_start_y = map_grid_conf["ybound"][0] + map_res_y / 2.0
        # Grid is rebuilt lazily: from_pretrained instantiates on the meta
        # device, where plain-attribute tensors would stay empty.
        self._map_params = (
            map_start_x,
            map_grid_conf["xbound"][1],
            map_res_x,
            map_start_y,
            map_grid_conf["ybound"][1],
            map_res_y,
            -float(grid_conf["xbound"][0] + grid_conf["xbound"][2] / 2.0),
            -float(grid_conf["ybound"][0] + grid_conf["ybound"][2] / 2.0),
        )
        self._map_grid = None

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        if self._map_grid is None or self._map_grid.device != x.device:
            sx0, sx1, dx, sy0, sy1, dy, nx, ny = self._map_params
            norm_x = torch.arange(sx0, sx1, dx, device=x.device) / nx
            norm_y = torch.arange(sy0, sy1, dy, device=x.device) / ny
            grid_y, grid_x = torch.meshgrid(norm_y, norm_x, indexing="ij")
            self._map_grid = torch.stack((grid_x, grid_y), dim=2)
        return self._map_grid

    def forward(self, x):
        grid = self._grid(x).unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1, 1)
        return nn.functional.grid_sample(x, grid=grid, mode="bilinear", align_corners=True)


class BEVFormerHead(nn.Module):
    """Detection queries + occupancy + map segmentation over the BEV embedding."""

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        num_query: int = 900,
        num_classes: int = 7,
        embed_dims: int = 256,
        code_size: int = 10,
        num_reg_fcs: int = 2,
        det_pc_range=None,
        det_voxel_size=None,
        map_grid_conf=None,
        det_grid_conf=None,
        occ_pc_range=None,
        nuplan_occ_pc_range=None,
        occ_voxel_size=None,
        nuplan_occ_voxel_size=None,
        occ_pillar_h: int = 16,
        occ_dim: int = 32,
        occ_num_classes: int = 10,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        post_center_range=None,
        max_num: int = 300,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.code_size = code_size
        self.cls_out_channels = num_classes

        self.transformer = PerceptionTransformer(
            embed_dims=embed_dims,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            det_pc_range=det_pc_range,
            occ_pc_range=occ_pc_range,
            nuplan_occ_pc_range=nuplan_occ_pc_range,
            det_voxel_size=det_voxel_size,
            occ_voxel_size=occ_voxel_size,
            nuplan_occ_voxel_size=nuplan_occ_voxel_size,
            occ_pillar_h=occ_pillar_h,
            occ_dim=occ_dim,
            occ_num_classes=occ_num_classes,
        )
        self.bbox_coder = NMSFreeCoder(
            pc_range=det_pc_range, post_center_range=post_center_range, max_num=max_num, num_classes=num_classes
        )
        self.pc_range = list(det_pc_range)
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=embed_dims // 2, row_num_embed=bev_h, col_num_embed=bev_w
        )

        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)]
        cls_branch.append(nn.Linear(embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch += [nn.Linear(embed_dims, embed_dims), nn.ReLU()]
        reg_branch.append(nn.Linear(embed_dims, code_size))
        reg_branch = nn.Sequential(*reg_branch)

        num_pred = num_decoder_layers
        self.cls_branches = nn.ModuleList([copy.deepcopy(fc_cls) for _ in range(num_pred)])
        self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(num_pred)])

        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)

        self.feat_cropper = BevFeatureSlicer(det_grid_conf, map_grid_conf)
        self.seg_decoder = MapSegEncode(inC=embed_dims, outC=6)

    # ------------------------------------------------------------- forward

    def forward(
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        vit_bev_feat=None,
        uvtr_bev_feat=None,
        uvtr_occ_feat=None,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)

        if uvtr_bev_feat is not None:
            bev_queries = bev_queries + uvtr_bev_feat.to(dtype)
        elif vit_bev_feat is not None:
            bev_queries = bev_queries + vit_bev_feat.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches,
            cls_branches=None,
            img_metas=img_metas,
            prev_bev=prev_bev,
            uvtr_occ_feat=uvtr_occ_feat,
        )
        bev_embed, hs, init_reference, inter_references, occ_pred, bev_ego_feat_2d = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        from .layers import inverse_sigmoid

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "occ_pred": occ_pred,
        }
        return self._append_map_preds(outs, bs, bev_ego_feat_2d)

    def _append_map_preds(self, outs, bs, bev_ego_feat_2d):
        seg_bev = bev_ego_feat_2d
        seg_bev = self.feat_cropper(seg_bev)
        outs["seg_preds"] = self.seg_decoder(seg_bev)
        return outs

    # ------------------------------------------------------------ decoding

    @torch.no_grad()
    def get_bboxes(self, preds_dicts, img_metas):
        """Decode ego-frame boxes into the lidar frame.

        Returns one dict with ``boxes`` ``(N, 9)``, ``scores`` and ``labels``
        per sample, with z shifted from gravity-center to bottom-center.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        results = []
        for i, preds in enumerate(preds_dicts):
            bboxes = preds["bboxes"]
            if img_metas[i].get("box_coord_system") == "ego":
                lidar2ego = img_metas[i]["lidar2ego"]
                if isinstance(lidar2ego, torch.Tensor):
                    lidar2ego = lidar2ego[0]
                else:
                    lidar2ego = torch.as_tensor(lidar2ego[0], dtype=torch.float32)
                bboxes = geometry.ego_to_lidar_boxes(bboxes, lidar2ego.to(bboxes.device))
            bboxes = bboxes.clone()
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            results.append(
                {
                    "boxes": bboxes.float().cpu().numpy(),
                    "scores": preds["scores"].float().cpu().numpy(),
                    "labels": preds["labels"].cpu().numpy(),
                }
            )
        return results

    @torch.no_grad()
    def get_occ(self, preds_dicts, img_metas):
        occ_score = preds_dicts["occ_pred"].softmax(-1)
        return occ_score.argmax(-1)

    @torch.no_grad()
    def get_map_seg(self, preds_dicts, img_metas):
        seg_preds = preds_dicts["seg_preds"].softmax(1)
        return seg_preds.argmax(1)
