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

"""UVTR view transform: depth prediction and voxel pooling into a 3D feature volume.

The frustum grid covers the 896x512 image plane with 16x16 bins and 0.5 m depth
steps up to 60 m (118 bins). For each frustum cell the predicted depth
distribution lifts the ViT features into ``[B, C, 16, 200, 200]`` ego-anchored
voxels via the fused ``voxel_pool_depth`` CUDA kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import BasicBlock, build_norm_layer

__all__ = ["ASPP", "DepthNet", "Uni3DVoxelPoolDepth"]


class _ASPPModule(nn.Module):
    def __init__(self, inplanes: int, planes: int, kernel_size: int, padding: int, dilation: int):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling with 32-group GroupNorm."""

    def __init__(self, inplanes: int, mid_channels: int):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(mid_channels * 5, inplanes, 1, bias=False)
        self.bn1 = nn.GroupNorm(32, inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


class DepthNet(nn.Module):
    """Depth distribution over the frustum for every image feature cell."""

    def __init__(self, in_channels: int, mid_channels: int, depth_channels: int, aspp_mid_channels: int):
        super().__init__()
        norm_cfg = dict(type="GN", num_groups=32)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            ASPP(mid_channels, aspp_mid_channels),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0),
        )
        self.depth_channels = depth_channels

    def forward(self, x):
        return self.depth_conv(self.reduce_conv(x))


class Uni3DVoxelPoolDepth(nn.Module):
    """Frustum unprojection + voxel pooling with three 3D conv layers."""

    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        frustum_range,
        frustum_size,
        embed_dim: int = 256,
        num_convs: int = 3,
        kernel_size=(3, 3, 3),
    ):
        super().__init__()
        self.pc_range = list(pc_range)
        self.voxel_size = list(voxel_size)
        self.voxel_shape = list(voxel_shape)
        self.frustum_size = list(frustum_size)
        self.frustum_range = list(frustum_range)
        self.depth_dim = int((frustum_range[5] - frustum_range[2]) / frustum_size[2])
        self._frustum = None

        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv_layer = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=True),
                nn.BatchNorm3d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_convs)
        )

    @property
    def frustum(self) -> torch.Tensor:
        """The (W, H, D, 3) frustum grid, in fp32 on the current device.

        Deliberately not a registered buffer: the original kept it as a plain
        attribute so ``model.to(bfloat16)`` never cast it, and the inverse
        projection in ``coord_preparing`` runs in fp32.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._frustum is None or self._frustum.device.type != device:
            self._frustum = torch.stack(
                torch.meshgrid(
                    [
                        torch.arange(self.frustum_range[i], self.frustum_range[i + 3], self.frustum_size[i], device=device)
                        for i in range(3)
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )
        return self._frustum

    def _format_lidar2img(self, lidar2img):
        if lidar2img.ndim == 4:
            lidar2img = lidar2img[:, :, None, ...]
        if lidar2img.ndim != 5:
            raise ValueError(f"Unexpected lidar2img shape: {tuple(lidar2img.shape)}")
        return lidar2img

    def forward(self, mlvl_feats, img_depth, img_metas):
        """``mlvl_feats`` is a list of one ``[B, N, C, H, W]`` tensor, ``img_depth`` the matching softmax depth."""
        for img_meta in img_metas:
            lidar2img = img_meta["lidar2img"][None, ...]
            img_meta["lidar2img"] = self._format_lidar2img(lidar2img)
        voxel_coords, mask = self.coord_preparing(img_metas)
        voxel_space = self.feat_sampling(mlvl_feats, img_depth, voxel_coords, mask)
        return self.feat_encoding(voxel_space)

    @staticmethod
    def _stack_lidar2img(img_metas, ref_tensor):
        lidar2img_list = []
        for img_meta in img_metas:
            lidar2img = img_meta["lidar2img"]
            if isinstance(lidar2img, torch.Tensor):
                lidar2img_mat = lidar2img.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
            else:
                lidar2img_mat = ref_tensor.new_tensor(lidar2img)
            lidar2img_list.append(lidar2img_mat)
        return torch.stack(lidar2img_list, dim=0)

    @staticmethod
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

    def coord_preparing(self, img_metas):
        """Unproject the frustum grid into ego-frame voxel indices."""
        B = len(img_metas)
        frustum = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        W, H, D = frustum.shape[1:-1]

        lidar2img = self._stack_lidar2img(img_metas, frustum)
        _, N, C = lidar2img.shape[:3]
        lidar2img = lidar2img.flatten(1, 2)

        frustum = (
            torch.cat([frustum, torch.ones_like(frustum[..., :1])], -1).flatten(1, 3).unsqueeze(1)
        )
        frustum[..., :2] *= frustum[..., 2:3]
        frustum = torch.matmul(torch.inverse(lidar2img).unsqueeze(2), frustum.unsqueeze(-1)).squeeze(-1)
        # Reference points live in the ego frame: bring the unprojected lidar
        # points back through inv(lidar2ego).
        lidar2ego = self._extract_lidar2ego(img_metas, frustum)
        frustum = torch.matmul(lidar2ego[:, None, None, :, :], frustum.unsqueeze(-1)).squeeze(-1)

        pc_range = frustum.new_tensor(self.pc_range)
        voxel_size = frustum.new_tensor(self.voxel_size)
        voxel_coords = ((frustum[..., :3] - pc_range[:3]) / voxel_size).int()
        batch_ix = torch.cat(
            [torch.full_like(voxel_coords[ix : ix + 1, ..., 0:1], ix) for ix in range(B)], dim=0
        )
        voxel_coords = torch.cat([batch_ix, voxel_coords], dim=-1)
        voxel_coords = (
            voxel_coords.view(B, N, C, W, H, D, 4).permute(0, 1, 2, 5, 4, 3, 6).contiguous()
        )

        mask = (
            (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < self.voxel_shape[0])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < self.voxel_shape[1])
            & (voxel_coords[..., 3] >= 0)
            & (voxel_coords[..., 3] < self.voxel_shape[2])
        )
        return voxel_coords, mask

    def feat_sampling(self, mlvl_feats, img_depth, voxel_coords, mask):
        from .ops import voxel_pool_depth

        assert len(mlvl_feats) == len(img_depth) == 1, "Only support single level feature"
        img_feats = mlvl_feats[0]
        img_depth = img_depth[0]
        B, X, Y, Z = len(img_feats), *self.voxel_shape
        return voxel_pool_depth(img_feats, img_depth, voxel_coords, mask, B, X, Y, Z).permute(0, 1, 5, 4, 3, 2)

    def feat_encoding(self, voxel_space):
        """Sum over the (single) sweep and run the 3D conv stack: [B,1,C,D,H,W] -> [B,C,D,H,W]."""
        B, num_sweep = voxel_space.shape[:2]
        voxel_space = voxel_space.flatten(0, 1)
        voxel_space = voxel_space.view(B, num_sweep, *voxel_space.shape[1:])
        voxel_space = voxel_space.sum(1)
        for layer in self.conv_layer:
            voxel_space = layer(voxel_space)
        return voxel_space
