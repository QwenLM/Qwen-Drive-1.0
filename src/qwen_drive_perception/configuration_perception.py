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

"""Frozen constants of the perception head.

The model was trained on nuScenes and OpenScene camera rigs at a fixed
input resolution of 896x512 per camera. Switching camera layouts or image
resolutions is not covered by the released weights and will likely degrade
results.
"""

from __future__ import annotations

from transformers import PretrainedConfig

DET_CLASS_NAMES = (
    "vehicle",
    "czone_sign",
    "bicycle",
    "generic_object",
    "pedestrian",
    "traffic_cone",
    "barrier",
)
"""Unified 7-class detection taxonomy (index order matters)."""

OCC_CLASS_NAMES = DET_CLASS_NAMES + ("driveable", "background", "empty")
"""Occupancy classes. ``empty`` is index 9 and is the argmax target for free voxels."""

MAP_CLASS_NAMES = (
    "background",
    "driveable_surface",
    "road_line",
    "road_edge",
    "crosswalk",
    "walkway",
)

OCC_PALETTE = (
    (255, 138, 0),
    (145, 80, 210),
    (35, 190, 105),
    (145, 110, 90),
    (235, 55, 85),
    (255, 214, 10),
    (112, 128, 144),
    (190, 190, 190),
    (110, 130, 105),
    (248, 248, 248),
)

MAP_PALETTE = (
    (250, 250, 250),
    (196, 205, 214),
    (255, 193, 7),
    (225, 95, 65),
    (75, 180, 170),
    (139, 195, 74),
)

DET_BOX_COLORS = {
    "vehicle": (255, 138, 0),
    "czone_sign": (155, 81, 224),
    "bicycle": (32, 191, 107),
    "generic_object": (141, 110, 99),
    "pedestrian": (235, 59, 90),
    "traffic_cone": (255, 214, 10),
    "barrier": (127, 140, 141),
}

GT_BOX_COLOR = (58, 175, 74)

OCC_EMPTY_LABEL = len(OCC_CLASS_NAMES) - 1
OCC_ROAD_LABEL = OCC_CLASS_NAMES.index("driveable")
OCC_BACKGROUND_LABEL = OCC_CLASS_NAMES.index("background")

DET_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 5.4)
DET_VOXEL_SIZE = (102.4 / 200, 102.4 / 200, 5.4 - -5.0)
NUSCENES_OCC_PC_RANGE = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
NUSCENES_OCC_VOXEL_SIZE = (80 / 200, 80 / 200, 6.4)
NUPLAN_OCC_PC_RANGE = (-50.0, -50.0, -4.0, 50.0, 50.0, 4.0)
NUPLAN_OCC_VOXEL_SIZE = (100.0 / 200, 100.0 / 200, 8.0 / 16)

MAP_XBOUND = (-30.0, 30.0, 0.15)
MAP_YBOUND = (-15.0, 15.0, 0.15)

FRUSTUM_RANGE = (0, 0, 1.0, 896, 512, 60.0)
FRUSTUM_SIZE = (16.0, 16.0, 0.5)

LLM_DIM = 2560
VIT_DIM = 1024
EMBED_DIM = 256
FFN_CHANNELS = 512
NUM_HEADS = 8
BEV_H = BEV_W = 200
OCC_PILLAR_H = 16
OCC_DIM = 32
NUM_QUERY = 900
CODE_SIZE = 10
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_FEATURE_LEVELS = 4
POINTS_IN_PILLAR = 4
SCA_NUM_POINTS = 8
TSA_NUM_POINTS = 4
DECODER_NUM_POINTS = 4
MAX_NUM_BOXES = 300
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
IMAGE_SIZE = (896, 512)  # (width, height) the model was trained with


class QwenDrivePerceptionConfig(PretrainedConfig):
    """Configuration of the perception head. All values are fixed."""

    model_type = "qwen_drive_perception"

    def __init__(self, **kwargs):
        self.llm_dim = LLM_DIM
        self.vit_dim = VIT_DIM
        self.embed_dim = EMBED_DIM
        self.det_pc_range = list(DET_PC_RANGE)
        self.det_voxel_size = list(DET_VOXEL_SIZE)
        self.nuscenes_occ_pc_range = list(NUSCENES_OCC_PC_RANGE)
        self.nuscenes_occ_voxel_size = list(NUSCENES_OCC_VOXEL_SIZE)
        self.nuplan_occ_pc_range = list(NUPLAN_OCC_PC_RANGE)
        self.nuplan_occ_voxel_size = list(NUPLAN_OCC_VOXEL_SIZE)
        self.map_xbound = list(MAP_XBOUND)
        self.map_ybound = list(MAP_YBOUND)
        self.frustum_range = list(FRUSTUM_RANGE)
        self.frustum_size = list(FRUSTUM_SIZE)
        self.bev_h = BEV_H
        self.bev_w = BEV_W
        self.occ_pillar_h = OCC_PILLAR_H
        self.occ_dim = OCC_DIM
        self.occ_num_classes = len(OCC_CLASS_NAMES)
        self.det_num_classes = len(DET_CLASS_NAMES)
        self.map_num_classes = len(MAP_CLASS_NAMES)
        self.num_query = NUM_QUERY
        self.code_size = CODE_SIZE
        self.num_encoder_layers = NUM_ENCODER_LAYERS
        self.num_decoder_layers = NUM_DECODER_LAYERS
        self.image_size = list(IMAGE_SIZE)
        super().__init__(**kwargs)
