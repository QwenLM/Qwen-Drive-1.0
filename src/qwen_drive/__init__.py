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

"""Qwen-Drive-1.0 inference package."""

from transformers import AutoConfig, AutoModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from .configuration_qwen_drive import PlanningExpertConfig, QwenDriveConfig
from .modeling_qwen_drive import InferenceMode, QwenDriveForPlanning, QwenDriveOutput
from .planning_expert import PlanningExpert
from .scene import CAMERA_VIEWS, NAV_COMMANDS, CameraFrame, DrivingScene, QwenDriveProcessor

__version__ = "1.0.0"

# Make a released model directory openable through the auto classes, which also stops
# transformers from warning about an unknown model type when it reads config.json.
if QwenDriveConfig.model_type not in CONFIG_MAPPING_NAMES:
    AutoConfig.register(QwenDriveConfig.model_type, QwenDriveConfig)
    AutoModel.register(QwenDriveConfig, QwenDriveForPlanning)

__all__ = [
    "CAMERA_VIEWS",
    "NAV_COMMANDS",
    "CameraFrame",
    "DrivingScene",
    "InferenceMode",
    "PlanningExpert",
    "PlanningExpertConfig",
    "QwenDriveConfig",
    "QwenDriveForPlanning",
    "QwenDriveOutput",
    "QwenDriveProcessor",
]
