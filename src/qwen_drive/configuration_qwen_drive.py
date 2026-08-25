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

"""Configuration for Qwen-Drive-1.0."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class PlanningExpertConfig(PretrainedConfig):
    """Geometry of the planning expert, a joint-attention diffusion transformer.

    The expert reads the post-rotary keys/values of the VLM's grouped-query
    attention layers. ``kv_head_dim`` and ``num_key_value_heads`` therefore have to
    match the VLM exactly, and ``layers_per_kv`` consecutive expert layers share
    one VLM cache.
    """

    model_type = "qwen_drive_planning_expert"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 3584,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        layers_per_kv: int = 4,
        rms_norm_eps: float = 1e-5,
        time_embed_dim: int = 128,
        time_embed_scale: float = 1000.0,
        fourier_num_features: int = 16,
        fourier_max_frequency: float = 16.0,
        nav_command_classes: int = 3,
        ego_status_dim: int = 8,
        history_dynamics_dim: int = 2,
        rope_theta: float = 1.0e7,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] = (11, 11, 10),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layers_per_kv = layers_per_kv
        self.rms_norm_eps = rms_norm_eps
        self.time_embed_dim = time_embed_dim
        self.time_embed_scale = time_embed_scale
        self.fourier_num_features = fourier_num_features
        self.fourier_max_frequency = fourier_max_frequency
        self.nav_command_classes = nav_command_classes
        self.ego_status_dim = ego_status_dim
        self.history_dynamics_dim = history_dynamics_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.mrope_section = list(mrope_section)

    @property
    def num_kv_sources(self) -> int:
        return self.num_hidden_layers // self.layers_per_kv

    @property
    def attention_hidden_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_hidden_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


class QwenDriveConfig(PretrainedConfig):
    """Top-level configuration: a Qwen3.5 VLM plus a planning expert.

    Trajectory conventions
    ----------------------
    Waypoints are ``(x, y, heading)`` in the ego frame of the current timestamp:
    ``x`` forward, ``y`` left, ``heading`` a left turn when positive. The expert
    predicts ``num_future_points`` waypoints at ``trajectory_hz``, and history is
    ``num_history_points`` waypoints at the same rate whose last entry is the
    current pose ``(0, 0, 0)``.

    Every trajectory is normalized by dividing each channel by ``trajectory_scale``
    before it enters the network, and multiplied back afterwards. The heading scale is
    the bfloat16 rounding of ``pi / 2``, which is the value training saw.
    """

    model_type = "qwen_drive"
    sub_configs = {"vlm_config": AutoConfig, "expert_config": PlanningExpertConfig}

    def __init__(
        self,
        vlm_config: dict[str, Any] | PretrainedConfig | None = None,
        expert_config: dict[str, Any] | PlanningExpertConfig | None = None,
        # --- trajectory geometry ---
        num_future_points: int = 50,
        num_history_points: int = 16,
        trajectory_point_dim: int = 3,
        trajectory_hz: float = 10.0,
        trajectory_scale: tuple[float, float, float] = (165.0, 25.0, 1.5703125),
        # --- flow-matching sampler ---
        num_inference_steps: int = 10,
        noise_init_std: float = 1.0,
        noise_seed: int = 42,
        min_one_minus_t: float = 0.1,
        # --- image preprocessing ---
        history_image_pixels: int = 174080,
        current_image_pixels: int = 921600,
        image_patch_size: int = 16,
        image_temporal_patch_size: int = 2,
        image_spatial_merge_size: int = 2,
        # --- text generation for the reasoning stage ---
        max_reasoning_tokens: int = 256,
        min_reasoning_tokens: int = 10,
        **kwargs: Any,
    ) -> None:
        if isinstance(vlm_config, dict):
            vlm_config = AutoConfig.for_model(**vlm_config)
        if isinstance(expert_config, dict):
            expert_config = PlanningExpertConfig(**expert_config)
        self.vlm_config = vlm_config
        self.expert_config = expert_config or PlanningExpertConfig()

        self.num_future_points = num_future_points
        self.num_history_points = num_history_points
        self.trajectory_point_dim = trajectory_point_dim
        self.trajectory_hz = trajectory_hz
        self.trajectory_scale = list(trajectory_scale)

        self.num_inference_steps = num_inference_steps
        self.noise_init_std = noise_init_std
        self.noise_seed = noise_seed
        self.min_one_minus_t = min_one_minus_t

        self.history_image_pixels = history_image_pixels
        self.current_image_pixels = current_image_pixels
        self.image_patch_size = image_patch_size
        self.image_temporal_patch_size = image_temporal_patch_size
        self.image_spatial_merge_size = image_spatial_merge_size

        self.max_reasoning_tokens = max_reasoning_tokens
        self.min_reasoning_tokens = min_reasoning_tokens
        super().__init__(**kwargs)

    @property
    def full_attention_layers(self) -> list[int]:
        """Indices of the VLM layers whose keys/values condition the expert."""
        layer_types = self.vlm_config.text_config.layer_types
        return [i for i, kind in enumerate(layer_types) if kind == "full_attention"]

    @property
    def history_query_points(self) -> int:
        """History poses seen by the expert: re-referenced to the oldest pose, which
        becomes the origin and is dropped."""
        return self.num_history_points - 1
