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

"""The Qwen-Drive planning expert: a joint-attention diffusion transformer.

The expert holds one token per future waypoint. Every layer attends over the
concatenation of the VLM's cached keys/values and the waypoint tokens' own
keys/values, so the waypoints read the driving scene and each other in a single
attention. Flow matching with a clean-endpoint (x) parameterization turns Gaussian
noise into a trajectory in ten Euler steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_qwen_drive import PlanningExpertConfig

__all__ = ["PlanningExpert"]


def wrap_heading(trajectory: torch.Tensor) -> torch.Tensor:
    """Wrap the heading channel of a ``(..., 3)`` trajectory into ``[-pi, pi)``."""
    heading = torch.remainder(trajectory[..., 2:3] + math.pi, 2 * math.pi) - math.pi
    return torch.cat([trajectory[..., :2], heading], dim=-1)


def _one_hot(index: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot encode, mapping out-of-range indices to an all-zero row."""
    valid = (index >= 0) & (index < num_classes)
    onehot = F.one_hot(index.clamp(0, num_classes - 1).long(), num_classes).to(dtype)
    return onehot * valid.to(dtype).unsqueeze(-1)


def _mlp(in_features: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(), nn.Linear(hidden, hidden))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float) -> None:
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        decay = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -decay)
        angles = self.scale * t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FourierFeatureEncoder(nn.Module):
    """Per-channel Fourier features of a waypoint, then an MLP.

    The frequency table is rebuilt in the module's compute dtype on every call: training
    stored it in bfloat16, so its rounding is part of the features the weights expect.
    """

    def __init__(self, point_dim: int, hidden: int, num_features: int, max_frequency: float) -> None:
        super().__init__()
        self.num_features = num_features
        self.max_frequency = max_frequency
        self.net = _mlp(point_dim * num_features * 2, hidden)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        weight = self.net[0].weight
        freqs = torch.logspace(
            0,
            math.log10(self.max_frequency),
            steps=self.num_features,
            device=weight.device,
            dtype=weight.dtype,
        )
        angles = waypoints.float().unsqueeze(-1) * freqs * (2 * math.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.net(features.to(waypoints.dtype))


class WaypointRotaryEmbedding(nn.Module):
    """Interleaved multi-section rotary embedding, matching the VLM's.

    Waypoint tokens are positioned immediately after the VLM prefix, so their
    rotary phases continue the language model's. All three mRoPE sections share the
    same anchor because the last prefix token is always a text token.

    The phases are computed in the module's compute dtype rather than float32. Training
    held the inverse-frequency table in bfloat16 and cast positions to it, which rounds
    positions above 256 to a coarser grid. The weights were fitted against those exact
    phases, so the rounding is reproduced here.
    """

    def __init__(self, config: PlanningExpertConfig) -> None:
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        if dim % 2 or sum(config.mrope_section) != dim // 2:
            raise ValueError(
                f"mrope_section {config.mrope_section} must sum to {dim // 2} frequency pairs"
            )
        self.rotary_dim = dim
        self.rope_theta = config.rope_theta
        self.sections = list(config.mrope_section)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``position_ids`` is ``[3, B, L]``, returns cos/sin of shape ``[B, L, 1, dim]``."""
        device = position_ids.device
        exponents = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
        inv_freq = (1.0 / (self.rope_theta ** (exponents / self.rotary_dim))).to(dtype)
        angles = position_ids.to(dtype).unsqueeze(-1) * inv_freq  # [3, B, L, pairs]
        merged = angles[0].clone()
        for offset, length in enumerate(self.sections[1:], start=1):
            merged[..., offset : length * 3 : 3] = angles[offset][..., offset : length * 3 : 3]
        emb = torch.cat([merged, merged], dim=-1).unsqueeze(2)  # [B, L, 1, dim]
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``cos.shape[-1]`` channels of ``[B, L, heads, head_dim]``."""
    rotary_dim = cos.shape[-1]
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat([rotated, passthrough], dim=-1)


class PlanningExpertLayer(nn.Module):
    """One diffusion-transformer layer with joint VLM/waypoint attention."""

    def __init__(self, config: PlanningExpertConfig, attn_implementation: str) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.heads_per_group = self.num_heads // self.num_kv_heads
        self.attn_implementation = attn_implementation

        # Fused projection in the VLM's layout: per key/value group, the query
        # heads and their output gates come first, then one key and one value head.
        qkv_out = self.num_kv_heads * (self.heads_per_group * 2 + 2) * self.head_dim
        self.input_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.qkv_proj = nn.Linear(hidden, qkv_out, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        self.post_attention_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.gate_up_proj = nn.Linear(hidden, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, hidden, bias=False)

        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.adaln_modulation[1].weight)
        nn.init.zeros_(self.adaln_modulation[1].bias)

    def _split_qkv(self, fused: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, length, _ = fused.shape
        head_dim, groups, per_group = self.head_dim, self.num_kv_heads, self.heads_per_group
        fused = fused.view(batch, length, groups, (per_group * 2 + 2) * head_dim)
        gated_query, key, value = torch.split(
            fused, [per_group * 2 * head_dim, head_dim, head_dim], dim=3
        )
        query, gate = torch.chunk(gated_query, 2, dim=-1)
        return (
            query.reshape(batch, length, self.num_heads, head_dim),
            gate.reshape(batch, length, self.num_heads, head_dim),
            key.reshape(batch, length, groups, head_dim),
            value.reshape(batch, length, groups, head_dim),
        )

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        if self.attn_implementation == "flash_attention_2":
            from flash_attn import flash_attn_func

            return flash_attn_func(query, key, value, causal=False)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return out.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        scene_key: torch.Tensor,
        scene_value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        modulation = self.adaln_modulation(condition).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            m.unsqueeze(1) for m in modulation
        )

        residual = hidden_states
        x = self.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
        query, gate, key, value = self._split_qkv(self.qkv_proj(x))
        query = _apply_rotary(self.q_norm(query), cos, sin)
        key = _apply_rotary(self.k_norm(key), cos, sin)

        attn = self._attend(
            query,
            torch.cat([scene_key, key], dim=1),
            torch.cat([scene_value, value], dim=1),
        ).reshape(batch, length, -1)
        attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))
        hidden_states = residual + (1 + gate_attn) * self.o_proj(attn)

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states) * (1 + scale_ffn) + shift_ffn
        swiglu_gate, swiglu_up = self.gate_up_proj(x).chunk(2, dim=-1)
        return residual + (1 + gate_ffn) * self.down_proj(F.silu(swiglu_gate) * swiglu_up)


class PlanningExpert(nn.Module):
    """Flow-matching trajectory decoder conditioned on the VLM's attention cache."""

    def __init__(
        self,
        config: PlanningExpertConfig,
        num_future_points: int,
        num_history_points: int,
        trajectory_point_dim: int,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_future_points = num_future_points
        self.num_history_points = num_history_points
        self.trajectory_point_dim = trajectory_point_dim
        hidden = config.hidden_size

        self.trajectory_proj = nn.Linear(trajectory_point_dim, hidden)
        self.fourier_encoder = FourierFeatureEncoder(
            trajectory_point_dim, hidden, config.fourier_num_features, config.fourier_max_frequency
        )
        self.waypoint_embed = nn.Embedding(num_future_points, hidden)
        self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim, config.time_embed_scale)
        self.time_mlp = _mlp(config.time_embed_dim, hidden)
        self.nav_mlp = _mlp(config.nav_command_classes, hidden)
        self.ego_mlp = _mlp(config.ego_status_dim, hidden)

        history_dim = (num_history_points - 1) * trajectory_point_dim + config.nav_command_classes
        dynamics_dim = num_history_points * config.history_dynamics_dim
        self.history_encoder = _mlp(history_dim, hidden)
        self.history_velocity_encoder = _mlp(dynamics_dim, hidden)
        self.history_acceleration_encoder = _mlp(dynamics_dim, hidden)
        # Waypoint queries fuse: noisy waypoint, its Fourier features, flow time,
        # history poses, waypoint index, history velocity and history acceleration.
        self.query_fusion = _mlp(hidden * 7, hidden)

        self.rotary_emb = WaypointRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            PlanningExpertLayer(config, attn_implementation)
            for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = RMSNorm(hidden, config.rms_norm_eps)
        self.out_proj = nn.Linear(hidden, trajectory_point_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.out_proj.weight.dtype

    def _waypoint_positions(self, anchor: torch.Tensor, length: int) -> torch.Tensor:
        """Positions ``anchor + 1 .. anchor + length`` for each mRoPE section."""
        steps = torch.arange(1, length + 1, device=anchor.device, dtype=anchor.dtype)
        return anchor.unsqueeze(-1) + steps

    def encode_history(
        self,
        history: torch.Tensor,
        nav_command: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the normalized history poses and the raw history dynamics."""
        dtype = self.dtype
        batch = history.shape[0]
        nav_onehot = _one_hot(nav_command, self.config.nav_command_classes, dtype)
        pose_query = self.history_encoder(
            torch.cat([history.to(dtype).reshape(batch, -1), nav_onehot], dim=-1)
        )
        velocity_query = self.history_velocity_encoder(velocity.to(dtype).reshape(batch, -1))
        acceleration_query = self.history_acceleration_encoder(
            acceleration.to(dtype).reshape(batch, -1)
        )
        return pose_query, velocity_query, acceleration_query

    def predict_endpoint(
        self,
        waypoints: torch.Tensor,
        flow_time: torch.Tensor,
        history_queries: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
        position_anchor: torch.Tensor,
        nav_command: torch.Tensor,
        ego_status: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the clean trajectory from the current noisy one, in float32."""
        dtype = self.dtype
        waypoints = waypoints.to(dtype)
        batch, length, _ = waypoints.shape
        pose_query, velocity_query, acceleration_query = history_queries

        time_condition = self.time_mlp(self.time_embed(flow_time).to(dtype))
        waypoint_index = torch.arange(length, device=waypoints.device)
        broadcast = [
            self.trajectory_proj(waypoints),
            self.fourier_encoder(waypoints),
            time_condition.unsqueeze(1).expand(-1, length, -1),
            pose_query.unsqueeze(1).expand(-1, length, -1),
            self.waypoint_embed(waypoint_index).unsqueeze(0).expand(batch, -1, -1).to(dtype),
            velocity_query.unsqueeze(1).expand(-1, length, -1),
            acceleration_query.unsqueeze(1).expand(-1, length, -1),
        ]
        hidden_states = self.query_fusion(torch.cat(broadcast, dim=-1))

        nav_onehot = _one_hot(nav_command, self.config.nav_command_classes, dtype)
        condition = (
            time_condition + self.nav_mlp(nav_onehot) + self.ego_mlp(ego_status.to(dtype))
        )

        positions = self._waypoint_positions(position_anchor, length)
        cos, sin = self.rotary_emb(positions, dtype)
        layers_per_kv = self.config.layers_per_kv
        for index, layer in enumerate(self.layers):
            scene_key, scene_value = scene_cache[index // layers_per_kv]
            hidden_states = layer(
                hidden_states,
                scene_key.expand(batch, -1, -1, -1),
                scene_value.expand(batch, -1, -1, -1),
                cos,
                sin,
                condition,
            )
        return self.out_proj(self.final_layernorm(hidden_states)).float()

    @torch.no_grad()
    def sample(
        self,
        scene_cache: list[tuple[torch.Tensor, torch.Tensor]],
        position_anchor: torch.Tensor,
        history: torch.Tensor,
        history_velocity: torch.Tensor,
        history_acceleration: torch.Tensor,
        nav_command: torch.Tensor,
        ego_status: torch.Tensor,
        noise: torch.Tensor,
        num_steps: int,
        min_one_minus_t: float,
    ) -> torch.Tensor:
        """Integrate the flow from ``noise`` to a normalized trajectory.

        The network predicts the clean endpoint, which is turned into a velocity by
        dividing by the remaining time. That divisor is floored at ``min_one_minus_t``
        so the last step lands on the prediction without amplifying its error.
        """
        history_queries = self.encode_history(
            history, nav_command, history_velocity, history_acceleration
        )
        waypoints = noise.float()
        step = 1.0 / num_steps
        for index in range(num_steps):
            flow_time = torch.full(
                (waypoints.shape[0],), index * step, device=waypoints.device, dtype=torch.float32
            )
            endpoint = self.predict_endpoint(
                waypoints,
                flow_time,
                history_queries,
                scene_cache,
                position_anchor,
                nav_command,
                ego_status,
            )
            remaining = max(1.0 - index * step, min_one_minus_t)
            waypoints = waypoints + (endpoint - waypoints) / remaining * step
        return waypoints
