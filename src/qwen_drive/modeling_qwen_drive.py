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

"""Qwen-Drive-1.0: a Qwen3.5 VLM driving a flow-matching planning expert."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging as transformers_logging

from .configuration_qwen_drive import QwenDriveConfig
from .planning_expert import PlanningExpert
from .scene import DrivingScene, QwenDriveProcessor
from .trajectory import denormalize_trajectory, normalize_history

__all__ = ["InferenceMode", "QwenDriveOutput", "QwenDriveForPlanning", "VQA_DECODE_DEFAULTS"]

_THINK_CLOSE = "</think>"

# Canonical decoding parameters for the VQA mode. With top_k=1 and a near-zero
# temperature this is effectively greedy, but it keeps the sampling path deterministic
# and matches the released evaluation protocol.
VQA_DECODE_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.01,
    "top_k": 1,
    "top_p": 0.001,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "max_new_tokens": 32768,
    "seed": 3407,
}


def _strip_thinking(text: str) -> str:
    """Drop a leading thinking block, keeping the answer that follows it."""
    return text.split(_THINK_CLOSE)[-1].strip()


class InferenceMode(str, Enum):
    """The three ways the model can be queried.

    ``VQA``
        Text only, through the unmodified vision-language interface. The planning
        expert is not used, so the model behaves exactly like its Qwen3.5 VLM.
    ``DIRECT_PLANNING``
        The expert plans straight from the scene. The user turn carries no reasoning
        request and the assistant turn is left empty, matching how the model was
        trained to plan without text.
    ``REASONING_PLANNING``
        The VLM first writes a one-sentence rationale, then the expert plans from
        the attention cache that generation produced, so the trajectory is conditioned
        on the model's own reasoning.
    """

    VQA = "vqa"
    DIRECT_PLANNING = "direct_planning"
    REASONING_PLANNING = "reasoning_planning"


@dataclass
class QwenDriveOutput:
    """Result of one query.

    ``trajectories`` is ``[num_samples, num_future_points, 3]`` of ``(x, y, heading)`` in
    metres and radians, in the current ego frame, at the model's 10 Hz output rate.
    """

    trajectories: np.ndarray | None = None
    reasoning: str | None = None
    text: str | None = None

    @property
    def trajectory(self) -> np.ndarray:
        """The first sampled trajectory."""
        return self.trajectories[0]


class QwenDriveForPlanning(PreTrainedModel):
    config_class = QwenDriveConfig
    base_model_prefix = "vlm"
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: QwenDriveConfig) -> None:
        super().__init__(config)
        self.vlm = AutoModelForImageTextToText.from_config(config.vlm_config)
        self.planning_expert = PlanningExpert(
            config.expert_config,
            num_future_points=config.num_future_points,
            num_history_points=config.num_history_points,
            trajectory_point_dim=config.trajectory_point_dim,
            attn_implementation=config._attn_implementation,
        )
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, planner: str | Path | None = None, **kwargs):
        """Load the VLM, optionally attaching a separately released planner head.

        The VLM is shared by every task, so it ships once and each task head
        ships on its own. ``planner`` points at a Planning Expert directory
        (SFT or RL); without it only the VQA mode is available.
        """
        # The expert lives in its own directory, so loading the VLM alone would
        # report every expert tensor as missing.
        verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            model = super().from_pretrained(*args, **kwargs)
        finally:
            transformers_logging.set_verbosity(verbosity)
        if planner is not None:
            model.load_planner(planner)
        return model

    def load_planner(self, path: str | Path) -> None:
        """Load Planning Expert weights from a released head directory."""
        from safetensors.torch import load_file

        weights = load_file(str(Path(path) / "model.safetensors"))
        prefix = "planning_expert."
        weights = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in weights.items()}
        target = next(self.planning_expert.parameters()).dtype
        self.planning_expert.load_state_dict(
            {k: v.to(target) for k, v in weights.items()}, strict=True
        )

    def trajectory_scale(self, device: torch.device | None = None) -> torch.Tensor:
        """Per-channel normalization constants, in metres and radians."""
        return torch.tensor(self.config.trajectory_scale, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------ helpers

    def _modality_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-token modality map the VLM needs for multimodal rotary positions."""
        return (input_ids == self.config.vlm_config.image_token_id).long()

    def _rope_positions(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Multimodal rotary positions ``[3, B, S]`` for a prompt."""
        positions, _ = self.vlm.model.get_rope_index(
            input_ids,
            mm_token_type_ids=self._modality_ids(input_ids),
            image_grid_thw=image_grid_thw,
        )
        return positions

    def _scene_cache(self, past_key_values) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Post-rotary keys/values of the VLM's grouped-query attention layers.

        Returned as ``[batch, sequence, kv_heads, head_dim]`` per layer, in layer order,
        one entry per cache the expert consumes.
        """
        cache = []
        for index in self.config.full_attention_layers:
            layer = past_key_values.layers[index]
            cache.append((layer.keys.transpose(1, 2), layer.values.transpose(1, 2)))
        return cache

    def _initial_noise(
        self, num_samples: int, num_points: int, seed: int, device: torch.device
    ) -> torch.Tensor:
        """One standard-normal draw per sample, each from its own reproducible seed."""
        rows = []
        for offset in range(num_samples):
            generator = torch.Generator(device=device).manual_seed(seed + offset)
            rows.append(
                torch.randn(
                    1,
                    num_points,
                    self.config.trajectory_point_dim,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(rows, dim=0)

    # ------------------------------------------------------------------ stages

    @torch.no_grad()
    def _prefill(self, inputs: dict) -> tuple[list, torch.Tensor]:
        """Read the whole prompt once and keep its attention cache."""
        input_ids = inputs["input_ids"]
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=self._modality_ids(input_ids),
            use_cache=True,
        )
        anchor = self._rope_positions(input_ids, inputs["image_grid_thw"])[:, :, -1]
        return self._scene_cache(outputs.past_key_values), anchor

    @torch.no_grad()
    def _prefill_with_reasoning(
        self, inputs: dict, max_new_tokens: int
    ) -> tuple[list, torch.Tensor, str]:
        """Generate the assistant turn, then extend its cache to the trained turn ending.

        Generation stops as soon as ``<|im_end|>`` is sampled, so the cache is short of
        the tokens that close the turn. Those are appended here, which puts the expert's
        waypoint tokens at the same rotary positions they occupied during training.
        """
        prompt_ids = inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]
        prompt_anchor = self._rope_positions(prompt_ids, inputs["image_grid_thw"])[:, :, -1]
        processor = self.processor
        terminators = [processor.im_end_id, self.config.vlm_config.text_config.eos_token_id]

        generated = self.vlm.generate(
            input_ids=prompt_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=self._modality_ids(prompt_ids),
            max_new_tokens=max_new_tokens,
            min_new_tokens=self.config.min_reasoning_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            use_cache=True,
            return_dict_in_generate=True,
            eos_token_id=terminators,
        )
        cache = generated.past_key_values
        new_ids = generated.sequences[0, prompt_length:].tolist()

        # The turn ends at the first terminator. Anything after it is runaway generation
        # and must not reach the cache, or the turn ending would appear twice.
        content = new_ids
        for position, token in enumerate(new_ids):
            if token in terminators:
                content = new_ids[:position]
                break
        # The cache keeps every generated token, which is what the expert was trained to
        # read. Only the text handed back to the caller drops the thinking block.
        reasoning = _strip_thinking(processor.tokenizer.decode(content, skip_special_tokens=True))

        closed_turn = content + [processor.im_end_id] + processor.newline_ids
        already_cached = cache.get_seq_length() - prompt_length
        pending = closed_turn[already_cached:]
        if pending:
            self.vlm(
                input_ids=torch.tensor([pending], device=prompt_ids.device),
                past_key_values=cache,
                cache_position=torch.arange(
                    prompt_length + already_cached,
                    prompt_length + len(closed_turn),
                    device=prompt_ids.device,
                ),
                use_cache=True,
            )
        anchor = prompt_anchor + len(closed_turn)
        return self._scene_cache(cache), anchor, reasoning

    @torch.no_grad()
    def _plan_from_cache(
        self,
        scene_cache: list,
        anchor: torch.Tensor,
        inputs: dict,
        num_samples: int,
        num_steps: int,
        seed: int,
    ) -> np.ndarray:
        expert = self.planning_expert
        device = anchor.device
        num_points = self.config.num_future_points
        scale = self.trajectory_scale(device)

        history = normalize_history(inputs["history"].float(), scale)

        def tile(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(num_samples, dim=0)

        normalized = expert.sample(
            scene_cache=scene_cache,
            position_anchor=anchor.expand(-1, num_samples),
            history=tile(history),
            history_velocity=tile(inputs["history_velocity"].float()),
            history_acceleration=tile(inputs["history_acceleration"].float()),
            nav_command=tile(inputs["nav_command"]),
            ego_status=tile(inputs["ego_status"].float()),
            noise=self.config.noise_init_std
            * self._initial_noise(num_samples, num_points, seed, device),
            num_steps=num_steps,
            min_one_minus_t=self.config.min_one_minus_t,
        )
        return denormalize_trajectory(normalized, scale).cpu().numpy()

    # ------------------------------------------------------------------ public API

    @property
    def processor(self) -> QwenDriveProcessor:
        """Tokenizer-backed processor, loaded from the model directory on demand."""
        if getattr(self, "_processor", None) is None:
            tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
            self._processor = QwenDriveProcessor(tokenizer, self.config)
        return self._processor

    @processor.setter
    def processor(self, processor: QwenDriveProcessor) -> None:
        self._processor = processor

    @torch.no_grad()
    def generate_trajectory(
        self,
        scene: DrivingScene,
        mode: InferenceMode | str = InferenceMode.DIRECT_PLANNING,
        num_samples: int = 1,
        num_steps: int | None = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> QwenDriveOutput:
        """Plan a trajectory, optionally letting the model reason about the scene first.

        ``num_samples`` draws that many trajectories from independent noise. Sample ``k``
        always uses seed ``seed + k``, so results are reproducible and a candidate is
        identical whether drawn alone or as part of a batch.
        """
        mode = InferenceMode(mode)
        inputs = self.processor(
            scene, with_reasoning=mode is InferenceMode.REASONING_PLANNING, device="cpu"
        )
        return self.plan_from_inputs(
            inputs,
            mode=mode,
            num_samples=num_samples,
            num_steps=num_steps,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def plan_from_inputs(
        self,
        inputs: dict,
        mode: InferenceMode | str = InferenceMode.DIRECT_PLANNING,
        num_samples: int = 1,
        num_steps: int | None = None,
        seed: int | None = None,
        max_new_tokens: int | None = None,
    ) -> QwenDriveOutput:
        """Plan from inputs already produced by the processor.

        The inputs may be built on CPU by a DataLoader worker so their preparation overlaps
        with the GPU work of earlier scenes, so they are moved to the model device here.
        """
        mode = InferenceMode(mode)
        if mode is InferenceMode.VQA:
            raise ValueError("use generate_text for the VQA mode; it produces no trajectory")

        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        reasoning = None
        if mode is InferenceMode.REASONING_PLANNING:
            scene_cache, anchor, reasoning = self._prefill_with_reasoning(
                inputs, max_new_tokens or self.config.max_reasoning_tokens
            )
        else:
            scene_cache, anchor = self._prefill(inputs)

        trajectories = self._plan_from_cache(
            scene_cache,
            anchor,
            inputs,
            num_samples=num_samples,
            num_steps=num_steps or self.config.num_inference_steps,
            seed=self.config.noise_seed if seed is None else seed,
        )
        return QwenDriveOutput(trajectories=trajectories, reasoning=reasoning)

    @torch.no_grad()
    def generate_text(
        self,
        images,
        question: str,
        **generate_kwargs,
    ) -> QwenDriveOutput:
        """Answer a question about a list of images, using only the VLM.

        Decoding defaults to the canonical VQA parameters (see
        :data:`VQA_DECODE_DEFAULTS`); pass any of those keys to override.
        """
        processor = self.processor
        inputs = processor.encode_vqa(images, question, device=self.device)
        params = dict(VQA_DECODE_DEFAULTS)
        params.update(generate_kwargs)
        seed = params.pop("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
        # presence_penalty is part of the released protocol but always 0.0, and
        # transformers' generate does not accept it.
        params.pop("presence_penalty", None)
        generated = self.vlm.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=self._modality_ids(inputs["input_ids"]),
            **params,
        )
        answer = processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return QwenDriveOutput(text=_strip_thinking(answer))

    @torch.no_grad()
    def run(
        self,
        mode: InferenceMode | str,
        scene: DrivingScene | None = None,
        images=None,
        question: str | None = None,
        **kwargs,
    ) -> QwenDriveOutput:
        """Single entry point for all three inference modes."""
        mode = InferenceMode(mode)
        if mode is InferenceMode.VQA:
            if question is None:
                raise ValueError("the VQA mode needs a question")
            frames = images if images is not None else scene.frames_in_order()
            return self.generate_text(frames, question, **kwargs)
        if scene is None:
            raise ValueError(f"mode {mode.value!r} needs a DrivingScene")
        return self.generate_trajectory(scene, mode=mode, **kwargs)
