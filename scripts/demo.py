#!/usr/bin/env python
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

"""Run all three inference modes on one scene.

    python scripts/demo.py \
        --model Qwen-Drive-1.0-4B \
        --planner Qwen-Drive-1.0-4B/planner-rl \
        --scenes scenes/waymo_e2e_val.jsonl \
        --image-root /data/waymo_e2e/extracted/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qwen_drive import InferenceMode, QwenDriveForPlanning
from qwen_drive.benchmarks import read_scene_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--planner", default=None, help="Planning Expert weights directory")
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--image-archive", default=None)
    parser.add_argument("--index", type=int, default=0, help="which scene in the file")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--question", default="Describe the traffic scene and the safest action.")
    parser.add_argument("--plot", type=Path, default=None, help="write a trajectory plot here")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    archive = None
    if args.image_archive:
        from qwen_drive.images import ImageArchive

        archive = ImageArchive.open(args.image_archive)

    model = QwenDriveForPlanning.from_pretrained(
        args.model,
        planner=args.planner,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(args.device).eval()

    samples = read_scene_file(
        args.scenes,
        image_root=args.image_root,
        image_archive=archive,
        num_history_points=model.config.num_history_points,
        limit=args.index + 1,
    )
    sample = list(samples)[args.index]
    scene = sample.scene
    print(f"scene {sample.token}  |  {scene.num_camera_frames} frames x {len(scene.views)} views")
    print(f"navigation command: {scene.nav_command}\n")

    answer = model.run(InferenceMode.VQA, scene=scene, question=args.question)
    print(f"[vqa] {answer.text}\n")

    direct = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=args.num_samples)
    print(f"[direct planning] {args.num_samples} trajectories, shape {direct.trajectories.shape}")
    print(f"  endpoint of sample 0: {np.round(direct.trajectory[-1], 3).tolist()}")

    reasoned = model.run(
        InferenceMode.REASONING_PLANNING, scene=scene, num_samples=args.num_samples
    )
    print(f"\n[reasoning planning] reasoning: {reasoned.reasoning}")
    print(f"  endpoint of sample 0: {np.round(reasoned.trajectory[-1], 3).tolist()}")

    if sample.future_trajectory is not None:
        for name, result in (("direct", direct), ("reasoning", reasoned)):
            error = np.linalg.norm(
                result.trajectory[:, :2] - sample.future_trajectory[: len(result.trajectory), :2],
                axis=-1,
            )
            print(f"  {name}: ADE {error.mean():.3f} m, FDE {error[-1]:.3f} m")

    if args.plot is not None:
        from qwen_drive.visualize import plot_scene_summary

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_scene_summary(
            scene,
            reasoned.trajectories,
            history=scene.history,
            ground_truth=sample.future_trajectory,
            reasoning=reasoned.reasoning,
            title=f"{sample.token} (reasoning planning)",
            output=args.plot,
        )
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
