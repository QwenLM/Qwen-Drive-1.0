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

"""Predict trajectories for every scene in a benchmark file.

The output is one JSON object per scene holding the sampled trajectories and the ground
truth copied from the input, which is all the ``eval_*.py`` scripts need.

Image reading and preprocessing run in DataLoader workers so they overlap with the GPU
work of earlier scenes; raise ``--num-workers`` to hide that cost.

    python scripts/run_planning.py \
        --model Qwen-Drive-1.0-4B \
        --scenes scenes/navsim_navtest.jsonl \
        --image-root /data/navsim/sensor_blobs \
        --output outputs/navsim/predictions.jsonl \
        --mode reasoning_planning --num-samples 6
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from qwen_drive import InferenceMode, QwenDriveForPlanning, QwenDriveProcessor
from qwen_drive.benchmarks import read_scene_file
from qwen_drive.images import ImageArchive

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class SceneInputs(Dataset):
    """Reads a scene and preprocesses it into model inputs, entirely on CPU.

    Returned in a worker process so image decoding and patchifying overlap with the GPU.
    """

    def __init__(self, samples, processor: QwenDriveProcessor, with_reasoning: bool) -> None:
        self.samples = samples
        self.processor = processor
        self.with_reasoning = with_reasoning

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        inputs = self.processor(
            sample.scene, with_reasoning=self.with_reasoning, device="cpu"
        )
        return {
            "inputs": inputs,
            "token": sample.token,
            "scene_token": sample.scene_token,
            "future_trajectory": sample.future_trajectory,
            "future_valid": sample.future_valid,
            "preference_trajectories": sample.preference_trajectories,
            "preference_scores": sample.preference_scores,
            "initial_speed": sample.initial_speed,
        }


def load_done_tokens(path: Path) -> set[str]:
    """Tokens already written to a predictions file, tolerating a truncated last line.

    A preempted run can leave a half-written final line; that line is dropped and the file
    is rewritten with only the complete records, so appending resumes cleanly.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    complete: list[str] = []
    with open(path) as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # truncated final line from an interrupted write
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break
            done.add(str(record.get("token", "")))
            complete.append(line)
    with open(path, "w") as handle:
        handle.writelines(complete)
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen-Drive-1.0-4B directory")
    parser.add_argument("--planner", default=None, help="Planning Expert weights directory")
    parser.add_argument("--scenes", required=True, help="benchmark scene file (jsonl)")
    parser.add_argument("--output", required=True, help="destination predictions file (jsonl)")
    parser.add_argument(
        "--mode",
        default=InferenceMode.DIRECT_PLANNING.value,
        choices=[InferenceMode.DIRECT_PLANNING.value, InferenceMode.REASONING_PLANNING.value],
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="trajectories drawn per scene (best-of-N)"
    )
    parser.add_argument("--num-steps", type=int, default=None, help="flow integration steps")
    parser.add_argument("--seed", type=int, default=None, help="base noise seed")
    parser.add_argument("--image-root", default=None, help="directory the frame paths are relative to")
    parser.add_argument(
        "--image-archive", default=None, help="Parquet shards from scripts/pack_images.py"
    )
    parser.add_argument(
        "--image-resolver",
        default=None,
        help="'module:factory' returning a callable that maps a frame path to an image",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers that preprocess ahead of the GPU"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="overwrite the output instead of skipping scenes already written to it",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="split scenes into this many shards (one per GPU)"
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="which shard this process handles (0-based)"
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after this many scenes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["sdpa", "flash_attention_2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = ImageArchive.open(args.image_archive) if args.image_archive else None
    resolver = None
    if args.image_resolver:
        module_name, _, factory = args.image_resolver.partition(":")
        resolver = getattr(importlib.import_module(module_name), factory)()

    model = QwenDriveForPlanning.from_pretrained(
        args.model,
        planner=args.planner,
        dtype=DTYPES[args.dtype],
        attn_implementation=args.attn_implementation,
    )
    model = model.to(args.device).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume:
        output.unlink(missing_ok=True)
        done_tokens: set[str] = set()
    else:
        done_tokens = load_done_tokens(output)

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")

    all_samples = list(
        read_scene_file(
            args.scenes,
            image_root=args.image_root,
            image_archive=archive,
            image_resolver=resolver,
            num_history_points=model.config.num_history_points,
            limit=args.limit,
        )
    )
    # Round-robin shard so every GPU gets a balanced, disjoint subset.
    shard_samples = all_samples[args.shard_index :: args.num_shards]
    samples = [sample for sample in shard_samples if sample.token not in done_tokens]
    if args.num_shards > 1:
        print(f"shard {args.shard_index}/{args.num_shards}: {len(shard_samples)} scenes assigned")
    if done_tokens:
        print(f"resuming: {len(done_tokens)} scenes already done, {len(samples)} remaining")
    if not samples:
        print(f"nothing to do; {len(done_tokens)} predictions already in {output}")
        return

    dataset = SceneInputs(
        samples, model.processor, with_reasoning=args.mode == InferenceMode.REASONING_PLANNING.value
    )
    # batch_size=1: the model processes one scene at a time, but workers prepare the next
    # scenes while the GPU runs this one. collate returns the single item unchanged.
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    started = time.perf_counter()
    count = 0
    total = len(dataset)
    log_every = max(1, min(50, total // 20))
    with open(output, "a") as handle:
        for item in loader:
            result = model.plan_from_inputs(
                item["inputs"],
                mode=args.mode,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                seed=args.seed,
            )
            future = item["future_trajectory"]
            valid = item["future_valid"]
            handle.write(
                json.dumps(
                    {
                        "token": item["token"],
                        "scene_token": item["scene_token"],
                        "trajectories": result.trajectories.round(5).tolist(),
                        "reasoning": result.reasoning,
                        "future_trajectory": None if future is None else future.round(5).tolist(),
                        "future_valid": None if valid is None else valid.tolist(),
                        "preference_trajectories": [
                            t.round(5).tolist() for t in item["preference_trajectories"]
                        ],
                        "preference_scores": item["preference_scores"],
                        "initial_speed": item["initial_speed"],
                    }
                )
                + "\n"
            )
            # Flush each record so a preemption loses at most the scene being written. The
            # next run's resume drops any truncated tail and continues from here.
            handle.flush()
            count += 1
            if count % log_every == 0 or count == total:
                elapsed = time.perf_counter() - started
                rate = elapsed / count
                eta = rate * (total - count)
                print(
                    f"[{args.mode}] {count}/{total} scenes "
                    f"({rate:.2f} s/scene, elapsed {int(elapsed // 60)}m{int(elapsed % 60):02d}s, "
                    f"ETA {int(eta // 60)}m{int(eta % 60):02d}s)",
                    flush=True,
                )
            os.fsync(handle.fileno())

    elapsed = time.perf_counter() - started
    total_done = count + len(done_tokens)
    print(
        f"wrote {count} predictions in {elapsed:.1f}s ({elapsed / max(count, 1):.2f}s each); "
        f"{total_done} total in {output}"
    )


if __name__ == "__main__":
    main()
