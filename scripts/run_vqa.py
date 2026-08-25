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

"""Ask the model a question about one or more images.

This is the plain vision-language interface: the planning expert is not involved, so the
model answers exactly as its Qwen3.5 VLM would.

    python scripts/run_vqa.py --model Qwen-Drive-1.0-4B \
        --image front.jpg --image front_left.jpg \
        --question "Is it safe to change into the left lane?"
"""

from __future__ import annotations

import argparse

import torch

from qwen_drive import QwenDriveForPlanning

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", action="append", required=True, help="image path; repeatable")
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.001)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    args = parser.parse_args()

    model = QwenDriveForPlanning.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa"
    )
    model = model.to(args.device).eval()

    result = model.generate_text(
        args.image,
        args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
    )
    print(result.text)


if __name__ == "__main__":
    main()
