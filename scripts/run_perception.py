"""Run perception inference on packed demo frames.

The perception head reads features from the Qwen-Drive VLM at the root of the
release directory. The head weights live in its ``perception`` subfolder.

Example:
    python scripts/run_perception.py \
        --vlm /path/to/Qwen-Drive-1.0-4B \
        --model /path/to/Qwen-Drive-1.0-4B/perception \
        --frames data/demo/perception \
        --output outputs/perception_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm", type=Path, required=True, help="Qwen-Drive-1.0-4B directory")
    parser.add_argument("--model", type=Path, required=True, help="Qwen-Drive-1.0-4B/perception directory")
    parser.add_argument("--frames", type=Path, default=Path("data/demo/perception"))
    parser.add_argument("--output", type=Path, default=Path("outputs/perception_demo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=["sdpa", "flash_attention_2"])
    args = parser.parse_args()

    holder = QwenDriveForPlanning.from_pretrained(
        args.vlm, dtype=torch.bfloat16, attn_implementation=args.attn_implementation
    )
    vlm = holder.vlm
    del holder.planning_expert

    model = QwenDrivePerception.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    from transformers import AutoTokenizer

    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(args.vlm))
    model.attach(vlm.to(args.device), processor)

    args.output.mkdir(parents=True, exist_ok=True)
    for frame_dir in sorted(p for p in args.frames.iterdir() if p.is_dir()):
        frame = PerceptionFrame(frame_dir)
        inputs, img_metas = processor(frame, device=args.device)
        result = model.infer(inputs, img_metas)

        np.savez(args.output / f"{frame.token}.npz", **result)
        print(f"{frame.token} ({frame.dataset_type}): {len(result['boxes'])} boxes written")


if __name__ == "__main__":
    main()
