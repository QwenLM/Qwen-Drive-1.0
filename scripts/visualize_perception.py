"""Render the packed demo frames (with predictions) as summary images.

Example:
    python scripts/visualize_perception.py \
        --frames data/demo/perception \
        --predictions outputs/perception_demo \
        --output outputs/perception_demo/vis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from qwen_drive_perception.dataset import PerceptionFrame
    from qwen_drive_perception.visualize import render_frame
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen_drive_perception.dataset import PerceptionFrame
    from qwen_drive_perception.visualize import render_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=Path("data/demo/perception"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/perception_demo"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    args = parser.parse_args()
    output = args.output or (args.predictions / "vis")
    output.mkdir(parents=True, exist_ok=True)

    for frame_dir in sorted(p for p in args.frames.iterdir() if p.is_dir()):
        pred_path = args.predictions / f"{frame_dir.name}.npz"
        if not pred_path.exists():
            print(f"skipping {frame_dir.name}: no predictions at {pred_path}")
            continue
        frame = PerceptionFrame(frame_dir)
        predictions = np.load(pred_path)
        result = {k: predictions[k] for k in predictions.files}
        image = render_frame(frame, result, score_threshold=args.score_threshold)
        cv2.imwrite(str(output / f"{frame_dir.name}.png"), image)
        print(f"rendered {output / (frame_dir.name + '.png')}")


if __name__ == "__main__":
    main()
