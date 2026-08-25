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

"""Open-loop scoring for the NVIDIA PhysicalAI splits.

Self-contained: the predictions file already carries the ground truth, so nothing beyond
numpy is needed. ``ADE`` averages over the sampled trajectories while ``minADE`` takes
the best one, which needs the ground truth to pick and is therefore an oracle bound.

    python scripts/eval_physical_ai.py --predictions outputs/physical_ai/predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qwen_drive.metrics import open_loop_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="directory for the reports")
    parser.add_argument("--num-poses", type=int, default=50, help="poses to score")
    args = parser.parse_args()

    per_token: list[dict] = []
    skipped = 0
    with open(args.predictions) as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # truncated final line from an interrupted run
            record = json.loads(line)
            if record.get("future_trajectory") is None:
                skipped += 1
                continue
            candidates = np.asarray(record["trajectories"])[:, : args.num_poses]
            ground_truth = np.asarray(record["future_trajectory"])[: args.num_poses]
            if not (np.isfinite(candidates).all() and np.isfinite(ground_truth).all()):
                skipped += 1
                continue
            row = {"token": record["token"], "num_samples": candidates.shape[0]}
            row.update(open_loop_metrics(candidates, ground_truth))
            per_token.append(row)

    if not per_token:
        raise SystemExit("no scorable predictions found")

    metric_names = [key for key in per_token[0] if key not in ("token", "num_samples")]
    summary = {name: float(np.mean([row[name] for row in per_token])) for name in metric_names}
    summary["num_scored"] = len(per_token)
    summary["num_skipped"] = skipped
    summary["num_samples"] = per_token[0]["num_samples"]

    width = max(len(name) for name in metric_names)
    for name in metric_names:
        print(f"  {name:<{width}}  {summary[name]:.4f}")
    print(f"  scored {len(per_token)} scenes, skipped {skipped}")

    output = args.output or args.predictions.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "physical_ai_metrics.json").write_text(json.dumps(summary, indent=2))
    with open(output / "physical_ai_per_token.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_token[0]))
        writer.writeheader()
        writer.writerows(per_token)
    print(f"wrote reports to {output}")


if __name__ == "__main__":
    main()
