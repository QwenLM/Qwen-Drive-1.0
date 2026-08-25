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

"""Scoring for the Waymo Open Dataset end-to-end driving split.

The 5 s / 10 Hz prediction is resampled onto the benchmark's 20-pose 4 Hz grid before
displacement errors are computed. Those need nothing but numpy. The rater feedback
score additionally needs the official implementation from the Waymo Open Dataset; pass
``--rater-feedback`` to compute it.

With several sampled trajectories per scene, the candidate closest to the highest-rated
preference trajectory is kept. That selection uses the labels, so it is an oracle bound.

    python scripts/eval_waymo.py --predictions outputs/waymo/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qwen_drive.metrics import rater_feedback_score, waymo_displacement_metrics
from qwen_drive.trajectory import resample_to_4hz


def _best_preference(record: dict) -> np.ndarray | None:
    """The highest-rated preference trajectory, padded to 20 poses."""
    scores = record.get("preference_scores") or []
    if not scores:
        return None
    best = np.asarray(record["preference_trajectories"][int(np.argmax(scores))], dtype=np.float64)
    if best.shape[0] >= 20:
        return best[:20]
    return np.concatenate([best, np.repeat(best[-1:], 20 - best.shape[0], axis=0)])


def _select_candidate(record: dict, official_grid: bool) -> np.ndarray:
    """Pick the candidate closest to the best-rated preference trajectory."""
    candidates = np.asarray(record["trajectories"], dtype=np.float64)
    reference = _best_preference(record)
    if candidates.shape[0] == 1 or reference is None:
        return candidates[0]
    errors = [
        np.linalg.norm(
            resample_to_4hz(candidate, 20, official_grid=official_grid)[:, :2] - reference[:, :2],
            axis=-1,
        ).mean()
        for candidate in candidates
    ]
    return candidates[int(np.argmin(errors))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--official-4hz-grid",
        action="store_true",
        help="resample onto the benchmark's absolute timestamps instead of a normalized index ramp",
    )
    parser.add_argument("--rater-feedback", action="store_true", help="also compute RFS")
    args = parser.parse_args()

    predictions, ground_truth = [], []
    preference_trajectories, preference_scores, initial_speeds = [], [], []
    skipped = 0
    with open(args.predictions) as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # truncated final line from an interrupted run
            record = json.loads(line)
            if record.get("future_trajectory") is None:
                skipped += 1
                continue
            predictions.append(_select_candidate(record, args.official_4hz_grid))
            ground_truth.append(np.asarray(record["future_trajectory"], dtype=np.float64))
            if record.get("preference_scores"):
                preference_trajectories.append(
                    [np.asarray(t, dtype=np.float64)[:20, :2] for t in record["preference_trajectories"]]
                )
                preference_scores.append(np.asarray(record["preference_scores"], dtype=np.float64))
                initial_speeds.append(record["initial_speed"])

    if not predictions:
        raise SystemExit("no scorable predictions found")

    metrics = waymo_displacement_metrics(
        np.stack(predictions), np.stack(ground_truth), official_4hz_grid=args.official_4hz_grid
    )
    metrics["num_scored"] = len(predictions)
    metrics["num_skipped"] = skipped

    if args.rater_feedback:
        if len(preference_trajectories) != len(predictions):
            raise SystemExit(
                f"only {len(preference_trajectories)} of {len(predictions)} scenes carry "
                "preference trajectories; RFS needs all of them"
            )
        resampled = np.stack(
            [resample_to_4hz(p, 20, official_grid=args.official_4hz_grid) for p in predictions]
        )
        result = rater_feedback_score(
            resampled, preference_trajectories, preference_scores, np.asarray(initial_speeds)
        )
        metrics["RFS"] = float(result["rater_feedback_score"].mean())
        trusted = result.get("is_fully_within_trust_region")
        if trusted is not None:
            metrics["RFS_within_trust_pct"] = float(trusted.any(axis=-1).mean() * 100)

    width = max(len(name) for name in metrics)
    for name, value in metrics.items():
        print(f"  {name:<{width}}  {value:.4f}" if isinstance(value, float) else f"  {name:<{width}}  {value}")

    output = args.output or args.predictions.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "waymo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote reports to {output}")


if __name__ == "__main__":
    main()
