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

"""Scoring for NAVSIM v1.1 navtest.

Displacement errors are computed from the predictions file alone. The PDM score is a
pseudo-closed-loop simulation and needs the NAVSIM package, the nuPlan maps and a
prebuilt metric cache, so it is opt-in via ``--metric-cache``:

    export NUPLAN_MAPS_ROOT=/data/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
    python scripts/eval_navsim.py \
        --predictions outputs/navsim/predictions.jsonl \
        --metric-cache /data/navsim/metric_cache_navtest

The model predicts 50 poses at 10 Hz while NAVSIM scores 4 s. Two conventions are
reported: ``interp`` feeds the 8 poses at 2 Hz that the simulator interpolates from, and
``direct`` feeds the leading 40 poses at their native 10 Hz. With several samples per
scene the highest-scoring one is kept, which is an oracle bound.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qwen_drive.metrics import navsim_displacement_metrics
from qwen_drive.trajectory import resample_10hz_to_2hz

PDM_FIELDS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)


def load_predictions(path: Path) -> list[dict]:
    records = []
    with open(path) as handle:
        for line in handle:
            if not line.endswith("\n"):
                break  # truncated final line from an interrupted run
            records.append(json.loads(line))
    return records


def displacement_report(records: list[dict]) -> dict[str, float]:
    scorable = [r for r in records if r.get("future_trajectory") is not None]
    if not scorable:
        return {}
    predictions = np.stack([np.asarray(r["trajectories"])[0] for r in scorable])
    ground_truth = np.stack([np.asarray(r["future_trajectory"]) for r in scorable])
    return navsim_displacement_metrics(predictions, ground_truth)


def pdm_report(records: list[dict], metric_cache: Path) -> dict[str, float]:
    """Score every prediction with NAVSIM's PDM simulator, keeping the best sample."""
    from navsim.common.dataclasses import Trajectory, TrajectorySampling
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)
    loader = MetricCacheLoader(metric_cache)

    best: dict[str, list[dict]] = {"interp": [], "direct": []}
    for record in tqdm(records, desc="pdm score"):
        try:
            cache = loader.get_from_token(record["token"])
        except (KeyError, FileNotFoundError):
            continue
        rows = {"interp": [], "direct": []}
        for candidate in np.asarray(record["trajectories"], dtype=np.float32):
            interp = Trajectory(poses=resample_10hz_to_2hz(candidate).astype(np.float32))
            rows["interp"].append(
                asdict(pdm_score(cache, interp, sampling, simulator, scorer))
            )
            direct = Trajectory(poses=candidate[:40], trajectory_sampling=sampling)
            rows["direct"].append(
                asdict(
                    pdm_score(cache, direct, sampling, simulator, scorer, skip_interpolation=True)
                )
            )
        for key, candidates in rows.items():
            best[key].append(max(candidates, key=lambda row: row["score"]))

    report: dict[str, float] = {}
    for key, rows in best.items():
        if not rows:
            continue
        suffix = "" if key == "interp" else "_direct10hz"
        for field in PDM_FIELDS:
            report[f"{field}{suffix}"] = float(np.mean([row[field] for row in rows]))
        report[f"num_scored{suffix}"] = len(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--metric-cache", type=Path, default=None, help="NAVSIM metric cache root")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_predictions(args.predictions)
    metrics = displacement_report(records)
    if args.metric_cache is not None:
        metrics.update(pdm_report(records, args.metric_cache))
    metrics["num_predictions"] = len(records)

    width = max(len(name) for name in metrics)
    for name, value in metrics.items():
        print(f"  {name:<{width}}  {value:.4f}" if isinstance(value, float) else f"  {name:<{width}}  {value}")

    output = args.output or args.predictions.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "navsim_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote reports to {output}")


if __name__ == "__main__":
    main()
