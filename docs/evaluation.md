# Evaluation

Evaluation runs in two steps. Predict trajectories for a split, then score the predictions
file. A slow inference pass therefore runs once and can be scored under several conventions
afterwards.

```bash
python scripts/run_planning.py --model Qwen-Drive-1.0-4B \
    --planner Qwen-Drive-1.0-4B/planner-rl \
    --scenes data/benchmarks/<split>.jsonl --image-root /data/<dataset> \
    --output outputs/<split>/predictions.jsonl \
    --mode reasoning_planning --num-samples 6

python scripts/eval_<benchmark>.py --predictions outputs/<split>/predictions.jsonl
```

## Reported results

`SFT` is the imitation-trained Planning Expert, `RL` the same expert after reward
optimization. `best-of-6` keeps the highest-scoring of six sampled trajectories. That
selection uses the ground truth, so it is an upper bound on what inference-time selection
could reach.

**NAVSIM v1.1 navtest, PDM score**

| | NC | DAC | EP | TTC | Comfort | PDMS |
| --- | --- | --- | --- | --- | --- | --- |
| SFT, w/o reasoning | 98.2 | 96.4 | 82.0 | 94.4 | 100.0 | 87.8 |
| SFT, w/ reasoning | 98.4 | 96.6 | 82.4 | 94.7 | 100.0 | 88.2 |
| SFT, w/ reasoning, best-of-6 | 98.7 | 97.2 | 83.2 | 95.5 | 100.0 | 89.3 |
| RL | 98.6 | 98.2 | 84.8 | 95.9 | 100.0 | 90.7 |
| RL, best-of-6 | 98.8 | 98.4 | 85.5 | 96.5 | 100.0 | 91.4 |

**Waymo Open Dataset end-to-end.** Displacement errors are measured against the recorded
trajectory. The rater feedback score instead matches the prediction to human-rated
candidates, so it can accept a good plan that differs from what was driven. The validation
ratings supervise the reward, so the validation RL row is in-sample.

| | ADE 3 s | ADE 5 s | RFS |
| --- | --- | --- | --- |
| test, SFT w/o reasoning | 1.20 | 2.66 | 7.76 |
| test, SFT w/ reasoning | 1.19 | 2.65 | 7.78 |
| test, RL | 1.19 | 2.67 | 7.91 |
| val, SFT w/ reasoning | 0.99 | 2.31 | 7.95 |
| val, RL | 0.62 | 1.27 | 8.45 |

**NVIDIA PhysicalAI open-loop, six trajectories per scene**

| | ADE 3 s | ADE 5 s | minADE 3 s | minADE 5 s |
| --- | --- | --- | --- | --- |
| 644-example split, SFT w/o reasoning | 0.38 | 1.07 | 0.34 | 0.96 |
| 644-example split, SFT w/ reasoning | 0.37 | 1.07 | 0.34 | 0.97 |
| 644-example split, RL | 0.42 | 1.11 | 0.38 | 1.00 |
| 700-frame subset, SFT w/ reasoning | 0.42 | 1.23 | 0.39 | 1.11 |
| 700-frame subset, RL | 0.47 | 1.27 | 0.43 | 1.15 |

Reward optimization trades a few centimetres of open-loop displacement for better
preference alignment and a higher pseudo-closed-loop score. Perception and VQA results are
in the paper.

## The predictions file

One JSON object per scene.

| Field | Meaning |
| --- | --- |
| `token`, `scene_token` | Identifiers from the scene file |
| `trajectories` | `[num_samples, 50, 3]` of `(x, y, heading)`, metres and radians, ego frame, 10 Hz |
| `reasoning` | The generated rationale, or `null` in direct mode |
| `future_trajectory`, `future_valid` | Ground truth copied from the scene file |
| `preference_trajectories`, `preference_scores` | Waymo only |
| `initial_speed` | Speed at the current timestamp, used by the Waymo rater score |

## Converting the 5 s / 10 Hz output

The model always predicts the same 50 poses. Each benchmark wants a different grid, and the
conversions live in `qwen_drive.trajectory`.

| Benchmark | Target grid | Conversion |
| --- | --- | --- |
| NAVSIM, `interp` | 8 poses at 2 Hz, t = 0.5 … 4.0 s | Take indices 4, 9, …, 39, the poses the simulator's interpolation grid lands on |
| NAVSIM, `direct` | 40 poses at 10 Hz | Use the leading 40 poses unchanged |
| Waymo | 20 poses at 4 Hz, t = 0.25 … 5.0 s | Interpolate, heading through its sine and cosine |
| PhysicalAI | 50 poses at 10 Hz | None |

Waymo offers two resampling conventions. The default maps both grids onto a normalized
index ramp, which preserves the endpoints and is what the reported numbers use.
`--official-4hz-grid` instead interpolates onto the benchmark's absolute timestamps from the
prediction's own `0.1 … 5.0 s`. The two differ slightly at intermediate poses.

## NVIDIA PhysicalAI

Fully self-contained. The predictions file carries the ground truth, so only numpy is
needed.

```bash
python scripts/eval_physical_ai.py --predictions outputs/physical_ai/predictions.jsonl
```

Reports, at 1 s through 5 s:

- `ADE_<h>`, the mean displacement over the horizon, averaged over the sampled trajectories.
- `minADE_<h>`, the same for the best sample. Selecting it needs the ground truth, so this
  is an **oracle upper bound**.
- `FDE_<h>` and `minFDE_<h>`, the displacement at the horizon's last pose.

The reported numbers use six samples per scene.

## Waymo Open Dataset end-to-end

Displacement errors need only numpy. The rater feedback score needs the official
implementation from the Waymo Open Dataset, which is not a dependency here.

```bash
python scripts/eval_waymo.py --predictions outputs/waymo/predictions.jsonl

# with the rater feedback score
export WAYMO_OPEN_DATASET_SRC=/path/to/waymo-open-dataset/src
python scripts/eval_waymo.py --predictions outputs/waymo/predictions.jsonl --rater-feedback
```

Reports `ADE`, `FDE` and `HeadingMAE` on the 4 Hz grid plus `ADE@{1,3,5}s` and
`FDE@{1,3,5}s`, the same on the native 10 Hz grid with a `_10hz` suffix, and `RFS` when
requested. `ADE@Xs` is the mean up to X seconds, `FDE@Xs` the single pose at X seconds. The
benchmark publishes the `ADE@3s`, `ADE@5s` and `RFS` keys of the 4 Hz grid, the remaining
keys are there for diagnosis.

With several samples per scene, the candidate closest to the highest-rated preference
trajectory is kept. That uses the rater labels, so it is an oracle bound.

## NAVSIM v1.1

Displacement errors over the 4 s window need only the predictions file. The PDM score is a
pseudo-closed-loop simulation and needs more:

- the NAVSIM package (`pip install -e navsim`) and `nuplan-devkit`,
- the nuPlan maps, via `NUPLAN_MAPS_ROOT` and `NUPLAN_MAP_VERSION`,
- a prebuilt metric cache for the split, passed as `--metric-cache`.

```bash
export NUPLAN_MAPS_ROOT=/data/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
python scripts/eval_navsim.py \
    --predictions outputs/navsim/predictions.jsonl \
    --metric-cache /data/navsim/metric_cache_navtest
```

Both conventions from the table above are reported. The plain field names are the `interp`
path and `_direct10hz` is the native-rate path. The sub-scores are no-at-fault collisions,
drivable-area compliance, ego progress, time-to-collision, comfort and driving-direction
compliance, plus the combined `score` (PDMS). With several samples per scene the
highest-scoring one is kept per convention, which is an oracle bound.

## Sampling several trajectories

`--num-samples N` costs one VLM pass plus N expert rollouts rather than N full passes. The
VLM runs once and its cache is broadcast while only the expert's conditioning is tiled. The
expert rollout is the cheap part, so wall clock grows far more slowly than N.

Sample `k` starts from seed `noise_seed + k`. Pass `--seed` to move the whole set.

## Reproducibility notes

- Sampling is deterministic given the seed. The noise comes from an explicitly seeded
  generator and the reasoning stage decodes greedily.
- Every benchmark scene file stores its prompt verbatim, so the text the model sees is the
  text it was evaluated with.
- Frames are resized to the resolution recorded per frame in the scene file before being
  snapped to the patch grid. A different resolution changes the vision tokens and therefore
  the trajectory.
- `--image-archive` and `--image-root` decode identical pixels. The archive only changes how
  the bytes are read.
