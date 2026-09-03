# Cookbook

Recipes for the common tasks. Commands assume the repository root as the working directory
and `PYTHONPATH=src` when running from a checkout instead of `pip install -e .`.

- [The three inference modes](#the-three-inference-modes)
- [Drawing several trajectories per scene](#drawing-several-trajectories-per-scene)
- [Asking questions about a scene](#asking-questions-about-a-scene)
- [Building a scene yourself](#building-a-scene-yourself)
- [Running a benchmark split](#running-a-benchmark-split)
- [Sharding across GPUs](#sharding-across-gpus)
- [Resuming after preemption](#resuming-after-preemption)
- [Packing frames for faster loading](#packing-frames-for-faster-loading)
- [Perception](#perception)
- [Visualizing results](#visualizing-results)

## The three inference modes

| Mode | What runs | Output |
| --- | --- | --- |
| `InferenceMode.VQA` | VLM only | Text |
| `InferenceMode.DIRECT_PLANNING` | One VLM pass, then the expert | Trajectories |
| `InferenceMode.REASONING_PLANNING` | VLM writes a rationale, then the expert | Trajectories and reasoning |

The mode changes two things. Whether the user turn asks for a rationale, and whether the
expert reads the cache of a prompt-only pass or of a pass that includes the generated text.
The network is the same in all three.

`planner-rl` was reward-optimized only under reasoning-conditioned rollouts, where the reward
scores trajectories drawn from the model's own sampled reasoning. Run it in
`REASONING_PLANNING`. `planner-sft` works in both `DIRECT_PLANNING` and `REASONING_PLANNING`.

```python
model.run(InferenceMode.VQA, scene=scene, question="Who has right of way here?")
model.run(InferenceMode.DIRECT_PLANNING, scene=scene)
model.run(InferenceMode.REASONING_PLANNING, scene=scene)
```

`scripts/demo.py` runs all three modes on one bundled scene and prints the answer, the
rationale and the trajectories side by side.

```bash
python scripts/demo.py --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-rl \
    --scenes data/demo/planning_scenes.jsonl --image-archive data/demo/frames.parquet \
    --plot demo.png
```

## Drawing several trajectories per scene

`num_samples=N` draws N trajectories from independent noise in one batched pass. The VLM
runs once and its cache is shared, only the expert's conditioning is tiled. Sample `k`
always starts from seed `noise_seed + k`, so a sample is identical whether drawn alone or
inside a batch.

```python
result = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=6)
result.trajectories.shape   # (6, 50, 3)
```

## Asking questions about a scene

```python
result = model.generate_text(
    images=[frame.load() for frame in scene.views["<FRONT VIEW>"]],
    question="What is the state of the traffic light ahead?",
)
print(result.text)
```

`generate_text` sees only the images passed to it, not the ego history or the navigation
command. The same kind of question from the command line:

```bash
python scripts/run_vqa.py --model Qwen-Drive-1.0-4B \
    --image front.jpg --image front_left.jpg \
    --question "Is it safe to change into the left lane?"
```

## Building a scene yourself

A `DrivingScene` needs three camera views at four timestamps (1.5 s of history at 2 Hz,
current frame last), 1.5 s of ego history at 10 Hz, and the navigation command.

```python
from qwen_drive import CameraFrame, DrivingScene

scene = DrivingScene(
    views={
        "<FRONT VIEW>": [CameraFrame(p) for p in front_paths],        # oldest to current
        "<FRONT LEFT VIEW>": [CameraFrame(p) for p in left_paths],
        "<FRONT RIGHT VIEW>": [CameraFrame(p) for p in right_paths],
    },
    history=history_poses,               # [16, 3], last row is (0, 0, 0)
    history_velocity=history_velocity,   # [16, 2] m/s
    history_acceleration=history_accel,  # [16, 2] m/s^2
    ego_velocity=(12.4, -0.2),
    ego_acceleration=(0.24, 0.33),
    driving_command=[0, 1, 0, 0],        # left / straight / right / unknown
    nav_command=0,                       # 0 straight, 1 left, 2 right
)
```

[data.md](data.md) has the exact conventions and the scene-file format.

## Running a benchmark split

The benchmark scene files are not shipped with the repository. Build the one for your
split from the dataset's official validation/test data as described in
[data.md](data.md#building-the-validation-scene-files) (record format and trajectory
processing), obtain the camera frames from the dataset provider (see
[data.md](data.md#obtaining-the-camera-frames)), then predict and score:

```bash
# 1. predict
python scripts/run_planning.py --model Qwen-Drive-1.0-4B \
    --planner Qwen-Drive-1.0-4B/planner-rl \
    --scenes <SCENES_JSONL> \
    --image-root <IMAGE_ROOT> \
    --output outputs/<SPLIT_NAME>/predictions.jsonl \
    --mode reasoning_planning --num-samples 6

# 2. score
python scripts/eval_navsim.py       --predictions outputs/<SPLIT_NAME>/predictions.jsonl  # NAVSIM navtest
python scripts/eval_waymo.py        --predictions outputs/<SPLIT_NAME>/predictions.jsonl  # Waymo E2E validation
python scripts/eval_physical_ai.py  --predictions outputs/<SPLIT_NAME>/predictions.jsonl  # NVIDIA PhysicalAI test
```

| Split | Scoring |
| --- | --- |
| NAVSIM navtest | Displacement errors standalone. The PDM score needs the NAVSIM package, nuPlan maps and a metric cache |
| Waymo E2E validation | Displacement errors standalone. The rater feedback score needs the Waymo Open Dataset metrics |
| NVIDIA PhysicalAI test | Standalone |

Each record keeps only what the pipeline consumes: the prompt with its camera frames, the
ego history and future series, and `token` with `scene_token`. One scene file serves both
planning modes, because the model appends the reasoning request when the reasoning mode is
selected.

A run appends to its output and flushes every record, so relaunching the same command skips
the scenes already present and continues where it stopped, repairing a truncated last line.
To use several GPUs, give each process `--num-shards` and its own `--shard-index`, then
concatenate the per-shard files:

```bash
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python scripts/run_planning.py \
      --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-rl \
      --scenes <SCENES_JSONL> \
      --image-root <IMAGE_ROOT> --output outputs/<SPLIT_NAME>/pred_$gpu.jsonl \
      --num-shards 4 --shard-index $gpu &
done
wait
cat outputs/<SPLIT_NAME>/pred_*.jsonl > outputs/<SPLIT_NAME>/predictions.jsonl
python scripts/eval_navsim.py --predictions outputs/<SPLIT_NAME>/predictions.jsonl
```

## Packing frames for faster loading

Twelve small image reads per scene dominate wall clock on network storage. Pack them once
into memory-mappable Parquet shards, see [data.md](data.md#packing-frames-for-faster-loading)
for what the archive holds.


```bash
python scripts/pack_images.py --scenes <SCENES_JSONL> \
    --image-root <IMAGE_ROOT> --output archives/<SPLIT_NAME>
python scripts/run_planning.py ... --image-archive archives/<SPLIT_NAME>
```

## Perception

3D detection, occupancy, and BEV map segmentation from the surround camera ring.

```bash
python scripts/run_perception.py \
    --vlm Qwen-Drive-1.0-4B \
    --model Qwen-Drive-1.0-4B/perception \
    --frames data/demo/perception \
    --output outputs/perception_demo
```

Each frame directory under `--frames` bundles the camera ring, the calibration and the
ground truth. [perception.md](perception.md) has the layout and the coordinate conventions.

## Visualizing results

```bash
# planning: trajectory plot for one scene
python scripts/demo.py ... --plot demo.png

# perception: one summary image per frame (BEV, camera ring, map, occupancy)
python scripts/visualize_perception.py \
    --frames data/demo/perception --predictions outputs/perception_demo
```
