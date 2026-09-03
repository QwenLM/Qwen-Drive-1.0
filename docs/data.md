# Data

A benchmark split is a JSON-lines file. Each line is one planning query and references its
camera frames by a path relative to the source dataset, so the same file works against any
copy of that dataset.

## Bundled files

The repository ships a runnable demo under `data/demo/`, which holds four WOD-E2E
validation scenes with their frames packed into `frames.parquet` plus six self-contained
perception frames under `perception/`. The demo files use paths relative to `data/demo/`
itself.
The benchmark scene files are not shipped. Each is built from the official validation
split of its dataset, see [building the validation scene
files](#building-the-validation-scene-files), and references frames relative to that
dataset's frame root.

## Scene record

```json
{
  "type": "chatml",
  "messages": [{"role": "user", "content": [ ... ]}, {"role": "assistant", "content": []}],
  "meta_info": {"dataset": "...", "token": "...", "scene_token": "...", "cam_order": [...]},
  "trajectory": { ... }
}
```

`messages[0].content` is the user turn. For each camera view it holds the view tag, then a
`frame: k` label and the image for every timestamp, and finally the instruction text. Views
are grouped, so all four timestamps of the front camera come first, then the front-left,
then the front-right.

```json
[
  {"text": "<FRONT VIEW>"},
  {"text": "frame: 0"},
  {"image": "val/<context>/FRONT/134.jpg", "resized_width": 384, "resized_height": 416},
  ...
  {"text": "The input images are organized by camera view. ..."}
]
```

`resized_width` and `resized_height` are the resolution each frame is resized to before it
is snapped to the patch grid. They keep history frames near 320p and the current frame near
720p, and they are part of reproducing the reported numbers.

The instruction text is stored verbatim and used as-is, so the evaluated prompt is
reproduced exactly. The reasoning request is **not** stored. The model appends it when the
reasoning mode is selected, which is why one scene file serves both planning modes.

## Trajectory fields

All quantities are in the ego frame of the current timestamp, with `x` forward, `y` left,
heading positive for a left turn, in metres and radians.

| Field | Shape | Meaning |
| --- | --- | --- |
| `hist_traj_10hz` | `[16, 3]` | Ego history over 1.5 s at 10 Hz, oldest first. The last row is the current pose `(0, 0, 0)` |
| `hist_vel_10hz` | `[16, 2]` | Velocity at those timestamps, m/s |
| `hist_acc_10hz` | `[16, 2]` | Acceleration at those timestamps, m/s² |
| `future_traj_10hz` | `[50, 3]` | Ground-truth future over 5 s at 10 Hz |
| `future_valid_mask_10hz` | `[50]` | 1 where the future pose is real |
| `ego_status.ego_velocity` | `[2]` | Current velocity |
| `ego_status.ego_acceleration` | `[2]` | Current acceleration |
| `ego_status.driving_command` | `[4]` | One-hot: left, straight, right, unknown |
| `nav_command` | scalar | 0 straight, 1 left, 2 right |
| `preference_trajectories` | list | Waymo only: rater-specified trajectories and their scores |

The Waymo split also carries `hist_*_1p5s_10hz` variants, which are preferred when present.
The expert receives the last 16 history points, re-references them to the oldest one, drops
that origin row and normalizes by the per-channel scale in the model config.

The ego status handed to the expert is the eight numbers
`[vx, vy, ax, ay, *driving_command]`, in that order.

## Trajectory sources
The 10 Hz trajectories in the scene files come from each dataset's raw data:
- **NAVSIM**: read from the original nuPlan data (pkl) at 10 Hz. The
  acceleration series is de-glitched with the official nuPlan Savitzky-Golay
  filter to remove spikes and drift, headings are wrapped to `[-pi, pi]`, and
  records whose yaw rate exceeds 1.2 rad/s or acceleration exceeds 1 g are
  rejected.
- **NVIDIA PhysicalAI**: read directly from the raw data at 10 Hz via the
  official egomotion interpolator, so acceleration comes from the
  interpolator itself instead of pose differentiation and carries no
  integration drift. Headings are wrapped to `[-pi, pi]` and the same yaw
  rate / acceleration guards apply.
- **Waymo Open Dataset end-to-end**: the raw metadata only carries the ego
  trajectory at 4 Hz (16 history points over 1.5 s, 20 future points over
  5 s), which is resampled to the 10 Hz fields above. Positions are fit with
  a cubic spline over the raw 4 Hz points (natural boundary for the future,
  anchored at the current pose) and evaluated on the 10 Hz grid. History
  headings come from the velocity direction, linearly interpolated to 10 Hz;
  future headings integrate the curvature of the position spline directly on
  the 10 Hz grid rather than interpolating a coarser 4 Hz heading curve.
  Velocities and accelerations are linearly resampled and rotated into the
  body frame of each heading. The interpolation function is
  [scripts/interpolate_waymo_trajectory.py](../scripts/interpolate_waymo_trajectory.py).


## Obtaining the camera frames

### NAVSIM

Frame paths look like `test/<log>/<CAM>/<token>.jpg`, the layout of NAVSIM's sensor blobs.
Download the OpenScene and NAVSIM test sensor blobs following the
[NAVSIM instructions](https://github.com/autonomousvision/navsim) and point `--image-root`
at the directory containing `test/`.

Cameras used: `CAM_F0` (front), `CAM_L0` (front-left), `CAM_R0` (front-right).

### Waymo Open Dataset end-to-end

Frame paths look like `val/<context_name>/<CAMERA>/<frame>.jpg`. Download the
[end-to-end driving dataset](https://waymo.com/open/) and extract the camera images per
context, camera and frame index. Point `--image-root` at the directory containing `val/`.

Cameras used: `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`.

### NVIDIA PhysicalAI

Frame paths look like `<clip_id>/<camera>/<frame>.jpg`, where `<frame>` is a zero-padded
frame **index into that clip's camera video**. This dataset ships videos rather than
images, so the frames have to be decoded.

1. Get the
   [PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
   dataset. Each clip has one video per camera, named `<clip_id>.<camera>.mp4`.
2. Decode every frame of each video, keeping the original frame order, and write frame `n`
   (zero-based) as `<clip_id>/<camera>/<n:08d>.jpg`. For example:

   ```bash
   ffmpeg -i "$clip.$camera.mp4" -qscale:v 2 -start_number 0 \
       "$root/$clip/$camera/%08d.jpg"
   ```

   Only the frames a scene file references are needed. The indices are the video's own
   frame numbers, so decoding must not drop or resample frames.

Cameras used: `camera_front_wide_120fov`, `camera_cross_left_120fov`,
`camera_cross_right_120fov`.

## Packing frames for faster loading

Each scene needs twelve small reads, which dominates wall clock on network storage.
`scripts/pack_images.py` copies exactly the frames a split references into uncompressed
Parquet shards that are memory-mapped on open. The stored bytes are the original encoded
frames, so decoded pixels are unchanged.

```bash
python scripts/pack_images.py \
    --scenes <SCENES_JSONL> \
    --image-root <IMAGE_ROOT> \
    --output archives/<SPLIT_NAME>
python scripts/run_planning.py ... --image-archive archives/<SPLIT_NAME>
```

## Reading frames from somewhere else

For storage that is neither a directory tree nor a Parquet archive, pass a callable that
maps a frame path to a `PIL.Image`, or to anything `CameraFrame` accepts, as
`image_resolver` to `read_scene_file`. From the command line, `--image-resolver
module:factory` imports a factory returning such a callable.

```bash
python scripts/run_planning.py ... --image-resolver my_storage:make_reader
```

## Building the validation scene files
Each benchmark scene file is a JSON-lines file derived from the official validation (or
test) split of its dataset, one record per planning query in the scene format above.
Every record needs:
- the three camera views at four timestamps, i.e. 1.5 s of history at 2 Hz
  plus the current frame, oldest to newest, referenced by paths relative to
  the dataset frame root
- the 10 Hz trajectory fields, i.e. 1.5 s history and 5 s future with their
  velocities and accelerations, the ego status and the navigation command,
  processed from the raw data as described under [trajectory
  sources](#trajectory-sources)
- `dataset`, `token` (the query) and `scene_token` (its scene) under `meta_info`, plus
  `cam_order` with the view tags in frame order
Per dataset, the validation queries are:
- **NAVSIM**: the keyframes of the official navtest split.
- **Waymo Open Dataset end-to-end**: only the validation samples that carry a
  human reference trajectory are evaluated. The 4 Hz raw trajectories are
  resampled with
  [scripts/interpolate_waymo_trajectory.py](../scripts/interpolate_waymo_trajectory.py).
- **NVIDIA PhysicalAI**: the keyframes of the clips listed in
  [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes). The
  trajectories are read directly from the raw data.


## Preparing your own scene file

To evaluate on your own data, write a scene file in the format above: image paths relative
to a frame root you pass as `--image-root`, the ego history and future series under
`trajectory`, and `token` with `scene_token` under `meta_info`. Building `DrivingScene`
objects directly in Python skips the file format entirely; see [cookbook.md](cookbook.md).

## Image preprocessing

Each frame goes through two resizes. First to the `resized_width` × `resized_height` in the
scene record, with bicubic interpolation. Then to the nearest multiple of
`patch_size * spatial_merge_size` (32) inside a pixel budget, again bicubic. Pixels are
scaled to `[0, 1]` and normalized with mean and standard deviation 0.5 per channel, then
grouped into patches of 16×16 over 2 duplicated temporal steps and merged 2×2, giving
`grid_h * grid_w / 4` tokens per frame.

For a scene built from scratch, leaving `CameraFrame.target_size` unset falls back to the
`history_image_pixels` and `current_image_pixels` budgets in the model config.
