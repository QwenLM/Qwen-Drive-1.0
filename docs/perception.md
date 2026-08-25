# Perception: 3D detection, occupancy and BEV map segmentation

Qwen-Drive-1.0 also runs as a single-frame perception model. The same Qwen3.5 VLM processes
the surround-camera ring and a BEVFormer-style head set predicts, in one forward pass:

* **3D detection**, up to 300 boxes over 7 unified classes in the lidar frame, with scores
  and velocities.
* **occupancy**, a 200x200x16 semantic voxel grid over 10 classes in the ego frame.
* **BEV map segmentation**, a 60 m x 30 m raster at 0.15 m resolution over 6 classes.

## Structure

```
src/qwen_drive_perception/     inference-only package, no mmcv/mmdet/mmdet3d
  configuration_perception.py  frozen ranges, grid sizes, classes, palettes
  modeling_perception.py       QwenDrivePerception: VLM feature taps and BEV stack
  fpn.py                       SimpleFPN adaptors for the LLM and ViT streams
  view_transform.py            DepthNet and UVTR voxel pooling
  attention.py                 temporal, spatial and deformable attentions
  bev_encoder.py               BEVFormer encoder and detection decoder
  perception_transformer.py    transformer with occupancy and UVTR fusion
  occ_refiner.py               3D U-Net occupancy refiner
  map_seg.py                   map decoder, resnet18 trunk with U-Net upsampling
  heads.py                     detection head, NMS-free coder, map head
  geometry.py                  calibration, ego and lidar box transforms, projection
  dataset.py                   frame loader with prompt and patch processing
  visualize.py                 summary rendering
  ops/                         CUDA kernels (ms_deform_attn_bf16, voxel_pool)
scripts/
  run_perception.py            inference on packed frames
  visualize_perception.py      render summary images
data/demo/perception/          six self-contained sample frames
```

The perception weights ship separately from the VLM. The VLM is `Qwen-Drive-1.0-4B` and the
heads are the `perception/` subfolder of the same release directory, about 0.5 GB.

## Environment

Perception runs in the same environment as planning and VQA. `opencv-python` and
`matplotlib` are part of the base dependencies.

Two CUDA kernels are compiled on first use, which needs an `nvcc` matching the installed
torch build.

```bash
export CUDA_HOME=/usr/local/cuda-12.9   # adjust to your toolkit
export PATH=$CUDA_HOME/bin:$PATH
```

Compilation takes a couple of minutes once and is cached under `~/.cache/torch_extensions`.

## Inference

```bash
export PYTHONPATH=src
python scripts/run_perception.py \
    --vlm /path/to/Qwen-Drive-1.0-4B \
    --model /path/to/Qwen-Drive-1.0-4B/perception \
    --frames data/demo/perception \
    --output outputs/perception_demo
```

Each frame directory under `--frames` packs everything a sample needs.

| file | contents |
| --- | --- |
| `frame.json` | ordered prompt content (view tag and image pairs), camera order, dataset type |
| `images/*.jpg` | original camera frames |
| `calib.npz` | `cam_intrinsic`, `sensor2lidar_*`, `lidar2ego` |
| `gt.npz` | occupancy, map and 3D-box ground truth, already label-mapped |
| `lidar.npy` | lidar sweep for visualization |

Predictions are written to `<output>/<token>.npz` with boxes, scores, labels, occupancy and
map rasters.

## Visualization

```bash
python scripts/visualize_perception.py \
    --frames data/demo/perception \
    --predictions outputs/perception_demo
```

This produces one summary image per frame. The camera ring wraps around the central BEV panel,
each camera placed in the ring cell matching its azimuth, so the montage reads like the rig seen
from above. The BEV shows the lidar sweep coloured by height with ground-truth boxes in green and
predictions coloured per class. Underneath, the occupancy pair is drawn as 3D voxels and the map
pair as BEV rasters, followed by the class legend. Ego-forward points up in the BEV, map and
occupancy panels. The occupancy panels leave out background voxels hanging over the road,
which would otherwise hide the road surface from an elevated view, and both panels of the
pair are treated the same way.

## Coordinate conventions

* Boxes are `[x, y, z, w, l, h, yaw, vx, vy]` in the **lidar** frame, with z at the box
  bottom, yaw around +Z and `w` along the heading direction.
* The detection head regresses in ego coordinates and converts back through `inv(lidar2ego)`.
  Only the planar rotation is applied to yaw and velocity.
* The occupancy grid and the map raster are indexed in ego coordinates with X forward, Y left
  and Z up. Occupancy axes are `[X, Y, Z]`.
* Predicted boxes come back in the lidar frame, while the packed ground-truth boxes are in the
  ego frame like the occupancy grid and the map raster. `geometry.lidar_to_ego_boxes` converts
  between the two, which is what the visualization uses to draw everything in one frame.
* `lidar2img` folds the 896x512 resize into the projection, so it maps lidar points directly
  onto the model-resolution image plane.

## Notes

* The model was trained on nuScenes (6 cameras) and OpenScene/nuPlan (8 cameras) at a fixed
  896x512 input resolution with fixed camera configurations. A different camera layout or
  resolution is not covered by the released weights and will likely degrade results.
* Inference is strictly single-frame, so the temporal self-attention degenerates to plain BEV
  self-attention, exactly as during evaluation.
