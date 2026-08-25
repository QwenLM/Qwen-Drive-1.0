# Model

Qwen-Drive-1.0 is a Qwen3.5 VLM with a Planning Expert attached. The VLM keeps the
pretrained architecture unchanged, so it keeps its language and vision-language abilities.
The expert is a separate 1.0 B parameter network that reads what the VLM already computed.

## Loading the weights

The release directory holds the VLM at its root and one subfolder per head. The VLM alone
answers questions, and a Planning Expert subfolder is attached when trajectories are needed.

```python
model = QwenDriveForPlanning.from_pretrained(
    "Qwen-Drive-1.0-4B", planner="Qwen-Drive-1.0-4B/planner-rl", dtype=torch.bfloat16
)

model.load_planner("Qwen-Drive-1.0-4B/planner-sft")   # or swap the expert later
```

The perception head lives in its own class, see [perception.md](perception.md).

## How the expert reads the scene

The VLM caches keys and values at its grouped-query softmax attention layers. The expert
reads those caches directly through cross-attention, so the scene enters the trajectory
network without any extra projection.

## The expert

One token per future waypoint, 50 of them. Each layer is a diffusion-transformer block whose
attention runs over the concatenation of the VLM's cached keys and values and the waypoint
tokens' own:

```
attention(Q_waypoints, [K_scene ; K_waypoints], [V_scene ; V_waypoints])
```

so the waypoints read the scene and each other at once. The block otherwise mirrors the
VLM's: a fused query/key/value projection with a per-head output gate, per-head RMS
normalization of queries and keys, grouped-query attention, and a SwiGLU feed-forward.

| | |
| --- | --- |
| Layers | 32 |
| Residual width | 1024 |
| Feed-forward width | 3584 |
| Query heads | 16, head dim 256 |
| Key/value heads | 4 |
| Caches consumed | 8, one per 4 layers |

**What each waypoint token is built from.** Seven signals are concatenated and fused by an
MLP: the current noisy waypoint, its Fourier features, the flow time, the encoded ego history
poses, a learned waypoint-index embedding, the encoded history velocity, and the encoded
history acceleration.

**What conditions each layer.** Flow time, the navigation command and the ego status are
summed and injected through adaptive layer normalization shared across the block, which
produces the six shift, scale and gate vectors used around the attention and the
feed-forward.

**Where the waypoints sit.** The waypoint tokens are given multimodal rotary positions
continuing right after the VLM's prefix, so their phases follow the language model's. The
last prefix token is text, so all three rotary sections share the same anchor.

## Trajectory representation and normalization

A trajectory is 50 waypoints of `(x, y, heading)` covering 5 s at 10 Hz, in the ego frame of
the current timestamp, with `x` forward, `y` left and heading positive for a left turn.

Normalization divides each channel by a fixed scale and denormalization multiplies back,
wrapping the heading into `[-pi, pi)`.

| Channel | Scale |
| --- | --- |
| `x` | 165 m |
| `y` | 25 m |
| `heading` | 1.5703125 rad |

History is handled slightly differently. The 16 poses are re-referenced to the oldest one so
history and future both progress in the driving direction, that origin row is dropped as it
carries no information, and the remaining 15 poses use the same scale.

## Sampling

Training uses flow matching with a clean-endpoint parameterization, so the network predicts
the finished trajectory rather than a velocity. Sampling starts from Gaussian noise and takes
10 Euler steps, converting the prediction to a velocity by dividing by the time remaining.

```
v = (x1_hat - x_t) / max(1 - t, 0.1)
x_t+dt = x_t + v * dt
```

The floor on the divisor keeps the last step from amplifying prediction error. Sample `k`
draws its noise from seed `noise_seed + k`, so results are reproducible and a sample is
identical whether drawn alone or inside a batch.

## VQA decoding

The VQA mode decodes with the evaluation parameters. With `top_k=1` and a near-zero
temperature this is effectively greedy, while keeping the sampling path deterministic.

| Parameter | Value |
| --- | --- |
| `do_sample` | `True` |
| `temperature` | 0.01 |
| `top_k` | 1 |
| `top_p` | 0.001 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 0.0 |
| `max_new_tokens` | 32768 |
| `seed` | 3407 |

They are the defaults of `generate_text` (`VQA_DECODE_DEFAULTS`) and any of them can be
overridden per call.

## Configuration reference

`QwenDriveConfig` holds the trajectory and sampler fields. `PlanningExpertConfig`, nested as
`expert_config`, holds the expert's geometry.

### Trajectory

| Field | Default | Meaning |
| --- | --- | --- |
| `num_future_points` | 50 | Waypoints predicted |
| `num_history_points` | 16 | History poses consumed |
| `trajectory_point_dim` | 3 | `(x, y, heading)` |
| `trajectory_hz` | 10.0 | Output rate |
| `trajectory_scale` | `[165.0, 25.0, 1.5703125]` | Per-channel normalization |

### Sampler

| Field | Default | Meaning |
| --- | --- | --- |
| `num_inference_steps` | 10 | Euler steps |
| `noise_init_std` | 1.0 | Scale of the initial noise |
| `noise_seed` | 42 | Seed of sample 0 |
| `min_one_minus_t` | 0.1 | Floor on the remaining-time divisor |
| `max_reasoning_tokens` | 256 | Cap on generated reasoning |

### Images

| Field | Default | Meaning |
| --- | --- | --- |
| `history_image_pixels` | 174080 | Pixel budget for history frames |
| `current_image_pixels` | 921600 | Pixel budget for the current frame |
| `image_patch_size` | 16 | Vision patch size |
| `image_temporal_patch_size` | 2 | Temporal patch size |
| `image_spatial_merge_size` | 2 | Spatial merge factor |

The budgets only apply when a frame carries no explicit `target_size`. Benchmark scene files
always do.

### Expert geometry

| Field | Default |
| --- | --- |
| `hidden_size` | 1024 |
| `intermediate_size` | 3584 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| `layers_per_kv` | 4 |
| `time_embed_dim` | 128 |
| `fourier_num_features` | 16 |
| `fourier_max_frequency` | 16.0 |
| `rope_theta` | 1e7 |
| `partial_rotary_factor` | 0.25 |
| `mrope_section` | `[11, 11, 10]` |

The attention geometry, the rotary parameters and `layers_per_kv` are dictated by the VLM and
should not be changed independently of it.
