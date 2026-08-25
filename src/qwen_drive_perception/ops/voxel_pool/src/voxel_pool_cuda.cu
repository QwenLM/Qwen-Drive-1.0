#include <stdio.h>
#include <stdlib.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<c10::Half>(const c10::Half* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

template <>
__device__ __forceinline__ float load_as_float<c10::BFloat16>(const c10::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__global__ void voxel_pool_kernel(int B, int X, int Y, int Z, int N, int C, int n_intervals,
                                  const float* feats, const int* coords, const int* interval_starts,
                                  const int* interval_lengths, float* out) {
  // feats: (N, C)
  // coords: (N, 4), [bs_idx, x, y, z]
  // out: (B, X, Y, Z, C)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / C;
  int cur_c = idx % C;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  const float* cur_feats = feats + interval_start * C + cur_c;
  float* cur_out = out + cur_coords[0] * X * Y * Z * C +
                   cur_coords[1] * Y * Z * C + 
                   cur_coords[2] * Z * C +
                   cur_coords[3] * C + 
                   cur_c;
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_feats[i * C];
  }
  *cur_out = psum;
}

__global__ void voxel_pool_grad_kernel(int B, int X, int Y, int Z, int N, int C, int n_intervals,
                                       const float* out_grad, const int* coords, const int* interval_starts,
                                       const int* interval_lengths, float* feats_grad) {
  // out_grad: (B, X, Y, Z, C)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / C;
  int cur_c = idx % C;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_coords = coords + interval_start * 4;
  float* cur_feats_grad = feats_grad + interval_start * C + cur_c;
  const float* cur_out_grad = out_grad + cur_coords[0] * X * Y * Z * C +
                              cur_coords[1] * Y * Z * C +
                              cur_coords[2] * Z * C +
                              cur_coords[3] * C +
                              cur_c;
  for(int i = 0; i < interval_length; i++){
    cur_feats_grad[i * C] = *cur_out_grad;
  }
}

void voxel_pool(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* feats,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* out) {
  voxel_pool_kernel<<<DIVUP(n_intervals * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    B, X, Y, Z, N, C, n_intervals, feats, coords, interval_starts, interval_lengths, out
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void voxel_pool_grad(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* out_grad,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* feats_grad) {
  voxel_pool_grad_kernel<<<DIVUP(n_intervals * C, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
    B, X, Y, Z, N, C, n_intervals, out_grad, coords, interval_starts, interval_lengths, feats_grad
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename feat_t, typename depth_t>
__global__ void voxel_pool_depth_forward_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int n_intervals, int camera_index,
    const feat_t* __restrict__ img_feats,
    const depth_t* __restrict__ img_depth,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    const int* __restrict__ sort_indices,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int interval_index = idx / feat_channels;
  int channel = idx % feat_channels;
  if (interval_index >= n_intervals) return;

  int interval_start = interval_starts[interval_index];
  int interval_length = interval_lengths[interval_index];
  int first_sorted = sort_indices[interval_start];
  const int* cur_coords = coords + first_sorted * 4;
  float psum = 0.0f;

  for (int i = 0; i < interval_length; ++i) {
    int valid_index = sort_indices[interval_start + i];
    int point_index = point_indices[valid_index];
    int p = point_index;
    int w = p % W;
    p /= W;
    int h = p % H;
    p /= H;
    int d = p % D;
    p /= D;
    int sweep = p % N_sweep;
    int batch = p / N_sweep;

    int image_index = (batch * N_sweep + sweep) * N_cam + camera_index;
    int feat_offset = ((image_index * feat_channels + channel) * H + h) * W + w;
    int depth_offset = ((image_index * D + d) * H + h) * W + w;
    psum += load_as_float(img_feats + feat_offset) * load_as_float(img_depth + depth_offset);
  }

  int out_offset = ((cur_coords[0] * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  out[(out_offset + cur_coords[3]) * feat_channels + channel] = psum;
}

template <typename feat_t, typename depth_t>
__global__ void voxel_pool_depth_forward_all_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int n_intervals,
    const feat_t* __restrict__ img_feats,
    const depth_t* __restrict__ img_depth,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    const int* __restrict__ sort_indices,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int interval_index = idx / feat_channels;
  int channel = idx % feat_channels;
  if (interval_index >= n_intervals) return;

  int interval_start = interval_starts[interval_index];
  int interval_length = interval_lengths[interval_index];
  int first_sorted = sort_indices[interval_start];
  int first_point = point_indices[first_sorted];
  int first = first_point;
  first /= W;
  first /= H;
  first /= D;
  int camera = first % N_cam;
  const int* cur_coords = coords + first_sorted * 4;
  float psum = 0.0f;

  for (int i = 0; i < interval_length; ++i) {
    int valid_index = sort_indices[interval_start + i];
    int point_index = point_indices[valid_index];
    int p = point_index;
    int w = p % W;
    p /= W;
    int h = p % H;
    p /= H;
    int d = p % D;
    p /= D;
    int cam = p % N_cam;
    p /= N_cam;
    int sweep = p % N_sweep;
    int batch = p / N_sweep;

    int image_index = (batch * N_sweep + sweep) * N_cam + cam;
    int feat_offset = ((image_index * feat_channels + channel) * H + h) * W + w;
    int depth_offset = ((image_index * D + d) * H + h) * W + w;
    psum += load_as_float(img_feats + feat_offset) * load_as_float(img_depth + depth_offset);
  }

  int out_offset = (((cur_coords[0] * N_cam + camera) * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  out[(out_offset + cur_coords[3]) * feat_channels + channel] = psum;
}

template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_feat_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int camera_index,
    const grad_t* __restrict__ out_grad,
    const depth_t* __restrict__ img_depth,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    float* __restrict__ img_feats_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int valid_index = idx / feat_channels;
  int channel = idx % feat_channels;
  if (valid_index >= n_valid) return;

  int point_index = point_indices[valid_index];
  int p = point_index;
  int w = p % W;
  p /= W;
  int h = p % H;
  p /= H;
  int d = p % D;
  p /= D;
  int sweep = p % N_sweep;
  int batch = p / N_sweep;

  const int* cur_coords = coords + valid_index * 4;
  int image_index = (batch * N_sweep + sweep) * N_cam + camera_index;
  int depth_offset = ((image_index * D + d) * H + h) * W + w;
  int grad_out_offset = ((cur_coords[0] * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  grad_out_offset = (grad_out_offset + cur_coords[3]) * feat_channels + channel;
  int grad_feat_offset = ((image_index * feat_channels + channel) * H + h) * W + w;

  float grad = load_as_float(out_grad + grad_out_offset) * load_as_float(img_depth + depth_offset);
  atomicAdd(img_feats_grad + grad_feat_offset, grad);
}

template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_all_feat_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels,
    const grad_t* __restrict__ out_grad,
    const depth_t* __restrict__ img_depth,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    float* __restrict__ img_feats_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int valid_index = idx / feat_channels;
  int channel = idx % feat_channels;
  if (valid_index >= n_valid) return;

  int point_index = point_indices[valid_index];
  int p = point_index;
  int w = p % W;
  p /= W;
  int h = p % H;
  p /= H;
  int d = p % D;
  p /= D;
  int cam = p % N_cam;
  p /= N_cam;
  int sweep = p % N_sweep;
  int batch = p / N_sweep;

  const int* cur_coords = coords + valid_index * 4;
  int image_index = (batch * N_sweep + sweep) * N_cam + cam;
  int depth_offset = ((image_index * D + d) * H + h) * W + w;
  int grad_out_offset = (((cur_coords[0] * N_cam + cam) * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  grad_out_offset = (grad_out_offset + cur_coords[3]) * feat_channels + channel;
  int grad_feat_offset = ((image_index * feat_channels + channel) * H + h) * W + w;

  float grad = load_as_float(out_grad + grad_out_offset) * load_as_float(img_depth + depth_offset);
  atomicAdd(img_feats_grad + grad_feat_offset, grad);
}

template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_depth_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int camera_index,
    const grad_t* __restrict__ out_grad,
    const feat_t* __restrict__ img_feats,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    float* __restrict__ img_depth_grad) {
  int valid_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (valid_index >= n_valid) return;

  int point_index = point_indices[valid_index];
  int p = point_index;
  int w = p % W;
  p /= W;
  int h = p % H;
  p /= H;
  int d = p % D;
  p /= D;
  int sweep = p % N_sweep;
  int batch = p / N_sweep;

  const int* cur_coords = coords + valid_index * 4;
  int image_index = (batch * N_sweep + sweep) * N_cam + camera_index;
  int grad_out_base = ((cur_coords[0] * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  grad_out_base = (grad_out_base + cur_coords[3]) * feat_channels;
  int feat_base = (image_index * feat_channels * H + h) * W + w;

  float grad = 0.0f;
  for (int channel = 0; channel < feat_channels; ++channel) {
    int feat_offset = feat_base + channel * H * W;
    grad += load_as_float(out_grad + grad_out_base + channel) * load_as_float(img_feats + feat_offset);
  }

  int depth_offset = ((image_index * D + d) * H + h) * W + w;
  img_depth_grad[depth_offset] = grad;
}

template <typename feat_t, typename depth_t, typename grad_t>
__global__ void voxel_pool_depth_backward_all_depth_kernel(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels,
    const grad_t* __restrict__ out_grad,
    const feat_t* __restrict__ img_feats,
    const int* __restrict__ coords,
    const int* __restrict__ point_indices,
    float* __restrict__ img_depth_grad) {
  int valid_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (valid_index >= n_valid) return;

  int point_index = point_indices[valid_index];
  int p = point_index;
  int w = p % W;
  p /= W;
  int h = p % H;
  p /= H;
  int d = p % D;
  p /= D;
  int cam = p % N_cam;
  p /= N_cam;
  int sweep = p % N_sweep;
  int batch = p / N_sweep;

  const int* cur_coords = coords + valid_index * 4;
  int image_index = (batch * N_sweep + sweep) * N_cam + cam;
  int grad_out_base = (((cur_coords[0] * N_cam + cam) * X + cur_coords[1]) * Y + cur_coords[2]) * Z;
  grad_out_base = (grad_out_base + cur_coords[3]) * feat_channels;
  int feat_base = (image_index * feat_channels * H + h) * W + w;

  float grad = 0.0f;
  for (int channel = 0; channel < feat_channels; ++channel) {
    int feat_offset = feat_base + channel * H * W;
    grad += load_as_float(out_grad + grad_out_base + channel) * load_as_float(img_feats + feat_offset);
  }

  int depth_offset = ((image_index * D + d) * H + h) * W + w;
  img_depth_grad[depth_offset] = grad;
}

template <typename feat_t, typename depth_t>
void voxel_pool_depth_forward_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int n_intervals, int camera_index,
    const feat_t* img_feats, const depth_t* img_depth, const int* coords,
    const int* point_indices, const int* sort_indices, const int* interval_starts,
    const int* interval_lengths, float* out) {
  voxel_pool_depth_forward_kernel<feat_t, depth_t>
      <<<DIVUP(n_intervals * feat_channels, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          n_intervals, camera_index, img_feats, img_depth, coords, point_indices,
          sort_indices, interval_starts, interval_lengths, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename feat_t, typename depth_t>
void voxel_pool_depth_forward_all_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int n_intervals,
    const feat_t* img_feats, const depth_t* img_depth, const int* coords,
    const int* point_indices, const int* sort_indices, const int* interval_starts,
    const int* interval_lengths, float* out) {
  voxel_pool_depth_forward_all_kernel<feat_t, depth_t>
      <<<DIVUP(n_intervals * feat_channels, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          n_intervals, img_feats, img_depth, coords, point_indices,
          sort_indices, interval_starts, interval_lengths, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename feat_t, typename depth_t, typename grad_t>
void voxel_pool_depth_backward_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int camera_index,
    const grad_t* out_grad, const feat_t* img_feats, const depth_t* img_depth,
    const int* coords, const int* point_indices, float* img_feats_grad,
    float* img_depth_grad) {
  voxel_pool_depth_backward_feat_kernel<feat_t, depth_t, grad_t>
      <<<DIVUP(n_valid * feat_channels, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          camera_index, out_grad, img_depth, coords, point_indices, img_feats_grad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  voxel_pool_depth_backward_depth_kernel<feat_t, depth_t, grad_t>
      <<<DIVUP(n_valid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          camera_index, out_grad, img_feats, coords, point_indices, img_depth_grad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename feat_t, typename depth_t, typename grad_t>
void voxel_pool_depth_backward_all_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels,
    const grad_t* out_grad, const feat_t* img_feats, const depth_t* img_depth,
    const int* coords, const int* point_indices, float* img_feats_grad,
    float* img_depth_grad) {
  voxel_pool_depth_backward_all_feat_kernel<feat_t, depth_t, grad_t>
      <<<DIVUP(n_valid * feat_channels, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          out_grad, img_depth, coords, point_indices, img_feats_grad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  voxel_pool_depth_backward_all_depth_kernel<feat_t, depth_t, grad_t>
      <<<DIVUP(n_valid, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
          B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels,
          out_grad, img_feats, coords, point_indices, img_depth_grad);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE_VOXEL_POOL_DEPTH_FORWARD(FEAT_T, DEPTH_T) \
  template void voxel_pool_depth_forward_all_cuda<FEAT_T, DEPTH_T>( \
      int, int, int, int, int, int, int, int, int, int, int, int, \
      const FEAT_T*, const DEPTH_T*, const int*, const int*, const int*, \
      const int*, const int*, float*);

#define INSTANTIATE_VOXEL_POOL_DEPTH_BACKWARD(FEAT_T, DEPTH_T, GRAD_T) \
  template void voxel_pool_depth_backward_all_cuda<FEAT_T, DEPTH_T, GRAD_T>( \
      int, int, int, int, int, int, int, int, int, int, int, \
      const GRAD_T*, const FEAT_T*, const DEPTH_T*, const int*, const int*, \
      float*, float*);

#define INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(FEAT_T, DEPTH_T) \
  INSTANTIATE_VOXEL_POOL_DEPTH_FORWARD(FEAT_T, DEPTH_T) \
  INSTANTIATE_VOXEL_POOL_DEPTH_BACKWARD(FEAT_T, DEPTH_T, float) \
  INSTANTIATE_VOXEL_POOL_DEPTH_BACKWARD(FEAT_T, DEPTH_T, c10::Half) \
  INSTANTIATE_VOXEL_POOL_DEPTH_BACKWARD(FEAT_T, DEPTH_T, c10::BFloat16)

INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(float, float)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(float, c10::Half)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(float, c10::BFloat16)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::Half, float)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::Half, c10::Half)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::Half, c10::BFloat16)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::BFloat16, float)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::BFloat16, c10::Half)
INSTANTIATE_VOXEL_POOL_DEPTH_GRADS(c10::BFloat16, c10::BFloat16)
