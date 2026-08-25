#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#define THREADS_PER_BLOCK 256
#define GET_BLOCKS(N, T) (((N) + (T)-1) / (T))

template <typename T>
__device__ __forceinline__ float load_float(const T* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_float<c10::BFloat16>(const c10::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

template <>
__device__ __forceinline__ float load_float<c10::Half>(const c10::Half* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

template <typename T>
__device__ __forceinline__ void store_float(T* ptr, float value) {
  *ptr = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_float<c10::BFloat16>(c10::BFloat16* ptr, float value) {
  *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

template <>
__device__ __forceinline__ void store_float<c10::Half>(c10::Half* ptr, float value) {
  *reinterpret_cast<__half*>(ptr) = __float2half(value);
}

__device__ __forceinline__ float load_bf16(const c10::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr));
}

__device__ __forceinline__ void store_bf16(c10::BFloat16* ptr, float value) {
  *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

template <typename LocT, typename WeightT>
__global__ void ms_deform_attn_bf16_forward_kernel(
    const int n,
    const c10::BFloat16* __restrict__ value,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const LocT* __restrict__ sampling_loc,
    const WeightT* __restrict__ attn_weight,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    c10::BFloat16* __restrict__ output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }

  int temp = index;
  const int c = temp % channels;
  temp /= channels;
  const int head = temp % num_heads;
  temp /= num_heads;
  const int query = temp % num_query;
  temp /= num_query;
  const int batch = temp;

  const int qid_stride = num_heads * channels;
  const int value_batch_base = batch * spatial_size * qid_stride;
  int sample_index = ((batch * num_query + query) * num_heads + head) * num_levels * num_point;

  float acc = 0.0f;
  for (int level = 0; level < num_levels; ++level) {
    const int spatial_h = static_cast<int>(spatial_shapes[level * 2]);
    const int spatial_w = static_cast<int>(spatial_shapes[level * 2 + 1]);
    const int level_start = static_cast<int>(level_start_index[level]);
    const int value_level_base = value_batch_base + level_start * qid_stride + head * channels + c;

    for (int point = 0; point < num_point; ++point, ++sample_index) {
      const float loc_w = load_float(sampling_loc + sample_index * 2);
      const float loc_h = load_float(sampling_loc + sample_index * 2 + 1);
      const float weight = load_float(attn_weight + sample_index);
      const float h_im = loc_h * spatial_h - 0.5f;
      const float w_im = loc_w * spatial_w - 0.5f;

      if (h_im > -1.0f && w_im > -1.0f && h_im < spatial_h && w_im < spatial_w) {
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = h_im - h_low;
        const float lw = w_im - w_low;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;

        float val = 0.0f;
        if (h_low >= 0 && w_low >= 0) {
          val += hh * hw * load_bf16(value + value_level_base + (h_low * spatial_w + w_low) * qid_stride);
        }
        if (h_low >= 0 && w_high <= spatial_w - 1) {
          val += hh * lw * load_bf16(value + value_level_base + (h_low * spatial_w + w_high) * qid_stride);
        }
        if (h_high <= spatial_h - 1 && w_low >= 0) {
          val += lh * hw * load_bf16(value + value_level_base + (h_high * spatial_w + w_low) * qid_stride);
        }
        if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1) {
          val += lh * lw * load_bf16(value + value_level_base + (h_high * spatial_w + w_high) * qid_stride);
        }
        acc += val * weight;
      }
    }
  }
  store_bf16(output + index, acc);
}

template <typename LocT, typename WeightT>
__global__ void ms_deform_attn_bf16_backward_kernel(
    const int n_blocks,
    const c10::BFloat16* __restrict__ grad_output,
    const c10::BFloat16* __restrict__ value,
    const int64_t* __restrict__ spatial_shapes,
    const int64_t* __restrict__ level_start_index,
    const LocT* __restrict__ sampling_loc,
    const WeightT* __restrict__ attn_weight,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    float* __restrict__ grad_value,
    LocT* __restrict__ grad_sampling_loc,
    WeightT* __restrict__ grad_attn_weight) {
  const int block_index = blockIdx.x;
  if (block_index >= n_blocks) {
    return;
  }
  const int c = threadIdx.x;
  if (c >= channels) {
    return;
  }

  int temp = block_index;
  const int head = temp % num_heads;
  temp /= num_heads;
  const int query = temp % num_query;
  temp /= num_query;
  const int batch = temp;

  extern __shared__ float shared[];
  float* cache_grad_w = shared;
  float* cache_grad_h = shared + channels;
  float* cache_grad_attn = shared + channels * 2;

  const int qid_stride = num_heads * channels;
  const int value_batch_base = batch * spatial_size * qid_stride;
  const int grad_out_base = ((batch * num_query + query) * num_heads + head) * channels;
  int sample_index = ((batch * num_query + query) * num_heads + head) * num_levels * num_point;

  const float top_grad = load_bf16(grad_output + grad_out_base + c);

  for (int level = 0; level < num_levels; ++level) {
    const int spatial_h = static_cast<int>(spatial_shapes[level * 2]);
    const int spatial_w = static_cast<int>(spatial_shapes[level * 2 + 1]);
    const int level_start = static_cast<int>(level_start_index[level]);
    const int value_level_base = value_batch_base + level_start * qid_stride + head * channels + c;

    for (int point = 0; point < num_point; ++point, ++sample_index) {
      const float loc_w = load_float(sampling_loc + sample_index * 2);
      const float loc_h = load_float(sampling_loc + sample_index * 2 + 1);
      const float weight = load_float(attn_weight + sample_index);
      const float h_im = loc_h * spatial_h - 0.5f;
      const float w_im = loc_w * spatial_w - 0.5f;

      float grad_w = 0.0f;
      float grad_h = 0.0f;
      float grad_attn = 0.0f;

      if (h_im > -1.0f && w_im > -1.0f && h_im < spatial_h && w_im < spatial_w) {
        const int h_low = floorf(h_im);
        const int w_low = floorf(w_im);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = h_im - h_low;
        const float lw = w_im - w_low;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;
        const float w1 = hh * hw;
        const float w2 = hh * lw;
        const float w3 = lh * hw;
        const float w4 = lh * lw;
        const float top_grad_value = top_grad * weight;

        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
        float v4 = 0.0f;

        if (h_low >= 0 && w_low >= 0) {
          const int ptr = value_level_base + (h_low * spatial_w + w_low) * qid_stride;
          v1 = load_bf16(value + ptr);
          grad_h -= hw * v1;
          grad_w -= hh * v1;
          atomicAdd(grad_value + ptr, w1 * top_grad_value);
        }
        if (h_low >= 0 && w_high <= spatial_w - 1) {
          const int ptr = value_level_base + (h_low * spatial_w + w_high) * qid_stride;
          v2 = load_bf16(value + ptr);
          grad_h -= lw * v2;
          grad_w += hh * v2;
          atomicAdd(grad_value + ptr, w2 * top_grad_value);
        }
        if (h_high <= spatial_h - 1 && w_low >= 0) {
          const int ptr = value_level_base + (h_high * spatial_w + w_low) * qid_stride;
          v3 = load_bf16(value + ptr);
          grad_h += hw * v3;
          grad_w -= lh * v3;
          atomicAdd(grad_value + ptr, w3 * top_grad_value);
        }
        if (h_high <= spatial_h - 1 && w_high <= spatial_w - 1) {
          const int ptr = value_level_base + (h_high * spatial_w + w_high) * qid_stride;
          v4 = load_bf16(value + ptr);
          grad_h += lw * v4;
          grad_w += lh * v4;
          atomicAdd(grad_value + ptr, w4 * top_grad_value);
        }

        const float interpolated = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
        grad_attn = top_grad * interpolated;
        grad_w = spatial_w * grad_w * top_grad_value;
        grad_h = spatial_h * grad_h * top_grad_value;
      }

      cache_grad_w[c] = grad_w;
      cache_grad_h[c] = grad_h;
      cache_grad_attn[c] = grad_attn;
      __syncthreads();

      for (int stride = channels >> 1; stride > 0; stride >>= 1) {
        if (c < stride) {
          cache_grad_w[c] += cache_grad_w[c + stride];
          cache_grad_h[c] += cache_grad_h[c + stride];
          cache_grad_attn[c] += cache_grad_attn[c + stride];
        }
        __syncthreads();
      }

      if (c == 0) {
        store_float(grad_sampling_loc + sample_index * 2, cache_grad_w[0]);
        store_float(grad_sampling_loc + sample_index * 2 + 1, cache_grad_h[0]);
        store_float(grad_attn_weight + sample_index, cache_grad_attn[0]);
      }
      __syncthreads();
    }
  }
}

template <typename LocT, typename WeightT>
void launch_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    at::Tensor& output) {
  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);
  const int n = batch * num_query * num_heads * channels;

  ms_deform_attn_bf16_forward_kernel<LocT, WeightT>
      <<<GET_BLOCKS(n, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
          n,
          value.data_ptr<c10::BFloat16>(),
          spatial_shapes.data_ptr<int64_t>(),
          level_start_index.data_ptr<int64_t>(),
          sampling_loc.data_ptr<LocT>(),
          attn_weight.data_ptr<WeightT>(),
          spatial_size,
          num_heads,
          channels,
          num_levels,
          num_query,
          num_point,
          output.data_ptr<c10::BFloat16>());
}

template <typename LocT, typename WeightT>
void launch_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    at::Tensor& grad_value,
    at::Tensor& grad_sampling_loc,
    at::Tensor& grad_attn_weight) {
  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);
  const int n_blocks = batch * num_query * num_heads;
  const size_t shared_bytes = static_cast<size_t>(channels) * 3 * sizeof(float);

  ms_deform_attn_bf16_backward_kernel<LocT, WeightT>
      <<<n_blocks, channels, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
          n_blocks,
          grad_output.data_ptr<c10::BFloat16>(),
          value.data_ptr<c10::BFloat16>(),
          spatial_shapes.data_ptr<int64_t>(),
          level_start_index.data_ptr<int64_t>(),
          sampling_loc.data_ptr<LocT>(),
          attn_weight.data_ptr<WeightT>(),
          spatial_size,
          num_heads,
          channels,
          num_levels,
          num_query,
          num_point,
          grad_value.data_ptr<float>(),
          grad_sampling_loc.data_ptr<LocT>(),
          grad_attn_weight.data_ptr<WeightT>());
}

void dispatch_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    at::Tensor& output) {
  const auto loc_type = sampling_loc.scalar_type();
  const auto weight_type = attn_weight.scalar_type();
  if (loc_type == at::kFloat && weight_type == at::kFloat) {
    launch_forward<float, float>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kFloat && weight_type == at::kBFloat16) {
    launch_forward<float, c10::BFloat16>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kBFloat16 && weight_type == at::kBFloat16) {
    launch_forward<c10::BFloat16, c10::BFloat16>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kBFloat16 && weight_type == at::kFloat) {
    launch_forward<c10::BFloat16, float>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kHalf && weight_type == at::kHalf) {
    launch_forward<c10::Half, c10::Half>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kFloat && weight_type == at::kHalf) {
    launch_forward<float, c10::Half>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else if (loc_type == at::kHalf && weight_type == at::kFloat) {
    launch_forward<c10::Half, float>(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  } else {
    TORCH_CHECK(false, "unsupported sampling_loc/attn_weight dtype combination");
  }
}

void dispatch_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    at::Tensor& grad_value,
    at::Tensor& grad_sampling_loc,
    at::Tensor& grad_attn_weight) {
  const auto loc_type = sampling_loc.scalar_type();
  const auto weight_type = attn_weight.scalar_type();
  if (loc_type == at::kFloat && weight_type == at::kFloat) {
    launch_backward<float, float>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kFloat && weight_type == at::kBFloat16) {
    launch_backward<float, c10::BFloat16>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kBFloat16 && weight_type == at::kBFloat16) {
    launch_backward<c10::BFloat16, c10::BFloat16>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kBFloat16 && weight_type == at::kFloat) {
    launch_backward<c10::BFloat16, float>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kHalf && weight_type == at::kHalf) {
    launch_backward<c10::Half, c10::Half>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kFloat && weight_type == at::kHalf) {
    launch_backward<float, c10::Half>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else if (loc_type == at::kHalf && weight_type == at::kFloat) {
    launch_backward<c10::Half, float>(
        value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output,
        grad_value, grad_sampling_loc, grad_attn_weight);
  } else {
    TORCH_CHECK(false, "unsupported sampling_loc/attn_weight dtype combination");
  }
}

at::Tensor ms_deform_attn_bf16_cuda_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int im2col_step) {
  (void)im2col_step;
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value must be bfloat16");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index must be contiguous");
  TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
  TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");

  const int batch = value.size(0);
  const int num_query = sampling_loc.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  TORCH_CHECK(channels > 0 && channels <= 1024, "unsupported head_dim");
  TORCH_CHECK((channels & (channels - 1)) == 0,
              "head_dim must be a power of two for the native bf16 MSDeformAttn kernel");

  auto output = at::empty({batch, num_query, num_heads, channels}, value.options());
  dispatch_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, output);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.view({batch, num_query, num_heads * channels});
}

std::vector<at::Tensor> ms_deform_attn_bf16_cuda_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    const int im2col_step) {
  (void)im2col_step;
  TORCH_CHECK(value.is_cuda(), "value must be CUDA");
  TORCH_CHECK(value.scalar_type() == at::kBFloat16, "value must be bfloat16");
  TORCH_CHECK(grad_output.scalar_type() == at::kBFloat16, "grad_output must be bfloat16");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index must be contiguous");
  TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc must be contiguous");
  TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight must be contiguous");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  TORCH_CHECK(channels > 0 && channels <= 1024, "unsupported head_dim");
  TORCH_CHECK((channels & (channels - 1)) == 0,
              "head_dim must be a power of two for the native bf16 MSDeformAttn kernel");

  auto grad_value = at::zeros({batch, spatial_size, num_heads, channels},
                              value.options().dtype(at::kFloat));
  auto grad_sampling_loc = at::empty_like(sampling_loc);
  auto grad_attn_weight = at::empty_like(attn_weight);

  at::Tensor grad_output_view = grad_output;
  if (grad_output.dim() == 3) {
    const int num_query = sampling_loc.size(1);
    grad_output_view = grad_output.view({batch, num_query, num_heads, channels});
  }

  dispatch_backward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
      grad_output_view, grad_value, grad_sampling_loc, grad_attn_weight);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_value, grad_sampling_loc, grad_attn_weight};
}
