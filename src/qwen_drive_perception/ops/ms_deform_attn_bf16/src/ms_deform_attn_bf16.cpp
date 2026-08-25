#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

at::Tensor ms_deform_attn_bf16_cuda_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int im2col_step);

std::vector<at::Tensor> ms_deform_attn_bf16_cuda_backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    const int im2col_step);

at::Tensor forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int im2col_step) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(value));
  return ms_deform_attn_bf16_cuda_forward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
}

std::vector<at::Tensor> backward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const at::Tensor& grad_output,
    const int im2col_step) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(value));
  return ms_deform_attn_bf16_cuda_backward(
      value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "bf16 MSDeformAttn forward (CUDA)");
  m.def("backward", &backward, "bf16 MSDeformAttn backward (CUDA)");
}
