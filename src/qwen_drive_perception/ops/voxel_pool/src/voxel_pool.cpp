#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

// CUDA function declarations
void voxel_pool(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* feats,
    const int* coords, const int* interval_starts, const int* interval_lengths, float* out);

void voxel_pool_grad(int B, int X, int Y, int Z, int N, int C, int n_intervals, const float* out_grad,
  const int* coords, const int* interval_starts, const int* interval_lengths, float* feats_grad);

template <typename feat_t, typename depth_t>
void voxel_pool_depth_forward_all_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels, int n_intervals,
    const feat_t* img_feats, const depth_t* img_depth, const int* coords,
    const int* point_indices, const int* sort_indices, const int* interval_starts,
    const int* interval_lengths, float* out);

template <typename feat_t, typename depth_t, typename grad_t>
void voxel_pool_depth_backward_all_cuda(
    int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W,
    int n_valid, int feat_channels,
    const grad_t* out_grad, const feat_t* img_feats, const depth_t* img_depth,
    const int* coords, const int* point_indices, float* img_feats_grad,
    float* img_depth_grad);

at::Tensor voxel_pool_forward(const at::Tensor _feats, const at::Tensor _coords, const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts, int B, int X, int Y, int Z
) {
  int N = _feats.size(0);
  int C = _feats.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_feats));
  const float* feats = _feats.data_ptr<float>();
  const int* coords = _coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options = torch::TensorOptions().dtype(_feats.dtype()).device(_feats.device());
  at::Tensor _out = torch::zeros({B, X, Y, Z, C}, options);
  float* out = _out.data_ptr<float>();

  voxel_pool(B, X, Y, Z, N, C, n_intervals,
    feats, coords, interval_starts, interval_lengths, out
  );
  
  return _out;
}

at::Tensor voxel_pool_backward(const at::Tensor _out_grad, const at::Tensor _coords, const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts, int B, int X, int Y, int Z
) {
  int N = _coords.size(0);
  int C = _out_grad.size(-1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* coords = _coords.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _feats_grad = torch::zeros({N, C}, options);
  float* feats_grad = _feats_grad.data_ptr<float>();

  voxel_pool_grad(
    B, X, Y, Z, N, C, n_intervals, out_grad,
    coords, interval_starts, interval_lengths, feats_grad
  );
  
  return _feats_grad;
}

at::Tensor voxel_pool_depth_forward_all(
  const at::Tensor _img_feats, const at::Tensor _img_depth, const at::Tensor _coords,
  const at::Tensor _point_indices, const at::Tensor _sort_indices,
  const at::Tensor _interval_lengths, const at::Tensor _interval_starts,
  int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W
) {
  int n_valid = _coords.size(0);
  int feat_channels = _img_feats.size(2);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_img_feats));

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_img_feats.device());
  at::Tensor _out = torch::zeros({B, N_cam, X, Y, Z, feat_channels}, options);
  if (n_valid == 0 || n_intervals == 0) {
    return _out;
  }

  const int* coords = _coords.data_ptr<int>();
  const int* point_indices = _point_indices.data_ptr<int>();
  const int* sort_indices = _sort_indices.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  float* out = _out.data_ptr<float>();

#define LAUNCH_VOXEL_POOL_DEPTH_FORWARD_ALL(FEAT_T, DEPTH_T) \
  voxel_pool_depth_forward_all_cuda<FEAT_T, DEPTH_T>( \
    B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels, n_intervals, \
    _img_feats.data_ptr<FEAT_T>(), _img_depth.data_ptr<DEPTH_T>(), \
    coords, point_indices, sort_indices, interval_starts, interval_lengths, out)

#define DISPATCH_VOXEL_POOL_DEPTH_FORWARD_ALL_DEPTH(FEAT_T) \
  switch (_img_depth.scalar_type()) { \
    case at::ScalarType::Float: LAUNCH_VOXEL_POOL_DEPTH_FORWARD_ALL(FEAT_T, float); break; \
    case at::ScalarType::Half: LAUNCH_VOXEL_POOL_DEPTH_FORWARD_ALL(FEAT_T, c10::Half); break; \
    case at::ScalarType::BFloat16: LAUNCH_VOXEL_POOL_DEPTH_FORWARD_ALL(FEAT_T, c10::BFloat16); break; \
    default: TORCH_CHECK(false, "voxel_pool_depth_forward_all: unsupported depth dtype ", _img_depth.scalar_type()); \
  }

  switch (_img_feats.scalar_type()) {
    case at::ScalarType::Float: DISPATCH_VOXEL_POOL_DEPTH_FORWARD_ALL_DEPTH(float); break;
    case at::ScalarType::Half: DISPATCH_VOXEL_POOL_DEPTH_FORWARD_ALL_DEPTH(c10::Half); break;
    case at::ScalarType::BFloat16: DISPATCH_VOXEL_POOL_DEPTH_FORWARD_ALL_DEPTH(c10::BFloat16); break;
    default: TORCH_CHECK(false, "voxel_pool_depth_forward_all: unsupported feature dtype ", _img_feats.scalar_type());
  }

#undef DISPATCH_VOXEL_POOL_DEPTH_FORWARD_ALL_DEPTH
#undef LAUNCH_VOXEL_POOL_DEPTH_FORWARD_ALL

  return _out;
}

std::vector<at::Tensor> voxel_pool_depth_backward_all(
  const at::Tensor _out_grad, const at::Tensor _img_feats, const at::Tensor _img_depth,
  const at::Tensor _coords, const at::Tensor _point_indices,
  int B, int N_sweep, int N_cam, int X, int Y, int Z, int D, int H, int W
) {
  int n_valid = _coords.size(0);
  int feat_channels = _img_feats.size(2);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_img_feats));

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_img_feats.device());
  at::Tensor _img_feats_grad = torch::zeros(_img_feats.sizes(), options);
  at::Tensor _img_depth_grad = torch::zeros(_img_depth.sizes(), options);
  if (n_valid == 0) {
    return {_img_feats_grad, _img_depth_grad};
  }

  const int* coords = _coords.data_ptr<int>();
  const int* point_indices = _point_indices.data_ptr<int>();
  float* img_feats_grad = _img_feats_grad.data_ptr<float>();
  float* img_depth_grad = _img_depth_grad.data_ptr<float>();

#define LAUNCH_VOXEL_POOL_DEPTH_BACKWARD_ALL(FEAT_T, DEPTH_T, GRAD_T) \
  voxel_pool_depth_backward_all_cuda<FEAT_T, DEPTH_T, GRAD_T>( \
    B, N_sweep, N_cam, X, Y, Z, D, H, W, n_valid, feat_channels, \
    _out_grad.data_ptr<GRAD_T>(), _img_feats.data_ptr<FEAT_T>(), _img_depth.data_ptr<DEPTH_T>(), \
    coords, point_indices, img_feats_grad, img_depth_grad)

#define DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_GRAD(FEAT_T, DEPTH_T) \
  switch (_out_grad.scalar_type()) { \
    case at::ScalarType::Float: LAUNCH_VOXEL_POOL_DEPTH_BACKWARD_ALL(FEAT_T, DEPTH_T, float); break; \
    case at::ScalarType::Half: LAUNCH_VOXEL_POOL_DEPTH_BACKWARD_ALL(FEAT_T, DEPTH_T, c10::Half); break; \
    case at::ScalarType::BFloat16: LAUNCH_VOXEL_POOL_DEPTH_BACKWARD_ALL(FEAT_T, DEPTH_T, c10::BFloat16); break; \
    default: TORCH_CHECK(false, "voxel_pool_depth_backward_all: unsupported grad dtype ", _out_grad.scalar_type()); \
  }

#define DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_DEPTH(FEAT_T) \
  switch (_img_depth.scalar_type()) { \
    case at::ScalarType::Float: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_GRAD(FEAT_T, float); break; \
    case at::ScalarType::Half: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_GRAD(FEAT_T, c10::Half); break; \
    case at::ScalarType::BFloat16: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_GRAD(FEAT_T, c10::BFloat16); break; \
    default: TORCH_CHECK(false, "voxel_pool_depth_backward_all: unsupported depth dtype ", _img_depth.scalar_type()); \
  }

  switch (_img_feats.scalar_type()) {
    case at::ScalarType::Float: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_DEPTH(float); break;
    case at::ScalarType::Half: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_DEPTH(c10::Half); break;
    case at::ScalarType::BFloat16: DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_DEPTH(c10::BFloat16); break;
    default: TORCH_CHECK(false, "voxel_pool_depth_backward_all: unsupported feature dtype ", _img_feats.scalar_type());
  }

#undef DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_DEPTH
#undef DISPATCH_VOXEL_POOL_DEPTH_BACKWARD_ALL_GRAD
#undef LAUNCH_VOXEL_POOL_DEPTH_BACKWARD_ALL

  return {_img_feats_grad, _img_depth_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_pool_forward", &voxel_pool_forward, "voxel_pool_forward");
  m.def("voxel_pool_backward", &voxel_pool_backward, "voxel_pool_backward");
  m.def("voxel_pool_depth_forward_all", &voxel_pool_depth_forward_all, "voxel_pool_depth_forward_all");
  m.def("voxel_pool_depth_backward_all", &voxel_pool_depth_backward_all, "voxel_pool_depth_backward_all");
}
