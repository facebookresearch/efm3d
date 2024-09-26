// @lint-ignore-every LICENSELINT

// Modified from
// https://github.com/lilanxiao/Rotated_IoU/blob/master/cuda_op/sort_vert.cpp
// License https://github.com/lilanxiao/Rotated_IoU/blob/master/LICENSE

#include "sort_vert.h"
#include "iou3d.h"
#include "utils.h"

void sort_vertices_wrapper(
    int b,
    int n,
    int m,
    const float* vertices,
    const bool* mask,
    const int* num_valid,
    int* idx);

at::Tensor
sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid) {
  CHECK_CONTIGUOUS(vertices);
  CHECK_CONTIGUOUS(mask);
  CHECK_CONTIGUOUS(num_valid);
  CHECK_CUDA(vertices);
  CHECK_CUDA(mask);
  CHECK_CUDA(num_valid);
  CHECK_IS_FLOAT(vertices);
  CHECK_IS_BOOL(mask);
  CHECK_IS_INT(num_valid);

  int b = vertices.size(0);
  int n = vertices.size(1);
  int m = vertices.size(2);
  at::Tensor idx = torch::zeros(
      {b, n, MAX_NUM_VERT_IDX},
      at::device(vertices.device()).dtype(at::ScalarType::Int));

  // fix issue with multi-gpu (kernel only works for cuda:0)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(idx));

  sort_vertices_wrapper(
      b,
      n,
      m,
      vertices.data_ptr<float>(),
      mask.data_ptr<bool>(),
      num_valid.data_ptr<int>(),
      idx.data_ptr<int>());

  return idx;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sort_vertices_forward",
      &sort_vertices,
      "sort vertices of a convex polygon. forward only");
  m.def(
      "boxes_overlap_bev_gpu",
      &boxes_overlap_bev_gpu,
      "oriented boxes overlap");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
  m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
  m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
}
