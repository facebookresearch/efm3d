// @lint-ignore-every LICENSELINT

// Modified from
// https://github.com/lilanxiao/Rotated_IoU/blob/master/cuda_op/utils.h
// License https://github.com/lilanxiao/Rotated_IoU/blob/master/LICENSE

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h> // @manual=//caffe2:torch-cpp

#define CHECK_CUDA(x)                                      \
  do {                                                     \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                            \
  do {                                                                 \
    TORCH_CHECK(x.is_contiguous(), #x " must ne a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                                                      \
  do {                                                                       \
    TORCH_CHECK(                                                             \
        x.scalar_type() == at::ScalarType::Int, #x " must be a int tensor"); \
  } while (0)

#define CHECK_IS_FLOAT(x)                         \
  do {                                            \
    TORCH_CHECK(                                  \
        x.scalar_type() == at::ScalarType::Float, \
        #x " must be a float tensor");            \
  } while (0)

#define CHECK_IS_BOOL(x)                                                       \
  do {                                                                         \
    TORCH_CHECK(                                                               \
        x.scalar_type() == at::ScalarType::Bool, #x " must be a bool tensor"); \
  } while (0)
