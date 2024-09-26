// @lint-ignore-every LICENSELINT

// Modified from
// https://github.com/lilanxiao/Rotated_IoU/blob/master/cuda_op/sort_vert.h
// License https://github.com/lilanxiao/Rotated_IoU/blob/master/LICENSE

#pragma once
#include <torch/extension.h> // @manual=//caffe2:torch-cpp

#define MAX_NUM_VERT_IDX 9

at::Tensor
sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid);
