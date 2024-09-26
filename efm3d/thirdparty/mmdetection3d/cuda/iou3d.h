// @lint-ignore-every LICENSELINT

// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.h
// License under Apache 2.0
// https://github.com/open-mmlab/OpenPCDet/blob/master/LICENSE

#pragma once
#include <torch/extension.h> // @manual=//caffe2:torch-cpp

int boxes_overlap_bev_gpu(
    at::Tensor boxes_a,
    at::Tensor boxes_b,
    at::Tensor ans_overlap);

int boxes_iou_bev_gpu(
    at::Tensor boxes_a,
    at::Tensor boxes_b,
    at::Tensor ans_iou);

int nms_gpu(
    at::Tensor boxes,
    at::Tensor keep,
    float nms_overlap_thresh,
    int device_id);

int nms_normal_gpu(
    at::Tensor boxes,
    at::Tensor keep,
    float nms_overlap_thresh,
    int device_id);
