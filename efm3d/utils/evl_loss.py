# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_OBB_PADDED,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.aria.obb import obb_filter_outside_volume, obb_time_union, ObbTW, PAD_VAL
from efm3d.thirdparty.mmdetection3d.iou3d import RotatedIoU3DLoss
from efm3d.utils.detection_utils import (
    compute_chamfer_loss,
    compute_focal_loss,
    obb2voxel,
    voxel2obb,
)
from efm3d.utils.pointcloud import get_points_world
from efm3d.utils.reconstruction import compute_occupancy_loss_subvoxel, compute_tv_loss


def get_gt_obbs(batch, voxel_extent, T_wv=None):
    """
    Get the GT Obbs from the batch.

    voxel_extent: used to filter GT Obbs outside of voxel grid.
    T_wv: if not None, filter GT Obbs outside of voxel grid.
    """
    if ARIA_OBB_PADDED not in batch:
        B = batch[ARIA_SNIPPET_T_WORLD_SNIPPET].shape[0]
        return ObbTW().view(1, -1).repeat(B, 1)
    obbs_gt = batch[ARIA_OBB_PADDED].clone()
    # Optionally filter GT.
    if batch[ARIA_OBB_PADDED].ndim == 4:
        # Filter by time Union.
        obbs_gt = obb_time_union(obbs_gt)
    if T_wv is not None:
        # Filter outside of voxel grid.
        T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET].squeeze(1)
        obbs_gt = obb_filter_outside_volume(
            obbs_gt, T_ws, T_wv, voxel_extent=voxel_extent
        )
    return obbs_gt


def obbs_to_7d(obbs):
    obbs_cent = obbs.bb3_center_world  # center in voxel coords
    wlh = obbs.bb3_max_object - obbs.bb3_min_object

    # Get gravity aligned rotation from obb
    T_voxel_object = obbs.T_world_object.clone()
    # HACK to avoid gimbal lock for padded entries.
    mask = obbs.get_padding_mask()
    T_voxel_object.R[mask] = PAD_VAL
    rpy = T_voxel_object.to_euler()
    yaw = rpy[..., 2].unsqueeze(-1)

    obbs_7d = torch.concat([obbs_cent, wlh, yaw], dim=-1)
    return obbs_7d


def iou_3d_loss(obbs_pr, obbs_gt, cent_pr, cent_gt, valid_gt):
    """
    obbs_pr: N x 34
    obbs_gt: N x 34
    """
    assert obbs_pr.ndim == 2 and obbs_gt.ndim == 2, "obbs dimension should be Nx34"
    obbs_pr_7d = obbs_to_7d(obbs_pr)
    obbs_gt_7d = obbs_to_7d(obbs_gt)
    iou_loss = RotatedIoU3DLoss(loss_weight=1.0)

    # weighted by validness and GT centerness
    obbs_weight = cent_gt.reshape(-1) * valid_gt.reshape(-1)
    valid_idx = torch.nonzero(obbs_weight > 0).squeeze()
    obbs_weight = obbs_weight[valid_idx]
    obbs_pr_7d = obbs_pr_7d[valid_idx, :]
    obbs_gt_7d = obbs_gt_7d[valid_idx, :]

    loss = obbs_weight * iou_loss.forward(obbs_pr_7d, obbs_gt_7d)
    loss = loss.mean()

    return loss


def compute_obb_losses(
    outputs,
    batch,
    voxel_extent,
    num_class,
    splat_sigma,
    cent_weight,
    clas_weight,
    iou_weight,
    bbox_weight,
    cham_weight,
):
    B, _, vD, vH, vW = outputs["cent_pr"].shape
    ve = voxel_extent
    N = vD * vH * vW
    T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET].squeeze(1)
    T_wv = outputs["voxel/T_world_voxel"]
    obb_gt_s = get_gt_obbs(batch, voxel_extent, T_wv)

    # Put GT in voxel coordinate frame.
    T_vs = T_wv.inverse() @ T_ws
    obb_gt_v = obb_gt_s.transform(T_vs.unsqueeze(1))

    # Create 3D GT tensors.
    cent_gt, bbox_gt, clas_gt, valid_gt = obb2voxel(
        obb_gt_v, vD, vH, vW, ve, num_class, splat_sigma
    )
    outputs["cent_gt"] = cent_gt
    outputs["bbox_gt"] = bbox_gt
    outputs["clas_gt"] = clas_gt
    outputs["valid_gt"] = valid_gt

    # Get Obbs from densified predictions + GT.
    cent_pr = outputs["cent_pr"]
    bbox_pr = outputs["bbox_pr"]
    clas_pr = outputs["clas_pr"]
    obbs_pr_dense = voxel2obb(cent_pr, bbox_pr, clas_pr, ve, top_k=None, thresh=None)
    obbs_gt_dense = voxel2obb(cent_gt, bbox_gt, clas_gt, ve, top_k=None, thresh=None)
    obbs_pr_dense = obbs_pr_dense.reshape(B * N, -1)
    obbs_gt_dense = obbs_gt_dense.reshape(B * N, -1)
    outputs["obbs_gt_dense"] = obbs_gt_dense

    losses = {"rgb": {}}
    total_loss = 0.0

    # Centerness loss.
    if cent_weight > 0:
        cent_pr = outputs["cent_pr"]
        cent_loss = compute_focal_loss(cent_pr, cent_gt)
        cent_loss = cent_loss.reshape(B, -1)
        cent_loss = cent_loss.mean()
        cent_loss = cent_loss * cent_weight
        losses["rgb"]["cent"] = cent_loss
        total_loss += cent_loss

    # Classification loss.
    if clas_weight > 0:
        clas_pr = outputs["clas_pr"]
        clas_loss = compute_focal_loss(clas_pr, clas_gt)
        clas_loss = clas_loss.sum(dim=1).reshape(-1)
        clas_loss[~valid_gt.reshape(-1)] = 0.0
        clas_loss = torch.sum(clas_loss) / (valid_gt.sum() + 1)
        clas_loss = clas_loss * clas_weight
        losses["rgb"]["clas"] = clas_loss
        total_loss += clas_loss

    # 3D IoU loss (gravity aligned 7 DoF loss).
    if iou_weight > 0:
        iou_loss = iou_3d_loss(obbs_pr_dense, obbs_gt_dense, cent_pr, cent_gt, valid_gt)
        iou_loss = iou_loss * iou_weight
        losses["rgb"]["iou"] = iou_loss
        total_loss += iou_loss

    # Supervise directly on D, H, W dimensions with L1 loss.
    if bbox_weight > 0:
        dhw_gt = obbs_gt_dense.bb3_diagonal
        dhw_pr = obbs_pr_dense.bb3_diagonal
        bbox_loss = torch.mean(torch.abs(dhw_pr - dhw_gt), dim=-1)
        bbox_loss[~valid_gt.reshape(-1)] = 0.0
        bbox_loss = torch.sum(bbox_loss) / (valid_gt.sum() + 1)
        bbox_loss = bbox_loss * bbox_weight
        losses["rgb"]["bbox"] = bbox_loss
        total_loss += bbox_loss

    # Chamfer loss for rotation.
    if cham_weight > 0:
        corners_pr = obbs_pr_dense.bb3corners_world  # world is voxel
        corners_gt = obbs_gt_dense.bb3corners_world  # world is voxel
        cham_loss = compute_chamfer_loss(corners_pr, corners_gt)
        cham_loss[~valid_gt.reshape(-1)] = 0.0
        cham_loss = torch.sum(cham_loss) / (valid_gt.sum() + 1)
        cham_loss = cham_loss * cham_weight
        losses["rgb"]["cham"] = cham_loss
        total_loss += cham_loss

    return losses, total_loss


def compute_occ_losses(
    outputs,
    batch,
    voxel_extent,
    occ_weight,
    tv_weight,
):
    B, T, vD, vH, vW = outputs["occ_pr"].shape
    T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET].squeeze(1)
    T_wv = outputs["voxel/T_world_voxel"]

    losses = {"rgb": {}}
    total_loss = 0.0
    p3s_w, dist_stds = get_points_world(batch)

    # Occupancy loss.
    cams = batch[ARIA_CALIB[0]]
    Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]]
    T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET]
    Ts_wr = T_ws @ Ts_sr
    Ts_cw = cams.T_camera_rig @ Ts_wr.inverse()
    Ts_wc = Ts_cw.inverse()

    occ = outputs["occ_pr"].squeeze(1)
    voxel_counts = outputs["voxel/counts"]

    B, D, H, W = occ.shape
    B, Df, Hf, Wf = voxel_counts.shape
    if D != Df or H != Hf or W != Wf:
        resize = torch.nn.Upsample(size=(D, H, W))
        voxel_counts = resize(voxel_counts.unsqueeze(1).float()).squeeze(1)

    visible = voxel_counts > 0

    if occ_weight > 0:
        occ_loss = compute_occupancy_loss_subvoxel(
            occ,
            visible,
            p3s_w,
            Ts_wc,
            cams,
            T_wv,
            voxel_extent,
            loss_type="l2",
        )
        occ_loss = occ_loss * occ_weight
        total_loss += occ_loss
        losses["rgb"]["occ"] = occ_loss.cpu().detach()

    if tv_weight > 0.0:
        tv_loss = compute_tv_loss(occ)
        tv_loss = tv_loss * tv_weight
        total_loss += tv_loss
        losses["rgb"]["tv"] = tv_loss.cpu().detach()

    return losses, total_loss
