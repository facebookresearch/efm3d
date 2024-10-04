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

from typing import Literal

import torch

from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_DISTANCE_M,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.utils.depth import dist_im_to_point_cloud_im
from efm3d.utils.detection_utils import compute_focal_loss
from efm3d.utils.pointcloud import (
    pointcloud_occupancy_samples,
    pointcloud_to_occupancy_snippet,
    pointcloud_to_voxel_ids,
)
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from einops import rearrange
from torch.nn import functional as F


def build_gt_occupancy(occ, visible, p3s_w, Ts_wc, cams, T_wv, voxel_extent):
    """
    build GT occupancy from GT point cloud, return batched occupancy with masks.
    """
    B, vD, vH, vW = occ.shape
    occ_gts, masks = [], []
    for b in range(B):
        occ_gt, mask = pointcloud_to_occupancy_snippet(
            p3s_w[b],
            Ts_wc[b],
            cams[b],
            T_wv[b],
            vW,
            vH,
            vD,
            voxel_extent,
            S=1,
        )
        mask = torch.logical_and(mask.bool(), visible[b])
        occ_gts.append(occ_gt)
        masks.append(mask)
    occ_gts = torch.stack(occ_gts)
    masks = torch.stack(masks)
    return occ_gts, masks


def get_fused_gt_feat(
    visible,
    p3s_w,
    Ts_wc,
    cams,
    T_wv,
    voxel_extent,
    img_feat_gt,
    feat_pred,
    gt_dists,
    vD,
    vH,
    vW,
):
    feat_dim = img_feat_gt.shape[2]
    gt_feat_volume = torch.zeros_like(feat_pred).detach()  # BxCxDxHxW
    gt_feat_volume = gt_feat_volume.permute(
        0, 2, 3, 4, 1
    )  # BxDxHxWxC for easier indexing
    gt_feat_volume_counts = (
        torch.zeros(*gt_feat_volume.shape[:4]).to(feat_pred).detach()
    )  # BxDxHxW

    dists = gt_dists.squeeze(2)
    B, T = cams.shape[:2]
    pc_c, valids = dist_im_to_point_cloud_im(dists, cams)
    pc_c = pc_c.reshape(B, T, -1, 3)  # BxTxNx3
    T_vc = T_wv.inverse() @ Ts_wc
    pc_v = T_vc * pc_c
    for b in range(B):
        pc_ids, valid_v = pointcloud_to_voxel_ids(pc_v[b], vW, vH, vD, voxel_extent)
        for t in range(T):
            valid = torch.logical_and(valid_v[t], valids[b, t].reshape(-1))
            pc_ids_t = pc_ids[t][valid]
            feat_gt_t = img_feat_gt[b, t].permute(1, 2, 0).reshape(-1, feat_dim)
            feat_gt_t = feat_gt_t[valid]
            gt_feat_volume_counts[b][
                pc_ids_t[:, 0], pc_ids_t[:, 1], pc_ids_t[:, 2]
            ] += 1.0
            gt_feat_volume[b][pc_ids_t[:, 0], pc_ids_t[:, 1], pc_ids_t[:, 2]] += (
                feat_gt_t
            )
        gt_feat_volume[b][gt_feat_volume_counts[b] > 1e-4] /= gt_feat_volume_counts[b][
            gt_feat_volume_counts[b] > 1e-4
        ].unsqueeze(-1)
    surface_mask = gt_feat_volume_counts > 1e-4  # BxDxHxW
    return gt_feat_volume, surface_mask


def get_feats_world(batch, tgt_feats):
    B = tgt_feats.shape[0]
    tgt_H, tgt_W = tgt_feats.shape[-2], tgt_feats.shape[-1]
    dists_ori = batch[ARIA_DISTANCE_M[0]]
    cams_ori = batch[ARIA_CALIB[0]]
    # rescale dist and camera to tgt feat size
    dists = rearrange(dists_ori, "b t c h w -> (b t) c h w")
    dists = F.interpolate(dists, [tgt_H, tgt_W], mode="nearest")
    dists = rearrange(dists, "(b t) c h w -> b t c h w", b=B).squeeze(2)
    cams = cams_ori.scale_to_size((tgt_W, tgt_H))

    B, T = cams.shape[:2]
    Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]]
    T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET]
    Ts_wr = T_ws @ Ts_sr
    Ts_cw = cams.T_camera_rig @ Ts_wr.inverse()
    Ts_wc = Ts_cw.inverse()
    pc_c, valids = dist_im_to_point_cloud_im(dists, cams)
    B, T, H, W = pc_c.shape[:4]
    pc_w = Ts_wc * pc_c.view(B, T, -1, 3)
    pc_w = pc_w.view(B, T, H, W, 3)
    pc_w[~valids] = float("nan")  # nan
    # remove all points that are invalid across all time and batches.
    all_valid = ~(~valids).all(0).all(0)
    all_valid = all_valid.view(1, 1, H, W).repeat(B, T, 1, 1)
    pc_w = pc_w[all_valid].view(B, T, -1, 3)
    feat_dim = tgt_feats.shape[2]
    tgt_feats = tgt_feats.permute(0, 1, 3, 4, 2)
    tgt_feats = tgt_feats[all_valid].view(B, T, -1, feat_dim)

    return pc_w, tgt_feats


def compute_tv_loss(occ):
    # B 1 D H W
    tv_d = (occ[:, 1:, :, :] - occ[:, :-1, :, :]).abs().mean()
    tv_h = (occ[:, :, 1:, :] - occ[:, :, :-1, :]).abs().mean()
    tv_w = (occ[:, :, :, 1:] - occ[:, :, :, :-1]).abs().mean()
    tv_loss = tv_d + tv_h + tv_w
    return tv_loss


def compute_occupancy_loss_subvoxel(
    occ,
    visible,
    p3s_w_all,
    Ts_wc,
    cams,
    T_wv,
    voxel_extent,
    S=1,
    sample_beyond=False,
    surf_val=0.5,
    subsample=1,
    free_surf_occ_weights=None,
    loss_type: Literal["l2", "l1", "logl1", "ce", "focal"] = "focal",
):
    """
    sample occupied, surface and freespace GT points
    obtain predictions at those sample points by sampling into the occ voxel
    grid via tri-linear interpolation.
    """
    assert p3s_w_all.ndim == 4, f"{p3s_w_all.shape}"  # B T N 3
    assert occ.ndim == 4, f"{occ.shape}"  # B D H W
    assert visible.ndim == 4, f"{visible.shape}"  # B D H W
    assert not sample_beyond, "not supported"
    device = occ.device
    B, vD, vH, vW = occ.shape

    if subsample > 1:
        # subsample
        B, T, N = p3s_w_all.shape[:3]
        ids = torch.randperm(N)[: N // subsample].to(device)
        p3s_w = p3s_w_all[:, :, ids]
        # print("subsample", subsample, p3s_w.shape, p3s_w_all.shape)
    else:
        p3s_w = p3s_w_all
    B, T, N = p3s_w.shape[:3]

    p3s_occ_w, p3s_surf_w, p3s_free_w, valid = pointcloud_occupancy_samples(
        p3s_w,
        Ts_wc,
        cams,
        vD,
        vH,
        vW,
        voxel_extent,
        S=S,
        sample_beyond=sample_beyond,
        vox_diag_scale=1.0,
        T_wv=T_wv,
    )
    Ts_vw = T_wv.inverse().view(B, 1, -1).repeat(1, T, 1)

    p3s_occ_v = Ts_vw * p3s_occ_w
    p3s_surf_v = Ts_vw * p3s_surf_w
    p3s_free_v = Ts_vw * p3s_free_w

    B, vD, vH, vW = occ.shape
    # free points
    p3s_free_vox, valid_free = pc_to_vox(p3s_free_v, vW, vH, vD, voxel_extent)
    valid_free = torch.logical_and(valid_free, valid)
    free_samples, valid_samples = sample_voxels(
        occ.unsqueeze(1), p3s_free_vox.view(B, -1, 3)
    )
    free_samples, valid_samples = (
        free_samples.view(B, T, -1),
        valid_samples.view(B, T, -1),
    )
    valid_free = torch.logical_and(valid_samples, valid_free)
    free_samples = free_samples[valid_free].clamp(0.0, 1.0)
    free_gt = torch.zeros_like(free_samples)

    # surface points
    p3s_surf_vox, valid_surf = pc_to_vox(p3s_surf_v, vW, vH, vD, voxel_extent)
    valid_surf = torch.logical_and(valid_surf, valid)
    surf_samples, valid_samples = sample_voxels(
        occ.unsqueeze(1), p3s_surf_vox.view(B, -1, 3)
    )
    surf_samples, valid_samples = (
        surf_samples.view(B, T, -1),
        valid_samples.view(B, T, -1),
    )
    valid_surf = torch.logical_and(valid_samples, valid_surf)
    surf_samples = surf_samples[valid_surf].clamp(0.0, 1.0)
    surf_gt = surf_val * torch.ones_like(surf_samples)

    # occupied points
    p3s_occ_vox, valid_occ = pc_to_vox(p3s_occ_v, vW, vH, vD, voxel_extent)
    valid_occ = torch.logical_and(valid_occ, valid)
    occ_samples, valid_samples = sample_voxels(
        occ.unsqueeze(1), p3s_occ_vox.view(B, -1, 3)
    )
    occ_samples, valid_samples = (
        occ_samples.view(B, T, -1),
        valid_samples.view(B, T, -1),
    )
    valid_occ = torch.logical_and(valid_samples, valid_occ)
    occ_samples = occ_samples[valid_occ].clamp(0.0, 1.0)
    occ_gt = torch.ones_like(occ_samples)

    if free_surf_occ_weights is None:
        num = free_samples.numel() + surf_samples.numel() + occ_samples.numel()
        if loss_type == "l2":
            # L2 loss
            pred = torch.cat([free_samples, surf_samples, occ_samples], -1)
            gt = torch.cat([free_gt, surf_gt, occ_gt], -1)
            loss = ((pred - gt) ** 2).sum()
        elif loss_type == "l1":
            # L1 loss
            pred = torch.cat([free_samples, surf_samples, occ_samples], -1)
            gt = torch.cat([free_gt, surf_gt, occ_gt], -1)
            loss = (pred - gt).abs().sum()
        elif loss_type == "logl1":
            # logl1 loss
            pred = torch.cat([free_samples, surf_samples, occ_samples], -1)
            gt = torch.cat([free_gt, surf_gt, occ_gt], -1)
            loss = (torch.log(pred + 1e-5) - torch.log(gt + 1e-5)).abs().sum()
        elif loss_type == "ce":
            # CE on free and occ and L1 on surf
            pred = torch.cat([free_samples, occ_samples], -1)
            gt = torch.cat([free_gt, occ_gt], -1)
            loss = F.binary_cross_entropy(pred, gt, reduction="sum")
            loss = loss + (surf_samples - surf_gt).abs().sum()
        elif loss_type == "focal":
            # like used to using focal loss
            pred = torch.cat([free_samples, surf_samples, occ_samples], -1)
            gt = torch.cat([free_gt, surf_gt, occ_gt], -1)
            loss = compute_focal_loss(pred, gt)
        assert (
            not loss.isnan().any()
        ), f"have nans in loss {loss.isnan().count_nonzero()}"
        # handle no samples case in mean
        num = max(1.0, num)
        loss = loss.sum() / num
        return loss

    assert loss_type == "focal", f"{loss_type} not supported"
    loss_free = compute_focal_loss(free_samples, free_gt).sum()
    loss_surf = compute_focal_loss(surf_samples, surf_gt).sum()
    loss_occ = compute_focal_loss(occ_samples, occ_gt).sum()

    loss_free = loss_free / max(1.0, loss_free.numel())
    loss_surf = loss_surf / max(1.0, loss_surf.numel())
    loss_occ = loss_occ / max(1.0, loss_occ.numel())

    loss = loss_free * free_surf_occ_weights[0] + loss_occ * free_surf_occ_weights[2]
    return loss
