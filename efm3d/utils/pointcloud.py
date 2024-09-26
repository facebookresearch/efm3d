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

import math

import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_DISTANCE_M,
    ARIA_IMG,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_WORLD,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.utils.depth import dist_im_to_point_cloud_im
from efm3d.utils.ray import sample_depths_in_grid, transform_rays
from efm3d.utils.voxel import tensor_wrap_voxel_extent
from torch.nn import functional as F


def get_points_world(batch, batch_idx=None, dist_std0=0.04, prefer_points=False):
    if ARIA_DISTANCE_M[0] in batch and not prefer_points:
        dists = batch[ARIA_DISTANCE_M[0]].squeeze(2)
        cams = batch[ARIA_CALIB[0]]
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

        dist_stds = torch.ones(pc_w.shape[:-1], device=pc_w.device) * dist_std0
    elif ARIA_POINTS_WORLD in batch:
        pc_w = batch[ARIA_POINTS_WORLD]

        if ARIA_POINTS_DIST_STD in batch:
            dist_stds = batch[ARIA_POINTS_DIST_STD]
        else:
            dist_stds = torch.ones(pc_w.shape[:-1], device=pc_w.device) * 0.01

    else:
        raise NotImplementedError(
            f"do need either points or depth image! {batch.keys()}"
        )

    if batch_idx is not None:
        return pc_w[batch_idx], dist_stds[batch_idx]
    return pc_w, dist_stds


def get_freespace_world(
    batch,
    batch_idx,
    T_wv,
    vW,
    vH,
    vD,
    voxel_extent,
    S=1,
    prefer_points=False,
    dropout_points=False,
    drop_points_rate_max=0.5,
):
    """
    Get points (semi-dense or GT points) of a snippet in the batch.
    """
    cams = batch[ARIA_CALIB[0]][batch_idx]
    T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET][
        batch_idx
    ]  # T_world_rig (one per snippet)
    Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]][
        batch_idx
    ]  # Ts_snippet_rig (T per snippet)
    Ts_wr = T_ws @ Ts_sr
    Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()  # Ts_world_cam

    # compute rays and max depths
    if ARIA_DISTANCE_M[0] in batch and not prefer_points:
        # get gt distances into world points
        gt_dist = batch[ARIA_DISTANCE_M[0]][batch_idx]
        cams = batch[ARIA_CALIB[0]][batch_idx]
        # invalid depth has values 0 or NaN (padding used by semidense stream).
        valid_depths = gt_dist.squeeze(1) > 1e-4
        p3cs, valids = dist_im_to_point_cloud_im(
            gt_dist.squeeze(1),
            cams,
        )
        valids = torch.logical_and(valids, valid_depths)
        p3cs = p3cs.reshape(p3cs.shape[0], -1, 3)
        T, N = p3cs.shape[:2]
        ds = torch.norm(p3cs, 2.0, dim=-1)
        dirs_c = F.normalize(p3cs, 2.0, dim=-1)
        rays_c = torch.cat([torch.zeros_like(dirs_c), dirs_c], dim=-1)
        T_vc = T_wv.inverse() @ Ts_wc
        rays_v = transform_rays(rays_c, T_vc)
        rays_v = rays_v.view(-1, 6)
        ds = ds.view(-1)
        valids = valids.reshape(-1)
        rays_v = rays_v[valids]
        ds = ds[valids]
    else:
        p_w = batch[ARIA_POINTS_WORLD][batch_idx]  # TxNx3
        T, N = p_w.shape[:2]
        p0_w = Ts_wc.t.unsqueeze(1)  # Tx1x3
        diff_w = p_w - p0_w
        ds = torch.norm(diff_w, 2.0, dim=-1)
        dir_w = F.normalize(diff_w, 2.0, dim=-1)
        # filter out nans
        good = ~p_w.isnan().any(dim=-1)
        p0_w = p0_w.repeat(1, N, 1)[good]
        ds = ds[good]
        dir_w = dir_w[good]
        rays_w = torch.cat([p0_w, dir_w], dim=-1)
        rays_v = transform_rays(rays_w, T_wv.inverse())

    # dropout rays if desired
    if dropout_points:
        N = rays_v.shape[0]
        p = drop_points_rate_max
        Ndrop = int(N * (torch.rand(1).item() * p + (1.0 - p)))
        print(f"dropout {Ndrop}/{N} points")
        rnd = torch.randperm(N, device=p_w.device)[:Ndrop]
        rays_v = rays_v[rnd, :]
        ds = ds[rnd]

    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
    dW = (x_max - x_min) / vW
    dH = (y_max - y_min) / vH
    dD = (z_max - z_min) / vD
    diag = math.sqrt(dW**2 + dH**2 + dD**2)
    # subtract diagonal of voxel size to not label the occupied voxel as free
    ds = ds - diag
    # sample depths that lie within the feature volume grid (same function as used for nerf3d!)
    depths, _, _ = sample_depths_in_grid(
        rays_v.view(1, 1, -1, 6),
        ds.view(1, 1, -1),
        voxel_extent,
        vW,
        vH,
        vD,
        S,
    )
    depths = depths.view(-1, S)
    rays_v = rays_v.view(-1, 1, 6)
    pts_v = rays_v[..., :3] + depths.unsqueeze(-1) * rays_v[..., 3:]
    pts_v = pts_v.view(-1, 3)
    return T_wv * pts_v


def collapse_pointcloud_time(pc_w):
    pc_w = pc_w.reshape(-1, 3)
    # filter out nans
    bad = pc_w.isnan().any(dim=-1)
    pc_w = pc_w[~bad]
    # filter out duplicates from the collapsing of the time dimension
    pc_w = torch.unique(pc_w, dim=0)
    pc_w = pc_w.reshape(-1, 3)
    return pc_w


def pointcloud_to_voxel_ids(pc_v, vW, vH, vD, voxel_extent):
    """
    converts a point cloud in voxel grid coordinates into voxel ids.
    """
    assert pc_v.ndim == 3, f"{pc_v.shape}"  # T N 3
    assert isinstance(voxel_extent, torch.Tensor)
    assert voxel_extent.ndim == 1, f"{voxel_extent.shape}"  # 6
    device = pc_v.device
    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent.tolist()
    valid = pc_v[..., 0] > x_min
    valid = torch.logical_and(pc_v[..., 0] < x_max, valid)
    valid = torch.logical_and(pc_v[..., 1] > y_min, valid)
    valid = torch.logical_and(pc_v[..., 1] < y_max, valid)
    valid = torch.logical_and(pc_v[..., 2] > z_min, valid)
    valid = torch.logical_and(pc_v[..., 2] < z_max, valid)
    dW = (x_max - x_min) / vW
    dH = (y_max - y_min) / vH
    dD = (z_max - z_min) / vD
    s = [1] * (pc_v.ndim - 1) + [3]
    dVox = torch.tensor([dW, dH, dD]).view(s).to(device)
    vox_min = torch.tensor([x_min, y_min, z_min]).view(s).to(device)
    pc_id = ((pc_v - vox_min) / dVox).floor().long()
    valid = torch.logical_and(pc_id[..., 0] >= 0, valid)
    valid = torch.logical_and(pc_id[..., 0] < vW, valid)
    valid = torch.logical_and(pc_id[..., 1] >= 0, valid)
    valid = torch.logical_and(pc_id[..., 1] < vH, valid)
    valid = torch.logical_and(pc_id[..., 2] >= 0, valid)
    valid = torch.logical_and(pc_id[..., 2] < vD, valid)
    # to match the D H W ordering of the voxel tensors
    pc_id = pc_id[..., [2, 1, 0]]
    return pc_id, valid


def pointcloud_to_occupancy_snippet(
    pcs_w, Ts_wc, cams, T_wv, vW, vH, vD, voxel_extent, S=1
):
    """
    converts a pointcloud to an occupancy grid (and mask where there are
    points).

    All voxels which have a point in them are marked occupied
    Along rays to the points of the cloud we sample S points and mark them as
    not occupied.
    """
    assert pcs_w.ndim == 3, f"{pcs_w.shape}"  # T N 3
    assert Ts_wc.ndim == 2, f"{Ts_wc.shape}"  # T C
    assert cams.ndim == 2, f"{cams.shape}"  # T C
    assert T_wv.ndim in [1, 2], f"{T_wv.shape}"  # 1 C
    voxel_extent = tensor_wrap_voxel_extent(voxel_extent)
    device = pcs_w.device
    occ = torch.zeros((vD, vH, vW), device=device)
    mask = torch.zeros_like(occ)

    # get invalid mask as the points that are nan and do not project into the
    # camera.
    Ts_vc = T_wv.inverse() @ Ts_wc
    pc_c = Ts_wc.inverse() * pcs_w
    invalid = pc_c.isnan().any(-1)  # T N
    pc_im, valid = cams.project(pc_c)
    invalid = torch.logical_or(invalid, ~valid)
    depth = torch.sqrt((pc_c**2).sum(-1))
    ray_c = pc_c / depth.unsqueeze(-1)

    # camera origins are not occupied
    rayP_c = torch.zeros_like(Ts_wc.t)
    rayP_v = Ts_vc * rayP_c
    pc_ids, valid = pointcloud_to_voxel_ids(rayP_v, vW, vH, vD, voxel_extent)
    pc_ids = pc_ids[valid]
    pc_ids = pc_ids.view(-1, 3)
    if pc_ids.numel() > 0:
        occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 0.0
        mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    # sample along the ray
    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
    dW = (x_max - x_min) / vW
    dH = (y_max - y_min) / vH
    dD = (z_max - z_min) / vD
    diag = math.sqrt(dW**2 + dH**2 + dD**2)
    T, N = ray_c.shape[:2]
    rayP_c = rayP_c.view(T, 1, 3).repeat(1, N, 1)
    # sample depths conservatively up to the depth - diagonal of a voxel
    ds = depth.unsqueeze(-1) - diag
    ds = torch.rand((T, N, S), device=device) * ds
    samples_c = rayP_c.unsqueeze(2) + ds.unsqueeze(3) * ray_c.unsqueeze(2)
    samples_c = samples_c.view(T, -1, 3)
    samples_v = Ts_vc * samples_c
    pc_ids, valid = pointcloud_to_voxel_ids(samples_v, vW, vH, vD, voxel_extent)
    invalid_ = invalid.unsqueeze(-1).repeat(1, 1, S).view(T, -1)
    valid = torch.logical_and(valid, ~invalid_)
    pc_ids = pc_ids[valid]
    pc_ids = pc_ids.view(-1, 3)
    if pc_ids.numel() > 0:
        occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 0.0
        mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    # add points as occupied
    pc_v = T_wv.inverse() * pcs_w
    pc_ids, valid = pointcloud_to_voxel_ids(pc_v, vW, vH, vD, voxel_extent)
    valid = torch.logical_and(valid, ~invalid)
    pc_ids = pc_ids[valid]
    pc_ids = pc_ids.view(-1, 3)
    if pc_ids.numel() > 0:
        occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0
        mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    return occ, mask


def pointcloud_occupancy_samples(
    p3s_w,
    Ts_wc,
    cams,
    vW,
    vH,
    vD,
    voxel_extent,
    S=16,
    sample_beyond=False,
    vox_diag_scale=1.0,
    T_wv=None,
    sample_mode="random",
):
    """
    compute occupied points and sample S freespace points along rays.
    """
    assert p3s_w.ndim == 4, f"{p3s_w.shape}"  # B T N 3
    assert Ts_wc.ndim == 3, f"{Ts_wc.shape}"  # B T C
    assert not sample_beyond, "not supported"
    B = p3s_w.shape[0]
    # precompute things
    pc_c = Ts_wc.inverse() * p3s_w
    invalid = pc_c.isnan().any(-1)  # B T N
    pc_im, valid = cams.project(pc_c)
    invalid = torch.logical_or(invalid, ~valid)
    depth = torch.sqrt((pc_c**2).sum(-1)).unsqueeze(-1)
    rayD_c = pc_c / depth
    B, T, N = rayD_c.shape[:3]
    rayP_c = torch.zeros_like(Ts_wc.t)
    rayP_c = rayP_c.view(B, T, 1, 3).repeat(1, 1, N, 1)
    T_vc = T_wv.inverse().unsqueeze(-2) @ Ts_wc
    voxel_extent = tensor_wrap_voxel_extent(voxel_extent, B, device=depth.device)
    diag = voxel_extent[..., 1::2] - voxel_extent[..., 0::2]
    diag = diag / torch.tensor([vW, vH, vD], device=voxel_extent.device)
    diag = torch.sqrt((diag**2).sum(-1)) * vox_diag_scale
    delta = diag.view(B, 1, 1, 1)
    ds_free_max = depth - delta  # BxTxNx1
    # sample depths conservatively up to the depth - diagonal of a voxel
    rays_c = torch.cat([rayP_c, rayD_c], dim=-1)
    rays_v = transform_rays(rays_c, T_vc)
    ds_free, _, _ = sample_depths_in_grid(
        rays_v,
        ds_free_max.squeeze(-1),
        voxel_extent,
        vW,
        vH,
        vD,
        S,
        d_near=0.01,
        d_far=10.0,
        sample_mode=sample_mode,
    )
    free_c = rayP_c.unsqueeze(3) + ds_free.unsqueeze(4) * rayD_c.unsqueeze(3)
    free_c = free_c.view(B, T, -1, 3)
    free_w = Ts_wc * free_c

    ds_occ = depth + delta
    occ_c = rayP_c + ds_occ * rayD_c
    occ_c = occ_c.view(B, T, -1, 3)
    occ_w = Ts_wc * occ_c
    # occupied, on surface, free space
    return occ_w, p3s_w, free_w, ~invalid


def pointcloud_to_occupancy(
    pc_w, T_wc, cam, T_wv, vW, vH, vD, voxel_extent, S=1, occ=None, mask=None
):

    device = pc_w.device
    if occ is None:
        occ = torch.zeros((vD, vH, vW), device=device)
    if mask is None:
        mask = torch.zeros_like(occ)

    T_vc = T_wv.inverse() @ T_wc
    pc_c = T_wc.inverse() * pc_w
    invalid = pc_c.isnan().any(-1)
    pc_c = pc_c[~invalid]
    pc_im, valid = cam.unsqueeze(0).project(pc_c.unsqueeze(0))
    pc_im, valid = pc_im.squeeze(0), valid.squeeze(0)
    depth = torch.sqrt((pc_c**2).sum(-1))
    ray_c = pc_c / depth.unsqueeze(-1)
    ray_c = ray_c[valid]
    depth = depth[valid]

    # camera origins are not occupied
    rayP_c = torch.zeros_like(T_wc.t)
    rayP_v = T_vc * rayP_c
    pc_ids, valid = pointcloud_to_voxel_ids(rayP_v, vW, vH, vD, voxel_extent)
    pc_ids = pc_ids[valid]
    occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 0.0
    mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    # sample along the ray
    N = ray_c.shape[0]
    rayP_c = rayP_c.view(1, 3).repeat(N, 1)
    ds = torch.rand((N, S), device=device) * depth.unsqueeze(1)
    samples_c = rayP_c.unsqueeze(1) + ds.unsqueeze(2) * ray_c.unsqueeze(1)
    samples_c = samples_c.view(-1, 3)
    samples_v = T_vc * samples_c
    pc_ids, valid = pointcloud_to_voxel_ids(samples_v, vW, vH, vD, voxel_extent)
    pc_ids = pc_ids[valid]
    occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 0.0
    mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    # add points as occupied
    pc_v = T_wv.inverse() * pc_w
    invalid = pc_v.isnan().any(-1)
    pc_v = pc_v[~invalid]
    pc_ids, valid = pointcloud_to_voxel_ids(pc_v, vW, vH, vD, voxel_extent)
    pc_ids = pc_ids[valid]
    occ[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0
    mask[pc_ids[:, 0], pc_ids[:, 1], pc_ids[:, 2]] = 1.0

    return occ, mask


def pointcloud_to_voxel_counts(points_v, voxel_extent, vW, vH, vD):
    """
    Convert a pointcloud in the voxel coordinate to a voxel grid where each voxel value indicates the number of points falling into this voxel.
    """
    assert points_v.ndim == 2, f"{points_v.shape}"
    voxel_extent = tensor_wrap_voxel_extent(voxel_extent).to(points_v.device)
    assert voxel_extent.ndim == 1, f"{voxel_extent.shape}"
    if points_v.shape[0] == 0:
        print("WARNING: No 3D points provided. ")
        return torch.zeros((1, vD, vH, vW), device=points_v.device, dtype=torch.int64)
    num_voxels_x, num_voxels_y, num_voxels_z = vW, vH, vD
    bb_min, bb_max = voxel_extent[..., 0::2], voxel_extent[..., 1::2]
    dim = torch.tensor([vW, vH, vD], device=points_v.device)
    voxel_sizes = (bb_max - bb_min) / dim
    voxel_min = bb_min
    point_count = torch.zeros(
        (num_voxels_x, num_voxels_y, num_voxels_z), device=points_v.device
    )
    voxel_indices = torch.floor((points_v - voxel_min) / voxel_sizes).to(torch.int64)
    # Filter out points that fall outside the voxel grid
    valid_indices = (voxel_indices >= 0) & (
        voxel_indices
        < torch.tensor([num_voxels_x, num_voxels_y, num_voxels_z]).to(voxel_indices)
    )
    valid_indices = valid_indices.all(dim=-1)
    voxel_indices = voxel_indices[valid_indices]

    # get flat index so we can use bincount to get counts
    voxel_indices_flat = (
        voxel_indices[..., 0]
        + voxel_indices[..., 1] * vW
        + voxel_indices[..., 2] * vW * vH
    )
    # get counts of how many points per voxel
    point_count = torch.bincount(voxel_indices_flat, minlength=vW * vH * vD)
    # reshape back to vD x vH x vW convention.
    point_count = point_count.view(1, vD, vH, vW)
    return point_count


def get_points_counts(
    batch,
    T_wv,
    vW,
    vH,
    vD,
    voxel_extent,
    prefer_points=True,
    MAX_NUM_POINTS_VOXEL=50,
    return_mask=False,
    dropout_points=False,
    dropout_points_rate_max=0.0,
):
    """
    Get points as voxel grid where each voxel is assigned a count of how many points are inside it.
    If return_mask is trued the function returns the binary occupancy instead of point counts.
    """
    B, T, _, H, W = batch[ARIA_IMG[0]].shape
    point_counts = []
    for b in range(B):
        p_w = get_points_world(batch, b, prefer_points=prefer_points)[0]
        p_w = collapse_pointcloud_time(p_w)
        if dropout_points:
            print("drop points ", p_w.shape)
            N = p_w.shape[0]
            p = dropout_points_rate_max
            Ndrop = int(N * (torch.rand(1).item() * p + (1.0 - p)))
            print(f"dropout {N-Ndrop}/{N} points")
            rnd = torch.randperm(N, device=p_w.device)[:Ndrop]
            p_w = p_w[rnd, :]
        # transform points into voxel coordinate.
        p_v = T_wv[b].inverse() * p_w
        if isinstance(voxel_extent, list):
            ve_b = voxel_extent
        else:
            ve_b = voxel_extent[b].tolist()
        point_count = pointcloud_to_voxel_counts(p_v, ve_b, vW, vH, vD)
        point_counts.append(point_count)
    point_counts = torch.stack(point_counts, dim=0)  # B x 1 x vD, vH, vW
    # Normalize
    point_counts = point_counts.clamp(0, MAX_NUM_POINTS_VOXEL) / MAX_NUM_POINTS_VOXEL
    if return_mask:
        # Only use as a mask. Comment out if want to use real point counts.
        point_counts[point_counts > 1e-4] = 1.0

    return point_counts


def get_freespace_counts(
    batch,
    T_wv,
    vW,
    vH,
    vD,
    voxel_extent,
    num_free_samples=1,
    prefer_points=True,
    MAX_NUM_POINTS_VOXEL=50,
    return_mask=False,
    dropout_points=False,
    dropout_points_rate_max=0.0,
):
    """
    Get points as voxel grid where each voxel is assigned a count of how many points are inside it.
    If return_mask is trued the function returns the binary occupancy instead of point counts.
    """
    B, T, _, H, W = batch[ARIA_IMG[0]].shape
    point_counts = []
    for b in range(B):
        if isinstance(voxel_extent, list):
            ve_b = voxel_extent
        else:
            ve_b = voxel_extent[b].tolist()
        p_w = get_freespace_world(
            batch,
            b,
            T_wv[b],
            vW,
            vH,
            vD,
            ve_b,
            num_free_samples,
            prefer_points,
            dropout_points,
            dropout_points_rate_max,
        )
        # transform points into voxel coordinate.
        p_v = T_wv[b].inverse() * p_w
        point_count = pointcloud_to_voxel_counts(p_v, ve_b, vW, vH, vD)
        point_counts.append(point_count)
    point_counts = torch.stack(point_counts, dim=0)  # B x 1 x vD, vH, vW
    # Normalize
    point_counts = point_counts.clamp(0, MAX_NUM_POINTS_VOXEL) / MAX_NUM_POINTS_VOXEL
    if return_mask:
        # Only use as a mask. Comment out if want to use real point counts.
        point_counts[point_counts > 1e-4] = 1.0

    return point_counts
