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

import numpy as np
import torch


def pc_to_vox(pc_v, vW, vH, vD, voxel_extent):
    device = pc_v.device
    if isinstance(voxel_extent, list):
        x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
        valid = pc_v[..., 0] > x_min
        valid = torch.logical_and(pc_v[..., 0] < x_max, valid)
        valid = torch.logical_and(pc_v[..., 1] > y_min, valid)
        valid = torch.logical_and(pc_v[..., 1] < y_max, valid)
        valid = torch.logical_and(pc_v[..., 2] > z_min, valid)
        valid = torch.logical_and(pc_v[..., 2] < z_max, valid)
        dW = (x_max - x_min) / vW
        dH = (y_max - y_min) / vH
        dD = (z_max - z_min) / vD
        dVox = torch.tensor([dW, dH, dD]).view(1, 3).to(device)
        vox_min = torch.tensor([x_min, y_min, z_min]).view(1, 3).to(device)
        pc_id = (pc_v - vox_min) / dVox
    else:
        s = pc_v.shape[:-1]
        B = s[0]
        vox_min = voxel_extent[..., 0::2].view(B, 1, 3)
        vox_max = voxel_extent[..., 1::2].view(B, 1, 3)
        dim = (
            torch.tensor([vW, vH, vD], device=voxel_extent.device)
            .view(1, 1, 3)
            .repeat(B, 1, 1)
        )
        dVox = (vox_max - vox_min) / dim
        pc_v = pc_v.view(B, -1, 3)
        valid = torch.logical_not(pc_v.isnan().any(-1))
        valid = torch.logical_and(valid, (pc_v > vox_min).all(-1))
        valid = torch.logical_and(valid, (pc_v < vox_max).all(-1))
        pc_id = (pc_v - vox_min) / dVox
        valid = valid.view(s)
        pc_id = pc_id.view(list(s) + [3])

    return pc_id, valid


def compute_factor(size):
    return 1.0 * size / 2


def convert_coordinates_to_voxel(coordinates, factor):
    return factor * (coordinates + 1.0)


def convert_voxel_to_coordinates(coordinates, factor):
    return (coordinates / factor) - 1.0


def normalize_keypoints(kpts, depth, height, width):
    # compute conversion factor
    x_factor = compute_factor(width)
    y_factor = compute_factor(height)
    z_factor = compute_factor(depth)
    factors = torch.tensor([x_factor, y_factor, z_factor], device=kpts.device).view(
        [1] * (kpts.ndim - 1) + [3]
    )
    pts_dst = convert_voxel_to_coordinates(kpts, factors)
    return pts_dst


def denormalize_keypoints(kpts, depth, height, width):
    # compute conversion factor
    x_factor = compute_factor(width)
    y_factor = compute_factor(height)
    z_factor = compute_factor(depth)
    if isinstance(kpts, torch.Tensor):
        pts_dst = kpts.clone()
    elif isinstance(kpts, np.ndarray):
        pts_dst = kpts.copy()
    else:
        raise TypeError("must be torch or numpy")
    factors = torch.tensor([x_factor, y_factor, z_factor], device=kpts.device).view(
        [1] * (kpts.ndim - 1) + [3]
    )
    pts_dst = convert_coordinates_to_voxel(kpts, factors)
    return pts_dst


def in_grid(pt_vox, depth, height, width):
    valid = pt_vox[..., 0] >= 0.5
    valid = torch.logical_and(pt_vox[..., 0] <= width - 0.5, valid)
    valid = torch.logical_and(pt_vox[..., 1] >= 0.5, valid)
    valid = torch.logical_and(pt_vox[..., 1] <= height - 0.5, valid)
    valid = torch.logical_and(pt_vox[..., 2] >= 0.5, valid)
    valid = torch.logical_and(pt_vox[..., 2] <= depth - 0.5, valid)
    return valid


def sample_voxels(feat3d, pts_v, differentiable=False, interp_mode="bilinear"):
    """
    Sample voxel grid of features at pts_v locations.
    Args:
        feat3d: feature volume batches B C D H W
        pts_v: 3d points in -1 to 1 range in shape compatible with B N 3
        differentiable: we need this to be differentiable wrt to the pts_v
    Returns:
        voxel grid samples in shape B C N
    """
    assert feat3d.ndim == 5, f"{feat3d.shape}"
    assert pts_v.ndim == 3, f"{pts_v.shape}"
    B, C, D, H, W = feat3d.shape
    valid = in_grid(pts_v, height=H, width=W, depth=D)
    # Sample into the 3D feature maps.
    norm_samp_pts = normalize_keypoints(pts_v.clone(), height=H, width=W, depth=D)
    if differentiable:
        # use differentiable implementation of 3d trilinear interpolation.
        samp_feats = diff_grid_sample(
            feat3d,
            norm_samp_pts.view(B, 1, 1, -1, 3),
            align_corners=False,  # B 1 1 N 3
        )
    else:
        # if we dont need differentiability wrt to sample points then we can use
        # the default implementation.
        samp_feats = torch.nn.functional.grid_sample(
            feat3d,
            norm_samp_pts.view(B, 1, 1, -1, 3),  # B 1 1 N 3
            align_corners=False,
            padding_mode="border",
            mode=interp_mode,  # important to be differentiable
        )
    # squeeze back down the dimension of 1 we unsqueezed for norm_samp_pts to comply with interface
    samp_feats = samp_feats.view(B, C, -1)
    return samp_feats, valid


def diff_grid_sample(feature_3d, pts_norm, align_corners=False):
    N, C, iD, iH, iW = feature_3d.shape
    _, D, H, W, _ = pts_norm.shape
    assert not pts_norm.isnan().any(), "have nan values in pts_norm! not supported"

    ix = pts_norm[..., 0]
    iy = pts_norm[..., 1]
    iz = pts_norm[..., 2]

    if align_corners:
        ix = ((ix + 1.0) * 0.5) * (iW - 1)
        iy = ((iy + 1.0) * 0.5) * (iH - 1)
        iz = ((iz + 1.0) * 0.5) * (iD - 1)
    else:
        ix = ((ix + 1.0) * 0.5) * iW - 0.5
        iy = ((iy + 1.0) * 0.5) * iH - 0.5
        iz = ((iz + 1.0) * 0.5) * iD - 0.5

    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)

    with torch.no_grad():
        torch.clamp(ix_bnw, 0, iW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, iH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, iD - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, iW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, iH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, iD - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, iW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, iH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, iD - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, iW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, iH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, iD - 1, out=iz_bse)

        torch.clamp(ix_tnw, 0, iW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, iH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, iD - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, iW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, iH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, iD - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, iW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, iH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, iD - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, iW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, iH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, iD - 1, out=iz_tse)

    feature_3d = feature_3d.reshape(N, C, -1)

    # D H W, z y x
    bnw_val = torch.gather(
        feature_3d,
        2,
        (iz_bnw * iH * iW + iy_bnw * iW + ix_bnw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bne_val = torch.gather(
        feature_3d,
        2,
        (iz_bne * iH * iW + iy_bne * iW + ix_bne)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bsw_val = torch.gather(
        feature_3d,
        2,
        (iz_bsw * iH * iW + iy_bsw * iW + ix_bsw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bse_val = torch.gather(
        feature_3d,
        2,
        (iz_bse * iH * iW + iy_bse * iW + ix_bse)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )

    tnw_val = torch.gather(
        feature_3d,
        2,
        (iz_tnw * iH * iW + iy_tnw * iW + ix_tnw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tne_val = torch.gather(
        feature_3d,
        2,
        (iz_tne * iH * iW + iy_tne * iW + ix_tne)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tsw_val = torch.gather(
        feature_3d,
        2,
        (iz_tsw * iH * iW + iy_tsw * iW + ix_tsw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tse_val = torch.gather(
        feature_3d,
        2,
        (iz_tse * iH * iW + iy_tse * iW + ix_tse)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )

    out_val = (
        bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W)
        + bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W)
        + bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W)
        + bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W)
        + tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W)
        + tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W)
        + tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W)
        + tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W)
    )

    return out_val
