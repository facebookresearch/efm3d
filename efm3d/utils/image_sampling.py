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

import einops
import torch


def compute_factor(size):
    return 1.0 * size / 2


def convert_pixel_to_coordinates(coordinates, factor):
    return (coordinates / factor) - 1.0


def normalize_keypoints(kpts, height, width):
    # compute conversion factor
    x_factor = compute_factor(width)
    y_factor = compute_factor(height)
    pts_dst = kpts
    pts_dst[..., 0] = convert_pixel_to_coordinates(pts_dst[..., 0], x_factor)
    pts_dst[..., 1] = convert_pixel_to_coordinates(pts_dst[..., 1], y_factor)
    return pts_dst


def sample_images(
    feat2d,
    query_pts_cam,
    cams,
    n_by_c=True,
    warn=True,
    padding_mode: Literal["border", "zeros", "reflection"] = "border",
    interp_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
    single_channel_mask: bool = False,
):
    """
    Uses 3D points and calibrated cameras to sample features from 2D feature maps.

    Inputs:
        feat2d: torch.tensor - feature maps to sample from shaped B(xT)xCxHxW
        query_pts_cam: torch.tensor - 3D points in camera coordinates shaped B(xT)xNx3
        cams: CameraTW - calibrated camera objects shaped B(xT)x15
        n_by_c: return shapes ending in NxC or CxN
    Returns:
        samp_feats: torch.tensor - sampled features from 2D feature maps shaped B(xT)xCxN
        valid: torch.tensor - boolean of whether there was a valid sampling B(xT)xCxN
    """
    assert query_pts_cam.dim() == feat2d.dim() - 1

    T = None
    if feat2d.dim() == 5:
        B, T, C, H, W = feat2d.shape
        feat2d = feat2d.view(-1, C, H, W)
        query_pts_cam = query_pts_cam.view(B * T, -1, 3)
        cams = cams.view(B * T, -1)
    elif feat2d.dim() == 4:
        B, C, H, W = feat2d.shape
    else:
        raise ValueError(f"feat2d.dim must be 5 or 4 {feat2d.shape}")

    camH = cams[0].size[1]
    featH = feat2d.shape[-2]
    camW = cams[0].size[0]
    featW = feat2d.shape[-1]

    # Cams may need to be rescaled to match the feature map spatial dimensions.
    if camH != featH or camW != featW:
        cams_resize = cams.scale_to(feat2d)
    else:
        cams_resize = cams

    assert (
        round(cams_resize[0].size[0].item()) == featW
    ), f"height of cam and feature image do not match. {cams_resize[0].size[0]}!= {feat2d.shape}"
    assert (
        round(cams_resize[0].size[1].item()) == featH
    ), f"width of cam and feature image do not match. {cams_resize[0].size[1]}!= {feat2d.shape}"

    samp_pts, valid = cams_resize.project(query_pts_cam)
    if warn:
        frac_valid = valid.count_nonzero() / valid.numel()
        if frac_valid < 0.05:
            print(
                f"[Warning] not many valids! {frac_valid} {valid.count_nonzero()} {valid.shape}"
            )
    samp_pts[~valid] = 0.0
    samp_pts = samp_pts.float()
    # Sample into the 2D feature maps.
    norm_samp_pts = normalize_keypoints(
        samp_pts.clone(), height=cams_resize[0].size[1], width=cams_resize[0].size[0]
    )
    samp_feats = torch.nn.functional.grid_sample(
        feat2d,
        norm_samp_pts.unsqueeze(-2),
        align_corners=False,
        padding_mode=padding_mode,
        mode=interp_mode,  # bilinear allows differentiating.
    )
    # squeeze back down the dimension of 1 we unsqueezed for norm_samp_pts to comply with interface
    samp_feats = samp_feats.squeeze(-1)

    # Overwrite invalid projections with zeros.
    BT = samp_feats.shape[0]
    valid = valid.reshape(BT, 1, -1)
    if single_channel_mask:
        samp_feats[(~valid).expand_as(samp_feats)] = 0.0
    else:
        valid = valid.repeat(1, C, 1)
        samp_feats[~valid] = 0.0
    if T is None:
        if n_by_c:
            samp_feats = einops.rearrange(samp_feats, "b c n -> b n c", b=B)
            valid = einops.rearrange(valid, "b c n -> b n c", b=B)[..., 0]
    else:
        if n_by_c:
            samp_feats = einops.rearrange(samp_feats, "(b t) c n -> b t n c", t=T, b=B)
            valid = einops.rearrange(valid, "(b t) c n -> b t n c", t=T, b=B)[..., 0]
        else:
            samp_feats = einops.rearrange(samp_feats, "(b t) c n -> b t c n", t=T, b=B)
            valid = einops.rearrange(valid, "(b t) c n -> b t c n", t=T, b=B)[..., 0]
    return samp_feats, valid
