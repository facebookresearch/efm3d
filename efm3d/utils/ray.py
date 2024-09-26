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
from efm3d.aria.camera import CameraTW, pixel_grid
from efm3d.utils.voxel import tensor_wrap_voxel_extent
from torch.nn import functional as F


def grid_ray(pixel_grid, camera):
    """
    grid_ray:
            Given a 2D grid size, this function creates a 2D grid and then unprojects the grid
            into rays in their respective rig coordinate systems.

    Args:
        grid_width: self-explanatory
        grid_height: self-explanatory
        camera: Batch of Camera objects [B x object_params]

    Returns:
        Rays: [B x grid_height x grid_width x 6] rays in their respective rig coordinates
        Each ray grid in a batch may have different rig coordinate systems.
        Valid: Valid rays in the batch
    """
    eps = 1e-6
    grid_height, grid_width = pixel_grid.shape[0], pixel_grid.shape[1]
    batch_size = camera.shape[0]
    pixel_grid = pixel_grid.reshape(-1, 2)
    pixel_grid = einops.repeat(pixel_grid, "n c -> b n c", b=batch_size)
    rays, valid = camera.double().unproject(pixel_grid.double())
    rays = rays.float()
    assert not torch.isnan(
        rays
    ).any(), f"have {torch.isnan(rays).count_nonzero().item()} nans in rays. Camera params: {camera.params}"
    rays = F.normalize(rays, p=2, dim=-1, eps=eps)
    rays = torch.where(valid.unsqueeze(-1), rays, torch.zeros_like(rays))
    T_rig_camera = camera.T_camera_rig.inverse()
    T_rig_camera = T_rig_camera.to(dtype=rays.dtype)
    rays = T_rig_camera.rotate(rays)
    ray_origins = einops.repeat(
        T_rig_camera.t, "b c -> b n c", n=grid_width * grid_height
    )

    # set invalid rays to zeros
    rays = F.normalize(rays, p=2, dim=-1, eps=eps)
    rays = torch.where(valid.unsqueeze(-1), rays, torch.zeros_like(rays))
    ray_origins = torch.where(
        valid.unsqueeze(-1), ray_origins, torch.zeros_like(ray_origins)
    )

    rays = torch.cat([ray_origins, rays], dim=-1)
    return rays.view([batch_size, grid_height, grid_width, -1]), valid.view(
        [batch_size, grid_height, grid_width]
    )


def ray_grid(cam: CameraTW):
    """
    rays returned are in rig coordinate system
    """
    if cam.ndim == 1:
        px = pixel_grid(cam)
        rays, valid = grid_ray(px, cam.unsqueeze(0))
        return rays.squeeze(0), valid.squeeze(0)
    elif cam.ndim == 2:
        px = pixel_grid(cam[0])  # assuming camera sizes are all the same in a batch!
        return grid_ray(px, cam)
    else:
        raise ValueError(f"Camera must be 1 or 2 dimensional: {cam.shape}")


def transform_rays(rays_old: torch.Tensor, T_new_old):
    """
    Expects rays to be in old coordinate frame
    """
    assert rays_old.shape[
        -1
    ], "Rays must be 6 dimensional in the following order: [ray_origins, ray_directions]"
    ray_origins = T_new_old.transform(rays_old[..., :3])
    ray_directions = T_new_old.rotate(rays_old[..., 3:])
    return torch.cat([ray_origins, ray_directions], dim=-1)


def ray_obb_intersection(
    rays_v, voxel_extent, t_min=-1e9, t_max=1e9, return_points=False
):
    assert rays_v.ndim == 3, f"{rays_v.shape}"
    assert rays_v.shape[-1] == 6, f"{rays_v.shape}"

    device = rays_v.device
    B, N = rays_v.shape[:2]
    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
    raysP_v = rays_v[..., :3]
    raysD_v = rays_v[..., 3:]  # assume normalized!

    ns_bb = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    ps_bb = [
        [x_max, 0.0, 0.0],
        [x_min, 0.0, 0.0],
        [0.0, y_max, 0.0],
        [0.0, y_min, 0.0],
        [0.0, 0.0, z_max],
        [0.0, 0.0, z_min],
    ]

    eps = 1e-3
    minmaxs_bb = [
        [x_max - eps, y_min, z_min, x_max + eps, y_max, z_max],
        [x_min - eps, y_min, z_min, x_min + eps, y_max, z_max],
        [x_min, y_max - eps, z_min, x_max, y_max + eps, z_max],
        [x_min, y_min - eps, z_min, x_max, y_min + eps, z_max],
        [x_min, y_min, z_max - eps, x_max, y_max, z_max + eps],
        [x_min, y_min, z_min - eps, x_max, y_max, z_min + eps],
    ]

    t_upper = torch.ones((B, N), device=device) * t_max
    t_lower = torch.ones((B, N), device=device) * t_min
    ts = torch.stack([t_upper, t_lower], dim=-1)
    for n_bb, p_bb, minmax_bb in zip(ns_bb, ps_bb, minmaxs_bb):
        n_bb = torch.tensor(n_bb).view(1, 1, 3).to(device)
        p_bb = torch.tensor(p_bb).view(1, 1, 3).to(device)
        min_bb = torch.tensor(minmax_bb[:3]).view(1, 1, 3).to(device)
        max_bb = torch.tensor(minmax_bb[3:]).view(1, 1, 3).to(device)
        # dot product
        denom = (raysD_v * n_bb).sum(-1)
        valid = denom.abs() > 1e-6
        dp = p_bb - raysP_v
        t = (dp * n_bb).sum(-1) / denom
        valid = torch.logical_and(valid, t > t_min)
        valid = torch.logical_and(valid, t < t_max)
        # points on surface
        ps_v = raysP_v + raysD_v * t.unsqueeze(-1)
        valid = torch.logical_and(valid, (ps_v > min_bb).all(-1))
        valid = torch.logical_and(valid, (ps_v < max_bb).all(-1))

        ts_min = torch.where(valid, t, t_upper)
        ts_max = torch.where(valid, t, t_lower)

        ts[..., 0] = torch.minimum(ts_min, ts[..., 0])
        ts[..., 1] = torch.maximum(ts_max, ts[..., 1])

    if return_points:
        one_int = ts[..., 0] == ts[..., 1]
        ts[..., 0] = torch.where(
            one_int, t_min * torch.ones_like(ts[..., 0]) * t_min, ts[..., 0]
        )
        no_int = ts[..., 0] > ts[..., 1]
        ts[no_int] = t_min

        ps_min_v = raysP_v + raysD_v * ts[..., 0].unsqueeze(-1)
        ps_max_v = raysP_v + raysD_v * ts[..., 1].unsqueeze(-1)
        return ts, ps_min_v, ps_max_v
    return ts


def sample_depths_in_grid(
    rays_v,
    ds_max,
    voxel_extent,
    W,
    H,
    D,
    num_samples,
    d_near=0.01,
    d_far=10.0,
    sample_mode: Literal["random", "uniform"] = "random",
    ds_min=None,
):
    assert rays_v.ndim == 4, f"{rays_v.shape}"  # BxTxNx6
    assert ds_max.ndim == 3, f"{ds_max.shape}"  # BxTxN
    B = rays_v.shape[0]
    voxel_extent = tensor_wrap_voxel_extent(voxel_extent, B).to(rays_v.device)

    def safe_extent(voxel_extent, W, H, D):
        # compute a "safe" voxel extent that is shrunk by half a voxel in all
        # directions
        bb_min, bb_max = voxel_extent[::2], voxel_extent[1::2]
        dim = torch.tensor([W, H, D], device=voxel_extent.device)
        dd = 0.5 * (bb_max - bb_min) / dim
        bb_min = bb_min + dd
        bb_max = bb_max - dd
        voxel_extent_safe = torch.zeros_like(voxel_extent)
        voxel_extent_safe[::2] = bb_min
        voxel_extent_safe[1::2] = bb_max
        return voxel_extent_safe

    B, T, N = rays_v.shape[:3]
    ts = []
    for b in range(B):
        voxel_extent_safe = safe_extent(voxel_extent[b], W, H, D)
        ts.append(
            ray_obb_intersection(
                rays_v[b].view(T, N, 6),
                voxel_extent_safe,
                t_min=d_near,
                t_max=d_far,
            )
        )
    ts = torch.stack(ts, 0)  # BxTxNx2

    no_int = ts[..., 0] > ts[..., 1]
    one_int = ts[..., 0] == ts[..., 1]
    depths_min = torch.where(
        one_int, torch.ones_like(ts[..., 0]) * d_near, ts[..., 0]
    )  # BxTxN
    depths_max = ts[..., 1]
    depths_min[no_int] = torch.nan
    depths_max[no_int] = torch.nan

    if ds_max is not None:
        depths_max = torch.minimum(ds_max, depths_max)
    if ds_min is not None:
        depths_min = torch.maximum(ds_min, depths_min)

    ddepths = depths_max - depths_min
    ddepths[ddepths < 1e-3] = torch.nan
    # go to d_min to d_max per ray
    depths = torch.linspace(0.0, 1.0, num_samples).to(rays_v.device)
    depths = depths.view(1, 1, 1, num_samples).repeat(B, T, N, 1)
    depths = depths_min.unsqueeze(-1) + ddepths.unsqueeze(-1) * depths
    if sample_mode == "uniform":
        return depths, depths_max, ~no_int.view(B, T, N)
    elif sample_mode == "random":
        # add noise
        noise = torch.rand((B, T, N, num_samples), device=rays_v.device)
        noise = noise * (ddepths.unsqueeze(-1) / num_samples)
        if num_samples > 1:
            noise[..., -1] = 0.0
        depths = depths + noise
        return depths, depths_max, ~no_int.view(B, T, N)
    else:
        raise ValueError(f"Unknown sample mode {sample_mode}")
