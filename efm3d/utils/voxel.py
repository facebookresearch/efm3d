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


def tensor_wrap_voxel_extent(voxel_extent, B=None, device="cpu"):
    if isinstance(voxel_extent, torch.Tensor):
        if B is not None:
            assert voxel_extent.shape[0] == B
        return voxel_extent
    elif isinstance(voxel_extent, list):
        if B is None:
            return torch.tensor(voxel_extent, device=device)
        else:
            return torch.tensor(voxel_extent, device=device).view(1, 6).repeat(B, 1)
    else:
        raise NotImplementedError(f"type {voxel_extent} not supported")


def create_voxel_grid(vW, vH, vD, voxel_extent, device="cpu"):
    """
    Given a bounding box range [x_min, x_max, y_min, y_max, z_min, z_max], and the
    number of voxels in each dimension [vW, vH, vD], return a voxel center positions.
    Note that the min and max coordinates are not [x_min, y_min, z_min] and [x_max, y_max, z_max],
    since they are the bounding range but not the center positions.

    vW: the number of voxels for x-dim
    vH: the number of voxels for y-dim
    vD: the number of voxels for z-dim
    voxel_extent: the bounding box range in [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
    dW = (x_max - x_min) / vW
    dH = (y_max - y_min) / vH
    dD = (z_max - z_min) / vD
    # take the center position of each voxel
    rng_x = torch.linspace(x_min + dW / 2, x_max - dW / 2, steps=vW, device=device)
    rng_y = torch.linspace(y_min + dH / 2, y_max - dH / 2, steps=vH, device=device)
    rng_z = torch.linspace(z_min + dD / 2, z_max - dD / 2, steps=vD, device=device)
    xx, yy, zz = torch.meshgrid(rng_x, rng_y, rng_z, indexing="ij")
    vox_v = torch.stack([xx, yy, zz], axis=-1)
    return vox_v


def erode_voxel_mask(mask):
    """
    Erode a given mask by one voxel i.e.
    0 0 0 0 0    0 0 0 0 0
    0 1 1 1 0    0 0 0 0 0
    0 1 1 1 0 -> 0 0 1 0 0
    0 1 1 1 0    0 0 0 0 0
    0 0 0 0 0    0 0 0 0 0
    """
    # B T D H W
    assert mask.ndim in [4, 5], f"mask dim needs to be 3 or 4 got {mask.shape}"
    kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device)
    mask = (
        1.0
        - torch.clamp(
            torch.nn.functional.conv3d(1.0 - mask.float(), kernel, padding="same"), 0, 1
        )
    ).bool()
    return mask
