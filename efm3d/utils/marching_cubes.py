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

import logging

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


def marching_cubes_scaled(values, isolevel, voxel_extent, voxel_mask):
    """
    Runs marching cubes on a values tensor (D H W) at the specified isolevel.
    Voxel_mask is used to tell marching cubes where to run in the voxel grid.
    Uses scikit implementation which runs only on CPU.

    Returns vertices, face ids, and normals in the voxel coordinate system
    scaled to the given voxel_extent.
    """

    from skimage.measure import marching_cubes as mc_scikit

    device = values.device
    values = values.cpu()  # CPU only
    assert values.ndim == 3, f"skicit can only do non-batched inputs, {values.shape}"
    isolevel = max(values.min(), min(isolevel, values.max()))
    logging.info(f"mc min {values.min()}, max {values.max()}, isolevel {isolevel}")
    voxel_mask = voxel_mask.cpu().numpy() if voxel_mask is not None else None
    try:
        if voxel_mask is not None:
            verts, faces, normals, _ = mc_scikit(
                values.contiguous().numpy(), isolevel, mask=voxel_mask
            )
        else:
            verts, faces, normals, _ = mc_scikit(values.contiguous().numpy(), isolevel)
        logging.info(f"{verts.shape}, {faces.shape}")
    except RuntimeError as e:
        logging.error(f"{e} {values.shape}, {voxel_mask.shape}")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    except Exception as e:
        logging.error(f"{e} {values.shape}, {voxel_mask.shape}")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # copy to get around negative stride
    # go back to x, y, z ordering
    verts, faces, normals = (
        torch.from_numpy(verts.copy()),
        torch.from_numpy(faces.copy()),
        torch.from_numpy(normals.copy()),
    )
    verts = verts[:, [2, 1, 0]]
    normals = normals[:, [2, 1, 0]]
    verts, faces, normals = verts.to(device), faces.to(device), normals.to(device)

    logging.info(f"{verts.shape}, {faces.shape}, {normals.shape}")

    vD, vH, vW = values.shape
    logging.info(f"{vD}, {vH}, {vW}, {voxel_extent}")
    x_min, x_max, y_min, y_max, z_min, z_max = voxel_extent
    dW = (x_max - x_min) / vW
    dH = (y_max - y_min) / vH
    dD = (z_max - z_min) / vD

    dVox = torch.tensor([dW, dH, dD]).view(1, 3).to(device)
    vox_min = torch.tensor([x_min, y_min, z_min]).view(1, 3).to(device)
    logging.info(f"{verts.shape}")
    verts = verts * dVox + vox_min + dVox * 0.5
    return verts, faces, normals
