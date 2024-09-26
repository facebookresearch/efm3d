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

import glob
import logging
import os
from typing import List

import numpy as np
import torch
import tqdm

import trimesh
from efm3d.aria.pose import PoseTW
from efm3d.utils.marching_cubes import marching_cubes_scaled
from efm3d.utils.reconstruction import pc_to_vox, sample_voxels
from efm3d.utils.voxel import create_voxel_grid


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


def set_boundary_value(x, val, thickness):
    if thickness == 0:
        return x
    x[..., :thickness, :, :] = val
    x[..., -thickness:, :, :] = val
    x[..., :, :thickness, :] = val
    x[..., :, -thickness:, :] = val
    x[..., :, :, :thickness] = val
    x[..., :, :, -thickness:] = val
    return x


def load_tensor(fname, device):
    data = torch.load(fname, map_location=device)
    if "_8b" in fname:
        data = data.dequantize()
    return data


class VolumeFusion:
    def __init__(
        self,
        voxel_size: List[float],
        voxel_extent: List[float],
        device: str = "cuda",
        dtype=torch.float32,
        w_min: float = 5.0,
        w_max: float = 100.0,
        init_value: float = 0.0,
        surface_thres: float = 0.99,
        boundary_thres: int = 1,
    ):

        self.voxel_size = voxel_size  # D x H x W
        self.voxel_extent = voxel_extent  # W x H x D
        self.vD, self.vH, self.vW = self.voxel_size
        self.vD = int(self.vD)
        self.vH = int(self.vH)
        self.vW = int(self.vW)
        self.w_max = w_max
        self.w_min = w_min
        self.surface_thres = surface_thres
        self.boundary_thres = boundary_thres

        self.global_volume = torch.ones(
            self.vD, self.vH, self.vW, device=device, dtype=dtype
        )  # D H W
        self.global_volume = self.global_volume * init_value
        self.global_volume_weights = torch.zeros_like(self.global_volume)  # D H W
        self.global_volume_points = create_voxel_grid(
            self.vW, self.vH, self.vD, self.voxel_extent, device
        ).to(
            dtype=dtype
        )  # W, H, D, 3
        # reshaping
        self.global_volume_points = self.global_volume_points.permute(
            2, 1, 0, 3
        )  # D H W 3
        self.global_volume_points = self.global_volume_points.reshape(
            -1, 3
        )  # (D*H*W) x 3
        self.global_volume_weights = self.global_volume_weights.reshape(-1)  # D*H*W
        self.global_volume = self.global_volume.reshape(-1)  # D*H*W

        self.device = device

    def set_boundary_mask(self, mask):
        thickness = self.boundary_thres
        mask[:thickness] = False  # Set the first 'thickness' layers in Height to zero
        mask[-thickness:] = False  # Set the last 'thickness' layers in Height to zero
        mask[:, :thickness] = False  # Set the first 'thickness' layers in Width to zero
        mask[:, -thickness:] = False  # Set the last 'thickness' layers in Width to zero
        mask[:, :, :thickness] = (
            False  # Set the first 'thickness' layers in Depth to zero
        )
        mask[:, :, -thickness:] = (
            False  # Set the last 'thickness' layers in Depth to zero
        )
        return mask

    def fuse(
        self,
        local_volume: torch.Tensor,
        local_extent: List[float],
        T_l_w: PoseTW,
        new_obs_w=1.5,
        visiblity_mask=None,
    ):
        local_volume = local_volume.to(self.global_volume.device)
        T_l_w = T_l_w.to(self.global_volume.device)

        vD, vH, vW = local_volume.shape
        # transform global_volume to local volume
        global_volume_l = T_l_w * self.global_volume_points
        global_volume_l_coord, valid_global_points = pc_to_vox(
            global_volume_l, vW, vH, vD, local_extent
        )
        local_samples, valid_samples = sample_voxels(
            local_volume.unsqueeze(0).unsqueeze(0).float(),
            global_volume_l_coord.view(1, -1, 3).float(),
        )
        local_samples = (
            local_samples.squeeze(0).squeeze(0).to(dtype=self.global_volume.dtype)
        )
        valid_samples = valid_samples.squeeze(0)

        # making a mask
        surface_mask = local_volume < self.surface_thres
        if visiblity_mask is not None:
            surface_mask &= visiblity_mask.to(surface_mask)
        # we don't trust the boundary voxels from CNNS
        if self.boundary_thres > 0:
            surface_mask = self.set_boundary_mask(surface_mask)
        surface_mask_f = surface_mask.float()
        surface_mask_f[~surface_mask] = torch.nan
        # sample the mask
        surface_mask_samples, _ = sample_voxels(
            surface_mask_f.unsqueeze(0).unsqueeze(0).float(),
            global_volume_l_coord.view(1, -1, 3).float(),
        )
        surface_mask = ~surface_mask_samples.isnan()
        valid_samples = valid_samples & surface_mask
        mask = valid_samples & valid_global_points
        mask = mask.squeeze()
        w = self.global_volume_weights[mask]

        self.global_volume[mask] = (
            self.global_volume[mask] * w + local_samples[mask] * 2.0
        ) / (w + 2.0)

        # update weights
        self.global_volume_weights[mask] = w + new_obs_w
        self.global_volume_weights[mask] = self.global_volume_weights[mask].clamp(
            max=self.w_max
        )

    def get_volume(self, reshape=True):
        if reshape:
            return self.global_volume.reshape(self.vD, self.vH, self.vW)
        else:
            return self.global_volume

    def get_weights(self, reshape=True):
        if reshape:
            return self.global_volume_weights.reshape(self.vD, self.vH, self.vW)
        else:
            self.global_volume_weights

    def get_mask(self, reshape=True):
        mask = self.global_volume_weights >= self.w_min
        if reshape:
            return mask.reshape(self.vD, self.vH, self.vW)
        else:
            mask

    def get_trimesh(self, iso_level=0.5):
        global_vol = self.get_volume()
        mask = self.get_mask()
        verts_w, faces, _ = marching_cubes_scaled(
            global_vol.cpu().detach().float(),
            iso_level,
            self.voxel_extent,
            mask,
        )
        sem_rgb = None
        mesh = trimesh.Trimesh(verts_w, faces, vertex_colors=sem_rgb)

        return mesh


class VolumetricFusion:
    def __init__(
        self,
        input_folder,
        w_min=5.0,
        w_max=9999999.0,
        voxel_res=0.04,
        device="cuda",
    ):
        self.input_folder = input_folder
        self.per_snip_folder = os.path.join(input_folder, "per_snip")
        f_vol_min = os.path.join(self.per_snip_folder, "scene_vol_min.pt")
        f_vol_max = os.path.join(self.per_snip_folder, "scene_vol_max.pt")
        assert os.path.exists(f_vol_min) and os.path.exists(
            f_vol_max
        ), "missing scene volume info"
        self.vol_min = load_tensor(f_vol_min, "cpu").numpy()
        self.vol_max = load_tensor(f_vol_max, "cpu").numpy()
        self.w_min = w_min
        self.w_max = w_max
        self.voxel_res = voxel_res
        self.device = device

        self.vis_norm_grad_occ_thr = 0.2
        # we remove a 1 voxel wide boundary on the volumes to remove cnn artifacts
        self.boundary_thresh = 1

        self.f_occ_preds = sorted(
            glob.glob(os.path.join(self.per_snip_folder, "occ_pr*.pt"))
        )
        Ts_wv_pt = os.path.join(self.per_snip_folder, "Ts_wv.pt")
        self.Ts_wv = torch.load(Ts_wv_pt, map_location="cpu")  # need to be on cpu
        assert (
            len(self.f_occ_preds) == self.Ts_wv.shape[0]
        ), f"occ snippets {len(self.f_occ_preds)} should match with Ts_wv {self.Ts_wv.shape[0]}"

        # load voxel extent for initialization
        ve_path = os.path.join(self.per_snip_folder, "voxel_extent.pt")
        self.local_extent = torch.load(ve_path).cpu()
        if self.local_extent.ndim == 2:
            self.local_extent = self.local_extent.squeeze(0)
        self.local_extent = self.local_extent.tolist()
        self.global_vol = None

        self.init_from_range(self.vol_min, self.vol_max)

    def reinit(self):
        # reinit with the same voxel extent
        if self.global_vol is not None:
            del self.global_vol
        self.init_from_range(self.vol_min, self.vol_max)

    def init_from_range(self, xyz_min, xyz_max):
        # Add a little buffer around the bounds.
        xyz_min -= 2 * self.voxel_res
        xyz_max += 2 * self.voxel_res
        if xyz_min.ndim == 2:
            xyz_min = xyz_min[0]
        if xyz_max.ndim == 2:
            xyz_max = xyz_max[0]

        global_extent = [
            xyz_min[0],
            xyz_max[0],
            xyz_min[1],
            xyz_max[1],
            xyz_min[2],
            xyz_max[2],
        ]
        voxel_size = np.ceil((xyz_max - xyz_min) / self.voxel_res).tolist()
        voxel_size.reverse()  # change to DxHxW
        self.global_vol = VolumeFusion(
            voxel_size,
            global_extent,
            device=self.device,
            w_min=self.w_min,
            w_max=self.w_max,
            init_value=1.0,
            surface_thres=0.99,
        )

    def get_trimesh(self):
        return self.global_vol.get_trimesh()

    def run_step(self, i):
        # run one step of volume fusion
        if i >= len(self.f_occ_preds):
            logger.info(
                f"{i}-th snippet exceeding the number of snippets {len(self.f_occ_preds)}"
            )
            return
        T_wv = self.Ts_wv[i]
        occ_pred = load_tensor(self.f_occ_preds[i], self.device)  # [1, 1, D, H, W]
        occ_pred = occ_pred[0][0]  # [D, H, W]

        self.global_vol.fuse(
            local_volume=occ_pred,
            local_extent=self.local_extent,
            T_l_w=T_wv.inverse(),
        )

    def run(self):
        logger.info("Fusing voxel occupancy using volume fusion...")
        for i, _ in tqdm.tqdm(enumerate(self.f_occ_preds), total=len(self.f_occ_preds)):
            self.run_step(i)
