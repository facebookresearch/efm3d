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
import math
from abc import ABC
from typing import List, Literal, Optional

import numpy as np
import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_IMG,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_POINTS_WORLD,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.model.cnn import UpsampleCNN

from efm3d.model.dpt import DPTOri

from efm3d.utils.gravity import gravity_align_T_world_cam, GRAVITY_DIRECTION_VIO
from efm3d.utils.image_sampling import sample_images
from efm3d.utils.pointcloud import pointcloud_to_voxel_counts
from efm3d.utils.ray import sample_depths_in_grid, transform_rays
from efm3d.utils.voxel import create_voxel_grid
from torch.nn import functional as F


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class VideoBackbone3d(torch.nn.Module, ABC):
    """
    Abstract Video Backbone that creates an explicit 3D feature volume from a video stream.
    """

    def __init__(
        self,
        feat_dim: int,
    ):
        """
        Args:
            feat_dim: number of channels in voxel grid, the C in BxCxDxHxW
        """
        super().__init__()
        self._feat_dim = feat_dim

    @property
    def feat_dim(self):
        return self._feat_dim

    def forward_impl(self, batch):
        pass

    def forward(self, batch):
        out = {}

        assert "rgb/feat" in batch, "must run 2d backbone to get rgb feature maps first"

        out.update(self.forward_impl(batch))

        # Shaped B x C x D x H x W
        assert "voxel/feat" in out, "3d backbone must output voxel features"

        # Shaped B x N x W (where N=D*H*W)
        assert "voxel/pts_world" in out, "3d backbone must output voxel positions"

        # Shaped B x 12 (PoseTW object)
        assert "voxel/T_world_voxel" in out, "3d backbone must output voxel coord frame"

        return out


class Lifter(VideoBackbone3d):
    """
    Abstract Video Backbone that creates an explicit 3D feature volume from a set of 2D features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        voxel_size: List[float],
        voxel_extent: List[float],
        head_type: Literal["none", "dpt_ori", "cnn"] = "cnn",
        streams: Optional[List[str]] = None,  # default is just rgb stream
        joint_slam_streams: bool = False,
        joint_streams: bool = False,  # joint all streams
    ):
        """
        Args:
            in_dim: input feature dimension (the 2d image or feature image channel dim)
            out_dim: output feature dimension (in 3d volume - FPN in 2D is used to get to that dim)
            patch_size: size of the patch to use for upsampling
            voxel_size: size of the voxel grid (D H W)
            voxel_extent: extent of the voxel grid (x_min, x_max y_min, y_max, z_min, z_max)
            streams: list of streams to use for the 2D features ("rgb", "slaml", "slamr"). Lifting gets run per stream (unless joint_slam_streams is True) and then concatenated in 3d.
            joint_slam_streams: if True, use the slaml and slamr streams as a single stream (i.e. dont concatenate lifted volumes from the two slam streams - lift them as if they were one camera)
        """

        super().__init__(in_dim)
        self.streams = streams
        if streams is None:
            self.streams = ["rgb"]  # default is just rgb stream
        self.stream2id = {"rgb": 0, "slaml": 1, "slamr": 2}
        # feature map upsampling network
        final_dim = out_dim

        if head_type == "none":
            self.head = None
            self.out_dim = in_dim
        elif head_type == "cnn":
            assert patch_size > 0, f"{patch_size} should be > 0 for UpsampleCNN"
            upsample_power = np.sqrt(patch_size)
            logger.info("True upsample_power: %f" % upsample_power)
            upsample_power = int(round(upsample_power))
            logger.info("Rounded upsample_power: %d" % upsample_power)
            self.head = UpsampleCNN(
                input_dim=in_dim,
                first_hidden_dim=-1,
                final_dim=final_dim,
                upsample_power=upsample_power,
                fix_hidden_dim=False,
            )
            self.out_dim = out_dim
        elif head_type == "dpt_ori":
            self.head = DPTOri(
                input_dim=in_dim,
                output_dim=final_dim,
                depth=False,
            )
            self.out_dim = out_dim
        else:
            raise ValueError(f"{head_type} is not supported")

        self.voxel_size = voxel_size  # D x H x W
        self.voxel_extent = list(voxel_extent)  # W x H x D
        self.joint_streams = joint_streams
        self.joint_slam_streams = (
            joint_slam_streams and "slaml" in self.streams and "slamr" in self.streams
        )

        x_meters = (voxel_extent[1] - voxel_extent[0]) / self.voxel_size[2]
        y_meters = (voxel_extent[3] - voxel_extent[2]) / self.voxel_size[1]
        z_meters = (voxel_extent[5] - voxel_extent[4]) / self.voxel_size[0]
        assert (
            abs(x_meters - y_meters) < 1e-5 and abs(x_meters - z_meters) < 1e-5
        ), f"Voxels should be cubes {x_meters}x{y_meters}x{z_meters}"
        self.voxel_meters = x_meters
        self.num_free_samples = 16

    def output_dim(self):
        num_streams = len(self.streams)
        if self.joint_slam_streams:
            num_streams -= 1
        if self.joint_streams:
            num_streams = 1
        out_dim = 0
        out_dim = self.out_dim * num_streams

        out_dim += 1  # point mask
        out_dim += 1  # freespace token
        return out_dim

    def get_freespace_world(self, batch, batch_idx, T_wv, vW, vH, vD, S=1):
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

        x_min, x_max, y_min, y_max, z_min, z_max = self.voxel_extent
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
            self.voxel_extent,
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

    def get_points_world(self, batch, batch_idx, keep_T=False):
        """
        Get points (semi-dense or GT points) of a snippet in the batch.
        """

        def filter_points(p_w):
            p_w = p_w.reshape(-1, 3)
            # filter out nans
            bad = p_w.isnan().any(dim=-1)
            p_w = p_w[~bad]
            # filter out duplicates from the collapsing of the time dimension
            p_w = torch.unique(p_w, dim=0)
            return p_w

        p_w_Ts = []
        p_w = batch[ARIA_POINTS_WORLD][batch_idx]
        if not keep_T:
            p_w = filter_points(p_w)
        else:
            T = p_w.shape[0]
            for t in range(T):
                p_w_t = p_w[t, ...]
                p_w_t = filter_points(p_w_t)
                p_w_Ts.append(p_w_t)

        if keep_T:
            return p_w_Ts
        else:
            return p_w

    def get_freespace_counts(
        self,
        batch,
        T_wv,
        vW,
        vH,
        vD,
        MAX_NUM_POINTS_VOXEL=50,
        return_mask=False,
    ):
        """
        Get points as voxel grid where each voxel is assigned a count of how many points are inside it.
        If return_mask is trued the function returns the binary occupancy instead of point counts.
        """
        B, T, _, H, W = batch[ARIA_IMG[0]].shape
        point_counts = []
        for b in range(B):
            p_w = self.get_freespace_world(
                batch, b, T_wv[b], vW, vH, vD, self.num_free_samples
            )
            # transform points into voxel coordinate.
            p_v = T_wv[b].inverse() * p_w
            point_count = pointcloud_to_voxel_counts(p_v, self.voxel_extent, vW, vH, vD)
            point_counts.append(point_count)
        point_counts = torch.stack(point_counts, dim=0)  # B x 1 x vD, vH, vW
        # Normalize
        point_counts = (
            point_counts.clamp(0, MAX_NUM_POINTS_VOXEL) / MAX_NUM_POINTS_VOXEL
        )
        if return_mask:
            # Only use as a mask. Comment out if want to use real point counts.
            point_counts[point_counts > 1e-4] = 1.0

        return point_counts

    def get_points_counts(
        self,
        batch,
        T_wv,
        vW,
        vH,
        vD,
        MAX_NUM_POINTS_VOXEL=50,
        return_mask=False,
        keep_T=False,
    ):
        """
        Get points as voxel grid where each voxel is assigned a count of how many points are inside it.
        If return_mask is trued the function returns the binary occupancy instead of point counts.
        """
        B, T, _, H, W = batch[ARIA_IMG[0]].shape
        point_counts = []
        for b in range(B):
            p_w = self.get_points_world(batch, b, keep_T)
            if not keep_T:
                assert isinstance(p_w, torch.Tensor)
                # transform points into voxel coordinate.
                p_v = T_wv[b].inverse() * p_w
                point_count = pointcloud_to_voxel_counts(
                    p_v, self.voxel_extent, vW, vH, vD
                )
            else:
                assert isinstance(p_w, list)
                point_count = []
                for p_w_t in p_w:
                    p_v_t = T_wv[b].inverse() * p_w_t
                    point_count_t = pointcloud_to_voxel_counts(
                        p_v_t, self.voxel_extent, vW, vH, vD
                    )
                    point_count.append(point_count_t)
                point_count = torch.cat(point_count, dim=0)
            point_counts.append(point_count)
        point_counts = torch.stack(point_counts, dim=0)  # B x 1 x vD, vH, vW
        # Normalize
        point_counts = (
            point_counts.clamp(0, MAX_NUM_POINTS_VOXEL) / MAX_NUM_POINTS_VOXEL
        )
        if return_mask:
            # Only use as a mask. Comment out if want to use real point counts.
            point_counts[point_counts > 1e-4] = 1.0

        return point_counts

    def get_voxelgrid_pose(self, cams, T_ws, Ts_sr):
        B, T = cams.shape[:2]
        Ts_wr = T_ws @ Ts_sr
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()  # Ts_world_cam
        # Select last frame in snippet.
        selectT = torch.tensor(T - 1).repeat(B).long()

        # Create the voxel grid by aligning selected frame with gravity.
        T_wc_select = Ts_wc[torch.arange(B), selectT, :]
        T_wv = gravity_align_T_world_cam(
            T_wc_select, gravity_w=GRAVITY_DIRECTION_VIO, z_grav=True
        )
        # T_wv should only have yaw value
        rpy = T_wv.to_euler()
        assert torch.allclose(torch.tensor(0.0), rpy[:, :2], atol=1e-4)
        return T_wv, selectT

    def lift(self, feats2d, vox_w, cam, Ts_wr, vD, vH, vW):
        B, T = cam.shape[:2]
        F = feats2d.shape[2]
        Ts_wc = Ts_wr @ cam.T_camera_rig.inverse()  # Ts_world_cam
        vox_w = torch.flatten(vox_w, 0, 1)
        cam = torch.flatten(cam, 0, 1)
        Ts_wc = torch.flatten(Ts_wc, 0, 1)
        feats2d = torch.flatten(feats2d, 0, 1)

        vox_cam = Ts_wc.inverse() * vox_w
        vox_feats, vox_valid = sample_images(
            feats2d, vox_cam, cam, n_by_c=False, warn=False, single_channel_mask=True
        )
        vox_feats = vox_feats.reshape(B, T, F, vD, vH, vW)
        vox_valid = vox_valid.reshape(B, T, 1, vD, vH, vW)
        return vox_feats, vox_valid

    def aggregate(self, vox_feats, vox_valid):
        def basic_mean(x, dim, valid, keepdim=False):
            count = torch.sum(valid, dim=dim, keepdim=True)  # B 1 C D H W
            invalid = (~valid).expand_as(x)
            x[invalid] = 0.0
            x_sum = torch.sum(x, dim=dim, keepdim=True)
            count[count == 0] = 1.0  # just so we dont divide by zero
            mean = x_sum / count
            del x_sum
            mean[count.expand_as(mean) < 1] = 0.0
            if not keepdim:
                return mean.squeeze(dim), count.squeeze(dim)
            return mean, count

        vox_feats, count_feats_m = basic_mean(
            vox_feats, 1, valid=vox_valid, keepdim=False
        )
        return vox_feats, count_feats_m[:, [0]]

    def lift_aggregate_centers(self, batch, feats2d, vox_w, Ts_wr, T_wv=None):
        vD, vH, vW = self.voxel_size
        B, T = batch[ARIA_IMG[0]].shape[:2]
        # Lift to 3D. Project 3D voxel centers into each image and sample.
        vox_w = vox_w.reshape(B, 1, -1, 3).repeat(1, T, 1, 1)
        vox_feats, vox_valid, stream2pos = [], [], {}
        for stream in self.streams:
            stream_id = self.stream2id[stream]
            cam = batch[ARIA_CALIB[stream_id]]
            _vox_feats, _vox_valid = self.lift(
                feats2d[stream], vox_w, cam, Ts_wr, vD, vH, vW
            )
            stream2pos[stream] = len(vox_feats)
            vox_feats.append(_vox_feats)
            vox_valid.append(_vox_valid)
        if self.joint_slam_streams:
            vox_feats_rgb, vox_valid_rgb = None, None
            if "rgb" in stream2pos:
                i = stream2pos["rgb"]
                vox_feats_rgb, vox_valid_rgb = vox_feats[i], vox_valid[i]
            vox_feats_slam = [
                vox_feats[stream2pos[stream]] for stream in ["slaml", "slamr"]
            ]
            vox_valid_slam = [
                vox_valid[stream2pos[stream]] for stream in ["slaml", "slamr"]
            ]
            vox_feats_slam = torch.cat(vox_feats_slam, 1)
            vox_valid_slam = torch.cat(vox_valid_slam, 1)
            count_feats = torch.sum(vox_valid_slam, dim=1, keepdim=True)  # B 1 C D H W
            if vox_valid_rgb is not None:
                count_feats = count_feats + torch.sum(
                    vox_valid_slam, dim=1, keepdim=True
                )
            vox_feats_m, vox_valid_m = vox_feats_slam, vox_valid_slam
            vox_feats, count_feats_m = self.aggregate(vox_feats_m, vox_valid_m)
            if vox_valid_rgb is not None:
                vox_feats_m, vox_valid_m = vox_feats_rgb, vox_valid_rgb
                vox_feats_rgb, count_feats_rgb_m = self.aggregate(
                    vox_feats_m, vox_valid_m
                )
                vox_feats = torch.cat([vox_feats, vox_feats_rgb], 1)
                count_feats_m = count_feats_m + count_feats_rgb_m
        elif self.joint_streams:
            vox_feats = torch.cat(vox_feats, 1)
            vox_valid = torch.cat(vox_valid, 1)
            # Sum up number of valid projections into each camera for each voxel.
            count_feats = torch.sum(vox_valid, dim=1, keepdim=True)  # B 1 C D H W
            vox_feats_m, vox_valid_m = vox_feats, vox_valid
            vox_feats, count_feats_m = self.aggregate(vox_feats_m, vox_valid_m)
        else:
            # concat lifted volumes for all selected video streams
            vox_feats = torch.cat(vox_feats, 2)
            vox_valid = torch.cat(vox_valid, 2)  # B T C D H W
            # Sum up number of valid projections into each camera for each voxel.
            count_feats = torch.sum(vox_valid, dim=1, keepdim=True)  # B 1 C D H W
            vox_feats_m, vox_valid_m = vox_feats, vox_valid
            vox_feats, count_feats_m = self.aggregate(vox_feats_m, vox_valid_m)
        count_feats = count_feats[:, :, 0]
        assert count_feats.shape == (B, 1, vD, vH, vW), f"{count_feats.shape}"
        assert count_feats_m.shape == (B, 1, vD, vH, vW), f"{count_feats_m.shape}"
        return vox_feats, count_feats, count_feats_m

    def forward(self, batch):
        B, T, _, H, W = batch[ARIA_IMG[0]].shape

        # Run CNN on EFM features to features back up to full resolution.
        feats2d = {}
        tokens2d = {}
        for stream in self.streams:
            feats2d[stream] = batch[f"{stream}/feat"]
            # for visualizations
            if not isinstance(feats2d[stream], list):
                tokens2d[stream] = feats2d[stream].detach().cpu()
            else:
                # multi-layer 2d features. Needed by DPT head in Lifter
                tokens2d[stream] = [f.detach().cpu() for f in feats2d[stream]]
            if self.head:
                feats2d[stream] = self.head.forward(feats2d[stream])

        # Compute voxel grid pose.
        cams = batch[ARIA_CALIB[0]]
        device = cams.device
        T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET]  # T_world_rig (one per snippet)
        Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]]  # Ts_snippet_rig (T per snippet)
        Ts_wr = T_ws @ Ts_sr
        T_wv, selectT = self.get_voxelgrid_pose(cams, T_ws, Ts_sr)

        # Generate voxel grid.
        vD, vH, vW = self.voxel_size
        point_info = []
        point_masks = self.get_points_counts(batch, T_wv, vW, vH, vD, return_mask=True)
        point_info.append(point_masks)
        free_masks = self.get_freespace_counts(
            batch, T_wv, vW, vH, vD, return_mask=True
        )
        point_info.append(free_masks)
        vox_v_orig = create_voxel_grid(vW, vH, vD, self.voxel_extent, device)
        vox_v_orig = vox_v_orig.permute(2, 1, 0, 3)  # D H W 3
        vox_v = vox_v_orig.reshape(-1, 3)
        vox_v = vox_v.unsqueeze(0).repeat(B, 1, 1)
        vox_w = T_wv * vox_v
        vox_w = vox_w.reshape(B, vD, vH, vW, 3)
        vox_w = vox_w.reshape(B, -1, 3)  # B DHW 3

        if len(feats2d) > 0:
            # Lift image features to 3D. Project 3D voxel centers into each
            # image and sample.
            vox_feats, count_feats, count_feats_m = self.lift_aggregate_centers(
                batch,
                feats2d,
                vox_w,
                Ts_wr,
                T_wv,
            )
            vox_feats = torch.concatenate([vox_feats] + point_info, dim=1)
        else:
            vox_feats = torch.concatenate(point_info, dim=1)
            count_feats = torch.ones(B, 1, vD, vH, vW, device=device)
            count_feats_m = torch.ones(B, 1, vD, vH, vW, device=device)
        out = {}

        # Don't use the masked out versions (_m) because loss functions later on need these.
        for stream, feat2d in feats2d.items():
            out[f"{stream}/feat2d_upsampled"] = feat2d
        for stream, token2d in tokens2d.items():
            out[f"{stream}/token2d"] = token2d

        out["voxel/feat"] = vox_feats  # B x F x D x H x W
        out["voxel/counts"] = count_feats[:, 0]  # B x D x H x W
        # Pass the masked version of counts for debugging.
        out["voxel/counts_m"] = count_feats_m[:, 0]  # B x D x H x W
        # We don't need the repeat across time anymore.
        vox_w = vox_w.reshape(B, vD * vH * vW, 3)
        out["voxel/pts_world"] = vox_w  # B x N x 3 (N=D*H*W)
        out["voxel/T_world_voxel"] = T_wv  # B x 12
        out["voxel/selectT"] = selectT  # B x 1 (frame that voxel grid is anchored to)
        out["voxel/occ_input"] = point_info[0]
        return out
