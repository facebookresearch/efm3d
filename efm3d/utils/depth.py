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
from efm3d.utils.ray import ray_grid


def dist_im_to_point_cloud_im(dist_m, cams):
    B, T = None, None
    if cams.ndim == 3:
        B, T, _ = cams.shape
        cams = cams.view(B * T, -1)
        dist_m = dist_m.flatten(0, 1)
    elif cams.ndim == 2:
        B, _ = cams.shape
    elif cams.ndim == 1:
        cams = cams.view(1, -1)
        H, W = dist_m.shape
        dist_m = dist_m.view(1, H, W)
    BT, H, W = dist_m.shape
    rays_rig, valids = ray_grid(cams)
    p3s_rig = rays_rig[..., :3] + rays_rig[..., 3:] * dist_m.unsqueeze(-1)
    p3s_c = cams.T_camera_rig * p3s_rig.view(BT, -1, 3)
    # distances > 0.0 are valid
    valids = torch.logical_and(valids, dist_m > 0.0)

    if T is not None:
        p3s_c = p3s_c.view(B, T, H, W, 3)
        valids = valids.view(B, T, H, W)
    elif B is not None:
        p3s_c = p3s_c.view(B, H, W, 3)
        valids = valids.view(B, H, W)
    else:
        p3s_c = p3s_c.view(H, W, 3)
        valids = valids.view(H, W)
    return p3s_c, valids
