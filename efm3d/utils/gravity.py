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
import torch.nn.functional as F
from efm3d.aria.pose import PoseTW, rotation_from_euler


GRAVITY_DIRECTION_DLR = np.array([0.0, -1.0, 0.0], np.float32)
GRAVITY_DIRECTION_VIO = np.array([0.0, 0.0, -1.0], np.float32)


def get_transform_to_vio_gravity_convention(gravity_direction: np.array):
    """
    Get transformation to map gravity_direction to (0,0,-1) as per our (and
    VIO/Temple) convention.
    """
    # gravity_direction = (d1, d2, d3) (0,0,-1)^T; d1, d2, d3 column vectors of rotation matrix R_gravity_vio
    # -d3 = gravity_direction
    d3 = -gravity_direction.copy()
    # now construct an orthonormal basis for the rotation matrix
    # d1 is a vector thats orthogonal to gravity_direction by construction
    d1 = np.array(
        [
            gravity_direction[2] - gravity_direction[1],
            gravity_direction[0],
            -gravity_direction[0],
        ]
    )
    # get d2 via orthogonal direction vector to d3 and d1
    d2 = np.cross(d3, d1)
    # get rotation matrix
    R_gravity_vio = np.concatenate(
        [d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis]], 1
    )
    assert (np.linalg.det(R_gravity_vio) - 1.0) < 1e-5
    assert (((R_gravity_vio @ R_gravity_vio.transpose()) - np.eye(3)) < 1e-5).all()
    R_gravity_vio = torch.from_numpy(R_gravity_vio)
    # normalize to unit length
    R_gravity_vio = F.normalize(R_gravity_vio, p=2, dim=-2)
    R_vio_gravity = R_gravity_vio.transpose(1, 0)
    T_vio_gravity = PoseTW.from_Rt(R_vio_gravity, torch.zeros(3))
    return T_vio_gravity


def correct_adt_mesh_gravity(mesh):
    """
    Change gravity direction of ADT mesh
    """
    gravity_direction = np.array([0.0, -1.0, 0.0], np.float32)
    T_vio_gravity = get_transform_to_vio_gravity_convention(gravity_direction).double()
    print("Changing ADT gravity convention to VIO convention.")
    mesh.apply_transform(T_vio_gravity.matrix.numpy())
    return mesh


def reject_vector_a_from_b(a, b):
    # https://en.wikipedia.org/wiki/Vector_projection
    b_norm = torch.sqrt((b**2).sum(-1, keepdim=True))
    b_unit = b / b_norm
    # batched dot product for variable dimensions
    a_proj = b_unit * (a * b_unit).sum(-1, keepdim=True)
    a_rej = a - a_proj
    return a_rej


def gravity_align_T_world_cam(
    T_world_cam, gravity_w=GRAVITY_DIRECTION_VIO, z_grav=False
):
    """
    get T_world_gravity from T_world_cam such that the x axis of T_world_gravity is gravity.
    """
    assert T_world_cam.dim() > 1, f"{T_world_cam} has wrong dimension; expected >1"
    dim = T_world_cam.dim()
    device = T_world_cam.device
    R_wc = T_world_cam.R
    dir_shape = [1] * (dim - 1) + [3]
    g_w = torch.from_numpy(gravity_w.copy()).view(dir_shape).to(R_wc)
    g_w = g_w.expand_as(R_wc[..., 1])
    # forward vector (z) that is orthogonal to gravity direction
    d3 = reject_vector_a_from_b(a=R_wc[..., 2], b=g_w)
    # optionally add a tiny offset to avoid cross product two identical vectors.
    d3_is_zeros = (d3 == 0.0).all(dim=-1).unsqueeze(-1).expand_as(d3)
    d3_offset = torch.zeros(*d3.shape).to(T_world_cam._data.device)
    d3_offset[..., 1] += 0.001
    d3 = torch.where(d3_is_zeros, d3 + d3_offset, d3)
    d2 = torch.linalg.cross(d3, g_w, dim=-1)
    # camera down vector is x direction since Aria cameras are rotated by 90 degree CW
    # hence the new x direction is gravity
    R_wcg = torch.cat([g_w.unsqueeze(-1), d2.unsqueeze(-1), d3.unsqueeze(-1)], -1)
    # normalize to unit length
    R_world_cg = torch.nn.functional.normalize(R_wcg, p=2, dim=-2)
    if z_grav:
        # add extra rotation to make z gravity direction, not x.
        R_cg_cgz = rotation_from_euler(
            torch.tensor([[-np.pi / 2.0, 0.0, np.pi / 2.0]])
        ).to(device)
        R_world_cgz = R_world_cg @ R_cg_cgz.inverse()
        T_world_cgz = PoseTW.from_Rt(R_world_cgz, T_world_cam.t)
        return T_world_cgz
    else:
        R_world_cg = R_world_cg
        T_world_cg = PoseTW.from_Rt(R_world_cg, T_world_cam.t)
        return T_world_cg
