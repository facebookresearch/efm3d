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
from typing import Dict, List, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F

from .tensor_wrapper import autocast, autoinit, smart_stack, TensorWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

IdentityPose = torch.tensor(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
).reshape(12)

PAD_VAL = -1


def get_T_rot_z(angle: float):
    T_rot_z = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return torch.from_numpy(T_rot_z).float()


def skew_symmetric(v):
    """Create a skew-symmetric matrix from a (batched) vector of size (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    M = torch.stack(
        [
            z,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            z,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            z,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))
    return M


def inv_skew_symmetric(V):
    """Create a (batched) vector from a skew-symmetric matrix of size (..., 3, 3)."""
    # average lower and uper triangular entries in case skew symmetric matrix
    # has numeric errors.
    VVT = 0.5 * (V - V.transpose(-2, -1))
    return torch.stack(
        [
            -VVT[..., 1, 2],
            VVT[..., 0, 2],
            -VVT[..., 0, 1],
        ],
        -1,
    )


def so3exp_map(w, eps: float = 1e-7):
    """Compute rotation matrices from batched twists.
    Args:
        w: batched 3D axis-angle vectors of size (..., 3).
    Returns:
        A batch of rotation matrices of size (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def so3log_map(R, eps: float = 1e-7):
    trace = torch.diagonal(R, dim1=-1, dim2=-2).sum(-1)
    cos = torch.clamp((trace - 1.0) * 0.5, -1, 1)
    theta = torch.acos(cos).unsqueeze(-1).unsqueeze(-1)
    ones = torch.ones_like(theta)
    small = theta < eps
    # compute factors and approximate them around 0 using second order
    # taylor expansion (from WolframAlpha)
    theta_over_sin_theta = torch.where(
        small,
        ones - (theta**2) / 6.0 + 7.0 * (theta**4) / 360.0,
        theta / torch.sin(theta),
    )
    # compute log-map W of rotation R first
    W = 0.5 * theta_over_sin_theta * (R - R.transpose(-1, -2))
    omega = inv_skew_symmetric(W)
    return omega


def interpolation_boundaries_alphas(times: torch.Tensor, interp_times: torch.Tensor):
    """
    find the ids in times tensor that bound each of the interp_times timestamps
    from below (lower_ids) and above (upper_ids).
    If interp_times are outside the interval spanned by times, upper and lower
    ids will both point to the boundary timestamps and the returned good boolean
    tensor will be False at those interpolation timestamps.

    Also return the alphas needed to interpolate a value as:
    interp_value = alpha * value[lower_id] + (1-alpha)* value[upper_id]

    Note that because the upper and lower ids are pointing to the boundary
    timestamps when the interpolation time is outside the time interval,
    applying the formula above will yield the values at the boundaries as a
    reasonable "interpolation". No extrapolation will be performed. Again the
    good values can be used to check which values are at the boundaries and
    which ones are interpolated.
    """
    times = times.unsqueeze(-2)
    interp_times = interp_times.unsqueeze(-1)
    dt = times - interp_times
    if dt.dtype == torch.long:
        dt_max = torch.iinfo(type=dt.dtype).max
    else:
        dt_max = torch.finfo(type=dt.dtype).max
    dt_upper = torch.where(dt < 0.0, torch.ones_like(dt) * dt_max, dt)
    dt_lower = torch.where(dt > 0.0, torch.ones_like(dt) * dt_max, -dt)
    upper_alpha, upper_ids = torch.min(dt_upper, dim=-1)
    lower_alpha, lower_ids = torch.min(dt_lower, dim=-1)
    good = torch.logical_and(lower_alpha < dt_max, upper_alpha < dt_max)
    upper_ids = torch.where(good, upper_ids, torch.maximum(lower_ids, upper_ids))
    lower_ids = torch.where(good, lower_ids, torch.minimum(lower_ids, upper_ids))
    assert (lower_ids <= upper_ids).all()
    # nan_to_num handles the case where time and interpolation time are the same
    # and hence this is a 0/0
    # okay to go to floats now since the critical bit is the computation of the time difference
    alpha = torch.nan_to_num(lower_alpha.float() / (lower_alpha + upper_alpha).float())
    alpha = torch.where(good, alpha, torch.zeros_like(alpha))
    return lower_ids, upper_ids, alpha, good


def quaternion_to_matrix(quaternions_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices. Input quaternions
    should be in wxyz format, with real part first, imaginary part last.

    The function is copied from `quaternion_to_matrix` in Pytorch3d:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions_wxyz, -1)
    two_s = 2.0 / (quaternions_wxyz * quaternions_wxyz).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions_wxyz.shape[:-1] + (3, 3))


class PoseTW(TensorWrapper):
    @autocast
    @autoinit
    def __init__(self, data: torch.Tensor = IdentityPose):
        assert isinstance(data, torch.Tensor)
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_qt(cls, quaternion_wxyz: torch.Tensor, t: torch.Tensor):
        """Pose from quaternion and translation vectors. Quaternion should
        be wxyz format, with real part first, and imaginary part last.

        Args:
            quaternion: quaternion with shape (..., 4).
            t: translation vector with shape (..., 3).
        """
        assert (
            quaternion_wxyz.shape[:-1] == t.shape[:-1]
        ), f"quaternion shape {quaternion_wxyz.shape[:-1]} must match translation shape {t.shape[:-1]} expect the last dim"
        assert quaternion_wxyz.shape[-1] == 4, "quaternion must be of shape (..., 4)"
        assert t.shape[-1] == 3, "translation must be of shape (..., 3)"

        R = quaternion_to_matrix(quaternion_wxyz)
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    @autocast
    def from_aa(cls, aa: torch.Tensor, t: torch.Tensor):
        """Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            aa: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        """
        assert aa.shape[-1] == 3
        assert t.shape[-1] == 3
        assert aa.shape[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(aa), t)

    @classmethod
    @autocast
    def from_matrix(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    @autocast
    def from_matrix3x4(cls, T_3x4: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 3, 4).
        """
        assert T_3x4.shape[-2:] == (3, 4)
        R, t = T_3x4[..., :3, :3], T_3x4[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    @autocast
    def exp(cls, u_omega: torch.Tensor, eps: float = 1e-7):
        """
        Compute the SE3 exponential map from input se3 vectors u_omega [....,6] where
        the last 3 entries are the so3 entires omega and the first 3 the entries
        for translation.
        """
        # following https://www.ethaneade.com/lie.pdf and http://people.csail.mit.edu/jstraub/download/straubTransformationCookbook.pdf
        u = u_omega[..., :3]
        omega = u_omega[..., 3:]
        theta = omega.norm(p=2, dim=-1, keepdim=True).unsqueeze(-1)
        small = theta < eps
        R = so3exp_map(omega, eps)
        # compute V
        shape = [1] * len(omega.shape[:-1])
        ones = torch.ones_like(theta)
        # compute factors and approximate them around 0 using second order
        # taylor expansion (from WolframAlpha)
        b = torch.where(
            small,
            0.5 * ones - theta**2 / 24.0 + theta**4 / 720.0,
            (ones - torch.cos(theta)) / theta**2,
        )
        c = torch.where(
            small,
            1.0 / 6.0 * ones - theta**2 / 120.0 + theta**4 / 5040.0,
            (theta - torch.sin(theta)) / theta**3,
        )
        Identity = (
            torch.eye(3).reshape(shape + [3, 3]).repeat(shape + [1, 1]).to(u_omega)
        )
        W = skew_symmetric(omega)
        V = Identity + b * W + c * W @ W
        # compute t
        t = (V @ u.unsqueeze(-1)).squeeze(-1)
        return cls.from_Rt(R, t)

    # @classmethod
    # def from_colmap(cls, image: NamedTuple):
    #    '''Pose from a COLMAP Image.'''
    #    return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self._data[..., -3:]

    @property
    def q(self) -> torch.Tensor:
        """
        Convert rotations of shape (..., 3, 3) to a quaternion (..., 4).
        The returned quaternions have real part first, as wxyz.
        The function is adapted from `matrix_to_quaternion` in Pytorch3d:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

        The major difference to the original pytorch3d function is that the returned
        quaternions are normalized and have positive real part.
        """

        def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
            """
            Returns torch.sqrt(torch.max(0, x))
            but with a zero subgradient where x is 0.
            """
            ret = torch.zeros_like(x)
            positive_mask = x > 0
            ret[positive_mask] = torch.sqrt(x[positive_mask])
            return ret

        matrix = self.R
        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_wxyz = torch.stack(
            [
                torch.stack(
                    [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
                ),
                torch.stack(
                    [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
                ),
                torch.stack(
                    [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
                ),
                torch.stack(
                    [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
                ),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_wxyz / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)
        best_quat = quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))

        # normalize quaternions and make the real part to be positive for all quaternions
        best_quat = best_quat.reshape(-1, 4)
        neg_ind = torch.nonzero(best_quat[:, 0] < 0).squeeze()
        best_quat[neg_ind, :] *= -1
        best_quat = best_quat.reshape(batch_dim + (4,))
        best_quat_normalized = F.normalize(best_quat, p=2, dim=-1)
        return best_quat_normalized

    @property
    def q_xyzw(self) -> torch.Tensor:
        """
        Get the quaternion representation similar to self.q, but the real part
        of the quaternion comes last rather than first. This is a handy function to increase
        interoperability, e.g. lietorch requires xyzw quaternions.
        """
        quat_wxyz = self.q
        return torch.concat([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], dim=-1)

    @property
    def matrix3x4(self) -> torch.Tensor:
        """Underlying transformation matrix with shape (..., 3, 4)."""
        rvec = self._data[..., :9]
        rmat = rvec.reshape(rvec.shape[:-1] + (3, 3))
        tvec = self._data[..., -3:].unsqueeze(-1)
        T = torch.cat([rmat, tvec], dim=-1)
        return T

    @property
    def matrix(self) -> torch.Tensor:
        """Underlying transformation matrix with shape (..., 4, 4)."""
        T_3x4 = self.matrix3x4
        bot_row = T_3x4.new_zeros(T_3x4.shape[:-2] + (1, 4))
        bot_row[..., 0, 3] = 1
        return torch.cat([T_3x4, bot_row], dim=-2)

    def to_euler(self, rad=True) -> torch.Tensor:
        """Convert the rotation matrix to Euler angles using ZYX convention."""
        """Reference: http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf"""
        # Test gimbal lock (ignore rotations that are all PAD_VAL).
        is_pad = torch.all(torch.all(self.R == PAD_VAL, dim=-1), dim=-1)
        assert (~torch.abs(self.R[~is_pad][..., 2, 0]).isclose(torch.tensor(1.0))).all()
        Y_angle = -torch.asin(self.R[..., 2, 0])
        euler_angles = (
            torch.atan2(self.R[..., 2, 1], self.R[..., 2, 2]),
            Y_angle,
            torch.atan2(self.R[..., 1, 0], self.R[..., 0, 0]),
        )
        if not rad:
            # return degree
            return torch.stack(euler_angles, -1) * 180.0 / torch.pi
        return torch.stack(euler_angles, -1)

    def to_ypr(self, rad=True) -> torch.Tensor:
        # yaw, pitch, roll from rotation matrix: http://lavalle.pl/planning/node103.html
        R = self.R
        yaw = torch.atan(R[..., 1, 0] / R[..., 0, 0])
        pitch = torch.atan(
            -R[..., 2, 0]
            / torch.sqrt(R[..., 2, 1] * R[..., 2, 1] + R[..., 2, 2] * R[..., 2, 2])
        )
        roll = torch.atan(R[..., 2, 1] / R[..., 2, 2])
        return yaw, pitch, roll

    def inverse(self) -> "PoseTW":
        """Invert an SE(3) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "PoseTW") -> "PoseTW":
        """Chain two SE(3) poses: T_C_B.compose(T_B_A) -> T_C_A."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        # use more efficient right multiply that avoids transpose of the points
        # according to the equality:
        # (Rp + t)^T = (Rp)^T + t^T = p^T R^T + t^T
        # where p^T = p3d, R = self.R and t = self.t
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    @autocast
    def batch_transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points each by the associated (in batch
        dimensions) transform in this PoseTW.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape == self.t.shape, f"shapes of p3d {p3d.shape}, t {self.t.shape}"
        # bmm assumes one batch dimension
        assert p3d.dim() == 2, f"{p3d.shape}"
        assert self.ndim == 2, f"{self.shape}"
        # use more efficient right multiply that avoids transpose of the points
        # according to the equality:
        # (Rp + t)^T = (Rp)^T + t^T = p^T R^T + t^T
        # where p^T = p3d, R = self.R and t = self.t
        return (
            torch.bmm(p3d.unsqueeze(-2), self.R.transpose(-1, -2)).squeeze(-2) + self.t
        )

    @autocast
    def rotate(self, p3d: torch.Tensor) -> torch.Tensor:
        """Rotate a set of 3D points. Useful for directional vectors which should not be translated.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        # use more efficient right multiply that avoids transpose of the points
        # according to the equality:
        # (Rp)^T = p^T R^T where p3d = p^T and self.R = R
        return p3d @ self.R.transpose(-1, -2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points: T_B_A * p3D_A -> p3D_B"""
        return self.transform(p3D)

    def __matmul__(self, other: "PoseTW") -> "PoseTW":
        """Chain two SE(3) poses: T_C_B @ T_B_A -> T_C_A."""
        return self.compose(other)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self, deg=True, eps=0) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(3) transformation. The `eps` has to be
        positive if you want to use this function as part of a training loop.

        Returns:
            dr: rotation angle in degrees (if deg=True) or in radians.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, min=-1.0 + eps, max=1.0 - eps)
        dr = torch.acos(cos)
        if deg:
            dr = dr * 180.0 / math.pi
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def so3_geodesic(self, other: "PoseTW", deg=False) -> "PoseTW":
        """Compute the geodesic distance for rotation between this pose and another pose"""
        pose_e = self.compose(other.inverse())
        dr, _ = pose_e.magnitude(deg=deg, eps=1e-6)
        return dr

    def log(self, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute the SE3 log map for these poses.
        Returns [...,6] where the last 3 entries are the so3 entires omega and
        the first 3 the entries for translation.
        """
        # following https://www.ethaneade.com/lie.pdf and http://people.csail.mit.edu/jstraub/download/straubTransformationCookbook.pdf
        R, t = self.R, self.t
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1.0) * 0.5, -1, 1)
        theta = torch.acos(cos).unsqueeze(-1).unsqueeze(-1)
        ones = torch.ones_like(theta)
        small = theta < eps
        # compute factors and approximate them around 0 using second order
        # taylor expansion (from WolframAlpha)
        theta_over_sin_theta = torch.where(
            small,
            ones - (theta**2) / 6.0 + 7.0 * (theta**4) / 360.0,
            theta / torch.sin(theta),
        )
        c = torch.where(
            small,
            0.08333333 + 0.001388889 * theta**2 + 0.0000330688 * theta**4,
            (ones - ((0.5 * theta * torch.sin(theta)) / (ones - torch.cos(theta))))
            / theta**2,
        )
        # compute log-map W of rotation R first
        W = 0.5 * theta_over_sin_theta * (R - R.transpose(-1, -2))
        # compute V_inv to be able to get u
        shape = [1] * len(R.shape[:-2])
        Identity = (
            torch.eye(3).reshape(shape + [3, 3]).repeat(shape + [1, 1]).to(self._data)
        )
        V_inv = Identity - 0.5 * W + c * W @ W
        u = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
        omega = inv_skew_symmetric(W)
        return torch.cat([u, omega], -1)

    def interpolate(self, times: torch.Tensor, interp_times: torch.Tensor):
        """
        Return poses at the given interpolation times interp_times based on the
        poses in this object and the provided associated timestamps times.

        If interpolation timestamps are outside the interval of times, the poses
        at the interval boundaries will be returned and the good boolean tensor
        will indicate those boundary values with a False.
        """
        assert (
            times.shape == self._data.shape[:-1]
        ), f"time stamps for the poses do not match poses shape {times.shape} vs {self._data.shape}"

        assert (
            times.dim() <= 2
        ), "The shape of the input times should be either BxT or T."
        times = times.to(self.device)
        interp_times = interp_times.to(self.device)
        # find the closest timestamps above and below for each interp_times in times
        lower_ids, upper_ids, alpha, good = interpolation_boundaries_alphas(
            times, interp_times
        )
        # get the bounding poses
        upper_ids = upper_ids.unsqueeze(-1)
        upper_ids = upper_ids.expand(*upper_ids.shape[0:-1], self._data.shape[-1])
        lower_ids = lower_ids.unsqueeze(-1)
        lower_ids = lower_ids.expand(*lower_ids.shape[0:-1], self._data.shape[-1])
        T_upper = self.__class__(self._data.gather(times.dim() - 1, upper_ids))
        T_lower = self.__class__(self._data.gather(times.dim() - 1, lower_ids))
        # get se3 element connecting the lower and upper poses
        dT = T_lower.inverse() @ T_upper
        dx = dT.log()
        # interpolate on se3
        dT = self.exp(dx * alpha.unsqueeze(-1))
        return T_lower @ dT, good

    def align(self, other, self_times=None, other_times=None):
        """Align two trajectories using the method of Horn (closed-form).

        Input:
            other -- second PoseTW (Nx12) trajectory to align to

        Output:
            T_self_other -- relative SE3 transform (Nx12)
            trans_error -- translational error per point (Nx1)

        code inspired by: https://github.com/symao/vio_evaluation/blob/master/align.py#L6-L38
        """
        if self.t.ndim != 2:
            raise ValueError(
                "Only Nx12 Pose supported in alignment, given {self.shape}"
            )
        if other.t.ndim != 2:
            raise ValueError(
                "Only Nx12 Pose supported in alignment, given {other.shape}"
            )
        dtype = torch.promote_types(self.dtype, other.dtype)

        # Optionally interpolate other to match the size of self.
        if self.shape[0] != other.shape[0]:
            if self_times is None or other_times is None:
                raise ValueError(
                    "Got different length PoseTW (self {self.shape} and other {other.shape}). Must provide timestamps to support interpolation"
                )
            # Do interpolation on temporal intersection.
            other, goods = other.interpolate(other_times, self_times)
            self2 = self.clone()[goods].to(dtype)
            other2 = other.clone()[goods].to(dtype)
        else:
            self2 = self.clone().to(dtype)
            other2 = other.clone().to(dtype)

        P = self2.t.transpose(0, 1)
        Q = other2.t.transpose(0, 1)

        if P.shape != Q.shape:
            raise ValueError("Matrices P and Q must be of the same dimensionality")

        centroids_P = torch.mean(P, dim=1)
        centroids_Q = torch.mean(Q, dim=1)
        A = P - torch.outer(centroids_P, torch.ones(P.shape[1], dtype=dtype))
        B = Q - torch.outer(centroids_Q, torch.ones(Q.shape[1], dtype=dtype))
        C = A @ B.transpose(0, 1)
        U, S, V = torch.linalg.svd(C)
        R = V.transpose(0, 1) @ U.transpose(0, 1)
        L = torch.eye(3, dtype=dtype)
        if torch.linalg.det(R) < 0:
            L[2][2] *= -1

        R = V.transpose(0, 1) @ (L @ U.transpose(0, 1))
        t = (-R @ centroids_P) + centroids_Q
        T_self_other = PoseTW.from_Rt(R, t).inverse().to(dtype)

        other_aligned = T_self_other @ other2

        error = torch.linalg.norm(other_aligned.t - self2.t, dim=-2)
        mean_error = error.mean(dim=-1)

        return T_self_other, mean_error

    def fit_to_SO3(self):
        # Math used from quora post and this berkeley pdf.
        # https://qr.ae/pKQaG5
        # https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
        assert self._data.ndim == 1
        Q = fit_to_SO3(self.R)
        return PoseTW.from_Rt(Q, self.t)

    def __repr__(self):
        return f"PoseTW: {self.shape} {self.dtype} {self.device}"


def interpolate_timed_poses(
    timed_poses: Dict[
        Union[float, int],
        Union[PoseTW, List[PoseTW], Dict[Union[float, int, str], PoseTW]],
    ],
    time: Union[float, int],
):
    """
    interpolate timed poses given as a dict[time:container[PoseTW]] to given
    time.  The poses container indexed by time can be given plain as poses, or a
    list or dict of poses.  If a list or dict of poses is given, the output will
    also be a list or dict of the interpolated poses. This allows batched
    interpolation.
    """
    ts_list = list(timed_poses.keys())
    ts = torch.from_numpy(np.array(ts_list))
    interp_time = torch.from_numpy(np.array([time]))
    lower_ids, upper_ids, _, _ = interpolation_boundaries_alphas(ts, interp_time)
    t_lower = ts_list[lower_ids[0]]
    t_upper = ts_list[upper_ids[0]]
    poses_lower, poses_upper = timed_poses[t_lower], timed_poses[t_upper]
    poses_interp = None
    times = torch.from_numpy(np.array([t_lower, t_upper])).float()
    if isinstance(poses_lower, PoseTW):
        poses = PoseTW(smart_stack([poses_lower, poses_upper]))
        if poses.dim() == 3:
            times = times.unsqueeze(-1).repeat(1, poses.shape[1])
        poses_interp = poses.interpolate(times, interp_time)[0].squeeze()
    elif isinstance(poses_lower, dict):
        keys_lower = set(poses_lower.keys())
        keys_upper = set(poses_upper.keys())
        keys = keys_lower & keys_upper
        poses_interp = {}
        for key in keys:
            poses = PoseTW(smart_stack([poses_lower[key], poses_upper[key]]))
            if poses.dim() == 3 and times.dim() == 1:
                times = times.unsqueeze(-1).repeat(1, poses.shape[1])
            poses_interp[key] = poses.interpolate(times, interp_time)[0].squeeze()
    elif isinstance(poses_lower, list):
        assert len(poses_lower) == len(poses_upper)
        poses_interp = []
        for i in range(len(poses_lower)):
            poses = PoseTW(smart_stack([poses_lower[i], poses_upper[i]]))
            if poses.dim() == 3 and times.dim() == 1:
                times = times.unsqueeze(-1).repeat(1, poses.shape[1])
            poses_interp.append(poses.interpolate(times, interp_time)[0].squeeze())
    return poses_interp


def lower_timed_poses(
    timed_poses: Dict[
        Union[float, int],
        Union[PoseTW, List[PoseTW], Dict[Union[float, int, str], PoseTW]],
    ],
    time: Union[float, int],
):
    """
    interpolate timed poses given as a dict[time:container[PoseTW]] to given
    time.  The poses container indexed by time can be given plain as poses, or a
    list or dict of poses.  If a list or dict of poses is given, the output will
    also be a list or dict of the interpolated poses. This allows batched
    interpolation.
    """
    ts_list = list(timed_poses.keys())
    ts = torch.from_numpy(np.array(ts_list))
    interp_time = torch.from_numpy(np.array([time]))
    lower_ids, _, alpha, good = interpolation_boundaries_alphas(ts, interp_time)
    t_lower = ts_list[lower_ids[0]]
    poses_lower = timed_poses[t_lower]
    return poses_lower, t_lower - time


def closest_timed_poses(
    timed_poses: Dict[
        Union[float, int],
        Union[PoseTW, List[PoseTW], Dict[Union[float, int, str], PoseTW]],
    ],
    time: Union[float, int],
):
    """
    interpolate timed poses given as a dict[time:container[PoseTW]] to given
    time.  The poses container indexed by time can be given plain as poses, or a
    list or dict of poses.  If a list or dict of poses is given, the output will
    also be a list or dict of the interpolated poses. This allows batched
    interpolation.
    """
    ts_list = list(timed_poses.keys())
    ts = torch.from_numpy(np.array(ts_list))
    interp_time = torch.from_numpy(np.array([time]))
    lower_ids, upper_ids, alpha, good = interpolation_boundaries_alphas(ts, interp_time)
    t_lower = ts_list[lower_ids[0]]
    t_upper = ts_list[upper_ids[0]]
    poses_lower, poses_upper = timed_poses[t_lower], timed_poses[t_upper]
    if time - t_lower < t_upper - time:
        return poses_lower, time - t_lower
    else:
        return poses_upper, t_upper - time


def all_rot90():
    # construct all possible 90 degree rotations
    dirs = torch.cat([torch.eye(3), -torch.eye(3)], dim=0)
    ids = torch.arange(0, 6).long()
    jds = torch.arange(0, 6).long()
    ids, jds = torch.meshgrid(ids, jds)
    ids, jds = ids.reshape(-1), jds.reshape(-1)
    a, b = dirs[ids, :], dirs[jds, :]
    c = torch.cross(a, b, -1)
    Rs = torch.cat([a.unsqueeze(2), b.unsqueeze(2), c.unsqueeze(2)], dim=2)
    # filter to valid rotations
    det = torch.linalg.det(Rs)
    Rs = Rs[det > 0.99]
    return Rs


def find_r90(Ta, Tb, R90s):
    N = None
    if Tb.ndim == 2:
        N = Tb.shape[0]
        # 24xNx3x3
        R90s = R90s.unsqueeze(1).repeat(1, N, 1, 1)
    Ra_inv, Rb = Ta.inverse().R.unsqueeze(0), Tb.R
    # 24x(Nx)3x3
    dR = Ra_inv @ Rb.unsqueeze(0) @ R90s
    w = so3log_map(dR)
    ang = torch.linalg.norm(w, 2, dim=-1)
    ang_min, id_min = torch.min(ang, dim=0)
    if N is None:
        R90min = R90s[id_min]
    else:
        R90min = R90s[id_min, torch.arange(N)]
    Rb = Rb @ R90min
    Tb = PoseTW.from_Rt(Rb, Tb.t)
    return Tb, R90min


def stereographic_unproject(a, axis=None):
    """
    Inverse of stereographic projection: https://en.wikipedia.org/wiki/Stereographic_projection
    This is from the paper "On the Continuity of Rotation Representations in Neural
    Networks" https://arxiv.org/pdf/1812.07035.pdf, equation [8,9],
    used in rotation_from_ortho_5d.
    """
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)
    ans = torch.autograd.Variable(torch.zeros(batch, a.shape[1] + 1).to(a))
    unproj = 2 * a / (s2 + 1).reshape(batch, 1).repeat(1, a.shape[1])
    if axis > 0:
        ans[:, :axis] = unproj[:, :axis]
    ans[:, axis] = (s2 - 1) / (s2 + 1)
    ans[:, axis + 1 :] = unproj[:, axis:]
    return ans


def rotation_from_ortho_6d(ortho6d):
    """
    Convert a 6-d rotation representation to rotation matrix

    From the paper "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/pdf/1812.07035.pdf
    """
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = F.normalize(x_raw, dim=-1, eps=1e-6)
    y = F.normalize(y_raw, dim=-1, eps=1e-6)

    z = torch.cross(x, y, -1)
    z = F.normalize(z, dim=-1, eps=1e-6)
    y = torch.cross(z, x, -1)

    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix


def rotation_from_ortho_5d(ortho5d):
    """
    Convert a 5-d rotation representation to rotation matrix

    From the paper "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/pdf/1812.07035.pdf
    """
    batch = ortho5d.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])
    proj_scale = (
        torch.autograd.Variable(torch.FloatTensor(proj_scale_np).to(ortho5d))
        .reshape(1, 3)
        .repeat(batch, 1)
    )

    u = stereographic_unproject(ortho5d[:, 2:5] * proj_scale, axis=0)
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))
    u = u / norm.reshape(batch, 1).repeat(1, u.shape[1])
    b = torch.cat((ortho5d[:, 0:2], u), 1)
    matrix = rotation_from_ortho_6d(b)
    return matrix


def rotation_from_euler(euler):
    """
    Convert a 3-d Euler angle representation to rotation matrix
    """
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).reshape(batch, 1)
    s1 = torch.sin(euler[:, 0]).reshape(batch, 1)
    c2 = torch.cos(euler[:, 2]).reshape(batch, 1)
    s2 = torch.sin(euler[:, 2]).reshape(batch, 1)
    c3 = torch.cos(euler[:, 1]).reshape(batch, 1)
    s3 = torch.sin(euler[:, 1]).reshape(batch, 1)

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).reshape(-1, 1, 3)
    row2 = torch.cat(
        (c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1
    ).reshape(-1, 1, 3)
    row3 = torch.cat(
        (s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1
    ).reshape(-1, 1, 3)

    matrix = torch.cat((row1, row2, row3), 1)
    return matrix


def fit_to_SO3(R):
    # Math used from quora post and this berkeley pdf.
    # https://qr.ae/pKQaG5
    # https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
    #
    # Input:
    #   R - torch 3x3 rotation matrix that is not quite orthogonal
    # Output:
    #   Q - torch 3x3 nearest valid rotation matrix
    assert R.ndim == 2
    assert R.shape[0] == 3 and R.shape[1] == 3
    B = R
    I = torch.eye(3)
    Y = B.transpose(-2, -1) @ B - I
    Q = B - B @ Y @ (
        I / 2.0 - (3.0 * Y) / 8.0 + (5 * Y @ Y) / 16 - (35 * Y @ Y @ Y @ Y) / 128
    )
    return Q
