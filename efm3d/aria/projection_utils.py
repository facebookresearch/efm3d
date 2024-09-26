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


def sign_plus(x):
    """
    return +1 for positive and for 0.0 in x. This is important for our handling
    of z values that should never be 0.0
    """
    sgn = torch.ones_like(x)
    sgn[sgn < 0.0] = -1.0
    return sgn


@torch.jit.script
def fisheye624_project(xyz, params):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model project() function.

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        params: Bx(T)x16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx(T)x15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane

    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model allows fu != fv.
    Specifically, the model is:
    uvDistorted = [x_r]  + tangentialDistortion  + thinPrismDistortion
                  [y_r]
    proj = diag(fu,fv) * uvDistorted + [cu;cv];
    where:
      a = x/z, b = y/z, r = (a^2+b^2)^(1/2)
      th = atan(r)
      cosPhi = a/r, sinPhi = b/r
      [x_r]  = (th+ k0 * th^3 + k1* th^5 + ...) [cosPhi]
      [y_r]                                     [sinPhi]
      the number of terms in the series is determined by the template parameter numK.
      tangentialDistortion = [(2 x_r^2 + rd^2)*p_0 + 2*x_r*y_r*p_1]
                             [(2 y_r^2 + rd^2)*p_1 + 2*x_r*y_r*p_0]
      where rd^2 = x_r^2 + y_r^2
      thinPrismDistortion = [s0 * rd^2 + s1 rd^4]
                            [s2 * rd^2 + s3 rd^4]
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (
        xyz.ndim == 4 and params.ndim == 3
    ), f"point dim {xyz.shape} does not match cam parameter dim {params}"
    assert xyz.shape[-1] == 3
    assert (
        params.shape[-1] == 16 or params.shape[-1] == 15
    ), "This model allows fx != fy"
    assert xyz.dtype == params.dtype, "data type must match"

    eps = 1e-9
    T = -1
    if xyz.ndim == 4:
        # has T dim
        T, N = xyz.shape[1], xyz.shape[2]
        xyz = xyz.reshape(-1, N, 3)  # (BxT)xNx3
        params = params.reshape(-1, params.shape[-1])  #  (BxT)x16

    B, N = xyz.shape[0], xyz.shape[1]

    # Radial correction.
    z = xyz[:, :, 2].reshape(B, N, 1)
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    ab = xyz[:, :, :2] / z
    # make sure abs are not too small or 0 otherwise gradients are nan
    ab = torch.where(torch.abs(ab) < eps, eps * sign_plus(ab), ab)
    r = torch.norm(ab, dim=-1, p=2, keepdim=True)
    th = torch.atan(r)
    th_divr = torch.where(r < eps, torch.ones_like(ab), ab / r)
    th_k = th.reshape(B, N, 1).clone()
    for i in range(6):
        th_k = th_k + params[:, -12 + i].reshape(B, 1, 1) * torch.pow(th, 3 + i * 2)
    xr_yr = th_k * th_divr
    uv_dist = xr_yr

    # Tangential correction.
    p0 = params[:, -6].reshape(B, 1)
    p1 = params[:, -5].reshape(B, 1)
    xr = xr_yr[:, :, 0].reshape(B, N)
    yr = xr_yr[:, :, 1].reshape(B, N)
    xr_yr_sq = torch.square(xr_yr)
    xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
    yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
    rd_sq = xr_sq + yr_sq
    uv_dist_tu = uv_dist[:, :, 0] + ((2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1)
    uv_dist_tv = uv_dist[:, :, 1] + ((2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0)
    uv_dist = torch.stack(
        [uv_dist_tu, uv_dist_tv], dim=-1
    )  # Avoids in-place complaint.

    # Thin Prism correction.
    s0 = params[:, -4].reshape(B, 1)
    s1 = params[:, -3].reshape(B, 1)
    s2 = params[:, -2].reshape(B, 1)
    s3 = params[:, -1].reshape(B, 1)
    rd_4 = torch.square(rd_sq)
    uv_dist[:, :, 0] = uv_dist[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
    uv_dist[:, :, 1] = uv_dist[:, :, 1] + (s2 * rd_sq + s3 * rd_4)

    # Finally, apply standard terms: focal length and camera centers.
    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)
    result = uv_dist * fx_fy + cx_cy

    if T > 0:
        result = result.reshape(B // T, T, N, 2)

    assert result.ndim == 4 or result.ndim == 3
    assert result.shape[-1] == 2

    return result


@torch.jit.script
def fisheye624_unproject(uv, params, max_iters: int = 5):
    """
    Batched implementation of the FisheyeRadTanThinPrism (aka Fisheye624) camera
    model. There is no analytical solution for the inverse of the project()
    function so this solves an optimization problem using Newton's method to get
    the inverse.

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels to be projected
        params: Bx(T)x16 tensor of Fisheye624 parameters formatted like this:
                [f_u f_v c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
                or Bx(T)x15 tensor of Fisheye624 parameters formatted like this:
                [f c_u c_v {k_0 ... k_5} {p_0 p_1} {s_0 s_1 s_2 s_3}]
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays of uv points with z = 1.

    Model for fisheye cameras with radial, tangential, and thin-prism distortion.
    This model assumes fu=fv. This unproject function holds that:

    X = unproject(project(X))     [for X=(x,y,z) in R^3, z>0]

    and

    x = project(unproject(s*x))   [for s!=0 and x=(u,v) in R^2]
    """

    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx2"
    assert uv.shape[-1] == 2
    assert (
        params.ndim == 2 or params.ndim == 3
    ), "Expected batched input shaped Bx(T)x16 or Bx(T)x15"
    assert (
        params.shape[-1] == 16 or params.shape[-1] == 15
    ), "This model allows fx != fy"
    assert uv.dtype == params.dtype, "data type must match"
    eps = 1e-6

    T = -1
    if uv.ndim == 4:
        # has T dim
        T, N = uv.shape[1], uv.shape[2]
        uv = uv.reshape(-1, N, 2)  # (BxT)xNx2
        params = params.reshape(-1, params.shape[-1])  #  (BxT)x16

    B, N = uv.shape[0], uv.shape[1]

    if params.shape[-1] == 15:
        fx_fy = params[:, 0].reshape(B, 1, 1)
        cx_cy = params[:, 1:3].reshape(B, 1, 2)
    else:
        fx_fy = params[:, 0:2].reshape(B, 1, 2)
        cx_cy = params[:, 2:4].reshape(B, 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    # Compute xr_yr using Newton's method.
    xr_yr = uv_dist.clone()  # Initial guess.
    for _ in range(max_iters):
        uv_dist_est = xr_yr.clone()
        # Tangential terms.
        p0 = params[:, -6].reshape(B, 1)
        p1 = params[:, -5].reshape(B, 1)
        xr = xr_yr[:, :, 0].reshape(B, N)
        yr = xr_yr[:, :, 1].reshape(B, N)
        xr_yr_sq = torch.square(xr_yr)
        xr_sq = xr_yr_sq[:, :, 0].reshape(B, N)
        yr_sq = xr_yr_sq[:, :, 1].reshape(B, N)
        rd_sq = xr_sq + yr_sq
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (
            (2.0 * xr_sq + rd_sq) * p0 + 2.0 * xr * yr * p1
        )
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (
            (2.0 * yr_sq + rd_sq) * p1 + 2.0 * xr * yr * p0
        )
        # Thin Prism terms.
        s0 = params[:, -4].reshape(B, 1)
        s1 = params[:, -3].reshape(B, 1)
        s2 = params[:, -2].reshape(B, 1)
        s3 = params[:, -1].reshape(B, 1)
        rd_4 = torch.square(rd_sq)
        uv_dist_est[:, :, 0] = uv_dist_est[:, :, 0] + (s0 * rd_sq + s1 * rd_4)
        uv_dist_est[:, :, 1] = uv_dist_est[:, :, 1] + (s2 * rd_sq + s3 * rd_4)
        # Compute the derivative of uv_dist w.r.t. xr_yr.
        duv_dist_dxr_yr = uv.new_ones(B, N, 2, 2)
        duv_dist_dxr_yr[:, :, 0, 0] = (
            1.0 + 6.0 * xr_yr[:, :, 0] * p0 + 2.0 * xr_yr[:, :, 1] * p1
        )
        offdiag = 2.0 * (xr_yr[:, :, 0] * p1 + xr_yr[:, :, 1] * p0)
        duv_dist_dxr_yr[:, :, 0, 1] = offdiag
        duv_dist_dxr_yr[:, :, 1, 0] = offdiag
        duv_dist_dxr_yr[:, :, 1, 1] = (
            1.0 + 6.0 * xr_yr[:, :, 1] * p1 + 2.0 * xr_yr[:, :, 0] * p0
        )
        xr_yr_sq_norm = xr_yr_sq[:, :, 0] + xr_yr_sq[:, :, 1]
        temp1 = 2.0 * (s0 + 2.0 * s1 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 0, 0] = duv_dist_dxr_yr[:, :, 0, 0] + (
            xr_yr[:, :, 0] * temp1
        )
        duv_dist_dxr_yr[:, :, 0, 1] = duv_dist_dxr_yr[:, :, 0, 1] + (
            xr_yr[:, :, 1] * temp1
        )
        temp2 = 2.0 * (s2 + 2.0 * s3 * xr_yr_sq_norm)
        duv_dist_dxr_yr[:, :, 1, 0] = duv_dist_dxr_yr[:, :, 1, 0] + (
            xr_yr[:, :, 0] * temp2
        )
        duv_dist_dxr_yr[:, :, 1, 1] = duv_dist_dxr_yr[:, :, 1, 1] + (
            xr_yr[:, :, 1] * temp2
        )
        # Compute 2x2 inverse manually here since torch.inverse() is very slow.
        # Because this is slow: inv = duv_dist_dxr_yr.inverse()
        # About a 10x reduction in speed with above line.
        mat = duv_dist_dxr_yr.reshape(-1, 2, 2)
        a = mat[:, 0, 0].reshape(-1, 1, 1)
        b = mat[:, 0, 1].reshape(-1, 1, 1)
        c = mat[:, 1, 0].reshape(-1, 1, 1)
        d = mat[:, 1, 1].reshape(-1, 1, 1)
        det = 1.0 / ((a * d) - (b * c))
        top = torch.cat([d, -b], dim=2)
        bot = torch.cat([-c, a], dim=2)
        inv = det * torch.cat([top, bot], dim=1)
        inv = inv.reshape(B, N, 2, 2)
        # Manually compute 2x2 @ 2x1 matrix multiply.
        # Because this is slow: step = (inv @ (uv_dist - uv_dist_est)[..., None])[..., 0]
        diff = uv_dist - uv_dist_est
        a = inv[:, :, 0, 0]
        b = inv[:, :, 0, 1]
        c = inv[:, :, 1, 0]
        d = inv[:, :, 1, 1]
        e = diff[:, :, 0]
        f = diff[:, :, 1]
        step = torch.stack([a * e + b * f, c * e + d * f], dim=-1)
        # Newton step.
        xr_yr = xr_yr + step

    # Compute theta using Newton's method.
    xr_yr_norm = xr_yr.norm(p=2, dim=2).reshape(B, N, 1)
    th = xr_yr_norm.clone()
    for _ in range(max_iters):
        th_radial = uv.new_ones(B, N, 1)
        dthd_th = uv.new_ones(B, N, 1)
        for k in range(6):
            r_k = params[:, -12 + k].reshape(B, 1, 1)
            th_radial = th_radial + (r_k * torch.pow(th, 2 + k * 2))
            dthd_th = dthd_th + ((3.0 + 2.0 * k) * r_k * torch.pow(th, 2 + k * 2))
        th_radial = th_radial * th
        step = (xr_yr_norm - th_radial) / dthd_th
        # handle dthd_th close to 0.
        step = torch.where(dthd_th.abs() > eps, step, sign_plus(step) * eps * 10.0)
        th = th + step
    # Compute the ray direction using theta and xr_yr.
    close_to_zero = torch.logical_and(th.abs() < eps, xr_yr_norm.abs() < eps)
    ray_dir = torch.where(close_to_zero, xr_yr, torch.tan(th) / xr_yr_norm * xr_yr)
    ray = torch.cat([ray_dir, uv.new_ones(B, N, 1)], dim=2)
    assert ray.shape[-1] == 3

    if T > 0:
        ray = ray.reshape(B // T, T, N, 3)

    return ray


def pinhole_project(xyz, params):
    """
    Batched implementation of the Pinhole (aka Linear) camera
    model project() function.

    Inputs:
        xyz: Bx(T)xNx3 tensor of 3D points to be projected
        params: Bx(T)x4 tensor of Pinhole parameters formatted like this:
                [f_u f_v c_u c_v]
    Outputs:
        uv: Bx(T)xNx2 tensor of 2D projections of xyz in image plane
    """

    assert (xyz.ndim == 3 and params.ndim == 2) or (xyz.ndim == 4 and params.ndim == 3)
    assert params.shape[-1] == 4
    eps = 1e-9

    # Focal length and principal point
    fx_fy = params[..., 0:2].reshape(*xyz.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*xyz.shape[:-2], 1, 2)
    # Make sure depth is not too close to zero.
    z = xyz[..., 2:]
    # Do not use torch.sign(z) it leads to 0.0 zs if z == 0.0 which leads to a
    # nan when we compute xy/z
    z = torch.where(torch.abs(z) < eps, eps * sign_plus(z), z)
    uv = (xyz[..., :2] / z) * fx_fy + cx_cy
    return uv


def pinhole_unproject(uv, params, max_iters: int = 5):
    """
    Batched implementation of the Pinhole (aka Linear) camera model.

    Inputs:
        uv: Bx(T)xNx2 tensor of 2D pixels to be projected
        params: Bx(T)x4 tensor of Pinhole parameters formatted like this:
                [f_u f_v c_u c_v]
    Outputs:
        xyz: Bx(T)xNx3 tensor of 3D rays of uv points with z = 1.

    """
    assert uv.ndim == 3 or uv.ndim == 4, "Expected batched input shaped Bx(T)xNx3"
    assert params.ndim == 2 or params.ndim == 3
    assert params.shape[-1] == 4
    assert uv.shape[-1] == 2

    # Focal length and principal point
    fx_fy = params[..., 0:2].reshape(*uv.shape[:-2], 1, 2)
    cx_cy = params[..., 2:4].reshape(*uv.shape[:-2], 1, 2)

    uv_dist = (uv - cx_cy) / fx_fy

    ray = torch.cat([uv_dist, uv.new_ones(*uv.shape[:-1], 1)], dim=-1)
    return ray