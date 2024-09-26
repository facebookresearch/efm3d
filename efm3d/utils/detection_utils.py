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
import torchvision

from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PAD_VAL, PoseTW, rotation_from_euler


def norm2ind(norm_xyz, vD, vH, vW):
    """Converts normalized xyz coords [-1,1] to DxHxW indices."""
    if isinstance(norm_xyz, np.ndarray):
        inds_dhw = norm_xyz.copy()
    else:
        inds_dhw = norm_xyz.clone()
    inds_dhw[..., 0] = torch.ceil((norm_xyz[..., 2] + 1.0) * vD / 2.0) - 1
    inds_dhw[..., 1] = torch.ceil((norm_xyz[..., 1] + 1.0) * vH / 2.0) - 1
    inds_dhw[..., 2] = torch.ceil((norm_xyz[..., 0] + 1.0) * vW / 2.0) - 1

    inds_dhw = inds_dhw.round()
    outside = (
        (inds_dhw[..., 0] <= 0)
        | (inds_dhw[..., 0] >= (vD - 1))
        | (inds_dhw[..., 1] <= 0)
        | (inds_dhw[..., 1] >= (vH - 1))
        | (inds_dhw[..., 2] <= 0)
        | (inds_dhw[..., 2] >= (vW - 1))
    )
    inside = ~outside
    if isinstance(inds_dhw, np.ndarray):
        inds_dhw = inds_dhw.astype(int)
    else:
        inds_dhw = inds_dhw.int()
    return inds_dhw, inside


def ind2norm(inds_dhw, vD, vH, vW):
    """Converts DxHxW indices to normalized xyz coords [-1,1]."""
    if isinstance(inds_dhw, np.ndarray):
        norm_xyz = inds_dhw.copy().astype(float)
    else:
        norm_xyz = inds_dhw.clone().float()
    norm_xyz[..., 0] = 2.0 * (inds_dhw[..., 2] + 0.5) / vW - 1.0
    norm_xyz[..., 1] = 2.0 * (inds_dhw[..., 1] + 0.5) / vH - 1.0
    norm_xyz[..., 2] = 2.0 * (inds_dhw[..., 0] + 0.5) / vD - 1.0

    return norm_xyz


def normalize_coord3d(xyz, extent):
    if isinstance(xyz, np.ndarray):
        xyz_n = xyz.copy()
    else:
        xyz_n = xyz.clone()
    x_min, x_max, y_min, y_max, z_min, z_max = extent
    xyz_n[..., 0] = ((xyz[..., 0] - x_min) / ((x_max - x_min) / 2.0)) - 1.0
    xyz_n[..., 1] = ((xyz[..., 1] - y_min) / ((y_max - y_min) / 2.0)) - 1.0
    xyz_n[..., 2] = ((xyz[..., 2] - z_min) / ((z_max - z_min) / 2.0)) - 1.0
    return xyz_n


def unnormalize_coord3d(xyz_n, extent):
    if isinstance(xyz_n, np.ndarray):
        xyz = xyz_n.copy()
    else:
        xyz = xyz_n.clone()
    x_min, x_max, y_min, y_max, z_min, z_max = extent
    xyz[..., 0] = ((xyz_n[..., 0] + 1.0) * ((x_max - x_min) / 2.0)) + x_min
    xyz[..., 1] = ((xyz_n[..., 1] + 1.0) * ((y_max - y_min) / 2.0)) + y_min
    xyz[..., 2] = ((xyz_n[..., 2] + 1.0) * ((z_max - z_min) / 2.0)) + z_min
    return xyz


def create_heatmap_gt(mu_xy, H, W, valid=None):
    """
    Inputs:
        mu_xy : torch.Tensor : shaped BxNx2 of pixel locations in range [0,H-1] and [0,W-1]
        H : image height
        W : image width:
        valid : torch.Tensor : optional boolean mask shaped BxNx2 or whether to use this point or not
    returns:
        heat_gt : torch.Tensor : Bx1xHxW tensor of splatted 2D points
    """

    B = mu_xy.shape[0]
    inside = (
        (mu_xy[..., 0] >= 0)
        & (mu_xy[..., 0] <= (H - 1))
        & (mu_xy[..., 1] >= 0)
        & (mu_xy[..., 1] <= (W - 1))
    )
    if valid is not None:  # if we have additional valid signal, use it
        inside = inside & valid
    inds_xy = mu_xy.round().long()
    inds_xy = inds_xy.reshape(B, -1, 2)
    inds = (
        inds_xy[:, :, 1] * W + inds_xy[:, :, 0]
    )  # flatten matrix index into vector index
    inds = torch.clip(inds, min=0, max=(H - 1) * (W - 1))
    inside = inside.reshape(B, -1).to(inds)
    heat_gt = torch.zeros((B, H * W)).to(inds)
    heat_gt.scatter_(1, inds, inside)
    heat_gt = heat_gt.reshape(B, H, W).float()
    blur = torchvision.transforms.functional.gaussian_blur
    kernel = 25
    heat_gt = blur(heat_gt, kernel)
    if heat_gt.sum() > 0:
        # Normalize such that peak is ~1.
        heat_gt = heat_gt * 100
        heat_gt = torch.clip(heat_gt, min=0, max=1)
    return heat_gt


def simple_nms(scores, nms_radius: int):
    """Approximate + Fast Non-maximum suppression to remove nearby points,
    works by running max pool twice on GPU."""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def simple_nms3d(scores, nms_radius: int):
    """Approximate + Fast Non-maximum suppression on 3D heatmap to remove nearby points,
    works by running max pool twice on GPU."""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool3d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def heatmap2obb(scores, threshold=0.3, size=20, max_elts=1000):
    """Runs argmax on a 2D heatmaps to return (x,y) positions
    in the heatmap in the ObbTW class, above a threshold. Creates
    fake 2D bounding boxes of size 20x20 by default."""
    # Extract keypoints
    hsize = int(round(size / 2))
    obbs = []
    dev = scores.device
    for score in scores:
        keypoint = torch.nonzero(score > threshold)
        ymin = keypoint[:, 0] - hsize
        ymax = keypoint[:, 0] + hsize
        xmin = keypoint[:, 1] - hsize
        xmax = keypoint[:, 1] + hsize
        bb2 = torch.stack([xmin, xmax, ymin, ymax], dim=1).float()
        obb = ObbTW().repeat(bb2.shape[0], 1).clone().to(dev)
        # Set bb2_rgb
        obb.set_bb2(cam_id=0, bb2d=bb2, use_mask=False)  # Set to RGB.
        # Set probability
        probs = score[tuple(keypoint.t())]
        obb.set_prob(probs)
        obbs.append(obb.add_padding(max_elts=max_elts))
    return torch.stack(obbs, dim=0)


# Centerness loss, binary cross entropy (evaluated densely per voxel position).
def compute_focal_loss(pred, gt, focal_gamma=2, focal_alpha=0.25):
    """focal loss for imbalanced classification
    https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
    Args:
        pred (torch.tensor): predicted probabilities
        gt (torch.tensor): GT probabilities
    Returns:
        nll_loss: negative log-likelihood loss
    """
    assert pred.shape == gt.shape
    gt = gt.double()
    pred = pred.double()
    eps = 1e-9
    # Simple negative log-likelihood (aka binary cross-entropy). Assume sigmoid already applied.
    nll = -(torch.log(pred + eps) * gt + torch.log((1.0 - pred) + eps) * (1.0 - gt))

    if focal_gamma > 0:
        p_t = pred * gt + (1 - pred) * (1 - gt)
        nll = nll * ((1 - p_t) ** focal_gamma)

    # class-wise balancing
    if focal_alpha >= 0:
        alpha_t = focal_alpha * gt + (1 - focal_alpha) * (1.0 - gt)
        nll = alpha_t * nll

    return nll.float()


def compute_chamfer_loss(vals, target):
    B = vals.shape[0]
    xx = vals.view(B, 8, 1, 3)
    yy = target.view(B, 1, 8, 3)
    l1_dist = (xx - yy).abs().sum(-1)

    gt_to_pred = l1_dist.min(1).values.mean(-1)
    pred_to_gt = l1_dist.min(2).values.mean(-1)
    l1 = 0.1 * pred_to_gt + gt_to_pred
    return l1


def obb2voxel(obb_v, vD, vH, vW, voxel_extent, num_class, splat_sigma=2):
    """
    Inputs:
        obb_v : ObbTW : shaped BxNx34 of obbs in voxel coordinates.
        vD : voxel depth
        vH : voxel height
        vW : voxel width:
        voxel_extent: size of voxel grid in meters, with order W, H, D
        num_class: number of classes to detect
        splat_sigma: how big to splat the Obbs
    returns:
        cent_gt : torch.Tensor : Bx1xDxHxW tensor of splatted 2D points
        bbox_gt : torch.Tensor : Bx7xDxHxW tensor of bounding box params
        clas_gt : torch.Tensor : Bxnum_classxDxHxW one hot tensor of class
        valid_gt : torch.Tensor : Bx1xDxHxW bool tensor of where splatting is valid
    """
    B = obb_v.shape[0]
    device = obb_v.device
    cent_gt = torch.zeros((B, 1, vD, vH, vW), device=device)
    bbox_gt = torch.zeros((B, 7, vD, vH, vW), device=device)
    clas_gt = torch.zeros((B, num_class, vD, vH, vW), device=device)
    # Where to apply non-centerness losses.
    valid_gt = torch.zeros((B, 1, vD, vH, vW), device=device, dtype=torch.bool)
    # Gaussian kernel for splatting.
    size = 2 * splat_sigma + 1
    rng = torch.arange(0, size, 1).to(device)
    xx, yy, zz = torch.meshgrid(rng, rng, rng, indexing="ij")
    x0 = y0 = z0 = size // 2
    eps = 1e-6
    gauss = torch.exp(
        -((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) / (2 * splat_sigma**2 + eps)
    )
    # Convert obb centers to voxel indices.
    cent_v = obb_v.bb3_center_world
    cent_vn = normalize_coord3d(cent_v, voxel_extent)
    inds, inside = norm2ind(cent_vn, vD, vH, vW)
    # Get index offsets for splatting.
    if splat_sigma == 0:
        dd = torch.tensor([0]).reshape(1, 1, 1).to(device)
        hh = torch.tensor([0]).reshape(1, 1, 1).to(device)
        ww = torch.tensor([0]).reshape(1, 1, 1).to(device)
    elif splat_sigma > 0:
        rng_d = torch.arange(start=-splat_sigma, end=splat_sigma + 1).to(device)
        rng_h = torch.arange(start=-splat_sigma, end=splat_sigma + 1).to(device)
        rng_w = torch.arange(start=-splat_sigma, end=splat_sigma + 1).to(device)
        dd, hh, ww = torch.meshgrid(rng_d, rng_h, rng_w, indexing="ij")
    else:
        raise ValueError("splat sigma most be non-negative")
    offsets_dhw = torch.stack((dd.reshape(-1), hh.reshape(-1), ww.reshape(-1)), dim=-1)
    offsets_dhw = offsets_dhw.unsqueeze(0).repeat(B, 1, 1)
    # Use broadcasting to apply the offset indices to the voxel indices.
    O = offsets_dhw.shape[1]
    N = inds.shape[1]
    inds_dhw = inds.reshape(B, N, 1, 3) + offsets_dhw.reshape(B, 1, O, 3)
    inds_dhw = inds_dhw.reshape(B, N * O, 3)
    inside = inside.reshape(B, N, 1).repeat(1, 1, O).reshape(B, N * O).float()
    # Avoid accessing OOB.
    ones = torch.ones_like(inds_dhw[:, :, 0])
    inds_dhw[:, :, 0] = torch.maximum(inds_dhw[:, :, 0], 0 * ones)
    inds_dhw[:, :, 1] = torch.maximum(inds_dhw[:, :, 1], 0 * ones)
    inds_dhw[:, :, 2] = torch.maximum(inds_dhw[:, :, 2], 0 * ones)
    inds_dhw[:, :, 0] = torch.minimum(inds_dhw[:, :, 0], (vD - 1) * ones)
    inds_dhw[:, :, 1] = torch.minimum(inds_dhw[:, :, 1], (vH - 1) * ones)
    inds_dhw[:, :, 2] = torch.minimum(inds_dhw[:, :, 2], (vW - 1) * ones)

    # keep the (d, h, w) indices before flattening
    inds_dhw_3d = inds_dhw.clone()

    # Convert D,H,W indices into flat array indices.
    inds_d = inds_dhw[:, :, 0]
    inds_h = inds_dhw[:, :, 1]
    inds_w = inds_dhw[:, :, 2]
    inds_dhw = inds_d * (vH * vW) + inds_h * vW + inds_w
    b_inds = torch.arange(B).reshape(-1, 1).repeat(1, N * O)
    # Set centerness GT.
    cent_gt = cent_gt.reshape(B, -1)
    gauss = gauss.reshape(1, 1, -1).repeat(B, N, 1).reshape(B, N * O)
    cent_gt[b_inds, inds_dhw] = gauss * inside
    cent_gt = cent_gt.reshape(B, 1, vD, vH, vW)
    # Semantic class.
    CL = num_class
    sem_id = torch.clip(obb_v.sem_id, 0, CL - 1).long()
    one_hot = torch.nn.functional.one_hot(sem_id, num_classes=CL)
    one_hot = one_hot.reshape(B, N, CL).permute(0, 2, 1)
    one_hot = one_hot.reshape(B, CL, N, 1).repeat(1, 1, 1, O)
    one_hot = one_hot.reshape(B, CL, N * O)
    val = one_hot * inside.reshape(B, 1, -1)
    clas_gt = clas_gt.reshape(B, -1, vD * vH * vW)
    b_inds_rep = b_inds.reshape(B, 1, -1).repeat(1, CL, 1)
    cl_inds_rep = torch.arange(CL).reshape(1, CL, 1).repeat(B, 1, O * N)
    inds_dhw_rep = inds_dhw.reshape(B, 1, -1).repeat(1, CL, 1)
    clas_gt[b_inds_rep, cl_inds_rep, inds_dhw_rep] = val
    clas_gt = clas_gt.reshape(B, -1, vD, vH, vW)
    # Get gravity aligned rotation from obb
    T_voxel_object = obb_v.T_world_object.clone()
    # HACK to avoid gimbal lock for padded entries.
    mask = obb_v.get_padding_mask()
    T_voxel_object.R[mask] = PAD_VAL
    rpy = T_voxel_object.to_euler()
    yaw = rpy[:, :, 2]
    # BBox size (in voxel coordinates.)
    bb3 = obb_v.bb3_object
    xsize = bb3[:, :, 1] - bb3[:, :, 0]
    ysize = bb3[:, :, 3] - bb3[:, :, 2]
    zsize = bb3[:, :, 5] - bb3[:, :, 4]
    # Discretized centers.
    centd_vn = ind2norm(inds_dhw_3d, vD, vH, vW)
    centd_v = unnormalize_coord3d(centd_vn, voxel_extent)
    # Compute offset between discretized centers and obb centers.
    cent_v_rep = cent_v.reshape(B, -1, 1, 3).repeat(1, 1, O, 1).reshape(B, N * O, 3)
    offsets = centd_v - cent_v_rep
    xoff = offsets[:, :, 0]
    yoff = offsets[:, :, 1]
    zoff = offsets[:, :, 2]
    # Splat via repeat.
    xsize = xsize.reshape(B, -1, 1).repeat(1, 1, O).reshape(B, N * O)
    ysize = ysize.reshape(B, -1, 1).repeat(1, 1, O).reshape(B, N * O)
    zsize = zsize.reshape(B, -1, 1).repeat(1, 1, O).reshape(B, N * O)
    yaw = yaw.reshape(B, -1, 1).repeat(1, 1, O).reshape(B, N * O)
    # Assign bbox parameters into voxel GT.
    bbox_gt = bbox_gt.reshape(B, 7, -1)
    BB = bbox_gt.shape[1]
    bb_inds = torch.arange(BB).reshape(-1, 1).repeat(1, N * O)
    bbox_gt[b_inds, bb_inds[0, :], inds_dhw] = xsize * inside
    bbox_gt[b_inds, bb_inds[1, :], inds_dhw] = ysize * inside
    bbox_gt[b_inds, bb_inds[2, :], inds_dhw] = zsize * inside
    bbox_gt[b_inds, bb_inds[3, :], inds_dhw] = xoff * inside
    bbox_gt[b_inds, bb_inds[4, :], inds_dhw] = yoff * inside
    bbox_gt[b_inds, bb_inds[5, :], inds_dhw] = zoff * inside
    bbox_gt[b_inds, bb_inds[6, :], inds_dhw] = yaw * inside
    bbox_gt = bbox_gt.reshape(B, 7, vD, vH, vW)
    # Set valid mask.
    valid_gt = valid_gt.reshape(B, -1)
    valid_gt[b_inds, inds_dhw] = inside.bool()
    valid_gt = valid_gt.reshape(B, 1, vD, vH, vW)
    return cent_gt, bbox_gt, clas_gt, valid_gt


def voxel2obb(
    cent_pr,
    bbox_pr,
    clas_pr,
    voxel_extent,
    top_k=None,
    thresh=None,
    return_full_prob=False,
):
    """Convert 3D centerness, size, rotation voxel grids to ObbTW objects,
    returning objects in the voxel coordinate frame. Can optionally threshold
    based on a topK predictions.
    """
    device = cent_pr.device
    assert cent_pr.ndim == 5
    B, _, vD, vH, vW = cent_pr.shape
    device = cent_pr.device
    # Get extent.
    xhalf = bbox_pr[:, 0] / 2.0
    yhalf = bbox_pr[:, 1] / 2.0
    zhalf = bbox_pr[:, 2] / 2.0
    bb3 = torch.stack(
        [
            -xhalf,
            +xhalf,
            -yhalf,
            +yhalf,
            -zhalf,
            +zhalf,
        ],
        dim=-1,
    )
    # Get rotation to set T_world_object.
    yaw = bbox_pr[:, 6]
    zeros = torch.zeros_like(yaw)
    e_angles = torch.stack([zeros, zeros, yaw], dim=-1)
    R = rotation_from_euler(e_angles.reshape(-1, 3))
    R = R.reshape(B, vD, vH, vW, 3, 3)
    t_zero = torch.zeros(B, vD, vH, vW, 3).to(device)
    T_voxel_object = PoseTW.from_Rt(R, t_zero)
    rngd = torch.arange(vD).to(device)
    rngh = torch.arange(vH).to(device)
    rngw = torch.arange(vW).to(device)
    xx, yy, zz = torch.meshgrid(rngd, rngh, rngw, indexing="ij")
    inds = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=-1)
    norm_centers = ind2norm(inds, vD, vH, vW)
    centers_v = unnormalize_coord3d(norm_centers, voxel_extent)
    centers_v = centers_v.reshape(1, vD, vH, vW, 3).repeat(B, 1, 1, 1, 1)
    # The center is defined as the voxel center + the offset.
    xoff = bbox_pr[:, 3]
    yoff = bbox_pr[:, 4]
    zoff = bbox_pr[:, 5]
    t_off = torch.stack([xoff, yoff, zoff], dim=-1)
    T_voxel_object.t[:] = centers_v - t_off
    # Get prob.
    prob = cent_pr.reshape(B, vD, vH, vW, 1)
    N = inds.shape[0]
    # Get instance id, use voxel location for this.
    inst_id = torch.arange(N).reshape(1, vD, vH, vW, 1).repeat(B, 1, 1, 1, 1)
    # Get semantic id
    sem_id = torch.argmax(clas_pr, dim=1).unsqueeze(-1)
    # Construct ObbTW object.
    obbs = ObbTW.from_lmc(
        bb3_object=bb3,
        T_world_object=T_voxel_object,
        prob=prob,
        inst_id=inst_id,
        sem_id=sem_id,
    )
    # Optionally remove detections below threshold.
    if thresh is not None:
        below = (obbs.prob < thresh).squeeze(-1)
        obbs._data[below, :] = PAD_VAL

    # Optionally subselect top K.
    if top_k is not None:
        prob = obbs.prob.reshape(B, N)
        s_vals, s_inds = torch.sort(prob, dim=1, descending=True)
        n_inds = s_inds[:, :top_k].reshape(-1)
        b_inds = torch.arange(B).reshape(B, 1).repeat(1, top_k).to(device).reshape(-1)
        obbs = obbs.reshape(B, N, -1)
        # B x K x 34
        obbs = obbs[b_inds, n_inds].reshape(B, top_k, -1)
        # B x K
        prob = prob[b_inds, n_inds].reshape(B, top_k)
        # B x K x C
        clas_pr = clas_pr.reshape(B, -1, N)[b_inds, :, n_inds].reshape(B, top_k, -1)

    if return_full_prob:
        return obbs, prob, clas_pr
    else:
        return obbs
