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

import cv2
import numpy as np
import torch
from efm3d.aria import CameraTW, ObbTW
from efm3d.aria.aria_constants import RESOLUTION_MAP


def get_crops_scale(
    W: int,
    H: int,
    cam_name: Literal["rgb", "slaml", "slamr"],
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
):
    # Pre-cropping is universal to all down_scaling.
    # Handle RGB properly with binning
    pre_crop = None
    if cam_name == "rgb" and W == 2880 and H == 2880:
        # crop image to 2816x2816
        pre_crop = [32, 32]
        H, W = H - 64, W - 64

    if down_scale in [0, 1, 2, 4]:
        factor = 1
        if cam_name == "rgb":
            if W == 2816 and H == 2816:
                # downsample to 1408x1408
                factor = 2
            if down_scale > 0:
                factor = 2 * down_scale * factor
        else:
            factor = down_scale
        if factor <= 1:
            factor = None
        if factor:
            # W, H after scaling down
            W, H = W // factor, H // factor

        # post-crop to reach size divisible by wh_multiple_of
        w_crop = (W % wh_multiple_of) // 2
        h_crop = (H % wh_multiple_of) // 2
        post_crop = [w_crop, h_crop]
        # set outputs none if they are not needed
        if w_crop == 0 and h_crop == 0:
            post_crop = None
    elif down_scale in RESOLUTION_MAP:
        if cam_name == "rgb":
            target_h = RESOLUTION_MAP[down_scale][0]
            target_w = RESOLUTION_MAP[down_scale][0]
        elif cam_name in ["slaml", "slamr"]:
            target_w = RESOLUTION_MAP[down_scale][1]
            target_h = RESOLUTION_MAP[down_scale][2]
        else:
            raise ValueError("Specified cam_name of %s is not supported" % down_scale)

        if target_h % wh_multiple_of != 0 or target_w % wh_multiple_of != 0:
            raise ValueError(
                f"only wh_multiple_of 16 is guaranteed when using scale_down == [5,6,7,8,9] {target_h} % {wh_multiple_of}"
            )

        # This rescale factor can be non-integer.
        factor_w = W / target_w
        factor_h = H / target_h
        assert (
            factor_w == factor_h
        ), "rescale factor must maintain original aspect ratio"
        factor = factor_w
        post_crop = None
    else:
        raise ValueError("Specified down_scale of %d is not supported" % down_scale)

    return pre_crop, factor, post_crop


def rescale_camera_tw(
    cam: CameraTW,
    cam_size_before,  # tuple of (height, width, ...)
    cam_name: Literal["rgb", "slaml", "slamr"],
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
):
    """
    Rescale CameraTW tensors by passing the camera size, camera name, and a down scale factor.
    cam shape should be [..., N] where N is the valid camera calibration dimension (25 or 33)
    """

    H, W = cam_size_before[:2]

    if (cam.c > 1000.0).any():
        # it can happen that the calibration was stored with respect to the full
        # 2880 x 2880 resolution although the rgb video stream is binned to 1408 x
        # 1408. We catch this by looking at the principal point which should be
        # about [704, 704] and fix the calibration.
        H, W = 2880, 2880
        if cam.valid_radius[0].item() < 1000.0:
            # it is likely that the valid_radius was set on the wrong cam
            # size (2x too small) so we fix it here.
            cam.set_valid_radius(cam.valid_radius * 2.0)

    pre_crop, factor, post_crop = get_crops_scale(
        W, H, cam_name, down_scale, wh_multiple_of
    )
    if pre_crop:
        # new width and height after center crop
        W, H = W - 2 * pre_crop[0], H - 2 * pre_crop[1]
        cam = cam.crop(pre_crop, (W, H))
    if factor:
        cam = cam.scale(1.0 / factor)
        # after scaling
        W, H = W // factor, H // factor
    if post_crop:
        # new width and height after center crop
        W, H = W - 2 * post_crop[0], H - 2 * post_crop[1]
        cam = cam.crop(post_crop, (W, H))

    return cam


def rescale_calib(
    calib,
    cam_size_before,  # tuple of (height, width, ...)
    cam_name: Literal["rgb", "slaml", "slamr"],
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
):
    """
    rescale raw camera parameters
    """
    # fisheye264
    assert calib.shape[-1] == 15

    H, W = cam_size_before[:2]
    # it can happen that the calibration was stored with respect to the full
    # 2880 x 2880 resolution although the rgb video stream is binned to 1408 x
    # 1408. We catch this by looking at the principal point which should be
    # about [704, 704] and fix the calibration.
    if (calib[1:3] > 1000.0).any():
        H, W = 2880, 2880

    pre_crop, factor, post_crop = get_crops_scale(
        W, H, cam_name, down_scale, wh_multiple_of
    )
    if pre_crop:
        calib[1:3] = calib[1:3] - np.array(pre_crop)
    if factor:
        calib[0] = calib[0] / factor
        calib[1:3] = (calib[1:3] + 0.5) / factor - 0.5
    if post_crop:
        calib[1:3] = calib[1:3] - np.array(post_crop)

    return calib


def rescale_image(
    img,
    cam_name: Literal["rgb", "slaml", "slamr"],
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
):
    H, W = img.shape[:2]
    pre_crop, factor, post_crop = get_crops_scale(
        W, H, cam_name, down_scale, wh_multiple_of
    )
    if pre_crop:
        img = img[pre_crop[1] : H - pre_crop[1], pre_crop[0] : W - pre_crop[0], ...]

    if factor:
        # When factor is integer, then cv2.INTER_AREA behaves identically
        # to skimage.downscale_local_mean, as described in the blog post:
        # https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
        H, W = img.shape[:2]
        target_wh = int(round(W / factor)), int(round(H / factor))
        orig_ndim = img.ndim
        img = cv2.resize(img, target_wh, interpolation=cv2.INTER_AREA)
        if orig_ndim == 3 and img.ndim == 2:
            img = np.expand_dims(img, axis=2)  # Preserve HxWx1 vs HxW to match input.
    if post_crop:
        H, W = img.shape[:2]
        img = img[post_crop[1] : H - post_crop[1], post_crop[0] : W - post_crop[0], ...]
    return img


def rescale_image_tensor(
    img,
    cam_name: Literal["rgb", "slaml", "slamr"],
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
    interpolate_mode: str = "bilinear",
):
    """
    Rescale the Aria image tensor. `img` is a torch Tensor, which is expected to have
    [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    `down_scale` specifies the degree of down-sampling.
    """
    from torchvision.transforms.functional import InterpolationMode, resize

    str2torchvision_mapping = {
        "nearest": InterpolationMode.NEAREST,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }

    H, W = img.shape[-2:]
    pre_crop, factor, post_crop = get_crops_scale(
        W, H, cam_name, down_scale, wh_multiple_of
    )
    if pre_crop:
        img = img[..., pre_crop[1] : H - pre_crop[1], pre_crop[0] : W - pre_crop[0]]

    if factor:
        H, W = img.shape[-2:]
        target_hw = int(round(H / factor)), int(round(W / factor))
        img = resize(
            img,
            target_hw,
            interpolation=str2torchvision_mapping[interpolate_mode],
            antialias=True,
        )
    if post_crop:
        H, W = img.shape[-2:]
        img = img[..., post_crop[1] : H - post_crop[1], post_crop[0] : W - post_crop[0]]
    return img


def rescale_depth_img(
    depth_img, scale_down, filter_boundary=True, valid=None, wh_multiple_of: int = 16
):
    # Use torch to re-scale since opencv doesn't re-scale.
    # And make sure it's 1xHxW
    depth_img = torch.tensor(depth_img).squeeze().unsqueeze(0)
    depth_img_rescale = rescale_image_tensor(
        depth_img,
        "rgb",
        scale_down,
        wh_multiple_of=wh_multiple_of,
        interpolate_mode="nearest",
    )
    if not filter_boundary:
        return depth_img_rescale

    # Change the mask to float to capture the boundaries of invalid area.
    if valid is None:
        d_mask = (depth_img > 0).float()
    else:
        d_mask = torch.tensor(valid).float().unsqueeze(0)

    d_mask_rescale = rescale_image_tensor(
        d_mask,
        "rgb",
        scale_down,
        wh_multiple_of=wh_multiple_of,
        interpolate_mode="nearest",
    )
    # only the mask pixels which are close to 1.0 is the valid ones.
    depth_img_rescale[abs(d_mask_rescale - 1.0) > 1e-5] = 0.0
    return depth_img_rescale


def rescale_obb_tw(
    obbs: ObbTW,
    cam_size_before_rgb,
    cam_size_before_slam,
    down_scale: Literal[0, 1, 2, 4, 5, 6, 7, 8, 9] = 0,
    wh_multiple_of: int = 16,
):
    """
    Rescale ObbTW 2d bb tensors by passing the camera size, camera name, and a down scale factor.
    """
    H_rgb, W_rgb = cam_size_before_rgb[:2]
    H_slam, W_slam = cam_size_before_slam[:2]
    pre_crop_rgb, factor_rgb, post_crop_rgb = get_crops_scale(
        W_rgb, H_rgb, "rgb", down_scale, wh_multiple_of
    )
    pre_crop_slam, factor_slam, post_crop_slam = get_crops_scale(
        W_slam, H_slam, "slaml", down_scale, wh_multiple_of
    )
    if pre_crop_rgb or pre_crop_slam:
        if not pre_crop_rgb:
            pre_crop_rgb = [0, 0]
        if not pre_crop_slam:
            pre_crop_slam = [0, 0]
        obbs = obbs.crop_bb2(left_top_rgb=pre_crop_rgb, left_top_slam=pre_crop_slam)
    if factor_rgb or factor_slam:
        if not factor_slam:
            factor_slam = 1.0
        if not factor_rgb:
            factor_slam = 1.0
        obbs = obbs.scale_bb2(scale_rgb=1.0 / factor_rgb, scale_slam=1.0 / factor_slam)
    if post_crop_rgb or post_crop_slam:
        if not post_crop_rgb:
            post_crop_rgb = [0, 0]
        if not post_crop_slam:
            post_crop_slam = [0, 0]
        obbs = obbs.crop_bb2(left_top_rgb=post_crop_rgb, left_top_slam=post_crop_slam)

    return obbs
