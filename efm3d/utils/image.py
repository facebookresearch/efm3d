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

from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Some globals for opencv drawing functions.
BLU = (255, 0, 0)
GRN = (0, 255, 0)
RED = (0, 0, 255)
WHT = (255, 255, 255)
BLK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_PT = (5, 15)
FONT_SZ = 0.5
FONT_TH = 1.0


def string2color(string):
    string = string.lower()
    if string == "white":
        return WHT
    elif string == "green":
        return GRN
    elif string == "red":
        return RED
    elif string == "black":
        return BLK
    elif string == "blue":
        return BLU
    else:
        raise ValueError("input color string %s not supported" % string)


def normalize(img, robust=0.0, eps=1e-6):
    if isinstance(img, torch.Tensor):
        vals = img.view(-1).cpu().numpy()
    elif isinstance(img, np.ndarray):
        vals = img.flatten()

    if robust > 0.0:
        v_min = np.quantile(vals, robust)
        v_max = np.quantile(vals, 1.0 - robust)
    else:
        v_min = vals.min()
        v_max = vals.max()
    # make sure we are not dividing by 0
    dv = max(eps, v_max - v_min)
    # normalize to 0-1
    img = (img - v_min) / dv
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
    elif isinstance(img, np.ndarray):
        img = img.clip(0, 1)
    return img


def put_text(
    img: np.ndarray,
    text: str,
    scale: float = 1.0,
    line: int = 0,
    color: Tuple[Tuple, str] = WHT,
    font_pt: Optional[Tuple[int, int]] = None,
    truncate: int = None,
):
    """Writes text with a shadow in the back at various lines and autoscales it.

    Args:
        image: image HxWx3 or BxHxWx3, should be uint8 for anti-aliasing to work
        text: text to write
        scale: 0.5 for small, 1.0 for normal, 1.5 for big font
        line: vertical line to write on (0: first, 1: second, -1: last, etc)
        color: text color, tuple of BGR integers between 0-255, e.g. (0,0,255) is red,
               can also be a few strings like "white", "black", "green", etc
        truncate: if not None, only show the first N characters
    Returns:
        image with text drawn on it

    """
    if isinstance(img, list) or len(img.shape) == 4:  # B x H x W x 3
        for i in range(len(img)):
            img[i] = put_text(img[i], text, scale, line, color, font_pt, truncate)
    else:  # H x W x 3
        if truncate and len(text) > truncate:
            text = text[:truncate] + "..."  # Add "..." to denote truncation.
        height = img.shape[0]
        scale = scale * (height / 320.0)
        wht_th = max(int(FONT_TH * scale), 1)
        blk_th = 2 * wht_th
        text_ht = 15 * scale
        if not font_pt:
            font_pt = int(FONT_PT[0] * scale), int(FONT_PT[1] * scale)
            font_pt = font_pt[0], int(font_pt[1] + line * text_ht)
        if line < 0:
            font_pt = font_pt[0], int(font_pt[1] + (height - text_ht * 0.5))
        cv2.putText(img, text, font_pt, FONT, FONT_SZ * scale, BLK, blk_th, lineType=16)

        if isinstance(color, str):
            color = string2color(color)

        cv2.putText(
            img, text, font_pt, FONT, FONT_SZ * scale, color, wht_th, lineType=16
        )
    return img


def rotate_image90(image: np.ndarray, k: int = 3):
    """Rotates an image and then re-allocates memory to avoid problems with opencv
    Input:
        image: numpy image, HxW or HxWxC
        k: number of times to rotate by 90 degrees counter clockwise
    Returns
        rotated image: numpy image, HxW or HxWxC
    """
    return np.ascontiguousarray(np.rot90(image, k=k))


def smart_resize(
    image: np.ndarray, height: int = -1, width: int = -1, pad_image: bool = False
):
    """Resize with opencv, auto-inferring height or width to maintain aspect ratio."""
    if image.ndim == 4:
        return np.stack([smart_resize(im, height, width, pad_image) for im in image])
    assert image.ndim == 3, "only three channel image currently supported"
    if width == -1 and height == -1:
        return image
    hh, ww = image.shape[0], image.shape[1]
    if width == -1:
        width = int(round((float(ww) / float(hh)) * height))
        width = int(width / 2) * 2  # enforce divisible by 2
    if height == -1:
        height = int(round((float(hh) / float(ww)) * width))
        height = int(height / 2) * 2  # enforce divisible by 2
    if pad_image:
        ar_orig = ww / hh
        ar_new = width / height

        if ar_new > ar_orig:  # pad the sides.
            h_scale = height / hh
            new_w = h_scale * ww
            pad = (width - new_w) / 2
            pad_before = int(pad / h_scale)
            dtype = image.dtype
            pad_img = np.zeros((hh, pad_before, 3), dtype=dtype)
            image = np.hstack([pad_img, image, pad_img])
        elif ar_new < ar_orig:  # pad the top and bottom
            w_scale = width / ww
            new_h = w_scale * hh
            pad = (height - new_h) / 2
            pad_before = int(pad / w_scale)
            dtype = image.dtype
            pad_img = np.zeros((pad_before, ww, 3), dtype=dtype)
            image = np.vstack([pad_img, image, pad_img])

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def torch2cv2(
    img: Union[np.ndarray, torch.Tensor],
    rotate: bool = False,
    rgb2bgr: bool = True,
    ensure_rgb: bool = False,
    apply_colormap: Optional[str] = None,
    robust_quant: float = 0.0,
):
    """
    Converts numpy/torch float32 image [0,1] CxHxW to numpy uint8 [0,255] HxWxC

    Args:
        img: image CxHxW float32 image
        rotate: if True, rotate image 90 degrees
        rgb2bgr: convert image to BGR
        ensure_rgb: ensure RGB if True (i.e. replicate the single color channel 3 times)
        apply_colormap: apply colormap if specified (matplotlib color map names
            i.e. "jet") to a single channel image. Overwrites ensure_rgb. This
            lets you display single channel images outside the 0-1 range. (image
            is normalized to [0,1] before applying the colormap.)
        robust_quant: quantile to robustly compute min and max for normalization of the image.
    """

    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            if img.shape[0] == 1:
                # pre-serve old way of just squeezing 0th dim
                img = img[0]
            else:
                # run torch2cv2 on all frames of the video
                return np.stack(
                    [
                        torch2cv2(
                            im,
                            rotate,
                            rgb2bgr,
                            ensure_rgb,
                            apply_colormap,
                            robust_quant,
                        )
                        for im in img
                    ]
                )
        img = img.data.cpu().float().numpy()
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    # CxHxW -> HxWxC
    img = img.transpose(1, 2, 0)
    if img.shape[2] == 1 and apply_colormap is not None:
        # make sure to normalize so min is 0 and max is 1.
        img = normalize(img, robust=robust_quant)
        cm = plt.cm.get_cmap(apply_colormap)
        img = cm(img[:, :, 0])[:, :, :3]
    img_cv2 = (img * 255.0).astype(np.uint8)

    if rgb2bgr:
        img_cv2 = img_cv2[:, :, ::-1]
    if rotate:
        img_cv2 = rotate_image90(img_cv2)
    else:
        img_cv2 = np.ascontiguousarray(img_cv2)
    if ensure_rgb and img_cv2.shape[2] == 1:
        img_cv2 = img_cv2[:, :, 0]
    if ensure_rgb and img_cv2.ndim == 2:
        img_cv2 = np.stack([img_cv2, img_cv2, img_cv2], -1)
    return img_cv2


def numpy2mp4(imgs, output_path, fps=10):
    """
    Convert a numpy array to mp4.

    imgs: T, H, W, 3
    """
    T, H, W, C = imgs.shape
    assert C == 3, "input image should be 3-channel"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        out.write(imgs[i])
    out.release()
