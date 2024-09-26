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
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from efm3d.aria.aria_constants import (
    ARIA_IMG,
    ARIA_POINTS_DIST_STD,
    ARIA_POINTS_INV_DIST_STD,
    ARIA_POINTS_WORLD,
)

from torchvision.transforms.v2._color import RandomAdjustSharpness
from webdataset import WebDataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class ColorJitter:
    """
    Applies photometric jitter to the images in the video sequence.
    """

    def __init__(
        self,
        brightness: Union[Tuple[float], float] = 0.5,
        contrast: Union[Tuple[float], float] = 0.3,
        saturation: Union[Tuple[float], float] = 0.3,
        hue: Union[Tuple[float], float] = 0.05,
        sharpness: Union[Tuple[float], float] = 2.0,
        snippet_jitter: bool = False,
    ):
        """
        Calls torchvision on the images independently in a video using:
        https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html


        brightness: how much to jitter brightness in range [0,val]
        contrast: how much to jitter contrast in range [0,val]
        saturation: how much to jitter contrast in range [0,val]
        hue: how much to jitter hue in range [-val,val]
        snippet_jitter: if true, jitter equally across the snippet
        """

        self.transform = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.snippet_jitter = snippet_jitter
        self.sharpness = sharpness

    def rnd_sharpen(self, im):
        factor = float(self.sharpness * torch.rand(1))
        sharp_fn = RandomAdjustSharpness(sharpness_factor=factor, p=1.0)
        return sharp_fn(im)

    def apply(self, im):
        im = self.transform(im)
        im = self.rnd_sharpen(im)
        return im

    def __call__(self, batch: Dict):
        for name in ARIA_IMG:
            if name in batch:
                batch[name] = batch[name].clone().detach()
                if self.snippet_jitter:
                    batch[name] = self.apply(batch[name])
                else:
                    for t in range(len(batch[name])):
                        batch[name][t] = self.apply(batch[name][t])
        return batch


class PointDrop:
    """
    Applies point drop augmentation based on the standard deviations of the points.
    A standard deviation is sampled within the provided range, and points exceeding
    the sampled standard deviation are dropped.
    Attributes:
        dropout_all_rate (float): The rate at which all points are dropped.
        inv_dist_std (List[float]): Range [min, max] of inverse distance standard deviations.
        dist_std (List[float]): Range [min, max] of distance standard deviations.
    """

    def __init__(
        self,
        dropout_all_rate: float = 0.2,
        inv_dist_std: Optional[List[float]] = None,
        dist_std: Optional[List[float]] = None,
    ):
        if inv_dist_std is None:
            inv_dist_std = [0.001, 0.03]
        if dist_std is None:
            dist_std = [0.01, 0.3]
        self.dropout_all_rate = dropout_all_rate
        self.inv_dist_std = inv_dist_std
        self.dist_std = dist_std

        assert inv_dist_std[1] >= inv_dist_std[0]
        assert dist_std[1] >= dist_std[0]

    def __call__(self, batch: Dict):
        if ARIA_POINTS_WORLD not in batch:
            return batch

        p_drop_all = torch.rand(1).item()
        if p_drop_all < self.dropout_all_rate:
            # drop all points
            batch[ARIA_POINTS_WORLD][:, :, :] = torch.nan
        else:
            # drop based on stds.
            p_w = batch[ARIA_POINTS_WORLD]
            T, N = p_w.shape[:2]

            # sample inv_dist_std
            rand_inv_dist_thres = torch.rand(1).item()
            rand_inv_dist_thres = (
                rand_inv_dist_thres * (self.inv_dist_std[1] - self.inv_dist_std[0])
                + self.inv_dist_std[0]
            )

            # sample dist_std
            rand_dist_thres = torch.rand(1).item()
            rand_dist_thres = (
                rand_dist_thres * (self.dist_std[1] - self.dist_std[0])
                + self.dist_std[0]
            )

            dropped = torch.zeros(T, N, dtype=torch.bool)
            if ARIA_POINTS_INV_DIST_STD in batch:
                drop_inv_dist_std = (
                    batch[ARIA_POINTS_INV_DIST_STD] > rand_inv_dist_thres
                )
                dropped |= drop_inv_dist_std

                logger.debug(f"drop points with max inv_dist_std {rand_inv_dist_thres}")
                logger.debug(f"drop {dropped.sum()} points.")
            if ARIA_POINTS_DIST_STD in batch:
                drop_dist_std = batch[ARIA_POINTS_DIST_STD] > rand_dist_thres
                dropped |= drop_dist_std

                logger.debug(f"drop points with max dist_std {rand_dist_thres}")
                logger.debug(f"drop {dropped.sum()} points.")
            p_w[dropped, :] = torch.nan
            batch[ARIA_POINTS_WORLD] = p_w

        return batch


class PointDropSimple:
    """
    simple point drop augmentation.
    """

    def __init__(
        self,
        max_dropout_rate: float = 0.8,
    ):
        self.max_dropout_rate = max_dropout_rate
        assert self.max_dropout_rate < 1.0 and self.max_dropout_rate > 0.0

    def __call__(self, batch: Dict):
        if ARIA_POINTS_WORLD not in batch:
            return batch

        dropout_rate = torch.rand(1).item()
        if dropout_rate > self.max_dropout_rate:
            return batch
        else:
            p_w = batch[ARIA_POINTS_WORLD]  # B, T, 3
            T, N = p_w.shape[:2]
            mask = torch.rand((T, N)) < dropout_rate
            p_w[mask, :] = torch.nan
            batch[ARIA_POINTS_WORLD] = p_w

        return batch


class PointJitter:
    """
    Applies point jitter augmentation.
    """

    def __init__(
        self,
        depth_std_scale_min: float = 1.0,
        depth_std_scale_max: float = 3.0,
    ):
        """
        Args:
            depth_std_scale_min: min scale factor for depth jitter based on depth_std
            depth_std_scale_max: max scale factor for depth jitter based on depth_std
        """
        self.depth_std_scale_max = depth_std_scale_max
        self.depth_std_scale_min = depth_std_scale_min

    def __call__(self, batch: Dict):
        if ARIA_POINTS_WORLD in batch and ARIA_POINTS_DIST_STD in batch:
            p_w = batch[ARIA_POINTS_WORLD]
            scale = (
                torch.rand(1).item()
                * (self.depth_std_scale_max - self.depth_std_scale_min)
                + self.depth_std_scale_min
            )
            std = batch[ARIA_POINTS_DIST_STD] * scale
            noise = torch.randn_like(p_w) * std.unsqueeze(-1)
            batch[ARIA_POINTS_WORLD] = p_w + noise
        return batch
