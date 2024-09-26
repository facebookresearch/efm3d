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

from typing import List

import einops
import torch

import torch.nn.functional as F
from efm3d.model.dinov2_utils import dino_name_mappings, DinoV2Wrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class ImageToDinoV2Tokens(torch.nn.Module):
    """
    Tokenize an image snippet using DinoV2.
    """

    def __init__(
        self,
        dinov2_name: str = "vit_small",
        freeze: bool = False,
        handle_rotated_data: bool = True,
        dim_out: int = 768,  # ignored if add_linear_layer = False
        add_lin_layer: bool = False,  # add a linear layer to get to any output dim
        out_patch_size: int = 14,  # 14 is default but can set to 16 to get resampled into a more compatible feature size
        multilayer_output: bool = False,  # if True, return a list of features
        ckpt_path: str = "",  # if not empty, load the pretrained weights from the given path
    ):
        super().__init__()
        assert dinov2_name in dino_name_mappings.keys()
        self.freeze = freeze
        self.handle_rotated_data = handle_rotated_data

        self.model = DinoV2Wrapper(
            dinov2_name, multilayer_output=multilayer_output, ckpt_path=ckpt_path
        )

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

        self.lin = None
        if not add_lin_layer:
            assert (
                dim_out == self.model.feat_dim
            ), f"dim_out must match DinoV2 feature dim if not adding linear layer, but get dim_out: {dim_out} and feat_dim: {self.model.feat_dim}."
        else:
            self.lin = torch.nn.Linear(self.model.feat_dim, dim_out)
            print(
                f"Add linear layer to project features from {self.model.feat_dim} to {dim_out}"
            )
        self.dim_out = dim_out

        logger.info(
            f"DinoV2 InputTokenizer {dinov2_name}, is frozen {freeze}, dim_out of {self.dim_out}"
        )
        self.out_patch_size = out_patch_size

    def feat_dim(self):
        return self.dim_out

    def patch_size(self):
        return self.out_patch_size

    def post_process(self, feats, B, T, out_size=None):
        """
        Post processing to convert Dino features, e.g. feature interpolation to the desired size,
        handling Aria image rotation, the linear mapping to increase feature dimension.

        Args:
            feats: [B x T x C x H x W]
            B: batch size
            T: number of frames
            out_size: (h, w) token feature map output size, if None, don't resize the feature map size.
        """
        if out_size is not None:
            # resize to desired size
            feats = F.interpolate(feats, out_size, mode="bilinear")
        if self.handle_rotated_data:
            feats = torch.rot90(feats, 1, [-2, -1])
        # to token sequence BxNxC
        feats = einops.rearrange(feats, "(b t) c h w -> b t h w c", b=B, t=T)
        if self.lin is not None:
            # increase feature dimension to desired output dimension
            feats = self.lin(feats)
        return feats

    def forward_resize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Return the round-up image size to match a multiple of patch size, which will be used as the input size
        to the DinoV2 model.

        Args:
            img: [..., H, W] image tensor
        """
        H_ori, W_ori = img.shape[-2:]
        # Dino models have a fixed patch size of 14
        H_new = math.ceil(H_ori / 14) * 14
        W_new = math.ceil(W_ori / 14) * 14
        if H_new != H_ori or W_new != W_ori:
            img = F.interpolate(
                img, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
        return img

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [B x T x C x H x W] A sequence / snippet of Image Frames (typically used for Pose Regression)
        """
        assert img.dim() == 5, f"expecting BxTxCxHxW but got {img.shape}"
        B, T, C, H, W = img.shape
        if self.handle_rotated_data:
            # rotate image 90 degrees clockwise to give it expected upright
            # orientation for pretrained resnet
            img = torch.rot90(img, 1, [-1, -2])
        # get batch image for resnet
        img = einops.rearrange(img, "b t c h w -> (b t) c h w")

        H_ori, W_ori = img.shape[-2:]
        img = self.forward_resize(img)
        feats = self.model.forward(img)

        out_size = None
        # if output_patch_size is not 14, then we need to resize the feature map to the desired size
        if self.patch_size() != 14:
            out_size = H_ori // self.patch_size(), W_ori // self.patch_size()
            if (
                out_size[0] * self.patch_size() != H_ori
                or out_size[1] * self.patch_size() != W_ori
            ):
                logger.warning(
                    f"Image size {(H_ori, W_ori)} not divisible by output patch size {self.patch_size()}"
                )

        if isinstance(feats, List):
            feats = [self.post_process(f, B, T, out_size) for f in feats]
        else:
            feats = self.post_process(feats, B, T, out_size)
        return feats
