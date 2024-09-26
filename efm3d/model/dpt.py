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

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class ResidualConvUnit(nn.Module):
    # From "Vision Transformers for Dense Prediction": https://arxiv.org/abs/2103.13413
    # adapted from https://github.com/isl-org/DPT/blob/main/dpt/blocks.py
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class FeatureFusionBlock(nn.Module):
    # Fro "Vision Transformers for Dense Prediction": https://arxiv.org/abs/2103.13413
    # adapted from https://github.com/isl-org/DPT/blob/main/dpt/blocks.py
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip, "Must init with with_skip=True"
            assert (
                skip_x.shape == x.shape
            ), f"skip {skip_x.shape} and x {x.shape} shape mismatch"
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class Interpolate(nn.Module):
    """
    Interpolation module. https://github.com/isl-org/DPT/blob/main/dpt/blocks.py#L138
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DPTOri(nn.Module):
    """
    Implementation of DPT according to the paper description https://arxiv.org/pdf/2103.13413
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=256, depth=False):
        """
        input_dim: dimension of the DinoV2 tokens (384/768/...)
        hidden_dim: dense feature dimension(=256, D^{hat} in the paper) in DPT
        output_dim: final output feature dimension
        """
        super().__init__()
        self.depth = depth
        if self.depth:
            # DPT depth head https://github.com/isl-org/DPT/blob/main/dpt/models.py#L89
            self.depth_head = nn.Sequential(
                nn.Conv2d(
                    hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1
                ),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(hidden_dim // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),  # require depth to be non-negative
                nn.Identity(),
            )
            output_dim = output_dim - 1  # last dim is depth

        # 1x1 convs to map (H/p x W/p x D) -> (H/s x W/s x D^{hat})
        self.conv_0 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)
        self.conv_1 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)
        self.conv_2 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)
        self.conv_3 = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)

        # (strided) convs for upsampling (feat_0/1/2) and downsample (feat_3)
        # image - WxW, padding - P, kernel - FxF, stride - S
        # conv size - (W-F+2P) / S + 1
        # transpose conv size - (H-1)*S+F-2P
        self.resample_conv0 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, 3, stride=4, padding=0
        )
        self.resample_conv1 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim, 3, stride=2, padding=1
        )
        self.resample_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1)
        self.resample_conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)

        # fusion blocks
        self.ref_0 = FeatureFusionBlock(hidden_dim, 3)
        self.ref_1 = FeatureFusionBlock(hidden_dim, 3)
        self.ref_2 = FeatureFusionBlock(hidden_dim, 3)
        self.ref_3 = FeatureFusionBlock(hidden_dim, 3, with_skip=False)

        # final upsample head
        self.conv_up1 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.conv_final = nn.Conv2d(
            hidden_dim, output_dim, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        """
        feats: tokens from multi-layers, for ViT-base these are [3,6,9,12] (starting from 1, not 0)
        """
        assert (
            len(feats) == 4
        ), "feats must be multi-level as a list of 4 tensors, probably set model.video_backbone.image_tokenizer.multilayer_output=True"
        ndim = feats[0].ndim
        if ndim == 5:
            T = feats[0].shape[1]
            feats = [einops.rearrange(f, "b t c h w -> (b t) c h w") for f in feats]

        # [T, D, H/p, W/p]
        feats[0] = self.conv_0(feats[0])
        feats[1] = self.conv_1(feats[1])
        feats[2] = self.conv_2(feats[2])
        feats[3] = self.conv_3(feats[3])

        # add single-side padding here after feat0 and feat1 to make upsampling 4x and 2x the token map size
        padding = (0, 1, 0, 1)  # left, right, top, bottom
        feats[0] = self.resample_conv0(feats[0])
        feats[0] = F.pad(feats[0], padding, mode="constant", value=0)
        feats[1] = self.resample_conv1(feats[1])
        feats[1] = F.pad(feats[1], padding, mode="constant", value=0)
        feats[2] = self.resample_conv2(feats[2])
        feats[3] = self.resample_conv3(feats[3])

        out = self.ref_3(feats[3], None)
        out = interpolate(
            out, size=feats[2].shape[-2:], mode="bilinear", align_corners=True
        )
        out = self.ref_2(feats[2], out)
        out = interpolate(
            out, size=feats[1].shape[-2:], mode="bilinear", align_corners=True
        )
        out = self.ref_1(feats[1], out)
        out = interpolate(
            out, size=feats[0].shape[-2:], mode="bilinear", align_corners=True
        )
        out = self.ref_0(feats[0], out)
        h, w = feats[0].shape[-2:]
        feat = interpolate(
            out, size=(h * 2, w * 2), mode="bilinear", align_corners=True
        )

        # upsample by 2x (In the paper DPT outputs 1/2 original size feature maps)
        out = self.relu(self.conv_up1(feat))
        h, w = out.shape[-2:]
        out = interpolate(out, size=(h * 2, w * 2), mode="bilinear", align_corners=True)
        out = self.conv_final(out)

        if self.depth:
            inv_depth = self.depth_head(feat) + 1e-3  # predict inv depth, add epsilon
            out = torch.cat([out, inv_depth], dim=1)

        if ndim == 5:
            out = einops.rearrange(out, "(b t) c h w -> b t c h w", t=T)
        return out
