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

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def cnn_weight_initialization(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors, taken from
    https://github.com/huggingface/pytorch-image-models/blob/d7b55a9429f3d56a991e604cbc2e9fdf1901612f/timm/models/layers/norm.py#L26
    """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)


class UpsampleCNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        first_hidden_dim: int = 32,
        final_dim: int = 1,
        upsample_power: int = 4,
        fix_hidden_dim: bool = True,
    ):
        """
        Upsample a feature map by a given factor = 2^upsample_power

        Args:
            input_dim (int): number of input channels
            first_hidden_dim (int): the first hidden layer output dimension. If set to -1, we use the input dimension.
            final_dim (int): number of output channels
            upsample_power (int): 2^upsample_power is the factor of image resolution upsampling
            fix_hidden_dim (bool): if True, all layers have the same hidden dims. Otherwise, hidden dims are subsequently halved by 2x starting from first_hidden_dim
        """
        super(UpsampleCNN, self).__init__()
        assert upsample_power <= 4, "only upsampling power <= 4 is supported"

        if fix_hidden_dim:
            # all layers have the same hidden dims
            c = [first_hidden_dim] * (upsample_power + 1)
        else:
            first_hidden_dim = first_hidden_dim if first_hidden_dim > 0 else input_dim
            assert (
                first_hidden_dim // 2 ** (upsample_power) >= 1
            ), f"first_hidden_dim must be at least {2 ** (upsample_power)}, but got {first_hidden_dim}."
            # subsequently halve the hidden dim by 2x
            c = [first_hidden_dim] + [
                first_hidden_dim // 2 ** (i + 1) for i in range(upsample_power)
            ]

        self.conv1 = nn.Conv2d(input_dim, c[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c[0])

        if upsample_power >= 1:
            self.conv1u = nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1)
            self.bn1u = nn.BatchNorm2d(c[1])
        if upsample_power >= 2:
            self.conv2u = nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1)
            self.bn2u = nn.BatchNorm2d(c[2])
        if upsample_power >= 3:
            self.conv3u = nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1)
            self.bn3u = nn.BatchNorm2d(c[3])
        if upsample_power >= 4:
            self.conv4u = nn.Conv2d(c[3], c[4], kernel_size=3, stride=1, padding=1)
            self.bn4u = nn.BatchNorm2d(c[4])
        self.conv_final = nn.Conv2d(
            c[-1], final_dim, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_power = upsample_power
        cnn_weight_initialization(self.modules())

        print(f"==> [UpsampleCNN]: intialized with hidden layers: {c}")

    def forward(self, x, force_hw=None):
        """
        Inputs:
            x : torch.Tensor : Bx(T)xCxhxw tensor
            force_hw: (int, int) : tuple of ints of height and width to be forced upsampled to
        Returns:
            x : torch.Tensor: Upsampled to Bx(T)xCxHxW, where H = h*(upsample_power**2) and W = w*(upsample_power**2)
        """
        ndim = x.ndim
        if ndim == 5:
            T = x.shape[1]
            x = einops.rearrange(x, "b t c h w -> (b t) c h w")
        x = self.relu(self.bn1(self.conv1(x)))
        if self.upsample_power >= 1:
            x = self.upsample(x)
            x = self.relu(self.bn1u(self.conv1u(x)))
        if self.upsample_power >= 2:
            x = self.upsample(x)
            x = self.relu(self.bn2u(self.conv2u(x)))
        if self.upsample_power >= 3:
            x = self.upsample(x)
            x = self.relu(self.bn3u(self.conv3u(x)))
        if self.upsample_power >= 4:
            x = self.upsample(x)
            x = self.relu(self.bn4u(self.conv4u(x)))

        # Force upsampling, useful for patch_size=14 ViTs for example.
        if force_hw is not None and (
            x.shape[-2] != force_hw[0] or x.shape[-1] != force_hw[1]
        ):
            x = torch.nn.functional.interpolate(x, size=force_hw, mode="bilinear")

        x = self.conv_final(x)

        if ndim == 5:
            x = einops.rearrange(x, "(b t) c h w -> b t c h w", t=T)
        return x


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm for channels of '3D' spatial NCDHW tensors, taken from
    https://github.com/huggingface/pytorch-image-models/blob/d7b55a9429f3d56a991e604cbc2e9fdf1901612f/timm/models/layers/norm.py#L26
    """

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 4, 1),  # NCHW
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 4, 1, 2, 3)


class UpConv3d(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )
        self.cnn_up = torch.nn.Conv3d(dim_in, dim_out, 3, stride=1, padding=1)

        self.norm = torch.nn.BatchNorm3d(dim_out)
        cnn_weight_initialization(self.modules())

    def forward(self, x_up):
        assert x_up.shape[1] == self.dim_in, f"{x_up.shape}, {self.dim_in}"
        x_up = self.upsample(x_up)
        return self.norm(self.cnn_up(x_up))


class FpnUpConv3d(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )
        self.cnn_up = torch.nn.Conv3d(dim_in, dim_out, 3, stride=1, padding=1)
        self.cnn_lat = torch.nn.Conv3d(dim_out, dim_in, 1)

        self.norm = torch.nn.BatchNorm3d(dim_out)
        cnn_weight_initialization(self.modules())

    def forward(self, x_up, x_lat):
        assert x_up.shape[1] == self.dim_in, f"{x_up.shape}, {self.dim_in}"
        assert x_lat.shape[1] == self.dim_out, f"{x_lat.shape}, {self.dim_out}"
        x_up = self.upsample(x_up)
        x_lat = self.cnn_lat(x_lat)
        return self.norm(self.cnn_up(x_up + x_lat))


class InvBottleNeck3d(torch.nn.Module):
    def __init__(self, dim_in, dim_out, stride: int = 1, expansion: float = 1.0):
        super().__init__()
        self.dim_hidden = int(math.floor(dim_in * expansion))
        self.stride = stride
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm3d(self.dim_out)

        self.cnn1 = torch.nn.Conv3d(dim_in, self.dim_hidden, 1)
        self.cnn2 = torch.nn.Conv3d(
            self.dim_hidden, self.dim_hidden, 3, stride=stride, padding=1
        )
        self.cnn3 = torch.nn.Conv3d(self.dim_hidden, dim_out, 1)
        cnn_weight_initialization(self.modules())

    def forward(self, x):
        y = self.relu(self.cnn1(x))
        y = self.relu(self.cnn2(y))
        y = self.cnn3(y)
        if self.stride != 1 or self.dim_in != self.dim_out:
            return self.norm(y)
        return self.norm(y + x)


class InvResnetBlock3d(torch.nn.Module):
    def __init__(
        self, dim_in, dim_out, num_bottles, in_stride: int = 1, expansion: float = 1.0
    ):
        super().__init__()
        self.inv_bottles = torch.nn.ModuleList(
            [InvBottleNeck3d(dim_in, dim_out, in_stride, expansion)]
        )
        for _ in range(1, num_bottles):
            self.inv_bottles.append(InvBottleNeck3d(dim_out, dim_out, 1, expansion))

        self.num_bottles = num_bottles

    def forward(self, x):
        for i in range(self.num_bottles):
            x = self.inv_bottles[i](x)
        return x


class InvResnetFpn3d(torch.nn.Module):
    def __init__(self, dims, num_bottles, strides, expansions, freeze=False):
        super().__init__()

        assert len(dims) == len(num_bottles) + 1
        assert len(dims) == len(strides) + 1
        assert len(dims) == len(expansions) + 1
        assert strides[0] == 1
        assert all([s == 2 for s in strides[1:]])

        self.block1 = InvResnetBlock3d(
            dims[0], dims[1], num_bottles[0], strides[0], expansions[0]
        )
        self.block2 = InvResnetBlock3d(
            dims[1], dims[2], num_bottles[1], strides[1], expansions[1]
        )
        self.block3 = InvResnetBlock3d(
            dims[2], dims[3], num_bottles[2], strides[2], expansions[2]
        )
        self.block4 = InvResnetBlock3d(
            dims[3], dims[4], num_bottles[3], strides[3], expansions[3]
        )
        self.fpn1 = FpnUpConv3d(dims[2], dims[1])
        self.fpn2 = FpnUpConv3d(dims[3], dims[2])
        self.fpn3 = FpnUpConv3d(dims[4], dims[3])

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x = self.block4(x3)

        x = self.fpn3(x, x3)
        del x3
        x = self.fpn2(x, x2)
        del x2
        x = self.fpn1(x, x1)
        del x1
        return x


class VolumeCNN(nn.Module):
    """A 3d UNet structure with take in a `hidden_dims` vector (e.g. [c0, c1, c2, c3],
    c0 <= c1 <= c2 <= c3). It outputs a shared feature layer with ReLU and BN applied.
    The shape on the channel dimension looks like c0->c1->c2->c3->c2->c1->c0.
    """

    def __init__(self, hidden_dims, conv3=nn.Conv3d, freeze=False):
        super(VolumeCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

        c0, c1, c2, c3 = tuple(hidden_dims)
        self.conv1 = conv3(c0, c1, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv3(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv3(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv2u = conv3(c2 + c3, c2, kernel_size=3, stride=1, padding=1)
        self.conv1u = conv3(c1 + c2, c1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(c1)
        self.bn2 = nn.BatchNorm3d(c2)
        self.bn3 = nn.BatchNorm3d(c3)
        self.bn2u = nn.BatchNorm3d(c2)
        self.bn1u = nn.BatchNorm3d(c1)

        cnn_weight_initialization(self.modules())
        self.out_dim = c1

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        # Simple U-Net like structure.
        conv1 = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(conv1)
        conv2 = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(conv2)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.relu(self.bn2u(self.conv2u(x)))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.relu(self.bn1u(self.conv1u(x)))
        return x


class VolumeCNNHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        final_dim,
        num_layers=2,
        name="",
        bias=None,
        freeze=False,
    ):
        super(VolumeCNNHead, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)

        assert num_layers in [2, 3, 4], f"num_layers {num_layers} must be 2, 3, or 4"

        # first conv layer is the same for all num_layers = {2,3,4}
        self.conv1 = torch.nn.Conv3d(
            input_dim, hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(hidden_dim)

        if num_layers == 2:
            self.conv2 = torch.nn.Conv3d(hidden_dim, final_dim, kernel_size=1)
        elif num_layers == 3:
            self.conv2 = torch.nn.Conv3d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
            )
            self.conv3 = torch.nn.Conv3d(hidden_dim, final_dim, kernel_size=1)
            self.bn2 = nn.BatchNorm3d(hidden_dim)
        elif num_layers == 4:
            self.conv2 = torch.nn.Conv3d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
            )
            self.conv3 = torch.nn.Conv3d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1
            )
            self.conv4 = torch.nn.Conv3d(hidden_dim, final_dim, kernel_size=1)
            self.bn2 = nn.BatchNorm3d(hidden_dim)
            self.bn3 = nn.BatchNorm3d(hidden_dim)

        cnn_weight_initialization(self.modules())
        model_msg = f"==> Init {num_layers}-layer 3DCNN with {hidden_dim} hidden dims and {final_dim} outputs"
        if name:
            model_msg += f", with name {name}"
        print(model_msg)

        if bias:
            print("overwriting bias to %f" % bias)
            self.conv2.bias.data.fill_(bias)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        if self.num_layers == 2:
            x = self.conv2(x)
        elif self.num_layers == 3:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.conv3(x)
        elif self.num_layers == 4:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
        return x


class ResidualConvUnit3d(nn.Module):
    # From "Vision Transformers for Dense Prediction": https://arxiv.org/abs/2103.13413
    # adapted from https://github.com/isl-org/DPT/blob/main/dpt/blocks.py
    def __init__(self, features, kernel_size):
        super().__init__()
        assert kernel_size % 1 == 0, "Kernel size needs to be odd"
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
            nn.Conv3d(features, features, kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x) + x


class FeatureFusionBlock3d(nn.Module):
    # Fro "Vision Transformers for Dense Prediction": https://arxiv.org/abs/2103.13413
    # adapted from https://github.com/isl-org/DPT/blob/main/dpt/blocks.py
    def __init__(self, features, kernel_size, with_skip=True):
        super().__init__()
        self.with_skip = with_skip
        if self.with_skip:
            self.resConfUnit1 = ResidualConvUnit3d(features, kernel_size)

        self.resConfUnit2 = ResidualConvUnit3d(features, kernel_size)

    def forward(self, x, skip_x=None):
        if skip_x is not None:
            assert self.with_skip and skip_x.shape == x.shape
            x = self.resConfUnit1(x) + skip_x

        x = self.resConfUnit2(x)
        return x


class VolumeResnet(nn.Module):
    def __init__(self, hidden_dims, conv3=nn.Conv3d, freeze=False):
        super(VolumeResnet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )

        c0, c1, c2, c3 = tuple(hidden_dims)
        self.resconv1 = ResidualConvUnit3d(c0, kernel_size=3)
        self.conv1 = conv3(c0, c1, kernel_size=3, stride=1, padding=1)

        self.resconv2 = ResidualConvUnit3d(c1, kernel_size=3)
        self.conv2 = conv3(c1, c2, kernel_size=3, stride=1, padding=1)

        self.resconv3 = ResidualConvUnit3d(c2, kernel_size=3)
        self.conv3 = conv3(c2, c3, kernel_size=3, stride=1, padding=1)

        self.conv2u = conv3(c2 + c3, c2, kernel_size=3, stride=1, padding=1)
        self.conv1u = conv3(c1 + c2, c1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(c1)
        self.bn2 = nn.BatchNorm3d(c2)
        self.bn3 = nn.BatchNorm3d(c3)
        self.bn2u = nn.BatchNorm3d(c2)
        self.bn1u = nn.BatchNorm3d(c1)

        cnn_weight_initialization(self.modules())
        self.out_dim = c1

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x):
        # Simple U-Net like structure.
        conv1 = self.relu(self.bn1(self.conv1(self.resconv1(x))))
        x = self.pool(conv1)
        conv2 = self.relu(self.bn2(self.conv2(self.resconv2(x))))
        x = self.pool(conv2)
        x = self.relu(self.bn3(self.conv3(self.resconv3(x))))
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.relu(self.bn2u(self.conv2u(x)))
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.relu(self.bn1u(self.conv1u(x)))
        return x
