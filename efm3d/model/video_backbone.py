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
from abc import ABC, abstractproperty
from typing import Dict, List, Optional

import einops
import torch
import torch.nn as nn

from efm3d.aria.aria_constants import ARIA_IMG
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class VideoBackbone(torch.nn.Module, ABC):
    """
    Snippet Feature Backbone runs image feature extractors for video snippets.
    This lets us easily try out various different backbones.
    """

    def __init__(
        self,
        video_streams: Optional[List[str]] = None,
        pass_batch: bool = True,
        feat_dim: Optional[int] = None,
        correct_vignette: bool = False,
        optimize_vignette: bool = False,
        ensure_rgb: bool = False,
    ):
        """
        Args:
            video_streams: a list of video streams to extract features for.
                Supported is "rgb", "slaml", "slamr".
            pass_batch: pass whole batch dict to the forward_impl if set to
                true. Otherwise passing the image tensors associated with the stream,
                instead of passing a dictionary of batch.
            correct_vignette: correct vignette for the image streams.
            optimize_vignette: optimize vignette correction for the image streams. This enables backpropagating into the vignettes.
            ensure_rgb: if set to true, will ensure that the output streams are all 3 channels.
        """
        super().__init__()
        self.ensure_rgb = ensure_rgb
        self._feat_dim = -1
        if feat_dim is not None:
            # Note that FPN will be constructed if feat_dim is passed in by construction (and fpn_levels > 0).
            self.feat_dim = feat_dim
        self.video_streams = video_streams
        if self.video_streams is None:
            self.video_streams = ["rgb"]
        self.pass_batch = pass_batch
        self.stream_to_id = {"rgb": 0, "slaml": 1, "slamr": 2}
        assert set(self.video_streams).issubset(
            set(self.stream_to_id.keys())
        ), f"{self.video_streams} are not all valid (need to be a subset of {self.stream_to_id.keys()})"

        self.vignette_correction = {}
        self.vignette_correction = nn.ModuleDict(self.vignette_correction)

    @property
    def feat_dim(self):
        return self._feat_dim

    @feat_dim.setter
    def feat_dim(self, _feat_dim: int):
        self._feat_dim = _feat_dim

    @abstractproperty
    def patch_size(self):
        pass

    def forward_impl(self, img, stream) -> Dict[str, torch.Tensor]:
        """
        forward_impl should return a dict with keys of the desired streams mapping to the extracted feature images.
        Other additional outputs can be added as well as needed. A suggested way
        to return additional outputs is to nest their keys under the
        corresponding streams such as: "rgb/feature_scale2" for additional
        feature outputs for the rgb stream.
        """
        pass

    def forward(self, batch):
        out = {}
        if self.pass_batch:
            out = self.forward_impl(batch, self.video_streams)
        else:
            for stream in self.video_streams:
                # if we have a batch dictionary retrieve the corresponding video. If not assume that we are just
                key = ARIA_IMG[self.stream_to_id[stream]]
                if isinstance(batch, dict) and key in batch:
                    im = batch[key]
                elif isinstance(batch, torch.Tensor) and len(self.video_streams) == 1:
                    im = batch
                else:
                    raise ValueError(
                        f"batch not passed correctly {type(batch)} for video streams {self.video_streams}, {key}"
                    )
                if self.ensure_rgb and stream in ["slaml", "slamr"]:
                    # greyscale -> rgb
                    im = torch.cat([im, im, im], 2)
                # correct vignette if desired
                if stream in self.vignette_correction:
                    im = self.vignette_correction[stream](im)
                # accumulate updates into one flat dict
                out.update(self.forward_impl(im, stream))

        assert isinstance(
            out, dict
        ), f"Output of forward must be of type dict, got {type(out)}"
        assert set(self.video_streams).issubset(set(out.keys()))
        return out


class VideoBackboneDinov2(VideoBackbone):
    """
    Get a snippet feature extractor from Dino v2.
    """

    def __init__(
        self,
        image_tokenizer: DictConfig,
        video_streams: Optional[List[str]] = None,
        freeze_encoder: bool = False,
        correct_vignette: bool = False,
        optimize_vignette: bool = False,
    ):
        super().__init__(
            video_streams=video_streams,
            pass_batch=False,
            correct_vignette=correct_vignette,
            optimize_vignette=optimize_vignette,
        )
        self.image_tokenizer = image_tokenizer
        if isinstance(image_tokenizer, DictConfig):
            self.image_tokenizer = instantiate(self.image_tokenizer)

        # assert freeze_encoder == self.image_tokenizer.freeze

        # get feature dimension
        self.feat_dim = self.image_tokenizer.feat_dim()
        self._patch_size = self.image_tokenizer.patch_size()
        logging.info("feature dim is %d" % self.feat_dim)
        logging.info("down_scale factor is %d" % self.patch_size)

    @property
    def patch_size(self):
        return self._patch_size

    def forward_impl(self, img, stream):
        # Run tokenizer. handles SLAM images internally.
        img_tokens = self.image_tokenizer.forward(img)
        # BxTxHxWxC -> B, T, C, H, W

        if isinstance(img_tokens, List):
            return {
                stream: [
                    einops.rearrange(t, "b t h w c -> b t c h w") for t in img_tokens
                ]
            }

        return {stream: einops.rearrange(img_tokens, "b t h w c -> b t c h w")}
