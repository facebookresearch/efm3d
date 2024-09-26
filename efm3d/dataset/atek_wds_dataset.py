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

import glob
import tarfile

import numpy as np
import torch
import webdataset as wds
from efm3d.aria import CameraTW, DEFAULT_CAM_DATA_SIZE, ObbTW, PoseTW, TensorWrapper
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_IMG,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_OBB_PADDED,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
    ARIA_POSE_T_SNIPPET_RIG,
    ARIA_POSE_T_WORLD_RIG,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm


def batchify(datum, device=None):
    # Add batch dimension
    for key in datum:
        if isinstance(datum[key], (torch.Tensor, TensorWrapper)):
            datum[key] = datum[key][None, ...].to(device)
            if device is not None:
                datum[key] = datum[key].to(device)
        else:
            datum[key] = [datum[key]]
    return datum


def unbatchify(datum):
    # Remove batch dimension
    for key in datum:
        if isinstance(datum[key], (torch.Tensor, TensorWrapper, list)):
            datum[key] = datum[key][0]
    return datum


class AtekWdsStreamDataset:
    """Sample 2s/1s WDS dataset to specified snippet length and stride"""

    def __init__(
        self,
        data_path,
        atek_to_efm_taxonomy,
        snippet_length_s=1.0,
        stride_length_s=0.1,
        wds_length_s=2.0,
        fps=10,
        max_snip=99999999,
    ):
        self.snippet_length_s = snippet_length_s
        self.stride_length_s = stride_length_s
        self.wds_length_s = wds_length_s
        # wds snippets should always be generated half overlapped
        self.wds_stride_s = wds_length_s // 2
        self.fps = fps
        self.max_snip = max_snip

        tar_list = sorted(glob.glob(f"{data_path}/*.tar"))
        sn = set()
        with tarfile.TarFile(tar_list[0], "r") as tar:
            for member in tar.getmembers():
                sn.add(member.name.split(".")[0])
        self.samples_per_tar = len(sn)
        self.num_tars = len(tar_list)

        self.dataset = load_atek_wds_dataset_as_efm(
            urls=tar_list,
            freq=fps,
            snippet_length_s=wds_length_s,  # Need to use `wds_length` for model adaptor!
            atek_to_efm_taxonomy_mapping_file=atek_to_efm_taxonomy,
        )
        self.dataloader = iter(self.dataset)

        self.frames_wds = int(self.fps * self.wds_length_s)
        self.frames_out = int(self.fps * self.snippet_length_s)
        self.frames_stride_wds = int(self.fps * self.wds_stride_s)
        self.frames_stride_out = int(self.fps * self.stride_length_s)

        self.num_rest = int(
            (self.wds_length_s - self.snippet_length_s) / self.stride_length_s
        )
        self.num_first = int(1 + self.num_rest)
        self.num_snippets = (
            self.num_first + (self.samples_per_tar * self.num_tars - 1) * self.num_rest
        )

        # for iteration
        self.first = True
        self.wds_snippet = None
        self.snip_idx = 0
        self.global_idx = 0

    def __len__(self):
        return min(self.num_snippets, self.max_snip)

    def sample_snippet_(self, snippet, start, end):
        # time crop
        sample = snippet.copy()
        for k in sample:
            if isinstance(sample[k], (torch.Tensor, TensorWrapper)):
                if k not in [
                    ARIA_SNIPPET_T_WORLD_SNIPPET,
                    ARIA_POINTS_VOL_MIN,
                    ARIA_POINTS_VOL_MAX,
                ]:
                    sample[k] = sample[k][start:end, ...]

        return sample

    def __iter__(self):
        return self

    def if_get_next_(self):
        if self.wds_snippet is None:
            return True

        if self.first:
            return self.snip_idx >= self.num_first
        else:
            return self.snip_idx >= self.num_rest

    def __next__(self):
        if self.global_idx >= self.max_snip:
            raise StopIteration

        if self.if_get_next_():
            if self.first and self.wds_snippet is not None:
                self.first = False
            self.wds_snippet = next(self.dataloader)
            self.snip_idx = 0

        if self.first:
            start = self.snip_idx * self.frames_stride_out
        else:
            start = (self.snip_idx + 1) * self.frames_stride_out

        end = start + self.frames_out
        sample = self.sample_snippet_(self.wds_snippet, start, end)
        self.snip_idx += 1
        self.global_idx += 1
        return sample
