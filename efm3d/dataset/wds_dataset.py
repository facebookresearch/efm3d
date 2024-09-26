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


def convert_to_aria_multimodal_dataset(sample):
    """
    Convert a data sample from Aria multimodal data in webdataset format
    to training/validation sample format.
    """

    def to_mm_key(k, end_separator="."):
        k = k[: k.rfind(end_separator)]  # remove suffix
        # move keys back to the "/" convention from the "-" separator needed for webdataset paths.
        k = k.replace("-", "/")
        return k

    image_snippets = {}
    mm_sample = {}
    for k, v in sample.items():
        # Compose images to image snippet
        if k.endswith(".jpg"):
            img_key = to_mm_key(k, "_")
            if img_key not in image_snippets:
                image_snippets[img_key] = [v]
            else:
                image_snippets[img_key].append(v)

        # np.float32 tensors
        elif k.endswith(".pyd"):
            k = to_mm_key(k, ".")
            if k in [
                ARIA_POSE_T_SNIPPET_RIG,
                ARIA_POSE_T_WORLD_RIG,
                ARIA_SNIPPET_T_WORLD_SNIPPET,
                ARIA_IMG_T_SNIPPET_RIG[0],
                ARIA_IMG_T_SNIPPET_RIG[1],
                ARIA_IMG_T_SNIPPET_RIG[2],
            ]:
                mm_sample[k] = PoseTW.from_matrix3x4(v.float())
            elif k in ARIA_CALIB:
                assert (
                    v.shape[-1] == DEFAULT_CAM_DATA_SIZE
                ), "only allow Fisheye624 cameras"
                mm_sample[k] = CameraTW(v)
            elif k == ARIA_OBB_PADDED:
                mm_sample[ARIA_OBB_PADDED] = ObbTW(v)
            elif k == ARIA_POINTS_WORLD:
                # load as float32
                mm_sample[ARIA_POINTS_WORLD] = v.float()
            elif isinstance(v, dict):
                # store dicts as (key, datum) lists in order to be able to collate them
                mm_sample[k] = [(kv, vv) for kv, vv in v.items()]
            else:
                mm_sample[k] = v

        # str
        elif k.endswith(".txt"):
            k = to_mm_key(k, ".")
            mm_sample[k] = v

        # int
        elif k.endswith(".cls"):
            k = to_mm_key(k, ".")
            mm_sample[k] = v

        else:
            pass  # silently ignore data field not used for training

    # images to image snippets
    for k, v in image_snippets.items():
        mm_sample[k] = np.transpose(np.stack(v, axis=0), (0, 3, 1, 2))
        # convert to one-channel for SLAM images
        if k == ARIA_IMG[1] or k == ARIA_IMG[2]:
            mm_sample[k] = mm_sample[k][:, :1, :, :]
        mm_sample[k] = torch.from_numpy(mm_sample[k])

    for key in mm_sample:
        if "time_s" in key:
            if isinstance(mm_sample[key], np.ndarray):
                mm_sample[key] = torch.from_numpy(mm_sample[key])
            assert mm_sample[key].dtype == torch.float32
            mm_sample[key] = mm_sample[key]
        if "time_ns" in key:
            if isinstance(mm_sample[key], np.ndarray):
                mm_sample[key] = torch.from_numpy(mm_sample[key])
            assert mm_sample[key].dtype == torch.int64
            mm_sample[key] = mm_sample[key]
    return mm_sample


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


def get_tar_sample_num(tar_file):
    sn = set()
    with tarfile.TarFile(tar_file, "r") as tar:
        for member in tar.getmembers():
            sn.add(member.name.split(".")[0])
    return len(sn)


class WdsStreamDataset:
    """Sample 2s/1s WDS dataset to specified snippet length and stride"""

    def __init__(
        self,
        data_path,
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
        self.samples_per_tar = get_tar_sample_num(tar_list[0])
        self.num_tars = len(tar_list)

        self.dataset = wds.DataPipeline(
            wds.SimpleShardList(tar_list),
            wds.tarfile_to_samples(),
            wds.decode("rgb"),
            wds.map(convert_to_aria_multimodal_dataset),
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
