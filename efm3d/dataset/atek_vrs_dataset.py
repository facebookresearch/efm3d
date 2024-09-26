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

# pyre-strict

import logging
import os

from typing import Dict, List, Optional

from atek.data_loaders.atek_wds_dataloader import select_and_remap_dict_keys
from atek.data_preprocess.atek_data_sample import AtekDataSample
from atek.data_preprocess.sample_builders.atek_data_paths_provider import (
    AtekDataPathsProvider,
)

from atek.data_preprocess.sample_builders.efm_sample_builder import EfmSampleBuilder
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)
from efm3d.dataset.efm_model_adaptor import EfmModelAdaptor
from omegaconf.omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtekRawDataloaderAsEfm:
    def __init__(
        self,
        vrs_file: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
        conf: DictConfig,
        freq_hz: int,
        snippet_length_s: float,
        semidense_points_pad_to_num: int = 50000,
        max_snippets=9999,
    ) -> None:
        self.max_snippets = max_snippets

        # initialize the sample builder
        self.sample_builder = EfmSampleBuilder(
            conf=conf.processors,
            vrs_file=vrs_file,
            mps_files=mps_files,
            gt_files=gt_files,
            depth_vrs_file="",
            sequence_name=os.path.basename(vrs_file),
        )

        self.subsampler = CameraTemporalSubsampler(
            vrs_file=vrs_file,
            conf=conf.camera_temporal_subsampler,
        )

        # Create a EFM model adaptor
        self.model_adaptor = EfmModelAdaptor(
            freq=freq_hz,
            snippet_length_s=snippet_length_s,
            semidense_points_pad_to_num=semidense_points_pad_to_num,
            atek_to_efm_taxonomy_mapping_file=f"{os.path.dirname(__file__)}/../config/taxonomy/atek_to_efm.csv",
        )

    def __len__(self):
        return min(self.subsampler.get_total_num_samples(), self.max_snippets)

    def get_timestamps_by_sample_index(self, index: int) -> List[int]:
        return self.subsampler.get_timestamps_by_sample_index(index)

    def get_atek_sample_at_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[AtekDataSample]:
        return self.sample_builder.get_sample_by_timestamps_ns(timestamps_ns)

    def get_model_specific_sample_at_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[Dict]:
        atek_sample = self.get_atek_sample_at_timestamps_ns(timestamps_ns)
        if atek_sample is None:
            logger.warning(
                f"Cannot retrieve valid atek sample at timestamp {timestamps_ns}"
            )
            return None

        # Flatten to dict
        atek_sample_dict = atek_sample.to_flatten_dict()

        # key remapping
        remapped_data_dict = select_and_remap_dict_keys(
            sample_dict=atek_sample_dict,
            key_mapping=self.model_adaptor.get_dict_key_mapping_all(),
        )

        # transform
        model_specific_sample_gen = self.model_adaptor.atek_to_efm([remapped_data_dict])

        # Obtain a dict from a generator object
        model_specific_sample = next(model_specific_sample_gen)

        return model_specific_sample

    def __getitem__(self, index):
        if index >= self.max_snippets:
            raise StopIteration

        timestamps = self.get_timestamps_by_sample_index(index)
        maybe_sample = self.get_model_specific_sample_at_timestamps_ns(timestamps)

        return maybe_sample


def create_atek_raw_data_loader_from_vrs_path(
    vrs_path: str,
    freq_hz: int,
    snippet_length_s,
    stride_length_s,
    skip_begin_seconds: float = 0.0,
    skip_end_seconds: float = 0.0,
    semidense_points_pad_to_num=50000,
    max_snippets=9999,
):
    vrs_dir = os.path.dirname(vrs_path)
    data_path_provider = AtekDataPathsProvider(data_root_path=vrs_dir)
    atek_data_paths = data_path_provider.get_data_paths()

    conf = OmegaConf.load("efm3d/config/efm_preprocessing_conf.yaml")

    # Update snippet / stride length
    conf.camera_temporal_subsampler.main_camera_target_freq_hz = float(freq_hz)
    conf.camera_temporal_subsampler.sample_length_in_num_frames = int(
        freq_hz * snippet_length_s
    )
    conf.camera_temporal_subsampler.stride_length_in_num_frames = int(
        freq_hz * stride_length_s
    )
    conf.camera_temporal_subsampler.update(
        {
            "skip_begin_seconds": skip_begin_seconds,
            "skip_end_seconds": skip_end_seconds,
        }
    )

    data_loader = AtekRawDataloaderAsEfm(
        vrs_file=atek_data_paths["video_vrs_file"],
        mps_files={
            "mps_closedloop_traj_file": atek_data_paths["mps_closedloop_traj_file"],
            "mps_semidense_points_file": atek_data_paths["mps_semidense_points_file"],
            "mps_semidense_observations_file": atek_data_paths[
                "mps_semidense_observations_file"
            ],
        },
        gt_files={
            "obb3_file": atek_data_paths["gt_obb3_file"],
            "obb3_traj_file": atek_data_paths["gt_obb3_traj_file"],
            "obb2_file": atek_data_paths["gt_obb2_file"],
            "instance_json_file": atek_data_paths["gt_instance_json_file"],
        },
        conf=conf,
        freq_hz=freq_hz,
        snippet_length_s=snippet_length_s,
        semidense_points_pad_to_num=semidense_points_pad_to_num,
        max_snippets=max_snippets,
    )

    return data_loader
