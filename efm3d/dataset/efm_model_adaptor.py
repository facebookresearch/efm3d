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

import csv
import logging
from functools import partial
from typing import Callable, Dict, List, Optional

import torch

import webdataset as wds
from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
    process_wds_sample,
    select_and_remap_dict_keys,
)
from atek.util.tensor_utils import fill_or_trim_tensor
from efm3d.aria import CameraTW, ObbTW, PoseTW, TensorWrapper
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_CALIB_SNIPPET_TIME_S,
    ARIA_OBB_PADDED,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POSE_SNIPPET_TIME_S,
    ARIA_POSE_T_SNIPPET_RIG,
    ARIA_SNIPPET_LENGTH_S,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
    ARIA_SNIPPET_TIME_NS,
)
from efm3d.aria.obb import transform_obbs
from efm3d.aria.tensor_wrapper import smart_stack

from webdataset.filters import pipelinefilter

logger = logging.getLogger(__name__)


def get_local_pose_helper(snippet_origin_time_s, batch, local_coordinate):
    """
    get the local coordinate system of the snippet as the pose at the
    snippet_origin_time_s under the specified coordinate system conventions (rig, or cam_rgb)
    """
    assert (
        ARIA_POSE_T_SNIPPET_RIG in batch.keys()
        and ARIA_POSE_SNIPPET_TIME_S in batch.keys()
        and ARIA_SNIPPET_T_WORLD_SNIPPET in batch.keys()
    ), f"keys not in batch keys {batch.keys()}"

    T_world_snippet = batch[ARIA_SNIPPET_T_WORLD_SNIPPET]
    Ts_world_rig = T_world_snippet @ batch[ARIA_POSE_T_SNIPPET_RIG]
    time_s = batch[ARIA_POSE_SNIPPET_TIME_S]
    assert Ts_world_rig.dim() in [2, 3], f"{Ts_world_rig.shape} should be (B)xTx12"

    if local_coordinate == "rig":
        T_world_local = get_snippet_cosy_from_rig(
            Ts_world_rig=Ts_world_rig,
            time=time_s,
            snippet_origin_time=snippet_origin_time_s,
        )
    elif local_coordinate == "cam_rgb":
        T_world_local = get_snippet_cosy_from_cam_rgb(
            Ts_world_rig=Ts_world_rig,
            time=time_s,
            snippet_origin_time=snippet_origin_time_s,
            cam_rgb=batch[ARIA_CALIB[0]],
            cam_rgb_time_s=batch[ARIA_CALIB_SNIPPET_TIME_S[0]],
        )
    else:
        raise NotImplementedError(
            f"{local_coordinate} is not a valid coordinate option"
        )

    return T_world_local


def run_local_cosy(
    batch,
    origin_ratio=0.5,
    local_coordinate="cam_rgb",
    align_to_gravity=False,
    snippet_origin_time_s=None,
):
    new_batch = {}

    if snippet_origin_time_s is None:
        assert ARIA_SNIPPET_LENGTH_S in batch.keys()
        # get new snippet time origin
        snippet_length_s = batch[ARIA_SNIPPET_LENGTH_S]
        snippet_origin_time_s = snippet_length_s * origin_ratio

    # New origin time in ns.
    snippet_origin_time_ns = (snippet_origin_time_s * 1e9).long()

    # modify all time stamps to account for snippet origin change
    new_batch[ARIA_SNIPPET_TIME_NS] = (
        batch[ARIA_SNIPPET_TIME_NS] + snippet_origin_time_ns
    )

    # modify all snippet_time_s timestamps to account for snippet origin change
    keys_time_s = [key for key in batch.keys() if key.endswith("/snippet_time_s")]
    for key in keys_time_s:
        new_batch[key] = batch[key] - snippet_origin_time_s

    # get new snippet pose origin
    if (
        ARIA_POSE_T_SNIPPET_RIG in batch
        and ARIA_POSE_SNIPPET_TIME_S in batch
        and ARIA_SNIPPET_TIME_NS in batch
        and ARIA_SNIPPET_T_WORLD_SNIPPET in batch
    ):

        T_world_snippet = get_local_pose_helper(
            snippet_origin_time_s,
            batch,
            local_coordinate,
        )

        # apply change of coordinates to snippet coordinate system
        T_snippet_new_old = (
            T_world_snippet.inverse() @ batch[ARIA_SNIPPET_T_WORLD_SNIPPET]
        )
        new_batch[ARIA_SNIPPET_T_WORLD_SNIPPET] = T_world_snippet
        # apply the coordinate change to t_snippet_rigs
        keys_t_snippet_rig = [
            key for key in batch.keys() if key.endswith("t_snippet_rig")
        ]
        for key in keys_t_snippet_rig:
            new_batch[key] = T_snippet_new_old @ batch[key]

        # transform obbs into the new snippet coordinate system as well
        if ARIA_OBB_PADDED in batch.keys():
            new_batch[ARIA_OBB_PADDED] = transform_obbs(
                batch[ARIA_OBB_PADDED], T_snippet_new_old
            )

    return new_batch


def get_snippet_cosy_from_rig(
    snippet_origin_time: torch.Tensor,
    Ts_world_rig: PoseTW,
    time: torch.Tensor,
):
    """
    simply interpolate the T_world_rig using the given time at the snippet_origin_time
    to get T_world_rig_origin
    """
    T_world_rig_origin, good = Ts_world_rig.interpolate(time, snippet_origin_time)
    T = T_world_rig_origin.shape[-1]
    if T > 1 and not good.all():
        logger.warn(
            f"WARNING some interpolated poses were not good: {good} time_s {time} snippet_time {snippet_origin_time}"
        )
    return T_world_rig_origin


def get_snippet_cosy_from_cam_rgb(
    snippet_origin_time: torch.Tensor,
    Ts_world_rig: PoseTW,
    time: torch.Tensor,
    cam_rgb: torch.Tensor,
    cam_rgb_time_s: torch.Tensor,
):
    """
    interpolate T_world_rig and T_camera_rig using the given time_s at the snippet_origin_time
    and then compose the interpolated centers to get T_world_camera_origin
    """
    # interpolate T_camera_rig
    Ts_camera_rig = cam_rgb.T_camera_rig
    T_camera_rig_origin, good = Ts_camera_rig.interpolate(
        cam_rgb_time_s, snippet_origin_time
    )

    T = Ts_camera_rig.shape[-1]
    if T > 1 and not good.all():
        logger.warn("WARNING: some interpolated camera extrinsics were not good:")
    logger.debug(
        f"Good: {good}\n time_s {cam_rgb_time_s}\n snip_center {snippet_origin_time}"
    )
    T_world_rig_origin = get_snippet_cosy_from_rig(
        Ts_world_rig=Ts_world_rig, time=time, snippet_origin_time=snippet_origin_time
    )
    return T_world_rig_origin @ T_camera_rig_origin.inverse()


class EfmModelAdaptor:
    ATEK_CAM_LABEL_TO_EFM_CAM_LABEL: Dict[str, str] = {
        "camera-rgb": "rgb",
        "camera-slam-left": "slaml",
        "camera-slam-right": "slamr",
    }
    EFM_CAM_LABELS = ["rgb", "slaml", "slamr"]

    EFM_GRAVITY_IN_WORLD = [0, 0, -9.81]

    def __init__(
        self,
        freq: int,
        snippet_length_s: float = 2.0,
        semidense_points_pad_to_num: int = 50000,
        atek_to_efm_taxonomy_mapping_file: Optional[str] = None,
    ):
        self.freq = torch.tensor([freq], dtype=torch.int32)

        # EFM samples have fields padded to a fixed shape.
        # Obtain the fixed shape dimentions
        self.fixed_num_frames = int(snippet_length_s * freq)
        self.fixed_semidense_num_points = semidense_points_pad_to_num

        # Load optional taxonomy mapping file
        if atek_to_efm_taxonomy_mapping_file is not None:
            self.atek_to_efm_category_mapping = self._load_taxonomy_mapping_file(
                atek_to_efm_taxonomy_mapping_file
            )
        else:
            self.atek_to_efm_category_mapping = None

    @staticmethod
    def get_dict_key_mapping_for_camera(atek_camera_label: str, efm_camera_label: str):
        return {
            f"mfcd#{atek_camera_label}+images": f"{efm_camera_label}/img",
            f"mfcd#{atek_camera_label}+projection_params": f"{efm_camera_label}/calib/projection_params",
            f"mfcd#{atek_camera_label}+frame_ids": f"{efm_camera_label}/frame_id_in_sequence",
            f"mfcd#{atek_camera_label}+capture_timestamps_ns": f"{efm_camera_label}/img/time_ns",
            f"mfcd#{atek_camera_label}+camera_model_name": f"{efm_camera_label}/calib/camera_model_name",
            f"mfcd#{atek_camera_label}+camera_valid_radius": f"{efm_camera_label}/calib/valid_radius",
            f"mfcd#{atek_camera_label}+exposure_durations_s": f"{efm_camera_label}/calib/exposure",
            f"mfcd#{atek_camera_label}+gains": f"{efm_camera_label}/calib/gain",
            f"mfcd#{atek_camera_label}+t_device_camera": f"{efm_camera_label}/calib/t_device_camera",
        }

    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            # mps data mappings
            "mtd#ts_world_device": "pose/t_world_rig",
            "mtd#capture_timestamps_ns": "pose/time_ns",
            "mtd#gravity_in_world": "pose/gravity_in_world",
            "msdpd#points_world": "points/p3s_world",
            "msdpd#points_inv_dist_std": "points/inv_dist_std",
            "msdpd#points_dist_std": "points/dist_std",
            "msdpd#capture_timestamps_ns": "points/time_ns",
            "msdpd#points_volumn_min": ARIA_POINTS_VOL_MIN,
            "msdpd#points_volumn_max": ARIA_POINTS_VOL_MAX,
            "msdpd#points": "points/time_ns",
            "mfcd#camera-rgb-depth+images": "rgb/distance_m",
            # gt mappings
            "gt_data": "gt_data",
        }
        # camera data related mappings
        for (
            atek_cam_label,
            efm_cam_label,
        ) in EfmModelAdaptor.ATEK_CAM_LABEL_TO_EFM_CAM_LABEL.items():
            dict_key_mapping.update(
                EfmModelAdaptor.get_dict_key_mapping_for_camera(
                    atek_camera_label=atek_cam_label, efm_camera_label=efm_cam_label
                )
            )

        return dict_key_mapping

    def _get_pose_to_align_gravity(self, sample_dict: Dict) -> Optional[PoseTW]:
        """
        A helper function to return a T_newWorld_oldWorld transformation to align world gravity to the EFM convention.
        This pose needs to be later applied to all poses that include world.
        """
        efm_gravity_in_world = torch.tensor(
            self.EFM_GRAVITY_IN_WORLD, dtype=torch.float32
        )
        current_gravity_in_world = sample_dict["pose/gravity_in_world"]
        if torch.allclose(efm_gravity_in_world, current_gravity_in_world, atol=1e-3):
            # print("gravity convention is already aligned.")
            return None
        else:
            if torch.allclose(current_gravity_in_world, torch.tensor([0, -9.81, 0])):
                return PoseTW.from_Rt(
                    torch.tensor(
                        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32
                    ),
                    torch.tensor([0, 0, 0], dtype=torch.float32),
                )
            else:
                raise ValueError(
                    f"unsupported gravity direction to align: {current_gravity_in_world}"
                )

    def _load_taxonomy_mapping_file(self, filename: str) -> Dict:
        """
        Load a taxonomy mapping csv file in the format of:
        ATEK_category_name, efm_category_name, efm_category_id

        returns a dict of {atek_cat_name -> (efm_cat_name, efm_cat_id)}
        """
        atek_to_efm_category_mapping = {}
        with open(filename, "r") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)

            for row in csv_reader:
                atek_name = row[0]
                value = (row[1], int(row[2]))  # Convert category id to an integer
                atek_to_efm_category_mapping[atek_name] = value

        return atek_to_efm_category_mapping

    def _fill_dict_with_freq(self, sample_dict: Dict) -> Dict:
        fields_to_fill = [
            "pose/hz",
            "points/hz",
            "rgb/img/hz",
            "slaml/img/hz",
            "slamr/img/hz",
        ]

        # only fill obb frequency if GT exists
        if self.gt_exists_flag:
            fields_to_fill += ["obbs/hz"]

        for field in fields_to_fill:
            sample_dict[field] = self.freq
        return sample_dict

    def _convert_to_batched_camera_tw(
        self, sample_dict: Dict, cam_label: str
    ) -> CameraTW:
        """
        A helper function to convert ATEK camera calibration to EFM camera tensor wrapper, where calibration params are replicated x `num_frames`.
        """
        # calibrations are replicated by `fixed_num_frames`
        batched_size = torch.Size((self.fixed_num_frames, 1))
        camera_tw = CameraTW.from_surreal(
            width=torch.full(
                size=batched_size, fill_value=sample_dict[f"{cam_label}/img"].shape[3]
            ),
            height=torch.full(
                size=batched_size, fill_value=sample_dict[f"{cam_label}/img"].shape[2]
            ),
            type_str=sample_dict[f"{cam_label}/calib/camera_model_name"],
            params=sample_dict[f"{cam_label}/calib/projection_params"].unsqueeze(
                0
            ),  # make tensor shape [1, 15], so that it can be expanded to [num_frames, 15]
            gain=fill_or_trim_tensor(
                tensor=sample_dict[f"{cam_label}/calib/gain"],
                dim_size=self.fixed_num_frames,
                dim=0,
            ),
            exposure_s=fill_or_trim_tensor(
                tensor=sample_dict[f"{cam_label}/calib/exposure"],
                dim_size=self.fixed_num_frames,
                dim=0,
            ),
            valid_radius=sample_dict[f"{cam_label}/calib/valid_radius"],
            T_camera_rig=PoseTW.from_matrix3x4(
                sample_dict[f"{cam_label}/calib/t_device_camera"]
            ).inverse(),
        )

        return camera_tw.float()

    def _update_efm_obb_gt(self, atek_gt_dict: Dict) -> Dict:
        """
        Helper function to convert ATEK obb gt to EFM obb gt.
        """
        efm_sub_dict = {}

        # loop over all timestamps
        timestamp_list = []
        efm_obb_all_timestamps = []
        semantic_id_to_name = {}
        for timestamp, obb3_dict in atek_gt_dict["efm_gt"].items():
            timestamp_list.append(int(timestamp))

            # Create a hash map to query which instance is visible in which camera.
            # The resulting map will look like: {
            #    "instance_1": {
            #    "cam_0": index_in_cam0,
            #    "cam_1": index_in_cam1,
            #       ...
            # },
            #    "instance_2":  {
            #    ...
            # }
            #    ...
            instance_visible_map = {}
            for camera_label, per_cam_dict in obb3_dict.items():
                for i in range(len(per_cam_dict["instance_ids"])):
                    instance_id = per_cam_dict["instance_ids"][i].item()
                    if instance_id not in instance_visible_map:
                        instance_visible_map[instance_id] = {}
                    instance_visible_map[instance_id][camera_label] = i

            efm_obb_tw_list = []
            # Loop over all instances from all cameras
            for instance_id, instance_mapping_info in instance_visible_map.items():
                # Create a ObbTW for this instance
                # get obb3 info from any visible camera
                cam_label_0, cam_index_0 = next(iter(instance_mapping_info.items()))
                atek_single_bb3_dict = obb3_dict[cam_label_0]
                bb3_dim = atek_single_bb3_dict["object_dimensions"][
                    cam_index_0
                ]  # tensor [3]
                object_half_sizes = bb3_dim / 2.0
                bb3_object = torch.tensor(
                    [
                        -object_half_sizes[0],
                        object_half_sizes[0],
                        -object_half_sizes[1],
                        object_half_sizes[1],
                        -object_half_sizes[2],
                        object_half_sizes[2],
                    ],
                    dtype=torch.float32,
                )
                T_world_object = PoseTW.from_matrix3x4(
                    atek_single_bb3_dict["ts_world_object"][cam_index_0]
                )
                inst_id = atek_single_bb3_dict["instance_ids"][cam_index_0]

                # perform taxonomy remapping if needed, but skip "other"
                sem_id = atek_single_bb3_dict["category_ids"][cam_index_0].item()
                category_name = atek_single_bb3_dict["category_names"][cam_index_0]
                if category_name == "other":
                    continue
                if self.atek_to_efm_category_mapping is not None:
                    category_name, sem_id = self.atek_to_efm_category_mapping[
                        category_name
                    ]

                # Also keep track of a sem_id_to_name mapping
                if sem_id not in semantic_id_to_name:
                    semantic_id_to_name[sem_id] = category_name

                bb2_rgb = -1 * torch.ones(4)
                bb2_slaml = -1 * torch.ones(4)
                bb2_slamr = -1 * torch.ones(4)
                # Commenting off because obb2 are not needed
                """
                if "camera-rgb" in instance_mapping_info:
                    cam_label = "camera-rgb"
                    cam_index = instance_mapping_info[cam_label]
                    bb2_rgb = atek_gt_dict["obb2"][cam_label]["bbox_ranges"][cam_index]
                
                if "camera-slam-left" in instance_mapping_info:
                    cam_label = "camera-slam-left"
                    cam_index = instance_mapping_info[cam_label]
                    bb2_slaml = atek_gt_dict["obb2"][cam_label]["bbox_ranges"][cam_index]
                
                if "camera-slam-right" in instance_mapping_info:
                    cam_label = "camera-slam-right"
                    cam_index = instance_mapping_info[cam_label]
                    bb2_slamr = atek_gt_dict["obb2"][cam_label]["bbox_ranges"][cam_index]
                """

                # Fill in padded obbs in EFM format
                efm_obb_tw_list.append(
                    ObbTW.from_lmc(
                        bb3_object=bb3_object,
                        bb2_rgb=bb2_rgb,
                        bb2_slaml=bb2_slaml,
                        bb2_slamr=bb2_slamr,
                        T_world_object=T_world_object,
                        sem_id=torch.tensor([sem_id], dtype=torch.int64),
                        inst_id=torch.tensor([inst_id], dtype=torch.int64),
                    )
                )
            # end for instance_id

            if len(efm_obb_tw_list) == 0:
                efm_obb_tw = ObbTW()
            else:
                efm_obb_tw = ObbTW(smart_stack(efm_obb_tw_list, dim=0))
            efm_obb_tw = efm_obb_tw.add_padding(max_elts=128)
            efm_obb_all_timestamps.append(efm_obb_tw)

        efm_sub_dict["obbs/padded_snippet"] = ObbTW(
            smart_stack(efm_obb_all_timestamps, dim=0)
        )
        efm_sub_dict["obbs/time_ns"] = torch.tensor(timestamp_list, dtype=torch.int64)
        efm_sub_dict["obbs/sem_id_to_name"] = semantic_id_to_name
        return efm_sub_dict

    def _pad_semidense_data(self, sample_dict: Dict) -> Dict:
        """
        A helper function to pad semidense data from List[Tensor, (K, 3 or 1)] to fixed shape of [numFrames, num_semidense_points, 3 or 1]
        """
        result_dict = {}

        fields_to_pad = ["points/p3s_world", "points/dist_std", "points/inv_dist_std"]
        for field in fields_to_pad:
            tensor_list = sample_dict[field]
            for i in range(len(tensor_list)):
                # First, pad each tensor in the list to fixed num points
                tensor_list[i] = fill_or_trim_tensor(
                    tensor=tensor_list[i],
                    dim_size=self.fixed_semidense_num_points,
                    dim=0,
                    fill_value=float("nan"),
                )

            # then stack
            stacked_tensor = torch.stack(tensor_list, dim=0)

            # then pad over frames
            result_dict[field] = fill_or_trim_tensor(
                tensor=stacked_tensor, dim_size=self.fixed_num_frames, dim=0
            )

        return result_dict

    def _pad_over_frames(self, sample_dict: Dict, fields_to_pad: List[str]) -> Dict:
        """
        A helper function to pad data over frames, by repeating the last element over frames.
        """
        result_dict = {}
        for field in fields_to_pad:
            result_dict[field] = fill_or_trim_tensor(
                tensor=sample_dict[field],
                dim_size=self.fixed_num_frames,
                dim=0,
            )
        return result_dict

    def _split_pose_over_snippet(self, sample_dict: Dict) -> Dict:
        """
        A helper function to split T_world_rig into T_world_snippet and T_snippet_rig.
        In the meantime, Align gravity to [0, 0, -9.81]
        """
        result_dict = {}

        # check if world coordinates needs to be re-aligned
        maybe_T_newWorld_oldWorld = self._get_pose_to_align_gravity(sample_dict)
        # maybe_T_newWorld_oldWorld = None

        Ts_world_rig = PoseTW.from_matrix3x4(sample_dict["pose/t_world_rig"])
        if maybe_T_newWorld_oldWorld:
            Ts_world_rig = maybe_T_newWorld_oldWorld @ Ts_world_rig

        result_dict["pose/t_world_rig"] = Ts_world_rig

        T_world_snippet = Ts_world_rig.clone()[0]
        T_world_snippet = T_world_snippet.unsqueeze(0)
        result_dict["snippet/t_world_snippet"] = T_world_snippet.clone()
        result_dict["pose/t_snippet_rig"] = Ts_world_rig[0].inverse() @ Ts_world_rig

        for camera_label in EfmModelAdaptor.EFM_CAM_LABELS:
            result_dict[f"{camera_label}/t_snippet_rig"] = result_dict[
                "pose/t_snippet_rig"
            ].clone()

        # Transform obbs poses, from old_world -> new_world -> snippet
        if ARIA_OBB_PADDED in sample_dict:
            if maybe_T_newWorld_oldWorld:
                T_snippet_world = T_world_snippet.inverse() @ maybe_T_newWorld_oldWorld
            else:
                T_snippet_world = T_world_snippet.inverse()

            result_dict[ARIA_OBB_PADDED] = transform_obbs(
                sample_dict[ARIA_OBB_PADDED], T_snippet_world
            )

        # Also transform semidense points
        if maybe_T_newWorld_oldWorld:
            result_dict["points/p3s_world"] = (
                maybe_T_newWorld_oldWorld * sample_dict["points/p3s_world"]
            )

        return result_dict

    def _split_timestamps_over_snippet(self, sample_dict: Dict) -> Dict:
        """
        A helper function to split capture_timestamps_ns into snippet/time_ns and */snippet_time_s
        """
        dict_keys_to_split_timestamps = [
            "pose/",
            "points/",
        ] + [f"{label}/img/" for label in EfmModelAdaptor.EFM_CAM_LABELS]

        # Also split obbs timestamps, if gt exists
        if self.gt_exists_flag:
            dict_keys_to_split_timestamps += ["obbs/"]

        result_dict = {}

        result_dict["snippet/time_ns"] = sample_dict["rgb/img/time_ns"][0].unsqueeze(0)
        for key in dict_keys_to_split_timestamps:
            result_dict[key + "snippet_time_s"] = (
                sample_dict[key + "time_ns"] - result_dict["snippet/time_ns"]
            ) / torch.tensor(1e9, dtype=torch.float32)

        return result_dict

    def atek_to_efm(self, data, train=False):
        """
        A helper data transform function to convert a ATEK webdataset data sample built by EfmSampleBuilder to EFM unbatched
        samples. Yield one unbatched sample a time to use the collation and batching mechanism in
        the webdataset properly.
        """
        for atek_wds_sample in data:
            efm_sample = atek_wds_sample

            # Check if GT exists in the sample. If not, all obb related operations will be skipped
            self.gt_exists_flag = (
                "gt_data" in atek_wds_sample and len(atek_wds_sample["gt_data"]) > 0
            )

            # Fill frequenze data from conf
            efm_sample = self._fill_dict_with_freq(efm_sample)

            # Pad semidense data, which requires 2-dim padding
            padded_dict = self._pad_semidense_data(efm_sample)
            efm_sample.update(padded_dict)

            # Convert ATEK calibration to EFM camera calibration, where calibration params are replicated x `num_frames`,
            # except gains and exposure_s which is per-frame.
            for cam_label in EfmModelAdaptor.EFM_CAM_LABELS:
                efm_sample[f"{cam_label}/calib"] = self._convert_to_batched_camera_tw(
                    efm_sample, cam_label
                )

            # Convert ATEK GT to EFM GT
            if self.gt_exists_flag:
                result_dict = self._update_efm_obb_gt(atek_wds_sample["gt_data"])
                efm_sample.update(result_dict)

            # split T_world_rig into T_world_snippet and T_snippet_rig
            result_dict = self._split_pose_over_snippet(efm_sample)
            efm_sample.update(result_dict)

            # split capture_timestamps_ns into snippet/time_ns and */snippet_time_s
            result_dict = self._split_timestamps_over_snippet(efm_sample)
            efm_sample.update(result_dict)

            # Pad some data over frames by repeating last element
            fields_to_pad = []
            fields_to_skip_padding = ["snippet/t_world_snippet"]
            for key, value in efm_sample.items():
                if key in fields_to_skip_padding:
                    continue
                if isinstance(value, torch.Tensor) or isinstance(value, TensorWrapper):
                    if value.shape[0] < self.fixed_num_frames:
                        # pad timestamp tensors, but not other 1-dim tensors
                        if (
                            key.endswith("time_ns")
                            or key.endswith("time_s")
                            or value.ndim > 1
                        ):
                            fields_to_pad.append(key)
            result_dict = self._pad_over_frames(efm_sample, fields_to_pad=fields_to_pad)
            efm_sample.update(result_dict)

            # Duplicate `camera/img/time` to `camera/calib/time`
            for camera_name in EfmModelAdaptor.EFM_CAM_LABELS:
                efm_sample[f"{camera_name}/calib/time_ns"] = efm_sample[
                    f"{camera_name}/img/time_ns"
                ]
                efm_sample[f"{camera_name}/calib/snippet_time_s"] = efm_sample[
                    f"{camera_name}/img/snippet_time_s"
                ]

            # Convert data types from int to float32
            fields_to_conv2float32 = [
                f"{label}/img" for label in EfmModelAdaptor.EFM_CAM_LABELS
            ] + [
                f"{label}/frame_id_in_sequence"
                for label in EfmModelAdaptor.EFM_CAM_LABELS
            ]
            for field in fields_to_conv2float32:
                efm_sample[field] = efm_sample[field].to(torch.float32)
                if field.endswith("img"):
                    # normalize
                    efm_sample[field] = efm_sample[field] / 255.0
                if field == "rgb/img":
                    # swap channels from [RGB] -> [BGR]
                    # efm_sample[field] = efm_sample[field][:, [2, 1, 0], :, :]
                    pass

            # Run local cosy to shift the origin
            # For testing only: patch snippet lenths
            efm_sample[ARIA_SNIPPET_LENGTH_S] = torch.tensor([2.0], dtype=torch.float32)
            result = run_local_cosy(batch=efm_sample, origin_ratio=0.5)
            efm_sample.update(result)

            # delete useless data
            if train:
                # keep only tensors
                remove_keys = []
                for key in efm_sample:
                    if not isinstance(efm_sample[key], (torch.Tensor, TensorWrapper)):
                        remove_keys.append(key)
                for k in remove_keys:
                    efm_sample.pop(k)

            yield efm_sample


def load_atek_wds_dataset_as_efm(
    urls: List,
    freq=10,
    snippet_length_s=2.0,
    semidense_points_pad_to_num=50000,
    atek_to_efm_taxonomy_mapping_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    collation_fn: Optional[Callable] = None,
):
    efm_model_adaptor = EfmModelAdaptor(
        freq=freq,
        snippet_length_s=snippet_length_s,
        semidense_points_pad_to_num=semidense_points_pad_to_num,
        atek_to_efm_taxonomy_mapping_file=atek_to_efm_taxonomy_mapping_file,
    )

    return load_atek_wds_dataset(
        urls,
        dict_key_mapping=EfmModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(efm_model_adaptor.atek_to_efm)(
            train=collation_fn is not None
        ),
        batch_size=batch_size,
        collation_fn=collation_fn,
    )


def load_atek_wds_dataset_as_efm_train(
    urls: List,
    freq=10,
    snippet_length_s=2.0,
    semidense_points_pad_to_num=50000,
    atek_to_efm_taxonomy_mapping_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    collation_fn: Optional[Callable] = None,
):

    efm_model_adaptor = EfmModelAdaptor(
        freq=freq,
        snippet_length_s=snippet_length_s,
        semidense_points_pad_to_num=semidense_points_pad_to_num,
        atek_to_efm_taxonomy_mapping_file=atek_to_efm_taxonomy_mapping_file,
    )

    wds_dataset = (
        wds.WebDataset(urls, nodesplitter=None, resampled=True, repeat=True)
        .decode(wds.imagehandler("torchrgb8"))
        .map(process_wds_sample)
    )
    wds_dataset = wds_dataset.map(
        partial(
            select_and_remap_dict_keys,
            key_mapping=EfmModelAdaptor.get_dict_key_mapping_all(),
        )
    )
    wds_dataset = wds_dataset.compose(
        pipelinefilter(efm_model_adaptor.atek_to_efm)(train=collation_fn is not None)
    )
    wds_dataset = wds_dataset.batched(batch_size, collation_fn=collation_fn)

    return wds_dataset
