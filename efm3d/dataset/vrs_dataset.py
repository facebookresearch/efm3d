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
import os
import random

from typing import Callable, List, Optional, Union

import numpy as np
import pyvrs
import torch
import torch.nn.functional as F

from efm3d.aria import CameraTW, ObbTW, PoseTW, smart_stack, transform_obbs
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_CALIB_SNIPPET_TIME_S,
    ARIA_CALIB_TIME_NS,
    ARIA_CAM_INFO,
    ARIA_FRAME_ID,
    ARIA_IMG,
    ARIA_IMG_SNIPPET_TIME_S,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_IMG_TIME_NS,
    ARIA_OBB_BB2,
    ARIA_OBB_PADDED,
    ARIA_OBB_SEM_ID_TO_NAME,
    ARIA_OBB_SNIPPET_TIME_S,
    ARIA_OBB_TIME_NS,
    ARIA_POINTS_SNIPPET_TIME_S,
    ARIA_POINTS_TIME_NS,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_POINTS_WORLD,
    ARIA_POSE_SNIPPET_TIME_S,
    ARIA_POSE_T_SNIPPET_RIG,
    ARIA_POSE_T_WORLD_RIG,
    ARIA_POSE_TIME_NS,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.utils.file_utils import (
    exists_nonzero_path,
    get_timestamp_list_ns,
    load_factory_calib,
    load_global_points_csv,
    load_obbs_gt,
    load_semidense_observations,
    load_trajectory,
    load_trajectory_adt,
    load_trajectory_aeo,
    read_image_snippet_from_vrs,
    sample_from_range,
    sample_times,
)
from efm3d.utils.obb_io import get_instance_id_in_frameset, next_obb_observations
from efm3d.utils.rescale import rescale_obb_tw
from torch.utils.data import Dataset


# gravity direction in ADT conventions
GRAVITY_DIRECTION_ADT = np.array([0.0, -1.0, 0.0], np.float32)


def is_adt(vrs_path):
    # get folder name
    if vrs_path.endswith(".vrs"):
        vrs_path = os.path.split(vrs_path)[0]
    return os.path.exists(os.path.join(vrs_path, "aria_trajectory.csv"))


def is_aeo(vrs_path):
    return "aeo_" in vrs_path


def get_transform_to_vio_gravity_convention(gravity_direction: np.array):
    """
    Get transformation to map gravity_direction to (0,0,-1) as per our (and
    VIO/Temple) convention.
    """
    # gravity_direction = (d1, d2, d3) (0,0,-1)^T; d1, d2, d3 column vectors of rotation matrix R_gravity_vio
    # -d3 = gravity_direction
    d3 = -gravity_direction.copy()
    # now construct an orthonormal basis for the rotation matrix
    # d1 is a vector thats orthogonal to gravity_direction by construction
    d1 = np.array(
        [
            gravity_direction[2] - gravity_direction[1],
            gravity_direction[0],
            -gravity_direction[0],
        ]
    )
    # get d2 via orthogonal direction vector to d3 and d1
    d2 = np.cross(d3, d1)
    # get rotation matrix
    R_gravity_vio = np.concatenate(
        [d1[:, np.newaxis], d2[:, np.newaxis], d3[:, np.newaxis]], 1
    )
    assert (np.linalg.det(R_gravity_vio) - 1.0) < 1e-5
    assert (((R_gravity_vio @ R_gravity_vio.transpose()) - np.eye(3)) < 1e-5).all()
    R_gravity_vio = torch.from_numpy(R_gravity_vio)
    # normalize to unit length
    R_gravity_vio = F.normalize(R_gravity_vio, p=2, dim=-2)
    R_vio_gravity = R_gravity_vio.transpose(1, 0)
    T_vio_gravity = PoseTW.from_Rt(R_vio_gravity, torch.zeros(3))
    return T_vio_gravity


def compute_time_intersection(time_lists):
    min_time = -math.inf
    max_time = math.inf
    for ts in time_lists:
        ts = np.array(ts)
        min_time = max(min_time, ts.min())
        max_time = min(max_time, ts.max())

    # add an offset to the timestamp
    safety_margin = 3_000_000  # 3ms
    min_time = min_time - safety_margin
    max_time = max_time - safety_margin

    return min_time, max_time


def preprocess_inference(batch):
    # tensor wrapper
    for k in batch:
        if not isinstance(batch[k], (torch.Tensor, PoseTW, CameraTW, ObbTW)):
            continue

        if k in [
            ARIA_SNIPPET_T_WORLD_SNIPPET,
            ARIA_POSE_T_WORLD_RIG,
            ARIA_POSE_T_SNIPPET_RIG,
        ] + ARIA_IMG_T_SNIPPET_RIG and not isinstance(batch[k], PoseTW):
            batch[k] = PoseTW(batch[k])
        elif k in ARIA_CALIB and not isinstance(batch[k], CameraTW):
            batch[k] = CameraTW(batch[k])
        elif k == ARIA_OBB_PADDED and not isinstance(batch[k], ObbTW):
            batch[k] = ObbTW(batch[k])

    return batch


def preprocess(
    batch,
    device,
    subsample: int = 10,
    aug_funcs: Optional[Union[Callable, List[Callable]]] = None,
):
    # tensor wrapper
    for k in batch:
        if not isinstance(batch[k], (torch.Tensor, PoseTW, CameraTW, ObbTW)):
            continue

        if k in [
            ARIA_SNIPPET_T_WORLD_SNIPPET,
            ARIA_POSE_T_WORLD_RIG,
            ARIA_POSE_T_SNIPPET_RIG,
        ] + ARIA_IMG_T_SNIPPET_RIG and not isinstance(batch[k], PoseTW):
            batch[k] = PoseTW(batch[k])
        elif k in ARIA_CALIB and not isinstance(batch[k], CameraTW):
            batch[k] = CameraTW(batch[k])
        elif k == ARIA_OBB_PADDED and not isinstance(batch[k], ObbTW):
            batch[k] = ObbTW(batch[k])

    # time crop
    T = batch[ARIA_IMG[0]].shape[1]
    if subsample != T:
        s = random.randint(0, T - subsample - 1)
        for k in batch:
            if (
                isinstance(batch[k], (torch.Tensor, PoseTW, CameraTW, ObbTW))
                and batch[k].shape[1] == T
            ):
                batch[k] = batch[k][:, s : s + subsample, ...]

    # move to GPU
    for k in batch:
        if isinstance(batch[k], (torch.Tensor, PoseTW, CameraTW, ObbTW)):
            batch[k] = batch[k].to(device)

    # data augmentations
    if aug_funcs is not None:
        if isinstance(aug_funcs, Callable):
            aug_funcs = [aug_funcs]
        for aug in aug_funcs:
            batch = aug(batch)

    return batch


def tensor_unify(tensor, dim_size: int, dim: int = 0):
    """Fill or trim a torch or numpy tensor to the given `dim_size`, along the given dim

    Inputs:
        tensor (torch or np.array): input tensor
        dim_size (int): the size to fill or trim to (e.g. predefined batch size)
        dim (int): the dimension to fill or trim

    Returns:
        tensor2 (a torch or np.array): output tensor with the dim size = `dim_size`.
    """
    assert tensor.shape[dim] > 0, "Input tensor must have at least 1 element"

    if isinstance(tensor, list):
        tensor = np.array(tensor)
    if isinstance(tensor, np.ndarray):
        np_tensor = tensor
        tensor_bs = np_tensor.shape[dim]
        if tensor_bs > dim_size:
            tensor2 = np.take(np_tensor, indices=np.arange(dim_size), axis=dim)
        elif tensor_bs < dim_size:
            last = np.take(np_tensor, tensor_bs - 1, dim)
            fill = np.expand_dims(last, axis=dim)  # fill with last element
            fill = np.repeat(fill, dim_size - tensor_bs, axis=dim)
            tensor2 = np.concatenate([np_tensor, fill], axis=dim)
        else:
            tensor2 = tensor
    else:
        tensor_bs = tensor.shape[dim]
        if tensor_bs > dim_size:
            indices = torch.arange(dim_size)
            for i in range(tensor.ndim):
                if i != dim:
                    indices = indices.unsqueeze(i)
            tensor2 = torch.take_along_dim(tensor, indices, dim)
        elif tensor_bs < dim_size:
            shape = [1 for _ in range(tensor.ndim)]
            indices = torch.ones(shape).long()
            indices[0] = tensor_bs - 1
            last = torch.take_along_dim(tensor, indices, dim)
            fill_shape = shape
            fill_shape[dim] = dim_size - tensor_bs
            fill = last.repeat(fill_shape)
            tensor2 = torch.cat([tensor, fill], dim=dim)
        else:
            tensor2 = tensor
    return tensor2


def run_sensor_poses(batch, num_notified=-1, max_notified=10):
    if (
        ARIA_POSE_T_SNIPPET_RIG in batch.keys()
        and ARIA_POSE_SNIPPET_TIME_S in batch.keys()
    ):
        new_batch = {}
        Ts_snippet_rig = batch[ARIA_POSE_T_SNIPPET_RIG]
        ts = batch[ARIA_POSE_SNIPPET_TIME_S]
        assert Ts_snippet_rig.dim() in [
            2,
            3,
        ], f"need to be of shape (B) x T x 12 but are {Ts_snippet_rig.shape}"
        for i, img_time_key in enumerate(ARIA_IMG_SNIPPET_TIME_S):
            if (
                img_time_key in batch.keys()
                and ARIA_IMG_T_SNIPPET_RIG[i] not in batch.keys()
            ):
                ts_interp = batch[img_time_key]
                Ts_world_rig_i, good = Ts_snippet_rig.interpolate(ts, ts_interp)
                new_batch[ARIA_IMG_T_SNIPPET_RIG[i]] = Ts_world_rig_i
                if not good.all():
                    counts = good.sum(dim=-1).squeeze()
                    if num_notified > 0 and num_notified < max_notified:
                        print(
                            f"some interpolated poses were bad (fraction good per batch: {counts/good.shape[-1]}); likely because tried to interpolated past given input timed poses."
                        )
        return new_batch


class VrsSequenceDataset(Dataset):
    def __init__(
        self,
        vrs_path,
        frame_rate,
        sdi,
        snippet_length_s,
        stride_length_s,
        max_snippets=9999,
        preprocess=None,
    ):
        self.frame_rate = frame_rate
        self.vrs_path = vrs_path
        self.vrs_folder = os.path.split(vrs_path)[0]
        self.reader = pyvrs.SyncVRSReader(
            vrs_path, auto_read_configuration_records=True
        )
        self.max_snippets = max_snippets
        self.sdi = sdi
        self.preprocess = preprocess
        self.cam_calib = load_factory_calib(self.reader)
        self.is_adt = is_adt(vrs_path)
        self.is_aeo = is_aeo(vrs_path)
        self.max_objects_per_frameset = 128

        fps = self.cam_calib["fps"]
        self.fps = [fps["rgb"], fps["slaml"], fps["slamr"]]

        ts_lists = []
        # Add images
        for idx in range(3):
            img_ts_list = get_timestamp_list_ns(self.reader, ARIA_CAM_INFO["id"][idx])
            ts_lists.append(img_ts_list)

        # Add poses
        timed_Ts_world_rig = self.load_poses(self.vrs_folder, subsample=1)
        pose_times_ns = list(timed_Ts_world_rig.keys())
        pose_freq = int(1.0 / (1e-9 * (pose_times_ns[1] - pose_times_ns[0])))
        pose_subsample = int(pose_freq / frame_rate)
        pose_times_ns = pose_times_ns[::pose_subsample]

        self.T_world_rig_time_ns = pose_times_ns
        self.Ts_world_rig = torch.stack(
            [timed_Ts_world_rig[key] for key in pose_times_ns]
        )
        ts_lists.append(pose_times_ns)

        # Add obbs GT if available
        self.obs = None
        if not self.is_adt:
            self.obs = self.load_objects()
        if self.obs is not None:
            obb_freq = int(1.0 / (1e-9 * (self.obb_times[1] - self.obb_times[0])))
            obb_subsample = int(obb_freq / frame_rate)
            self.obb_times = self.obb_times[::obb_subsample]

        # Add points
        self.load_semidense(self.vrs_folder)

        # intersect all data modalities
        min_time, max_time = compute_time_intersection(ts_lists)

        play_times_ns = get_timestamp_list_ns(self.reader, ARIA_CAM_INFO["id"][idx])
        play_times_ns = [
            ts for ts in play_times_ns if (ts > min_time and ts < max_time)
        ]
        play_times_ns = np.unique(play_times_ns).tolist()

        # compute snippets start and end time
        seq_start_time = play_times_ns[0]
        seq_end_time = play_times_ns[-1]
        snip_start = seq_start_time
        snip_end = snip_start + snippet_length_s * 1e9
        self.snippet_times = []

        while snip_end < seq_end_time:
            self.snippet_times.append((snip_start, snip_end))
            snip_start += stride_length_s * 1e9
            snip_end = snip_start + snippet_length_s * 1e9

    def load_objects(self):
        self.obs = load_obbs_gt(
            self.vrs_folder,
            load_2d_bbs=True,
            filter_outside_2d_bbs=True,
            rgb_only=False,
        )
        if len(self.obs) == 0:
            return None

        # inverse map from proto to a linear id and filter the interested objects if given.
        instance2proto = self.obs["inst2proto"]
        unique_proto_names = np.unique(list(instance2proto.values())).tolist()
        self.obs["proto2id"] = {name: i for i, name in enumerate(unique_proto_names)}

        if self.is_aeo:
            aeo_to_efm = (
                f"{os.path.dirname(__file__)}/../config/taxonomy/aeo_to_efm.csv"
            )
            self.global_name_to_id = {}
            with open(aeo_to_efm, "r") as f:
                lines = f.readlines()
            for li in lines[1:]:
                ori_name, class_name, class_id = li.strip().split(",")
                self.global_name_to_id[str(ori_name)] = (str(class_name), int(class_id))

            filtered_proto_names = set(self.global_name_to_id.keys()).intersection(
                set(unique_proto_names)
            )

            # remap the proto names and semantic ids given the taxonomy mapping
            self.obs["proto2id"] = {
                self.global_name_to_id[name][0]: self.global_name_to_id[name][1]
                for name in filtered_proto_names
            }
            self.obs["inst2proto"] = {
                inst: self.global_name_to_id[name][0]
                for inst, name in instance2proto.items()
                if name in filtered_proto_names
            }
        else:
            # use the class name to id mapping in the sequence
            self.obs["proto2id"] = {
                name: i for i, name in enumerate(unique_proto_names)
            }

        # compute inverse map
        self.obs["id2proto"] = {id: name for name, id in self.obs["proto2id"].items()}

        timedTs_world_object = self.obs["timedTs_world_object"]
        static_Ts_world_object = {}
        assert (
            len(timedTs_world_object) != 0
        ), "Warning: no observations found for entire sequence"
        # timedTs_world_object captures static object at the -1 timestamp
        if -1 in timedTs_world_object.keys():
            static_Ts_world_object = timedTs_world_object[-1]
        self.obs["static_Ts_world_object"] = static_Ts_world_object
        self.obb_times = sorted(set(self.obs[ARIA_OBB_BB2[0]].keys()))

        if self.is_adt:
            T_vio_gravity = get_transform_to_vio_gravity_convention(
                GRAVITY_DIRECTION_ADT
            )
            for time, idT_wo in self.obs["timedTs_world_object"].items():
                for inst, T_wo in idT_wo.items():
                    # we go from gravity world coordinate system to the new one that follows vio conventions
                    self.obs["timedTs_world_object"][time][inst] = (
                        T_vio_gravity @ T_wo.float()
                    )

        return self.obs

    def load_semidense(self, vrs_path, max_inv_depth_std=0.005, max_depth_std=0.05):
        possible_global_points_paths = [
            os.path.join(vrs_path, "multi_global_points.csv.gz"),
            os.path.join(vrs_path, "multi_global_points.csv"),
            os.path.join(vrs_path, "global_points.csv.gz"),
            os.path.join(vrs_path, "global_points.csv"),
            os.path.join(vrs_path, "semidense_points.csv.gz"),
            os.path.join(vrs_path, "maps/maps_v1/globalcloud_GT.csv"),  # ASE
            os.path.join(vrs_path, "mps/slam/semidense_points.csv.gz"),  # ADT
        ]
        possible_obs_paths = [
            os.path.join(vrs_path, "semidense_observations.csv.gz"),
            os.path.join(vrs_path, "semidense_observations.csv"),
            os.path.join(vrs_path, "maps/maps_v1/observations.csv"),  # ASE
            os.path.join(vrs_path, "semidense_points.csv"),
            os.path.join(vrs_path, "mps/slam/semidense_observations.csv.gz"),  # ADT
        ]
        global_points_path = exists_nonzero_path(possible_global_points_paths)
        self.uid_to_p3, self.uid_to_inv_dist_std, self.uid_to_dist_std = (
            load_global_points_csv(global_points_path, max_inv_depth_std, max_depth_std)
        )

        if self.is_adt:
            T_vio_gravity = get_transform_to_vio_gravity_convention(
                GRAVITY_DIRECTION_ADT
            ).double()
            for uid, p3 in self.uid_to_p3.items():
                self.uid_to_p3[uid] = (T_vio_gravity * p3).reshape(-1)

        semidense_obs_path = exists_nonzero_path(possible_obs_paths)
        self.time_to_uids, self.uid_to_times = load_semidense_observations(
            semidense_obs_path
        )

        if self.time_to_uids is not None:
            self.pts_times_ns = sorted(self.time_to_uids.keys())
            (
                self.time_to_pc,
                self.time_to_dist_std,
                self.time_to_inv_dist_std,
                no_points_times,
            ) = ({}, {}, {}, [])
            for time in self.pts_times_ns:
                uids = self.time_to_uids[time]
                p3s = [self.uid_to_p3[uid] for uid in uids if uid in self.uid_to_p3]
                if len(p3s) > 0:
                    # sort by inv dist std to make any cropping later use the best points
                    inv_dist_std = [
                        self.uid_to_inv_dist_std[uid]
                        for uid in uids
                        if uid in self.uid_to_inv_dist_std
                    ]
                    inv_dist_std = np.array(inv_dist_std)
                    dist_std = [
                        self.uid_to_dist_std[uid]
                        for uid in uids
                        if uid in self.uid_to_dist_std
                    ]
                    dist_std = np.array(dist_std)
                    ids = np.argsort(inv_dist_std)
                    p3s = [p3s[i] for i in ids]
                    p3s = torch.stack(p3s)
                    inv_dist_std = torch.from_numpy(inv_dist_std[ids])
                    dist_std = torch.from_numpy(dist_std[ids])
                else:
                    no_points_times.append(time)
                    p3s = torch.zeros((0, 3), dtype=torch.float32)
                    inv_dist_std = torch.zeros((0), dtype=torch.float32)
                    dist_std = torch.zeros((0), dtype=torch.float32)
                self.time_to_pc[time] = p3s
                self.time_to_dist_std[time] = dist_std
                self.time_to_inv_dist_std[time] = inv_dist_std
        print(
            f"Found {len(self.uid_to_p3)} semidense points; time range {min(self.pts_times_ns)/1e9}s-{max(self.pts_times_ns)/1e9}s"
        )

        # aggregate all the points
        all_p3s = [self.uid_to_p3[uid] for uid in self.uid_to_p3]
        all_inv_dist_std = [
            self.uid_to_inv_dist_std[uid] for uid in self.uid_to_inv_dist_std
        ]
        ids = np.argsort(all_inv_dist_std)
        # ranked by inverse depth std
        self.all_p3s = torch.stack([all_p3s[i] for i in ids])  # [N, 3]
        assert self.all_p3s.shape[0] > 0, "no points loaded"

        # compute a [q, 1-q] percentile as the global range
        q = 0.001
        self.vol_min = torch.quantile(self.all_p3s, q, dim=0)
        self.vol_max = torch.quantile(self.all_p3s, 1 - q, dim=0)
        self.vol_min = self.vol_min.detach()
        self.vol_max = self.vol_max.detach()

    def load_poses(self, vrs_path, subsample):
        timed_Ts_world_rig = None
        # ADT sequences
        timed_Ts_world_rig = load_trajectory_adt(vrs_path, subsample=subsample)
        if timed_Ts_world_rig is not None:
            # handle ADT sequence gravity rotation
            T_vio_gravity = get_transform_to_vio_gravity_convention(
                GRAVITY_DIRECTION_ADT
            ).double()
            for k, T_wr in timed_Ts_world_rig.items():
                timed_Ts_world_rig[k] = T_vio_gravity @ T_wr
            return timed_Ts_world_rig

        # AEO sequences
        timed_Ts_world_rig = load_trajectory_aeo(
            vrs_path,
            time_in_secs=False,
            load_torch=True,
            subsample=subsample,
        )
        if timed_Ts_world_rig is not None:
            return timed_Ts_world_rig

        # Other sequences
        timed_Ts_world_rig = load_trajectory(
            vrs_path,
            time_in_secs=False,
            load_torch=True,
            subsample=subsample,
        )

        return timed_Ts_world_rig

    def load_snippet_pose(self, start, end):
        idx_i, idx_j = sample_times(self.T_world_rig_time_ns, start, end)
        Ts_wr = self.Ts_world_rig[idx_i:idx_j, :]
        pose_times_ns = torch.LongTensor(self.T_world_rig_time_ns[idx_i:idx_j])

        T_ws = Ts_wr[0].clone().unsqueeze(0)
        Ts_sr = T_ws.inverse() @ Ts_wr
        pose_times_s = (
            pose_times_ns - torch.tensor(start, dtype=torch.long)
        ).float() * 1e-9
        return T_ws, Ts_wr, Ts_sr, pose_times_ns, pose_times_s

    def load_snippet_semidense(self, start, end, max_size=20000):
        idx_i, idx_j = sample_times(self.pts_times_ns, start, end)
        points_times_ns = self.pts_times_ns[idx_i:idx_j]
        points_world = [self.time_to_pc[time] for time in points_times_ns]

        for idx, ps in enumerate(points_world):
            ps = ps[:max_size, :]
            pad_num = max_size - ps.shape[0]
            assert pad_num >= 0, f"padding must be non-negative, but got {pad_num}"

            points_world[idx] = F.pad(
                ps,
                (0, 0, 0, pad_num),
                "constant",
                float("nan"),
            )
        points_world = torch.stack(points_world)
        points_times_ns = torch.LongTensor(points_times_ns)
        points_times_s = (
            points_times_ns - torch.tensor(start, dtype=torch.long)
        ).float() * 1e-9
        return points_world, points_times_ns, points_times_s

    def load_snippet_objects(self, start, end):
        def get_obbs_for_time(t: int, inst_ids: List):
            (
                bb2s_rgb,
                bb2s_slaml,
                bb2s_slamr,
                bb3s,
                Ts_world_object,
                sem_ids,
                inst_ids,
            ) = next_obb_observations(
                obs=self.obs,
                time=t,
                inst_ids=inst_ids,
                cam_names=["rgb", "slaml", "slamr"],
                load_dynamic_objects=True,
                interpolate_poses=True,
                dt_threshold_ns=10_000_000,
            )
            obbs = ObbTW.from_lmc(
                bb3s,
                bb2s_rgb,
                bb2s_slaml,
                bb2s_slamr,
                Ts_world_object,
                sem_ids,
                inst_ids,
            )
            # scale 2d bbs to image size
            obbs = rescale_obb_tw(
                obbs,
                cam_size_before_rgb=[1408, 1408, 3],  # Aria rgb size
                cam_size_before_slam=[480, 640, 1],  # Aria slam size
                down_scale=self.sdi,
                wh_multiple_of=16,
            )
            # center object bounding box in the object coordinate system
            # T_world_object so that origin is the center of the object
            obbs = obbs.center()

            # get object sem_id to name mapping
            sem_id_to_name = {
                self.obs["proto2id"][self.obs["inst2proto"][iid.item()]]: self.obs[
                    "inst2proto"
                ][iid.item()]
                for iid in inst_ids
            }
            return obbs, sem_id_to_name

        obbs_snippet, sem_id_to_name, snippet_times = [], {}, []
        probably_snippet_times = [t for t in self.obb_times if start < t and t <= end]
        for t in probably_snippet_times:
            # we get only the instances that are visibile as indicated by them having 2d bb annotations
            inst_ids = get_instance_id_in_frameset(
                self.obs,
                t,
                load_dynamic_objects=True,
                interpolate_poses=True,
                dt_threshold_ns=10_000_000,
            )
            snippet_times.append(t)
            if len(inst_ids) == 0:
                obbs_snippet.append(
                    ObbTW(-1 * torch.ones(self.max_objects_per_frameset, 34))
                )
                continue
            obbs, sem2names = get_obbs_for_time(t, inst_ids)
            obbs_snippet.append(obbs.add_padding(self.max_objects_per_frameset))
            sem_id_to_name.update(sem2names)

        if len(obbs_snippet) > 0:
            obbs_padded = ObbTW(smart_stack(obbs_snippet))
        else:
            obbs_padded = ObbTW(-1 * torch.ones((0, self.max_objects_per_frameset, 34)))
            print(f"could not find obbs for snippet times {snippet_times}")
        obbs_time_ns = torch.LongTensor(snippet_times)
        obbs_time_s = (
            obbs_time_ns - torch.tensor(start, dtype=torch.long)
        ).float() * 1e-9

        # subsample
        obj_idxs = sample_from_range(
            0, len(obbs_padded), sample_rate=1, add_random=False
        )
        obbs_padded = obbs_padded[obj_idxs].contiguous()
        obbs_time_ns = obbs_time_ns[obj_idxs].contiguous()
        obbs_time_s = obbs_time_s[obj_idxs].contiguous()

        return obbs_padded, sem_id_to_name, obbs_time_ns, obbs_time_s

    def __len__(self):
        return min(len(self.snippet_times), self.max_snippets)

    def __getitem__(self, index):
        if index >= self.max_snippets:
            raise StopIteration

        sample = {}
        start, end = self.snippet_times[index]

        rgb_calib = {key: self.cam_calib[key]["rgb"] for key in self.cam_calib}

        # img
        for i in range(3):
            subsample = int(self.fps[i] / self.frame_rate)
            imgs, img_times_ns, cam_tws, frame_ids = read_image_snippet_from_vrs(
                self.reader,
                ARIA_CAM_INFO["id"][i],
                start,
                end,
                rgb_calib,
                subsample=subsample,
                scale_down_images=self.sdi,
            )
            img_times_s = (
                img_times_ns - torch.tensor(start, dtype=torch.long).float()
            ) * 1e-9

            sample.update(
                {
                    ARIA_IMG[i]: imgs,
                    ARIA_IMG_TIME_NS[i]: img_times_ns,
                    ARIA_IMG_SNIPPET_TIME_S[i]: img_times_s,
                    ARIA_FRAME_ID[i]: frame_ids,
                    ARIA_CALIB[i]: cam_tws,
                    ARIA_CALIB_TIME_NS[i]: img_times_ns,
                    ARIA_CALIB_SNIPPET_TIME_S[i]: img_times_s,
                }
            )

        # pose
        T_ws, Ts_wr, Ts_sr, pose_times_ns, pose_times_s = self.load_snippet_pose(
            start, end
        )
        sample.update(
            {
                ARIA_SNIPPET_T_WORLD_SNIPPET: T_ws,
                ARIA_POSE_T_WORLD_RIG: Ts_wr,
                ARIA_POSE_T_SNIPPET_RIG: Ts_sr,
                ARIA_POSE_TIME_NS: pose_times_ns,
                ARIA_POSE_SNIPPET_TIME_S: pose_times_s,
            }
        )

        # interpolate slam poses to get img poses
        sample.update(run_sensor_poses(sample))

        # semidense points
        pts_world, pts_times_ns, pts_times_s = self.load_snippet_semidense(start, end)
        sample.update(
            {
                ARIA_POINTS_WORLD: pts_world,
                ARIA_POINTS_TIME_NS: pts_times_ns,
                ARIA_POINTS_SNIPPET_TIME_S: pts_times_s,
                ARIA_POINTS_VOL_MIN: self.vol_min,
                ARIA_POINTS_VOL_MAX: self.vol_max,
            }
        )

        # objects
        if self.obs:
            obbs_padded, sem_id_to_name, obbs_time_ns, obbs_time_s = (
                self.load_snippet_objects(start, end)
            )
            # transform obbs into snippet coordinate system
            obbs_padded = transform_obbs(obbs_padded, T_ws.float().inverse())
            sample.update(
                {
                    ARIA_OBB_PADDED: obbs_padded,
                    ARIA_OBB_SEM_ID_TO_NAME: sem_id_to_name,
                    ARIA_OBB_TIME_NS: obbs_time_ns,
                    ARIA_OBB_SNIPPET_TIME_S: obbs_time_s,
                }
            )

        for key in sample:
            if isinstance(sample[key], (PoseTW, CameraTW, ObbTW)):
                sample[key] = sample[key].tensor()

            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].float()

            if key not in [
                ARIA_SNIPPET_T_WORLD_SNIPPET,
                ARIA_POINTS_VOL_MIN,
                ARIA_POINTS_VOL_MAX,
                ARIA_OBB_SEM_ID_TO_NAME,
            ]:
                sample[key] = tensor_unify(sample[key], self.frame_rate)

        if self.preprocess:
            sample = self.preprocess(sample)

        return sample
