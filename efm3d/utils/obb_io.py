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

import numpy as np
import torch
from efm3d.aria.aria_constants import ARIA_OBB_BB2, ARIA_OBB_BB3
from efm3d.aria.pose import closest_timed_poses, interpolate_timed_poses, PoseTW
from efm3d.utils.common import find_nearest


def bb2extent(bb):
    if bb.ndim == 1:
        bb = bb.reshape(1, -1)
    x_min = bb[:, 0].min()
    x_max = bb[:, 0].max()
    y_min = bb[:, 1].min()
    y_max = bb[:, 1].max()
    z_min = bb[:, 2].min()
    z_max = bb[:, 2].max()
    out = np.stack([x_min, x_max, y_min, y_max, z_min, z_max], axis=0)
    return out


def extent2bb(extent):
    if extent.ndim == 1:
        extent = extent.reshape(1, -1)

    x_min, x_max = extent[:, 0], extent[:, 1]
    y_min, y_max = extent[:, 2], extent[:, 3]
    z_min, z_max = extent[:, 4], extent[:, 5]
    arr = (
        [
            x_min,
            y_min,
            z_min,
            x_max,
            y_min,
            z_min,
            x_max,
            y_max,
            z_min,
            x_min,
            y_max,
            z_min,
            x_min,
            y_min,
            z_max,
            x_max,
            y_min,
            z_max,
            x_max,
            y_max,
            z_max,
            x_min,
            y_max,
            z_max,
        ],
    )
    if torch.is_tensor(extent):
        bb3d = torch.stack(arr, dim=-1).reshape(-1, 8, 3)
    elif isinstance(extent, np.ndarray):
        bb3d = np.stack(arr, axis=-1).reshape(-1, 8, 3)
    else:
        raise TypeError("Unknown type")

    return bb3d.squeeze()


def get_all_Ts_world_object_for_time(
    obs,
    time,
    load_dynamic_objects=True,
    interpolate_poses=True,
    dt_threshold_ns: int = 10_000_000,
):
    # concat static obb poses and dynamic ones at the current time
    static_Ts_world_object = obs["static_Ts_world_object"]
    have_dynamic_objects = len(obs["timedTs_world_object"]) > 1
    if load_dynamic_objects and have_dynamic_objects:

        if time in obs["timedTs_world_object"].keys():
            dynamic_Ts_world_object = obs["timedTs_world_object"][time]
        else:
            if interpolate_poses:
                dynamic_Ts_world_object = interpolate_timed_poses(
                    obs["timedTs_world_object"], time
                )
                print(
                    f"Warning: did not find time {time} in dynamic objects pose map - so interpolated poses"
                )
            else:
                dynamic_Ts_world_object, dt = closest_timed_poses(
                    obs["timedTs_world_object"], time
                )
                if abs(dt) > dt_threshold_ns:
                    dynamic_Ts_world_object = {}
                else:
                    print(
                        f"Warning: no time {time} in dynamic objects pose map - picked closest pose before in time {dt}"
                    )
    else:
        dynamic_Ts_world_object = {}
    all_Ts_world_object = {}
    all_Ts_world_object.update(static_Ts_world_object)
    all_Ts_world_object.update(dynamic_Ts_world_object)
    static_inst = set(static_Ts_world_object.keys())
    dynamic_inst = set(dynamic_Ts_world_object.keys())
    if len(static_inst.intersection(dynamic_inst)):
        print(
            "Warning: static and dynamic instances overlap overwriting static poses with dynamic ones! "
        )

    return all_Ts_world_object


def get_inst_id_in_camera(
    bb2s_camera,
    time: int,
    camera_name: str,
):
    if bb2s_camera and time in bb2s_camera.keys():
        inst_ids = [line[0] for line in bb2s_camera[time]]
    else:
        bb2_times = list(bb2s_camera.keys())
        nearest_idx = find_nearest(bb2_times, float(time), return_index=True)
        nearest_time = bb2_times[nearest_idx]
        if abs(time - nearest_time) >= 1_000_000:
            print(
                f"Error: {camera_name}: target time {time}ns has too large gap from the found nearest time {nearest_time}ns with gap {abs(time-nearest_time)}ns, skip this frame."
            )
            return []
        print(
            f"{camera_name}:",
            time,
            nearest_time,
            time - nearest_time,
        )
        inst_ids = [line[0] for line in bb2s_camera[nearest_time]]
    return inst_ids


def get_instance_id_in_frameset(
    obs,
    time: int,
    load_dynamic_objects: bool,
    interpolate_poses: bool = True,
    dt_threshold_ns: int = 10_000_000,
):
    # Get 3D object transforms that are visible in this frameset.
    bb2s_rgb = obs[ARIA_OBB_BB2[0]]
    bb2s_slaml = obs[ARIA_OBB_BB2[1]]
    bb2s_slamr = obs[ARIA_OBB_BB2[2]]
    bb2_time_rgb = time

    all_Ts_world_object = get_all_Ts_world_object_for_time(
        obs,
        bb2_time_rgb,
        load_dynamic_objects,
        interpolate_poses=interpolate_poses,
        dt_threshold_ns=dt_threshold_ns,
    )
    instance2proto = obs["inst2proto"]
    local_extents = obs[ARIA_OBB_BB3]

    inst_ids_rgb = get_inst_id_in_camera(bb2s_rgb, bb2_time_rgb, "rgb")

    # Support having visibility for only RGB.
    if len(bb2s_slaml) == 0:
        inst_ids_slaml = []
    else:
        inst_ids_slaml = get_inst_id_in_camera(bb2s_slaml, bb2_time_rgb, "slaml")
    if len(bb2s_slamr) == 0:
        inst_ids_slamr = []
    else:
        inst_ids_slamr = get_inst_id_in_camera(bb2s_slamr, bb2_time_rgb, "slamr")

    # Get union of all instance ids.
    inst_ids = list(
        set(inst_ids_rgb).union(set(inst_ids_slaml)).union(set(inst_ids_slamr))
    )
    # Make sure that all 2D BB instance ids have a 3D pose, prototype and local extent.
    warning_ids = [
        id
        for id in inst_ids
        if id not in all_Ts_world_object
        or id not in instance2proto
        or id not in local_extents
    ]
    if len(warning_ids) > 0:
        [inst_ids.remove(warning_id) for warning_id in warning_ids]

    inst_ids = np.unique(inst_ids)
    return inst_ids


def get_bb2s_for_instances(obs, time, inst_ids, cam_names, cam_scales=None):
    """
    Args:
        obs (dict): observation dict from Hive table
        time (int): nanoseconds timestamp of observation
        inst_ids (list): list of instance ids to get 2D BBs for
        cam_names (list): list of camera names
        cam_scales (dict): dict of camera scale for each camera (via cam_name) {cam_name:[x_scal, y_scale]}
    """
    # visible bounding boixes are >=0; invisible ones are < 0
    no_bb2 = [-1, -1, -1, -1]
    bb2_time_rgb = time
    bb2s = {cam_name: [] for cam_name in cam_names}
    for bb2_name, cam_name in zip(ARIA_OBB_BB2, cam_names):
        if bb2_time_rgb not in obs[bb2_name].keys():
            bb2_insts = [no_bb2] * len(inst_ids)
        else:
            bb2_obs_at_time = obs[bb2_name][bb2_time_rgb]
            bb2_insts = bb2s[cam_name]
            for iid in inst_ids:
                bb2 = None
                for line in bb2_obs_at_time:
                    if line[0] == iid:
                        bb2 = line[1:]
                        break
                if bb2:
                    bb2_insts.append(bb2)
                else:
                    bb2_insts.append(no_bb2)
        bb2_insts = torch.from_numpy(np.array(bb2_insts)).float()
        if cam_scales:
            bb2_insts[:2] = bb2_insts[:2] * cam_scales[cam_name][0]
            bb2_insts[2:] = bb2_insts[2:] * cam_scales[cam_name][1]
        bb2s[cam_name] = bb2_insts
    return bb2s


def next_obb_observations(
    obs,
    time,
    inst_ids,
    cam_names,
    cam_scales=None,
    load_dynamic_objects: bool = True,
    interpolate_poses: bool = True,
    dt_threshold_ns: int = 10_000_000,
):
    """
    Args:
        obs (dict): observation dict from Hive table
        time (float): timestamp of observation
        inst_ids (list): list of instance ids to get 2D BBs for
        cam_names (list): list of camera names
        cam_scales (dict): dict of camera scale for each camera (via cam_name) {cam_name:[x_scal, y_scale]}
    """
    all_Ts_world_object = get_all_Ts_world_object_for_time(
        obs,
        time,
        load_dynamic_objects=load_dynamic_objects,
        interpolate_poses=interpolate_poses,
        dt_threshold_ns=dt_threshold_ns,
    )
    # make sure we have a pose for all instances at this time.
    inst_ids = list(set(inst_ids).intersection(set(all_Ts_world_object.keys())))
    # make sure we have instances for all obb extends
    inst_ids = list(set(inst_ids).intersection(set(obs[ARIA_OBB_BB3].keys())))

    # get data
    Ts_wo = [all_Ts_world_object[iid] for iid in inst_ids]
    proto_names = [obs["inst2proto"][iid] for iid in inst_ids]
    proto_ids = [obs["proto2id"][name] for name in proto_names]
    exs = [obs[ARIA_OBB_BB3][iid] for iid in inst_ids]
    bbs_object = np.array([extent2bb(ex) for ex in exs])
    bbs_object = torch.tensor(bbs_object).float()
    # handle no obbs case.
    if Ts_wo:
        Ts_world_object = torch.stack(Ts_wo).float()
    else:
        Ts_world_object = PoseTW(torch.zeros(0, 12))
    inst_ids = torch.tensor(inst_ids)
    sem_ids = torch.tensor(proto_ids)
    bb3 = torch.from_numpy(np.array([bb2extent(bb) for bb in bbs_object]))
    # get 2D BBs for this frame
    bb2s = get_bb2s_for_instances(obs, time, inst_ids, cam_names, cam_scales)
    return (
        bb2s["rgb"],
        bb2s["slaml"],
        bb2s["slamr"],
        bb3,
        Ts_world_object,
        sem_ids,
        inst_ids,
    )
