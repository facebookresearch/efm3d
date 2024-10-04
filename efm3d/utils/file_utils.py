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

import gzip
import json
import os
import pickle
import random
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import numpy as np
import pandas as pd
import pyvrs
import torch
import tqdm

from efm3d.aria import CameraTW, PoseTW
from efm3d.aria.aria_constants import ARIA_CAM_INFO, ARIA_OBB_BB2, ARIA_OBB_BB3
from efm3d.utils.rescale import rescale_camera_tw, rescale_image
from pyquaternion import Quaternion
from pyvrs import SyncVRSReader
from vrsbindings import ImageConversion, RecordType


def load_gt_calibration(
    calib_path: Union[str, dict], load_torch=False, timestamps=None
):
    """load ground truth calibration json from simulation"""

    if isinstance(calib_path, str):
        with fsspec.open(calib_path, "r") as f:
            calib = json.load(f)
    elif isinstance(calib_path, dict):
        calib = calib_path
    else:
        raise IOError("calib_path must be str or dict")
    gt_calib = {}
    gt_calib["T_rig_views"] = {}
    gt_calib["intr_type"] = {}
    gt_calib["intr_params"] = {}

    cam_names = ARIA_CAM_INFO["name"]
    # Maps names from the gt_calib.json file to the ARIA_CAM_INFO convention.
    name_map = {
        "camera-rgb": cam_names[0],
        "camera-slam-left": cam_names[1],
        "camera-slam-right": cam_names[2],
    }
    for camera in calib["CameraCalibrations"]:
        cn = camera["Label"]
        if cn not in name_map:  # Ignore other cameras like eye tracking.
            continue
        cam_name = name_map[cn]
        [tx, ty, tz] = camera["T_Device_Camera"]["Translation"]
        [qw, [qx, qy, qz]] = camera["T_Device_Camera"]["UnitQuaternion"]

        rot_mat = Quaternion(qw, qx, qy, qz).rotation_matrix
        translation = torch.tensor([tx, ty, tz]).view(3, 1)
        T_rig_view = torch.concat([torch.tensor(rot_mat), translation], dim=1)
        T_rig_view = PoseTW.from_matrix3x4(T_rig_view)
        T_rig_view = T_rig_view.fit_to_SO3()
        if not load_torch:
            T_rig_view = T_rig_view.numpy()
        gt_calib["T_rig_views"][cam_name] = T_rig_view

        intr_type = camera["Projection"]["Name"]
        # This is the case for Fisheye62 which has 6+2+3=11 parameters, morphed as Fisheye624
        # Add zeros to make it 15 params (same as Fisheye624)
        if intr_type == "Fisheye624":
            N = 15 - len(camera["Projection"]["Params"])
            if N > 0:
                for _i in range(N):
                    camera["Projection"]["Params"].append(0)
        intr_params = np.array(camera["Projection"]["Params"])
        if load_torch:
            intr_params = torch.from_numpy(intr_params)
        gt_calib["intr_type"][cam_name] = intr_type
        gt_calib["intr_params"][cam_name] = intr_params

    if timestamps is not None:
        time2calib = {}
        for timestamp in timestamps:
            time2calib[timestamp] = gt_calib
        return time2calib

    return gt_calib


def get_image_info(image_reader: SyncVRSReader) -> Tuple[Dict, Dict]:
    """
    Get image info such as sizes and frame rate. These fields are not
    part of calibration so we have to query them through VRSReader.
    """
    image_sizes = {}
    fps = {}
    image_config_reader = image_reader.filtered_by_fields(
        record_types=["configuration"]
    )
    for image_config in image_config_reader:
        assert image_config.record_type == "configuration"
        stream_id = image_config.stream_id
        if stream_id not in ARIA_CAM_INFO["id_to_name"]:
            continue
        name = ARIA_CAM_INFO["id_to_name"][stream_id]
        metadata = image_config.metadata_blocks[0]
        image_sizes[name] = metadata["image_height"], metadata["image_width"]
        fps[name] = metadata["nominal_rate"]
    return image_sizes, fps


def load_factory_calib(
    reader: SyncVRSReader,
    calib: Optional[str] = None,
    map_radius_to_cam_height: bool = False,
):
    """
    Augment `load_gt_calibration` by adding `image_sizes`, `camera_tw`
    (CameraTW objects), and `fps` for each camera. The reader has to be
    an image VRSReader.
    video_stream_name is needed for eye tracking images. Unlike slaml and slamr where their vrs ids are 1201-1 and 1201-2,
    eye tracking vrs id is only 211-1 for both left and right eye images
    """
    image_sizes, fps = get_image_info(reader)
    if "calib_json" in reader.file_tags:
        calib = json.loads(reader.file_tags["calib_json"])
    elif calib is None:
        return None
    cam_calib = load_gt_calibration(calib, load_torch=True, timestamps=None)
    cam_calib["image_sizes"] = image_sizes
    cam_calib["fps"] = fps
    cam_calib["camera_tw"] = {}

    # Hack to override the camera model instead of using cam_calib["intr_type"][cam_name] which is set to "Fisheye62"
    for cam_name in image_sizes:
        if map_radius_to_cam_height:
            cam_calib["camera_tw"][cam_name] = CameraTW.from_surreal(
                height=image_sizes[cam_name][0],
                width=image_sizes[cam_name][1],
                type_str=cam_calib["intr_type"][cam_name],
                params=cam_calib["intr_params"][cam_name],
                T_camera_rig=cam_calib["T_rig_views"][cam_name].inverse(),
                valid_radius=image_sizes[cam_name][0],
            )
        else:
            cam_calib["camera_tw"][cam_name] = CameraTW.from_surreal(
                height=image_sizes[cam_name][0],
                width=image_sizes[cam_name][1],
                type_str=cam_calib["intr_type"][cam_name],
                params=cam_calib["intr_params"][cam_name],
                T_camera_rig=cam_calib["T_rig_views"][cam_name].inverse(),
            )
    return cam_calib


def load_2d_bounding_boxes(bb2d_path, time_in_secs=False):
    bb2ds = {}

    try:
        with fsspec.open(bb2d_path).open() as f:
            # genfromtxt handles missing values and lets us specify dtypes.
            # #Object_UID, timestamp [nanoseconds], x_min [pixel], x_max [pixel], y_min [pixel], y_max [pixel]
            lines = np.genfromtxt(
                f,
                dtype=[int] * 2 + [float] * 4,
                names=True,
                delimiter=",",
                usecols=range(6),
            )
    except Exception:
        try:
            # sometimes the last row is bad for some reason so we just skip it
            with fsspec.open(bb2d_path).open() as f:
                # genfromtxt handles missing values and lets us specify dtypes.
                # #Object_UID, timestamp [nanoseconds], x_min [pixel], x_max [pixel], y_min [pixel], y_max [pixel]
                lines = np.genfromtxt(
                    f,
                    dtype=[int] * 2 + [float] * 4,
                    names=True,
                    delimiter=",",
                    usecols=range(6),
                    skip_footer=1,
                )
        except Exception as e:
            print(f"could not load {bb2d_path}; error {e}")
            return bb2ds

    count = 0
    for line in lines:
        object_id = line[0]
        timestamp_ns = line[1]
        if time_in_secs:
            timestamp = timestamp_ns / 1e9
        else:
            timestamp = timestamp_ns
        x_min = max(0, line[2])
        x_max = max(0, line[3])
        y_min = max(0, line[4])
        y_max = max(0, line[5])
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [x_min, x_max, y_min, y_max]):
            continue
        if timestamp not in bb2ds:
            bb2ds[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
        else:
            bb2ds[timestamp].append((object_id, x_min, x_max, y_min, y_max))
        count += 1
    print(f"loaded {count} 2d bbs for {len(bb2ds)} timestamps from {bb2d_path}")
    return bb2ds


def load_2d_bounding_boxes_adt(bb2d_path):
    bb2ds_rgb = {}
    bb2ds_slaml = {}
    bb2ds_slamr = {}

    with fsspec.open(bb2d_path).open() as f:
        lines = f.readlines()

    # expected header:
    # stream_id,object_uid,timestamp[ns],x_min[pixel],x_max[pixel],y_min[pixel],y_max[pixel],visibility_ratio[%]\n'

    count = 0
    for ii, line in enumerate(lines):
        if ii == 0:
            continue  # skip header
        line = line.decode("utf-8").rstrip().split(",")
        device_id = str(line[0])
        object_id = int(line[1])
        timestamp = int(line[2])  # ns
        x_min = max(0, float(line[3]))
        x_max = max(0, float(line[4]))
        y_min = max(0, float(line[5]))
        y_max = max(0, float(line[6]))
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [x_min, x_max, y_min, y_max]):
            continue

        if device_id == "214-1":
            if timestamp not in bb2ds_rgb:
                bb2ds_rgb[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_rgb[timestamp].append((object_id, x_min, x_max, y_min, y_max))

        elif device_id == "1201-1":
            if timestamp not in bb2ds_slaml:
                bb2ds_slaml[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_slaml[timestamp].append((object_id, x_min, x_max, y_min, y_max))

        elif device_id == "1201-2":
            if timestamp not in bb2ds_slamr:
                bb2ds_slamr[timestamp] = [(object_id, x_min, x_max, y_min, y_max)]
            else:
                bb2ds_slamr[timestamp].append((object_id, x_min, x_max, y_min, y_max))
        else:
            raise IOError("unexpected device id {device_id} in 2d observations")

        count += 1
    print(
        f"loaded {count} 2d bbs for {len(bb2ds_rgb)}[rgb] {len(bb2ds_slaml)}[slaml] {len(bb2ds_slamr)}[slamr] timestamps from {bb2d_path}"
    )
    return bb2ds_rgb, bb2ds_slaml, bb2ds_slamr


def remove_invalid_2d_bbs(timed_bb2s, filter_bb2_area=-1):
    """
    remove bbs with x, y <= 0. In some datasets (DlrSim) these 2d bbs indicate
    object is not visible!
    """
    bb2s_filtered = defaultdict(list)
    for time, bb2s in timed_bb2s.items():
        for bb2 in bb2s:
            if not ((bb2[1] <= 0 and bb2[2] <= 0) or (bb2[3] <= 0 and bb2[4] <= 0)):
                if filter_bb2_area > 0:
                    bb2_area = (bb2[2] - bb2[1]) * (bb2[4] - bb2[3])
                    if bb2_area >= filter_bb2_area:
                        bb2s_filtered[time].append(bb2)
                else:
                    bb2s_filtered[time].append(bb2)
    return bb2s_filtered


def load_instances(instances_path):
    instance2proto = {}

    assert os.path.exists(
        instances_path
    ), f"instances path {instances_path} does not exist"
    with open(instances_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip first line
        line = line.rstrip().split(",")
        instance_uid = int(line[0])
        prototype_uid = str(line[1]).strip()
        instance2proto[instance_uid] = prototype_uid

    return instance2proto


def load_instances_adt(instances_path):
    instance2proto = {}
    with fsspec.open(instances_path).open() as f:
        content = json.load(f)
    for inst_id in content:
        instance2proto[int(inst_id)] = content[inst_id]["category"]

    # lot of other info available, for example:
    #  {'instance_id': 5691266090916432, 'instance_name': 'Hook_4',
    #  'prototype_name': 'Hook', 'category': 'hook', 'category_uid': 643,
    #  'motion_type': 'static', 'instance_type': 'object', 'rigidity': 'rigid',
    #  'rotational_symmetry': {'is_annotated': False},
    #  'canonical_pose': {'up_vector': [0, 1, 0], 'front_vector': [0, 0, 1]}}

    return instance2proto


def load_3d_bounding_box_transforms(scene_path, time_in_secs=False, load_torch=False):
    T_world_object = {}

    with fsspec.open(scene_path).open() as f:
        lines = np.genfromtxt(
            f,
            dtype=[int] * 2 + [float] * 7,
            names=True,
            delimiter=",",
            usecols=range(9),
        )
        if lines.size == 1:
            lines = lines[np.newaxis]

    for line in lines:
        object_id = line[0]
        timestamp_ns = line[1]
        if time_in_secs and timestamp_ns != -1:
            timestamp = timestamp_ns / 1e9
        else:
            timestamp = timestamp_ns
        tx = line[2]
        ty = line[3]
        tz = line[4]
        qw = line[5]
        qx = line[6]
        qy = line[7]
        qz = line[8]
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [tx, ty, tz, qw, qx, qy, qz]):
            continue

        rot_mat = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix
        translation = torch.tensor([tx, ty, tz]).view(3, 1)
        T_wo = torch.concat([torch.tensor(rot_mat), translation], dim=1)
        T_wo = PoseTW.from_matrix3x4(T_wo)
        T_wo = T_wo.fit_to_SO3()
        if not load_torch:
            T_wo = T_wo.numpy()

        if timestamp not in T_world_object:
            T_world_object[timestamp] = {}
        T_world_object[timestamp][object_id] = T_wo
    return T_world_object


def load_3d_bounding_box_local_extents(bb3d_path, load_torch=False):
    bb3ds_local = {}
    with fsspec.open(bb3d_path).open() as f:
        # Object UID, Timestamp ( ns ), p_local_obj.xmin, p_local_obj.xmax, p_local_obj.ymin, p_local_obj.ymax, p_local_obj.zmin, p_local_obj.zmax
        lines = np.genfromtxt(
            f,
            dtype=[int] * 2 + [float] * 6,
            names=True,
            delimiter=",",
            usecols=range(8),
        )
        if lines.size == 1:
            lines = lines[np.newaxis]
    for line in lines:
        object_id = line[0]
        xmin = line[2]
        xmax = line[3]
        ymin = line[4]
        ymax = line[5]
        zmin = line[6]
        zmax = line[7]
        # invalid entries will have nan as fill value; we skip them.
        if any(x != x for x in [xmin, xmax, ymin, ymax, zmin, zmax]):
            continue
        local = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        if load_torch:
            local = torch.from_numpy(local)
        bb3ds_local[object_id] = local
    return bb3ds_local


def load_obbs_gt(
    input_dir,
    load_2d_bbs=True,
    filter_outside_2d_bbs: bool = False,
    rgb_only=False,
    filter_bb2_area=-1,
):
    obs = {}
    if load_2d_bbs:
        # Load 2d bbs from CSV.
        bb2s_path_rgb = exists_nonzero_path(
            [
                os.path.join(input_dir, "2d_bounding_box.csv"),
                os.path.join(input_dir, "2d_bounding_box_rgb.csv"),
                os.path.join(input_dir, "sensor_0_2d_bounding_box.csv"),
            ]
        )
        bb2s_path_slaml, bb2s_path_slamr = False, False
        if not rgb_only:
            bb2s_path_slaml = exists_nonzero_path(
                [
                    os.path.join(input_dir, "2d_bounding_box_2.csv"),
                    os.path.join(input_dir, "2d_bounding_box_left_slam.csv"),
                    os.path.join(input_dir, "sensor_1_2d_bounding_box.csv"),
                ]
            )
            bb2s_path_slamr = exists_nonzero_path(
                [
                    os.path.join(input_dir, "2d_bounding_box_3.csv"),
                    os.path.join(input_dir, "2d_bounding_box_right_slam.csv"),
                    os.path.join(input_dir, "sensor_2_2d_bounding_box.csv"),
                ]
            )

        bb2_loaded = False
        if bb2s_path_rgb:
            # ADT dataset packs all three bb2 observations into one file
            with fsspec.open(bb2s_path_rgb).open() as f:
                header = f.readline()
                header = str(header).split(",")
                if len(header) == 8:
                    (
                        obs[ARIA_OBB_BB2[0]],
                        obs[ARIA_OBB_BB2[1]],
                        obs[ARIA_OBB_BB2[2]],
                    ) = load_2d_bounding_boxes_adt(bb2s_path_rgb)
                    bb2_loaded = True

        if not bb2_loaded and bb2s_path_rgb and bb2s_path_slaml and bb2s_path_slamr:
            # Load 2d bounding boxes separately for three cameras
            obs[ARIA_OBB_BB2[0]] = load_2d_bounding_boxes(
                bb2s_path_rgb, time_in_secs=False
            )
            obs[ARIA_OBB_BB2[1]] = load_2d_bounding_boxes(
                bb2s_path_slaml, time_in_secs=False
            )
            obs[ARIA_OBB_BB2[2]] = load_2d_bounding_boxes(
                bb2s_path_slamr, time_in_secs=False
            )
            bb2_loaded = True
        elif not bb2_loaded and bb2s_path_rgb:
            # sometimes we only have RGB 2d bounding boxes.
            obs[ARIA_OBB_BB2[0]] = load_2d_bounding_boxes(
                bb2s_path_rgb, time_in_secs=False
            )
            obs[ARIA_OBB_BB2[1]] = {}
            obs[ARIA_OBB_BB2[2]] = {}
            bb2_loaded = True
        elif not bb2_loaded:
            print("Warning: could not find 2d bbs")
            return {}
    else:
        obs[ARIA_OBB_BB2[0]] = {}
        obs[ARIA_OBB_BB2[1]] = {}
        obs[ARIA_OBB_BB2[2]] = {}
        print("not loading 2d bb information")

    # most of the time bbs with x, y <= 0 indicate object is visible but we dont
    # know where. In the DlrSim dataset it indicates object not observed!
    if filter_outside_2d_bbs:
        for bb2_key in ARIA_OBB_BB2:
            obs[bb2_key] = remove_invalid_2d_bbs(obs[bb2_key], filter_bb2_area)

    # Load bounding box local 3D extents.
    bb3d_path = exists_nonzero_path(
        [
            os.path.join(input_dir, "scene/3d_bounding_box.csv"),
            os.path.join(input_dir, "3d_bounding_box.csv"),
        ]
    )
    if bb3d_path:
        obs[ARIA_OBB_BB3] = load_3d_bounding_box_local_extents(bb3d_path)

    # Load scene object centers + object_ids from scene_objects.csv
    scene_path = exists_nonzero_path(
        [
            os.path.join(input_dir, "scene/scene_objects.csv"),
            os.path.join(input_dir, "scene_objects.csv"),
        ]
    )
    if scene_path:
        obs["timedTs_world_object"] = load_3d_bounding_box_transforms(
            scene_path, time_in_secs=False, load_torch=True
        )
    # Load label mapping from instances to prototypes.
    instance_path = exists_nonzero_path(
        [
            # fixed some wrong 'rug' labels
            os.path.join(input_dir, "scene/instances_fix.csv"),
            os.path.join(input_dir, "scene/instances.csv"),
            os.path.join(input_dir, "instances.json"),
        ]
    )
    if instance_path:
        if instance_path.endswith(".csv"):
            obs["inst2proto"] = load_instances(instance_path)
        elif instance_path.endswith(".json"):
            obs["inst2proto"] = load_instances_adt(instance_path)
        else:
            raise IOError("Unknown instances extension")

    return obs


def load_trajectory_adt(
    traj_path,
    subsample: Union[float, int] = 1,
    load_first_n=99999999999,
):
    print("checking " + traj_path)
    fs = fsspec.get_mapper(traj_path).fs
    if not fs.exists(traj_path):
        return None
    if not fs.isfile(traj_path):
        traj_path = exists_nonzero_path(
            [
                os.path.join(traj_path, "aria_trajectory.csv"),  # ADT ground truth
            ]
        )
    if traj_path is None:
        return None
    print("loading " + traj_path)

    T_world_rigs = {}
    # check for number of columns first
    with fsspec.open(traj_path, "r").open() as f:
        header = f.readline()
        num_cols = len(header.split(","))
        if num_cols not in [20]:
            return None

    # load data without header
    with fsspec.open(traj_path, "rb").open() as f:
        lines = f.readlines()

    N = min(len(lines), load_first_n)
    idxs = sample_from_range(0, N, subsample)
    for ii in idxs:
        if ii == 0:
            continue  # skip header
        line = lines[ii]
        line = str(line).split(",")
        timestamp_us = int(line[1])
        timestamp_ns = timestamp_us * 1000
        timestamp = timestamp_ns
        sub_line = line[3:10]
        tx, ty, tz, qx, qy, qz, qw = [float(e) for e in sub_line]
        rot_mat = Quaternion(qw, qx, qy, qz).rotation_matrix
        translation = torch.tensor([tx, ty, tz]).view(3, 1)
        T_world_rig = torch.concat([torch.tensor(rot_mat), translation], dim=1)
        T_world_rig = PoseTW.from_matrix3x4(T_world_rig)
        T_world_rig = T_world_rig.fit_to_SO3()
        T_world_rigs[timestamp] = T_world_rig

    return T_world_rigs


def load_trajectory_aeo(
    csv_path: str,
    load_torch: bool = False,
    subsample: Union[float, int] = 1,
    time_in_secs: bool = False,
    load_first_n: int = 99999999999,
):
    assert not time_in_secs, "Only support time in ns for now"
    vio_filenames = [
        "closed_loop_framerate_trajectory.csv",
        "closed_loop_trajectory.csv",
        "mps/slam/closed_loop_trajectory.csv",
    ]
    lines = None
    for vio_filename in vio_filenames:
        traj_csv_path = os.path.join(csv_path, vio_filename)
        print("checking " + traj_csv_path)
        if os.path.exists(traj_csv_path):
            with open(traj_csv_path, "r") as f:
                lines = f.readlines()
            print(f"loaded {len(lines)} from " + traj_csv_path)
            break

    if lines is None:
        print(f"No file found in {csv_path}.")
        return None

    T_world_rigs = {}
    header = lines[0].strip().split(",")
    if len(header) not in {26, 28, 29}:
        print(f"Invalid header, expected 26, 28 or 29 columns, but got {len(header)}")
        print(header)
        return None

    start_index = 0
    if len(header) == 28:  # no recording_source field in this version
        start_index = -1

    N = min(len(lines), load_first_n)
    idxs = sample_from_range(1, N, subsample)
    for ii in idxs:
        line = lines[ii]
        # Handle data error
        line = line.strip()
        if len(line) == 0:
            continue

        cols = line.split(",")
        timestamp_ns = int(cols[start_index + 2]) * 1000
        tx, ty, tz, qx, qy, qz, qw = [
            float(num) for num in cols[start_index + 4 : start_index + 11]
        ]
        rot_mat = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix
        translation = torch.tensor([tx, ty, tz]).view(3, 1)
        T_world_rig = torch.concat([torch.tensor(rot_mat), translation], dim=1)
        T_world_rig = PoseTW.from_matrix3x4(T_world_rig)
        # T_world_rig = T_world_rig.fit_to_SO3()
        if not load_torch:
            T_world_rig = T_world_rig.numpy()

        T_world_rigs[timestamp_ns] = T_world_rig

    return T_world_rigs


def load_trajectory(
    traj_path,
    time_in_secs=False,
    load_torch=False,
    subsample: Union[float, int] = 1,
    load_quaternion=False,
    load_first_n=99999999999,
):
    print("checking " + traj_path)
    fs = fsspec.get_mapper(traj_path).fs
    if not fs.exists(traj_path):
        return None
    if not fs.isfile(traj_path):
        traj_path = exists_nonzero_path(
            [
                os.path.join(traj_path, "trajectory.csv"),  # default
                os.path.join(traj_path, "traj000.csv"),  # ASE
            ]
        )
    if traj_path is None:
        return None
    print("loading " + traj_path)

    T_world_rigs = {}
    # check for number of columns first
    with fsspec.open(traj_path, "r").open() as f:
        header = f.readline()
        num_cols = len(header.split(","))
        if num_cols not in [8, 14, 17]:
            return None
    # load data without header
    with fsspec.open(traj_path, "rb").open() as f:
        lines = np.loadtxt(f, delimiter=",", skiprows=1)

    N = min(len(lines), load_first_n)
    idxs = sample_from_range(0, N, subsample)
    for ii in idxs:
        line = lines[ii]
        timestamp_ns = int(line[0])
        if time_in_secs:
            timestamp = timestamp_ns / 1e9
        else:
            timestamp = timestamp_ns
        sub_line = line[1:8]
        tx, ty, tz, qw, qx, qy, qz = [float(e) for e in sub_line]
        rot_mat = Quaternion(w=qw, x=qx, y=qy, z=qz).rotation_matrix

        if load_quaternion:
            # allow "raw" loading
            T_world_rig = np.array([tx, ty, tz, qw, qx, qy, qz])
        else:
            translation = torch.tensor([tx, ty, tz]).view(3, 1)
            T_world_rig = torch.concat([torch.tensor(rot_mat), translation], dim=1)
            T_world_rig = PoseTW.from_matrix3x4(T_world_rig)
            T_world_rig = T_world_rig.fit_to_SO3()
            if not load_torch:
                T_world_rig = T_world_rig.numpy()
        T_world_rigs[timestamp] = T_world_rig

    return T_world_rigs


def parse_global_name_to_id_csv(csv_path: str, verbose: bool = True) -> Dict[str, int]:
    """
    Loads a csv with 2 columns: old_sem_name, sem_id and returns it as
    a dictionary of {old_sem_name: sem_id}
    """
    global_name_to_id = None
    if len(csv_path) > 0:
        if verbose:
            print(f"trying to load taxonomy from csv at {csv_path}")
        with fsspec.open(csv_path) as f:
            global_name_to_id = dict(
                np.loadtxt(
                    f,
                    delimiter=",",
                    skiprows=1,
                    dtype={
                        "names": ("Object Name", "Object cls ID"),
                        "formats": ("U30", int),
                    },
                    ndmin=1,
                )
            )
        if verbose:
            print(f"loaded {len(global_name_to_id)} name-to-id mappings from csv.")
    return global_name_to_id


def exists_nonzero_path(path: Union[str, list]) -> Optional[str]:
    """Helper function, iterate through paths to make sure exists;

    Input:
        paths - can be str or list of str
    Returns:
        found - if not found, return False, if found, return the path
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    # Iterate through each path, breaking if good file is found.
    found = None
    for path in paths:
        try:
            fs = fsspec.core.url_to_fs(path)[0]
        except Exception as e:
            print(f"skipping {path}: {e}")
            continue
        if fs.exists(path):
            found = path
            break
    return found


def get_timestamp_list_ns(reader, stream_id=None):
    if stream_id is None:
        filtered_reader = reader
    else:
        filtered_reader = reader.filtered_by_fields(
            stream_ids=[stream_id], record_types=["data"]
        )
    time_list = filtered_reader.get_timestamp_list()
    # go from vrs output of float times in seconds to long times in nanoseconds
    timestamp_list = [int(t * 1e9) for t in time_list]
    return timestamp_list


def sample_times(time_list: List, start_time: int, end_time: int) -> Tuple[int, int]:
    """
    Sample timestamps within the interval [start_time,end_time] using binary
    search, making sure that at least one sample is taken.

    Inputs:
        time_list: list of sorted times
        start_time: float of start of range to sample
        end_time: float of end of range to sample
    Returns:
        (idx_i,idx_j): tuple of indices into time_list of sampled range.

    Suppose the anchor modality is IMU, Using `bisect_left` would give
    a time order like this:
    image1, image2, image3, image4,   image5
       |      |       |       |          |
            img_i    ...   img_(j-1)   img_j
          |                       |
         imu1                    imu2
          |                       |
         audio1                  audio2
    """
    idx_i = bisect_left(time_list, start_time)
    idx_j = bisect_left(time_list, end_time)
    # Make sure sampled image data is in between the start and end time
    if idx_j > idx_i:
        assert (
            start_time <= time_list[idx_i]
            and time_list[idx_i] <= time_list[idx_j - 1]
            and time_list[idx_j - 1] < end_time
        ), f"start {start_time} end {end_time}, time_list[idx_i], {time_list[idx_i]}, time_list[idx_j], {time_list[idx_j]}"
    else:
        # make sure idx_j is greater than idx_i
        idx_j = max(idx_j, idx_i + 1)
    return idx_i, idx_j


def sample_from_range(
    start: int,
    end: int,
    sample_rate: Union[float, int],
    add_random: bool = True,
) -> List[int]:
    """
    sample from a range using defined sample_rate.
    Args:
        start (int): start of the range (inclusive).
        end (int): end of the range (exclusive).
        sample_rate (Union[float, int]): target sampling rate.
        add_random (bool): whether to add randomness to the final samples.
    Returns:
        list: a list of integers sampled from the range, in increasing order.
    Example:
        1. sample_rate is integer. We just return range(start, end, sample_rate).
        2. sample_rate is float.
            We first round up the sample_rate and then add the missing numbers from the reminder of the entire list.
            For example, we'd like to sample from [0, 1, 2, ..., 9] with sample rate 1.25, which will result in 8 samples.
            We first round the sample rate to 2, then get the list of numbers by range(0, 10, 2) which is [0, 2, 4, 6, 8].
            And then we randomly get 3 numbers from the reminder of the list [1, 3, 5, 7, 9] to add to the final sample list.
    """
    assert end >= start, "the end of the range must be greater than the start."
    assert sample_rate > 0, "sample rate must be positive."

    if end == start:
        print(f"[Warn] end equals start ({start}, {end}), return emply list")
        return []

    # if sample rate is an integer, we just return the sampling by using sample_rate as the step size.
    if type(sample_rate) is int or sample_rate.is_integer():
        return list(range(start, end, int(sample_rate)))

    # Otherwise, we do sampling with non-integer sampling rate.
    if (end - start) % sample_rate != 0:
        print(
            f"[WARN] sample_rate not divisible by for the range: got sample_rate {sample_rate}, start {start}, end {end}. Can not achieve the desired sampling rate in the end."
        )

    step = int(np.ceil(sample_rate))  # round-up the sampling rate
    num = int((end - start) / sample_rate)  # number of final samples

    # Generate the evenly spaced integers
    integers = list(range(start, end, step))

    # If we don't have enough integers, sample the missing ones randomly
    if len(integers) < num:
        missing_num = num - len(integers)
        # Create a list of potential candidates that excludes already selected integers
        candidates = [i for i in range(start, end) if i not in integers]
        # Add the missing integers
        if add_random:
            integers.extend(random.sample(candidates, missing_num))
        else:
            integers = list(
                np.linspace(start, end, num, endpoint=False).round().astype(int)
            )

    return sorted(integers)


def read_image_from_vrs(
    reader: pyvrs.filter.FilteredVRSReader,
    cam_id: str,
    image_ts_ns: int,
    intr_type: str,
    intr_params: Union[List, np.array],
    T_rig_camera: PoseTW,
    scale_down_images: int = 0,
    valid_radius: Optional[torch.Tensor] = None,
    wh_multiple_of: int = 16,
):
    """
    Expect all the input time is in vrs capture time domain.
    """
    cam_name = ARIA_CAM_INFO["id_to_name"][cam_id]

    # Read image from time-associated VRS block.
    ret_error = (None, None, None)
    try:
        # convert from nanoseconds to seconds for vrs reader
        image_ts = image_ts_ns / 1e9
        record = reader.read_record_by_time(
            cam_id, image_ts, record_type=RecordType.DATA
        )
    except ValueError as e:
        return ret_error
    if record is None:
        return ret_error
    if len(record.image_blocks) < 1:  # Bad image block.
        return ret_error
    else:
        image = record.image_blocks[0]
    cam_hw_before = image.shape

    exposure_s = record.metadata_blocks[0]["exposure_duration_s"]
    gain = record.metadata_blocks[0]["gain"]
    # note that currently capture_time_ns is equal to image_ts_ns but this might
    # change (?) so we rely on this meta data instead and pass it back out.
    capture_time_ns = record.metadata_blocks[0]["capture_timestamp_ns"]

    cam = CameraTW.from_surreal(
        height=cam_hw_before[0],
        width=cam_hw_before[1],
        type_str=intr_type,
        params=intr_params,
        T_camera_rig=T_rig_camera.inverse(),
        exposure_s=exposure_s,
        gain=gain,
        valid_radius=valid_radius,
    )

    image = rescale_image(image, cam_name, scale_down_images, wh_multiple_of)
    cam = rescale_camera_tw(
        cam, cam_hw_before, cam_name, scale_down_images, wh_multiple_of
    )

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    image = image.transpose(2, 0, 1)  # HxWxC -> CxHxW
    image = torch.tensor(image.astype(np.float32) / 255.0)

    return image, cam, capture_time_ns


def read_image_snippet_from_vrs(
    image_reader: SyncVRSReader,
    cam_id: str,
    start_time_ns: int,
    end_time_ns: int,
    cam_calib,
    subsample: Union[float, int] = 1,
    scale_down_images: int = 0,
    valid_radius: Optional[torch.Tensor] = None,
    wh_multiple_of: int = 16,
):
    """
    If time code mapping provided, assume the input time is the timecode time domain.
    Need to convert it to capture time domain to read data.
    Otherwise, the start_time_ns and end_time_ns need to be in the capture time domain.
    Output time domain is always aligned with the input time domain.
    """
    image_reader.set_image_conversion(conversion=ImageConversion.NORMALIZE)
    filtered_reader = image_reader.filtered_by_fields(
        stream_ids=[cam_id], record_types=["data"]
    )
    capture_time_list_ns = get_timestamp_list_ns(filtered_reader)

    img_i, img_j = sample_times(capture_time_list_ns, start_time_ns, end_time_ns)

    images = []
    times_ns = []
    cam_tws = []
    frame_ids = []
    sample_range = sample_from_range(
        img_i, img_j, sample_rate=subsample, add_random=False
    )
    for i in sample_range:
        image, cam_tw, capture_image_time_ns = read_image_from_vrs(
            reader=filtered_reader,
            cam_id=cam_id,
            image_ts_ns=capture_time_list_ns[i],
            intr_type=cam_calib["intr_type"],
            intr_params=cam_calib["intr_params"],
            T_rig_camera=cam_calib["T_rig_views"],
            scale_down_images=scale_down_images,
            valid_radius=valid_radius,
            wh_multiple_of=wh_multiple_of,
        )
        if (
            image is not None
            and capture_image_time_ns is not None
            and cam_tw is not None
        ):
            images.append(image)
            times_ns.append(capture_image_time_ns)
            cam_tws.append(cam_tw)
            frame_ids.append(i)

    images = torch.stack(images)
    # Long to hold timestamp in ns to not lose accuracy
    times_ns = torch.LongTensor(times_ns)
    cam_tws = torch.stack(cam_tws)
    frame_ids = torch.LongTensor(frame_ids)
    return images, times_ns, cam_tws, frame_ids


def load_global_points_csv(
    path: str,
    max_inv_depth_std: float = 0.001,
    min_observations: int = 5,
):
    print(f"loading global points from {path}")
    uid_to_p3 = {}
    uid_to_inv_dist_std = {}
    uid_to_dist_std = {}
    if path.split(".")[-1] == "gz" or "maps/maps_v1" in path:
        compression = "gzip"
    else:
        compression = None

    cache_path = path + ".pickle.gz"
    if not os.path.exists(cache_path):
        with fsspec.open(path, "rb") as f:
            csv = pd.read_csv(f, compression=compression)
            # filter by inverse distance std
            csv = csv[csv.inv_dist_std < max_inv_depth_std]
            if "num_observations" in csv.columns:
                csv = csv[csv.num_observations > min_observations]
            print(csv.columns)
            # select points and uids and return mapping
            uid_pts = csv[
                ["uid", "inv_dist_std", "dist_std", "px_world", "py_world", "pz_world"]
            ]

            for row in tqdm.tqdm(uid_pts.values):
                uid = int(row[0])
                inv_dist_std = float(row[1])
                dist_std = float(row[2])
                p3 = row[3:]
                uid_to_p3[uid] = p3
                uid_to_inv_dist_std[uid] = inv_dist_std
                uid_to_dist_std[uid] = dist_std

        try:
            # cache points
            with gzip.open(cache_path, "wb") as f:
                pickle.dump(uid_to_p3, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(uid_to_inv_dist_std, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(uid_to_dist_std, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Cached global points to {cache_path}")
        except:
            print("Failed to cache the semidense points, like a write permission issue")
    else:
        # load from the cached file
        with gzip.open(cache_path, "rb") as f:
            uid_to_p3 = pickle.load(f)
            uid_to_inv_dist_std = pickle.load(f)
            uid_to_dist_std = pickle.load(f)
        print(f"Loaded global points from cached file {cache_path}")

    uid_to_p3 = {uid: torch.from_numpy(p3) for uid, p3 in uid_to_p3.items()}
    return uid_to_p3, uid_to_inv_dist_std, uid_to_dist_std


def load_semidense_observations(path: str):
    print(f"loading semidense observations from {path}")
    time_to_uids = defaultdict(list)
    uid_to_times = defaultdict(list)
    if path.split(".")[-1] == "gz" or "maps/maps_v1" in path:
        compression = "gzip"
    else:
        compression = None

    cache_path = path + ".pickle.gz"
    if not os.path.exists(cache_path):
        with fsspec.open(path, "rb") as f:
            csv = pd.read_csv(f, compression=compression)
            csv = csv[["uid", "frame_tracking_timestamp_us"]]
            for row in tqdm.tqdm(csv.values):
                uid = int(row[0])
                time_ns = int(row[1]) * 1000
                time_to_uids[time_ns].append(uid)
                uid_to_times[uid].append(time_ns)

        try:
            with gzip.open(cache_path, "wb") as f:
                pickle.dump(time_to_uids, f, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(uid_to_times, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached semidense observations to {cache_path}")
        except:
            print(
                "Failed to cache the semidense observations, like a write permission issue"
            )
    else:
        with gzip.open(cache_path, "rb") as f:
            time_to_uids = pickle.load(f)
            uid_to_times = pickle.load(f)
        print(f"Loaded semidense observations from cached file {cache_path}")

    return time_to_uids, uid_to_times
