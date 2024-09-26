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

# High level organization of the constants:
# - */time_ns is timestamp with respect to the aria clock in nanoseconds stored as torch.long()
# - */snippet_time_s is the timestamp with respect to the start of the snippet in seconds stored as torch.float32()
# - */t_A_B is a pose transformation from coordinate system B to A
# - path-like key strings designate hierarchical relationships of data. I.e.
#   rgb/img/... is all data relating to the rgb image information. rgb/calib/...
#   is all about the calibration data. And all rgb/... is data relating to the
#   rgb video stream.

# ---------------------------------------------------------------------
# sequence level information
# ---------------------------------------------------------------------
ARIA_SEQ_ID = "sequence/id"
# start of the sequence in ns relative to global Aria timestamp
ARIA_SEQ_TIME_NS = "sequence/time_ns"

# ---------------------------------------------------------------------
# snippet level information
# ---------------------------------------------------------------------
ARIA_SNIPPET_ID = "snippet/id_in_sequence"
ARIA_SNIPPET_LENGTH_S = "snippet/length_s"
# start of sequence in ns relative to global Aria timestamp (sometimes unix 0)
ARIA_SNIPPET_TIME_NS = "snippet/time_ns"
# offset of snippet coordinate system to sequence coordinate system
ARIA_SNIPPET_T_WORLD_SNIPPET = "snippet/t_world_snippet"
# Ratio of where in the snippet is the origin of cosy relative to the
# snippet length. E.g. 0.5 for a 10 sec snippet would mean that 5 sec is origin,
# was previously known as "frame_selection" in LocalCosyPreprocessor.
ARIA_SNIPPET_ORIGIN_RATIO = "snippet/origin_ratio"

# ---------------------------------------------------------------------
# streamer playback time information
# ---------------------------------------------------------------------
ARIA_PLAY_TIME_NS = "play/time_ns"
ARIA_PLAY_SEQUENCE_TIME_S = "play/sequence_time_s"
ARIA_PLAY_SNIPPET_TIME_S = "play/snippet_time_s"
ARIA_PLAY_FREQUENCY_HZ = "play/hz"

# ---------------------------------------------------------------------
# aria video stream information
# ---------------------------------------------------------------------
# frame id in the sequence
ARIA_FRAME_ID = [
    "rgb/frame_id_in_sequence",
    "slaml/frame_id_in_sequence",
    "slamr/frame_id_in_sequence",
]
# timestamp within snippet
ARIA_IMG_SNIPPET_TIME_S = [
    "rgb/img/snippet_time_s",
    "slaml/img/snippet_time_s",
    "slamr/img/snippet_time_s",
]
# timestamp within sequence
ARIA_IMG_TIME_NS = [
    "rgb/img/time_ns",
    "slaml/img/time_ns",
    "slamr/img/time_ns",
]
# poses of the rig at the time of the respective frame capture
# T x 12
ARIA_IMG_T_SNIPPET_RIG = [
    "rgb/t_snippet_rig",
    "slaml/t_snippet_rig",
    "slamr/t_snippet_rig",
]
# image tensors
ARIA_IMG = ["rgb/img", "slaml/img", "slamr/img"]
ARIA_IMG_FREQUENCY_HZ = [
    "rgb/img/hz",
    "slaml/img/hz",
    "slamr/img/hz",
]

# ---------------------------------------------------------------------
# calibration information
# ---------------------------------------------------------------------
ARIA_CALIB = [
    "rgb/calib",
    "slaml/calib",
    "slamr/calib",
]
# timestamp within the snippet
ARIA_CALIB_SNIPPET_TIME_S = [
    "rgb/calib/snippet_time_s",
    "slaml/calib/snippet_time_s",
    "slamr/calib/snippet_time_s",
]
# timestamp within the sequence
ARIA_CALIB_TIME_NS = [
    "rgb/calib/time_ns",
    "slaml/calib/time_ns",
    "slamr/calib/time_ns",
]

# ---------------------------------------------------------------------
# pose information
# ---------------------------------------------------------------------
# pose timestamp within snippet
ARIA_POSE_SNIPPET_TIME_S = "pose/snippet_time_s"
# pose timestamp within sequence
ARIA_POSE_TIME_NS = "pose/time_ns"
# transformation from rig to snippet coordinate system
ARIA_POSE_T_SNIPPET_RIG = "pose/t_snippet_rig"
# transformation from rig to world coordinate system
ARIA_POSE_T_WORLD_RIG = "pose/t_world_rig"
# frequency of poses
ARIA_POSE_FREQUENCY_HZ = "pose/hz"

# ---------------------------------------------------------------------
# semidense points information
# ---------------------------------------------------------------------
ARIA_POINTS_WORLD = "points/p3s_world"
ARIA_POINTS_TIME_NS = "points/time_ns"
ARIA_POINTS_SNIPPET_TIME_S = "points/snippet_time_s"
ARIA_POINTS_FREQUENCY_HZ = "points/hz"
ARIA_POINTS_INV_DIST_STD = "points/inv_dist_std"
ARIA_POINTS_DIST_STD = "points/dist_std"

# ---------------------------------------------------------------------
# imu information
# ---------------------------------------------------------------------
ARIA_IMU = ["imur", "imul"]
ARIA_IMU_CHANNELS = [
    ["imur/lin_acc_ms2", "imur/rot_vel_rads"],
    ["imul/lin_acc_ms2", "imul/rot_vel_rads"],
]
ARIA_IMU_SNIPPET_TIME_S = ["imur/snippet_time_s", "imul/snippet_time_s"]
ARIA_IMU_TIME_NS = ["imur/time_ns", "imul/time_ns"]
ARIA_IMU_FACTORY_CALIB = ["imur/factory_calib", "imul/factory_calib"]
ARIA_IMU_FREQUENCY_HZ = ["imur/hz", "imul/hz"]

# ---------------------------------------------------------------------
# audio data
# ---------------------------------------------------------------------
ARIA_AUDIO = "audio"
# snippet time within snippet of audio sample
ARIA_AUDIO_SNIPPET_TIME_S = "audio/snippet_time_s"
# timestamp of audio sample in sequence
ARIA_AUDIO_TIME_NS = "audio/time_ns"
# frequency of audio sample
ARIA_AUDIO_FREQUENCY_HZ = "audio/hz"

# ---------------------------------------------------------------------
# OBB
# ---------------------------------------------------------------------
# padded ObbTW tensor for oriented object bounding boxes given in *snippet coordinate system*
ARIA_OBB_PADDED = "obbs/padded_snippet"
# mapping of semantic id of the obb to a string name
ARIA_OBB_SEM_ID_TO_NAME = "obbs/sem_id_to_name"
# snippet time within the sequence
ARIA_OBB_SNIPPET_TIME_S = "obbs/snippet_time_s"
# timestamp within the sequence
ARIA_OBB_TIME_NS = "obbs/time_ns"
# frequency of object detection information
ARIA_OBB_FREQUENCY_HZ = "obbs/hz"

# predicted ObbTW tensor for oriented object bounding boxes
ARIA_OBB_PRED = "obbs/pred"  # raw predictions from the networks.
ARIA_OBB_PRED_VIZ = "obbs/pred_viz"  # predictions for visualization (e.g. raw predictions filtered by some criteria.)
ARIA_OBB_PRED_SEM_ID_TO_NAME = "obbs/pred/sem_id_to_name"
ARIA_OBB_PRED_PROBS_FULL = "obbs/pred/probs_full"
ARIA_OBB_PRED_PROBS_FULL_VIZ = "obbs/pred/probs_ful_viz"
# tracked ObbTW tensor for oriented object bounding boxes
ARIA_OBB_TRACKED = "obbs/tracked"
ARIA_OBB_TRACKED_PROBS_FULL = "obbs/tracked/probs_full"
# tracked but not instantiated ObbTW tensor for oriented object bounding boxes
ARIA_OBB_UNINST = "obbs/uninst"

ARIA_OBB_BB2 = ["bb2s_rgb", "bb2s_slaml", "bb2s_slamr"]
ARIA_OBB_BB3 = "bb3s_object"

# ---------------------------------------------------------------------
# depth information
# ---------------------------------------------------------------------
# for depth images (z-depth) in meters
ARIA_DEPTH_M = ["rgb/depth_m", "slaml/depth_m", "slamr/depth_m"]
# for distance images (distance along ray) in meters
ARIA_DISTANCE_M = ["rgb/distance_m", "slaml/distance_m", "slamr/distance_m"]
ARIA_DEPTH_TIME_NS = [
    "rgb/depth/time_ns",
    "slaml/depth/time_ns",
    "slamr/depth/time_ns",
]
ARIA_DEPTH_SNIPPET_TIME_S = [
    "rgb/depth/snippet_time_s",
    "slaml/depth/snippet_time_s",
    "slamr/depth/snippet_time_s",
]

ARIA_DEPTH_M_PRED = ["rgb/pred/depth_m", "slaml/pred/depth_m", "slamr/pred/depth_m"]
# for distance images (distance along ray) in meters
ARIA_DISTANCE_M_PRED = [
    "rgb/pred/distance_m",
    "slaml/pred/distance_m",
    "slamr/pred/distance_m",
]

# ---------------------------------------------------------------------
# SDF information
# ---------------------------------------------------------------------
ARIA_SDF = "snippet/sdf/sdf"
ARIA_SDF_EXT = "snippet/sdf/extent"
ARIA_SDF_COSY_TIME_NS = "snippet/sdf/cosy_time_ns"
ARIA_SDF_MASK = "snippet/sdf/mask"
ARIA_SDF_T_WORLD_VOXEL = "snippet/sdf/T_world_voxel"

# ---------------------------------------------------------------------
# GT Mesh information
# ---------------------------------------------------------------------
ARIA_MESH_VERTS_W = "snippet/mesh/verts_w"
ARIA_MESH_FACES = "snippet/mesh/faces"
ARIA_MESH_VERT_NORMS_W = "snippet/mesh/v_norms_w"
ARIA_SCENE_MESH_VERTS_W = "scene/mesh/verts_w"
ARIA_SCENE_MESH_FACES = "scene/mesh/faces"
ARIA_SCENE_MESH_VERT_NORMS_W = "scene/mesh/v_norms_w"

# ---------------------------------------------------------------------
# Scene volume information (can be acquired from mesh or semidense points)
# --------------------------------------------------------------------
ARIA_MESH_VOL_MIN = "scene/mesh/vol_min"
ARIA_MESH_VOL_MAX = "scene/mesh/vol_max"
ARIA_POINTS_VOL_MIN = "scene/points/vol_min"
ARIA_POINTS_VOL_MAX = "scene/points/vol_max"

# ---------------------------------------------------------------------
# additional image constants
# ---------------------------------------------------------------------

# Fixed mapping of resolutions, tuple has three numbers: (RGB_HW, SLAM_W, SLAM_H)
RESOLUTION_MAP = {
    0: (1408, 640, 480),
    1: (704, 640, 480),
    2: (352, 320, 240),
    # 3: there is none
    4: (176, 160, 112),  # there is some cropping in SLAM image height
    5: (480, 640, 480),
    6: (336, 448, 336),  # match typical internet image FOV (assume 70 deg)
    7: (240, 320, 240),  # match typical internet pixels e.g. ImageNet
    8: (192, 256, 192),
    9: (144, 192, 144),
    # divisible by 14 for ViTs that use patch size 14
    10: (
        1400,
        560,
        420,
    ),  # similar to 0  560x420 instead of 616x462 so that we can also get half the resolution for equivalent to 7
    11: (700, 560, 420),  # similar to 1
    12: (420, 560, 420),  # similar to 5
    13: (210, 280, 210),  # similar to 7
}
# Fixed mapping of corresponding wh_multiple_of, for each resolution
WH_MULTIPLE_OF_MAP = {
    0: 16,
    1: 16,
    2: 16,
    # 3: there is none
    4: 16,
    5: 16,
    6: 16,
    7: 16,
    8: 16,
    9: 16,
    10: 14,
    11: 14,
    12: 14,
    13: 14,
}

# Helper constants for managing valid radius of the fisheye images, valid radius
# defines a circle from the center of projection where project/unproject is valid
RGB_RADIUS_FACTOR = 760.0 / 1408.0
SLAM_RADIUS_FACTOR = 320.0 / 640.0

ARIA_RGB_WIDTH_TO_RADIUS = {
    RESOLUTION_MAP[key][0]: RESOLUTION_MAP[key][0] * RGB_RADIUS_FACTOR
    for key in RESOLUTION_MAP
}
ARIA_SLAM_WIDTH_TO_RADIUS = {
    RESOLUTION_MAP[key][1]: RESOLUTION_MAP[key][1] * SLAM_RADIUS_FACTOR
    for key in RESOLUTION_MAP
}

ARIA_RGB_SCALE_TO_WH = {
    key: [RESOLUTION_MAP[key][0], RESOLUTION_MAP[key][0]] for key in RESOLUTION_MAP
}
ARIA_SLAM_SCALE_TO_WH = {
    key: [RESOLUTION_MAP[key][1], RESOLUTION_MAP[key][2]] for key in RESOLUTION_MAP
}

ARIA_IMG_MIN_LUX = 30.0
ARIA_IMG_MAX_LUX = 150000.0
ARIA_IMG_MAX_PERC_OVEREXPOSED = 0.02
ARIA_IMG_MAX_PERC_UNDEREXPOSED = 0.0001

# ---------------------------------------------------------------------
# EFM Constants
# ---------------------------------------------------------------------
ARIA_EFM_OUTPUT = "efm/output"

ARIA_CAM_INFO = {
    "name": ["rgb", "slaml", "slamr"],
    "stream_id": [0, 1, 2],
    "name_to_stream_id": {
        "rgb": 0,
        "slaml": 1,
        "slamr": 2,
    },
    "width_height": {
        "rgb": (1408, 1408),
        "slaml": (640, 480),
        "slamr": (640, 480),
    },
    # vrs id
    "id": ["214-1", "1201-1", "1201-2"],
    "id_to_name": {
        "214-1": "rgb",
        "1201-1": "slaml",
        "1201-2": "slamr",
    },
    # display names
    "display": [
        "RGB",
        "SLAM Left",
        "SLAM Right",
    ],
    # Physical position on glasses from left to right.
    "spatial_order": [1, 0, 2],
}
