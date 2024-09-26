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

import os
from bisect import bisect_left
from typing import Optional

import cv2
import numpy as np
import torch
import tqdm
import trimesh
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_IMG,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_IMG_TIME_NS,
    ARIA_MESH_FACES,
    ARIA_MESH_VERT_NORMS_W,
    ARIA_MESH_VERTS_W,
    ARIA_OBB_PADDED,
    ARIA_OBB_PRED_VIZ,
    ARIA_OBB_TRACKED,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.aria.obb import ObbTW
from efm3d.inference.fuse import VolumetricFusion
from efm3d.utils.image import put_text, smart_resize, torch2cv2
from efm3d.utils.obb_csv_writer import ObbCsvReader
from efm3d.utils.render import draw_obbs_snippet
from efm3d.utils.viz import draw_snippet_scene_3d, SceneView


VIZ_RGB = "RGB/GT"
VIZ_SLAM = "SLAM"
VIZ_PRED_OBB = "Snippet Prediction"
VIZ_TRACKED_OBB = "Tracked Prediction"
VIZ_GT_OBB = "Ground Truth"


def find_nearest(array, value):
    """Find the index of the nearest value in an array."""
    idx = bisect_left(array, value)
    if idx == len(array):
        return idx - 1
    if idx == 0:
        return 0
    before = array[idx - 1]
    after = array[idx]
    if after - value < value - before:
        return idx
    return idx - 1


def fill_obbs_to_snippet(obbs, rgb_ts, T_ws):
    obbs_out = []
    obbs_ts = sorted(obbs.keys())
    for ts in rgb_ts:
        if ts in obbs:
            obbs_out.append(obbs[ts].add_padding(128))
        else:
            # find the nearest timestamp within 1s
            nidx = find_nearest(obbs_ts, ts)
            if abs(obbs_ts[nidx] - ts) / 1e9 < 1:
                obbs_out.append(obbs[obbs_ts[nidx]].add_padding(128))
            else:
                obbs_out.append(ObbTW().add_padding(128))
    obbs_w = torch.stack(obbs_out, dim=0)
    obbs_s = obbs_w.transform(T_ws.inverse())
    return obbs_s


def compose_views(view_dict, keys, vertical=True):
    """stack snippet images into a single image, vertical or horizontal"""
    keys = [k for k in keys if k in view_dict]
    if len(keys) == 0:
        return None
    if len(keys) == 1:
        return view_dict[keys[0]]

    output_imgs = []
    T = len(view_dict[keys[0]])
    for i in range(T):
        img_list = [view_dict[key][i] for key in keys]
        axis = 0 if vertical else 1
        combine_img = np.concatenate(img_list, axis=axis)
        output_imgs.append(combine_img)
    return output_imgs


def draw_scene_with_mesh_and_obbs(
    snippet,
    w,
    h,
    scene,
    snip_obbs=None,
    tracked_obbs=None,
    gt_obbs=None,
    mesh=None,
    sem_ids_to_names=None,
):
    """
    Draw 3d scene view of a snippet, with optionally obbs and mesh.
    """
    # put pred obbs into the snippet
    rgb_ts = snippet[ARIA_IMG_TIME_NS[0]]
    rgb_ts = [ts.item() for ts in rgb_ts]
    T_ws = snippet[ARIA_SNIPPET_T_WORLD_SNIPPET]

    if snip_obbs is not None:
        snippet[ARIA_OBB_PRED_VIZ] = fill_obbs_to_snippet(snip_obbs, rgb_ts, T_ws)
    if tracked_obbs is not None:
        snippet[ARIA_OBB_TRACKED] = fill_obbs_to_snippet(tracked_obbs, rgb_ts, T_ws)
    if gt_obbs is not None:
        snippet[ARIA_OBB_PADDED] = fill_obbs_to_snippet(gt_obbs, rgb_ts, T_ws)
    if mesh is not None and mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0:
        snippet[ARIA_MESH_VERTS_W] = torch.tensor(mesh.vertices)
        snippet[ARIA_MESH_FACES] = torch.tensor(mesh.faces)
        # normals for pred should be minus due to marching cube
        snippet[ARIA_MESH_VERT_NORMS_W] = -torch.tensor(mesh.vertex_normals)

    scene_imgs = draw_snippet_scene_3d(
        snippet, sem_ids_to_names=sem_ids_to_names, width=w, height=h, scene=scene
    )
    return scene_imgs


def render_views(snippet, h, w, pred_sem_ids_to_names, gt_sem_ids_to_names):
    Ts_sr = snippet[ARIA_IMG_T_SNIPPET_RIG[0]]
    cams = snippet[ARIA_CALIB[0]]
    T_ws = snippet[ARIA_SNIPPET_T_WORLD_SNIPPET]
    Ts_wr = T_ws @ Ts_sr
    rgb_ts = snippet[ARIA_IMG_TIME_NS[0]]
    time_s = [f"{ts.item()*1e-9:.02f}s" for ts in rgb_ts]

    imgs = {}
    # RGB and SLAM
    rgb_imgs = snippet[ARIA_IMG[0]].clone().numpy()
    rgb_imgs = [
        torch2cv2(im, rotate=True, ensure_rgb=True, rgb2bgr=False) for im in rgb_imgs
    ]
    imgs[VIZ_RGB] = rgb_imgs

    if ARIA_IMG[1] in snippet and ARIA_IMG[2] in snippet:
        slaml_imgs = snippet[ARIA_IMG[1]].clone().numpy()
        slamr_imgs = snippet[ARIA_IMG[2]].clone().numpy()
        slaml_imgs = [
            torch2cv2(im, rotate=True, ensure_rgb=True, rgb2bgr=False)
            for im in slaml_imgs
        ]
        slamr_imgs = [
            torch2cv2(im, rotate=True, ensure_rgb=True, rgb2bgr=False)
            for im in slamr_imgs
        ]
        imgs[VIZ_SLAM] = []
        for iml, imr in zip(slaml_imgs, slamr_imgs):
            imgs[VIZ_SLAM].append(np.concatenate([iml, imr], axis=1))

    if ARIA_OBB_PRED_VIZ in snippet:
        imgs[VIZ_PRED_OBB] = draw_obbs_snippet(
            snippet[ARIA_IMG[0]].clone(),
            snippet[ARIA_OBB_PRED_VIZ].transform(T_ws),
            Ts_wr,
            cams,
            rgb2bgr=False,
            draw_cosy=False,
            white_backing_line=False,
            draw_bb2=False,
            sem_id_to_name_mapping=pred_sem_ids_to_names,
            draw_label=True,
            draw_score=True,
            prob_threshold=0.001,  # keep this very low, obbs are already thresholded.
        )

    if ARIA_OBB_TRACKED in snippet:
        imgs[VIZ_TRACKED_OBB] = draw_obbs_snippet(
            snippet[ARIA_IMG[0]].clone(),
            snippet[ARIA_OBB_TRACKED].transform(T_ws),
            Ts_wr,
            cams,
            rgb2bgr=False,
            draw_cosy=False,
            white_backing_line=False,
            draw_bb2=False,
            sem_id_to_name_mapping=pred_sem_ids_to_names,
            draw_label=True,
            draw_score=True,
            prob_threshold=0.001,  # keep this very low, obbs are already thresholded.
        )

    if ARIA_OBB_PADDED in snippet:
        # if gt obb (VIZ_GT_OBB) is present, overlay it on top of the RGB view
        imgs[VIZ_RGB] = draw_obbs_snippet(
            snippet[ARIA_IMG[0]].clone(),
            snippet[ARIA_OBB_PADDED].transform(T_ws),
            Ts_wr,
            cams,
            rgb2bgr=False,
            draw_cosy=False,
            white_backing_line=False,
            draw_bb2=False,
            sem_id_to_name_mapping=gt_sem_ids_to_names,
            draw_label=True,
            draw_inst_id=True,
            draw_score=True,
        )

    # add text to the images
    for text, grid_imgs in imgs.items():
        for i, img in enumerate(grid_imgs):
            img = smart_resize(img, h, w, pad_image=True)
            img = put_text(img, text)
            imgs[text][i] = put_text(img, time_s[i], line=-1)
    return imgs


def generate_video(
    streamer,
    output_dir,
    fps=10,
    vol_fusion: Optional[VolumetricFusion] = None,
    stride_s: float = 0.1,
):
    """
    streamer: AriaStreamer object, assuming input snippets are 1s at 10 FPS.
    output_dir: the output folder for the video, will also load obbs and per_snip artifacts from the same folder
    fps: the output video fps
    vol_fusion: A volumetric fusion class instance. If not None, will use it to show the incremental mesh, updated as 1s frame rate.
    """

    # read snippet obbs
    snip_obbs_csv = os.path.join(output_dir, "snippet_obbs.csv")
    snip_obbs = None
    sem_ids_to_names = None
    if os.path.exists(snip_obbs_csv):
        snip_obb_reader = ObbCsvReader(snip_obbs_csv)
        snip_obbs = snip_obb_reader.obbs
        sem_ids_to_names = snip_obb_reader.sem_ids_to_names

    # read tracked obbs
    tracked_obbs_csv = os.path.join(output_dir, "tracked_obbs.csv")
    tracked_obbs = None
    if os.path.exists(tracked_obbs_csv):
        tracked_obb_reader = ObbCsvReader(tracked_obbs_csv)
        tracked_obbs = tracked_obb_reader.obbs

    # read GT obbs
    gt_obbs_csv = os.path.join(output_dir, "gt_obbs.csv")
    gt_obbs = None
    gt_sem_ids_to_names = None
    if os.path.exists(gt_obbs_csv):
        gt_obb_reader = ObbCsvReader(gt_obbs_csv)
        gt_obbs = gt_obb_reader.obbs
        gt_sem_ids_to_names = gt_obb_reader.sem_ids_to_names

    # read fused mesh
    fused_mesh = os.path.join(output_dir, "fused_mesh.ply")
    pred_mesh = None
    if os.path.exists(fused_mesh):
        pred_mesh = trimesh.load(fused_mesh)

    # write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, "video.mp4")

    # two columns for 2d views (RGB+SLAM, output), 1 column for 3d scene
    gW, gH = 360, 360  # 2d grid size
    sH = 2 * gH
    sW = sH
    W = sW + 2 * gW
    H = sH

    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    scene = SceneView(width=sW, height=sH)
    num_snip_per_s = int(1.0 / stride_s)
    for idx, snippet in tqdm.tqdm(enumerate(streamer), total=len(streamer)):
        # show incremental fusion if vol_fusion is given
        if vol_fusion is not None:
            for i in range(num_snip_per_s):
                vol_fusion.run_step(idx * num_snip_per_s + i)
            pred_mesh = vol_fusion.get_trimesh()

        scene_imgs = draw_scene_with_mesh_and_obbs(
            snippet,
            w=sW,
            h=sH,
            scene=scene,
            snip_obbs=snip_obbs,
            tracked_obbs=tracked_obbs,
            gt_obbs=gt_obbs,
            mesh=pred_mesh,
            sem_ids_to_names=sem_ids_to_names,
        )
        view_imgs = render_views(
            snippet,
            gH,
            gW,
            pred_sem_ids_to_names=sem_ids_to_names,
            gt_sem_ids_to_names=gt_sem_ids_to_names,
        )

        input_col = compose_views(view_imgs, [VIZ_RGB, VIZ_SLAM])
        output_col = compose_views(view_imgs, [VIZ_PRED_OBB, VIZ_TRACKED_OBB])

        for i, scene_img in enumerate(scene_imgs):
            final_img = np.zeros((H, W, 3), dtype=np.uint8)  # black background
            h, w = input_col[i].shape[:2]
            final_img[:h, :w] = input_col[i]
            final_img[:sH, gW : gW + sW, :] = scene_img
            if output_col is not None:
                h, w = output_col[i].shape[:2]
                final_img[:h, gW + sW : gW + sW + w] = output_col[i]

            out.write(final_img[:, :, ::-1])  # convert rgb to bgr before writing
    out.release()
    return output_path
