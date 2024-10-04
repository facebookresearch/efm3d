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

import colorsys
from typing import Dict, Literal

import cv2
import numpy as np
import torch
from efm3d.aria import CameraTW, ObbTW, PoseTW
from efm3d.utils.image import put_text, torch2cv2


AXIS_COLORS_RGB = {
    0: (255, 0, 0),  # red
    3: (0, 255, 0),  # green
    8: (0, 0, 255),  # blue
}  # use RGB for xyz axes respectively


def get_colors(num_colors: int, scale_to_255: bool = False):
    assert num_colors > 0, f"Number of colors {num_colors} has to be positive."
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Spread out the colors in the hue space
        saturation = 1.0  # Use maximum saturation for bright colors
        value = 1.0  # Use maximum value for bright colors
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        if scale_to_255:
            colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        else:
            colors.append((rgb[0], rgb[1], rgb[2]))
    return colors


# RGB values in [0, 1] used in Static Structure Index
SSI_SEM_COLORS = {
    "floor": (1, 0.75, 0.75),
    "mirror": (0.5, 0.5, 0.5),
    "ceiling": (1, 1, 0.75),
    "chair": (0.2, 0.6, 1),
    "bench": (0.2, 0.6, 1),
    "ottoman": (0.2, 0.6, 1),
    "table": (1, 1, 0),
    "desk": (1, 1, 0),
    "storage": (0.7, 0.4, 0.05),
    "plant": (0, 1, 0),
    "plant_or_flower_pot": (0, 1, 0),
    "vase": (0, 1, 0),
    "screen": (1, 0, 0),
    "wallart": (0.6, 0.3, 0.95),
    "picture_frame_or_painting": (0.6, 0.3, 0.95),
    "bed": (0.55, 0.9, 0),
    # "couch": (0, 1, 1),   # SSI color
    "couch": (0.1, 0.5, 0.1),  # dark green
    # "sofa": (0, 1, 1),    # SSI color
    "sofa": (0.1, 0.5, 0.1),  # dark green
    "wall": (1, 1, 1),
    "lamp": (1, 0.8, 0.25),
    "door": (0.95, 0.25, 0.85),
    "window": (0.5, 1, 1),
    "unknown": (0.4, 0.4, 0.8),
    "other": (0.6, 0.6, 0.6),
    # hard code 'floor_mat' to dark red
    "floor_mat": (0.8, 0.15, 0.15),  # dark red
}


def get_colors_from_sem_map(
    sem_ids_to_names: Dict[int, str],
    scale_to_255: bool = True,
    match_with_ssi: bool = True,
):
    """
    sem_ids_to_names: taxonomy map from semantic id to semantic name.
    scale_to_255: whether to scale the colors to [0, 255].
    match_with_ssi: whether to match the colors with the Static Structure Index taxonomy for
    the overlapped classes.
    """

    if len(sem_ids_to_names) == 0:
        num_sem_ids = 1
    else:
        num_sem_ids = max(sem_ids_to_names.keys()) + 1
    colors = get_colors(num_sem_ids, scale_to_255=scale_to_255)

    if match_with_ssi:
        for sem_id, sem_name in sem_ids_to_names.items():
            sn = sem_name.lower()
            if sn in SSI_SEM_COLORS:
                clr = SSI_SEM_COLORS[sn]
                if scale_to_255:
                    clr2 = (
                        int(round(clr[0] * 255)),
                        int(round(clr[1] * 255)),
                        int(round(clr[2] * 255)),
                    )
                else:
                    clr2 = clr
                colors[sem_id] = clr2

    return colors


def draw_bb2s(
    viz,
    bb2s,
    line_type=cv2.LINE_AA,
    bb2s_center=None,
    labels=None,
    rotate_text=True,
    color=None,
    text_size=0.6,
):
    """
    Args:
        viz: numpy array image
        bb2s: a list of bounding boxes as numpy array Nx 4 where (x_min, x_max, y_min, y_max) per row
        color: either a 3-tuple/list or a list 3-tuples, or an np.array shaped Nx3
    """
    height = viz.shape[0]
    if height > 320:
        thickness = 2
    else:
        thickness = 1

    if color is None:
        color = (255, 100, 100)  # brighter red

    if bb2s.shape[0] == 0:
        return viz

    def _draw_bb2_line(img, p1, p2, clr):
        cv2.line(img, p1, p2, clr, thickness, lineType=line_type)

    if isinstance(color[0], (list, tuple, np.ndarray)):
        assert len(color) == len(
            bb2s
        ), "need either single color or same # of colors as bb2s"
        if isinstance(color[0], np.ndarray):
            colors = [clr.tolist() for clr in color]
        else:
            colors = color
    elif isinstance(color[0], (int, float)):
        colors = [color for _ in range(len(bb2s))]
    else:
        raise TypeError("Unknown type for 'color' argument of draw_bb2s()")

    for i, (bb2, clr) in enumerate(zip(bb2s, colors)):
        x_min, y_min = int(round(bb2[0].item())), int(round(bb2[2].item()))  # min pt
        x_max, y_max = int(round(bb2[1].item())), int(round(bb2[3].item()))  # max pt
        # if x_min < 0 or y_min < 0:
        #    print("WARNING line point outside image")
        _draw_bb2_line(viz, (x_min, y_min), (x_min, y_max), clr)
        _draw_bb2_line(viz, (x_min, y_max), (x_max, y_max), clr)
        _draw_bb2_line(viz, (x_max, y_max), (x_max, y_min), clr)
        _draw_bb2_line(viz, (x_max, y_min), (x_min, y_min), clr)
        if bb2s_center is not None:
            cx = int(round(float(bb2s_center[i, 0])))
            cy = int(round(float(bb2s_center[i, 1])))
            cv2.circle(viz, (cx, cy), 1, clr, 1, lineType=line_type)
        if labels is not None:
            text = labels[i]
            x = int(round((x_min + x_max) / 2.0))
            y = int(round((y_min + y_max) / 2.0))

            if rotate_text:
                viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
                center_rot90 = (height - y, x)
                x, y = center_rot90
            ((txt_w, txt_h), _) = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, text_size, 1
            )
            x = x - int(round(txt_w / 4))
            y = y + int(round(txt_h / 4))
            put_text(viz, text, scale=text_size, font_pt=(x, y))
            if rotate_text:
                viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return viz


def draw_bb3_lines(
    viz,
    T_world_cam: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_cosy: bool,
    T: int,
    line_type=cv2.LINE_AA,
    colors=None,
    thickness=1,
):
    bb3corners_world = obbs.T_world_object * obbs.bb3edge_pts_object(T)
    bb3corners_cam = T_world_cam.inverse() * bb3corners_world
    B = bb3corners_cam.shape[0]
    pt3s_cam = bb3corners_cam.view(B, -1, 3)
    pt2s, valids = cam.project(pt3s_cam)
    sem_ids = obbs.sem_id.int()
    # reshape to lines each composed of T segments
    pt2s = pt2s.round().int().view(B * 12, T, 2)
    valids = valids.view(B * 12, T)
    for line in range(pt2s.shape[0]):
        line_id = line % 12
        obb_id = line // 12
        sem_id = sem_ids[obb_id]
        # if colors is not None and sem_id >= len(colors):
        #     print("warning sem_id too big", sem_id, len(colors))
        if colors is None or sem_id >= len(colors):
            color = (255, 255, 255)
        else:
            color = colors[sem_id]
        for i in range(T - 1):
            j = i + 1
            if valids[line, i] and valids[line, j]:
                # check if we should color this line in a special way
                if draw_cosy and line_id in AXIS_COLORS_RGB:
                    color = AXIS_COLORS_RGB[line_id]
                pt1 = (
                    int(round(float(pt2s[line, i, 0]))),
                    int(round(float(pt2s[line, i, 1]))),
                )
                pt2 = (
                    int(round(float(pt2s[line, j, 0]))),
                    int(round(float(pt2s[line, j, 1]))),
                )
                cv2.line(
                    viz,
                    pt1,
                    pt2,
                    color,
                    thickness,
                    lineType=line_type,
                )


def draw_bb3s(
    viz,
    T_world_rig: PoseTW,
    cam: CameraTW,
    obbs: ObbTW,
    draw_bb3_center=False,
    draw_bb3=True,
    draw_label=False,
    draw_cosy=True,
    draw_score=True,
    render_obb_corner_steps=10,
    line_type=cv2.LINE_AA,
    sem_id_to_name_mapping: Dict[int, str] = None,
    rotate_label=True,
    colors=None,
    white_backing_line=True,
    draw_inst_id=False,
):
    # Get pose of camera.
    T_world_cam = T_world_rig.float() @ cam.T_camera_rig.inverse()
    # Project the 3D BB center into the image.
    if draw_bb3:
        # auto set the thickness of the bb3 lines
        thickness = 1

        # draw white background lines
        if white_backing_line:
            draw_bb3_lines(
                viz,
                T_world_cam,
                cam,
                obbs,
                draw_cosy=draw_cosy,
                T=render_obb_corner_steps,
                line_type=cv2.LINE_AA,
                colors=None,
                thickness=thickness + 1,
            )
        # draw semantic colors
        draw_bb3_lines(
            viz,
            T_world_cam,
            cam,
            obbs,
            draw_cosy=draw_cosy,
            T=render_obb_corner_steps,
            line_type=cv2.LINE_AA,
            colors=colors,
            thickness=thickness,
        )

    if draw_label or draw_bb3_center:
        bb3center_cam = T_world_cam.inverse() * obbs.bb3_center_world
        bb2center_im, valids = cam.unsqueeze(0).project(bb3center_cam.unsqueeze(0))
        bb2center_im, valids = bb2center_im.squeeze(0), valids.squeeze(0)
        for idx, (pt2, valid) in enumerate(zip(bb2center_im, valids)):
            if valid:
                center = (int(pt2[0]), int(pt2[1]))
                if draw_bb3_center:
                    cv2.circle(viz, center, 3, (255, 0, 0), 1, lineType=line_type)
                if draw_label:
                    height = viz.shape[0]
                    sem_id = int(obbs.sem_id.squeeze(-1)[idx])
                    if sem_id_to_name_mapping and sem_id in sem_id_to_name_mapping:
                        text = sem_id_to_name_mapping[sem_id]
                    else:
                        # display sem_id if no mapping is provided.
                        text = str(sem_id)
                    if draw_inst_id:
                        inst_id = int(obbs.inst_id.squeeze(-1)[idx])
                        text = f"{inst_id}: {text}"
                    # rot 90 degree before drawing the text
                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
                        center_rot90 = (height - center[1], center[0])
                        x, y = center_rot90
                    else:
                        x, y = center
                    ((txt_w, txt_h), _) = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1
                    )
                    x = x - txt_w // 4
                    y = y + txt_h // 4

                    # Show text on top of the 3d boxes
                    bb2_ymin = obbs.bb2_rgb[idx][2]
                    bb2_ymax = obbs.bb2_rgb[idx][3]
                    up = int((bb2_ymax - bb2_ymin) / 2.0)
                    if y - up > 0:
                        put_text(viz, text, scale=0.8, font_pt=(x, y - up))
                        if draw_score and obbs.prob is not None:
                            score = float(obbs.prob.squeeze(-1)[idx])
                            score_text = f"{score:.2f}"
                            score_pos = (x, y + int(txt_h + 0.5) - up)
                            put_text(
                                viz,
                                score_text,
                                scale=0.5,
                                font_pt=score_pos,
                                color=(200, 200, 200),
                            )

                    if rotate_label:
                        viz = cv2.rotate(viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return viz


def draw_obbs_image(
    img: torch.Tensor,
    obbs_padded: ObbTW,
    T_world_rig: PoseTW = None,
    cam: CameraTW = None,
    aria_cam_id: Literal[0, 1, 2] = 0,
    draw_bb2=False,
    draw_bb3=True,
    draw_bb3_center=False,
    draw_label=False,
    draw_cosy=True,
    draw_score=False,
    render_obb_corner_steps=10,
    post_rotate_viz=True,  # whether to rotate the image 90 degrees before (pre) or after (post) rendering 3d bbs; only for debugging. resulting image should be the same!
    rgb2bgr=True,
    rotate_viz=True,
    background_sem_id: int = None,
    prob_threshold: float = 0.5,
    sem_id_to_name_mapping: Dict[int, str] = None,
    draw_label_2d: bool = False,  # Draw label on 2D viz also.
    white_backing_line: bool = True,
    draw_inst_id: bool = False,
    draw_conic: bool = False,
):
    assert img.dim() == 3, f"image input must be 3D tensor {img.shape}"
    assert (
        obbs_padded.dim() == 2
    ), f"assuming one set of obbs per frame {obbs_padded.shape}"

    viz = torch2cv2(img, rotate=False, ensure_rgb=True, rgb2bgr=rgb2bgr)
    if not post_rotate_viz and rotate_viz:
        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)

    # get valid obbs
    obbs = obbs_padded.remove_padding()
    if obbs.shape[0] == 0:  # Handle no valid OBBs.
        if post_rotate_viz and rotate_viz:
            viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
        return viz

    # filter out low probability obbs
    good = obbs.prob >= prob_threshold
    colors = None
    if sem_id_to_name_mapping is not None:
        colors = get_colors_from_sem_map(sem_id_to_name_mapping)
    if obbs.shape[0] > 0 and good.any():
        obbs = obbs[good.squeeze(-1), :]
        # if we have background id given, then filter out background obbs
        if background_sem_id is not None:
            background = obbs.sem_id == background_sem_id
            obbs = obbs[~background.squeeze(-1), :]
        if obbs.shape[0] > 0:
            # Draw 2D bounding box.
            if not draw_label_2d or sem_id_to_name_mapping is None:
                labels = None
            else:
                sem_id = obbs.sem_id.squeeze(-1)
                labels = [sem_id_to_name_mapping[int(si)] for si in sem_id]
                if draw_inst_id:
                    inst_ids = obbs.inst_id.squeeze(-1)
                    labels = [f"{inst}:{n}" for inst, n in zip(inst_ids, labels)]
            if draw_bb2:
                viz = draw_bb2s(
                    viz,
                    obbs.bb2(aria_cam_id),
                    bb2s_center=obbs.get_bb2_centers(aria_cam_id),
                    labels=labels,
                )

            if draw_conic and cam and T_world_rig:
                pass

            # Draw 3D bounding box (requires poses from VIO).
            if cam and T_world_rig and (draw_bb3 or draw_bb3_center):
                if not post_rotate_viz and rotate_viz:
                    cam = cam.rotate_90_cw()
                viz = draw_bb3s(
                    viz,
                    T_world_rig,
                    cam,
                    obbs,
                    draw_bb3_center=draw_bb3_center,
                    draw_bb3=draw_bb3,
                    draw_label=draw_label,
                    draw_cosy=draw_cosy,
                    draw_score=draw_score,
                    render_obb_corner_steps=render_obb_corner_steps,
                    sem_id_to_name_mapping=sem_id_to_name_mapping,
                    rotate_label=rotate_viz,
                    colors=colors,
                    white_backing_line=white_backing_line,
                    draw_inst_id=draw_inst_id,
                )

    # Rotate everything before displaying.
    if post_rotate_viz and rotate_viz:
        viz = cv2.rotate(viz, cv2.ROTATE_90_CLOCKWISE)
    return viz


def draw_obbs_snippet(
    imgs: torch.Tensor,
    obbs_padded: ObbTW,
    Ts_world_rig: PoseTW = None,
    cams: CameraTW = None,
    aria_cam_id: Literal[0, 1, 2] = 0,
    draw_bb2=True,
    draw_bb3=True,
    draw_bb3_center=False,
    render_obb_corner_steps=10,
    post_rotate_viz=True,  # whether to rotate the image 90 degrees before (pre) or after (post) rendering 3d bbs; only for debugging. resulting image should be the same!
    rgb2bgr=True,
    rotate_viz=True,
    background_sem_id: int = None,
    prob_threshold: float = 0.5,
    sem_id_to_name_mapping: Dict[int, str] = None,
    draw_label: bool = False,
    draw_label_2d: bool = False,  # Draw label on 2D viz also.
    white_backing_line: bool = True,
    draw_cosy: bool = True,
    draw_score: bool = False,
    draw_inst_id: bool = False,
    draw_conic: bool = False,
):
    assert imgs.dim() == 4, f"snippet input must be 4D tensor {imgs.shape}"
    T = imgs.shape[0]
    viz = []
    for t in range(T):
        if obbs_padded.dim() == 2:
            cur_obbs_padded = obbs_padded
        elif obbs_padded.dim() == 3:
            cur_obbs_padded = obbs_padded[t]
        else:
            raise ValueError(
                f"obbs_padded must have 2 or 3 dimensions {obbs_padded.shape}"
            )

        viz.append(
            draw_obbs_image(
                img=imgs[t],
                obbs_padded=cur_obbs_padded,
                T_world_rig=Ts_world_rig[t],
                cam=cams[t],
                aria_cam_id=aria_cam_id,
                draw_bb2=draw_bb2,
                draw_bb3=draw_bb3,
                draw_bb3_center=draw_bb3_center,
                render_obb_corner_steps=render_obb_corner_steps,
                post_rotate_viz=post_rotate_viz,
                rgb2bgr=rgb2bgr,
                rotate_viz=rotate_viz,
                background_sem_id=background_sem_id,
                prob_threshold=prob_threshold,
                sem_id_to_name_mapping=sem_id_to_name_mapping,
                draw_label=draw_label,
                draw_label_2d=draw_label_2d,
                white_backing_line=white_backing_line,
                draw_cosy=draw_cosy,
                draw_score=draw_score,
                draw_inst_id=draw_inst_id,
                draw_conic=draw_conic,
            )
        )
    return viz


def discretize_values(values: torch.Tensor, precision: int):
    """
    Discretize the values of an input tensor with a certain precision. The lower the precision, the coarser the output.
    The function is added to better rendering a dense pointcloud.
    """
    d_values = (values * precision).int()
    d_values = (torch.unique(d_values, dim=0) / precision).float()
    return d_values
