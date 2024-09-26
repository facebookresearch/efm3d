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

from typing import Optional, Tuple, Union

import moderngl

import numpy as np
import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_CALIB_TIME_NS,
    ARIA_DISTANCE_M,
    ARIA_DISTANCE_M_PRED,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_MESH_FACES,
    ARIA_MESH_VERT_NORMS_W,
    ARIA_MESH_VERTS_W,
    ARIA_OBB_PADDED,
    ARIA_OBB_PRED_VIZ,
    ARIA_OBB_TRACKED,
    ARIA_OBB_UNINST,
    ARIA_POINTS_WORLD,
    ARIA_POSE_T_SNIPPET_RIG,
    ARIA_POSE_TIME_NS,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import BB3D_LINE_ORDERS, OBB_LINE_INDS, OBB_MESH_TRI_INDS, ObbTW
from efm3d.aria.pose import PoseTW
from efm3d.utils.common import sample_nearest
from efm3d.utils.depth import dist_im_to_point_cloud_im
from efm3d.utils.gravity import gravity_align_T_world_cam, GRAVITY_DIRECTION_VIO
from efm3d.utils.render import discretize_values, get_colors_from_sem_map
from PIL import Image
from torch.nn import functional as F

# mapping from edge ids to colors for visualizing the xyz axes
AXIS_COLORS_GL = {
    0: (1.0, 0.0, 0.0, 1.0),  # red
    3: (0.0, 1.0, 0.0, 1.0),  # green
    8: (0.0, 0.0, 1.0, 1.0),  # blue
}  # use RGB for xyz axes respectively


def render_points(pts, rgba, prog=None, ctx=None, point_size=1.0, scene=None):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if pts.shape[0] == 0:
        return
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx
    prog["global_color"].value = rgba
    prog["point_size"].value = point_size
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f", "in_vert")])
    vao.render(moderngl.POINTS)

    vao.release()
    vbo.release()


def render_cubes(centers, bb3_halfdiag, prog, ctx, rgb=None):
    cs = centers.reshape(-1, 3)
    offs = [
        torch.tensor([-1.0, -1.0, -1.0], device=cs.device),
        torch.tensor([1.0, -1.0, -1.0], device=cs.device),
        torch.tensor([1.0, 1.0, -1.0], device=cs.device),
        torch.tensor([-1.0, 1.0, -1.0], device=cs.device),
        torch.tensor([-1.0, -1.0, 1.0], device=cs.device),
        torch.tensor([1.0, -1.0, 1.0], device=cs.device),
        torch.tensor([1.0, 1.0, 1.0], device=cs.device),
        torch.tensor([-1.0, 1.0, 1.0], device=cs.device),
    ]

    offs = torch.stack(offs, dim=0)
    corners = (
        cs.unsqueeze(1) + (offs * bb3_halfdiag.unsqueeze(0)).unsqueeze(0)
    ).clone()
    tris = (
        torch.tensor(OBB_MESH_TRI_INDS, dtype=torch.int32, device=cs.device)
        .transpose(1, 0)
        .unsqueeze(0)
    )
    tris_offset = 8 * torch.arange(
        0, corners.shape[0], dtype=torch.int32, device=cs.device
    ).view(-1, 1, 1)
    tris = (tris + tris_offset).clone()
    normals = F.normalize((offs * bb3_halfdiag.unsqueeze(0)), 2.0, -1)
    normals = normals.unsqueeze(0).repeat(corners.shape[0], 1, 1).clone()

    # render_rgb_points(corners, normals, prog, ctx)
    if rgb is not None:
        render_rgb_tri_mesh(corners, normals, tris, rgb, prog, ctx)
    else:
        render_tri_mesh(corners, normals, tris, prog, ctx)


def render_tri_mesh(pts, normals, tris, prog, ctx):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if isinstance(tris, torch.Tensor):
        tris = tris.detach().cpu().numpy()
    if isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().float().numpy()
    if pts.shape[0] == 0:
        return
    prog["point_size"].value = 1.0
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    nbo = ctx.buffer(normals.astype("float32").tobytes())
    ibo = ctx.buffer(tris.astype("int32").tobytes())
    vao = ctx.vertex_array(
        prog, [(vbo, "3f", "in_vert"), (nbo, "3f", "in_normal")], ibo
    )
    vao.render(moderngl.TRIANGLES)

    vao.release()
    ibo.release()
    nbo.release()
    vbo.release()


def render_rgb_tri_mesh(pts, normals, tris, rgb, prog, ctx):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if isinstance(tris, torch.Tensor):
        tris = tris.detach().cpu().numpy()
    if isinstance(normals, torch.Tensor):
        normals = normals.detach().cpu().float().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().float().numpy()
    if pts.shape[0] == 0:
        return
    prog["point_size"].value = 1.0
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    nbo = ctx.buffer(normals.astype("float32").tobytes())
    cbo = ctx.buffer(rgb.astype("float32").tobytes())
    ibo = ctx.buffer(tris.astype("int32").tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo, "3f", "in_vert"), (nbo, "3f", "in_normal"), (cbo, "3f", "in_rgb")],
        ibo,
    )
    vao.render(moderngl.TRIANGLES)

    vao.release()
    ibo.release()
    cbo.release()
    nbo.release()
    vbo.release()


def render_scalar_field_points(
    pts,
    values,
    prog,
    ctx,
    val_min=0.0,
    val_max=1.0,
    point_size=1.0,
    alphas=None,
):
    assert pts.shape[-1] == 3, f"only support 3d points {pts.shape}"
    assert (
        pts.numel() == 3 * values.numel()
    ), f"pts and values must have same numel {pts.numel()} {values.numel()}, {pts.shape} and {values.shape}"

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().numpy()
    if pts.shape[0] == 0:
        return
    if alphas is None:
        alphas = np.ones_like(values)
    else:
        if isinstance(alphas, torch.Tensor):
            alphas = alphas.detach().cpu().float().numpy()
        if isinstance(alphas, torch.Tensor):
            alphas = alphas.detach().cpu().float().numpy()
    prog["max_value"].value = val_max
    prog["min_value"].value = val_min
    prog["point_size"].value = point_size
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    vbv = ctx.buffer(values.astype("float32").tobytes())
    vba = ctx.buffer(alphas.astype("float32").tobytes())
    vao = ctx.vertex_array(
        prog, [(vbo, "3f", "in_vert"), (vbv, "1f", "in_value"), (vba, "1f", "in_alpha")]
    )
    vao.render(moderngl.POINTS)

    vao.release()
    vba.release()
    vbv.release()
    vbo.release()


def render_rgb_points(
    pts,
    rgb,
    prog,
    ctx,
    point_size=1.0,
):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().float().numpy()
    if pts.shape[0] == 0:
        return
    prog["point_size"].value = point_size
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    cbo = ctx.buffer(rgb.astype("float32").tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f", "in_vert"), (cbo, "3f", "in_rgb")])
    vao.render(moderngl.POINTS)

    vao.release()
    cbo.release()
    vbo.release()


def render_linestrip(pts, rgba, prog=None, ctx=None, scene=None):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if rgba is None:
        rgba = (0.0, 0.0, 0.0, 1.0)
    if pts.shape[0] == 0:
        return
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx
    prog["global_color"].value = rgba
    vbo = ctx.buffer(pts.astype("float32").tobytes())
    vao = ctx.vertex_array(prog, vbo, "in_vert")
    vao.render(moderngl.LINE_STRIP)

    vao.release()
    vbo.release()


def render_line(p0, p1, rgba, prog=None, ctx=None, scene=None):
    if isinstance(p0, list):
        p0 = np.array(p0)
    if isinstance(p1, list):
        p1 = np.array(p1)
    if isinstance(p0, torch.Tensor):
        p0 = p0.detach().cpu().numpy()
    if isinstance(p1, torch.Tensor):
        p1 = p1.detach().cpu().numpy()
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx
    pts = np.stack([p0, p1])
    render_linestrip(pts, rgba=rgba, prog=prog, ctx=ctx)


def render_cosy(
    T: Optional[PoseTW] = None, prog=None, ctx=None, scale: float = 0.1, scene=None
):
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx
    if T is None:
        T = PoseTW.from_Rt(torch.eye(3), torch.zeros(3))
    T = T.cpu().detach()
    ex = (T * torch.tensor([scale, 0.0, 0.0])).squeeze(0)
    ey = (T * torch.tensor([0.0, scale, 0.0])).squeeze(0)
    ez = (T * torch.tensor([0.0, 0.0, scale])).squeeze(0)
    render_line(T.t, ex, rgba=(1.0, 0.0, 0.0, 1.0), prog=prog, ctx=ctx)
    render_line(T.t, ey, rgba=(0.0, 1.0, 0.0, 1.0), prog=prog, ctx=ctx)
    render_line(T.t, ez, rgba=(0.0, 0.0, 1.0, 1.0), prog=prog, ctx=ctx)


def render_frustum(
    T_wr: PoseTW,
    cam: CameraTW,
    prog=None,
    ctx=None,
    rgba=(0, 0, 0, 1.0),
    scale=0.2,
    scene=None,
):
    """
    Draw the camera frustum of the given camera cam at the rig pose T_wr.
    """
    assert T_wr.dim() == 1
    assert cam.dim() == 1
    cam = cam.cpu().detach()
    T_wr = T_wr.cpu().detach()
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx

    def scaled_unproject(cam, pt2, scale):
        pt3 = cam.unproject(pt2)[0]
        pt3 = pt3 / torch.linalg.norm(pt3, dim=-1, keepdim=True)
        return pt3 * scale

    T_wc = T_wr @ cam.T_camera_rig.inverse()
    T_wc = T_wc.detach().cpu()
    c = cam.c
    rs = cam.valid_radius * 0.7071  # multiply by sqrt(0.5) to get the diagonal
    # valid get image corners
    tl = (c + torch.FloatTensor([-rs[0], -rs[1]])).view(1, 1, -1)
    tr = (c + torch.FloatTensor([-rs[0], rs[1]])).view(1, 1, -1)
    br = (c + torch.FloatTensor([rs[0], rs[1]])).view(1, 1, -1)
    bl = (c + torch.FloatTensor([rs[0], -rs[1]])).view(1, 1, -1)
    # unproject to 3d
    tl_w = (T_wc * scaled_unproject(cam, tl, scale)).squeeze()
    tr_w = (T_wc * scaled_unproject(cam, tr, scale)).squeeze()
    br_w = (T_wc * scaled_unproject(cam, br, scale)).squeeze()
    bl_w = (T_wc * scaled_unproject(cam, bl, scale)).squeeze()
    c_w = T_wc.t
    # get line_strip
    p3_w = torch.stack(
        [tl_w, tr_w, br_w, bl_w, tl_w, c_w, tr_w, c_w, br_w, c_w, bl_w, c_w, tl_w], 0
    )
    return render_linestrip(p3_w.numpy(), rgba=rgba, prog=prog, ctx=ctx)


def render_obbs_line(
    obbs: ObbTW,
    prog=None,
    ctx=None,
    rgba=(0.0, 0.0, 0.0, 1.0),
    colors=None,
    color_alpha=1.0,
    line_width=3.0,
    draw_cosy=False,
    scene=None,
):
    """
    Draw multiple oriented bounding boxes (obbs) each as a set of lines. obbs should be of shape N x C.
    """
    assert obbs.dim() == 2, f"{obbs.shape}"
    if scene is not None:
        prog, ctx = scene.prog, scene.ctx
    old_line_width = ctx.line_width
    ctx.line_width = line_width
    for obb in obbs:
        sem_id = int(obb.sem_id)
        if colors is not None and sem_id < len(colors):
            rgb = colors[sem_id]
            rgba = (rgb[0], rgb[1], rgb[2], color_alpha)
        if obb.sem_id.item() >= 0:
            render_obb_line(
                obb,
                prog,
                ctx,
                rgba=rgba,
                draw_cosy=draw_cosy,
            )
    ctx.line_width = old_line_width


def get_color_from_id(sem_id, max_sem_id, rgba=None):
    if sem_id:
        rgba = (0.0, 0.0, 0.0, 1.0)
    return rgba


def render_obb_line(obb: ObbTW, prog, ctx, rgba=None, draw_cosy=False):
    """
    Draw line-based oriented bounding box (obb) for a single obb.
    """
    assert obb.dim() == 1
    p3_w = obb.bb3corners_world
    if not draw_cosy:
        # Draw with linestrip.
        p3_w_strip = p3_w[OBB_LINE_INDS, :]
        render_linestrip(p3_w_strip, rgba=rgba, prog=prog, ctx=ctx)
    else:
        # Draw lines one by one.
        p3_w_all = p3_w[BB3D_LINE_ORDERS, :]
        for i, p3 in enumerate(p3_w_all):
            if i in AXIS_COLORS_GL:
                cur_rgba = AXIS_COLORS_GL[i]
            else:
                cur_rgba = rgba
            render_linestrip(p3, rgba=cur_rgba, prog=prog, ctx=ctx)


class SceneView:
    """
    SceneView is a simple 3D scene renderer using OpenGL.
    Simply follow the pattern:

    # init the scene
    sceneView = SceneView(...)

    while something:
        # clear render buffer
        sceneView.clear()

        # set view to camera pose
        sceneView.set_follow_view(T_world_camera)
        # OR set view to model view matrix (any matrix you want to)
        sceneView.set_view(MV)

        # do call any rendering functions using scene.ctx and scene.prog
        ...

        # finish the rendering and obtain the rendered image
        img = sceneView.finish()
        # display or save image

    Here is a simple example to render a coordinate system at the origin:

    ```
    scene = SceneView(width=320, height=320)
    scene.clear()
    T_wc = PoseTW()
    scene.set_default_view(PoseTW(), zoom_factor=6)
    render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)
    img = np.array(scene.finish())
    ```

    """

    def __init__(
        self,
        width: int,
        height: int,
        z_near: float = 0.1,
        z_far: float = 1000.0,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Args:
            width (int): width of rendered image.
            height (int): height of rendered image.
            z_near (float): near clipping plane.
            z_far (float): far clipping plane.
            bg_color (Tuple[float, float, float]): background color (0-1 range)
        """
        self.width = width
        self.height = height
        self.z_near = z_near
        self.z_far = z_far
        self.bg_color = bg_color
        self.ctx = init_egl_context()
        if self.ctx is not None:
            self.prog = simple_shader_program(self.ctx)
            self.prog_scalar_field = scalar_field_shader_program(self.ctx)
            self.prog_rgb_point_cloud = rgb_point_cloud_shader_program(self.ctx)
            self.prog_mesh = mesh_normal_shader_program(self.ctx)
            self.prog_mesh_rgb = mesh_rgb_shader_program(self.ctx)
            # attach frame and depth buffer. Depth buffer is important to be able to
            # do z-buffering!
            self.fbo1 = self.ctx.framebuffer(
                self.ctx.renderbuffer((width, height), samples=4),
                self.ctx.depth_renderbuffer((width, height), samples=4),
            )
            self.fbo2 = self.ctx.framebuffer(
                self.ctx.renderbuffer((width, height)),
                self.ctx.depth_renderbuffer((width, height)),
            )

        # setup camera projection for rendering
        fu, fv = self.width / 0.5, self.height / 0.5
        self.f = min(fu, fv)
        self.P = projection_matrix_rdf_top_left(
            self.width,
            self.height,
            self.f,
            self.f,
            (self.width - 1.0) / 2,
            (self.height - 1.0) / 2,
            self.z_near,
            self.z_far,
        )

    def valid(self):
        return self.ctx is not None

    def clear(self, bg_color: Optional[Tuple[float, float, float]] = None):
        """
        clear the scene rendering buffer.
        (call before any rendering!)

        if bg_color is specified then this is used over the one specified during
        construction.
        """
        self.fbo1.use()
        if bg_color is not None:
            self.ctx.clear(
                red=bg_color[0], green=bg_color[1], blue=bg_color[2], depth=1e4
            )
        else:
            self.ctx.clear(
                red=self.bg_color[0],
                green=self.bg_color[1],
                blue=self.bg_color[2],
                depth=1e4,
            )
        # enable depth test, point size, blending, and cull backfacing mesh triangles
        self.ctx.enable(
            moderngl.DEPTH_TEST
            | moderngl.PROGRAM_POINT_SIZE
            | moderngl.BLEND
            | moderngl.CULL_FACE
        )

    def set_default_view(self, T_world_camera: PoseTW, zoom_factor: float = 4.0):
        """
        set view to follow given T_world_camera behind and to the right of the T_wc.
        """
        mv = get_mv(T_world_camera, zoom_factor=zoom_factor, position=[-1, -1, -2])
        self.set_view(mv)

    def set_follow_view(self, T_world_camera: PoseTW, zoom_factor: float = 4.0):
        """
        set view to follow given T_world_camera.
        """
        mv = get_mv(T_world_camera, zoom_factor=zoom_factor, position=[-1, 0, -2])
        self.set_view(mv)

    def set_birds_eye_view(self, T_world_camera: PoseTW, zoom_factor: float = 6.0):
        """
        set view to a birds eye view given T_world_camera.
        """
        mv = get_mv(T_world_camera, zoom_factor=zoom_factor, position=[-2, 0, -0.0001])
        T_ahead = PoseTW.from_Rt(torch.eye(3), torch.tensor([0, -2, 0]))
        mv = T_ahead.matrix.numpy() @ mv
        self.set_view(mv)

    def set_side_view(self, T_world_camera: PoseTW, zoom_factor: float = 6.0):
        """
        set view to the left side of T_world_camera
        """
        mv = get_mv(T_world_camera, zoom_factor=zoom_factor, position=[-0, 2, -0.0001])
        T_ahead = PoseTW.from_Rt(torch.eye(3), torch.tensor([-2.5, 0, 0]))
        mv = T_ahead.matrix.numpy() @ mv
        self.set_view(mv)

    def set_birds_eye_view_from_bb(
        self, bb_scene_xyzxyz: torch.Tensor, zoom_factor: float = 6.0
    ):
        """
        set view to a birds eye view given bounding volume of scene
        assumes gravity aligned coordinate system with z=up
        """
        bb_min = bb_scene_xyzxyz[:3]
        bb_max = bb_scene_xyzxyz[3:]
        bb_diag = bb_max - bb_min
        bb_center = (bb_max + bb_min) * 0.5
        up = torch.tensor([0, 0, 1])
        dz = bb_diag[0] * self.f / self.width
        dz = max(dz, bb_diag[1] * self.f / self.height)
        dz += bb_diag[2] * 0.5
        dir_max = bb_diag / F.normalize(bb_diag, p=2, dim=0)
        eye = bb_center + up * zoom_factor * dz
        eye = bb_center + dir_max * zoom_factor * dz
        eye = bb_max + up * zoom_factor * dz
        mv = model_view_look_at_rdf(eye.numpy(), bb_center.numpy(), -up.numpy())
        self.set_view(mv)

    def set_view(self, mv: Union[PoseTW, np.array]):
        """
        set view to model view matrix.
        """
        if isinstance(mv, PoseTW):
            mv = mv.matrix.numpy()
        MVP = self.P @ mv
        # important to transpose MVP since opengl is column-major!
        self.prog["mvp"].write(MVP.transpose().astype("float32").tobytes())
        self.prog_scalar_field["mvp"].write(MVP.transpose().astype("float32").tobytes())
        self.prog_rgb_point_cloud["mvp"].write(
            MVP.transpose().astype("float32").tobytes()
        )
        self.prog_mesh["mvp"].write(MVP.transpose().astype("float32").tobytes())
        self.prog_mesh["mv"].write(mv.transpose().astype("float32").tobytes())
        self.prog_mesh_rgb["mvp"].write(MVP.transpose().astype("float32").tobytes())
        self.prog_mesh_rgb["mv"].write(mv.transpose().astype("float32").tobytes())

    def finish(self):
        """
        finish the scene rendering and return the rendered image as a PIL image.
        (call after all rendering!)
        """
        self.ctx.copy_framebuffer(self.fbo2, self.fbo1)
        data = self.fbo2.read(components=3, alignment=1)
        img = Image.frombytes("RGB", self.fbo2.size, data)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


def draw_obb_scene_3d(
    tgt,
    T_ws,
    Ts_sr,
    cams,
    frame_id=0,
    tgt_removed=None,
    sem_ids_to_names=None,
    prd=None,
    width=512,
    height=512,
    draw_origin=False,
    draw_trajectory=True,
    draw_frustum=True,
    matcher=None,
    prd_logits=None,
    p3s_world=None,
    p3s_pred_world=None,
    depth_pred=None,
    # optional scene object - if you want constant GPU memory allocate them once
    # outside and pass them in to reuse.
    scene: Optional[SceneView] = None,
    z_height_clip=None,
    render_raw_pred=True,
    render_removed_pred=True,
    cams_slaml=None,
    cams_slamr=None,
    zoom_factor=4.0,
    bird_eye_view=False,
    scene_mesh_v=None,
    scene_mesh_f=None,
    scene_mesh_n=None,
    scene_mesh_T_wv=None,
):
    """
    Draw a 3D scene of obbs, camera trajectory and camera frustum. The scene is
    selected via the frame_id which indexes into the snippet variables Ts_wr,
    cams which are TxC.

    Args:
      tgt: target obbs whose bounding boxes are to be drawn
      Ts_wr: camera trajectory
      cams: camera calibrations
      frame_id: frame index to select from Ts_wr and cams
      sem_ids_to_names: a dict mapping sem ids to names
      prd:  predicted obbs (if any) who are to be drawn. These are optional and
            meant to allow comparing two sets of bounding boxes.
      width: width of figure (only needed if scene is not provided)
      height: height of figure (only needed if scene is not provided)
      draw_origin: if True, draw the origin of the scene
      draw_trajectory: if True, draw the camera trajectory
      draw_frustum: if True, draw the camera frustum
      matcher: a function that takes tgt ObbTW, prd ObbTW and prd_logits and returns a list of matching ids to draw (HungarianMatcher)
      prd_logits: a list of logits matching the prd ObbTWs for the matcher
      bg_color: background color for the rendered scene
      scene: optional scene to draw into (if not provided instantiated internally)
      z_height_clip: z clip to limit the points of the scene to below this height (remove ceilings for better viz)
      cams_slaml: camera calibrations of slam left camera
      cams_slamr: camera calibrations of slam right camera
    Returns:
      fig: plotly figure with all the drawings
    """

    if scene is None:
        scene = SceneView(width=width, height=height)

    cam = cams[frame_id].cpu()
    cam_slaml = cams_slaml[frame_id].cpu() if cams_slaml is not None else None
    cam_slamr = cams_slamr[frame_id].cpu() if cams_slamr is not None else None
    if p3s_world is not None:
        p3s_world = p3s_world[frame_id].cpu() if p3s_world.ndim == 3 else p3s_world
    if p3s_pred_world is not None:
        p3s_pred_world = (
            p3s_pred_world[frame_id].cpu()
            if isinstance(p3s_pred_world, list)
            else p3s_pred_world
        )
    if depth_pred is not None:
        depth_pred = (
            depth_pred[frame_id].cpu() if isinstance(depth_pred, list) else depth_pred
        )
    pose_id = frame_id
    Ts_wr = T_ws @ Ts_sr
    Ts_wr = Ts_wr.cpu()
    if cams.shape[0] != Ts_wr.shape[0]:
        pose_id = round(Ts_wr.shape[0] * (float(frame_id) / float(cams.shape[0])))
    Ts_wc = Ts_wr @ cam.T_camera_rig.inverse()
    T_wr = Ts_wr[pose_id]
    T_wc = Ts_wc[pose_id]

    if tgt is not None and tgt.ndim == 3:
        tgt = tgt[frame_id]
    tgt = tgt.cpu() if tgt is not None else None

    colors = None
    if sem_ids_to_names is not None:
        # needs to color to be in scale [0,1]
        colors = get_colors_from_sem_map(sem_ids_to_names, scale_to_255=False)

    # setup framebuffer for rendering
    scene.clear()
    if not bird_eye_view:
        scene.set_follow_view(T_wc, zoom_factor=zoom_factor)
    else:
        scene.set_birds_eye_view(T_wc, zoom_factor=zoom_factor)

    # draw target obbs
    if tgt is not None and tgt.shape[0] > 0:
        render_obbs_line(
            tgt,
            scene.prog,
            scene.ctx,
            rgba=(1.0, 0.0, 0.0, 1.0),
            colors=colors,
        )
    if render_removed_pred and tgt_removed is not None:
        if tgt_removed.ndim == 3:
            tgt_removed = tgt_removed[frame_id]
        render_obbs_line(
            tgt_removed.cpu(),
            scene.prog,
            scene.ctx,
            rgba=(0.75, 0.75, 0.75, 0.3),
        )

    if render_raw_pred and prd is not None:
        if prd.ndim == 3:
            prd = prd[frame_id]
        # change the alpha value of the predictions when we have target obbs.
        if tgt is not None and tgt.shape[0] > 0:
            color_alpha = 0.3
        else:
            color_alpha = 1.0
        # draw predicted obbs
        render_obbs_line(
            prd.cpu(),
            scene.prog,
            scene.ctx,
            colors=colors,
            color_alpha=color_alpha,
        )

    if draw_trajectory:
        # draw rig trajectory
        render_linestrip(
            Ts_wr.t, rgba=(0.0, 0.0, 0.0, 1.0), prog=scene.prog, ctx=scene.ctx
        )
        # draw the current rig pose
        render_cosy(T_wr, ctx=scene.ctx, prog=scene.prog, scale=0.3)
        # draw the snippet origin
        # n the case of frames coming from different snippets, e.g. T_ws is [10, 12]
        # take the first T_ws as the snippet origin.
        if T_ws.shape[0] > 0:
            T_ws = T_ws[0:1]
        render_cosy(T_ws, ctx=scene.ctx, prog=scene.prog, scale=0.3)

    if p3s_world is not None:
        if z_height_clip is not None:
            keep = p3s_world[:, 2] < z_height_clip
            p3s_world = p3s_world[keep]
        # draw 3d points
        render_points(
            p3s_world,
            (0.1, 0.1, 0.1, 1.0),
            prog=scene.prog,
            ctx=scene.ctx,
            point_size=1.2,
        )

    if scene_mesh_v is not None:
        verts_w = scene_mesh_T_wv * scene_mesh_v.to(scene_mesh_T_wv.device)
        normals_w = scene_mesh_T_wv.rotate(scene_mesh_n.to(scene_mesh_T_wv.device))
        render_tri_mesh(
            verts_w,
            normals_w,
            scene_mesh_f,
            prog=scene.prog_mesh,
            ctx=scene.ctx,
        )

    if p3s_pred_world is not None:
        if depth_pred is None:
            # draw 3d points
            render_points(
                p3s_pred_world,
                (0.0, 1.0, 0.0, 1.0),
                prog=scene.prog,
                ctx=scene.ctx,
                point_size=2.0,
            )
        else:
            # draw 3d points colored by depth
            render_scalar_field_points(
                p3s_pred_world,
                depth_pred,
                prog=scene.prog_scalar_field,
                ctx=scene.ctx,
                val_min=0.0,
                val_max=3.0,
                point_size=2.0,
            )

    if draw_frustum:
        # draw the current frustum
        render_frustum(
            T_wr, cam, prog=scene.prog, ctx=scene.ctx, rgba=(0.0, 0.0, 0.0, 1.0)
        )
        render_line(
            T_wr.t, T_wc.t, rgba=(0.0, 0.0, 1.0, 1.0), prog=scene.prog, ctx=scene.ctx
        )

        if draw_trajectory:
            # Show smaller frustums along trajectory.
            for twr in Ts_wr:
                render_frustum(
                    twr,
                    cam,
                    prog=scene.prog,
                    ctx=scene.ctx,
                    rgba=(0.0, 0.0, 0.0, 1.0),
                    scale=0.08,
                )

        if cam_slaml is not None:
            # draw the current frustum
            render_frustum(
                T_wr,
                cam_slaml,
                prog=scene.prog,
                ctx=scene.ctx,
                rgba=(0.0, 0.0, 0.0, 1.0),
            )
            T_wcsl = T_wr @ cam_slaml.T_camera_rig.inverse()
            render_line(
                T_wr.t,
                T_wcsl.t,
                rgba=(0.0, 0.0, 1.0, 1.0),
                prog=scene.prog,
                ctx=scene.ctx,
            )
        if cam_slamr is not None:
            # draw the current frustum
            render_frustum(
                T_wr,
                cam_slamr,
                prog=scene.prog,
                ctx=scene.ctx,
                rgba=(0.0, 0.0, 0.0, 1.0),
            )
            T_wcsr = T_wr @ cam_slamr.T_camera_rig.inverse()
            render_line(
                T_wr.t,
                T_wcsr.t,
                rgba=(0.0, 0.0, 1.0, 1.0),
                prog=scene.prog,
                ctx=scene.ctx,
            )

    if draw_origin:
        # draw the origin cosy
        render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)

    if matcher is not None and prd is not None and prd_logits is not None:
        # draw matches under matcher
        tgt_sem_id = [tgt.sem_id.squeeze(-1)]
        indices = matcher(
            prd_logits.unsqueeze(0),
            prd.bb3_center_world.unsqueeze(0),
            tgt_sem_id,
            [tgt.bb3_center_world],
        )
        for p, t in zip(indices[0][0], indices[0][1]):
            pt0 = prd.bb3_center_world[p]
            pt1 = tgt.bb3_center_world[t]
            render_line(pt0, pt1, (0.0, 0.0, 0.0, 1.0), scene.ctx, scene.prog)

    # finish and obtain image
    img = scene.finish()
    return img


def draw_snippet_scene_3d(
    snippet,
    sem_ids_to_names=None,
    width=512,
    height=512,
    draw_origin=False,
    frame_id: Optional[int] = None,
    batch_id: int = 0,
    # optional scene object - if you want constant GPU memory allocate them once
    # outside and pass them in to reuse.
    scene: Optional[SceneView] = None,
    clean_viz: bool = False,
    viz_gt_points: bool = True,
):
    """
    Draw a 3D scene of obbs and camera trajectory.

    Args:
      snippet: a AriaStreamer snippet dict containing all relevant information for drawing
      sem_ids_to_names: a dict mapping sem ids to names
      width: width of figure (only needed if scene is not provided)
      height: height of figure (only needed if scene is not provided)
      draw_origin: if True, draw the origin of the scene
      draw_center: if True, draw the center of the scene
      return_plotly: if True, return the plotly figures, otherwise return the rendered images.
      frame_id: if set, only return the image/plotly plot for this frame.
      batch_id: if we are passing batched inputs, select the batch with this id for rendering.
      scene: optional scene to draw into (if not provided instantiated internally)
      viz_gt_points: if there is ground truth depth in the batch, visualize the GT depth instead of the semi-dense points.
    Returns:
      fig: plotly figure with all the drawings
    """

    if scene is None:
        scene = SceneView(width=width, height=height)

    has_slaml = ARIA_CALIB[1] in snippet
    has_slamr = ARIA_CALIB[2] in snippet

    cams = snippet[ARIA_CALIB[0]].cpu()
    if has_slaml:
        cams_slaml = snippet[ARIA_CALIB[1]].cpu()
    else:
        cams_slaml = None
    if has_slamr:
        cams_slamr = snippet[ARIA_CALIB[2]].cpu()
    else:
        cams_slamr = None
    T_ws = snippet[ARIA_SNIPPET_T_WORLD_SNIPPET].cpu()
    if ARIA_IMG_T_SNIPPET_RIG[0] in snippet:
        Ts_sr = snippet[ARIA_IMG_T_SNIPPET_RIG[0]].cpu()
    elif ARIA_POSE_T_SNIPPET_RIG in snippet:
        Ts_sr = snippet[ARIA_POSE_T_SNIPPET_RIG].cpu()
        if Ts_sr.shape[0] != cams.shape[0]:
            cam_times_ns = snippet[ARIA_CALIB_TIME_NS[0]].tolist()
            pose_times_ns = snippet[ARIA_POSE_TIME_NS].tolist()
            Ts_sr = sample_nearest(cam_times_ns, pose_times_ns, Ts_sr)
    if T_ws.ndim == 3:
        T_ws = T_ws[batch_id]
        Ts_sr = Ts_sr[batch_id]
        cams = cams[batch_id]
        if has_slaml:
            cams_slaml = cams_slaml[batch_id]
        if has_slamr:
            cams_slamr = cams_slamr[batch_id]

    obbs, prd, uninst = None, None, None
    if ARIA_OBB_PADDED in snippet:
        obbs = snippet[ARIA_OBB_PADDED].cpu()
        obbs = obbs[batch_id] if obbs.ndim == 4 else obbs
    have_tracked = ARIA_OBB_TRACKED in snippet
    if have_tracked:
        obbs = snippet[ARIA_OBB_TRACKED].cpu()
        obbs = obbs[batch_id] if obbs.ndim == 4 else obbs
    if ARIA_OBB_PRED_VIZ in snippet:
        prd = snippet[ARIA_OBB_PRED_VIZ].cpu()
        prd = prd[batch_id] if prd.ndim == 4 else prd
    if ARIA_OBB_UNINST in snippet:
        uninst = snippet[ARIA_OBB_UNINST].cpu()
        uninst = uninst[batch_id] if uninst.ndim == 4 else uninst
    p3s_world = None

    # If GT depth exists, visualize GT depth pointcloud instead of semi-dense points.
    if viz_gt_points and ARIA_DISTANCE_M[0] in snippet:
        # Note: we only visualize GT depth map of RGB images now.
        valid_depths = snippet[ARIA_DISTANCE_M[0]].squeeze(1) > 1e-4
        p3cs, valids = dist_im_to_point_cloud_im(
            snippet[ARIA_DISTANCE_M[0]].squeeze(1),
            snippet[ARIA_CALIB[0]],
        )
        valids = torch.logical_and(valids, valid_depths)
        p3cs = p3cs.reshape(p3cs.shape[0], -1, 3)
        T_s_c = (
            snippet[ARIA_IMG_T_SNIPPET_RIG[0]]
            @ snippet[ARIA_CALIB[0]].T_camera_rig.inverse()
        )
        T_w_c = snippet[ARIA_SNIPPET_T_WORLD_SNIPPET] @ T_s_c
        p3ws = T_w_c * p3cs
        p3ws = p3ws.reshape(-1, 3)
        valids = valids.reshape(-1)
        p3ws = p3ws[valids]
        p3s_world = discretize_values(p3ws, precision=70)

    if ARIA_POINTS_WORLD in snippet:
        p3s_world = snippet[ARIA_POINTS_WORLD]
        p3s_world = p3s_world[batch_id] if p3s_world.ndim == 4 else p3s_world

    p3s_pred_world, depth_pred = None, None
    if ARIA_DISTANCE_M_PRED[0] in snippet:
        dist_m = snippet[ARIA_DISTANCE_M_PRED[0]].cpu()
        dist_m = dist_m[batch_id] if dist_m.ndim == 4 else dist_m
        # scale camera to fit the depth image (in case depth image is at a lower res)
        cams_depth = cams.scale_to(dist_m)
        Ts_wc = T_ws @ Ts_sr @ cams.T_camera_rig.inverse()
        p3s_pred_world, depth_pred = [], []
        for t in range(dist_m.shape[0]):
            p3s_c, valids = dist_im_to_point_cloud_im(dist_m[t], cams_depth[t])
            p3s_pred_world.append(Ts_wc[t] * p3s_c[valids])
            depth_pred.append(dist_m[t][valids])

    Ts_wr = T_ws @ Ts_sr
    obbs = obbs.remove_padding() if obbs is not None else None
    prd = prd.remove_padding() if prd is not None else None
    uninst = uninst.remove_padding() if uninst is not None else None

    # clip the point cloud 1m above the rig coordinates
    z_height_clip = Ts_wr.t[..., 2].max() + 1.0

    assert (
        Ts_wr.shape[0] == cams.shape[0]
    ), f"poses and cameras must have the same length but got {Ts_wr.shape[0]} and {cams.shape[0]}"
    if obbs is not None:
        assert Ts_wr.shape[0] == len(
            obbs
        ), f"poses and obbs must have the same length {len(obbs)} but got {Ts_wr.shape}"

    if frame_id:
        assert frame_id >= 0 and frame_id < Ts_wr.shape[0]
    frame_ids = [frame_id] if frame_id else range(Ts_wr.shape[0])

    scene_mesh_v = None
    scene_mesh_f = None
    scene_mesh_n = None
    scene_mesh_T_wv = None

    if ARIA_MESH_VERTS_W in snippet:
        scene_mesh_v = snippet[ARIA_MESH_VERTS_W].squeeze().cpu().detach().float()
        scene_mesh_f = snippet[ARIA_MESH_FACES].squeeze().cpu().detach().float()
        # flip normals to visualize better.
        scene_mesh_n = -snippet[ARIA_MESH_VERT_NORMS_W].squeeze().cpu().detach().float()
        scene_mesh_T_wv = PoseTW()

    imgs = []
    for t in frame_ids:
        # transform obbs into world coordinates too
        tgt_w = obbs[t].transform(T_ws) if obbs is not None else None
        prd_w = prd[t].transform(T_ws) if prd is not None else None
        uninst_w = uninst[t].transform(T_ws) if uninst is not None else None
        img = draw_obb_scene_3d(
            tgt=tgt_w,
            prd=prd_w,
            tgt_removed=uninst_w,
            T_ws=T_ws,
            Ts_sr=Ts_sr,
            cams=cams,
            cams_slaml=cams_slaml,
            cams_slamr=cams_slamr,
            frame_id=t,
            p3s_world=p3s_world,
            p3s_pred_world=p3s_pred_world,
            depth_pred=depth_pred,
            sem_ids_to_names=sem_ids_to_names,
            width=width,
            height=height,
            draw_origin=draw_origin,
            scene=scene,
            z_height_clip=z_height_clip,
            render_raw_pred=(not clean_viz) or (not have_tracked and clean_viz),
            render_removed_pred=not clean_viz,
            scene_mesh_v=scene_mesh_v,
            scene_mesh_f=scene_mesh_f,
            scene_mesh_n=scene_mesh_n,
            scene_mesh_T_wv=scene_mesh_T_wv,
        )
        imgs.append(np.array(img))
    return imgs


def normalize(x):
    return x / (np.linalg.norm(x, 2) + 1e-6)


# https://github.com/stevenlovegrove/Pangolin/blob/7776a567f5c7b074668b8abb2316aba3f4b8b568/components/pango_opengl/src/opengl_render_state.cpp#L621
# e=eye is the eye location in world coordinates (camera position)
# l=look_at is the look at direction (projects to image center)
# u=up is the up direction
def model_view_look_at_rdf(e, look_at, u):
    z = normalize(look_at - e)
    if np.allclose(u - z, np.zeros(3), atol=1e-5):
        # Add some tiny offset so that cross product is non-zero.
        z[1] = z[1] + 0.001
    x = normalize(np.cross(z, u))
    y = normalize(np.cross(z, x))

    M = np.zeros((4, 4))
    M[0, 0] = x[0]
    M[0, 1] = x[1]
    M[0, 2] = x[2]
    M[1, 0] = y[0]
    M[1, 1] = y[1]
    M[1, 2] = y[2]
    M[2, 0] = z[0]
    M[2, 1] = z[1]
    M[2, 2] = z[2]
    M[3, 0] = 0.0
    M[3, 1] = 0.0
    M[3, 2] = 0.0
    M[0, 3] = -(M[0, 0] * e[0] + M[0, 1] * e[1] + M[0, 2] * e[2])
    M[1, 3] = -(M[1, 0] * e[0] + M[1, 1] * e[1] + M[1, 2] * e[2])
    M[2, 3] = -(M[2, 0] * e[0] + M[2, 1] * e[1] + M[2, 2] * e[2])
    M[3, 3] = 1.0
    return M


def get_mv(T_world_cam: PoseTW, zoom_factor: float = 3.0, position=[-1, 0, -2]):
    """
    T_world_cam is the camera pose in world coordinates that the ModelView Matrix will "follow".
    zoom_factor is the zoom factor for the ModelView Matrix. I.e. from how far
    above and behind the camera pose we will render the scene. 1.0 is very
    close, 3.0 is medium (good default) and 6.0 is farther away.
    """
    # gravity align the camera pose to make rendering videos smoother.
    T_world_cam = gravity_align_T_world_cam(
        T_world_cam.clone().unsqueeze(0), gravity_w=GRAVITY_DIRECTION_VIO
    ).squeeze(0)
    T_world_cam = T_world_cam.detach().cpu()
    # center is where "look at" position; will project to center of rendering
    center = T_world_cam.t
    # eye is the position of the camera center (translation)
    eye = T_world_cam * (torch.FloatTensor(position) * zoom_factor)
    # eye = T_world_cam * (torch.FloatTensor([-1,0,-1]) * zoom_factor)
    eye = eye.squeeze(0)
    # up is the up direction for the rendering camera. We choose it to be the
    # negative x-axis of the camera pose. Which works for our 90-deg rotated
    # cameras on Aria.
    up = T_world_cam.R[:, 0]
    # model view matrix
    mv = model_view_look_at_rdf(eye.numpy(), center.numpy(), up.numpy())
    return mv


# https://github.com/stevenlovegrove/Pangolin/blob/7776a567f5c7b074668b8abb2316aba3f4b8b568/components/pango_opengl/src/opengl_render_state.cpp#L462
# Camera Axis:
#   X - Right, Y - Down, Z - Forward
# Image Origin:
#   Top Left
# Pricipal point specified with image origin (0,0) at top left of top-left pixel (not center)
def projection_matrix_rdf_top_left(w, h, fu, fv, u0, v0, zNear, zFar):
    # http://www.songho.ca/opengl/gl_projectionmatrix.html
    L = -(u0) * zNear / fu
    R = +(w - u0) * zNear / fu
    T = -(v0) * zNear / fv
    B = +(h - v0) * zNear / fv

    P = np.zeros((4, 4))
    P[0, 0] = 2 * zNear / (R - L)
    P[1, 1] = 2 * zNear / (T - B)
    P[0, 2] = (R + L) / (L - R)
    P[1, 2] = (T + B) / (B - T)
    P[2, 2] = (zFar + zNear) / (zFar - zNear)
    P[3, 2] = 1.0
    P[2, 3] = (2 * zFar * zNear) / (zNear - zFar)
    return P


def init_egl_context():
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
    except Exception as e:
        print(f"{e}")
        return None
    return ctx


def simple_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;
    uniform float point_size;
    in vec3 in_vert;

    void main() {
        gl_Position = mvp * vec4(in_vert, 1.0);
        gl_PointSize = point_size;
    }"""
    fragment_shader_source = """#version 330
    uniform vec4 global_color;
    out vec4 f_color;

    void main() {
        f_color = global_color;
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog


def mesh_normal_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform float point_size;
    in vec3 in_vert;
    in vec3 in_normal;

    out vec3 n_c;

    void main() {
        gl_Position = mvp * vec4(in_vert, 1.0);
        gl_PointSize = point_size;
        n_c = transpose(inverse(mat3(mv))) * in_normal;
    }"""
    fragment_shader_source = """#version 330
    in vec3 n_c;
    out vec4 f_color;

    void main() {
        f_color = vec4((normalize(n_c) + vec3(1.0, 1.0, 1.0)) / 2.0, 1.0f);
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog


def mesh_rgb_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform float point_size;
    in vec3 in_vert;
    in vec3 in_normal;
    in vec3 in_rgb;

    out vec3 n_c;
    out vec3 rgb;

    void main() {
        gl_Position = mvp * vec4(in_vert, 1.0);
        gl_PointSize = point_size;
        n_c = transpose(inverse(mat3(mv))) * in_normal;
        rgb = in_rgb;
    }"""
    fragment_shader_source = """#version 330
    in vec3 n_c;
    in vec3 rgb;
    out vec4 f_color;

    void main() {
        f_color = vec4(rgb * max(n_c.z, 0.0), 1.0);
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog


def rgb_point_cloud_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;
    uniform mat4 mv;
    uniform float point_size;
    in vec3 in_vert;
    in vec3 in_rgb;

    out vec3 rgb;

    void main() {
        gl_Position = mvp * vec4(in_vert, 1.0);
        gl_PointSize = point_size;
        rgb = in_rgb;
    }"""
    fragment_shader_source = """#version 330
    in vec3 rgb;
    out vec4 f_color;

    void main() {
        f_color = vec4(rgb, 1.0f);
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog


def scalar_field_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;
    uniform float point_size;
    uniform float max_value;
    uniform float min_value;
    in vec3 in_vert;
    in float in_value;
    in float in_alpha;
    out vec3 frag_rgb;
    out float frag_a;

    // https://thebookofshaders.com/06/
    vec3 hsb2rgb( in vec3 c ){
        vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
        rgb = rgb*rgb*(3.0-2.0*rgb);
        return c.z * mix( vec3(1.0), rgb, c.y);
    }
    vec3 hsv(float v) {
        return hsb2rgb(vec3(v, 1.0, 1.0));
    }

    // https://github.com/kbinani/colormap-shaders/tree/master
    // The MIT License (MIT)
    // Copyright (c) 2015 kbinani
    // Permission is hereby granted, free of charge, to any person obtaining a copy
    // of this software and associated documentation files (the "Software"), to deal
    // in the Software without restriction, including without limitation the rights
    // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    // copies of the Software, and to permit persons to whom the Software is
    // furnished to do so, subject to the following conditions:
    // The above copyright notice and this permission notice shall be included in all
    // copies or substantial portions of the Software.
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    // SOFTWARE.

    float jet_red(float x) {
        if (x < 0.7) {
            return 4.0 * x - 1.5;
        } else {
            return -4.0 * x + 4.5;
        }
    }
    float jet_green(float x) {
        if (x < 0.5) {
            return 4.0 * x - 0.5;
        } else {
            return -4.0 * x + 3.5;
        }
    }
    float jet_blue(float x) {
        if (x < 0.3) {
           return 4.0 * x + 0.5;
        } else {
           return -4.0 * x + 2.5;
        }
    }
    vec3 jet(float x) {
        float r = clamp(jet_red(x), 0.0, 1.0);
        float g = clamp(jet_green(x), 0.0, 1.0);
        float b = clamp(jet_blue(x), 0.0, 1.0);
        return vec3(r, g, b);
    }

    void main() {
        float f_value = (in_value - min_value) / (max_value - min_value);
        f_value = clamp(f_value, 0.0, 1.0);
        frag_rgb = jet(f_value);
        frag_a = in_alpha;
        gl_Position = mvp * vec4(in_vert, 1.0);
        gl_PointSize = point_size;
    }
    """
    fragment_shader_source = """#version 330
    in vec3 frag_rgb;
    in float frag_a;
    out vec4 f_color;

    void main() {
        f_color = vec4(frag_rgb, frag_a);
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog


def semantic_color_shader_program(ctx):
    vertex_shader_source = """#version 330
    uniform mat4 mvp;

    in int in_sem_id;
    in vec3 in_vert;
    out int sem_id;
    out vec3 v_vert;

    void main() {
        v_vert = in_vert;
        gl_Position = mvp * vec4(v_vert, 1.0);
        sem_id = in_sem_id;
    }"""

    fragment_shader_source = """#version 330
    uniform int sem_max;

    in int sem_id;
    in vec3 v_vert;
    out vec4 f_color;

    // https://thebookofshaders.com/06/
    vec3 hsb2rgb( in vec3 c ){
        vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
        rgb = rgb*rgb*(3.0-2.0*rgb);
        return c.z * mix( vec3(1.0), rgb, c.y);
    }

    void main() {
        sem_hue = sem_id / sem_max;
        f_color = vec4(hsb2rgb(vec3(sem_hue, 1.0, 1.0)), 1.0);
    }
    """
    prog = ctx.program(
        vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source
    )
    return prog
