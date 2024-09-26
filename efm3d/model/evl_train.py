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

from typing import List, Optional, Union

import numpy as np

import torch
from efm3d.aria.aria_constants import (
    ARIA_CALIB,
    ARIA_IMG,
    ARIA_IMG_T_SNIPPET_RIG,
    ARIA_OBB_PADDED,
    ARIA_OBB_PRED,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW

from efm3d.model.evl import EVL
from efm3d.model.lifter import VideoBackbone3d

from efm3d.model.video_backbone import VideoBackbone
from efm3d.utils.evl_loss import compute_obb_losses, compute_occ_losses, get_gt_obbs
from efm3d.utils.image import put_text
from efm3d.utils.marching_cubes import marching_cubes_scaled
from efm3d.utils.obb_utils import prec_recall_bb3
from efm3d.utils.pointcloud import (
    get_points_world,
    pointcloud_occupancy_samples,
    pointcloud_to_occupancy_snippet,
)
from efm3d.utils.render import draw_obbs_snippet, get_colors_from_sem_map
from efm3d.utils.viz import (
    render_cosy,
    render_frustum,
    render_linestrip,
    render_obb_line,
    render_obbs_line,
    render_points,
    render_rgb_tri_mesh,
    render_scalar_field_points,
    render_tri_mesh,
    SceneView,
)
from efm3d.utils.voxel import erode_voxel_mask
from efm3d.utils.voxel_sampling import pc_to_vox, sample_voxels
from omegaconf import DictConfig


class EVLTrain(EVL):
    def __init__(
        self,
        video_backbone: Union[VideoBackbone, DictConfig],
        video_backbone3d: Union[VideoBackbone3d, DictConfig],
        neck_hidden_dims: Optional[List] = None,
        head_hidden_dim: int = 128,
        head_layers: int = 2,
        taxonomy_file: Optional[str] = None,
        det_thresh: float = 0.2,
        yaw_max: float = 1.6,
    ):
        super().__init__(
            video_backbone,
            video_backbone3d,
            neck_hidden_dims,
            head_hidden_dim,
            head_layers,
            taxonomy_file,
            det_thresh,
            yaw_max,
        )

    def compute_losses(self, outputs, batch):
        total_loss = 0
        losses = {"rgb": {}}

        self.occ_weight = 10.0
        self.tv_weight = 0.01
        occ_losses, occ_total_loss = compute_occ_losses(
            outputs,
            batch,
            self.ve,
            occ_weight=self.occ_weight,
            tv_weight=self.tv_weight,
        )
        for k in losses:  # for ['rgb', 'slaml', 'slamr']
            losses[k].update(occ_losses[k])
        total_loss += occ_total_loss

        self.cent_weight = 10.0
        self.bbox_weight = 0.0
        self.cham_weight = 0.0
        self.clas_weight = 0.1
        self.iou_weight = 0.5
        obb_losses, obb_total_loss = compute_obb_losses(
            outputs,
            batch,
            self.ve,
            self.num_class,
            self.splat_sigma,
            cent_weight=self.cent_weight,
            clas_weight=self.clas_weight,
            iou_weight=self.iou_weight,
            bbox_weight=self.bbox_weight,
            cham_weight=self.cham_weight,
        )
        for k in losses:  # for ['rgb', 'slaml', 'slamr']
            losses[k].update(obb_losses[k])
        total_loss += obb_total_loss

        return losses, total_loss

    def render2d(self, imgs, obbs, Ts_wr, cams):
        """Render a 2D visualization overlaid on the RGB image of the given obbs."""
        # Draw the 3D bb overlaid on the image.
        obb_img = draw_obbs_snippet(
            imgs.clone(),
            obbs,
            Ts_wr,
            cams,
            rgb2bgr=False,
            draw_cosy=True,
            white_backing_line=False,
            draw_bb2=False,
            sem_id_to_name_mapping=self.sem2name,
            draw_label=True,
            draw_score=True,
            prob_threshold=0.001,  # keep this very low, obbs are already thresholded.
        )
        return np.array(obb_img)

    def log_single_obb(self, batch, outputs, batch_idx):
        """Log a single element from the batch based on "batch_idx"."""
        log_ims = {}

        # Get stuff.
        rgb_img = batch[ARIA_IMG[0]][batch_idx].cpu().detach()
        T = rgb_img.shape[0]
        cams = batch[ARIA_CALIB[0]][batch_idx].cpu().detach()
        Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]][batch_idx].cpu().detach()
        T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET][batch_idx].cpu().detach()
        voxel_w = outputs["voxel/pts_world"][batch_idx].cpu().detach()
        T_wv = outputs["voxel/T_world_voxel"][batch_idx]
        obbs_gt = get_gt_obbs(batch, self.ve, T_wv)
        obbs_gt = obbs_gt[batch_idx].cpu()
        T_wv = T_wv.cpu().detach()
        cent_pr = outputs["cent_pr"][batch_idx].cpu().detach()
        obbs_pr = outputs["obbs_pr_nms"][batch_idx].cpu().detach()
        occ_input = None

        # Get some convenience transforms.
        Ts_wr = T_ws @ Ts_sr

        # Transform Objects to world coords.
        obbs_pr = obbs_pr.transform(T_wv)  # T_wo = T_wv @ T_vo

        # Transform lifter volume obb to world coordinates.
        extent = torch.tensor(self.ve).to(T_wv._data)
        voxel_obb = ObbTW()[0]
        voxel_obb.set_bb3_object(extent, use_mask=False)
        voxel_obb.set_T_world_object(T_wv)

        occ_input = outputs["voxel/occ_input"][batch_idx].cpu().detach().reshape(-1)
        mask = occ_input > 1e-4
        log_ims["voxel/occ_input"] = self.render3d_obb(
            occ_input[mask],
            obbs_pr,
            Ts_wr,
            T_ws,
            cams,
            voxel_w[mask],
            voxel_obb,
            view="follow",
            alpha_min=0.1,
        )

        # compute precision and recall and add text to the pred 2d
        log_ims["rgb_pred"] = self.render2d(rgb_img, obbs_pr, Ts_wr, cams)
        if ARIA_OBB_PADDED in batch:
            obbs_pr_nms = outputs[ARIA_OBB_PRED][batch_idx].cpu()
            prec, rec, match_mat = prec_recall_bb3(
                obbs_pr_nms.remove_padding(),
                obbs_gt.remove_padding(),
                iou_thres=self.iou_thres,
            )
            if match_mat is not None:
                num_tp = match_mat.any(-1).sum().item()
                num_pred = match_mat.shape[0]
                num_gt = match_mat.shape[1]
                precision = f"Prec@{self.iou_thres}: {prec:.2f} ({num_tp}/{num_pred})"
                recall = f"Recall@{self.iou_thres}: {rec:.2f} ({num_tp}/{num_gt})"
            else:
                precision = f"Prec@{self.iou_thres}: {prec:.2f}"
                recall = f"Recall@{self.iou_thres}: {rec:.2f}"
            imgs_pred = log_ims["rgb_pred"]
            imgs_pred = [put_text(img, precision, line=-2) for img in imgs_pred]
            imgs_pred = [put_text(img, recall, line=-1) for img in imgs_pred]
            log_ims["rgb_pred"] = np.array(imgs_pred)

        log_ims["3D_pred"] = self.render3d_obb(
            cent_pr,
            obbs_pr,
            Ts_wr,
            T_ws,
            cams,
            voxel_w,
            voxel_obb,
            view="follow",
            alpha_min=0.1,
        )

        if "cent_gt" in outputs:
            self.compute_losses(outputs, batch)

            obbs_gt = obbs_gt[~obbs_gt.get_padding_mask()]
            obbs_gt = obbs_gt.transform(T_ws)  # T_wo = T_ws @ T_so
            log_ims["rgb_gt"] = self.render2d(rgb_img, obbs_gt, Ts_wr, cams)

            cent_gt = outputs["cent_gt"][batch_idx].cpu().reshape(-1)
            log_ims["3D_gt"] = self.render3d_obb(
                cent_gt,
                obbs_gt,
                Ts_wr,
                T_ws,
                cams,
                voxel_w,
                voxel_obb,
                alpha_min=0.1,
            )

        return log_ims

    def log_single(self, batch, outputs, batch_idx):
        """Log a single element from the batch based on "batch_idx"."""
        log_ims = self.log_single_obb(batch, outputs, batch_idx)

        cams = batch[ARIA_CALIB[0]][batch_idx].cpu().detach()
        Ts_sr = batch[ARIA_IMG_T_SNIPPET_RIG[0]][batch_idx].cpu().detach()
        T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET][batch_idx].cpu().detach()
        voxel_w = outputs["voxel/pts_world"][batch_idx].cpu().detach()
        T_wv = outputs["voxel/T_world_voxel"][batch_idx].cpu().detach()
        Ts_wr = T_ws @ Ts_sr
        T = cams.shape[0]

        occ = outputs["occ_pr"].squeeze(1)
        voxel_counts = outputs["voxel/counts"][batch_idx].cpu().detach()

        B, D, H, W = occ.shape
        Df, Hf, Wf = voxel_counts.shape
        if D != Df or H != Hf or W != Wf:
            resize = torch.nn.Upsample(size=(D, H, W))
            voxel_w = voxel_w.view(Df, Hf, Wf, 3).permute(3, 0, 1, 2)
            voxel_w = resize(voxel_w.unsqueeze(0)).squeeze(0)
            voxel_w = voxel_w.permute(1, 2, 3, 0).view(-1, 3)
            voxel_counts = resize(voxel_counts.unsqueeze(0).unsqueeze(0).float())
            voxel_counts = voxel_counts.squeeze(0).squeeze(0)

        visible = voxel_counts > 0
        visible = erode_voxel_mask(visible.unsqueeze(0)).squeeze(0)

        # Get some convenience transforms.
        Ts_wr = T_ws @ Ts_sr
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()

        # Transform lifter volume obb to world coordinates.
        extent = torch.tensor(self.ve).to(T_wv._data)
        voxel_obb = ObbTW()[0]
        voxel_obb.set_bb3_object(extent, use_mask=False)
        voxel_obb.set_T_world_object(T_wv)

        # -------------------- draw occ -----------------------
        occ_pr = outputs["occ_pr"][batch_idx].cpu().detach().squeeze(0)
        alpha_min = 0.5 if self.occ_weight > 0.0 else 0.04
        log_ims["occ/mesh_pred"] = self.render3d_mesh(
            occ_pr,
            Ts_wr,
            T_ws,
            cams,
            voxel_obb,
            view="follow",
            alpha_min=alpha_min,
            T_wv=T_wv,
            voxel_mask=visible,
        )

        log_ims["occ/occ_pred"] = self.render3d_occ(
            occ_pr,
            Ts_wr,
            T_ws,
            cams,
            voxel_w,
            voxel_obb,
            view="follow",
            alpha_min=alpha_min,
            voxel_mask=visible,
        )

        vD, vH, vW = occ_pr.shape
        pc_w = get_points_world(batch, batch_idx)[0].cpu().detach()
        (
            p3s_occ_w,
            p3s_surf_w,
            p3s_free_w,
            valid,
        ) = pointcloud_occupancy_samples(
            pc_w.unsqueeze(0),
            Ts_wc.unsqueeze(0),
            cams.unsqueeze(0),
            vW,
            vH,
            vD,
            self.ve,
            S=1,
            T_wv=T_wv,
        )
        p3s_occ_w[~valid] = float("nan")
        p3s_surf_w[~valid] = float("nan")
        p3s_free_w[~valid] = float("nan")

        log_ims["occ/occ_gt_samples"] = self.render3d_points(
            p3s_surf_w.squeeze(0),
            Ts_wr,
            T_ws,
            cams,
            voxel_obb,
            view="follow",
            more_p3s_w=p3s_free_w.squeeze(0),
            more2_p3s_w=p3s_occ_w.squeeze(0),
        )

        # get occ gt
        occ_gt, mask = pointcloud_to_occupancy_snippet(
            pc_w,
            Ts_wc,
            cams,
            T_wv,
            vW,
            vH,
            vD,
            self.ve,
            S=1,
        )
        mask = torch.logical_and(mask.bool(), visible)
        log_ims["occ/mesh_gt"] = self.render3d_mesh(
            occ_gt,
            Ts_wr,
            T_ws,
            cams,
            voxel_obb,
            view="follow",
            alpha_min=alpha_min,
            T_wv=T_wv,
            voxel_mask=mask,
        )

        return log_ims

    @torch.no_grad()
    def render3d_mesh(
        self,
        voxel_vals,
        Ts_wr,
        T_ws,
        cams,
        voxel_obb,
        view="follow",
        alpha_min=0.5,
        T_wv=None,
        voxel_mask=None,
        volume_feat=None,
    ):
        """Render a 3D visualization of the given voxel values and obbs."""
        if self.scene is None:
            self.scene = SceneView(width=320, height=320)
        scene = self.scene
        lifter_imgs = []
        verts_v, faces, normals_v = marching_cubes_scaled(
            voxel_vals.cpu().detach().float(),
            alpha_min,
            self.ve,
            voxel_mask,
        )
        feats = torch.tensor([])
        if volume_feat is not None and len(verts_v) > 0:
            vD, vH, vW = voxel_vals.shape
            p3s_surf_vox, _ = pc_to_vox(verts_v, vW, vH, vD, self.ve)
            feats, _ = sample_voxels(
                volume_feat.unsqueeze(0), p3s_surf_vox.unsqueeze(0)
            )
            feats = feats.squeeze(0).permute(1, 0)
            print("[WARN] No PCA compressor provided. Take the first 3 channels.")
            rgb = feats[:, :3]
            maxs = rgb.max(dim=-1, keepdim=True)[0]
            mins = rgb.min(dim=-1, keepdim=True)[0]
            rgb = (rgb - mins) / (maxs - mins + 1e-4)
        black = (0.0, 0.0, 0.0, 1.0)
        green = (0.0, 1.0, 0.0, 1.0)
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()
        for t, (T_wr, T_wc, cam) in enumerate(zip(Ts_wr, Ts_wc, cams)):
            scene.clear()
            if view == "follow":
                scene.set_follow_view(T_wc, zoom_factor=4)
            elif view == "bird":
                scene.set_birds_eye_view(T_wc, zoom_factor=8)
            else:
                raise ValueError("bad option for 3d view style")
            if len(verts_v) > 0:
                verts_w = T_wv * verts_v.to(T_wv.device)
                normals_w = T_wv.rotate(normals_v.to(T_wv.device))
                if volume_feat is not None:
                    render_rgb_tri_mesh(
                        verts_w,
                        -normals_w,
                        faces,
                        rgb=rgb,
                        prog=scene.prog_mesh_rgb,
                        ctx=scene.ctx,
                    )
                else:
                    render_tri_mesh(
                        verts_w,
                        normals_w,
                        faces,
                        prog=scene.prog_mesh,
                        ctx=scene.ctx,
                    )

            # draw voxel bounding volume
            render_obb_line(
                voxel_obb, scene.prog, scene.ctx, rgba=black, draw_cosy=True
            )
            # draw trajectory.
            render_linestrip(Ts_wr.t, prog=scene.prog, ctx=scene.ctx, rgba=black)
            render_frustum(T_wr, cam, prog=scene.prog, ctx=scene.ctx, rgba=black)
            # Render snippet origin.
            render_cosy(T_ws, ctx=scene.ctx, prog=scene.prog, scale=0.3)
            # Render world origin.
            render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)
            img = scene.finish()
            lifter_imgs.append(np.array(img))
        lifter_imgs = np.array(lifter_imgs)
        if volume_feat is None:
            return lifter_imgs
        else:
            return lifter_imgs, feats, verts_v, faces, normals_v

    @torch.no_grad()
    def render3d_points(
        self,
        p3s_w,
        Ts_wr,
        T_ws,
        cams,
        voxel_obb,
        view="follow",
        values=None,
        alpha_min=0.01,
        mask=None,
        more_p3s_w=None,
        more2_p3s_w=None,
    ):
        """Render a 3D visualization of the given voxel values and obbs."""
        if self.scene is None:
            self.scene = SceneView(width=320, height=320)
        scene = self.scene
        lifter_imgs = []
        black = (0.0, 0.0, 0.0, 1.0)
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()

        for t, (T_wr, T_wc, cam) in enumerate(zip(Ts_wr, Ts_wc, cams)):
            scene.clear()
            if view == "follow":
                scene.set_follow_view(T_wc, zoom_factor=8)
            elif view == "bird":
                scene.set_birds_eye_view(T_wc, zoom_factor=12)
            else:
                raise ValueError("bad option for 3d view style")

            if values is not None:
                alphas = torch.ones_like(values[t])
                if alpha_min is not None:
                    alphas[values[t] < alpha_min] = 0
                else:
                    alpha_min = 0.0
            if mask is not None:
                p3s_wt = p3s_w[t][mask[t]]
                if values is not None:
                    values_t = values[t][mask[t]]
                    alphas_t = alphas[mask[t]]
            else:
                p3s_wt = p3s_w[t]
                if values is not None:
                    values_t = values[t]
                    alphas_t = alphas

            if values is not None:
                render_scalar_field_points(
                    p3s_wt,
                    values_t.float(),
                    prog=scene.prog_scalar_field,
                    ctx=scene.ctx,
                    point_size=1.0,
                    alphas=alphas_t,
                    val_min=alpha_min,
                )
            else:
                render_points(p3s_wt, (1.0, 0, 0, 1.0), scene.prog, scene.ctx, 1.0)

            if more_p3s_w is not None:
                render_points(
                    more_p3s_w[t], (0.0, 1.0, 0, 0.5), scene.prog, scene.ctx, 1.0
                )
            if more2_p3s_w is not None:
                render_points(
                    more2_p3s_w[t], (0.0, 0.0, 1.0, 1.0), scene.prog, scene.ctx, 1.0
                )
            # draw voxel bounding volume
            render_obb_line(
                voxel_obb, scene.prog, scene.ctx, rgba=black, draw_cosy=True
            )
            # draw trajectory.
            render_linestrip(Ts_wr.t, prog=scene.prog, ctx=scene.ctx, rgba=black)
            render_frustum(T_wr, cam, prog=scene.prog, ctx=scene.ctx, rgba=black)
            # Render snippet origin.
            render_cosy(T_ws, ctx=scene.ctx, prog=scene.prog, scale=0.3)
            # Render world origin.
            render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)
            img = scene.finish()
            lifter_imgs.append(np.array(img))
        lifter_imgs = np.array(lifter_imgs)
        return lifter_imgs

    @torch.no_grad()
    def render3d_obb(
        self,
        voxel_vals,
        obb,
        Ts_wr,
        T_ws,
        cams,
        voxel_w,
        voxel_obb,
        view="follow",
        alpha_min=None,
    ):
        """Render a 3D visualization of the given voxel values and obbs."""
        if self.scene is None:
            self.scene = SceneView(width=320, height=320)
        scene = self.scene
        lifter_imgs = []
        alphas = torch.ones_like(voxel_vals)
        if alpha_min is not None:
            alphas[voxel_vals < alpha_min] = 0
        blue = (0.1, 0.1, 1.0, 1.0)
        black = (0.0, 0.0, 0.0, 1.0)
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()
        for T_wr, T_wc, cam in zip(Ts_wr, Ts_wc, cams):
            scene.clear()
            if view == "follow":
                scene.set_follow_view(T_wc, zoom_factor=8)
            elif view == "bird":
                scene.set_birds_eye_view(T_wc, zoom_factor=8)
            else:
                raise ValueError("bad option for 3d view style")
            # draw obbs
            if obb:
                colors = get_colors_from_sem_map(self.sem2name, scale_to_255=False)
                render_obbs_line(
                    obb,
                    scene.prog,
                    scene.ctx,
                    line_width=2,
                    colors=colors,
                    draw_cosy=True,
                )
            # draw voxel bounding volume
            render_obb_line(
                voxel_obb, scene.prog, scene.ctx, rgba=black, draw_cosy=True
            )
            # draw trajectory.
            render_linestrip(Ts_wr.t, prog=scene.prog, ctx=scene.ctx, rgba=black)
            render_frustum(T_wr, cam, prog=scene.prog, ctx=scene.ctx, rgba=black)
            # Render snippet origin.
            render_cosy(T_ws, ctx=scene.ctx, prog=scene.prog, scale=0.3)
            # Render world origin.
            render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)
            # "scalar_field_points" supports colored point cloud, will rescale based on min/max.
            render_scalar_field_points(
                voxel_w,
                voxel_vals,
                prog=scene.prog_scalar_field,
                ctx=scene.ctx,
                point_size=3,
                alphas=alphas,
            )
            img = scene.finish()
            lifter_imgs.append(np.array(img))
        lifter_imgs = np.array(lifter_imgs)
        return lifter_imgs

    @torch.no_grad()
    def render3d_occ(
        self,
        voxel_vals,
        Ts_wr,
        T_ws,
        cams,
        voxel_w,
        voxel_obb,
        view="follow",
        alpha_min=None,
        voxel_mask=None,
    ):
        """Render a 3D visualization of the given voxel values and obbs."""
        if self.scene is None:
            self.scene = SceneView(width=320, height=320)
        scene = self.scene
        lifter_imgs = []
        black = (0.0, 0.0, 0.0, 1.0)
        Ts_wc = Ts_wr @ cams.T_camera_rig.inverse()
        for t, (T_wr, T_wc, cam) in enumerate(zip(Ts_wr, Ts_wc, cams)):
            if voxel_vals.ndim == 4:
                v_vals = voxel_vals[t]
            else:
                v_vals = voxel_vals
            alp = torch.ones_like(v_vals)
            if alpha_min is not None:
                if isinstance(alpha_min, torch.Tensor):
                    alp_min = alpha_min[t]
                else:
                    alp_min = alpha_min
                alp[v_vals < alp_min] = 0
            else:
                alp_min = 0.0

            scene.clear()
            if view == "follow":
                scene.set_follow_view(T_wc, zoom_factor=4)
            elif view == "bird":
                scene.set_birds_eye_view(T_wc, zoom_factor=8)
            else:
                raise ValueError("bad option for 3d view style")
            # "scalar_field_points" supports colored point cloud, will rescale based on min/max.
            if voxel_mask is not None:
                render_scalar_field_points(
                    voxel_w[voxel_mask.view(-1)],
                    v_vals[voxel_mask].float(),
                    prog=scene.prog_scalar_field,
                    ctx=scene.ctx,
                    point_size=3,
                    alphas=alp[voxel_mask].float(),
                    val_min=alp_min,
                )
            else:
                render_scalar_field_points(
                    voxel_w,
                    v_vals.float(),
                    prog=scene.prog_scalar_field,
                    ctx=scene.ctx,
                    point_size=3,
                    alphas=alp.float(),
                    val_min=alp_min,
                )
            # draw voxel bounding volume
            render_obb_line(
                voxel_obb, scene.prog, scene.ctx, rgba=black, draw_cosy=True
            )
            # draw trajectory.
            render_linestrip(Ts_wr.t, prog=scene.prog, ctx=scene.ctx, rgba=black)
            render_frustum(T_wr, cam, prog=scene.prog, ctx=scene.ctx, rgba=black)
            # Render snippet origin.
            render_cosy(T_ws, ctx=scene.ctx, prog=scene.prog, scale=0.3)
            # Render world origin.
            render_cosy(PoseTW(), ctx=scene.ctx, prog=scene.prog, scale=1.0)
            img = scene.finish()
            lifter_imgs.append(np.array(img))
        lifter_imgs = np.array(lifter_imgs)
        return lifter_imgs

    def get_log_images(self, batch, outputs):
        B = len(batch["rgb/img"])
        with torch.no_grad():
            # Visualize one random element from the batch.
            batch_idx = torch.randint(low=0, high=B, size=(1,)).item()
            log_ims = self.log_single(batch, outputs, batch_idx=batch_idx)
        return log_ims

    def reset_metrics(self):
        self.metrics = {}

        # obb
        self.metrics[f"precision@{self.iou_thres}"] = []
        self.metrics[f"recall@{self.iou_thres}"] = []

        # occ
        self.metrics["mesh/acc"] = []
        self.metrics["mesh/comp"] = []
        self.metrics["mesh/prec"] = []
        self.metrics["mesh/recall"] = []

    def update_metrics(self, outputs, batch):
        # don't compute metrics on training since it takes long to compute.
        if self.training:
            return

        obbs_pred = outputs[ARIA_OBB_PRED]
        T_wv = outputs["voxel/T_world_voxel"]
        obbs_gt = get_gt_obbs(batch, self.ve, T_wv)
        precs, recs = [], []
        for obbs_pred_s, obbs_gt_s in zip(obbs_pred, obbs_gt):
            prec, rec, _ = prec_recall_bb3(
                obbs_pred_s.remove_padding(),
                obbs_gt_s.remove_padding(),
                iou_thres=self.iou_thres,
            )

            if prec != -1.0 and rec != -1.0:
                precs.append(prec)
                recs.append(rec)
        self.metrics[f"precision@{self.iou_thres}"].extend(precs)
        self.metrics[f"recall@{self.iou_thres}"].extend(recs)

    def compute_metrics(self):
        metrics = {}
        if self.training:
            return metrics

        metrics["rgb"] = {}
        metrics["rgb"]["metrics"] = {}
        for key in self.metrics:
            val = torch.tensor(self.metrics[key]).mean()
            metrics["rgb"]["metrics"][key] = val
        return metrics
