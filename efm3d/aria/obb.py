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

import logging
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from .camera import CameraTW
from .pose import IdentityPose, PAD_VAL, PoseTW, rotation_from_euler
from .tensor_wrapper import autocast, autoinit, smart_cat, smart_stack, TensorWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


# OBB corner numbering diagram for this implementation (the same as pytorch3d
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py#L111)
#
# (4) +---------+. (5)
#     | ` .     |  ` .
#     | (0) +---+-----+ (1)
#     |     |   |     |
# (7) +-----+---+. (6)|
#     ` .   |     ` . |
#     (3) ` +---------+ (2)
#
# NOTE: Throughout this implementation, we assume that boxes
# are defined by their 8 corners exactly in the order specified in the
# diagram above for the function to give correct results. In addition
# the vertices on each plane must be coplanar.
# As an alternative to the diagram, this is a unit bounding
# box which has the correct vertex ordering:
# box_corner_vertices = [
#     [0, 0, 0],  #   (0)
#     [1, 0, 0],  #   (1)
#     [1, 1, 0],  #   (2)
#     [0, 1, 0],  #   (3)
#     [0, 0, 1],  #   (4)
#     [1, 0, 1],  #   (5)
#     [1, 1, 1],  #   (6)
#     [0, 1, 1],  #   (7)
# ]

# triangle indices to draw an OBB mesh from bb3corners_*
OBB_MESH_TRI_INDS = [
    [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
    [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
    [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
]

# line indices to draw an OBB line strip frame from bb3corners_*
OBB_LINE_INDS = [0, 1, 2, 3, 0, 3, 7, 4, 0, 1, 5, 6, 5, 4, 7, 6, 2, 1, 5]

# corner indices to construct all edge lines
BB3D_LINE_ORDERS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]

_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]

DOT_EPS = 1e-3
AREA_EPS = 1e-4


class ObbTW(TensorWrapper):
    """
    Oriented 3D Bounding Box observation in world coordinates (via
    T_world_object) for Aria headsets.
    """

    @autocast
    @autoinit
    def __init__(self, data: torch.Tensor = PAD_VAL * torch.ones((1, 34))):
        assert isinstance(data, torch.Tensor)
        assert data.shape[-1] == 34
        super().__init__(data)

    @classmethod
    @autoinit
    def from_lmc(
        cls,
        bb3_object: torch.Tensor = PAD_VAL * torch.ones(6),
        bb2_rgb: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slaml: torch.Tensor = PAD_VAL * torch.ones(4),
        bb2_slamr: torch.Tensor = PAD_VAL * torch.ones(4),
        T_world_object: Union[torch.Tensor, PoseTW] = IdentityPose,  # 1x12.
        sem_id: torch.Tensor = PAD_VAL * torch.ones(1),
        inst_id: torch.Tensor = PAD_VAL * torch.ones(1),
        prob: torch.Tensor = 1 * torch.ones(1),
        moveable: torch.Tensor = 0 * torch.ones(1),
    ):
        # Concatenate into one big data tensor, handles TensorWrapper objects.
        # make sure that its on the same device (fails if IdentityPose is used)
        device = bb3_object.device
        data = smart_cat(
            [
                bb3_object,
                bb2_rgb.to(device),
                bb2_slaml.to(device),
                bb2_slamr.to(device),
                T_world_object.to(device),
                sem_id.to(device),
                inst_id.to(device),
                prob.to(device),
                moveable.to(device),
            ],
            dim=-1,
        )
        return cls(data)

    @property
    def bb3_object(self) -> torch.Tensor:
        """3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame, with shape (..., 6)."""
        return self._data[..., :6]

    @property
    def bb3_min_object(self) -> torch.Tensor:
        """3D bounding box minimum corner [xmin,ymin,zmin] in object coord frame, with shape (..., 3)."""
        return self._data[..., 0:6:2]

    @property
    def bb3_max_object(self) -> torch.Tensor:
        """3D bounding box maximum corner [xmax,ymax,zmax] in object coord frame, with shape (..., 3)."""
        return self._data[..., 1:6:2]

    @property
    def bb3_center_object(self) -> torch.Tensor:
        """3D bounding box center in object coord frame, with shape (..., 3)."""
        return 0.5 * (self.bb3_min_object + self.bb3_max_object)

    @property
    def bb3_center_world(self) -> torch.Tensor:
        """3D bounding box center in world coord frame, with shape (..., 3)."""
        s = self.bb3_center_object.shape
        _bb3_center_world = self.T_world_object.view(-1, 12).batch_transform(
            self.bb3_center_object.view(-1, 3)
        )
        return _bb3_center_world.view(s)

    @property
    def bb3_diagonal(self) -> torch.Tensor:
        """3D bounding box diagonal, with shape (..., 3)."""
        return self.bb3_max_object - self.bb3_min_object

    @property
    def bb3_volumes(self) -> torch.Tensor:
        """3D bounding box volumes, with shape (..., 1)."""
        diags = self.bb3_diagonal
        return diags.prod(dim=-1, keepdim=True)

    @property
    def bb2_rgb(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in RGB image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 6:10]

    def visible_bb3_ind(self, cam_id) -> torch.Tensor:
        """Indices of visible 3D bounding boxes in camera cam_id"""
        bb2_cam = self.bb2(cam_id)
        vis_ind = torch.all(bb2_cam > 0, dim=-1)
        return vis_ind

    @property
    def bb2_slaml(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in SLAM Left image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 10:14]

    @property
    def bb2_slamr(self) -> torch.Tensor:
        """2D bounding box [xmin,xmax,ymin,ymax] as visible in SLAM Right image, -1's if not visible, with shape (..., 4)."""
        return self._data[..., 14:18]

    def bb2(self, cam_id) -> torch.Tensor:
        """
        2D bounding box [xmin,xmax,ymin,ymax] as visible in camera with given
        cam_id, -1's if not visible, with shape (..., 4).
        cam_id == 0 for rgb
        cam_id == 1 for slam left
        cam_id == 2 for slam right
        """
        return self._data[..., 6 + cam_id * 4 : 10 + cam_id * 4]

    def set_bb2(self, cam_id, bb2d, use_mask=True):
        """
        Set 2D bounding box [xmin,xmax,ymin,ymax] in camera with given
        cam_id == 0 for rgb
        cam_id == 1 for slam left
        cam_id == 2 for slam right
        """
        padding_mask = self.get_padding_mask()
        self._data[..., 6 + cam_id * 4 : 10 + cam_id * 4] = bb2d
        if use_mask:
            self._data[padding_mask] = PAD_VAL

    def set_bb3_object(self, bb3_object, use_mask=True) -> torch.Tensor:
        """set 3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame, with shape (..., 6)."""
        padding_mask = self.get_padding_mask()
        self._data[..., :6] = bb3_object
        if use_mask:
            self._data[padding_mask] = PAD_VAL

    def set_prob(self, prob, use_mask=True):
        """Set probability score"""
        padding_mask = self.get_padding_mask()
        self._data[..., 32] = prob
        if use_mask:
            self._data[padding_mask] = PAD_VAL

    @property
    def T_world_object(self) -> torch.Tensor:
        """3D SE3 transform from object to world coords, with shape (..., 12)."""
        return PoseTW(self._data[..., 18:30])

    def get_padding_mask(self) -> torch.Tensor:
        """get boolean mask indicating which Obbs are valid/non-padded."""
        return (self._data == PAD_VAL).all(dim=-1, keepdim=False)

    def set_T_world_object(self, T_world_object: PoseTW):
        """set 3D SE3 transform from object to world coords."""
        invalid_mask = self.get_padding_mask()
        self._data[..., 18:30] = T_world_object._data
        self._data[invalid_mask] = PAD_VAL

    @property
    def sem_id(self) -> torch.Tensor:
        """semantic id, with shape (..., 1)."""
        return self._data[..., 30].unsqueeze(-1).int()

    def set_sem_id(self, sem_id: torch.Tensor):
        """set semantic id to sem_id"""
        self._data[..., 30] = sem_id.squeeze()

    @property
    def inst_id(self) -> torch.Tensor:
        """instance id, with shape (..., 1)."""
        return self._data[..., 31].unsqueeze(-1).int()

    def set_inst_id(self, inst_id: torch.Tensor):
        """set instance id to inst_id"""
        self._data[..., 31] = inst_id.squeeze()

    @property
    def prob(self) -> torch.Tensor:
        """probability of detection, with shape (..., 1)."""
        return self._data[..., 32].unsqueeze(-1)

    @property
    def moveable(self) -> torch.Tensor:
        """boolean if moveable, with shape (..., 1)."""
        return self._data[..., 33].unsqueeze(-1)

    @property
    def bb3corners_world(self) -> torch.Tensor:
        return self.T_world_object * self.bb3corners_object

    @property
    def bb3corners_object(self) -> torch.Tensor:
        """return the 8 corners of the 3D BB in object coord frame (..., 8, 3)."""
        ids = [0, 2, 4, 1, 2, 4, 1, 3, 4, 0, 3, 4, 0, 2, 5, 1, 2, 5, 1, 3, 5, 0, 3, 5]
        b3o = self.bb3_object
        c3o = b3o[..., ids]
        c3o = c3o.reshape(*c3o.shape[:-1], 8, 3)
        return c3o

    def bb3edge_pts_object(self, num_samples_per_edge: int = 10) -> torch.Tensor:
        """
        return the num_samples_per_edge points per 3D BB edge in object coord
        frame (..., num_samples_per_edge * 12, 3).

        num_samples_per_edge == 1 will result in a list of corners (with some duplicates)
        num_samples_per_edge == 2 will result in a list of corners (with some more duplicates)
        num_samples_per_edge == 3 will result in a list of corners and edge midpoints
        ...
        """
        bb3corners = self.bb3corners_object
        shape = bb3corners.shape
        alphas = torch.linspace(0, 1, num_samples_per_edge, device=bb3corners.device)
        alphas = alphas.view([1] * len(shape[:-2]) + [num_samples_per_edge, 1])
        alphas = alphas.repeat(list(shape[:-2]) + [1, 3])
        betas = torch.ones_like(alphas) - alphas
        bb3edge_pts = []
        for edge_ids in BB3D_LINE_ORDERS:
            bb3edge_pts.append(
                bb3corners[..., edge_ids[0], :].unsqueeze(-2) * betas
                + bb3corners[..., edge_ids[1], :].unsqueeze(-2) * alphas
            )
        return torch.cat(bb3edge_pts, dim=-2)

    def center(self):
        """
        Returns a ObbTW object where the 3D OBBs are centered in their local coordinate system.
        I.e. bb3_min_object == - bb3_max_object.
        """

        T_wo = self.T_world_object
        center_o = self.bb3_center_object
        # compute centered bb3_object and obb pose T_world_object
        centered_T_wo = PoseTW.from_Rt(T_wo.R, T_wo.batch_transform(center_o))
        centered_bb3_min_o = self.bb3_min_object - center_o
        centered_bb3_max_o = self.bb3_max_object - center_o
        centered_bb3_o = torch.stack(
            [
                centered_bb3_min_o[..., 0],
                centered_bb3_max_o[..., 0],
                centered_bb3_min_o[..., 1],
                centered_bb3_max_o[..., 1],
                centered_bb3_min_o[..., 2],
                centered_bb3_max_o[..., 2],
            ],
            dim=-1,
        )
        return ObbTW.from_lmc(
            bb3_object=centered_bb3_o,
            bb2_rgb=self.bb2_rgb,
            bb2_slaml=self.bb2_slaml,
            bb2_slamr=self.bb2_slamr,
            T_world_object=centered_T_wo,
            sem_id=self.sem_id,
            inst_id=self.inst_id,
            prob=self.prob,
            moveable=self.moveable,
        )

    def add_padding(self, max_elts: int = 1000) -> "ObbTW":
        """
        Adds padding to Obbs, useful for returning batches with a varying number
        of Obbs. E.g. if in one batch we have 4 Obbs and another one we have 2,
        setting max_elts=4 will add 2 pads (consisting of all -1s) to the second
        element in the batch.
        """
        assert self._data.ndim <= 2, "higher than order 2 add_padding not supported yet"
        elts = self._data
        num_to_pad = max_elts - len(elts)
        # All -1's denotes a pad element.
        pad_elt = PAD_VAL * self._data.new_ones(self._data.shape[-1])
        if num_to_pad > 0:
            rep_elts = torch.stack([pad_elt for _ in range(num_to_pad)], dim=0)
            elts = torch.cat([elts, rep_elts], dim=0)
        elif num_to_pad < 0:
            elts = elts[:max_elts]
            logger.warning(
                f"Warning: some obbs have been clipped (actual/max {len(elts)}/{max_elts}) in ObbTW.add_padding()"
            )
        return self.__class__(elts)

    def remove_padding(self) -> List["ObbTW"]:
        """
        Removes any padding by finding Obbs with all -1s. Returns a list.
        """
        assert self.ndim <= 4, "higher than order 4 remove_padding not supported yet"

        if self.ndim == 1:
            return self  # Nothing to be done in this case.

        # All -1's denotes a pad element.
        pad_elt = (PAD_VAL * self._data.new_ones(self._data.shape[-1])).unsqueeze(-2)
        is_not_pad = ~torch.all(self._data == pad_elt, dim=-1)

        if self.ndim == 2:
            new_data = self.__class__(self._data[is_not_pad])
        elif self.ndim == 3:
            B = self._data.shape[0]
            new_data = []
            for b in range(B):
                new_data.append(self.__class__(self._data[b][is_not_pad[b]]))
        else:  # self.ndim == 4:
            B, T = self._data.shape[:2]
            new_data = []
            for b in range(B):
                new_data.append([])
                for t in range(T):
                    new_data[-1].append(
                        self.__class__(self._data[b, t][is_not_pad[b, t]])
                    )
        return new_data

    def _mark_invalid(self, invalid_mask: torch.Tensor) -> "ObbTW":
        """
        in place mark obbs in this ObbTW as invalid via mask
        """
        assert invalid_mask.ndim == self.ndim - 1, "invalid_mask must match ObbTW"
        assert (
            invalid_mask.shape[:-1] == self.shape[:-1]
        ), "invalid_mask must match ObbTW"
        self._data[invalid_mask] = PAD_VAL

    def _mark_invalid_ids(self, invalid_ids: torch.Tensor) -> "ObbTW":
        """
        in place mark obbs in this ObbTW as invalid via mask
        """
        assert self.ndim == 2, "invalid_ids only supported for 2d ObbTW"
        assert invalid_ids.ndim == 1, "invalid_ids must be 1d"
        assert invalid_ids.dtype == torch.int64, "invalid_ids must be int64"
        self._data[invalid_ids] = PAD_VAL

    def num_valid(self) -> int:
        """
        Returns the number of valid Obbs in this collection.
        """
        if self.ndim == 1:
            is_pad = torch.all(self._data == PAD_VAL, dim=-1)
            return 0 if is_pad.item() else 1
        elif self.ndim == 2:
            is_pad = torch.all(self._data == PAD_VAL, dim=-1)
            return self.shape[0] - is_pad.sum()
        elif self.ndim == 3:
            is_pad = torch.all(self._data == PAD_VAL, dim=-1)
            return self.shape[0] * self.shape[1] - is_pad.sum()
        elif self.ndim == 4:
            is_pad = torch.all(self._data == PAD_VAL, dim=-1)
            return self.shape[0] * self.shape[1] * self.shape[2] - is_pad.sum()
        else:
            raise NotImplementedError(f"{self.shape}")

    def scale_bb2(self, scale_rgb: float, scale_slam: float):
        """Update the 2d bb parameters after resizing the underlying images.
        All 2d bbs are scaled by the same scale specified for the frame of the
        2d bb (RGB vs SLAM)."""

        # Check for padded values and leave those unchanged.
        pad_rgb = (
            torch.all(self.bb2_rgb == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_rgb.shape)
        )
        pad_slamr = (
            torch.all(self.bb2_slamr == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_slamr.shape)
        )
        pad_slaml = (
            torch.all(self.bb2_slaml == PAD_VAL, dim=-1)
            .unsqueeze(-1)
            .expand(*self.bb2_slaml.shape)
        )
        sc_rgb = scale_rgb * torch.ones_like(self.bb2_rgb)
        sc_slamr = scale_slam * torch.ones_like(self.bb2_slamr)
        sc_slaml = scale_slam * torch.ones_like(self.bb2_slaml)
        # If False, multiply by scale, if True multiply by 1.
        sc_rgb = torch.where(pad_rgb, torch.ones_like(sc_rgb), sc_rgb)
        sc_slamr = torch.where(pad_slamr, torch.ones_like(sc_slamr), sc_slamr)
        sc_slaml = torch.where(pad_slaml, torch.ones_like(sc_slaml), sc_slaml)

        data = smart_cat(
            [
                self.bb3_object,
                self.bb2_rgb * sc_rgb,
                self.bb2_slaml * sc_slaml,
                self.bb2_slamr * sc_slamr,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def crop_bb2(self, left_top_rgb: Tuple[float], left_top_slam: Tuple[float]):
        """Update the 2d bb parameters after cropping the underlying images.
        All 2d bbs are cropped by the same crop specified for the frame of the
        2d bb (RGB vs SLAM).
        left_top_* is assumed to be a 2D tuple of the left top corner of te crop.
        """
        # accumulate 2d bb formatting of (xmin, xmax, ymin, ymax)
        left_top_rgb = self._data.new_tensor(
            (left_top_rgb[0], left_top_rgb[0], left_top_rgb[1], left_top_rgb[1])
        )
        left_top_slam = self._data.new_tensor(
            (left_top_slam[0], left_top_slam[0], left_top_slam[1], left_top_slam[1])
        )

        # Expand the dimension if self._data is a tensor of CameraTW
        if len(self._data.shape) > 1:
            expand_dim = list(self._data.shape[:-1]) + [1]
            left_top_rgb = left_top_rgb.repeat(expand_dim)
            left_top_slam = left_top_slam.repeat(expand_dim)

        data = smart_cat(
            [
                self.bb3_object,
                self.bb2_rgb - left_top_rgb,
                self.bb2_slaml - left_top_slam,
                self.bb2_slamr - left_top_slam,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def rotate_bb2_cw(self, image_sizes: List[Tuple[int]]):
        """Update the 2d bb parameters after rotating the underlying images.
        Args:
          image_sizes: List of original image sizes before the rotation.
                       The order of the images sizes should be [(w_rgb, h_rgb), (w_slaml, h_slaml), (w_slamr, h_slamr)].
        """
        ## Early check the input input sizes
        assert (
            len(image_sizes) == 3
        ), f"the image sizes of 3 video stream should be given, but only got {len(image_sizes)}"
        for s in image_sizes:
            assert len(s) == 2

        # rotate the obbs stream by stream
        bb2_rgb_cw = rot_obb2_cw(self.bb2_rgb.clone(), image_sizes[0])
        bb2_slaml_cw = rot_obb2_cw(self.bb2_slaml.clone(), image_sizes[1])
        bb2_slamr_cw = rot_obb2_cw(self.bb2_slamr.clone(), image_sizes[2])

        data = smart_cat(
            [
                self.bb3_object,
                bb2_rgb_cw,
                bb2_slaml_cw,
                bb2_slamr_cw,
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def rectify_obb2(self, fisheye_cams: List[CameraTW], pinhole_cams: List[CameraTW]):
        rect_bb2s = []
        for idx, (fisheye_cam, pinhole_cam) in enumerate(
            zip(fisheye_cams, pinhole_cams)
        ):
            if idx == 0:  # rgb
                bb2 = self.bb2_rgb
            elif idx == 1:  # slaml
                bb2 = self.bb2_slaml
            else:  # slamr
                bb2 = self.bb2_slamr

            tl_points = bb2[..., [0, 2]].clone()  # top-left
            bl_points = bb2[..., [0, 3]].clone()  # bottom-left
            br_points = bb2[..., [1, 3]].clone()  # bottom-right
            tr_points = bb2[..., [1, 2]].clone()  # top-right
            visible_points = self.visible_bb3_ind(idx)

            tl_rays, _ = fisheye_cam.unproject(tl_points)
            br_rays, _ = fisheye_cam.unproject(br_points)
            bl_rays, _ = fisheye_cam.unproject(bl_points)
            tr_rays, _ = fisheye_cam.unproject(tr_points)

            rect_tl_pts, valid = pinhole_cam.project(tl_rays)
            rect_br_pts, valid = pinhole_cam.project(br_rays)
            rect_tl_pts, valid = pinhole_cam.project(bl_rays)
            rect_tr_pts, valid = pinhole_cam.project(tr_rays)
            rect_concat = torch.cat(
                [rect_tl_pts, rect_br_pts, rect_tl_pts, rect_tr_pts], dim=-1
            )
            xmin, _ = torch.min(rect_concat[..., 0::2], dim=-1, keepdim=True)
            xmax, _ = torch.max(rect_concat[..., 0::2], dim=-1, keepdim=True)
            ymin, _ = torch.min(rect_concat[..., 1::2], dim=-1, keepdim=True)
            ymax, _ = torch.max(rect_concat[..., 1::2], dim=-1, keepdim=True)

            # trim
            width = pinhole_cam.size.reshape(-1, 2)[0][0]
            height = pinhole_cam.size.reshape(-1, 2)[0][1]
            xmin = torch.clamp(xmin, min=0, max=width - 1)
            xmax = torch.clamp(xmax, min=0, max=width - 1)
            ymin = torch.clamp(ymin, min=0, max=height - 1)
            ymax = torch.clamp(ymax, min=0, max=height - 1)

            rect_bb2 = torch.cat([xmin, xmax, ymin, ymax], dim=-1)

            # remove the ones without any area
            areas = (rect_bb2[..., 1] - rect_bb2[..., 0]) * (
                rect_bb2[..., 3] - rect_bb2[..., 2]
            )
            areas = areas.unsqueeze(-1)
            areas = areas.repeat(*([1] * (areas.ndim - 1)), 4)
            rect_bb2[areas <= 0] = PAD_VAL
            rect_bb2[~visible_points] = PAD_VAL
            rect_bb2s.append(rect_bb2)

        data = smart_cat(
            [
                self.bb3_object,
                rect_bb2s[0],
                rect_bb2s[1],
                rect_bb2s[2],
                self.T_world_object,
                self.sem_id,
                self.inst_id,
                self.prob,
                self.moveable,
            ],
            dim=-1,
        )
        return self.__class__(data)

    def get_pseudo_bb2(
        self,
        cam: CameraTW,
        T_world_rig: PoseTW,
        num_samples_per_edge: int = 1,
        return_frac_valids: bool = False,
    ):
        """
        get the 2d bbs of the projection of the 3d bbs into all given camera view points.
        This is done by sampling points on the 3d bb edges (see
        bb3edge_pts_object), projecting them and then computing the 2d bbs from
        the valid projected points. The caller has to make sure the ObbTW has valid
        3d bbs data

        num_samples_per_edge == 1 and num_samples_per_edge == 2 are equivalent
        (in both cases we project the obb corners into the frames to compute 2d bbs)
        """
        assert self._data.shape[-2] > 0, "No valid 3d bbs data found!"
        return bb2d_from_project_bb3d(
            self, cam, T_world_rig, num_samples_per_edge, return_frac_valids
        )

    def get_bb2_heights(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        heights = bb2s[..., 3] - bb2s[..., 2]
        heights[~valid_bb2s] = -1
        return heights

    def get_bb2_widths(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        widths = bb2s[..., 1] - bb2s[..., 0]
        widths[~valid_bb2s] = -1
        return widths

    def get_bb2_areas(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        areas = (bb2s[..., 1] - bb2s[..., 0]) * (bb2s[..., 3] - bb2s[..., 2])
        areas[~valid_bb2s] = -1
        return areas

    def get_bb2_centers(self, cam_id):
        bb2s = self.bb2(cam_id)
        valid_bb2s = self.visible_bb3_ind(cam_id)
        center_x = (bb2s[..., 0:1] + bb2s[..., 1:2]) / 2.0
        center_y = (bb2s[..., 2:3] + bb2s[..., 3:4]) / 2.0
        center_2d = torch.cat([center_x, center_y], -1)
        center_2d[~valid_bb2s] = -1
        return center_2d

    def batch_points_inside_bb3(self, pts_world: torch.Tensor) -> torch.Tensor:
        """
        checks if a set of points is inside the 3d bounding box
        expected input shape is N x 3 where N is the number of points and the
        number of obbs in self.
        """
        assert pts_world.shape == self.T_world_object.t.shape
        pts_object = self.T_world_object.inverse().batch_transform(pts_world)
        inside_min = (pts_object > self.bb3_min_object).all(-1)
        inside_max = (pts_object < self.bb3_max_object).all(-1)
        return torch.logical_and(inside_min, inside_max)

    def points_inside_bb3(
        self, pts_world: torch.Tensor, scale_obb: float = 1.0
    ) -> torch.Tensor:
        """
        checks if a set of points is inside the 3d bounding box
        """
        assert self.ndim == 1 and pts_world.ndim == 2
        pts_object = self.T_world_object.inverse().transform(pts_world)
        inside_min = (pts_object > self.bb3_min_object * scale_obb).all(-1)
        inside_max = (pts_object < self.bb3_max_object * scale_obb).all(-1)
        return torch.logical_and(inside_min, inside_max)

    def _transform(self, T_new_world):
        """
        in place transform T_world_object as T_new_object = T_new_world @ T_world_object
        """
        T_world_object = self.T_world_object
        T_new_object = T_new_world @ T_world_object
        self.set_T_world_object(T_new_object)

    def transform(self, T_new_world):
        """
        transform T_world_object as T_new_object = T_new_world @ T_world_object
        """
        obb_new = self.clone()
        obb_new._transform(T_new_world)
        return obb_new

    def _transform_object(self, T_object_new):
        """
        in place transform T_world_object as T_world_new = T_world_object @ T_object_new
        """
        T_world_object = self.T_world_object
        T_world_new = T_world_object @ T_object_new
        self.set_T_world_object(T_world_new)

    def filter_by_sem_id(self, keep_sem_ids):
        valid = self._data.new_zeros(self.shape[:-1]).bool()
        for si in keep_sem_ids:
            valid = valid | (self.sem_id == si)[..., 0]
        self._data[~valid] = PAD_VAL
        return self

    def filter_by_prob(self, prob_thr: float):
        # since PAD_VAL is -1 this will work fine with padded entries
        invalid = self.prob.squeeze(-1) < prob_thr
        self._data[invalid] = PAD_VAL
        return self

    def filter_bb2_center_by_radius(self, calib, cam_id):
        """
        Inputs
            calib: CameraTW : shaped ... x 34, matching leading dims with self
            cam_id : int : integer corresponding to which bb2ds to use (0: rgb, 1: slaml, 2: slamr)
        """
        # Remove detections centers outside of valid_radius.
        centers = self.get_bb2_centers(cam_id)
        inside = calib.in_radius(centers)
        self._data[~inside, :] = PAD_VAL
        return self

    def voxel_grid(self, vD: int, vH: int, vW: int):
        """
        Input: Works on obbs shaped (B) x 34
        Output: world points sampled uniformly in a voxel grid (B) x vW*vH*vD x 3
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bb3_object.unbind(-1)
        dW = (x_max - x_min) / vW
        dH = (y_max - y_min) / vH
        dD = (z_max - z_min) / vD
        # take the center position of each voxel
        rng_x = tensor_linspace(
            x_min + dW / 2, x_max - dW / 2, steps=vW, device=self.device
        )
        rng_y = tensor_linspace(
            y_min + dH / 2, y_max - dH / 2, steps=vH, device=self.device
        )
        rng_z = tensor_linspace(
            z_min + dD / 2, z_max - dD / 2, steps=vD, device=self.device
        )
        if self.ndim > 1:
            if self.ndim > 2:
                raise NotImplementedError
            B = self.shape[0]
            xs, ys, zs = [], [], []
            for b in range(B):
                xx, yy, zz = torch.meshgrid(rng_x[b], rng_y[b], rng_z[b], indexing="ij")
                xs.append(xx)
                ys.append(yy)
                zs.append(zz)
            xx = torch.stack(xs)
            yy = torch.stack(ys)
            zz = torch.stack(zs)
        else:
            xx, yy, zz = torch.meshgrid(rng_x, rng_y, rng_z, indexing="ij")
        vox_v = torch.stack([xx, yy, zz], axis=-1)
        vox_v = vox_v.reshape(B, -1, 3)
        # vox_v = vox_v.unsqueeze(0).repeat(B, 1, 1)
        T_wv = self.T_world_object
        vox_w = T_wv * vox_v
        return vox_w

    def __repr__(self):
        return f"ObbTW {self.shape} {self.dtype} {self.device}"


def _single_transform_obbs(obbs_padded, Ts_other_world):
    assert obbs_padded.ndim == 3  # T x N x C
    assert Ts_other_world.ndim == 2 and Ts_other_world.shape[0] == 1  # 1 x C
    T, N, C = obbs_padded.shape
    if T == 0:
        # Directly return the input since T=0 and there are no obbs to transform.
        return obbs_padded
    obbs_transformed = []
    for t in range(T):
        # clone so that we get a new transformed obbs object.
        obbs = obbs_padded[t, ...].remove_padding().clone()
        obbs._transform(Ts_other_world)
        obbs_transformed.append(obbs.add_padding(N))
    obbs_transformed = ObbTW(smart_stack(obbs_transformed))
    return obbs_transformed


def _batched_transform_obbs(obbs_padded, Ts_other_world):
    assert obbs_padded.ndim == 4  # B x T x N x C
    assert Ts_other_world.ndim == 3  # T x 1 x C
    B, T, N, C = obbs_padded.shape
    obbs_transformed = []
    for b in range(B):
        obbs_transformed.append(
            _single_transform_obbs(obbs_padded[b], Ts_other_world[b])
        )
    obbs_transformed = ObbTW(smart_stack(obbs_transformed))
    return obbs_transformed


def transform_obbs(obbs_padded, Ts_other_world):
    """
    transform padded obbs from the world coordinate system to a "other"
    coordinate system.
    """
    if obbs_padded.ndim == 4:
        return _batched_transform_obbs(obbs_padded, Ts_other_world)
    return _single_transform_obbs(obbs_padded, Ts_other_world)


def rot_obb2_cw(bb2: torch.Tensor, size: Tuple[int]):
    bb2_ori = bb2.clone()
    # exchange (xmin, xmax, ymin, ymax) -> (ymax, ymin, xmin, xmax)
    bb2 = bb2[..., [3, 2, 0, 1]]
    # x_new = height - x_new
    bb2[..., 0:2] = size[1] - bb2[..., 0:2] - 1
    # bring back the invalid entries.
    bb2[bb2_ori < 0] = bb2_ori[bb2_ori < 0]
    return bb2


def project_bb3d_onto_image(
    obbs: ObbTW, cam: CameraTW, T_world_rig: PoseTW, num_samples_per_edge: int = 1
):
    """
    project 3d bb edge points into snippet images defined by T_world_rig and
    camera cam. The assumption is that obbs are in the "world" coordinate system
    of T_world_rig.
    Supports batched operation.

    Args:
        obbs (ObbTW): obbs to project; shape is (Bx)(Tx)Nx34
        cam (CameraTW): camera to project to; shape is (Bx)TxC where T is the snippet dimension;
        T_world_rig (PoseTW): T_world_rig defining where the camera rig is; shape is (Bx)Tx12
        num_samples_per_edge (int): how many points to sample per edge to
            compute 2d bb (1, and 2 means only corners)
    Returns:
        bb3_corners_im (Tensor): bb3 corners in the image coordinate system; shape is (Bx)TxNx8x2
        bb3_valids (Tensor): valid indices of bb3_corners_im (indicates which
            corners lie within the images); shape is (Bx)TxNx8
    """
    obb_dim = obbs.dim()
    # support 3 sets of input shapes
    if obb_dim == 2:  # Nx34
        # cam: TxC, T_world_rig: Tx12
        assert (
            cam.dim() == 2
            and T_world_rig.dim() == 2
            and cam.shape[0]
            == T_world_rig.shape[0]  # T dim should be the same for cam and T_world_rig
        ), f"Unsupported input shapes: obb: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}."

        # To the consistent shapes
        obbs = obbs.unsqueeze(0).unsqueeze(0)  # expand to B(1)xT(1)xNx34
        cam = cam[None, ...]  # expand to B(1)xTxC
        T_world_rig = T_world_rig[None, ...]  # expand to B(1)xTx12
        B, T = cam.shape[0:2]
        N = obbs.shape[-2]
        obbs = obbs.expand(B, T, *obbs.shape[-2:])  # repeat to real T: B(1)xTxNx34

    elif obb_dim == 3:  # BxNx34
        # cam: BxTxC, T_world_rig: BxTx12
        assert cam.dim() == 3 and T_world_rig.dim() == 3
        # B dim should be the same
        assert obbs.shape[0] == cam.shape[0] and obbs.shape[0] == T_world_rig.shape[0]
        # T dim of cam and pose should be the same
        assert cam.shape[1] == T_world_rig.shape[1]

        # To the consistent shapes
        obbs = obbs.unsqueeze(1)  # expand to BxT(1)xNx34
        B, T = cam.shape[0:2]
        obbs = obbs.expand(B, T, *obbs.shape[-2:])

    elif obb_dim == 4:  # BxTxNx34
        pass
    else:
        raise ValueError(
            f"Unsupported input shapes: obb: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}."
        )

    # check if all tensors are of correct shapes.
    assert (
        obbs.dim() == 4 and cam.dim() == 3 and T_world_rig.dim() == 3
    ), f"The shapes of obbs, cam and T_world_rig should be BxTxNx34, BxTxC, and BxTx12, respectively. However, we got obbs: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}"
    assert (
        obbs.shape[0:2] == cam.shape[0:2] and obbs.shape[0:2] == T_world_rig.shape[0:2]
    ), f"The BxT dims should be the same for all tensors, but got obbs: {obbs.shape}, cam: {cam.shape}, T_world_rig: {T_world_rig.shape}"

    B, T = cam.shape[0:2]
    N = obbs.shape[-2]
    assert N > 0, "obbs have to exist for this frame"
    # Get pose of camera.
    T_world_cam = T_world_rig @ cam.T_camera_rig.inverse()
    # Project the 3D BB corners into the image.
    # BxTxNx8x3 -> BxTxN*8x3
    if num_samples_per_edge <= 2:
        bb3pts_world = obbs.bb3corners_world.view(B, T, -1, 3)
    else:
        bb3pts_object = obbs.bb3edge_pts_object(num_samples_per_edge)
        bb3pts_world = obbs.T_world_object * bb3pts_object
        bb3pts_world = bb3pts_world.view(B, T, -1, 3)
    Npt = bb3pts_world.shape[2]
    T_world_cam = T_world_cam.unsqueeze(2).repeat(1, 1, Npt, 1)
    bb3pts_cam = (
        T_world_cam.inverse()
        .view(-1, 12)
        .batch_transform(bb3pts_world.view(-1, 3))
        .view(B, T, -1, 3)
    )
    bb3pts_im, bb3pts_valids = cam.project(bb3pts_cam)
    bb3pts_im = bb3pts_im.view(B, T, N, -1, 2)
    bb3pts_valids = bb3pts_valids.detach().view(B, T, N, -1)

    if obb_dim == 2:
        # remove B dim if it didn't exist before.
        bb3pts_im = bb3pts_im.squeeze(0)
        bb3pts_valids = bb3pts_valids.squeeze(0)
    return bb3pts_im, bb3pts_valids


def bb2d_from_project_bb3d(
    obbs: ObbTW,
    cam: CameraTW,
    T_world_rig: PoseTW,
    num_samples_per_edge: int = 1,
    return_frac_valids: bool = False,
):
    """
    get 2d bbs around the 3d bb corners of obbs projected into the image coordinate system
    defined by T_world_rig and camera cam. The assumption is that obbs are in the
    "world" coordinate system of T_world_rig.

    This is done by sampling points on the 3d bb edges (see bb3edge_pts_object),
    projecting them and then computing the 2d bbs from the valid projected
    points.

    Supports batched operation.

    Args:
        obbs (ObbTW): obbs to project; shape is (Bx)Nx34
        cam (CameraTW): camera to project to; shape is (Bx)TxC where T is the snippet dimension;
        T_world_rig (PoseTW): T_world_rig defining where the camera rig is; shape is (Bx)Tx12
    Returns:
        bb2s (Tensor): 2d bounding boxes in the image coordinate system; shape is (Bx)TxNx4
        bb2s_valid (Tensor): valid indices of bb2s; shape is (Bx)TxN
    """
    from torchvision.ops.boxes import box_iou

    bb3corners_im, bb3corners_valids = project_bb3d_onto_image(
        obbs, cam, T_world_rig, num_samples_per_edge
    )
    # get image points that will min and max reduce correctly given the valid masks
    bb3corners_im_min = torch.where(
        bb3corners_valids.unsqueeze(-1).expand_as(bb3corners_im),
        bb3corners_im,
        999999 * torch.ones_like(bb3corners_im),
    )
    bb3corners_im_max = torch.where(
        bb3corners_valids.unsqueeze(-1).expand_as(bb3corners_im),
        bb3corners_im,
        -999999 * torch.ones_like(bb3corners_im),
    )
    # compute 2d bounding boxes
    bb2s_min = torch.min(bb3corners_im_min, dim=-2)[0]
    bb2s_max = torch.max(bb3corners_im_max, dim=-2)[0]
    bb2s = torch.stack(
        [bb2s_min[..., 0], bb2s_max[..., 0], bb2s_min[..., 1], bb2s_max[..., 1]], dim=-1
    )
    # min < max so that it's a valid box.
    non_empty_boxes = (bb2s[..., 0] < bb2s[..., 1]) & (bb2s[..., 2] < bb2s[..., 3])
    if cam.is_linear:
        bb2s_full = bb2s.clone()
        # Clamp based on the camera size for linear cameras.
        # Note that this could generate very big/loose bounding boxes if the object is badly truncated due to out of view.
        bb2s[..., 0:2] = torch.clamp(
            bb2s[..., 0:2], min=0, max=cam.size.view(-1, 2)[0, 0] - 1
        )
        bb2s[..., 2:4] = torch.clamp(
            bb2s[..., 2:4], min=0, max=cam.size.view(-1, 2)[0, 1] - 1
        )
        # filter out empty boxes.
        bb2s_valid = torch.logical_and(non_empty_boxes, bb3corners_valids.any(-1))
        if return_frac_valids:
            frac_valid = torch.zeros_like(bb2s_valid).float()
            frac_valid[non_empty_boxes] = box_iou(
                bb2_xxyy_to_xyxy(bb2s[non_empty_boxes]),
                bb2_xxyy_to_xyxy(bb2s_full[non_empty_boxes]),
            ).diagonal()
    else:
        # count number of valid points
        num_points = bb3corners_valids.count_nonzero(-1)
        # valid 2d bbs are non-empty and have at least 1/6 of the edge sample
        # points in the valid image region
        bb2s_valid = torch.logical_and(
            non_empty_boxes, num_points > num_samples_per_edge * 2
        )
        if return_frac_valids:
            frac_valid = num_points / bb3corners_valids.shape[-1]
            frac_valid[~non_empty_boxes] = 0.0
    if return_frac_valids:
        return bb2s, bb2s_valid, frac_valid
    return bb2s, bb2s_valid


def bb2_xxyy_to_xyxy(bb2s):
    # check if the input is xxyy
    is_xxyy = torch.logical_and(
        bb2s[..., 0] <= bb2s[..., 1], bb2s[..., 2] <= bb2s[..., 3]
    )
    is_xxyy = is_xxyy.all()
    if not is_xxyy:
        logger.warning("Input 2d bbx doesn't follow xxyy convention.")
    return bb2s[..., [0, 2, 1, 3]]


def bb2_xyxy_to_xxyy(bb2s):
    # check if the input is xxyy
    is_xyxy = torch.logical_and(
        bb2s[..., 0] <= bb2s[..., 2], bb2s[..., 1] <= bb2s[..., 3]
    )
    is_xyxy = is_xyxy.all()
    if not is_xyxy:
        logger.warning("Input 2d bbx doesn't follow xyxy convention.")
    return bb2s[..., [0, 2, 1, 3]]


def bb3_xyzxyz_to_xxyyzz(bb3s):
    """
    take bb3 in xyzxyz format and return xxyyzz format.
    """
    return bb3s[..., [0, 3, 1, 4, 2, 5]]


def bb3_xyz_xyz_to_xxyyzz(bb3s_min, bb3s_max):
    """
    take min and max points of the bb3 and return xxyyzz format
    """
    return torch.cat([bb3s_min, bb3s_max], -1)[..., [0, 3, 1, 4, 2, 5]]


def rnd_obbs(N: int = 1, num_semcls: int = 10, bb3_min_diag=0.1, bb2_min_diag=10):
    pts3_min = torch.randn(N, 3)
    pts3_max = pts3_min + bb3_min_diag + torch.randn(N, 3).abs()
    pts2_min = torch.randn(N, 2)
    pts2_max = pts2_min + bb2_min_diag + torch.randn(N, 2).abs()

    obb = ObbTW.from_lmc(
        bb3_object=bb3_xyzxyz_to_xxyyzz(torch.cat([pts3_min, pts3_max], -1)),
        prob=torch.ones(N),
        bb2_rgb=bb2_xyxy_to_xxyy(torch.cat([pts2_min, pts2_max], -1)),
        sem_id=torch.randint(low=0, high=num_semcls - 1, size=[N]),
        T_world_object=PoseTW.from_aa(torch.randn(N, 3), 10.0 * torch.randn(N, 3)),
    )
    return obb


def obb_time_union(obbs, pad_size=128):
    """
    Take frame level ground truth shaped BxTxNxC and take the union
    over the time dimensions using the instance id to extend to snippet level
    obbs shaped BxNxC.
    """
    # T already merged somewhere else.
    if obbs.ndim == 3:
        return obbs

    assert obbs.ndim == 4, "Only B x T x N x C supported"
    new_obbs = []
    for obb in obbs:
        new_obb = []
        flat_time_obb = obb.clone().reshape(-1, 34)
        unique = flat_time_obb.inst_id.unique()
        for uni in unique:
            if uni == PAD_VAL:
                continue
            found = int(torch.argwhere(flat_time_obb.inst_id == uni)[0, 0])
            found_obb = flat_time_obb[found].clone()
            new_obb.append(found_obb)
        if len(new_obb) == 0:
            print(f"Adding empty OBB in time_union {obbs.shape}")
            new_obb.append(ObbTW().reshape(-1).to(obbs._data))
        new_obbs.append(torch.stack(new_obb).add_padding(pad_size))
    new_obbs = torch.stack(new_obbs)
    # Remove all bb2 observations since we no longer know which frame in time it came from.
    # Note: we set the visibility for the merged obbs in order to do the evaluation on those theses obbs.
    pad_mask = new_obbs.get_padding_mask()
    new_obbs.set_bb2(cam_id=0, bb2d=1)
    new_obbs.set_bb2(cam_id=1, bb2d=1)
    new_obbs.set_bb2(cam_id=2, bb2d=1)
    new_obbs._data[pad_mask] = PAD_VAL
    return new_obbs


def obb_filter_outside_volume(obbs, T_ws, T_wv, voxel_extent, border=0.1):
    """
    Remove obbs outside a volume of size voxel_extent, e.g. from a lifter volume.
    Obbs are filtered based on their center point being inside the volume, and
    are additionally filtered near the border.
    """
    assert obbs.ndim == 3, "Only B x N x C supported"
    T_vs = T_wv.inverse() @ T_ws
    obbs_v = obbs.transform(T_vs.unsqueeze(1))
    centers_v = obbs_v.bb3_center_world
    cx = centers_v[:, :, 0]
    cy = centers_v[:, :, 1]
    cz = centers_v[:, :, 2]
    x_min = voxel_extent[0]
    x_max = voxel_extent[1]
    y_min = voxel_extent[2]
    y_max = voxel_extent[3]
    z_min = voxel_extent[4]
    z_max = voxel_extent[5]
    valid = (obbs_v.inst_id != PAD_VAL).squeeze(-1)
    inside = (
        (cx > (x_min + border))
        & (cy > (y_min + border))
        & (cz > (z_min + border))
        & (cx < (x_max - border))
        & (cy < (y_max - border))
        & (cz < (z_max - border))
    )
    remove = valid & ~inside
    obbs._data[remove, :] = PAD_VAL
    return obbs


def tensor_linspace(start, end, steps, device):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=device).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=device).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def make_obb(sz, position, prob=1.0, roll=0.0, pitch=0.0, yaw=0.1):
    e_angles = torch.tensor([roll, pitch, yaw]).reshape(-1, 3)
    R = rotation_from_euler(e_angles).reshape(3, 3)
    T_voxel_object = PoseTW.from_Rt(R, torch.tensor(position))
    bb3 = [
        -sz[0] / 2.0,
        sz[0] / 2.0,
        -sz[1] / 2.0,
        sz[1] / 2.0,
        -sz[2] / 2.0,
        sz[2] / 2.0,
    ]
    return ObbTW.from_lmc(
        bb3_object=torch.tensor(bb3),
        prob=[prob],
        T_world_object=T_voxel_object,
    )


# =====> Main function for 3D IoU computation. <=======
def obb_iou3d(obb1: ObbTW, obb2: ObbTW, samp_per_dim=32):
    """
    Computes the intersection of two boxes by sampling points uniformly in
    x,y,z dims.

    samp_per_dim: int, number of samples per dimension, e.g. if 8, then 8x8x8
                       increase for more accuracy but less speed
                       8: fast but not so accurate
                       32: medium
                       128: most accurate but slow
    """
    assert obb1.ndim == 2
    assert obb2.ndim == 2

    B1 = obb1.shape[0]
    B2 = obb2.shape[0]
    vol1 = obb1.bb3_volumes
    vol2 = obb2.bb3_volumes

    dim = samp_per_dim
    points1_w = obb1.voxel_grid(vD=dim, vH=dim, vW=dim)
    points2_w = obb2.voxel_grid(vD=dim, vH=dim, vW=dim)
    num_samples = points1_w.shape[1]

    isin21 = is_point_inside_box(points2_w, obb1.bb3corners_world, verbose=True)
    num21 = isin21.sum(dim=-1)
    isin12 = is_point_inside_box(points1_w, obb2.bb3corners_world, verbose=True)
    num12 = isin12.sum(dim=-1)

    inters12 = vol1.view(B1, 1) * num12.view(B1, B2)
    inters21 = vol2.view(B2, 1) * num21.view(B2, B1)
    inters = (inters12 + inters21.transpose(1, 0)) / 2.0
    union = (vol1.view(B1, 1) * num_samples) + (vol2.view(1, B2) * num_samples) - inters
    iou = inters / union
    return iou


def is_point_inside_box(points: torch.Tensor, box: torch.Tensor, verbose=False):
    """
    Determines whether points are inside the boxes
    Args:
        points: tensor of shape (B1, P, 3) of the points
        box: tensor of shape (B2, 8, 3) of the corners of the boxes
    Returns:
        inside: bool tensor of whether point (row) is in box (col) shape (B1, B2, P)
    """
    device = box.device
    B1 = points.shape[0]
    B2 = box.shape[0]
    P = points.shape[1]

    normals = box_planar_dir(box)  # (B2, 6, 3)
    box_planes = get_plane_verts(box)  # (B2, 6, 4, 3)
    NP = box_planes.shape[1]  # = 6

    # a point p is inside the box if it "inside" all planes of the box
    # so we run the checks
    ins = torch.zeros((B1, B2, P, NP), device=device, dtype=torch.bool)
    # ins = []
    for i in range(NP):
        is_in = is_inside(points, box_planes[:, i], normals[:, i])
        ins[:, :, :, i] = is_in
        # ins.append(is_in)
    # ins = torch.stack(ins, dim=-1)

    ins = ins.all(dim=-1)
    return ins


def box_planar_dir(
    box: torch.Tensor, dot_eps: float = DOT_EPS, area_eps: float = AREA_EPS
) -> torch.Tensor:
    """
    Finds the unit vector n which is perpendicular to each plane in the box
    and points towards the inside of the box.
    The planes are defined by `_box_planes`.
    Since the shape is convex, we define the interior to be the direction
    pointing to the center of the shape.
    Args:
       box: tensor of shape (B, 8, 3) of the vertices of the 3D box
    Returns:
       n: tensor of shape (B, 6) of the unit vector orthogonal to the face pointing
          towards the interior of the shape
    """
    assert box.shape[1] == 8 and box.shape[2] == 3
    # center point of each box
    box_ctr = box.mean(dim=1).view(-1, 1, 3)
    # box planes
    plane_verts = get_plane_verts(box)  # (B, 6, 4, 3)
    v0, v1, v2, v3 = plane_verts.unbind(2)
    plane_ctr, n = get_plane_center_normal(plane_verts)
    # Check all verts are coplanar
    normv = F.normalize(v3 - v0, dim=-1).unsqueeze(2).reshape(-1, 1, 3)
    nn = n.unsqueeze(3).reshape(-1, 3, 1)
    dists = normv @ nn
    if not (dists.abs() < dot_eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)
    # Check all faces have non zero area
    area1 = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    area2 = torch.cross(v3 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    if (area1 < area_eps).any().item() or (area2 < area_eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)
    # We can write:  `box_ctr = plane_ctr + a * e0 + b * e1 + c * n`, (1).
    # With <e0, n> = 0 and <e1, n> = 0, where <.,.> refers to the dot product,
    # since that e0 is orthogonal to n. Same for e1.
    """
    # Below is how one would solve for (a, b, c)
    # Solving for (a, b)
    numF = verts.shape[0]
    A = torch.ones((numF, 2, 2), dtype=torch.float32, device=device)
    B = torch.ones((numF, 2), dtype=torch.float32, device=device)
    A[:, 0, 1] = (e0 * e1).sum(-1)
    A[:, 1, 0] = (e0 * e1).sum(-1)
    B[:, 0] = ((box_ctr - plane_ctr) * e0).sum(-1)
    B[:, 1] = ((box_ctr - plane_ctr) * e1).sum(-1)
    ab = torch.linalg.solve(A, B)  # (numF, 2)
    a, b = ab.unbind(1)
    # solving for c
    c = ((box_ctr - plane_ctr - a.view(numF, 1) * e0 - b.view(numF, 1) * e1) * n).sum(-1)
    """
    # Since we know that <e0, n> = 0 and <e1, n> = 0 (e0 and e1 are orthogonal to n),
    # the above solution is equivalent to
    direc = F.normalize(box_ctr - plane_ctr, dim=-1)  # (6, 3)
    c = (direc * n).sum(-1)
    # If c is negative, then we revert the direction of n such that n points "inside"
    negc = c < 0.0
    n[negc] *= -1.0
    # c[negc] *= -1.0
    # Now (a, b, c) is the solution to (1)
    return n


def get_plane_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the planes of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (B, 8, 3)
    Returns:
        plane_verts: tensor of shape (B, 6, 4, 3)
    """
    device = box.device
    B = box.shape[0]
    faces = torch.tensor(_box_planes, device=device, dtype=torch.int64)  # (6, 4)
    plane_verts = torch.stack([box[b, faces] for b in range(B)])  # (B, 6, 4, 3)
    return plane_verts


def is_inside(
    points: torch.Tensor,
    plane: torch.Tensor,
    normal: torch.Tensor,
    return_proj: bool = True,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |

    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      points: tensor of shape (B1, P, 3) of coordinates of a point
      plane: tensor of shape (B2, 4, 3) of vertices of a box plane
      normal: tensor of shape (B2, 3) of the unit "inside" direction on the plane
      return_proj: bool whether to return the projected point on the plane
    Returns:
      is_inside: bool of shape (B2, P) of whether point is inside
    """
    device = plane.device
    assert plane.ndim == 3
    assert normal.ndim == 2
    assert points.ndim == 3
    assert points.shape[2] == 3
    B1 = points.shape[0]
    B2 = plane.shape[0]
    P = points.shape[1]
    v0, v1, v2, v3 = plane.unbind(dim=1)
    plane_ctr = plane.mean(dim=1)
    e0 = F.normalize(v0 - plane_ctr, dim=1)
    e1 = F.normalize(v1 - plane_ctr, dim=1)

    dot1 = (e0.unsqueeze(1) @ normal.unsqueeze(2)).reshape(B2)
    if not torch.allclose(dot1, torch.zeros((B2,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")
    dot2 = (e1.unsqueeze(1) @ normal.unsqueeze(2)).reshape(B2)
    if not torch.allclose(dot2, torch.zeros((B2,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")

    # Every point p can be written as p = ctr + a e0 + b e1 + c n
    # solving for c
    # c = (point - ctr - a * e0 - b * e1).dot(n)
    pts = points.view(B1, 1, P, 3)
    ctr = plane_ctr.view(1, B2, 1, 3)
    e0 = e0.view(1, B2, 1, 3)
    e1 = e1.view(1, B2, 1, 3)
    normal = normal.view(1, B2, 1, 3)

    direc = torch.sum((pts - ctr) * normal, dim=-1)
    ins = direc >= 0.0
    return ins


def get_plane_center_normal(planes: torch.Tensor) -> torch.Tensor:
    """
    Returns the center and normal of planes
    Args:
        planes: tensor of shape (B, P, 4, 3)
    Returns:
        center: tensor of shape (B, P, 3)
        normal: tensor of shape (B, P, 3)
    """
    B = planes.shape[0]

    add_dim1 = False
    if planes.ndim == 3:
        planes = planes.unsqueeze(1)
        add_dim1 = True

    ctr = planes.mean(dim=2)  # (B, P, 3)
    normals = torch.zeros_like(ctr)

    v0, v1, v2, v3 = planes.unbind(dim=2)  # 4 x (B, P, 3)

    P = planes.shape[1]
    for t in range(P):
        ns = torch.zeros((B, 6, 3), device=planes.device)
        ns[:, 0] = torch.cross(v0[:, t] - ctr[:, t], v1[:, t] - ctr[:, t], dim=-1)
        ns[:, 1] = torch.cross(v0[:, t] - ctr[:, t], v2[:, t] - ctr[:, t], dim=-1)
        ns[:, 2] = torch.cross(v0[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ns[:, 3] = torch.cross(v1[:, t] - ctr[:, t], v2[:, t] - ctr[:, t], dim=-1)
        ns[:, 4] = torch.cross(v1[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ns[:, 5] = torch.cross(v2[:, t] - ctr[:, t], v3[:, t] - ctr[:, t], dim=-1)
        ii = torch.argmax(torch.norm(ns, dim=-1), dim=-1)
        normals[:, t] = ns[torch.arange(B), ii]

    if add_dim1:
        ctr = ctr[:, 0]
        normals = normals[:, 0]
    normals = F.normalize(normals, dim=-1)
    return ctr, normals
