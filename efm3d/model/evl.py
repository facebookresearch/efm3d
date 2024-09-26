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

import torch
from efm3d.aria.aria_constants import (
    ARIA_OBB_PRED,
    ARIA_OBB_PRED_PROBS_FULL,
    ARIA_OBB_PRED_PROBS_FULL_VIZ,
    ARIA_OBB_PRED_SEM_ID_TO_NAME,
    ARIA_OBB_PRED_VIZ,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.model.cnn import InvResnetFpn3d, VolumeCNNHead
from efm3d.model.lifter import VideoBackbone3d

from efm3d.model.video_backbone import VideoBackbone
from efm3d.utils.detection_utils import simple_nms3d, voxel2obb
from efm3d.utils.file_utils import parse_global_name_to_id_csv
from hydra.utils import instantiate
from omegaconf import DictConfig


class EVL(torch.nn.Module):
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
        """
        Args:
            video_backbone: 2D backbone to extract features from images.
            video_backbone3d: 3D backbone to lift 2d to 3d voxels.
            neck_hidden_dims: hidden dims of the 3D CNN neck.
            head_hidden_dim: hidden dim of the 3D CNN head.

            # obb params
            det_thresh: Detection threshold for NMS.
            yaw_max: Maximum yaw angle for object orientation.
        """
        super().__init__()

        if neck_hidden_dims is None:
            neck_hidden_dims = [64, 128, 256]

        self.backbone2d = video_backbone
        self.backbone3d = video_backbone3d
        self.head_layers = head_layers

        if isinstance(video_backbone, DictConfig):
            self.backbone2d = instantiate(video_backbone)
        if isinstance(video_backbone3d, DictConfig):
            self.backbone3d = instantiate(video_backbone3d)

        backbone3d_out_dim = self.backbone3d.output_dim()

        # 3d U-Net
        c = backbone3d_out_dim  # c = 66 (64 + 1 + 1)
        dims = [c, 64, 96, 128, 160]
        neck_final = dims[1]
        print(f"==> Init 3D InvResnetFpn3d neck with hidden layers: {dims}")
        self.neck = InvResnetFpn3d(
            dims=dims,
            num_bottles=[2, 2, 2, 2],
            strides=[1, 2, 2, 2],
            expansions=[2.0, 2.0, 2.0, 2.0],
        )

        print(
            f"==> Init 3D CNN Head with final dim = {neck_final}, hidden dim = {head_hidden_dim}"
        )
        # occpuancy head
        self.occ_head = VolumeCNNHead(
            neck_final,
            head_hidden_dim,
            final_dim=1,
            num_layers=self.head_layers,
            name="Occupancy",
        )

        # obb part
        if taxonomy_file is not None:
            taxonomy = parse_global_name_to_id_csv(taxonomy_file)
            self.sem2name = {int(sem_id): name for name, sem_id in taxonomy.items()}
        self.num_class = len(self.sem2name)

        # Centerness head (center of the bounding box).
        self.cent_head = VolumeCNNHead(
            neck_final,
            head_hidden_dim,
            final_dim=1,
            name="Centerness",
            bias=-5,
        )
        # Box size head (height, width, depth, offset_h, offset_w, offset_d, yaw rotation of box).
        self.bbox_head = VolumeCNNHead(
            neck_final,
            head_hidden_dim,
            final_dim=7,
            name="BoundingBox",
        )
        self.clas_head = VolumeCNNHead(
            neck_final,
            head_hidden_dim,
            final_dim=self.num_class,
            name="Class",
        )

        self.det_thresh = det_thresh
        self.bbox_min = 0.1  # Min bbox dim
        self.bbox_max = 6.0  # Max bbox dim
        # Scale the bbox offset max based on voxel size in meters.
        self.offset_max = 2 * self.backbone3d.voxel_meters
        self.splat_sigma = max(1, int(0.12 / self.backbone3d.voxel_meters))
        self.iou_thres = 0.2
        self.ve = self.backbone3d.voxel_extent  # voxel extent
        self.yaw_max = yaw_max
        self.scene = None

    def post_process(self, batch, out):
        cent_pr = out["cent_pr"]
        bbox_pr = out["bbox_pr"]
        clas_pr = out["clas_pr"]
        # Run NMS + convert voxel outputs to ObbTW.
        with torch.no_grad():
            # First NMS is a simple heatmap suppression.
            cent_pr_nms = simple_nms3d(cent_pr, nms_radius=self.splat_sigma + 1)
            vD, vH, vW = cent_pr.shape[-3:]
            # Convert dense predicitions to sparse ObbTW predictions.
            obbs_pr_nms, _, clas_prob_nms = voxel2obb(
                cent_pr_nms,
                bbox_pr,
                clas_pr,
                self.ve,
                top_k=128,
                thresh=self.det_thresh,
                return_full_prob=True,
            )
            out["obbs_pr_nms"] = obbs_pr_nms
            out["cent_pr_nms"] = cent_pr_nms

        # Fill these fields for efm_streamer
        # obb tracker expects ARIA_OBB_PRED and ARIA_OBB_PRED_VIZ to be in snippet coord system
        obbs_pr_nms_s = obbs_pr_nms.clone()
        T_ws = batch[ARIA_SNIPPET_T_WORLD_SNIPPET]  # B x 1 x 12
        T_wv = out["voxel/T_world_voxel"].unsqueeze(1)  # B x 1 x 12
        T_sv = T_ws.inverse() @ T_wv
        obbs_pr_nms_s = obbs_pr_nms_s.transform(T_sv)  # transform to snippet coords
        out[ARIA_OBB_PRED_SEM_ID_TO_NAME] = self.sem2name
        out[ARIA_OBB_PRED] = obbs_pr_nms_s
        out[ARIA_OBB_PRED_VIZ] = obbs_pr_nms_s
        out[ARIA_OBB_PRED_PROBS_FULL] = [item for item in clas_prob_nms]
        out[ARIA_OBB_PRED_PROBS_FULL_VIZ] = out[ARIA_OBB_PRED_PROBS_FULL]
        return out

    def forward(self, batch):
        out = {}
        # Run 2D backbone on images to get on 2D feature map per image.
        backbone2d_out_all = self.backbone2d(batch)
        for stream in ["rgb", "slaml", "slamr"]:
            if stream in backbone2d_out_all:
                # add to batch for lifter
                batch[f"{stream}/feat"] = backbone2d_out_all[stream]

        # Run explicit 3D backbone to lift 2D features to a 3D voxel grid.
        backbone3d_out = self.backbone3d(batch)
        voxel_feats = backbone3d_out["voxel/feat"]

        # Run 3D encoder-decoder CNN, acting as a "neck" to the heads.
        neck_feats1 = self.neck(voxel_feats)
        neck_feats2 = neck_feats1

        # ---------- Run the occ head ------------
        occ_logits = self.occ_head(neck_feats1)
        occ_pr = torch.sigmoid(occ_logits)  # logits => prob.
        out["occ_pr"] = occ_pr
        out["voxel_extent"] = torch.tensor(self.ve).to(occ_pr)

        # ---------- Run the obb head ------------
        # Run the centerness head.
        cent_logits = self.cent_head(neck_feats2)
        cent_pr = torch.sigmoid(cent_logits)  # logits => prob.

        # Run the box size head.
        bbox_pr = self.bbox_head(neck_feats2)
        bbox_pr[:, 0:3] = (self.bbox_max - self.bbox_min) * torch.sigmoid(
            bbox_pr[:, :3]
        ) + self.bbox_min
        bbox_pr[:, 3:6] = self.offset_max * torch.tanh(bbox_pr[:, 3:6])
        bbox_pr[:, 6] = self.yaw_max * torch.tanh(bbox_pr[:, 6])

        # Run the classification head.
        clas_pr = self.clas_head(neck_feats2)
        clas_pr = torch.nn.functional.softmax(clas_pr, dim=1)

        out.update(backbone3d_out)
        out["neck/occ_feat"] = neck_feats1
        out["neck/obb_feat"] = neck_feats2
        #  Copy data from head outputs.
        out["cent_pr"] = cent_pr
        out["bbox_pr"] = bbox_pr
        out["clas_pr"] = clas_pr

        out = self.post_process(batch, out)

        return out
