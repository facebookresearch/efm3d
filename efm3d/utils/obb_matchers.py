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

import torch
from efm3d.aria.obb import bb2_xxyy_to_xyxy, ObbTW
from efm3d.utils.obb_utils import box3d_overlap_wrapper
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from torchvision.ops.boxes import box_area

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class HungarianMatcher2d3d(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox2: float = 1,
        cost_giou2: float = 1,
        cost_bbox3: float = 1,
        cost_iou3: float = 1,
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox2 = cost_bbox2
        self.cost_bbox3 = cost_bbox3
        self.cost_giou2 = cost_giou2
        self.cost_iou3 = cost_iou3
        assert (
            cost_class != 0
            or cost_bbox2 != 0
            or cost_bbox3 != 0
            or cost_giou2 != 0
            or cost_iou3 != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward_obbs(
        self,
        prd: ObbTW,
        tgt: ObbTW,
        prd_logits=None,
        logits_is_prob: bool = False,
    ):
        if prd.ndim == 2:
            return self.forward(
                pred_logits=prd_logits.unsqueeze(0),
                pred_bb2s=prd.bb2_rgb.unsqueeze(0),
                pred_bb3s=prd.bb3corners_world.unsqueeze(0),
                pred_center_world=prd.bb3_center_world.unsqueeze(0),
                tgt_labels=[tgt.sem_id.squeeze(-1)],
                tgt_bb2s=[tgt.bb2_rgb],
                tgt_bb3s=[tgt.bb3corners_world],
                tgt_center_world=[tgt.bb3_center_world],
                logits_is_prob=logits_is_prob,
            )[0]
        elif prd.ndim == 3:
            if isinstance(tgt, ObbTW):
                tgt = tgt.remove_padding()
            return self.forward(
                pred_logits=prd_logits,
                pred_bb2s=prd.bb2_rgb,
                pred_bb3s=prd.bb3corners_world,
                pred_center_world=prd.bb3_center_world,
                tgt_labels=[tt.sem_id.squeeze(-1) for tt in tgt],
                tgt_bb2s=[tt.bb2_rgb for tt in tgt],
                tgt_bb3s=[tt.bb3corners_world for tt in tgt],
                tgt_center_world=[tt.bb3_center_world for tt in tgt],
                logits_is_prob=logits_is_prob,
            )
        else:
            raise ValueError(f"Unsupported shape {prd.shape}")

    @torch.no_grad()
    def forward(
        self,
        pred_logits=None,
        pred_bb2s=None,
        pred_center_world=None,
        pred_bb3s=None,
        tgt_labels=None,
        tgt_bb2s=None,
        tgt_center_world=None,
        tgt_bb3s=None,
        logits_is_prob: bool = False,
    ):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, snippet_frames, num_queries, num_semcls] with the classification logits
                 "pred_bb2s": Tensor of dim [batch_size, snippet_frames, num_queries, 4] with the predicted 2d box coordinates
            targets: This is a list of batch_size targets:
                 "tgt_labels": Tensor of dim [snippet_frames, num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "tgt_bb2s": Tensor of dim [snippet_frames, num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        assert pred_bb2s.dim() == 3, f"{pred_bb2s.shape}"
        assert pred_center_world.dim() == 3, f"{pred_center_world.shape}"
        B, N = pred_bb2s.shape[:2]
        assert len(tgt_bb2s) == B, "number of targets should be equal to batch size"
        assert (
            len(tgt_center_world) == B
        ), "number of targets should be equal to batch size"

        cost_class = None
        if pred_logits is not None:
            assert pred_logits.dim() == 3, f"{pred_logits.shape}"
            assert (
                len(tgt_labels) == B
            ), "number of targets should be equal to batch size"
            # We flatten to compute the cost matrices in a batch
            # [batch_size * num_queries, num_semcls]
            out_prob = pred_logits.flatten(0, 1)
            if not logits_is_prob:
                out_prob = out_prob.softmax(-1)
            tgt_ids = torch.cat(tgt_labels)
            assert tgt_ids.ndim == 1, f"{tgt_ids.shape} is not right"

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            cost_class = -out_prob[:, tgt_ids]
            if cost_class.isnan().any():
                logger.warning(
                    f"have {cost_class.isnan().sum()} nan values in cost_class"
                )
            cost_class = torch.nan_to_num(cost_class, nan=1e6)

        # [batch_size * num_queries, 4]
        pred_bb2s = pred_bb2s.flatten(0, 1)
        pred_center_world = pred_center_world.flatten(0, 1)
        # remember sizes for later
        sizes = [len(v) for v in tgt_bb2s]
        # Also concat the target boxes
        tgt_bb2s = torch.cat(tgt_bb2s)
        tgt_center_world = torch.cat(tgt_center_world)

        # Compute the L1 cost between boxes
        cost_bbox2 = torch.cdist(pred_bb2s, tgt_bb2s, p=1)
        if cost_bbox2.isnan().any():
            logger.warning(f"have {cost_bbox2.isnan().sum()} nan values in cost_bbox")
        cost_bbox2 = torch.nan_to_num(cost_bbox2, nan=1e6)
        # 3d bbs
        cost_bbox3 = torch.cdist(pred_center_world, tgt_center_world, p=1)
        if cost_bbox3.isnan().any():
            logger.warning(f"have {cost_bbox3.isnan().sum()} nan values in cost_bbox")
        cost_bbox3 = torch.nan_to_num(cost_bbox3, nan=1e6)

        # 3d bbs iou
        cost_iou3 = None
        if pred_bb3s is not None and tgt_bb3s is not None and self.cost_iou3 > 0.0:
            pred_bb3s = pred_bb3s.flatten(0, 1)
            tgt_bb3s = torch.cat(tgt_bb3s)
            cost_iou3 = -box3d_overlap_wrapper(pred_bb3s, tgt_bb3s).iou
            if cost_iou3.isnan().any():
                logger.warning(
                    f"have {cost_iou3.isnan().sum()} nan values in cost_iou3"
                )
            cost_iou3 = torch.nan_to_num(cost_iou3, nan=1e6)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            bb2_xxyy_to_xyxy(pred_bb2s), bb2_xxyy_to_xyxy(tgt_bb2s)
        )
        # set invalid costs to high value so they are not chosen in linear assignment
        # invalid predictions have size 0.0
        pred_areas = box_area(bb2_xxyy_to_xyxy(pred_bb2s))
        pred_invalid = pred_areas <= 0.0
        cost_giou[pred_invalid, :] = 1e6
        if cost_giou.isnan().any():
            logger.warning(f"have {cost_giou.isnan().sum()} nan values in cost_giou")
        cost_giou = torch.nan_to_num(cost_giou, nan=1e6)

        # Final cost matrix
        C = (
            self.cost_bbox2 * cost_bbox2
            + self.cost_bbox3 * cost_bbox3
            + self.cost_giou2 * cost_giou
        )
        if cost_class is not None:
            C = C + self.cost_class * cost_class
        if cost_iou3 is not None:
            C = C + self.cost_iou3 * cost_iou3
        C = C.view(B, N, -1).cpu()

        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(row_id, dtype=torch.int64),
                torch.as_tensor(col_id, dtype=torch.int64),
            )
            for row_id, col_id in indices
        ]
