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
from time import time
from typing import Dict, List, Optional

import torch

from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import ObbTW
from efm3d.utils.file_utils import parse_global_name_to_id_csv
from efm3d.utils.obb_utils import MeanAveragePrecision3D

ARIA_CAM_IDS = list(range(3))
ARIA_CAM_NAMES = ["rgb", "slaml", "slamr"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class ObbMetrics(torch.nn.Module):
    """
    Metrics that directly work with our ObbTW class
    It is a torch.nn.Module to be able to behave like a torchmetrics object
    """

    def __init__(
        self,
        cam_ids=ARIA_CAM_IDS,
        cam_names=ARIA_CAM_NAMES,
        class_metrics: bool = False,
        volume_range_metrics: bool = False,
        eval_2d: bool = True,
        eval_3d: bool = False,
        ignore_bb2d_visibility: bool = False,
        global_name_to_id_file: Optional[str] = None,
        global_name_to_id: Optional[Dict] = None,
        ret_all_prec_rec: Optional[bool] = False,
        max_detection_thresholds: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            cam_ids (list): list of camera ids to evaluate
            cam_names (list): list of camera names to evaluate
            class_metrics (bool): if True, computes per-class metrics
            volume_range_metrics (bool): if True, computes volume range metrics
            eval_2d (bool): if True, evaluate 2d detections
            eval_3d (bool): if True, evaluate 3d detections
        """
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        super().__init__()
        assert (
            eval_2d or eval_3d
        ), "At least eval_2d or eval_3d needs to be set to True."
        self.eval_2d = eval_2d
        self.eval_3d = eval_3d
        self.ignore_bb2d_visibility = ignore_bb2d_visibility

        self.metric_2d = torch.nn.ModuleDict(
            {
                cam_name: MeanAveragePrecision(class_metrics=class_metrics)
                for cam_name in cam_names
            }
            if eval_2d
            else {}
        )
        bbox_area_ranges = None
        if volume_range_metrics:
            # Using category statistics from SUN_3D dataset: D42985037.
            bbox_area_ranges = {
                "all": (0, 1e5),
                "small": (0, 1e-2),  # pen, remote, toilet paper, etc.
                "medium": (1e-2, 1),  # chair, bin, monitor, etc.
                "large": (1, 1e5),  # bed, sofa, etc.
            }
        if max_detection_thresholds is None:
            # max number of detections to evaluate - 220 is sufficient for ASE scenes
            max_detection_thresholds = [220]

        self.metric_3d = torch.nn.ModuleDict(
            {
                cam_name: MeanAveragePrecision3D(
                    class_metrics=class_metrics,
                    bbox_area_ranges=bbox_area_ranges,
                    max_detection_thresholds=max_detection_thresholds,
                    ret_all_prec_rec=ret_all_prec_rec,
                )
                for cam_name in cam_names
            }
            if eval_3d
            else {}
        )
        self.cam_ids = cam_ids
        self.cam_names = cam_names
        self.cam_id_to_name = {id: name for id, name in zip(cam_ids, cam_names)}
        self.sem_id_to_name = None

        if global_name_to_id_file is not None:
            global_name_to_id = parse_global_name_to_id_csv(global_name_to_id_file)
        if global_name_to_id is not None:
            self.sem_id_to_name = {
                int(sem_id): name for name, sem_id in global_name_to_id.items()
            }

    def update(self, prediction: ObbTW, target: ObbTW, cam: Optional[CameraTW] = None):
        """ """
        for cam_id in self.cam_ids:
            if self.eval_2d:
                self.update_2d(
                    prediction.bb2(cam_id),
                    prediction.prob.squeeze(),
                    prediction.sem_id.squeeze(),
                    target.bb2(cam_id),
                    target.sem_id.squeeze(),
                    cam_id,
                )
            if self.eval_3d:
                visible_predictions_ind = prediction.visible_bb3_ind(cam_id)
                visible_targets_ind = target.visible_bb3_ind(cam_id)
                if self.ignore_bb2d_visibility:
                    visible_predictions_ind[:] = True
                    visible_targets_ind[:] = True
                    if not visible_predictions_ind.any():
                        print("WARNING: no predictions are visible")
                    if not visible_targets_ind.any():
                        print("WARNING: no targets are visible")

                # Use visible boxes in the camera for evaluation
                self.update_3d(
                    prediction.bb3corners_world[visible_predictions_ind],
                    prediction.prob[visible_predictions_ind].view(-1),
                    prediction.sem_id[visible_predictions_ind].view(-1),
                    target.bb3corners_world[visible_targets_ind],
                    target.sem_id[visible_targets_ind].view(-1),
                    cam_id,
                )

    def forward(self, prediction: ObbTW, target: ObbTW):
        self.update(prediction, target)
        return self.compute()

    def update_3d(
        self,
        pred_bb3corners: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        tgt_bb3corners: torch.Tensor,
        tgt_labels: torch.Tensor,
        cam_id: int = 0,
    ):
        assert pred_bb3corners.dim() == 3
        assert tgt_bb3corners.dim() == 3
        assert pred_scores.dim() == 1
        assert pred_labels.dim() == 1
        assert tgt_labels.dim() == 1
        p = [
            {
                "boxes": pred_bb3corners,
                "scores": pred_scores,
                "labels": pred_labels,
            }
        ]
        t = [
            {
                "boxes": tgt_bb3corners,
                "labels": tgt_labels,
            }
        ]
        self.metric_3d[self.cam_id_to_name[cam_id]].update(p, t)

    def update_2d(
        self,
        pred_bb2: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        tgt_bb2: torch.Tensor,
        tgt_labels: torch.Tensor,
        cam_id: int = 0,
    ):
        assert pred_scores.dim() == 1
        assert pred_labels.dim() == 1
        assert tgt_labels.dim() == 1
        p = [
            {
                "boxes": pred_bb2,
                "scores": pred_scores,
                "labels": pred_labels,
            }
        ]
        t = [
            {
                "boxes": tgt_bb2,
                "labels": tgt_labels,
            }
        ]
        self.metric_2d[self.cam_id_to_name[cam_id]].update(p, t)

    def update_2d_instances(
        self,
        preds,  #: List[Instances],
        tgts,  #: List[Instances],
        cam_id: int = 0,
    ):
        for pred, tgt in zip(preds, tgts):
            self.update_2d(
                pred_bb2=pred.pred_boxes.tensor,
                pred_scores=pred.scores,
                pred_labels=pred.pred_classes,
                tgt_bb2=tgt.gt_boxes.tensor,
                tgt_labels=tgt.gt_classes,
                cam_id=cam_id,
            )

    def compute(self):
        metrics = {}
        for cam_name in self.cam_names:
            if self.eval_2d:
                m2d = self.metric_2d[cam_name].compute()
                for metric_name, val in m2d.items():
                    if (
                        "small" not in metric_name
                        and "medium" not in metric_name
                        and "large" not in metric_name
                    ):
                        metrics[f"{cam_name}/{metric_name}_2D"] = val
            if self.eval_3d:
                logger.info(f"Computing metric {self.metric_3d[cam_name]}")
                t0 = time()
                m3d = self.metric_3d[cam_name].compute(self.sem_id_to_name)
                t1 = time()
                logger.info(
                    f"DONE Computing metric {self.metric_3d[cam_name]} in {t1-t0} seconds"
                )
                for metric_name, val in m3d.items():
                    metrics[f"{cam_name}/{metric_name}_3D"] = val
        return metrics

    def reset(self):
        for metric in self.metric_2d.values():
            metric.reset()
        for metric in self.metric_3d.values():
            metric.reset()
