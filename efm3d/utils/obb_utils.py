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
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from efm3d.aria.obb import ObbTW
from pytorch3d.ops.iou_box3d import (
    _box3d_overlap,
    _box_planes,
    _box_triangles,
    _check_nonzero,
)
from torch import IntTensor, Tensor
from torch.nn import functional as F
from torchmetrics.detection.mean_ap import (
    _fix_empty_tensors,
    BaseMetricResults,
    MARMetricResults,
    MeanAveragePrecision,
)
from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_8

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


@dataclass
class IouOutputs:
    vol: torch.Tensor
    iou: torch.Tensor


def input_validator_box3d(  # noqa
    preds: Sequence[Dict[str, Tensor]], targets: Sequence[Dict[str, Tensor]]
) -> None:
    """Ensure the correct input format of `preds` and `targets`"""
    if not isinstance(preds, Sequence):
        raise ValueError("Expected argument `preds` to be of type Sequence")
    if not isinstance(targets, Sequence):
        raise ValueError("Expected argument `target` to be of type Sequence")
    if len(preds) != len(targets):
        raise ValueError(
            "Expected argument `preds` and `target` to have the same length"
        )

    for k in ["boxes", "scores", "labels"]:
        if any(k not in p for p in preds):
            raise ValueError(f"Expected all dicts in `preds` to contain the `{k}` key")

    for k in ["boxes", "labels"]:
        if any(k not in p for p in targets):
            raise ValueError(f"Expected all dicts in `target` to contain the `{k}` key")

    if any(type(pred["boxes"]) is not Tensor for pred in preds):
        raise ValueError("Expected all boxes in `preds` to be of type Tensor")
    if any(type(pred["scores"]) is not Tensor for pred in preds):
        raise ValueError("Expected all scores in `preds` to be of type Tensor")
    if any(type(pred["labels"]) is not Tensor for pred in preds):
        raise ValueError("Expected all labels in `preds` to be of type Tensor")
    if any(type(target["boxes"]) is not Tensor for target in targets):
        raise ValueError("Expected all boxes in `target` to be of type Tensor")
    if any(type(target["labels"]) is not Tensor for target in targets):
        raise ValueError("Expected all labels in `target` to be of type Tensor")

    for i, item in enumerate(targets):
        if item["boxes"].size(0) != item["labels"].size(0):
            raise ValueError(
                f"Input boxes and labels of sample {i} in targets have a"
                f" different length (expected {item['boxes'].size(0)} labels, got {item['labels'].size(0)})"
            )
        if item["boxes"].shape[-2:] != (8, 3):
            raise ValueError(
                f"Input boxes of sample {i} in targets have a"
                f" wrong shape (expected (...,8, 3), got {item['boxes'].shape})"
            )
    for i, item in enumerate(preds):
        if not (
            item["boxes"].size(0) == item["labels"].size(0) == item["scores"].size(0)
        ):
            raise ValueError(
                f"Input boxes, labels and scores of sample {i} in predictions have a"
                f" different length (expected {item['boxes'].size(0)} labels and scores,"
                f" got {item['labels'].size(0)} labels and {item['scores'].size(0)})"
            )


class MAPMetricResults3D(BaseMetricResults):
    """Class to wrap the final mAP results."""

    __slots__ = (
        "map",
        "map_25",
        "map_50",
        "map_small",
        "map_medium",
        "map_large",
    )


def box3d_volume(boxes: Tensor) -> Tensor:
    """
    Computes the volume of a set of 3d bounding boxes.

    Args:
        boxes (Tensor[N, 8, 3]): 3d boxes for which the volume will be computed.

    Returns:
        Tensor[N]: the volume for each box
    """
    if boxes.numel() == 0:
        return torch.zeros(0).to(boxes)
    # Triple product to calculate volume
    a = boxes[:, 1, :] - boxes[:, 0, :]
    b = boxes[:, 3, :] - boxes[:, 0, :]
    c = boxes[:, 4, :] - boxes[:, 0, :]
    vol = torch.abs(torch.cross(a, b, dim=-1) @ c.T)[0]
    return vol


def box3d_convert(boxes: Tensor, in_fmt: str, out_fmt: str) -> Tensor:
    """
    Convert 3d box coordinate conventions.
    """
    assert in_fmt == "xyz8"
    assert out_fmt == "xyz8"
    return boxes


class MeanAveragePrecision3D(MeanAveragePrecision):
    def __init__(
        self,
        box_format: str = "xyz8",
        bbox_area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,  # compute per class metrics
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        ret_all_prec_rec: bool = False,
    ) -> None:  # type: ignore
        # Use Omni3D iOU thresholds by default
        iou_thresholds = (
            iou_thresholds
            or torch.linspace(
                0.05, 0.5, round((0.5 - 0.05) / 0.05) + 1, dtype=torch.float64
            ).tolist()
        )
        rec_thresholds = (
            rec_thresholds
            or torch.linspace(
                0.0, 1.00, round(1.00 / 0.01) + 1, dtype=torch.float64
            ).tolist()
        )
        super().__init__(
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not _TORCHVISION_GREATER_EQUAL_0_8:
            raise ModuleNotFoundError(
                "`MeanAveragePrecision` metric requires that `torchvision` version 0.8.0 or newer is installed."
                " Please install with `pip install torchvision>=0.8` or `pip install torchmetrics[detection]`."
            )
        allowed_box_formats = ["xyz8"]
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format
        max_det_thr, _ = torch.sort(IntTensor(max_detection_thresholds or [1, 10, 100]))
        self.max_detection_thresholds = max_det_thr.tolist()

        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a boolean")

        self.class_metrics = class_metrics
        # important to overwrite after the __init__() call since they are otherwise overwritten by super().__init__()
        self.bbox_area_ranges = bbox_area_ranges
        if bbox_area_ranges is None:
            self.bbox_area_ranges = {"all": (0, 1e5)}

        self.ret_all_prec_rec = ret_all_prec_rec
        self.eval_imgs = [] if self.ret_all_prec_rec else None

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:  # type: ignore
        """Add detections and ground truth to the metric.

        Args:
            preds: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 8, 3] containing `num_boxes` detection boxes of the format
                specified in the constructor. By default, this method expects
                (4) +---------+. (5)
                    | ` .     |  ` .
                    | (0) +---+-----+ (1)
                    |     |   |     |
                (7) +-----+---+. (6)|
                    ` .   |     ` . |
                    (3) ` +---------+ (2)
                box_corner_vertices = [
                    [xmin, ymin, zmin],
                    [xmax, ymin, zmin],
                    [xmax, ymax, zmin],
                    [xmin, ymax, zmin],
                    [xmin, ymin, zmax],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmax],
                    [xmin, ymax, zmax],
                ]
            - ``scores``: ``torch.FloatTensor`` of shape
                [num_boxes] containing detection scores for the boxes.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 0-indexed detection classes for the boxes.

            target: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 8, 3] containing `num_boxes` ground truth boxes of the format
                specified in the constructor.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 1-indexed ground truth classes for the boxes.

        Raises:
            ValueError:
                If ``preds`` is not of type List[Dict[str, Tensor]]
            ValueError:
                If ``target`` is not of type List[Dict[str, Tensor]]
            ValueError:
                If ``preds`` and ``target`` are not of the same length
            ValueError:
                If any of ``preds.boxes``, ``preds.scores``
                and ``preds.labels`` are not of the same length
            ValueError:
                If any of ``target.boxes`` and ``target.labels`` are not of the same length
            ValueError:
                If any box is not type float and of length 4
            ValueError:
                If any class is not type int and of length 1
            ValueError:
                If any score is not type float and of length 1
        """
        input_validator_box3d(preds, target)

        for item in preds:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box3d_convert(boxes, in_fmt=self.box_format, out_fmt="xyz8")
            if hasattr(self, "detection_boxes"):
                self.detection_boxes.append(boxes)
            else:
                self.detections.append(boxes)

            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            boxes = _fix_empty_tensors(item["boxes"])
            boxes = box3d_convert(boxes, in_fmt=self.box_format, out_fmt="xyz8")
            if hasattr(self, "groundtruth_boxes"):
                self.groundtruth_boxes.append(boxes)
            else:
                self.groundtruths.append(boxes)
            self.groundtruth_labels.append(item["labels"])

    def _compute_iou(self, id: int, class_id: int, max_det: int) -> Tensor:
        """Computes the Intersection over Union (IoU) for ground truth and detection bounding boxes for the given
        image and class.

        Args:
            id:
                Image Id, equivalent to the index of supplied samples
            class_id:
                Class Id of the supplied ground truth and detection labels
            max_det:
                Maximum number of evaluated detection bounding boxes
        """
        if hasattr(self, "detection_boxes"):
            gt = self.groundtruth_boxes[id]
            det = self.detection_boxes[id]
        else:
            gt = self.groundtruths[id]
            det = self.detections[id]
        gt_label_mask = self.groundtruth_labels[id] == class_id
        det_label_mask = self.detection_labels[id] == class_id
        if len(gt_label_mask) == 0 or len(det_label_mask) == 0:
            return Tensor([])
        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if len(gt) == 0 or len(det) == 0:
            return Tensor([])

        # Sort by scores and use only max detections
        scores = self.detection_scores[id]
        scores_filtered = scores[self.detection_labels[id] == class_id]
        inds = torch.argsort(scores_filtered, descending=True)
        det = det[inds]
        if len(det) > max_det:
            det = det[:max_det]

        # generalized_box_iou
        # both det and gt are List of "boxes"
        ious = box3d_overlap_wrapper(det, gt).iou
        return ious

    def _evaluate_image(
        self,
        id: int,
        class_id: int,
        area_range: Tuple[int, int],
        max_det: int,
        ious: dict,
    ) -> Optional[dict]:
        """Perform evaluation for single class and image.

        Args:
            id:
                Image Id, equivalent to the index of supplied samples.
            class_id:
                Class Id of the supplied ground truth and detection labels.
            area_range:
                List of lower and upper bounding box area threshold.
            max_det:
                Maximum number of evaluated detection bounding boxes.
            ious:
                IoU results for image and class.
        """
        if hasattr(self, "detection_boxes"):
            gt = self.groundtruth_boxes[id]
            det = self.detection_boxes[id]
        else:
            gt = self.groundtruths[id]
            det = self.detections[id]

        gt_label_mask = self.groundtruth_labels[id] == class_id
        det_label_mask = self.detection_labels[id] == class_id
        if len(gt_label_mask) == 0 or len(det_label_mask) == 0:
            return None
        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if len(gt) == 0 and len(det) == 0:
            return None

        areas = box3d_volume(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])

        # sort dt highest score first, sort gt ignore last
        ignore_area_sorted, gtind = torch.sort(ignore_area.to(torch.uint8))
        # Convert to uint8 temporarily and back to bool, because "Sort currently does not support bool dtype on CUDA"
        ignore_area_sorted = ignore_area_sorted.to(torch.bool)
        gt = gt[gtind]
        scores = self.detection_scores[id]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = det[dtind]
        if len(det) > max_det:
            det = det[:max_det]
        # load computed ious
        ious = (
            ious[id, class_id][:, gtind]
            if len(ious[id, class_id]) > 0
            else ious[id, class_id]
        )

        nb_iou_thrs = len(self.iou_thresholds)
        nb_gt = len(gt)
        nb_det = len(det)
        gt_matches = torch.zeros(
            (nb_iou_thrs, nb_gt), dtype=torch.bool, device=det.device
        )
        det_matches = torch.zeros(
            (nb_iou_thrs, nb_det), dtype=torch.bool, device=det.device
        )
        gt_ignore = ignore_area_sorted
        det_ignore = torch.zeros(
            (nb_iou_thrs, nb_det), dtype=torch.bool, device=det.device
        )

        if torch.numel(ious) > 0:
            for idx_iou, t in enumerate(self.iou_thresholds):
                for idx_det, _ in enumerate(det):
                    m = MeanAveragePrecision._find_best_gt_match(
                        t, gt_matches, idx_iou, gt_ignore, ious, idx_det
                    )
                    if m != -1:
                        det_ignore[idx_iou, idx_det] = gt_ignore[m]
                        det_matches[idx_iou, idx_det] = 1
                        gt_matches[idx_iou, m] = 1

        # set unmatched detections outside of area range to ignore
        det_areas = box3d_volume(det)
        det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
        ar = det_ignore_area.reshape((1, nb_det))
        det_ignore = torch.logical_or(
            det_ignore,
            torch.logical_and(
                det_matches == 0, torch.repeat_interleave(ar, nb_iou_thrs, 0)
            ),
        )
        det_matches = det_matches.cpu()
        gt_matches = gt_matches.cpu()
        scores_sorted = scores_sorted.cpu()
        gt_ignore = gt_ignore.cpu()
        det_ignore = det_ignore.cpu()

        ret = {
            "dtMatches": det_matches,
            "gtMatches": gt_matches,
            "dtScores": scores_sorted,
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore,
        }

        if self.ret_all_prec_rec:
            self.eval_imgs.append(ret)

        return ret

    def _summarize_results(
        self, precisions: Tensor, recalls: Tensor
    ) -> Tuple[MAPMetricResults3D, MARMetricResults]:
        """Summarizes the precision and recall values to calculate mAP/mAR.

        Args:
            precisions:
                Precision values for different thresholds
            recalls:
                Recall values for different thresholds
        """
        results = dict(precision=precisions, recall=recalls)
        map_metrics = MAPMetricResults3D()
        last_max_det_thr = self.max_detection_thresholds[-1]
        map_metrics.map = self._summarize(results, True, max_dets=last_max_det_thr)
        if 0.25 in self.iou_thresholds:
            map_metrics.map_25 = self._summarize(
                results, True, iou_threshold=0.25, max_dets=last_max_det_thr
            )
        if 0.5 in self.iou_thresholds:
            map_metrics.map_50 = self._summarize(
                results, True, iou_threshold=0.5, max_dets=last_max_det_thr
            )

        mar_metrics = MARMetricResults()
        for max_det in self.max_detection_thresholds:
            mar_metrics[f"mar_{max_det}"] = self._summarize(
                results, False, max_dets=max_det
            )

        if "small" in self.bbox_area_ranges:
            map_metrics.map_small = self._summarize(
                results, True, area_range="small", max_dets=last_max_det_thr
            )
            mar_metrics.mar_small = self._summarize(
                results, False, area_range="small", max_dets=last_max_det_thr
            )
        if "medium" in self.bbox_area_ranges:
            map_metrics.map_medium = self._summarize(
                results, True, area_range="medium", max_dets=last_max_det_thr
            )
            mar_metrics.mar_medium = self._summarize(
                results, False, area_range="medium", max_dets=last_max_det_thr
            )
        if "large" in self.bbox_area_ranges:
            map_metrics.map_large = self._summarize(
                results, True, area_range="large", max_dets=last_max_det_thr
            )
            mar_metrics.mar_large = self._summarize(
                results, False, area_range="large", max_dets=last_max_det_thr
            )

        return map_metrics, mar_metrics

    def compute(self, sem_id_to_name_mapping: Optional[Dict[int, str]] = None) -> dict:
        metrics = MeanAveragePrecision.compute(self)
        final_results = {}

        # resemble class-based results.
        if self.class_metrics:
            seen_classes = self._get_classes()
            if sem_id_to_name_mapping is None:
                logger.warning("No sem_id to name mapping. Falling back on id=name")
                sem_id_to_name_mapping = {
                    sem_id: str(sem_id) for sem_id in seen_classes
                }

            for k, v in metrics.items():
                # Deal with per-class metrics
                if "per_class" in k:
                    # populate per class numbers
                    mapped, unmapped = set(), set()
                    for idx, pcr in enumerate(v):
                        if seen_classes[idx] not in sem_id_to_name_mapping:
                            unmapped.add(seen_classes[idx])
                        else:
                            mapped.add(seen_classes[idx])
                            final_results[
                                f"{k}@{sem_id_to_name_mapping[seen_classes[idx]]}"
                            ] = pcr
                    if len(unmapped) > 0:
                        logger.warning(
                            f"Mapped sem_ids {mapped} but DID NOT MAP sem_ids {unmapped}"
                        )
                else:
                    final_results[k] = v
        else:
            final_results = metrics
        return final_results


def coplanar_mask(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    good = (mat1.bmm(mat2).abs() < eps).view(-1)
    return good


def nonzero_area_mask(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2
    return (face_areas > eps).all(-1)


def bb3_valid(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the box is valid
    """
    # Check that the box is not degenerate
    return nonzero_area_mask(boxes, eps) & coplanar_mask(boxes, eps)


def box3d_overlap_wrapper(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-3
) -> IouOutputs:
    """
    only compute ious and volumes for good boxes and recompose with 0s for all bad boxes.
    its better because it can handle if a subset of boxes is bad. But it costs more compute.
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")
    m1 = bb3_valid(boxes1, eps)
    m2 = bb3_valid(boxes2, eps)
    b1_good = boxes1[m1]
    b2_good = boxes2[m2]
    vol = torch.zeros(boxes1.shape[0], boxes2.shape[0], device=boxes1.device)
    iou = torch.zeros_like(vol)
    if b1_good.shape[0] == 0 or b2_good.shape[0] == 0:
        logger.info("no valid bbs returning 0 volumes and ious")
    else:
        try:
            vol_good, iou_good = _box3d_overlap.apply(b1_good, b2_good)
            m_good = m1.unsqueeze(-1) & m2.unsqueeze(0)
            vol[m_good] = vol_good.view(-1)
            iou[m_good] = iou_good.view(-1)
        except Exception:
            logger.exception("returning 0 volumes and ious because of an exception")
    return IouOutputs(vol=vol, iou=iou)


def remove_invalid_box3d(obbs: ObbTW, mark_in_place: bool = False) -> torch.Tensor:
    boxes = obbs.bb3corners_world
    assert boxes.dim() == 3
    assert (8, 3) == boxes.shape[1:]
    valid_ind, invalid_ind = [], []
    for b in range(boxes.shape[0]):
        try:
            # no need for co planarity check since our obbs are good by construction.
            # _check_coplanar(boxes[b : b + 1, :, :])
            _check_nonzero(boxes[b : b + 1, :, :])
            valid_ind.append(b)
        except Exception:
            invalid_ind.append(b)

    if mark_in_place:
        obbs._mark_invalid_ids(torch.tensor(invalid_ind, dtype=torch.long))
        return valid_ind
    return obbs[valid_ind], valid_ind


def prec_recall_bb3(
    padded_pred: ObbTW,
    padded_target: ObbTW,
    iou_thres=0.2,
    return_ious=False,
    per_class=False,
):
    """Compute precision and recall based on 3D IoU."""
    assert (
        padded_pred.ndim == 2 and padded_target.ndim == 2
    ), f"input ObbTWs must be Nx34, but got {padded_pred.shape} and {padded_target.shape}"

    pred = padded_pred.remove_padding()
    target = padded_target.remove_padding()
    pred_shape = pred.shape
    target_shape = target.shape

    pred, _ = remove_invalid_box3d(pred)
    target, _ = remove_invalid_box3d(target)
    if pred.shape != pred_shape:
        logging.warning(
            f"Warning: predicted obbs filtered from {pred_shape[0]} to {pred.shape[0]}"
        )
    if target.shape != target_shape:
        logging.warning(
            f"Warning: target obbs filtered from {target_shape[0]} to {target.shape[0]}"
        )

    prec_recall = (-1.0, -1.0, None)
    # deal with edge cases first
    if pred.shape[0] == 0:
        # invalid precision and 0 recall
        prec_recall = (-1.0, 0.0, None)
        return prec_recall
    elif target.shape[0] == 0:
        # invalid recall and 0 precision
        prec_recall = (0.0, -1.0, None)
        return prec_recall

    pred_sems = pred.sem_id
    target_sems = target.sem_id.squeeze(-1).unsqueeze(0)
    # 1. Match classes
    sem_id_match = pred_sems == target_sems
    # 2. Match IoUs
    ious = box3d_overlap_wrapper(pred.bb3corners_world, target.bb3corners_world).iou
    iou_match = ious > iou_thres
    # 3. Match both
    sem_iou_match = torch.logical_and(sem_id_match, iou_match)
    # make final matching matrix
    final_sem_iou_match = torch.zeros_like(sem_iou_match).bool()
    num_pred = sem_iou_match.shape[0]  # TP + FP
    num_target = sem_iou_match.shape[1]  # TP + FN
    # 4. Deal with the case where one prediction correspond to multiple GTs.
    # In this case, only the GT with highest IoU is considered the match.
    for pred_idx in range(int(num_pred)):
        if sem_iou_match[pred_idx, :].sum() <= 1:
            final_sem_iou_match[pred_idx, :] = sem_iou_match[pred_idx, :].clone()
        else:
            tgt_ious = ious[pred_idx, :].clone()
            tgt_ious[~sem_iou_match[pred_idx, :]] = -1.0
            sorted_ids = torch.argsort(tgt_ious, descending=True)
            tp_id = sorted_ids[0]
            # Set the pred with highest iou
            final_sem_iou_match[pred_idx, :] = False
            final_sem_iou_match[pred_idx, tp_id] = True

    # 5. Deal with the case where one GT correspond to multiple predictions.
    # In this case, if the predictions contain probabilities, we take the one with the highest score, otherwise, we take the one with the highest iou.
    for gt_idx in range(int(num_target)):
        if final_sem_iou_match[:, gt_idx].sum() <= 1:
            continue
        else:
            pred_scores = pred.prob.squeeze(-1).clone()
            if torch.all(pred_scores.eq(-1.0)):
                # go with highest iou
                pred_ious = ious[:, gt_idx].clone()
                pred_ious[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_ious, descending=True)
                tp_id = sorted_ids[0]
                # Set the pred with highest iou
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True
            else:
                # go with the highest score
                pred_scores[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_scores, descending=True)
                tp_id = sorted_ids[0]
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True

    TPs = final_sem_iou_match.any(-1)
    # precision = TP / (TP + FP) = TP / #Preds
    num_tp = TPs.sum().item()
    prec = num_tp / num_pred
    # recall = TP / (TP + FN) = TP / #GTs
    rec = num_tp / num_target

    ret = (prec, rec, final_sem_iou_match)
    if return_ious:
        ret = ret + (ious,)

    if per_class:
        # per class prec and recalls
        per_class_results = {}
        all_sems = torch.cat([pred_sems.squeeze(-1), target_sems.squeeze(0)], dim=0)
        unique_classes = torch.unique(all_sems.squeeze(-1))
        for sem_id in unique_classes:
            pred_obbs_sem = pred_sems.squeeze(-1) == sem_id
            TPs_sem = (TPs & pred_obbs_sem).sum().item()
            num_pred_sem = pred_obbs_sem.sum().item()
            gt_obbs_sem = target_sems.squeeze(0) == sem_id
            num_gt_sem = gt_obbs_sem.sum().item()
            prec_sem = TPs_sem / num_pred_sem if num_pred_sem > 0 else -1.0
            rec_sem = TPs_sem / num_gt_sem if num_gt_sem > 0 else -1.0
            per_class_results[sem_id] = {}
            per_class_results[sem_id]["num_true_positives"] = TPs_sem
            per_class_results[sem_id]["num_dets"] = num_pred_sem
            per_class_results[sem_id]["num_gts"] = num_gt_sem
            per_class_results[sem_id]["precision"] = prec_sem
            per_class_results[sem_id]["recall"] = rec_sem
        ret = ret + (per_class_results,)

    return ret


def prec_recall_curve(
    pred_gt_pairs: List[Tuple[ObbTW, ObbTW]], iou_thres=0.2, interp=True
):
    # get all probs
    probs = torch.empty(0)
    for pred, _ in pred_gt_pairs:
        pred_no_padding = pred.cpu().remove_padding()
        ps = pred_no_padding.prob.squeeze(-1)
        probs = torch.concatenate([probs, ps])

    # truncate
    probs = (probs * 100).int() / 100.0
    # combine too close probs
    probs = torch.unique(probs)
    probs = probs.tolist()
    probs.sort(reverse=True)

    precs = []
    recalls = []

    eps = 1e-6
    for prob in probs:
        tps = 0
        dets = 0
        gts = 0
        for pred, gt in pred_gt_pairs:
            pred_no_padding = pred.remove_padding()
            gt_no_padding = gt.remove_padding()
            # thresholding
            pred_no_padding = pred_no_padding[pred_no_padding.prob.squeeze(-1) >= prob]
            dets += pred_no_padding.shape[0]
            gts += gt_no_padding.shape[0]
            pred_no_padding = (
                pred_no_padding.cuda() if torch.cuda.is_available() else pred_no_padding
            )
            gt_no_padding = (
                gt_no_padding.cuda() if torch.cuda.is_available() else gt_no_padding
            )
            _, _, match_mat = prec_recall_bb3(
                pred_no_padding, gt_no_padding, iou_thres=iou_thres
            )
            if match_mat is None:
                continue
            tps += match_mat.any(-1).sum().item()
        prec = tps / (dets + eps)
        rec = tps / (gts + eps)
        precs.append(prec)
        recalls.append(rec)

    if interp:
        precs = torch.Tensor(precs)
        precs_interp = []
        for idx, _ in enumerate(precs):
            precs_interp.append(precs[idx:].max().item())
        precs = precs_interp
    return precs, recalls, probs


def draw_prec_recall_curve(
    prec: List,
    recall: List,
    save_folder: str,
    name: str = "pr_curve.png",
    iou_thres: Optional[float] = None,
):
    import matplotlib.pyplot as plt

    fig_title = "Prec-Recall Curve"
    if iou_thres is not None:
        fig_title += f" @IoU={iou_thres:.2f}"
    figure_path = os.path.join(save_folder, name)
    plt.figure(figsize=(4, 4))
    plt.title(fig_title)
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("recall")
    plt.ylabel("precision")
    # append prec recall if the last recall is not 1
    if recall[-1] != 1:
        prec.append(0)
        recall.append(recall[-1])

    plt.plot(recall, prec)
    plt.savefig(figure_path)
    print(f"Save precision recall curve to {figure_path}")
    return figure_path
