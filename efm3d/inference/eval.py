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

import numpy as np
import torch
from efm3d.utils.obb_csv_writer import ObbCsvReader
from efm3d.utils.obb_metrics import ObbMetrics
from efm3d.utils.obb_utils import (
    draw_prec_recall_curve,
    prec_recall_bb3,
    prec_recall_curve,
)


def check_sem_id_conflict(ids_pred, ids_gt):
    all_sem_ids = set(list(ids_pred.keys()) + list(ids_gt.keys()))
    for sem_id in all_sem_ids:
        if sem_id in ids_pred and sem_id in ids_gt:
            assert (
                ids_pred[sem_id] == ids_gt[sem_id]
            ), f"Mismatch id to name for sem id {sem_id}, {ids_pred[sem_id]} in pred but {ids_gt[sem_id]} in GT"
        elif sem_id not in ids_pred:
            print(f"sem_id {sem_id} not found in pred")
        else:
            print(f"sem_id {sem_id} not found in GT")


def evaluate_obb_csv(
    pred_csv: str,
    gt_csv: str,
    iou: float = 0.2,
    pr_curve: bool = False,
):
    pred_reader = ObbCsvReader(pred_csv)
    gt_reader = ObbCsvReader(gt_csv)
    pred_obbs = pred_reader.obbs
    gt_obbs = gt_reader.obbs

    sem_id_to_name = pred_reader.sem_ids_to_names.copy()
    sem_id_to_name_gt = gt_reader.sem_ids_to_names.copy()
    check_sem_id_conflict(sem_id_to_name, sem_id_to_name_gt)
    sem_id_to_name.update(sem_id_to_name_gt)

    result = {}
    mAP = ObbMetrics(
        cam_ids=[0],
        cam_names=["rgb"],
        class_metrics=True,
        eval_2d=False,
        eval_3d=True,
        global_name_to_id={
            name: int(sem_id) for sem_id, name in sem_id_to_name.items()
        },
    )

    ts = list(pred_obbs.keys()) + list(gt_obbs.keys())
    ts = list(set(ts))
    ts.sort()

    gt_ts_miss = 0
    pred_ts_miss = 0
    for t in ts:
        if t not in pred_obbs:
            print(f"pred obbs not found for {t}")
            pred_ts_miss += 1
            continue
        if t not in gt_obbs:
            print(f"gt obbs not found for {t}")
            gt_ts_miss += 1
            continue

        # we should not have any paddings
        assert pred_obbs[t].shape[0] == pred_obbs[t].remove_padding().shape[0]
        assert gt_obbs[t].shape[0] == gt_obbs[t].remove_padding().shape[0]

        # always do precision recall calculation
        prec, rec, match_mat, ious, per_class_results = prec_recall_bb3(
            pred_obbs[t],
            gt_obbs[t],
            iou_thres=iou,
            return_ious=True,
            per_class=True,
        )
        tps = match_mat.any(-1)
        fps = (~match_mat).all(-1)
        result[f"precision@IoU{iou}"] = float(prec)
        result[f"recall@IoU{iou}"] = float(rec)
        result[f"num_true_positives@IoU{iou}"] = int(tps.sum())
        result["num_dets"] = match_mat.shape[0]
        result["num_gts"] = match_mat.shape[1]
        for sem_id, per_class_result in per_class_results.items():
            result[f"precision@IoU{iou}@Class_{sem_id_to_name[sem_id.item()]}"] = float(
                per_class_result["precision"]
            )
            result[f"recall@IoU{iou}@Class_{sem_id_to_name[sem_id.item()]}"] = float(
                per_class_result["recall"]
            )
        # check if the preds contain probabilities
        prob = pred_obbs[t].prob.squeeze()
        assert not torch.all(
            prob.eq(-1.0)
        ), "the obbs don't contain valid probabilities for mAP calculation."
        # add pred/gt pair to mAP calculator.
        mAP.update(pred_obbs[t], gt_obbs[t])

        output_dir = os.path.dirname(pred_csv)
        if pr_curve and len(ts) == 1:
            precs, recalls, probs = prec_recall_curve([(pred_obbs[t], gt_obbs[t])])
            draw_prec_recall_curve(
                precs, recalls, save_folder=output_dir, iou_thres=iou
            )

    result["num_timestamps"] = len(ts)
    result["num_timestamp_miss_pred"] = pred_ts_miss
    result["num_timestamp_miss_gt"] = gt_ts_miss

    result_map = mAP.compute()
    # ignore average recall
    result_map = {
        k: v.item() for k, v in result_map.items() if not k.startswith("rgb/mar_")
    }
    result.update(result_map)
    return result


def obb_eval_dataset(input_folder: str, iou: float = 0.2):
    """
    Obb eval at dataset-level
    """

    GT_OBB_FILENAME = "gt_scene_obbs.csv"
    PRED_OBB_FILENAME = "tracked_scene_obbs.csv"

    # get all the pred and gt csv files
    pred_csv_paths, gt_csv_paths = [], []
    filenames = os.listdir(input_folder)
    dirs = [os.path.join(input_folder, f) for f in filenames]
    dirs = [d for d in dirs if os.path.isdir(d)]
    for d in dirs:
        pred_csv = os.path.join(d, PRED_OBB_FILENAME)
        gt_csv = os.path.join(d, GT_OBB_FILENAME)
        if os.path.exists(gt_csv) and os.path.exists(pred_csv):
            pred_csv_paths.append(pred_csv)
            gt_csv_paths.append(gt_csv)

    result = {}
    result["num_seqs"] = len(pred_csv_paths)
    if len(pred_csv_paths) == 0 or len(gt_csv_paths) == 0:
        return result

    pred_obbs, gt_obbs = [], []
    sem_id_to_name = {}

    for pred_csv, gt_csv in zip(pred_csv_paths, gt_csv_paths):
        pred_reader = ObbCsvReader(pred_csv)
        gt_reader = ObbCsvReader(gt_csv)
        p_obbs = pred_reader.obbs
        g_obbs = gt_reader.obbs
        # p_obbs, g_obbs are single-item dicts
        p_obbs = next(iter(p_obbs.values()))
        g_obbs = next(iter(g_obbs.values()))
        pred_obbs.append(p_obbs)
        gt_obbs.append(g_obbs)

        sem_id_to_name_pred = pred_reader.sem_ids_to_names.copy()
        sem_id_to_name_gt = gt_reader.sem_ids_to_names.copy()
        check_sem_id_conflict(sem_id_to_name_pred, sem_id_to_name_gt)
        sem_id_to_name.update(sem_id_to_name_gt)

    mAP = ObbMetrics(
        cam_ids=[0],
        cam_names=["rgb"],
        class_metrics=True,
        eval_2d=False,
        eval_3d=True,
        global_name_to_id={
            name: int(sem_id) for sem_id, name in sem_id_to_name.items()
        },
    )

    precs, recs = [], []
    for p_obbs, g_obbs in zip(pred_obbs, gt_obbs):
        prec, rec, match_mat, ious, per_class_results = prec_recall_bb3(
            p_obbs,
            g_obbs,
            iou_thres=iou,
            return_ious=True,
            per_class=True,
        )
        precs.append(prec)
        recs.append(rec)
        mAP.update(p_obbs, g_obbs)
    result[f"precision@IoU{iou}"] = np.mean(precs)
    result[f"recall@IoU{iou}"] = np.mean(recs)

    precs, recalls, probs = prec_recall_curve(
        [(p_obbs, g_obbs) for p_obbs, g_obbs in zip(pred_obbs, gt_obbs)]
    )

    # save precision-recall curve to png
    save_dir = input_folder
    draw_prec_recall_curve(precs, recalls, save_folder=save_dir, iou_thres=iou)

    result_map = mAP.compute()
    # ignore average recall (e.g. "rgb/mar_220_3D")
    result_map = {
        k: v.item() for k, v in result_map.items() if not k.startswith("rgb/mar_")
    }
    result.update(result_map)
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run EFM eval pipeline")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="The input folder that contains the gt and pred obbs csv files. If this is provided, the eval will be done at dataset-level",
        default=None,
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        help="The prediction obbs csv file, can be snippet-level snippet_obbs.csv or scene-level tracked_scene_obbs.csv",
        default=None,
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        help="The ground truth obbs csv file, can be snippet-level gt_obbs.csv or scene-level gt_scene_obbs.csv",
        default=None,
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--pr_curve",
        action="store_true",
        help="Whether to draw precision recall curve",
    )
    args = parser.parse_args()

    if args.input_folder:
        metrics = obb_eval_dataset(args.input_folder)
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        assert args.pred_csv is not None, "pred_csv is required"
        assert args.gt_csv is not None, "gt_csv is required"

        metrics = evaluate_obb_csv(args.pred_csv, args.gt_csv, args.iou, args.pr_curve)
        output_dir = os.path.dirname(args.pred_csv)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
