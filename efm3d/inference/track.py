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

import torch

from efm3d.utils.obb_csv_writer import ObbCsvReader, ObbCsvWriter
from efm3d.utils.obb_trackers import ObbTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


def track_obbs(input_path, prob_inst_thr=0.3, prob_assoc_thr=0.25):
    """
    Run ObbTracker on input csv file.

    input_path: path to input folder or obbs csv file. if folder, will look for 'snippet_obbs.csv'
    as the input obbs csv file.
    prob_inst_thr: minimum probability threshold for instantiating a new world obb
    prob_assoc_thr: minimum probability threshold for associating a new obb with existing world obbs
    """
    if not os.path.exists(input_path):
        logger.error(f"Input folder {input_path} does not exist")
        return

    if input_path.endswith(".csv"):
        obb_csv_path = input_path
        obb_folder = os.path.dirname(input_path)
    else:
        obb_csv_path = os.path.join(input_path, "snippet_obbs.csv")
        obb_folder = input_path
    assert os.path.exists(obb_csv_path), f"No obb csv file found {obb_csv_path}"

    tracked_obbs_path = os.path.join(obb_folder, "tracked_obbs.csv")
    reader = ObbCsvReader(obb_csv_path)
    writer = ObbCsvWriter(tracked_obbs_path)
    tracker = ObbTracker(
        track_best=False,
        track_running_average=True,
        max_assoc_dist=0.1,
        max_assoc_iou2=0.0,  # disabled
        max_assoc_iou3=0.2,
        prob_inst_thr=prob_inst_thr,
        prob_assoc_thr=prob_assoc_thr,
        nms_iou3_thr=0.1,
        nms_iou2_thr=0.0,  # disabled
        w_max=30,
        w_min=5,
        dt_max_inst=1.0,
        dt_max_occ=999999.0,  # never delete
    )

    # write snippet-level tracked obbs
    for i, (t_ns, obbs) in enumerate(reader):
        tracked_obbs, unviz_obbs = tracker.track(obbs)
        # seq_obb_eval use both tracked and unviz obbs
        all_tracked_obbs = torch.cat([tracked_obbs, unviz_obbs], dim=-2)
        writer.write(all_tracked_obbs, t_ns, reader.sem_ids_to_names)

    # write scene-level tracked obbs
    tracked_scene_obbs_path = os.path.join(obb_folder, "tracked_scene_obbs.csv")
    scene_writer = ObbCsvWriter(tracked_scene_obbs_path)
    final_scene_obbs, unviz_obbs = tracker.obbs_world
    final_scene_obbs_all = torch.cat([final_scene_obbs, unviz_obbs], dim=-2)
    scene_writer.write(final_scene_obbs_all, -1, reader.sem_ids_to_names)
    logger.info(f"Wrote scene-level tracked obbs to {tracked_scene_obbs_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Obb tracker on obbs csv file.")
    parser.add_argument(
        "--input",
        type=str,
        help="The input folder to look for the per-snippet obbs csv file",
        required=True,
    )
    parser.add_argument(
        "--prob_inst_thr",
        type=float,
        default=0.3,
        help="minimum probability threshold for instantiating a new world obb",
    )
    parser.add_argument(
        "--prob_assoc_thr",
        type=float,
        default=0.25,
        help="minimum probability threshold for associating a new obb with existing world obbs",
    )

    args = parser.parse_args()
    track_obbs(input_path=args.input)


if __name__ == "__main__":
    main()
