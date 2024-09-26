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
import shutil
import time

import torch
import tqdm
from efm3d.aria.aria_constants import (
    ARIA_IMG_TIME_NS,
    ARIA_OBB_PADDED,
    ARIA_OBB_PRED_SEM_ID_TO_NAME,
    ARIA_OBB_PRED_VIZ,
    ARIA_OBB_SEM_ID_TO_NAME,
    ARIA_POINTS_VOL_MAX,
    ARIA_POINTS_VOL_MIN,
    ARIA_SNIPPET_T_WORLD_SNIPPET,
)
from efm3d.aria.obb import obb_time_union
from efm3d.dataset.wds_dataset import batchify
from efm3d.utils.obb_csv_writer import ObbCsvWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class EfmInference:
    def __init__(self, streamer, model, output_dir, device, zip):
        self.streamer = streamer
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.zip = zip
        self.metadata_saved = False
        self.Ts_wv = []  # all T_world_voxel as one tensor

        self.obb_csv_path = os.path.join(output_dir, "snippet_obbs.csv")
        self.obb_writer = None
        self.per_snip_dir = os.path.join(output_dir, "per_snip")
        shutil.rmtree(self.per_snip_dir, ignore_errors=True)
        os.makedirs(self.per_snip_dir, exist_ok=True)

        # obb GT
        self.gt_obb_csv_path = os.path.join(output_dir, "gt_obbs.csv")
        self.gt_obb_writer = None
        self.scene_gt_obbs_w = []

    def __del__(self):
        if not self.zip:
            return

        # compress the output folder
        if os.path.exists(self.output_dir) and os.listdir(self.output_dir):
            logger.info(f"zipping file to {self.output_dir}.zip")
            shutil.make_archive(
                self.output_dir.rstrip("/"), "zip", self.output_dir, verbose=True
            )
        logger.info(f"zip file saved to {self.output_dir}.zip")

    def save_tensor(self, tensor, key, idx=None, output_dir=""):
        if idx is not None:
            pt_name = os.path.join(output_dir, f"{key}_{idx:06}.pt")
        else:
            pt_name = os.path.join(output_dir, f"{key}.pt")
        torch.save(tensor.cpu(), pt_name)

    def save_output(self, data, idx, output_dir):
        """
        Save per-snippet 3D obb output and occupancy tensor to disk.
        """
        # assuming single sample batch
        bid = 0

        # 3d obb predictions
        if ARIA_OBB_PRED_VIZ in data:
            obb_preds_s = data[ARIA_OBB_PRED_VIZ][bid].remove_padding()
            T_ws = data[ARIA_SNIPPET_T_WORLD_SNIPPET][bid]
            obb_preds_w = obb_preds_s.transform(T_ws)
            first_rgb_time_ns = data[ARIA_IMG_TIME_NS[0]][bid, 0].item()
            if self.obb_writer is None:
                self.obb_writer = ObbCsvWriter(self.obb_csv_path)
            self.obb_writer.write(
                obb_preds_w, first_rgb_time_ns, data[ARIA_OBB_PRED_SEM_ID_TO_NAME]
            )

            if ARIA_OBB_PADDED in data and ARIA_OBB_SEM_ID_TO_NAME in data:
                gt_obbs_s = obb_time_union(data[ARIA_OBB_PADDED])[bid].remove_padding()
                gt_obbs_w = gt_obbs_s.transform(T_ws)
                self.scene_gt_obbs_w.append(gt_obbs_w.add_padding(128))

                if self.gt_obb_writer is None:
                    self.gt_obb_writer = ObbCsvWriter(self.gt_obb_csv_path)

                gt_sem_id_to_name = {}
                gt_sem_id_to_name.update(data[ARIA_OBB_SEM_ID_TO_NAME][bid])
                self.gt_obb_writer.write(
                    gt_obbs_w,
                    first_rgb_time_ns,
                    sem_id_to_name=gt_sem_id_to_name,
                )

        # occupancy predictions
        if (
            "occ_pr" in data
            and ARIA_POINTS_VOL_MIN in data
            and ARIA_POINTS_VOL_MAX in data
        ):
            if not self.metadata_saved:
                self.save_tensor(
                    data["voxel_extent"],
                    "voxel_extent",
                    idx=None,
                    output_dir=output_dir,
                )
                self.metadata_saved = True
                self.save_tensor(
                    data[ARIA_POINTS_VOL_MIN][0],  # tensor(3)
                    "scene_vol_min",
                    idx=None,
                    output_dir=output_dir,
                )
                self.save_tensor(
                    data[ARIA_POINTS_VOL_MAX][0],  # tensor(3)
                    "scene_vol_max",
                    idx=None,
                    output_dir=output_dir,
                )

            self.save_tensor(data["occ_pr"], "occ_pr", idx, output_dir)
            self.Ts_wv.append(data["voxel/T_world_voxel"][0])

    def run(self):
        # feed the per-snippet data to the model
        gt_sem_id = {}
        idx = 0

        start = time.time()
        for batch in tqdm.tqdm(self.streamer, total=len(self.streamer)):
            # convert single sample to batch and move to GPU
            batchify(batch, device=self.device)

            with torch.no_grad():
                output = self.model(batch)
                batch.update(output)
                self.save_output(batch, idx, self.per_snip_dir)
            if ARIA_OBB_SEM_ID_TO_NAME in batch:
                gt_sem_id.update(batch[ARIA_OBB_SEM_ID_TO_NAME][0])
            idx += 1

        torch.cuda.synchronize()
        print(f"\ninference speed {idx / (time.time() - start) :.02f} sample/s")

        # save all T_wv as one tensor to avoid writing small files
        if len(self.Ts_wv) > 0:
            Ts_wv = torch.stack(self.Ts_wv, dim=0)
            self.save_tensor(Ts_wv, "Ts_wv", None, self.per_snip_dir)

        # write scene-level obbs
        if len(self.scene_gt_obbs_w) > 0:
            max_obbs = 512
            merged_gts = torch.stack(self.scene_gt_obbs_w, dim=0)
            merged_gts = obb_time_union(merged_gts.unsqueeze(0), pad_size=max_obbs)
            merged_gts = merged_gts[0].remove_padding()

            gt_scene_obb_csv_path = os.path.join(self.output_dir, "gt_scene_obbs.csv")
            gt_scene_obb_writer = ObbCsvWriter(gt_scene_obb_csv_path)
            gt_scene_obb_writer.write(merged_gts, -1, gt_sem_id)
