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

import json
import math
import os
import shutil

import hydra
import numpy as np
import omegaconf
import torch
import trimesh
from efm3d.dataset.vrs_dataset import preprocess_inference, VrsSequenceDataset
from efm3d.inference.fuse import VolumetricFusion
from efm3d.inference.model import EfmInference
from efm3d.inference.viz import generate_video
from efm3d.utils.gravity import correct_adt_mesh_gravity
from efm3d.utils.mesh_utils import eval_mesh_to_mesh


def get_gt_mesh_ply(data_path):
    """
    Return ASE or ADT GT mesh path. If not exist, return empty str.
    """
    if data_path.endswith(".vrs"):
        seq_name = os.path.basename(os.path.dirname(data_path))
    else:
        seq_name = os.path.basename(data_path.strip("/"))

    adt_mesh_ply = f"./data/adt_mesh/{seq_name}/gt_mesh.ply"
    ase_mesh_ply = f"./data/ase_mesh/scene_ply_{seq_name}.ply"
    if os.path.exists(adt_mesh_ply):
        return adt_mesh_ply
    elif os.path.exists(ase_mesh_ply):
        return ase_mesh_ply
    return ""


def compute_avg_metrics(paths):
    """
    Given metrics path list, compute the average metrics
    Note that simply averaging is not a good way to compute mAP metrics.
    """
    avg_ret = {}

    for path in paths:
        with open(path, "r") as f:
            metrics = json.load(f)
            for k, v in metrics.items():
                if k not in avg_ret:
                    avg_ret[k] = [v]
                else:
                    avg_ret[k].append(v)
    for k, v in avg_ret.items():
        avg_ret[k] = np.mean(v)
    return avg_ret


def create_streamer(data_path, snippet_length_s, stride_length_s, max_snip):
    # infer data type
    def is_atek_wds_input(data_path):
        ATEK_WDS_TAR = "shards-0000.tar"
        first_tar = os.path.join(data_path, ATEK_WDS_TAR)
        return os.path.exists(first_tar)

    if is_atek_wds_input(data_path):
        from efm3d.dataset.atek_wds_dataset import AtekWdsStreamDataset

        streamer = AtekWdsStreamDataset(
            data_path,
            atek_to_efm_taxonomy=f"{os.path.dirname(__file__)}/../config/taxonomy/atek_to_efm.csv",
            snippet_length_s=snippet_length_s,
            stride_length_s=stride_length_s,
            max_snip=max_snip,
        )
    elif data_path.endswith(".vrs"):
        # Use the native vrs sequence processor
        streamer = VrsSequenceDataset(
            data_path,
            frame_rate=10,
            sdi=2,
            snippet_length_s=snippet_length_s,
            stride_length_s=stride_length_s,
            max_snippets=max_snip,
            preprocess=preprocess_inference,
        )

        # (optional) use the ATEK data loader If it is installed
        # from efm3d.dataset.atek_vrs_dataset import create_atek_raw_data_loader_from_vrs_path
        # streamer = create_atek_raw_data_loader_from_vrs_path(
        #     vrs_path=data_path,
        #     freq_hz=10,
        #     snippet_length_s=snippet_length_s,
        #     stride_length_s=stride_length_s,
        #     skip_begin_seconds=20.0,
        #     skip_end_seconds=5.0,
        #     max_snippets=max_snip,
        # )
    else:
        print(
            f"Input error {data_path}, expect the input to be a folder to WDS tars or a .vrs file"
        )
        exit(-1)
    return streamer


def create_output_dir(output_dir, model_ckpt, data_path):
    # create output path from model ckpt and data path
    # e.g. result will be output to <output_dir>/<ckpt_name>/<seq_name>
    model_name = os.path.basename(os.path.splitext(model_ckpt)[0])
    seq_name = data_path
    if data_path.endswith(".vrs"):
        seq_name = os.path.basename(os.path.dirname(data_path))
    else:
        seq_name = os.path.basename(data_path.strip("/"))
    output_dir = os.path.join(output_dir, f"{model_name}", f"{seq_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_one(
    data_path,
    model_ckpt,
    model_cfg,
    max_snip=9999,
    snip_stride=0.1,
    voxel_res=0.04,
    output_dir="./output",
):
    output_dir = create_output_dir(output_dir, model_ckpt, data_path)

    # create model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    checkpoint = torch.load(model_ckpt, weights_only=True, map_location=device)
    model_config = omegaconf.OmegaConf.load(model_cfg)
    model = hydra.utils.instantiate(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    print("model init done")

    # create dataset
    streamer = create_streamer(
        data_path, snippet_length_s=1.0, stride_length_s=snip_stride, max_snip=max_snip
    )

    # per-snippet inference
    efm_inf = EfmInference(streamer, model, output_dir, device=device, zip=False)
    efm_inf.run()
    del efm_inf
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # track obbs
    try:
        from efm3d.inference.track import track_obbs

        track_obbs(output_dir)
    except:
        print(f"Skip tracking obb due to missing dependency, please see INSTALL.md")

    # eval obb
    metrics = {}
    pred_csv = os.path.join(output_dir, "tracked_scene_obbs.csv")
    gt_csv = os.path.join(output_dir, "gt_scene_obbs.csv")
    if os.path.exists(pred_csv) and os.path.exists(gt_csv):
        try:
            from efm3d.inference.eval import evaluate_obb_csv

            obb_metrics = evaluate_obb_csv(pred_csv=pred_csv, gt_csv=gt_csv, iou=0.2)
            metrics.update(obb_metrics)
        except:
            print(
                f"Skip obb evaluation due to missing dependency, please see INSTALL.md"
            )

    # fuse mesh
    vol_fusion = VolumetricFusion(output_dir, voxel_res=voxel_res, device=device)
    vol_fusion.run()
    fused_mesh = vol_fusion.get_trimesh()
    pred_mesh_ply = os.path.join(output_dir, "fused_mesh.ply")
    if fused_mesh.vertices.shape[0] > 0 and fused_mesh.faces.shape[0] > 0:
        fused_mesh.export(pred_mesh_ply)

    # eval mesh
    gt_mesh_ply = get_gt_mesh_ply(data_path)
    if os.path.exists(pred_mesh_ply) and os.path.exists(gt_mesh_ply):
        pred_trimesh = trimesh.load(pred_mesh_ply)
        gt_trimesh = trimesh.load(gt_mesh_ply)
        if "adt" in gt_mesh_ply:
            gt_trimesh = correct_adt_mesh_gravity(gt_trimesh)

        mesh_metrics, _, _ = eval_mesh_to_mesh(
            pred=pred_trimesh,
            gt=gt_trimesh,
            sample_num=1000,
        )
        metrics.update(mesh_metrics)

    # write metrics
    if len(metrics) > 0:
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(json.dumps(metrics, indent=2, sort_keys=True))

    # viz
    streamer = create_streamer(
        data_path=data_path,
        snippet_length_s=1.0,
        stride_length_s=1.0,
        max_snip=math.ceil((max_snip - 1) * snip_stride),
    )
    vol_fusion.reinit()
    viz_path = generate_video(
        streamer, output_dir=output_dir, vol_fusion=vol_fusion, stride_s=snip_stride
    )
    print(f"output viz file to {os.path.abspath(viz_path)}")

    # rm per-snippet occupancy tensors
    shutil.rmtree(os.path.join(output_dir, "per_snip"))
