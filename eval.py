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

import argparse
import json
import os

from efm3d.inference.eval import obb_eval_dataset
from efm3d.inference.pipeline import compute_avg_metrics, run_one


ASE_DATA_PATH = "./data/ase_eval"
ADT_DATA_PATH = "./data/adt"
AEO_DATA_PATH = "./data/aeo"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EFM3D evaluation benchmark")
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=9999,
        help="number of sequences to evaluate, by default evaluate all sequences",
    )
    parser.add_argument(
        "--num_snips",
        type=int,
        default=9999,
        help="number of snippets per sequence, by default evaluate the full sequence",
    )
    parser.add_argument(
        "--snip_stride",
        type=float,
        default=0.1,
        help="overlap between snippets in second, default to 0.1 (recommend to set it between 0.1-0.5), set it larger will make performance worse but run faster",
    )
    parser.add_argument(
        "--voxel_res",
        type=float,
        default=0.04,
        help="voxel resolution in meter for volumetric fusion",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="./ckpt/model_release.pth",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="./efm3d/config/evl_inf.yaml",
        help="model config file",
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="output dir")
    parser.add_argument(
        "--ase", action="store_true", help="Evaluate the model on ASE dataset"
    )
    parser.add_argument(
        "--adt", action="store_true", help="Evaluate the model on ADT dataset"
    )
    parser.add_argument(
        "--aeo", action="store_true", help="Evaluate the model on AEO dataset"
    )
    args = parser.parse_args()

    input_paths = []
    if args.ase:
        with open("./data/ase_splits.json", "r") as f:
            seq_ids = json.load(f)["test_sequences"]
            seq_ids = [seq.strip() for seq in seq_ids]
        input_paths = [
            os.path.join(ASE_DATA_PATH, seq.strip()) for seq in seq_ids[: args.num_seqs]
        ]
    elif args.adt:
        with open("./data/adt_sequences.txt", "r") as f:
            seq_ids = f.readlines()
            seq_ids = [seq.strip() for seq in seq_ids]
        input_paths = [
            os.path.join(ADT_DATA_PATH, seq.strip(), "video.vrs")
            for seq in seq_ids[: args.num_seqs]
        ]
    elif args.aeo:
        with open("./data/aeo_sequences.txt", "r") as f:
            seq_ids = f.readlines()
            seq_ids = [seq.strip() for seq in seq_ids]
        input_paths = [
            os.path.join(AEO_DATA_PATH, seq.strip(), "main.vrs")
            for seq in seq_ids[: args.num_seqs]
        ]
    else:
        assert (
            args.ase or args.adt or args.aeo
        ), "Specify eval dataset, for example, --ase"

    for input_path in input_paths:
        run_one(
            input_path,
            args.model_ckpt,
            model_cfg=args.model_cfg,
            max_snip=args.num_snips,
            snip_stride=args.snip_stride,
            voxel_res=args.voxel_res,
            output_dir=args.output_dir,
        )

    # aggregate results
    if len(seq_ids) > 1:
        dirs = []
        model_name = os.path.splitext(os.path.basename(args.model_ckpt))[0]
        output_dir = os.path.join(args.output_dir, model_name)
        for seq_id in seq_ids:
            seq_output_dir = os.path.join(output_dir, seq_id)
            dirs.append(seq_output_dir)

        metrics_paths = [os.path.join(folder, "metrics.json") for folder in dirs]
        metrics_paths = [p for p in metrics_paths if os.path.exists(p)]
        if len(metrics_paths) > 0:
            avg_ret = compute_avg_metrics(metrics_paths)
            print("==> mean results")
            print(json.dumps(avg_ret, indent=2, sort_keys=True))
            with open(os.path.join(output_dir, "mean_metrics.json"), "w") as f:
                json.dump(avg_ret, f, indent=2, sort_keys=True)

            # aggregate mAP for 3D object detection
            if args.ase or args.aeo:
                joint_map = obb_eval_dataset(output_dir)
                print("==> joint mAP")
                print(json.dumps(joint_map, indent=2, sort_keys=True))
                with open(
                    os.path.join(args.output_dir, "joint_metrics.json"), "w"
                ) as f:
                    json.dump(joint_map, f, indent=2, sort_keys=True)
