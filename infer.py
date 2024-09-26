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

from efm3d.inference.pipeline import run_one


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EVL model inference on Aria sequences"
    )
    parser.add_argument("--input", type=str, required=True, help="input data")
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
    args = parser.parse_args()

    run_one(
        args.input,
        args.model_ckpt,
        model_cfg=args.model_cfg,
        max_snip=args.num_snips,
        snip_stride=args.snip_stride,
        voxel_res=args.voxel_res,
        output_dir=args.output_dir,
    )
