#!/usr/bin/env bash
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

set -e

if ! ls infer.py | grep -q "infer.py"; then
  echo "Error: Can't find infer.py under the current directory. Make sure to run this script under <EFM3D_DIR>"
  exit 1
fi

# download DinoV2 weights
wget -O ckpt/dinov2_vitb14_reg4_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth

if [ ! -f "ckpt/evl_model_ckpt.zip" ]; then
  echo "Error: File evl_model_ckpt.zip does not exist. Make sure it's put under EFM3D_DIR/ckpt"
  exit 1
fi

# model
cd ckpt
unzip evl_model_ckpt.zip
mv evl_model_ckpt/*.pth .
mv evl_model_ckpt/seq136_sample.zip ../data
rmdir evl_model_ckpt

# data
cd ../data
unzip seq136_sample.zip
rm seq136_sample.zip

echo "Done preparing for inference"
