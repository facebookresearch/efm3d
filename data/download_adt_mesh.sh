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

if ! pwd | grep -q "/data"; then
  echo "Error: Make sure to run this script under <EFM3D_DIR>/data/"
  exit 1
fi

wget https://github.com/facebookresearch/efm3d/releases/download/v1.0/adt_mesh.zip
unzip adt_mesh.zip
