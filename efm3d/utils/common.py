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

import numpy as np
import torch


def sample_nearest(value_a, value_b, array_b):
    array_b_at_a = []
    for v_a in value_a:
        idx = find_nearest(value_b, v_a, return_index=True)
        array_b_at_a.append(array_b[idx])
    return torch.stack(array_b_at_a)


def find_nearest(array, value, return_index=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if return_index:
        return idx
    else:
        return array[idx]
