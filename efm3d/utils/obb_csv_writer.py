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

import csv
from typing import Dict, Optional

import fsspec
import torch
from efm3d.aria.obb import ObbTW
from efm3d.aria.pose import PoseTW
from pyquaternion import Quaternion


class ObbCsvReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_reader = fsspec.open(self.file_name, "r").open()
        self.csv_reader = csv.DictReader(self.file_reader)
        try:
            self.next_row = next(self.csv_reader)
        except Exception:  # StopIteration
            self.next_row = None
        self.all_obbs = None
        self.sem_ids_to_names = {}

    def parse_row(self, row):
        t_ns = int(row["time_ns"])
        tx_wo = float(row["tx_world_object"])
        ty_wo = float(row["ty_world_object"])
        tz_wo = float(row["tz_world_object"])
        qw_wo = float(row["qw_world_object"])
        qx_wo = float(row["qx_world_object"])
        qy_wo = float(row["qy_world_object"])
        qz_wo = float(row["qz_world_object"])
        sx = float(row["scale_x"])
        sy = float(row["scale_y"])
        sz = float(row["scale_z"])
        if "instance" in row:
            inst_id = int(row["instance"])
        else:
            inst_id = -1
        sem_id = int(row["sem_id"])
        name = row["name"]
        if sem_id not in self.sem_ids_to_names:
            self.sem_ids_to_names[sem_id] = name
        else:
            assert name == self.sem_ids_to_names[sem_id]
        if "prob" in row:
            prob = float(row["prob"])
        else:
            # methods like ObjectMapper may not have probabilities
            prob = -1.0

        # create obbs
        xmin = -sx / 2.0
        xmax = sx / 2.0
        ymin = -sy / 2.0
        ymax = sy / 2.0
        zmin = -sz / 2.0
        zmax = sz / 2.0
        bb3s = torch.tensor([xmin, xmax, ymin, ymax, zmin, zmax])

        # create poses
        rot_mat = Quaternion(w=qw_wo, x=qx_wo, y=qy_wo, z=qz_wo).rotation_matrix
        translation = torch.tensor([tx_wo, ty_wo, tz_wo]).view(3, 1)
        T_wo = torch.concat([torch.tensor(rot_mat), translation], dim=1)
        T_wo = PoseTW.from_matrix3x4(T_wo)
        T_wo = T_wo.fit_to_SO3()
        T_world_object = T_wo._data

        # sem ids
        sem_ids = sem_id
        inst_ids = inst_id
        probs = prob
        # moveable: assuming static for now.
        # bb2s also decide the visibility of the 3D obbs in the corresponding camera.
        # we now just assume the obbs are visible in all the cameras
        bb2_rgbs = torch.ones(4)
        bb2_slamls = torch.ones(4)
        bb2_slamrs = torch.ones(4)
        # assume everything is static for now.
        moveables = torch.zeros(1)

        obb_tw = ObbTW.from_lmc(
            bb3_object=bb3s,
            bb2_rgb=bb2_rgbs,
            bb2_slaml=bb2_slamls,
            bb2_slamr=bb2_slamrs,
            T_world_object=T_world_object,
            sem_id=sem_ids,
            inst_id=inst_ids,
            prob=probs,
            moveable=moveables,
        ).float()
        return t_ns, obb_tw

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next obbs set with the same timestamp.
        """
        if self.next_row is None:
            raise StopIteration

        t0_ns, obb = self.parse_row(self.next_row)
        obbs = [obb]
        for row in self.csv_reader:
            t_ns = int(row["time_ns"])
            if t_ns != t0_ns:
                self.next_row = row
                return t0_ns, torch.stack(obbs)
            t_ns, obb = self.parse_row(row)
            obbs.append(obb)

        self.next_row = None
        return t0_ns, torch.stack(obbs)

    @property
    def obbs(self):
        if self.all_obbs is not None:
            return self.all_obbs

        all_obbs = {}
        for t_ns, obbs in self:
            all_obbs[t_ns] = obbs
        self.all_obbs = all_obbs
        return all_obbs


class ObbCsvWriter:
    def __init__(self, file_name=""):
        if not file_name:
            file_name = "/tmp/obbs.csv"

        print(f"starting obb writer to {file_name}")
        self.file_name = file_name
        self.file_writer = fsspec.open(self.file_name, "w").open()
        headers_str = "time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,name,instance,sem_id,prob"
        headers = headers_str.split(",")
        self.num_cols = len(headers)
        header_row = ",".join(headers)
        self.file_writer.write(header_row + "\n")
        self.rows = []

    def write_rows(self):
        for row in self.rows:
            self.file_writer.write(row + "\n")
        self.file_writer.flush()
        self.rows = []

    def write(
        self,
        obb_padded: ObbTW,
        timestamps_ns: int = -1,
        sem_id_to_name: Optional[Dict[int, str]] = None,
        flush_at_end: bool = True,
    ):
        obb = obb_padded.remove_padding().clone().cpu()
        time_ns = str(int(timestamps_ns))

        N = obb.shape[0]
        if N == 0:
            # write all -1 to indicate the obbs for this timestamp is missing
            # null_row = [time_ns] + ["-1" for _ in range(self.num_cols - 1)]
            # self.file_writer.write(",".join(null_row) + "\n")
            return

        obbs_poses = obb.T_world_object
        obbs_dims = obb.bb3_diagonal.numpy()
        obb_sems = obb.sem_id.squeeze(-1).numpy()
        obb_inst = obb.inst_id.squeeze(-1).numpy()
        obb_prob = obb.prob.squeeze(-1).numpy()
        for i in range(N):
            sem_id = obb_sems[i]
            if sem_id_to_name and sem_id in sem_id_to_name:
                name = sem_id_to_name[sem_id]
            else:
                name = str(int(sem_id))

            qwxyz = obbs_poses[i].q  # torch.Tensor [4]
            qwxyz = ",".join(qwxyz.numpy().astype(str))

            txyz = obbs_poses[i].t  # torch.Tensor [3]
            txyz = ",".join(txyz.numpy().astype(str))

            sxyz = ",".join(obbs_dims[i].astype(str))
            self.file_writer.write(
                f"{time_ns},{txyz},{qwxyz},{sxyz},{name},{obb_inst[i]},{obb_sems[i]},{obb_prob[i]}\n"
            )
        if flush_at_end:
            self.file_writer.flush()

    def flush(self):
        self.file_writer.flush()

    def __del__(self):
        if hasattr(self, "file_writer"):
            self.file_writer.close()
