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

import copy
import random
from typing import Union

import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt


def point_to_closest_vertex_dist(pts, verts, tris):
    # pts N 3 float
    # verts M 3 float
    # norms M 3 float
    # tris O 3 int
    assert verts.ndim == 2, f"{verts.shape}"
    assert tris.ndim == 2, f"{tris.shape}"
    assert pts.ndim == 2, f"{pts.shape}"
    v0s = verts[None, tris[:, 0], :]
    v1s = verts[None, tris[:, 1], :]
    v2s = verts[None, tris[:, 2], :]
    pts = pts.unsqueeze(1)
    # compute distance to closest vertex
    vs = torch.cat([v0s, v1s, v2s], 0)
    dist_vs = torch.linalg.norm(vs.unsqueeze(1) - pts.unsqueeze(0), 2.0, -1)  # 3, N, M
    dist_vs = torch.min(dist_vs, 0)[0]
    dist_vs = torch.min(dist_vs, 1)[0]  # N
    return dist_vs


def point_to_closest_tri_dist(pts, verts, tris):
    """
    Compute the min distance of points to triangles. If a point doesn't intersect with any triangles
    return a big number (1e6) for that point.
    """
    assert verts.ndim == 2, f"{verts.shape}"
    assert tris.ndim == 2, f"{tris.shape}"
    assert pts.ndim == 2, f"{pts.shape}"

    def dot(a, b):
        return (a * b).sum(-1, keepdim=True)

    # pts N 3 float
    # verts M 3 float
    # norms M 3 float
    # tris O 3 int
    v0s = verts[None, tris[:, 0], :]
    v1s = verts[None, tris[:, 1], :]
    v2s = verts[None, tris[:, 2], :]
    pts = pts.unsqueeze(1)

    # compute if point projects inside triangle
    u = v1s - v0s
    v = v2s - v0s
    n = torch.cross(u, v)
    w = pts - v0s
    nSq = dot(n, n)
    gamma = dot(torch.cross(u, w, -1), n) / nSq
    beta = dot(torch.cross(w, v, -1), n) / nSq
    alpha = 1.0 - gamma - beta
    valid_alpha = torch.logical_and(0.0 <= alpha, alpha <= 1.0)
    valid_beta = torch.logical_and(0.0 <= beta, beta <= 1.0)
    valid_gamma = torch.logical_and(0.0 <= gamma, gamma <= 1.0)
    projs_to_tri = torch.logical_and(valid_alpha, valid_beta)
    projs_to_tri = torch.logical_and(projs_to_tri, valid_gamma)
    num_proj = projs_to_tri.count_nonzero(1)
    projs_to_tri = projs_to_tri.squeeze(-1)

    # compute distance to triangle plane
    n = n / torch.sqrt(nSq)
    dist_tri = dot(n, w).squeeze(-1).abs()
    # set distance to large for point-triangle combinations that do not project
    dist_tri[~projs_to_tri] = 1e6

    dist_tri = torch.min(dist_tri, 1)[0]  # N
    num_proj = num_proj.squeeze(-1)

    return dist_tri, num_proj


def compute_pts_to_mesh_dist(pts, faces, verts, step):
    dev = pts.device
    N = pts.shape[0]
    err = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    dist_tri = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    dist_ver = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    num_proj = torch.zeros(N).to(dev)
    for i in range(0, faces.shape[0], step):
        dist_tri_i, num_proj_i = point_to_closest_tri_dist(
            pts, verts, faces[i : i + step]
        )
        dist_ver_i = point_to_closest_vertex_dist(pts, verts, faces[i : i + step])
        dist_tri = torch.min(dist_tri_i, dist_tri)
        dist_ver = torch.min(dist_ver_i, dist_ver)
        num_proj = num_proj + num_proj_i

        prog_perc = min((i + step) / faces.shape[0] * 100, 100)
        print(f"Compute pts to mesh progress: {prog_perc:.01f}%", end="\r")
    err = torch.where(num_proj == 0, dist_ver, dist_tri)
    err = err.detach().cpu().numpy()
    return err


def eval_mesh_to_mesh(
    pred: Union[str, trimesh.Trimesh],
    gt: Union[str, trimesh.Trimesh],
    threshold=0.05,
    sample_num=10000,
    step=50000,
    cut_height=None,
):
    """
    Eval point to faces distance using `point_to_closest_tri_dist`.
    """
    rnd_seed = 0
    random.seed(0)
    np.random.seed(0)

    if isinstance(gt, str):
        print(f"load gt mesh {gt}")
        gt_mesh = trimesh.load_mesh(gt)
    else:
        gt_mesh = gt
    if isinstance(pred, str):
        print(f"load pred mesh {pred}")
        pred_mesh = trimesh.load_mesh(pred)
    else:
        pred_mesh = pred

    if cut_height is not None:
        cutting_plane = [[0, 0, -1], [0, 0, cut_height]]
        gt_mesh = gt_mesh.slice_plane(
            plane_origin=cutting_plane[1], plane_normal=cutting_plane[0]
        )
        pred_mesh = pred_mesh.slice_plane(
            plane_origin=cutting_plane[1], plane_normal=cutting_plane[0]
        )

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    print(f"==> [eval_mesh_to_mesh] use device {dev}")

    pred_vertices = torch.from_numpy(pred_mesh.vertices.view(np.ndarray)).to(dev)
    gt_vertices = torch.from_numpy(gt_mesh.vertices.view(np.ndarray)).to(dev)
    pred_faces = torch.from_numpy(pred_mesh.faces.view(np.ndarray)).to(dev)
    gt_faces = torch.from_numpy(gt_mesh.faces.view(np.ndarray)).to(dev)
    print(f"gt vertices and faces {gt_vertices.shape}, {gt_faces.shape}")
    print(f"pred vertices and faces {pred_vertices.shape}, {pred_faces.shape}")

    # accuracy (from sampled point in pred to GT)
    acc = torch.from_numpy(np.array(sample_num, np.finfo(np.float32).max)).to(dev)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, sample_num, seed=rnd_seed)
    pred_pts = torch.from_numpy(pred_pts.view(np.ndarray)).to(dev)
    acc = compute_pts_to_mesh_dist(pred_pts, gt_faces, gt_vertices, step)

    # completeness
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, sample_num, seed=rnd_seed)
    gt_pts = torch.from_numpy(gt_pts.view(np.ndarray)).to(dev)
    comp = compute_pts_to_mesh_dist(gt_pts, pred_faces, pred_vertices, step)

    precision5 = np.mean((acc < 0.05).astype("float"))
    recal5 = np.mean((comp < 0.05).astype("float"))
    precision1 = np.mean((acc < 0.01).astype("float"))
    recal1 = np.mean((comp < 0.01).astype("float"))
    fscore5 = 2 * precision5 * recal5 / (precision5 + recal5)
    fscore1 = 2 * precision1 * recal1 / (precision1 + recal1)
    # sort to get percentile numbers.
    acc_sorted = np.sort(acc)
    comp_sorted = np.sort(comp)
    metrics = {
        "acc_mean": np.mean(acc),
        "comp_mean": np.mean(comp),
        "prec@0.05": precision5,
        "recal@0.05": recal5,
        "fscore@0.05": fscore5,
    }

    # Create some visualizations for debugging.
    cmap = plt.cm.jet
    # accuracy heatmap (as a pointcloud) on predicted mesh
    norm = plt.Normalize(acc.min(), acc.max())
    colors = cmap(norm(acc))
    acc_pc = trimesh.points.PointCloud(pred_pts.detach().cpu().numpy())
    acc_pc.colors = colors
    # completeness heatmap (as a pointcloud) on gt mesh
    norm = plt.Normalize(comp.min(), comp.max())
    colors = cmap(norm(comp))
    com_pc = trimesh.points.PointCloud(gt_pts.detach().cpu().numpy())
    com_pc.colors = colors

    viz = {
        "acc_pc": acc_pc,
        "comp_pc": com_pc,
        "gt_mesh": gt_mesh,
    }

    for threshold in [0.01, 0.05]:
        prec_inliers = acc < threshold
        recall_inliers = comp < threshold

        # create visualizations for precision and recall
        prec_pc = copy.deepcopy(acc_pc)
        recal_pc = copy.deepcopy(com_pc)
        prec_pc.colors[prec_inliers] = [0, 255, 0, 255]  # green
        prec_pc.colors[~prec_inliers] = [255, 0, 0, 255]  # red
        recal_pc.colors[recall_inliers] = [0, 255, 0, 255]  # green
        recal_pc.colors[~recall_inliers] = [255, 0, 0, 255]  # red
        viz[f"prec@{threshold:.2}_pc"] = prec_pc
        viz[f"recal@{threshold:.2}_pc"] = recal_pc

    raw_data = {"acc": acc, "comp": comp}
    return metrics, viz, raw_data
