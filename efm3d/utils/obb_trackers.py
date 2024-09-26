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
from typing import Optional

import torch
from efm3d.aria.camera import CameraTW
from efm3d.aria.obb import bb2_xxyy_to_xyxy, bb3_xyzxyz_to_xxyyzz, ObbTW
from efm3d.aria.pose import all_rot90, find_r90, PoseTW
from efm3d.utils.obb_matchers import HungarianMatcher2d3d
from efm3d.utils.obb_utils import box3d_overlap_wrapper, remove_invalid_box3d
from torch.nn import functional as F
from torchvision.ops.boxes import box_iou

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def nms_3d(
    obbs,
    nms_iou3_thr: float = 0.1,
    nms_iou3_max_thr: float = 0.15,
    verbose: bool = False,
    mark_in_place: bool = False,
):
    """
    NMS based on 3D bbs. When a duplicate is found the obb with higher probability is retained.
    """
    ids_keep, ids_bad = list(range(obbs.shape[0])), []
    if mark_in_place:
        if obbs is None or obbs.shape[0] == 0:
            return (ids_keep, ids_bad)
        remove_invalid_box3d(obbs, mark_in_place)
        if obbs.shape[0] == 0:
            return (ids_keep, ids_bad)
    else:
        if obbs is None or obbs.shape[0] == 0:
            return obbs, None, (ids_keep, ids_bad)
        obbs, _ = remove_invalid_box3d(obbs, mark_in_place)
        if obbs.shape[0] == 0:
            return obbs, None, (ids_keep, ids_bad)

    bb3s = obbs.bb3corners_world
    N = bb3s.shape[0]
    # iou and make the diagonal negative (since we dont want to check overlap with self.)
    iou3 = box3d_overlap_wrapper(bb3s, bb3s).iou - 2.0 * torch.eye(
        N, device=bb3s.device
    )
    # we want bb3s to overlap and be in the same class
    same_ids = obbs.sem_id == obbs.sem_id.view(1, -1)
    overlap = iou3 > nms_iou3_thr
    overlap_overwrite = iou3 > nms_iou3_max_thr
    nms = torch.logical_or(torch.logical_and(overlap, same_ids), overlap_overwrite)

    if verbose and overlap.count_nonzero() > 0:
        logger.debug("overlap", iou3[overlap])
    if verbose and overlap_overwrite.count_nonzero() > 0:
        logger.debug("overlap_overwrite", iou3[overlap_overwrite])

    # ids where we want to NMS
    ids = torch.nonzero(nms, as_tuple=True)
    # ids of the obbs we want to remove (lower probability)
    ids_bad = torch.where(
        (obbs[ids[0]].prob < obbs[ids[1]].prob).squeeze(-1), ids[0], ids[1]
    )
    ids_bad = set(ids_bad.tolist())
    if len(ids_bad) > 0:
        # have some obbs to remove; compute which to keep and return those.
        if verbose:
            logger.debug(
                f"NMS3d: found {len(ids_bad)} non-maxima to suppress. {ids_bad}/ {bb3s.shape[0]}"
            )
        ids_keep = list(set(range(bb3s.shape[0])) - ids_bad)
        ids_bad = list(ids_bad)
        if mark_in_place:
            obbs._mark_invalid_ids(torch.tensor(ids_bad, dtype=torch.long))
            return (ids_keep, ids_bad)
        return obbs[ids_keep], obbs[ids_bad], (ids_keep, ids_bad)
    if mark_in_place:
        return (ids_keep, ids_bad)
    return obbs, None, (ids_keep, ids_bad)


def nms_2d(obbs, nms_iou2_thr: float, verbose: bool = False):
    """
    NMS based on 2D bbs. When a duplicate is found the obb with higher probability is retained.
    """
    if obbs is None or obbs.shape[0] == 0:
        return obbs, None
    bb2s = bb2_xxyy_to_xyxy(obbs.bb2_rgb)
    N = bb2s.shape[0]
    # iou and make the diagonal negative (since we dont want to check overlap with self.)
    iou2 = box_iou(bb2s, bb2s) - 2.0 * torch.eye(N, device=bb2s.device)
    # we want bb2s to overlap and be in the same class
    same_ids = obbs.sem_id == obbs.sem_id.view(1, -1)
    overlap = torch.logical_and(iou2 > nms_iou2_thr, same_ids)
    # ids where we have overlap
    ids = torch.nonzero(overlap, as_tuple=True)
    # ids of the obbs we want to remove (lower probability)
    ids_bad = torch.where(
        (obbs[ids[0]].prob < obbs[ids[1]].prob).squeeze(-1), ids[0], ids[1]
    )
    ids_bad = set(ids_bad.tolist())
    if len(ids_bad) > 0:
        # have some obbs to remove; compute which to keep and return those.
        if verbose:
            logger.debug(
                f"NMS2d: found {len(ids_bad)} non-maxima to suppress. {ids_bad}/ {bb2s.shape[0]}"
            )
        ids_keep = list(set(range(bb2s.shape[0])) - ids_bad)
        return obbs[ids_keep], obbs[list(ids_bad)]
    return obbs, None


class ObbTracker:
    """
    A simple obb tracker that uses Hungarian matching to associate new detected
    obbs with a set of "world"-state obbs it maintains incrementally.
    """

    def __init__(
        self,
        track_best: bool = False,
        track_running_average: bool = True,
        max_assoc_dist: float = 0.1,
        max_assoc_iou2: float = 0.2,
        max_assoc_iou3: float = 0.2,
        prob_inst_thr: float = 0.3,
        prob_assoc_thr: float = 0.25,
        nms_iou3_thr: float = 0.1,
        nms_iou2_thr: float = 0.5,
        w_max: int = 30,
        w_min: int = 5,
        dt_max_inst: float = 1.0,
        dt_max_occ: float = 5.0,
    ):
        """
        Args:
            track_best: choose the highest probability obb for obbs that have
                been associated. This is the most basic fusion strategy.
            track_running_average: maintain a running average of obbs that have
                been associated. This allows denoising obb parameters but
                struggles with detections that are not consistently in the same
                canonical orientation.
            max_assoc_dist: maximum distance to associate an obb with another
                obb. Obbs that are further are assumed to be distinct and lead
                to a new instantiation.
            max_assoc_iou2: maximum 2D IoU to associate an obb with another obb;
                beyond we instantiate a new obb.
            max_assoc_iou3: maximum 3D IoU to associate an obb with another obb;
                beyond we instantiate a new obb.
            prob_inst_thr: minimum probability threshold for instantiating a new
                world obb.
            prob_assoc_thr: minimum probability threshold for associating a new
                obb with existing world obbs.
            nms_iou3_thr: 3D IoU threshold to consider an world obb to be a
                duplicate and suppress it.
            nms_iou2_thr: 2D IoU threshold to consider an world obb to be a
                duplicate and suppress it.
            w_max: maximum weight accumulated in the running average.
            w_min: minimum weight needed to return the scene_obb.
            dt_max_inst: how long it can take for an object to be instantiated; in seconds
            dt_max_occ: how long it is okay for an instantiated object to be occluded; in seconds
        """
        self.matcher = HungarianMatcher2d3d(
            cost_class=8.0,
            cost_bbox2=0.0,
            cost_bbox3=1.0,
            cost_giou2=4.0,
            cost_iou3=0.0,
            # cost_class=8.0,
            # cost_bbox2=0.0,
            # cost_bbox3=0.0,
            # cost_giou2=4.0,
            # cost_iou3=4.0,
        )
        # the set of scene obbs
        self.scene_obbs_w = None
        # w is the weight (count) of each of the scene obbs
        self.w = None
        self.scene_probs_full = None
        self.num_semcls = 128

        # when last got an observation associated
        self.last_obs_time = None
        # when last possible to observe (based on 2d bb in frame)
        self.last_possible_obs_time = None
        # time of tracker (pseudo time incremented by 1 each track() call)
        self.time = 0
        self.hz = 10.0
        # how long it can take for an object to be instantiated; in seconds
        self.dt_max_inst = dt_max_inst
        # how long it is okay for an instantiated object to be occluded; in seconds
        self.dt_max_occ = dt_max_occ

        self.w_max = w_max
        self.w_min = w_min
        self.track_best = track_best
        self.track_running_average = track_running_average
        self.max_assoc_dist = max_assoc_dist
        self.max_assoc_iou2 = max_assoc_iou2
        self.max_assoc_iou3 = max_assoc_iou3
        self.prob_inst_thr = prob_inst_thr
        self.prob_assoc_thr = prob_assoc_thr
        self.nms_iou3_thr = nms_iou3_thr
        self.nms_iou3_max_thr = 0.15
        self.nms_iou2_thr = nms_iou2_thr
        self.R90s = all_rot90()
        self.counts_as_prob = False
        self.device = torch.device("cpu")
        self.num_instances_so_far = 0

    def reset(self):
        self.scene_obbs_w = None
        self.w = None
        self.scene_probs_full = None
        self.last_obs_time = None
        self.last_possible_obs_time = None
        self.time = 0

    def set_hz(self, hz: float):
        # adjust obb framerate
        self.hz = float(hz)

    @property
    def obbs_world(self):
        """
        The main function to access the tracked obbs that pass a set of gates.
        The returned objects are a subset of the full set of world obbs.
        """
        if self.scene_obbs_w is None:
            return ObbTW().to(self.device), ObbTW().to(self.device)
        sem_ids = self.scene_probs_full.argmax(dim=1)
        if (sem_ids != self.scene_obbs_w.sem_id.squeeze(-1)).any():
            change = sem_ids != self.scene_obbs_w.sem_id.squeeze(-1)
            logger.debug(
                "semantic id has changed because of probs_full averaging ",
                sem_ids[change].tolist(),
                self.scene_obbs_w.sem_id.squeeze(-1)[change].tolist(),
            )
        self.scene_obbs_w.set_sem_id(sem_ids)
        # which obbs have we seen recently?
        dt = self.last_possible_obs_time - self.last_obs_time
        seen_uninst = dt < self.dt_max_inst
        seen_occlusion = dt < self.dt_max_occ
        # remove obbs that do not have enough observations
        enough_observations = self.w > self.w_min

        # categories of obbs
        good_visible = torch.logical_and(enough_observations, seen_occlusion)
        good_invisible = torch.logical_and(enough_observations, ~seen_occlusion)
        uninst_visible = torch.logical_and(~enough_observations, seen_uninst)
        uninst_delete = torch.logical_and(~enough_observations, ~seen_uninst)

        # return all good ones
        obbs_w = self.scene_obbs_w[good_visible]
        # return the stale visible ones for debugging
        obbs_invis_w = self.scene_obbs_w[
            torch.logical_or(uninst_visible, good_invisible)
        ]

        # delete uninst obbs
        if uninst_delete.count_nonzero() > 0:
            logger.debug(
                f"removing un-instantiated obbs {uninst_delete.count_nonzero()}"
            )
            self.scene_obbs_w = self.scene_obbs_w[~uninst_delete]
            self.scene_probs_full = self.scene_probs_full[~uninst_delete]
            self.last_obs_time = self.last_obs_time[~uninst_delete]
            self.last_possible_obs_time = self.last_possible_obs_time[~uninst_delete]
            self.w = self.w[~uninst_delete]

        # NMS based on 3D IoU
        if self.nms_iou3_thr > 0.0:
            obbs_w, obbs_non_max_w = self.nms_3d(obbs_w)
        # NMS based on 2D IoU
        if self.nms_iou2_thr > 0.0:
            obbs_w, obbs_non_max_w = self.nms_2d(obbs_w)
        return obbs_w, obbs_invis_w

    def track(
        self,
        obbs_w: ObbTW,
        probs_full: Optional[torch.Tensor] = None,
        cam: Optional[CameraTW] = None,
        T_world_rig: Optional[PoseTW] = None,
    ):
        """
        Args:
            obbs_w: new obb detections to track. shape: Nx34
            probs_full: full probability distribution over the classes of each of the obb detections.
        """
        self.device = obbs_w.device
        assert obbs_w.ndim == 2, f"{obbs_w.shape}"
        # if we dont have any good new obbs return
        if obbs_w.shape[0] == 0:
            return self.obbs_world
        # set 2d bbs
        obbs_w = self.set_2d_bbs(obbs_w, cam, T_world_rig)
        # filter out obbs that are too low probability to be associated
        assoc = obbs_w.prob.squeeze(-1) > self.prob_assoc_thr
        # remove probs_full padding
        if probs_full is not None:
            probs_full = probs_full[: obbs_w.shape[0], :]
        else:
            # create one-hot probability encoding based on semantic id
            probs_full = F.one_hot(
                obbs_w.sem_id.squeeze(-1).long(), num_classes=self.num_semcls
            ).float()

        obbs_w = obbs_w[assoc]
        probs_full = probs_full[assoc] if probs_full is not None else None
        # if we dont have any good new obbs return
        if obbs_w.shape[0] == 0:
            return self.obbs_world
        # if we dont have any scene obbs yet (at the beginning) initialize the
        # tracker state and return it.
        if self.scene_obbs_w is None:
            self.add_new_obbs(obbs_w, probs_full)
            return self.obbs_world
        # find matches
        indices = self.matcher.forward_obbs(
            prd=obbs_w,
            tgt=self.scene_obbs_w,
            prd_logits=probs_full,
            logits_is_prob=True,
        )
        # if we have not matches we return
        if len(indices[0]) == 0:
            return self.obbs_world
        # get matched obbs
        pids, tids = indices[0], indices[1]
        pobbs, tobbs = obbs_w[pids], self.scene_obbs_w[tids]
        pprobs_full = probs_full[pids] if probs_full is not None else None
        # find good associations based on the 2d and 3d iou
        dist = torch.linalg.norm(
            pobbs.bb3_center_world - tobbs.bb3_center_world, 2, dim=-1
        ).cpu()
        if self.max_assoc_iou2 > 0:
            iou2 = (
                box_iou(
                    bb2_xxyy_to_xyxy(pobbs.bb2_rgb), bb2_xxyy_to_xyxy(tobbs.bb2_rgb)
                )
                .cpu()
                .diagonal()
            )
        else:
            iou2 = None

        # filter out invalid bboxes, if we can't compute iou3 we return
        pobbs, valid_ind = remove_invalid_box3d(pobbs)
        pprobs_full = probs_full[valid_ind] if pprobs_full is not None else None
        if pobbs.shape[0] == 0:
            return self.obbs_world

        if iou2 is not None:
            iou2 = iou2[valid_ind]
        tobbs = tobbs[valid_ind]
        dist = dist[valid_ind]
        pids = pids[valid_ind]
        tids = tids[valid_ind]

        # this function could fail due to thin object (ValueError: Planes have zero areas).
        # if we can't compute iou3 we return
        try:
            iou3 = (
                box3d_overlap_wrapper(pobbs.bb3corners_world, tobbs.bb3corners_world)
                .iou.cpu()
                .diagonal()
            )
        except Exception as e:
            print(e)
            return self.obbs_world

        # assoc = torch.logical_or(dist < self.max_assoc_dist, iou2 > self.max_assoc_iou)
        assoc = iou3 > self.max_assoc_iou3
        if self.max_assoc_iou2 > 0.0:
            assoc = torch.logical_or(assoc, iou2 > self.max_assoc_iou2)

        # new obbs
        new_ids = list(set(range(obbs_w.shape[0])) - set(pids.tolist()))
        if assoc.count_nonzero() > 0:
            logger.debug(
                f"{assoc.count_nonzero()} associated",
                "dist",
                dist[assoc],
                "iou2",
                iou2[assoc] if iou2 is not None else None,
                "iou3",
                iou3[assoc],
            )
        if (~assoc).count_nonzero() > 0:
            logger.debug(
                f"{(~assoc).count_nonzero()} not associated",
                "dist",
                dist[~assoc],
                "iou2",
                iou2[assoc] if iou2 is not None else None,
                "iou3",
                iou3[~assoc],
            )
        new_obbs = torch.cat([pobbs[~assoc].clone(), obbs_w[new_ids]])
        new_insts = new_obbs.prob.squeeze(-1) > self.prob_inst_thr
        new_obbs = new_obbs[new_insts]
        if pprobs_full is not None:
            new_probs_full = torch.cat(
                [pprobs_full[~assoc].clone(), probs_full[new_ids]]
            )
            new_probs_full = new_probs_full[new_insts]
        # associated obbs
        pids, tids = pids[assoc], tids[assoc]
        pobbs, tobbs = pobbs[assoc], tobbs[assoc]
        pprobs_full = pprobs_full[assoc] if pprobs_full is not None else None
        # deal with associations
        if self.track_best and tids.shape[0] > 0:
            better_pred = (pobbs.prob > tobbs.prob).squeeze(-1).cpu()
            # update better obbs
            better_tids = tids[better_pred]
            self.scene_obbs_w._data[better_tids] = pobbs._data[better_pred]
            # increment weights
            self.w[tids] = self.w[tids] + 1.0
            # update times
            self.last_obs_time[tids] = self.time

            # update counts as probabilities
            if self.counts_as_prob:
                scene_obbs = self.scene_obbs_w[tids].clone()
                scene_obbs.set_prob(self.w[tids])
                self.scene_obbs_w._data[tids] = scene_obbs._data

        elif self.track_running_average and tids.shape[0] > 0:
            wpp = (self.w[tids] + 1.0).unsqueeze(-1)
            pdiag = pobbs.bb3_diagonal
            # running average T_world_object
            dT_tobj_pobj = tobbs.T_world_object.inverse() @ pobbs.T_world_object
            xi_tobj_pobj = dT_tobj_pobj.log()
            # check if any relative pose is further than 45 degree which
            # indicates that there is a 90 deg rotation that is closer.
            dr = xi_tobj_pobj[..., 3:]
            dr_norm = torch.linalg.norm(dr, 2, dim=-1)
            too_big = dr_norm > 3.14 * 0.25
            if too_big.any():
                # find closest 90 degree rotation
                pT_wo, R90min = find_r90(
                    tobbs[too_big].T_world_object,
                    pobbs[too_big].T_world_object,
                    self.R90s.to(tobbs.device),
                )
                # update xi with the 90 deg closest rotation
                dT_tobj_pobj = tobbs[too_big].T_world_object.inverse() @ pT_wo
                xi_tobj_pobj[too_big] = dT_tobj_pobj.log()
                # also permute the diagonal according to the 90 deg rotation
                pdiag[too_big] = (
                    (R90min @ pdiag[too_big].unsqueeze(-1)).squeeze(-1).abs()
                )
            # apply updates
            ppT_world_object = tobbs.T_world_object @ PoseTW.exp(xi_tobj_pobj / wpp)
            # running average over scale / diagonal of obb
            ppdiag = (tobbs.bb3_diagonal * self.w[tids].unsqueeze(-1) + pdiag) / wpp
            ppbb3 = bb3_xyzxyz_to_xxyyzz(
                torch.cat([-ppdiag * 0.5, ppdiag * 0.5], dim=-1)
            )
            # running average over prob
            ppprob = (tobbs.prob * self.w[tids].unsqueeze(-1) + pobbs.prob) / wpp

            if pprobs_full is not None:
                # running average over the full probability distribution
                pprobs_full = (
                    self.scene_probs_full[tids] * self.w[tids].unsqueeze(-1)
                    + pprobs_full
                ) / wpp
                self.scene_probs_full[tids] = pprobs_full

            # update target parameters
            tobbs.set_T_world_object(ppT_world_object)
            tobbs.set_bb3_object(ppbb3)
            tobbs.set_prob(ppprob.squeeze(-1))
            self.scene_obbs_w._data[tids] = tobbs._data
            # update weights
            self.w[tids] = wpp.clamp(max=self.w_max).squeeze(-1)
            # update times
            self.last_obs_time[tids] = self.time

            # update counts as probabilities
            if self.counts_as_prob:
                scene_obbs = self.scene_obbs_w[tids].clone()
                scene_obbs.set_prob(self.w[tids])
                self.scene_obbs_w._data[tids] = scene_obbs._data

        # add new obbs
        if new_obbs.shape[0] > 0:
            self.add_new_obbs(new_obbs, new_probs_full)

        # update last possible obs time based on 2d visibility
        self.update_last_obs_time(cam, T_world_rig)

        # update time
        self.time += 1.0 / self.hz
        return self.obbs_world

    def update_last_obs_time(self, cam, T_world_rig):
        if cam is None or T_world_rig is None:
            # mark all visible
            self.last_possible_obs_time[:] = self.time
            return
        # compute visibility of scene obbs:
        # - at least 50% of object has to be in 2d bb
        # - 2d bb has to be at least 100 pixel area and each side has to be at least 10 pixels
        bb2s, _, frac = self.scene_obbs_w.get_pseudo_bb2(
            cam.unsqueeze(0), T_world_rig.unsqueeze(0), 10, return_frac_valids=True
        )
        bb2s, frac = bb2s.squeeze(0), frac.squeeze(0)
        visible = frac > 0.5
        # area = box_area(bb2_xxyy_to_xyxy(bb2s))
        # visible = torch.logical_and(visible, area > 100)
        visible = torch.logical_and(visible, bb2s[..., 1] - bb2s[..., 0] > 10)
        visible = torch.logical_and(visible, bb2s[..., 3] - bb2s[..., 2] > 10)
        # update last possible times
        self.last_possible_obs_time[visible] = self.time

    def add_new_obbs(self, new_obbs, new_probs_full):
        new_w = torch.ones(new_obbs.shape[0], device=new_obbs.device)
        new_obbs_time = self.time * torch.ones(
            new_obbs.shape[0], device=new_obbs.device
        )
        # Set instance ids for new obbs
        new_obbs.set_inst_id(
            torch.arange(
                self.num_instances_so_far,
                self.num_instances_so_far + new_obbs.shape[0],
                device=new_obbs.device,
            )
        )
        # Increment number of instances we have seen so far
        self.num_instances_so_far += new_obbs.shape[0]

        if self.scene_obbs_w is None:
            self.scene_obbs_w = new_obbs
            self.scene_probs_full = new_probs_full
            self.w = new_w
            self.last_obs_time = new_obbs_time
            self.last_possible_obs_time = new_obbs_time.clone()
        else:
            self.scene_obbs_w = torch.cat([self.scene_obbs_w, new_obbs], dim=0)
            self.scene_probs_full = torch.cat(
                [self.scene_probs_full, new_probs_full], dim=0
            )
            self.w = torch.cat([self.w, new_w])
            self.last_obs_time = torch.cat([self.last_obs_time, new_obbs_time])
            self.last_possible_obs_time = torch.cat(
                [self.last_possible_obs_time, new_obbs_time]
            )

    def nms_3d(self, obbs):
        obbs_keep, obbs_rm, _ = nms_3d(obbs, self.nms_iou3_thr, self.nms_iou3_max_thr)
        return obbs_keep, obbs_rm

    def nms_2d(self, obbs):
        return nms_2d(obbs, self.nms_iou2_thr)

    def set_2d_bbs(self, obbs_w: ObbTW, cam: CameraTW, T_world_rig: PoseTW):
        if cam is None or T_world_rig is None:
            return obbs_w
        if obbs_w.shape[0] > 0:
            bb2s, valids, frac = obbs_w.get_pseudo_bb2(
                cam.unsqueeze(0), T_world_rig.unsqueeze(0), 10, return_frac_valids=True
            )
            invisible = ~valids  # frac < 0.1
            bb2s[invisible] = -1.0
            obbs_w.set_bb2(0, bb2s.squeeze(0))
        if self.scene_obbs_w is not None and self.scene_obbs_w.shape[0] > 0:
            bb2s, valids, frac = self.scene_obbs_w.get_pseudo_bb2(
                cam.unsqueeze(0), T_world_rig.unsqueeze(0), 10, return_frac_valids=True
            )
            invisible = ~valids  # frac < 0.1
            bb2s[invisible] = -1.0
            self.scene_obbs_w.set_bb2(0, bb2s.squeeze(0))
        return obbs_w
