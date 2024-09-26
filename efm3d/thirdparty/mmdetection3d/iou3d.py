# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py  # noqa
# Adapted from https://github.com/lilanxiao/Rotated_IoU/blob/master/oriented_iou_loss.py  # noqa
# License https://github.com/lilanxiao/Rotated_IoU/blob/master/LICENSE

from typing import Tuple

import mmdet_iou3d

import torch
from torch import Tensor
from torch.autograd import Function

EPSILON = 1e-8


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = mmdet_iou3d.sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


def box_intersection(corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor: (B, N, 4, 4, 2) Intersections.
         - Tensor: (B, N, 4, 4) Valid intersections mask.
    """
    # build edges from corners
    # B, N, 4, 4: Batch, Box, edge, point
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3)
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3)
    line2_ext = line2.unsqueeze(2)
    x1, y1, x2, y2 = line1_ext.split([1, 1, 1, 1], dim=-1)
    x3, y3, x4, y4 = line2_ext.split([1, 1, 1, 1], dim=-1)
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    numerator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denumerator_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = denumerator_t / numerator
    t[numerator == 0.0] = -1.0
    mask_t = (t > 0) & (t < 1)  # intersection on line segment 1
    denumerator_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -denumerator_u / numerator
    u[numerator == 0.0] = -1.0
    mask_u = (u > 0) & (u < 1)  # intersection on line segment 2
    mask = mask_t * mask_u
    # overwrite with EPSILON. otherwise numerically unstable
    t = denumerator_t / (numerator + EPSILON)
    intersections = torch.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: Tensor, corners2: Tensor) -> Tensor:
    """Check if corners of box1 lie in box2.
    Convention: if a corner is exactly on the edge of the other box,
    it's also a valid point.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tensor: (B, N, 4) Intersection.
    """
    # a, b, c, d - 4 vertices of box2
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    # ab, am, ad - vectors between corresponding vertices
    ab = b - a  # (B, N, 1, 2)
    am = corners1 - a  # (B, N, 4, 2)
    ad = d - a  # (B, N, 1, 2)
    prod_ab = torch.sum(ab * am, dim=-1)  # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)  # (B, N, 1)
    prod_ad = torch.sum(ad * am, dim=-1)  # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)  # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes
    # are exactly the same also stable with different scale of bboxes
    cond1 = (prod_ab / norm_ab > -1e-6) * (prod_ab / norm_ab < 1 + 1e-6)  # (B, N, 4)
    cond2 = (prod_ad / norm_ad > -1e-6) * (prod_ad / norm_ad < 1 + 1e-6)  # (B, N, 4)
    return cond1 * cond2


def box_in_box(corners1: Tensor, corners2: Tensor) -> Tuple[Tensor, Tensor]:
    """Check if corners of two boxes lie in each other.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor: (B, N, 4) True if i-th corner of box1 is in box2.
         - Tensor: (B, N, 4) True if i-th corner of box2 is in box1.
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(
    corners1: Tensor,
    corners2: Tensor,
    c1_in_2: Tensor,
    c2_in_1: Tensor,
    intersections: Tensor,
    valid_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Find vertices of intersection area.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.
        c1_in_2 (Tensor): (B, N, 4) True if i-th corner of box1 is in box2.
        c2_in_1 (Tensor): (B, N, 4) True if i-th corner of box2 is in box1.
        intersections (Tensor): (B, N, 4, 4, 2) Intersections.
        valid_mask (Tensor): (B, N, 4, 4) Valid intersections mask.

    Returns:
        Tuple:
         - Tensor: (B, N, 24, 2) Vertices of intersection area;
               only some elements are valid.
         - Tensor: (B, N, 24) Mask of valid elements in vertices.
    """
    # NOTE: inter has elements equals zero and has zeros gradient
    # (masked by multiplying with 0); can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    # (B, N, 4 + 4 + 16, 2)
    vertices = torch.cat([corners1, corners2, intersections.view([B, N, -1, 2])], dim=2)
    # Bool (B, N, 4 + 4 + 16)
    mask = torch.cat([c1_in_2, c2_in_1, valid_mask.view([B, N, -1])], dim=2)
    return vertices, mask


def sort_indices(vertices: Tensor, mask: Tensor) -> Tensor:
    """Sort indices.
    Note:
        why 9? the polygon has maximal 8 vertices.
        +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitrary elements in the last
        16 (intersections not corners) with value 0 and mask False.
        (cause they have zero value and zero gradient)

    Args:
        vertices (Tensor): (B, N, 24, 2) Box vertices.
        mask (Tensor): (B, N, 24) Mask.

    Returns:
        Tensor: (B, N, 9) Sorted indices.

    """
    num_valid = torch.sum(mask.int(), dim=2).int()  # (B, N)
    mean = torch.sum(
        vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True
    ) / num_valid.unsqueeze(-1).unsqueeze(-1)
    vertices_normalized = vertices - mean  # normalization makes sorting easier
    return SortVertices.apply(vertices_normalized, mask, num_valid).long()


def calculate_area(idx_sorted: Tensor, vertices: Tensor) -> Tuple[Tensor, Tensor]:
    """Calculate area of intersection.

    Args:
        idx_sorted (Tensor): (B, N, 9) Sorted vertex ids.
        vertices (Tensor): (B, N, 24, 2) Vertices.

    Returns:
        Tuple:
         - Tensor (B, N): Area of intersection.
         - Tensor: (B, N, 9, 2) Vertices of polygon with zero padding.
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1, 1, 1, 2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = (
        selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1]
        - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    )
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(
    corners1: Tensor, corners2: Tensor
) -> Tuple[Tensor, Tensor]:
    """Calculate intersection area of 2d rotated boxes.

    Args:
        corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
        corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

    Returns:
        Tuple:
         - Tensor (B, N): Area of intersection.
         - Tensor (B, N, 9, 2): Vertices of polygon with zero padding.
    """
    intersections, valid_mask = box_intersection(corners1, corners2)
    c12, c21 = box_in_box(corners1, corners2)
    vertices, mask = build_vertices(
        corners1, corners2, c12, c21, intersections, valid_mask
    )
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners(box: Tensor) -> Tensor:
    """Convert rotated 2d box coordinate to corners.

    Args:
        box (Tensor): (B, N, 5) with x, y, w, h, alpha.

    Returns:
        Tensor: (B, N, 4, 2) Corners.
    """
    B = box.size()[0]
    x, y, w, h, alpha = box.split([1, 1, 1, 1, 1], dim=-1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).to(box.device)
    x4 = x4 * w  # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).to(box.device)
    y4 = y4 * h  # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)  # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1, 4, 2]), rot_T.view([-1, 2, 2]))
    rotated = rotated.view([B, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def diff_iou_rotated_2d(box1: Tensor, box2: Tensor) -> Tensor:
    """Calculate differentiable iou of rotated 2d boxes.

    Args:
        box1 (Tensor): (B, N, 5) First box.
        box2 (Tensor): (B, N, 5) Second box.

    Returns:
        Tensor: (B, N) IoU.
    """
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou


def diff_iou_rotated_3d(box3d1: Tensor, box3d2: Tensor) -> Tensor:
    """Calculate differentiable iou of rotated 3d boxes.

    Args:
        box3d1 (Tensor): (B, N, 3+3+1) First box (x,y,z,w,h,l,alpha).
        box3d2 (Tensor): (B, N, 3+3+1) Second box (x,y,z,w,h,l,alpha).

    Returns:
        Tensor: (B, N) IoU.
    """
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_(min=0.0)
    intersection_3d = intersection * z_overlap
    volume1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    volume2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    union_3d = volume1 + volume2 - intersection_3d
    return intersection_3d / union_3d


def rotated_iou_3d_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of rotated bounding boxes.
    Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [N, 7]
            (x, y, z, w, l, h, alpha).
        target (torch.Tensor): Bbox targets (gt) with shape [N, 7]
            (x, y, z, w, l, h, alpha).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    iou_loss = 1 - diff_iou_rotated_3d(pred.unsqueeze(0), target.unsqueeze(0))[0]
    return iou_loss


class RotatedIoU3DLoss(torch.nn.Module):
    """Calculate the IoU loss (1-IoU) of rotated bounding boxes.

    Args:
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
    ):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 7]
                (x, y, z, w, l, h, alpha).

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        # print(pred.shape, target.shape)
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return 0.0 * pred.sum()
        loss = self.loss_weight * rotated_iou_3d_loss(pred, target)
        return loss


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    mmdet_iou3d.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = mmdet_iou3d.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = mmdet_iou3d.nms_normal_gpu(boxes, keep, thresh, boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
