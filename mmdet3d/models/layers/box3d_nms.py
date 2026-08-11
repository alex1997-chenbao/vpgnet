# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numba
import numpy as np
import torch
from mmcv.ops import nms, nms_rotated
from torch import Tensor


def box3d_multiclass_nms(
        mlvl_bboxes: Tensor,
        mlvl_bboxes_for_nms: Tensor,
        mlvl_scores: Tensor,
        score_thr: float,
        max_num: int,
        cfg: dict,
        mlvl_dir_scores: Optional[Tensor] = None,
        mlvl_attr_scores: Optional[Tensor] = None,
        mlvl_bboxes2d: Optional[Tensor] = None) -> Tuple[Tensor]:
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (Tensor): Multi-level boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (Tensor): Multi-level boxes with shape (N, C + 1).
            N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (Tensor, optional): Multi-level scores of direction
            classifier. Defaults to None.
        mlvl_attr_scores (Tensor, optional): Multi-level scores of attribute
            classifier. Defaults to None.
        mlvl_bboxes2d (Tensor, optional): Multi-level 2D bounding boxes.
            Defaults to None.

    Returns:
        Tuple[Tensor]: Return results after nms, including 3D bounding boxes,
        scores, labels, direction scores, attribute scores (optional) and
        2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = mlvl_scores.shape[1] - 1
    bboxes = []
    scores = []
    labels = []
    dir_scores = []
    attr_scores = []
    bboxes2d = []
    for i in range(0, num_classes):
        # get bboxes and scores of this class
        cls_inds = mlvl_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        _scores = mlvl_scores[cls_inds, i]
        _bboxes_for_nms = mlvl_bboxes_for_nms[cls_inds, :]

        if cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev

        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)
        _mlvl_bboxes = mlvl_bboxes[cls_inds, :]
        bboxes.append(_mlvl_bboxes[selected])
        scores.append(_scores[selected])
        cls_label = mlvl_bboxes.new_full((len(selected), ),
                                         i,
                                         dtype=torch.long)
        labels.append(cls_label)

        if mlvl_dir_scores is not None:
            _mlvl_dir_scores = mlvl_dir_scores[cls_inds]
            dir_scores.append(_mlvl_dir_scores[selected])
        if mlvl_attr_scores is not None:
            _mlvl_attr_scores = mlvl_attr_scores[cls_inds]
            attr_scores.append(_mlvl_attr_scores[selected])
        if mlvl_bboxes2d is not None:
            _mlvl_bboxes2d = mlvl_bboxes2d[cls_inds]
            bboxes2d.append(_mlvl_bboxes2d[selected])

    if bboxes:
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        if mlvl_dir_scores is not None:
            dir_scores = torch.cat(dir_scores, dim=0)
        if mlvl_attr_scores is not None:
            attr_scores = torch.cat(attr_scores, dim=0)
        if mlvl_bboxes2d is not None:
            bboxes2d = torch.cat(bboxes2d, dim=0)
        if bboxes.shape[0] > max_num:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            scores = scores[inds]
            if mlvl_dir_scores is not None:
                dir_scores = dir_scores[inds]
            if mlvl_attr_scores is not None:
                attr_scores = attr_scores[inds]
            if mlvl_bboxes2d is not None:
                bboxes2d = bboxes2d[inds]
    else:
        bboxes = mlvl_scores.new_zeros((0, mlvl_bboxes.size(-1)))
        scores = mlvl_scores.new_zeros((0, ))
        labels = mlvl_scores.new_zeros((0, ), dtype=torch.long)
        if mlvl_dir_scores is not None:
            dir_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_attr_scores is not None:
            attr_scores = mlvl_scores.new_zeros((0, ))
        if mlvl_bboxes2d is not None:
            bboxes2d = mlvl_scores.new_zeros((0, 4))

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores, )
    if mlvl_attr_scores is not None:
        results = results + (attr_scores, )
    if mlvl_bboxes2d is not None:
        results = results + (bboxes2d, )

    return results


import torch
from torch import Tensor

# def aligned_3d_nms(boxes: Tensor, scores: Tensor, classes: Tensor,
#                    thresh: float) -> Tensor:
#     """
#     双策略 NMS：实现“类内严格，类间宽松”。
    
#     Args:
#         strict_thresh (float): 类内阈值 (建议 0.2~0.3)。
#                                值越小越严格，同类物体只要有一点重叠就删。
#         loose_thresh (float):  类间阈值 (建议 0.8~1.0)。
#                                值越大越宽松，不同类物体允许高度重叠。
#     """
#     strict_thresh=0.25
#     loose_thresh=0.5
#     x1, y1, z1 = boxes[:, 0], boxes[:, 1], boxes[:, 2]
#     x2, y2, z2 = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    
#     # 1. 计算体积
#     area = (x2 - x1) * (y2 - y1) * (z2 - z1)
#     zero = boxes.new_zeros(1, )
    
#     score_sorted = torch.argsort(scores)
#     pick = []

#     while score_sorted.shape[0] > 0:
#         last = score_sorted.shape[0]
#         i = score_sorted[-1]
#         pick.append(i)

#         # 2. 计算 AABB 交集 (不考虑方向角)
#         xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
#         yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
#         zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
#         xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
#         yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
#         zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
        
#         inter_l = torch.max(zero, xx2 - xx1)
#         inter_w = torch.max(zero, yy2 - yy1)
#         inter_h = torch.max(zero, zz2 - zz1)
#         inter = inter_l * inter_w * inter_h

#         # --- 核心逻辑：双轨制计算 ---

#         # A. 针对同类别 (类内)：使用 IoM + 严格阈值
#         # 只要交集占了小框的一小部分，就视为重复预测
#         min_area = torch.min(area[i], area[score_sorted[:last - 1]])
#         metric_strict = inter / (min_area + 1e-6) # IoM

#         # B. 针对不同类别 (类间)：使用 IoU + 宽松阈值
#         # 允许桌子下有椅子，IoU 分母大，计算值小，不容易被删
#         metric_loose = inter / (area[i] + area[score_sorted[:last - 1]] - inter + 1e-6) # IoU

#         # --- 动态选择阈值 ---
#         classes1 = classes[i]
#         classes2 = classes[score_sorted[:last - 1]]
#         same_class_mask = (classes1 == classes2)

#         # 1. 组合 Metric：同类用 IoM，异类用 IoU
#         final_metric = torch.where(same_class_mask, metric_strict, metric_loose)

#         # 2. 组合 Threshold：同类用低阈值(0.25)，异类用高阈值(0.8)
#         # 构造一个与候选框数量一致的阈值向量
#         dynamic_thresh = torch.where(same_class_mask, 
#                                      torch.tensor(strict_thresh).to(boxes), 
#                                      torch.tensor(loose_thresh).to(boxes))

#         # 筛选：如果 Metric > 对应的 Threshold，则删除
#         # (同类：IoM > 0.25 删;  异类：IoU > 0.8 删)
#         score_sorted = score_sorted[torch.nonzero(
#             final_metric <= dynamic_thresh, as_tuple=False).flatten()]

#     return boxes.new_tensor(pick, dtype=torch.long)

def aligned_3d_nms(boxes: Tensor, scores: Tensor, classes: Tensor,
                   thresh: float) -> Tensor:
    """3D NMS for aligned boxes.

    Args:
        boxes (Tensor): Aligned box with shape [N, 6].
        scores (Tensor): Scores of each box.
        classes (Tensor): Class of each box.
        thresh (float): IoU threshold for nms.

    Returns:
        Tensor: Indices of selected boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)
    zero = boxes.new_zeros(1, )

    score_sorted = torch.argsort(scores)
    pick = []
    while (score_sorted.shape[0] != 0):
        last = score_sorted.shape[0]
        i = score_sorted[-1]
        pick.append(i)

        xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
        yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
        zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
        xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
        yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
        zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
        classes1 = classes[i]
        classes2 = classes[score_sorted[:last - 1]]
        inter_l = torch.max(zero, xx2 - xx1)
        inter_w = torch.max(zero, yy2 - yy1)
        inter_h = torch.max(zero, zz2 - zz1)

        inter = inter_l * inter_w * inter_h
        iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
        iou = iou * (classes1 == classes2).float()
        score_sorted = score_sorted[torch.nonzero(
            iou <= thresh, as_tuple=False).flatten()]

    indices = boxes.new_tensor(pick, dtype=torch.long)
    return indices
# def aligned_3d_nms(boxes: Tensor, scores: Tensor, classes: Tensor,
#                    thresh: float) -> Tensor:
#     """
#     修改版：基于 BEV (俯视图) 平面的 3D NMS。
#     保持参数签名不变，通过投影到 XY 平面计算 IoU，提高去重鲁棒性。
#     """
#     if boxes.shape[0] == 0:
#         return boxes.new_zeros(0, dtype=torch.long)

#     # 提取坐标：假设输入 boxes 为 [N, 6]，格式为 [x1, y1, z1, x2, y2, z2]
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     # z 轴在 BEV NMS 中不参与交集计算，但保留引用以防万一
#     x2 = boxes[:, 3]
#     y2 = boxes[:, 4]

#     # 计算 BEV 面积 (XY 平面投影面积)
#     # 使用 clamp(min=0) 确保即便预测框坐标反向也不会出现负面积
#     area_bev = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
#     # 按照分数从低到高排序，argsort 默认升序
#     score_sorted = torch.argsort(scores)
#     pick = []
#     zero = boxes.new_zeros(1, )

#     while (score_sorted.shape[0] != 0):
#         # 弹出当前分数最高的索引 (最后一个)
#         i = score_sorted[-1]
#         pick.append(i)

#         if score_sorted.shape[0] == 1:
#             break

#         # 计算当前高分框与剩余所有框在 BEV 上的交集区域
#         # xx1, yy1 为交集左下角；xx2, yy2 为交集右上角
#         xx1 = torch.max(x1[i], x1[score_sorted[:-1]])
#         yy1 = torch.max(y1[i], y1[score_sorted[:-1]])
#         xx2 = torch.min(x2[i], x2[score_sorted[:-1]])
#         yy2 = torch.min(y2[i], y2[score_sorted[:-1]])

#         # 计算交集宽和高
#         inter_w = torch.max(zero, xx2 - xx1)
#         inter_h = torch.max(zero, yy2 - yy1)
#         inter_area = inter_w * inter_h

#         # 计算 BEV IoU
#         # IoU = 交集 / (面积A + 面积B - 交集)
#         union = area_bev[i] + area_bev[score_sorted[:-1]] - inter_area
#         iou = inter_area / (union + 1e-6)

#         # 类别约束：只有同类别的物体才会互相抑制
#         # 如果类别不同，iou 会被乘 0，从而被保留
#         classes_match = (classes[i] == classes[score_sorted[:-1]]).float()
#         iou = iou * classes_match

#         # 核心筛选：保留 IoU 小于阈值的框（即不重叠或重叠较少的框）
#         mask = iou <= thresh
#         score_sorted = score_sorted[:-1][mask]

#     # 将结果转换为长整型张量返回
#     indices = boxes.new_tensor(pick, dtype=torch.long)
#     return indices

@numba.jit(nopython=True)
def circle_nms(dets: Tensor, thresh: float, post_max_size: int = 83) -> Tensor:
    """Circular NMS.

    An object is only counted as positive if no other center with a higher
    confidence exists within a radius r using a bird-eye view distance metric.

    Args:
        dets (Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept.
            Defaults to 83.

    Returns:
        Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
                i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return keep[:post_max_size]

    return keep


# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
def nms_bev(boxes: Tensor,
            scores: Tensor,
            thresh: float,
            pre_max_size: Optional[int] = None,
            post_max_size: Optional[int] = None) -> Tensor:
    """NMS function GPU implementation (for BEV boxes). The overlap of two
    boxes for IoU calculation is defined as the exact overlapping area of the
    two boxes. In this function, one can also set ``pre_max_size`` and
    ``post_max_size``.

    Args:
        boxes (Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (Tensor): Scores of boxes with the shape of [N].
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Defaults to None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Defaults to None.

    Returns:
        Tensor: Indexes after NMS.
    """
    assert boxes.size(1) == 5, 'Input boxes shape should be [N, 5]'
    order = scores.sort(0, descending=True)[1]
    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = boxes[order].contiguous()
    scores = scores[order]

    # xyxyr -> back to xywhr
    # note: better skip this step before nms_bev call in the future
    boxes = torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
        dim=-1)

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


# This function duplicates functionality of mmcv.ops.iou_3d.nms_normal_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms.
# Nms api will be unified in mmdetection3d one day.
def nms_normal_bev(boxes: Tensor, scores: Tensor, thresh: float) -> Tensor:
    """Normal NMS function GPU implementation (for BEV boxes). The overlap of
    two boxes for IoU calculation is defined as the exact overlapping area of
    the two boxes WITH their yaw angle set to 0.

    Args:
        boxes (Tensor): Input boxes with shape (N, 5).
        scores (Tensor): Scores of predicted boxes with shape (N).
        thresh (float): Overlap threshold of NMS.

    Returns:
        Tensor: Remaining indices with scores in descending order.
    """
    assert boxes.shape[1] == 5, 'Input boxes shape should be [N, 5]'
    return nms(boxes[:, :-1], scores, thresh)[1]
