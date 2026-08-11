"""Compatibility exports for legacy ``mmdet3d.core.bbox`` imports."""

from mmdet3d.registry import TASK_UTILS
from mmdet3d.structures import (AxisAlignedBboxOverlaps3D, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, bbox3d2roi,
                                bbox3d_mapping_back, bbox_overlaps_3d,
                                bbox_overlaps_nearest_3d, points_cam2img,
                                xywhr2xyxyr)
from mmdet3d.structures.ops import box_np_ops


def build_bbox_coder(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

__all__ = [
    'AxisAlignedBboxOverlaps3D', 'Box3DMode', 'CameraInstance3DBoxes',
    'Coord3DMode', 'DepthInstance3DBoxes', 'LiDARInstance3DBoxes',
    'bbox3d2result', 'bbox3d2roi', 'bbox3d_mapping_back', 'bbox_overlaps_3d',
    'bbox_overlaps_nearest_3d', 'box_np_ops', 'build_bbox_coder',
    'points_cam2img', 'xywhr2xyxyr'
]
