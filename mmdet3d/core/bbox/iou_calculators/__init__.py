"""Compatibility exports for legacy IoU calculator imports."""

from .iou3d_calculator import (AxisAlignedBboxOverlaps3D, BboxOverlaps3D,
                               BboxOverlapsNearest3D,
                               axis_aligned_bbox_overlaps_3d,
                               bbox_overlaps_3d, bbox_overlaps_nearest_3d)

__all__ = [
    'AxisAlignedBboxOverlaps3D', 'BboxOverlaps3D', 'BboxOverlapsNearest3D',
    'axis_aligned_bbox_overlaps_3d', 'bbox_overlaps_3d',
    'bbox_overlaps_nearest_3d'
]

