"""Compatibility exports for legacy post-processing imports."""

from mmdet3d.models.layers.box3d_nms import nms_bev, nms_normal_bev

__all__ = ['nms_bev', 'nms_normal_bev']
