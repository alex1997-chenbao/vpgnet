"""Compatibility exports for legacy ``mmdet3d.core.bbox.coders`` imports."""

from mmdet3d.registry import TASK_UTILS

BBOX_CODERS = TASK_UTILS


def build_bbox_coder(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)

__all__ = ['BBOX_CODERS', 'build_bbox_coder']
