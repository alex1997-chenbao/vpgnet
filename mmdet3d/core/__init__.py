"""Compatibility exports for legacy ``mmdet3d.core`` imports."""

from mmdet3d.core.bbox import *  # noqa: F401,F403


def show_result(*args, **kwargs):
    raise NotImplementedError('Legacy show_result is not available in this '
                              'checkout.')

