"""Compatibility exports for legacy ``mmdet3d.ops`` imports."""

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

__all__ = ['SparseBasicBlock', 'make_sparse_convmodule',
           'IS_SPCONV2_AVAILABLE']

