# Copyright (c) OpenMMLab. All rights reserved.

from mmdet3d.registry import DATASETS
from .sunrgbd_dataset import SUNRGBDDataset


@DATASETS.register_module()
class LuggageSUNRGBDDataset(SUNRGBDDataset):
    """Single-class luggage SUNRGBD dataset."""

    METAINFO = {
        'classes': ('luggage', ),
        'palette': [(255, 187, 120)]
    }
