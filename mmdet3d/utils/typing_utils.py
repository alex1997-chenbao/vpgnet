# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in MMDetection3D."""
from typing import Any, List, Optional, Union

try:
    from mmdet.models.task_modules.samplers import SamplingResult
except ImportError:
    from mmdet.core.bbox.samplers import SamplingResult
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

try:
    from mmdet3d.structures.det3d_data_sample import Det3DDataSample
except ImportError:
    Det3DDataSample = Any

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]

SamplingResultList = List[SamplingResult]

OptSamplingResultList = Optional[SamplingResultList]
SampleList = List[Det3DDataSample]
OptSampleList = Optional[SampleList]
