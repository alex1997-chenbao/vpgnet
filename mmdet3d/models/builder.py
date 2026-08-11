"""Compatibility builders for legacy projects."""

from mmdet3d.registry import MODELS, TASK_UTILS

BACKBONES = MODELS
DETECTORS = MODELS
FUSION_LAYERS = MODELS
HEADS = MODELS
LOSSES = MODELS
MIDDLE_ENCODERS = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SEGMENTORS = MODELS
SHARED_HEADS = MODELS
VOXEL_ENCODERS = MODELS

BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
BBOX_CODERS = TASK_UTILS


def build(cfg, default_args=None):
    return MODELS.build(cfg, default_args=default_args)


def build_backbone(cfg):
    return MODELS.build(cfg)


def build_neck(cfg):
    return MODELS.build(cfg)


def build_head(cfg):
    return MODELS.build(cfg)


def build_loss(cfg):
    return MODELS.build(cfg)


def build_assigner(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_sampler(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_bbox_coder(cfg, default_args=None):
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    default_args = {}
    if train_cfg is not None:
        default_args['train_cfg'] = train_cfg
    if test_cfg is not None:
        default_args['test_cfg'] = test_cfg
    return MODELS.build(cfg, default_args=default_args or None)
