auto_scale_lr = dict(base_batch_size=128, enable=False)
backend_args = None
class_names = ('luggage', )
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet3d.datasets.sunrgbd_luggage_dataset',
        'mmdet3d.models.my_module',
    ])
data_root = 'data/sunrgbd'
dataset_type = 'LuggageSUNRGBDDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=-1,
        rule='greater',
        save_best='mAP_0.50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0008
luggage_mean_size = [
    [
        0.558169,
        0.376041,
        0.05,
    ],
]
metainfo = dict(classes=('luggage', ))
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='Det3DDataPreprocessor'),
    fc_cross_attn=dict(
        img_in_dim=256,
        k_size=7,
        out_dim=256,
        pts_in_dim=256,
        type='FullLocalCrossAttention'),
    freeze_img_branch=True,
    fusion_layer=dict(
        max_imvote_per_pixel=3, num_classes=1, type='VoteFusion'),
    geometric_refiner=dict(
        embed_dim=256,
        k=16,
        radius=0.2,
        tau=5.0,
        type='LocalGeometricRefiner'),
    img_backbone=dict(
        depth=50,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='mmdet.ResNet'),
    img_mlp=dict(
        act_cfg=dict(type='ReLU'),
        conv_cfg=dict(type='Conv1d'),
        conv_channels=(
            256,
            256,
        ),
        in_channel=18,
        norm_cfg=dict(type='BN1d')),
    img_neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.FPN'),
    img_roi_head=dict(
        _scope_='mmdet',
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    img_rpn_head=dict(
        _scope_='mmdet',
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    num_sampled_seed=1024,
    pts_backbone=dict(
        fp_channels=(
            (
                256,
                256,
            ),
            (
                256,
                256,
            ),
        ),
        in_channels=4,
        norm_cfg=dict(type='BN2d'),
        num_points=(
            2048,
            1024,
            512,
            256,
        ),
        num_samples=(
            64,
            32,
            16,
            16,
        ),
        radius=(
            0.2,
            0.4,
            0.8,
            1.2,
        ),
        sa_cfg=dict(
            normalize_xyz=True,
            pool_mod='max',
            type='PointSAModule',
            use_xyz=True),
        sa_channels=(
            (
                64,
                64,
                128,
            ),
            (
                128,
                128,
                256,
            ),
            (
                128,
                128,
                256,
            ),
            (
                128,
                128,
                256,
            ),
        ),
        type='PointNet2SASSG'),
    pts_bbox_heads=dict(
        common=dict(
            bbox_coder=dict(
                mean_sizes=[
                    [
                        0.584916,
                        0.396432,
                        0.2,
                    ],
                ],
                num_dir_bins=12,
                num_sizes=1,
                type='PartialBinBasedBBoxCoder',
                with_rot=True),
            center_loss=dict(
                loss_dst_weight=10.0,
                loss_src_weight=10.0,
                mode='l2',
                reduction='sum',
                type='ChamferDistance'),
            dir_class_loss=dict(
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.CrossEntropyLoss'),
            dir_res_loss=dict(
                loss_weight=10.0, reduction='sum', type='mmdet.SmoothL1Loss'),
            iou_loss=dict(
                loss_weight=1.0, reduction='sum', type='AxisAlignedIoULoss'),
            num_classes=1,
            objectness_loss=dict(
                class_weight=[
                    0.2,
                    0.8,
                ],
                loss_weight=5.0,
                reduction='sum',
                type='mmdet.CrossEntropyLoss'),
            pred_layer_cfg=dict(
                bias=True, in_channels=128, shared_conv_channels=(
                    128,
                    128,
                )),
            semantic_loss=dict(
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.CrossEntropyLoss'),
            size_class_loss=dict(
                loss_weight=1.0,
                reduction='sum',
                type='mmdet.CrossEntropyLoss'),
            size_res_loss=dict(
                loss_weight=3.3333333333333335,
                reduction='sum',
                type='mmdet.SmoothL1Loss'),
            type='VoteHead'),
        img=dict(
            vote_aggregation_cfg=dict(
                mlp_channels=[
                    256,
                    128,
                    128,
                    128,
                ],
                normalize_xyz=True,
                num_point=256,
                num_sample=16,
                radius=0.3,
                type='PointSAModule',
                use_xyz=True),
            vote_module_cfg=dict(
                conv_cfg=dict(type='Conv1d'),
                conv_channels=(
                    256,
                    256,
                ),
                gt_per_seed=3,
                in_channels=256,
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    loss_dst_weight=10.0,
                    mode='l1',
                    reduction='none',
                    type='ChamferDistance'),
                vote_per_seed=1)),
        joint=dict(
            vote_aggregation_cfg=dict(
                mlp_channels=[
                    256,
                    128,
                    128,
                    128,
                ],
                normalize_xyz=True,
                num_point=256,
                num_sample=16,
                radius=0.3,
                type='PointSAModule',
                use_xyz=True),
            vote_module_cfg=dict(
                conv_cfg=dict(type='Conv1d'),
                conv_channels=(
                    256,
                    256,
                ),
                gt_per_seed=3,
                in_channels=256,
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    loss_dst_weight=10.0,
                    mode='l1',
                    reduction='none',
                    type='ChamferDistance'),
                vote_per_seed=1)),
        loss_weights=[
            1.0,
        ],
        pts=dict(
            vote_aggregation_cfg=dict(
                mlp_channels=[
                    256,
                    128,
                    128,
                    128,
                ],
                normalize_xyz=True,
                num_point=256,
                num_sample=16,
                radius=0.3,
                type='PointSAModule',
                use_xyz=True),
            vote_module_cfg=dict(
                conv_cfg=dict(type='Conv1d'),
                conv_channels=(
                    256,
                    256,
                ),
                gt_per_seed=3,
                in_channels=256,
                norm_cfg=dict(type='BN1d'),
                norm_feats=True,
                vote_loss=dict(
                    loss_dst_weight=10.0,
                    mode='l1',
                    reduction='none',
                    type='ChamferDistance'),
                vote_per_seed=1))),
    sam_fre=dict(
        depth=4,
        embed_dim=256,
        init_cfg=dict(layer='Conv2d', type='Kaiming'),
        scale_factor=4,
        type='FrequencyPromptAdapter'),
    test_cfg=dict(
        img_rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.1),
        img_rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=1000,
            nms_pre=1000),
        pts=dict(
            nms_thr=0.25,
            per_class_proposal=True,
            sample_mode='seed',
            score_thr=0.05)),
    train_cfg=dict(
        _scope_='mmdet',
        img_rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        img_rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        img_rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=1000,
            nms_pre=2000),
        pts=dict(
            neg_distance_thr=0.6, pos_distance_thr=0.3, sample_mode='vote')),
    type='ImVoteNet',
    use_aux_heads=False)
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.0008, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=50,
        gamma=0.1,
        milestones=[
            20,
            40,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=8)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file='sunrgbd_trainval/sunrgbd_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root=data_root,
        metainfo=dict(classes=('luggage', )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                backend_args=None,
                coord_type='DEPTH',
                load_dim=6,
                shift_height=True,
                type='LoadPointsFromFile',
                use_dim=[
                    0,
                    1,
                    2,
                ]),
            dict(
                key='sam_feat_path',
                output_key='sam_feat',
                type='LoadSamFeature'),
            dict(keep_ratio=True, scale=(
                1333,
                600,
            ), type='Resize'),
            dict(num_points=20000, type='PointSample'),
            dict(keys=[
                'img',
                'points',
                'sam_feat',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='LuggageSUNRGBDDataset'),
    num_workers=8,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='IndoorMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        backend_args=None,
        coord_type='DEPTH',
        load_dim=6,
        shift_height=True,
        type='LoadPointsFromFile',
        use_dim=[
            0,
            1,
            2,
        ]),
    dict(key='sam_feat_path', output_key='sam_feat', type='LoadSamFeature'),
    dict(keep_ratio=True, scale=(
        1333,
        600,
    ), type='Resize'),
    dict(num_points=20000, type='PointSample'),
    dict(keys=[
        'img',
        'points',
        'sam_feat',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=7,
    dataset=dict(
        dataset=dict(
            ann_file='sunrgbd_trainval/sunrgbd_infos_train.pkl',
            backend_args=None,
            box_type_3d='Depth',
            data_root=data_root,
            filter_empty_gt=False,
            metainfo=dict(classes=('luggage', )),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='DEPTH',
                    load_dim=6,
                    shift_height=True,
                    type='LoadPointsFromFile',
                    use_dim=[
                        0,
                        1,
                        2,
                    ]),
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(
                    key='sam_feat_path',
                    output_key='sam_feat',
                    type='LoadSamFeature'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox=False,
                    with_bbox_3d=True,
                    with_label=False,
                    with_label_3d=True),
                dict(keep_ratio=True, scale=(
                    1333,
                    600,
                ), type='Resize'),
                dict(
                    flip_ratio_bev_horizontal=0.5,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.523599,
                        0.523599,
                    ],
                    scale_ratio_range=[
                        0.85,
                        1.15,
                    ],
                    shift_height=True,
                    type='GlobalRotScaleTrans'),
                dict(num_points=20000, type='PointSample'),
                dict(
                    keys=[
                        'img',
                        'gt_bboxes',
                        'gt_bboxes_labels',
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                        'sam_feat',
                    ],
                    type='Pack3DDetInputs'),
            ],
            type='LuggageSUNRGBDDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=8,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='DEPTH',
        load_dim=6,
        shift_height=True,
        type='LoadPointsFromFile',
        use_dim=[
            0,
            1,
            2,
        ]),
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(key='sam_feat_path', output_key='sam_feat', type='LoadSamFeature'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=False,
        with_bbox_3d=True,
        with_label=False,
        with_label_3d=True),
    dict(keep_ratio=True, scale=(
        1333,
        600,
    ), type='Resize'),
    dict(flip_ratio_bev_horizontal=0.5, sync_2d=False, type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.523599,
            0.523599,
        ],
        scale_ratio_range=[
            0.85,
            1.15,
        ],
        shift_height=True,
        type='GlobalRotScaleTrans'),
    dict(num_points=20000, type='PointSample'),
    dict(
        keys=[
            'img',
            'gt_bboxes',
            'gt_bboxes_labels',
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'sam_feat',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=10,
    dataset=dict(
        ann_file='sunrgbd_trainval/sunrgbd_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root=data_root,
        metainfo=dict(classes=('luggage', )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                backend_args=None,
                coord_type='DEPTH',
                load_dim=6,
                shift_height=True,
                type='LoadPointsFromFile',
                use_dim=[
                    0,
                    1,
                    2,
                ]),
            dict(
                key='sam_feat_path',
                output_key='sam_feat',
                type='LoadSamFeature'),
            dict(keep_ratio=True, scale=(
                1333,
                600,
            ), type='Resize'),
            dict(num_points=20000, type='PointSample'),
            dict(keys=[
                'img',
                'points',
                'sam_feat',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='LuggageSUNRGBDDataset'),
    num_workers=8,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='IndoorMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/vpgnet_airport_luggage'
