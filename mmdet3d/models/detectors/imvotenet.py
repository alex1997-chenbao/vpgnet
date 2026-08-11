# Copyright (c) OpenMMLab. All rights reserved.
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from ..layers import MLP
from .base import Base3DDetector


def sample_valid_seeds(mask: Tensor, num_sampled_seed: int = 1024) -> Tensor:
    r"""Randomly sample seeds from all imvotes.

    Modified from `<https://github.com/facebookresearch/imvotenet/blob/a8856345146bacf29a57266a2f0b874406fd8823/models/imvotenet.py#L26>`_

    Args:
        mask (torch.Tensor): Bool tensor in shape (
            seed_num*max_imvote_per_pixel), indicates
            whether this imvote corresponds to a 2D bbox.
        num_sampled_seed (int): How many to sample from all imvotes.

    Returns:
        torch.Tensor: Indices with shape (num_sampled_seed).
    """  # noqa: E501
    device = mask.device
    batch_size = mask.shape[0]
    sample_inds = mask.new_zeros((batch_size, num_sampled_seed),
                                 dtype=torch.int64)
    for bidx in range(batch_size):
        # return index of non zero elements
        valid_inds = torch.nonzero(mask[bidx, :]).squeeze(-1)
        if len(valid_inds) < num_sampled_seed:
            # compute set t1 - t2
            t1 = torch.arange(num_sampled_seed, device=device)
            t2 = valid_inds % num_sampled_seed
            combined = torch.cat((t1, t2))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]

            rand_inds = torch.randperm(
                len(difference),
                device=device)[:num_sampled_seed - len(valid_inds)]
            cur_sample_inds = difference[rand_inds]
            cur_sample_inds = torch.cat((valid_inds, cur_sample_inds))
        else:
            rand_inds = torch.randperm(
                len(valid_inds), device=device)[:num_sampled_seed]
            cur_sample_inds = valid_inds[rand_inds]
        sample_inds[bidx, :] = cur_sample_inds
    return sample_inds


@MODELS.register_module()
class ImVoteNet(Base3DDetector):
    r"""`ImVoteNet <https://arxiv.org/abs/2001.10692>`_ for 3D detection.

    ImVoteNet is based on fusing 2D votes in images and 3D votes in point
    clouds, which explicitly extract both geometric and semantic features
    from the 2D images. It leverage camera parameters to lift these
    features to 3D. A multi-tower training scheme also improve the synergy
    of 2D-3D feature fusion.

    """

    def __init__(self,
                 geometric_refiner: Optional[dict] = None,
                 fusion_cfg: Optional[dict] = None,
                 prompt_neck: Optional[dict] = None,
                 fc_cross_attn: Optional[dict] = None,
                 sam_fre: Optional[dict] = None,
                 use_fusion_gate: bool = True,
                 use_aux_heads: bool = False,
                 pts_backbone: Optional[dict] = None,
                 pts_bbox_heads: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 img_mlp: Optional[dict] = None,
                 freeze_img_branch: bool = False,
                 fusion_layer: Optional[dict] = None,
                 num_sampled_seed: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:

        super(ImVoteNet, self).__init__(init_cfg=init_cfg, **kwargs)
        self.local_atten = MODELS.build(
            fc_cross_attn) if fc_cross_attn is not None else None
        self.sam_fre_fused = MODELS.build(
            sam_fre) if sam_fre is not None else None
        self.local_refiner = MODELS.build(
            geometric_refiner) if geometric_refiner is not None else None
        # self.freprompts=MODELS.build(prompt_neck)
        # self.fusion_layer=MODELS.build(fusion_cfg)
        # point branch
        if pts_backbone is not None:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = MODELS.build(pts_neck)
        if pts_bbox_heads is not None:
            pts_bbox_head_common = pts_bbox_heads.common
            pts_bbox_head_common.update(
                train_cfg=train_cfg.pts if train_cfg is not None else None)
            pts_bbox_head_common.update(test_cfg=test_cfg.pts)
            pts_bbox_head_joint = pts_bbox_head_common.copy()
            pts_bbox_head_joint.update(pts_bbox_heads.joint)
            self.pts_bbox_head_joint = MODELS.build(pts_bbox_head_joint)
            self.pts_bbox_heads = [self.pts_bbox_head_joint]
            self.loss_weights = [1.0]
            self.pts_bbox_head_pts = None

            if use_aux_heads:
                pts_bbox_head_pts = pts_bbox_head_common.copy()
                pts_bbox_head_pts.update(pts_bbox_heads.pts)
                self.pts_bbox_head_pts = MODELS.build(pts_bbox_head_pts)
                self.pts_bbox_heads.append(self.pts_bbox_head_pts)
                self.loss_weights = list(pts_bbox_heads.loss_weights[:len(
                    self.pts_bbox_heads)])

        # image branch
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_rpn_head is not None:
            rpn_train_cfg = train_cfg.img_rpn if train_cfg \
                is not None else None
            img_rpn_head_ = img_rpn_head.copy()
            img_rpn_head_.update(
                train_cfg=rpn_train_cfg, test_cfg=test_cfg.img_rpn)
            self.img_rpn_head = MODELS.build(img_rpn_head_)
        if img_roi_head is not None:
            rcnn_train_cfg = train_cfg.img_rcnn if train_cfg \
                is not None else None
            img_roi_head.update(
                train_cfg=rcnn_train_cfg, test_cfg=test_cfg.img_rcnn)
            self.img_roi_head = MODELS.build(img_roi_head)

        # fusion
        # if fusion_layer is not None:
        #     self.fusion_layer = MODELS.build(fusion_layer)
        #     self.max_imvote_per_pixel = fusion_layer.max_imvote_per_pixel

        self.freeze_img_branch = freeze_img_branch
        if freeze_img_branch:
            self.freeze_img_branch_params()

        if img_mlp is not None:
            self.img_mlp = MLP(**img_mlp)

        self.num_sampled_seed = num_sampled_seed

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_fusion_gate = use_fusion_gate
        self.gate_net = None
        if self.use_fusion_gate:
            self.gate_net = nn.Sequential(
                nn.Conv1d(256 * 2, 256, 1),
                nn.Sigmoid())

            # 初始化技巧：让 Gate 一开始关闭
            nn.init.constant_(self.gate_net[0].bias, -4.0)
        self._sam_loader_pool: Optional[ThreadPoolExecutor] = None

    def _forward(self):
        raise NotImplementedError

    def freeze_img_branch_params(self):
        """Freeze all image branch parameters."""
        if self.with_img_bbox_head:
            for param in self.img_bbox_head.parameters():
                param.requires_grad = False
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.with_img_rpn:
            for param in self.img_rpn_head.parameters():
                param.requires_grad = False
        if self.with_img_roi_head:
            for param in self.img_roi_head.parameters():
                param.requires_grad = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load img network ckpts into img branch."""
        module_names = ['backbone', 'neck', 'roi_head', 'rpn_head']
        for key in list(state_dict):
            for module_name in module_names:
                if key.startswith(module_name) and ('img_' +
                                                    key) not in state_dict:
                    state_dict['img_' + key] = state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def train(self, mode=True):
        """Overload in order to keep image branch modules in eval mode."""
        super(ImVoteNet, self).train(mode)
        if self.freeze_img_branch:
            if self.with_img_bbox_head:
                self.img_bbox_head.eval()
            if self.with_img_backbone:
                self.img_backbone.eval()
            if self.with_img_neck:
                self.img_neck.eval()
            if self.with_img_rpn:
                self.img_rpn_head.eval()
            if self.with_img_roi_head:
                self.img_roi_head.eval()

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return ((hasattr(self, 'img_roi_head') and self.img_roi_head.with_bbox)
                or (hasattr(self, 'img_bbox_head')
                    and self.img_bbox_head is not None))

    @property
    def with_img_bbox_head(self):
        """bool: Whether the detector has a 2D image box head (not roi)."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    def extract_feat(self, imgs):
        """Just to inherit from abstract method."""
        pass

    def extract_img_feat(self, img: Tensor) -> Sequence[Tensor]:
        """Directly extract features from the img backbone+neck."""
        if not self.with_img_backbone:
            raise RuntimeError('img_backbone is required for image branch.')
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_pts_feat(self, pts: Tensor) -> Tuple[Tensor]:
        """Extract features of points."""
        x = self.pts_backbone(pts)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        seed_points = x['fp_xyz'][-1]
        seed_features = x['fp_features'][-1]
        seed_indices = x['fp_indices'][-1]


        support_points = x['fp_xyz'][-2]
        support_features = x['fp_features'][-2]


        return (seed_points, seed_features, seed_indices, support_points, support_features)

    def _get_sam_loader_pool(self) -> ThreadPoolExecutor:
        """Lazily create the disk IO pool used by the fallback SAM loader."""
        if self._sam_loader_pool is None:
            self._sam_loader_pool = ThreadPoolExecutor(max_workers=12)
        return self._sam_loader_pool

    @staticmethod
    def _load_torch_tensor_cpu(path: str) -> Tensor:
        """Load a tensor from disk with fast fallbacks across torch versions."""
        load_variants = (
            dict(map_location='cpu', weights_only=True, mmap=True),
            dict(map_location='cpu', weights_only=True),
            dict(map_location='cpu', mmap=True),
            dict(map_location='cpu'))
        for kwargs in load_variants:
            try:
                return torch.load(path, **kwargs)
            except TypeError:
                continue
        return torch.load(path, map_location='cpu')

    @staticmethod
    def _ensure_4d_sam_feature(s_feat: Tensor) -> Tensor:
        """Normalize SAM features to `(1, C, H, W)`."""
        if s_feat.dim() == 2:
            return s_feat.unsqueeze(0).unsqueeze(0)
        if s_feat.dim() == 3:
            return s_feat.unsqueeze(0)
        if s_feat.dim() == 4 and s_feat.size(0) == 1:
            return s_feat
        raise ValueError(f'Unexpected SAM feature shape: {tuple(s_feat.shape)}')

    @staticmethod
    def _resolve_batch_sam_shape(
            valid_results: List[Tuple[Tensor, Optional[dict]]]) -> Tuple[int, int]:
        """Infer the common raw spatial shape used by a feature batch."""
        target_h, target_w = 0, 0
        for s_feat, _ in valid_results:
            if s_feat.dim() == 2:
                h, w = s_feat.shape
            else:
                h, w = s_feat.shape[-2:]
            target_h = max(target_h, int(h))
            target_w = max(target_w, int(w))

        return target_h, target_w

    def _extract_joint_vote_features(
            self,
            points: List[Tensor],
            imgs: Optional[Tensor],
            batch_data_samples: List[Det3DDataSample],
            sam_inputs: Optional[Union[List[Tensor], Tensor]] = None) -> dict:
        """Prepare the shared point-image features used by train/test heads."""
        stack_points = torch.stack(points)
        seeds_3d, seed_3d_features, seed_indices, support_points, \
            support_features = self.extract_pts_feat(stack_points)

        fused_feat = seed_3d_features
        if self.local_atten is not None:
            sam_feat_batch, sam_mask = self.load_sam_and_mask_batch(
                batch_data_samples,
                device=points[0].device,
                sam_inputs=sam_inputs)
            img_guided_feat, _ = self.local_atten(
                pts_feat=seed_3d_features,
                img_feat=sam_feat_batch,
                seeds_3d=seeds_3d,
                metas=batch_data_samples,
                mask_labels=sam_mask,
                imgs=imgs)
            if self.use_fusion_gate:
                gate = self.gate_net(
                    torch.cat([seed_3d_features, img_guided_feat], dim=1))
                fused_feat = seed_3d_features + gate * img_guided_feat
            else:
                fused_feat = seed_3d_features + img_guided_feat

        if self.local_refiner is not None:
            refined_feat = self.local_refiner(
                query_xyz=seeds_3d,
                query_feat=fused_feat,
                support_xyz=support_points,
                support_feat=support_features)
            fused_feat = seed_3d_features + refined_feat

        return dict(
            seed_points=seeds_3d,
            seed_features=fused_feat,
            seed_indices=seed_indices)

    def load_sam_and_mask_batch(self,
                                batch_data_samples: List[Det3DDataSample],
                                device: Optional[torch.device] = None,
                                sam_inputs: Optional[Union[List[Tensor],
                                                           Tensor]] = None):
        B = len(batch_data_samples)
        if device is None:
            device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        meta_list = [data_sample.metainfo for data_sample in batch_data_samples]

        if sam_inputs is not None:
            if torch.is_tensor(sam_inputs):
                if sam_inputs.dim() == 4 and sam_inputs.size(0) == B:
                    target_shape = self._resolve_batch_sam_shape(
                        [(sam_inputs[i], meta_list[i]) for i in range(B)])
                    no_flip = all(not meta.get('flip', False)
                                  for meta in meta_list)
                    if sam_inputs.shape[-2:] == target_shape and no_flip:
                        return sam_inputs.to(
                            device=device,
                            dtype=model_dtype,
                            non_blocking=(device.type == 'cuda')), None
                if sam_inputs.dim() == 4 and sam_inputs.size(0) == B:
                    sam_raw_list = [sam_inputs[i] for i in range(B)]
                elif sam_inputs.dim() == 3 and B == 1:
                    sam_raw_list = [sam_inputs]
                else:
                    raise ValueError(
                        f'Unexpected sam_inputs tensor shape: {tuple(sam_inputs.shape)}, '
                        f'batch_size={B}')
            elif isinstance(sam_inputs, (list, tuple)):
                if len(sam_inputs) != B:
                    raise ValueError(
                        f'Length mismatch between sam_inputs ({len(sam_inputs)}) and '
                        f'batch_data_samples ({B}).')
                sam_raw_list = list(sam_inputs)
            else:
                raise TypeError(
                    f'Unsupported sam_inputs type: {type(sam_inputs)}')
            valid_results = [(feat, data_sample.metainfo)
                             for feat, data_sample in zip(sam_raw_list,
                                                          batch_data_samples)]
        else:
            def load_raw_from_disk(data_sample):
                """Only handle feature IO in the slow fallback path."""
                img_path = getattr(data_sample, 'img_path', None)
                if img_path is None:
                    img_path = data_sample.metainfo.get('img_path', None)
                if img_path is None:
                    return None, None, 'missing img_path in data sample'
                sam_path = img_path.replace('/image/',
                                            '/sam_f/').replace('.jpg', '.pt')
                try:
                    s = self._load_torch_tensor_cpu(sam_path)
                    return s, None, data_sample.metainfo
                except Exception as e:
                    return None, sam_path, str(e)

            raw_results = list(
                self._get_sam_loader_pool().map(load_raw_from_disk,
                                                batch_data_samples))
            failed = [res for res in raw_results if res[0] is None]
            if failed:
                fail_msg = '; '.join(
                    [f'{path}: {err}' for _, path, err in failed[:3]])
                raise RuntimeError(
                    f'Failed to load SAM features for batch: {fail_msg}')
            valid_results = [(feat, meta) for feat, _, meta in raw_results]

        batch_h, batch_w = self._resolve_batch_sam_shape(valid_results)
        sam_feat_batch = None

        for sample_idx, (s_feat, meta) in enumerate(valid_results):
            meta = meta or {}
            s_feat = self._ensure_4d_sam_feature(s_feat)
            if s_feat.device != device:
                s_feat = s_feat.to(
                    device=device,
                    dtype=model_dtype,
                    non_blocking=(device.type == 'cuda'))
            elif s_feat.dtype != model_dtype:
                s_feat = s_feat.to(dtype=model_dtype)

            if meta.get('flip', False):
                s_feat = s_feat.flip(dims=[-1])

            sample_h, sample_w = s_feat.shape[-2:]

            if sam_feat_batch is None:
                sam_feat_batch = s_feat.new_zeros((B, s_feat.shape[1], batch_h,
                                                   batch_w))
            sam_feat_batch[sample_idx, :, :sample_h, :sample_w] = s_feat[0]

        return sam_feat_batch.contiguous(), None

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points', 'imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (list[torch.Tensor]): Image tensor with shape
                  (N, C, H ,W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        sam_inputs = batch_inputs_dict.get('sam_feat', None)
        if points is None:
            x = self.extract_img_feat(imgs)
            losses = dict()
            # RPN forward and loss
            if self.with_img_rpn:
                proposal_cfg = self.train_cfg.get('img_rpn_proposal',
                                                  self.test_cfg.img_rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = \
                    self.img_rpn_head.loss_and_predict(
                        x, rpn_data_samples,
                        proposal_cfg=proposal_cfg, **kwargs)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for
                # the second stage
                # to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            roi_losses = self.img_roi_head.loss(x, rpn_results_list,
                                                batch_data_samples, **kwargs)
            losses.update(roi_losses)
            return losses
        else:
            feat_dict_joint = self._extract_joint_vote_features(
                points=points,
                imgs=imgs,
                batch_data_samples=batch_data_samples,
                sam_inputs=sam_inputs)
            # fused_pts_feat = self.fusion_layer(
            #     pts_feat=seed_3d_features,
            #     imgs=imgs,
            #     sam_feat=sam3_feat,
            #     seeds_3d=seeds_3d,
            #     img_metas=img_metas
            # )

            # print(seed_3d_features.shape,img_feat_sampled.shape,fused_feat.shape,sam3_fuse_fre.shape)
            # tensors = {
            #     "seed_3d_features": seed_3d_features,
            #     "img_feat_sampled": img_feat_sampled,
            #     "fused_feat": fused_feat,
            #     "sam3_fuse_fre": sam3_fuse_fre,
            # }

            # print("-" * 50)

            # # --- 2. 批量计算并显示全局均值和方差 ---

            # for name, tensor in tensors.items():
                
            #     # 打印张量形状以进行确认
            #     print(f"张量名称: {name}")
            #     print(f"  张量形状: {tensor.shape}")
                
            #     # 计算全局均值 (Mean)
            #     # torch.mean() 默认在所有元素上求平均，返回一个标量
            #     mean_val = torch.mean(tensor)
                
            #     # 计算全局方差 (Variance)
            #     # unbiased=True 计算样本方差 (除以 n-1)，这是常用的统计量
            #     var_val = torch.var(tensor, unbiased=True)
                
            #     print(f"  全局均值 (Global Mean): {mean_val.item():.6f}")
            #     print(f"  全局方差 (Global Variance): {var_val.item():.6f}")
            #     print("-" * 50)
            # del fre_sam_cross
            
            # img_metas = [item.metainfo for item in batch_data_samples]
            # img_features, masks = self.fusion_layer(
            #     imgs, pred_bboxes_with_label_list, seeds_3d, img_metas)

            # inds = sample_valid_seeds(masks, self.num_sampled_seed)



            # batch_size, img_feat_size = img_features.shape[:2]
            # pts_feat_size = seed_3d_features.shape[1]
            # inds_img = inds.view(batch_size, 1,
            #                      -1).expand(-1, img_feat_size, -1)
            # img_features = img_features.gather(-1, inds_img)
            # inds = inds % inds.shape[1]
            # inds_seed_xyz = inds.view(batch_size, -1, 1).expand(-1, -1, 3)
            # seeds_3d = seeds_3d.gather(1, inds_seed_xyz)
            # inds_seed_feats = inds.view(batch_size, 1,
            #                             -1).expand(-1, pts_feat_size, -1)
            # seed_3d_features = seed_3d_features.gather(-1, inds_seed_feats)
            # seed_indices = seed_indices.gather(1, inds)

            # img_features = self.img_mlp(img_features)
            # fused_features = torch.cat([seed_3d_features, img_features], dim=1)
            # feat_dict_pts = dict(
            #     seed_points=seeds_3d,
            #     seed_features=seed_3d_features,
            #     seed_indices=seed_indices)
            # feat_dict_img = dict(
            #     seed_points=seeds_3d,
            #     seed_features=img_feat_sampled.squeeze(-1),
            #     seed_indices=seed_indices)

            losses_towers = []
            losses_joint = self.pts_bbox_head_joint.loss(
                points, feat_dict_joint, batch_data_samples)
            # losses_pts = self.pts_bbox_head_pts.loss(points, feat_dict_pts,
            #                                          batch_data_samples)
            # losses_img = self.pts_bbox_head_img.loss(points, feat_dict_img,
            #                                          batch_data_samples)
            losses_towers.append(losses_joint)
            # losses_towers.append(losses_pts)
            # losses_towers.append(losses_img)
            combined_losses = dict()
            for loss_term in losses_joint:
                if 'loss' in loss_term:
                    combined_losses[loss_term] = 0
                    for i in range(len(losses_towers)):
                        combined_losses[loss_term] += \
                            losses_towers[i][loss_term] * \
                            self.loss_weights[i]
                else:
                    # only save the metric of the joint head
                    # if it is not a loss
                    combined_losses[loss_term] = \
                        losses_towers[0][loss_term]

            return combined_losses

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
            batch_data_samples: List[Det3DDataSample],
            **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and 'imgs keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (list[torch.Tensor]): Tensor of Images.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        sam_inputs = batch_inputs_dict.get('sam_feat', None)
        
        # 纯图像检测模式 (逻辑不变)
        if points is None:
            assert imgs is not None
            results_2d = self.predict_img_only(imgs, batch_data_samples)
            return self.add_pred_to_datasample(
                batch_data_samples, data_instances_2d=results_2d)

        else:
            feat_dict = self._extract_joint_vote_features(
                points=points,
                imgs=imgs,
                batch_data_samples=batch_data_samples,
                sam_inputs=sam_inputs)

            # --- 6. 3D 检测头预测 ---
            # 只使用联合塔 (pts_bbox_head_joint) 进行最终预测
            results_3d = self.pts_bbox_head_joint.predict(
                batch_inputs_dict['points'],
                feat_dict,
                batch_data_samples,
                rescale=True,
                use_nms=True)
            return self.add_pred_to_datasample(
                batch_data_samples, data_instances_3d=results_3d)

    def predict_img_only(self,
                         imgs: Tensor,
                         batch_data_samples: List[Det3DDataSample],
                         rescale: bool = True) -> List[InstanceData]:
        """Predict results from a batch of imgs with post- processing."""

        assert self.with_img_bbox, 'Img bbox head must be implemented.'
        assert self.with_img_backbone, 'Img backbone must be implemented.'
        assert self.with_img_rpn, 'Img rpn must be implemented.'
        assert self.with_img_roi_head, 'Img roi head must be implemented.'
        x = self.extract_img_feat(imgs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.img_rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.img_roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        return results_list
