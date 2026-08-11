# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# from mmcv.ops import furthest_point_sample
# from mmdet.models.utils import multi_apply
# from mmengine import ConfigDict
# from mmengine.model import BaseModule
# from mmengine.structures import InstanceData
# from torch import Tensor
# from torch.nn import functional as F

# from mmdet3d.models.layers import VoteModule, aligned_3d_nms, build_sa_module
# from mmdet3d.models.losses import chamfer_distance
# from mmdet3d.registry import MODELS, TASK_UTILS
# from mmdet3d.structures import Det3DDataSample
# from .base_conv_bbox_head import BaseConvBboxHead
# from torch_scatter import scatter_mean
# from scipy.optimize import linear_sum_assignment

# @MODELS.register_module()
# class VoteHead(BaseModule):
#     r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.

#     Args:
#         num_classes (int): The number of class.
#         bbox_coder (ConfigDict, dict): Bbox coder for encoding and
#             decoding boxes. Defaults to None.
#         train_cfg (dict, optional): Config for training. Defaults to None.
#         test_cfg (dict, optional): Config for testing. Defaults to None.
#         vote_module_cfg (dict, optional): Config of VoteModule for
#             point-wise votes. Defaults to None.
#         vote_aggregation_cfg (dict, optional): Config of vote
#             aggregation layer. Defaults to None.
#         pred_layer_cfg (dict, optional): Config of classification
#             and regression prediction layers. Defaults to None.
#         objectness_loss (dict, optional): Config of objectness loss.
#             Defaults to None.
#         center_loss (dict, optional): Config of center loss.
#             Defaults to None.
#         dir_class_loss (dict, optional): Config of direction
#             classification loss. Defaults to None.
#         dir_res_loss (dict, optional): Config of direction
#             residual regression loss. Defaults to None.
#         size_class_loss (dict, optional): Config of size
#             classification loss. Defaults to None.
#         size_res_loss (dict, optional): Config of size
#             residual regression loss. Defaults to None.
#         semantic_loss (dict, optional): Config of point-wise
#             semantic segmentation loss. Defaults to None.
#         iou_loss (dict, optional): Config of IOU loss for
#             regression. Defaults to None.
#         init_cfg (dict, optional): Config of model weight
#             initialization. Defaults to None.
#     """

#     def __init__(self,
#                  num_classes: int,
#                  bbox_coder: Union[ConfigDict, dict],
#                  train_cfg: Optional[dict] = None,
#                  test_cfg: Optional[dict] = None,
#                  vote_module_cfg: Optional[dict] = None,
#                  vote_aggregation_cfg: Optional[dict] = None,
#                  pred_layer_cfg: Optional[dict] = None,
#                  objectness_loss: Optional[dict] = None,
#                  center_loss: Optional[dict] = None,
#                  dir_class_loss: Optional[dict] = None,
#                  dir_res_loss: Optional[dict] = None,
#                  size_class_loss: Optional[dict] = None,
#                  size_res_loss: Optional[dict] = None,
#                  semantic_loss: Optional[dict] = None,
#                  iou_loss: Optional[dict] = None,
#                  init_cfg: Optional[dict] = None):
#         super(VoteHead, self).__init__(init_cfg=init_cfg)
#         self.num_classes = num_classes
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#         self.gt_per_seed = vote_module_cfg['gt_per_seed']
#         self.num_proposal = vote_aggregation_cfg['num_point']

#         self.loss_objectness = MODELS.build(objectness_loss)
#         self.loss_center = MODELS.build(center_loss)
#         self.loss_dir_res = MODELS.build(dir_res_loss)
#         self.loss_dir_class = MODELS.build(dir_class_loss)
#         self.loss_size_res = MODELS.build(size_res_loss)
#         if size_class_loss is not None:
#             self.size_class_loss = MODELS.build(size_class_loss)
#         if semantic_loss is not None:
#             self.semantic_loss = MODELS.build(semantic_loss)
#         if iou_loss is not None:
#             self.iou_loss = MODELS.build(iou_loss)
#         else:
#             self.iou_loss = None

#         self.bbox_coder = TASK_UTILS.build(bbox_coder)
#         self.num_sizes = self.bbox_coder.num_sizes
#         self.num_dir_bins = self.bbox_coder.num_dir_bins

#         self.vote_module = VoteModule(**vote_module_cfg)
#         self.vote_aggregation = build_sa_module(vote_aggregation_cfg)
#         import torch.nn as nn
#         # 假设输入是 256，输出对齐你的 pred_layer_cfg 要求的 128
#         self.channel_mapper = nn.Conv1d(256, 128, kernel_size=1)
#         # Bbox classification and regression
#         self.conv_pred = BaseConvBboxHead(
#             **pred_layer_cfg,
#             num_cls_out_channels=self._get_cls_out_channels(),
#             num_reg_out_channels=self._get_reg_out_channels())

#     @property
#     def sample_mode(self):
#         if self.training:
#             sample_mode = self.train_cfg.sample_mode
#         else:
#             sample_mode = self.test_cfg.sample_mode
#         assert sample_mode in ['vote', 'seed', 'random', 'spec']
#         return sample_mode

#     def _get_cls_out_channels(self):
#         """Return the channel number of classification outputs."""
#         # Class numbers (k) + objectness (2)
#         return self.num_classes + 2

#     def _get_reg_out_channels(self):
#         """Return the channel number of regression outputs."""
#         # Objectness scores (2), center residual (3),
#         # heading class+residual (num_dir_bins*2),
#         # size class+residual(num_sizes*4)
#         return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

#     def _extract_input(self, feat_dict: dict) -> tuple:
#         """Extract inputs from features dictionary.

#         Args:
#             feat_dict (dict): Feature dict from backbone.

#         Returns:
#             tuple[Tensor]: Arrage as following three tensor.

#                 - Coordinates of input points.
#                 - Features of input points.
#                 - Indices of input points.
#         """

#         # for imvotenet
#         if 'seed_points' in feat_dict and \
#            'seed_features' in feat_dict and \
#            'seed_indices' in feat_dict:
#             seed_points = feat_dict['seed_points']
#             seed_features = feat_dict['seed_features']
#             seed_indices = feat_dict['seed_indices']
#             seed_masks = feat_dict.get('seed_masks', None)
#         # for votenet
#         else:
#             seed_points = feat_dict['fp_xyz'][-1]
#             seed_features = feat_dict['fp_features'][-1]
#             seed_indices = feat_dict['fp_indices'][-1]
#             seed_masks = None # 显式添加这一行，确保返回值始终包含 4 个元素

#         return seed_points, seed_features, seed_indices, seed_masks

#     def predict(self,
#                 points: List[torch.Tensor],
#                 feats_dict: Dict[str, torch.Tensor],
#                 batch_data_samples: List[Det3DDataSample],
#                 use_nms: bool = True,
#                 **kwargs) -> List[InstanceData]:
#         """
#         Args:
#             points (list[tensor]): Point clouds of multiple samples.
#             feats_dict (dict): Features from FPN or backbone..
#             batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
#                 Samples. It usually includes meta information of data.
#             use_nms (bool): Whether do the nms for predictions.
#                 Defaults to True.

#         Returns:
#             list[:obj:`InstanceData`]: List of processed predictions. Each
#             InstanceData contains 3d Bounding boxes and corresponding
#             scores and labels.
#         """
#         preds_dict = self(feats_dict)
#         # `preds_dict` can be used in H3DNET
#         feats_dict.update(preds_dict)

#         batch_size = len(batch_data_samples)
#         batch_input_metas = []
#         for batch_index in range(batch_size):
#             metainfo = batch_data_samples[batch_index].metainfo
#             batch_input_metas.append(metainfo)

#         results_list = self.predict_by_feat(
#             points, preds_dict, batch_input_metas, use_nms=use_nms, **kwargs)
#         return results_list

#     def loss_and_predict(self,
#                          points: List[torch.Tensor],
#                          feats_dict: Dict[str, torch.Tensor],
#                          batch_data_samples: List[Det3DDataSample],
#                          ret_target: bool = False,
#                          proposal_cfg: dict = None,
#                          **kwargs) -> Tuple:
#         """
#         Args:
#             points (list[tensor]): Points cloud of multiple samples.
#             feats_dict (dict): Predictions from backbone or FPN.
#             batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
#                 contains the meta information of each sample and
#                 corresponding annotations.
#             ret_target (bool): Whether return the assigned target.
#                 Defaults to False.
#             proposal_cfg (dict): Configure for proposal process.
#                 Defaults to True.

#         Returns:
#             tuple:  Contains loss and predictions after post-process.
#         """
#         preds_dict = self.forward(feats_dict)
#         feats_dict.update(preds_dict)
#         batch_gt_instance_3d = []
#         batch_gt_instances_ignore = []
#         batch_input_metas = []
#         batch_pts_semantic_mask = []
#         batch_pts_instance_mask = []
#         for data_sample in batch_data_samples:
#             batch_input_metas.append(data_sample.metainfo)
#             batch_gt_instance_3d.append(data_sample.gt_instances_3d)
#             batch_gt_instances_ignore.append(
#                 data_sample.get('ignored_instances', None))
#             batch_pts_semantic_mask.append(
#                 data_sample.gt_pts_seg.get('pts_semantic_mask', None))
#             batch_pts_instance_mask.append(
#                 data_sample.gt_pts_seg.get('pts_instance_mask', None))

#         loss_inputs = (points, preds_dict, batch_gt_instance_3d)
#         losses = self.loss_by_feat(
#             *loss_inputs,
#             batch_pts_semantic_mask=batch_pts_semantic_mask,
#             batch_pts_instance_mask=batch_pts_instance_mask,
#             batch_input_metas=batch_input_metas,
#             batch_gt_instances_ignore=batch_gt_instances_ignore,
#             ret_target=ret_target,
#             **kwargs)

#         results_list = self.predict_by_feat(
#             points,
#             preds_dict,
#             batch_input_metas,
#             use_nms=proposal_cfg.use_nms,
#             **kwargs)

#         return losses, results_list

#     def loss(self,
#              points: List[torch.Tensor],
#              feats_dict: Dict[str, torch.Tensor],
#              batch_data_samples: List[Det3DDataSample],
#              ret_target: bool = False,
#              **kwargs) -> dict:
#         """
#         Args:
#             points (list[tensor]): Points cloud of multiple samples.
#             feats_dict (dict): Predictions from backbone or FPN.
#             batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
#                 contains the meta information of each sample and
#                 corresponding annotations.
#             ret_target (bool): Whether return the assigned target.
#                 Defaults to False.

#         Returns:
#             dict:  A dictionary of loss components.
#         """
#         preds_dict = self.forward(feats_dict)
#         batch_gt_instance_3d = []
#         batch_gt_instances_ignore = []
#         batch_input_metas = []
#         batch_pts_semantic_mask = []
#         batch_pts_instance_mask = []
#         for data_sample in batch_data_samples:
#             batch_input_metas.append(data_sample.metainfo)
#             batch_gt_instance_3d.append(data_sample.gt_instances_3d)
#             batch_gt_instances_ignore.append(
#                 data_sample.get('ignored_instances', None))
#             batch_pts_semantic_mask.append(
#                 data_sample.gt_pts_seg.get('pts_semantic_mask', None))
#             batch_pts_instance_mask.append(
#                 data_sample.gt_pts_seg.get('pts_instance_mask', None))

#         loss_inputs = (points, preds_dict, batch_gt_instance_3d)
#         losses = self.loss_by_feat(
#             *loss_inputs,
#             batch_pts_semantic_mask=batch_pts_semantic_mask,
#             batch_pts_instance_mask=batch_pts_instance_mask,
#             batch_input_metas=batch_input_metas,
#             batch_gt_instances_ignore=batch_gt_instances_ignore,
#             ret_target=ret_target,
#             **kwargs)
#         return losses

#     def forward(self, feat_dict: dict) -> dict:
#         from torch_scatter import scatter_mean 
#         seed_points, seed_features, seed_indices, seed_masks = self._extract_input(
#             feat_dict)

#         # 1. 生成原始投票
#         vote_points, vote_features, vote_offset = self.vote_module(
#             seed_points, seed_features)
        
#         results = dict(
#             seed_points=seed_points,
#             seed_indices=seed_indices,
#             seed_masks=seed_masks, 
#             vote_points=vote_points,
#             vote_features=vote_features,
#             vote_offset=vote_offset)

#         # 2. 实例级聚合逻辑
#         if seed_masks is not None:
#             instance_indices = seed_masks.long() # [B, 1024]

#             # A. 处理坐标: [B, 1024, 3] -> 聚合维度 dim=1 -> 结果 [B, Max_ID+1, 3]
#             idx_pts = instance_indices.unsqueeze(-1)
#             aggregated_points_raw = scatter_mean(vote_points, idx_pts, dim=1)
#             aggregated_points = aggregated_points_raw[:, 1:, :] # 过滤 ID=0 背景

#             # B. 处理特征: [B, 256, 1024] -> 聚合维度 dim=2 -> 结果 [B, 256, Max_ID+1]
#             idx_feat = instance_indices.unsqueeze(1)
#             aggregated_features_raw = scatter_mean(vote_features, idx_feat, dim=2)
#             features = aggregated_features_raw[:, :, 1:] # [B, 256, M]
            
#             # --- 【核心修改：解决通道不匹配】 ---
#             # 如果你的 conv_pred 期待 128 通道，但这里是 256
#             if features.shape[1] != 128:
#                 features = self.channel_mapper(features)
#             # ----------------------------------

#             aggregated_indices = None
#         else:
#             # 原生 FPS 采样逻辑
#             if self.sample_mode == 'vote':
#                 aggregation_inputs = dict(points_xyz=vote_points, features=vote_features)
#             else:
#                 raise NotImplementedError(f'Sample mode {self.sample_mode} is not supported!')

#             vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
#             aggregated_points, features, aggregated_indices = vote_aggregation_ret

#         results['aggregated_points'] = aggregated_points
#         results['aggregated_features'] = features
#         results['aggregated_indices'] = aggregated_indices

#         # 3. 预测检测框属性 (BBox and Score)
#         # 此时 features 形状为 [B, 128, M]，完美匹配 conv_pred 的 conv1d 要求
#         cls_predictions, reg_predictions = self.conv_pred(features)

#         # 4. 解码预测结果
#         decode_res = self.bbox_coder.split_pred(cls_predictions,
#                                                 reg_predictions,
#                                                 aggregated_points)
#         results.update(decode_res)
#         return results

#     def loss_by_feat(self, points, bbox_preds_dict, batch_gt_instances_3d, **kwargs) -> dict:
#         """完全基于匈牙利匹配的损失函数，废弃 get_targets 依赖。"""
#         vote_points = bbox_preds_dict['vote_points']
#         seed_masks = bbox_preds_dict['seed_masks']
#         batch_size = vote_points.shape[0]

#         zero = torch.tensor(0.0, device=vote_points.device)
#         total_losses = {k: zero.clone() for k in ['vote', 'sem', 'dir_cls', 'dir_res', 'size_cls', 'size_res', 'obj']}
#         total_matched = 0

#         for i in range(batch_size):
#             gt_inst = batch_gt_instances_3d[i]
#             # 获取当前 Batch 的所有 Mask ID (排除背景 0)
#             unique_ids = torch.unique(seed_masks[i])
#             valid_mask_ids = unique_ids[unique_ids > 0].long()
#             num_proposals = len(valid_mask_ids)
            
#             if num_proposals == 0: continue
            
#             # 建立密集索引映射
#             dense_indices = torch.arange(num_proposals, device=vote_points.device)
#             # 聚合预测中心
#             pred_centers = scatter_mean(vote_points[i], seed_masks[i].long().unsqueeze(-1), dim=0)[valid_mask_ids]

#             # --- 核心逻辑：如果没有 GT，所有有效 Mask 都是负样本 ---
#             if len(gt_inst) == 0:
#                 # 所有 Mask 的 Objectness 目标都是 0 (背景)
#                 obj_targets = torch.zeros(num_proposals, dtype=torch.long, device=vote_points.device)
#                 total_losses['obj'] += F.cross_entropy(bbox_preds_dict['obj_scores'][i, dense_indices], obj_targets, reduction='sum')
#                 continue

#             # --- 匈牙利匹配 ---
#             gt_centers = gt_inst.bboxes_3d.gravity_center
#             dist_matrix = torch.cdist(pred_centers, gt_centers).detach().cpu().numpy()
#             p_idx_np, g_idx_np = linear_sum_assignment(dist_matrix)
            
#             p_idx = torch.from_numpy(p_idx_np).to(pred_centers.device)
#             g_idx = torch.from_numpy(g_idx_np).to(pred_centers.device)
#             m_dense_idx = dense_indices[p_idx]

#             # 1. Objectness 损失 (基于匹配结果)
#             # 初始认为全是背景(0)，匹配上的设为前景(1)
#             obj_targets = torch.zeros(num_proposals, dtype=torch.long, device=vote_points.device)
#             obj_targets[p_idx] = 1 
#             total_losses['obj'] += F.cross_entropy(bbox_preds_dict['obj_scores'][i, dense_indices], obj_targets, reduction='sum')

#             # 2. 属性回归与分类 (仅针对匹配上的实例)
#             gt_center, gt_size_cls, gt_size_res, gt_dir_cls, gt_dir_res = \
#                 self.bbox_coder.encode(gt_inst.bboxes_3d[g_idx], gt_inst.labels_3d[g_idx])
            
#             # 缩放对齐
#             gt_dir_res /= (np.pi / self.num_dir_bins)
#             mean_sizes = torch.tensor(self.bbox_coder.mean_sizes, device=gt_size_res.device)
#             gt_size_res /= mean_sizes[gt_size_cls.long()] 

#             # 分类
#             total_losses['sem'] += F.cross_entropy(bbox_preds_dict['sem_scores'][i, m_dense_idx], 
#                                                 torch.clamp(gt_inst.labels_3d[g_idx].long(), 0, self.num_classes-1), reduction='sum')
#             total_losses['dir_cls'] += F.cross_entropy(bbox_preds_dict['dir_class'][i, m_dense_idx], 
#                                                     gt_dir_cls.long(), reduction='sum')
            
#             # 残差回归 (Gather)
#             matched_dir_res = torch.gather(bbox_preds_dict['dir_res_norm'][i, m_dense_idx], 1, gt_dir_cls.long().unsqueeze(-1)).squeeze(-1)
#             total_losses['dir_res'] += F.smooth_l1_loss(matched_dir_res, gt_dir_res, reduction='sum')
            
#             size_idx = gt_size_cls.long().view(-1, 1, 1).expand(-1, 1, 3)
#             matched_size_res = torch.gather(bbox_preds_dict['size_res_norm'][i, m_dense_idx], 1, size_idx).squeeze(1)
#             total_losses['size_res'] += F.smooth_l1_loss(matched_size_res, gt_size_res, reduction='sum')
            
#             # 中心投票
#             total_losses['vote'] += F.smooth_l1_loss(pred_centers[p_idx], gt_centers[g_idx], reduction='sum')
#             total_matched += len(p_idx)

#         # 归一化输出
#         denom = total_matched + 1e-6
#         # 注意：Objectness 损失通常根据总 Proposal 数归一化
#         res_losses = {f"{k}_loss": v / denom for k, v in total_losses.items() if k != 'obj'}
#         res_losses['objectness_loss'] = total_losses['obj'] / (batch_size * self.num_proposal + 1e-6)
        
#         return res_losses
#     def get_targets(
#         self,
#         points,
#         bbox_preds: dict = None,
#         batch_gt_instances_3d: List[InstanceData] = None,
#         batch_pts_semantic_mask: List[torch.Tensor] = None,
#         batch_pts_instance_mask: List[torch.Tensor] = None,
#     ):
#         """Generate targets of vote head.

#         Args:
#             points (list[torch.Tensor]): Points of each batch.
#             bbox_preds (torch.Tensor): Bounding box predictions of vote head.
#             batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
#                 gt_instances. It usually includes ``bboxes`` and ``labels``
#                 attributes.
#             batch_pts_semantic_mask (list[tensor]): Semantic gt mask for
#                 point clouds. Defaults to None.
#             batch_pts_instance_mask (list[tensor]): Instance gt mask for
#                 point clouds. Defaults to None.

#         Returns:
#             tuple[torch.Tensor]: Targets of vote head.
#         """
#         # find empty example
#         valid_gt_masks = list()
#         gt_num = list()
#         batch_gt_labels_3d = [
#             gt_instances_3d.labels_3d
#             for gt_instances_3d in batch_gt_instances_3d
#         ]
#         batch_gt_bboxes_3d = [
#             gt_instances_3d.bboxes_3d
#             for gt_instances_3d in batch_gt_instances_3d
#         ]
#         for index in range(len(batch_gt_labels_3d)):
#             if len(batch_gt_labels_3d[index]) == 0:
#                 fake_box = batch_gt_bboxes_3d[index].tensor.new_zeros(
#                     1, batch_gt_bboxes_3d[index].tensor.shape[-1])
#                 batch_gt_bboxes_3d[index] = batch_gt_bboxes_3d[index].new_box(
#                     fake_box)
#                 batch_gt_labels_3d[index] = batch_gt_labels_3d[
#                     index].new_zeros(1)
#                 valid_gt_masks.append(batch_gt_labels_3d[index].new_zeros(1))
#                 gt_num.append(1)
#             else:
#                 valid_gt_masks.append(batch_gt_labels_3d[index].new_ones(
#                     batch_gt_labels_3d[index].shape))
#                 gt_num.append(batch_gt_labels_3d[index].shape[0])
#         max_gt_num = max(gt_num)

#         aggregated_points = [
#             bbox_preds['aggregated_points'][i]
#             for i in range(len(batch_gt_labels_3d))
#         ]

#         (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
#          dir_class_targets, dir_res_targets, center_targets,
#          assigned_center_targets, mask_targets,
#          objectness_targets, objectness_masks) = multi_apply(
#              self._get_targets_single, points, batch_gt_bboxes_3d,
#              batch_gt_labels_3d, batch_pts_semantic_mask,
#              batch_pts_instance_mask, aggregated_points)

#         # pad targets as original code of votenet.
#         for index in range(len(batch_gt_labels_3d)):
#             pad_num = max_gt_num - batch_gt_labels_3d[index].shape[0]
#             center_targets[index] = F.pad(center_targets[index],
#                                           (0, 0, 0, pad_num))
#             valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

#         vote_targets = torch.stack(vote_targets)
#         vote_target_masks = torch.stack(vote_target_masks)
#         center_targets = torch.stack(center_targets)
#         valid_gt_masks = torch.stack(valid_gt_masks)

#         assigned_center_targets = torch.stack(assigned_center_targets)
#         objectness_targets = torch.stack(objectness_targets)
#         objectness_weights = torch.stack(objectness_masks)
#         objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
#         box_loss_weights = objectness_targets.float() / (
#             torch.sum(objectness_targets).float() + 1e-6)
#         valid_gt_weights = valid_gt_masks.float() / (
#             torch.sum(valid_gt_masks.float()) + 1e-6)
#         dir_class_targets = torch.stack(dir_class_targets)
#         dir_res_targets = torch.stack(dir_res_targets)
#         size_class_targets = torch.stack(size_class_targets)
#         size_res_targets = torch.stack(size_res_targets)
#         mask_targets = torch.stack(mask_targets)

#         return (vote_targets, vote_target_masks, size_class_targets,
#                 size_res_targets, dir_class_targets, dir_res_targets,
#                 center_targets, assigned_center_targets, mask_targets,
#                 valid_gt_masks, objectness_targets, objectness_weights,
#                 box_loss_weights, valid_gt_weights)

#     def _get_targets_single(self,
#                             points,
#                             gt_bboxes_3d,
#                             gt_labels_3d,
#                             pts_semantic_mask=None,
#                             pts_instance_mask=None,
#                             aggregated_points=None):
#         """Generate targets of vote head for single batch.

#         Args:
#             points (torch.Tensor): Points of each batch.
#             gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
#                 boxes of each batch.
#             gt_labels_3d (torch.Tensor): Labels of each batch.
#             pts_semantic_mask (torch.Tensor): Point-wise semantic
#                 label of each batch.
#             pts_instance_mask (torch.Tensor): Point-wise instance
#                 label of each batch.
#             aggregated_points (torch.Tensor): Aggregated points from
#                 vote aggregation layer.

#         Returns:
#             tuple[torch.Tensor]: Targets of vote head.
#         """
#         assert self.bbox_coder.with_rot or pts_semantic_mask is not None

#         gt_bboxes_3d = gt_bboxes_3d.to(points.device)

#         # generate votes target
#         num_points = points.shape[0]
#         if self.bbox_coder.with_rot:
#             vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
#             vote_target_masks = points.new_zeros([num_points],
#                                                  dtype=torch.long)
#             vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
#             box_indices_all = gt_bboxes_3d.points_in_boxes_all(points)
#             for i in range(gt_labels_3d.shape[0]):
#                 box_indices = box_indices_all[:, i]
#                 indices = torch.nonzero(
#                     box_indices, as_tuple=False).squeeze(-1)
#                 selected_points = points[indices]
#                 vote_target_masks[indices] = 1
#                 vote_targets_tmp = vote_targets[indices]
#                 votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
#                     0) - selected_points[:, :3]

#                 for j in range(self.gt_per_seed):
#                     column_indices = torch.nonzero(
#                         vote_target_idx[indices] == j,
#                         as_tuple=False).squeeze(-1)
#                     vote_targets_tmp[column_indices,
#                                      int(j * 3):int(j * 3 +
#                                                     3)] = votes[column_indices]
#                     if j == 0:
#                         vote_targets_tmp[column_indices] = votes[
#                             column_indices].repeat(1, self.gt_per_seed)

#                 vote_targets[indices] = vote_targets_tmp
#                 vote_target_idx[indices] = torch.clamp(
#                     vote_target_idx[indices] + 1, max=2)
#         elif pts_semantic_mask is not None:
#             vote_targets = points.new_zeros([num_points, 3])
#             vote_target_masks = points.new_zeros([num_points],
#                                                  dtype=torch.long)
#             for i in torch.unique(pts_instance_mask):
#                 indices = torch.nonzero(
#                     pts_instance_mask == i, as_tuple=False).squeeze(-1)
#                 if pts_semantic_mask[indices[0]] < self.num_classes:
#                     selected_points = points[indices, :3]
#                     center = 0.5 * (
#                         selected_points.min(0)[0] + selected_points.max(0)[0])
#                     vote_targets[indices, :] = center - selected_points
#                     vote_target_masks[indices] = 1
#             vote_targets = vote_targets.repeat((1, self.gt_per_seed))
#         else:
#             raise NotImplementedError

#         (center_targets, size_class_targets, size_res_targets,
#          dir_class_targets,
#          dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

#         proposal_num = aggregated_points.shape[0]
#         distance1, _, assignment, _ = chamfer_distance(
#             aggregated_points.unsqueeze(0),
#             center_targets.unsqueeze(0),
#             reduction='none')
#         assignment = assignment.squeeze(0)
#         euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

#         objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
#         objectness_targets[
#             euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1

#         objectness_masks = points.new_zeros((proposal_num))
#         objectness_masks[
#             euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
#         objectness_masks[
#             euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

#         dir_class_targets = dir_class_targets[assignment]
#         dir_res_targets = dir_res_targets[assignment]
#         dir_res_targets /= (np.pi / self.num_dir_bins)
#         size_class_targets = size_class_targets[assignment]
#         size_res_targets = size_res_targets[assignment]

#         one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
#             (proposal_num, self.num_sizes))
#         one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
#         one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
#             1, 1, 3)
#         mean_sizes = size_res_targets.new_tensor(
#             self.bbox_coder.mean_sizes).unsqueeze(0)
#         pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
#         size_res_targets /= pos_mean_sizes

#         mask_targets = gt_labels_3d[assignment]
#         assigned_center_targets = center_targets[assignment]

#         return (vote_targets, vote_target_masks, size_class_targets,
#                 size_res_targets, dir_class_targets,
#                 dir_res_targets, center_targets, assigned_center_targets,
#                 mask_targets.long(), objectness_targets, objectness_masks)

#     def predict_by_feat(self, points, bbox_preds_dict, batch_input_metas, use_nms=True, **kwargs) -> List[InstanceData]:
#         """修正后的推理函数：确保 Origin 与 密集索引完全对齐"""
#         obj_scores = F.softmax(bbox_preds_dict['obj_scores'], dim=-1)[..., -1]
#         sem_scores = F.softmax(bbox_preds_dict['sem_scores'], dim=-1)
#         bbox3d = self.bbox_coder.decode(bbox_preds_dict) #

#         results_list = []
#         for i in range(bbox3d.shape[0]):
#             res = InstanceData()
#             score_thr = kwargs.get('score_thr', self.test_cfg.get('score_thr', 0.1))
#             keep = obj_scores[i] > score_thr
            
#             if not keep.any():
#                 empty_bbox = bbox3d.new_zeros((0, bbox3d.shape[-1]))
#                 res.bboxes_3d = batch_input_metas[i]['box_type_3d'](empty_bbox, box_dim=bbox3d.shape[-1], with_yaw=self.bbox_coder.with_rot)
#                 res.scores_3d = obj_scores.new_zeros(0)
#                 res.labels_3d = obj_scores.new_zeros(0, dtype=torch.long)
#                 results_list.append(res)
#                 continue

#             # 关键修改：封装 Box 时强制 origin=(0.5, 0.5, 0.5)
#             res.bboxes_3d = batch_input_metas[i]['box_type_3d'](
#                 bbox3d[i][keep], 
#                 box_dim=bbox3d.shape[-1], 
#                 with_yaw=self.bbox_coder.with_rot,
#                 origin=(0.5, 0.5, 0.5)) # 解决框偏高的问题
#             res.scores_3d = obj_scores[i][keep]
#             res.labels_3d = sem_scores[i][keep].argmax(dim=-1)
#             results_list.append(res)
#         return results_list
    
#     def multiclass_nms_single(self, obj_scores: Tensor, sem_scores: Tensor,
#                               bbox: Tensor, points: Tensor,
#                               input_meta: dict) -> Tuple:
#         """Multi-class nms in single batch.

#         Args:
#             obj_scores (torch.Tensor): Objectness score of bounding boxes.
#             sem_scores (torch.Tensor): semantic class score of bounding boxes.
#             bbox (torch.Tensor): Predicted bounding boxes.
#             points (torch.Tensor): Input points.
#             input_meta (dict): Point cloud and image's meta info.

#         Returns:
#             tuple[torch.Tensor]: Bounding boxes, scores and labels.
#         """
#         bbox = input_meta['box_type_3d'](
#             bbox,
#             box_dim=bbox.shape[-1],
#             with_yaw=self.bbox_coder.with_rot,
#             origin=(0.5, 0.5, 0.5))
#         box_indices = bbox.points_in_boxes_all(points)

#         corner3d = bbox.corners
#         minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
#         minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
#         minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

#         nonempty_box_mask = box_indices.T.sum(1) > 5

#         bbox_classes = torch.argmax(sem_scores, -1)
#         nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
#                                       obj_scores[nonempty_box_mask],
#                                       bbox_classes[nonempty_box_mask],
#                                       self.test_cfg.nms_thr)

#         # filter empty boxes and boxes with low score
#         scores_mask = (obj_scores > self.test_cfg.score_thr)
#         nonempty_box_inds = torch.nonzero(
#             nonempty_box_mask, as_tuple=False).flatten()
#         nonempty_mask = torch.zeros_like(bbox_classes).scatter(
#             0, nonempty_box_inds[nms_selected], 1)
#         selected = (nonempty_mask.bool() & scores_mask.bool())

#         if self.test_cfg.per_class_proposal:
#             bbox_selected, score_selected, labels = [], [], []
#             for k in range(sem_scores.shape[-1]):
#                 bbox_selected.append(bbox[selected].tensor)
#                 score_selected.append(obj_scores[selected] *
#                                       sem_scores[selected][:, k])
#                 labels.append(
#                     torch.zeros_like(bbox_classes[selected]).fill_(k))
#             bbox_selected = torch.cat(bbox_selected, 0)
#             score_selected = torch.cat(score_selected, 0)
#             labels = torch.cat(labels, 0)
#         else:
#             bbox_selected = bbox[selected].tensor
#             score_selected = obj_scores[selected]
#             labels = bbox_classes[selected]

#         return bbox_selected, score_selected, labels





# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.ops import furthest_point_sample
from mmdet.models.utils import multi_apply
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.layers import VoteModule, aligned_3d_nms, build_sa_module
from mmdet3d.models.losses import chamfer_distance
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample
from .base_conv_bbox_head import BaseConvBboxHead


@MODELS.register_module()
class VoteHead(BaseModule):
    r"""Bbox head of `Votenet <https://arxiv.org/abs/1904.09664>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (ConfigDict, dict): Bbox coder for encoding and
            decoding boxes. Defaults to None.
        train_cfg (dict, optional): Config for training. Defaults to None.
        test_cfg (dict, optional): Config for testing. Defaults to None.
        vote_module_cfg (dict, optional): Config of VoteModule for
            point-wise votes. Defaults to None.
        vote_aggregation_cfg (dict, optional): Config of vote
            aggregation layer. Defaults to None.
        pred_layer_cfg (dict, optional): Config of classification
            and regression prediction layers. Defaults to None.
        objectness_loss (dict, optional): Config of objectness loss.
            Defaults to None.
        center_loss (dict, optional): Config of center loss.
            Defaults to None.
        dir_class_loss (dict, optional): Config of direction
            classification loss. Defaults to None.
        dir_res_loss (dict, optional): Config of direction
            residual regression loss. Defaults to None.
        size_class_loss (dict, optional): Config of size
            classification loss. Defaults to None.
        size_res_loss (dict, optional): Config of size
            residual regression loss. Defaults to None.
        semantic_loss (dict, optional): Config of point-wise
            semantic segmentation loss. Defaults to None.
        iou_loss (dict, optional): Config of IOU loss for
            regression. Defaults to None.
        init_cfg (dict, optional): Config of model weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 bbox_coder: Union[ConfigDict, dict],
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 vote_module_cfg: Optional[dict] = None,
                 vote_aggregation_cfg: Optional[dict] = None,
                 pred_layer_cfg: Optional[dict] = None,
                 objectness_loss: Optional[dict] = None,
                 center_loss: Optional[dict] = None,
                 dir_class_loss: Optional[dict] = None,
                 dir_res_loss: Optional[dict] = None,
                 size_class_loss: Optional[dict] = None,
                 size_res_loss: Optional[dict] = None,
                 semantic_loss: Optional[dict] = None,
                 iou_loss: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super(VoteHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.gt_per_seed = vote_module_cfg['gt_per_seed']
        self.num_proposal = vote_aggregation_cfg['num_point']

        self.loss_objectness = MODELS.build(objectness_loss)
        self.loss_center = MODELS.build(center_loss)
        self.loss_dir_res = MODELS.build(dir_res_loss)
        self.loss_dir_class = MODELS.build(dir_class_loss)
        self.loss_size_res = MODELS.build(size_res_loss)
        if size_class_loss is not None:
            self.size_class_loss = MODELS.build(size_class_loss)
        if semantic_loss is not None:
            self.semantic_loss = MODELS.build(semantic_loss)
        if iou_loss is not None:
            self.iou_loss = MODELS.build(iou_loss)
        else:
            self.iou_loss = None

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_sizes = self.bbox_coder.num_sizes
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.vote_module = VoteModule(**vote_module_cfg)
        self.vote_aggregation = build_sa_module(vote_aggregation_cfg)

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **pred_layer_cfg,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

    @property
    def sample_mode(self):
        if self.training:
            sample_mode = self.train_cfg.sample_mode
        else:
            sample_mode = self.test_cfg.sample_mode
        assert sample_mode in ['vote', 'seed', 'random', 'spec']
        return sample_mode

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (2)
        return self.num_classes + 2

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_dir_bins*2),
        # size class+residual(num_sizes*4)
        return 3 + self.num_dir_bins * 2 + self.num_sizes * 4

    def _extract_input(self, feat_dict: dict) -> tuple:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            tuple[Tensor]: Arrage as following three tensor.

                - Coordinates of input points.
                - Features of input points.
                - Indices of input points.
        """

        # for imvotenet
        if 'seed_points' in feat_dict and \
           'seed_features' in feat_dict and \
           'seed_indices' in feat_dict:
            seed_points = feat_dict['seed_points']
            seed_features = feat_dict['seed_features']
            seed_indices = feat_dict['seed_indices']
        # for votenet
        else:
            seed_points = feat_dict['fp_xyz'][-1]
            seed_features = feat_dict['fp_features'][-1]
            seed_indices = feat_dict['fp_indices'][-1]

        return seed_points, seed_features, seed_indices

    def predict(self,
                points: List[torch.Tensor],
                feats_dict: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                use_nms: bool = True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            points (list[tensor]): Point clouds of multiple samples.
            feats_dict (dict): Features from FPN or backbone..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            use_nms (bool): Whether do the nms for predictions.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        preds_dict = self(feats_dict)
        # `preds_dict` can be used in H3DNET
        feats_dict.update(preds_dict)

        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        results_list = self.predict_by_feat(
            points, preds_dict, batch_input_metas, use_nms=use_nms, **kwargs)
        return results_list

    def loss_and_predict(self,
                         points: List[torch.Tensor],
                         feats_dict: Dict[str, torch.Tensor],
                         batch_data_samples: List[Det3DDataSample],
                         ret_target: bool = False,
                         proposal_cfg: dict = None,
                         **kwargs) -> Tuple:
        """
        Args:
            points (list[tensor]): Points cloud of multiple samples.
            feats_dict (dict): Predictions from backbone or FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each sample and
                corresponding annotations.
            ret_target (bool): Whether return the assigned target.
                Defaults to False.
            proposal_cfg (dict): Configure for proposal process.
                Defaults to True.

        Returns:
            tuple:  Contains loss and predictions after post-process.
        """
        preds_dict = self.forward(feats_dict)
        feats_dict.update(preds_dict)
        batch_gt_instance_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        batch_pts_semantic_mask = []
        batch_pts_instance_mask = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))
            batch_pts_semantic_mask.append(
                data_sample.gt_pts_seg.get('pts_semantic_mask', None))
            batch_pts_instance_mask.append(
                data_sample.gt_pts_seg.get('pts_instance_mask', None))

        loss_inputs = (points, preds_dict, batch_gt_instance_3d)
        losses = self.loss_by_feat(
            *loss_inputs,
            batch_pts_semantic_mask=batch_pts_semantic_mask,
            batch_pts_instance_mask=batch_pts_instance_mask,
            batch_input_metas=batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            ret_target=ret_target,
            **kwargs)

        results_list = self.predict_by_feat(
            points,
            preds_dict,
            batch_input_metas,
            use_nms=proposal_cfg.use_nms,
            **kwargs)

        return losses, results_list

    def loss(self,
             points: List[torch.Tensor],
             feats_dict: Dict[str, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             ret_target: bool = False,
             **kwargs) -> dict:
        """
        Args:
            points (list[tensor]): Points cloud of multiple samples.
            feats_dict (dict): Predictions from backbone or FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each sample and
                corresponding annotations.
            ret_target (bool): Whether return the assigned target.
                Defaults to False.

        Returns:
            dict:  A dictionary of loss components.
        """
        preds_dict = self.forward(feats_dict)
        batch_gt_instance_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        batch_pts_semantic_mask = []
        batch_pts_instance_mask = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))
            batch_pts_semantic_mask.append(
                data_sample.gt_pts_seg.get('pts_semantic_mask', None))
            batch_pts_instance_mask.append(
                data_sample.gt_pts_seg.get('pts_instance_mask', None))

        loss_inputs = (points, preds_dict, batch_gt_instance_3d)
        losses = self.loss_by_feat(
            *loss_inputs,
            batch_pts_semantic_mask=batch_pts_semantic_mask,
            batch_pts_instance_mask=batch_pts_instance_mask,
            batch_input_metas=batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            ret_target=ret_target,
            **kwargs)
        return losses

    def forward(self, feat_dict: dict) -> dict:
        """Forward pass.

        Note:
            The forward of VoteHead is divided into 4 steps:

                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            dict: Predictions of vote head.
        """

        seed_points, seed_features, seed_indices = self._extract_input(
            feat_dict)

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if self.sample_mode == 'vote':
            # use fps in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=vote_points, features=vote_features)
        elif self.sample_mode == 'seed':
            # FPS on seed and choose the votes corresponding to the seeds
            sample_indices = furthest_point_sample(seed_points,
                                                   self.num_proposal)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif self.sample_mode == 'random':
            # Random sampling from the votes
            batch_size, num_seed = seed_points.shape[:2]
            sample_indices = seed_points.new_tensor(
                torch.randint(0, num_seed, (batch_size, self.num_proposal)),
                dtype=torch.int32)
            aggregation_inputs = dict(
                points_xyz=vote_points,
                features=vote_features,
                indices=sample_indices)
        elif self.sample_mode == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                points_xyz=seed_points,
                features=seed_features,
                target_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {self.sample_mode} is not supported!')

        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret

        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(features)

        # 4. decode predictions
        decode_res = self.bbox_coder.split_pred(cls_predictions,
                                                reg_predictions,
                                                aggregated_points)
        results.update(decode_res)
        return results

    def loss_by_feat(
            self,
            points: List[torch.Tensor],
            bbox_preds_dict: dict,
            batch_gt_instances_3d: List[InstanceData],
            batch_pts_semantic_mask: Optional[List[torch.Tensor]] = None,
            batch_pts_instance_mask: Optional[List[torch.Tensor]] = None,
            ret_target: bool = False,
            **kwargs) -> dict:
        """Compute loss.

        Args:
            points (list[torch.Tensor]): Input points.
            bbox_preds_dict (dict): Predictions from forward of vote head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_pts_semantic_mask (list[tensor]): Semantic mask
                of points cloud. Defaults to None.
            batch_pts_semantic_mask (list[tensor]): Instance mask
                of points cloud. Defaults to None.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            ret_target (bool): Return targets or not. Defaults to False.

        Returns:
            dict: Losses of Votenet.
        """

        targets = self.get_targets(points, bbox_preds_dict,
                                   batch_gt_instances_3d,
                                   batch_pts_semantic_mask,
                                   batch_pts_instance_mask)
        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets,
         assigned_center_targets, mask_targets, valid_gt_masks,
         objectness_targets, objectness_weights, box_loss_weights,
         valid_gt_weights) = targets

        # calculate vote loss
        vote_loss = self.vote_module.get_loss(bbox_preds_dict['seed_points'],
                                              bbox_preds_dict['vote_points'],
                                              bbox_preds_dict['seed_indices'],
                                              vote_target_masks, vote_targets)

        # calculate objectness loss
        objectness_loss = self.loss_objectness(
            bbox_preds_dict['obj_scores'].transpose(2, 1),
            objectness_targets,
            weight=objectness_weights)

        # calculate center loss
        source2target_loss, target2source_loss = self.loss_center(
            bbox_preds_dict['center'],
            center_targets,
            src_weight=box_loss_weights,
            dst_weight=valid_gt_weights)
        center_loss = source2target_loss + target2source_loss

        # calculate direction class loss
        dir_class_loss = self.loss_dir_class(
            bbox_preds_dict['dir_class'].transpose(2, 1),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        batch_size, proposal_num = size_class_targets.shape[:2]
        heading_label_one_hot = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        dir_res_norm = torch.sum(
            bbox_preds_dict['dir_res_norm'] * heading_label_one_hot, -1)
        dir_res_loss = self.loss_dir_res(
            dir_res_norm, dir_res_targets, weight=box_loss_weights)

        # calculate size class loss
        size_class_loss = self.size_class_loss(
            bbox_preds_dict['size_class'].transpose(2, 1),
            size_class_targets,
            weight=box_loss_weights)

        # calculate size residual loss
        one_hot_size_targets = vote_targets.new_zeros(
            (batch_size, proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(2, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets_expand = one_hot_size_targets.unsqueeze(
            -1).repeat(1, 1, 1, 3).contiguous()
        size_residual_norm = torch.sum(
            bbox_preds_dict['size_res_norm'] * one_hot_size_targets_expand, 2)
        box_loss_weights_expand = box_loss_weights.unsqueeze(-1).repeat(
            1, 1, 3)
        size_res_loss = self.loss_size_res(
            size_residual_norm,
            size_res_targets,
            weight=box_loss_weights_expand)

        # calculate semantic loss
        semantic_loss = self.semantic_loss(
            bbox_preds_dict['sem_scores'].transpose(2, 1),
            mask_targets,
            weight=box_loss_weights)

        losses = dict(
            vote_loss=vote_loss,
            objectness_loss=objectness_loss,
            semantic_loss=semantic_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=size_class_loss,
            size_res_loss=size_res_loss)

        if self.iou_loss:
            # 1. 准备基础数据
            # mean_sizes: [NumClasses, 3] -> [B, N, 3]
            mean_sizes = bbox_preds_dict['center'].new_tensor(self.bbox_coder.mean_sizes)
            mean_sizes_targets = mean_sizes[size_class_targets]

            # 2. 恢复预测框尺寸 (Predicted Size)
            # size = mean_size * (1 + predicted_residual)
            pred_size = mean_sizes_targets * (1 + size_residual_norm)
            
            # 3. 恢复真实框尺寸 (Target Size)
            # size = mean_size * (1 + target_residual)
            target_size = mean_sizes_targets * (1 + size_res_targets)

            # 4. 计算 AABB (Axis Aligned Bounding Box)
            # [B, N, 3]
            pred_min = bbox_preds_dict['center'] - pred_size * 0.5
            pred_max = bbox_preds_dict['center'] + pred_size * 0.5
            
            target_min = assigned_center_targets - target_size * 0.5
            target_max = assigned_center_targets + target_size * 0.5

            # 5. 拼接坐标 [x1, y1, z1, x2, y2, z2] -> [B, N, 6]
            pred_aabb = torch.cat([pred_min, pred_max], dim=-1)
            target_aabb = torch.cat([target_min, target_max], dim=-1)

            # 6. 展平并筛选正样本 (Filter Positive Samples)
            pred_aabb_flat = pred_aabb.reshape(-1, 6)
            target_aabb_flat = target_aabb.reshape(-1, 6)
            weights_flat = box_loss_weights.reshape(-1)

            # 只保留有 GT 匹配的 Proposal (mask > 0)
            pos_mask = weights_flat > 0

            if pos_mask.any():
                pos_pred = pred_aabb_flat[pos_mask]   # [M, 6]
                pos_target = target_aabb_flat[pos_mask] # [M, 6]

                # --- 3D GIoU 计算核心逻辑 (安全修复版) ---
                eps = 1e-6 # 安全常量，防止所有除 0 报错

                # [核心修复 1]：计算单个框体积时，强制限制长宽高 >= 0
                # 原来的 prod() 会导致长宽高为负数时“负负得正”算出假体积
                pred_whd = torch.clamp(pos_pred[:, 3:] - pos_pred[:, :3], min=0.0)
                target_whd = torch.clamp(pos_target[:, 3:] - pos_target[:, :3], min=0.0)
                vol_pred = pred_whd[:, 0] * pred_whd[:, 1] * pred_whd[:, 2]
                vol_target = target_whd[:, 0] * target_whd[:, 1] * target_whd[:, 2]
                
                # A. 计算交集 (Intersection)
                inter_min = torch.max(pos_pred[:, :3], pos_target[:, :3])
                inter_max = torch.min(pos_pred[:, 3:], pos_target[:, 3:])
                inter_whd = torch.clamp(inter_max - inter_min, min=0.0)
                intersection = inter_whd[:, 0] * inter_whd[:, 1] * inter_whd[:, 2]

                # B. 计算并集 (Union)
                union = vol_pred + vol_target - intersection

                # C. 计算 IoU
                iou = intersection / (union + eps)

                # D. 计算最小外接框 (Smallest Enclosing Box)
                enclose_min = torch.min(pos_pred[:, :3], pos_target[:, :3])
                enclose_max = torch.max(pos_pred[:, 3:], pos_target[:, 3:])
                enclose_whd = torch.clamp(enclose_max - enclose_min, min=0.0)
                enclose_vol = enclose_whd[:, 0] * enclose_whd[:, 1] * enclose_whd[:, 2]

                # [核心修复 2]：外接框体积理论上必然 >= 并集体积
                # 加 max 防止浮点误差或异常框导致分子出现负数
                enclose_vol = torch.max(enclose_vol, union + eps)

                # E. 计算 GIoU
                giou = iou - (enclose_vol - union) / enclose_vol
                
                # [核心修复 3]：彻底封死负 Loss 漏洞！
                # 强制将 1.0 - giou 限制在 [0.0, 2.0] 的理论区间内
                loss_giou = torch.clamp(1.0 - giou, min=0.0, max=4.0)

                # 7. 自动归一化 (Mean Reduction)
                num_pos = pos_mask.sum().float().clamp(min=1.0)
                
                giou_weight = float(getattr(self.iou_loss, 'loss_weight', 1.0))
                losses['iou_loss'] = (loss_giou.sum() / num_pos) * giou_weight
            else:
                # 如果没有正样本，Loss 为 0 但保留梯度图
                losses['iou_loss'] = bbox_preds_dict['center'].sum() * 0.0

        if ret_target:
            losses['targets'] = targets

        return losses

    def get_targets(
        self,
        points,
        bbox_preds: dict = None,
        batch_gt_instances_3d: List[InstanceData] = None,
        batch_pts_semantic_mask: List[torch.Tensor] = None,
        batch_pts_instance_mask: List[torch.Tensor] = None,
    ):
        """Generate targets of vote head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of vote head.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_pts_semantic_mask (list[tensor]): Semantic gt mask for
                point clouds. Defaults to None.
            batch_pts_instance_mask (list[tensor]): Instance gt mask for
                point clouds. Defaults to None.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        batch_gt_labels_3d = [
            gt_instances_3d.labels_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        batch_gt_bboxes_3d = [
            gt_instances_3d.bboxes_3d
            for gt_instances_3d in batch_gt_instances_3d
        ]
        for index in range(len(batch_gt_labels_3d)):
            if len(batch_gt_labels_3d[index]) == 0:
                fake_box = batch_gt_bboxes_3d[index].tensor.new_zeros(
                    1, batch_gt_bboxes_3d[index].tensor.shape[-1])
                batch_gt_bboxes_3d[index] = batch_gt_bboxes_3d[index].new_box(
                    fake_box)
                batch_gt_labels_3d[index] = batch_gt_labels_3d[
                    index].new_zeros(1)
                valid_gt_masks.append(batch_gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(batch_gt_labels_3d[index].new_ones(
                    batch_gt_labels_3d[index].shape))
                gt_num.append(batch_gt_labels_3d[index].shape[0])
        max_gt_num = max(gt_num)

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(batch_gt_labels_3d))
        ]

        (vote_targets, vote_target_masks, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, center_targets,
         assigned_center_targets, mask_targets,
         objectness_targets, objectness_masks) = multi_apply(
             self._get_targets_single, points, batch_gt_bboxes_3d,
             batch_gt_labels_3d, batch_pts_semantic_mask,
             batch_pts_instance_mask, aggregated_points)

        # pad targets as original code of votenet.
        for index in range(len(batch_gt_labels_3d)):
            pad_num = max_gt_num - batch_gt_labels_3d[index].shape[0]
            center_targets[index] = F.pad(center_targets[index],
                                          (0, 0, 0, pad_num))
            valid_gt_masks[index] = F.pad(valid_gt_masks[index], (0, pad_num))

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)
        center_targets = torch.stack(center_targets)
        valid_gt_masks = torch.stack(valid_gt_masks)

        assigned_center_targets = torch.stack(assigned_center_targets)
        objectness_targets = torch.stack(objectness_targets)
        objectness_weights = torch.stack(objectness_masks)
        objectness_weights /= (torch.sum(objectness_weights) + 1e-6)
        box_loss_weights = objectness_targets.float() / (
            torch.sum(objectness_targets).float() + 1e-6)
        valid_gt_weights = valid_gt_masks.float() / (
            torch.sum(valid_gt_masks.float()) + 1e-6)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_class_targets = torch.stack(size_class_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets, dir_class_targets, dir_res_targets,
                center_targets, assigned_center_targets, mask_targets,
                valid_gt_masks, objectness_targets, objectness_weights,
                box_loss_weights, valid_gt_weights)

    def _get_targets_single(self,
                            points,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            pts_semantic_mask=None,
                            pts_instance_mask=None,
                            aggregated_points=None):
        """Generate targets of vote head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                vote aggregation layer.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if self.bbox_coder.with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes_all(points)
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                        selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        (center_targets, size_class_targets, size_res_targets,
         dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        proposal_num = aggregated_points.shape[0]
        distance1, _, assignment, _ = chamfer_distance(
            aggregated_points.unsqueeze(0),
            center_targets.unsqueeze(0),
            reduction='none')
        assignment = assignment.squeeze(0)
        euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)

        objectness_targets = points.new_zeros((proposal_num), dtype=torch.long)
        objectness_targets[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1

        objectness_masks = points.new_zeros((proposal_num))
        objectness_masks[
            euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1.0
        objectness_masks[
            euclidean_distance1 > self.train_cfg['neg_distance_thr']] = 1.0

        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        dir_res_targets /= (np.pi / self.num_dir_bins)
        size_class_targets = size_class_targets[assignment]
        size_res_targets = size_res_targets[assignment]

        one_hot_size_targets = gt_bboxes_3d.tensor.new_zeros(
            (proposal_num, self.num_sizes))
        one_hot_size_targets.scatter_(1, size_class_targets.unsqueeze(-1), 1)
        one_hot_size_targets = one_hot_size_targets.unsqueeze(-1).repeat(
            1, 1, 3)
        mean_sizes = size_res_targets.new_tensor(
            self.bbox_coder.mean_sizes).unsqueeze(0)
        pos_mean_sizes = torch.sum(one_hot_size_targets * mean_sizes, 1)
        size_res_targets /= pos_mean_sizes

        mask_targets = gt_labels_3d[assignment]
        assigned_center_targets = center_targets[assignment]

        return (vote_targets, vote_target_masks, size_class_targets,
                size_res_targets, dir_class_targets,
                dir_res_targets, center_targets, assigned_center_targets,
                mask_targets.long(), objectness_targets, objectness_masks)

    def predict_by_feat(self,
                        points: List[torch.Tensor],
                        bbox_preds_dict: dict,
                        batch_input_metas: List[dict],
                        use_nms: bool = True,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from vote head predictions.

        Args:
            points (List[torch.Tensor]): Input points of multiple samples.
            bbox_preds_dict (dict): Predictions from vote head.
            batch_input_metas (list[dict]): Each item
                contains the meta information of each sample.
            use_nms (bool): Whether to apply NMS, skip nms postprocessing
                while using vote head in rpn stage.

        Returns:
            list[:obj:`InstanceData`] or Tensor: Return list of processed
            predictions when `use_nms` is True. Each InstanceData cantains
            3d Bounding boxes and corresponding scores and labels.
            Return raw bboxes when `use_nms` is False.
        """
        # decode boxes
        stack_points = torch.stack(points)
        obj_scores = F.softmax(bbox_preds_dict['obj_scores'], dim=-1)[..., -1]
        sem_scores = F.softmax(bbox_preds_dict['sem_scores'], dim=-1)
        bbox3d = self.bbox_coder.decode(bbox_preds_dict)

        batch_size = bbox3d.shape[0]
        results_list = list()
        if use_nms:
            for batch_index in range(batch_size):
                temp_results = InstanceData()
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(
                        obj_scores[batch_index],
                        sem_scores[batch_index],
                        bbox3d[batch_index],
                        stack_points[batch_index, ..., :3],
                        batch_input_metas[batch_index])
                bbox = batch_input_metas[batch_index]['box_type_3d'](
                    bbox_selected,
                    box_dim=bbox_selected.shape[-1],
                    with_yaw=self.bbox_coder.with_rot)
                temp_results.bboxes_3d = bbox
                temp_results.scores_3d = score_selected
                temp_results.labels_3d = labels
                # print(labels, score_selected)
                results_list.append(temp_results)
            # print(batch_input_metas[0]['sample_idx'])
            return results_list
        else:
            # TODO unify it when refactor the Augtest
            return bbox3d

    def multiclass_nms_single(self, obj_scores: Tensor, sem_scores: Tensor,
                              bbox: Tensor, points: Tensor,
                              input_meta: dict) -> Tuple:
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        bbox = input_meta['box_type_3d'](
            bbox,
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        box_indices = bbox.points_in_boxes_all(points)

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        nonempty_box_mask = box_indices.T.sum(1) > 5

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = aligned_3d_nms(minmax_box3d[nonempty_box_mask],
                                      obj_scores[nonempty_box_mask],
                                      bbox_classes[nonempty_box_mask],
                                      self.test_cfg.nms_thr)

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores > self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(
            nonempty_box_mask, as_tuple=False).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(obj_scores[selected] *
                                      sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels
