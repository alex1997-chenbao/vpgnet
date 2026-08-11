# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import copy
import glob
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet.evaluation import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from scipy.optimize import linear_sum_assignment

from mmdet3d.evaluation import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type


@METRICS.register_module()
class IndoorMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 instance_match_filter: bool = False,
                 instance_pointclouds_dir: str = '',
                 match_ratio_thr: float = 0.0,
                 match_score_mode: str = 'binary',
                 match_ratio_power: float = 1.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.instance_match_filter = instance_match_filter
        self.instance_pointclouds_dir = instance_pointclouds_dir
        self.match_ratio_thr = match_ratio_thr
        assert match_score_mode in ('binary', 'ratio', 'ratio_mul_det')
        self.match_score_mode = match_score_mode
        self.match_ratio_power = match_ratio_power
        self._instance_cache = {}

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            img_path = data_sample.get('img_path', None)
            if img_path is None:
                img_path = data_sample.get('metainfo', {}).get('img_path', None)
            sample_idx = data_sample.get('sample_idx', None)
            if sample_idx is None:
                sample_idx = data_sample.get('metainfo', {}).get(
                    'sample_idx', None)

            if self.instance_match_filter:
                scene_id = self._extract_scene_id(img_path, sample_idx)
                if scene_id is not None and self.instance_pointclouds_dir:
                    instance_points = self._load_instance_points(scene_id)
                    pred_3d = self._filter_pred_by_instance_points(
                        pred_3d, instance_points)

            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def _extract_scene_id(self, img_path: Optional[str],
                          sample_idx: Optional[int]) -> Optional[str]:
        if img_path is not None:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            if stem.isdigit():
                return stem.zfill(6)
        if sample_idx is not None:
            return f'{int(sample_idx) + 1:06d}'
        return None

    def _load_instance_points(self, scene_id: str) -> List[np.ndarray]:
        if scene_id in self._instance_cache:
            return self._instance_cache[scene_id]

        pattern = os.path.join(self.instance_pointclouds_dir,
                               f'{scene_id}_inst_*.bin')
        files = sorted(glob.glob(pattern))
        points_list = []
        for file_path in files:
            arr = np.fromfile(file_path, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.size % 6 == 0:
                arr = arr.reshape(-1, 6)[:, :3]
            elif arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            else:
                continue
            points_list.append(arr.astype(np.float32, copy=False))
        self._instance_cache[scene_id] = points_list
        return points_list

    def _filter_pred_by_instance_points(self, pred: dict,
                                        instance_points: List[np.ndarray]) -> dict:
        filtered = copy.deepcopy(pred)
        bboxes_3d = filtered['bboxes_3d']
        scores_3d = filtered['scores_3d']
        orig_scores = scores_3d.float()

        num_det = len(scores_3d)
        if num_det == 0:
            return filtered

        # default all unmatched -> score 0
        new_scores = scores_3d.new_zeros(scores_3d.shape)

        num_inst = len(instance_points)
        if num_inst == 0:
            filtered['scores_3d'] = new_scores
            return filtered

        if hasattr(bboxes_3d, 'volume'):
            volumes = bboxes_3d.volume
        else:
            volumes = bboxes_3d.tensor[:, 3:6].prod(dim=-1)
        volumes = torch.clamp(volumes.float(), min=1e-6).cpu().numpy()

        ratio_mat = np.zeros((num_inst, num_det), dtype=np.float32)
        for inst_idx, points_np in enumerate(instance_points):
            if bboxes_3d.tensor.device.type == 'cpu':
                counts = self._count_points_in_boxes_cpu(points_np, bboxes_3d)
            else:
                points_t = torch.from_numpy(points_np).to(scores_3d.device)
                in_box_mask = bboxes_3d.points_in_boxes_all(points_t)
                counts = in_box_mask.T.sum(dim=1).float().cpu().numpy()
            ratio_mat[inst_idx] = counts / volumes

        row_ind, col_ind = linear_sum_assignment(-ratio_mat)
        for r, c in zip(row_ind, col_ind):
            if ratio_mat[r, c] >= self.match_ratio_thr:
                ratio_val = float(max(ratio_mat[r, c], 0.0))
                ratio_score = ratio_val**self.match_ratio_power
                if self.match_score_mode == 'binary':
                    score_val = 1.0
                elif self.match_score_mode == 'ratio':
                    score_val = ratio_score
                else:  # ratio_mul_det
                    score_val = ratio_score * float(orig_scores[c].item())
                new_scores[c] = score_val

        filtered['scores_3d'] = new_scores
        return filtered

    @staticmethod
    def _count_points_in_boxes_cpu(points_xyz: np.ndarray,
                                   boxes_3d) -> np.ndarray:
        boxes = boxes_3d.tensor.cpu().numpy()
        num_boxes = boxes.shape[0]
        counts = np.zeros((num_boxes, ), dtype=np.float32)
        if num_boxes == 0 or points_xyz.size == 0:
            return counts

        centers = boxes[:, :3]
        dims = np.maximum(boxes[:, 3:6], 1e-6)
        if boxes.shape[1] > 6:
            yaws = boxes[:, 6]
        else:
            yaws = np.zeros((num_boxes, ), dtype=np.float32)

        px = points_xyz[:, 0]
        py = points_xyz[:, 1]
        pz = points_xyz[:, 2]
        for i in range(num_boxes):
            rel_x = px - centers[i, 0]
            rel_y = py - centers[i, 1]
            rel_z = pz - centers[i, 2]
            c = np.cos(-yaws[i])
            s = np.sin(-yaws[i])
            local_x = rel_x * c - rel_y * s
            local_y = rel_x * s + rel_y * c
            inside = (np.abs(local_x) <= dims[i, 0] * 0.5) & \
                     (np.abs(local_y) <= dims[i, 1] * 0.5) & \
                     (np.abs(rel_z) <= dims[i, 2] * 0.5)
            counts[i] = float(np.sum(inside))
        return counts

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = []
        pred_results = []

        for eval_ann, single_pred_results in results:
            ann_infos.append(eval_ann)
            pred_results.append(single_pred_results)

        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = indoor_eval(
            ann_infos,
            pred_results,
            self.iou_thr,
            self.dataset_meta['classes'],
            logger=logger,
            box_mode_3d=box_mode_3d)

        return ret_dict


@METRICS.register_module()
class Indoor2DMetric(BaseMetric):
    """indoor 2d predictions evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: Union[float, List[float]] = [0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(Indoor2DMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            eval_ann_info = data_sample['eval_ann_info']
            ann = dict(
                labels=eval_ann_info['gt_bboxes_labels'],
                bboxes=eval_ann_info['gt_bboxes'])

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        annotations, preds = zip(*results)
        eval_results = OrderedDict()
        for iou_thr_2d_single in self.iou_thr:
            mean_ap, _ = eval_map(
                preds,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr_2d_single,
                dataset=self.dataset_meta['classes'],
                logger=logger)
            eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
        return eval_results
