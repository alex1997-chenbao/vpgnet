#!/usr/bin/env python3
"""Evaluate luggage AP on quality-based instance subsets.

Non-target luggage instances in the same scene are treated as ignored GT, so a
correct detection on another subset is not counted as a false positive.
"""

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from mmdet3d.structures import DepthInstance3DBoxes
except ImportError:
    from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes


SUBSETS = [
    ('normal', {'normal'}),
    ('occluded', {'occluded'}),
    ('reflective', {'reflective'}),
    ('occluded_reflective', {'occluded', 'reflective'}),
]
IOU_THR_LIST = (0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate AP on luggage normal/occluded/reflective subsets.')
    parser.add_argument('--gt-ann-file', required=True)
    parser.add_argument('--quality-csv', required=True)
    parser.add_argument(
        '--pred',
        action='append',
        default=[],
        metavar='NAME=JSONL',
        help='Prediction JSONL. Can be repeated.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=None,
        help='Optional extra score threshold applied after loading JSONL.')
    parser.add_argument(
        '--topk',
        type=int,
        default=0,
        help='Optional top-k predictions per sample after score filtering.')
    parser.add_argument(
        '--pred-z-plus-half-height',
        action='store_true',
        help='Add dz/2 to prediction z after loading. This is useful only for '
        'checking a center-z interpretation of JSONL boxes; mmdet3d depth '
        'evaluation normally expects bottom-center boxes.')
    return parser.parse_args()


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        return default
    try:
        return int(value)
    except Exception:
        return default


def normalize_box(box: Sequence[Any]) -> Optional[List[float]]:
    values = [float(v) for v in box]
    if len(values) < 6:
        return None
    if len(values) == 6:
        values.append(0.0)
    return values[:7]


def load_quality_map(path: str) -> Dict[Tuple[int, int], str]:
    quality = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') != 'val':
                continue
            sample_idx = to_int(row.get('sample_idx'))
            inst_idx = to_int(row.get('inst_idx'))
            rule_class = (row.get('rule_class') or 'unknown').strip()
            if sample_idx is None or inst_idx is None:
                continue
            quality[(sample_idx, inst_idx)] = rule_class
    return quality


def load_gt(path: str, quality: Dict[Tuple[int, int], str]) -> Dict[int, Dict[str, Any]]:
    data = load_pickle(path)
    if not isinstance(data, dict) or 'data_list' not in data:
        raise TypeError('This evaluator expects a mmdet3d v1 info pkl with data_list.')

    gt_by_sample = {}
    missing_quality = 0
    for dataset_index, item in enumerate(data['data_list']):
        sample_idx = to_int(item.get('sample_idx'), dataset_index)
        boxes = []
        labels = []
        rules = []
        for inst_idx, inst in enumerate(item.get('instances', [])):
            box = normalize_box(inst.get('bbox_3d', []))
            if box is None:
                continue
            label = to_int(inst.get('bbox_label_3d', inst.get('bbox_label', 0)), 0)
            rule = quality.get((sample_idx, inst_idx))
            if rule is None:
                missing_quality += 1
                rule = 'unknown'
            boxes.append(box)
            labels.append(label)
            rules.append(rule)
        gt_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
        if gt_boxes.shape[0] > 0:
            # SUN RGB-D info files store boxes at center origin. The official
            # dataset parser converts them to depth bottom-center boxes before
            # evaluation, so mirror that here for subset AP.
            gt_boxes = DepthInstance3DBoxes(
                gt_boxes,
                box_dim=7,
                with_yaw=True,
                origin=(0.5, 0.5, 0.5)).tensor.detach().cpu().numpy()
        gt_by_sample[sample_idx] = {
            'boxes': gt_boxes,
            'labels': np.asarray(labels, dtype=np.int64),
            'rules': rules,
        }

    if missing_quality:
        print(f'warning: {missing_quality} val GT boxes have no quality label')
    return gt_by_sample


def sample_idx_from_path(path_value: Any) -> Optional[int]:
    if not path_value:
        return None
    stem = Path(str(path_value)).stem
    if stem.isdigit():
        return int(stem)
    return None


def row_sample_idx(row: Dict[str, Any]) -> Optional[int]:
    for key in ('sample_idx', 'sample_index', 'lidar_idx'):
        value = to_int(row.get(key))
        if value is not None:
            return value
    for key in ('lidar_path', 'pts_path', 'point_path'):
        value = sample_idx_from_path(row.get(key))
        if value is not None:
            return value
    value = to_int(row.get('dataset_index'))
    return value


def boxes_from_row(row: Dict[str, Any]) -> Tuple[List[List[float]], List[float], List[int]]:
    if 'boxes_3d' in row:
        boxes = [normalize_box(box) for box in row.get('boxes_3d', [])]
        scores = [float(v) for v in row.get('scores_3d', [])]
        labels = [int(v) for v in row.get('labels_3d', [])]
        out_boxes = [box for box in boxes if box is not None]
        if len(out_boxes) != len(scores):
            keep = [i for i, box in enumerate(boxes) if box is not None]
            scores = [scores[i] for i in keep]
            labels = [labels[i] for i in keep]
        return out_boxes, scores, labels

    if 'boxes' in row:
        out_boxes = []
        scores = []
        labels = []
        for item in row.get('boxes', []):
            if isinstance(item, dict):
                if 'bbox_3d' in item:
                    box = normalize_box(item['bbox_3d'])
                else:
                    center = item.get('center', [])
                    size = item.get('size', [])
                    yaw = item.get('yaw', 0.0)
                    box = normalize_box(list(center) + list(size) + [yaw])
                if box is None:
                    continue
                out_boxes.append(box)
                scores.append(float(item.get('score', item.get('scores_3d', 0.0))))
                labels.append(int(item.get('label', item.get('labels_3d', 0))))
        return out_boxes, scores, labels

    return [], [], []


def load_predictions(path: str, score_thr: Optional[float], topk: int,
                     pred_z_plus_half_height: bool = False
                     ) -> Dict[int, List[Dict[str, Any]]]:
    pred_by_sample = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_idx = row_sample_idx(row)
            if sample_idx is None:
                raise ValueError(f'Cannot infer sample_idx at {path}:{line_idx + 1}')
            boxes, scores, labels = boxes_from_row(row)
            kept = []
            for box, score, label in zip(boxes, scores, labels):
                if int(label) != 0:
                    continue
                if score_thr is not None and float(score) < score_thr:
                    continue
                if pred_z_plus_half_height:
                    box = list(box)
                    box[2] += box[5] * 0.5
                kept.append({
                    'sample_idx': int(sample_idx),
                    'box': box,
                    'score': float(score),
                    'label': int(label),
                })
            if topk > 0 and len(kept) > topk:
                kept.sort(key=lambda x: x['score'], reverse=True)
                kept = kept[:topk]
            pred_by_sample[int(sample_idx)].extend(kept)
    return dict(pred_by_sample)


def build_iou_matrices(gt_by_sample: Dict[int, Dict[str, Any]],
                       pred_by_sample: Dict[int, List[Dict[str, Any]]]
                       ) -> Dict[int, np.ndarray]:
    matrices = {}
    for sample_idx, preds in pred_by_sample.items():
        gt = gt_by_sample.get(sample_idx)
        if gt is None or len(preds) == 0 or gt['boxes'].shape[0] == 0:
            matrices[sample_idx] = np.zeros((len(preds), 0), dtype=np.float32)
            continue
        pred_tensor = torch.as_tensor(
            [pred['box'] for pred in preds], dtype=torch.float32)
        gt_tensor = torch.as_tensor(gt['boxes'], dtype=torch.float32)
        pred_boxes = DepthInstance3DBoxes(pred_tensor, box_dim=7, with_yaw=True)
        gt_boxes = DepthInstance3DBoxes(gt_tensor, box_dim=7, with_yaw=True)
        with torch.no_grad():
            ious = DepthInstance3DBoxes.overlaps(
                pred_boxes, gt_boxes).detach().cpu().numpy()
        matrices[sample_idx] = ious.astype(np.float32, copy=False)
    return matrices


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def count_positives(gt_by_sample: Dict[int, Dict[str, Any]],
                    target_rules: Iterable[str]) -> int:
    target_rules = set(target_rules)
    total = 0
    for gt in gt_by_sample.values():
        for label, rule in zip(gt['labels'], gt['rules']):
            if int(label) == 0 and rule in target_rules:
                total += 1
    return total


def evaluate_one_thr(gt_by_sample: Dict[int, Dict[str, Any]],
                     pred_by_sample: Dict[int, List[Dict[str, Any]]],
                     iou_matrices: Dict[int, np.ndarray],
                     detections: List[Tuple[int, int, float]],
                     target_rules: Iterable[str],
                     iou_thr: float) -> Dict[str, Any]:
    target_rules = set(target_rules)
    npos = count_positives(gt_by_sample, target_rules)
    matched = set()
    tp = []
    fp = []
    ignored_dets = 0

    for sample_idx, pred_idx, _score in detections:
        gt = gt_by_sample.get(sample_idx)
        if gt is None or gt['boxes'].shape[0] == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = iou_matrices.get(sample_idx)
        if ious is None or pred_idx >= ious.shape[0] or ious.shape[1] == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        row = ious[pred_idx]
        above = np.flatnonzero(row > iou_thr)
        if above.size == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        target = np.asarray([
            int(gt['labels'][gt_idx]) == 0 and
            gt['rules'][gt_idx] in target_rules
            for gt_idx in above
        ], dtype=bool)
        if target.any():
            target_above = above[target]
            best_gt = int(target_above[np.argmax(row[target_above])])
        else:
            ignored_dets += 1
            continue

        gt_id = (sample_idx, best_gt)
        if gt_id in matched:
            tp.append(0.0)
            fp.append(1.0)
        else:
            matched.add(gt_id)
            tp.append(1.0)
            fp.append(0.0)

    if npos == 0:
        return {
            'num_gt': 0,
            'num_matched_gt': 0,
            'num_counted_predictions': int(len(tp)),
            'num_ignored_predictions': int(ignored_dets),
            'ap': math.nan,
            'ar': math.nan,
        }

    if not tp:
        return {
            'num_gt': int(npos),
            'num_matched_gt': 0,
            'num_counted_predictions': 0,
            'num_ignored_predictions': int(ignored_dets),
            'ap': 0.0,
            'ar': 0.0,
        }

    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_cum / max(float(npos), np.finfo(np.float64).eps)
    precisions = tp_cum / np.maximum(
        tp_cum + fp_cum, np.finfo(np.float64).eps)
    return {
        'num_gt': int(npos),
        'num_matched_gt': int(tp_cum[-1]),
        'num_counted_predictions': int(len(tp)),
        'num_ignored_predictions': int(ignored_dets),
        'ap': average_precision(recalls, precisions),
        'ar': float(recalls[-1]),
    }


def evaluate_model(gt_by_sample: Dict[int, Dict[str, Any]],
                   pred_by_sample: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    iou_matrices = build_iou_matrices(gt_by_sample, pred_by_sample)
    detections = []
    for sample_idx, preds in pred_by_sample.items():
        for pred_idx, pred in enumerate(preds):
            detections.append((sample_idx, pred_idx, float(pred['score'])))
    detections.sort(key=lambda x: x[2], reverse=True)

    rows = []
    num_predictions = len(detections)
    for subset_name, target_rules in SUBSETS:
        result = {
            'subset': subset_name,
            'num_predictions': int(num_predictions),
        }
        for iou_thr in IOU_THR_LIST:
            metrics = evaluate_one_thr(
                gt_by_sample, pred_by_sample, iou_matrices, detections,
                target_rules, iou_thr)
            suffix = f'{iou_thr:.2f}'
            result[f'num_gt_{suffix}'] = metrics['num_gt']
            result[f'num_matched_gt_{suffix}'] = metrics['num_matched_gt']
            result[f'num_counted_predictions_{suffix}'] = (
                metrics['num_counted_predictions'])
            result[f'num_ignored_predictions_{suffix}'] = (
                metrics['num_ignored_predictions'])
            result[f'AP_{suffix}'] = metrics['ap']
            result[f'AR_{suffix}'] = metrics['ar']
        rows.append(result)
    return rows


def parse_pred_arg(value: str) -> Tuple[str, str]:
    if '=' not in value:
        raise ValueError(f'--pred must be NAME=JSONL, got: {value}')
    name, path = value.split('=', 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f'--pred must be NAME=JSONL, got: {value}')
    return name, path


def format_float(value: Any) -> str:
    if isinstance(value, float) and math.isnan(value):
        return 'nan'
    return f'{float(value):.4f}'


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'subset_ap_summary.csv'
    txt_path = out_dir / 'subset_ap_summary.txt'
    json_path = out_dir / 'subset_ap_details.json'

    fieldnames = [
        'model', 'subset', 'pred_jsonl', 'num_predictions',
        'num_gt_0.25', 'AP_0.25', 'AR_0.25',
        'num_gt_0.50', 'AP_0.50', 'AR_0.50',
        'num_matched_gt_0.25', 'num_matched_gt_0.50',
        'num_counted_predictions_0.25', 'num_counted_predictions_0.50',
        'num_ignored_predictions_0.25', 'num_ignored_predictions_0.50',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with json_path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = []
    header = (
        f'{"Model":<12} {"Subset":<20} {"GT":>6} '
        f'{"AP25":>8} {"AP50":>8} {"AR25":>8} {"AR50":>8}'
    )
    lines.append(header)
    lines.append('-' * len(header))
    for row in rows:
        gt_count = row.get('num_gt_0.50', row.get('num_gt_0.25', 0))
        lines.append(
            f'{row["model"]:<12} {row["subset"]:<20} {gt_count:>6} '
            f'{format_float(row["AP_0.25"]):>8} '
            f'{format_float(row["AP_0.50"]):>8} '
            f'{format_float(row["AR_0.25"]):>8} '
            f'{format_float(row["AR_0.50"]):>8}')
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'saved: {csv_path}')
    print(f'saved: {txt_path}')
    print(f'saved: {json_path}')


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    quality = load_quality_map(args.quality_csv)
    gt_by_sample = load_gt(args.gt_ann_file, quality)

    all_rows = []
    for pred_arg in args.pred:
        model_name, pred_path = parse_pred_arg(pred_arg)
        print(f'loading predictions: {model_name} -> {pred_path}', flush=True)
        pred_by_sample = load_predictions(
            pred_path, args.score_thr, args.topk,
            args.pred_z_plus_half_height)
        rows = evaluate_model(gt_by_sample, pred_by_sample)
        for row in rows:
            row['model'] = model_name
            row['pred_jsonl'] = str(Path(pred_path).resolve())
        all_rows.extend(rows)

    write_outputs(all_rows, out_dir)


if __name__ == '__main__':
    main()
