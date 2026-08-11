#!/usr/bin/env python3
"""Evaluate luggage subset precision/recall and GT-first paired PR."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from eval_luggage_subsets_ap import (build_iou_matrices, load_gt,
                                     load_predictions, load_quality_map)


SUBSETS = [
    ('all', {'normal', 'reflective', 'occluded'}),
    ('normal', {'normal'}),
    ('reflective', {'reflective'}),
    ('occluded', {'occluded'}),
    ('reflective_or_occluded', {'reflective', 'occluded'}),
]
IOU_THR_LIST = (0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate luggage subset P/R and GT-first paired P/R.')
    parser.add_argument('--gt-ann-file', required=True)
    parser.add_argument('--quality-csv', required=True)
    parser.add_argument('--pred-jsonl', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument(
        '--pred-z-plus-half-height',
        action='store_true',
        help='Add dz/2 to prediction z before computing IoU.')
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, )):
        return int(value)
    if isinstance(value, (np.floating, )):
        return float(value)
    return value


def count_target_gt(gt_by_sample: Dict[int, Dict[str, Any]],
                    target_rules: Iterable[str]) -> int:
    target_rules = set(target_rules)
    return sum(
        1 for gt in gt_by_sample.values()
        for label, rule in zip(gt['labels'], gt['rules'])
        if int(label) == 0 and rule in target_rules)


def build_score_detections(
    pred_by_sample: Dict[int, List[Dict[str, Any]]]
) -> List[Tuple[int, int, float]]:
    detections = []
    for sample_idx, preds in pred_by_sample.items():
        for pred_idx, pred in enumerate(preds):
            detections.append((sample_idx, pred_idx, float(pred['score'])))
    detections.sort(key=lambda item: item[2], reverse=True)
    return detections


def evaluate_detection_pr_one_thr(
    gt_by_sample: Dict[int, Dict[str, Any]],
    iou_matrices: Dict[int, np.ndarray],
    detections: Sequence[Tuple[int, int, float]],
    target_rules: Iterable[str],
    iou_thr: float,
) -> Dict[str, Any]:
    target_rules = set(target_rules)
    num_gt = count_target_gt(gt_by_sample, target_rules)
    matched = set()
    tp = 0
    fp = 0
    ignored = 0

    for sample_idx, pred_idx, _score in detections:
        gt = gt_by_sample.get(sample_idx)
        ious = iou_matrices.get(sample_idx)
        if gt is None or ious is None or pred_idx >= ious.shape[0] or ious.shape[1] == 0:
            fp += 1
            continue
        row = ious[pred_idx]
        best_gt = int(np.argmax(row))
        best_iou = float(row[best_gt])
        if best_iou <= float(iou_thr):
            fp += 1
            continue
        if int(gt['labels'][best_gt]) != 0 or gt['rules'][best_gt] not in target_rules:
            ignored += 1
            continue
        gt_id = (sample_idx, best_gt)
        if gt_id in matched:
            fp += 1
        else:
            matched.add(gt_id)
            tp += 1

    return {
        'num_gt': int(num_gt),
        'tp': int(tp),
        'fp': int(fp),
        'ignored_predictions_best_match_other_subset': int(ignored),
        'counted_predictions': int(tp + fp),
        'precision': float(tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall': float(tp / num_gt) if num_gt else math.nan,
    }


def gt_first_pairs_for_sample(ious: np.ndarray) -> List[Tuple[int, int, float]]:
    if ious.size == 0 or ious.shape[0] == 0 or ious.shape[1] == 0:
        return []

    gt_order = []
    for gt_idx in range(ious.shape[1]):
        best_pred = int(np.argmax(ious[:, gt_idx]))
        gt_order.append((float(ious[best_pred, gt_idx]), gt_idx))
    gt_order.sort(reverse=True)

    used_preds = set()
    pairs = []
    for _best_iou, gt_idx in gt_order:
        pred_order = np.argsort(-ious[:, gt_idx])
        for pred_idx in pred_order:
            pred_idx = int(pred_idx)
            if pred_idx in used_preds:
                continue
            used_preds.add(pred_idx)
            pairs.append((gt_idx, pred_idx, float(ious[pred_idx, gt_idx])))
            break
    return pairs


def build_gt_first_pairs(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
    iou_matrices: Dict[int, np.ndarray],
) -> Dict[int, List[Tuple[int, int, float]]]:
    out = {}
    for sample_idx, gt in gt_by_sample.items():
        ious = iou_matrices.get(sample_idx)
        if ious is None:
            out[sample_idx] = []
            continue
        out[sample_idx] = gt_first_pairs_for_sample(ious)
    return out


def evaluate_gt_first_one_thr(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pairs_by_sample: Dict[int, List[Tuple[int, int, float]]],
    target_rules: Iterable[str],
    iou_thr: float,
) -> Dict[str, Any]:
    target_rules = set(target_rules)
    num_gt = count_target_gt(gt_by_sample, target_rules)
    selected = 0
    tp = 0
    fp = 0

    for sample_idx, pairs in pairs_by_sample.items():
        gt = gt_by_sample.get(sample_idx)
        if gt is None:
            continue
        for gt_idx, _pred_idx, iou in pairs:
            if int(gt['labels'][gt_idx]) != 0 or gt['rules'][gt_idx] not in target_rules:
                continue
            selected += 1
            if float(iou) > float(iou_thr):
                tp += 1
            else:
                fp += 1

    return {
        'num_gt': int(num_gt),
        'tp': int(tp),
        'fp': int(fp),
        'selected_pairs': int(selected),
        'unpaired_gt': int(num_gt - selected),
        'precision': float(tp / (tp + fp)) if (tp + fp) else 0.0,
        'recall': float(tp / num_gt) if num_gt else math.nan,
    }


def evaluate_all(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    iou_matrices = build_iou_matrices(gt_by_sample, pred_by_sample)
    detections = build_score_detections(pred_by_sample)
    pairs_by_sample = build_gt_first_pairs(gt_by_sample, pred_by_sample,
                                           iou_matrices)

    detection_pr = {}
    gt_first_pr = {}
    for subset_name, target_rules in SUBSETS:
        detection_pr[subset_name] = {}
        gt_first_pr[subset_name] = {}
        for iou_thr in IOU_THR_LIST:
            key = f'iou_{iou_thr:.2f}'
            detection_pr[subset_name][key] = evaluate_detection_pr_one_thr(
                gt_by_sample, iou_matrices, detections, target_rules, iou_thr)
            gt_first_pr[subset_name][key] = evaluate_gt_first_one_thr(
                gt_by_sample, pairs_by_sample, target_rules, iou_thr)

    return {
        'num_predictions': int(sum(len(v) for v in pred_by_sample.values())),
        'selected_pairs_total': int(sum(len(v) for v in pairs_by_sample.values())),
        'unpaired_gt_total': int(
            sum(len(gt['boxes']) for gt in gt_by_sample.values()) -
            sum(len(v) for v in pairs_by_sample.values())),
        'detection_pr': detection_pr,
        'gt_first_pr': gt_first_pr,
    }


def flatten_rows(metrics: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
    rows = []
    for subset, by_thr in metrics[section].items():
        for thr_key, values in by_thr.items():
            row = {
                'section': section,
                'subset': subset,
                'iou_thr': thr_key.replace('iou_', ''),
            }
            row.update(values)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        'section', 'subset', 'iou_thr', 'num_gt', 'tp', 'fp',
        'ignored_predictions_best_match_other_subset', 'counted_predictions',
        'selected_pairs', 'unpaired_gt', 'precision', 'recall',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quality = load_quality_map(args.quality_csv)
    gt_by_sample = load_gt(args.gt_ann_file, quality)
    pred_by_sample = load_predictions(
        args.pred_jsonl,
        args.score_thr,
        args.topk,
        pred_z_plus_half_height=args.pred_z_plus_half_height)
    metrics = evaluate_all(gt_by_sample, pred_by_sample)
    metrics['args'] = vars(args)

    json_path = out_dir / 'subset_pr_gt_first.json'
    csv_path = out_dir / 'subset_pr_gt_first.csv'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(to_builtin(metrics), f, ensure_ascii=False, indent=2)
    write_csv(csv_path, flatten_rows(metrics, 'detection_pr') +
              flatten_rows(metrics, 'gt_first_pr'))

    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2))
    print(f'saved: {json_path}')
    print(f'saved: {csv_path}')


if __name__ == '__main__':
    main()
