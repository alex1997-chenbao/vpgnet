#!/usr/bin/env python3
"""Compute IoU and yaw statistics for GT-first matched predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from eval_luggage_subset_pr_gtfirst import build_gt_first_pairs
from eval_luggage_subsets_ap import (build_iou_matrices, load_gt,
                                     load_predictions, load_quality_map)


SUBSETS = {
    'all': {'normal', 'reflective', 'occluded'},
    'normal': {'normal'},
    'reflective': {'reflective'},
    'occluded': {'occluded'},
    'reflective_or_occluded': {'reflective', 'occluded'},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute IoU/yaw stats for GT-first matched boxes.')
    parser.add_argument('--gt-ann-file', required=True)
    parser.add_argument('--quality-csv', required=True)
    parser.add_argument('--pred-jsonl', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--score-thr', type=float, default=None)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument(
        '--pred-z-plus-half-height',
        action='store_true',
        help='Add dz/2 to prediction z before matching.')
    return parser.parse_args()


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_error_deg(pred_yaw: float, gt_yaw: float) -> tuple[float, float]:
    raw = abs(wrap_pi(pred_yaw - gt_yaw))
    sym = min(raw, abs(math.pi - raw))
    return math.degrees(raw), math.degrees(sym)


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


def quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    out = {}
    for q in qs:
        out[f'p{q:02d}'] = float(np.percentile(values, q))
    return out


def basic_stats(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            'count': 0,
            'mean': math.nan,
            'std': math.nan,
            'min': math.nan,
            'max': math.nan,
            'quantiles': {},
        }
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'quantiles': quantiles(values),
    }


def iou_hist(values: np.ndarray) -> Dict[str, Any]:
    bins = np.asarray([0.0, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 1.000001])
    counts, _ = np.histogram(values, bins=bins)
    labels = ['[0,.25)', '[.25,.50)', '[.50,.70)', '[.70,.80)',
              '[.80,.90)', '[.90,.95)', '[.95,1]']
    return {
        label: int(count)
        for label, count in zip(labels, counts)
    }


def threshold_rates(values: np.ndarray, thresholds: Iterable[float]) -> Dict[str, Any]:
    if values.size == 0:
        return {f'>={thr:g}': math.nan for thr in thresholds}
    return {
        f'>={thr:g}': float(np.mean(values >= float(thr)))
        for thr in thresholds
    }


def yaw_rates(values: np.ndarray, thresholds: Iterable[float]) -> Dict[str, Any]:
    if values.size == 0:
        return {f'<={thr:g}deg': math.nan for thr in thresholds}
    return {
        f'<={thr:g}deg': float(np.mean(values <= float(thr)))
        for thr in thresholds
    }


def summarize(rows: List[Dict[str, Any]], target_rules: set[str]) -> Dict[str, Any]:
    selected = [row for row in rows if row['rule_class'] in target_rules]
    ious = np.asarray([row['iou_3d'] for row in selected], dtype=np.float64)
    yaw = np.asarray([row['yaw_error_deg'] for row in selected], dtype=np.float64)
    yaw_sym = np.asarray(
        [row['yaw_error_sym180_deg'] for row in selected], dtype=np.float64)
    center = np.asarray(
        [row['center_error_3d_m'] for row in selected], dtype=np.float64)
    center_xy = np.asarray(
        [row['center_error_xy_m'] for row in selected], dtype=np.float64)
    center_z = np.asarray(
        [row['center_error_z_m'] for row in selected], dtype=np.float64)
    return {
        'num_pairs': int(len(selected)),
        'iou_3d': {
            **basic_stats(ious),
            'histogram': iou_hist(ious),
            'threshold_rates': threshold_rates(
                ious, [0.25, 0.50, 0.70, 0.80, 0.90, 0.95]),
        },
        'yaw_error_deg': {
            **basic_stats(yaw),
            'threshold_rates': yaw_rates(yaw, [1, 2, 5, 10, 20, 45, 90]),
        },
        'yaw_error_sym180_deg': {
            **basic_stats(yaw_sym),
            'threshold_rates': yaw_rates(yaw_sym, [1, 2, 5, 10, 20, 45, 90]),
        },
        'center_error_3d_m': basic_stats(center),
        'center_error_xy_m': basic_stats(center_xy),
        'center_error_z_m': basic_stats(center_z),
    }


def build_match_rows(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    iou_matrices = build_iou_matrices(gt_by_sample, pred_by_sample)
    pairs_by_sample = build_gt_first_pairs(gt_by_sample, pred_by_sample,
                                           iou_matrices)
    rows: List[Dict[str, Any]] = []
    for sample_idx in sorted(gt_by_sample):
        gt = gt_by_sample[sample_idx]
        preds = pred_by_sample.get(sample_idx, [])
        for gt_idx, pred_idx, iou in pairs_by_sample.get(sample_idx, []):
            gt_box = gt['boxes'][gt_idx]
            pred = preds[pred_idx]
            pred_box = np.asarray(pred['box'], dtype=np.float64)
            yaw_raw, yaw_sym = yaw_error_deg(pred_box[6], gt_box[6])
            gt_center = gt_box[:3].astype(np.float64).copy()
            pred_center = pred_box[:3].astype(np.float64).copy()
            gt_center[2] += float(gt_box[5]) * 0.5
            pred_center[2] += float(pred_box[5]) * 0.5
            center_delta = pred_center - gt_center
            rows.append({
                'sample_idx': int(sample_idx),
                'gt_idx': int(gt_idx),
                'pred_idx': int(pred_idx),
                'rule_class': str(gt['rules'][gt_idx]),
                'iou_3d': float(iou),
                'score': float(pred['score']),
                'yaw_error_deg': float(yaw_raw),
                'yaw_error_sym180_deg': float(yaw_sym),
                'center_error_3d_m': float(np.linalg.norm(center_delta)),
                'center_error_xy_m': float(np.linalg.norm(center_delta[:2])),
                'center_error_z_m': float(abs(center_delta[2])),
                'gt_yaw_rad': float(gt_box[6]),
                'pred_yaw_rad': float(pred_box[6]),
                'gt_box': [float(v) for v in gt_box[:7]],
                'pred_box': [float(v) for v in pred_box[:7]],
            })
    return rows


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        'sample_idx', 'gt_idx', 'pred_idx', 'rule_class', 'iou_3d', 'score',
        'yaw_error_deg', 'yaw_error_sym180_deg', 'center_error_3d_m',
        'center_error_xy_m', 'center_error_z_m', 'gt_yaw_rad', 'pred_yaw_rad',
        'gt_box', 'pred_box',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out['gt_box'] = json.dumps(out['gt_box'])
            out['pred_box'] = json.dumps(out['pred_box'])
            writer.writerow(out)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    fieldnames = [
        'subset', 'num_pairs', 'iou_mean', 'iou_std', 'iou_min', 'iou_p25',
        'iou_p50', 'iou_p75', 'iou_p90', 'iou_p95', 'iou_max',
        'yaw_mean_deg', 'yaw_std_deg', 'yaw_p50_deg', 'yaw_p90_deg',
        'yaw_p95_deg', 'yaw_max_deg', 'yaw_sym_mean_deg',
        'yaw_sym_std_deg', 'yaw_sym_p50_deg', 'yaw_sym_p90_deg',
        'yaw_sym_p95_deg', 'yaw_sym_max_deg', 'center_error_mean_m',
        'center_error_std_m', 'center_error_p50_m', 'center_error_p90_m',
        'center_error_p95_m', 'center_error_max_m',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset, data in summary['subsets'].items():
            iou = data['iou_3d']
            yaw = data['yaw_error_deg']
            sym = data['yaw_error_sym180_deg']
            center = data['center_error_3d_m']
            writer.writerow({
                'subset': subset,
                'num_pairs': data['num_pairs'],
                'iou_mean': iou['mean'],
                'iou_std': iou['std'],
                'iou_min': iou['min'],
                'iou_p25': iou['quantiles'].get('p25'),
                'iou_p50': iou['quantiles'].get('p50'),
                'iou_p75': iou['quantiles'].get('p75'),
                'iou_p90': iou['quantiles'].get('p90'),
                'iou_p95': iou['quantiles'].get('p95'),
                'iou_max': iou['max'],
                'yaw_mean_deg': yaw['mean'],
                'yaw_std_deg': yaw['std'],
                'yaw_p50_deg': yaw['quantiles'].get('p50'),
                'yaw_p90_deg': yaw['quantiles'].get('p90'),
                'yaw_p95_deg': yaw['quantiles'].get('p95'),
                'yaw_max_deg': yaw['max'],
                'yaw_sym_mean_deg': sym['mean'],
                'yaw_sym_std_deg': sym['std'],
                'yaw_sym_p50_deg': sym['quantiles'].get('p50'),
                'yaw_sym_p90_deg': sym['quantiles'].get('p90'),
                'yaw_sym_p95_deg': sym['quantiles'].get('p95'),
                'yaw_sym_max_deg': sym['max'],
                'center_error_mean_m': center['mean'],
                'center_error_std_m': center['std'],
                'center_error_p50_m': center['quantiles'].get('p50'),
                'center_error_p90_m': center['quantiles'].get('p90'),
                'center_error_p95_m': center['quantiles'].get('p95'),
                'center_error_max_m': center['max'],
            })


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
    rows = build_match_rows(gt_by_sample, pred_by_sample)
    summary = {
        'args': vars(args),
        'num_prediction_boxes': int(sum(len(v) for v in pred_by_sample.values())),
        'num_gt_boxes': int(sum(len(v['boxes']) for v in gt_by_sample.values())),
        'num_gt_first_pairs': int(len(rows)),
        'subsets': {
            name: summarize(rows, rules)
            for name, rules in SUBSETS.items()
        },
    }

    json_path = out_dir / 'gt_first_match_iou_yaw_stats.json'
    rows_csv = out_dir / 'gt_first_match_iou_yaw_pairs.csv'
    summary_csv = out_dir / 'gt_first_match_iou_yaw_summary.csv'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(to_builtin(summary), f, ensure_ascii=False, indent=2)
    write_rows_csv(rows_csv, rows)
    write_summary_csv(summary_csv, summary)

    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))
    print(f'saved: {json_path}')
    print(f'saved: {summary_csv}')
    print(f'saved: {rows_csv}')


if __name__ == '__main__':
    main()
