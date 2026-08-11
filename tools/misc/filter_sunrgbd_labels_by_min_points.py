#!/usr/bin/env python3
"""Remove SUNRGBD label boxes whose enclosed point count is below a threshold."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Filter SUNRGBD-style labels and info pkl instances by point count.')
    parser.add_argument(
        '--standard-root',
        type=Path,
        default=Path('data/sunrgbd'),
        help='Standard dataset root containing points/ and sunrgbd_trainval/.')
    parser.add_argument(
        '--flat-root',
        type=Path,
        default=Path('data/sunrgbd_flat'),
        help='Optional flat source root whose label/<id>.txt files will be synced.')
    parser.add_argument('--min-points', type=int, default=1000)
    parser.add_argument(
        '--quality-csv',
        type=Path,
        default=None,
        help='Optional quality CSV with bbox_3d/rule_class columns.')
    parser.add_argument(
        '--target-rules',
        nargs='*',
        default=[],
        help='If set, only boxes whose quality rule is in this list can be removed.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return value


def load_points(path: Path, num_features: int = 6) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, num_features), dtype=np.float32)
    candidates = [num_features, 6, 4, 3]
    seen = set()
    for dim in candidates:
        if dim in seen:
            continue
        seen.add(dim)
        if dim > 0 and arr.size % dim == 0:
            return arr.reshape(-1, dim)
    raise ValueError(f'Cannot infer point dimension for {path}, values={arr.size}')


def count_points_in_box(points_xyz: np.ndarray, box: Sequence[float]) -> int:
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    rel_x = points_xyz[:, 0] - cx
    rel_y = points_xyz[:, 1] - cy
    rel_z = points_xyz[:, 2] - cz
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    local_x = rel_x * c - rel_y * s
    local_y = rel_x * s + rel_y * c
    inside = (
        (np.abs(local_x) <= dx * 0.5) &
        (np.abs(local_y) <= dy * 0.5) &
        (np.abs(rel_z) <= dz * 0.5))
    return int(np.count_nonzero(inside))


def read_label_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        line for line in path.read_text(encoding='utf-8', errors='ignore').splitlines()
        if line.strip()
    ]


def write_label(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def backup_dataset_files(standard_root: Path, flat_root: Path | None,
                         tag: str) -> Dict[str, str]:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_root = standard_root / 'filter_label_minpoints_backups' / f'before_{tag}_{stamp}'
    backup_root.mkdir(parents=True, exist_ok=False)

    std_tv = standard_root / 'sunrgbd_trainval'
    shutil.copytree(std_tv / 'label', backup_root / 'standard_label')
    for name in ['sunrgbd_infos_train.pkl', 'sunrgbd_infos_val.pkl']:
        for base in [standard_root, std_tv]:
            src = base / name
            if src.exists():
                dst = backup_root / base.relative_to(standard_root) / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    flat_backup = None
    if flat_root is not None and flat_root.exists() and (flat_root / 'label').exists():
        flat_backup = backup_root / 'flat_label'
        shutil.copytree(flat_root / 'label', flat_backup)

    return {
        'backup_root': str(backup_root),
        'standard_label_backup': str(backup_root / 'standard_label'),
        'flat_label_backup': str(flat_backup) if flat_backup else '',
    }


def quality_key(sample_idx: int | str, box: Sequence[float]) -> tuple[int, tuple[float, ...]]:
    return int(sample_idx), tuple(round(float(v), 6) for v in box[:7])


def load_quality_rules(path: Path | None) -> Dict[tuple[int, tuple[float, ...]], str]:
    if path is None:
        return {}
    import csv

    rules = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') and row.get('split') != 'val':
                continue
            sample_idx = row.get('sample_idx')
            box_text = row.get('bbox_3d')
            rule = (row.get('rule_class') or '').strip()
            if sample_idx is None or not box_text or not rule:
                continue
            box = [float(v) for v in box_text.split()[:7]]
            rules[quality_key(sample_idx, box)] = rule
    return rules


def main() -> None:
    args = parse_args()
    standard_root = args.standard_root.resolve()
    flat_root = args.flat_root.resolve() if args.flat_root else None
    trainval_root = standard_root / 'sunrgbd_trainval'
    info_path = trainval_root / 'sunrgbd_infos_val.pkl'
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    infos = load_pickle(info_path)
    data_list = infos['data_list']
    quality_rules = load_quality_rules(args.quality_csv)
    target_rules = set(args.target_rules)
    keep_by_sample: Dict[int, List[bool]] = {}
    count_by_sample: Dict[int, List[int]] = {}
    rule_by_sample: Dict[int, List[str]] = {}
    label_mismatches = []
    quality_missing = []

    for dataset_idx, info in enumerate(data_list):
        sample_idx = int(info.get('sample_idx', dataset_idx))
        lidar_path = info.get('lidar_points', {}).get('lidar_path', f'{sample_idx:05d}.bin')
        point_path = standard_root / 'points' / lidar_path
        num_features = int(info.get('lidar_points', {}).get('num_pts_feats', 6))
        points_xyz = load_points(point_path, num_features)[:, :3].astype(np.float64, copy=False)

        counts = []
        keeps = []
        rules = []
        for inst in info.get('instances', []):
            box = inst.get('bbox_3d')
            if box is None or len(box) < 7:
                counts.append(0)
                keeps.append(False)
                rules.append('missing_box')
                continue
            rule = quality_rules.get(quality_key(sample_idx, box), '')
            if target_rules and not rule:
                quality_missing.append({'sample_idx': sample_idx, 'bbox_3d': box})
            count = count_points_in_box(points_xyz, box)
            eligible = not target_rules or rule in target_rules
            counts.append(count)
            keeps.append((not eligible) or count >= int(args.min_points))
            rules.append(rule or 'unscoped')
        keep_by_sample[sample_idx] = keeps
        count_by_sample[sample_idx] = counts
        rule_by_sample[sample_idx] = rules

        label_path = trainval_root / 'label' / f'{sample_idx:05d}.txt'
        num_label_lines = len(read_label_lines(label_path))
        if num_label_lines != len(keeps):
            label_mismatches.append({
                'sample_idx': sample_idx,
                'label_lines': num_label_lines,
                'instances': len(keeps),
            })

    all_counts = [count for counts in count_by_sample.values() for count in counts]
    kept_counts = [count for counts, keeps in zip(count_by_sample.values(), keep_by_sample.values())
                   for count, keep in zip(counts, keeps) if keep]
    removed_counts = [count for counts, keeps in zip(count_by_sample.values(), keep_by_sample.values())
                      for count, keep in zip(counts, keeps) if not keep]
    removed_rules = [
        rule
        for rules, keeps in zip(rule_by_sample.values(), keep_by_sample.values())
        for rule, keep in zip(rules, keeps) if not keep
    ]
    if label_mismatches:
        raise RuntimeError(f'Label/instance count mismatch: {label_mismatches[:5]}')
    if target_rules and quality_missing:
        raise RuntimeError(f'Missing quality labels for scoped filter: {quality_missing[:5]}')

    tag_rules = '_'.join(sorted(target_rules)) if target_rules else 'all'
    backup_tag = f'minpoints{args.min_points}_{tag_rules}'
    backup = {} if args.dry_run else backup_dataset_files(
        standard_root, flat_root, backup_tag)

    per_sample_rows = []
    if not args.dry_run:
        for sample_idx, keeps in keep_by_sample.items():
            std_label = trainval_root / 'label' / f'{sample_idx:05d}.txt'
            lines = read_label_lines(std_label)
            new_lines = [line for line, keep in zip(lines, keeps) if keep]
            write_label(std_label, new_lines)

            if flat_root is not None and (flat_root / 'label').exists():
                flat_label = flat_root / 'label' / f'{sample_idx}.txt'
                if flat_label.exists():
                    write_label(flat_label, new_lines)

        for name in ['sunrgbd_infos_train.pkl', 'sunrgbd_infos_val.pkl']:
            for base in [standard_root, trainval_root]:
                pkl_path = base / name
                if not pkl_path.exists():
                    continue
                data = load_pickle(pkl_path)
                for dataset_idx, info in enumerate(data['data_list']):
                    sample_idx = int(info.get('sample_idx', dataset_idx))
                    keeps = keep_by_sample[sample_idx]
                    info['instances'] = [
                        inst for inst, keep in zip(info.get('instances', []), keeps)
                        if keep
                    ]
                dump_pickle(data, pkl_path)

    for sample_idx in sorted(keep_by_sample):
        keeps = keep_by_sample[sample_idx]
        counts = count_by_sample[sample_idx]
        per_sample_rows.append({
            'sample_idx': sample_idx,
            'before': len(keeps),
            'after': int(sum(keeps)),
            'removed': int(len(keeps) - sum(keeps)),
            'min_count_before': min(counts) if counts else None,
            'min_count_after': min([c for c, k in zip(counts, keeps) if k], default=None),
        })

    summary = {
        'standard_root': str(standard_root),
        'flat_root': str(flat_root) if flat_root else '',
        'min_points': int(args.min_points),
        'target_rules': sorted(target_rules),
        'dry_run': bool(args.dry_run),
        'backup': backup,
        'total_before': len(all_counts),
        'total_after': len(kept_counts),
        'total_removed': len(removed_counts),
        'min_count_before': min(all_counts) if all_counts else None,
        'min_count_after': min(kept_counts) if kept_counts else None,
        'max_removed_count': max(removed_counts) if removed_counts else None,
        'removed_by_rule': dict(sorted(Counter(removed_rules).items())),
        'removed_count_histogram_100pt_bins': dict(
            sorted(Counter((count // 100) * 100 for count in removed_counts).items())),
        'per_sample': per_sample_rows,
    }

    out_dir = trainval_root / 'label_filter_reports'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'minpoints_{args.min_points}_{stamp}.json'
    out_path.write_text(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2),
                        encoding='utf-8')
    summary['report_path'] = str(out_path)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
