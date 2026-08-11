#!/usr/bin/env python3
"""Randomly browse VPGNet JSONL detections with Open3D windows."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d


DEFAULT_PRED_JSONL = (
    'work_dirs/vpgnet_predictions/val_predictions.jsonl')
DEFAULT_POINTS_ROOT = 'data/sunrgbd/points'
DEFAULT_GT_ANN_FILE = 'data/sunrgbd/sunrgbd_trainval/sunrgbd_infos_val.pkl'

GT_COLOR = np.array([0.0, 0.9, 0.1], dtype=np.float64)
GT_YAW_COLOR = np.array([0.0, 0.55, 0.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Open random VPGNet detection frames in Open3D.')
    parser.add_argument('--pred-jsonl', type=str, default=DEFAULT_PRED_JSONL)
    parser.add_argument('--points-root', type=str, default=DEFAULT_POINTS_ROOT)
    parser.add_argument(
        '--gt-ann-file',
        type=str,
        default=DEFAULT_GT_ANN_FILE,
        help='SUNRGBD annotation pkl for GT boxes.')
    parser.add_argument('--no-gt', action='store_true')
    parser.add_argument('--sample-ids', nargs='*', default=None)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260712)
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--load-dim', type=int, default=6)
    parser.add_argument('--point-size', type=float, default=1.8)
    parser.add_argument('--line-width', type=float, default=8.0)
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--bg-white', action='store_true')
    parser.add_argument(
        '--display-seconds',
        type=float,
        default=0.0,
        help='Auto-close each frame after N seconds. <=0 waits for manual close.'
    )
    parser.add_argument(
        '--top-view',
        action='store_true',
        help='Open the window from a top-down view.')
    parser.add_argument(
        '--show-yaw',
        action='store_true',
        help='Draw a short yellow yaw direction line for each box.')
    return parser.parse_args()


def normalize_sample_id(value: int | str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError('Empty sample id')
    if text.isdigit():
        return str(int(text))
    return str(int(Path(text).stem))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gt_map(path: Path) -> dict[str, list[tuple[np.ndarray, np.ndarray, float]]]:
    with path.open('rb') as f:
        data = pickle.load(f)
    infos = data['data_list'] if isinstance(data, dict) and 'data_list' in data else data

    gt_map: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for info in infos:
        sample_id = normalize_sample_id(info['sample_idx'])
        boxes = []
        for inst in info.get('instances', []):
            bbox = np.asarray(inst.get('bbox_3d', []), dtype=np.float64)
            if bbox.size < 7:
                continue
            center = bbox[:3].astype(np.float64)
            size = bbox[3:6].astype(np.float64)
            if np.any(size <= 0):
                continue
            boxes.append((center, size, float(bbox[6])))
        gt_map[sample_id] = boxes
    return gt_map


def row_sample_id(row: dict) -> str:
    for key in ('sample_idx', 'sample_id', 'dataset_index', 'index'):
        if row.get(key) is not None:
            return normalize_sample_id(row[key])
    for key in ('img_path', 'pts_path', 'lidar_path'):
        if row.get(key):
            return normalize_sample_id(Path(str(row[key])).stem)
    raise KeyError(f'Cannot infer sample id from row keys: {sorted(row)}')


def resolve_point_path(points_root: Path, sample_id: str) -> Path:
    value = int(sample_id)
    candidates = [
        points_root / f'{value:05d}.bin',
        points_root / f'{value:06d}.bin',
        points_root / f'{value}.bin',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Missing point cloud. Tried:\n' + '\n'.join(str(p) for p in candidates))


def load_points(path: Path, load_dim: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % load_dim != 0:
        raise ValueError(f'{path} size {raw.size} is not divisible by {load_dim}')
    points = raw.reshape(-1, load_dim)
    xyz = points[:, :3].astype(np.float64)
    if load_dim >= 6:
        rgb = points[:, 3:6].astype(np.float64)
        if rgb.max(initial=0.0) > 1.0:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.full((len(xyz), 3), 0.55, dtype=np.float64)
    return xyz, rgb


def rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def box_color(score: float) -> np.ndarray:
    if score >= 0.95:
        return np.array([1.0, 0.05, 0.05], dtype=np.float64)
    if score >= 0.5:
        return np.array([1.0, 0.55, 0.0], dtype=np.float64)
    return np.array([0.1, 0.75, 1.0], dtype=np.float64)


def build_box(center: np.ndarray, size: np.ndarray, yaw: float,
              color: np.ndarray) -> o3d.geometry.LineSet:
    obb = o3d.geometry.OrientedBoundingBox(center, rotz(float(yaw)), size)
    obb.color = color
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color(color)
    return lines


def build_yaw_line(center: np.ndarray, size: np.ndarray, yaw: float,
                   color: np.ndarray) -> o3d.geometry.LineSet:
    length = max(float(size[0]), float(size[1])) * 0.65
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    start = center.copy()
    end = center + direction * length
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.vstack([start, end]))
    lines.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    lines.paint_uniform_color(color)
    return lines


def iter_pred_boxes(
    row: dict,
    score_thr: float,
    topk: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, float, float]]:
    boxes = np.asarray(row.get('boxes_3d', []), dtype=np.float64)
    scores = np.asarray(row.get('scores_3d', []), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] < 7 or scores.size == 0:
        return []

    order = np.argsort(-scores)
    out: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    for idx in order:
        score = float(scores[idx])
        if score < score_thr:
            continue
        box = boxes[idx]
        center = box[:3].astype(np.float64)
        size = box[3:6].astype(np.float64)
        if np.any(size <= 0):
            continue
        center = center.copy()
        center[2] += size[2] * 0.5
        out.append((center, size, float(box[6]), score))
        if topk > 0 and len(out) >= topk:
            break
    return out


def make_geometries(
    xyz: np.ndarray,
    rgb: np.ndarray,
    pred_boxes: Iterable[tuple[np.ndarray, np.ndarray, float, float]],
    gt_boxes: Iterable[tuple[np.ndarray, np.ndarray, float]],
    show_yaw: bool,
) -> list[o3d.geometry.Geometry]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geometries: list[o3d.geometry.Geometry] = [pcd]

    for center, size, yaw, score in pred_boxes:
        color = box_color(score)
        geometries.append(build_box(center, size, yaw, color))
        if show_yaw:
            geometries.append(
                build_yaw_line(center, size, yaw,
                               np.array([1.0, 0.9, 0.0], dtype=np.float64)))
    for center, size, yaw in gt_boxes:
        geometries.append(build_box(center, size, yaw, GT_COLOR))
        if show_yaw:
            geometries.append(build_yaw_line(center, size, yaw, GT_YAW_COLOR))
    return geometries


def apply_top_view(vis: o3d.visualization.Visualizer, xyz: np.ndarray) -> None:
    center = xyz.mean(axis=0)
    extent = np.ptp(xyz, axis=0)
    ctr = vis.get_view_control()
    ctr.set_lookat(center.tolist())
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_zoom(max(0.35, min(0.9, 1.8 / max(float(extent[0]), float(extent[1]), 1.0))))


def show_row(
    row: dict,
    sample_id: str,
    gt_map: dict[str, list[tuple[np.ndarray, np.ndarray, float]]],
    args: argparse.Namespace,
) -> None:
    point_path = resolve_point_path(Path(args.points_root), sample_id)
    xyz, rgb = load_points(point_path, args.load_dim)
    pred_boxes = list(iter_pred_boxes(row, args.score_thr, args.topk))
    gt_boxes = [] if args.no_gt else gt_map.get(sample_id, [])

    title = (
        f'VPGNet detections | sample {int(sample_id):05d} | '
        f'pred {len(pred_boxes)} | GT {len(gt_boxes)} | close window for next')
    print(f'[Open3D] {title} | points={len(xyz)} | {point_path}', flush=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=args.width, height=args.height)
    for geom in make_geometries(xyz, rgb, pred_boxes, gt_boxes, args.show_yaw):
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    if hasattr(opt, 'line_width'):
        opt.line_width = float(args.line_width)
    opt.background_color = np.asarray(
        [1.0, 1.0, 1.0] if args.bg_white else [0.0, 0.0, 0.0],
        dtype=np.float64)

    if args.top_view:
        vis.poll_events()
        vis.update_renderer()
        apply_top_view(vis, xyz)

    if args.display_seconds > 0:
        end_time = time.time() + float(args.display_seconds)
        while time.time() < end_time:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
    else:
        vis.run()
    vis.destroy_window()


def select_rows(rows: list[dict], args: argparse.Namespace) -> list[tuple[str, dict]]:
    by_id = {row_sample_id(row): row for row in rows}
    if args.sample_ids:
        selected = []
        for raw_id in args.sample_ids:
            sample_id = normalize_sample_id(raw_id)
            if sample_id not in by_id:
                raise KeyError(f'Sample {sample_id} not found in {args.pred_jsonl}')
            selected.append((sample_id, by_id[sample_id]))
        return selected

    sample_ids = sorted(by_id, key=lambda item: int(item))
    rng = random.Random(args.seed)
    count = min(max(int(args.count), 0), len(sample_ids))
    picked = rng.sample(sample_ids, count)
    return [(sample_id, by_id[sample_id]) for sample_id in picked]


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.pred_jsonl))
    gt_map = {} if args.no_gt else load_gt_map(Path(args.gt_ann_file))
    selected = select_rows(rows, args)
    print(
        f'[Info] pred_jsonl={args.pred_jsonl}\n'
        f'[Info] gt_ann_file={None if args.no_gt else args.gt_ann_file}\n'
        f'[Info] selected={",".join(f"{int(s):05d}" for s, _ in selected)}',
        flush=True)
    for sample_id, row in selected:
        show_row(row, sample_id, gt_map, args)
    print('[Done] Finished Open3D browsing.', flush=True)


if __name__ == '__main__':
    main()
