#!/usr/bin/env python3
"""Sample luggage instances and visualize XY projection coverage."""

import argparse
import csv
import math
import pickle
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize random luggage XY projections for coverage rules.')
    parser.add_argument(
        '--data-root',
        default='data/sunrgbd',
        help='SUN RGB-D luggage data root.')
    parser.add_argument(
        '--ann-files',
        nargs='+',
        default=[
            'sunrgbd_trainval/sunrgbd_infos_train.pkl',
            'sunrgbd_trainval/sunrgbd_infos_val.pkl',
        ])
    parser.add_argument('--points-dir', default='points')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260709)
    parser.add_argument('--canvas-size', type=int, default=512)
    parser.add_argument('--padding-m', type=float, default=0.08)
    parser.add_argument('--z-margin', type=float, default=0.03)
    parser.add_argument(
        '--morph-radius-m',
        type=float,
        default=0.018,
        help='Metric radius for dilate+erode on projected point mask.')
    parser.add_argument('--normal-thr', type=float, default=90.0)
    parser.add_argument('--occlusion-iou-thr', type=float, default=30.0)
    parser.add_argument('--draw-overlap-thr', type=float, default=1.0)
    parser.add_argument('--sheet-cols', type=int, default=5)
    parser.add_argument('--sheet-rows', type=int, default=5)
    return parser.parse_args()


def load_infos(data_root, ann_files):
    all_items = []
    for ann_file in ann_files:
        ann_path = Path(data_root) / ann_file
        with ann_path.open('rb') as f:
            data = pickle.load(f)
        items = data.get('data_list', data) if isinstance(data, dict) else data
        split_name = ann_path.stem.replace('sunrgbd_infos_', '')
        for local_idx, info in enumerate(items):
            all_items.append((split_name, local_idx, info))
    return all_items


def gather_instances(items):
    records = []
    for split, local_idx, info in items:
        instances = info.get('instances', [])
        for inst_idx, inst in enumerate(instances):
            if int(inst.get('bbox_label_3d', inst.get('bbox_label', 0))) != 0:
                continue
            box = inst.get('bbox_3d')
            if box is None or len(box) < 7:
                continue
            records.append((split, local_idx, inst_idx, info, inst))
    return records


def load_points(data_root, points_dir, info):
    rel_path = info.get('lidar_points', {}).get('lidar_path')
    if rel_path is None:
        rel_path = f'{int(info["sample_idx"]):05d}.bin'
    path = Path(data_root) / points_dir / rel_path
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 6 == 0:
        pts = pts.reshape(-1, 6)
    elif pts.size % 4 == 0:
        pts = pts.reshape(-1, 4)
    else:
        raise ValueError(f'Unexpected point file shape: {path}, values={pts.size}')
    return pts, path


def rotation_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def box_corners_xy(box):
    cx, cy, _, dx, dy, _, yaw = map(float, box[:7])
    local = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2],
         [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
        dtype=np.float64)
    return local @ rotation_matrix(yaw).T + np.array([cx, cy], dtype=np.float64)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, dtype=np.float64)
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1))) * 0.5)


def signed_polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly, dtype=np.float64)
    return float((np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                  np.dot(p[:, 1], np.roll(p[:, 0], -1))) * 0.5)


def ensure_ccw(poly):
    poly = np.asarray(poly, dtype=np.float64)
    return poly if signed_polygon_area(poly) >= 0 else poly[::-1].copy()


def _inside(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-9


def _line_intersection(p1, p2, a, b):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return p2
    t = ((a - p1)[0] * s[1] - (a - p1)[1] * s[0]) / denom
    return p1 + t * r


def polygon_clip(subject, clipper):
    output = [np.asarray(p, dtype=np.float64) for p in subject]
    clipper = [np.asarray(p, dtype=np.float64) for p in clipper]
    for i in range(len(clipper)):
        if not output:
            break
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if _inside(e, a, b):
                if not _inside(s, a, b):
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b):
                output.append(_line_intersection(s, e, a, b))
            s = e
    return np.asarray(output, dtype=np.float64)


def oriented_box_iou_xy(box_a, box_b):
    poly_a = ensure_ccw(box_corners_xy(box_a))
    poly_b = ensure_ccw(box_corners_xy(box_b))
    inter = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter)
    if inter_area <= 0:
        return 0.0
    union = polygon_area(poly_a) + polygon_area(poly_b) - inter_area
    return 0.0 if union <= 0 else inter_area / union


def points_in_oriented_box(points_xyz, box, z_margin=0.0):
    cx, cy, cz, dx, dy, dz, yaw = map(float, box[:7])
    rel_xy = points_xyz[:, :2] - np.array([cx, cy], dtype=np.float64)
    local_xy = rel_xy @ rotation_matrix(yaw)
    local_z = points_xyz[:, 2] - cz
    mask = (
        (np.abs(local_xy[:, 0]) <= dx / 2) &
        (np.abs(local_xy[:, 1]) <= dy / 2) &
        (np.abs(local_z) <= dz / 2 + z_margin))
    return local_xy, local_z, mask


def make_transform(box, canvas_size, padding_m):
    _, _, _, dx, dy, _, _ = map(float, box[:7])
    margin = 48
    min_x, max_x = -dx / 2 - padding_m, dx / 2 + padding_m
    min_y, max_y = -dy / 2 - padding_m, dy / 2 + padding_m
    sx = (canvas_size - 2 * margin) / max(max_x - min_x, 1e-6)
    sy = (canvas_size - 2 * margin) / max(max_y - min_y, 1e-6)
    scale = min(sx, sy)

    def to_px(xy):
        xy = np.asarray(xy, dtype=np.float64)
        u = margin + (xy[..., 0] - min_x) * scale
        v = canvas_size - margin - (xy[..., 1] - min_y) * scale
        return np.stack([u, v], axis=-1)

    return to_px, scale


def draw_text(img, lines, x=12, y=22):
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (25, 25, 25), 1, cv2.LINE_AA)
        y += 18


def render_case(record, data_root, points_dir, out_path, args):
    split, local_idx, inst_idx, info, inst = record
    target_box = np.asarray(inst['bbox_3d'], dtype=np.float64)
    points, points_path = load_points(data_root, points_dir, info)
    points_xyz = points[:, :3].astype(np.float64)

    local_xy_all, local_z_all, inside_mask = points_in_oriented_box(
        points_xyz, target_box, z_margin=args.z_margin)
    target_local_xy = local_xy_all[inside_mask]

    canvas = args.canvas_size
    img = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    overlay = img.copy()
    to_px, scale = make_transform(target_box, canvas, args.padding_m)

    _, _, _, dx, dy, _, _ = map(float, target_box[:7])
    target_rect = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2],
         [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
        dtype=np.float64)
    target_poly_px = np.round(to_px(target_rect)).astype(np.int32)

    box_mask = np.zeros((canvas, canvas), dtype=np.uint8)
    cv2.fillPoly(box_mask, [target_poly_px], 255)
    cv2.fillPoly(overlay, [target_poly_px], (235, 245, 255))

    point_mask = np.zeros((canvas, canvas), dtype=np.uint8)
    if len(target_local_xy) > 0:
        pts_px = np.round(to_px(target_local_xy)).astype(np.int32)
        valid = (
            (pts_px[:, 0] >= 0) & (pts_px[:, 0] < canvas) &
            (pts_px[:, 1] >= 0) & (pts_px[:, 1] < canvas))
        for u, v in pts_px[valid]:
            cv2.circle(point_mask, (int(u), int(v)), 1, 255, -1)

    kernel_radius = max(1, int(round(args.morph_radius_m * scale)))
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morph_mask = cv2.erode(cv2.dilate(point_mask, kernel, iterations=1),
                           kernel, iterations=1)

    cover_mask = (morph_mask > 0) & (box_mask > 0)
    box_area_px = max(1, int(np.count_nonzero(box_mask)))
    raw_coverage = 100.0 * np.count_nonzero((point_mask > 0) & (box_mask > 0)) / box_area_px
    morph_coverage = 100.0 * np.count_nonzero(cover_mask) / box_area_px

    overlay[cover_mask] = (90, 90, 245)
    img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
    cv2.polylines(img, [target_poly_px], True, (255, 80, 0), 2, cv2.LINE_AA)

    if len(target_local_xy) > 0:
        pts_px = np.round(to_px(target_local_xy)).astype(np.int32)
        valid = (
            (pts_px[:, 0] >= 0) & (pts_px[:, 0] < canvas) &
            (pts_px[:, 1] >= 0) & (pts_px[:, 1] < canvas))
        for u, v in pts_px[valid][::max(1, len(pts_px[valid]) // 2500)]:
            cv2.circle(img, (int(u), int(v)), 1, (0, 0, 0), -1)

    target_center = target_box[:2]
    target_rot = rotation_matrix(float(target_box[6]))
    max_lower_iou = 0.0
    max_any_iou = 0.0
    occluder_match_count = 0
    other_instances = info.get('instances', [])
    for other_idx, other in enumerate(other_instances):
        if other_idx == inst_idx or other.get('bbox_3d') is None:
            continue
        other_box = np.asarray(other['bbox_3d'], dtype=np.float64)
        iou = oriented_box_iou_xy(target_box, other_box)
        max_any_iou = max(max_any_iou, iou)
        is_occluder = float(other_box[2]) < float(target_box[2])
        if is_occluder:
            max_lower_iou = max(max_lower_iou, iou)
            if iou >= args.occlusion_iou_thr / 100.0:
                occluder_match_count += 1
        if iou * 100 < args.draw_overlap_thr:
            continue
        other_world = box_corners_xy(other_box)
        other_local = (other_world - target_center) @ target_rot
        other_px = np.round(to_px(other_local)).astype(np.int32)
        color = (40, 170, 40) if is_occluder else (150, 150, 150)
        width = 2 if is_occluder else 1
        cv2.polylines(img, [other_px], True, color, width, cv2.LINE_AA)
        label_pos = tuple(other_px[0])
        cv2.putText(img, f'{other_idx}:{iou * 100:.0f}',
                    (int(label_pos[0]), int(label_pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    if morph_coverage >= args.normal_thr:
        cls = 'normal'
    elif max_lower_iou * 100 >= args.occlusion_iou_thr:
        cls = 'occluded'
    else:
        cls = 'reflective'

    sample_idx = int(info.get('sample_idx', -1))
    draw_text(img, [
        f'{split} sample={sample_idx:05d} inst={inst_idx}',
        f'cls={cls} cov={morph_coverage:.1f}% raw={raw_coverage:.1f}%',
        f'pts={len(target_local_xy)} occIoU={max_lower_iou * 100:.1f}% anyIoU={max_any_iou * 100:.1f}%',
        f'z={target_box[2]:.3f} dz={target_box[5]:.3f} morph_r={args.morph_radius_m:.3f}m',
    ])
    cv2.putText(img, 'blue=target box  red=morph points  green=occluder-overlap',
                (12, canvas - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (25, 25, 25), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return {
        'image': str(out_path),
        'split': split,
        'sample_idx': sample_idx,
        'local_info_idx': local_idx,
        'inst_idx': inst_idx,
        'points_path': str(points_path),
        'bbox_3d': ' '.join(f'{v:.6f}' for v in target_box.tolist()),
        'num_points_in_box': len(target_local_xy),
        'raw_coverage_pct': f'{raw_coverage:.4f}',
        'morph_coverage_pct': f'{morph_coverage:.4f}',
        'max_occluder_iou_pct': f'{max_lower_iou * 100:.4f}',
        'max_any_iou_pct': f'{max_any_iou * 100:.4f}',
        'occluder_match_count': occluder_match_count,
        'rule_class': cls,
        'kernel_size_px': kernel_size,
        'pixel_scale_per_m': f'{scale:.4f}',
    }, img


def make_contact_sheets(images, out_dir, cols, rows):
    per_sheet = cols * rows
    thumbs = []
    for img in images:
        thumbs.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))
    for sheet_idx in range(0, len(thumbs), per_sheet):
        chunk = thumbs[sheet_idx:sheet_idx + per_sheet]
        if not chunk:
            continue
        while len(chunk) < per_sheet:
            chunk.append(np.full_like(chunk[0], 255))
        grid_rows = []
        for r in range(rows):
            grid_rows.append(np.hstack(chunk[r * cols:(r + 1) * cols]))
        sheet = np.vstack(grid_rows)
        out_path = Path(out_dir) / f'contact_sheet_{sheet_idx // per_sheet:02d}.png'
        cv2.imwrite(str(out_path), sheet)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    items = load_infos(data_root, args.ann_files)
    records = gather_instances(items)
    if len(records) < args.count:
        raise RuntimeError(f'Only found {len(records)} instances, count={args.count}')

    rng = random.Random(args.seed)
    sampled = rng.sample(records, args.count)

    rows = []
    rendered = []
    for idx, record in enumerate(sampled):
        split, _, inst_idx, info, _ = record
        sample_idx = int(info.get('sample_idx', -1))
        out_name = f'{idx:03d}_{split}_{sample_idx:05d}_inst{inst_idx:03d}.png'
        row, img = render_case(record, data_root, args.points_dir,
                               image_dir / out_name, args)
        row['sample_order'] = idx
        rows.append(row)
        rendered.append(img)
        print(f'[{idx + 1:03d}/{args.count}] {row["rule_class"]} '
              f'sample={sample_idx:05d} inst={inst_idx} '
              f'cov={row["morph_coverage_pct"]} occIoU={row["max_occluder_iou_pct"]}')

    csv_path = out_dir / 'sample_index.csv'
    fieldnames = [
        'sample_order', 'image', 'split', 'sample_idx', 'local_info_idx',
        'inst_idx', 'rule_class', 'num_points_in_box', 'raw_coverage_pct',
        'morph_coverage_pct', 'max_occluder_iou_pct', 'max_any_iou_pct',
        'occluder_match_count', 'kernel_size_px', 'pixel_scale_per_m',
        'bbox_3d', 'points_path'
    ]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheets(rendered, out_dir, args.sheet_cols, args.sheet_rows)

    summary = {}
    for row in rows:
        summary[row['rule_class']] = summary.get(row['rule_class'], 0) + 1
    with (out_dir / 'summary.txt').open('w') as f:
        f.write(f'total_instances={len(records)}\n')
        f.write(f'sampled={len(rows)}\n')
        f.write(f'seed={args.seed}\n')
        f.write(f'normal_thr={args.normal_thr}\n')
        f.write(f'occlusion_iou_thr={args.occlusion_iou_thr}\n')
        f.write(f'morph_radius_m={args.morph_radius_m}\n')
        for k in sorted(summary):
            f.write(f'{k}={summary[k]}\n')
        f.write(f'csv={csv_path}\n')
        f.write(f'images={image_dir}\n')

    print(f'Wrote {csv_path}')
    print(f'Wrote images to {image_dir}')
    print(f'Class counts: {summary}')


if __name__ == '__main__':
    main()
