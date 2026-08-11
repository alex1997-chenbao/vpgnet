#!/usr/bin/env python3
"""Classify all luggage instances by XY projection coverage and occluders."""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from sample_luggage_xy_projection_cases import (
    box_corners_xy,
    gather_instances,
    load_infos,
    load_points,
    make_contact_sheets,
    make_transform,
    oriented_box_iou_xy,
    render_case,
    rotation_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify luggage instances into normal/occluded/reflective.')
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
    parser.add_argument('--seed', type=int, default=20260709)
    parser.add_argument('--save-random-images', type=int, default=2000)
    parser.add_argument(
        '--max-instances',
        type=int,
        default=0,
        help='Debug only: classify at most this many instances.')
    parser.add_argument('--canvas-size', type=int, default=512)
    parser.add_argument('--padding-m', type=float, default=0.08)
    parser.add_argument('--z-margin', type=float, default=0.03)
    parser.add_argument('--morph-radius-m', type=float, default=0.018)
    parser.add_argument('--normal-thr', type=float, default=90.0)
    parser.add_argument('--occlusion-iou-thr', type=float, default=30.0)
    parser.add_argument('--draw-overlap-thr', type=float, default=1.0)
    parser.add_argument('--contact-sheet-count', type=int, default=100)
    parser.add_argument('--sheet-cols', type=int, default=5)
    parser.add_argument('--sheet-rows', type=int, default=5)
    return parser.parse_args()


def instance_key(info, inst_idx):
    return int(info.get('sample_idx', -1)), int(inst_idx)


def box_local_points_prefilter(points_xyz, box, z_margin):
    cx, cy, cz, dx, dy, dz, yaw = map(float, box[:7])
    corners = box_corners_xy(box)
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    z_min = cz - dz / 2 - z_margin
    z_max = cz + dz / 2 + z_margin

    pre_mask = (
        (points_xyz[:, 0] >= min_xy[0]) & (points_xyz[:, 0] <= max_xy[0]) &
        (points_xyz[:, 1] >= min_xy[1]) & (points_xyz[:, 1] <= max_xy[1]) &
        (points_xyz[:, 2] >= z_min) & (points_xyz[:, 2] <= z_max))
    if not np.any(pre_mask):
        return np.empty((0, 2), dtype=np.float64)

    cand = points_xyz[pre_mask]
    local_xy = (cand[:, :2] - np.array([cx, cy], dtype=np.float64)) @ rotation_matrix(yaw)
    inside = (
        (np.abs(local_xy[:, 0]) <= dx / 2) &
        (np.abs(local_xy[:, 1]) <= dy / 2))
    return local_xy[inside]


def projection_coverage(local_xy, box, args):
    canvas = args.canvas_size
    to_px, scale = make_transform(box, canvas, args.padding_m)
    _, _, _, dx, dy, _, _ = map(float, box[:7])
    target_rect = np.array(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2],
         [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
        dtype=np.float64)
    target_poly_px = np.round(to_px(target_rect)).astype(np.int32)

    box_mask = np.zeros((canvas, canvas), dtype=np.uint8)
    cv2.fillPoly(box_mask, [target_poly_px], 255)

    point_mask = np.zeros((canvas, canvas), dtype=np.uint8)
    if len(local_xy) > 0:
        pts_px = np.round(to_px(local_xy)).astype(np.int32)
        valid = (
            (pts_px[:, 0] >= 0) & (pts_px[:, 0] < canvas) &
            (pts_px[:, 1] >= 0) & (pts_px[:, 1] < canvas))
        pts_px = pts_px[valid]
        if len(pts_px) > 0:
            point_mask[pts_px[:, 1], pts_px[:, 0]] = 255
            point_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3))
            point_mask = cv2.dilate(point_mask, point_kernel, iterations=1)

    kernel_radius = max(1, int(round(args.morph_radius_m * scale)))
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    morph_mask = cv2.erode(cv2.dilate(point_mask, kernel, iterations=1),
                           kernel, iterations=1)

    box_area_px = max(1, int(np.count_nonzero(box_mask)))
    raw_coverage = (
        100.0 * np.count_nonzero((point_mask > 0) & (box_mask > 0)) /
        box_area_px)
    morph_coverage = (
        100.0 * np.count_nonzero((morph_mask > 0) & (box_mask > 0)) /
        box_area_px)
    return raw_coverage, morph_coverage, kernel_size, scale


def classify_instance(points_xyz, info, inst_idx, args):
    instances = info.get('instances', [])
    inst = instances[inst_idx]
    target_box = np.asarray(inst['bbox_3d'], dtype=np.float64)
    local_xy = box_local_points_prefilter(points_xyz, target_box, args.z_margin)
    raw_cov, morph_cov, kernel_size, scale = projection_coverage(
        local_xy, target_box, args)

    max_occluder_iou = 0.0
    max_any_iou = 0.0
    occluder_match_count = 0
    occluder_indices = []
    for other_idx, other in enumerate(instances):
        if other_idx == inst_idx or other.get('bbox_3d') is None:
            continue
        other_box = np.asarray(other['bbox_3d'], dtype=np.float64)
        iou = oriented_box_iou_xy(target_box, other_box)
        max_any_iou = max(max_any_iou, iou)
        if float(other_box[2]) < float(target_box[2]):
            max_occluder_iou = max(max_occluder_iou, iou)
            if iou * 100 >= args.occlusion_iou_thr:
                occluder_match_count += 1
                occluder_indices.append(str(other_idx))

    if morph_cov >= args.normal_thr:
        rule_class = 'normal'
    elif max_occluder_iou * 100 >= args.occlusion_iou_thr:
        rule_class = 'occluded'
    else:
        rule_class = 'reflective'

    return {
        'rule_class': rule_class,
        'num_points_in_box': len(local_xy),
        'raw_coverage_pct': f'{raw_cov:.4f}',
        'morph_coverage_pct': f'{morph_cov:.4f}',
        'max_occluder_iou_pct': f'{max_occluder_iou * 100:.4f}',
        'max_any_iou_pct': f'{max_any_iou * 100:.4f}',
        'occluder_match_count': occluder_match_count,
        'occluder_inst_indices': ' '.join(occluder_indices),
        'kernel_size_px': kernel_size,
        'pixel_scale_per_m': f'{scale:.4f}',
        'bbox_3d': ' '.join(f'{v:.6f}' for v in target_box.tolist()),
    }


def write_class_csvs(rows, out_dir):
    class_dir = Path(out_dir) / 'class_csv'
    class_dir.mkdir(parents=True, exist_ok=True)
    for cls in ['normal', 'occluded', 'reflective']:
        cls_rows = [r for r in rows if r['rule_class'] == cls]
        if not cls_rows:
            continue
        with (class_dir / f'{cls}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(cls_rows)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / 'random_projection_images'
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    items = load_infos(data_root, args.ann_files)
    records = gather_instances(items)
    if args.max_instances > 0:
        records = records[:args.max_instances]
    rng = random.Random(args.seed)
    save_count = min(args.save_random_images, len(records))
    selected_ids = set(rng.sample(range(len(records)), save_count))

    rows = []
    selected_records = []
    selected_rows_by_id = {}
    cached_sample = None
    cached_points_xyz = None
    cached_points_path = None

    for global_id, record in enumerate(records):
        split, local_idx, inst_idx, info, _ = record
        sample_idx = int(info.get('sample_idx', -1))
        current_sample = (split, sample_idx)
        if current_sample != cached_sample:
            points, points_path = load_points(data_root, args.points_dir, info)
            cached_points_xyz = points[:, :3].astype(np.float64, copy=False)
            cached_points_path = points_path
            cached_sample = current_sample

        cls_row = classify_instance(cached_points_xyz, info, inst_idx, args)
        cls_row.update({
            'global_instance_id': global_id,
            'split': split,
            'sample_idx': sample_idx,
            'local_info_idx': local_idx,
            'inst_idx': inst_idx,
            'points_path': str(cached_points_path),
        })
        rows.append(cls_row)

        if global_id in selected_ids:
            selected_records.append((global_id, record))
            selected_rows_by_id[global_id] = cls_row

        if (global_id + 1) % 1000 == 0 or global_id + 1 == len(records):
            counts = {}
            for r in rows:
                counts[r['rule_class']] = counts.get(r['rule_class'], 0) + 1
            print(f'classified {global_id + 1}/{len(records)} {counts}', flush=True)

    all_csv = out_dir / 'all_luggage_projection_quality.csv'
    fieldnames = [
        'global_instance_id', 'split', 'sample_idx', 'local_info_idx',
        'inst_idx', 'rule_class', 'num_points_in_box', 'raw_coverage_pct',
        'morph_coverage_pct', 'max_occluder_iou_pct', 'max_any_iou_pct',
        'occluder_match_count', 'occluder_inst_indices', 'kernel_size_px',
        'pixel_scale_per_m', 'bbox_3d', 'points_path'
    ]
    with all_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_class_csvs(rows, out_dir)

    image_rows = []
    rendered_for_sheets = []
    sheet_limit = max(0, min(args.contact_sheet_count, len(selected_records)))
    for image_order, (global_id, record) in enumerate(selected_records):
        base = selected_rows_by_id[global_id]
        split, _, inst_idx, info, _ = record
        sample_idx = int(info.get('sample_idx', -1))
        cls = base['rule_class']
        cls_dir = image_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        out_name = (
            f'{image_order:04d}_gid{global_id:05d}_{cls}_'
            f'{split}_{sample_idx:05d}_inst{inst_idx:03d}.png')
        row, img = render_case(record, data_root, args.points_dir,
                               cls_dir / out_name, args)
        row['global_instance_id'] = global_id
        row['image_order'] = image_order
        row['rule_class_from_full_pass'] = cls
        image_rows.append(row)
        if image_order < sheet_limit:
            rendered_for_sheets.append(img)
        if (image_order + 1) % 100 == 0 or image_order + 1 == len(selected_records):
            print(f'rendered {image_order + 1}/{len(selected_records)} images',
                  flush=True)

    image_csv = out_dir / 'random_projection_images.csv'
    if image_rows:
        keys = list(image_rows[0].keys())
        with image_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(image_rows)

    if rendered_for_sheets:
        sheet_dir = out_dir / 'contact_sheets_first_random_images'
        sheet_dir.mkdir(parents=True, exist_ok=True)
        make_contact_sheets(rendered_for_sheets, sheet_dir, args.sheet_cols,
                            args.sheet_rows)

    counts = {}
    for row in rows:
        counts[row['rule_class']] = counts.get(row['rule_class'], 0) + 1
    img_counts = {}
    for row in image_rows:
        cls = row['rule_class_from_full_pass']
        img_counts[cls] = img_counts.get(cls, 0) + 1

    summary_path = out_dir / 'summary.txt'
    with summary_path.open('w') as f:
        f.write(f'total_instances={len(rows)}\n')
        f.write(f'seed={args.seed}\n')
        f.write(f'saved_random_images={len(image_rows)}\n')
        f.write(f'normal_thr={args.normal_thr}\n')
        f.write(f'occlusion_iou_thr={args.occlusion_iou_thr}\n')
        f.write(f'morph_radius_m={args.morph_radius_m}\n')
        f.write('class_counts:\n')
        for cls in ['normal', 'occluded', 'reflective']:
            f.write(f'  {cls}={counts.get(cls, 0)}\n')
        f.write('random_image_counts:\n')
        for cls in ['normal', 'occluded', 'reflective']:
            f.write(f'  {cls}={img_counts.get(cls, 0)}\n')
        f.write(f'all_csv={all_csv}\n')
        f.write(f'image_csv={image_csv}\n')
        f.write(f'image_dir={image_dir}\n')

    print(f'Wrote {all_csv}')
    print(f'Wrote {image_csv}')
    print(f'Wrote images to {image_dir}')
    print(f'Class counts: {counts}')


if __name__ == '__main__':
    main()
