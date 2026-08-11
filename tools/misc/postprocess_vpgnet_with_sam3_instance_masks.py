#!/usr/bin/env python3
"""Filter 3D luggage detections with SAM3 instance-mask point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import types
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[2]
SAM3_ROOT = ROOT / 'sam3'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from mmdet3d.evaluation.functional.indoor_eval import indoor_eval  # noqa: E402
from mmdet3d.structures import DepthInstance3DBoxes, get_box_type  # noqa: E402


SUBSETS = [
    ('normal', {'normal'}),
    ('reflective', {'reflective'}),
    ('occluded', {'occluded'}),
    ('reflective_or_occluded', {'reflective', 'occluded'}),
]
IOU_THR_LIST = (0.25, 0.50)


def install_torch_attention_compat() -> None:
    """Expose torch.nn.attention for SAM3 on older PyTorch builds."""
    if hasattr(torch, 'compiler') and not hasattr(torch.compiler, 'is_dynamo_compiling'):
        torch.compiler.is_dynamo_compiling = lambda: False

    if 'torch.nn.attention' in sys.modules:
        return
    if hasattr(torch.nn, 'attention'):
        return
    if not hasattr(torch.backends, 'cuda'):
        return
    if not hasattr(torch.backends.cuda, 'sdp_kernel'):
        return
    if not hasattr(torch.backends.cuda, 'SDPBackend'):
        return

    def sdpa_kernel(backends):
        if not torch.cuda.is_available():
            return nullcontext()
        if not isinstance(backends, (list, tuple, set)):
            backends = [backends]
        names = {
            getattr(backend, 'name', str(backend).split('.')[-1])
            for backend in backends
        }
        enable_flash = 'FLASH_ATTENTION' in names
        enable_math = 'MATH' in names
        enable_mem_efficient = (
            'EFFICIENT_ATTENTION' in names or
            'MEM_EFFICIENT_ATTENTION' in names or
            'MEM_EFFICIENT' in names)
        return torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient)

    module = types.ModuleType('torch.nn.attention')
    module.sdpa_kernel = sdpa_kernel
    module.SDPBackend = torch.backends.cuda.SDPBackend
    sys.modules['torch.nn.attention'] = module


def parse_args() -> argparse.Namespace:
    default_quality = (
        ROOT /
        'work_dirs/luggage_projection_quality_real120_occlusion_priority_iou30_cov85_20260711_220025'
        '/all_luggage_projection_quality_occlusion_priority_iou30_cov85.csv')
    default_pred = (
        ROOT /
        'work_dirs/real_sunrgbd_merged_120_640x480_vpgnet_numpts300_epoch31_20260711_204821'
        '/val_predictions_epoch_31_bs10_score005_numpts_gt500_plus03.jsonl')
    default_data_root = ROOT / 'data/sunrgbd_merged_120_640x480_standard'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(
        description='Generate SAM3 suitcase instance masks, keep one 3D box per mask instance, and evaluate.')
    parser.add_argument('--data-root', type=Path, default=default_data_root)
    parser.add_argument(
        '--ann-file',
        type=Path,
        default=default_data_root / 'sunrgbd_trainval/sunrgbd_infos_val.pkl')
    parser.add_argument('--quality-csv', type=Path, default=default_quality)
    parser.add_argument('--pred-jsonl', type=Path, default=default_pred)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=ROOT / 'work_dirs' / f'vpgnet_sam3_instance_mask_postprocess_{stamp}')
    parser.add_argument(
        '--mask-dir',
        type=Path,
        default=default_data_root / 'sunrgbd_trainval/instance_masks_sam3_suitcase_score06')
    parser.add_argument('--sam-checkpoint', type=Path, default=SAM3_ROOT / 'sam3.pt')
    parser.add_argument('--prompt', default='suitcase')
    parser.add_argument('--sam-score-thr', type=float, default=0.6)
    parser.add_argument('--generate-masks', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite-masks', action='store_true')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--min-instance-points', type=int, default=1)
    parser.add_argument('--min-match-points', type=int, default=1)
    parser.add_argument('--depth-eps', type=float, default=1e-4)
    parser.add_argument('--mask-background-id', type=int, default=0)
    parser.add_argument('--report-every', type=int, default=20)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open('rb') as f:
        return pickle.load(f)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


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


def load_infos(ann_file: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    data = load_pickle(ann_file)
    if not isinstance(data, dict) or 'data_list' not in data:
        raise TypeError(f'Expected mmdet3d v1 info pkl with data_list: {ann_file}')
    items = data['data_list']
    by_sample = {}
    for dataset_index, info in enumerate(items):
        sample_idx = to_int(info.get('sample_idx'), dataset_index)
        if sample_idx is None:
            raise ValueError(f'Cannot infer sample_idx for dataset index {dataset_index}')
        by_sample[int(sample_idx)] = info
    return items, by_sample


def image_path_for_info(data_root: Path, info: Dict[str, Any]) -> Path:
    img_path = None
    images = info.get('images') or {}
    if isinstance(images, dict):
        cam = images.get('CAM0') or next(iter(images.values()), {})
        img_path = cam.get('img_path') if isinstance(cam, dict) else None
    if img_path is None:
        img_path = f'{int(info["sample_idx"]):05d}.jpg'
    path = Path(str(img_path))
    if path.is_absolute():
        return path
    return data_root / 'sunrgbd_trainval' / 'image' / path


def point_path_for_info(data_root: Path, info: Dict[str, Any]) -> Path:
    rel_path = info.get('lidar_points', {}).get('lidar_path')
    if rel_path is None:
        rel_path = f'{int(info["sample_idx"]):05d}.bin'
    path = Path(str(rel_path))
    if path.is_absolute():
        return path
    return data_root / 'points' / path


def depth2img_for_info(info: Dict[str, Any]) -> np.ndarray:
    images = info.get('images') or {}
    cam = images.get('CAM0') if isinstance(images, dict) else None
    if cam is None and isinstance(images, dict) and images:
        cam = next(iter(images.values()))
    if isinstance(cam, dict) and 'depth2img' in cam:
        return np.asarray(cam['depth2img'], dtype=np.float64).reshape(3, 3)
    raise KeyError(f'No images.CAM0.depth2img for sample {info.get("sample_idx")}')


def load_points(path: Path, expected_dim: Optional[int] = None) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, expected_dim or 3), dtype=np.float32)
    candidates = []
    if expected_dim:
        candidates.append(int(expected_dim))
    candidates.extend([6, 4, 3])
    seen = set()
    for dim in candidates:
        if dim in seen:
            continue
        seen.add(dim)
        if dim > 0 and arr.size % dim == 0:
            return arr.reshape(-1, dim)
    raise ValueError(f'Cannot infer point dimension for {path}, values={arr.size}')


def project_points_to_mask_ids(
    points_xyz: np.ndarray,
    depth2img: np.ndarray,
    instance_map: np.ndarray,
    depth_eps: float,
    background_id: int,
) -> np.ndarray:
    ids = np.full((points_xyz.shape[0], ), int(background_id), dtype=np.int64)
    if points_xyz.size == 0:
        return ids
    depth = points_xyz[:, 2]
    valid = np.isfinite(points_xyz).all(axis=1) & (depth > depth_eps)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return ids
    pts = points_xyz[valid_idx]
    proj = pts @ depth2img.T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], depth_eps)
    height, width = instance_map.shape[:2]
    px = np.rint(uv[:, 0]).astype(np.int64)
    py = np.rint(uv[:, 1]).astype(np.int64)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if np.any(inside):
        ids[valid_idx[inside]] = instance_map[py[inside], px[inside]].astype(np.int64)
    return ids


def load_instance_map(path: Path) -> np.ndarray:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    if torch.is_tensor(payload):
        arr = payload.detach().cpu().numpy()
    elif isinstance(payload, np.ndarray):
        arr = payload
    else:
        raise TypeError(f'Unsupported mask payload {type(payload)!r}: {path}')
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D instance map, got {arr.shape}: {path}')
    return arr.astype(np.int64, copy=False)


def build_sam3_processor(checkpoint: Path):
    install_torch_attention_compat()
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = build_sam3_image_model(checkpoint_path=str(checkpoint))
    model.to(device)
    model.eval()
    return Sam3Processor(model), device


def mask_tensor_to_2d(mask: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    mask = mask.detach().float()
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f'Unsupported SAM3 mask shape: {tuple(mask.shape)}')
    mask = torch.nn.functional.interpolate(
        mask[None, None], size=out_hw, mode='bilinear', align_corners=False)
    return mask[0, 0]


def generate_missing_masks(
    data_root: Path,
    infos: Sequence[Dict[str, Any]],
    mask_dir: Path,
    checkpoint: Path,
    prompt: str,
    score_thr: float,
    overwrite: bool,
    report_every: int,
) -> Dict[str, Any]:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    mask_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for info in infos:
        sample_idx = int(info.get('sample_idx'))
        out_path = mask_dir / f'{sample_idx:05d}.pt'
        if overwrite or not out_path.exists():
            missing.append((sample_idx, info, out_path))
    summary = {
        'mask_dir': str(mask_dir),
        'prompt': prompt,
        'sam_score_thr': float(score_thr),
        'num_expected': len(infos),
        'num_to_generate': len(missing),
        'generated': 0,
        'skipped_existing': len(infos) - len(missing),
        'failures': [],
    }
    if not missing:
        return summary

    processor, device = build_sam3_processor(checkpoint)
    for idx, (sample_idx, info, out_path) in enumerate(missing, start=1):
        try:
            image = Image.open(image_path_for_info(data_root, info)).convert('RGB')
            width, height = image.size
            with torch.no_grad():
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output.get('masks')
            scores = output.get('scores')
            if masks is None or scores is None:
                raise KeyError('SAM3 output has no masks/scores')
            scores_cpu = scores.detach().float().cpu()
            keep = torch.nonzero(scores_cpu >= float(score_thr), as_tuple=False).flatten()
            instance_map = np.zeros((height, width), dtype=np.int16)
            if keep.numel() > 0:
                keep_list = keep.tolist()
                keep_list.sort(key=lambda i: float(scores_cpu[i]))
                for inst_id, mask_idx in enumerate(keep_list, start=1):
                    mask_2d = mask_tensor_to_2d(masks[mask_idx].to(device), (height, width))
                    instance_map[(mask_2d > 0.5).detach().cpu().numpy()] = int(inst_id)
            torch.save(torch.from_numpy(instance_map).contiguous(), out_path)
            summary['generated'] += 1
        except Exception as exc:
            summary['failures'].append({'sample_idx': sample_idx, 'error': repr(exc)})
            print(f'[mask fail] sample={sample_idx:05d} error={exc}', flush=True)
        if idx % max(1, report_every) == 0 or idx == len(missing):
            print(f'generated masks {idx}/{len(missing)}', flush=True)
    return summary


def boxes_from_row(row: Dict[str, Any]) -> Tuple[List[List[float]], List[float], List[int], List[int]]:
    boxes = [normalize_box(box) for box in row.get('boxes_3d', [])]
    scores = [float(v) for v in row.get('scores_3d', [])]
    labels = [int(v) for v in row.get('labels_3d', [])]
    keep = [i for i, box in enumerate(boxes) if box is not None]
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep], keep


def filter_row_lists(row: Dict[str, Any], valid_indices: List[int], selected_valid: Iterable[int]) -> Dict[str, Any]:
    selected_valid = sorted(set(int(i) for i in selected_valid))
    selected_orig = [valid_indices[i] for i in selected_valid]
    selected_orig_set = set(selected_orig)
    out = dict(row)
    original_len = len(row.get('boxes_3d', []))
    for key, value in row.items():
        if isinstance(value, list) and len(value) == original_len:
            out[key] = [value[i] for i in selected_orig if i < len(value)]
    out['num_boxes'] = len(out.get('boxes_3d', []))
    out['sam3_instance_postprocess'] = {
        'selected_original_indices': selected_orig,
        'dropped_original_indices': [i for i in range(original_len) if i not in selected_orig_set],
    }
    return out


def points_in_box_mask(points_xyz: np.ndarray, box: Sequence[float]) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    rel_x = points_xyz[:, 0] - cx
    rel_y = points_xyz[:, 1] - cy
    rel_z = points_xyz[:, 2] - cz
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    local_x = rel_x * c - rel_y * s
    local_y = rel_x * s + rel_y * c
    return (
        (np.abs(local_x) <= dx * 0.5) &
        (np.abs(local_y) <= dy * 0.5) &
        (np.abs(rel_z) <= dz * 0.5))


def select_boxes_by_instance_points(
    points_xyz: np.ndarray,
    point_instance_ids: np.ndarray,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    background_id: int,
    min_instance_points: int,
    min_match_points: int,
) -> Tuple[List[int], Dict[str, Any]]:
    valid_instance_ids = point_instance_ids[point_instance_ids != int(background_id)]
    if valid_instance_ids.size == 0 or not boxes:
        return [], {
            'num_sam_instances': 0,
            'num_sam_instances_with_enough_points': 0,
            'num_instances_matched': 0,
            'selected_detection_indices': [],
            'match_points_by_detection': {},
        }

    unique_instance_ids = np.unique(valid_instance_ids.astype(np.int64))
    instance_point_counts = np.bincount(valid_instance_ids.astype(np.int64))
    enough_instances = {
        int(i) for i, count in enumerate(instance_point_counts)
        if i != int(background_id) and int(count) >= int(min_instance_points)
    }
    best_by_inst: Dict[int, Tuple[int, int, float]] = {}
    for det_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if int(label) != 0:
            continue
        inside = points_in_box_mask(points_xyz, box)
        ids = point_instance_ids[inside]
        ids = ids[ids != int(background_id)]
        if ids.size == 0:
            continue
        counts = np.bincount(ids.astype(np.int64))
        for inst_id in np.flatnonzero(counts):
            inst_id = int(inst_id)
            count = int(counts[inst_id])
            if inst_id not in enough_instances or count < int(min_match_points):
                continue
            old = best_by_inst.get(inst_id)
            if old is None or count > old[1] or (count == old[1] and float(score) > old[2]):
                best_by_inst[inst_id] = (int(det_idx), count, float(score))

    selected = sorted({item[0] for item in best_by_inst.values()})
    match_points_by_detection: Dict[int, int] = defaultdict(int)
    for det_idx, count, _score in best_by_inst.values():
        match_points_by_detection[int(det_idx)] += int(count)
    return selected, {
        'num_sam_instances': int(len(unique_instance_ids)),
        'num_sam_instances_with_enough_points': int(len(enough_instances)),
        'num_instances_matched': int(len(best_by_inst)),
        'selected_detection_indices': selected,
        'match_points_by_detection': {str(k): int(v) for k, v in sorted(match_points_by_detection.items())},
    }


def row_sample_idx(row: Dict[str, Any]) -> Optional[int]:
    for key in ('sample_idx', 'sample_index', 'lidar_idx', 'dataset_index'):
        value = to_int(row.get(key))
        if value is not None:
            return value
    return None


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row_sample_idx(row) is None:
                raise ValueError(f'Cannot infer sample_idx at {path}:{line_idx}')
            rows.append(row)
    return rows


def postprocess_predictions(args: argparse.Namespace,
                            infos_by_sample: Dict[int, Dict[str, Any]]) -> Tuple[Path, Dict[str, Any]]:
    rows = load_prediction_rows(args.pred_jsonl)
    out_jsonl = args.out_dir / 'val_predictions_sam3_instance_mask_keep_max_points.jsonl'
    out_rows = []
    summary = {
        'pred_jsonl': str(args.pred_jsonl),
        'mask_dir': str(args.mask_dir),
        'score_thr': float(args.score_thr) if args.score_thr is not None else None,
        'min_instance_points': int(args.min_instance_points),
        'min_match_points': int(args.min_match_points),
        'num_samples': len(rows),
        'num_predictions_before_score_thr': 0,
        'num_predictions_after_score_thr': 0,
        'num_predictions_after_sam3_filter': 0,
        'num_sam_instances': 0,
        'num_sam_instances_with_enough_points': 0,
        'num_sam_instances_matched': 0,
        'samples_missing_info': [],
        'samples_missing_mask': [],
    }
    for row_idx, row in enumerate(rows, start=1):
        sample_idx = int(row_sample_idx(row))
        info = infos_by_sample.get(sample_idx)
        if info is None:
            summary['samples_missing_info'].append(sample_idx)
            out_rows.append(filter_row_lists(row, [], []))
            continue
        boxes_all, scores_all, labels_all, valid_indices = boxes_from_row(row)
        score_keep = [
            i for i, (score, label) in enumerate(zip(scores_all, labels_all))
            if int(label) == 0 and (args.score_thr is None or float(score) >= float(args.score_thr))
        ]
        boxes = [boxes_all[i] for i in score_keep]
        scores = [scores_all[i] for i in score_keep]
        labels = [labels_all[i] for i in score_keep]
        valid_after_score = [valid_indices[i] for i in score_keep]
        summary['num_predictions_before_score_thr'] += len(boxes_all)
        summary['num_predictions_after_score_thr'] += len(boxes)

        mask_path = args.mask_dir / f'{sample_idx:05d}.pt'
        if not mask_path.exists():
            summary['samples_missing_mask'].append(sample_idx)
            selected_after_score = []
            match_summary = {
                'num_sam_instances': 0,
                'num_sam_instances_with_enough_points': 0,
                'num_instances_matched': 0,
                'selected_detection_indices': [],
                'match_points_by_detection': {},
            }
        else:
            points = load_points(
                point_path_for_info(args.data_root, info),
                expected_dim=info.get('lidar_points', {}).get('num_pts_feats'))
            points_xyz = points[:, :3].astype(np.float64, copy=False)
            instance_map = load_instance_map(mask_path)
            point_instance_ids = project_points_to_mask_ids(
                points_xyz,
                depth2img_for_info(info),
                instance_map,
                depth_eps=float(args.depth_eps),
                background_id=int(args.mask_background_id))
            selected_after_score, match_summary = select_boxes_by_instance_points(
                points_xyz,
                point_instance_ids,
                boxes,
                scores,
                labels,
                background_id=int(args.mask_background_id),
                min_instance_points=int(args.min_instance_points),
                min_match_points=int(args.min_match_points))

        selected_valid_indices = [score_keep[i] for i in selected_after_score]
        out_row = filter_row_lists(row, valid_indices, selected_valid_indices)
        out_row['sam3_instance_postprocess'].update(match_summary)
        out_rows.append(out_row)
        summary['num_predictions_after_sam3_filter'] += len(out_row.get('boxes_3d', []))
        summary['num_sam_instances'] += int(match_summary['num_sam_instances'])
        summary['num_sam_instances_with_enough_points'] += int(
            match_summary['num_sam_instances_with_enough_points'])
        summary['num_sam_instances_matched'] += int(match_summary['num_instances_matched'])
        if row_idx % max(1, args.report_every) == 0 or row_idx == len(rows):
            print(
                f'postprocessed {row_idx}/{len(rows)} '
                f'kept={summary["num_predictions_after_sam3_filter"]}/'
                f'{summary["num_predictions_after_score_thr"]}',
                flush=True)

    with out_jsonl.open('w', encoding='utf-8') as f:
        for row in out_rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + '\n')
    return out_jsonl, summary


def load_quality_map(path: Path) -> Dict[Tuple[int, int], str]:
    quality = {}
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('split') and row.get('split') != 'val':
                continue
            sample_idx = to_int(row.get('sample_idx'))
            inst_idx = to_int(row.get('inst_idx'))
            rule_class = (row.get('rule_class') or 'unknown').strip()
            if sample_idx is None or inst_idx is None:
                continue
            quality[(int(sample_idx), int(inst_idx))] = rule_class
    return quality


def build_gt_annos_and_subset_gt(
    infos: Sequence[Dict[str, Any]],
    quality: Dict[Tuple[int, int], str],
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    gt_annos = []
    subset_gt = {}
    for dataset_index, info in enumerate(infos):
        sample_idx = int(to_int(info.get('sample_idx'), dataset_index))
        boxes = []
        labels = []
        rules = []
        for inst_idx, inst in enumerate(info.get('instances', [])):
            box = normalize_box(inst.get('bbox_3d', []))
            if box is None:
                continue
            label = int(to_int(inst.get('bbox_label_3d', inst.get('bbox_label', 0)), 0))
            boxes.append(box)
            labels.append(label)
            rules.append(quality.get((sample_idx, inst_idx), 'unknown'))
        gt_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
        if len(gt_np) > 0:
            gt_boxes = DepthInstance3DBoxes(
                torch.as_tensor(gt_np, dtype=torch.float32),
                box_dim=7,
                with_yaw=True,
                origin=(0.5, 0.5, 0.5))
        else:
            gt_boxes = DepthInstance3DBoxes(
                torch.empty((0, 7), dtype=torch.float32),
                box_dim=7,
                with_yaw=True)
        labels_np = np.asarray(labels, dtype=np.int64)
        gt_annos.append({
            'gt_bboxes_3d': gt_boxes,
            'gt_labels_3d': labels_np,
        })
        subset_gt[sample_idx] = {
            'boxes': gt_boxes.tensor.detach().cpu().numpy(),
            'labels': labels_np,
            'rules': rules,
        }
    return gt_annos, subset_gt


def predictions_from_jsonl(path: Path,
                           score_thr: Optional[float] = None
                           ) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    rows = load_prediction_rows(path)
    pred_by_sample = defaultdict(list)
    for row in rows:
        sample_idx = int(row_sample_idx(row))
        boxes, scores, labels, _valid = boxes_from_row(row)
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != 0:
                continue
            if score_thr is not None and float(score) < float(score_thr):
                continue
            pred_by_sample[sample_idx].append({
                'sample_idx': sample_idx,
                'box': box,
                'score': float(score),
                'label': int(label),
            })
    return rows, dict(pred_by_sample)


def build_dt_annos(infos: Sequence[Dict[str, Any]],
                   pred_by_sample: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    dt_annos = []
    for dataset_index, info in enumerate(infos):
        sample_idx = int(to_int(info.get('sample_idx'), dataset_index))
        preds = pred_by_sample.get(sample_idx, [])
        boxes_np = np.asarray([p['box'] for p in preds], dtype=np.float32).reshape(-1, 7)
        if len(boxes_np) > 0:
            boxes_3d = DepthInstance3DBoxes(
                torch.as_tensor(boxes_np, dtype=torch.float32),
                box_dim=7,
                with_yaw=True)
        else:
            boxes_3d = DepthInstance3DBoxes(
                torch.empty((0, 7), dtype=torch.float32),
                box_dim=7,
                with_yaw=True)
        dt_annos.append({
            'bboxes_3d': boxes_3d,
            'scores_3d': torch.as_tensor([p['score'] for p in preds], dtype=torch.float32),
            'labels_3d': torch.as_tensor([p['label'] for p in preds], dtype=torch.long),
        })
    return dt_annos


def evaluate_overall(
    infos: Sequence[Dict[str, Any]],
    gt_annos: List[Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, float]:
    dt_annos = build_dt_annos(infos, pred_by_sample)
    _box_type_3d, box_mode_3d = get_box_type('depth')
    return indoor_eval(
        gt_annos,
        dt_annos,
        list(IOU_THR_LIST),
        {0: 'luggage'},
        logger='current',
        box_mode_3d=box_mode_3d)


def build_iou_matrices(gt_by_sample: Dict[int, Dict[str, Any]],
                       pred_by_sample: Dict[int, List[Dict[str, Any]]]) -> Dict[int, np.ndarray]:
    matrices = {}
    for sample_idx, preds in pred_by_sample.items():
        gt = gt_by_sample.get(sample_idx)
        if gt is None or len(preds) == 0 or gt['boxes'].shape[0] == 0:
            matrices[sample_idx] = np.zeros((len(preds), 0), dtype=np.float32)
            continue
        pred_tensor = torch.as_tensor([pred['box'] for pred in preds], dtype=torch.float32)
        gt_tensor = torch.as_tensor(gt['boxes'], dtype=torch.float32)
        pred_boxes = DepthInstance3DBoxes(pred_tensor, box_dim=7, with_yaw=True)
        gt_boxes = DepthInstance3DBoxes(gt_tensor, box_dim=7, with_yaw=True)
        with torch.no_grad():
            ious = DepthInstance3DBoxes.overlaps(pred_boxes, gt_boxes).detach().cpu().numpy()
        matrices[sample_idx] = ious.astype(np.float32, copy=False)
    return matrices


def count_target_gt(gt_by_sample: Dict[int, Dict[str, Any]], target_rules: Iterable[str]) -> int:
    target_rules = set(target_rules)
    total = 0
    for gt in gt_by_sample.values():
        for label, rule in zip(gt['labels'], gt['rules']):
            if int(label) == 0 and rule in target_rules:
                total += 1
    return total


def evaluate_subset_one_thr(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
    iou_matrices: Dict[int, np.ndarray],
    detections: List[Tuple[int, int, float]],
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

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / num_gt) if num_gt > 0 else math.nan
    return {
        'num_gt': int(num_gt),
        'tp': int(tp),
        'fp': int(fp),
        'ignored_predictions_best_match_other_subset': int(ignored),
        'counted_predictions': int(tp + fp),
        'precision': precision,
        'recall': recall,
    }


def evaluate_subsets(
    gt_by_sample: Dict[int, Dict[str, Any]],
    pred_by_sample: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    iou_matrices = build_iou_matrices(gt_by_sample, pred_by_sample)
    detections = []
    for sample_idx, preds in pred_by_sample.items():
        for pred_idx, pred in enumerate(preds):
            detections.append((sample_idx, pred_idx, float(pred['score'])))
    detections.sort(key=lambda x: x[2], reverse=True)

    out = {}
    for subset_name, target_rules in SUBSETS:
        subset_result = {}
        for iou_thr in IOU_THR_LIST:
            subset_result[f'iou_{iou_thr:.2f}'] = evaluate_subset_one_thr(
                gt_by_sample, pred_by_sample, iou_matrices,
                detections, target_rules, iou_thr)
        out[subset_name] = subset_result
    return out


def write_subset_csv(path: Path, subset_metrics: Dict[str, Any]) -> None:
    fieldnames = [
        'subset', 'iou_thr', 'num_gt', 'tp', 'fp',
        'ignored_predictions_best_match_other_subset', 'counted_predictions',
        'precision', 'recall',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset, by_thr in subset_metrics.items():
            for thr_key, values in by_thr.items():
                row = {'subset': subset, 'iou_thr': thr_key.replace('iou_', '')}
                row.update(values)
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    infos, infos_by_sample = load_infos(args.ann_file)
    mask_summary = {}
    if args.generate_masks:
        mask_summary = generate_missing_masks(
            args.data_root,
            infos,
            args.mask_dir,
            args.sam_checkpoint,
            args.prompt,
            args.sam_score_thr,
            args.overwrite_masks,
            args.report_every)
    pred_out, post_summary = postprocess_predictions(args, infos_by_sample)

    quality = load_quality_map(args.quality_csv)
    gt_annos, gt_by_sample = build_gt_annos_and_subset_gt(infos, quality)
    _rows, pred_by_sample = predictions_from_jsonl(pred_out, score_thr=None)
    overall = evaluate_overall(infos, gt_annos, pred_by_sample)
    subset = evaluate_subsets(gt_by_sample, pred_by_sample)

    result = {
        'mask_generation': mask_summary,
        'postprocess': post_summary,
        'overall': overall,
        'subset': subset,
        'outputs': {
            'postprocessed_jsonl': str(pred_out),
            'summary_json': str(args.out_dir / 'sam3_instance_mask_postprocess_metrics.json'),
            'subset_csv': str(args.out_dir / 'sam3_instance_mask_postprocess_subset_precision_recall.csv'),
        },
    }
    summary_path = args.out_dir / 'sam3_instance_mask_postprocess_metrics.json'
    subset_csv = args.out_dir / 'sam3_instance_mask_postprocess_subset_precision_recall.csv'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(to_builtin(result), f, ensure_ascii=False, indent=2)
    write_subset_csv(subset_csv, subset)

    print(json.dumps(to_builtin(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
