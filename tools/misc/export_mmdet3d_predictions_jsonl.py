#!/usr/bin/env python3
"""Export mmdet3d v1 test predictions to a compact JSONL file."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import import_modules_from_strings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmdet3d.registry import MODELS  # noqa: E402
from mmdet3d.utils import register_all_modules  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export mmdet3d v1 model predictions to JSONL.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
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


def patch_dataloader(cfg: Config, batch_size: int, num_workers: int) -> None:
    if 'test_dataloader' not in cfg:
        raise KeyError('config has no test_dataloader')
    dataloader = cfg.test_dataloader
    dataloader.batch_size = batch_size
    dataloader.num_workers = num_workers
    dataloader.persistent_workers = False
    if 'sampler' in dataloader:
        dataloader.sampler.shuffle = False
    if isinstance(dataloader.get('dataset', None), dict):
        dataloader.dataset.test_mode = True


def read_meta(data_sample: Any) -> Dict[str, Any]:
    meta = {}
    if hasattr(data_sample, 'metainfo') and isinstance(data_sample.metainfo, dict):
        meta.update(data_sample.metainfo)
    return meta


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return value
    return value


def serialize_prediction(data_sample: Any, dataset_index: int) -> Dict[str, Any]:
    meta = read_meta(data_sample)
    pred = getattr(data_sample, 'pred_instances_3d', None)
    if pred is None and hasattr(data_sample, 'get'):
        pred = data_sample.get('pred_instances_3d', None)
    if pred is None:
        raise RuntimeError('pred_instances_3d not found in model output')

    boxes = []
    scores = []
    labels = []
    if hasattr(pred, 'bboxes_3d'):
        boxes = pred.bboxes_3d.tensor.detach().cpu().tolist()
    if hasattr(pred, 'scores_3d'):
        scores = pred.scores_3d.detach().cpu().tolist()
    if hasattr(pred, 'labels_3d'):
        labels = pred.labels_3d.detach().cpu().tolist()

    sample_idx = scalar(meta.get('sample_idx'))
    if isinstance(sample_idx, str) and sample_idx.isdigit():
        sample_idx = int(sample_idx)

    return dict(
        dataset_index=int(dataset_index),
        sample_idx=sample_idx,
        lidar_path=str(meta.get('lidar_path')) if meta.get('lidar_path') else None,
        pts_path=str(meta.get('pts_path')) if meta.get('pts_path') else None,
        img_path=str(meta.get('img_path')) if meta.get('img_path') else None,
        num_boxes=len(scores),
        boxes_3d=boxes,
        scores_3d=scores,
        labels_3d=labels,
    )


def build_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)
    patch_dataloader(cfg, args.batch_size, args.num_workers)
    cfg.model.train_cfg = None
    cfg.load_from = None
    cfg.launcher = 'none'
    return cfg


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = build_cfg(args)

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    out_path = Path(args.out_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(dataloader.dataset)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if args.max_samples > 0 and len(rows) >= args.max_samples:
                break
            outputs = model.test_step(data_batch)
            for offset, output in enumerate(outputs):
                if args.max_samples > 0 and len(rows) >= args.max_samples:
                    break
                dataset_index = batch_idx * args.batch_size + offset
                rows.append(serialize_prediction(output, dataset_index))
            if len(rows) % args.log_interval == 0 or len(rows) == total:
                print(f'processed {len(rows)} / {total}', flush=True)

    with out_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + '\n')

    print(json.dumps({
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'out_jsonl': str(out_path),
        'num_samples': len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
