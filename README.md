# VPGNet Airport Luggage Detection

Clean release package for VPGNet, a single-class top-view 3D detector for
airport luggage scenes. This package keeps the modified MMDetection3D source,
the VPGNet config, evaluation utilities, and metadata for the best checkpoint
used in the current experiments.

## Included

- VPGNet source code in `mmdet3d/`
  - detector integration: `mmdet3d/models/detectors/imvotenet.py`
  - GCA and feature fusion modules: `mmdet3d/models/my_module/`
  - luggage dataset and SAM feature loading: `mmdet3d/datasets/`
- Best config: `configs/vpgnet/vpgnet_airport_luggage.py`
- Best checkpoint metadata: `checkpoints/vpgnet_epoch31_best.json`
- Core tools:
  - `tools/train.py`
  - `tools/test.py`
  - `tools/misc/export_mmdet3d_predictions_jsonl.py`
  - `tools/misc/eval_luggage_subsets_ap.py`
  - `tools/misc/eval_luggage_subset_pr_gtfirst.py`
  - `tools/misc/compute_gt_first_match_iou_yaw_stats.py`
  - `tools/misc/postprocess_vpgnet_with_sam3_instance_masks.py`
  - `tools/misc/visualize_vpgnet_predictions_o3d.py`

Dataset files are not included in this package.

## Quick Start

Install this package in the same style as MMDetection3D:

```bash
cd VPGNet-Airport-Luggage-OpenSource
pip install -v -e .
```

Before evaluation, download or copy the checkpoint release asset to:

```text
checkpoints/vpgnet_epoch31_best.pth
```

Run evaluation:

```bash
bash scripts/run_eval_vpgnet.sh
```

Train from the released config:

```bash
bash scripts/run_train_vpgnet.sh
```

Export predictions to JSONL:

```bash
bash scripts/export_vpgnet_predictions.sh
```

## Model

The best checkpoint is selected by validation AP@0.50:

| Model | Epoch | AP@0.25 | AP@0.50 | Checkpoint |
|---|---:|---:|---:|---|
| VPGNet | 31 | 0.9803 | 0.9737 | `checkpoints/vpgnet_epoch31_best.pth` |

The checkpoint is about 203 MiB. It is intentionally not committed to git
because GitHub blocks ordinary git blobs larger than 100 MiB. Upload it as a
GitHub Release asset or publish it with Git LFS.

## Data Layout

By default the config expects:

```text
data/sunrgbd/
  points/
  sunrgbd_trainval/
    image/
    sam_f/
    sunrgbd_infos_train.pkl
    sunrgbd_infos_val.pkl
```

The SAM feature loader reads `.pt` feature files. If `sam_feat_prefix` is not
specified, the dataset maps each image path from `image/*.jpg` to
`sam_f/*.pt`.

More notes are in `docs/DATASET.md`, `docs/MODEL_ZOO.md`, and
`docs/REPRODUCE.md`.
