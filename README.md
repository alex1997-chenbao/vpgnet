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
- Best checkpoint metadata: `checkpoints/vpgnet_sam3fp1_best_epoch28.json`
- Core tools:
  - `tools/train.py`
  - `tools/test.py`
  - `tools/misc/export_mmdet3d_predictions_jsonl.py`
  - `tools/misc/eval_luggage_subsets_ap.py`
  - `tools/misc/eval_luggage_subset_pr_gtfirst.py`
  - `tools/misc/compute_gt_first_match_iou_yaw_stats.py`
  - `tools/misc/postprocess_vpgnet_with_sam3_instance_masks.py`
  - `tools/misc/visualize_vpgnet_predictions_o3d.py`

The public training and test dataset is available on Hugging Face:

```text
https://huggingface.co/datasets/alex-chenbao1997/vpgnet-airport-luggage
```

## Quick Start

Install this package in the same style as MMDetection3D:

```bash
cd VPGNet-Airport-Luggage-OpenSource
pip install -v -e .
```

SAM3 feature extraction may be run in a separate SAM3 environment. The saved
`.pt` features are environment-independent and are then consumed by the VPGNet
training environment.

Download the released train/test dataset:

```bash
bash scripts/download_dataset.sh
```

Prepare the SAM3-FP1 visual prior used by the released model:

```bash
git clone https://github.com/facebookresearch/sam3.git ../sam3
pip install -e ../sam3
hf auth login  # required by facebook/sam3 if you do not provide sam3.pt locally
bash scripts/extract_sam3_features.sh
```

Before downloading the SAM3 checkpoint from Hugging Face, request access to
`facebook/sam3` on the Hugging Face model page.

The extractor writes features to:

```text
data/sunrgbd/sunrgbd_trainval/sam_f/
```

Run the whole data-preparation and training sequence:

```bash
bash scripts/prepare_and_train_vpgnet.sh
```

Download the pretrained checkpoint:

```bash
bash scripts/download_checkpoint.sh
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

The released VPGNet setting uses the SAM3-FP1 visual prior. The reported
airport-luggage test metrics are:

| Model | Visual prior | AP@0.25 | AP@0.50 | Checkpoint |
|---|---|---:|---:|---|
| VPGNet | SAM3-FP1 | 0.9869 | 0.9805 | [Hugging Face](https://huggingface.co/alex-chenbao1997/vpgnet-airport-luggage/blob/main/vpgnet_sam3fp1_best_epoch28.pth) |

## Data Layout

By default the config expects:

```text
data/sunrgbd/
  points/
  sunrgbd_trainval/
    calib/
    image/
    label/
    sam_f/
    sunrgbd_infos_train.pkl
    sunrgbd_infos_val.pkl
  train_data_idx.txt
  val_data_idx.txt
```

The SAM feature loader reads `.pt` feature files. If `sam_feat_prefix` is not
specified, the dataset maps each image path from `image/*.jpg` to
`sam_f/*.pt`. The released VPGNet setting uses SAM3-FP1 features saved as
`float16` tensors with shape `(256, 144, 144)`.

More notes are in `docs/DATASET.md`, `docs/MODEL_ZOO.md`, and
`docs/REPRODUCE.md`.
