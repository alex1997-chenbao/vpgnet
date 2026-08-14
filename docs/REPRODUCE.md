# Reproduce

This repository provides two validation paths:

1. public SUN RGB-D training with the released code;
2. self-collected airport-luggage validation with the released data,
   checkpoint, and evaluation utilities.

## 1. Public SUN RGB-D Training Path

For public-dataset code validation, first prepare SUN RGB-D using the standard
MMDetection3D data-preparation pipeline. Then point a SUN RGB-D VPGNet config
to the generated `sunrgbd_infos_train.pkl` and `sunrgbd_infos_val.pkl` files.

Train with the common VPGNet training entry:

```bash
cd VPGNet-Airport-Luggage-OpenSource

CONFIG=/path/to/vpgnet_sunrgbd.py \
WORK_DIR=work_dirs/vpgnet_sunrgbd \
bash scripts/run_train_vpgnet.sh
```

If the SUN RGB-D config uses SAM3 visual priors, extract SAM3-FP1 features
before training:

```bash
DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
bash scripts/extract_sam3_features.sh
```

Evaluate a trained SUN RGB-D checkpoint with:

```bash
CONFIG=/path/to/vpgnet_sunrgbd.py \
CHECKPOINT=/path/to/checkpoint.pth \
WORK_DIR=work_dirs/vpgnet_sunrgbd_eval \
bash scripts/run_eval_vpgnet.sh
```

No SUN RGB-D checkpoint or SUN RGB-D pretrained model is released here.

## 2. Self-collected Airport-luggage Validation Path

### Prepare Data

Download the airport-luggage train/test dataset from Hugging Face:

```bash
cd VPGNet-Airport-Luggage-OpenSource
bash scripts/download_dataset.sh
```

This creates:

```text
data/sunrgbd/
  points/
  sunrgbd_trainval/calib/
  sunrgbd_trainval/image/
  sunrgbd_trainval/label/
  sunrgbd_trainval/sunrgbd_infos_train.pkl
  sunrgbd_trainval/sunrgbd_infos_val.pkl
  train_data_idx.txt
  val_data_idx.txt
```

Then extract SAM3-FP1 image features:

```bash
git clone https://github.com/facebookresearch/sam3.git ../sam3
pip install -e ../sam3
hf auth login
bash scripts/extract_sam3_features.sh
```

If you let the extractor download `sam3.pt` automatically, request access to
`facebook/sam3` on Hugging Face first.

The extractor writes:

```text
data/sunrgbd/sunrgbd_trainval/sam_f/*.pt
```

Use a local SAM3 checkpoint if you have one:

```bash
SAM_CHECKPOINT=/path/to/sam3.pt bash scripts/extract_sam3_features.sh
```

SAM3 feature extraction can be done in a separate SAM3 environment. After
`data/sunrgbd/sunrgbd_trainval/sam_f/*.pt` is generated, switch back to the
VPGNet/MMDetection3D environment for training and evaluation.

One-shot preparation and training:

```bash
bash scripts/prepare_and_train_vpgnet.sh
```

### Evaluate

Download the pretrained checkpoint:

```bash
bash scripts/download_checkpoint.sh
```

```bash
cd VPGNet-Airport-Luggage-OpenSource
bash scripts/run_eval_vpgnet.sh
```

The script runs:

```bash
python tools/test.py \
  configs/vpgnet/vpgnet_airport_luggage.py \
  checkpoints/best.pth \
  --work-dir work_dirs/vpgnet_airport_luggage_eval
```

### Train

```bash
cd VPGNet-Airport-Luggage-OpenSource
bash scripts/run_train_vpgnet.sh
```

The released config trains for 50 epochs with one luggage class, fixed height
configuration, 20000 sampled points, and the VPGNet fusion/refinement modules.

### Export JSONL Predictions

Use the checkpoint downloaded by `scripts/download_checkpoint.sh`.

```bash
bash scripts/export_vpgnet_predictions.sh
```

The output path is `work_dirs/vpgnet_predictions/val_predictions.jsonl`.

### Subset Evaluation Utilities

After exporting predictions and preparing a quality CSV:

```bash
python tools/misc/eval_luggage_subset_pr_gtfirst.py \
  --gt-ann-file data/sunrgbd/sunrgbd_trainval/sunrgbd_infos_val.pkl \
  --quality-csv work_dirs/luggage_quality/quality.csv \
  --pred-jsonl work_dirs/vpgnet_predictions/val_predictions.jsonl \
  --out-dir work_dirs/vpgnet_subset_pr
```

IoU and yaw/center statistics:

```bash
python tools/misc/compute_gt_first_match_iou_yaw_stats.py \
  --gt-ann-file data/sunrgbd/sunrgbd_trainval/sunrgbd_infos_val.pkl \
  --quality-csv work_dirs/luggage_quality/quality.csv \
  --pred-jsonl work_dirs/vpgnet_predictions/val_predictions.jsonl \
  --out-dir work_dirs/vpgnet_match_stats
```
