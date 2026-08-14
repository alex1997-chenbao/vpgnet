# Reproduce

This repository provides two validation paths:

1. public SUN RGB-D training with the released code;
2. self-collected airport-luggage validation with the released data,
   checkpoint, and evaluation utilities.

## 1. Public SUN RGB-D Training Path

For public-dataset code validation, first prepare SUN RGB-D using the standard
MMDetection3D data-preparation pipeline. A more detailed standalone version is
provided in `docs/SUNRGBD.md`.

Download the official SUN RGB-D files from:

```text
http://rgbd.cs.princeton.edu/data/
```

Place these files under `data/sunrgbd/OFFICIAL_SUNRGBD/` in an official
MMDetection3D checkout:

```text
SUNRGBD.zip
SUNRGBDMeta2DBB_v2.mat
SUNRGBDMeta3DBB_v2.mat
SUNRGBDtoolbox.zip
```

Unzip the two `.zip` files, then run the official MATLAB extraction scripts:

```bash
cd data/sunrgbd/matlab
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
```

Generate MMDetection3D point clouds and info files:

```bash
cd /path/to/official/mmdetection3d
python tools/create_data.py sunrgbd \
  --root-path ./data/sunrgbd \
  --out-dir ./data/sunrgbd \
  --extra-tag sunrgbd
```

The processed public SUN RGB-D root should contain:

```text
data/sunrgbd/
  points/
  sunrgbd_trainval/
    calib/
    depth/
    image/
    label/
    train_data_idx.txt
    val_data_idx.txt
  sunrgbd_infos_train.pkl
  sunrgbd_infos_val.pkl
```

Use this processed root in the VPGNet repository, or symlink it as
`data/sunrgbd`.

If the SUN RGB-D config uses SAM3 visual priors, install SAM3 and extract
SAM3-FP1 features:

```bash
cd VPGNet-Airport-Luggage-OpenSource
git clone https://github.com/facebookresearch/sam3.git ../sam3
pip install -e ../sam3
hf auth login  # if SAM3 needs to download sam3.pt from Hugging Face

DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
SAM3_REPO=../sam3 \
FEATURE_LEVEL=1 \
SAM_RESOLUTION=1008 \
SAVE_DTYPE=float16 \
bash scripts/extract_sam3_features.sh
```

Use a local SAM3 checkpoint if available:

```bash
SAM_CHECKPOINT=/path/to/sam3.pt \
DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
bash scripts/extract_sam3_features.sh
```

Train with the common VPGNet training entry:

```bash
CONFIG=/path/to/vpgnet_sunrgbd.py \
WORK_DIR=work_dirs/vpgnet_sunrgbd \
bash scripts/run_train_vpgnet.sh
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
