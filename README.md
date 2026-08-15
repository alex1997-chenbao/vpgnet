# VPGNet Airport Luggage Detection

Clean release package for VPGNet, a visual-prior-guided 3D detector. This
package keeps the modified MMDetection3D source, the VPGNet airport-luggage
config, evaluation utilities, and metadata for the released airport-luggage
checkpoint.

The repository is organized around two validation paths:

1. public SUN RGB-D training with the released code, for code-level validation;
2. self-collected airport-luggage validation, with the released data,
   checkpoint, and evaluation scripts.

## Included

- VPGNet source code in `mmdet3d/`
  - detector integration: `mmdet3d/models/detectors/imvotenet.py`
  - GCA and feature fusion modules: `mmdet3d/models/my_module/`
  - luggage dataset and SAM feature loading: `mmdet3d/datasets/`
- Airport-luggage config: `configs/vpgnet/vpgnet_airport_luggage.py`
- Airport-luggage checkpoint metadata: `checkpoints/best.json`
- Core tools:
  - `tools/train.py`
  - `tools/test.py`
  - `tools/misc/export_mmdet3d_predictions_jsonl.py`
  - `tools/misc/eval_luggage_subsets_ap.py`
  - `tools/misc/eval_luggage_subset_pr_gtfirst.py`
  - `tools/misc/compute_gt_first_match_iou_yaw_stats.py`
  - `tools/misc/postprocess_vpgnet_with_sam3_instance_masks.py`
  - `tools/misc/visualize_vpgnet_predictions_o3d.py`

The self-collected airport-luggage train/test dataset is available on Hugging
Face:

```text
https://huggingface.co/datasets/alex135632/vpgnet-airport-luggage
```

## Quick Start

Install this package in the same style as MMDetection3D:

```bash
git clone https://github.com/alex135632/vpgnet.git
cd vpgnet
pip install -v -e .
```

### 1. Public SUN RGB-D training path

For public-dataset code validation, prepare SUN RGB-D with the standard
MMDetection3D preprocessing pipeline, extract SAM3-FP1 image features if the
config uses visual priors, then train with a SUN RGB-D VPGNet config through
the same training entry point. These SUN RGB-D commands are templates: replace
`CONFIG` with your own SUN RGB-D VPGNet config. The full data-preparation notes
are in `docs/SUNRGBD.md`.

```bash
# In an official MMDetection3D checkout:
# 1) Download SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat,
#    SUNRGBDMeta3DBB_v2.mat, and SUNRGBDtoolbox.zip.
# 2) Run the official MATLAB extraction scripts.
# 3) Generate the MMDetection3D infos:
python tools/create_data.py sunrgbd \
  --root-path ./data/sunrgbd \
  --out-dir ./data/sunrgbd \
  --extra-tag sunrgbd
```

Use the processed `data/sunrgbd` root in this repository, then extract SAM3-FP1
features:

```bash
DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
SAM3_REPO=../sam3 \
FEATURE_LEVEL=1 \
SAM_RESOLUTION=1008 \
SAVE_DTYPE=float16 \
bash scripts/extract_sam3_features.sh
```

If the SAM3 checkpoint is already available locally, pass it with
`SAM_CHECKPOINT=/path/to/sam3.pt`.

Train on SUN RGB-D:

```bash
CONFIG=/path/to/your_vpgnet_sunrgbd_config.py \
WORK_DIR=work_dirs/vpgnet_sunrgbd \
bash scripts/run_train_vpgnet.sh
```

Evaluate a trained SUN RGB-D checkpoint with:

```bash
CONFIG=/path/to/your_vpgnet_sunrgbd_config.py \
CHECKPOINT=/path/to/checkpoint.pth \
WORK_DIR=work_dirs/vpgnet_sunrgbd_eval \
bash scripts/run_eval_vpgnet.sh
```

No SUN RGB-D checkpoint is released in this repository.

### 2. Self-collected airport-luggage validation path

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
| VPGNet | SAM3-FP1 | 0.9846 | 0.9794 | [Hugging Face](https://huggingface.co/alex135632/vpgnet-airport-luggage/blob/main/best.pth) |

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
