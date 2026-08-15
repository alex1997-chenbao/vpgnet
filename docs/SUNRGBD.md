# Public SUN RGB-D Training Data Preparation

This page describes the code-level SUN RGB-D validation path. It provides data
preparation, SAM3 feature extraction, and training/evaluation command templates
only. Replace `CONFIG` with your own SUN RGB-D VPGNet config before training or
evaluation. No SUN RGB-D checkpoint or pretrained SUN RGB-D model is released
here.

## 1. Prepare the official SUN RGB-D data

Download the official SUN RGB-D files from the SUN RGB-D project page:

```text
http://rgbd.cs.princeton.edu/data/
```

Place the following files under `data/sunrgbd/OFFICIAL_SUNRGBD/` in an
official MMDetection3D checkout:

```text
data/sunrgbd/
  OFFICIAL_SUNRGBD/
    SUNRGBD.zip
    SUNRGBDMeta2DBB_v2.mat
    SUNRGBDMeta3DBB_v2.mat
    SUNRGBDtoolbox.zip
```

Unzip `SUNRGBD.zip` and `SUNRGBDtoolbox.zip` in that folder. The expected
intermediate layout is:

```text
data/sunrgbd/
  matlab/
    extract_rgbd_data_v1.m
    extract_rgbd_data_v2.m
    extract_split.m
  OFFICIAL_SUNRGBD/
    SUNRGBD/
    SUNRGBDMeta2DBB_v2.mat
    SUNRGBDMeta3DBB_v2.mat
    SUNRGBDtoolbox/
```

Run the official SUN RGB-D MATLAB extraction scripts:

```bash
cd data/sunrgbd/matlab
matlab -nosplash -nodesktop -r 'extract_split;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v2;quit;'
matlab -nosplash -nodesktop -r 'extract_rgbd_data_v1;quit;'
```

Then generate the MMDetection3D point clouds and annotation infos:

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

Use this processed `data/sunrgbd` root directly in this repository, or create a
symlink to it:

```bash
git clone https://github.com/alex135632/vpgnet.git
cd vpgnet
ln -s /path/to/official/mmdetection3d/data/sunrgbd data/sunrgbd
```

If you copy the data instead of symlinking it, keep the same relative paths used
by the SUN RGB-D config.

## 2. Extract SAM3-FP1 visual features

Install SAM3 in a separate environment if needed. The saved `.pt` feature files
are then consumed by the VPGNet/MMDetection3D training environment.

```bash
cd vpgnet
git clone https://github.com/facebookresearch/sam3.git ../sam3
pip install -e ../sam3
```

If the SAM3 checkpoint is not already available locally, request access to
`facebook/sam3` on Hugging Face and log in:

```bash
hf auth login
```

Extract SAM3-FP1 features for all SUN RGB-D images:

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

If `sam3.pt` has already been downloaded, pass it explicitly:

```bash
SAM_CHECKPOINT=/path/to/sam3.pt \
DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
bash scripts/extract_sam3_features.sh
```

For a quick sanity check before extracting all features:

```bash
DATA_DIR=data/sunrgbd \
IMAGE_DIR=data/sunrgbd/sunrgbd_trainval/image \
SAM_OUTPUT_DIR=data/sunrgbd/sunrgbd_trainval/sam_f \
bash scripts/extract_sam3_features.sh --limit 10 --overwrite
```

The expected output is one `.pt` file per image:

```text
data/sunrgbd/sunrgbd_trainval/sam_f/*.pt
```

The default released setting uses SAM3-FP1, image resolution `1008`, and
`float16` tensors.

## 3. Train and evaluate with the released code

Point a SUN RGB-D VPGNet config to the processed public SUN RGB-D root and the
generated SAM3 feature directory. Then train with:

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
