# Airport-luggage Dataset Notes

The self-collected airport-luggage dataset is not included in git. The
train/test files needed by the released airport-luggage VPGNet config are
published on Hugging Face and can be downloaded from:

```text
https://huggingface.co/datasets/alex135632/vpgnet-airport-luggage
```

The provided script downloads the same dataset to the expected local layout:

```bash
bash scripts/download_dataset.sh
```

The released airport-luggage config uses a SUN RGB-D-style directory layout for
compatibility with MMDetection3D data pipelines. The directory name
`data/sunrgbd` is a layout convention here; the files downloaded by
`scripts/download_dataset.sh` are the self-collected airport-luggage data.

The expected root is:

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

The dataset class is `LuggageSUNRGBDDataset`. It uses one class:

```text
luggage
```

The default pipeline loads:

- point cloud files from `points/`
- images from `sunrgbd_trainval/image/`
- SAM3 features from `sunrgbd_trainval/sam_f/`
- annotations from `sunrgbd_trainval/sunrgbd_infos_train.pkl` and
  `sunrgbd_trainval/sunrgbd_infos_val.pkl`

## SAM3 Feature Extraction

SAM3 features are generated from the released RGB images before VPGNet
training and evaluation.

Install SAM3 and extract the visual prior:

```bash
git clone https://github.com/facebookresearch/sam3.git ../sam3
pip install -e ../sam3
hf auth login
bash scripts/extract_sam3_features.sh
```

If the checkpoint is not provided through `SAM_CHECKPOINT`, request access to
`facebook/sam3` on Hugging Face before running the extractor.

By default this extracts the same prior used by the released checkpoint:

```text
feature level: SAM3-FP1
resolution: 1008
dtype: float16
output: data/sunrgbd/sunrgbd_trainval/sam_f/*.pt
shape per file: (256, 144, 144)
```

If you already downloaded `sam3.pt`, pass it explicitly:

```bash
SAM_CHECKPOINT=/path/to/sam3.pt bash scripts/extract_sam3_features.sh
```

SAM feature files should be `.pt` tensors or dict payloads containing one of
these tensor keys: `sam_feat`, `patch_features`, `feature`, `features`,
`fpn_0`, `fpn_1`, or `fpn_2`.
