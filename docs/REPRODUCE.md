# Reproduce

## Prepare Data

Download the train/test dataset from Hugging Face:

```bash
cd VPGNet-Airport-Luggage-OpenSource
hf auth login  # required if the dataset is private
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

## Evaluate

First place the checkpoint release asset at:

```text
checkpoints/vpgnet_sam3fp1_best_epoch28.pth
```

```bash
cd VPGNet-Airport-Luggage-OpenSource
bash scripts/run_eval_vpgnet.sh
```

The script runs:

```bash
python tools/test.py \
  configs/vpgnet/vpgnet_airport_luggage.py \
  checkpoints/vpgnet_sam3fp1_best_epoch28.pth \
  --work-dir work_dirs/vpgnet_airport_luggage_eval
```

## Train

```bash
cd VPGNet-Airport-Luggage-OpenSource
bash scripts/run_train_vpgnet.sh
```

The released config trains for 40 epochs with one luggage class, fixed height
configuration, 20000 sampled points, and the VPGNet fusion/refinement modules.

## Export JSONL Predictions

First place the checkpoint release asset at
`checkpoints/vpgnet_sam3fp1_best_epoch28.pth`.

```bash
bash scripts/export_vpgnet_predictions.sh
```

The output path is `work_dirs/vpgnet_predictions/val_predictions.jsonl`.

## Subset Evaluation Utilities

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
