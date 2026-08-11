# Reproduce

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
