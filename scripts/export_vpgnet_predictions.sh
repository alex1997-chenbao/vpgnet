#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/vpgnet/vpgnet_airport_luggage.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vpgnet_sam3fp1_best_epoch24.pth}"
OUT_JSONL="${OUT_JSONL:-work_dirs/vpgnet_predictions/val_predictions.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Run: bash scripts/download_checkpoint.sh" >&2
  exit 1
fi

python tools/misc/export_mmdet3d_predictions_jsonl.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --out-jsonl "$OUT_JSONL" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  "$@"
