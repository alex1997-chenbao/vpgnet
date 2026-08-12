#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/sunrgbd}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_DIR/sunrgbd_trainval/image}"
SAM_OUTPUT_DIR="${SAM_OUTPUT_DIR:-$DATA_DIR/sunrgbd_trainval/sam_f}"
SAM3_REPO="${SAM3_REPO:-../sam3}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-}"
SAM_RESOLUTION="${SAM_RESOLUTION:-1008}"
FEATURE_LEVEL="${FEATURE_LEVEL:-1}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
REPORT_EVERY="${REPORT_EVERY:-25}"

if [[ -z "$SAM_CHECKPOINT" && -f "$SAM3_REPO/sam3.pt" ]]; then
  SAM_CHECKPOINT="$SAM3_REPO/sam3.pt"
fi

args=(
  --image-dir "$IMAGE_DIR"
  --output-dir "$SAM_OUTPUT_DIR"
  --sam-resolution "$SAM_RESOLUTION"
  --feature-level "$FEATURE_LEVEL"
  --save-dtype "$SAVE_DTYPE"
  --report-every "$REPORT_EVERY"
)

if [[ -d "$SAM3_REPO" ]]; then
  args+=(--sam3-repo "$SAM3_REPO")
fi
if [[ -n "$SAM_CHECKPOINT" ]]; then
  args+=(--sam-checkpoint "$SAM_CHECKPOINT")
fi

"$PYTHON" tools/misc/extract_sam3_sunrgbd_features.py "${args[@]}" "$@"

echo "[DONE] SAM3-FP${FEATURE_LEVEL} features are ready under $SAM_OUTPUT_DIR"
