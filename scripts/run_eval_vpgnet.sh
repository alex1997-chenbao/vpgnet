#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/vpgnet/vpgnet_airport_luggage.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vpgnet_epoch31_best.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/vpgnet_airport_luggage_eval}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Download the GitHub Release asset and place it at checkpoints/vpgnet_epoch31_best.pth" >&2
  exit 1
fi

python tools/test.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR" "$@"
