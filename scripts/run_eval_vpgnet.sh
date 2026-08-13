#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/vpgnet/vpgnet_airport_luggage.py}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/vpgnet_airport_luggage_eval}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Run: bash scripts/download_checkpoint.sh" >&2
  exit 1
fi

python tools/test.py "$CONFIG" "$CHECKPOINT" --work-dir "$WORK_DIR" "$@"
