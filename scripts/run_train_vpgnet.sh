#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/vpgnet/vpgnet_airport_luggage.py}"
WORK_DIR="${WORK_DIR:-work_dirs/vpgnet_airport_luggage}"

python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "$@"

