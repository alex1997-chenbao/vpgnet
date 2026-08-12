#!/usr/bin/env bash
set -euo pipefail

RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_SAM3="${RUN_SAM3:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"

if [[ "$RUN_DOWNLOAD" == "1" ]]; then
  bash scripts/download_dataset.sh
fi

if [[ "$RUN_SAM3" == "1" ]]; then
  bash scripts/extract_sam3_features.sh
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  bash scripts/run_train_vpgnet.sh
fi
