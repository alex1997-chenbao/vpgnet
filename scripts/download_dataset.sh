#!/usr/bin/env bash
set -euo pipefail

HF_BIN="${HF_BIN:-hf}"
DATASET_REPO="${DATASET_REPO:-alex-chenbao1997/vpgnet-airport-luggage}"
DATA_DIR="${DATA_DIR:-data/sunrgbd}"
MAX_WORKERS="${MAX_WORKERS:-8}"
USE_PROXY="${USE_PROXY:-0}"

if [[ "$HF_BIN" == "hf" ]] && ! command -v hf >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/hf" ]]; then
    HF_BIN="$HOME/.local/bin/hf"
  fi
fi

if ! command -v "$HF_BIN" >/dev/null 2>&1 && [[ ! -x "$HF_BIN" ]]; then
  echo "Hugging Face CLI not found: $HF_BIN" >&2
  echo "Install it with: pip install -U 'huggingface_hub[hf_xet]'" >&2
  exit 1
fi

if [[ "$USE_PROXY" == "0" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  unset http_proxy https_proxy all_proxy
fi

mkdir -p "$DATA_DIR"

echo "[INFO] Dataset repo: $DATASET_REPO"
echo "[INFO] Local data dir: $DATA_DIR"
echo "[INFO] Download workers: $MAX_WORKERS"

"$HF_BIN" download "$DATASET_REPO" \
  --repo-type dataset \
  --local-dir "$DATA_DIR" \
  --max-workers "$MAX_WORKERS" \
  --include "points/**" \
  --include "sunrgbd_trainval/calib/**" \
  --include "sunrgbd_trainval/image/**" \
  --include "sunrgbd_trainval/label/**" \
  --include "sunrgbd_trainval/sunrgbd_infos_train.pkl" \
  --include "sunrgbd_trainval/sunrgbd_infos_val.pkl" \
  --include "train_data_idx.txt" \
  --include "val_data_idx.txt"

echo "[DONE] Dataset is ready under $DATA_DIR"
