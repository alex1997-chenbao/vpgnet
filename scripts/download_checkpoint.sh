#!/usr/bin/env bash
set -euo pipefail

HF_BIN="${HF_BIN:-hf}"
MODEL_REPO="${MODEL_REPO:-alex-chenbao1997/vpgnet-airport-luggage}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vpgnet_sam3fp1_best_epoch24.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
MAX_WORKERS="${MAX_WORKERS:-4}"
USE_PROXY="${USE_PROXY:-0}"

if ! command -v "$HF_BIN" >/dev/null 2>&1 && [[ ! -x "$HF_BIN" ]]; then
  echo "Hugging Face CLI not found: $HF_BIN" >&2
  echo "Install it with: pip install -U 'huggingface_hub[hf_xet]'" >&2
  exit 1
fi

if [[ "$USE_PROXY" == "0" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
  unset http_proxy https_proxy all_proxy
fi

mkdir -p "$CHECKPOINT_DIR"

"$HF_BIN" download "$MODEL_REPO" "$CHECKPOINT_NAME" \
  --repo-type model \
  --local-dir "$CHECKPOINT_DIR" \
  --max-workers "$MAX_WORKERS"

echo "[DONE] Checkpoint is ready at $CHECKPOINT_DIR/$CHECKPOINT_NAME"
