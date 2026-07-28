#!/usr/bin/env bash
set -euo pipefail

python main_v.py \
  --data-root "${DATA_ROOT:-./dataset}" \
  --output-dir "${OUTPUT_DIR:-./results}" \
  --dataset "${DATASET:-houston}" \
  --Experiment_num "${EXPERIMENT_NUM:-10}" \
  --split_type "${SPLIT_TYPE:-fixed}" \
  --train_num "${TRAIN_NUM:-30}" \
  --patch_size "${PATCH_SIZE:-13}"
