#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/probe.CLaMP3-layers.GTZANGenre.yaml"
NUM_LAYERS=13
MODEL_TAG="CLaMP3"
TASK_TAG="GTZANGenre"
SWEEP_DIR="configs/sweeps/CLaMP3.GTZANGenre"

python scripts/gen_sweep_configs.py \
    --base-config "$BASE_CONFIG" \
    --num-layers  "$NUM_LAYERS" \
    --model-tag   "$MODEL_TAG" \
    --task-tag    "$TASK_TAG" \
    --out-dir     "$SWEEP_DIR"

for LAYER in $(seq 0 $((NUM_LAYERS - 1))); do
    CFG="$SWEEP_DIR/sweep.${MODEL_TAG}.${TASK_TAG}.layer${LAYER}.yaml"
    echo "=== Layer ${LAYER} ==="
    python cli.py fit  -c "$CFG"
    python cli.py test -c "$CFG"
done
