#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/probe.OMARRQ-multifeature25hz.GTZANGenre.yaml"
NUM_LAYERS=12
MODEL_TAG="OMARRQ-multifeature25hz"
TASK_TAG="GTZANGenre"
SWEEP_DIR="configs/sweeps/OMARRQ-multifeature25hz.GTZANGenre"

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
