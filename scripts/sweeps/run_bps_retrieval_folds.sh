#!/usr/bin/env bash
# Comprehensive BPS-Motif Retrieval sweep: all 5 CV folds x 13 CLaMP3 layers.
#
# run_sweep_local.py sweeps LAYERS, not folds, so this wraps it: for each fold it
# patches fold_idx + save_dir into a per-fold base config, then runs the 13-layer
# sweep. Retrieval is zero-shot (max_epochs=0) so NO checkpoints are written.
#
# Usage:
#   scripts/sweeps/run_bps_retrieval_folds.sh [--accelerator mps|gpu|cpu] [--folds "0 1 2 3 4"]
#
# Mac:  scripts/sweeps/run_bps_retrieval_folds.sh --accelerator mps
# CUDA: scripts/sweeps/run_bps_retrieval_folds.sh --accelerator gpu
#
# Requires the symbolic-midi extra:  uv pip install mido pretty_midi
set -uo pipefail

ACCEL="mps"
FOLDS="0 1 2 3 4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"
BASE="configs/probe.CLaMP3-symbolic-layers.BPSMotifRetrieval.yaml"

for F in $FOLDS; do
  CFG="configs/_bps_retr_fold${F}.yaml"
  # Patch only the CV split; the per-fold output dir is appended by
  # run_sweep_local via --dir-suffix (layer-primary: ...-layers.layer{N}.fold{F}).
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" "$BASE" > "$CFG"
  echo "===== FOLD ${F} (accelerator=${ACCEL}) ====="
  "$PY" scripts/sweeps/run_sweep_local.py \
    --base-config "$CFG" \
    --num-layers 13 --model-tag CLaMP3-symbolic --task-tag BPSMotifRetrieval \
    --accelerator "$ACCEL" --skip-fit-if-no-train --skip-meanall \
    --extra-tag "fold${F}" --dir-suffix ".fold${F}"
done

echo "===== aggregating ====="
"$PY" scripts/sweeps/bps_retrieval_summary.py
