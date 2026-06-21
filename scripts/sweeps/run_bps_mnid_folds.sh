#!/usr/bin/env bash
# BPS-Motif MNID (motif vs non-motif window classification) sweep:
# all 5 CV folds x 13 CLaMP3 layers. SUPERVISED — trains an MLP head per run
# (40 epochs, early-stop on val/f1) and writes checkpoints.
#
# Checkpoints are tiny (~0.9 MB) thanks to the frozen-encoder strip in BaseTask
# (the 552M-param CLaMP3 encoder is dropped on save + re-injected on load), so
# the full sweep's checkpoints total ~113 MB — internal disk is always fine.
#
# Usage:
#   scripts/sweeps/run_bps_mnid_folds.sh [--accelerator mps|gpu] [--folds "0 1 2 3 4"] \
#       [--layers "0 1 ... 12"]
#
# Mac:  scripts/sweeps/run_bps_mnid_folds.sh --accelerator mps
# PC :  scripts/sweeps/run_bps_mnid_folds.sh --accelerator gpu
set -uo pipefail

ACCEL="gpu"
FOLDS="0 1 2 3 4"
LAYERS_ARG=""
CONCURRENCY="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;  # run N layers in parallel (CUDA)
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"
BASE="configs/probe.CLaMP3-symbolic-layers.BPSMotifMNID.yaml"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

for F in $FOLDS; do
  CFG="configs/_bps_mnid_fold${F}.yaml"
  # Patch only the CV split here; the per-fold output dir is appended by
  # run_sweep_local via --dir-suffix so it lands AFTER .layer{N}
  # (layer-primary: ...-layers.layer{N}.fold{F}).
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" "$BASE" > "$CFG"
  echo "===== MNID FOLD ${F} (accelerator=${ACCEL}, concurrency=${CONCURRENCY}) ====="
  "$PY" scripts/sweeps/run_sweep_local.py \
    --base-config "$CFG" \
    --num-layers 13 --model-tag CLaMP3-symbolic --task-tag BPSMotifMNID \
    --accelerator "$ACCEL" --skip-meanall --concurrency "$CONCURRENCY" \
    --extra-tag "fold${F}" --dir-suffix ".fold${F}" "${EXTRA_LAYERS[@]}"
done
echo "===== MNID sweep complete ====="
