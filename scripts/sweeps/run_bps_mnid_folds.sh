#!/usr/bin/env bash
# BPS-Motif MNID (motif vs non-motif window classification) sweep:
# all 5 CV folds x 13 CLaMP3 layers. SUPERVISED — trains an MLP head per run
# (40 epochs, early-stop on val/f1) and writes checkpoints.
#
# !! STORAGE !! Each checkpoint bundles the frozen 552M-param CLaMP3 encoder
# (~2.2 GB). With save_top_k=1 + save_last that is ~4.4 GB/run x 65 runs ~= 286 GB.
# Choose a checkpoint root with --ckpt-root (e.g. an external drive) OR apply the
# encoder-stripping fix first (see docs). On the PC, internal disk is usually fine.
#
# Usage:
#   scripts/sweeps/run_bps_mnid_folds.sh [--accelerator mps|gpu] [--folds "0 1 2 3 4"] \
#       [--ckpt-root /Volumes/WD\ Black/marble] [--layers "0 1 ... 12"]
#
# Mac:  scripts/sweeps/run_bps_mnid_folds.sh --accelerator mps --ckpt-root "/Volumes/WD Black/marble"
# PC :  scripts/sweeps/run_bps_mnid_folds.sh --accelerator gpu
set -uo pipefail

ACCEL="gpu"
FOLDS="0 1 2 3 4"
CKPT_ROOT=""
LAYERS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
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
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" \
      -e "s#CLaMP3-symbolic-layers/#CLaMP3-symbolic-layers.fold${F}/#g" \
      "$BASE" > "$CFG"
  # Optionally redirect checkpoints off the internal disk.
  if [[ -n "$CKPT_ROOT" ]]; then
    sed -i.bak \
      -e "s#dirpath: \"\./output/#dirpath: \"${CKPT_ROOT}/output/#g" \
      "$CFG" && rm -f "${CFG}.bak"
  fi
  echo "===== MNID FOLD ${F} (accelerator=${ACCEL}, ckpt_root=${CKPT_ROOT:-./output}) ====="
  "$PY" scripts/sweeps/run_sweep_local.py \
    --base-config "$CFG" \
    --num-layers 13 --model-tag CLaMP3-symbolic --task-tag BPSMotifMNID \
    --accelerator "$ACCEL" --skip-meanall \
    --extra-tag "fold${F}" "${EXTRA_LAYERS[@]}"
done
echo "===== MNID sweep complete ====="
