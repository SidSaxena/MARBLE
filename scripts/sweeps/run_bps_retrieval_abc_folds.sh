#!/usr/bin/env bash
# BPS-Motif Retrieval — ABC INPUT (Option B) sweep: all 5 CV folds x 13 CLaMP3
# layers + a per-fold meanall baseline. Zero-shot (max_epochs=0) so NO
# checkpoints are written. ABC counterpart of run_bps_retrieval_folds.sh; the
# only differences are the base config (ABC), --model-tag CLaMP3-symbolic-abc
# (so output dirs + wandb don't collide with the MTF sweep), and the per-fold
# temp config names.
#
# Build the ABC dataset first:
#   uv run python scripts/data/build_bps_motif_abc.py
#
# Usage:
#   scripts/sweeps/run_bps_retrieval_abc_folds.sh [--accelerator mps|gpu|cpu] \
#       [--folds "0 1 2 3 4"] [--layers "0 1 ... 12"]
set -uo pipefail

ACCEL="gpu"
FOLDS="0 1 2 3 4"
LAYERS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"
BASE="configs/probe.CLaMP3-symbolic-abc-layers.BPSMotifRetrieval.yaml"
MEANALL_BASE="configs/probe.CLaMP3-symbolic-abc-meanall.BPSMotifRetrieval.yaml"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

for F in $FOLDS; do
  # ── per-fold meanall baseline (zero-shot test only): patch fold_idx + output
  #    dir to .meanall.fold{F}. The per-layer sweep below passes --skip-meanall.
  MCFG="configs/_bps_retr_abc_meanall_fold${F}.yaml"
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" \
      -e "s#CLaMP3-symbolic-abc-meanall/#CLaMP3-symbolic-abc-meanall.fold${F}/#g" \
      "$MEANALL_BASE" > "$MCFG"
  echo "===== RETR-ABC MEANALL FOLD ${F} (accelerator=${ACCEL}) ====="
  # Match the established meanall naming convention (see the MTF arm): name
  # carries the fold so LogSweepCoordsCallback stamps sweep/fold; job_type=test.
  "$PY" cli.py test -c "$MCFG" --trainer.accelerator "$ACCEL" \
      --trainer.logger.init_args.name "meanall-fold${F}-test" \
      --trainer.logger.init_args.job_type "test"

  CFG="configs/_bps_retr_abc_fold${F}.yaml"
  # Patch only the CV split; the per-fold output dir is appended by
  # run_sweep_local via --dir-suffix (layer-primary: ...-abc-layers.layer{N}.fold{F}).
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" "$BASE" > "$CFG"
  echo "===== RETR-ABC FOLD ${F} (accelerator=${ACCEL}) ====="
  # --run-name-suffix (NOT --extra-tag): foldF in the test run NAME, job_type
  # stays the clean stage 'test'. The fold coord is stamped authoritatively by
  # LogSweepCoordsCallback (reads datamodule fold_idx), so runs group by
  # sweep/layer and are recoverable by fold. (Retrieval is zero-shot: test only.)
  "$PY" scripts/sweeps/run_sweep_local.py \
    --base-config "$CFG" \
    --num-layers 13 --model-tag CLaMP3-symbolic-abc --task-tag BPSMotifRetrieval \
    --accelerator "$ACCEL" --skip-fit-if-no-train --skip-meanall \
    --run-name-suffix "fold${F}" --dir-suffix ".fold${F}" "${EXTRA_LAYERS[@]}"
done

echo "===== RETR-ABC sweep complete ====="
