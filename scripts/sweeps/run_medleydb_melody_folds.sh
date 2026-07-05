#!/usr/bin/env bash
# MedleyDB melody extraction — all 5 CV folds × layer sweep for the priority
# encoders (MuQ 13 → MERT-v1-95M 13 → OMARRQ-multifeature-25hz 24) + a per-fold
# meanall baseline. SUPERVISED — trains an MLP head per run (40 epochs, early-stop
# on val/acc_rpa) and writes checkpoints. Mirrors run_bps_mnid_abc_folds.sh (the
# canonical fold convention): --run-name-suffix puts foldF in BOTH fit and test
# run names while job_type stays the clean stage, and LogSweepCoordsCallback
# stamps sweep/fold authoritatively from the datamodule's fold_idx.
#
# Build the 5-fold JSONLs first (needs a local MedleyDB copy):
#   uv run python scripts/data/build_medleydb_melody_jsonl.py \
#       --audio-root <Audio> --annotation-root <Annotations> --out-dir data/MedleyDB
#
# num_workers: default 6. The old "workers deadlock at spawn on WSL" note is
# STALE — verified 2026-07-06: 8 fork-mode workers ran a full VGMIDITVar-timbre
# extraction on this WSL box without issue (the historical deadlock was the
# *Windows-native* spawn path / SSH-session desktop-heap case, see
# docs/local_sweeps.md). Override with NUM_WORKERS=0 only if launching from
# Windows-native Python rather than WSL.
#
# Usage:
#   scripts/sweeps/run_medleydb_melody_folds.sh [--accelerator gpu|mps] \
#       [--folds "0 1 2 3 4"] [--layers "0 6 12"] [--concurrency N] \
#       [--models "MuQ:13 MERT-v1-95M:13 OMARRQ-multifeature-25hz:24"]
set -uo pipefail

ACCEL="gpu"
FOLDS="0 1 2 3 4"
LAYERS_ARG=""
CONCURRENCY="1"
NUM_WORKERS="${NUM_WORKERS:-6}"
MODELS_ARG="MuQ:13 MERT-v1-95M:13 OMARRQ-multifeature-25hz:24"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --models) MODELS_ARG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

read -ra MODELS <<< "$MODELS_ARG"
for entry in "${MODELS[@]}"; do
  M="${entry%%:*}"; NL="${entry##*:}"
  BASE="configs/probe.${M}-layers.MedleyDBMelody.yaml"
  MEANALL_BASE="configs/probe.${M}-meanall.MedleyDBMelody.yaml"
  for F in $FOLDS; do
    # ── per-fold meanall baseline: patch fold_idx + the output dir to land at
    #    .meanall.fold{F}, then fit+test directly (the layer sweep below passes
    #    --skip-meanall). name carries the fold so LogSweepCoordsCallback stamps
    #    sweep/fold; job_type is the clean stage.
    MCFG="configs/_medleydb_${M}_meanall_fold${F}.yaml"
    sed -e "s/fold_idx: 0/fold_idx: ${F}/g" \
        -e "s#MedleyDBMelody.${M}-meanall/#MedleyDBMelody.${M}-meanall.fold${F}/#g" \
        "$MEANALL_BASE" > "$MCFG"
    echo "===== MedleyDBMelody ${M} MEANALL FOLD ${F} (accelerator=${ACCEL}) ====="
    "$PY" cli.py fit  -c "$MCFG" --trainer.accelerator "$ACCEL" \
        --data.init_args.num_workers "$NUM_WORKERS" \
        --trainer.logger.init_args.name "meanall-fold${F}-fit"  --trainer.logger.init_args.job_type "fit"
    "$PY" cli.py test -c "$MCFG" --trainer.accelerator "$ACCEL" \
        --data.init_args.num_workers "$NUM_WORKERS" \
        --trainer.logger.init_args.name "meanall-fold${F}-test" --trainer.logger.init_args.job_type "test"

    # ── per-layer sweep: patch only the CV fold; the per-fold output dir is
    #    appended by run_sweep_local via --dir-suffix (layer-primary:
    #    ...-layers.layer{N}.fold{F}).
    CFG="configs/_medleydb_${M}_fold${F}.yaml"
    sed -e "s/fold_idx: 0/fold_idx: ${F}/g" "$BASE" > "$CFG"
    echo "===== MedleyDBMelody ${M} FOLD ${F} (accelerator=${ACCEL}, concurrency=${CONCURRENCY}, num_workers=${NUM_WORKERS}) ====="
    "$PY" scripts/sweeps/run_sweep_local.py \
      --base-config "$CFG" \
      --num-layers "$NL" --model-tag "$M" --task-tag MedleyDBMelody \
      --accelerator "$ACCEL" --skip-meanall --concurrency "$CONCURRENCY" \
      --num-workers "$NUM_WORKERS" \
      --run-name-suffix "fold${F}" --dir-suffix ".fold${F}" ${EXTRA_LAYERS[@]+"${EXTRA_LAYERS[@]}"}
  done
done
echo "===== MedleyDBMelody 5-fold sweep complete ====="
