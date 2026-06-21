#!/usr/bin/env bash
# BPS-Motif MNID — ABC INPUT (Option B) sweep: all 5 CV folds x 13 CLaMP3 layers
# + a meanall baseline per fold. SUPERVISED — trains an MLP head per run (40
# epochs, early-stop on val/f1) and writes checkpoints. ABC counterpart of
# run_bps_mnid_folds.sh; the only differences are the base config (ABC),
# --model-tag CLaMP3-symbolic-abc (so output dirs + wandb don't collide with
# the MTF sweep), and the per-fold temp config name.
#
# Build the ABC dataset first:
#   uv run python scripts/data/build_bps_motif_abc.py
#
# Usage:
#   scripts/sweeps/run_bps_mnid_abc_folds.sh [--accelerator mps|gpu] \
#       [--folds "0 1 2 3 4"] [--layers "0 1 ... 12"] [--concurrency N]
set -uo pipefail

ACCEL="gpu"
FOLDS="0 1 2 3 4"
LAYERS_ARG=""
# Run 2 layers in parallel as separate subprocesses, matching how the MTF MNID
# sweep was run (each layer-fit needs ~5-6 GB VRAM; the 16 GB RTX 5060 Ti fits
# 2). run_sweep_local auto-halves per-proc dataloader workers to bound CPU.
CONCURRENCY="2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"
BASE="configs/probe.CLaMP3-symbolic-abc-layers.BPSMotifMNID.yaml"
MEANALL_BASE="configs/probe.CLaMP3-symbolic-abc-meanall.BPSMotifMNID.yaml"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

for F in $FOLDS; do
  # ── per-fold meanall baseline (mirrors how the MTF meanall.fold{F} dirs were
  #    produced): patch fold_idx + the output dir to land at .meanall.fold{F},
  #    then fit+test directly. The per-layer sweep below passes --skip-meanall.
  MCFG="configs/_bps_mnid_abc_meanall_fold${F}.yaml"
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" \
      -e "s#CLaMP3-symbolic-abc-meanall/#CLaMP3-symbolic-abc-meanall.fold${F}/#g" \
      "$MEANALL_BASE" > "$MCFG"
  echo "===== MNID-ABC MEANALL FOLD ${F} (accelerator=${ACCEL}) ====="
  # Match the established meanall naming/grouping/job_type convention exactly
  # (see the MTF arm `CLaMP3-symbolic / BPSMotifMNID`): name carries the fold
  # (`meanall-fold{F}-{fit,test}`) so LogSweepCoordsCallback stamps sweep/fold,
  # and job_type is the clean stage (`fit`/`test`) rather than the YAML default.
  "$PY" cli.py fit  -c "$MCFG" --trainer.accelerator "$ACCEL" \
      --trainer.logger.init_args.name "meanall-fold${F}-fit" \
      --trainer.logger.init_args.job_type "fit"
  "$PY" cli.py test -c "$MCFG" --trainer.accelerator "$ACCEL" \
      --trainer.logger.init_args.name "meanall-fold${F}-test" \
      --trainer.logger.init_args.job_type "test"

  CFG="configs/_bps_mnid_abc_fold${F}.yaml"
  # Patch only the CV split; the per-fold output dir is appended by
  # run_sweep_local via --dir-suffix (layer-primary: ...-abc-layers.layer{N}.fold{F}).
  sed -e "s/fold_idx: 0/fold_idx: ${F}/g" "$BASE" > "$CFG"
  echo "===== MNID-ABC FOLD ${F} (accelerator=${ACCEL}, concurrency=${CONCURRENCY}) ====="
  # --run-name-suffix (NOT --extra-tag): puts foldF in BOTH the fit and test run
  # NAMES while keeping job_type the clean stage (fit/test) — so runs group by
  # sweep/layer and EVERY run (fit included) is recoverable by fold. The fold
  # coord itself is stamped authoritatively by LogSweepCoordsCallback (which now
  # reads the datamodule's fold_idx), so grouping works even on fit runs.
  "$PY" scripts/sweeps/run_sweep_local.py \
    --base-config "$CFG" \
    --num-layers 13 --model-tag CLaMP3-symbolic-abc --task-tag BPSMotifMNID \
    --accelerator "$ACCEL" --skip-meanall --concurrency "$CONCURRENCY" \
    --run-name-suffix "fold${F}" --dir-suffix ".fold${F}" "${EXTRA_LAYERS[@]}"
done
echo "===== MNID-ABC sweep complete ====="
