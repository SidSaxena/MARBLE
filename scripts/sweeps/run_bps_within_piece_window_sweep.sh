#!/usr/bin/env bash
# BPS-Motif WITHIN-PIECE window-size breaking-point sweep — fold-style.
#
# The window analog of run_bps_retrieval_abc_folds.sh: for each encoding arm
# (clip-isolated `BPSMotifWithinPiece` and whole-piece `BPSMotifWithinPieceWhole`)
# and each window size N (bars), run all 13 CLaMP3 layers + a meanall baseline,
# zero-shot (max_epochs=0, no checkpoints). All windows of an arm share ONE wandb
# group (neutral, set in the per-N configs); LogSweepCoordsCallback stamps
# sweep/window (from the test JSONL `.N{W}.`) and sweep/layer, so the dashboard
# groups by sweep/layer and plots metric vs sweep/window — exactly as the fold
# sweep groups by layer and reads metric vs sweep/fold.
#
# The prevalence permutation null runs inside each probe (test/map_centered_lift
# = real - null), so every (arm, window, layer) cell carries its honest,
# prevalence-controlled lift. Set BPS_NULL_PERMS to change perm count (default 100).
#
# Build the per-window datasets + configs first:
#   for N in 1 2 3 4 6 8 12 16 24 32; do
#     uv run python scripts/data/build_bps_motif_within_piece.py --window $N --workers 8
#     uv run python scripts/data/build_bps_motif_within_piece.py --window $N --whole --workers 8
#   done
#   python scripts/sweeps/gen_within_piece_n_configs.py --windows 1 2 3 4 6 8 12 16 24 32 --include-n4
#
# Usage:
#   scripts/sweeps/run_bps_within_piece_window_sweep.sh \
#       [--accelerator gpu|mps|cpu] [--concurrency 4] \
#       [--windows "1 2 3 4 6 8 12 16 24 32"] [--arms "clip whole"] \
#       [--layers "0 1 ... 12"]
set -uo pipefail

ACCEL="gpu"
CONCURRENCY="4"
WINDOWS="1 2 3 4 6 8 12 16 24 32"
ARMS="clip whole"
LAYERS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --windows) WINDOWS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

for ARM in $ARMS; do
  case "$ARM" in
    clip)  TASK_BASE="BPSMotifWithinPiece";      INFIX="BPSMotifWithinPiece" ;;
    whole) TASK_BASE="BPSMotifWithinPieceWhole"; INFIX="BPSMotifWithinPieceWhole" ;;
    *) echo "unknown arm: $ARM"; exit 2 ;;
  esac

  for W in $WINDOWS; do
    LAYERS_CFG="configs/probe.CLaMP3-symbolic-abc-layers.${INFIX}N${W}.yaml"
    MEANALL_CFG="configs/probe.CLaMP3-symbolic-abc-meanall.${INFIX}N${W}.yaml"
    if [[ ! -f "$LAYERS_CFG" ]]; then
      echo "!! missing $LAYERS_CFG — run gen_within_piece_n_configs.py --windows $W --include-n4; skipping ${ARM} N=$W" >&2
      continue
    fi

    # ── per-window meanall baseline (zero-shot test only). Name carries the
    #    window so the coord is recoverable on this single-run baseline too;
    #    sweep/window is ALSO stamped authoritatively from the JSONL.
    if [[ -f "$MEANALL_CFG" ]]; then
      echo "===== ${ARM} WITHIN-PIECE MEANALL N=${W} (accel=${ACCEL}) ====="
      "$PY" cli.py test -c "$MEANALL_CFG" --trainer.accelerator "$ACCEL" \
        --trainer.logger.init_args.name "meanall-window${W}-test" \
        --trainer.logger.init_args.job_type "test"
    fi

    # ── per-window 13-layer sweep. Single neutral wandb group (from the config);
    #    --task-tag carries N for per-window output-dir + resume-skip isolation;
    #    --run-name-suffix window{W} puts the window in the run NAME (job_type
    #    stays clean 'test'); the window coord is stamped authoritatively.
    echo "===== ${ARM} WITHIN-PIECE N=${W} sweep (accel=${ACCEL}, conc=${CONCURRENCY}) ====="
    "$PY" scripts/sweeps/run_sweep_local.py \
      --base-config "$LAYERS_CFG" \
      --num-layers 13 --model-tag CLaMP3-symbolic-abc --task-tag "${TASK_BASE}N${W}" \
      --accelerator "$ACCEL" --concurrency "$CONCURRENCY" \
      --skip-fit-if-no-train --skip-meanall \
      --run-name-suffix "window${W}" "${EXTRA_LAYERS[@]}"
    echo "===== ${ARM} WITHIN-PIECE N=${W} sweep complete ====="
  done
done

echo "===== WITHIN-PIECE window-size sweep complete ====="
