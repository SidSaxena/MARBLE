#!/usr/bin/env bash
# BPS-Motif WITHIN-PIECE WHOLE-PIECE-CONTEXT same-motif retrieval — ABC INPUT
# sweep: one single zero-shot test pool over all 32 movements x 13 CLaMP3 layers
# + a meanall baseline. The whole-piece-context counterpart of
# run_bps_within_piece_abc.sh (clip-isolated): SAME windows/labels/metric, but
# the probe encodes each WHOLE movement once at per-patch resolution (segmenting
# >512) and pools each window's bar-patches IN MOVEMENT CONTEXT. Zero-shot
# (max_epochs=0) so NO checkpoints are written. SINGLE-POOL (no fold loop): the
# metric is per-movement-gallery, so the whole corpus is one test pool.
#
# Build the WHOLE-movement dataset first:
#   uv run python scripts/data/build_bps_motif_within_piece.py --window 4 --whole
#
# Usage:
#   scripts/sweeps/run_bps_within_piece_whole_abc.sh [--accelerator mps|gpu|cpu] \
#       [--layers "0 1 ... 12"] [--concurrency N]
set -uo pipefail

ACCEL="gpu"
CONCURRENCY="4"
LAYERS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator) ACCEL="$2"; shift 2 ;;
    --layers) LAYERS_ARG="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
[[ "$ACCEL" == "mps" ]] && export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="$REPO/.venv/bin/python"
BASE="configs/probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPieceWholeN4.yaml"

EXTRA_LAYERS=()
[[ -n "$LAYERS_ARG" ]] && EXTRA_LAYERS=(--layers $LAYERS_ARG)

echo "===== WITHIN-PIECE-WHOLE-N4-ABC sweep (accelerator=${ACCEL}, concurrency=${CONCURRENCY}) ====="
# Single pool, all 13 layers + meanall (the sibling meanall config is auto-run
# unless --skip-meanall). --skip-fit-if-no-train: zero-shot, test only.
"$PY" scripts/sweeps/run_sweep_local.py \
  --base-config "$BASE" \
  --num-layers 13 --model-tag CLaMP3-symbolic-abc --task-tag BPSMotifWithinPieceWholeN4 \
  --accelerator "$ACCEL" --concurrency "$CONCURRENCY" --skip-fit-if-no-train \
  "${EXTRA_LAYERS[@]}"

echo "===== WITHIN-PIECE-WHOLE-N4-ABC sweep complete ====="
