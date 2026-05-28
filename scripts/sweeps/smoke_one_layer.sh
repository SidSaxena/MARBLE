#!/usr/bin/env bash
#
# scripts/sweeps/smoke_one_layer.sh — single-layer smoke test for the
# zero-shot retrieval sweep, with the launcher patches that bypass two
# Windows-specific issues:
#
#   1. --skip-fit-if-no-train → skips the cli.py fit no-op stage for
#      max_epochs=0 configs (a no-op that still pays full setup cost
#      and on Windows num_workers>0 deadlocks at worker spawn).
#
#   2. --num-workers ${NUM_WORKERS:-0} → forces a known-working worker
#      count regardless of what the config says. Default 0 because on
#      this Windows + 9700X + RTX 5060 Ti box we've observed:
#        - num_workers=0: 50 it/s on cache-hit-only test_dataloader
#        - num_workers=8: deadlock at LOCAL_RANK before iteration starts
#        - num_workers=4/6: untested in clean state, may work after
#          reboot — run with `NUM_WORKERS=6 ./smoke_one_layer.sh` to
#          test systematically.
#
# Picks MERT-v1-95M × VGMIDITVar-timbre × layer 0 as the smoke target
# — same fixture that crashed yesterday, so completion verifies the
# whole stack end-to-end. ~14:25 wall-clock with --num-workers=0 vs
# yesterday's ~63 min.
#
# Usage:
#   bash scripts/sweeps/smoke_one_layer.sh              # num_workers=0
#   NUM_WORKERS=4 bash scripts/sweeps/smoke_one_layer.sh # try 4
#   WANDB_MODE=offline bash scripts/sweeps/smoke_one_layer.sh  # local wandb only
#
# Reads the metric values from the log at the end; non-zero exit = fail.

set -uo pipefail
cd "$(dirname "$0")/../.."

NUM_WORKERS="${NUM_WORKERS:-0}"
LAYER="${LAYER:-0}"
LOG="${LOG:-tmp/smoke_layer${LAYER}_nw${NUM_WORKERS}.log}"

mkdir -p tmp
echo "smoke: layer=${LAYER}  num_workers=${NUM_WORKERS}  log=${LOG}"
date | tee "${LOG}"

uv run python scripts/sweeps/run_sweep_local.py \
  --base-config configs/probe.MERT-v1-95M-layers.VGMIDITVar-timbre.yaml \
  --num-layers 13 \
  --model-tag MERT-v1-95M \
  --task-tag VGMIDITVar-timbre \
  --layers "${LAYER}" \
  --no-skip \
  --skip-meanall \
  --skip-fit-if-no-train \
  --num-workers "${NUM_WORKERS}" \
  --accelerator gpu \
  --precision bf16-mixed \
  2>&1 | tee -a "${LOG}"

rc=${PIPESTATUS[0]}
echo "---"
echo "smoke done (exit=${rc})  log=${LOG}"
echo "key metrics:"
grep -E "CoverRetrieval|map_same|map_cross|MAP \(|median_rank|condition grid" "${LOG}" | head -20
exit "${rc}"
