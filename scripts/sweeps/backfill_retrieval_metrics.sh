#!/usr/bin/env bash
# scripts/sweeps/backfill_retrieval_metrics.sh
#
# Backfill the new retrieval metrics (Recall@K, Hit Rate@K, median rank,
# R-Precision, cross-condition MAP, anisotropy diagnostics) onto
# already-completed per-layer sweeps via `--retest` on run_sweep_local.py.
#
# Every backfill run gets the wandb tag `backfill-recall` so the dashboard
# can filter / group the new runs apart from the original `layer-N-test`
# runs. The original wandb summaries are NOT modified — fresh runs are
# created.
#
# Scope (current): Covers80 × 4 audio encoders × per-layer = 63 layers.
# Cost estimate: with N=160 and (if `cache_embeddings: true` is set
# in the configs) cache-warm by the second layer, ~1-3 min/layer. Total
# ~1-1.5 h. Without cache (fresh encoder forward each layer), ~30-50 min
# per encoder. Either way, tractable.
#
# Deferred (audio not currently on this box):
#   - SHS100K  × 4 encoders         (~63 layers)
#   - VGMIDITVar (base)             (~76 layers, with CLaMP3-symbolic)
#   - VGMIDITVar-multisf            (~63 layers, needs JSONL regen first)
#   - VGMIDITVar-leitmotif          (~76 layers, with CLaMP3-symbolic)
#
# Add those when storage is freed + audio restored.
#
# Usage:
#   bash scripts/sweeps/backfill_retrieval_metrics.sh
#   # OR target one specific encoder:
#   bash scripts/sweeps/backfill_retrieval_metrics.sh MuQ
#
# Resume-safe: each layer-task-encoder triple writes its own wandb run; if
# the script is interrupted, re-running picks up where it left off (no
# completion-check beyond what run_sweep_local does internally via --retest).

set -euo pipefail
cd "$(dirname "$0")/../.."

BACKFILL_TAG="backfill-recall"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

# Filter: optional first arg picks one encoder. Default = all 4 audio
# encoders. (CLaMP3-symbolic does NOT apply to Covers80 — it's symbolic-
# only; not in this scope.)
ENCODER_FILTER="${1:-}"

# (config, model-tag, num-layers) tuples. Format: cfg|model|layers
declare -a COVERS80=(
  "configs/probe.MuQ-layers.Covers80.yaml|MuQ|13"
  "configs/probe.MERT-v1-95M-layers.Covers80.yaml|MERT-v1-95M|13"
  "configs/probe.OMARRQ-multifeature-25hz.Covers80.yaml|OMARRQ-multifeature-25hz|24"
  "configs/probe.CLaMP3-layers.Covers80.yaml|CLaMP3|13"
)

run_one() {
  local cfg="$1"
  local model="$2"
  local layers="$3"
  local task="Covers80"
  local log_file="${LOG_DIR}/backfill_${task}_${model}.log"

  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  Backfill: ${task} × ${model}  (${layers} layers)"
  echo "  Log:      ${log_file}"
  echo "════════════════════════════════════════════════════════════════"

  if [[ ! -f "$cfg" ]]; then
    echo "  ✗ SKIP: config not found: $cfg"
    return 0
  fi

  uv run python scripts/sweeps/run_sweep_local.py \
    --base-config "$cfg" \
    --num-layers "$layers" \
    --model-tag "$model" \
    --task-tag "$task" \
    --retest \
    --skip-meanall \
    --extra-tag "$BACKFILL_TAG" 2>&1 | tee -a "$log_file"
}

for entry in "${COVERS80[@]}"; do
  IFS='|' read -r cfg model layers <<< "$entry"
  if [[ -n "$ENCODER_FILTER" && "$model" != *"$ENCODER_FILTER"* ]]; then
    continue
  fi
  run_one "$cfg" "$model" "$layers"
done

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Backfill complete (Covers80 only)."
echo "  Logs in: ${LOG_DIR}/backfill_Covers80_*.log"
echo "  Filter wandb dashboard by tag: ${BACKFILL_TAG}"
echo "════════════════════════════════════════════════════════════════"
