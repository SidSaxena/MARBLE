#!/bin/bash
# cleanup_mert330m.sh
# Delete checkpoints and embedding cache for MERT-v1-330M while preserving logs
#
# Usage:
#   bash cleanup_mert330m.sh                  # dry-run (show what would be deleted)
#   bash cleanup_mert330m.sh --apply          # actually delete

ENCODER_PATTERN="MERT-v1-330M"
MODE="${1:---dry-run}"

echo "════════════════════════════════════════════════════════════════════════"
echo "  MERT-v1-330M Cleanup"
echo "════════════════════════════════════════════════════════════════════════"
echo "Pattern: $ENCODER_PATTERN"
echo "Mode:    $MODE"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Find checkpoints
# ─────────────────────────────────────────────────────────────────────────────

echo "📋 Checkpoints to delete:"
CKPT_DIRS=()
while IFS= read -r dir; do
  if [ -d "$dir" ]; then
    SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "   • $dir ($SIZE)"
    CKPT_DIRS+=("$dir")
  fi
done < <(find ./output -type d -name "checkpoints" -path "*${ENCODER_PATTERN}*" 2>/dev/null)

if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
  echo "   (none found)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Find embedding cache
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "📋 Embedding cache to delete:"
CACHE_DIRS=()
while IFS= read -r dir; do
  if [ -d "$dir" ]; then
    SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "   • $dir ($SIZE)"
    CACHE_DIRS+=("$dir")
  fi
done < <(find ./output/.emb_cache -maxdepth 1 -type d -name "*${ENCODER_PATTERN}*" 2>/dev/null)

if [ ${#CACHE_DIRS[@]} -eq 0 ]; then
  echo "   (none found)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Show logs (preserving)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "📋 Logs (PRESERVING):"
LOG_COUNT=0
while IFS= read -r file; do
  if [ -f "$file" ]; then
    SIZE=$(du -sh "$file" 2>/dev/null | cut -f1)
    echo "   • $file ($SIZE)"
    ((LOG_COUNT++))
  fi
done < <(find ./output/logs -type f -name "*${ENCODER_PATTERN}*" 2>/dev/null)

if [ $LOG_COUNT -eq 0 ]; then
  echo "   (none found)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════════"
TOTAL_SIZE=$(du -sh output/.emb_cache/*${ENCODER_PATTERN}* ./output/*/*${ENCODER_PATTERN}*/checkpoints 2>/dev/null | tail -1 | cut -f1)
echo "Total to delete: ~$TOTAL_SIZE"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$MODE" = "--dry-run" ]; then
  echo "✓ DRY RUN — nothing deleted"
  echo ""
  echo "To actually delete, run:"
  echo "  bash cleanup_mert330m.sh --apply"
  echo ""
  exit 0
fi

if [ "$MODE" != "--apply" ]; then
  echo "❌ Unknown mode: '$MODE'"
  echo "Use: --dry-run (default) or --apply"
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# DESTRUCTIVE: Actually delete
# ─────────────────────────────────────────────────────────────────────────────

echo "⚠️  DELETING..."
echo ""

# Delete checkpoints
for dir in "${CKPT_DIRS[@]}"; do
  echo "🗑️  rm -rf '$dir'"
  rm -rf "$dir"
done

# Delete cache directories
for dir in "${CACHE_DIRS[@]}"; do
  echo "🗑️  rm -rf '$dir'"
  rm -rf "$dir"
done

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "✅ CLEANUP COMPLETE"
echo "════════════════════════════════════════════════════════════════════════"
echo "Deleted ${#CKPT_DIRS[@]} checkpoint dirs + ${#CACHE_DIRS[@]} cache dirs"
echo "Logs preserved at: ./output/logs/"
echo ""
