#!/usr/bin/env bash
# iter_postprocess_subset.sh
# ──────────────────────────
# Iterate on reverb / loudness configs for VGMIDITVar-timbre on a small
# 16-file subset (Mipha's Theme + Pokemon Sky Tower, theme variation
# only, 8 GM programs each). Produces 4 sibling output dirs you can
# A/B listen to.
#
# Run on my-pc:
#   bash scripts/data/iter_postprocess_subset.sh
#
# Output layout (under data/VGMIDITVar-timbre-iter/):
#   src/                ← clean source copies of the 16 chosen FLACs
#   config_A_loudonly/  ← no reverb, just loudness norm
#   config_B_plate5/    ← Bricasti Vocal Plate, ~5% wet
#   config_C_room5/     ← Bricasti Small Room, ~5% wet
#   config_D_chamber3/  ← Bricasti Small Chamber, ~3% wet
#
# After listening, pick a config; the final full-corpus pass uses the
# same --ir / --wet-db settings as that config.

set -euo pipefail
cd ~/Developer/Python/marble

# ── 1. Set up the 16-file subset under a clean src dir ─────────────────────
SUBSET=data/VGMIDITVar-timbre-iter/src
mkdir -p "$SUBSET"
echo "── Copying 16 chosen FLACs to $SUBSET ──"
for prog in 0 24 48 52 60 73 80 89; do
    for stem in "e0_real_The Legend of Zelda_Multiplatform_The Legend of Zelda Breath of the Wild_Miphas Theme_A_0" \
                "e0_real_Pokemon Mystery Dungeon_Multiplatform_Pokemon Mystery Dungeon Red Rescue Team & Pokemon Mystery Dungeon Blue Rescue Team_Sky Tower_A_0"; do
        src="data/VGMIDITVar-timbre/audio/${stem}_p${prog}.flac"
        if [ -f "$src" ]; then
            cp -n "$src" "$SUBSET/"
        else
            echo "  !! missing $src"
        fi
    done
done
echo "  subset has $(ls "$SUBSET" | wc -l) files"

# ── 2. Downmix the 3 needed Bricasti IRs to mono ───────────────────────────
mkdir -p data/ir
IR_DIR="/c/Users/Sid/Downloads/Samplicity - Bricasti IRs version 2023-10/Samplicity - Bricasti IRs version 2023-10, left-right files, 44.1 Khz"
for src_name in \
    "2 Plates 06 Vocal Plate, 44K L.wav:vocal_plate" \
    "3 Rooms 27 Small Room, 44K L.wav:small_room" \
    "4 Chambers 03 Small Chamber, 44K L.wav:small_chamber"; do
    src_file="${src_name%%:*}"
    short="${src_name##*:}"
    dst="data/ir/bricasti_m7_${short}_mono.wav"
    if [ ! -f "$dst" ]; then
        echo "── Downmixing $short ──"
        ffmpeg -y -i "$IR_DIR/$src_file" -ac 1 -ar 44100 -c:a pcm_s24le "$dst" 2>&1 | tail -2
    fi
done

# ── 3. Run the 4 configs in parallel-ish (each is fast for 16 files) ───────
PP="uv run python scripts/data/postprocess_vgmiditvar.py --src-dir $SUBSET --workers 4 --force"

echo ""
echo "── A: no reverb, loudness only ──"
$PP --dst-dir data/VGMIDITVar-timbre-iter/config_A_loudonly

echo ""
echo "── B: Vocal Plate, --wet-db=-3 (~5% wet) ──"
$PP --ir data/ir/bricasti_m7_vocal_plate_mono.wav --wet-db -3 \
    --dst-dir data/VGMIDITVar-timbre-iter/config_B_plate5

echo ""
echo "── C: Small Room, --wet-db=-3 (~5% wet) ──"
$PP --ir data/ir/bricasti_m7_small_room_mono.wav --wet-db -3 \
    --dst-dir data/VGMIDITVar-timbre-iter/config_C_room5

echo ""
echo "── D: Small Chamber, --wet-db=-6 (~3% wet) ──"
$PP --ir data/ir/bricasti_m7_small_chamber_mono.wav --wet-db -6 \
    --dst-dir data/VGMIDITVar-timbre-iter/config_D_chamber3

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " Done. A/B-listen here:"
echo "   data/VGMIDITVar-timbre-iter/src/                ← original (dry)"
echo "   data/VGMIDITVar-timbre-iter/config_A_loudonly/  ← no reverb"
echo "   data/VGMIDITVar-timbre-iter/config_B_plate5/    ← Vocal Plate 5% wet"
echo "   data/VGMIDITVar-timbre-iter/config_C_room5/     ← Small Room 5% wet"
echo "   data/VGMIDITVar-timbre-iter/config_D_chamber3/  ← Small Chamber 3% wet"
echo "════════════════════════════════════════════════════════════════════════"
