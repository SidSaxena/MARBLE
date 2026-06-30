# Impulse responses (`data/ir/`)

Convolution-reverb IRs used to add realism to the **VGMIDITVar-timbre** rendered
audio benchmark (see `scripts/data/postprocess_vgmiditvar.py`).

## Production IR
- **`bricasti_m7_vocal_plate_mono.wav`** — the production IR for VGMIDITVar-timbre.
- `bricasti_m7_vocal_plate_44kL_orig.wav` — the unmodified source (left channel, 44.1 kHz).

**Source:** Samplicity "Bricasti M7" free IR pack (*version 2023-10, left-right files,
44.1 kHz*), preset **`2 Plates 06 Vocal Plate`** (left channel). Free download from Samplicity.

**Downmix command (original → mono):**
```bash
ffmpeg -y -i "2 Plates 06 Vocal Plate, 44K L.wav" \
    -ac 1 -ar 44100 -c:a pcm_s24le bricasti_m7_vocal_plate_mono.wav
```

## How it is applied
`scripts/data/postprocess_vgmiditvar.py` runs, per rendered clip, a single ffmpeg pass of
**convolution reverb (`afir`) + EBU R128 loudness normalisation**:
```
[0:a][1:a]afir=dry=10:wet=0.5[wet];[wet]loudnorm=I=<LUFS>:TP=<TP>:LRA=<LRA>[out]
```
- `dry=10 wet=0.5` → **~5% wet** (mix = wet / (dry + wet)); linear gains in [0, 10], **not dB**.
- The **same IR is applied to every instrument**, so reverb character is held constant rather
  than a confound for the cross-timbre retrieval task.

## Alternates compared (not adopted)
The reverb config was picked by A/B over three Bricasti presets
(`scripts/data/iter_postprocess_subset.sh`):

| config | preset | wet |
|---|---|---|
| **B (adopted)** | `2 Plates 06 Vocal Plate` | ~5% (dry 10 / wet 0.5) |
| C | `3 Rooms 27 Small Room` | ~5% (dry 10 / wet 0.5) |
| D | `4 Chambers 03 Small Chamber` | ~3% (dry 10 / wet 0.3) |

The Small Room / Small Chamber mono IRs can be regenerated from the Samplicity pack with the
same `ffmpeg ... -ac 1` command.
