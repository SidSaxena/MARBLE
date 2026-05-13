# VGMIDI-TVar dataset setup

Standalone runbook for building the rendered VGMIDI-TVar audio dataset
that the MARBLE VGMIDITVar probes consume.

**Run this on a machine with adequate disk (~5–10 GB free).**
The smoke-tested pipeline produces ~150 MB of WAV per 100 MIDIs, so the
full 12,870-MIDI render is on the order of **15–20 GB** of audio (lossless
WAV at 44.1 kHz stereo).  If you'd rather store compressed, see the
"Optional: re-encode to FLAC/MP3" section below.

---

## Prerequisites

| | |
|---|---|
| **Python environment** | The project venv via `uv sync` |
| **fluidsynth** binary on PATH | `winget install FluidSynth.FluidSynth` (Windows); `brew install fluid-synth` (macOS); `sudo apt install fluidsynth` (Linux) |
| **ffprobe** on PATH | Comes with ffmpeg; install ffmpeg if missing |
| **A General-MIDI SoundFont** (.sf2) | See "Getting a SoundFont" below |
| **Source zip** | `VGMIDI-TVar.zip` from https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model |

---

## Getting a SoundFont

The renderer needs at least one `.sf2` file.  Recommended: **FluidR3_GM**
(free, ~148 MB, classic General-MIDI coverage).

```bash
# Linux / macOS / WSL
mkdir -p ~/.local/share/sf2
curl -L -o ~/.local/share/sf2/FluidR3_GM.sf2 \
    "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2"

# Windows PowerShell
mkdir $env:USERPROFILE\sf2 -Force
Invoke-WebRequest `
    -Uri "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2" `
    -OutFile $env:USERPROFILE\sf2\FluidR3_GM.sf2
```

Verify with `file FluidR3_GM.sf2` — should report
`RIFF (little-endian) data, SoundFont/Bank`, not HTML.

### Alternative SoundFonts

If you already have one of these locally, point `--soundfont` at it:

- **MuseScore_General.sf2 / .sf3** — installed with MuseScore 4 at
  `/Applications/MuseScore 4.app/Contents/Resources/sound/`
  (macOS) or `C:\Program Files\MuseScore 4\sound\` (Windows)
- **GeneralUser GS** — http://schristiancollins.com/generaluser.php
- **Salamander Grand Piano** — piano-only, but VGMIDI is mostly piano so
  this is a strong match if you want smaller-footprint renders

You can pass multiple `--soundfont` flags; the script rotates through
them deterministically per piece to avoid timbre-overfitting:

```bash
python scripts/build_vgmiditvar_dataset.py \
    --soundfont /path/to/FluidR3_GM.sf2 \
    --soundfont /path/to/GeneralUser-GS.sf2 \
    --soundfont /path/to/SalamanderGrandPiano.sf2 \
    ...
```

---

## Building the dataset (the canonical command)

Download the source zip:

```
https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model/raw/main/dataset/VGMIDI-TVar.zip
```

Drop it somewhere accessible, e.g. `data/source/VGMIDI-TVar.zip`.

Then run:

```bash
python scripts/build_vgmiditvar_dataset.py \
    --midi-zip /path/to/VGMIDI-TVar.zip \
    --soundfont /path/to/FluidR3_GM.sf2 \
    --data-dir data/VGMIDITVar \
    --audio-dir data/VGMIDITVar/audio \
    --workers 8
```

What happens:

1. **Extracts** all `.mid` files from the zip into
   `data/VGMIDITVar/midi/{train,test}/*.mid` (~10 MB).
2. **Renders** each MIDI to mono 44.1 kHz WAV via fluidsynth
   (~1–2 s per file with 8 workers; the full 12,870 MIDIs take
   ~30–60 min on a modern CPU). Output goes to `--audio-dir`.
3. **Probes metadata** (sample rate, frames, channels) via ffprobe.
4. **Writes** `data/VGMIDITVar/VGMIDITVar.jsonl` — one record per render
   with `audio_path`, `work_id` (md5 of piece_id + section, groups
   theme + variations), `variation` (0 = theme, ≥1 = variation index),
   `piece_id`, `section`, `split` (train/test from the zip's directory
   structure), `sample_rate`, `num_samples`, `channels`, `duration`.

### Disk usage estimate

| Stage | Size |
|---|---|
| Source zip | ~10 MB |
| Extracted MIDIs | ~10 MB |
| Rendered WAVs (44.1 kHz stereo, raw) | **~15–20 GB** |
| JSONL | ~3 MB |
| **Total** | **~15–20 GB** |

If you don't have room for raw WAV, see the optional re-encoding step.

---

## Verifying the build

After the renderer finishes:

```bash
# JSONL exists and has roughly 12,500–12,870 entries (some MIDIs may
# fail to render; expect ~97% success)
wc -l data/VGMIDITVar/VGMIDITVar.jsonl

# Spot-check one rendered file
ls -la data/VGMIDITVar/audio | head -5
ffprobe data/VGMIDITVar/audio/<one-file>.wav 2>&1 | head -20

# Smoke-load via the datamodule
python -c "
from marble.tasks.VGMIDITVar.datamodule import VGMIDITVarAudioAll
ds = VGMIDITVarAudioAll(
    jsonl='data/VGMIDITVar/VGMIDITVar.jsonl',
    sample_rate=24000, channels=1, clip_seconds=15.0,
    min_clip_ratio=0.5, channel_mode='mix',
)
print(f'{len(ds)} clips from {len(ds.meta)} files')
wav, work_id, path = ds[0]
print(f'waveform shape={wav.shape}  work_id={work_id}')
"
```

Expected output: a 4-digit clip count, a (1, 360000) waveform tensor,
and a non-zero work_id.

---

## Running the layer sweeps

Once the JSONL + audio are in place, the four registered sweeps will
work:

```bash
python scripts/run_all_sweeps.py --tasks VGMIDITVar
```

This runs:
1. `CLaMP3 × VGMIDITVar`           (13 layers, audio path)
2. `MERT-v1-95M × VGMIDITVar`      (13 layers, audio path)
3. `OMARRQ-multifeature25hz × VGMIDITVar` (24 layers, audio path)
4. `CLaMP3-symbolic × VGMIDITVar`  (13 layers, MIDI native — no audio needed)

All four are zero-shot retrieval (`max_epochs=0`); each layer runs
test-only and reports MAP.  The symbolic sweep (#4) reads from
`data/VGMIDITVar/midi/<split>/<stem>.mid` (the extracted MIDIs from
step 1) so it works independently of the audio render.

If disk is *really* tight, you can run just the symbolic sweep:

```bash
python scripts/run_all_sweeps.py --tasks VGMIDITVar --models CLaMP3-symbolic
```

That only needs the ~10 MB of extracted MIDI; no audio render required.

---

## Optional: re-encode to FLAC to save space

After rendering, the WAVs can be compressed to FLAC losslessly,
reducing footprint by ~40–60%:

```bash
# Re-encode in place — back up first if you want
cd data/VGMIDITVar/audio
for f in *.wav; do
    ffmpeg -nostdin -loglevel error -y \
        -i "$f" -c:a flac "${f%.wav}.flac"
    rm "$f"
done

# Update JSONL paths to .flac
python -c "
import json
records = [json.loads(l) for l in open('data/VGMIDITVar/VGMIDITVar.jsonl')]
for r in records:
    r['audio_path'] = r['audio_path'].replace('.wav', '.flac')
with open('data/VGMIDITVar/VGMIDITVar.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
"
```

torchaudio reads FLAC natively, so the datamodule needs no changes.

---

## Cross-modal semantic test (after build)

Once the dataset exists, validate the CLaMP3 cross-modal API on real
data:

```bash
python scripts/test_clamp3_crossmodal_semantic.py \
    --jsonl data/VGMIDITVar/VGMIDITVar.jsonl \
    --midi-dir data/VGMIDITVar/midi \
    --num-pairs 5
```

Exit code 0 means same-MIDI audio scores higher than other-MIDI audio
in ≥4/5 trials — the cross-modal API is semantically working.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `fluidsynth not found` | Tool not on PATH | install via package manager (see Prerequisites) |
| `Not a SoundFont or MIDI file` on the .sf2 | Download returned HTML | re-download from a different mirror; verify `file foo.sf2` |
| Rate 0 files / sec, stuck | fluidsynth blocked on first call | check with `fluidsynth --version`; reinstall if needed |
| `Permission denied` on output dir | Permissions issue | `mkdir -p` and check writability |
| Renders all fail with same MIDI error | SoundFont missing instrument banks | try a different soundfont; FluidR3_GM has full GM coverage |
| Disk fills up mid-render | Underestimated WAV size | render in batches; consider FLAC reencoding (above) |
