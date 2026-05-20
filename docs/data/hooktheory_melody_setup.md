# HookTheoryMelody — data setup runbook

End-to-end commands for getting HookTheoryMelody training data onto a
machine + (optionally) to Modal. Includes the optional yt-dlp recovery
path for the ~5K songs missing from the HuggingFace dataset.

## Prerequisites (one-time)

```bash
# 1. Request access to the gated dataset (may take a day to approve):
#    https://huggingface.co/datasets/m-a-p/HookTheory

# 2. Authenticate the HuggingFace CLI:
huggingface-cli login

# 3. Disk budget:
#    - Tars:  104 GB (deletable after extraction)
#    - Audio: ~100 GB (10,061 mp3s as of 2026-05)
#    - With yt-dlp recovery: +25-50 GB
#    Plan for 200 GB working set; 90 GB final after cleanup.
```

## Step 1 — Download from HuggingFace

Pulls the gated `m-a-p/HookTheory` repo and extracts the full-song audio.
Resumable: re-runs skip files already on disk.

```bash
uv run python scripts/data/download_hooktheory.py --data-dir data --with-full-audio
```

Outputs:
- `data/HookTheory/audio/<ytid>.mp3` — extracted mp3s (~10,061 files)
- `data/HookTheory/zips/audio/part_*.tar` — original tar parts (deletable later)
- `data/HookTheory/hooktheory_clips/` — clips for Key/Structure tasks
- `data/HookTheory/Hooktheory.json.gz` — source metadata

## Step 2 — Build the per-split JSONL

```bash
uv run python scripts/data/build_hooktheory_melody_jsonl.py \
    --audio-dir data/HookTheory/audio \
    --filter-by-audio \
    --out-dir data/HookTheory
```

Outputs `HookTheory.{train,val,test}.jsonl`. Expected: ~14,993 records
spanning ~8,907 unique YouTube videos (multiple HookTheory annotations
per song are common).

## Step 3 (optional) — yt-dlp recovery of missing songs

The HF dataset ships only ~10K mp3s; ~5K more songs have annotations
but their audio is gone from YouTube / wasn't fetchable at scrape time.
Recovers ~50% of those (~2.5K extra songs ≈ ~4.2K extra records).
Skip if you don't need the lift.

### 3a — Identify which ytids are missing

```bash
uv run python -c "
import gzip, json
from pathlib import Path
src = Path.home() / '.cache/huggingface/hub/datasets--m-a-p--HookTheory/snapshots'
src = next(src.iterdir()) / 'Hooktheory.json.gz'
data = json.loads(gzip.decompress(src.read_bytes()).decode())
all_ytids = set()
for hid, song in data.items():
    if not song.get('annotations', {}).get('melody'): continue
    yt = song.get('youtube', {}).get('id')
    if yt: all_ytids.add(yt)
have = {p.stem for p in Path('data/HookTheory/audio').glob('*.mp3')}
missing = all_ytids - have
out = Path('data/HookTheory/missing_ytids.txt')
out.write_text('\n'.join(sorted(missing)))
print(f'{len(missing):,} ytids written to {out}')
"
```

### 3b — Convert to URLs (yt-dlp's `--batch-file` wants URLs)

```bash
uv run python -c "
from pathlib import Path
ids = [i.strip() for i in Path('data/HookTheory/missing_ytids.txt').read_text().strip().split('\n') if i.strip()]
urls = [f'https://www.youtube.com/watch?v={i}' for i in ids]
Path('data/HookTheory/missing_urls.txt').write_text('\n'.join(urls))
print(f'wrote {len(urls):,} URLs')
"
```

### 3c — Download with yt-dlp

Resumable via `--download-archive`. ~10 hours wall-clock for 5K videos
with the 2-5s rate-limit sleeps.

```bash
uv run yt-dlp \
    --batch-file data/HookTheory/missing_urls.txt \
    --output "data/HookTheory/audio/%(id)s.%(ext)s" \
    --extract-audio --audio-format mp3 --audio-quality 0 \
    --no-overwrites \
    --ignore-errors \
    --sleep-interval 2 --max-sleep-interval 5 \
    --concurrent-fragments 4 \
    --no-warnings \
    --cookies-from-browser firefox \
    --download-archive data/HookTheory/yt_dlp_archive.txt \
    2>&1 | tee data/HookTheory/yt_dlp.log
```

Notes:
- `--cookies-from-browser firefox` — boosts success rate on
  age-restricted videos. Swap to `chrome` / `edge` if Firefox isn't
  logged into YouTube, or drop the flag entirely.
- `--ignore-errors` — keeps the batch alive across per-video failures.
- Ctrl-C + re-run is safe; the archive file is the resume marker.

### 3d — Rebuild the JSONL to include the recovered mp3s

```bash
uv run python scripts/data/build_hooktheory_melody_jsonl.py \
    --audio-dir data/HookTheory/audio \
    --filter-by-audio \
    --out-dir data/HookTheory
```

Expect "Skipped (no audio)" to drop by roughly 2× the yt-dlp success
count, and "Output records" to climb accordingly.

## Step 4 — Clean up disk

After Step 2 (or Step 3d) succeeds, two safe deletions:

```bash
# Tars are extracted — frees ~104 GB:
rm -rf data/HookTheory/zips

# Orphan mp3s: files for songs without melody annotation or split
# assignment. Freed ~12 GB at original 10,061-mp3 baseline; less if
# you ran the yt-dlp recovery.
uv run python -c "
import json
from pathlib import Path
used = set()
for split in ['train', 'val', 'test']:
    with open(f'data/HookTheory/HookTheory.{split}.jsonl') as f:
        for line in f:
            ytid = json.loads(line).get('youtube', {}).get('id')
            if ytid: used.add(ytid)
orphans = [p for p in Path('data/HookTheory/audio').glob('*.mp3') if p.stem not in used]
size_gb = sum(p.stat().st_size for p in orphans) / (1024**3)
print(f'  deleting {len(orphans):,} orphan mp3s ({size_gb:.1f} GB)')
for p in orphans: p.unlink()
"
```

## Step 5 — Upload to Modal (`marble-data` volume)

Costs Modal credits proportional to bandwidth. Skip if you only need
the data on one local machine.

```bash
# Required: audio + the 3 melody JSONLs
modal volume put marble-data data/HookTheory/audio HookTheory/audio
for split in train val test; do
  modal volume put marble-data data/HookTheory/HookTheory.$split.jsonl \
                                HookTheory/HookTheory.$split.jsonl
done

# Optional: source metadata (~20 MB) so the JSONL can be rebuilt
# on the volume without re-downloading from HF.
modal volume put marble-data data/HookTheory/Hooktheory.json.gz \
                              HookTheory/Hooktheory.json.gz

# Optional: clips + JSONLs for HookTheoryKey + HookTheoryStructure (~4 GB).
modal volume put marble-data data/HookTheory/hooktheory_clips \
                              HookTheory/hooktheory_clips
for f in HookTheoryKey HookTheoryStructure; do
  for split in train val test; do
    modal volume put marble-data data/HookTheory/$f.$split.jsonl \
                                  HookTheory/$f.$split.jsonl
  done
done
```

## Step 6 — Download from Modal (on a different machine)

If the data is already on `marble-data` and you want to sync it to a
fresh machine:

```bash
mkdir -p data/HookTheory
modal volume get marble-data HookTheory/audio data/HookTheory/audio
for split in train val test; do
  modal volume get marble-data HookTheory/HookTheory.$split.jsonl \
                                data/HookTheory/HookTheory.$split.jsonl
done
```

## Alternative — download directly on Modal (skips local-disk + upload step)

If you'd rather pay Modal's compute for the HF download than upload
from local, use the existing Modal function (only run on a machine
with credits available):

```bash
modal run modal_marble.py::setup_hooktheory_full
```

This does Steps 1-2 server-side, writing to `marble-data:HookTheory/`.
Note: this path does **not** run the yt-dlp recovery (Step 3).

## Running sweeps

```bash
# Smoke check — list what would run:
uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryMelody --dry-run

# meanall only across the 3 priority encoders (recommended first):
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks HookTheoryMelody --models meanall

# Full per-layer sweep (3 encoders × 13/24 layers):
uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryMelody

# One encoder only (e.g. MuQ first while OMARRQ + MERT queue):
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks HookTheoryMelody --models MuQ
```

CLaMP3 is intentionally not in the priority encoder list — its audio
token rate is 1 Hz (1 frame per second), too coarse for frame-level
pitch prediction at the 25-75 Hz the other encoders provide. See
`docs/vgm_encoder_selection.md` for the analysis.
