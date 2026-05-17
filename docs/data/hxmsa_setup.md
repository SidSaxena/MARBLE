# HXMSA (Harmonix Set Music Structure Analysis) — dataset setup

Standalone runbook for building the HXMSA dataset that the MARBLE HXMSA
probes consume.

**Wall-clock budget:** ~3–6 h for yt-dlp downloads (rate-limited by
YouTube) + ~20 min for ffmpeg segment slicing. Plan to do this overnight
on the first run.

**Disk budget:** ~5.5 GB total (~4.6 GB full-track FLACs + ~0.9 GB
per-segment FLACs + ~10 MB upstream repo). Pass `--cleanup-full-tracks`
to drop the full-tracks dir after slicing if disk is tight.

---

## What you get

912 Western pop tracks from the Harmonix Set (Nieto et al. ISMIR 2019),
each with professional functional segment annotations. After our
pipeline:

- ~9,000 per-segment FLAC files (24 kHz mono) at
  `data/HXMSA/segments/<file_id>/<seg_idx>_<label>.flac`
- Three JSONL splits at `data/HXMSA/HXMSA.{train,val,test}.jsonl`
  (80/10/10 by track ID, seed 1234) — one record per segment.
- 13 functional classes after dropping the `end` terminator and
  merging `instrumental` → `inst` (see the paper's §3.3 note and
  `scripts/data/build_hxmsa_dataset.py:RAW_TO_CANONICAL`).

---

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| **yt-dlp** | Download per-track audio from YouTube | `python -m yt_dlp -U` (already a project dep via `uv sync`) |
| **ffmpeg** | Segment slicing + metadata probe | macOS: `brew install ffmpeg`; Linux: `sudo apt install ffmpeg`; Windows: `winget install Gyan.FFmpeg` |
| **git** | Clone the upstream annotation repo | already required for project clone |

### YouTube cookies (strongly recommended)

The yt-dlp `android_vr` client we use can usually download without
sign-in, but a small fraction of tracks (~5–10%) hit bot-check or
age-gate walls. Exporting cookies once removes both issues:

```bash
# One-time: export Firefox cookies to a portable file
python -m yt_dlp --cookies-from-browser firefox \
                 --cookies cookies.txt \
                 --skip-download \
                 "https://youtube.com/watch?v=iBHNgV6_znU"
```

Then pass `--cookies-file cookies.txt` to the build script. Same as the
SHS100K workflow — same caveats: close Firefox before exporting to
avoid the cookie-db lock error.

---

## Build (the canonical command)

```bash
# Pilot first — smoke-test on 5 tracks (~5 min total)
uv run python scripts/data/build_hxmsa_dataset.py --max-tracks 5

# Full build (run overnight)
uv run python scripts/data/build_hxmsa_dataset.py \
    --cookies-file cookies.txt \
    --workers 2
```

What happens:

1. **Clones** the upstream `urinieto/harmonixset` repo (~10 MB) into
   `data/HXMSA/_upstream/`. Re-runs `git pull` if already present.
2. **Parses** `dataset/metadata.csv` and `dataset/youtube_urls.csv` to
   get the (file_id → YouTube URL) and (file_id → metadata) maps.
3. **Downloads** audio for each track via yt-dlp (multi-client
   fallback for format-gated videos; cookies for bot-check bypass).
   Stores at `data/HXMSA/full_tracks/<youtube_id>.<ext>` (typically
   .m4a or .webm).
4. **Parses** each track's segment annotation file
   (`dataset/segments/<file_id>.txt`). Space-separated `<timestamp>
   <label>` rows. Maps raw labels to the 13-class canonical inventory
   (see "Label inventory" below). Drops the `end` terminator.
5. **Slices** per segment via `ffmpeg -ss <start> -t <dur> -ar 24000
   -ac 1 -c:a flac`. Outputs `data/HXMSA/segments/<file_id>/<seg_idx
   :03d>_<label>.flac`.
6. **Splits** track IDs 80/10/10 with seed 1234 (deterministic;
   reproducible across machines).
7. **Emits** `data/HXMSA/HXMSA.{train,val,test}.jsonl`. Each JSONL
   record has: `audio_path`, `ori_uid`, `work_id`, `label`,
   `seg_idx`, `seg_start`, `seg_end`, `duration`, `sample_rate`,
   `num_samples`, `channels`, `title`, `artist`, `genre`.

### Render-plan preamble

Before the slow loop the script prints a planner block:

```
────────────────────────────────────────────────────────────
Build plan
────────────────────────────────────────────────────────────
  data-dir         : data/HXMSA
  audio-dir        : data/HXMSA/full_tracks
  segments-dir     : data/HXMSA/segments
  candidate tracks : 912  (912 have annotations, 912 have URLs, 912 have metadata)
  audio present    : 0
  audio to fetch   : 912
  workers          : 2
  cookies          : file=cookies.txt
  segment slicing  : enabled
  target sr        : 24000 Hz (mono FLAC)
────────────────────────────────────────────────────────────
```

If "audio to fetch" is 0 when you expected 912, or if "candidate tracks"
is suspiciously low, abort and check the dependency / dir paths
before the slow loop starts.

---

## Idempotency + resume

- Existing downloaded audio files are skipped (`_find_audio` check).
- Existing per-segment FLACs are skipped (size-aware existence check).
- The JSONL is always rewritten from the current on-disk state — so a
  resumed build with more tracks just produces a fresh JSONL covering
  everything that's present.

To force re-render existing audio (e.g., after a yt-dlp update):
delete `data/HXMSA/full_tracks/` first, then re-run.

To force re-slice (e.g., changed `--target-sr`):
delete `data/HXMSA/segments/` first.

---

## Disk-tight mode

Drop full-track audio after segmentation:

```bash
uv run python scripts/data/build_hxmsa_dataset.py \
    --cookies-file cookies.txt \
    --cleanup-full-tracks
```

Reduces total footprint from ~5.5 GB to ~1 GB. Trade-off: you can't
re-slice without re-downloading.

---

## Label inventory

13 canonical classes after two adjustments to the native 15-label
Harmonix vocabulary:

| Drop | Reason |
|---|---|
| `end` | Terminator timestamp marking file end, not a section label |

| Merge | Reason |
|---|---|
| `instrumental` → `inst` | Paper's §3.3 explicitly notes these are the same word, repeated by accident |

| Final 13 classes (alphabetical) |
|---|
| `break`, `bridge`, `chorus`, `inst`, `intro`, `other`, `outro`, `postchorus`, `prechorus`, `silence`, `solo`, `transition`, `verse` |

Class indices are alphabetical 0..12 — see
`marble/tasks/HXMSA/datamodule.py:LABEL2IDX`. The same order is mirrored
in `scripts/data/build_hxmsa_dataset.py:CANONICAL_LABELS`; the two
**must** stay in sync.

### Class imbalance

The distribution is long-tailed: `chorus` is ~60× more common than
`silence`. Two things matter:

1. **Use macro-F1 alongside accuracy.** Accuracy can be high even when
   the minority classes are fully missed. The configs log both. Watch
   `val/macro_f1` during training.
2. **Class-weighted loss is a future option.** If specific minority
   classes look collapsed in the per-class confusion matrix after
   training, swap `CrossEntropyLoss` for `CrossEntropyLoss(weight=...)`
   with inverse-frequency weights. Not done by default — start with
   the unweighted run.

---

## Running the layer sweeps

After build completes:

```bash
# Verify data is detected by the sweep runner
uv run python scripts/sweeps/run_all_sweeps.py --tasks HXMSA --dry-run

# Meanall first — gives a baseline number in <30 min × 4 encoders
uv run python scripts/sweeps/run_all_sweeps.py --tasks HXMSA --only-meanall

# Full per-layer sweep
uv run python scripts/sweeps/run_all_sweeps.py --tasks HXMSA
```

Active encoders: CLaMP3, MERT-v1-95M, MuQ, OMARRQ-multifeature-25hz.
CLaMP3-symbolic is deliberately excluded — Harmonix has no MIDI
distribution. MERT-v1-330M + MusicFM stay decommissioned per the prior
sweep-registry decision.

---

## Verification before launching the sweep

| Check | How |
|---|---|
| JSONL has ~9,000 records | `wc -l data/HXMSA/HXMSA.*.jsonl` |
| Class distribution roughly matches paper Fig. 5 | `jq -r .label data/HXMSA/HXMSA.train.jsonl \| sort \| uniq -c \| sort -rn` |
| No track appears in multiple splits | `jq -r .work_id data/HXMSA/HXMSA.*.jsonl \| sort -u \| wc -l` should equal the sum of `wc -l` per split for unique track-ids |
| Configs parse cleanly | `uv run python cli.py test -c configs/probe.MuQ-layers.HXMSA.yaml --print_config \| head -20` |
| Cache audit passes | `uv run python scripts/embeddings/audit_cache_integration.py` reports 175/175 |
| Sweep planner sees data | `uv run python scripts/sweeps/run_all_sweeps.py --tasks HXMSA --dry-run` shows `✓ data` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `git clone` of harmonixset fails | Network / firewall | Clone manually then pass `--harmonixset-dir /path/to/harmonixset` |
| Many `[bot-check]` warnings from yt-dlp | YouTube requires sign-in for those tracks | Export cookies (see Prerequisites) and pass `--cookies-file` |
| Many `[format-gated]` warnings | Specific YouTube videos region-locked or token-gated | Some tracks unavoidable; the build proceeds with whatever does download |
| `[timeout]` errors | Slow network or large videos | Already at 180 s timeout per video; check connection |
| `Unknown label` raised during build | Annotation file has a label outside the canonical map | Add a defensive alias to `RAW_TO_CANONICAL` in the build script and re-run with `--skip-download --skip-slice` |
| Audio downloaded but ffmpeg slice fails | Source format not seekable | Reportedly rare; if frequent, switch to two-pass ffmpeg (decode-then-slice) — file an issue |
| Disk fills mid-build | Underestimated WAV size (rare with FLAC) | Re-run with `--cleanup-full-tracks` |
| Datamodule raises `Unknown label: …` at training start | Stale JSONL from before a label-map change | Re-run build with `--skip-download --skip-slice` to regenerate the JSONL with the current canonical map |

---

## Files this dataset touches

- `data/HXMSA/_upstream/harmonixset/` — clone of the upstream repo (~10 MB)
- `data/HXMSA/full_tracks/` — yt-dlp downloads (~4.6 GB)
- `data/HXMSA/segments/<file_id>/` — per-segment FLACs (~0.9 GB)
- `data/HXMSA/HXMSA.{train,val,test}.jsonl` — annotation files
- `marble/tasks/HXMSA/{datamodule,probe}.py` — Lightning task
- `configs/probe.<encoder>-{layers,meanall}.HXMSA.yaml` — 8 sweep configs
- `output/.emb_cache/<encoder>/HXMSA__<hash>/` — per-clip embedding cache
  (auto-populated during sweeps)

---

## References

- Nieto, McCallum, Davies, Robertson, Stark, Egozy. *"The Harmonix Set:
  Beats, Downbeats, and Functional Segment Annotations of Western
  Popular Music."* ISMIR 2019.
  [Paper](https://archives.ismir.net/ismir2019/paper/000068.pdf) ·
  [Repo](https://github.com/urinieto/harmonixset)
- Companion docs:
  [`docs/data/vgmiditvar_setup.md`](vgmiditvar_setup.md) (same yt-dlp
  + ffmpeg pattern for VGMIDI-TVar),
  [`docs/embedding_cache.md`](../embedding_cache.md),
  [`docs/layer_analysis.md`](../layer_analysis.md).
