# SuperMarioStructure — dataset setup

Standalone runbook for building the SuperMarioStructure dataset that
the MARBLE SuperMarioStructure probes consume.

**Status:** implemented (2026-05-17). Configs + datamodule + build
script all landed. You bring your own audio; the build script handles
the rest (annotation clone, MIDI download for bar→time mapping, ffmpeg
segment slicing, JSONL emission).

**Wall-clock budget:** ~5 min to clone annotations + download all 554
MIDIs, then ~10–20 min for ffmpeg segment slicing depending on
storage speed. Total: under 30 min once you have the audio.

**Disk budget:** ~0.5 GB total (~5 MB annotation repo + ~5 MB cached
MIDIs + ~0.4 GB per-segment FLACs). Your audio is read-only — we
don't copy or duplicate it.

---

## What you get

- 554 Super Mario pieces from the upstream
  [supermario-structure-annotation](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
  repo (Function-level annotations, ~3,500 segments after filtering).
- Per-segment FLAC files (24 kHz mono) at
  `data/SuperMarioStructure/segments/<piece_id>/<seg_idx>_<label>.flac`.
- Three JSONL splits at
  `data/SuperMarioStructure/SuperMarioStructure.{train,val,test}.jsonl`
  (70/15/15 by piece, honouring upstream split where present, seed 1234
  for the rest).
- 6 functional classes (VGM-native):

| Class | Raw code | Description |
|---|---|---|
| `intro` | `In` | Opening segment, distinct from main loop |
| `loop` | `Lp` | Main repeating section (dominant in most VGM) |
| `transition` | `Tr` | Connecting passage between sections |
| `bridge` | `Br` | Contrasting middle section |
| `outro` | `Ou` | Closing segment |
| `stinger` | `St` | Short punctuation cue (often <2 s) |

---

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| **ffmpeg** | Segment slicing + metadata probe | macOS: `brew install ffmpeg`; Linux: `sudo apt install ffmpeg`; Windows: `winget install Gyan.FFmpeg` |
| **git** | Clone the upstream annotation repo | already required |
| **pretty_midi** | Bar→time conversion via MIDI tempo events | already a MARBLE dep |

### User audio

**You must provide audio yourself** — the upstream repo doesn't
distribute it. The build script expects files named by `piece_id`
(zero-padded 5-digit) in your `--audio-dir`. Supported extensions:
`.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`, `.opus` (first match wins).

Example:

```
/my/audio/dir/
  00001.flac    ← Captain Toad - Retro RampUp
  00002.mp3     ← Dr Mario - Chill
  00003.wav     ← Dr Mario - Endings
  ...
```

Pieces without audio in the dir are skipped (with a list of the first
5 missing piece_ids logged for sanity).

### Critical assumption: tempo alignment

The build script extracts bar→time mapping from each piece's source
MIDI (auto-downloaded from NinSheetMusic). If your audio is performed
at a different tempo than the MIDI (e.g., human performance), segment
boundaries will drift. For MIDI-rendered audio (the expected v1 case),
the mapping is exact.

**Tested OK:** audio rendered via fluidsynth + SGM-Pro 14 from the
same source MIDIs (matches our VGMIDI pipeline).

**Likely OK with caveats:** professional VGM rips at the original game
tempo.

**Will drift:** human piano performances, tempo-modified arrangements.

---

## Build (the canonical command)

```bash
# Pilot first — smoke-test on 5 pieces (~30 s)
uv run python scripts/data/build_supermario_dataset.py \
    --audio-dir /path/to/your/audio --max-pieces 5

# Full build (under 30 min once audio is in place)
uv run python scripts/data/build_supermario_dataset.py \
    --audio-dir /path/to/your/audio
```

What happens:

1. **Clones** the upstream `ShxLuo-Saxon/supermario-structure-annotation`
   repo (~5 MB) into `data/SuperMarioStructure/_upstream/`. Re-runs
   `git pull` if already present.
2. **Parses** `metadata/pieces.csv` (piece IDs + MIDI URLs) and
   `metadata/pairs.csv` (upstream train/val/test split for the 334
   pieces covered by the similarity dataset; the build script honours
   these and assigns the remaining pieces randomly with seed 1234).
3. **Downloads** the source MIDI for each piece from NinSheetMusic
   (purely as the bar→time clock, not for audio). Cached at
   `data/SuperMarioStructure/midi/<piece_id>.mid`.
4. **For each piece with both audio + MIDI:**
   - Loads the MIDI via `pretty_midi`, gets downbeat times via
     `get_downbeats()` (handles tempo + time-signature changes).
   - Parses `annotations/<piece_id>.json`. Uses the `Function` array
     (coarse, 6-class). Skips the `Section` array (reserved for v2
     section-similarity task).
   - For each `(bar_start, bar_end, label)` triple, computes
     `(start_sec, end_sec)` from the MIDI downbeat table.
   - Slices user audio via
     `ffmpeg -ss <start> -t <dur> -ar 24000 -ac 1 -c:a flac`.
5. **Emits** per-segment JSONL records (audio_path, ori_uid, work_id,
   label, seg_idx, bar_start/end, seg_start/end, duration, sample_rate,
   num_samples, channels, title, ninsheetmusic_id).
6. **Splits** train/val/test at piece level (no segment leakage).

### Render-plan preamble

Before any slow work, the script prints:

```
────────────────────────────────────────────────────────────
Build plan
────────────────────────────────────────────────────────────
  audio-dir         : /path/to/your/audio
  data-dir          : data/SuperMarioStructure
  segments-dir      : data/SuperMarioStructure/segments
  midi-dir          : data/SuperMarioStructure/midi
  candidate pieces  : 554  (554 annotations available, 0 have no audio in --audio-dir)
  MIDI already cached  : 0
  MIDI to download now : 554
  segment slicing   : enabled
  target sr         : 24000 Hz (mono FLAC)
  upstream splits   : honoured for 334 pieces; seed-1234 70/15/15 for the rest
────────────────────────────────────────────────────────────
```

If "candidate pieces" is suspiciously low, check the audio-dir naming
convention before the slow loop starts.

---

## Idempotency + resume

- Existing audio files in `--audio-dir` are never modified.
- Existing per-segment FLACs are skipped (size-aware existence check).
- Existing source MIDIs are skipped.
- The JSONL is always rewritten from the current on-disk state.

To force re-slice (e.g., changed `--target-sr`): delete
`data/SuperMarioStructure/segments/` first.

---

## Running the layer sweeps

```bash
# Verify
uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure --dry-run

# Meanall first — baseline in <30 min × 4 encoders
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks SuperMarioStructure --only-meanall

# Full per-layer sweep
uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure
```

Active encoders: CLaMP3, MERT-v1-95M, MuQ, OMARRQ-multifeature-25hz.

**Symbolic note:** the source MIDIs are available via the cached
`data/SuperMarioStructure/midi/` directory, so a CLaMP3-symbolic
variant is possible as a follow-up. Not implemented in v1 — would need
a separate config that points at the MIDI dir instead of audio. The
symbolic comparison would be especially interesting given the strong
showing of CLaMP3-symbolic on VGMIDITVar / leitmotif tasks.

---

## Verification before launching the sweep

| Check | How |
|---|---|
| JSONL has ~3500 records | `wc -l data/SuperMarioStructure/SuperMarioStructure.*.jsonl` |
| Class distribution looks reasonable | `jq -r .label data/SuperMarioStructure/SuperMarioStructure.train.jsonl \| sort \| uniq -c \| sort -rn` — expect `loop` to dominate |
| No piece appears in multiple splits | `for f in data/SuperMarioStructure/SuperMarioStructure.*.jsonl; do jq -r .work_id $f \| sort -u; done \| sort \| uniq -c \| awk '$1>1 {print}'` — should be empty |
| Configs parse cleanly | `uv run python cli.py test -c configs/probe.MuQ-layers.SuperMarioStructure.yaml --print_config \| head -20` |
| Cache audit passes | `uv run python scripts/embeddings/audit_cache_integration.py` reports 183/183 |
| Sweep planner sees data | `uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure --dry-run` shows `✓ data` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "candidate pieces: 0" | Audio file naming doesn't match piece_id convention | Rename audio files to `<5-digit-piece_id>.<ext>` matching pieces.csv |
| Many pieces in "no audio" list | Same — partial match | Inspect the listed names, fix naming convention |
| MIDI download fails (NinSheetMusic down) | Network issue | Retry later; use `--skip-midi-download` if MIDIs already cached |
| `pretty_midi could not load X.mid` | Malformed MIDI | Skip happens automatically; piece gets dropped with `dropped_no_midi` count |
| `BarRange out of range for MIDI` | Annotation references bar number beyond MIDI's downbeat count | Likely a MIDI/annotation mismatch (different versions of the score); piece's affected segments dropped, counted as `dropped_oor` |
| `Unknown Function code` at parse time | Annotation has a Function code outside {In, Lp, Tr, Br, Ou, St} | Add the new code to `RAW_TO_CANONICAL` in the build script and re-run |
| `ffmpeg slice failed` | Source audio unseekable / corrupted | Re-encode source to FLAC first; or substitute that piece |
| Datamodule `Unknown label: …` at train start | Stale JSONL from before LABEL2IDX change | Re-run build with `--skip-slice` to rewrite JSONL |
| Segments very short (<2 s dropped) | Stinger label hit `--min-segment-sec` filter | Lower `--min-segment-sec 1.0` if you want to keep them |

---

## Files this dataset touches

- `data/SuperMarioStructure/_upstream/supermario-structure-annotation/` — clone (~5 MB)
- `data/SuperMarioStructure/midi/<piece_id>.mid` — source MIDIs (~5 MB)
- `data/SuperMarioStructure/segments/<piece_id>/<seg_idx>_<label>.flac` — per-segment audio (~0.4 GB)
- `data/SuperMarioStructure/SuperMarioStructure.{train,val,test}.jsonl` — annotation files
- `marble/tasks/SuperMarioStructure/{datamodule,probe}.py` — Lightning task
- `configs/probe.<encoder>-{layers,meanall}.SuperMarioStructure.yaml` — 8 configs
- `output/.emb_cache/<encoder>/SuperMarioStructure__<hash>/` — embedding cache (auto)

---

## Why this dataset matters (research framing)

- **Only VGM-native structure benchmark** with labels (`loop`, `stinger`)
  that don't exist in pop-music structure datasets (Harmonix /
  HookTheoryStructure).
- **Highest annotation agreement** of any structure dataset we surveyed:
  95.77% function-boundary and 97.68% section-boundary inter-rater
  agreement on the 50-piece validation subset.
- **Companion to leitmotif work.** Same domain (VGM); different level
  (structure vs motif). End-to-end pipeline becomes possible: detect
  structural boundaries → extract leitmotifs within sections.
- **Small but high-fidelity.** 554 pieces is smaller than HXMSA (912)
  or SALAMI (1400), but with much better label agreement and
  domain-targeted labels.

See [`docs/structure_datasets_survey.md`](../structure_datasets_survey.md)
for the full survey context.

---

## References

- [SuperMario Structure Annotation repo](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
- [NinSheetMusic](https://www.ninsheetmusic.org) — source of the MUS/MXL transcriptions and MIDIs
- Companion docs:
  - [`hxmsa_setup.md`](hxmsa_setup.md) — closest comparable task (Harmonix structure)
  - [`vgmiditvar_setup.md`](vgmiditvar_setup.md) — MIDI-rendering pipeline (relevant if you want to render audio yourself)
  - [`../structure_datasets_survey.md`](../structure_datasets_survey.md) — full MSA dataset survey
