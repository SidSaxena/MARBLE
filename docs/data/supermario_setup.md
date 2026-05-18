# SuperMarioStructure — dataset setup

Standalone runbook for building the SuperMarioStructure dataset for
MARBLE probes.

> **Symbolic (MIDI) is the primary representation.** The annotations
> are bar-based, derived from the source MUS/MXL scores — MIDI is the
> exact-match input domain. CLaMP3-symbolic is the lead encoder.
> Audio is a derivation (rendered or recorded from the score) and
> runs as the secondary cross-encoder comparison. You can build
> symbolic-only without supplying any audio.

**Status:** implemented (2026-05-17). Build script, datamodule (both
symbolic + audio paths), probe, 10 configs (CLaMP3-symbolic × 2 +
CLaMP3-audio × 2 + MERT-v1-95M × 2 + MuQ × 2 + OMARRQ-25hz × 2), and
sweep registry all landed.

**Wall-clock budget:** ~5 min annotation clone + MIDI download, then
~5 min for MIDI segment slicing (always runs). Audio slicing (only if
you provide `--audio-dir`) adds ~10–20 min. Total: well under an hour.

**Disk budget (symbolic-only):** ~15 MB. **With audio:** add ~0.4 GB
for per-segment FLACs (your source audio is read-only).

---

## What you get

- 554 Super Mario pieces from the upstream
  [supermario-structure-annotation](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
  repo (Function-level annotations, ~3,500 segments after filtering).
- Per-segment MIDI files at
  `data/SuperMarioStructure/midi_segments/<piece_id>/<seg_idx>_<label>.mid`
  (**primary**, always present).
- Optional per-segment FLAC files at
  `data/SuperMarioStructure/segments/<piece_id>/<seg_idx>_<label>.flac`
  (only if `--audio-dir` provided).
- Three JSONL splits at
  `data/SuperMarioStructure/SuperMarioStructure.{train,val,test}.jsonl`
  (70/15/15 by piece, upstream split for the 334 paired pieces, seed
  1234 70/15/15 for the rest).
- 6 functional classes (VGM-native, shared between symbolic + audio
  datamodules):

| Class | Raw code | Description |
|---|---|---|
| `intro` | `In` | Opening segment, distinct from main loop |
| `loop` | `Lp` | Main repeating section (dominant in most VGM) |
| `transition` | `Tr` | Connecting passage between sections |
| `bridge` | `Br` | Contrasting middle section |
| `outro` | `Ou` | Closing segment |
| `stinger` | `St` | Short punctuation cue (often <2 s) |

### Per-record JSONL fields

**Always present** (symbolic-primary fields):

```
midi_path  ori_uid  work_id  label  seg_idx  bar_start  bar_end
seg_start  seg_end  title  ninsheetmusic_id
```

**Audio-derived fields** (only when `--audio-dir` was supplied AND
audio slice succeeded for that segment):

```
audio_path  duration  sample_rate  num_samples  channels  bit_depth
```

This means symbolic configs work on all records; audio configs filter
to records with `audio_path` present. Cleanly separated by the
respective datamodule.

---

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| **git** | Clone the upstream annotation repo | already required |
| **pretty_midi** | Bar→time mapping + MIDI segment slicing | already a MARBLE dep |
| **ffmpeg + ffprobe** | Audio segment slicing | macOS: `brew install ffmpeg`; only required if you pass `--audio-dir` |

### Sourcing the MIDIs (the one manual step)

**NinSheetMusic actively blocks scrapers** — every automated HTTP
request returns 403, regardless of User-Agent / Referer / cookies (we
verified across stock Playwright, patchright, and direct urllib). The
site is fronted by Cloudflare's anti-bot, which reliably detects
headless Chromium and pins the browser at the "Just a moment..."
challenge page indefinitely. The upstream dataset README also
explicitly says to download manually.

So our build script does **not** include a working auto-downloader;
it consumes a directory of MIDIs you sourced separately. Recommended
path:

#### Option A — `scripts/data/download_ninsheetmusic.py` (built-in Playwright downloader, recommended)

We ship a Playwright-based downloader that runs Chromium *visibly*
(headed mode) — this is the most reliable way to get past Cloudflare's
anti-bot, because it IS a real browser. Setup is one-time:

```bash
# 1. Install the optional Playwright dep + browser (~250 MB)
uv sync --extra ninsheetmusic
uv run playwright install chromium
# Optional but recommended (drop-in Playwright fork with stealth patches):
uv pip install patchright && uv run patchright install chromium

# 2. Download MIDIs (visible browser opens; ~15–30 min for all 554)
uv run python scripts/data/download_ninsheetmusic.py \
    --csv data/SuperMarioStructure/_upstream/supermario-structure-annotation/metadata/pieces.csv \
    --out-dir data/SuperMarioStructure/midi_user \
    --kind mid \
    --persistent-context-dir data/SuperMarioStructure/.playwright_profile
```

What happens: Chromium opens, visits the NSM homepage, Cloudflare's
JS challenge runs (you may briefly see "Just a moment..." then it
clears automatically — if it doesn't, click through any CAPTCHA
manually), then the script iterates the CSV and downloads each MIDI
via the browser's HTTP API (with the now-valid `cf_clearance`
cookie). `--persistent-context-dir` saves the session so re-runs
don't need to re-clear Cloudflare.

The downloader saves files as `<piece_id>.mid` ready to feed straight
into `build_supermario_dataset.py --midi-source-dir`. Pilot first
with `--max-pieces 5` to verify your browser session works.

#### Option B — `ohsheet` Rust CLI

[crates.io/crates/ohsheet](https://crates.io/crates/ohsheet) — a
purpose-built NinSheetMusic scraper. Requires `cargo install ohsheet`
(installs Rust if you don't have it). Output naming may not match
our `<piece_id>.mid` convention; rename as needed.

#### Option C — manual click-through

Open
`data/SuperMarioStructure/_upstream/supermario-structure-annotation/metadata/pieces.csv`
in a spreadsheet, click each `url_mid` link in a browser, save with
the piece_id stem. Tedious for all 554; only practical for a small
pilot.

#### Option D — any other scraper

The build script just consumes a directory of `<piece_id>.<ext>`
files (5-digit zero-padded). Supported extensions: `.mid`, `.midi`,
`.smf`. Whatever fills that dir works.

Without any sourced MIDIs, the build script attempts the urllib
fallback download and gracefully fails (logging the 403 count).
Pieces without MIDIs get dropped — no symbolic record, no audio
record.

### User audio (optional)

If you do want to also build the audio path, files must be named
`<piece_id>.<ext>` (zero-padded 5-digit). Supported extensions
(first match wins): `.flac`, `.wav`, `.mp3`, `.m4a`, `.ogg`, `.opus`.

```
/my/audio/dir/
  00001.flac    ← Captain Toad - Retro RampUp
  00002.mp3     ← Dr Mario - Chill
  ...
```

Pieces with no matching audio still get symbolic records — they just
don't get an `audio_path` field. The audio datamodule silently skips
those records at load time.

### Critical assumption: audio tempo-alignment (only matters if using audio)

The bar→time mapping comes from each piece's source MIDI (auto-
downloaded from NinSheetMusic). If your audio is performed at a
different tempo than the MIDI (e.g., human performance), audio segment
boundaries will drift. **This does not affect the symbolic build** —
MIDI segment slicing uses the exact same MIDI as the source.

---

## Build (the canonical commands)

### Symbolic-only build (recommended for first run)

```bash
# Pilot — 5 pieces, ~1 min (assumes you've populated --midi-source-dir)
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir /path/to/your/midis --max-pieces 5

# Full symbolic build (~10 min for 554 pieces, no audio needed)
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir /path/to/your/midis
```

### Symbolic + audio build

```bash
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir /path/to/your/midis \
    --audio-dir /path/to/your/mario/audio
```

What happens (in order):

1. **Clones** the upstream annotation repo (~5 MB) into
   `data/SuperMarioStructure/_upstream/`.
2. **Parses** `metadata/pieces.csv` (piece IDs + MIDI URLs) and
   `metadata/pairs.csv` (upstream train/val/test split for 334
   pieces).
3. **Downloads** each piece's source MIDI from NinSheetMusic (cached
   at `data/SuperMarioStructure/midi/<piece_id>.mid`). Used as the
   bar→time clock AND as the source for symbolic slicing.
4. **For each piece:**
   - Loads the source MIDI via `pretty_midi` → downbeat times.
   - Parses `annotations/<piece_id>.json` Function array (6 classes).
   - For each `(bar_start, bar_end, label)`:
     - Computes `(start_sec, end_sec)` from the downbeat table.
     - **Slices the source MIDI to a per-segment .mid file** (using
       `pretty_midi`, preserving notes + program-changes per
       instrument; tempo is taken from the source MIDI's local
       tempo at `start_sec`).
     - If `--audio-dir` provided AND audio for the piece is found AND
       ffmpeg slice succeeds: also produces a per-segment FLAC.
5. **Splits** train/val/test at piece level.
6. **Emits** the three JSONL files.

### Render-plan preamble

Before any slow work the script prints (symbolic-only example):

```
────────────────────────────────────────────────────────────
Build plan
────────────────────────────────────────────────────────────
  data-dir          : data/SuperMarioStructure
  midi-dir          : data/SuperMarioStructure/midi   (source MIDIs from NinSheetMusic)
  midi-segments-dir : data/SuperMarioStructure/midi_segments   (per-segment sliced MIDIs — primary symbolic path)
  audio-dir         : (none — symbolic-only mode)
  segments-dir      : (skipped)   (per-segment sliced audio FLACs — only if --audio-dir given)
  candidate pieces  : 554  (554 annotations available; audio missing for 0 of these)
  MIDI already cached  : 0
  MIDI to download now : 554
  MIDI segment slice : enabled
────────────────────────────────────────────────────────────
```

---

## Running the layer sweeps

The 5 sweeps registered for SuperMarioStructure, in the order
`run_all_sweeps.py` will execute them:

| # | Encoder | Layers | Notes |
|---|---|---:|---|
| 1 | **CLaMP3-symbolic** | 13 | **PRIMARY** — symbolic; exact-match domain |
| 2 | CLaMP3 (audio) | 13 | audio derivation (only runs if you built audio) |
| 3 | MERT-v1-95M | 13 | audio derivation |
| 4 | MuQ | 13 | audio derivation |
| 5 | OMARRQ-multifeature-25hz | 24 | audio derivation |

### Symbolic-only sweep (works without audio)

```bash
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks SuperMarioStructure --models CLaMP3-symbolic
```

### Full cross-encoder comparison (requires audio build)

```bash
# Meanall first — baseline in <30 min × 5 encoders
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks SuperMarioStructure --only-meanall

# Full per-layer sweep
uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure
```

---

## Verification before launching the sweep

| Check | How |
|---|---|
| JSONL has ~3500 records, all with `midi_path` | `head -1 data/SuperMarioStructure/SuperMarioStructure.train.jsonl \| jq 'keys'` (look for `midi_path`) |
| Class distribution looks reasonable | `jq -r .label data/SuperMarioStructure/SuperMarioStructure.train.jsonl \| sort \| uniq -c \| sort -rn` — expect `loop` to dominate |
| No piece appears in multiple splits | `for f in data/SuperMarioStructure/SuperMarioStructure.*.jsonl; do jq -r .work_id $f \| sort -u; done \| sort \| uniq -c \| awk '$1>1 {print}'` — should be empty |
| Configs parse cleanly | `uv run python cli.py test -c configs/probe.CLaMP3-symbolic-layers.SuperMarioStructure.yaml --print_config \| head -20` |
| Cache audit passes | `uv run python scripts/embeddings/audit_cache_integration.py` reports 185/185 |
| Sweep planner sees data | `uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure --dry-run` shows `✓ data` |
| (Audio only) `audio_path` present on most records | `jq 'select(.audio_path != null) | .audio_path' data/SuperMarioStructure/SuperMarioStructure.train.jsonl \| wc -l` |

---

## Idempotency + resume

- Source MIDIs: skipped if cached at `<midi-dir>/<piece_id>.mid`.
- MIDI segments: skipped if present (size-aware).
- Audio segments: skipped if present.
- JSONL: always rewritten from current on-disk state.

To force re-slice MIDIs (e.g., after a build-script bugfix): delete
`data/SuperMarioStructure/midi_segments/` first.

To rebuild JSONL only (no new slicing):

```bash
uv run python scripts/data/build_supermario_dataset.py \
    --skip-midi-slice --skip-midi-download --skip-slice
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `candidate pieces: 0` | First-run: upstream repo not yet cloned | The script clones automatically; if this fails, check git + network |
| `MIDI download failed for ... HTTP Error 403: Forbidden` | NinSheetMusic blocks all scrapers — see Prerequisites § Sourcing the MIDIs | Populate `--midi-source-dir` via manual download or `ohsheet`; re-run the build. The 403 is expected, not a bug |
| `pretty_midi could not load X.mid` | Malformed source MIDI | Piece is dropped automatically; counted as `dropped_no_midi` |
| `BarRange out of range for MIDI` | Annotation references bar number beyond MIDI's downbeat count | Likely a MIDI/annotation version mismatch; piece's affected segments dropped, counted as `dropped_oor` |
| `Unknown Function code` | Annotation has a Function code outside {In, Lp, Tr, Br, Ou, St} | Add a defensive alias to `RAW_TO_CANONICAL` in the build script |
| `pretty_midi write failed for X.mid` | Disk full / permission issue | Check disk + permissions on `midi-segments-dir` |
| MIDI segments very short (<2 s) dropped | Stinger label hit `--min-segment-sec` | Lower `--min-segment-sec 1.0` if you want to keep them |
| Datamodule `Unknown label: …` at train start | Stale JSONL from before LABEL2IDX change | Re-run build with `--skip-midi-slice --skip-slice` to rewrite JSONL only |
| (Audio only) "audio missing for N pieces" warning | Some pieces have no matching audio file in `--audio-dir` | Either rename audio files to match `<piece_id>.<ext>` or accept the partial coverage (symbolic still works for all pieces) |

---

## Files this dataset touches

```
data/SuperMarioStructure/
  _upstream/supermario-structure-annotation/   ← clone (~5 MB)
  midi/<piece_id>.mid                          ← source MIDIs (~5 MB)
  midi_segments/<piece_id>/<seg_idx>_<label>.mid   ← per-segment MIDIs (~10 MB)
  segments/<piece_id>/<seg_idx>_<label>.flac   ← per-segment audio (~0.4 GB, optional)
  SuperMarioStructure.{train,val,test}.jsonl   ← annotation files

marble/tasks/SuperMarioStructure/
  datamodule.py   ← audio + symbolic base classes
  probe.py        ← shared probe (BaseTask + per-segment aggregation)

configs/probe.<encoder>-{layers,meanall}.SuperMarioStructure.yaml   ← 10 configs

output/.emb_cache/<encoder>/SuperMarioStructure__<hash>/   ← embedding cache (auto)
```

---

## Why this dataset matters (research framing)

- **Only VGM-native structure benchmark** with labels (`loop`,
  `stinger`) that don't exist in pop-music structure datasets
  (Harmonix / HookTheoryStructure).
- **Highest annotation agreement** of any structure dataset we surveyed:
  95.77% function-boundary and 97.68% section-boundary inter-rater
  agreement on the 50-piece validation subset.
- **Direct comparison of symbolic vs audio paths** on a single
  dataset. Most of our other datasets are either audio-only
  (HookTheoryStructure, HXMSA) or where symbolic doesn't add much
  (VGMIDI/leitmotif where the MIDIs were already piano-only). Here,
  the symbolic input is the exact match to the annotation domain, so
  the cross-encoder comparison is genuinely interesting:
  - Does CLaMP3-symbolic dominate the audio encoders here the way it
    did on the leitmotif task?
  - Or does the small dataset (vs the leitmotif task's larger pool)
    favour the more parameter-efficient audio encoders?
- **Companion to leitmotif work.** Same domain (VGM); different level
  (structure vs motif). End-to-end pipeline becomes possible: detect
  structural boundaries → extract leitmotifs within sections.

See [`docs/structure_datasets_survey.md`](../structure_datasets_survey.md)
for the full survey context and ranking.

---

## References

- [SuperMario Structure Annotation repo](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
- [NinSheetMusic](https://www.ninsheetmusic.org) — source of the MUS/MXL transcriptions and MIDIs
- Companion docs:
  - [`hxmsa_setup.md`](hxmsa_setup.md) — closest comparable audio task (Harmonix structure)
  - [`vgmiditvar_setup.md`](vgmiditvar_setup.md) — MIDI-rendering pipeline (if you want to render audio from the source MIDIs)
  - [`../structure_datasets_survey.md`](../structure_datasets_survey.md) — full MSA dataset survey
  - [`../leitmotif_findings.md`](../leitmotif_findings.md) — CLaMP3-symbolic vs audio findings on the leitmotif task (informs expectations)
