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
| `bridge` | `Br` | Distinct transition section between Lp/Ln |
| `intro` | `In` | Opening segment |
| `linear` | `Ln` | Main through-composed body (non-looped piece) |
| `loop` | `Lp` | Main repeating section (loop-marked) |
| `outro` | `Ou` | Closing segment |
| `stinger` | `St` | Short event-triggered cue (often <2 s) |

> The upstream README lists `Tr` (Transition) instead of `Ln`. The
> README is wrong — the annotations were generated from the prompt
> at `scripts/prompts/prompt_v1.2.md`, which defines `Ln (Linear
> Body)` and has no `Tr` tag. Grepping all 554 annotations confirms
> 0 `Tr` entries, 107 `Ln` entries. Always trust the prompt + data
> over the README.

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

The downloader saves files as `<piece_id>_<title-slug>.mid` (since
commit `db8f376`). The build script handles both this naming and the
legacy `<piece_id>.mid` automatically — no rename step needed. Pilot
first with `--max-pieces 5` to verify your browser session works.

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

The build script consumes a directory of `<piece_id>.<ext>` OR
`<piece_id>_<slug>.<ext>` files (5-digit zero-padded `piece_id`).
Supported extensions: `.mid`, `.midi`, `.smf`. Whatever fills that
dir works.

Without any sourced MIDIs, the build script attempts the urllib
fallback download and gracefully fails (logging the 403 count).
Pieces without MIDIs get dropped — no symbolic record, no audio
record.

#### Option E — convert from .mxl / .musicxml (RECOMMENDED for CLaMP3-symbolic)

If you've sourced the SuperMarioAnnotation scores in MusicXML (.mxl /
.musicxml) — for example by opening the upstream .mus files in
MuseScore/Finale and exporting — the .mxl path unlocks two things the
.mid path can't:

1. **A clean offline pipeline.** No NSM Cloudflare-bot block, no
   Playwright session, no `ohsheet` — your local .mxl files become the
   source of truth.
2. **ABC input to CLaMP3-symbolic (the bar-level path).** CLaMP3's M3
   patchiliser is bimodal: MTF mode packs MIDI events into 64-byte
   patches; ABC mode emits one patch per bar using ABC barline
   delimiters. ABC is the format CLaMP3 was primarily trained on, so
   bar-level patches sit closer to the model's training distribution.
   Expect meaningfully better numbers than the secondary MTF-mode
   MIDI path on retrieval + classification probes.

Two pipelines, depending on how far you want to go:

##### (E1) .mxl → ABC → CLaMP3-symbolic (bar-level patches, primary)

```bash
# One-time: install the optional dep
uv sync --extra symbolic-abc

# Build with --build-abc: emits per-segment .abc files alongside the
# MIDI segments, and adds an `abc_path` field to every JSONL record
# where ABC slicing succeeded.
uv run python scripts/data/build_supermario_dataset.py \
    --build-abc \
    --mxl-source-dir data/SuperMarioStructure/mxl
```

What this produces:

- `data/SuperMarioStructure/abc_segments/<piece_id>/<seg_idx>_<label>.abc` per segment.
- Each JSONL record gains an `abc_path` field (records without a
  matching .mxl skip the field; `input_format: abc` configs filter
  those out at load time with a warning).

##### Interleaved ABC (training-faithful preprocessing)

Marble's `--build-abc` now applies the same post-xml2abc cleanup
CLaMP3 used during its symbolic-branch training:

1. `xml2abc -d 8 -x` (eighth-note default unit, stdout output — matches
   `vendor/clamp3/preprocessing/abc/batch_xml2abc.py` in upstream CLaMP3).
2. Strip metadata fields (`X:`, `T:`, `C:`, `Z:`, `W:`, `w:`, `%%MIDI`)
   that don't carry musical content.
3. Strip `%N` bar-number annotations xml2abc leaves at line ends.
4. Strip embedded barline characters inside ABC quote annotations
   (rare but breaks `rotate_abc`).
5. `strip_empty_bars` — drop bars with no notes.
6. `rotate_abc` — interleave voices bar-by-bar so the body becomes
   `[V:1]bar1|[V:2]bar1|[V:1]bar2|[V:2]bar2|...` (instead of raw
   `V:1 [all bars]\nV:2 [all bars]`).

The interleaved form is what CLaMP3 was actually trained on. The
implementation lives in
[`scripts/data/build_supermario_dataset.py::_abc_to_interleaved`](../../scripts/data/build_supermario_dataset.py)
and was ported from the leitmotifs project's symbolic adapter
(commit `a9d0ce0` of `feat/clamp3`), which mirrors CLaMP3's own
`preprocessing/abc/batch_interleaved_abc.py` via the
[`abctoolkit`](https://pypi.org/project/abctoolkit/) PyPI package
(`uv sync --extra symbolic-abc` installs it).

If `abctoolkit` isn't installed, `_abc_to_interleaved` falls back
to raw ABC with a one-shot warning — non-symbolic-abc workflows are
not broken.

Use the ABC variant configs:

- [`configs/probe.CLaMP3-symbolic-abc-meanall.SuperMarioStructure.yaml`](../../configs/probe.CLaMP3-symbolic-abc-meanall.SuperMarioStructure.yaml)
- [`configs/probe.CLaMP3-symbolic-abc-layers.SuperMarioStructure.yaml`](../../configs/probe.CLaMP3-symbolic-abc-layers.SuperMarioStructure.yaml)

Pipeline internals: `music21.converter.parse(.mxl)` →
`score.measures(bar_start, bar_end)` slices the requested bar range →
write temp .musicxml → vendored
[`scripts/data/_vendor/xml2abc.py`](../../scripts/data/_vendor/xml2abc.py)
emits the final ABC. The datamodule reads the .abc text and feeds it
straight to the patchiliser's ABC mode (no `midi_to_mtf` step).

##### (E2) .mxl → .mid (use existing MIDI pipeline, no ABC)

If you don't want to switch input formats but still want to bypass
NSM's bot block:

```bash
# One-time: install music21 (the symbolic-abc extra includes it; or
# `uv pip install music21` for the bare minimum)
uv pip install music21

# Bulk convert .mxl → .mid alongside, preserving the stem so the
# build script's `<piece_id>(_<slug>)?.mid` matcher still works.
uv run python scripts/data/convert_mxl_to_midi.py \
    --in-dir  /path/to/your/mxl_files \
    --out-dir data/SuperMarioStructure/midi_user

# Then run the regular build (no --build-abc, no changes to its CLI)
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir data/SuperMarioStructure/midi_user
```

This still benefits from music21's higher-fidelity MIDI export
(strictly cleaner than scraped .mid from NSM), but stays on the
MTF-mode patchiliser path that everything ran on before.

##### When to prefer (E1) over (E2)

- (E1) is the path for CLaMP3-symbolic specifically — bar-level
  patches match its training distribution.
- (E2) is fine for any downstream consumer that needs MIDI bytes
  (rendering audio, feeding a MIDI-only encoder like Aria or
  MidiBERT-Piano if/when added — see [`../symbolic_encoder_landscape.md`](../symbolic_encoder_landscape.md)).
- You can do both: `--build-abc --mxl-source-dir <dir>` PLUS supplying
  the converted MIDIs via the standard `--midi-source-dir` flag.
  The build script will emit both segment trees + both fields per
  record, and the datamodule picks `input_format: abc` or `midi`.

What .mxl does NOT unlock:

- The audio path. .mxl is symbolic; the audio encoders (MERT / MuQ /
  OMARRQ) still need real recordings (or rendered audio, which we
  don't ship).
- BPS-Motif. That dataset's symbolic source is `csv_notes/` CSVs from
  its own upstream — not MusicXML, not affected by this change.

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

### Symbolic build with ABC (recommended for CLaMP3-symbolic)

If you have .mxl files (e.g. from converting the upstream .mus via
Finale/MuseScore), add `--build-abc` to also emit per-segment ABC
alongside the MIDI segments. The ABC variant feeds CLaMP3-symbolic in
its bar-level patch mode — the primary mode the model was trained on.
See § Option E for the full rationale.

```bash
uv sync --extra symbolic-abc    # one-time: installs music21

uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir /path/to/your/midis \
    --build-abc \
    --mxl-source-dir /path/to/your/mxl_files
```

Adds: `data/SuperMarioStructure/abc_segments/<piece>/<seg>.abc` per
segment + an `abc_path` field on every JSONL record that successfully
sliced. Use `configs/probe.CLaMP3-symbolic-abc-*.SuperMarioStructure.yaml`
to run sweeps against the ABC path (the MIDI configs continue to use
the MIDI/MTF path — both coexist).

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

> **Important:** the 4 audio configs (`probe.{CLaMP3,MERT-v1-95M,MuQ,
> OMARRQ-multifeature-25hz}-*.SuperMarioStructure.yaml`) require the
> JSONL to have an `audio_path` field on every record. This only
> happens when the build was run **with `--audio-dir`**. If you ran
> a symbolic-only build (no `--audio-dir`), the audio datamodule will
> raise `KeyError: 'audio_path'` on the first batch. Either rebuild
> with `--audio-dir <wav-dir>` first, OR restrict the sweep to
> `--models CLaMP3-symbolic`.

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
| `Unknown Function code` | Annotation has a Function code outside {In, Lp, Ln, Br, Ou, St} | Add a defensive alias to `RAW_TO_CANONICAL` in the build script |
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
  - *../leitmotif_findings.md (deprecated, variant dropped)* — CLaMP3-symbolic vs audio findings on the leitmotif task (informs expectations)
