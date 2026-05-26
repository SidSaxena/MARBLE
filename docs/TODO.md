# MARBLE — open follow-ups

Living list of things we've decided to defer rather than not do. Add an
entry per item with (a) the motivation, (b) the rough design, (c) the
cost estimate, and (d) what would trigger us to actually do it.

---

## Research scope

The probing work here serves two goals, in priority order:

1. **Leitmotif analysis** in video-game and film music — does a frozen
   self-supervised encoder layer expose a representation that lets
   you recognise a recurring motif under timbral / orchestration /
   register variation?
2. **Structure analysis** in video-game and film music — boundary
   detection + functional-segment labelling (intro / loop / bridge / …)
   on game and film soundtracks specifically.

Everything else (emotion recognition, generic tagging, genre, speech,
broad pop classification) is **out of scope** for the active sweep
queue. We keep configs for those tasks for future reuse but don't
allocate GPU-time on them right now.

---

## Sweep status by priority (2026-05-22 snapshot)

WandB completion verified across all 1,292 finished runs in the
`MARBLE` project. "Complete" below means at least one encoder has
test/* metrics for ≥1 layer; "fully complete" means the canonical
priority encoder set (MERT-95M, MuQ, OMAR-RQ, CLaMP3) all have
their full per-layer + meanall runs.

### Tier A — top priority, in-flight or about to start

| Task | Domain | Status | Next action |
|---|---|---|---|
| **HookTheoryMelody** | frame-level pitch / melody on real songs (leitmotif foundation) | 🚧 in-flight on PC (MuQ ~36 h in); zero WandB completions yet | Let PC sweep finish. Then MERT-95M + OMARRQ. Per [data/hooktheory_melody_setup.md](data/hooktheory_melody_setup.md). |
| **NSynth** | isolated-note pitch (cleanest pitch foundation; meaningful for motif identity) | 0 / 4 priority encoders complete | Configs exist for all 4. Data on Modal volume: ❌ missing. Run `modal run modal_marble.py::download_marble_datasets` (NSynth is in the m-a-p set) then sweep. |

### Tier B — structure analysis (audio + symbolic + multimodal)

| Task | Domain | Status | Next action |
|---|---|---|---|
| **HookTheoryStructure** | Western pop functional segments | ✅ all 4 audio encoders complete (CLaMP3, MERT-95M, MuQ, OMARRQ × 13–24 layers + meanall) | MERT-330M optional; analysis already written ([layer_analysis.md](layer_analysis.md)) |
| **VGMIDITVar** | VGM theme-variation retrieval (cover-style baseline for game music) | ✅ all 7 audio + symbolic encoders complete | Done — first-author reference for the leitmotif story |
| **VGMIDITVar-leitmotif** | VGM leitmotif (recurring-motif slice of VGMIDITVar) | ✅ CLaMP3 audio + symbolic + MERT-95M + MuQ + OMARRQ complete | Optional: MERT-330M, OMARRQ-fsq. Analysis: [leitmotif_findings.md](leitmotif_findings.md) |
| **VGMIDITVar-multisf** | multi-soundfont VGM (timbre-invariance test for motif recognition) | ✅ CLaMP3 + MERT-95M/330M + MuQ + OMARRQ complete | Optional: OMARRQ-fsq |
| **SuperMarioStructure** | game-music functional segments (intro/loop/bridge/stinger) — symbolic ✅, audio missing | ⚠️ CLaMP3-symbolic complete (13 layers + meanall); audio path 0/4 | Audio sweep: needs the audio_source uploaded to Modal volume + MERT-95M/MuQ/OMARRQ/CLaMP3-audio configs. See [supermario_findings.md](supermario_findings.md) |
| **HXMSA** | Harmonix Set structure (general pop) — comparison baseline for game-music structure | 0 / 4 encoders complete | New `setup_hxmsa` on Modal — needs yt-dlp + cookies.txt on volume. Per [data/hxmsa_setup.md](data/hxmsa_setup.md). |

### Tier C — supporting context (foundational signals)

| Task | Why relevant | Status | Next action |
|---|---|---|---|
| **HookTheoryKey** | harmonic context (key/mode) helps interpret structural functions in film music | ⚠️ partial: CLaMP3 (3L), MuQ (1L) — abandoned mid-sweep | Decide: finish, or de-prioritise (HookTheoryStructure already covers this signal indirectly) |
| **Covers80** | cover-song retrieval — generic-pop melodic-identity baseline, calibrates SHS100K + VGMIDITVar retrieval numbers | ✅ all 6 encoders complete (CLaMP3, MERT-95M/330M, MuQ, OMARRQ + OMARRQ-fsq) | Done |
| **SHS100K** | cover-song retrieval — melodic-identity proxy at scale, dual-use for leitmotif validation | ✅ all 5 priority encoders complete (CLaMP3, MERT-95M/330M, MuQ, OMARRQ) | Done |
| **Chords1217** | frame-level chord recognition — harmonic-progression signal for structure | 0 / any encoder | Tier C, not blocking. Frame-level cache off (correctly). Worth one MERT-95M baseline if cycles available. |
| **GTZANBeatTracking** | downbeat detection helps structural inference | ⚠️ OMARRQ-fsq only, 4 layers (partial) | Tier C. Defer unless we want a strong downbeat baseline. |
| **GS** (GiantSteps Key) | global key as alternate to HookTheoryKey | ✅ CLaMP3, MERT-95M, OMARRQ-fsq complete | Done for our purposes |

### Out of scope (have configs, will not sweep)

EMO (emotion), GTZANGenre, MTG{Genre, Instrument, Mood, Top50}, MTT
(general tagging), LibriSpeechASR (speech). Configs stay in the tree
for future re-use but no GPU-time is allocated.

### Symbolic encoder candidates (none yet integrated)

CLaMP3-symbolic is the only symbolic encoder in MARBLE today. Full
landscape report current to 22 May 2026 in
[`symbolic_encoder_landscape.md`](symbolic_encoder_landscape.md) —
covers what's downloadable, integration cost sketches, what's new
post-Nov-2025 (MuseTok, PianoRoll-Event, BACHI, MIDI-LLaMA, motif-CRF,
SAVGM), and a recommended priority order. TL;DR when we do add more:

1. **Aria-medium-embedding** (solo-piano contrastive — top pick).
2. **MidiBERT-Piano** (literature baseline for BPS-Motif).
3. **Moonbeam-839M** (multi-instrument, modern AMT replacement).

Skip CLaMP 2 / MuPT / MuseTok (not yet packaged) / MusicBERT-original
(fairseq pain).

### Potential additions (not implemented yet)

From [`structure_datasets_survey.md`](structure_datasets_survey.md),
ordered by fit to the leitmotif + VGM/film scope:

1. ~~**BPS-Motif**~~ — **shipped 2026-05-22 (symbolic v1).** See [data/bps_motif_setup.md](data/bps_motif_setup.md). MNID + Retrieval probes against CLaMP3-symbolic, 5-fold CV. Audio variant deferred until user sources original Beethoven recordings.
2. **NES-VMDB** — 474 hours of VGM (largest VGM dataset). Strongest scaling test for VGM-specific layer profiles. ~8 h.
3. **SongFormBench** (Oct 2025) — expert-verified general-pop structure (300 tracks). SOTA benchmark for boundary detection. Worth a comparison point for HXMSA. ~3 h.
4. **TAVERN** — classical theme + variation parallel to VGMIDITVar. Tests whether the encoder behaviour generalises from VGM to classical theme-variation. ~5 h.
5. **BPSD** — Beethoven cover-style dataset; classical analogue to SHS100K + the VGMIDITVar retrieval task. Tests retrieval generalisation from VGM to classical. ~6 h.
6. **JSD** (Jazz Structure Dataset) — extends genre coverage; less directly relevant but useful for triangulation. ~4 h.

**Lower priority for the current scope:** Raveform, Annotated
Mozart Sonatas, YM2413-MDB, OSSL, Mozart Texture, Isophonics, RWC,
SALAMI (mostly general-pop / broad coverage; weak fit to VGM/film
leitmotif goals).

### Data-on-Modal status (priority datasets only)

| Dataset | On `marble-data` volume? | Setup function |
|---|---|---|
| HookTheoryMelody | ❌ missing | `setup_hooktheory_full` |
| NSynth | ❌ missing | `_download_marble_datasets` |
| HookTheoryStructure | ❌ missing | `setup_hooktheory_full` (rebuilds same audio) |
| HXMSA | ❌ missing | `setup_hxmsa` (new) |
| SuperMarioStructure | ✅ symbolic side complete; audio_source missing | `setup_supermario_structure` (new) |
| VGMIDITVar / -leitmotif / -multisf | ❌ missing | TODO: no Modal setup function yet — local-built data needs `modal volume put` |
| SHS100K | ✅ | `setup_shs100k_jsonl` |
| Chords1217 | ✅ | `_download_marble_datasets` |
| GS, Covers80, EMO, GTZAN*, MTG*, MTT | ❌ missing (mostly out of scope) | `_download_marble_datasets` covers most |

For VGMIDITVar* on Modal: needs either (a) a new `setup_vgmiditvar` function ported from the local build script, or (b) one-time `modal volume put` from the local data dir. Worth adding once we run anything on Modal that needs it.

---

## Done

### ✅ Per-layer embedding cache (retrieval tasks)

**Shipped 2026-05-14** in commit `58871bc` + `bdaef8b`. See full
documentation at [`docs/embedding_cache.md`](embedding_cache.md).

- Cache utility (`marble/utils/emb_cache.py`) with atomic writes,
  deterministic key derivation, batch helpers.
- Cache integration in `CoverRetrievalTask` (used by Covers80, SHS100K,
  VGMIDITVar via re-export). `cache_embeddings: true` opt-in on all 44
  retrieval configs.
- Datamodule batches return `clip_id` as a 4th tuple element.
- Pre-warm + inspection CLIs: `scripts/embeddings/extract.py`,
  `scripts/embeddings/manage.py`.

Validation: disk math exact (96 KB/clip predicted, 100 KB observed on
OMARRQ-25hz with `torch.save` header); meanall + per-layer queries served
from same cache by the commutativity property.

### ✅ Cache extension to clip-level supervised tasks (11 tasks)

**Shipped 2026-05-14** as the follow-up to the retrieval cache.

- Cache plumbing factored into `EmbeddingCacheMixin`
  (`marble/utils/emb_cache.py`). Both `BaseTask` and `CoverRetrievalTask`
  inherit; supervised tasks (`GS`, `EMO`, `GTZANGenre`, `NSynth`,
  `HookTheoryKey`, `HookTheoryStructure`, `MTGGenre/Instrument/Mood/Top50/MTT`)
  get caching transparently through `BaseTask`.
- `BaseTask.forward(x, clip_ids=...)` routes through the mixin's
  hit/miss paths; `_shared_step`/`test_step` unpack 4-tuple batches.
- All 11 supervised datamodules now emit `clip_id` as the 4th tuple
  element (using `make_clip_id`).
- ~97 supervised configs updated with `cache_embeddings: true`.

### ✅ Audio-I/O bypass on cache hits

**Shipped 2026-05-14** with the supervised extension. Pushes warm-cache
wall-clock from ~10 min per layer (audio decode dominated) to
estimated <1 min per layer.

- `BaseAudioDataset.cache_check_fn` (optional `Callable[[str], bool]`)
  injected by the task at `setup()` time via
  `EmbeddingCacheMixin._inject_cache_check_into_datasets`.
- On hit, dataset returns a zero-placeholder waveform and skips
  `torchaudio.load + resample + pad` entirely. The task's `forward()`
  ignores `x` on cache hits and uses the cached tensor.
- Same pattern added to the 7 custom-dataset classes (Covers80,
  SHS100K, VGMIDITVar, GS, EMO, GTZANGenre, NSynth).

---

## Open

### 0. Music structure analysis dataset queue

Live priority list for adding new MSA tasks to MARBLE. Full survey + per-dataset profiles in [`structure_datasets_survey.md`](structure_datasets_survey.md).

| Status | Dataset | Domain | Why |
|---|---|---|---|
| 🚧 Queued (configs landed) | **HXMSA** | Western pop (Harmonix) | 912 tracks, 13-class functional segments. See [`data/hxmsa_setup.md`](data/hxmsa_setup.md). Next sweep candidate. |
| 📝 Planned (runbook drafted) | **SuperMario Structure** | Video game | 554 Mario pieces, 6-class VGM-native labels (`loop`, `stinger`). See [`data/supermario_setup.md`](data/supermario_setup.md). MIDI-render via SGM-Pro. |
| 📋 Recommended next | **SongFormBench** (Oct 2025) | General pop | 300 expert-verified tracks; SOTA benchmark from current paper. ~3 h impl. |
| 📋 Recommended next | **BPS-Motif** | Classical (Beethoven) | Direct leitmotif analogue — 263 motifs in Beethoven sonatas. ~6 h impl. |
| 📋 Strong candidate | **JSD** (Jazz Structure Dataset) | Jazz | 340 tracks; extends genre coverage. ~4 h impl. |
| 📋 Strong candidate | **TAVERN** | Classical | Theme+variation parallel to VGMIDITVar in the classical canon. ~5 h impl. |
| 📋 Strong candidate | **BPSD** | Classical (Beethoven) | Multi-version retrieval analogue to Covers80 for classical. ~6 h impl. |
| 📋 Large but expensive | **NES-VMDB** | Video game (NES) | 474 hours; largest VGM dataset. ~8 h impl. |
| 📋 Large but expensive | **SALAMI** | Multi-genre | 1356 tracks, gold-standard boundary benchmark. ~8 h impl. |
| 📋 Lower priority | Raveform, Annotated Mozart Sonatas, YM2413-MDB, OSSL, Mozart Texture, Isophonics/RWC | Various | Niche or partial fit — see survey doc for details. |

**Recommended implementation sequence:** finish HXMSA → SuperMario → SongFormBench → BPS-Motif. That gives 4 distinct axes (large pop / VGM / SOTA general / classical motif) in ~14 hours of focused work.

### 1. Leitmotifs matrix-profile result cache (separate repo)

In `/Users/sid/leitmotifs/`. The matrix-profile cosine-similarity step
takes 4–5 hours of GPU compute over 8M pair-wise comparisons (5-second
windows × 259 tracks). DTW after that is another 2 hours. Per-window
embeddings are already cached in `embeddings/<model>/L<N>/*.pt`, so
MARBLE-style embedding caching won't help. What WILL help is caching
the **matrix-profile results** themselves:

- Per-pair `(pair_id, peak_score, peak_pos)` → ~100 MB for 8M pairs
- Per-pair DTW scalar → ~32 MB

Cache key: `sha256(encoder_id + layer + window_seconds + step_seconds + sample_rate)[:8]`.
Re-iterations on downstream filter thresholds or clustering
parameters reuse the cached matrix-profile output instead of paying
the 4–5h GPU cost again.

**Cost.** ~50 LOC wrapping the existing matrix-profile script.
**Trigger.** Next time you iterate on the leitmotifs pipeline.

### 2. Frame-level task caching

**Shipped 2026-05-20** in commit `279cf41` as the `pool_time: bool`
flag on `EmbeddingCache` + `EmbeddingCacheMixin` + `cache_pool_time`
on `BaseTask` configs. Off-by-default (clip-level `(L, H)` stays the
norm); set `cache_pool_time: false` alongside `cache_embeddings: true`
to switch to the `(L, T, H)` frame-level layout. Disk cost ~1.5 MB
per slice for MuQ at 25 Hz × 15 s × H=1024 vs ~50 KB pooled. Currently
opted in only on the 6 HookTheoryMelody configs. See
[embedding_cache_correctness.md §10](embedding_cache_correctness.md).

Open extension: keep the cache layout but compress with fp16 / int8
(section 3). Currently no plans to backfill cached frame-level data
for the rest of the frame-level tasks (`GTZANBeatTracking`,
`Chords1217`, `LibriSpeechASR`) — those are either lower priority or
out of scope for the current focus.

### 3. Tensor compression (fp16 / int8 quantization)

The current cache is fp32. fp16 would halve disk usage (96 KB → 48 KB
per clip) with negligible precision impact on cosine similarity. int8
would quarter it but needs careful per-clip scale storage. Not urgent
at current disk usage — defer until aggregate cache passes ~10 GB.

### 4. Encoder fine-tuning

The cache assumes a frozen encoder; if `train_mode != "freeze"`, the
cache key would need to include encoder weight hashes and invalidate
every epoch. Not in scope for the current probe-only experiments.

---

## Background-leitmotif level-mismatch experiment (deferred)

**Motivation.** When listening to game/film soundtracks, a leitmotif is
often re-introduced quieter, in the background under dialogue or louder
foreground instruments. A retrieval system that only works at uniform
levels misses that real use case. We want to know whether the encoders
can match a leitmotif against a quieter restatement of itself.

**Design.** Same VGMIDITVar-timbre rendered + normalized audio (post the
reverb + LUFS-normalize pass), but generate copies attenuated by
{−6, −12, −18, −24} dB. Build a JSONL variant `VGMIDITVar-timbre-levels`
where each (work, program) pair appears at 5 levels (0 dB plus the 4
attenuations). Cross-condition MAP grid spans both program AND level
axes — same metric infrastructure (`compute_perpair_map`,
`condition_gap`), just two condition fields concatenated.

**Why deferred.** Distinct from the cross-instrument timbre test. We
want to attribute timbre robustness BEFORE confounding it with level
robustness. Run this after the timbre sweep is interpreted; if
encoders are already failing on cross-instrument, level-mismatch
results will be even worse and uninformative.

**Cost.** 4 attenuated copies of 102,960 files = 411,840 extra renders.
Pure scalar gain, ~5 min with 8 ffmpeg workers. Disk ~3-4 GB additional
FLAC.

**Trigger.** Once VGMIDITVar-timbre sweep is interpreted AND we want a
follow-up "ecological deployment" signal.

---

## Ecological per-instrument reverb experiment (deferred)

**Motivation.** Real-world music applies different reverb signatures
per instrument family: acoustic instruments in concert halls, lead
synths with short bright reverbs, pad synths drenched in long ambient
reverbs. Pre-trained encoders saw these conventions in training. The
current VGMIDITVar-timbre uses ONE IR for all 8 programs to isolate
timbre invariance as the only varying axis, which is methodologically
clean but ecologically unrealistic for synth instruments (P80 Lead,
P89 Pad).

**Design.** Render-side: keep the existing mono renders. Postprocess
with a per-program IR map instead of a single IR:
- P0 Piano / P24 Guitar / P48 Strings / P52 Choir / P60 Horn / P73
  Flute → small/medium concert hall (e.g. Bricasti M7 Small & Near)
- P80 Lead → bright short plate reverb (e.g. Bricasti M7 Bright Plate,
  ~1s decay)
- P89 Pad → long ambient hall (e.g. Bricasti M7 Large & Deep, ~4s decay
  with high wet mix)

JSONL variant `VGMIDITVar-timbre-ecological` with the same audio paths
swapped to the ecologically-mixed audio dir.

**Why deferred.** Distinct experiment. The current uniform-IR test
answers "can the encoder generalize across timbre?". The ecological
variant answers "can the encoder retrieve under realistic deployment
conditions?". Run AFTER the uniform-IR test is interpreted so we can
attribute any per-program regression to either the timbre itself or
the reverb mismatch.

**Cost.** Re-render the timbre audio with per-program IR. Same wall
time as the uniform-IR pass (~30-40 min for 102,960 files at 8
workers). Disk: a parallel copy or in-place rewrite of the already-
processed audio.

**Trigger.** After VGMIDITVar-timbre (uniform) sweep results are
analyzed and we want a follow-up "deployment realism" signal.

---

## OMARRQ-multifeature-nonfsq config naming + checkpoint-family ambiguity

**Motivation.** `configs/probe.OMARRQ-multifeature-nonfsq*.VGMIDITVar.yaml`
have an internal inconsistency: the filename says `multifeature-nonfsq`
(no `-25hz-`) AND the `model_id` is `mtg-upf/omar-rq-multifeature` (the
75-Hz base checkpoint) — but the wandb `group`, `save_dir`, and some
tags read `OMARRQ-multifeature-25hz-nonfsq` (would imply the 25-Hz
variant). Auditor (issue #11) reported this as "two different group
names for the same variant" — but the two names probably refer to
DIFFERENT model checkpoints.

**Resolution required.** Decide which checkpoint we actually want to
probe at this config name, then fix the labels to match. Options:
1. The intent is the 75-Hz base non-FSQ → fix `group`/`save_dir` to
   drop the `-25hz-` substring.
2. The intent is a 25-Hz non-FSQ variant → change `model_id` to the
   matching HF checkpoint (verify it exists).

**Why deferred.** Not a mechanical fix; needs us to confirm which OMAR-RQ
checkpoint we ran (or intended to run) under this name. Audit cleanup
branch leaves these configs untouched.

**Cost.** ~15 min once we decide. No wandb runs exist under either
group name yet, so no dashboard regression either way.

**Trigger.** When the next OMARRQ-multifeature-nonfsq sweep is
scheduled, or when collating final OMARRQ variant comparisons.

---

## LeitmotifDetection embedding-cache integration

**Motivation.** `marble/tasks/LeitmotifDetection/probe.py` has zero
cache wiring while every other new task uses `EmbeddingCacheMixin`.
Every layer-sweep run re-encodes audio from scratch → ~10× the
compute cost it should be. LeitmotifDetection is listed as the #1
research priority in this doc.

**Design (verified via Phase-1 exploration in
`/Users/sid/.claude/plans/no-i-think-we-re-structured-quilt.md`).**

Probe edits (mirror Covers80/probe.py pattern, since `BaseTask` already
mixes `EmbeddingCacheMixin`):
- `__init__`: add `cache_embeddings: bool = False` + `cache_pool_time:
  bool = True` params; set as attrs; call `self._init_cache_state()`.
- `setup()`: add hook calling `super().setup(stage); self._ensure_cache();
  self._inject_cache_check_into_datasets()`.
- `test_step()`: unpack 4-tuple `(x, labels, file_paths, clip_ids)`
  and pass `clip_ids=list(clip_ids)` to `self(...)`.

Datamodule edits:
- Import `make_clip_id` from `marble.utils.emb_cache`.
- Change `__getitem__` return signature from 3-tuple → 4-tuple
  `(waveform, label, path, clip_id)`.
- Compute `clip_id = make_clip_id(path, slice_idx)` per item.
- Add `self.cache_check_fn = None` in `__init__` for audio-I/O bypass.

**Why deferred (user decision).** Larger structural change (alters the
3-tuple → 4-tuple contract); wants to land it as its own focused
branch rather than bundling into a mechanical-fixes PR.

**Cost.** ~1-2 hours implementation + 1 smoke sweep to validate.

**Trigger.** Before the next active LeitmotifDetection sweep run.
