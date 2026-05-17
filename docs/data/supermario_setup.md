# SuperMario Structure — dataset setup (research-stage, not yet implemented)

Runbook + design notes for a future MARBLE task using the
[`ShxLuo-Saxon/supermario-structure-annotation`](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
dataset.

**Status:** NOT IMPLEMENTED. This doc captures the research and the
planned design so the next implementation pass can move directly to
coding without re-doing the analysis.

---

## What the dataset is

| | |
|---|---|
| **Source** | https://github.com/ShxLuo-Saxon/supermario-structure-annotation |
| **Scope** | 554 Super Mario video game music pieces from NinSheetMusic (transcribed scores). Plus a 3,304-pair section-similarity dataset derived from 334 of those pieces (39 excluded for instrumentation reasons). |
| **License** | CC BY 4.0 on the annotations. Source transcriptions remain under NinSheetMusic copyright (not redistributed by the repo). |
| **Validation** | Human inter-rater agreement on 50 pieces: 95.77% on function boundaries, 97.68% on section boundaries. Highest agreement of any structure-annotation dataset we surveyed. |
| **Format** | JSON annotations. Source notation in MUS / MXL format. Provided MXL → ABC conversion scripts in the repo. |
| **Audio** | NOT distributed. Would need to be rendered from the MUS/MXL source. |

## Annotation schema (two levels)

1. **Function-level** — coarse boundaries with one of 6 functional
   labels: `intro`, `loop`, `transition`, `bridge`, `outro`, `stinger`.
   Each annotation includes bar ranges and a human-written rationale.
2. **Section-level** — finer divisions with alphabetic labels
   (A / B / C / ...). Also bar-ranged with rationales.

The two-level scheme is unique among our candidates — it lets us probe
both functional structure (like Harmonix / HookTheoryStructure) AND
similarity-based section structure (more like the SALAMI label-id
problem).

Plus the `compound_sim` similarity matrix (40% chroma + 30% duration +
10% register + 10% density) per piece — useful as ground truth for a
boundary detection task or a similarity-retrieval probe.

## Splits

Pre-defined 70/15/15 train/val/test at the **piece** level.
Reproducibility = high (just use the upstream split as-is).

## Why it's interesting (the music-encoder angle)

- **Specifically VGM-coded labels.** "Loop" and "stinger" are
  game-music-native and don't appear in pop-music datasets (Harmonix /
  HookTheoryStructure). This is the only VGM-native structure
  benchmark we know of.
- **High annotation quality.** 95–97% human agreement is exceptional;
  most pop-music structure datasets are 70–85%.
- **Pairs well with the leitmotif work.** Same domain (VGM), different
  level (structure vs motif). A solid full-pipeline test would be:
  detect structural boundaries → extract leitmotifs within sections.
- **Small but high-fidelity.** 554 pieces is small vs Harmonix (912)
  or SALAMI (1400) but with much better label agreement and
  domain-targeted labels.

---

## Implementation design (proposed, not yet built)

### Audio sourcing — the hard part

Audio is not distributed. Three plausible paths, ranked by cost:

**Path A — MIDI render (matches our VGMIDI infrastructure).** Convert
MUS/MXL → MIDI (using MuseScore CLI or `music21`), then render via
fluidsynth + Shan SGM-Pro 14 (the SoundFont we already use for VGMIDI).
~30 min total compute. Reuses everything from
`scripts/data/build_vgmiditvar_dataset.py`. Most viable v1 path.

**Path B — MuseScore CLI render direct.** MuseScore can render MXL
straight to WAV/MP3 with its own bundled SoundFont. Higher fidelity but
introduces a dependency on a desktop application. Skip unless v1 results
need a quality bump.

**Path C — Hand-curated audio.** Find existing Super Mario soundtrack
recordings online, align to the score annotations. Highest fidelity but
licensing nightmare; not viable.

**Decision:** Path A (MIDI render). Already proven for VGMIDI; same
SoundFont; same renderer.

### Task framing

Two natural tasks:

1. **Function-level classification (6 classes)** — clone the
   HookTheoryStructure / HXMSA pattern. Each section is one labeled
   clip; per-segment evaluation. Simpler, comparable to existing
   structure tasks.
2. **Section-pair similarity retrieval** — use the 3,304 pre-computed
   similarity pairs (or the `compound_sim` matrix) as ground truth.
   Each query section retrieves the K most similar sections; evaluate
   via MAP. Maps to Covers80 / SHS100K / VGMIDITVar machinery.

**Decision for v1:** start with task (1) — function classification.
Task (2) is a stronger probe but requires non-trivial datamodule work
(pairwise sampling, similarity-aware retrieval). Defer to v2.

### Label inventory

The 6 function-level labels:

| Label | Description |
|---|---|
| `intro` | Opening segment, distinguishable from main loop |
| `loop` | Main repeating section (the bulk of most VGM pieces) |
| `transition` | Connecting passage between sections |
| `bridge` | Contrasting middle section |
| `outro` | Closing segment |
| `stinger` | Short punctuation cue (e.g., game-event sound) |

Compared to HXMSA's 13-class inventory:
- `loop` is unique to VGM (no equivalent in pop)
- `stinger` is unique to VGM (no equivalent in pop)
- `intro`, `outro`, `bridge`, `transition` overlap with HXMSA
- VGM has no `verse` / `chorus` / `prechorus` / `postchorus` (lyric-section-specific labels)

The non-overlap means SuperMario gives genuinely complementary
information to HXMSA. Cross-task analysis (does an HXMSA-trained encoder
generalise to VGM structure? Or vice versa?) is a meaningful research
question.

### Encoders to sweep

Same as HXMSA — audio-only:
- CLaMP3 (audio)
- MERT-v1-95M
- MuQ
- OMARRQ-multifeature-25hz

**Plus CLaMP3-symbolic should also work here** since SuperMario has
MIDI input via the MUS/MXL conversion. If we go Path A (MIDI render),
we get symbolic-encoder support for free. **This is a meaningful
difference vs HXMSA** — it lets us compare the symbolic vs audio path
for structure analysis directly.

### Estimated implementation cost

| Step | Hours |
|---|---|
| Write `scripts/data/build_supermario_dataset.py` (MUS/MXL → MIDI → fluidsynth render → JSONL) | 2 |
| Clone HookTheoryStructure/HXMSA datamodule + probe for 6-class output | 1 |
| Clone 10 configs (5 encoders × {layers, meanall}; includes symbolic) | 0.5 |
| Add 5 SweepDef + jsonl_map entry to run_all_sweeps.py | 0.5 |
| Write `docs/data/supermario_setup.md` (this doc replaces stub) | 0.5 |
| Cache audit + verification | 0.5 |
| Total | **~5 hours of focused work** |

Plus ~30 min for actual audio rendering, then the sweep itself
(~3–6 h depending on encoder × layer count).

---

## Outstanding questions

These need to be resolved during implementation:

1. **MUS file format support.** MuseScore's native `.mscz` format
   converts cleanly to MXL. Plain `.mus` files (older Finale format)
   may not — need a converter. The upstream repo provides MXL → ABC
   scripts; check whether MXL → MIDI conversion is in there too, or
   whether we need to write it.
2. **Score → MIDI fidelity.** Are tempo markings preserved? Are
   ornaments (trills, mordents) expanded correctly? If not, the
   rendered audio will diverge from the annotated structure in
   unpredictable ways. May need a `--simplify` preprocessing step.
3. **Section duration distribution.** Loops in Mario music can be very
   short (~5 s) or long (~60 s). Our `clip_seconds=15` default may
   under-sample short loops. May need to lower `min_clip_ratio` or
   adjust `clip_seconds` per section length distribution.
4. **Splits — at piece level or by game?** The dataset's piece-level
   split may put different pieces from the same game on different
   sides. May want a stricter game-level split for harder
   out-of-distribution generalisation testing.

---

## Once implemented (future runbook section)

This section will become the actual runbook (mirrors `hxmsa_setup.md`
structure) once the build script lands:

- Prerequisites (MuseScore? music21? mido? — TBD per Path-A specifics)
- Build command (pilot + full)
- Idempotency notes
- Sweep launch commands
- Pre-flight verification checklist
- Troubleshooting matrix

For now, see [`hxmsa_setup.md`](hxmsa_setup.md) for the template.

---

## References

- [SuperMario Structure Annotation repo](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
- [NinSheetMusic](https://www.ninsheetmusic.org) — source of the MUS/MXL transcriptions
- [`hxmsa_setup.md`](hxmsa_setup.md) — template runbook for the closest
  comparable structure task
- [`vgmiditvar_setup.md`](vgmiditvar_setup.md) — MIDI rendering pipeline
  to reuse for Path A
- [`structure_datasets_survey.md`](../structure_datasets_survey.md) —
  comprehensive comparison of all music structure analysis datasets
  surveyed (including the rationale for prioritising SuperMario)
