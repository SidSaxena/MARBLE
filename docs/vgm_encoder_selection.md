# Encoder & layer selection for VGM structure / melody analysis

**Scope:** which encoder + layer combination to use for music
structure or melody analysis on video-game soundtracks, depending
on what you have (audio, MIDI, both). Practical deployment guide,
not an experimental writeup — for the underlying results see
[`supermario_findings.md`](supermario_findings.md),
[`leitmotif_findings.md`](leitmotif_findings.md), and the canonical
cross-encoder tables in [`layer_analysis.md`](layer_analysis.md).

**Caveat upfront:** every "audio + VGM + structure" recommendation in
this doc is a **transfer assumption from pop** until the
SuperMarioStructure audio sweep lands. The symbolic side and the
VGM-domain audio retrieval side are measured directly.

---

## Quick decision matrix

| You have | Task | Recommended | Confidence |
|---|---|---|---|
| **Symbolic only** | Structure | CLaMP3-symbolic **L4 or L11** | HIGH (direct data) |
| **Symbolic only** | Melody / motif retrieval | CLaMP3-symbolic **L11** | HIGH (direct data) |
| **Audio only** | Structure | MuQ **L10** | MED (transfer from pop) |
| **Audio only** | Melody / motif retrieval | MuQ **L8** | MED (have VGM data, no melody-specific) |
| **Both** | Structure (best perf) | **L11(symbolic) + L10(MuQ)**, late-fused | LOW (untested but principled) |
| **Both** | Structure (cross-modal queries) | **L12** on both sides | LOW (untested coherence on VGM) |
| **Both** | Melody retrieval | CLaMP3-symbolic **L11** alone | HIGH (symbolic dominates 5×) |

---

## Case A — audio only (the common VGM deployment case)

### What the data says

No direct VGM-audio-structure result yet. Closest signals from the
existing sweeps:

| Reference task | Best audio encoder | Layer | Score |
|---|---|---|---|
| HookTheoryStructure (audio, pop, 7-class structure) | **MuQ** | L10 | 0.591 acc |
| HookTheoryStructure | OMARRQ-25hz | L17 | 0.589 acc (tied) |
| HookTheoryStructure | CLaMP3-audio | L3 | 0.568 acc |
| HookTheoryStructure | MERT-95M | L4 | 0.558 acc |
| VGMIDITVar (audio, VGM, retrieval) | **MuQ** | L8 | 0.196 MAP |
| VGMIDITVar (audio) | OMARRQ-25hz | L20 | 0.195 MAP |

Both signals agree: **MuQ is the right audio encoder for VGM.**
Layer in the L8–L10 band depending on task type — L10 for
structure-classification (extrapolated from pop), L8 for
retrieval (measured on VGM).

### Alternatives if pure audio isn't cutting it

**Option 1 — transcribe audio → MIDI, then use CLaMP3-symbolic L11.**

[`basic-pitch`](https://github.com/spotify/basic-pitch) (Spotify,
2022) is the current open-source polyphonic transcription baseline.

- **Works well on:** chiptune, monophonic melodies, simple synth
  textures, piano-like timbres. Most pre-PS3 VGM falls here.
- **Struggles on:** dense orchestration, choirs, ambient/soundscape
  layers. Modern AAA OSTs lose too much.
- **Quality bound:** transcription introduces ~5–25% pitch/onset
  error depending on material; the symbolic encoder's accuracy is
  capped by that fidelity.

Decision rule (based on round-trip test, §"Tests to run" #3 below):

| Round-trip drop | Recommendation |
|---|---|
| ≤ 5 pp | Adopt transcribe → CLaMP3-symbolic path |
| 5–15 pp | Marginal; only worth it if downstream needs symbolic specifically |
| > 15 pp | Don't bother; use audio encoders directly |

**Material-dependent heuristic** for predicting where basic-pitch
will land:
- ✓ Chiptune, NES/SNES, monophonic melodies → likely < 5 pp drop
- ~ 16-bit / Genesis-era, simple synth orchestration → 5–15 pp drop
- ✗ Modern AAA full-orchestral OST → > 15 pp drop

**Alternative transcribers** (not wired in MARBLE, listed for
reference):

| Model | Best for | Notes |
|---|---|---|
| [Onsets & Frames](https://magenta.tensorflow.org/onsets-frames) | Solo piano | Historically the strongest piano transcriber; weaker on other instruments |
| [MT3](https://github.com/magenta/mt3) | Multi-instrument polyphonic | Heavier than basic-pitch, better on dense textures |
| [`hierarchical-encoder-music-transcription`](https://github.com/cwitkowitz/timbre-trap) (Cwitkowitz) | Research-grade | Newer; less battle-tested |

**Option 2 — accept the audio-only ceiling.**
Quickest path, fewest moving parts. The gap between best
audio-encoder structure and best symbolic-encoder structure on the
SAME task is likely small (we'll know after the SuperMario audio
sweep). For pop structure, no symbolic baseline exists for direct
comparison.

---

## Case B — symbolic (MIDI) only

### What the data says

CLaMP3-symbolic is the only symbolic encoder MARBLE currently runs,
and it dominates audio encoders by 5–10× on motif retrieval.
Measured directly on three symbolic tasks:

| Task | Best layer | Test value | meanall | L12 (contrastive output) |
|---|---|---|---|---|
| VGMIDITVar (retrieval) | **L11** | 0.198 MAP | — | 0.180 (−1.8 pp) |
| VGMIDITVar-leitmotif (cross-instr retrieval) | **L11** | 0.195 MAP | 0.195 | 0.176 (−1.8 pp) |
| SuperMarioStructure (classification) | **L4 ≈ L11** | 0.599 acc | 0.571 | 0.566 (−3.3 pp) |

Pick L11 across the board as the default. L4 is tied with L11 on
SuperMarioStructure accuracy with marginally better macro-F1; pick
L4 specifically if minority classes matter. L11 + L4 ensemble is
the next thing to try if a marginal lift is needed.

### Alternatives if symbolic-only doesn't cut it

**Option 1 — render audio from the MIDI, then use audio encoders.**

This is what VGMIDITVar / VGMIDITVar-leitmotif builds already do.
Workflow:

```bash
# SoundFont: SGM-Pro 14 (default across MARBLE's VGM benchmarks)
fluidsynth -F out.wav -r 24000 -O s16 SGM-Pro_14.sf2 in.mid
# Then run audio encoder over out.wav
```

Cost: ~5 sec/piece for the render; encoder pass adds 1–2 sec.

**Caveats:**
- **SoundFont choice is a hyperparameter.** VGMIDITVar-multisf was
  built specifically to test this — audio MAP dropped from 0.196
  (single SF) to 0.182 (multi SF). Pick one and stick.
- Render fidelity is an upper bound on the audio encoder's signal —
  it can't be more informative than the audio it consumes.

**Option 2 — just use CLaMP3-symbolic and stop.**
Cheapest, highest-quality option for symbolic tasks. The only
reason to add the audio path is if a specific downstream task needs
timbral / performance features the score doesn't carry.

---

## Case C — both audio + symbolic available

No direct fusion experiments measured yet, but the per-modality
picture is clear and the recipes are principled.

### Recipe 1 — best single-pass performance (default)

```python
import torch.nn.functional as F

emb = torch.cat([
    F.normalize(clamp3_symbolic(midi, layer=11)),  # 768-dim
    F.normalize(muq_audio(audio, layer=10)),       # 1024-dim
], dim=-1)  # 1792-dim
# Train a single MLP on top of this fused embedding.
```

**Why these layers:** L11(symbolic) is the proven dominant layer
for symbolic structure + retrieval. L10(MuQ) is the proven dominant
layer for audio structure (on pop; transfer to VGM assumed). Their
embedding dimensions don't match — that's fine, the concatenated
1792-dim vector is what the MLP probes.

**Expected lift:** +1–2 pp on structure-classification compared
to symbolic alone. Mostly because audio captures timbre /
performance dynamics the score doesn't have. (Untested on VGM.)

### Recipe 2 — cross-modal queries (multidomain use case)

```python
emb_audio  = clamp3_audio(audio, layer=12)       # 768-dim
emb_symbol = clamp3_symbolic(midi, layer=12)     # 768-dim
# These live in the SAME contrastive embedding space.
# cosine(emb_audio, emb_symbol) is now meaningful.
```

**Cost:** 1.8–3.3 pp performance loss vs L11 on single-modality
tasks (see [`supermario_findings.md` §5](supermario_findings.md#5-l12-deep-dive--when-to-use-the-contrastive-output))
and [`layer_analysis.md` "When to use L12"](layer_analysis.md#when-to-use-l12--the-contrastive-output-layer).

**Buys you:** the ability to do cross-modal retrieval — "find the
MIDI score for this audio recording" or vice-versa via cosine
ranking. The only valid use case for L12 over L11 is when this
shared-space property is what you need.

### Recipe 3 — triple-layer maximum-information (research only)

```python
emb = torch.cat([
    F.normalize(clamp3_symbolic(midi, layer=4)),   # local syntactic
    F.normalize(clamp3_symbolic(midi, layer=11)),  # discourse-level
    F.normalize(clamp3_symbolic(midi, layer=12)),  # cross-modal aligned
    F.normalize(muq_audio(audio, layer=10)),       # audio structural
], dim=-1)  # 3328-dim
```

Overkill for most deployments. Worth piloting only when retrieval
ceiling matters more than embedding size.

---

## Melody analysis — where we're thin

Melody is a stated priority but the data is partial. State of the
play:

| Task | Audio | Symbolic | Best layer (if known) |
|---|---|---|---|
| **HookTheoryKey** (key classification, 24-class) | ✓ CLaMP3 L0=0.639, MuQ L0=0.626 | — | **L0** wins — pitch lives in early layers |
| **HookTheoryMelody** (f0 extraction) | NOT RUN | — | unknown |
| **GS** (GuitarSet key, 24-class) | ✓ CLaMP3 L0=0.586, MERT L9=0.599 | — | mixed: L0 (CLaMP3) vs L9 (MERT) |
| **VGMIDITVar-leitmotif** (motif retrieval) | ✓ all 4 audio encoders | ✓ CLaMP3-symbolic | L11 (symbolic), L8–L11 (audio) |
| **VGM melody dataset** | not built | not built | — |

**Practical inference for VGM melody:**

- **Pitch / key / chord-level features:** L0 of any audio encoder.
  Robust pattern across HookTheoryKey, GS.
- **Melodic line retrieval (motif / theme):** CLaMP3-symbolic L11 if
  you have MIDI, MuQ L8 if you only have audio.
- **Melody-conditioned generation / similarity:** L11 either side;
  concat at L12 for cross-modal.

**Missing:** a melody-specific benchmark on VGM. The closest task
that exists (VGMIDITVar-leitmotif) is cross-instrument motif
retrieval, which is melody-adjacent but not melody-extraction. For
melody-specific evaluation, either:

1. Wire HookTheoryMelody as a pop baseline (~3 hr sweep, dataset
   already documented in [`data/hooktheory.md`](data/hooktheory.md)).
2. Build a VGM melody dataset (real new task — needs melody
   annotations, probe template, splits).

---

## Tests to run to firm any of this up

Priority order. Each is independently valuable; later tests depend
on earlier ones.

### 1. SuperMarioStructure audio sweep (highest-value, lowest-risk)

**What it gives:** direct measurement of audio encoders on VGM
structure. Currently extrapolating from HookTheoryStructure (pop).
Confirms whether MuQ L10 transfers or surfaces a different winner
for VGM specifically.

**Cost:** ~15 min fluidsynth render of 554 MIDIs + ~3–4 hr GPU
sweep (4 encoders × {layers, meanall}).

**How:**
```bash
# 1. Build with audio (re-uses the existing data/SuperMarioStructure/mid)
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir data/SuperMarioStructure/mid \
    --audio-dir data/SuperMarioStructure/audio_sgmpro14

# 2. Sweep
uv run python scripts/sweeps/run_all_sweeps.py --tasks SuperMarioStructure
```

### 2. HookTheoryMelody sweep (gives audio-melody data)

**What it gives:** melody-specific baseline on audio (pop). Tests
whether the "L0 wins on melody" pattern from key estimation holds.

**Cost:** dataset already documented; ~3 hr sweep.

### 3. basic-pitch round-trip on SuperMario (critical for audio-only)

**What it gives:** measures the cost of audio → MIDI → symbolic
workaround. If small, the workaround is defensible; if large, audio
encoders alone are better.

**Cost:** ~30 min basic-pitch transcription of 50 sample tracks +
re-build pilot + compare CLaMP3-symbolic L11 accuracy on original
vs transcribed.

**How:**
```bash
# 0. Pre-req: pip install basic-pitch
# 1. Render audio from MIDI (we'd render anyway for test #1)
fluidsynth -F /tmp/orig/00001.flac SGM-Pro_14.sf2 mid/00001_*.mid
# 2. Transcribe back
python -c "from basic_pitch.inference import predict_and_save; \
    predict_and_save(['/tmp/orig/00001.flac'], '/tmp/transcribed/', ...)"
# 3. Build pilot from transcribed MIDIs
uv run python scripts/data/build_supermario_dataset.py \
    --midi-source-dir /tmp/transcribed --max-pieces 50 \
    --data-dir data/SuperMarioStructure/_basicpitch_pilot
# 4. Compare CLaMP3-symbolic L11 accuracy on original vs transcribed
```

### 4. L12 cross-modal coherence test on VGM

**What it gives:** empirical check that CLaMP3's contrastive
training actually aligns audio + symbolic on VGM. If
`cosine(L12_audio, L12_symbolic)` is high for matching pieces and
low for mismatched, L12 is a valid cross-modal embedding for VGM.

**Cost:** needs the audio sweep first. Then ~10 min: load both
per-piece L12 embeddings, compute pairwise cosines, plot
intra-piece vs inter-piece histograms.

### 5. SoundFont robustness for structure (lower priority)

**What it gives:** tells you whether the rendered-audio path is
stable. VGMIDITVar-multisf showed retrieval MAP dropped 0.196 →
0.182 across soundfonts; structure response is unknown.

**Cost:** render SuperMario with 2–3 soundfonts (SGM-Pro,
FluidR3-GM, Sonatina), run audio sweep on each, compare. ~1 day.

---

## Should we render audio from MIDI?

**For benchmark builds: YES** when MIDI is the source-of-truth.
This is what VGMIDITVar / VGMIDITVar-leitmotif already do (SGM-Pro
14, single SF).

**Pros:**
- Lets you test audio encoders on tasks that wouldn't otherwise
  have audio
- Removes acoustic variability, clean cross-piece comparison
- Cheap (~5 sec/piece) and reproducible

**Cons:**
- SoundFont choice is a hyperparameter; results don't always
  transfer to "real" audio
- Numbers will be optimistic compared to deployment on actual
  game audio

**Workflow for SuperMario:** render once with SGM-Pro 14 (matches
the rest of MARBLE's VGM benchmarks); add a second SF later for
test #5 if needed.

---

## Should we transcribe audio to MIDI (basic-pitch)?

**Maybe.** Conditional on the round-trip test (#3 above):

- **≤ 5 pp drop on a 50-piece sample**: adopt the path for
  audio-only deployments
- **5–15 pp drop**: marginal; only worth it if downstream needs
  the symbolic representation specifically
- **> 15 pp drop**: don't bother; use audio encoders directly

Material-dependent heuristic as in Case A above.

If you decide to seriously evaluate this, the natural
implementation is a `scripts/data/transcribe_audio_to_midi.py`
wrapper around basic-pitch that produces files in the same
`<piece_id>_<slug>.mid` naming convention the SuperMario build
script consumes. One short script away from a one-command pipeline.

---

## One-line summary

- **Symbolic** = solved: CLaMP3-symbolic L11 default; L4 + L11 ensemble for squeezed classification accuracy.
- **Audio** = MuQ L10 is the bet, but it's a transfer assumption from pop until the SuperMario audio sweep runs.
- **Both** = run in parallel; ensemble at L11 for performance, L12-on-both for cross-modal queries.
- **Melody on VGM** = biggest data gap. CLaMP3-symbolic L11 works if you have MIDI; otherwise guess from key estimation + retrieval results.
- **Single most valuable next test** = audio sweep on SuperMarioStructure. Everything else is hypothesis until that lands.

---

## See also

- [`supermario_findings.md`](supermario_findings.md) — SuperMarioStructure CLaMP3-symbolic sweep results
- [`leitmotif_findings.md`](leitmotif_findings.md) — VGMIDITVar-leitmotif cross-instrument retrieval
- [`layer_analysis.md`](layer_analysis.md) — canonical cross-encoder per-task layer-selection reference
- [`structure_datasets_survey.md`](structure_datasets_survey.md) — other structure datasets we could add
- [`data/supermario_setup.md`](data/supermario_setup.md) — build commands + class inventory
- [`data/vgmiditvar_setup.md`](data/vgmiditvar_setup.md) — MIDI → audio render pipeline reference
