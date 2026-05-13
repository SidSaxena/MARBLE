# Leitmotif Detection on Star Wars: CLaMP3 + SWTC

End-to-end plan for using CLaMP3's cross-modal capabilities together with
the [Star Wars Thematic Corpus](https://github.com/Computational-Cognitive-Musicology-Lab/Star-Wars-Thematic-Corpus)
(SWTC) and Frank Lehman's [Star Wars Thematic Catalogue](https://franklehman.com/starwars/)
to detect, localise, and analyse leitmotifs in John Williams's
Star Wars scores.

This document is the project-specific companion to [`leitmotif_clamp3.md`](./leitmotif_clamp3.md).
That doc covers the general methodology; this one is about *this dataset*.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Can CLaMP3 detect Star Wars leitmotifs from MIDI templates? | Yes, at 5-sec resolution. Workflow 1 below is the canonical pipeline. |
| Do I need MARBLE to do this? | No — the upstream CLaMP3 CLI handles the basic pipeline in ~1 day. |
| When is MARBLE worth integrating? | When you want layer sweeps, cross-encoder comparison, reproducible benchmarks. |
| Can I share the audio side? | No — Williams's recordings are copyrighted. You provide your own copy locally. |
| Can I share the symbolic side? | Yes — SWTC is publicly distributed on GitHub. The MARBLE task ships theme MIDIs only; audio paths point at the user's local files. |
| What's the ground truth? | Lehman's catalogue: timestamps for theme appearances in each film. |

---

## 1. The corpus

### 1.1 SWTC GitHub repository

Source: [Computational-Cognitive-Musicology-Lab/Star-Wars-Thematic-Corpus](https://github.com/Computational-Cognitive-Musicology-Lab/Star-Wars-Thematic-Corpus)

| | |
|---|---|
| Themes | **67** distinct leitmotifs from Episodes I–IX |
| Original Trilogy | 25 themes (incl. Main Theme A/B, Force Theme, Imperial March, Force Theme, Han & Leia, Yoda, etc.) |
| Prequel Trilogy | 17 themes (Anakin's Theme, Duel of the Fates, Battle of the Heroes, ...) |
| Sequel Trilogy | 25 themes (Rey A/B, Kylo A/B/C, March of the Resistance, ...) |
| Formats | `.sib` (Sibelius), `.musicxml`, `.krn` (humdrum) |
| Primary format | `.krn` is the canonical human-and-computer-readable form |

### 1.2 The kern spines (annotations)

Each `.krn` file may contain multiple parallel "spines":

| Spine | Content |
|---|---|
| `**kern` | Melodic information (always present) |
| `**harte` | Chord labels (Harte 2010 format) |
| `**harm` | Roman numeral analysis |
| `**altharm` | Alternative Roman numeral interpretations |
| `**pedal` | Pedal tone annotations |
| `**cadence` | Cadence types + formal boundary markings |
| `**text` | Free-form analytical notes (modulation type, textural notes...) |

For leitmotif detection we primarily care about `**kern` (the actual notes).
The other spines are scholarly value-adds for the analyses in §6.

### 1.3 Lehman's thematic catalogue

The catalogue document (Lehman, 2022) is the *scholarly* counterpart to
the corpus. It includes:

- A score engraving for each theme
- The films each theme appears in
- **Timestamps for the theme's first presentation in the associated film**
- Analytical commentary on each theme's function and transformations

The timestamps are the **ground truth for localisation evaluation** — see §5.

### 1.4 What the corpus does NOT contain

- **No audio.** The actual John Williams orchestral recordings are
  copyrighted and not redistributed. You must legally own a copy of
  the Star Wars OSTs (or any other Star Wars recordings) to run
  any audio-side workflow.
- **No film-music alignments.** Lehman's timestamps are *film*
  timestamps. OST track timings differ from film timings — there's a
  one-time manual mapping step needed for evaluation.
- **No labels for recurrences in every cue.** Lehman documented
  notable / first appearances of each theme. He did NOT exhaustively
  label every quotation across every cue. Expect that any system you
  build will find "false positives" that are actually true positives
  that aren't in the catalogue.

---

## 2. Why this dataset is uniquely valuable for leitmotif research

1. **Gold-standard ground truth.** A musicology PhD's hand-curated
   catalogue, peer-reviewed and edited. You will not find this
   quality for game soundtracks, anime, or other film franchises.
2. **Closed musical universe.** All 67 themes were composed by the
   same composer in a roughly consistent stylistic idiom. Controlled
   stylistic dimension — the only sources of variation are
   *theme identity* and *arrangement/transformation*, which is
   exactly what a leitmotif detector should track.
3. **Bounded scale.** 67 themes is large enough for meaningful
   evaluation but small enough that **per-theme analysis is feasible**.
   You can manually inspect every detection's failure mode.
4. **Multimodal symbolic richness.** Beyond pitch, the kern annotations
   make several extra scholarly experiments possible (see §6).
5. **Public scholarly support.** Lehman has published extensively on
   Williams's thematic technique. Your results have a peer-reviewed
   theoretical literature to compare against, which is rare in MIR.

---

## 3. CLaMP3's role: what it can and can't do here

### 3.1 What CLaMP3 brings

CLaMP3's contrastively-trained shared embedding space means
**a MIDI theme and an audio cue containing that theme** land near each
other in the same 768-dim space — without any task-specific training.

For leitmotif detection in Star Wars specifically:

- **Symbolic encoder** consumes the 67 theme MIDIs (after `.krn` →
  MIDI conversion). One 768-dim vector per theme.
- **Audio encoder** consumes 5-second sliding windows over your OST.
  One 768-dim vector per window.
- **Cosine similarity** in the shared space tells you where in the
  audio each theme is most likely playing.

No labelled audio examples needed. **One MIDI per theme is sufficient
to produce a detection curve.**

### 3.2 What CLaMP3 cannot do here

| Want | Why CLaMP3 can't | Alternative |
|---|---|---|
| Sub-second timing (e.g. "Imperial March enters at 0:42.3") | 5-sec chunk ceiling | Coarse detection + frame-level encoder for refinement (§7.3) |
| Detect heavily fragmented theme quotes (2–3 notes of a melody) | Chunk-level pooling washes out short cells | Hybrid with melody-extraction probes |
| Recognise themes outside Williams's stylistic envelope | Limited pop/contemporary bias in training data | Domain-specific fine-tuning (out of scope) |
| Distinguish two themes that share intervallic DNA (e.g. Empire family) | Embeddings may cluster | Inspect confusion matrix, possibly add a fine-tuned head |

### 3.3 The cross-modal API (now implemented in MARBLE)

As of the latest commits, the three modality methods are available on
`CLaMP3_Encoder` and inherited by `CLaMP3_Symbolic_Encoder`:

```python
from marble.encoders.CLaMP3.model import CLaMP3_Symbolic_Encoder

enc = CLaMP3_Symbolic_Encoder().eval().cuda()

# Each method returns (B, 768) L2-normalised shared-space embeddings:
e_themes = enc.embed_symbolic(theme_patches)        # (67, 768)
e_audio  = enc.embed_audio(audio_windows)           # (N, 768)
e_text   = enc.embed_text(["heroic fanfare in C"])  # (M, 768)

# Cross-modal similarity is just inner product
detection_curves = e_audio @ e_themes.T             # (N_windows, 67)
```

All three methods have been **end-to-end tested with the real CLaMP3
checkpoint** (see `marble/encoders/CLaMP3/model.py:CLaMP3CrossModalMixin`).

---

## 4. Path A: CLaMP3 CLI workflow (no MARBLE needed)

This is the **fast scientific-validation path**. Goal: see whether the
approach produces meaningful detections on your data before investing
in framework integration.

### 4.1 Setup

```bash
# Clone CLaMP3 upstream
git clone https://github.com/sanderwood/clamp3 ~/clamp3
cd ~/clamp3 && pip install -r requirements.txt

# Clone SWTC (just for the .krn files)
git clone https://github.com/Computational-Cognitive-Musicology-Lab/Star-Wars-Thematic-Corpus ~/swtc

# Install humlib for kern→MIDI conversion
brew install humdrum-tools   # macOS; on Linux build from source
```

### 4.2 Convert SWTC `.krn` → MIDI

```bash
mkdir -p ~/swtc-midi
for f in ~/swtc/*Triology*/Krns/*.krn; do
    name=$(basename "$f" .krn)
    humdrum2mid -i "$f" -o ~/swtc-midi/"$name".mid
done

# Sanity check
ls ~/swtc-midi | wc -l    # should be 67
```

If humdrum2mid is unavailable, alternative: use Music21 or any
`.musicxml` → `.mid` converter on the `Xmls/` folder instead.

### 4.3 Embed every theme

```bash
cd ~/clamp3
python clamp3_embd.py ~/swtc-midi/ --get_global
# Output: ~/swtc-midi/*.npy (67 vectors, each (768,))
```

`--get_global` returns the L2-normalised shared-space embedding from
the symbolic projection head — exactly what you want for cross-modal
search.

### 4.4 Slice your OST into 5-sec windows

```bash
mkdir -p ~/sw-audio-windows
# For each track, slide 5-sec windows with 1-sec hop
for track in ~/sw-ost/*.flac; do
    name=$(basename "$track" .flac)
    duration=$(ffprobe -v error -show_entries format=duration \
               -of csv=p=0 "$track")
    end=$(awk "BEGIN{print int($duration - 5)}")
    for start in $(seq 0 1 "$end"); do
        ffmpeg -nostdin -ss "$start" -t 5 -i "$track" \
               -ar 24000 -ac 1 \
               "~/sw-audio-windows/${name}_${start}.flac" 2>/dev/null
    done
done
```

This produces hundreds-to-thousands of 5-second windows per OST track.

### 4.5 Embed all audio windows

```bash
python clamp3_embd.py ~/sw-audio-windows/ --get_global
# Output: ~/sw-audio-windows/*.npy (one vector per window)
```

### 4.6 Search themes against audio (or vice versa)

```bash
# For each theme, rank audio windows by similarity
for theme in ~/swtc-midi/*.npy; do
    name=$(basename "$theme" .npy)
    python clamp3_search.py "$theme" ~/sw-audio-windows/ \
           > ~/results/"$name".txt
done
```

Output: ranked list of windows per theme, with similarity scores.

### 4.7 Inspect for sanity

Pick 5 themes you know well and listen to the top-10 windows the system
proposes:

- Imperial March → expect Death Star scenes, Vader entrances
- Force Theme → expect Luke-on-Tatooine moments, hero moments
- Han & Leia → expect romantic cues from Empire
- Rey's Theme → expect Force Awakens / Last Jedi character moments
- Duel of the Fates → expect prequel lightsaber battles

If the top-10 contains 6+ correct matches, the approach works.
If it's mostly noise, see §3.2 — Williams's fragmentation may be too
aggressive for chunk-level matching, and you'll need the hybrid
approach in §7.3.

**Estimated time end-to-end: 4–8 hours**, dominated by audio-window
extraction and CLaMP3 inference (GPU-accelerated, ~50 windows/sec).

---

## 5. Path B: MARBLE integration (the reproducible benchmark)

Do this **after** Path A confirms the science works. MARBLE integration
buys you:

- **Layer sweep** across CLaMP3's 13 BERT layers (Path A only uses the
  final projection-head output)
- **Cross-encoder comparison** (CLaMP3 vs MERT vs OMARRQ on the same
  data, same splits)
- **YAML-tracked configuration** (no ad-hoc shell pipelines)
- **WandB run tracking** (filter by encoder, model, layer; share
  result dashboards)
- **Reproducible-by-someone-else** — your future self in 12 months
  will be glad

### 5.1 Two task variants

**Variant A — `SWTCRetrieval`: theme → audio cue MAP**

Direct adaptation of the existing retrieval pattern (Covers80,
SHS-100K, VGMIDI-TVar), but cross-modal:

| | |
|---|---|
| Query items | 67 theme MIDIs from SWTC `.krn` (converted to MIDI) |
| Target items | User's OST audio, sliced into 5-sec sliding windows |
| Positive pair | (theme_i, window_j) where window_j contains theme_i per Lehman |
| Metric | MAP — for each theme query, AP over all target windows |
| Encoder | CLaMP3 only (uniquely cross-modal) |
| Layer sweep | Yes — finds best CLaMP3 layer for leitmotif identity |
| Stage | `test` only (zero-shot, no probe training) |

This is the canonical Workflow 1 from §6.1, packaged as a MARBLE task.

**Variant B — `SWTCLocalization`: event-level F1**

Closer to the actual deployable system:

| | |
|---|---|
| Per-track input | full audio cue + ground-truth (theme, start, end) list |
| Per-track output | predicted detections from peak-picking on per-theme similarity curves |
| Metric | event-level F1 with ±2.5 s tolerance |
| Implementation | Separate evaluation script that consumes MARBLE-saved embeddings |

Variant B is scientifically more interesting but doesn't fit MARBLE's
existing probe contract (single-vector-per-clip evaluation). Build it
as a **separate evaluation pipeline** that consumes saved embeddings
from Variant A.

### 5.2 Design challenges unique to this task

These haven't applied to any previous MARBLE task and need careful
handling:

**1. Two-side dataset.** All existing MARBLE retrieval tasks have a
single dataset where all items share a contract. SWTC has **two**
item types — queries (theme MIDIs) and targets (audio windows) — with
different encoders. Cleanest approach:

```python
class SWTCThemeQueries(_VGMIDITVarSymbolicBase):
    """The 67 themes — uses M3Patchilizer pipeline."""
    pass

class SWTCAudioTargets(_LeitmotifAudioBase):
    """Audio windows from user's OST."""
    pass

class CrossModalRetrievalTask(LightningModule):
    """Runs queries through symbolic encoder, targets through audio,
    computes pairwise similarity, reports per-theme MAP."""
    ...
```

**2. The "work_id" assumption breaks.** In Covers80, each item has *one*
work_id and "positive pair" means "same work_id". Here, an audio
window may contain **multiple** themes (Williams stacks themes
constantly), and each theme query is its own "work" with N relevant
windows. The metric is **per-query MAP** (rank all windows per theme,
compute AP per theme, average across themes).

**3. Catalogue → OST timing alignment.** Lehman gives film timestamps.
Your OST tracks have a different timing structure (commercial album
tracks vs scene-by-scene film cues). One-time manual step:

```csv
# data/SWTC/ost_to_film_map.csv
ost_track,ost_start,ost_end,film,film_start,film_end
"Star Wars Main Title",0:00,4:08,IV,0:00,4:08
"Imperial Attack",0:00,3:38,V,0:18:33,0:23:42
...
```

Maybe 2–3 hours of work per OST you want to evaluate against.

**4. Open-set / sparse positives.** Most 5-sec windows contain no
catalogued theme. The retrieval metric treats these as "irrelevant
for all themes" — which is correct, but means the absolute MAP
numbers will be lower than for closed-set retrieval like Covers80.
Don't compare MAP numbers across tasks of different sparsity.

### 5.3 File layout

```
marble/tasks/SWTCLeitmotif/
  __init__.py
  datamodule.py        # SWTCThemeQueries + SWTCAudioTargets
  probe.py             # CrossModalRetrievalTask

scripts/
  build_swtc_dataset.py  # .krn → MIDI; parse Lehman catalogue;
                         # OST → JSONL with timestamps
  swtc_localise.py       # full-track sliding window + peak-picking
  swtc_evaluate.py       # event-level F1 against ground truth

data/SWTC/               # gitignored, user-local
  themes/{1..67}.mid     # converted from SWTC .krn
  audio/                 # user's OST files (.flac, .wav)
  catalogue.csv          # parsed Lehman catalogue
  ost_to_film_map.csv    # OST→film timing alignment

configs/
  probe.CLaMP3-symbolic-layers.SWTCLeitmotif.yaml
```

### 5.4 Implementation checklist

1. **`scripts/convert_swtc_to_midi.py`** — humlib wrapper, ~30 lines
2. **`scripts/build_swtc_dataset.py`** —
   - Reads `swtc_catalogue.xlsx` (or manual parse of the catalogue PDF)
   - Reads `ost_to_film_map.csv` (one-time manual)
   - Outputs `SWTC.queries.jsonl` (67 themes → MIDI paths)
   - Outputs `SWTC.targets.jsonl` (audio windows with `themes_present: [theme_id, ...]`)
3. **`marble/tasks/SWTCLeitmotif/datamodule.py`** — two `Dataset`
   classes + a `DataModule` that yields both
4. **`marble/tasks/SWTCLeitmotif/probe.py`** — `CrossModalRetrievalTask`
   extending the pattern from `CoverRetrievalTask`
5. **`configs/probe.CLaMP3-symbolic-layers.SWTCLeitmotif.yaml`** —
   wires the symbolic and audio sides into the cross-modal task
6. **`scripts/swtc_localise.py`** — sliding-window inference,
   peak-picking, JSON detection output
7. **`scripts/swtc_evaluate.py`** — event-level F1 + confusion matrix
   + per-theme P/R breakdown

Estimated effort: **1–2 days** once Path A has validated the science.

---

## 6. The full ten workflows

Repeating here in compact form so this doc is self-contained. Each
maps onto a tractable next step.

| # | Workflow | What it does | Effort | Novelty |
|---|---|---|---|---|
| 1 | **Theme localisation in OST** | Find every theme occurrence using MIDI templates | ★ | low |
| 2 | **Theme-family embedding map** | UMAP all 67 theme embeddings; compare to Lehman's groupings | ★ | medium |
| 3 | **Open-set detection** | Test that non-SW orchestral music scores below threshold | ★ | low |
| 4 | **Theme transformation analysis** | Quantify how transformed a theme is at each occurrence | ★★ | high |
| 5 | **Cross-trilogy invariance** | Same theme across 40 years of recordings — do embeddings stay close? | ★ | medium |
| 6 | **Anti-leitmotif** | Detect minor-mode / inverted variants of major-mode themes | ★★ | high |
| 7 | **Text-grounded query** | Embed catalogue descriptions; check if they find right moments | ★ | low |
| 8 | **Embedding similarity vs human ratings** | Correlate CLaMP3 cosine with listener pairwise similarity | ★★ | medium |
| 9 | **Catalogue inference for new scores** | Cluster a new score's windows, present to musicologist | ★★ | medium |
| 10 | **Audio-to-MIDI alignment via DTW** | Use cross-modal similarity matrix for score-following | ★★ | low |

### 6.1 Workflow 1 (canonical): Theme localisation

**Inputs:** SWTC theme MIDIs + your OST audio + Lehman timestamps
**Method:** symbolic embed themes, audio embed sliding windows, cosine
similarity matrix, peak-pick per theme
**Output:** ranked time-stamped detections per theme
**Evaluation:** event-level F1 against Lehman's catalogue
**Expected:** F1 0.3–0.5 for full themes; higher for verbatim quotes,
lower for fragmented transformations

### 6.2 Workflow 2: Theme-family embedding map

**Inputs:** SWTC theme MIDIs only — no audio needed
**Method:** symbolic embed all 67 themes, UMAP/t-SNE to 2D, plot
**Output:** scatter plot of the 67 themes in 2D embedding space
**Evaluation:** qualitative — do Williams's known family relationships
appear (Imperial/First Order cluster, Force/Resistance cluster,
prequel modal echoes)?
**Why interesting:** CLaMP3 has never seen Star Wars in training. If
it recovers the family structure as a side effect, that's a
genuinely novel result about its representation.

### 6.3 Workflow 3: Open-set detection

**Inputs:** non-Star-Wars orchestral music + your trained detector
**Method:** run detector on the OOD audio, check that all theme
scores stay below threshold
**Output:** false-positive rate on OOD audio
**Evaluation:** lower is better — should be << 5%
**Why interesting:** Many published "leitmotif detectors" fail this
because they're really detecting "orchestral music similarity" rather
than theme identity. This is a *real* test of leitmotif specificity.

### 6.4 Workflow 4: Theme transformation analysis

**Inputs:** known cue where a known theme appears in transformed form
**Method:** symbolic embed clean theme MIDI, audio embed sliding
windows of the cue, similarity curve over time
**Output:** characterise local maxima as "verbatim" / "transposed" /
"fragmented" / "mode-shifted" — compare to musicological analyses
**Evaluation:** qualitative + by example
**Why interesting:** Provides a **quantitative method for theme
transformation analysis**, currently done entirely by hand.

### 6.5 Workflow 5: Cross-trilogy invariance

**Inputs:** Force Theme appearances across all 9 films (you provide
the audio segments)
**Method:** audio embed each segment; pairwise similarity
**Output:** 9×9 similarity matrix
**Evaluation:** off-diagonal similarities should be high
**Why interesting:** Same source material, different actual orchestral
performances spanning 1977–2019. Tests the most genuine kind of
arrangement invariance.

### 6.6 Workflow 6: Anti-leitmotif (theme inversions / mode shifts)

**Inputs:** known mode-shifted recurrences of themes (Han & Leia in
*Empire* minor mode, etc.)
**Method:** measure similarity to the canonical theme; expect it to
be high but lower than verbatim
**Output:** distribution of "transformation depth" scores
**Evaluation:** correlation with musicological "how transformed is
this" labels

### 6.7 Workflow 7: Text-grounded query

**Inputs:** Lehman's catalogue descriptions; OST audio
**Method:** `enc.embed_text(description)`; similarity to audio windows
**Output:** ranked moments per textual description
**Honest caveat:** CLaMP3's text training is general music captions.
Catalogue-specific language won't match. Generic descriptions
("ascending fourth motif representing nobility" → "stately ascending
fourth motif on horns") might work.

### 6.8 Workflow 8: CLaMP3 similarity vs human similarity

**Inputs:** pairs of themes; human similarity ratings from
listeners (you'd need to run a small study)
**Method:** correlate CLaMP3 cosine with human ratings
**Output:** Pearson r
**Evaluation:** > 0.5 is interesting, > 0.7 is a paper

### 6.9 Workflow 9: Catalogue inference for new scores

**Inputs:** a non-SW film score; no prior theme labels
**Method:** sliding window over score → embed → cluster → present
clusters to a musicologist for labelling
**Output:** candidate themes for a new score
**Evaluation:** musicologist agreement / utility

### 6.10 Workflow 10: Audio-to-MIDI alignment via cross-modal DTW

**Inputs:** a film cue + the MIDI version of its score
**Method:** per-window audio embed, per-position MIDI patches; cosine
similarity matrix; DTW backtracking
**Output:** time-warped alignment path
**Evaluation:** compare to ground-truth alignment from
synctoolbox/chroma-DTW; expect CLaMP3 to win on orchestrational
robustness, lose on fine timing

---

## 7. Practical considerations

### 7.1 Audio sourcing

You must legally own (or have appropriate research access to) any
audio you analyse. Recommended audio sources, ranked by usefulness:

1. **Commercial OST releases** on FLAC/WAV — best timing precision,
   highest production quality. Track ordering ≠ film ordering, but
   alignment is one-time.
2. **Film audio** (ripped from your owned disc) — matches Lehman's
   timestamps directly. Lower production quality (mixed with
   dialogue/SFX), but better alignment.
3. **Concert recordings** (Williams's live performances) — different
   tempos, occasional cuts, but real-orchestra timbre, no dialogue.
4. **Cover/fan arrangements** — useful for cross-arrangement testing
   (Workflow 5), but not for ground-truth evaluation.

For your thesis, the most defensible setup is: legally-owned
commercial OSTs + manual OST→film timing map.

### 7.2 Computational cost

| Operation | Time on M-series Mac (CPU) | Time on Modal T4 |
|---|---|---|
| Embed one theme (symbolic) | ~0.2 s | ~0.05 s |
| Embed one 5-sec audio window | ~0.3 s | ~0.05 s |
| Full OST (4 films × 60 windows/min × 60 min) | ~80 min | ~12 min |
| Pairwise similarity 67 × N_windows | <1 s | <1 s |

For Path A on local CPU, ~2 hours of inference per OST. Path B on
Modal: ~20 min per OST.

### 7.3 Hybrid for sub-second timing

When you need finer timing than CLaMP3's 5-sec ceiling:

```
1. CLaMP3 cross-modal finds the region (5-sec resolution)
2. Inside each candidate region, extract MERT features (75 Hz)
3. Compute MERT chroma similarity to the theme's chroma profile
4. Peak-pick on the fine-grained curve
```

This stays within MARBLE's encoder zoo and doesn't require leaving
the framework.

---

## 8. Scientific questions worth answering

Beyond detection, the corpus enables several research-paper-shaped
questions:

### 8.1 Does CLaMP3 recover Williams's theme families?

Setup: §6.2 + comparison to Lehman's analytical groupings.
Hypothesis: clusters in CLaMP3 embedding space match Lehman's
character/faction/family groupings.
Why interesting: tests whether contrastive music-text training
yields perceptually meaningful musical similarity *without* needing
the target composer in training data.

### 8.2 Can CLaMP3 separate theme identity from arrangement?

Setup: Force Theme across 9 films (1977–2019), pairwise similarity.
Hypothesis: within-theme similarity > between-theme similarity
across all decades and orchestrations.
Why interesting: a clean version of the "is the embedding really
content-invariant" question that's hard to test with synthetic data.

### 8.3 What kinds of theme transformations defeat CLaMP3?

Setup: §6.4 across a curated set of transformations
(transposition, mode shift, augmentation, diminution, fragmentation,
rhythmic alteration).
Hypothesis: pitch-content-preserving transformations are
recognised; rhythmic / fragmentation transformations are not.
Why interesting: maps the *failure boundary* of contrastive audio
embeddings.

### 8.4 Do the kern annotations align with CLaMP3's representations?

Setup: cluster CLaMP3 symbolic embeddings → are clusters explained
by `**harm` Roman numerals? `**cadence` formal positions?
Hypothesis: the model has learned *some* harmonic structure; clusters
partially align with functional analysis.
Why interesting: opens the door to probing what specific musical
properties are represented in which layer.

These four are paper-shaped on their own. None requires you to build
a deployed leitmotif detection system.

---

## 9. Suggested order of work

**Day 1 (CLI validation):**
- Convert SWTC `.krn` → MIDI
- Embed all 67 themes
- Embed one OST track (say, A New Hope)
- Manually inspect top-10 detections for 5 well-known themes
- **Decision point**: does it work?

**Day 2–3 (MARBLE scaffold, if Day 1 succeeded):**
- Build `SWTCLeitmotif` task + datamodule
- Build `build_swtc_dataset.py` (parse catalogue, build JSONLs)
- Run the 13-layer sweep on CLaMP3-symbolic × SWTC
- Look at per-layer MAP — which CLaMP3 layer best captures leitmotif identity?

**Week 1 (full evaluation):**
- Manual OST→film timing alignment for all OSTs you have
- Run `swtc_evaluate.py` for event-level F1 + per-theme breakdown
- Workflow 3 (open-set) on non-Star-Wars orchestral audio

**Week 2+ (scholarly work):**
- Workflow 2 (theme-family embedding map) — paper-shaped
- Workflow 5 (cross-trilogy invariance) — paper-shaped
- Workflow 4 (theme transformation analysis) — paper-shaped

---

## 10. Open questions / known limitations

| Limitation | Mitigation |
|---|---|
| 5-sec chunk granularity | §7.3 hybrid approach |
| Williams's fragmentation defeats some matches | Live with reduced recall; characterise where |
| Catalogue is not exhaustive labelling | Treat unlabelled high-similarity as "candidate", not "false positive" |
| OST/film timing mismatch | One-time manual mapping (~3 h per OST) |
| Audio copyright | Keep audio local; never commit; never share derived embeddings beyond cosine values |
| Lehman's catalogue + the corpus are still recent (~2022); future updates may revise themes | Pin to a corpus commit hash for reproducibility |

---

## 11. Reading list

- Lehman, F. (2022). *Star Wars Thematic Catalogue*. franklehman.com/starwars
- Lehman, F. (2018). *Hollywood Harmony: Musical Wonder and the Sound of Cinema* (book)
- Arthur, C., McNamara, J. & Lehman, F. The SWTC repository readme.
- Wu, S., et al. (2024). *CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages*. ISMIR.
- Gao, C., et al. (2024). *Variation Transformer: Theme-and-Variation Generation*. ISMIR.
- Bohnenstiehl, K. & Beauchamp, J. (2025). *MARBLE: Music Audio Representation Benchmark for Universal Evaluation*.
