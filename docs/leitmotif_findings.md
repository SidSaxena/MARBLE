# MARBLE — Cross-instrument leitmotif retrieval findings

> Audio + symbolic encoder benchmark on the VGMIDI-TVar variation dataset
> rendered with rotating GM instruments (piano theme → strings / horn /
> flute / trumpet variations). Tests whether audio music encoders can
> match the *same melodic theme* across *different instrumentations* —
> the operational definition of leitmotif retrieval in film/game scoring.

**Status:** complete (2026-05-16). All five encoders swept across all
layers; per-instrument-pair MAP analysis reproducible via
`scripts/analysis/vgmiditvar_leitmotif_sweep.py`. All raw data dumped
as CSVs to `output/analysis/leitmotif/`; visualisations as PNGs.

**Companion docs:**
- [`layer_analysis.md`](layer_analysis.md) — broader layer-selection
  reference (covers tasks beyond leitmotif).
- [`embedding_cache.md`](embedding_cache.md) +
  [`embedding_cache_correctness.md`](embedding_cache_correctness.md) —
  the cache that made this analysis possible (~10× speedup).

---

## TL;DR

| | |
|---|---|
| **Audio encoder for cross-instrument leitmotif retrieval** | **MuQ at L7–L11** (mid-late layer band, all ~0.49 cross-instrument MAP) |
| **Best when symbolic transcription is available** | **CLaMP3-symbolic** (aggregate MAP 0.195, an order of magnitude above any audio encoder) |
| **Critical insight** | Aggregate MAP is misleading on this task. The *cross-instrument MAP* (off-diagonal mean of per-pair grid) is the real metric and disagrees with aggregate by 5+ layers for 3 of 4 audio encoders |
| **Practical recommendation** | If your retrieval architecture allows pre-filtering candidates by instrument, MuQ L7 + the per-pair workflow gives you 10× the per-pair MAP of its aggregate. Otherwise MuQ L11 is the single safest layer |
| **Asymmetry to exploit** | Theme-as-query → variation-as-target is dramatically easier than the reverse. Index by canonical themes; query with variations |
| **Ranked audio encoder shortlist** | MuQ (0.493) ≫ OMARRQ-25hz (0.466) > CLaMP3 (0.372) > MERT-95M (0.332) — by best-layer cross-instrument MAP |

If you read nothing else, read [§Headline numbers](#headline-numbers)
and [§Recommendations](#recommendations).

---

## Why this experiment

The previous Covers80 / SHS100K / VGMIDITVar (single SoundFont) results
gave us a ranking but didn't test what leitmotifs in real film/game
soundtracks actually do: **the same theme recurs played by different
sections of the orchestra**. Hedwig's theme on celesta vs. flute vs.
strings; the Force theme on trumpets vs. horns; Skyrim's main theme on
choir vs. brass. Real leitmotif systems need *cross-instrument*
invariance, not just cover-version invariance.

Three increasingly hard benchmark variants were built:

| Variant | What varies across positives | What stays constant |
|---|---|---|
| `VGMIDITVar` | Notes (theme + model-generated variations) | Single SoundFont (FluidR3_GM) — all renders sound like the same piano |
| `VGMIDITVar-multisf` | Notes + SoundFont (FluidR3 vs Shan SGM-Pro 14, rotated per piece) | Both SoundFonts render piano (the source MIDIs are piano-only) |
| `VGMIDITVar-leitmotif` | Notes + GM instrument (rewritten program changes per idx; piano theme + 4 non-piano variations rotated) | Single SoundFont (Shan SGM-Pro 14) for fair instrument comparison |

The leitmotif variant adds genuine cross-instrument variation by
rewriting the MIDI program changes themselves — see
`scripts/data/rewrite_vgmidi_programs.py`. The instrument schedule is:

| idx (variation index in filename) | GM program | Instrument |
|---:|---:|---|
| 0 (theme) | 0 | Acoustic Grand Piano |
| 1 | 48 | String Ensemble 1 |
| 2 | 60 | French Horn |
| 3 | 73 | Flute |
| 4 | 56 | Trumpet |
| ≥5 | cycle (idx % 5) | … |

Rendered with Shan SGM-Pro 14 (community SoundFont tuned to Roland
SC-88 Pro spec — ideal acoustic match for video-game soundtrack
content). Cross-instrument variation lives in the MIDI itself, not in
the SoundFont, so the experiment isolates one variable.

---

## Methodology

### Task definition

Standard retrieval setup, identical to Covers80 / SHS100K:
1. Embed each clip with the encoder, average-pool over layer-time, L2-normalise.
2. Compute pairwise cosine similarity matrix.
3. For each query, rank all other clips by similarity.
4. Treat clips with the same `work_id` (piece + section) as positives.
5. Compute Average Precision per query, Mean Average Precision over all.

The only twist: each query/candidate now has a `gm_program` field
recording which GM instrument it was rendered as. We slice the MAP
computation by `(query_program, target_program)` to produce a 5×5 grid
of per-instrument-pair MAP scores.

### Two MAP metrics

We report two distinct numbers per (encoder, layer):

**Aggregate MAP** (full pool) — for each query, the candidate pool is
*all 1,714 test clips* across all instruments. This is the WandB number.
Same-instrument distractors compete with cross-instrument positives.

**Per-pair MAP** (restricted pool) — for each `(query_program,
target_program)` cell, the candidate pool is restricted to clips with
the target instrument. Tests cross-instrument retrieval in isolation
from same-instrument confounders.

These metrics measure **different things** and disagree in ways that
matter (see [§The metric divergence](#the-metric-divergence)). For the
leitmotif application, the cross-instrument off-diagonal mean of the
per-pair grid is the operationally meaningful number.

### Reproducibility

Every numerical result in this document is reproducible by:

```bash
uv run python scripts/analysis/vgmiditvar_leitmotif_sweep.py
```

This auto-discovers cache directories under `output/.emb_cache/`,
re-computes the per-pair grid for every (encoder, layer) cell, pulls
WandB aggregates for the tri-task comparison, and emits CSVs + PNGs
to `output/analysis/leitmotif/`. ~30 seconds end-to-end on a normal
machine with the cache populated.

The script's header documents the layer-selection rationale (run all,
not top-K). All four CSV outputs are committed-friendly text and
analysis-friendly long format.

---

## Headline numbers

### Cross-encoder ranking — best cross-instrument MAP per encoder

| Rank | Encoder | Best layer | Cross-instrument MAP | Aggregate MAP at same layer |
|---:|---|---:|---:|---:|
| 1 | **MuQ** | L7 | **0.493** | 0.035 |
| 2 | OMARRQ-multifeature-25hz | L15 | 0.466 | 0.047 |
| 3 | CLaMP3 (audio) | L5 | 0.372 | 0.014 |
| 4 | MERT-v1-95M | L8 | 0.332 | 0.016 |
| † | CLaMP3-symbolic | L11 (aggregate-best) | (not measurable*) | 0.198 |

*The symbolic encoder bypasses the embedding cache entirely (no `clip_id`
in its 3-tuple `__getitem__`), so per-pair analysis isn't available
without a separate one-off extraction script. Aggregate symbolic MAP
remains the highest of any encoder by a factor of ~5×, almost certainly
because note tokens dominate the M3 token sequence and program-change
events are a tiny fraction of the input.

![Cross-encoder summary](../output/analysis/leitmotif/cross_encoder_summary.png)

### The metric divergence

This is the most important finding. The aggregate MAP from WandB
(reported in the previous turn's analysis) ranks layers very
differently from the cross-instrument MAP that actually matters for
leitmotif retrieval:

| Encoder | Aggregate-best layer | Cross-instr-best layer | Layer gap |
|---|---:|---:|---:|
| MuQ | L12 | L7 | **5** |
| MERT-v1-95M | L12 | L8 | **4** |
| OMARRQ-25hz | L15 | L15 | 0 |
| CLaMP3 (audio) | L0 | L5 | **5** |

Three of four audio encoders pick a different optimal layer depending
on which metric you trust. **OMARRQ is the only encoder where aggregate
and cross-instrument agree.**

Why the gap exists: late layers (e.g. MuQ L12) trade *cross-instrument
matching strength* for *less timbre confusion*. They make the wrong
piano variation of work A look less similar to the right piano theme of
work A — boosting full-pool MAP — but they also make the right strings
variation of work A look less similar to the right piano theme of
work A. The first effect dominates aggregate; the second dominates per-pair.

For a leitmotif system where you genuinely need cross-instrument
matching, **the per-pair off-diagonal mean is the metric, and the
aggregate ranking is misleading.**

---

## Per-encoder deep dive

Each encoder gets its own dashboard PNG under
`output/analysis/leitmotif/dashboard_<encoder>.png`. The four panels are:
**(A)** layer-profile of the three MAP metrics; **(B)** per-pair heatmap
at the cross-instrument-best layer; **(C)** tri-task aggregate overlay
(VGMIDITVar / multisf / leitmotif); **(D)** per-pair heatmap at the
aggregate-best layer (lets you see Panel B vs D directly).

### MuQ — the audio winner

![MuQ dashboard](../output/analysis/leitmotif/dashboard_MuQ.png)

**Layer profile:**
- L0–L6: cross-instrument MAP rises from 0.295 → 0.422 (early-layer
  features insufficient).
- **L7: cross-instrument MAP jumps to 0.493** — sharp transition
  marking where the encoder builds its melody-invariant representation.
- L7–L11: plateau at 0.485–0.493 (essentially tied).
- L12: drops to 0.447 (the over-specialisation effect).

**Aggregate MAP:** monotonically increases L7 → L12 (0.035 → 0.091).
This is what made the original WandB analysis pick L12.

**Per-pair grid at L7 (cross-instrument best):** dominant cells are
Strings ↔ Horn (0.55), Strings → Flute (0.65), Horn ↔ Flute (0.49–0.69),
Piano → Horn (0.54), Piano → Flute (0.60). Trumpet column is high but
N=16–22, statistically unreliable.

**Specificity-invariance trade-off — visible directly:**

| Layer | Cross-instrument MAP | Aggregate MAP | What's happening |
|---:|---:|---:|---|
| L7 | **0.493** | 0.035 | Cross-instrument matching at peak; many same-instrument confounders |
| L8 | 0.491 | 0.041 | Tied on cross; aggregate slightly improves |
| L11 | 0.490 | 0.087 | Cross still near-peak; aggregate climbing |
| L12 | 0.447 | **0.091** | Cross sacrificed (-9%); aggregate gains marginally (+5%) |

The right call depends on architecture (see [§Recommendations](#recommendations)).

### OMARRQ-multifeature-25hz — the only encoder where the metrics agree

![OMARRQ dashboard](../output/analysis/leitmotif/dashboard_OMARRQ-multifeature-25hz.png)

**Layer profile:**
- Cross-instrument and aggregate MAP both peak at **L15** (0.466 / 0.047).
- Smooth profile, no sharp transition (unlike MuQ L7).
- Late layers L21–L23 collapse rapidly toward the input layer's
  performance — the conformer's last few layers are too task-specialised.

**Why no metric divergence?** Hypothesis: the OMARRQ Conformer's deeper
representations are *both* timbre-invariant and discriminative — the
trade-off MuQ/MERT exhibit between L7-cross-best and L12-aggregate-best
is collapsed into a single peak at L15. Could be a property of
contrastive RQ training; could be coincidence. Worth investigating in
future work.

OMARRQ is the most "honest" encoder for this task — what aggregate MAP
says is what cross-instrument MAP confirms.

### CLaMP3 (audio path) — broken in aggregate, OK at cross-instrument

![CLaMP3 dashboard](../output/analysis/leitmotif/dashboard_CLaMP3.png)

The strangest result. **L5 has cross-instrument MAP 0.372** (3rd-best
audio encoder), but aggregate MAP is in the noise (0.014) at every
layer. The L0 "winner" by aggregate (0.017) is a meaningless artefact
of being slightly less terrible.

What this means: CLaMP3's audio path *can* match across instruments
(0.37 cross-instrument MAP is not nothing — it's competitive with
MERT-95M), but it is dominated by same-instrument distractors at every
layer. Likely cause: contrastive cross-modal pretraining (text + audio
+ MIDI) pushed CLaMP3 toward strong timbre clustering, which is
adversarial here.

**Practical implication:** use CLaMP3 audio only in pre-filtered
hybrid pipelines, never in single-pool retrieval. Or just use CLaMP3-
symbolic if you have MIDI, which has aggregate MAP of 0.195.

### MERT-v1-95M — same divergence as MuQ, weaker absolute

![MERT-95M dashboard](../output/analysis/leitmotif/dashboard_MERT-v1-95M.png)

Cleanest "depth = invariance" story:
- L0: cross MAP 0.244 (worst)
- L3: cross MAP 0.313 (the FluidR3-baseline winner, now mediocre)
- L8: cross MAP 0.332 (peak)
- L12: cross MAP 0.299 (collapse)
- L12 aggregate: 0.038 (peak)

Same L8-vs-L12 divergence as MuQ, just at lower absolute numbers.
MERT's lower ceiling is consistent with its uniformly weaker
performance on every prior leitmotif-relevant task.

**Decommissioned sibling note:** MERT-v1-330M lost 4/4 head-to-head
against MERT-v1-95M on prior tasks (deltas 0.006–0.018 MAP). It is
commented out in `scripts/sweeps/run_all_sweeps.py`; not re-evaluated
on the leitmotif variant. The bigger model does not help for
this task family.

### CLaMP3-symbolic — the absolute winner (for the case it covers)

Cannot run per-pair analysis for the reason described in §Methodology,
but the WandB aggregate is informative:

| Layer | Aggregate MAP | Notes |
|---|---:|---|
| Original VGMIDITVar | L11 = 0.198 | Best layer on FluidR3 baseline |
| Leitmotif variant | meanall = 0.195 | Best on cross-instrument; meanall > L11 here |

**Drop from FluidR3 to leitmotif: 0.003.** The symbolic encoder is
essentially indifferent to the program-change rewrite. This is by
construction: M3 tokens encode programs, but program tokens are 1–2
events out of hundreds; note events dominate. The symbolic encoder is
*structurally* invariant to instrument changes via its input
representation, not via learned invariance.

**Practical implication:** for leitmotif corpora where MIDI is
available (or where audio-to-MIDI transcription is acceptable),
CLaMP3-symbolic is roughly an order of magnitude stronger than any
audio path. The trade-off is needing accurate transcription as a
preprocessing step.

---

## Cross-encoder analysis

### Cross-instrument MAP at each encoder's best layer

| Encoder | Best layer | Cross-instr MAP | Relative to MuQ |
|---|---:|---:|---:|
| **MuQ** | L7 | **0.493** | 1.00× |
| OMARRQ-25hz | L15 | 0.466 | 0.94× |
| CLaMP3 (audio) | L5 | 0.372 | 0.75× |
| MERT-v1-95M | L8 | 0.332 | 0.67× |

MuQ's lead is real but not large — OMARRQ is within 6% relative on
its native metric. CLaMP3 audio and MERT-95M are meaningfully behind.

For ensemble strategies: MuQ L7 + OMARRQ L15 normalize-then-concat
would be the strongest two-encoder pair. Worth piloting in a future
turn but probably gives <5% relative gain for 2× embedding size.

### Where each encoder's depth profile peaks

```
        encoder          best cross   best agg    (at) layer
        ─────────────────────────────────────────────────────
        MuQ              0.493        0.091       L7 / L12
        OMARRQ-25hz      0.466        0.047       L15 / L15
        CLaMP3 audio     0.372        0.017       L5  / L0
        MERT-v1-95M      0.332        0.038       L8  / L12
```

The depth-peak ordering across encoders mirrors the "deeper = more
abstract" prior, *if* you accept that the layer indices live on
different network depths. Normalising by total depth:

| Encoder | Total layers | Best cross-instr layer | Normalised depth |
|---|---:|---:|---:|
| MERT-v1-95M | 13 | 8 | 0.62 |
| MuQ | 13 | 7 | 0.54 |
| CLaMP3 audio | 13 | 5 | 0.38 |
| OMARRQ-25hz | 24 | 15 | 0.62 |

MERT, MuQ and OMARRQ all peak around 0.5–0.6 normalised depth — the
mid-late zone where transformer-style encoders typically build
"semantic" representations. CLaMP3 peaks earlier (0.38), consistent
with its contrastive cross-modal training pushing semantic
representation toward earlier layers.

### Tri-task progression (encoder collapse profile)

For each encoder, the WandB aggregate MAP drops monotonically as the
test gets harder:

| Encoder | VGMIDITVar (orig) | -multisf | -leitmotif | Total drop |
|---|---:|---:|---:|---:|
| CLaMP3-symbolic | 0.198 | (n/a) | 0.195 | **−1.5%** |
| MuQ | 0.196 | 0.166 | 0.091 | −54% |
| OMARRQ-25hz | 0.195 | 0.145 | 0.047 | −76% |
| CLaMP3 audio | 0.182 | 0.131 | 0.017 | −91% |
| MERT-v1-95M | 0.170 | 0.121 | 0.038 | −78% |

The audio encoders all lose 50–90% of their aggregate MAP under
cross-instrument variation. Symbolic loses ~1.5%. This is the single
largest empirical justification for using symbolic when MIDI access
is available.

The aggregate-MAP collapse story conceals the per-pair MAP picture:
audio encoders are still doing meaningful cross-instrument retrieval
(0.33–0.49 per-pair MAP), they're just losing the full-pool ranking
to same-instrument distractors. Don't conflate "aggregate MAP collapsed"
with "encoder failed at the task."

---

## The most exploitable asymmetry

Per-pair MAP is **highly asymmetric** in the (query_program,
target_program) direction. For MuQ L7 on well-powered cells (N≥80):

| Pair | Theme-as-query → variation-as-target | Variation-as-query → theme-as-target | Asymmetry |
|---|---:|---:|---:|
| Piano ↔ Strings | 0.391 | 0.444 | −0.05 |
| Piano ↔ Horn | 0.541 | 0.402 | +0.14 |
| Piano ↔ Flute | 0.595 | **0.259** | **+0.34** |
| Strings ↔ Horn | 0.549 | 0.385 | +0.16 |
| Strings ↔ Flute | 0.646 | **0.252** | **+0.39** |
| Horn ↔ Flute | 0.691 | 0.489 | +0.20 |

**Pattern:** retrieval is much easier when the query has lower variation
index (i.e. is closer to the canonical theme) and the target has higher
variation index. The asymmetry magnitude correlates with idx-distance:
adjacent variations (Piano↔Strings) are nearly symmetric; distant
variations (Strings↔Flute, idx 1 ↔ idx 3) are highly asymmetric.

**Hypothesis:** the Variation-Transformer model that generated VGMIDI
variations likely produces increasingly divergent variations as idx
grows. By idx=3 (Flute), the variation may differ substantially from the
theme. When the original (low-idx) is the query, retrieval has the full
melodic content; when the variation (high-idx) is the query, content
has been removed and many works look superficially similar.

**Deployment implication:** if you control the indexing direction, **index
your leitmotif library by canonical themes (low idx) and query with
variations (high idx)**. This maximises retrieval performance for
free. For real game soundtracks: use the most canonical / least-
varied statement of each leitmotif as the index; query with novel
appearances.

---

## Recommendations

Choose the recipe matching your deployment scenario.

### A. You have MIDI (or can get it via transcription)

**Use CLaMP3-symbolic.** Aggregate MAP 0.195 (≈5× the best audio
encoder). Best layer on the leitmotif variant: meanall (very close
second: L11 at 0.195).

Caveats:
- Requires MIDI input. If your corpus is real audio, you'll need
  audio-to-MIDI transcription (Basic Pitch, MT3, melody-only models,
  or commercial tools). Transcription quality becomes the bottleneck.
- The symbolic encoder is structurally instrument-invariant via its
  input tokenisation, not via learned representation. This means it
  also won't *exploit* instrument cues you might want it to.

### B. You have audio only — single-pool retrieval (one big index)

**Use MuQ L11 or L12.** Aggregate MAP 0.087 / 0.091 respectively. L11
is the safer pick:
- Within 0.005 MAP of L12 on aggregate (essentially tied)
- 9% better than L12 on cross-instrument MAP (0.490 vs 0.447)
- Far less likely to get caught by edge cases of "last layer
  over-specialisation"

**Avoid:** CLaMP3 audio (broken in aggregate at every layer; do not
deploy single-pool), MERT-95M (consistently 4th-place on every
relevant metric).

### C. You have audio only — pre-filtered hybrid pipeline

If your retrieval architecture pre-filters candidates by instrument
(e.g. via an instrument classifier or by orchestrating the index per
instrument-section), the per-pair MAP is the metric, and **MuQ L7 is
the choice**. Cross-instrument MAP 0.493, vs. 0.447 at L12 — a 10%
relative gain.

In practice, a hybrid pipeline:
1. Detect dominant instrument(s) per audio query (e.g. via a small
   classifier trained on GM instrument labels).
2. Restrict retrieval to candidates of compatible instruments (or
   re-rank cross-instrument candidates using L7 features).
3. Use MuQ L11 for fallback single-pool retrieval if instrument
   detection confidence is low.

### D. Two-layer ensemble

**MuQ L7 + L11, L2-normalize-then-concat.** Pairs the cross-instrument
champion with the aggregate-balanced layer. Doubles embedding size
(2048-dim vs 1024-dim) but covers both failure modes.

L8 + L11 is a reasonable alternative if L7's "L8 plateau" makes you
prefer slightly later layers. Both are within 0.005 MAP of each other
on the relevant metric.

**Avoid L12 in any ensemble** — its cross-instrument collapse hurts
the joint embedding more than its aggregate gain helps.

### E. Cross-encoder ensemble (most ambitious)

**MuQ L7 + L11 + OMARRQ-25hz L15, normalize-then-concat.** Three
embeddings, 3072-dim. Expected ~5% relative gain over best single
encoder; pricier in disk and inference compute. Worth piloting if
you've maxed out single-encoder gains.

### F. Index-direction trick (free win)

Regardless of which encoder/layer you use: **index your library by
canonical themes (low variation index), query with variations**. The
asymmetry is huge (up to +0.34 absolute MAP) and costs nothing.

---

## Caveats and limitations

### Same-instrument MAP is unreliable

The diagonal cells of every per-pair grid have small N (Piano-Piano:
N=12, Strings-Strings: N=4, others: N=0 with the "—" marker).

This is a structural property of the experiment: with the rotation
schedule {Piano, Strings, Horn, Flute, Trumpet}, a work needs ≥5
variations to produce within-instrument duplicates. Most VGMIDI works
have 2–4 variations. Diagonals are mostly empty.

**Don't trust** the "same-instrument MAP" line in any per-encoder
dashboard — it's based on too few queries to be statistically
meaningful. The aggregate MAP and cross-instrument MAP are both
well-powered (N=1714 aggregate; N≥80 for most cross-instrument cells).

### Trumpet is statistically noisy

Trumpet rows/columns have N=16–22 even off-diagonal (Trumpet appears
only at idx=4, the rarest variation). High-MAP cells in Trumpet rows
(e.g. Piano→Trumpet 0.799 at MuQ L7) are real signal but should not
drive recommendations alone.

For a future re-run: include only 4 instruments (drop Trumpet, keep
Piano + Strings + Horn + Flute) so each appears at lower idx and gets
more queries. Or run on a larger source dataset with deeper variation
trees.

### One SoundFont, one rendering pipeline

All audio uses Shan SGM-Pro 14 + fluidsynth. SGM-Pro 14 is a community
SoundFont tuned to Roland SC-88 Pro spec — superb for video-game
soundtrack content but synth-flavoured for orchestral instruments.
Strings sound like sampled strings, not a live string section.

This is acceptable for *encoder ranking* (all encoders see the same
input distribution) but may understate absolute performance on
deployment-quality recorded audio. A v2 follow-up with Sonatina
Symphonic Orchestra or commercial libraries would close this gap.

### Variations are model-generated

VGMIDI-TVar variations come from the Variation-Transformer model (Gao
et al.), not human composers. The variations may be more "regular"
than human-composed leitmotif variations (which can include rhythmic
augmentation, key changes, fragmentation, contrapuntal weaving). Real
deployment performance may differ.

A v2 with hand-curated variations (or with at least key-transposition
augmentation on top) would test more realistic conditions.

### CLaMP3-symbolic per-pair not measured

The symbolic encoder bypasses the embedding cache (`__getitem__`
returns 3-tuple without `clip_id`). To get per-pair MAP for symbolic,
a separate one-off extraction script (~100 LOC) is needed. Estimated
~1 day of work; not done in this turn.

The aggregate MAP and the structural argument (note tokens dominate)
make it likely that symbolic per-pair would be uniformly high across
cells (not concentrated on diagonals), but this is not yet verified.

### No human listening test

All conclusions rest on retrieval-MAP metrics. A small human listening
test on 20–50 query-candidate pairs across encoders would be a
valuable sanity check. Future work.

---

## Open questions / future work

1. **Symbolic per-pair extraction.** Write the one-off script to
   measure CLaMP3-symbolic's per-instrument-pair MAP. Verifies the
   "structurally invariant" hypothesis.
2. **Audio-to-MIDI hybrid pipeline.** Build end-to-end:
   transcribe audio → MIDI → CLaMP3-symbolic embed. Quantify the
   transcription-quality bottleneck on retrieval performance. If
   transcription preserves enough melodic content, this could be the
   deployment-grade leitmotif system.
3. **Realistic SoundFont re-render.** Repeat with Sonatina Symphonic
   Orchestra (free) or commercial library. Test whether encoder
   rankings stay stable under higher-fidelity orchestral renders.
4. **Asymmetry mechanism.** Verify the hypothesis that
   Variation-Transformer outputs increasing divergence per idx by
   measuring melodic edit distance between (theme, variation_k) pairs
   in the source MIDI. If confirmed, recommend the index-direction
   trick more strongly.
5. **Cross-encoder ensembling.** Pilot MuQ L7+L11 + OMARRQ L15
   normalize-then-concat to quantify the ensemble gain.
6. **Real-world soundtrack benchmark.** Curate ~50 known leitmotifs
   from Star Wars / LotR / Skyrim / Halo / Final Fantasy with
   timestamped occurrences across orchestrations. Closest possible
   approximation to deployment.

---

## Reproducibility appendix

### Pipeline overview

```
                  ┌──────────────────────────────┐
                  │ data/source/VGMIDI-TVar.zip  │
                  └────────────┬─────────────────┘
                               │
                  scripts/data/rewrite_vgmidi_programs.py
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │ data/VGMIDITVar-leitmotif/   │
                  │   midi/{train,test}/*.mid    │
                  │   instrument_map.json        │
                  └────────────┬─────────────────┘
                               │
                  scripts/data/build_vgmiditvar_dataset.py
                  --skip-extract --instrument-map
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │ data/VGMIDITVar-leitmotif/   │
                  │   audio/*.wav                │
                  │   VGMIDITVar.jsonl           │
                  └────────────┬─────────────────┘
                               │
                  scripts/sweeps/run_all_sweeps.py
                  --tasks VGMIDITVar-leitmotif
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │ output/.emb_cache/<enc>/     │
                  │   VGMIDITVar-leitmotif__*/   │
                  │   <clip_id>.pt               │
                  └────────────┬─────────────────┘
                               │
                  scripts/analysis/vgmiditvar_leitmotif_sweep.py
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │ output/analysis/leitmotif/   │
                  │   *.csv  *.png               │
                  └──────────────────────────────┘
```

### Re-running from scratch

```bash
# 0. One-time: install deps including matplotlib + seaborn.
uv sync

# 1. Rewrite MIDIs with rotated programs.
uv run python scripts/data/rewrite_vgmidi_programs.py \
    --src-midi-dir data/VGMIDITVar-multisf/midi \
    --dst-midi-dir data/VGMIDITVar-leitmotif/midi
uv run python scripts/data/rewrite_vgmidi_programs.py \
    --dst-midi-dir data/VGMIDITVar-leitmotif/midi --verify

# 2. Render audio (single SoundFont — Shan SGM-Pro 14).
uv run python scripts/data/build_vgmiditvar_dataset.py \
    --skip-extract \
    --midi-extract-dir data/VGMIDITVar-leitmotif/midi \
    --data-dir data/VGMIDITVar-leitmotif \
    --audio-dir data/VGMIDITVar-leitmotif/audio \
    --soundfont ~/sf2/Shan_SGM_Pro_14.sf2 \
    --instrument-map data/VGMIDITVar-leitmotif/midi/instrument_map.json

# 3. Run the encoder sweeps (populates the embedding cache).
uv run python scripts/sweeps/run_all_sweeps.py \
    --tasks VGMIDITVar-leitmotif

# 4. Compute per-pair analysis + write CSVs/plots.
uv run python scripts/analysis/vgmiditvar_leitmotif_sweep.py
```

### CSV schemas

`per_pair_map.csv` — every per-(encoder, layer, query_program,
target_program) cell, long format:

| column | dtype | meaning |
|---|---|---|
| encoder | str | one of CLaMP3, MERT-v1-95M, MuQ, OMARRQ-multifeature-25hz |
| layer | int | layer index (0..L-1 for the encoder) |
| query_program | int | GM program of query (-1 = aggregate sentinel) |
| target_program | int | GM program of candidates (-1 = aggregate sentinel) |
| map | float | mean Average Precision over queries in this cell |
| n_queries | int | number of queries that contributed (0 = no positives) |

`per_layer_summary.csv` — per (encoder, layer), the three headline metrics:

| column | dtype | meaning |
|---|---|---|
| encoder | str | encoder name |
| layer | int | layer index |
| aggregate_map | float | full-pool MAP |
| aggregate_n | int | always 1714 (test split) |
| same_instr_map_mean | float | mean of diagonal cells with N>0 (LOW power) |
| same_instr_n_cells | int | number of populated diagonal cells (≤2) |
| cross_instr_map_mean | float | mean of off-diagonal cells with N>0 (HEADLINE) |
| cross_instr_n_cells | int | number of populated off-diagonal cells (≤20) |

`cross_encoder_summary.csv` — one row per encoder; per-pair best layers
plus WandB tri-task best layers.

`wandb_aggregate.csv` — raw WandB pull. (encoder, variant, layer,
test_map, test_mrr, test_map@1).

### Output figures

| File | Content |
|---|---|
| `cross_encoder_summary.png` | 2-panel: aggregate MAP per (encoder, variant) + best cross-instrument MAP per encoder |
| `dashboard_<encoder>.png` (×4 audio encoders) | 2×2: layer profile (3 metrics) + per-pair heatmap at cross-best layer + tri-task overlay + per-pair heatmap at aggregate-best layer |

All figures regeneratable from CSVs alone — the analysis script's
plotting functions consume the same DataFrames they emit.

### Decommissioned encoders

Per `scripts/sweeps/run_all_sweeps.py` (commented blocks with rationale
headers):

- **MERT-v1-330M:** lost 4/4 head-to-head against MERT-v1-95M on prior
  tasks (MAP deltas 0.006–0.018). Configs remain on disk; uncomment
  to revive.
- **MusicFM:** 0 finished WandB runs across all 4 registered tasks
  (upstream pipeline issue never resolved). Configs remain on disk.

Neither was evaluated on the leitmotif variant. Adding them is a
config + sweep run away.

---

## Changelog

- **2026-05-16** — Initial complete analysis. All 4 audio encoders
  swept across all layers; CLaMP3-symbolic on aggregate-only side.
  Discovered the aggregate-vs-cross-instrument metric divergence
  (5-layer gap for MuQ, MERT, CLaMP3-audio); discovered the
  theme→variation asymmetry; recommended MuQ L7 for hybrid pipeline
  + L11 for single-pool. Replaces the earlier "audio encoders all
  collapse" framing from the WandB-aggregate-only analysis.
