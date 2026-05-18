# MARBLE — layer analysis & leitmotif layer-selection guide

Single source of truth for everything we know about which transformer
layer to extract from each frozen encoder. Used to drive the
leitmotifs project's per-track embedding strategy and any
downstream MARBLE probe work.

Last updated: 2026-05-18 (after HookTheoryStructure full-encoder sweep —
4-encoder structural classification data lands the L10 + L11 mixed-task
ensemble recommendation; see § Supervised classification / structure
and § Two-layer ensemble).
For implementation details on caching / extracting, see
[`embedding_cache.md`](embedding_cache.md).

**For the deep-dive on cross-instrument leitmotif retrieval (the headline
deployment scenario): see [`leitmotif_findings.md`](leitmotif_findings.md).**
That doc covers the per-instrument-pair MAP analysis, the
aggregate-vs-cross-instrument metric divergence, the theme→variation
asymmetry, and per-encoder dashboards. This doc is the broader layer-
selection reference covering all tasks.

---

## TL;DR

| Encoder | L total | Default single-layer pick | Cross-instr peak | "Theme/motif" peak | "Song identity" peak | "Structure section" peak | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **MuQ** (primary) | 13 | **L11** (Pareto-optimal) | L7 (≈L11) | L8 | L11/L12 | **L10** | Default for real-audio retrieval. Best on HookTheoryStructure too (0.591 acc) |
| **OMARRQ-multifeature-25hz** (alternative) | 24 | **L15** (broad-peak winner) | L15 | L14 | L20 | **L17** | Use for multi-task deployments — broad peak (L8–L17 plateau on structure, L11–L21 on cross-instr) |
| MERT-v1-95M | 13 | L12 (agg) / L8 (cross) / L4 (struct) | L8 | L3 | L7 | L4 | Consistently 4th audio encoder; mid-block peak across tasks |
| CLaMP3 (audio path) | 13 | not for single-pool | L5 | L5 | L11 | L3 | Aggregate broken at every retrieval layer; OK for hybrid retrieval AND classification (0.568 on HookTheoryStructure, 3rd) |
| **CLaMP3-symbolic** (when MIDI available) | 13 | **meanall ≈ L11** | (≈ aggregate) | (uniform) | (uniform) | (not tested — HookTheory has no MIDI) | ~5× aggregate MAP of any audio encoder on retrieval. Structure-classification ranking TBD via SuperMario sweep |
| MERT-v1-330M | 25 | DECOMMISSIONED — lost 4/4 vs 95M | — | — | — | — | Configs commented out in run_all_sweeps.py |

- **MuQ L11 is Pareto-optimal as a single layer:** strictly dominates OMARRQ L15 on retrieval (+5% cross-instrument MAP / +85% aggregate MAP at same 1024-dim). On classification (HookTheoryStructure) MuQ L10 (0.591) wins narrowly over MuQ L11 (0.589) and OMARRQ L17 (0.589) — all three essentially tied. Within MuQ's own L7/L10/L11/L12 set, L11 is the joint-optimal single layer for retrieval; L10 is the joint-optimal single layer for classification; they're within 0.002 acc of each other.
- **OMARRQ L15 / L17 stays competitive on classification, broad peak holds.** Same "metric-agreement + robustness > absolute performance" trade-off as on retrieval. For multi-task pipelines that span retrieval and structure, OMARRQ degrades more gracefully under layer-choice drift than MuQ.
- **Two-layer pair recommendation refreshed:** for **retrieval-only** L11 alone is sufficient; for **mixed retrieval + structure** (the leitmotif-in-soundtracks deployment), **L10 + L11** is now defensible (L10 captures structure, L11 captures song identity). 2× cost but the second layer is well-motivated rather than marginal.
- **CLaMP3-symbolic dominates when symbolic input is available.** Structurally instrument-invariant via M3 token representation on retrieval. Whether it also dominates on supervised classification is open (SuperMario sweep will answer). See [`leitmotif_findings.md`](leitmotif_findings.md).

---

## Background: where each MIR task probes

Different MIR tasks measure different invariances and tap different
layers as a result. The probing literature plus our own benchmarks
suggest the following alignment (layers are normalized as a fraction
of total layers per encoder):

| Musical primitive | Typical layer depth | Probe task that measures it |
|---|---:|---|
| Pitch / timbre / spectral detail | ≤ 25% (very early) | HookTheoryMelody (frame-level pitch transcription) |
| Beat / rhythm | ~25–30% | GTZANBeatTracking |
| Chord / local harmony | ~30–50% (early-mid) | Chords1217 |
| Melody contour / motif-level | ~50–60% (mid) | VGMIDITVar (theme→variation retrieval) |
| Key / global tonality | ~50–70% (mid-late) | GS, HookTheoryKey |
| Song / theme identity | ≥ 75% (late) | Covers80, SHS100K (cover retrieval) |
| Song structure (verse/chorus/...) | latest (>80%) | HookTheoryStructure |

Our benchmark coverage so far:

- ✅ Cover retrieval (Covers80, SHS100K) — late-layer probe
- ✅ Theme/variation (VGMIDITVar) — mid-layer probe
- ⏳ Structure classification (HookTheoryStructure) — sweep configs ready, not yet run for MuQ
- ⏳ Melody pitch transcription (HookTheoryMelody) — sweep configs ready, not yet run
- ❌ Beat tracking (GTZANBeatTracking) — partial data only (OMARRQ-fsq variant)
- ❌ Chord recognition (Chords1217), Key detection (HookTheoryKey/GS) — partial coverage across encoders

The two missing tasks (HookTheoryStructure + HookTheoryMelody on MuQ) are the highest-priority follow-ups for verifying the recommendations below.

---

## Raw results — best layers per (encoder, task)

Pulled live from WandB via `scripts/analysis/best_layer.py`. See that
script for the up-to-date version.

### Audio retrieval / theme-variation invariance

| Encoder | Covers80 | SHS100K | VGMIDITVar (audio) |
|---|---|---|---|
| **MuQ** | **L12 / 0.198** (top-3: 12,11,10) | **L11 / 0.190** (11,12,10) | **L8 / 0.196** (8,7,9) |
| OMARRQ-25hz | L14 / 0.103 (14,15,16) | L14 / 0.086 (14,15,16) | L20 / 0.195 (20,15,14) |
| OMARRQ-25hz-fsq | L23 / 0.027 (23,22,21) | — | L5 / 0.074 (5,4,3) |
| MERT-v1-95M | L5 / 0.101 (5,3,4) | L7 / 0.084 (7,6,5) | L3 / 0.170 (3,4,7) |
| MERT-v1-330M | L5 / 0.085 (5,1,6) | — | — |
| CLaMP3 (audio) | L12 / 0.144 (12,11,9) | L11 / 0.146 (11,10,12) | L5 / 0.182 (5,6,4) |
| CLaMP3-symbolic (MIDI direct) | — | — | L11 / 0.198 (11,10,9) |

### Supervised classification / structure

| Encoder | GS (key, 24-cls) | HookTheoryKey | HookTheoryStructure | GTZANBeatTracking |
|---|---|---|---|---|
| CLaMP3 (audio) | L0 / 0.642 | L0 / 0.717 | L3 / 0.568 (3rd) | n/a |
| MERT-v1-95M | L9 / 0.651 | (not run) | L4 / 0.558 (4th, last) | (not run) |
| OMARRQ-25hz | L17 / 0.175 (fsq) | (not run) | **L17 / 0.589** (2nd, tied) | L3 / 0.361 (fsq) |
| **MuQ** | (not run) | (queued) | **L10 / 0.591** (1st) | n/a |
| CLaMP3-symbolic | n/a (audio task) | n/a | (no MIDI in dataset) | n/a |

**HookTheoryStructure full-encoder update (2026-05-18):** sweep
completed for all 4 audio encoders. Key findings:

| Encoder | Best layer | test/acc | Peak depth |
|---|---:|---:|---:|
| MuQ | L10 | **0.591** | 0.77 |
| OMARRQ-multifeature-25hz | L17 | 0.589 | 0.71 |
| CLaMP3 (audio) | L3 | 0.568 | 0.23 |
| MERT-v1-95M | L4 | 0.558 | 0.31 |

- **MuQ wins by a hair (+0.002 over OMARRQ)** — essentially tied at
  the top. Compare to the retrieval tasks where MuQ leads by
  0.04–0.10 MAP. Classification has a higher floor than retrieval
  (random=1/7≈0.143 for 7 classes; all encoders ≥55%) so the spread
  compresses; differences that matter on retrieval often don't on
  classification.
- **OMARRQ's broad peak holds here too** — stays ≥0.57 from L8
  through L17 (10-layer plateau, 42% of the network). MuQ's plateau
  is only L8–L11 (4 layers, 31%). For multi-task deployments where
  one cache serves both structure and retrieval, OMARRQ degrades
  more gracefully under layer-choice drift.
- **CLaMP3 (audio) at L3 is 3rd** — beats MERT-95M despite being
  consistently last on retrieval. Contrastive cross-modal training
  apparently produces strong local section-discriminating features
  even where it fails at song-level identity.
- **MERT-95M peaks shallowest** (L4, depth 0.31) — same
  early-peak / late-decline pattern we've seen on every other task.
  4th place, monotonically declines after L6.
- **MuQ L10 vs the L7-vs-L11 leitmotif story:** HookTheoryStructure
  peaks at L10, sitting between the L7 cross-instrument peak and
  the L11/L12 single-pool retrieval peak. This is consistent with
  the "L7 = motif identity, L11 = song identity" story —
  structural-function classification needs slightly less abstraction
  than song identity (L11) but more than motif identity (L7).

### Cross-encoder layer-depth pattern (normalized)

For tasks where multiple encoders have data, normalizing best-layer
by total-layers shows different internal organization across encoders:

| Task | MuQ (/ 13) | OMARRQ-25hz (/ 24) | MERT-95M (/ 13) | CLaMP3 (/ 13) |
|---|---:|---:|---:|---:|
| Covers80 (retrieval) | 0.92 | 0.58 | 0.38 | 0.92 |
| SHS100K (retrieval) | 0.85 | 0.58 | 0.54 | 0.85 |
| VGMIDITVar (retrieval) | 0.62 | 0.83 | 0.23 | 0.38 |
| -multisf (retrieval) | 0.62 | 0.83 | 0.85 | 0.38 |
| -leitmotif (retrieval, aggregate) | 0.92 | 0.62 | 0.92 | 0.00 |
| -leitmotif (per-pair cross-instr) | 0.54 (L7) | 0.62 (L15) | 0.62 (L8) | 0.38 (L5) |
| HookTheoryStructure (classification) | 0.77 | 0.71 | 0.31 | 0.23 |

Patterns that hold across the matrix:

- **MuQ and CLaMP3 prefer late layers for cover retrieval** (likely
  contrastive/semantic objectives push abstract features deep).
- **OMARRQ-25hz and MERT-v1-95M peak earlier in relative depth** on
  most tasks (masked-prediction-style objectives preserve more
  acoustic structure mid-block) — except MERT on -multisf and
  -leitmotif aggregate, which jump late as the task gets harder.
- **CLaMP3 is the outlier on cross-instrument**: peaks at depth 0.38
  on per-pair leitmotif (much shallower than the other three at
  0.54–0.62). Contrastive cross-modal pretraining (text + audio +
  MIDI) pushes "semantic" representations earlier in the network.
- **HookTheoryStructure clusters all three masked-prediction encoders
  in the mid-late zone** (MuQ 0.77, OMARRQ 0.71, MERT 0.31).
  Structure classification needs more abstraction than mid-block
  features but less than the deepest layer.

### Sparse-data row

MERT-v1-330M only has Covers80 data so far. Don't draw firm
conclusions on this encoder until the full sweep finishes (planned).

---

## Per-encoder layer-selection cheat sheet

For leitmotif extraction on real soundtracks, recommended layer picks
in priority order:

### MuQ (primary recommendation — winner across all retrieval tasks)

```
Hidden states 0–12, hidden dim 1024, token rate 25 Hz, sample rate 24 kHz
```

| Pick | Covers80 MAP | SHS100K MAP | VGMIDITVar MAP | Leitmotif cross-instr MAP | HookTheoryStructure acc | Use when |
|---|---:|---:|---:|---:|---:|---|
| **L11** | 0.193 (2nd) | **0.190** (1st) | 0.175 | 0.490 | 0.589 (2nd) | **Primary single-pool pick** — peaks on real-audio cover retrieval and near-peak on cross-instrument leitmotif. |
| **L7** | 0.148 | 0.134 | 0.196 (3rd) | **0.493 (1st)** | 0.545 | **Primary cross-instrument pick** — best layer when retrieval is restricted to per-instrument pools (hybrid pipeline). |
| L8 | 0.151 | 0.139 | **0.196 (1st)** | 0.491 | 0.582 | MIDI-rendered theme/variation peak; tied with L7/L11 on cross-instrument. |
| L12 | **0.198 (1st)** | 0.183 (3rd) | 0.164 (worst) | 0.447 | 0.571 | Covers80 + leitmotif-aggregate peak. Collapses on cross-instrument and VGMIDITVar. **Avoid** unless you specifically need single-pool retrieval and cannot use L11. |
| L10 | 0.175 (4th) | 0.174 (4th) | 0.181 | 0.478 | **0.591 (1st)** | Structure/section optimum; defensible for context-aware retrieval. |

**MuQ L11 is the default.** It is Pareto-optimal as a single layer — within 0.6% of L7's cross-instrument MAP and within 4.4% of L12's aggregate MAP, while beating each of them by ~150% / ~10% on the *other* metric. Pick L11 for any deployment where you don't have a specific reason to favour an extreme.

**When to consider L7 instead:** only if your retrieval pipeline pre-filters candidates by instrument so the cross-instrument MAP is the dominant metric. Even then, L7's edge over L11 is 0.003 absolute (0.6% relative) — not consequential for most deployments.

**When to consider L10 instead:** if your task is **structural section classification** (HookTheoryStructure / HXMSA / SuperMario) rather than retrieval. L10 wins HookTheoryStructure (0.591 vs L11's 0.589 — basically tied) and L11 is 2nd. For multi-task deployments combining structure + retrieval, L11 is still the cleaner default (gives up only 0.002 accuracy on structure but stays single-pool retrieval optimal).

**When to consider L12 instead:** only if you specifically need to maximise aggregate (single-pool) retrieval MAP and you accept the 10% cross-instrument MAP loss as a fair trade. Rare.

**Two-layer ensembles for MuQ:**

- **For retrieval-only pipelines: L11 single-layer is sufficient.** The previous L7+L11 recommendation traded 2× embedding cost for ~0.003 MAP improvement — not worthwhile.
- **For mixed retrieval + structure pipelines (the leitmotif-in-soundtracks use case): L10 + L11 is now defensible.** L10 captures structural-section identity (HookTheoryStructure: 0.591), L11 captures song identity (SHS100K: 0.190). Pair covers both regimes at 2× storage. Storage trade-off is the same as before; the difference is the second layer is now well-motivated rather than marginal.
- **For cross-instrument retrieval specifically: L7 + L11.** L7 brings cross-instrument MAP (0.493) that L11 narrowly misses (0.490 — basically tied, so the pair gain is small), L11 brings aggregate stability. Useful only when the deployment is dominated by cross-instrument hybrid retrieval.

Re-evaluate any ensemble choice case-by-case via the breakdown script's per-pair output before committing to it.

**Why this changed (2026-05-18 third pass):** the previous (2026-05-16) recommendation said "L11 only, ensembles never justified." That was correct for leitmotif retrieval but undersold the structure-task case. With HookTheoryStructure data showing L10 wins structure, L10+L11 becomes a defensible mixed-task ensemble. See [`leitmotif_findings.md`](leitmotif_findings.md) § Second-pass observations + the HookTheoryStructure note above for the full reasoning chain.

### OMARRQ-multifeature-25hz

```
Hidden states 0–23, hidden dim 1024, token rate 25 Hz, sample rate 24 kHz
```

| Pick | Covers80 | SHS100K | VGMIDITVar | Leitmotif cross-instr | Leitmotif aggregate | HookTheoryStructure |
|---|---:|---:|---:|---:|---:|---:|
| **L15** (single-layer leitmotif pick) | 0.098 | 0.083 | 0.193 | **0.466 (1st)** | **0.047 (1st)** | 0.564 |
| **L17** (structure pick) | ~0.097 | ~0.082 | 0.187 | 0.422 | 0.032 | **0.589 (1st-tied)** |
| L14 | **0.103** | **0.086** | 0.184 | 0.460 | 0.045 | 0.579 |
| L20 (theme-variation peak on prior tasks) | ~0.097 | ~0.082 | **0.195** | 0.426 | 0.033 | 0.563 |

**OMARRQ's distinguishing property #1: peak agreement.** It is the only encoder where aggregate-MAP and cross-instrument-MAP both peak at the same layer (L15) on the leitmotif task. No specificity-invariance trade-off to manage on retrieval. Same pattern on HookTheoryStructure (peak at L17 is within noise of L15).

**OMARRQ's distinguishing property #2: broad peak.** Cross-instrument MAP stays ≥0.42 across L11–L21 (11 layers / 42% of the network); MuQ's plateau is only L7–L11 (5 layers / 31%). HookTheoryStructure acc stays ≥0.57 across L8–L17 (same 10-layer plateau). For multi-task deployments where one cache serves both structure and retrieval, OMARRQ degrades more gracefully under layer-choice drift than MuQ.

**When OMARRQ L15 is preferable to MuQ L11:**
- You value metric agreement over absolute performance — clean defence of "we used the best layer."
- You want robustness to slight layer-choice errors (broad peak vs MuQ's narrow plateau).
- Operational risk hedge against MuQ checkpoint availability.
- Existing OMARRQ infrastructure makes inference cost amortise better.

**When MuQ L11 is preferable to OMARRQ L15:**
- You want the absolute best performance — MuQ L11 wins by 5% cross-instrument and 85% aggregate, same dimensionality.
- You want lower cache footprint (13 MuQ layers vs 24 OMARRQ layers).
- You want faster inference (half the depth).

**Two-layer pair (only if you need it):** L14 + L20 (mid-deep + very-deep). 6-layer spread; bimodality is more pronounced than MuQ. As with MuQ, a single layer (L15) is usually sufficient for this task.

### MERT-v1-95M

```
Hidden states 0–12, hidden dim 768, token rate 75 Hz, sample rate 24 kHz
```

| Pick | Covers80 | SHS100K | VGMIDITVar |
|---|---:|---:|---:|
| **L7** (single-layer pick — top-3 on all three) | 0.094 (2nd) | **0.084** | 0.166 (3rd) |
| L5 | **0.101** | 0.077 | 0.155 |
| L3 (theme-variation peak) | 0.087 | 0.072 | **0.170** |

**Two-layer pair:** L3 + L7 (early-mid + mid). Compact spread.

### CLaMP3 (audio path)

```
Hidden states 0–12, hidden dim 768
```

L11 — wins on cover retrieval, top-3 on VGMIDITVar via the symbolic path
(which isn't applicable to real audio anyway).

Note: CLaMP3-symbolic (feeding MIDI directly through CLaMP3's M3 sub-network) wins on VGMIDITVar at 0.198 — but that's a separate pipeline that bypasses audio rendering entirely. Not usable on real soundtracks.

---

## Combination strategies for two-layer extraction

### Option 1 — L2-normalize each, then concatenate (recommended default)

```python
e_a = F.normalize(emb_at_layer_a, dim=-1)
e_b = F.normalize(emb_at_layer_b, dim=-1)
emb = torch.cat([e_a, e_b], dim=-1)   # (T, 2H)
```

Pros: single pipeline, equalized layer contributions, drop-in
replacement for single-layer extraction. Cosine similarity on the
concat vector becomes a blended cosine.

Cons: doubles disk + DTW cost per pair. No tunable weight.

### Option 2 — Late fusion at the scoring stage

Compute similarity / matrix profile / DTW score per layer, then
combine the scalars:

```
s_combined = α · s_layer_a + (1 − α) · s_layer_b   with α ∈ [0, 1]
```

Pros: each layer's geometry is preserved through similarity
computation. Tunable α via grid search on a validation set. Robust to
one layer firing on a false positive.

Cons: 2× compute per pair (run the entire matrix profile / DTW
pipeline twice). Requires labeled validation data to tune α.

**Implementation pattern for the leitmotifs project:**

```python
# After embedding extraction:
embs = {layer: load_cached(track, layer) for layer in [8, 11]}

def fused_score(track_a, track_b, alpha=0.5):
    profiles = {}
    for L, e in embs.items():
        # Compute matrix profile per layer (independently)
        profiles[L] = matrix_profile_score(e[track_a], e[track_b])
    return alpha * profiles[8].max() + (1 - alpha) * profiles[11].max()
```

Grid search α on known positive / negative pairs:

```python
for α in [0.0, 0.1, 0.2, ..., 1.0]:
    pos_scores = [fused_score(a, b, α) for a, b in known_positives]
    neg_scores = [fused_score(a, b, α) for a, b in known_negatives]
    auc[α] = roc_auc(pos_scores, neg_scores)
best_α = argmax(auc)
```

### Option 3 — Mean averaging the embeddings

```python
emb = (emb_at_layer_a + emb_at_layer_b) / 2
```

Pros: cheapest. Same output dim as single layer.

Cons: mashes layer geometries; usually worse than concat by a few
percent. Only use if you can't afford the doubled dim.

### When NOT to use late fusion

If your validation set is too small to tune α reliably (say <50 known
positives), default to **L2-normalize-then-concat**. Late fusion's
main advantage is the tunable α; without enough data to tune it, you're
just doing more compute for no gain.

---

## Suggestions and recommendations

> **For deployment-scenario-specific recipes covering single-pool vs hybrid
> retrieval, the index-direction asymmetry trick, and per-encoder-pair
> ensemble guidance, see [`leitmotif_findings.md`](leitmotif_findings.md)
> § Recommendations.** This section covers the broader retrieval setup;
> the leitmotif doc has the deployment-quality detail.

### Default single-layer pick (covers >90% of use cases)

**MuQ at L11.** Pareto-optimal for retrieval: cross-instrument MAP 0.490, aggregate MAP 0.087, both within 5% of their respective peaks at neighbouring layers. Strictly dominates OMARRQ L15 on both metrics at the same dimensionality. For structure classification (HookTheoryStructure), L11 is also second-best (0.589 vs L10's 0.591) — within noise. So L11 is the right default for both retrieval and classification.

### When the task is structural classification specifically

**MuQ at L10.** HookTheoryStructure peak (0.591). Beats L11 by 0.002 accuracy. Only switch from L11 to L10 if structural classification is the primary metric AND you don't also need retrieval-quality embeddings — otherwise stay on L11.

### Principled-simple alternative (when metric agreement matters)

**OMARRQ-multifeature-25hz at L15.** The only encoder where aggregate-best and cross-instrument-best agree at one layer on the leitmotif task. Broader peak (more robust to layer choice) — L8–L17 stays ≥0.57 on HookTheoryStructure (10-layer plateau), L11–L21 stays ≥0.42 on cross-instrument MAP. Loses 5% cross-instrument and 46% aggregate vs MuQ L11 on retrieval; ties MuQ on classification (0.589 at L17). Defensible for multi-task deployments.

### Cross-instrument-only deployment (rare)

**MuQ L7 OR OMARRQ L15.** Both within 0.005 cross-instrument MAP of MuQ L11 — if the per-pair MAP is the only metric, the absolute best is MuQ L7 (0.493) but the gap over MuQ L11 is small.

### Two-layer ensemble (case-by-case)

- **Retrieval-only pipelines:** L11 single-layer is sufficient. The previously-recommended **MuQ L7 + L11** ensemble buys ~0.003 cross-instrument MAP at 2× embedding cost — not worth it.
- **Mixed retrieval + structure pipelines (the leitmotif-in-soundtracks use case):** **MuQ L10 + L11**, normalize-then-concat. L10 captures structural-section identity (HookTheoryStructure: 0.591), L11 captures song identity (SHS100K: 0.190). The second layer is now well-motivated by HookTheoryStructure evidence rather than marginal.

### Cross-encoder ensemble (speculative — pilot before deploying)

**MuQ L11 + OMARRQ-25hz L15, normalize-then-concat.** Combines the two strongest audio encoders. Plausible 3–5% relative gain over MuQ L11 alone; not validated empirically. Worth piloting if you've exhausted single-encoder optimisation, but don't deploy without measuring against your specific corpus.

### When MIDI is available — use symbolic instead

**CLaMP3-symbolic** at meanall (or L11; basically tied). Aggregate MAP 0.195 on the cross-instrument leitmotif benchmark — roughly 5× the best audio encoder. Trade-off: requires symbolic input. Pair with audio-to-MIDI transcription (Basic Pitch / MT3) for an end-to-end audio pipeline that's likely better than any pure-audio approach. See leitmotif doc § Recommendations recipe G.

### Free win for any recipe — exploit the index-direction asymmetry

Index your library by canonical themes (low variation index); query with variations. Up to +0.34 absolute MAP on the right pair at zero cost. Applies to all recommended encoders/layers. See leitmotif doc § The most exploitable asymmetry.

---

## Bad ideas — things to skip

### ❌ OMARRQ-multifeature-25hz-fsq

Worse than the non-fsq variant on every task we've tested (Covers80 0.027 vs 0.103; VGMIDITVar 0.074 vs 0.195). The FSQ quantization destroys representational fidelity. Use non-fsq always.

### ❌ MERT-v1-95M as the primary leitmotif encoder

Roughly half the retrieval MAP of MuQ on real audio (Covers80 0.101 vs 0.198). Smaller, faster, but on this task domain it's just not competitive. Keep it as a fallback for compute-constrained scenarios.

### ❌ Mean averaging two layers

Always strictly worse than L2-normalize-then-concat for retrieval. Same compute, lower performance. The only reason to mean-average is if you have a hard 1×H output-dim constraint downstream.

### ❌ Late fusion without α tuning

If you can't afford labeled validation data to grid-search α, just use concat. α=0.5 (the obvious default) often performs worse than concat. Late fusion's whole value proposition is the tunable α; using a fixed α throws that away.

### ❌ Single-layer extraction at a "random middle" layer

Several encoders (MERT-95M, OMARRQ-25hz) have peaks at non-obvious depths (L7, L14). Defaulting to "middle layer" without checking the per-task profile leaves 30–50% MAP on the table for retrieval tasks. Always start from the table above.

### ❌ Caching frame-level features for leitmotif

Each MuQ frame at 25 Hz × 30 sec clip is 13 × 750 × 1024 × 4 = ~40 MB/clip. Across 259 tracks of 3-min audio = ~620 MB per track × 259 = ~160 GB. Trivial bog if you accidentally cache per-frame instead of post-time-pool. Always pool to `(L, H)` before caching for retrieval; reserve frame-level for matrix-profile pipelines that need per-window similarity.

### ❌ Running fsq + non-fsq variant in the same sweep

The fsq variant exists for historical reasons (original OMARRQ release). It is uniformly worse. If you find yourself benchmarking both, drop the fsq and double-check no downstream config references it.

### ❌ MIDI-direct path for real-audio leitmotif

CLaMP3-symbolic wins VGMIDITVar (0.198) because it skips audio rendering — feeds MIDI through CLaMP3's M3 sub-network directly. **This is not transferable to real soundtracks**, which are audio, not MIDI. Don't confuse the symbolic-path benchmark numbers with audio-path applicability.

---

## Verification status & next experiments

### Completed

1. ✅ **MuQ × HookTheoryStructure** (2026-05-15). Peak at **L10 / 0.591 acc**, top-3 = {10, 11, 8}. Late-layer dominance confirmed. The L11→L12 dip mirrors the SHS100K dip — consistent with the last layer being slightly over-specialized to the pretraining objective.

### Open

2. **MuQ × HookTheoryKey** (config: `probe.MuQ-layers.HookTheoryKey.yaml`) — queued. If the late-layer hypothesis generalizes, peak should be in the L8–L11 band. Key estimation needs tonal abstraction over the full clip, so a mid-late peak (closer to L8) would also be consistent.
3. **MuQ × HookTheoryMelody** (config: `probe.MuQ-layers.HookTheoryMelody.yaml`) — peak layer should be mid (L5–8 range) if the melody hypothesis is correct. ~6–8 hours (no cache — frame-level task).
4. **Multi-soundfont VGMIDITVar re-render.** Re-render the VGMIDI-TVar dataset with 3–4 SoundFonts (FluidR3 + MuseScore_General + SGM-V2.01 + Salamander) using the existing `--soundfont` rotation in `scripts/data/build_vgmiditvar_dataset.py`. If the layer-8 VGMIDITVar peak is real timbre-invariance, it should stay at L8. If it was an artifact of single-soundfont rendering, the peak should shift toward L10–L11. ~2 hours of rendering + a re-sweep.
5. (Bonus) Cross-encoder verification: HookTheoryStructure + HookTheoryKey on MERT-95M + OMARRQ-25hz.

If the structure peak ≠ late layer, OR the melody peak ≠ mid layer,
revisit the recommendations.

After those experiments, update this doc's "TL;DR" table with the
actual layer numbers. If results contradict the analysis, mark
the contradiction explicitly here — don't quietly update the table.

### Changelog

- **2026-05-15** — Corrected primary MuQ leitmotif recommendation from L8 → L11. Previous recommendation was triangulated from VGMIDITVar alone, but the larger and closer-to-deployment Covers80/SHS100K benchmarks both peak at L11–L12. VGMIDITVar's narrower invariance profile (single-soundfont MIDI rendering) makes it a weaker proxy for real-audio leitmotif matching than the cover-retrieval tasks. Updated TL;DR table, MuQ cheat sheet, and recommendation sections accordingly. Added HookTheoryStructure result to the supervised-classification table.

---

## Implementation reference

- Embedding cache: see [`embedding_cache.md`](embedding_cache.md)
- Sweep runner: `scripts/sweeps/run_all_sweeps.py`, `run_sweep_local.py`
- Layer-stats query: `scripts/analysis/best_layer.py`
- Pre-warm CLI: `scripts/embeddings/extract.py`
- Cache inspection: `scripts/embeddings/manage.py`

For the leitmotifs project (separate repo at `/Users/sid/leitmotifs/`)
the relevant snippet is in
[`docs/TODO.md`](TODO.md#1-leitmotifs-matrix-profile-result-cache-separate-repo).
