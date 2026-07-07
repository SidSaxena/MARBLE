# Layer choice for leitmotif discovery — open questions & queued experiments

> Consolidated, prioritized, effort-estimated version of ALL open TODOs
> (this doc's included): [post_defense_master_todos.md](post_defense_master_todos.md).

*2026-07-07. Context: the deployed leitmotif-discovery pipeline uses MuQ **L11**,
selected on CSI-style / VGMIDITVar-timbre retrieval. The melody campaigns
(MedleyDB 5-fold, HookTheory) show L11 is NOT a top melody-tracking layer —
this doc records what that does and does not imply, and the experiments that
settle the remaining questions. All source numbers are in committed CSVs.*

## The two-axis numbers (MuQ)

Melody tracking (probe test RPA):

| corpus | L2 | L11 | Δ (L2−L11) |
|---|---|---|---|
| MedleyDB (instrumental; closest domain to VGM) | 0.645 | 0.558 | **+0.088 (+15.7 %)** |
| HookTheory (pop) | 0.473 | 0.463 | +0.011 (+2.3 %) |

Theme-identity retrieval (VGMIDITVar-timbre, the selection axis;
`docs/figures/vgmiditvar_timbre_muq_varctl/summary_table.csv`):

| metric | L2 | L11 | L8 (argmax) |
|---|---|---|---|
| cross-instrument MAP (varctl) | 0.143 | **0.217** | 0.226 |
| within-instrument MAP | 0.291 | 0.290 | 0.324 |
| condition gap (timbre sensitivity) | 0.148 | **0.073** | 0.098 |
| separability (varctl) | 0.296 | **0.424** | 0.391 |

## Standing hypothesis (thesis/defense framing)

**Refined (a): leitmotif matching at L11 does not rely on a linearly-readable
melody representation.** If melodic surface drove discovery, L2 (best melody
layer) would retrieve best — it loses 34 % cross-instrument MAP and doubles
the timbre sensitivity. L11 encodes an orchestration-invariant *theme code*
in which melody is entangled (with harmony/rhythm), not surface-accessible.
The melody↔invariance depth dissociation therefore **validates** the L11
choice; it is a finding, not an oversight.

Caveats to state: (i) RPA measures probe *readability*, not information
content — L11's melody info is degraded (−16 % MedleyDB), not gone (−2 %
HTM); (ii) within the invariance axis the benchmark argmax is L8, with L11
best on separability/gap → "wrong layer" no, "one notch inside the plateau"
possibly (see TODO 4).

Key wrinkle motivating the union idea: **within-instrument MAP at L2 equals
L11** (0.291 vs 0.290) — L11's edge is entirely cross-timbre. Equal MAP need
not mean the same retrieved items, so L2 may nominate *different* true
matches (same-orchestration restatements ranked by melodic surface).

## TODOs — retrieval-side (all on cached VGMIDITVar embeddings; minutes each, no training)

> **Structured plan:** items 1–5 are subsumed by the phased method plan in
> [layer_combination_search_plan.md](layer_combination_search_plan.md)
> (score-space search + work-disjoint holdout + Zelda validation assets +
> pooled-labeling A/B). Execute via that doc.

1. **Nomination-overlap analysis** — top-K lists from L2/L8/L11: Jaccard
   overlap + *oracle union recall@K* (relevant items L2 finds that L11
   misses). Upper-bounds what any multi-layer scheme can add, before
   building one. Break down by (query,target) instrument pair (does L2's
   gain concentrate on the same-instrument diagonal?).
2. **Rank fusion** — RRF / score-sum over {L2, L8, L11} vs single-L11 MAP.
   The practical union, operationalized.
3. **Frozen melody-gate mix → zero-shot retrieval** — apply the HTM-learned
   gates (0.604·L2 + 0.229·L1 + 0.168·L12, `LayerSoftmaxSum(learnable=False)`
   mechanism, commit 41cd456) as a fixed mix and score retrieval.
   *Prediction: worse than L11* — the crossed-axis transfer test; a clean
   falsifiable confirmation of the dissociation (matched-axis transfer would
   need a melody-like retrieval target). Unexpectedly good ⇒ bigger result.
4. **L8 vs L11 deployment A/B** — benchmark argmax is L8; L11 wins
   separability + condition gap. One discovery-pipeline run on real
   soundtracks comparing nomination sets.
5. **Deployed-pipeline nomination diversity** — L2 vs L11 nominations on the
   real (unlabelled) VGM corpus: overlap statistics tell whether a union
   adds candidates at all in production.

## TODOs — melody-side follow-ups (from the campaign discussions)

6. **NSynth monophonic pitch probe** (per encoder) — discriminates "objective
   buries spectral surface" vs "L9 melody peak is about predominance" for
   MERT; isolates pitch-readout from voice-selection.
7. **Linear-vs-MLP probe at MuQ L1/L2** (Zaiem sensitivity check) — is pitch
   surface-linear early? Cheap on the MedleyDB cache.
8. **Top-k 5-fold extension** — if a fold-0 top-k arm from the running matrix
   (tmux `topk`) beats/ties full-13 or best-single, extend that arm to 5
   folds (~75 min/arm) for thesis-grade error bars.
9. **HTM frozen-mix caching infra** (only if HTM-side top-k ever matters):
   post-transform cache flag whose key includes transform config (~40 lines
   + validation; the current cache is pre-transform, key excludes
   emb_transforms — verified in `compute_config_hash`). Enables HTM top-k at
   ~3 h instead of ~11 h.
10. **eff_rank-for-supervised analogue** — does unsupervised effective rank
    predict the best *probing* layer (as it tracked the retrieval story)?
    Computable offline from cached embeddings.

## In flight (not TODOs)

- Top-k 5-arm × {MuQ, MERT} MedleyDB fold-0 matrix — tmux `topk`.
- OMAR-RQ HookTheory fit → retest (`-fix`) → 3-encoder HTM figures + thesis
  appendix section (`hooktheory_melody_report.py` renders whatever encoders
  the CSVs contain).
