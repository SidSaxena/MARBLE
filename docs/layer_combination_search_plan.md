# Layer-combination search for unsupervised retrieval (VGMIDITVar + deployed Zelda pipeline)

*2026-07-07. Companion to [leitmotif_layer_transfer_todos.md](leitmotif_layer_transfer_todos.md)
(the L2-vs-L11 numbers and standing hypothesis live there). This doc is the
method plan: how to find the best layer combination for zero-shot theme
retrieval, honestly, on both the labeled benchmark and the unlabeled
deployment corpus.*

## Problem statement

The deployed leitmotif-discovery pipeline uses MuQ **L11** (single layer,
mean-centered, cosine ranking). The melody campaigns show melody-tracking
information peaks much earlier (L1–L2), and within-instrument retrieval MAP
at L2 equals L11 — so multi-layer combinations *might* add recall of
same-orchestration restatements. Two settings, different constraints:

| Setting | Labels? | Selection method |
|---|---|---|
| VGMIDITVar-timbre | yes (work/variation/instrument) | direct search — but overfittable |
| Zelda corpus (deployed) | no | transfer + on-domain validation assets |

## Key simplification: search in score space

With per-layer L2-normalized embeddings, **cosine-on-concatenation with
per-layer weights √w ≡ weighted sum of per-layer cosine scores**. So
concatenation and score-level fusion collapse into one search space:
*weights w over the 13 per-layer similarity scores*. (Embedding-level mixing
before normalization is a genuinely different operator — it entangles layers
and re-raises the mix-vs-center ordering question; treat it as a separate,
secondary arm.)

Consequence — evaluation is nearly free:

1. **Score cache** (once): per layer, similarity rows for ~4–5k stratified
   queries × 103k candidates, fp16 ≈ 1.6 GB/layer → ~21 GB on disk. ~30 min
   GPU total, reads the existing clip-embedding cache.
2. Any candidate (subset, weight vector, RRF, transferred gates) then
   evaluates in **seconds**: weighted sum of cached score blocks → MAP.
3. Per-layer post-processing (centering / documented-transductive whitening)
   stays per-layer and validated — fusion happens after it.

## Overfitting control (non-negotiable)

Combination search on the benchmark = model selection on the test set.
Protocol: **split by work** (~60 % of the 5,040 works for tuning, the rest
held out), report holdout MAP only, and report the tune→holdout shrinkage
explicitly. "No combination robustly beats L11/L8 on holdout" is a valid,
defensible outcome. Repeat top candidates over query subsamples to estimate
selection noise.

## Phase 1 — benchmark search (½ day infra, minutes per search)

In order of increasing freedom:

1. **Fixed candidate points** (hypothesis tests, no search):
   every single layer; uniform-13; RRF over {L2, L8, L11}; **HTM melody
   gates frozen** (0.604·L2 + 0.229·L1 + 0.168·L12 — the crossed-axis
   transfer; prediction: worse than L11 → dissociation confirmation);
   MedleyDB gates likewise.
2. **Greedy forward selection**: start at best single layer; add the layer
   that most improves tuned-split MAP; stop below noise. ≤ 91 evaluations.
   The interpretable output is *which layers enter and when* — directly
   answers "does adding L2 (melody) to L8/L11 (invariance) help".
3. **Continuous weights**: Dirichlet random search (~2k draws) + Nelder-Mead
   polish around the greedy solution. If this barely beats greedy-uniform:
   "subsets matter, precise weights don't" — also a result.

Deliverables: holdout MAP table (combo vs L11 vs L8), selected layer set,
shrinkage, per-instrument-pair breakdown (does any gain concentrate on the
same-timbre diagonal, i.e. melody layers recalling within-orchestration
restatements?).

## Phase 2 — make the Zelda corpus evaluable (½ day curation, no compute)

Rendered-GM-MIDI benchmark ≠ orchestral recordings; the deployment decision
needs on-domain ground truth. Two cheap sources:

- **Motif annotations**: 40–80 verified theme-restatement pairs (from the
  leitmotifs project / domain knowledge) → recall@K on the real corpus.
  Highest-leverage asset in the whole plan.
- **Cross-version pairs**: OST vs concert/anniversary recordings of the same
  pieces (e.g. Symphony of the Goddesses) — natural orchestration variation
  with certain identity; annotation effort = matching track lists.

## Phase 3 — deployment A/B with pooled labeling (~1 day, mostly listening)

3–4 surviving configs (L11 baseline, L8, benchmark-best fusion,
union-of-nominations). Run each on the Zelda corpus, **pool their top-K
nominations, label the pool blind** (standard IR pooling — one pass, no
per-config bias), score precision / recall-in-pool per config. This number
decides what ships. Union-of-nominations is judged on discovery's own
objective — **recall at reviewable precision** — where a config that loses
on MAP can still win on "novel true restatements surfaced per review-hour".

## Phase 4 (optional, thesis garnish) — unsupervised selection criteria

Once Phase 1's cache exists, test whether label-free diagnostics would have
picked the winner: **eff_rank** (tracked the retrieval story before),
**hubness** (k-occurrence skew, known retrieval degrader, computable
unlabeled), **top-K stability** under query/clip subsampling. Rank-correlate
each with holdout MAP across candidates. Either outcome is content: a
label-free layer-selection recipe, or the documented caveat that none works.

## Predictions on record (2026-07-07)

- Greedy picks 2–3 plateau layers (L8–L11) first; early layers enter late or
  never for cross-timbre MAP but help the within-timbre diagonal and
  union-recall (per L2-within 0.291 ≈ L11-within 0.290).
- Frozen melody-gate transfer underperforms L11.
- Modest fused gain over L8 alone (+0.01–0.02 MAP); larger gain in
  union-recall for discovery.

## Status / dependencies

- Blocked on GPU only for the ~30-min score cache (queue: top-k matrix, then
  OMAR fit until ~16:00 on 2026-07-07).
- Reuses: VGMIDITVar clip-embedding cache; `LayerSoftmaxSum`
  frozen-gates mechanism (41cd456) for the embedding-mix arm; committed gate
  CSVs (`docs/figures/{hooktheory,medleydb}_melody_multihead/`).
- TODO cross-refs: items 1–5 in
  [leitmotif_layer_transfer_todos.md](leitmotif_layer_transfer_todos.md) are
  subsumed by Phases 1 and 3 of this plan.
