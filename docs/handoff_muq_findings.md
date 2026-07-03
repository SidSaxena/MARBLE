# Handoff: MuQ findings for the thesis draft & defense slides

**Audience:** the agent drafting the thesis chapters and slides.
**Scope:** everything established for **MuQ (MuQ-large-msd-iter, frozen)** as of 2026-07-03.
MERT-v1-95M is being run through the identical pipeline now; OMAR-RQ follows. The structure
below is encoder-agnostic — extending each finding to MERT/OMAR-RQ is a row-append, not a rewrite.

**Thesis frame:** evaluating frozen music-foundation-model layers for **leitmotif discovery and
identification** in orchestral video-game music (Breath of the Wild). Two proxy axes:
(A) *melody/pitch extraction* (MedleyDB) and (B) *theme-identity retrieval across
orchestration* (VGMIDITVar-timbre). All probing on the MARBLE fork, branch `feat/bps-within-piece`.

---

## Finding 1 (headline, methodological): the "negative timbre gap" was a benchmark artifact

**The claim that died.** VGMIDITVar-timbre (102,960 renders = 5,040 works × ~2.55 variations ×
8 GM instruments; zero-shot cosine retrieval, centered embeddings, MAP) appeared to show MuQ's
late layers retrieving a theme in a *different* instrument more easily than in the *same*
instrument (condition_gap = within-MAP − cross-MAP = **−0.187** at L11). This was previously
reported as evidence of exceptional timbre invariance.

**The confound.** Relevance was defined by `work_id` only. Because the corpus is fully crossed,
every cross-instrument relevant set contains the query's **own variation re-rendered in the
target timbre** (a "same-composition twin"): the *same notes* in a different instrument. This is
**not** an audio duplicate — a piano→guitar twin scores cos≈0.79, and the twin's mean cosine
across all instrument pairs is ~0.35 (early layers) → ~0.63 (deep), rising with depth. But it is
still a systematically *easier* positive than a genuinely-different variation of the theme (which
scores ~0.24→0.40), and — crucially — the within-instrument set *cannot* contain it (there the
same-notes-same-timbre item is the query itself, self-excluded). So cross cells got an extra,
easier class of positive that within cells structurally cannot, biasing the comparison.

**The control.** Variation-controlled relevance: a candidate counts as relevant (and stays in
the gallery) only if it is the same work **and a different variation index** — "retrieve a
different variation of the theme", apples-to-apples for within vs cross. Zero selection bias
here: every work has ≥2 variations, so all 102,960 queries survive
(cross varctl n = 720,720 = 102,960×7 exactly).

**Verdict (full 13-layer sweep):** the gap **flips positive at every layer**. No depth shows a
genuine cross-easier effect.

| layer | map_centered | map_whitened | within | cross (conf) | **gap (conf)** | cross (ctl) | **gap (ctl)** | twin inflation | eff_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | .026 | .095 | .270 | .144 | **+.126** | .064 | **+.206** | .079 | 30 |
| 1 | .032 | .125 | .279 | .267 | +.012 | .118 | +.161 | .149 | 44 |
| 2 | .033 | .146 | .291 | .315 | −.025 | .142 | +.148 | .173 | 56 |
| 3 | .036 | .165 | .296 | .328 | −.032 | .150 | +.146 | .178 | 62 |
| 4 | .042 | .209 | .300 | .347 | −.047 | .160 | +.140 | .187 | 62 |
| 5 | .047 | .252 | .300 | .353 | −.052 | .161 | +.139 | .192 | 61 |
| 6 | .053 | .294 | .303 | .371 | −.067 | .169 | +.134 | .201 | 62 |
| 7 | .077 | .318 | .323 | .439 | −.116 | .223 | +.101 | .216 | 68 |
| **8** | .089 | .350 | .324 | .454 | −.130 | **.226** | +.098 | .228 | 71 |
| 9 | .103 | .371 | .317 | .461 | −.144 | .219 | +.098 | .242 | 74 |
| 10 | .136 | .372 | .302 | .473 | −.171 | .218 | +.084 | .255 | 81 |
| 11 | .178 | .382 | .290 | .477 | **−.187** | .217 | **+.072** | .259 | 79 |
| 12 | .185 | **.384** | .269 | .443 | −.174 | .185 | +.084 | .257 | 69 |
| meanall | .040 | .272 | .291 | .317 | −.025 | .140 | +.151 | .176 | 50 |

(`twin inflation` = cross_conf − cross_ctl, the MAP injected by the twin alone.
`within` is identical confounded/controlled — the diagonal has no twins, an internal
consistency check.)

**The subtle second-order point (great for defense):** twin inflation grows monotonically with
depth (0.079 → 0.259). The artifact got *worse* with depth precisely because deeper layers ARE
more timbre-invariant — the twin (same notes, different instrument) becomes a stronger match as
depth abstracts away timbre — but that is a different, easier capability than the intended one
(retrieve a *different* arrangement across timbre). The confounded trend was interpretable, just
about the wrong thing.

**Finding 1b (from the score-distribution dump — the raw geometry behind the ranking, novel-feeling
and defense-worthy).** The opt-in score dump records, per (query,target)-instrument cell, the full
cosine distribution split into distractor / honest-relevant (different variation) / twin (same
composition). It reveals a clean **timbre → composition representational shift with depth**
(fig `figS3_timbre_composition_shift`):

| MuQ layer | same-timbre, diff-variation (within +) | same-composition, cross-timbre (twin) | diff-variation, cross-timbre (honest) | distractor |
|---|---|---|---|---|
| 0 | **0.81** | 0.35 | 0.24 | −0.05 |
| 6 | 0.75 | 0.44 | 0.29 | −0.04 |
| 11 | 0.69 | 0.62 | 0.40 | −0.02 |
| 12 | 0.66 | **0.63** | 0.39 | −0.02 |

Early layers are **timbre-dominated** (same-instrument similarity 0.81, same-notes-across-instrument
only 0.35). With depth, same-timbre similarity *falls* while same-notes-across-timbre *rises* — they
**cross around layer ~12**, i.e. deep MuQ judges two clips more by their notes than their
instrument. This is exactly why the twin confound intensifies with depth, and it is an independent,
unsupervised characterization of *where and how* timbre invariance emerges — a result the MAP curve
alone cannot show. (The distractor floor also drifts up with depth, −0.05→−0.02: the mild
cone-anisotropy that motivates the centered/whitened variants.)

**Independent corroboration in the score geometry** (no ranking involved): off-diagonal cells
lose exactly 12,870 twin relevants each (n_rel 36,906→24,036); the cross relevant-mean drops
0.598→0.503; diagonal cells are bit-identical under control. Aggregate
`score_sep_cross` 0.501→0.424, `score_sep_within` 0.530 unchanged.

**How to phrase in the thesis:** "same-composition twin" confound; the control is
"variation-controlled relevance" / `condition_gap_varctl`. Related in spirit to cover-song
benchmark near-duplicate leakage; we have not claimed novelty for the control itself
(standard confound-removal), the contribution is *identifying* the artifact in the fully
crossed timbre-rendering design and quantifying it per layer.

## Finding 2: the honest cross-orchestration capability lives in mid-late layers (peak L8)

The variation-controlled cross-timbre MAP ("retrieve a different variation of the theme in a
different instrument" — the leitmotif-operational metric) rises 3.5× from layer 0 (0.064) to a
plateau at **L7–L11 (~0.22, peak L8 = 0.226)**, dipping at L12. Meanwhile within-timbre MAP is
nearly flat (~0.27–0.32) across all depth. **Depth buys cross-timbre generalization
specifically** — it never makes cross easier than within, but it closes the gap from 0.206 to
0.072.

## Finding 3: MedleyDB melody — pitch is an early-layer feature (5-fold, supervised probe)

Frame-level predominant-melody probe (128-MIDI classes, RPA/RCA, artist-conditional 5-fold CV,
108 tracks). Layer curve (fold0, cached, full 13 layers): **peak at layer 1** (RPA .638/RCA
.717), monotonic decline to a trough at L8–9 (~.51), mild uptick L10–12. Five-fold results:

| config | RPA (mean±sd over 5 folds) | RCA |
|---|---|---|
| layer 11 | .557 ± .040 | .648 ± .034 |
| **meanall** | **.635 ± .044** | **.706 ± .033** |
| layer 1 (fold0 only) | .638 | .717 |

Layer 1 beats meanall on every fold in paired comparison (~+0.02 RPA; meanall beats L11 by
~+0.078 on every fold). RCA−RPA ≈ 0.09 → consistent octave confusions (typical for frozen SSL).

## Finding 4: the depth dissociation (the slide story)

**Pitch peaks at layer 1; theme identity across orchestration peaks at layer 8.** Two probes,
opposite layer curves, one hierarchy: low-level acoustic pitch is encoded early and washed out
with depth; abstract theme identity robust to orchestration is assembled mid-late. This is the
cleanest one-slide narrative of the whole MuQ analysis and directly motivates layer choice per
downstream use (melody transcription vs leitmotif retrieval need different layers — and neither
is the final layer or meanall).

## Finding 5: meanall — good for supervised probes, bad for zero-shot retrieval

meanall **wins** the supervised MedleyDB melody probe (a trained head selects from concatenated
depth) but **loses badly** in zero-shot retrieval (map_centered 0.040 vs L11's 0.178;
honest cross-timbre 0.140 vs L8's 0.226) — unweighted layer-averaging dilutes cosine geometry.
Scope any "meanall is good/bad" claim by evaluation regime.

## Finding 6: transductive whitening lifts retrieval 2–4× at every layer

ZCA whitening (α=1.0, fit on the test corpus — transductive, same protocol as centering) raises
MAP to 0.384 at L12 (from 0.185 centered). Known technique (BERT-whitening — Su et al. 2021,
All-But-The-Top, Spectral Tempering); the transductive caveat is **load-bearing** and must be
stated wherever these numbers appear. See `docs/whitening_ablation.md` for the ablation grid
and prior-art positioning.

## Finding 7: effective_rank predicts the best retrieval layer without labels

Pearson r = **0.947** between `anisotropy/effective_rank` (entropy of the centered singular-value
spectrum — fully unsupervised) and the honest cross-timbre MAP across MuQ's 13 layers.
Replicates the observation made on CLaMP3 (layer 4 peak). Practical recipe: compute
effective_rank per layer on unlabeled embeddings → shortlist retrieval layers. Verify it holds
on MERT/OMAR-RQ before claiming generality.

---

## Method & validation notes (for the methods chapter)

- **Retrieval scoring:** cosine-similarity *ranking* evaluated by MAP. No threshold, no trained
  classifier; "same" is the ground-truth work_id — the model only supplies geometric proximity.
- **Grid metrics** are computed on **centered** (mean-subtracted, re-normalized) embeddings; the
  headline `test/map` is raw; `test/map_whitened` is the whitened variant.
- **Variation control** masks same-(work, variation) candidates from **both** gallery and
  relevance (per-cell, batched, and streaming implementations agree; unit-pinned).
- **Trust chain** (say this at the defense): 146 unit tests incl. hand-computed fixtures →
  full-scale oracle: the fast pipeline reproduces the audited live L11 run to |Δ| ≤ 4e-4 on
  every metric → the confounded gaps independently match the May-28 live sweep to ≤ 0.0014 on
  all 13 layers.
- **Infrastructure** (one methods paragraph + reproducibility): metric aggregation was moved
  from CPU to GPU and all passes fused (`marble/utils/retrieval_fused.py`,
  `scripts/analysis/vgm_timbre_sweep_from_cache.py`); a 13-layer full-suite sweep takes ~46 min
  instead of ~13 h on a single RTX 5060 Ti. Embedding cache stores the full (L, H) layer stack
  per clip, so one extraction serves all layers.

## Assets

Figures (PNG for slides, PDF vector for the paper), all in
`docs/figures/vgmiditvar_timbre_muq_varctl/`:

| file | content | suggested use |
|---|---|---|
| `fig1_gap_confound_vs_controlled` | confounded vs controlled gap by layer, zero line | **the** confound slide/figure |
| `fig2_within_vs_cross_panels` | within/cross MAP, 2 panels (conf vs ctl) | paper companion to fig1 |
| `fig3_cross_timbre_varctl_by_layer` | honest cross-timbre curve, best layer + meanall ref | layer-selection figure |
| `fig4_geometry_treatments` | raw/centered/whitened MAP by layer | whitening finding |
| `fig5_grids_best_layer` | 8×8 instrument grids at L8, conf vs ctl, shared scale | defense visual (diagonal flips) |
| `fig6_score_separation` | cross score-separation with/without twin | score-geometry corroboration |
| `figS1_score_distributions_crosstimbre` | cosine distributions (distractor/honest/twin), L11 | **the confound at the score level** (Finding 1b) |
| `figS2_within_vs_cross_distributions` | relevant-score distributions, within vs cross | twin is cross-only |
| `figS3_timbre_composition_shift` | the four pool means by layer, timbre⇄composition crossover | **the depth-shift finding** (Finding 1b) |
| `summary_table.csv` | all numbers in Finding 1's table + score seps | source of truth for tables |

Also relevant: `docs/figures/vgmiditvar_crossinstrument_treatments.png` (earlier 3-encoder
treatment comparison — **pre-control numbers; do not cite its gap values without the confound
caveat**).

## Docs & experiment tracking

- `docs/vgmiditvar_timbre_3encoder_analysis.md` — confound mechanics + 3-encoder background.
  ⚠️ Its "control pending" sections predate the sweep; this handoff supersedes the numbers.
- `docs/medleydb_leitmotif_eval_strategy.md` — task-battery design (R1–R9 requirements, T1–T8),
  audit verdicts (remix cut, instrument demoted).
- `docs/whitening_ablation.md` — whitening prior art + ablations.
- `docs/medleydb_melody_sweep_plan.md` — MedleyDB melody task design.
- **wandb** (project `marble`, entity sidsaxena-universitat-pompeu-fabra):
  - VGMIDITVar-timbre MuQ fast sweep: group `MuQ / VGMIDITVar-timbre`, runs `layer-{0..12}-test`
    + `layer-meanall-test`, tags `from-cache, varctl` (2026-07-03). Audited live L11 anchor:
    run `di81drwb` (`MuQ-L11-timbre-audit`).
  - MedleyDB melody: group `MuQ / MedleyDBMelody`, runs `layer-11-{fit,test}-fold{0..4}`,
    `meanall-fold{0..4}-{fit,test}`, and the fold0 layer curve `layer-{N}-{fit,test}-fold0cache`.

## Caveats the thesis MUST carry

1. **Single encoder so far** for the full controlled curve (MuQ). The earlier cross-encoder
   "invariance ordering" (MERT ≈ 0, CLaMP3 positive, MuQ negative) is **retracted** until
   MERT/CLaMP3 are re-run under variation control — the twin was handed to every encoder.
   (MERT is running now; will be a row-append here.)
2. **Whitening/centering are transductive** (fit on the test corpus). State it everywhere.
3. VGMIDITVar-timbre is **synthetic-render** (program-byte rewrite, one SoundFont family);
   ecological validity for real orchestration is an inference, not a measurement — the BotW
   downstream task is the real test.
4. MedleyDB fold variance is large (RPA fold spread ~0.09); single-fold comparisons within
   ~0.05 are noise — always report the 5-fold mean±sd.
5. meanall claims are regime-scoped (Finding 5).
