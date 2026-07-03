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
only 0.35). With depth, same-timbre similarity *falls* while same-notes-across-timbre *rises* — the
gap between them **closes from +0.45 (L0) to +0.03 (L12): they converge at the deepest layers**
(they do NOT strictly cross within 13 layers — say "converge", not "cross"), i.e. deep MuQ judges
two clips almost as much by their notes as their instrument. This is exactly why the twin confound
intensifies with depth, and it is an independent,
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

1. **The earlier cross-encoder "invariance ordering" (MERT ≈ 0, CLaMP3 positive, MuQ negative) is
   retracted.** It was a twin artifact: the confounded gap was handed to every encoder. Under
   variation control **both encoders now show a positive gap at every layer** — within-timbre
   always beats cross-timbre. MERT-v1-95M has been run end-to-end (see **Appendix B**); the honest
   verdict is the *opposite* of the retracted one — **MuQ is the more timbre-invariant encoder**
   (honest cross-orchestration MAP peaks 0.226 vs MERT 0.113; MuQ's within-vs-twin cosine gap
   converges to +0.03 while MERT's stays wide at +0.18). CLaMP3 still pending under control.
2. **Whitening/centering are transductive** (fit on the test corpus). State it everywhere.
3. VGMIDITVar-timbre is **synthetic-render** (program-byte rewrite, one SoundFont family);
   ecological validity for real orchestration is an inference, not a measurement — the BotW
   downstream task is the real test.
4. MedleyDB fold variance is large (RPA fold spread ~0.09); single-fold comparisons within
   ~0.05 are noise — always report the 5-fold mean±sd.
5. meanall claims are regime-scoped (Finding 5).

---

# APPENDIX A — Instructions for the thesis-drafting & slides agent

**You (the drafting agent) should treat this document as the single source of truth for the
MuQ results.** Do not recompute or re-derive any number — cite what is written here. Every claim
below ships with a caveat; the caveats are load-bearing and must travel with the claim into both
the thesis text and the slides. Figures live in `docs/figures/vgmiditvar_timbre_muq_varctl/`
(PNG for slides, PDF vector for LaTeX `\includegraphics`). MERT-v1-95M and OMAR-RQ are pending and
will be appended as parallel sections; write the MuQ sections so an encoder-comparison row/panel
can be slotted in later without restructuring.

## A.1 Finding → thesis-section → claim → figure → caveat (the map)

| # | Thesis home | One-sentence claim to make | Figure(s) | Non-negotiable caveat |
|---|---|---|---|---|
| 1 | Evaluation/Methodology (confound) | A fully-crossed timbre-rendering benchmark silently rewards retrieving the *same* composition re-orchestrated; correcting it flips the apparent "cross-timbre is easier" result. | fig1, fig2, fig5 | Numbers are *confounded* unless labelled `_varctl`. |
| 1b | Interpretability / representation analysis | With depth, MuQ's embedding shifts from timbre-dominated to composition-dominated — same-instrument similarity falls while same-notes-across-instruments rises, converging at the deepest layers. | figS3 (star), figS1, figS2 | Single encoder so far; synthetic renders. |
| 2 | Results (leitmotif proxy) | Honest cross-orchestration retrieval improves ~3.5× with depth and plateaus at layers 7–11 (best L8, MAP 0.226); depth never makes cross *easier* than within. | fig3 | varctl (twin-masked) numbers only. |
| 3 | Results (melody proxy) | Predominant-melody probing peaks at **layer 1** and declines with depth — pitch is an early, low-level feature. | (MedleyDB curve — regenerate if needed) | 5-fold mean±sd; fold variance ~0.09. |
| 4 | Discussion (the through-line) | Pitch peaks early, orchestration-invariant theme identity peaks deep — two probes, opposite layer curves, one representational hierarchy. | fig3 + figS3 + MedleyDB curve | Both above caveats. |
| 5 | Discussion (aggregation) | Mean-of-all-layers helps a *trained* probe but hurts *zero-shot* retrieval; scope any meanall claim by regime. | (table) | Regime-scoped. |
| 6 | Method / appendix | Transductive ZCA whitening lifts retrieval MAP 2–4× at every layer. | fig4 | **Transductive** (fit on test corpus) — state every time. |
| 7 | Method / appendix | `effective_rank` (unsupervised) predicts the best retrieval layer (r=0.947 on MuQ). | (scatter — optional) | MuQ-only until replicated. |

## A.2 Plain-word explanations to reuse verbatim (jury-level, no ML jargon)

**Retrieval + MAP.** "We test the frozen model as a *search engine*: hand it a short clip of a
game-music theme and it ranks every other clip by how similar its internal fingerprint is.
Success means other renditions of the *same theme* rise to the top, even across instruments and
variations. MAP is the single grade for how well the right answers cluster near the top."

**The confound (Finding 1).** "In this benchmark every theme is rendered in eight instruments.
When we ask 'find this piano theme among the guitar clips,' the correct-answer set includes the
*exact same piece*, just played on guitar — the same notes. The model matches that easily. But
the *same-instrument* version of that test can't contain such a freebie (there, the identical clip
*is* the query, which we exclude). So the cross-instrument test was quietly easier by
construction, which made the model look like it found themes *more* easily across instruments than
within one. Once we require the retrieved clip to be a genuinely *different* variation, that
illusion disappears and the expected order returns."

**The depth shift (Finding 1b — the memorable one).** "We looked inside the network layer by
layer. Early on, it mostly hears the *instrument*: two different melodies on the same instrument
look more alike to it than the same melody on two instruments. As you go deeper, this reverses —
it increasingly hears the *tune* regardless of instrument. By the deepest layers the two are
nearly equal. In short: shallow layers hear the instrument, deep layers hear the melody. That is
exactly why deep layers are the right place to look for a recurring theme that gets re-orchestrated
— which is what a leitmotif is."

## A.3 The numbers (final, cite directly)

- **Confounded vs controlled timbre gap (within-MAP − cross-MAP), by layer:** see the big table in
  Finding 1. Headline: L11 confounded **−0.187** → controlled **+0.072**; *every* layer's
  controlled gap is positive (min +0.072 at L11, max +0.206 at L0). meanall: −0.025 → +0.151.
- **Honest cross-timbre retrieval (varctl):** rises 0.064 (L0) → plateau ~0.22 at L7–11, **best
  L8 = 0.226**; within-timbre flat ~0.27–0.32.
- **Score-geometry means (cross-timbre cells), the figS3 data:** same-timbre-diff-variation
  0.81→0.66; same-composition-cross-timbre (twin) 0.35→0.63; different-variation-cross-timbre
  (honest) 0.24→0.40; distractor −0.05→−0.02. Same-work gap (within − twin) closes from **+0.45
  (L0) to +0.03 (L12)** — converge, **do not strictly cross**.
- **Whitening:** map_whitened up to 0.384 (L12), 2–4× the centered MAP at every layer.
- **Reproducibility anchors:** fast pipeline reproduces the audited live L11 run to ≤4e-4; all 13
  confounded gaps match the independent May-28 live sweep to ≤0.0014.

## A.4 Slide-deck guidance

- **Slide "the benchmark had a trap":** fig1 (gap flips) as the hook + fig5 (8×8 grids: the
  diagonal goes from lightest to darkest under control) as the visual proof. One line: *"cross-
  instrument only looked easier because the correct answers included the same piece re-orchestrated."*
- **Slide "what the model actually encodes" (the money slide):** figS3. One line: *"shallow layers
  hear the instrument; deep layers hear the tune."* This is the most memorable, least-technical
  slide of the whole MuQ story — lead the interpretability section with it.
- **Optional backup slide:** figS1/figS2 for anyone who asks "how do you know it's real timbre
  invariance, not duplicate-matching?" — the score distributions answer it directly (graded
  same-notes-across-timbre at 0.6–0.8, genuinely-different variations at ~0.4, distractors ~0).
- **Slide "pitch vs identity live at different depths":** fig3 (deep peak) beside the MedleyDB
  melody curve (layer-1 peak). One line: *"opposite layer preferences → one hierarchy."*

## A.5 Precision rules — DO NOT overclaim (these will be probed at a defense)

1. Say **"converge,"** never "cross" — the same-timbre and same-composition lines do not strictly
   cross within MuQ's 13 layers (0.66 vs 0.63 at L12).
2. The twin is a **same-composition re-orchestration** (mean cos ~0.35→0.63, up to ~0.79 for
   harmonically-similar instrument pairs) — **not** a "cos≈1 audio duplicate." Never call it a
   duplicate.
3. The controlled gap is **positive at every layer** — do not say "meanall/MuQ retrieves cross-
   timbre better than within." It does not, once the twin is masked.
4. **meanall** is good for the *supervised* MedleyDB probe, bad for *zero-shot* retrieval — always
   name the regime.
5. **Whitening and centering are transductive** (fit on the test set) — state it every single time
   a whitened/centered number appears.
6. The `effective_rank` predictor and the whole depth-shift story are **MuQ-only** until MERT/
   OMAR-RQ confirm — write "for MuQ" and leave a slot for the cross-encoder result.
7. VGMIDITVar-timbre is **synthetic** (one SoundFont family, program-byte rewrite); the timbre-
   invariance claim is about this controlled setting — the BotW downstream is the real ecological test.

---

# APPENDIX B — MERT-v1-95M results (parallel to the MuQ findings above)

**Status:** full 13-layer + meanall variation-controlled sweep complete (2026-07-03), same offline
`vgm_timbre_sweep_from_cache.py` pipeline, same hardened fused single-pass metric code, same
VGMIDITVar-timbre corpus (N=102,960 = 5,040 works × ~2.55 variations × 8 GM instruments). Numbers
below are the source of truth for MERT — do not recompute; cite directly. Figures live in
`docs/figures/vgmiditvar_timbre_mert_varctl/` (same file names as the MuQ set: `fig1..fig7`,
`figS1..figS3`, `summary_table.csv`, `pool_means_by_layer.csv`).

## B.1 The one-line verdict

**MERT-v1-95M shows the *same qualitative story* as MuQ but *much weaker in magnitude*, and it never
produces the negative-gap artifact.** The timbre→composition depth shift is **general in direction**
(both encoders: same-timbre similarity falls, same-notes-across-timbre rises with depth) but
**MuQ-specific in magnitude** (MuQ's within-vs-twin gap converges 0.45→0.03; MERT's only narrows
0.34→0.18 and stays wide). This directly explains why **MuQ is ~2× better than MERT at honest
cross-orchestration retrieval** (best varctl cross MAP: MuQ L8 = 0.226 vs MERT L11 = 0.113).
→ **MuQ is the more timbre-invariant encoder**, the reverse of the retracted pre-control ordering.

## B.2 Finding 1 (MERT): the confounded gap is positive at *every* layer — the twin never flips it

Unlike MuQ (whose confounded gap goes negative from L2 on, reaching −0.187 at L11), **MERT's
confounded gap stays positive at every layer** (min **+0.006** at L11, max **+0.133** at L1). The
twin still inflates cross-timbre MAP and *narrows* the gap with depth, but never enough to flip it
negative — because MERT is not timbre-invariant enough for the same-notes-different-instrument twin
to become a top match. So for MERT the artifact is **milder and never produced a wrong-signed
result**; the confound correction still matters (it removes real inflation) but there was no
"cross is easier" illusion to overturn.

| layer | map_centered | map_whitened | within | cross (conf) | **gap (conf)** | cross (ctl) | **gap (ctl)** | twin inflation | eff_rank |
|---|---|---|---|---|---|---|---|---|---|
| 0 | .024 | .072 | .204 | .075 | **+.130** | .030 | **+.174** | .044 | 23 |
| 1 | .026 | .083 | .223 | .090 | +.133 | .037 | +.186 | .052 | 31 |
| 2 | .030 | .092 | .249 | .126 | +.123 | .055 | +.195 | .071 | 39 |
| 3 | .035 | .097 | .270 | .182 | +.088 | .081 | +.188 | .100 | 53 |
| 4 | .037 | .109 | .271 | .210 | +.061 | .091 | +.179 | .119 | 56 |
| 5 | .042 | .123 | .270 | .235 | +.035 | .098 | +.172 | .137 | 63 |
| 6 | .042 | .129 | .269 | .231 | +.038 | .093 | +.176 | .138 | 64 |
| 7 | .044 | .127 | .266 | .218 | +.048 | .088 | +.178 | .130 | 64 |
| 8 | .043 | .128 | .265 | .216 | +.049 | .087 | +.178 | .129 | 62 |
| 9 | .046 | .133 | .266 | .225 | +.041 | .091 | +.175 | .134 | 62 |
| 10 | .050 | .132 | .267 | .237 | +.030 | .098 | +.169 | .139 | 63 |
| **11** | .060 | .130 | .273 | .267 | **+.006** | **.113** | +.160 | .154 | 77 |
| 12 | .067 | .130 | .267 | .246 | +.021 | .103 | +.165 | .143 | 76 |
| meanall | .040 | .139 | .260 | .211 | +.049 | .087 | +.174 | .125 | 47 |

(Same column semantics as the MuQ table. `within` is again confounded≡controlled — the diagonal
has no twins. Twin inflation grows with depth here too, 0.044→0.154, for the same reason: deeper
layers make the same-notes twin a somewhat stronger match — just far less so than in MuQ.)

Figures: `fig1_gap_confound_vs_controlled` (gap stays above zero — visually the opposite of MuQ's
fig1), `fig2_within_vs_cross_panels`, `fig5_grids_best_layer` (L11 8×8 grids; off-diagonal
dominance *shrinks* under control but was never dominant to begin with).

## B.3 Finding 1b (MERT): the timbre→composition shift is present but much weaker (no convergence)

Same score-distribution dump, same four pools. MERT shows the **same direction** as MuQ but a
**much shallower** shift — same-timbre similarity barely falls, same-notes-across-timbre rises only
modestly, and the two **never converge** (gap stays ~+0.18):

| MERT layer | same-timbre, diff-variation (within +) | same-composition, cross-timbre (twin) | diff-variation, cross-timbre (honest) | distractor |
|---|---|---|---|---|
| 0 | **0.73** | 0.39 | 0.26 | −0.04 |
| 6 | 0.70 | 0.43 | 0.29 | −0.03 |
| 11 | 0.68 | 0.47 | 0.31 | −0.02 |
| 12 | 0.67 | **0.49** | 0.33 | −0.02 |

Within-vs-twin gap: **L0 +0.34 → L12 +0.18** — it narrows but stays wide (contrast MuQ: +0.45 →
+0.03). MERT stays **timbre-biased even at its deepest layer**: same-instrument-different-melody
(0.67) is still clearly closer than same-melody-different-instrument (0.49). Figures:
`figS3_timbre_composition_shift` (annotated "gap stays wide — does NOT converge, cf. MuQ +0.03"),
`figS1`, `figS2` (illustration layer L11).

## B.4 Finding 2 (MERT): honest cross-orchestration capability — ~half of MuQ, peaks deepest

Variation-controlled cross-timbre MAP (the leitmotif-operational metric) rises 0.030 (L0) → **best
0.113 at L11**, then dips at L12 (0.103); meanall 0.087. The peak is **deepest** (L11 vs MuQ's L8)
and roughly **half MuQ's height** (0.113 vs 0.226). Within-timbre MAP is flat ~0.27 across depth
(same shape as MuQ, slightly lower). **Depth buys cross-timbre generalization for MERT too, just
less of it, and later.** Figure: `fig3_cross_timbre_varctl_by_layer`.

## B.5 Finding 6 & 7 (MERT): whitening and the effective_rank predictor both replicate

- **Transductive whitening lifts MAP at every layer** (map_whitened ≈ 2–3× map_centered; peak
  ~0.139 at meanall / ~0.133 at L9). Same direction as MuQ, smaller absolute values. Figure:
  `fig4_geometry_treatments`. Same transductive caveat.
- **`effective_rank` predicts the best retrieval layer without labels — and *more strongly* than for
  MuQ**: corr(effective_rank, cross-varctl-MAP) = **0.976** (MuQ was 0.947). Both encoders' honest
  cross-timbre capability tracks unsupervised anisotropy. This upgrades Finding 7 from "MuQ-only" to
  "replicated on a second encoder" (n=2 encoders — still not a law, but no longer a single point).

## B.6 The cross-encoder comparison figure

`fig7_muq_vs_mert` (two panels, VIOLET=MuQ, ORANGE=MERT):
- **Left (capability):** honest cross-orchestration MAP by layer — MuQ's curve sits ~2× above
  MERT's at every depth; MuQ peaks L8=0.226, MERT peaks L11=0.113.
- **Right (mechanism):** within-timbre − same-composition-twin cosine gap by layer — MuQ descends
  to +0.03 (converges), MERT plateaus at +0.18 (stays timbre-biased). This one figure carries the
  whole comparison: *the depth shift is general in direction, MuQ-specific in magnitude.*

## B.7 Where MERT slots into Appendix A

- **A.1 map:** add an encoder column / comparison row. Findings 1, 1b, 2, 6, 7 all have a MERT
  counterpart above; the claims are the same *shape* with weaker numbers, plus the new
  cross-encoder claim (`fig7`): "MuQ is the more timbre-invariant encoder; the timbre→composition
  shift is general in direction but MuQ-specific in magnitude."
- **A.3 numbers:** MERT headline — confounded gap **positive at every layer** (min +0.006 L11),
  controlled gap **+0.160 to +0.195**; honest cross-timbre **best L11 = 0.113**; within-vs-twin
  score gap **+0.34 → +0.18 (does not converge)**; whitening 2–3×; eff_rank corr **0.976**.
- **A.4 slides:** `fig7` is the encoder-comparison slide ("MuQ hears the tune deep; MERT keeps
  hearing the instrument"). Use MuQ's `figS3` and MERT's `figS3` side by side for the mechanism.
- **A.5 precision rules (add these for MERT):**
  1. MERT's confounded gap is **positive at every layer** — do **not** say "MERT retrieves
     cross-timbre more easily" in any view. It never does.
  2. MERT's within-vs-twin gap **does not converge** (stays ~+0.18) — say "narrows," never
     "converges" for MERT (that word is MuQ-only).
  3. The MuQ>MERT ordering is on **honest, variation-controlled** cross-timbre MAP — always cite the
     controlled number (0.226 vs 0.113), never a confounded one.
  4. Everything is still **synthetic-render, transductive, single-SoundFont** — all MuQ caveats
     carry over verbatim.

## B.8 Assets (MERT)

Same layout as the MuQ asset table (§Assets). All in `docs/figures/vgmiditvar_timbre_mert_varctl/`:
`fig1`–`fig6` (MERT analogues of the MuQ figures), **`fig7_muq_vs_mert`** (the cross-encoder
comparison, MERT-only asset), `figS1`–`figS3` (score-geometry), `summary_table.csv` (all numbers in
B.2 + score seps), `pool_means_by_layer.csv` (the figS3 four-pool data, machine-readable). wandb:
group `MERT-v1-95M / VGMIDITVar-timbre`, runs `layer-{0..12}-test` + `layer-meanall-test`, tags
`from-cache, varctl` (2026-07-03).
