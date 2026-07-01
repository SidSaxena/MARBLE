# MedleyDB → leitmotif FM-evaluation strategy (synthesis)

Synthesises four independent studies (task-landscape brainstorm, invariance/disentanglement design,
adversarial audit of the instrument + remix plans, and a training bonus). Goal: use MedleyDB to
evaluate — and optionally improve — how well frozen music FMs (MERT / MuQ / OMAR-RQ, layer-wise)
support **leitmotif discovery + identification** for the Breath of the Wild OST.

> **Governing caveat (applies to everything below).** These are **layer diagnostics, not selectors**.
> The real selector is the downstream `LeitmotifDetection` task (BotW `test/file_acc`). Every
> MedleyDB→BotW claim carries the "layer locator, not benchmark" caveat and must be confirmed on
> `LeitmotifDetection`. And MedleyDB is **57% vocal / pop-skewed** → the **instrumental-melody subset
> (~47 tracks) is the *primary* lens** for any orchestral-proxy claim, full-108 secondary.

## 1. What leitmotif ability actually requires (the requirement vector)

| ID | Requirement | Discovery | Identification |
|----|-------------|:--:|:--:|
| R1 | **Salience** — pull the predominant line out of dense polyphony | ● | ● |
| R2 | **Transposition identity** (contour/interval, key-invariant) | ● | |
| R3 | **Tempo/rhythm invariance** | ● | |
| R4 | **Timbre/orchestration invariance** (horn vs strings) | | ● |
| R5 | **Fragmentation** (partial/truncated statements) | | ● |
| R6 | **Segment thematic similarity** (retrievable phrase embedding) | ● | ● |
| R7 | **Handoff continuity** (lead passes instrument→instrument mid-line) | | ● |
| R8 | **Boundary/repetition sensitivity** | ● | |
| R9 | **Melody⊥timbre disentanglement** | | ● |

R4 and R7 are exactly what generic melody benchmarks never test and where orchestral VGM is hardest —
so that's where this suite should be distinctive.

## 2. The task suite (prioritised, with audit verdicts)

Ordered by (leitmotif-relevance × reuse × validity). ✅ built, 🟢 build, 🟡 conditional, 🔴 cut/defer.

| # | Task | Measures | Framing | Verdict |
|---|------|----------|---------|---------|
| **T1** | **MedleyDBMelody** (predominant f0 → 128-MIDI) | R1, R6 | frame probe, RPA/RCA | ✅ **built** — backbone locator |
| **T2** | **Salience-stratified melody** (MELODY3 polyphony density) | R1 under load | eval-time stratification of T1 | 🟢 **do next** — near-free, novel |
| **T3** | **Melody-through-handoff** (MELODY2 changing-lead + per-stem pitch) | **R7 + R4** | handoff-pair retrieval + ΔRPA@handoff | 🟢 **flagship** — most novel, most leitmotif-faithful |
| **T5** | **Intra-song recurring-theme retrieval** (f0-mined cliques) | R6, R2, R8 | zero-shot retrieval (whitening live) | 🟢 **discovery proxy** |
| **T4** | **Instrument-activation** (families) | R4 *locator* | frame multi-label probe | 🟡 **keep as diagnostic, with fixes (§3)** |
| **T7** | **Transposition / tempo** (pitch-shift / time-stretch) | R2, R3 | perpair retrieval surface | 🟡 cheap synthetic diagnostic, on demand |
| **T6** | **Stem-remix invariance** | accompaniment-inv. (not R4) | zero-shot retrieval | 🔴 **cut/gate (§4)** — redundant + leaky |
| **T8** | **Melody⊥timbre disentanglement** (2-axis plot + linear probe) | R9 | analysis notebook | 🟢 **interpretive payoff (§5)** |

**Minimal high-value program:** **T1 (have) → T2 → T3 → T5**, with **T4** as the interpretive
timbre-contrast and **T8** as the disentanglement read. That quartet answers: *find the melody in a
tutti (T1/T2), keep its identity through instrument handoffs (T3), and cluster its recurrences across
transformations (T5)* — precisely leitmotif discovery + identification.

### The two novel flagships (detail)
- **T3 Melody-through-handoff.** MELODY2 explicitly lets the lead source change (that's why MARBLE
  uses it). Cross-reference **MedleyDB-Pitch** per-stem f0 against the MELODY2 predominant f0 to find
  the *current lead stem*; when it switches while the melody stays continuous → a **handoff event**
  (instrument A→B from metadata). Readouts: (1) retrieval — is the melodic embedding just-before vs
  just-after a handoff a *positive* (same line, different timbre) that beats same-timbre-different-line
  negatives? (2) ΔRPA in a ±0.5 s window around handoffs. **No known MIR benchmark tests this.** Needs
  MedleyDB-Pitch (small download) + a handoff miner; sparse events → use the instrumental ~47 subset.
- **T2 Salience-stratified.** Per frame, count concurrent lines from **MELODY3** → bins {1,2,3,4+};
  report T1's RPA *as a function of polyphony*. Headline = **RPA@(4+) − RPA@(1)** slope. A flat layer
  = the salient-melody layer that survives a BotW tutti. Pure eval-time derivation of T1 — no new run.

## 3. T4 Instrument-activation — revised (audit-driven)

Keep, but **as a descriptive diagnostic, not a selector**, with these corrections:
- **Taxonomy fix.** MedleyDB's native top level is **6** families (Strings, Winds, Voices, Percussion,
  Electric, Other) — **Brass is nested under Winds**. For BotW, **break Brass out** into its own class
  (horn calls are the most leitmotif-relevant timbre) → a **custom ~7-way grouping**. My earlier plan
  said "7 families" but listed 6 and omitted Brass — that's the bug.
- **Class imbalance is real and unhandled.** MTGInstrument uses **unweighted BCE, macro-only metrics,
  and is clip-level (TimeAvgPool)** — so the "thin crib" claim is wrong: frame-level metrics + per-class
  AP are **new code**, and we must add **`pos_weight` / focal loss** or the rare orchestral families
  (Brass/Strings/Winds) collapse to "always-off."
- **The pop/vocal skew guts the *orchestral* axis.** The families that carry BotW leitmotifs are the
  **rarest and noisiest** in MedleyDB (the sample track has 3× singer + 2× guitar, zero orch). Rare-class
  variance is governed by **track count, not frame count** (~5 tracks / 5-fold → ~1/fold → per-family AP
  meaningless). So: **report per-family AP; split "orchestral" (Brass/Strings/Winds) from "pop"
  (Voices/Electric)**; consider **V2 (196 tracks)** *specifically* to gain Brass/Wind frames (accepting
  loss of melody-clip alignment for that variant).
- **Prefer a linear probe** (logistic-per-family, or CKA/MI on frozen features) over a trained MLP head:
  "how *linearly decodable* is instrument-identity at layer L" is a cleaner, cheaper "where does timbre
  live" reading; a trained `MLPDecoderKeepTime` can inflate weak layers and blur the axis.
- **Rename the claim:** it's an **instrument-identity / instrumentation** axis, not literally "timbre."

## 4. T6 Stem-remix — cut or gate (audit-driven)

**Recommendation: do not build as a layer selector.** Reasons:
1. **It likely can't rank layers at all** — every clique variant shares **bit-identical melody-stem
   audio**, so time-pooled cosine saturates ~1.0 at *every* layer. You can't both keep the shared
   content (to define the clique) and remove the shared audio (to make retrieval non-trivial).
2. **It measures accompaniment/mix invariance, not orchestration invariance** (the melody's own timbre
   is fixed) — i.e. *not* the leitmotif re-orchestration property.
3. **Whitening (its advertised free win) is in the weak N<H regime** here (108×K≈432 clips ≪ 768/1024
   dims → rank-deficient, drops to α=0.6 + ridge) — degraded exactly where we'd lean on it.
4. **Redundant.** The already-built **`VGMIDITVar-timbre`** computes the *real* cross-instrument
   invariance — `test/map_cross_condition`, `condition_gap`, an 8×8 GM-program grid, with soundfont
   rotation + constant reverb + EBU-R128 loudness controls; its own code comments call
   `map_cross_condition` "THE leitmotif-relevant metric." Real-audio texture isn't worth the 30–50 GB
   stems + a data-generation pipeline + the leakage.
- **If you still want it:** run a **triviality go/no-go on the free sample first** — render a few
  cliques, compute a raw-audio baseline (MFCC/chroma cosine) + one encoder layer; **if the baseline MAP
  already saturates, abandon** (no residual signal). Only then pull stems. Use the ~47 instrumental
  subset for a purer test.

## 5. Disentanglement — the interpretive payoff (T8)

Two mutually-reinforcing, high-reuse reads:
- **2-axis layer plot** (one point per layer per encoder): **x = melodic invariance** (T3/T7
  cross-transform retrieval MAP, or T1 RPA) vs **y = instrument-identity content** (T4 family-mAP on the
  *same tracks/folds*). Leitmotif-optimal layer = **high melody-invariance, comparatively low
  instrument-ID**. Hypothesis — confirm on `LeitmotifDetection`.
- **Pitch-vs-instrument linear probe** (the classic): train a *linear* head on the same frozen layer to
  predict pitch-class vs instrument-family; where each peaks localises the two attributes in depth.
  **This is an established result** — pitch-early / timbre-late per-layer probing: Pasad et al. (ASRU
  2021) for speech; MERT (arXiv 2306.00107) and MARBLE (NeurIPS 2023) for music; Castellon et al.
  (ISMIR 2021, Jukebox mid-layer optima). **Our contribution is the *setting*** (predominant-melody-in-
  polyphony, not isolated NSynth notes) **and the leitmotif framing**, not the phenomenon — per the
  standing no-overclaim rule, cite these before any novelty claim.

## 6. Invariance axes — build notes

All reframed as **zero-shot retrieval on pooled `(L,H)` clip embeddings via `CoverRetrievalTask`**
(so the centered/whitened metrics + cheap cache apply; the melody RPA probe stays the *locator*, not
the invariance reader). Reuse `compute_perpair_map` (cross-condition matrix = "same melody / different
X"), `compute_masked_map` (hard-distractor + stratification), and `compute_within_group_multilabel_map_with_null`
(the **permutation null** — the right tool for any shared-content leakage). Transposition (Axis 1) and
tempo (Axis 2) share one "render variants → clique jsonl" script and are the highest-*novelty* axes;
orchestration (Axis 3) is best served by **VGMIDITVar-timbre**, not remix; fragmentation (Axis 5) and
harmonization (Axis 6) are **weak on MedleyDB** (no motif/reharmonization labels) → run those on the
BPS/leitmotif symbolic corpora instead.

## 7. Bonus — training to improve leitmotif ability

Frame this as **cheap, gated, invariance-shaping adapters**, not backbone fine-tuning. Grounded in the
team's own prior results:
- **Frozen ceiling first (do before any training).** `vgmiditvar_timbre_3encoder_analysis.md` shows
  **MuQ L11 has a *negative* timbre gap** (cross-instrument MAP > within-instrument) suggesting the
  orchestration-invariance leitmotifs need is **largely present untrained** — but note the **−0.19
  magnitude is inflated by a construction confound** (the cross-instrument relevant set contains the
  query's own composition re-rendered in the target timbre, an audio near-duplicate the diagonal is
  denied by self-exclusion; see that doc's "same-composition twin" correction). The *direction* is
  real (a timbre-dominated encoder can't produce it) but run the variation-controlled control before
  quoting the number as the ceiling to beat. And
  `whitening_ablation.md`: post-hoc **whitening lifts cross-timbre MAP +109–425%**. So the bar every
  training move must beat is **whitened-frozen-MuQ-L11**, not raw-frozen.
- **Priority 1 — stem-remix / VGMIDITVar contrastive head** (InfoNCE on clique = same passage /
  different orchestration, frozen encoder → projection head). Reuse the union of MedleyDB-remix (real,
  accompaniment axis) **∪ VGMIDITVar-timbre (synthetic, true timbre-swap)** cliques — assemblable from
  shipped code. <1 day.
- **Priority 2 — melody + instrument-family LoRA with a gradient-reversal (adversarial) timbre arm** —
  the *trainable* version of "melody-decodable / timbre-not": actively removes timbre while keeping
  melody. LoRA r=8, cheap. Produces the melody-up / instrument-down signature that *is* the evidence.
- **Priority ≥4 — full melody-extraction backbone fine-tuning: deprioritise.** Invalidates the cache,
  1–4 days/encoder, and MedleyDB's vocal skew risks teaching *vocal*-salience features that hurt
  orchestral transfer.
- **Port the CLaMP3 contrastive playbook** (`clamp3_contrastive_training_plan.md`): cheapest-first
  ladder, P-K/InfoNCE sampling, inductive train-set centering, and — mandatory — the **cross-domain
  transfer GATE**: a gain that appears only on MedleyDB's own domain is memorisation (NO-GO); require
  it to transfer to VGMIDITVar (game-domain) and not regress HookTheoryMelody, and to beat
  whitened-frozen, before shipping. Final arbiter = Δ`LeitmotifDetection` `test/file_acc` on BotW.

## 8. Dependencies & sequencing

- **V1 audio (mixes only, ~4–5 GB) + GitHub annotations** cover T1, T2, T4, T5 (and the melody backbone).
- **MedleyDB-Pitch** (small Zenodo record) unlocks **T3** (handoff) and T8's clean templates — front-load it.
- **Stems (~30–50 GB)** only for T6 — and T6 is gated/cut, so **don't pull stems unless the triviality
  test passes**.
- **V2 (196 tracks)** only helps T4's rare orchestral families — optional, breaks melody-clip alignment.
- **Build order:** T2 (free) → T5 (miner + retrieval) → T3 (needs MedleyDB-Pitch) → T4 (fixed) → T8 plot.
  T6 gated; T7 on demand. Training bonus after the frozen+whitened ceiling is measured.
