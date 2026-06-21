# Symbolic-motif benchmark + CLaMP3 fine-tuning — strategic roadmap

**Status:** planning / synthesis. No code or experiments in this doc — it sequences
the work that follows from the CLaMP3-symbolic layer sweep.
**Scope:** which representation the symbolic motif-discovery pipeline should read,
how to confirm that choice generalizes beyond Beethoven, and how to lift the
supervised ceiling with a contrastive PEFT re-head.
**Audience:** anyone picking up the symbolic-motif effort — this should be enough
to act without re-deriving the findings.

---

## 1. Recap — what the layer sweep established

We probed *every* CLaMP3-symbolic layer (0–12) on BPS-Motif (the 32 Beethoven
first movements with ground-truth motif annotations, movement-level 5-fold CV)
two ways:

- **Supervised separability — MNID** (motif-window vs non-motif-window, frozen
  encoder + small MLP head): signal rises off L0, plateaus mid-network, and
  **peaks at L6 (`auc_roc` 0.836 ± 0.027)**. See
  [`bps_motif_mnid_clamp3_layersweep.md`](bps_motif_mnid_clamp3_layersweep.md).
- **Zero-shot geometry — Retrieval** (within-(piece, motif-letter)
  nearest-neighbour, no training): a slightly later but adjacent **peak at L7
  (MAP 0.474 raw / 0.488 centered)**. See
  [`bps_motif_retrieval_clamp3_layersweep.md`](bps_motif_retrieval_clamp3_layersweep.md).

Three facts that drive everything below:

1. **The motif signal lives mid-network; the final layer is the worst.** L12 — the
   contrastive projection head CLaMP3 was trained on for text↔audio↔symbolic
   alignment — is the single weakest probe on *both* tasks (MNID 0.697, Retrieval
   0.381). That objective rewards collapsing a whole piece into one alignable
   vector, which discards the fine-grained, position-dependent structure a motif
   task depends on. The default "use the pooled output embedding" choice is the
   wrong one.
2. **Pick one layer; never mean-pool.** L6 beats the mean-over-all-13-layers
   baseline by **+0.05 `auc_roc`** — the weak late layers actively drag the average
   down.
3. **Center, don't whiten.** Subtracting a common mean before cosine added ~+1.4 MAP
   points at *every* layer; ZCA-whitening lowered it. Free gain, no whitening.

What it means in one sentence: **the leitmotifs symbolic discovery pipeline today
reads the worst layer (L12-projected); switching it to the L6–L7 hidden band,
centered, is the single highest-leverage change available** — and it costs nothing
(same forward pass, different tap). This also refines the older "embedding is the
ceiling" framing: the ceiling is materially *higher* than the final-layer embedding
implied, once you read the right layer.

**Caveat that motivates §2:** all of the above is measured on **all-Beethoven solo
piano**. The *architectural* claim (mid-network ≫ output) is a property of CLaMP3
and should transfer; the *absolute* numbers will not. We need cross-texture
evidence before we trust the layer choice as a general rule — and certainly before
we spend fine-tuning budget on it.

---

## 2. Benchmark-expansion plan — does the layer finding generalize?

The goal here is **breadth, not depth**: confirm that "L6–L7 ≫ L12" holds across
composers, textures, and genres. Each dataset below is rated by the diversity it
adds and its MARBLE-integration cost. The integration template already exists — the
JKUPDD datamodule subclasses `_BPSMotifSymbolicBase` and reuses the BPS-Motif
MIDI→MTF→M3 tokenisation base verbatim; only the per-item relevance key
(`work_id`) and the build script differ. So any dataset that can be expressed as
"MIDI windows + a same-motif grouping key" is cheap to add.

### 2.1 JKUPDD — cross-composer confirmation (DONE — L7 peak confirmed) — **high value, near-zero cost**

- **What:** the JKU Patterns Development Database (Collins 2013). 5 pieces across
  **5 composers** (Bach, Beethoven, Chopin, Gibbons, Mozart), with multiple
  annotators per piece. Already built in `data/JKUPDD/` as a retrieval task
  (`JKUPDDRetrieval.test.jsonl`, **165 occurrence windows**, no CV folds), task
  code under `marble/tasks/JKUPDDRetrieval/`, config
  `configs/probe.CLaMP3-symbolic-layers.JKUPDDRetrieval.yaml`.
- **Diversity it adds:** breaks the all-Beethoven monoculture along the one axis
  that matters most right now — **composer / harmonic idiom**. Bach fugue, Gibbons
  Renaissance counterpoint, Chopin chromaticism, and Mozart classicism are
  genuinely different surface textures than Beethoven sonata-allegro. If the L6–L7
  peak survives across these five, the architectural claim is much harder to
  dismiss as a Beethoven artifact.
- **Honest limitation:** **statistically thin.** 5 pieces / 165 windows means
  per-layer MAP will be noisy and there are no CV folds (the cross-fold averaging
  that stabilizes BPS-Motif isn't available). Treat it as a *directional*
  confirmation — "the curve still bends down after L7" — not a precise re-estimate
  of the peak. Multiple annotators per piece partly compensate by multiplying the
  query set.
- **Effort:** done — sweep run + written up in
  [`jkupdd_retrieval_clamp3_layersweep.md`](jkupdd_retrieval_clamp3_layersweep.md).
- **RESULT (2026-06-21):** mixed, exactly as the thinness predicted.
  **Peak generalizes — best layer = 7** (MAP 0.894), the *same depth* as
  BPS-Motif, rising monotonically from an L1/L2 trough. So "mid-stack carries
  motif identity" is confirmed cross-composer, not a Beethoven artifact.
  **But the L12 *collapse* does not replicate:** L12 = 0.879, only −1.7% below the
  peak (vs −20% on BPS). JKUPDD's tiny, easy pool (165 windows, recall@50 = 1.0)
  saturates MAP near 0.9 at every layer, compressing the curve; centering/whitening
  are a wash for the same reason (noisy single-pool anisotropy). **Takeaway: the
  *actionable* half holds (use mid-layer L7), and the layer penalty is
  dataset-difficulty-dependent — bigger on harder, distractor-rich data (BPS, and
  presumably real game-music corpora) than on this saturated toy set.** This makes
  the harder MTC-ANN test (§2.2) more important, not less.

### 2.2 MTC-ANN / Meertens Tune Collection — Dutch folk monophony — **highest-value add**

- **What:** the annotated subset of the Meertens Tune Collection (MTC-ANN, ~360
  Dutch folk-song melodies) with **tune-family** membership and **phrase / motif
  annotations**. Monophonic vocal melodies.
- **Diversity it adds:** the **single biggest texture jump** available. Everything
  above is keyboard polyphony; MTC-ANN is monophonic oral-tradition folk song,
  where "the same motif" means melodic-contour identity under heavy oral variation
  (ornamentation, rhythmic stretching, transposition) rather than literal polyphonic
  repetition. If L6–L7 still wins *here*, the layer claim is about CLaMP3's motivic
  abstraction in general, not a polyphonic-keyboard quirk. Tune-family labels also
  give a *cross-tune* relevance signal that BPS-Motif's within-movement letters
  lack — a complementary retrieval framing.
- **Effort (moderate):** MTC-ANN ships as `**kern`/MIDI + a phrase/family annotation
  layer. A build script that slices phrase-marked windows and assigns a
  `(tune_family)` or `(tune, phrase)` relevance key, then a datamodule subclassing
  `_BPSMotifSymbolicBase` exactly like JKUPDD. The MIDI→MTF→M3 path is monophonic-
  safe. Main cost is the slicer + license/provenance check, not new infra. ~1 day
  to first sweep.
- **Verdict: prioritize this immediately after JKUPDD.** It is the highest-leverage
  generalization test in the plan, and it doubles as a future LoRA corpus (§3a) with
  clean tune-family positives.

### 2.3 Full BPS / Beethoven Piano Sonatas beyond BPS-Motif — **cheap incremental, lower value**

- **What:** the full BPS-FH / Beethoven-sonata corpus carries **phrase- and
  harmony-level annotations** beyond the motif labels BPS-Motif exposes. The build
  machinery (`build_bps_motif_dataset.py`, MIDI synth at 60 QPM, window slicer) is
  already in place.
- **Diversity it adds:** **little along the axis we care about** — it is still
  Beethoven solo piano, so it does *not* test cross-composer/cross-texture
  generalization. Its value is *vertical*: phrase/harmony annotations let us ask
  whether the same L6–L7 band that's best for *motifs* is also best for *phrase
  boundaries* and *harmonic function* — i.e. whether "mid-network" is a motif-
  specific or a general structure-level property of CLaMP3. Useful for the thesis's
  representation story, not for the breadth question.
- **Effort (cheap):** the M3 pipeline exists; this is mostly extending the build
  script to emit phrase/harmony windows + relevance keys. Low cost, but **defer** —
  it doesn't move the generalization needle and competes with MTC-ANN for time.

### 2.4 Essen Folksong Collection — large monophonic folk — **marginal; only if MTC-ANN is blocked**

- **What:** the Essen Folksong Collection (~6,000 `**kern` folk melodies) with
  **phrase markings**. Large, monophonic.
- **Diversity it adds:** more monophonic folk, but **MTC-ANN already covers this
  texture** with *better* (tune-family + motif) annotations. Essen's phrase marks
  are coarser than motif occurrences, so the within-group relevance signal is
  weaker for a motif-retrieval framing. Its one genuine advantage is **scale** —
  enough phrase pairs to matter as a *LoRA corpus* (§3) even if it's redundant as a
  *benchmark*.
- **Effort (moderate, mechanical):** `**kern` → MIDI → phrase-window slicer, same
  datamodule pattern. The volume makes the build longer but not harder.
- **Verdict: marginal as a benchmark — skip in the breadth phase.** Reconsider it
  later purely as augmentation-pair volume for the contrastive re-head if MTC-ANN's
  ~360 tunes prove too few.

### Benchmark priority, one line each

1. **JKUPDD** — in flight, ~free, cross-composer. Run + write up.
2. **MTC-ANN** — highest-value texture jump; ~1 day; do next.
3. **Full BPS phrase/harmony** — cheap but doesn't test breadth; defer to the
   thesis representation chapter.
4. **Essen** — redundant as a benchmark; hold as a LoRA-corpus fallback only.

---

## 3. LoRA / PEFT corpus plan — lifting the ~0.84 supervised ceiling

The frozen-encoder ceiling is **0.84 `auc_roc` (MNID) / ~0.49 centered MAP
(retrieval)** at the best layer. The retrieval doc's recall@K is the sharper
diagnostic of what's missing: `mrr` 0.94 and `hit_rate@10` 0.98 mean CLaMP3 almost
always finds *a* correct same-motif neighbour, but `recall@100` 0.61 means it
**pulls only ~half of a motif's full occurrence set together**. The encoder has
the right local geometry but doesn't make *all* occurrences of a motif mutually
near under the transformations they undergo (transposition, rhythm change,
ornamentation). That invariance is exactly what a **contrastive re-head** can teach.

**Approach (PEFT, not full fine-tune):** insert **LoRA adapters** into the M3
symbolic transformer blocks (attention + MLP projections; rank 8–16 is plenty for
a re-head), keep the base frozen, and train an **InfoNCE / supervised-contrastive
objective on motif-occurrence pairs**: occurrences of the *same* motif are
positives, everything else in the batch is a negative. **Train and read out at the
L6–L7 hidden tap** (the same one §2/the layer switch picks) — *not* the L12
projection — so the optimization targets the operating layer. Re-measure on the
exact BPS-Motif / JKUPDD retrieval tasks, so the lift is reported on the same ruler
as the frozen baseline.

Positive-pair sources, in order of cleanliness:

### (a) Labeled same-motif occurrences — BPS-Motif + JKUPDD — clean but small

- **What:** the existing ground-truth groupings. BPS-Motif gives ~4,944
  occurrences across 263 motifs; JKUPDD adds its 165 cross-composer occurrences.
  Same-`(piece, letter)` (or `(piece, annotator, pattern)`) = positive.
- **Pro:** zero label noise, already tokenised, already in MARBLE.
- **Con:** **small and Beethoven-skewed.** 263 motifs is thin for contrastive
  training; the all-Beethoven bias risks teaching a composer-specific invariance.
  This is the *anchor* set, not the whole corpus. **Critical discipline: hold out
  whole motifs (and ideally whole movements) for eval** — never let an occurrence of
  a train motif leak into test, or the lift is illusory.

### (b) Synthetic positives via transposition / tempo / ornamentation — the volume multiplier

- **What:** take each motif window and generate label-free positives by
  **transposition** (shift all pitches by ±k semitones), **tempo / duration scaling**
  (augment/diminish note durations), and **light ornamentation** (passing tones,
  neighbour tones, turn/mordent insertion). Each variant is, by construction, the
  same motif → a guaranteed positive pair. This is the cheapest way to multiply 263
  motifs into a large pair set and is precisely the invariance the recall gap calls
  for.
- **Reuse from the leitmotifs project — with a caveat.** leitmotifs already has a
  **transformation-stratified synthetic harness**, but it is **audio-domain**:
  `scripts/diagnostics/make_augmented_audio.py` does waveform pitch-shift /
  time-stretch and `plot_breaking_points.py` builds "breaking-point" survival
  curves (false-positive rate vs perturbation band, stratified by transposition and
  varispeed). **What transfers is the *methodology and the perturbation bands***
  (the exact ±semitone / ±rate grids already tuned there), and the
  stratified-evaluation idea — measure the LoRA lift *per transformation band* so we
  can see which invariances it actually buys. **What does NOT transfer is the code
  path:** symbolic LoRA needs **score-level** augmentation (operate on the MIDI /
  MTF tokens before M3), not on audio. So: borrow the transformation taxonomy and
  bands; write a small symbolic augmenter (transpose/scale/ornament on
  `pretty_midi`) feeding the existing MIDI→MTF→M3 pipeline.
- **Pro:** effectively unlimited positives, controllable difficulty, directly
  targets the measured invariance gap.
- **Con:** synthetic positives are *easier* than real occurrence variation — a model
  can overfit to "undo a clean transposition" without learning real motivic
  identity. Mitigate by mixing real (a) pairs in every batch and by stratifying
  augmentation strength.

### (c) In-domain game-music pairs — by-ear BotW annotations — small, on-target

- **What:** the leitmotifs project's **by-ear BotW motif annotations** (~111 label
  files under `leitmotifs/data/labels/`) mark recurring themes in the actual target
  domain. Same-theme spans → positive pairs in-domain.
- **Pro:** the *only* positives drawn from the real target distribution (game
  orchestration, repetition idiom, motif length) — the thing BPS-Motif's domain-gap
  caveat keeps flagging.
- **Con:** **smallest and noisiest** (by-ear, not score-aligned ground truth), and
  many are audio-annotated rather than symbolic. Use it to **adapt last** (§4 step
  4) and to **validate**, not to pre-train — it's the in-domain finishing set, not
  the bulk corpus.

### Rough data volumes & effort

| source | raw positives | after augmentation | effort |
|---|---|---|---|
| (a) BPS-Motif + JKUPDD | ~5.1k occ / ~428 motif groups | n/a (anchors) | ~0 — already built |
| (b) symbolic augmentation | — | 10–50× (a) | ~2–3 days: write symbolic augmenter + reuse leitmotifs bands |
| (c) BotW by-ear | ~111 label files | small | ~1–2 days: symbolicize / align, hold for final adapt |

LoRA training itself is cheap (adapters only; single-GPU, hours). The cost is in
the **corpus-building and the held-out-by-motif eval discipline**, not the compute.

---

## 4. Sequencing recommendation (and why this order)

1. **Free layer switch in leitmotifs (IN PROGRESS) → re-measure recall.**
   Tap the L6–L7 hidden layer (centered) in the symbolic-only discovery path,
   keep L12-projected only for the cross-modal symbolic×audio mode, and re-run
   within-piece recall on the by-ear BotW slice. See the handoff,
   `docs/2026-06-21-clamp3-layer-switch-handoff.md` (leitmotifs-symbolic repo).
   **Why first:** it is free (same forward pass), it is the single highest-leverage
   change, and **it sets the layer that everything downstream optimizes**. Doing a
   LoRA *before* the switch would tune adapters on the L12 projection — the worst
   layer — and we'd be optimizing the wrong space. Bank the free win first.

2. **Confirm breadth: JKUPDD → MTC-ANN.**
   Run the JKUPDD layer sweep (cross-composer), then add MTC-ANN (cross-texture
   monophony). **Why before fine-tuning:** if "L6–L7 ≫ L12" is a Beethoven artifact,
   we want to discover that *now*, on a one-day benchmark add, not after spending
   days building a contrastive corpus on the wrong premise. Confirming generalization
   is the cheap insurance that the fine-tuning targets a real, stable layer.

3. **Contrastive LoRA on pooled motif pairs (a + b).**
   Train LoRA adapters at the confirmed L6–L7 tap with InfoNCE on real (a) +
   symbolic-augmented (b) pairs; re-measure on BPS-Motif / JKUPDD retrieval with
   held-out motifs. **Why here:** only now is the target layer fixed and confirmed,
   and only now do we have a multi-texture corpus (BPS + JKUPDD + ideally MTC-ANN)
   that won't bake in a Beethoven-specific invariance. This is the step that
   actually attacks the 0.84 / 0.49 ceiling and the recall@100 ≈ 0.61 gap.

4. **Game-music LoRA last (c).**
   Finish-adapt (or co-train a small in-domain head) on the BotW by-ear pairs, and
   validate on the held-out game-music slice. **Why last:** it is the smallest and
   noisiest set, so it belongs as a *domain-adaptation finish* on top of a model
   that already learned general motivic invariance from (3) — not as a from-scratch
   target. Adapting to it first would overfit ~111 noisy labels.

This order matches the thesis's future-work framing — **layer/model fusion →
contrastive re-head (lead) → symbolic** — with the layer switch and breadth checks
as the "fusion/selection" prerequisites that must land before the contrastive
re-head (the lead intervention), and symbolic/in-domain adaptation trailing.

---

## 5. Cross-links

- **Layer-sweep evidence (this repo):**
  - [`bps_motif_mnid_clamp3_layersweep.md`](bps_motif_mnid_clamp3_layersweep.md) —
    supervised MNID, per-layer profile, L6 peak, mean-pool loss.
  - [`bps_motif_retrieval_clamp3_layersweep.md`](bps_motif_retrieval_clamp3_layersweep.md)
    — zero-shot retrieval, L7 peak, centered>raw>whitened, recall@K ceiling.
  - [`data/bps_mnid_leaderboard.csv`](data/bps_mnid_leaderboard.csv) — per-layer
    ranking (regenerate via `scripts/sweeps/bps_mnid_summary.py --out-csv`).
- **Forthcoming cross-composer confirmation (this repo, produced separately):**
  [`jkupdd_retrieval_clamp3_layersweep.md`](jkupdd_retrieval_clamp3_layersweep.md)
  — JKUPDD layer profile; the next external check that L6–L7 ≫ L12 generalizes.
- **Leitmotifs layer-switch handoff (leitmotifs-symbolic repo):**
  `docs/2026-06-21-clamp3-layer-switch-handoff.md` — the implementation plan for
  step 1 (expose M3 hidden states, tap L7, center; keep projected for cross-modal).
- **Setup / integration template:** [`data/bps_motif_setup.md`](data/bps_motif_setup.md)
  — build → JSONL splits → config → sweep; the pattern every new benchmark above
  reuses (subclass `_BPSMotifSymbolicBase`, swap the relevance key + build script).
