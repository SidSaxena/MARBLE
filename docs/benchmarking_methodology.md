# Benchmarking methodology — what MARBLE probe numbers mean

Canonical reference for *what the accuracy / MAP numbers we report actually
measure*, and how to describe them correctly in a paper or thesis. Written
after a long debug session where it wasn't obvious whether NSynth numbers
were "the encoder's pitch ability" or "an MLP's pitch ability given the
encoder's features" — answer: the second, and the field calls that
*probe-based evaluation* or *linear evaluation*. The numbers are still
meaningful and publishable; they just need the right framing.

If you're trying to write a methods section: jump to [How to describe the
numbers in a paper](#how-to-describe-the-numbers-in-a-paper).

## The one-line summary

> MARBLE measures **how well a small head can extract task-relevant
> information from frozen pre-trained encoder features.** That's an
> "MLP probe" benchmark — the standard methodology for evaluating music
> foundation models.

It is **not** zero-shot, **not** fine-tuning, and **not** the encoder
producing predictions by itself.

## Three evaluation modes MARBLE actually uses

| Mode | What trains | What's frozen | Tasks that use it | What it measures |
|---|---|---|---|---|
| **MLP probe** (supervised) | Small head (1 hidden layer, ~280 K params) | Encoder + feature extractor | NSynth, HookTheoryMelody, HookTheoryKey, HookTheoryStructure, GTZAN, EMO, GS, Chords1217, MTT, MTG\*, BPSMotif, SuperMarioStructure | "How linearly-separable is task X in the encoder's representations?" |
| **Zero-shot retrieval** | Nothing | Everything | Covers80, SHS100K, VGMIDITVar, HXMSA-retrieval, LeitmotifDetection | "Do similar musical inputs land close in the encoder's embedding space?" |
| **(Fine-tuning)** | Whole model | Nothing | *Not used in MARBLE's benchmark configs* — would conflate encoder quality with parameter count and ruin head-to-head comparison | n/a |

Each mode answers a different question. Use **MLP probe** numbers as
"information-extraction" benchmarks; use **zero-shot retrieval** numbers
as "embedding-space-structure" benchmarks. They can rank encoders
differently — that's expected and informative, not contradictory.

## Anatomy of an MLP-probe run (NSynth as worked example)

For the `probe.MuQ-meanall.NSynth.yaml` config:

```
Raw WAV (B, 1, T_samples)
    ↓
OMARRQ_FeatureExtractor          [squeeze (B,1,T)→(B,T) only]
    ↓
MuQ_Encoder                       [FROZEN, .eval(), train_mode=freeze]
    ↓
13 layer outputs (1 input + 12 transformer), each (B, T_tokens, 1024)
    ↓
LayerSelector(mode=mean)          [average across layers → (B, T, 1024)]
    ↓
TimeAvgPool                       [average across time → (B, 1024)]
    ↓
MLPDecoder                        [TRAINABLE: Linear(1024, 256) → ReLU →
                                   Dropout(0.2) → Linear(256, 88)]
    ↓
Logits (B, 88)
    ↓
CrossEntropyLoss                  [vs true MIDI pitch in 21..108]
```

**Trainable parameters**: ~280 K (just the MLP head). MuQ itself is
~370 M parameters, all frozen.

**Forward/backward**: encoder runs in `torch.no_grad()` (via
`train_mode=freeze`). Backward only flows through the MLP head.

**Optimizer**: `Adam(lr=1e-3)` updates the MLP head's ~280 K weights.

**Training trajectory**: 25–40 epochs on 50 K NSynth-train clips
(stratified subsample of the full 289 K), `EarlyStopping(patience=7,
monitor=val/acc)`.

**Reported number**: test accuracy on the held-out 4 K NSynth-test
clips, after the trained MLP head loads the best (lowest val/loss)
checkpoint.

The number quantifies: *given the encoder's frozen features and a tiny
linear-ish head trained on this exact task, how many test clips get
their MIDI pitch right?* It does NOT quantify the encoder alone, nor
the encoder if you fine-tuned it.

## Linear probe vs MLP probe vs fine-tuning

A spectrum, increasing in trainable params + nonlinearity:

| Method | Head architecture | Trainable params | What it tests |
|---|---|---|---|
| **Linear probe** | `Linear(H, n_classes)` | H × n_classes | Strict linear separability of features |
| **MLP probe** (MARBLE) | `Linear(H, 256) → ReLU → Linear(256, n_classes)` | ~(H × 256 + 256 × n_classes) | Mostly-linear separability + small nonlinear correction |
| **Fine-tuning** | Same MLP head + encoder unfrozen | Head + full encoder | End-to-end task performance (can mask weak features by training around them) |

MARBLE picked **MLP probe** as a middle ground:
- More expressive than linear probe (captures e.g. interactions between
  feature dimensions that pitch classes need).
- Still small enough that "task accuracy" is dominated by the encoder,
  not the head. A 280 K-param MLP can't memorise 50 K clips.
- Sweeps cheaply: 63-layer × 4-encoder × 4-task layer sweeps are
  tractable in days, not months.

If you want a *more* purely-encoder evaluation, swap `MLPDecoder` for
`LinearDecoder` (single affine layer) in the configs. Numbers will be
lower across the board but rankings tend to be similar.

## Task-specific gotchas

### NSynth is clip-level, not frame-level

Each NSynth clip is exactly 4 s of **one sustained musical note**. One
class label per clip (the MIDI pitch). The probe head is `MLPDecoder`
(clip-level), preceded by `TimeAvgPool` which collapses the time axis.

Contrast with HookTheoryMelody, which IS frame-level (per-frame melodic
pitch over polyphonic audio, 25 Hz labels). HTM uses
`MLPDecoderKeepTime` and does NOT TimeAvgPool. Don't conflate the two
when designing experiments or writing methods.

### CLaMP3 is a two-stage encoder

CLaMP3 isn't a single audio transformer like MuQ or MERT. It's:

```
Raw audio
    ↓
Stage 1: frozen MERT-v1-95M  [sliding-window 5 s, min chunk 1 s]
    ↓
Per-chunk MERT features (averaged over MERT's 13 layers)
    ↓
Stage 2: CLaMP3's 12-layer BERT-style audio model
    ↓
13 hidden states (embedding + 12 BERT layers) — these are the "layers"
the probe selects from.
```

**The 5 s "minimum" you may have heard is actually a nominal window
size, not a hard floor**: see `extract_mert_features_batch` in
`marble/encoders/CLaMP3/model.py`. The hard floor is 1 s. A 4 s NSynth
clip is processed as a single sub-5s chunk without padding (MERT
handles variable-length input natively).

**Caveat:** CLaMP3's BERT was pre-trained on long-form music with
*many* chunks per song; its job is modelling inter-chunk relationships.
On a single-chunk NSynth input it's operating at a degenerate input
length and can't use most of its capacity. Lower NSynth accuracy
than MuQ/MERT/OMARRQ is consistent with this design mismatch, not a
fundamental encoder weakness. Watch what CLaMP3 does on Covers80 /
SHS100K (song-level retrieval) — its sweet spot.

## How to describe the numbers in a paper

A safe, accurate methods paragraph:

> *We follow the standard linear-evaluation methodology for
> self-supervised audio encoders. For each task, the pre-trained
> encoder is frozen (parameters non-trainable, ``eval()`` mode) and a
> small MLP probe head (one hidden layer of 256 units with ReLU and
> dropout 0.2, ~280 K trainable parameters) is trained on the task's
> training split with Adam (lr=1e-3). The head receives either
> (a) the time-averaged mean of all transformer-layer outputs
> (``meanall`` configuration) or (b) a single specified layer's
> time-averaged output (per-layer sweep). Training runs for up to 40
> epochs with early stopping on validation accuracy (patience 7).
> Reported numbers are test-set accuracy at the best validation
> checkpoint.*

For zero-shot tasks (Covers80, SHS100K):

> *We report standard MAP@R cosine-similarity retrieval over frozen
> encoder embeddings, with no probe head trained. Each clip is
> embedded via the encoder's selected layer (per-layer sweep) or
> mean-of-all-layers (meanall), time-averaged, and L2-normalised; we
> rank query–gallery pairs by cosine similarity.*

Both are valid representation-evaluation methodologies — they answer
different questions, and ranking encoders across both is more
informative than either alone.

## Frequently asked variants

**"Why not linear probe?"** Pitch (and most MIR tasks) often benefits
from a small bit of nonlinearity over the encoder features. Linear
probe accuracy is a useful additional measurement (it's a stricter
test) but MLP probe is the standard practical reporting point.

**"Why not zero-shot for NSynth?"** NSynth doesn't have a canonical
zero-shot evaluation. You'd need either:
- A text-aligned encoder + class-name embeddings (CLIP-style — only
  CLaMP3 supports this), or
- An unsupervised clustering + cluster-naming pipeline.
Neither is a standard NSynth benchmark.

**"Why not fine-tune for the headline numbers?"** Fine-tuning is the
right call when you want to actually deploy a system. But for
*comparing encoders across tasks*, fine-tuning conflates encoder
quality with downstream parameter count (MuQ has 370 M, MERT-v1-95M
has 95 M — a head-to-head FT comparison would be unfair). The MARBLE
benchmark is a *representation evaluation*, and probing is the
representation-evaluation methodology.

**"Does the caching change what we're measuring?"** No (with the
NSynth-specific caveat that random crops would). Cache stores the
frozen encoder's deterministic output; cached and uncached probes
produce identical results modulo numerical precision. See
[embedding_cache_correctness.md](embedding_cache_correctness.md) for
details.

## See also

- [`layer_analysis.md`](layer_analysis.md) — per-encoder, per-task layer-selection guide
- [`embedding_cache_correctness.md`](embedding_cache_correctness.md) — when caching is and isn't safe
- [`performance_optimizations.md`](performance_optimizations.md) — sweep wall-clock optimisation reference
- *VGMIDITVar-leitmotif findings (deprecated, variant dropped)* — applied retrieval-evaluation methodology for the headline use case

---

## MAP self-exclusion fix — pre-fix vs post-fix numbers (2026-05-27)

A code audit found that `CoverRetrievalTask._compute_map` excluded self
via `sims_i[i] = -2.0` (a finite sentinel) rather than `-inf` + last-column
drop. With the finite sentinel, `argsort(descending=True)` placed self at
rank N rather than removing it; the `is_rel` mask matched self
(`work_ids[i] == work_ids[i]` is always True), inflating `n_relevant`
by 1 and adding a spurious hit at rank N.

**Per-query bias** scaled as `1 − n_true / (n_true + 1)`:

| Task | True n_relevant per query | test/map under-report |
|---|---|---|
| Covers80 | 1 (exactly 2 versions per work) | ~50% |
| SHS100K — canonical 2025 test split | ~63 (heavily skewed: 111 works, max 595, min 17) | **~1.6%** |
| SHS100K — our local set | ~61 (mean 61.45 versions/work after 3% YouTube attrition; same 111 works as upstream) | **~1.6%** |
| VGMIDITVar-timbre | ~7 (per-condition diagonal cells) | ~12.5% |
| VGMIDITVar base | varies | depends on per-work variation count |

The earlier draft of this table listed SHS100K at "~50%" — that figure
was an erroneous copy from the Covers80 row (assumed `n_true=1`). The
corrected number for our SHS-100K test set on disk is ~1.6% because
download attrition left a heavily-skewed distribution (median 32
versions per surviving work, see ``data/SHS100K/SHS100K.test.jsonl``).

**Fixed in** commit `ac121f0` on branch `fix/audit-cleanup-2` (merged
into main as part of the audit-2 batch). After this commit:

- `test/map`, `test/map_centered`, `test/map@1`, `test/map@1_centered`,
  `test/mrr`, `test/mrr_centered` produce numbers consistent with the
  standard IR-textbook MAP / MRR formulae.
- `test/recall@K`, `test/r_precision`, `test/median_rank`,
  `test/anisotropy/*`, `test/map_same_condition`,
  `test/map_cross_condition`, `test/condition_gap` are **unchanged**
  (they already used the correct -inf pattern via
  `marble.utils.retrieval_metrics._ranking_order`).

### Comparability

- **Pre-fix wandb runs vs post-fix wandb runs**: `test/map` values are
  NOT comparable. Post-fix is the correct number; pre-fix is
  systematically under-reported by the multiplicative factor in the
  table above.
- **Pre-fix relative rankings** (which layer is best, which encoder is
  best on a given task): preserved. The bias was multiplicative and
  uniform across encoders within a task.
- **Cross-task comparisons** in pre-fix data are particularly off
  (Covers80 was reported at half its true MAP while VGMIDITVar-timbre
  was off by only ~12.5%); avoid mixing pre-fix and post-fix numbers
  in any table.

### Action taken

All zero-shot retrieval sweeps (Covers80, SHS100K, all VGMIDITVar
variants) were re-run after the fix. The audit-2 cleanup branch
documentation enumerates which wandb groups were re-run.
