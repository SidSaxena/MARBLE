# JKUPDD Retrieval — CLaMP3-symbolic layer sweep

Within-piece motif **retrieval** on **JKUPDD** (JKU Patterns Development
Database, Collins 2013): the canonical MIREX motif-discovery ground truth, **5
cross-composer pieces** — J.S. Bach (BWV 889 fugue), Beethoven (Op. 2 No. 1, mvt
3), Chopin (Op. 24 No. 4 mazurka), Gibbons (*Silver Swan*), Mozart (K. 282, mvt
2). Zero-shot probe of CLaMP3-symbolic: embed every annotated
pattern-occurrence window, rank by cosine, score relevance as **same
`(piece, annotator, pattern)` group**.

This is the **cross-composer companion** to
[`bps_motif_retrieval_clamp3_layersweep.md`](bps_motif_retrieval_clamp3_layersweep.md)
(Beethoven-only, 32 sonatas). Its job is to test whether that sweep's headline —
**mid-layers L6/L7 ≫ final layer L12** — is a Beethoven artifact or a property
of CLaMP3's depth that holds across composers and eras (Baroque → Classical →
Romantic).

## ⚠️ Correction: the original MAP was inflated by trivial byte-duplicates

An earlier version of this sweep ran on the **raw 165 occurrence windows** and
reported a headline **L7 MAP ≈ 0.894**. That number was **inflated by a
trivial-duplicate artifact** and has been corrected. The cause:

- **JKUPDD ships every occurrence MIDI time-zeroed to t=0.** A literal
  pitch+rhythm phrase-repeat — extremely common in this corpus — therefore
  collapses to a **byte-identical** file inside its `(piece, annotator, pattern)`
  group.
- CLaMP3 maps identical bytes to an **identical embedding (cosine = 1.000)**, so
  a byte-twin is a **guaranteed rank-1 hit independent of encoder quality**. The
  duplicates were inflating MAP at *every* layer.
- Severity (raw 165): only **88 of 165 occurrences are byte-distinct**;
  **117/165 queries (71 %) have a byte-identical same-group twin**; **12 of 32
  groups are entirely byte-identical** across all their occurrences.

**Fix — within-group byte-dedup** (now default in
`scripts/data/build_jkupdd_retrieval.py`, `--no-dedup-within-group` to reproduce
the old build): keep **one representative per distinct byte-content per group**.
We dedup **byte-identical only** — transposed / rhythmically-varied repeats hash
differently and are **kept** (that is the legitimate motif variation the encoder
must be tested on). A retrieval group needs **≥2 distinct contents** to form a
valid query pool, so the **12 groups that collapse to a single distinct content
are dropped** (they can only produce trivial self-relevant queries).

**Surviving dedup'd dataset: 78 occurrence windows / 20 groups** (from 165 / 32;
**12 groups dropped**, all of them entirely-byte-identical ones — 5 Beethoven
`sectionalRepetitions`/`tomCollins` + 7 Mozart sectional/`barlowAndMorgenstern`).
Note: per-group dedup keeps **90** distinct contents, but **12** of those live in
the dropped single-content groups, leaving **78** in valid query pools (a further
2-vs-global gap is cross-annotator byte-twins, see *Caveats*). All numbers below
are on this clean 78-window set.

### As-run (inflated) vs dedup'd — per-layer raw MAP

| layer | raw 165 (inflated) | dedup 78 (clean) | Δ |
|------:|-------------------:|-----------------:|----:|
| 0  | 0.8233 | 0.7189 | −0.104 |
| 1  | 0.7992 | 0.6823 | −0.117 |
| 2  | 0.7967 | 0.6772 | −0.120 |
| 3  | 0.8110 | 0.6990 | −0.112 |
| 4  | 0.8348 | 0.7383 | −0.097 |
| 5  | 0.8567 | 0.7745 | −0.082 |
| 6  | 0.8762 | 0.8092 | −0.067 |
| **7** | **0.8939** | **0.8434** | **−0.051** |
| 8  | 0.8901 | 0.8337 | −0.056 |
| 9  | 0.8927 | 0.8400 | −0.053 |
| 10 | 0.8912 | 0.8350 | −0.056 |
| 11 | 0.8920 | 0.8373 | −0.055 |
| 12 | 0.8790 | 0.8247 | −0.054 |
| meanall | 0.8780 | 0.8138 | −0.064 |

Dedup costs **~0.05 MAP at the mid-stack peak and 0.10–0.12 at the surface**.
**L7 is still the argmax** (0.8434), and the headline ordering is unchanged — but
the *honest* number is **0.843, not 0.894**, and the surface layers fall
*further* (the duplicates were propping up the weak layers most), so the
mid-stack-over-surface gap **widens** on the clean set.

![per-layer MAP](jkupdd_retrieval_clamp3_layersweep.png)

## Run

- **13 layers × 1 run + 1 meanall = 14 runs**, all completed (rc=0), ~4 min wall
  on an RTX 5060 Ti (CUDA). Zero-shot (`max_epochs: 0`) → **no checkpoints**;
  the encoder forwards the 78 windows once, the 13 layer-jobs read the `(L, H)`
  embedding cache.
- **No CV folds.** JKUPDD is small (20 groups / 78 dedup'd occurrences), and
  motif identity is within-piece, so the whole benchmark is **one test set** —
  there is exactly one run per layer (contrast BPS-Motif's 5 folds).
- Date: 2026-06-21 (clean re-run). wandb: project `marble`, group
  `CLaMP3-symbolic / JKUPDDRetrieval`, names `layer-0-test` … `layer-12-test` +
  `layer-meanall-test`. Sweep coords (`sweep/layer`, `sweep/stage`) logged at run
  time via `LogSweepCoordsCallback`. (The earlier *inflated* runs remain in the
  wandb cloud, superseded — they are not deleted here.)
- Full metric suite via `log_extended_retrieval_metrics: true`.

## Results — `test/map` by layer: raw | centered | whitened (dedup'd)

| layer | raw | centered | whitened |
|------:|----:|---------:|---------:|
| 0  | 0.7189 | 0.7145 | 0.7181 |
| 1  | 0.6823 | 0.6808 | 0.6845 |
| 2  | 0.6772 | 0.6780 | 0.6837 |
| 3  | 0.6990 | 0.6943 | 0.7108 |
| 4  | 0.7383 | 0.7312 | 0.7561 |
| 5  | 0.7745 | 0.7676 | 0.7920 |
| 6  | 0.8092 | 0.8020 | 0.8202 |
| **7**  | **0.8434** | **0.8393** | **0.8498** |
| 8  | 0.8337 | 0.8245 | 0.8360 |
| 9  | 0.8400 | 0.8433 | 0.8376 |
| 10 | 0.8350 | 0.8398 | 0.8291 |
| 11 | 0.8373 | 0.8439 | 0.8219 |
| 12 | 0.8247 | 0.8413 | 0.8186 |
| **meanall** | 0.8138 | 0.8105 | 0.8190 |

**Best layer: 7 (MAP 0.8434 raw, 0.8393 centered).** Beats `meanall` by +0.030
raw / +0.029 centered (a *larger* margin than the inflated set's +0.016, because
removing the easy twins demotes `meanall` along with the weak layers).

## Does L6/7 peak + L12 collapse hold on this 5-composer set? (clean)

**The peak generalizes and is now *sharper*; the collapse is still mild.**

1. **Mid-layer peak — YES, at the *same depth*, and stronger after dedup.** The
   raw-MAP maximum is at **layer 7**, identical to BPS-Motif's best layer (7).
   Layers rise from an L1/L2 surface trough (**0.68**) to the L7 peak (**0.843**)
   — a **+0.16 mid-vs-surface gap**, roughly **double** the inflated set's gap
   (the trivial twins had been propping up the weak surface layers the most). The
   *location* of the motif-identity signal — mid-stack, around layer 7 — is
   **reproduced across composers and eras**, not a Beethoven artifact. This is
   the load-bearing cross-validation of the BPS finding, and it survives dedup
   intact.

2. **Mid-stack ≫ surface — YES, decisively.** L7–L11 form a high plateau
   (0.825–0.843); L0–L2 sit at 0.68–0.72. Surface layers are **~0.14–0.16 MAP
   below** the mid-stack on the clean set. The cross-composer evidence for
   "use a mid-layer, not the surface" is *stronger* after removing the
   duplicate floor.

3. **Last-layer taper — still mild, not a collapse.** On BPS-Motif, L12 was the
   *worst* layer (0.381, a −0.093 / −20 % drop from the L7 peak). On clean
   JKUPDD, L12 is **0.825 — −0.019 (−2.2 %) below the L7 peak** and still **above
   `meanall`**. The L7–L12 dip is real and consistent in *direction* (final-layer
   specialization for CLaMP3's *global contrastive* objective costs
   motif-occurrence locality), but on JKUPDD's small, easy pool it costs ~2 MAP
   points, not ~9. (Dedup did **not** change this picture: the L12 taper is the
   same ~2-point shape before and after.)

4. **Why the magnitude discrepancy with BPS is expected.** JKUPDD is
   **statistically thin and easy**: 78 windows, no folds, per-group relevant sets
   of 2–8 occurrences (vs BPS-Motif's long occurrence tails of dozens). With so
   few distractors, MAP stays high (the clean floor is 0.68, not ~0.4) and the
   whole curve is compressed, so inter-layer gaps — including L12's — are smaller
   than BPS's. **Read JKUPDD for the cross-composer *shape* (peak location,
   monotone rise, last-layer taper direction), not the magnitudes.** On *shape*,
   it agrees with BPS-Motif: mid-stack peak at L7, weakest at the surface
   (L1/L2), final layer below the peak.

**Verdict:** the BPS-Motif depth finding **partially generalizes**, and the
actionable half is now on *firmer* footing. *Use a mid-layer (L7), not the
all-layer mean or the final layer* holds across composers, and the
mid-vs-surface gap is **larger** once the trivial duplicates are removed. The
dramatic L12 collapse remains BPS-specific in *magnitude* (driven by hard, long
occurrence tails) but consistent in *direction* (L12 < peak everywhere).

## raw vs centered vs whitened (dedup'd)

The three variants stay **within ~0.02 MAP of each other at every layer** and
trade the lead back and forth (raw best at L0/L7/L8, centered at L9/L11/L12,
whitened on top at L3–L8). With only 78 windows and a single pool, the
anisotropy estimate is too noisy for centering/whitening to do reliable work; the
post-hoc transforms remain a **wash** here. Use **raw, layer 7**. (Whitening
edges raw at the peak by +0.006 — within noise; the centering benefit on
BPS-Motif came from much larger per-fold pools where the common-mean direction is
well estimated.)

## recall@K + secondary metrics (best layer 7, dedup'd)

| K | raw | centered | whitened |
|---:|----:|---------:|---------:|
| 1   | 0.3403 | 0.3446 | 0.3540 |
| 5   | 0.7784 | 0.7766 | 0.7778 |
| 10  | 0.9217 | 0.9251 | 0.9274 |
| 50  | 1.0000 | 1.0000 | 1.0000 |
| 100 | — | — | — |

| metric | raw / centered / whitened | reading |
|---|---|---|
| `map` | 0.8434 / 0.8393 / 0.8498 | high on this easy pool, but no longer near-1 |
| `r_precision` | 0.7802 / 0.7689 / 0.7753 | ~78 % of relevant retrieved at the R cutoff |
| `median_rank` | 1.0 / 1.0 / 1.0 | first relevant hit is still usually rank 1 |
| `mrr` | 0.8919 / 0.8979 / 0.9246 | first hit at avg rank ≈ 1.1 |
| `map@1` / `recall@1` | 0.3403 / 0.3403 / 0.3540 | top-1 ≈ 1/R of the occurrence set |
| `hit_rate@10` | 1.0 / 1.0 / 1.0 | every query has ≥1 relevant in top-10 |

**recall@1 rises from 0.268 (inflated) to 0.340 (clean)**: the old top-1 was
artificially low because each query's *own* byte-twin sat at cosine 1.0 and could
out-rank a genuine variant — removing the twins makes the top-1 a real
encoder-quality signal. **recall@100 is now N/A** (the dedup'd pool is only 78
windows < 100). **recall@50 = 1.0** still holds: with 78 windows total, the
top-50 always contains every same-pattern occurrence. JKUPDD measures *peak
location*, not the occurrence-tail recall that makes BPS-Motif (recall@100 only
~0.61) the harder, more discriminating benchmark. The two are complementary:
JKUPDD = cross-composer breadth, BPS-Motif = occurrence-tail depth.

## Caveats

- **Trivial-duplicate artifact (fixed).** The raw build double-counted
  time-zeroed byte-identical repeats as guaranteed rank-1 hits; this is now
  removed by default within-group byte-dedup. Re-build with
  `--no-dedup-within-group` only to reproduce the old inflated numbers.
- **Statistically thin.** 78 items, 20 groups, 5 pieces, **no folds** → no error
  bars. Treat every number as **directional**. The value is the **cross-composer
  spread** (does the *shape* survive a 5-composer, 4-century set?), not precision.
- **Still an easy pool → high MAP.** Few distractors compress the layer curve;
  gaps here are smaller than BPS-Motif's. Don't compare JKUPDD and BPS-Motif MAP
  *magnitudes* — only their *shapes*.
- Symbolic only (CLaMP3-symbolic, MIDI→MTF→M3 tokenisation); one clip-level
  vector per occurrence window (`TimeAvgPool` over patches, single layer).
- **Cross-annotator byte-twins (unchanged, conservative).** A few occurrence
  MIDIs are byte-identical across *different* annotator passes (e.g. Mozart K282
  `barlowAndMorgenstern|A` ≡ `barlowAndMorgensternRevised|C`). Within-group dedup
  does **not** touch these — they remain in *different* groups and are scored
  non-relevant despite identical content → a small, *conservative* MAP deflation
  (it under-states, never inflates). This is why per-group distinct (90) exceeds
  the global byte-distinct count (88). Pass `--annotators <one-per-piece>` for a
  single-source pool to avoid it.

## Reproduce

```bash
cd ~/developer/python/marble    # PC: /home/sid/developer/marble (WSL)
# 0. build the DEDUP'D dataset (default; 78 windows / 20 groups):
uv run python scripts/data/build_jkupdd_retrieval.py --jkupdd-root <path>
#    (add --no-dedup-within-group to reproduce the inflated 165-window build)
# 1. 13 layers + meanall on a CUDA box (~4 min, zero-shot):
.venv/bin/python scripts/sweeps/run_sweep_local.py \
  --base-config configs/probe.CLaMP3-symbolic-layers.JKUPDDRetrieval.yaml \
  --num-layers 13 --model-tag CLaMP3-symbolic --task-tag JKUPDDRetrieval \
  --accelerator gpu --skip-fit-if-no-train
# 2. aggregate (no folds → one run per layer):
.venv/bin/python scripts/sweeps/jkupdd_retrieval_summary.py \
  --out-csv docs/jkupdd_retrieval_clamp3_leaderboard.csv
# 3. per-layer MAP figure:
.venv/bin/python scripts/sweeps/plot_jkupdd_retrieval.py \
  --csv docs/jkupdd_retrieval_clamp3_leaderboard.csv \
  --out docs/jkupdd_retrieval_clamp3_layersweep.png
```

Requires the `symbolic-midi` extra and the built dataset under `data/JKUPDD/`
(`scripts/data/build_jkupdd_retrieval.py --jkupdd-root <path>`).
