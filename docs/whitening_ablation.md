# Whitening ablation — VGMIDITVar-timbre cross-timbre retrieval

**Headline:** post-hoc **ZCA whitening** of frozen music-encoder
embeddings improves cross-timbre retrieval MAP by **+109 % to +425 %**,
with zero retraining and no new data. Pure whitening (`Σ^(−1/2)(e−μ)`,
exponent α=1.0, unregularised) is the best treatment for every encoder
tested.

> **Status:** result verified by an independent audit (see §5). The
> *magnitude* carries a generalisation caveat (§6) — VGMIDITVar-timbre's
> clean 8-timbre balance makes timbre a cleanly-removable variance
> direction; real-corpus gains may be smaller. The cross-instrument
> (off-diagonal) confirmation via the per-condition grid is **pending**
> (the per-pair run was deliberately held).
>
> **Date:** 2026-05-28. **Provenance:** `scripts/analysis/whitening_ablation.py`,
> figures + CSVs in `docs/figures/whitening_ablation/`.

---

## 1. What was measured

For each encoder's best layer (by overall centered MAP from the
VGMIDITVar-timbre sweep), we applied a sequence of linear
post-processing transforms to the frozen per-file embeddings and
recomputed overall retrieval MAP / recall@10 / r-precision:

| treatment | definition |
|---|---|
| `raw` | L2-normalise → plain cosine (encoder's native geometry) |
| `centered` | subtract corpus mean μ, renormalise (= the probe's `map_centered`) |
| `abtt-K` | subtract projection onto top-K principal components (Mu & Viswanath, ICLR 2018) |
| `whiten-aα` | ZCA whitening `U Λ^(−α/2) Uᵀ (e−μ)`, renormalise; α∈{0.5,0.7,1.0} |
| `whiten-a1.0-erel-E` | full whitening with relative Tikhonov ridge `Λ + E·λ_max·I` |

All treatments are **content-agnostic global linear maps** (they never
see work_ids), fit on the corpus covariance.

Best layers used: CLaMP3 L5, MERT-v1-95M L12, MuQ L12, OMARRQ-25hz L15.

---

## 2. Results

![treatment curves](figures/whitening_ablation/whitening_treatment_curves.png)

Overall centered MAP per treatment (best layer per encoder):

| treatment | CLaMP3 | MERT-v1-95M | MuQ | OMARRQ-25hz |
|---|---:|---:|---:|---:|
| raw | 0.0419 | 0.0637 | 0.1812 | 0.0709 |
| centered | 0.0446 | 0.0683 | 0.1856 | 0.0759 |
| abtt-1 | 0.0639 | 0.0618 | 0.1875 | 0.0790 |
| abtt-3 | 0.0800 | 0.0680 | 0.2468 | 0.1577 |
| abtt-10 | 0.1154 | 0.1081 | 0.3018 | 0.2724 |
| whiten-a0.5 | 0.1021 | 0.1119 | 0.3273 | 0.2655 |
| whiten-a0.7 | 0.1371 | 0.1265 | 0.3671 | 0.3423 |
| **whiten-a1.0** | **0.1651** | **0.1329** | **0.3854** | **0.3725** |
| whiten-a1.0-erel-1e-3 | 0.1465 | 0.1282 | 0.3733 | 0.3473 |
| whiten-a1.0-erel-1e-2 | 0.1252 | 0.1175 | 0.3466 | 0.3078 |

**raw → whiten-a1.0 gains:** CLaMP3 +294 %, MERT +109 %, MuQ +113 %,
OMARRQ +425 %.

### Key observations

1. **Pure α=1.0 wins for every encoder.** The progression is monotonic
   through α, and the regularised (`-erel`) variants score *lower* —
   the Tikhonov ridge damps exactly the low-variance directions that
   carry the discriminative signal. The "noise-amplification" worry
   (whitening blows up tiny-eigenvalue directions) did not materialise,
   even for MuQ whose condition number is 1.7e6.

2. **Whitening reorders the encoders.**
   - Raw: `MuQ (0.181) > OMARRQ (0.071) > MERT (0.064) > CLaMP3 (0.042)`
   - Whitened: `MuQ (0.385) > OMARRQ (0.372) > CLaMP3 (0.165) > MERT (0.133)`

   OMARRQ leaps from 4th-ish to nearly tying MuQ for first. CLaMP3 and
   MERT swap. MuQ stays champion and its lead in absolute terms grows.

3. **Encoders respond differently — diagnostic of their cone.**
   - **OMARRQ (+425 %)** benefits most: its variance is the most
     "nuisance-dominated" (timbre/cone directions carry little
     work-identity), so removing them is almost pure gain. `abtt-1`
     alone already helps it.
   - **CLaMP3 (+294 %)** is similar — its top PCs were pure cone/text
     directions (`abtt-1` gave +53 %).
   - **MERT (+109 %)** benefits least, despite having the *highest*
     cone (`mean_vec_norm ≈ 0.89`). Tellingly, MERT's `abtt-1` *hurts*
     (0.062 < 0.064 raw): its top principal component carries useful
     signal, so naively removing it is counterproductive. Whitening
     (which downweights rather than deletes) still helps, but less.
   - **MuQ (+113 %)** — already the least cone-collapsed and the raw
     champion, still gains substantially.

4. **This is the inverse of the leitmotifs PCA-256 result.** That study
   (`/Users/sid/leitmotifs/docs/pca_analysis.md`) showed *dropping*
   low-variance directions (PCA-256) destroyed cross-timbre similarity.
   Whitening *up-weights* those same directions and maximally helps.
   Two sides of one coin: the cross-timbre discriminative signal lives
   in the low-variance tail.

---

## 3. Why this works (mechanism)

VGMIDITVar-timbre renders every MIDI piece with 8 GM instruments, so
**timbre is the single largest source of variance across the corpus**.
The covariance Σ is dominated by timbre directions. Plain cosine
similarity is therefore dominated by timbre agreement — a query
retrieves same-*timbre* items, not same-*work* items.

Whitening rescales every principal direction to unit variance:
`Σ^(−1/2)(e−μ)`. This **downweights the high-variance timbre
directions and up-weights the low-variance melodic/harmonic
directions** — exactly the directions that stay constant when the same
melody is played on different instruments. The resulting Mahalanobis
cosine `(e_a−μ)ᵀ Σ^(−1)(e_b−μ)` measures relative musical similarity
with timbre's contribution flattened out.

---

## 4. How this was computed — NO encoder re-runs

The original layer sweeps already ran the expensive part (encoder
forward over 102,960 audio files) and **cached the per-clip
embeddings** to `output/.emb_cache/<encoder>/VGMIDITVar-timbre__<hash>/*.pt`
(one `(L, H)` tensor per clip, L = all hidden-state layers).

`whitening_ablation.py` **never touches the encoder**. It:

1. Loads the cached `(L, H)` tensors (just file reads — no GPU forward).
2. Slices layer N, L2-normalises each clip, mean-pools clips per file,
   renormalises → `(N=102960, H)` per-file matrix. This reproduces the
   probe's `forward` + `on_test_epoch_end` aggregation exactly.
3. Fits PCA on the corpus (μ, Σ = U Λ Uᵀ, in fp64 for stability).
4. Applies a transform, then recomputes MAP via the same streaming
   argsort the probe uses.

The only thing "re-run" is the cheap metric computation under different
post-processing. The cost is dominated by the cache I/O (102k file
reads), not compute.

---

## 5. Audit — how we know +425 % is real, not a bug

A +425 % gain demands skepticism. Every link in the chain was
independently verified:

| check | method | result |
|---|---|---|
| **Pipeline** (load+aggregate+metric) matches the live probe | `--verify` compares raw + centered MAP to the sweep's logged `test/map` / `test/map_centered` | Δ ≤ 1.5e-6 (CLaMP3, MuQ) |
| **Mechanism** is real | synthetic cone (timbre 8× work amplitude), independent numpy whitening + brute-force MAP | whitening recovers work-identity, 11× on ground truth |
| **Transform** is correct | script's `whiten-a1.0` vs independent numpy ZCA, cosine geometry on a subset | Δ = 7.4e-7 |
| **Metric** is correct | script streaming MAP vs independent brute-force MAP | Δ = 0.0000 |
| **Real-data number** reproduces | independent numpy whitening + brute-force MAP on the actual CLaMP3 L5 matrix | raw 0.043 (script 0.042), whiten 0.166 (script 0.165), ×3.89 (script ×3.94) |

The one discrepancy found during the audit was a **self-exclusion bug
in the audit code itself** (counting the query as its own relevant item),
not in the script — once fixed, the independent and script MAP agreed
to 0.0000.

The keystone synthetic check is baked into CI as
`tests/test_whitening_ablation.py::test_whitening_recovers_buried_signal_vs_independent_numpy`.

**Verdict:** the gains are correctly computed.

---

## 6. Caveats (read before claiming this transfers)

1. **Generalisation / corpus structure.** VGMIDITVar-timbre has
   *exactly* 8 timbres per piece, making timbre a cleanly-dominant,
   cleanly-removable variance direction. On a real soundtrack corpus
   the nuisance variation is messier; the *magnitude* of the gain may
   shrink. **Must be tested on SHS100K / Covers80** before claiming the
   number transfers.

2. **Transductive fitting.** μ and Σ are fit on the same corpus the
   retrieval is evaluated over. This is not label leakage (the
   transform is content-agnostic — it never sees work_ids and applies
   the same matrix to every embedding), and it matches the existing
   `map_centered` protocol (which also uses the corpus mean). But a
   *strict* zero-shot deployment would fit μ, Σ on a disjoint reference
   set; the gain might be slightly lower.

3. **Layer choice.** We only whitened each encoder's *best-raw* layer.
   Whitening could change *which* layer is best (a heavier-cone layer
   might overtake once whitened). Untested. A focused follow-up: whiten
   3-4 candidate layers per encoder (e.g. MuQ {L8, L11, L12}).

4. **Cross-instrument confirmation pending.** These are *overall* MAP
   numbers (mixing within- and cross-timbre). The overall gain almost
   certainly includes a large cross-timbre component (off-diagonal cells
   are 56 of 64), but the per-condition grid for the whitened
   embeddings hasn't been run yet — it would quantify the gain on the
   leitmotif-specific off-diagonal.

---

## 7. Recommendations / next steps

In rough priority:

1. **Run the per-condition grid for whiten-a1.0** (the held per-pair
   run) — confirm and quantify the cross-instrument gain.
2. **Layer study** — whiten MuQ {L8, L11, L12} and the other encoders'
   candidate layers; see whether the optimal layer shifts.
3. **Bake `test/map_whitened` into the probe** — mirror the existing
   `map_centered` path so every future sweep logs the whitened metric
   as a first-class number (~30-line addition to
   `marble/tasks/Covers80/probe.py`).
4. **Generalisation test** — whitening ablation on SHS100K / Covers80
   to see whether the gain holds when the nuisance variation isn't a
   clean timbre axis.
5. **Deployment** — for a fixed retrieval database, whitening is free
   (fit μ, Σ once). For a growing database, fit on a representative
   reference set.

---

## 8. Reproducibility

```bash
# Fast pass (overall metrics, all treatments) for one encoder/layer:
uv run python scripts/analysis/whitening_ablation.py \
  --encoder CLaMP3 --encoder-tag CLaMP3-layers --task-tag VGMIDITVar-timbre \
  --layer 5 \
  --jsonl data/VGMIDITVar-timbre/VGMIDITVar.jsonl \
  --cache-dir output/.emb_cache/CLaMP3/VGMIDITVar-timbre__<hash> \
  --skip-perpair --verify

# Drop --skip-perpair to also compute the 8x8 cross-instrument grid.
```

| artefact | content |
|---|---|
| `docs/figures/whitening_ablation/<encoder>_L<N>_overall.csv` | per-treatment metrics, one encoder |
| `docs/figures/whitening_ablation/whitening_treatment_curves.png` | the curves above |
| `scripts/analysis/whitening_ablation.py` | the ablation script |
| `tests/test_whitening_ablation.py` | 14 tests incl. the independent-numpy audit |

### Related

- `docs/vgmiditvar_timbre_3encoder_analysis.md` — the underlying
  4-encoder layer-sweep comparison (raw embeddings).
- `docs/anisotropy.md` — the cone-collapse metrics that motivated this.
- `/Users/sid/leitmotifs/docs/pca_analysis.md` — the PCA-256 result this
  inverts.
