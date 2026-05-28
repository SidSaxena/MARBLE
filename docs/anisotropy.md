# Anisotropy in MARBLE retrieval probes

What "anisotropy" means here, how each of the four logged metrics
measures a different facet of it, what each one *cannot* tell you, and
how they translate into practical decisions about centering and
encoder selection.

> **Audience**: anyone reading `test/anisotropy/*` numbers on a wandb
> run, trying to decide whether `test/map_centered` is the headline
> number, or deciding which encoder layer to pick for downstream
> retrieval.
>
> **Provenance**: written 2026-05-28 after an audit of
> `anisotropy_metrics` triggered by the CLaMP3 × VGMIDITVar-timbre
> sweep. The audit corrected two pieces of folklore captured in older
> docs and the diagnostic script's verdict heuristic.

---

## The intuition in one paragraph

If an encoder is **isotropic**, random pairs of unit-norm embeddings
have cosine similarity ~0 — there is no shared direction, no preferred
axis, and the cosine between two embeddings really does measure their
specific relationship. If the encoder is **anisotropic**, most pairs
look artificially similar because all embeddings live in a thin region
of the hypersphere (a "cone"). Cosine retrieval then surfaces items
because they share that cone, not because they share the *content* the
user cares about. MARBLE measures four properties of the embedding
cloud to diagnose this and the related rank-collapse failure mode.

---

## The four metrics

All four are computed once per probe run, in
`marble.utils.retrieval_metrics.anisotropy_metrics`, on the L2-normalised
per-file mean-pooled embeddings (`embs`, shape `(N, H)`). Logged at
`test/anisotropy/{mean_vec_norm, avg_pair_cos, top1_sv_share, effective_rank}`.

### 1. `mean_vec_norm` — the cone-collapse detector

**Definition.** For each row `e_i`, L2-normalise it. Take the mean over
N rows. Report the L2 norm of that mean vector:

```
mean_vec_norm = || (1/N) Σ_i (e_i / ||e_i||) ||
```

**Range.** `[0, 1]`. Equal to 0 iff the unit vectors sum to the zero
vector (perfectly balanced); equal to 1 iff every unit vector points
in the same direction.

**Interpretation.**

| value | meaning |
|---|---|
| `~ 1/√N` | isotropic — the headline reference value |
| `< 0.1` | weakly anisotropic |
| `0.1 – 0.5` | moderate cone |
| `0.5 – 0.9` | strong cone — centering matters for retrieval |
| `> 0.9` | severe cone — virtually all directions share an axis |

**Theoretical baseline.** For N iid unit vectors drawn uniformly on
S^(H-1), `E[mean_vec_norm] ≈ 1/√N`. This is the "random walk on the
sphere" result — a useful sanity floor. At `N = 102 960` the isotropic
floor is `1/√102960 ≈ 0.003`. CLaMP3 at every layer measures ~0.85 —
**280× the isotropic baseline**, an unambiguous strong cone.

**What it does NOT tell you.** Whether the residual variance (after
removing the corpus mean) is concentrated in a few directions or spread
across many. For that you need `top1_sv_share` and `effective_rank`.

---

### 2. `avg_pair_cos` — independent cross-check on `mean_vec_norm`

**Definition.** Sample `n_pairs` (default 5000) random off-diagonal
index pairs `(a, b)`. Compute `cos(e_a, e_b)` for each, take the mean.

**Range.** `[-1, 1]`.

**Why it exists.** For L2-normalised vectors there is a closed-form
relationship:

```
E[cos(e_a, e_b)]  ≈  ||μ||²  =  mean_vec_norm²
```

where `μ = (1/N) Σ (e_i / ||e_i||)`. So `avg_pair_cos` *should* equal
`mean_vec_norm²` up to Monte Carlo noise of `O(1/√(n_pairs · H))`.

**Use it as a sanity check.** If the two numbers diverge meaningfully —
say `avg_pair_cos` is 0.4 but `mean_vec_norm²` is 0.7 — the
implementation has a bug: maybe a row wasn't normalised, maybe the pair
sampling is biased, maybe the corpus mean is computed before
normalisation instead of after. The relationship is pinned by
`tests/test_retrieval_metrics.py::test_anisotropy_pair_cos_matches_mvn_squared`
to `abs=0.005`.

**`avg_pair_cos` is informationally redundant** with `mean_vec_norm`,
but cheap (~1 ms) and a real correctness signal. Log it.

---

### 3. `top1_sv_share` — second-direction dominance

**Definition.** Compute SVD of the **centered** embedding matrix
`X̃ = X − X̄`. Take the first singular value squared, divide by the sum
of all squared singular values:

```
top1_sv_share  =  σ_1²  /  Σ σ_k²
```

**Range.** `[0, 1]`.

**Interpretation.**

| value | meaning |
|---|---|
| `~ 1/H` (Marchenko-Pastur edge) | isotropic residual |
| `< 0.1` | residual spread across many directions |
| `0.1 – 0.3` | one direction starts to dominate the residual |
| `> 0.3` | strong secondary direction (e.g. a binary cluster split) |

**Crucial subtlety.** Because the SVD is on **centered** embeddings,
this metric **does NOT** detect cone collapse — the corpus mean is
the first direction of variance, and centering removes it. What
`top1_sv_share` detects is a *second* shared direction that survives
mean removal: a binary cluster split, a domain marker (e.g., live vs
studio recording), or any orthogonal-to-the-cone structural bias.

**Worked example.** Suppose all embeddings are `+e_0 + 0.3·δ` where δ
is per-item Gaussian noise. `mean_vec_norm ≈ 1` (severe cone) but
`top1_sv_share ≈ 1/(H−1)` (no post-centering structure). Inverse case:
two equal-size clusters at `+e_1` and `−e_1`. `mean_vec_norm ≈ 0`
(perfectly balanced — no cone) but `top1_sv_share ≈ 1` (the e_1 axis
explains all variance).

---

### 4. `effective_rank` — residual structural diversity

**Definition.** With singular values `σ_k` of the centered matrix:

```
share_k = σ_k² / Σ_j σ_j²
H = − Σ_k share_k · log(share_k + 1e-12)   # entropy in nats
effective_rank = exp(H)
```

This is the **exponential entropy** of the normalised post-centering
variance spectrum. It generalises "matrix rank" to a continuous measure:
if `n_eff` of the singular values are equal and the rest are zero,
`effective_rank = n_eff`.

**Range.** `[1, min(N, H)]`.

**Interpretation.**

| value | meaning |
|---|---|
| `≈ min(N, H)` | isotropic residual (with MP edge reduction → ~0.85·min(N,H) at our sample sizes) |
| `H · (0.1 – 0.5)` | moderate structural compression |
| `< H/10` | severe rank compression |
| `≈ 1` | residual collapses to a single direction |

**Crucial subtlety #1.** `effective_rank` is computed on the
**centered** matrix. It does **NOT** drop for pure cone collapse — the
cone direction is exactly what centering removes. After centering a
"pure cone + iid Gaussian noise" matrix, the residual is uniform across
the remaining H−1 dimensions and `effective_rank ≈ H − 1`. This is
pinned by
`tests/test_retrieval_metrics.py::test_anisotropy_effective_rank_high_for_pure_cone_collapse`.

**Crucial subtlety #2.** A low `effective_rank` indicates **structural
collapse** — many features live in a low-dimensional subspace, possibly
because the encoder over-specialised. For retrieval tasks, **the
best-MAP layer often sits at the peak of `effective_rank`**: enough
expressive capacity to encode varied content, not so collapsed that
items become indistinguishable.

---

## How the four metrics relate

The four metrics decompose the anisotropy diagnosis into independent
axes:

| question | metric |
|---|---|
| "Is there a shared common direction?" | `mean_vec_norm` |
| "Confirms that?" | `avg_pair_cos` |
| "After removing the common direction, is there a *second* dominant axis?" | `top1_sv_share` |
| "After removing the common direction, how broad is the residual?" | `effective_rank` |

Equivalently: `mean_vec_norm` measures the **0th moment** (the mean),
while `top1_sv_share` and `effective_rank` measure the **1st moment**
(the covariance) of the centered distribution.

### Four-quadrant taxonomy

|  | Low cone (`mean_vec_norm` small) | High cone (`mean_vec_norm` large) |
|---|---|---|
| **High effective rank** | Isotropic. Best case. Centered ≈ Raw cosine MAP. | Cone collapse, high diversity. Centering helps moderately. CLaMP3 lives here. |
| **Low effective rank** | Multi-cluster but no global cone. Rare. Inspect `top1_sv_share`. | Cone + rank collapse. Severe. Both centering AND whitening helpful. OMARRQ historically lived here. |

---

## Practical implications

### When should you use centered cosine retrieval?

**Always log both**, but trust `test/map_centered` for headline
comparisons when the encoder is anisotropic (any of):

- `mean_vec_norm > 0.3`, OR
- `effective_rank < min(N, H) / 3`

For CLaMP3 (`mean_vec_norm ≈ 0.85`, `effective_rank ≈ 50/768`),
centered MAP runs **7–11 % higher** than raw MAP across layers. The
improvement is real but moderate — the cone shifts pair cosines
uniformly upward, but the ranking signal `(δ_a · δ_b)` survives mostly
intact. For encoders with worse anisotropy the gap can be much larger.

### Will centering change which items I retrieve, or just inflate MAP?

**Both, but mostly the latter.** Centering subtracts a *constant* from
every cosine (the cone contribution), which preserves rankings exactly
*on average*. The deviation comes from the per-item `(μ · δ_a)` and
`(μ · δ_b)` cross terms — these shuffle close-rank pairs but rarely
move an item across the top-10/top-50 boundary. For most queries the
top-10 set under centered cosine and raw cosine overlaps by ~80%.

If your downstream task is "is the very best item right?" (top-1
retrieval, hard-negative mining), centered cosine helps less. If it's
"surface the relevant items in the top-K" (MAP/recall@K), centered
cosine matches MAP improvements directly.

### When does centering NOT help?

Three cases:

1. **Classification probes** (any task with an MLP head + bias
   terms). The bias absorbs the mean shift; centering input has zero
   effect on the optimum.
2. **Anisotropy dominated by `top1_sv_share`** rather than
   `mean_vec_norm`. Centering removes the cone but not a binary cluster
   split. **Whitening** (subtract mean *and* multiply by `Σ^(-1/2)`)
   handles this; we don't currently log a whitened MAP.
3. **Already-isotropic encoders** (`mean_vec_norm < 0.05`). Centering
   is a no-op up to numerical noise; raw MAP and centered MAP will
   match to ~3 decimal places.

---

## Reference values from MARBLE sweeps

Snapshot from completed runs as of 2026-05-28. Updated as new sweeps
land.

### CLaMP3 × VGMIDITVar-timbre (13 layers, N=102 960)

| layer | mean_vec_norm | effective_rank | MAP raw | MAP centered | Δ |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.914 | 46.4 | 0.0379 | 0.0419 | +10.6% |
| 4 (best) | 0.848 | 61.0 | 0.0416 | 0.0445 | +7.0% |
| 12 | 0.856 | 47.3 | 0.0345 | 0.0365 | +5.7% |

CLaMP3 has heavy cone collapse at every layer but moderate residual
diversity. Layer 4 maximises both `effective_rank` and cross-condition
MAP — the connection between residual diversity and retrieval quality
is clearly visible.

### OMAR-RQ × Covers80 (24 layers, N≈200, frame-level)

| layer | avg_pair_cos | mean_vec_norm | effective_rank | n_dim |
|---:|---:|---:|---:|---:|
| 0 | 0.208 | 0.457 | 110 | 1024 |
| 12 | 0.244 | 0.494 | 124 | 1024 |
| 23 | 0.275 | 0.525 | 136 | 1024 |

OMAR-RQ shows **both** cone collapse (`mean_vec_norm ≈ 0.5`) and
significant rank compression (`effective_rank ≈ 110-135 out of 1024`,
~11-13% of full dim). It sits in the bottom-right of the four-quadrant
taxonomy. Centering helps but whitening would help further. (See
`docs/data/omar_rq_audit.md` for the full audit.)

---

## Common interpretation pitfalls

### "Low `effective_rank` means the encoder is anisotropic"

Partially. `effective_rank` measures residual structural diversity
after the cone is removed. A cone-collapsed encoder can have a *high*
`effective_rank` (CLaMP3 layer 0 has both `mean_vec_norm = 0.91` AND
`effective_rank = 46/768`). Use both metrics together — they answer
different questions.

### "`mean_vec_norm` is small → centering is unnecessary"

True for the cone-removal use case, but `top1_sv_share` may still flag
a problematic axis that centering doesn't fix. Inspect all three of
`mean_vec_norm`, `top1_sv_share`, `effective_rank` before declaring an
encoder isotropic.

### "Comparing `mean_vec_norm` across encoders with different N"

The isotropic baseline is `1/√N`, so the headline number depends on
corpus size. For cross-encoder comparison, use `mean_vec_norm × √N`
(unit-less ratio above isotropic) instead of the raw number.

### "`mean_vec_norm = 0.85` means embeddings are useless"

No. A strong cone shifts absolute cosine values upward but preserves
relative rankings approximately. CLaMP3 achieves MAP > 0.04 on a 100k-
file retrieval task despite the heavy cone — the ranking signal lives
in the per-item residuals. Centering recovers some headroom (~10%) but
the encoder is usable as-is.

### "`top1_sv_share = 0.04 → no rank collapse`"

This is what `docs/data/omar_rq_audit.md` originally claimed (now
corrected). `top1_sv_share` measures the **first** singular value's
share. A rank-collapsed embedding can have its variance spread across
~10 directions with no single one dominating — `top1_sv_share` stays
low but `effective_rank` reveals the compression. Always check
`effective_rank / min(N, H)` for true rank diversity.

---

## Test contract

Anisotropy assertions are pinned by the following tests in
`tests/test_retrieval_metrics.py`:

| test | claim |
|---|---|
| `test_anisotropy_metrics_isotropic_baseline` | Random Gaussian embeddings land near isotropic on all four metrics |
| `test_anisotropy_metrics_cone_collapse` | Cone-aligned embeddings show `mean_vec_norm > 0.95`, `avg_pair_cos > 0.95` |
| `test_anisotropy_metrics_keys_complete` | Function returns all four documented keys |
| `test_anisotropy_metrics_degenerate_returns_nan` | N < 2 returns NaN dict, no crash |
| `test_anisotropy_metrics_deterministic_under_seed` | Same input + seed → bit-identical output |
| `test_anisotropy_mean_vec_norm_matches_theoretical_for_isotropic` | Pins `mean_vec_norm ≈ 1/√N` (catches regressions that normalise after mean, etc.) |
| `test_anisotropy_pair_cos_matches_mvn_squared` | Pins `avg_pair_cos ≈ mean_vec_norm²` to `abs=0.005` |
| `test_anisotropy_effective_rank_high_for_pure_cone_collapse` | Documents that effective_rank stays HIGH for pure cone — guards against the naive "low rank = anisotropic" intuition |

If any anisotropy interpretation breaks one of these claims, fix the
implementation, not the test.

---

## Code references

- Implementation: `marble.utils.retrieval_metrics.anisotropy_metrics`
- Production logging: `marble.tasks.Covers80.probe.CoverRetrievalTask.on_test_epoch_end`
  (around lines 460-490 — all four metrics are unconditionally logged
  as of commit `5ff913e`)
- Offline diagnostic on frame-level features:
  `scripts/diagnostics/anisotropy_diag.py`
- Tests: `tests/test_retrieval_metrics.py`, section "anisotropy_metrics"

---

## See also

- `docs/benchmarking_methodology.md` — the retrieval metric suite at
  large; this doc covers only the anisotropy subset.
- `docs/data/omar_rq_audit.md` — encoder-specific anisotropy
  investigation that motivated logging these metrics live.
- `docs/layer_analysis.md` — cross-encoder layer-sweep methodology.
