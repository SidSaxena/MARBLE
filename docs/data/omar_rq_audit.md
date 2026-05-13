# OMAR-RQ integration audit (2026-05-14)

Live audit of MARBLE's OMAR-RQ encoder vs upstream `omar_rq` package
and the OMAR-RQ paper (arXiv:2507.03482), prompted by probe results
running 2–5× below the paper's reported benchmarks.

| Task | Our best layer | Our metric | Paper benchmark | Ratio |
|---|---:|---:|---:|---:|
| Beat tracking (GTZAN-BT, F1) | 0.361 (4/24 layers) | beat_f1 | 0.855 | **0.42×** |
| Key estimation (GS, weighted_score) | 0.175 | weighted_score | n/a (different task) | n/a |
| Retrieval (Covers80, MAP) | 0.027 (24 layers) | MAP | n/a | n/a (MERT gets 0.10) |

The Beat tracking gap is the cleanest apples-to-apples comparison.

---

## Verified-correct (audit passed)

Step 1: `scripts/diagnostics/probe_omarrq_shapes.py` confirmed via live
forward pass on 8 real audio clips (5 s @ 24 kHz mono):

| Property | Expected | Observed | Status |
|---|---|---|---|
| Output tuple length | 24 | 24 | ✓ |
| Per-layer shape | (B, T, 1024) | (8, 125, 1024) | ✓ |
| Token count for 5 s | 125 (= 5 × 25 Hz) | 125 | ✓ |
| Feature dim | 1024 | 1024 | ✓ |
| Per-layer mean | ≈ 0 (LayerNorm output) | 0.0000 across all 24 | ✓ |
| Per-layer std | ≈ 1.0 | 1.0000 across all 24 | ✓ |
| Per-frame L2 norm | √1024 ≈ 32.0 | 31.9998 across all 24 | ✓ |
| Layer ordering | input → output | cos(L0, Lₖ) drops monotonically 1.0 → 0.93 | ✓ |
| Sample rate | 24 kHz per `config.gin` | dataloader resamples to 24 kHz | ✓ |
| Pretrained weights | 792 net + 2 embedding | logs confirm | ✓ |

The "stale 512 comment" at `model.py:137` was a red herring — it's a
docstring typo, not a runtime bug.

**Conclusion**: `marble/encoders/OMAR_RQ/model.py` is functionally correct.

---

## Diagnostic finding: mild cone-effect anisotropy

Step 2: `scripts/diagnostics/anisotropy_diag.py` on 16 leitmotif clips
(5 s each = 2000 frames per layer):

| Layer | avg_pair_cos (raw) | mean_vec_norm | effective_rank | avg_pair_cos (centered) | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 0.208 | 0.457 | 110 | 0.0007 | MILD anisotropy |
| 6 | 0.231 | 0.482 | 118 | 0.0006 | MILD anisotropy |
| 12 | 0.244 | 0.494 | 124 | 0.0006 | MILD anisotropy |
| 18 | 0.265 | 0.516 | 130 | 0.0005 | cone — fixable by centering |
| 23 | 0.275 | 0.525 | 136 | 0.0006 | cone — fixable by centering |

Isotropic baseline for 1024-D: `1/√1024 ≈ 0.031`. Our avg_pair_cos
runs 0.21–0.28 — **~7–9× above isotropic**, comfortably "anisotropic"
in the literature sense but not "severe".

After centering (subtracting the corpus mean): collapses to ~0.0006 →
the anisotropy is a pure shift (cone effect), not rank collapse. Top-1
singular value share is ~0.04 (no dominant direction); effective rank
is ~110–135 out of 1024 (10–13% of full dim, moderate compression).

**What this explains:**
- Retrieval results (Covers80 MAP 0.027). Cosine similarity on cone-
  shaped embeddings squashes the discrimination range — all pairs end
  up looking artificially similar. MERT's identical pipeline gets
  0.10 on Covers80 (4× better), consistent with MERT being more
  isotropic (`/Users/sid/leitmotifs/anisotropy_diagnosis.json` showed
  avg_cos ≈ 0.22, similar — though MERT shows it from frame-level
  embeddings, which we average before cosine).

**What this does NOT explain:**
- Classification results (GS 0.175, GTZAN-BT 0.36). An MLP with bias
  terms is invariant to a mean shift in the input. Centering won't
  help classification.

So **centering would help retrieval but not classification** —
mathematically. The 2–5× gap on classification has a different cause.

---

## Critique of the existing anisotropy script

The user's original `/Users/sid/leitmotifs/scripts/diagnose_anisotropy.py`
implements the standard Ethayarajh-style cone metrics correctly. The
new `scripts/diagnostics/anisotropy_diag.py` extends it with:

| Added | Why |
|---|---|
| Top-1 singular value share | Rank-collapse detector (cone metric misses this) |
| Effective rank `exp(H(σ²))` | Continuous "intrinsic dimensionality" measure |
| Isotropic baseline `1/√C` | Replaces the arbitrary 0.4/0.5 threshold |
| Built-in encoder loading | No need to pre-dump `.pt` files; works on any MARBLE encoder |

The original script's main limitations:
1. Arbitrary thresholds (`avg_cos > 0.4 or mean_norm > 0.5`) — should
   be relative to dim (1/√C) and corpus size (1/√N).
2. Missing rank-collapse diagnostic.
3. `replace=True` in random sampling — mild bias from self-correlations.
4. Assumes pre-saved `.pt` files in a specific layout — works for the
   leitmotif project's frame dumps, but not for ad-hoc encoder probing.

For deciding "is this encoder usable", the original script's MuQ +
MERT outputs (both showing `is_anisotropic: false`) are reliable. Our
verdict for OMAR-RQ: **mild anisotropy, mostly the cone effect**.

---

## Open hypotheses (not yet investigated)

The verified-correct parts above narrow the search. Remaining
candidates for the 2–5× gap:

1. **Probe protocol mismatch** (HIGHEST LIKELIHOOD).
   - Paper Beat F1 = 0.855 is from THEIR probe protocol. Upstream
     `omar_rq/probe/data/nsynth_pitch.py:133` shows multi-layer
     aggregation (`mean(dim=0)` over all 24 layers) — MARBLE picks ONE
     layer per run. If the paper used multi-layer mean, our best-of-24
     is artificially low.
   - Paper probe head might be deeper / wider. Need to read the paper
     supplementary or `probe/modules/sequence_classifiers.py` config.

2. **Different metric or post-processing.**
   - Beat tracking F1 often uses a tolerance window (e.g. ±70 ms in
     mir_eval). MARBLE's `BeatF1Score(tol=0.07)` matches the convention.
   - But paper might apply HMM/CRF dynamic-programming post-processing
     on the frame-level activations, which can ~2× F1 over raw
     thresholding.

3. **Multi-layer aggregation as the canonical eval.**
   - Adding `LayerWeightedSum` or `LayerMean` to MARBLE's emb_transforms
     stack is the natural experiment.

4. **A different OMAR-RQ variant.**
   - We use `multifeature-25hz-fsq`. The paper's Beat F1 = 0.855 is for
     this variant. But the README shows `multifeature` (no -25hz)
     reaches 0.833 — possibly more robust to our probe protocol due to
     lower frame rate (18.75 Hz vs 25 Hz = fewer tokens to learn from).

5. **Audio scaling / normalization assumptions.**
   - Upstream `Waveform.norm_mean = None` confirms no input normalization
     expected. Our audio is in [-1, 1], matches PyTorch default. Unlikely
     to be the issue.

---

## Concrete next steps (in expected-payoff order)

1. **Implement multi-layer mean aggregation** in `LayerSelector` (extend
   to support `mean` mode over a list of layer indices). Re-run GS and
   GTZAN-BT with `layers=[6,12,18,23], mode='mean'`. **Highest
   expected payoff** — matches upstream eval convention.

2. **Try `mtg-upf/omar-rq-multifeature`** (no -25hz, no -fsq) as a
   model_id parameter to `OMARRQ_Multifeature25hz_Encoder`. The class
   name lies but the wrapper accepts arbitrary `model_id`. If results
   improve, we've isolated the issue to the -25hz variant.

3. **Apply centering** at encoder output for retrieval tasks specifically
   (Covers80, SHS100K, VGMIDITVar). New `EncoderOutputCentering`
   emb_transform that L2-normalizes after subtracting a precomputed
   corpus mean. Expected: Covers80 MAP from 0.027 toward 0.10.

4. **Read the OMAR-RQ paper supplementary** (arXiv:2507.03482) for the
   exact probe head config and training schedule. If they use a much
   larger probe than ours, port that.

5. **Defer**: training a larger probe head end-to-end. This is a
   pipeline change, not a fix; do this only if step 1+2+3 confirm
   the integration is otherwise correct.

---

## Files referenced

| File | Purpose |
|---|---|
| `marble/encoders/OMAR_RQ/model.py` | Encoder wrapper (audited — functionally correct) |
| `scripts/diagnostics/probe_omarrq_shapes.py` | Step 1 shape + statistics audit |
| `scripts/diagnostics/anisotropy_diag.py` | Step 2 isotropy + rank-collapse diagnostic |
| `.venv/lib/python3.10/site-packages/omar_rq/modules/maskingmodel.py:439` | Upstream `extract_embeddings` |
| `.venv/lib/python3.10/site-packages/omar_rq/probe/data/nsynth_pitch.py:133` | Upstream probe with multi-layer aggregation |
| `~/.cache/huggingface/hub/models--mtg-upf--omar-rq-multifeature-25hz-fsq/.../config.gin` | Model's training-time gin config |
| `/Users/sid/leitmotifs/scripts/diagnose_anisotropy.py` | The user's original anisotropy script (preserved as-is) |
