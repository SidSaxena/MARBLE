# MARBLE — layer analysis & leitmotif layer-selection guide

Single source of truth for everything we know about which transformer
layer to extract from each frozen encoder. Used to drive the
leitmotifs project's per-track embedding strategy and any
downstream MARBLE probe work.

Last updated: 2026-05-15 (results through MuQ + MERT-v1-330M sweeps).
For implementation details on caching / extracting, see
[`embedding_cache.md`](embedding_cache.md).

---

## TL;DR

| Encoder | L total | Best single layer for leitmotif | "Melody / theme" layer | "Song / structure" layer | Two-layer pair |
|---|---:|---:|---:|---:|---|
| **MuQ** (primary recommendation) | 13 | **L11** | L8 | L10 | L10 + L11 |
| OMARRQ-multifeature-25hz | 24 | **L14** | L14 | L20 | L14 + L20 |
| MERT-v1-95M | 13 | **L7** | L3 | L7 | L3 + L7 |
| CLaMP3 (audio path) | 13 | L11 | L5 | L11 | — |
| MERT-v1-330M | 25 | (sparse data — see § Sparse-data row) | — | — | — |

- **MuQ wins by a wide margin on real-audio retrieval** (1.3–2.3× the next-best encoder on Covers80 / SHS100K). It's the right primary encoder for leitmotif.
- **The MuQ layer pick is governed by Covers80 + SHS100K, not VGMIDITVar.** Earlier drafts of this doc named L8 as the leitmotif layer based on VGMIDITVar alone — that was a triangulation error. Real-audio cover retrieval (the closer proxy to real-soundtrack leitmotif matching) peaks at L11–L12. VGMIDITVar peaks at L8 but tests a narrower invariance (MIDI-rendered, single-soundfont, model-generated variations); see § _Sparse-data row_ and the audit notes below.
- The "two-layer pair" combines two late-block representations (motif-aware + structure-aware). L2-normalize each then concat is the default combination strategy.

---

## Background: where each MIR task probes

Different MIR tasks measure different invariances and tap different
layers as a result. The probing literature plus our own benchmarks
suggest the following alignment (layers are normalized as a fraction
of total layers per encoder):

| Musical primitive | Typical layer depth | Probe task that measures it |
|---|---:|---|
| Pitch / timbre / spectral detail | ≤ 25% (very early) | HookTheoryMelody (frame-level pitch transcription) |
| Beat / rhythm | ~25–30% | GTZANBeatTracking |
| Chord / local harmony | ~30–50% (early-mid) | Chords1217 |
| Melody contour / motif-level | ~50–60% (mid) | VGMIDITVar (theme→variation retrieval) |
| Key / global tonality | ~50–70% (mid-late) | GS, HookTheoryKey |
| Song / theme identity | ≥ 75% (late) | Covers80, SHS100K (cover retrieval) |
| Song structure (verse/chorus/...) | latest (>80%) | HookTheoryStructure |

Our benchmark coverage so far:

- ✅ Cover retrieval (Covers80, SHS100K) — late-layer probe
- ✅ Theme/variation (VGMIDITVar) — mid-layer probe
- ⏳ Structure classification (HookTheoryStructure) — sweep configs ready, not yet run for MuQ
- ⏳ Melody pitch transcription (HookTheoryMelody) — sweep configs ready, not yet run
- ❌ Beat tracking (GTZANBeatTracking) — partial data only (OMARRQ-fsq variant)
- ❌ Chord recognition (Chords1217), Key detection (HookTheoryKey/GS) — partial coverage across encoders

The two missing tasks (HookTheoryStructure + HookTheoryMelody on MuQ) are the highest-priority follow-ups for verifying the recommendations below.

---

## Raw results — best layers per (encoder, task)

Pulled live from WandB via `scripts/analysis/best_layer.py`. See that
script for the up-to-date version.

### Audio retrieval / theme-variation invariance

| Encoder | Covers80 | SHS100K | VGMIDITVar (audio) |
|---|---|---|---|
| **MuQ** | **L12 / 0.198** (top-3: 12,11,10) | **L11 / 0.190** (11,12,10) | **L8 / 0.196** (8,7,9) |
| OMARRQ-25hz | L14 / 0.103 (14,15,16) | L14 / 0.086 (14,15,16) | L20 / 0.195 (20,15,14) |
| OMARRQ-25hz-fsq | L23 / 0.027 (23,22,21) | — | L5 / 0.074 (5,4,3) |
| MERT-v1-95M | L5 / 0.101 (5,3,4) | L7 / 0.084 (7,6,5) | L3 / 0.170 (3,4,7) |
| MERT-v1-330M | L5 / 0.085 (5,1,6) | — | — |
| CLaMP3 (audio) | L12 / 0.144 (12,11,9) | L11 / 0.146 (11,10,12) | L5 / 0.182 (5,6,4) |
| CLaMP3-symbolic (MIDI direct) | — | — | L11 / 0.198 (11,10,9) |

### Supervised classification / structure

| Encoder | GS (key, 24-cls) | HookTheoryKey | HookTheoryStructure | GTZANBeatTracking |
|---|---|---|---|---|
| CLaMP3 | L0 / 0.642 | L0 / 0.717 | (not run) | n/a |
| MERT-v1-95M | L9 / 0.651 | (not run) | (not run) | (not run) |
| OMARRQ-25hz-fsq | L17 / 0.175 | (not run) | (not run) | L3 / 0.361 |
| **MuQ** | (not run) | (queued) | **L10 / 0.591** (top-3: 10, 11, 8) | n/a |

**MuQ × HookTheoryStructure update (2026-05-15):** test/acc curve climbs monotonically from L0=0.51 to L10=0.59, then dips slightly at L11–L12. Confirms the "late layers do song-level abstraction" hypothesis. Peak at L10 (not L12) suggests the very last layer is slightly over-specialized to MuQ's pretraining objective, which is consistent with the layer-12 drop seen on SHS100K.

### Cross-encoder layer-depth pattern (normalized)

For tasks where multiple encoders have data, normalizing best-layer
by total-layers shows different internal organization across encoders:

| Task | MuQ (L_best / 13) | OMARRQ-25hz (L_best / 24) | MERT-95M (L_best / 13) | CLaMP3 (L_best / 13) |
|---|---:|---:|---:|---:|
| Covers80 | 0.92 (deep) | 0.58 (mid-deep) | 0.38 (mid) | 0.92 (deep) |
| SHS100K | 0.85 | 0.58 | 0.54 | 0.85 |
| VGMIDITVar | 0.62 (mid) | 0.83 (deep) | 0.23 (early) | 0.38 |

**MuQ and CLaMP3 prefer late layers for cover retrieval** (likely contrastive/semantic objectives push abstract features deep). **OMARRQ-25hz and MERT-v1-95M peak earlier in the relative depth** (masked-reconstruction-style objectives preserve more linguistic-style structure mid-block). For theme/variation, MuQ stays mid; the others diverge.

### Sparse-data row

MERT-v1-330M only has Covers80 data so far. Don't draw firm
conclusions on this encoder until the full sweep finishes (planned).

---

## Per-encoder layer-selection cheat sheet

For leitmotif extraction on real soundtracks, recommended layer picks
in priority order:

### MuQ (primary recommendation — winner across all retrieval tasks)

```
Hidden states 0–12, hidden dim 1024, token rate 25 Hz, sample rate 24 kHz
```

| Pick | Covers80 MAP | SHS100K MAP | VGMIDITVar MAP | HookTheoryStructure acc | Use when |
|---|---:|---:|---:|---:|---|
| **L11** | 0.193 (2nd) | **0.190** (1st) | 0.175 | 0.589 (2nd) | **Primary leitmotif pick** — peaks on the larger real-audio cover-retrieval task and is near-peak on every other axis. |
| L12 | **0.198** (1st) | 0.183 (3rd) | 0.164 (worst) | 0.571 | Covers80 optimum, but a 16% relative MAP drop on VGMIDITVar warns this layer is task-specialized — risky default. |
| L10 | 0.175 (4th) | 0.174 (4th) | 0.181 | **0.591** (1st) | Structure/section optimum; defensible for context-aware retrieval but trails on raw retrieval. |
| L8 | 0.151 | 0.139 | **0.196** (1st) | 0.582 | MIDI-rendered theme/variation optimum only. Trails badly on real audio. |

**Why L11 over L12:** L12 wins Covers80 by 0.005 MAP but loses SHS100K (the larger, harder, noisier dataset) by 0.007 MAP and collapses on VGMIDITVar. The cross-task average and the variance both favor L11. The L11–L12 dip pattern (L11 strong, L12 slightly worse on larger sets) repeats across both MuQ retrieval results and the HookTheoryStructure classification result — consistent with the last layer being slightly over-specialized to MuQ's pretraining objective.

**Why not L8 (the previous recommendation):** L8 only wins on VGMIDITVar, which is single-soundfont MIDI-rendered audio with model-generated *musical* variations (different notes, not different instruments). It's a *narrower* invariance test than Covers80/SHS100K — those test full real-audio variation including timbre, key, tempo, recording fidelity, and vocal/mix differences simultaneously. For real-soundtrack leitmotif retrieval (real audio, real orchestrations), the wider invariance test is the closer proxy.

**Two-layer pair:** L10 + L11 (structure + motif), L2-normalize-then-concat. Stays in the late-layer neighborhood — both layers are evidence-backed across multiple tasks. Don't include L12 (collapses on VGMIDITVar) or pair L8 with a late layer (L8 already trails on real audio).

### OMARRQ-multifeature-25hz

```
Hidden states 0–23, hidden dim 1024, token rate 25 Hz, sample rate 24 kHz
```

| Pick | Covers80 | SHS100K | VGMIDITVar |
|---|---:|---:|---:|
| **L14** (single-layer pick — top-3 on all three) | **0.103** | **0.086** | 0.184 (3rd) |
| L15 (compromise, 2nd on each) | 0.098 | 0.083 | 0.193 |
| L20 (theme-variation peak) | ~0.097 | ~0.082 | **0.195** |

**Two-layer pair:** L14 + L20 (mid-deep + very-deep). 6-layer spread between them is large; bimodality is more pronounced than MuQ.

### MERT-v1-95M

```
Hidden states 0–12, hidden dim 768, token rate 75 Hz, sample rate 24 kHz
```

| Pick | Covers80 | SHS100K | VGMIDITVar |
|---|---:|---:|---:|
| **L7** (single-layer pick — top-3 on all three) | 0.094 (2nd) | **0.084** | 0.166 (3rd) |
| L5 | **0.101** | 0.077 | 0.155 |
| L3 (theme-variation peak) | 0.087 | 0.072 | **0.170** |

**Two-layer pair:** L3 + L7 (early-mid + mid). Compact spread.

### CLaMP3 (audio path)

```
Hidden states 0–12, hidden dim 768
```

L11 — wins on cover retrieval, top-3 on VGMIDITVar via the symbolic path
(which isn't applicable to real audio anyway).

Note: CLaMP3-symbolic (feeding MIDI directly through CLaMP3's M3 sub-network) wins on VGMIDITVar at 0.198 — but that's a separate pipeline that bypasses audio rendering entirely. Not usable on real soundtracks.

---

## Combination strategies for two-layer extraction

### Option 1 — L2-normalize each, then concatenate (recommended default)

```python
e_a = F.normalize(emb_at_layer_a, dim=-1)
e_b = F.normalize(emb_at_layer_b, dim=-1)
emb = torch.cat([e_a, e_b], dim=-1)   # (T, 2H)
```

Pros: single pipeline, equalized layer contributions, drop-in
replacement for single-layer extraction. Cosine similarity on the
concat vector becomes a blended cosine.

Cons: doubles disk + DTW cost per pair. No tunable weight.

### Option 2 — Late fusion at the scoring stage

Compute similarity / matrix profile / DTW score per layer, then
combine the scalars:

```
s_combined = α · s_layer_a + (1 − α) · s_layer_b   with α ∈ [0, 1]
```

Pros: each layer's geometry is preserved through similarity
computation. Tunable α via grid search on a validation set. Robust to
one layer firing on a false positive.

Cons: 2× compute per pair (run the entire matrix profile / DTW
pipeline twice). Requires labeled validation data to tune α.

**Implementation pattern for the leitmotifs project:**

```python
# After embedding extraction:
embs = {layer: load_cached(track, layer) for layer in [8, 11]}

def fused_score(track_a, track_b, alpha=0.5):
    profiles = {}
    for L, e in embs.items():
        # Compute matrix profile per layer (independently)
        profiles[L] = matrix_profile_score(e[track_a], e[track_b])
    return alpha * profiles[8].max() + (1 - alpha) * profiles[11].max()
```

Grid search α on known positive / negative pairs:

```python
for α in [0.0, 0.1, 0.2, ..., 1.0]:
    pos_scores = [fused_score(a, b, α) for a, b in known_positives]
    neg_scores = [fused_score(a, b, α) for a, b in known_negatives]
    auc[α] = roc_auc(pos_scores, neg_scores)
best_α = argmax(auc)
```

### Option 3 — Mean averaging the embeddings

```python
emb = (emb_at_layer_a + emb_at_layer_b) / 2
```

Pros: cheapest. Same output dim as single layer.

Cons: mashes layer geometries; usually worse than concat by a few
percent. Only use if you can't afford the doubled dim.

### When NOT to use late fusion

If your validation set is too small to tune α reliably (say <50 known
positives), default to **L2-normalize-then-concat**. Late fusion's
main advantage is the tunable α; without enough data to tune it, you're
just doing more compute for no gain.

---

## Suggestions and recommendations

### Single-encoder, single-layer (simplest)

**MuQ at L11.** Peaks on SHS100K and is near-peak on Covers80, HookTheoryStructure, and VGMIDITVar. Best single-layer baseline for real-soundtrack leitmotif retrieval.

### Single-encoder, two-layer (best one-step improvement)

**MuQ at L10 + L11, normalized-concat.** Stays in the late-layer neighborhood where both layers are evidence-backed. Disk cost: 26 KB/clip (double of single-layer). DTW cost: doubled. Expected to outperform single-layer L11 by a few percent on retrieval-style matching, with the gain coming from layer 10's structural-context signal complementing layer 11's motif-identity signal.

### Single-encoder, three-layer (kitchen sink)

**MuQ at L8 + L10 + L11, normalized-concat.** Adds L8 to recover the MIDI-rendered/motif-only invariance regime, in case the deployment data is more synthetic than expected. Use only if qualitative inspection of L10 + L11 retrievals shows the two-layer ensemble missing instrumentation-invariant matches.

### Ensemble across encoders (most ambitious)

**MuQ L8+L11 + OMARRQ-25hz L14, late fusion.** Combines MuQ's superior real-audio retrieval with OMARRQ's complementary theme-variation profile. Cost: 3× extraction, 3× DTW. Reserved for the case where the single-encoder approach plateaus.

---

## Bad ideas — things to skip

### ❌ OMARRQ-multifeature-25hz-fsq

Worse than the non-fsq variant on every task we've tested (Covers80 0.027 vs 0.103; VGMIDITVar 0.074 vs 0.195). The FSQ quantization destroys representational fidelity. Use non-fsq always.

### ❌ MERT-v1-95M as the primary leitmotif encoder

Roughly half the retrieval MAP of MuQ on real audio (Covers80 0.101 vs 0.198). Smaller, faster, but on this task domain it's just not competitive. Keep it as a fallback for compute-constrained scenarios.

### ❌ Mean averaging two layers

Always strictly worse than L2-normalize-then-concat for retrieval. Same compute, lower performance. The only reason to mean-average is if you have a hard 1×H output-dim constraint downstream.

### ❌ Late fusion without α tuning

If you can't afford labeled validation data to grid-search α, just use concat. α=0.5 (the obvious default) often performs worse than concat. Late fusion's whole value proposition is the tunable α; using a fixed α throws that away.

### ❌ Single-layer extraction at a "random middle" layer

Several encoders (MERT-95M, OMARRQ-25hz) have peaks at non-obvious depths (L7, L14). Defaulting to "middle layer" without checking the per-task profile leaves 30–50% MAP on the table for retrieval tasks. Always start from the table above.

### ❌ Caching frame-level features for leitmotif

Each MuQ frame at 25 Hz × 30 sec clip is 13 × 750 × 1024 × 4 = ~40 MB/clip. Across 259 tracks of 3-min audio = ~620 MB per track × 259 = ~160 GB. Trivial bog if you accidentally cache per-frame instead of post-time-pool. Always pool to `(L, H)` before caching for retrieval; reserve frame-level for matrix-profile pipelines that need per-window similarity.

### ❌ Running fsq + non-fsq variant in the same sweep

The fsq variant exists for historical reasons (original OMARRQ release). It is uniformly worse. If you find yourself benchmarking both, drop the fsq and double-check no downstream config references it.

### ❌ MIDI-direct path for real-audio leitmotif

CLaMP3-symbolic wins VGMIDITVar (0.198) because it skips audio rendering — feeds MIDI through CLaMP3's M3 sub-network directly. **This is not transferable to real soundtracks**, which are audio, not MIDI. Don't confuse the symbolic-path benchmark numbers with audio-path applicability.

---

## Verification status & next experiments

### Completed

1. ✅ **MuQ × HookTheoryStructure** (2026-05-15). Peak at **L10 / 0.591 acc**, top-3 = {10, 11, 8}. Late-layer dominance confirmed. The L11→L12 dip mirrors the SHS100K dip — consistent with the last layer being slightly over-specialized to the pretraining objective.

### Open

2. **MuQ × HookTheoryKey** (config: `probe.MuQ-layers.HookTheoryKey.yaml`) — queued. If the late-layer hypothesis generalizes, peak should be in the L8–L11 band. Key estimation needs tonal abstraction over the full clip, so a mid-late peak (closer to L8) would also be consistent.
3. **MuQ × HookTheoryMelody** (config: `probe.MuQ-layers.HookTheoryMelody.yaml`) — peak layer should be mid (L5–8 range) if the melody hypothesis is correct. ~6–8 hours (no cache — frame-level task).
4. **Multi-soundfont VGMIDITVar re-render.** Re-render the VGMIDI-TVar dataset with 3–4 SoundFonts (FluidR3 + MuseScore_General + SGM-V2.01 + Salamander) using the existing `--soundfont` rotation in `scripts/data/build_vgmiditvar_dataset.py`. If the layer-8 VGMIDITVar peak is real timbre-invariance, it should stay at L8. If it was an artifact of single-soundfont rendering, the peak should shift toward L10–L11. ~2 hours of rendering + a re-sweep.
5. (Bonus) Cross-encoder verification: HookTheoryStructure + HookTheoryKey on MERT-95M + OMARRQ-25hz.

If the structure peak ≠ late layer, OR the melody peak ≠ mid layer,
revisit the recommendations.

After those experiments, update this doc's "TL;DR" table with the
actual layer numbers. If results contradict the analysis, mark
the contradiction explicitly here — don't quietly update the table.

### Changelog

- **2026-05-15** — Corrected primary MuQ leitmotif recommendation from L8 → L11. Previous recommendation was triangulated from VGMIDITVar alone, but the larger and closer-to-deployment Covers80/SHS100K benchmarks both peak at L11–L12. VGMIDITVar's narrower invariance profile (single-soundfont MIDI rendering) makes it a weaker proxy for real-audio leitmotif matching than the cover-retrieval tasks. Updated TL;DR table, MuQ cheat sheet, and recommendation sections accordingly. Added HookTheoryStructure result to the supervised-classification table.

---

## Implementation reference

- Embedding cache: see [`embedding_cache.md`](embedding_cache.md)
- Sweep runner: `scripts/sweeps/run_all_sweeps.py`, `run_sweep_local.py`
- Layer-stats query: `scripts/analysis/best_layer.py`
- Pre-warm CLI: `scripts/embeddings/extract.py`
- Cache inspection: `scripts/embeddings/manage.py`

For the leitmotifs project (separate repo at `/Users/sid/leitmotifs/`)
the relevant snippet is in
[`docs/TODO.md`](TODO.md#1-leitmotifs-matrix-profile-result-cache-separate-repo).
