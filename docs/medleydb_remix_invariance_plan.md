# MedleyDB Remix — orchestration-invariance probe (stems)

Status: **planning — CUT/GATED by audit.** Purpose: a **retrieval** probe intended to measure
orchestration-invariance.

> **AUDIT VERDICT (see `medleydb_leitmotif_eval_strategy.md` §4): do not build as a layer selector.**
> (1) It likely **can't rank layers** — every clique variant shares **bit-identical melody-stem audio**,
> so time-pooled cosine saturates ~1.0 at *every* layer (can't keep the shared content *and* remove the
> shared audio). (2) It measures **accompaniment/mix invariance, not orchestration invariance** (the
> melody's own timbre is fixed) — not the leitmotif re-orchestration property. (3) Its whitening "free
> win" is in the **weak N<H regime** (~432 clips ≪ 768/1024 dims). (4) **Redundant** — the already-built
> **`VGMIDITVar-timbre`** delivers the real cross-instrument invariance (`map_cross_condition`,
> `condition_gap`, soundfont/reverb/loudness controls). **If still wanted:** run a **triviality go/no-go
> on the free sample first** (raw-audio MFCC/chroma baseline — if it saturates MAP, abandon) *before*
> pulling 30–50 GB of stems.

## 0. Why (and what this does / doesn't test)

Leitmotifs recur **re-orchestrated**: same melody/harmony/structure, different instrumentation and
mix. We want the encoder layer whose representation of a passage is **most stable across
orchestration changes**. Construction: from a track's **stems**, render several **remix variants** of
the *same musical content* (same notes/timing) with different orchestration/balance, then ask —
**do variants of the same passage embed close together?** A layer with high within-passage retrieval
= orchestration-invariant.

**Honest scope.** Stem remixes can vary *which accompaniment is present and its balance*, but the
**melody stem's own timbre is fixed** (we only have the one recording). So this tests invariance to
**accompaniment / mixing / instrumentation-of-the-backing** — real and leitmotif-relevant, but *not*
the full "same melody on horn vs strings" timbre swap (that needs re-synthesis; the existing
**VGMIDITVar-timbre** MIDI task is the complement that covers melody-timbre swaps). State this
limitation up front — the two together bracket the invariance question.

## 1. Framing — zero-shot retrieval (reuses CoverRetrievalTask)

- **Clique = (track, segment)**: all remix variants of the same time-segment of the same track are
  "relevant" to each other; everything else is a negative.
- **Zero-shot**: embed each variant (frozen encoder, per layer) → cosine retrieval. **No training** →
  fast, and — crucially — the **centered / whitened / ABTT metrics apply here** (retrieval is exactly
  where that machinery lives), so we get `map`, `map_centered`, `map_whitened` per layer for free.
- **Metric:** within-clique **MAP / MRR** (+ centered/whitened variants). High MAP at a layer =
  orchestration-invariant representation of that passage.

## 2. Variant generation (the real work)

Per track (metadata tells us which stem is `component: melody`):
- **Always keep the melody stem** (preserves the content/leitmotif line).
- Generate **K variants** by remixing the **accompaniment** stems differently, e.g.:
  - random accompaniment subsets (drop/keep each backing stem),
  - leave-one-instrument-family-out,
  - gain/balance perturbations (rebalance stems ±6 dB).
- Slice each variant into fixed segments (e.g. 10–15 s) aligned across variants (same time window →
  same clique).
- **Pitfall to control:** all variants share the melody-stem audio, so they're partly identical →
  retrieval could be "too easy" (shared melody dominates). Mitigations: vary accompaniment
  *aggressively*, down-weight the melody stem in some variants, and report the **lift over a trivial
  baseline** (e.g. vs the raw same-segment audio autocorrelation). This is the single biggest design
  risk — the audit must scrutinize it.

## 3. Reuse vs new

| Piece | Source | New? |
|---|---|---|
| Retrieval task + MAP/whitening metrics | `CoverRetrievalTask` (+ `retrieval_metrics.py`) | reuse |
| Zero-shot sweep infra (`--skip-fit-if-no-train`) | existing retrieval sweeps | reuse |
| Frame cache | not needed (clip-level pooled retrieval → the standard `(L,H)` cache **applies**) | reuse |
| Whitening ablations | `log_extended_retrieval_metrics: true` | reuse |
| **Remix generator** (stems → K variant wavs + clique jsonl) | **new** | build |
| **Retrieval datamodule** (clique = track+segment; `work_id` encodes it) | fork BPSMotifRetrieval-style | build |
| 6 configs + sweep registration | clone the retrieval-config pattern | clone |

Note this is **clip-level pooled** retrieval → the cheap `(L,H)` cache serves it (not the heavy frame
cache), and whitening/centering/ABTT are all live.

## 4. Data + cost

- **Needs STEMS** (the ~30–50 GB V1 multitrack download) — the one task that justifies pulling stems.
- Tracks: the 108 melody tracks (or the ~47 instrumental-melody subset, where the melody line is on a
  *swappable* instrument — arguably a purer test). Decision below.
- Storage: K variants × 108 tracks × segment wavs — modest if K~4 and we render mono 24 kHz.

## 5. Open decisions

1. **Variant strategy:** random accompaniment subsets vs leave-one-family-out vs gain perturbations
   (or a mix). How many variants K?
2. **Clique granularity:** per-segment (tight, many small cliques) vs per-track (coarse).
3. **Track set:** all 108 vs instrumental-melody subset (~47).
4. **Shared-melody-stem leakage:** how aggressively to vary accompaniment / down-weight melody so the
   task isn't trivial — needs an empirical trivial-baseline check.
5. Is this worth it vs leaning on the existing MIDI **VGMIDITVar-timbre** task for invariance? (The
   audit + brainstorm should weigh real-audio remix invariance against the MIDI analog.)
