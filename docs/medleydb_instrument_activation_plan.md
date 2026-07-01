# MedleyDB Instrument Activation — dataset breakdown + probing plan

Status: **planning — revised by audit**. Purpose: a complementary **instrument-identity** probe to
pair with the MedleyDB melody probe as a leitmotif layer *diagnostic* (not selector).

> **AUDIT VERDICT (see `medleydb_leitmotif_eval_strategy.md` §3):** keep, but **demote to a descriptive
> diagnostic**. Required corrections: (1) **taxonomy bug** — MedleyDB's native top level is **6**
> families (Brass is nested under Winds); this doc's "7 families" listed only 6 and omitted Brass →
> use a **custom ~7-way grouping that breaks Brass out** (horn calls = most leitmotif-relevant).
> (2) **Add `pos_weight`/focal loss + per-family AP** — MTGInstrument is clip-level, unweighted-BCE,
> macro-only, so the reuse below is *overstated* (frame metrics + per-class AP are new code) and plain
> BCE would learn "always-off" on the rare orchestral families. (3) **Pop/vocal skew guts the
> orchestral axis** — Brass/Strings/Winds are the rarest/noisiest (rare-class variance ∝ *track* count,
> ~1/fold) → report orchestral vs pop families separately; consider V2 for more brass/wind frames.
> (4) Prefer a **linear** probe over a trained MLP head for a cleaner "where does timbre live" read.

## 0. Why (the leitmotif rationale)

Leitmotifs recur **re-orchestrated** — the same motif on horn vs strings vs harp. The representation
we want for leitmotif discovery is **melodically discriminative but relatively timbre-invariant**.
The melody probe (RPA) tells us where *melodic* content lives; instrument-activation probing tells us
where *instrument/timbre* content lives. Plotting both layer curves reveals the sweet spot: a layer
**high on melody, comparatively low on instrument-ID** = the most orchestration-invariant melodic
layer. (This is a **hypothesis**, not established — treat as an informative contrast, validate against
the real `LeitmotifDetection` task.)

## 1. The annotations

- **File:** `Annotations/Instrument_Activations/ACTIVATION_CONF/<Track>_ACTIVATION_CONF.lab`
- **Format:** CSV with header `time,S01,S02,…,SNN` — one row per frame, columns = per-**stem**
  activation *confidence* in [0,1]. (Verified on the sample: `LizNelson_Rainfall` = 6,136 frames × 5
  stems, spanning 284.9 s.)
- **Frame rate:** hop = 2048/44100 = **46.4 ms (~21.5 fps)** — coarser than melody's 5.8 ms.
- **Confidence → active:** continuous [0,1]; MedleyDB's standard "active" threshold is **0.5** (the
  activation confidence is a smoothed energy envelope per stem).
- **Coverage: all 196 tracks** (V1 122 + V2 74) — unlike melody (108, V1-only). **So V2 audio *is*
  usable here** if we want the larger corpus.

### Stems → instruments
Activation columns are per-**stem** (`S01`…). Each stem's instrument label lives in
`<Track>_METADATA.yaml` under `stems[].instrument` (e.g. `female singer`, `acoustic guitar`). A track
can have several stems of the same instrument → we **aggregate stems by instrument** (a frame is
"instrument active" if *any* of its stems is active).

### Taxonomy (the label space)
MedleyDB ships a 3-tier taxonomy: **7 families** (Strings, Winds, Voices, Percussion, Electric,
Other) → performance/construction sub-groups → **~110 leaf instruments**. Label-space options:

| Option | #classes | Pros | Cons |
|---|---|---|---|
| Leaf instruments | ~110 | most specific | very sparse, long tail, imbalanced |
| Mid-level groups | ~15–20 | balanced-ish | some arbitrariness |
| **Families** | **7** | balanced, robust, most relevant to *orchestration* | coarse |
| Top-K leaf by freq | ~20 | covers the common instruments | drops rare ones |

**Recommendation:** **instrument families (7)** as the primary label space — for the timbre-invariance
question, family (strings vs winds vs voice…) is exactly the orchestration axis, and it's balanced
enough to train cleanly on 108 tracks. Optionally also run a **top-~20 leaf** variant for a finer view.

## 2. Task framing — frame-level multi-label

- **Prediction:** per-frame **multi-hot** over the vocabulary (multiple instruments active at once).
- **Label build:** per stem, threshold conf > 0.5 (or keep raw conf as a **soft target** — see §5) →
  aggregate stems → instrument/family → resample from 21.5 fps to the encoder token rate
  (`label_freq` 25/75) by nearest → multi-hot vector per frame. Reuses the melody datamodule's
  clip-slicing + nearest-resample machinery; **no `-1` silence mask** (an all-zeros frame — nothing
  active — is a valid target, unlike melody).
- **Decoder:** `MLPDecoderKeepTime`, `out_dim = n_classes`, no output activation.
- **Loss:** `torch.nn.BCEWithLogitsLoss` (multi-label). Reused from **MTGInstrument** (the repo's
  existing multi-label instrument-tagging task).
- **Metrics (frame-flattened):** macro/micro **F1** + **mAP** (`torchmetrics.AveragePrecision`),
  plus **per-class AP** so we can read the family-wise curve. Also cribbed from MTGInstrument.

## 3. What we reuse (small new surface)

| Piece | Source | New? |
|---|---|---|
| Audio/clip machinery | `BaseAudioDataset` | reuse |
| Artist-conditional 5-fold split | `marble/tasks/MedleyDBMelody/split.py` | reuse |
| Frame `(L,T,H)` cache + pre-warm | `emb_cache.py` + fixed `extract.py` | reuse |
| Multi-label loss + metrics | `MTGInstrument` (BCEWithLogits, F1, AveragePrecision) | reuse |
| Fold wrapper / sweep registration | clone `run_medleydb_melody_folds.sh` + `run_all_sweeps` | clone |
| **Label builder** (stems→multi-hot/frame) | **new** | build |
| **Probe** (keep-time multi-label head + frame metrics) | **new** (thin) | build |
| **JSONL builder** (adds `activation_lab` + `metadata_yaml` paths) | **new** (fork melody builder) | build |

**Cache note:** on the same clips + transforms, the encoder features are **identical** to the melody
task — only the labels differ. The cache key is task-scoped (`… / MedleyDBInstrument`) so it re-warms
by default; if we want, we can share one cache across both probes by keying on the shared
(encoder, sr, clip, pipeline, pool_time) tuple. Minor; decide later.

## 4. Data + split

- **Recommended: the same 108 melody tracks (V1)** so the instrument and melody layer curves are on
  the *same tracks/clips/folds* → directly comparable (the whole point of the contrast).
- **Alternative: all 196 (V1+V2)** for a stronger instrument probe — but then not clip-aligned with
  melody, and needs V2 audio staged. Decision below.
- **Split:** same artist-conditional 5-fold; same fold wrapper convention (`--run-name-suffix foldF`,
  `LogSweepCoordsCallback`).

## 5. Decisions (locked)

1. **Label space: 7 instrument families** (Strings, Winds, Voices, Percussion, Electric, Other).
2. **Tracks: the same 108 melody tracks (V1)** — melody-vs-instrument curves directly comparable.
3. **Targets: soft** — raw activation confidence [0,1] fed to BCE. Per-family soft target =
   **max** over that family's stems' confidences (the soft analog of "any stem active"), resampled
   to the token grid. **Loss uses the soft [0,1] target; metrics (F1/mAP) threshold at 0.5** for a
   binary ground truth (`AveragePrecision`/`F1Score` need binary targets). Frame-rate mismatch
   (21.5 fps → 25/75) handled by nearest upsampling.

## 6. Build breakdown (mirrors the melody task)

New `marble/tasks/MedleyDBInstrument/`:
- **`family_labels.py`** (pure, TDD) — `STEM_INSTRUMENT → FAMILY` map (from the taxonomy),
  `stem_conf_matrix + stem→family → per-frame (F,)-family soft matrix` (max-aggregate), and
  `clip_family_labels(track_family_conf, native_rate, clip_start, label_freq, label_len)` (nearest
  resample → `(label_len, 7)` float target). Analogous to `melody_labels.py`.
- **`datamodule.py`** — fork `MedleyDBMelody` datamodule; `get_targets` returns the `(T, 7)` soft
  matrix; JSONL carries `activation_lab` + `metadata_yaml` (for stem→instrument). Same
  fold/template/`fold_idx` wiring.
- **`probe.py`** — thin: `MLPDecoderKeepTime(out_dim=7)`, `BCEWithLogitsLoss`, and frame-level
  multi-label `F1`/`AveragePrecision` (crib `MTGInstrument`), keep-time (flatten frames as samples).
- **`scripts/data/build_medleydb_instrument_jsonl.py`** — fork the melody builder; discover tracks
  with MIX + ACTIVATION_CONF + METADATA; reuse `fold_split` on the same 108 tracks (**same seed** so
  folds align with melody).
- **6 configs** (`probe.{MuQ,MERT-v1-95M,OMARRQ}-{layers,meanall}.MedleyDBInstrument.yaml`) — fork
  melody configs; `out_dim: 7`, monitor `val/mAP` (or `val/macro_f1`), rest identical.
- **`run_all_sweeps` SweepDef + `run_medleydb_instrument_folds.sh`** — clone.
- Tests: `test_medleydb_family_labels.py` (+ datamodule) — TDD the pure label core against a
  synthetic stem-confidence matrix and the real sample `.lab`.

Reuse `fold_split`, `BaseAudioDataset`, the frame cache + fixed `extract.py`, `LogSweepCoordsCallback`.
