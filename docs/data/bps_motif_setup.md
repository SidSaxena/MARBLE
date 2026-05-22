# BPS-Motif — data setup runbook

End-to-end commands for getting the BPS-Motif dataset (leitmotif
annotations on the first movements of all 32 Beethoven piano sonatas)
onto a machine + (optionally) to Modal.

## Source

- Upstream repo: <https://github.com/Wiilly07/Beethoven_motif>
- Paper: Hsiao, Hung, Chen, Su — *BPS-Motif: A Dataset for Repeated Pattern
  Discovery of Polyphonic Symbolic Music*, ISMIR 2023.
  ([archives.ismir.net/ismir2023/paper/000032.pdf](https://archives.ismir.net/ismir2023/paper/000032.pdf))
- License: CC-BY-4.0 (per the Zenodo record at
  <https://zenodo.org/records/10265277>). The GitHub repo itself does not
  ship a `LICENSE` file; we treat it as research-use-with-attribution.
- Scale: 32 movements, ~127k notes, 263 distinct motifs, 4,944 motif
  occurrences. **Symbolic only** — the dataset ships CSVs + motif-only
  MIDIs at 60 QPM, NOT full-movement MIDIs or audio.

## Prerequisites

```bash
# 1. git (for cloning the upstream repo) — usually pre-installed
# 2. pretty_midi (for MIDI synthesis + slicing) — already a project dep
# 3. Disk budget:
#    - Upstream clone: ~230 MB
#    - Synthesised + sliced output: ~52 MB
#    Total: ~290 MB
```

## Step 1 — Build locally

```bash
uv run python scripts/data/build_bps_motif_dataset.py
```

Outputs (under `data/BPS-Motif/`):

- `_upstream/Beethoven_motif/` — the cloned annotation repo (CSVs + motif MIDIs).
- `midi/<id>-1.mid` — 32 full-movement MIDIs synthesised at 60 QPM from `csv_notes/`.
- `midi_windows/<piece_id>__<letter>__<idx>.mid` — per-window sliced MIDIs (positives = motif spans, negatives = sampled non-motif spans).
- `BPSMotif.MNID.fold{0..4}.{train,val,test}.jsonl` — 5-fold CV splits for MNID (window-level binary classification: motif vs. non-motif). ~5,500 records per fold.
- `BPSMotif.Retrieval.fold{0..4}.{train,val,test}.jsonl` — 5-fold CV splits for retrieval. Positives only. ~4,900 records per fold.

Pilot on a few movements first to verify the build:

```bash
uv run python scripts/data/build_bps_motif_dataset.py --max-movements 4
```

## Step 2 — Run sweeps

```bash
# MNID — clip-level binary classification (supervised, max_epochs=40)
uv run python cli.py fit  -c configs/probe.CLaMP3-symbolic-meanall.BPSMotifMNID.yaml
uv run python cli.py test -c configs/probe.CLaMP3-symbolic-meanall.BPSMotifMNID.yaml

# Retrieval — within-piece within-letter motif retrieval (zero-shot, max_epochs=0)
uv run python cli.py test -c configs/probe.CLaMP3-symbolic-meanall.BPSMotifRetrieval.yaml

# Or full layer sweep across all 13 CLaMP3-symbolic layers
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-symbolic-layers.BPSMotifMNID.yaml \
    --num-layers  13 \
    --model-tag   CLaMP3-symbolic \
    --task-tag    BPSMotifMNID
```

### 5-fold cross-validation

The configs default to `fold_idx=0`. To run the other 4 folds, override
the dataset's `fold_idx` per stage:

```bash
for fold in 0 1 2 3 4; do
  uv run python cli.py fit -c configs/probe.CLaMP3-symbolic-meanall.BPSMotifMNID.yaml \
    --data.init_args.train.init_args.fold_idx=$fold \
    --data.init_args.val.init_args.fold_idx=$fold \
    --data.init_args.test.init_args.fold_idx=$fold
  uv run python cli.py test -c configs/probe.CLaMP3-symbolic-meanall.BPSMotifMNID.yaml \
    --data.init_args.test.init_args.fold_idx=$fold
done
```

The reportable headline is the macro-average F1 across the 5 folds.

## Step 3 (optional) — Upload to Modal (`marble-data` volume)

Two paths. **Either** build directly on Modal (cheaper for credits):

```bash
modal run modal_marble.py::setup_bps_motif
```

This runs the same `build_bps_motif_dataset.py` server-side, writing to
`marble-data:/BPS-Motif/`. ~30 s wall-clock for the full 32-movement
build.

**Or** upload the locally-built data:

```bash
modal volume put marble-data data/BPS-Motif/midi BPS-Motif/midi
modal volume put marble-data data/BPS-Motif/midi_windows BPS-Motif/midi_windows
for probe in MNID Retrieval; do
  for fold in 0 1 2 3 4; do
    for split in train val test; do
      modal volume put marble-data \
        data/BPS-Motif/BPSMotif.${probe}.fold${fold}.${split}.jsonl \
        BPS-Motif/BPSMotif.${probe}.fold${fold}.${split}.jsonl
    done
  done
done
```

## Probe design — important caveats

### MNID is window-level, NOT per-note like the literature

Hsiao TISMIR'24 reports per-NOTE F1=0.721 using MidiBERT-Piano on
BPS-Motif. That requires per-note alignment of encoder embeddings to
score-position notes — feasible with CP-tokenised MidiBERT but fragile
with CLaMP3's M3 patches (which tokenise at the bar level via MTF, and
bar boundaries don't always survive midi_to_mtf cleanly across
Beethoven's varying time sigs: 2/2, 2/4, 4/4, 6/8, 3/4, 12/8, 3/8).

**v1 of MNID is window-level**: each clip is either a positive (the MIDI
slice of a motif occurrence) or a negative (an equal-length sliced MIDI
from a non-motif region of the same movement). The model predicts
"contains a motif?" per clip. F1 is reported on this binary task and is
NOT directly comparable to Hsiao's per-note F1. A per-note v2 can layer
on top once we have v1 results.

### Retrieval is within-piece within-letter

Each motif window has a `(piece_id, motif_letter)` tuple. We encode
this jointly into the `work_id` integer that `CoverRetrievalTask` uses
as the relevance key. Same `(piece_id, motif_letter)` → relevant;
anything else (different piece OR different letter) → not relevant.

Motif letters are MOVEMENT-LOCAL in BPS-Motif — motif "a" in Op.2 No.1
is unrelated to motif "a" in Op.2 No.2. The joint encoding handles this
correctly.

## Audio variant — DEFERRED

The audio path (probing MERT/MuQ/OMAR-RQ on Beethoven sonata recordings)
is a separate follow-up. Requirements:

1. The user sources **real recordings** of all 32 first movements (do
   NOT synthesise from these MIDIs — interpretive tempo won't survive).
2. Align each recording to the score-time MIDI via DTW (one-time per
   movement; or, if BPS-FH's existing audio-aligned annotations cover
   the same recordings, reuse those).
3. Augment the build script with `--audio-source-dir` and per-occurrence
   `audio_path` fields, mirroring the SuperMarioStructure pattern at
   [scripts/data/build_supermario_dataset.py](../../scripts/data/build_supermario_dataset.py).
4. Add audio probe configs:
   `probe.MERT-v1-95M-{layers,meanall}.BPSMotifMNID.yaml` etc.

Out of scope for the symbolic v1 release; flagged here so the
audio-extension trajectory is clear.

## SOTA reference numbers (symbolic, literature)

- **MidiBERT-Piano + pseudo-labels** (Hsiao TISMIR'24 §5.3,
  <https://transactions.ismir.net/articles/10.5334/tismir.250>):
  per-note motif identification — acc 0.839 / F1 0.721 / P 0.701 / R 0.763.
- Heuristic baselines on the upstream RPD task (CSA): Establishment F1
  0.676 / Occurrence F1 0.372 / Three-layer F1 0.276.

Neither number is directly comparable to MARBLE's window-level MNID v1;
both are per-note. They're listed for orientation only.

## Recommended encoders to add (not yet integrated)

For deeper coverage on this task, see the encoder landscape report in
the plan file at `/Users/sid/.claude/plans/okay-check-the-bugs-snug-feather.md`
(updated 22 May 2026). Top candidates:

1. **Aria-medium-embedding** (`loubb/aria-medium-embedding`) — best
   solo-piano symbolic encoder available; contrastive head fits the
   retrieval probe natively.
2. **MidiBERT-Piano** (`wazenmai/MIDI-BERT`) — literature baseline for
   BPS-Motif; would unlock direct per-note F1 comparability.
3. **Moonbeam-839M** (`guozixunnicolas/moonbeam-midi-foundation-model`)
   — modern multi-instrument alternative to AMT.

CLaMP3-symbolic is the v1 baseline because it's already integrated.
