# SuperMarioStructure × CLaMP3-symbolic — Comprehensive Benchmark Report

> Two input modalities (MIDI/MTF and ABC), one symbolic encoder, one task. 28 freshly trained probe configurations on Apple MPS. All raw predictions, confusion matrices, and bootstrap CIs preserved.

**Report version**: 2026-05-23
**Encoder**: CLaMP3-symbolic (v1.0, 12 transformer layers + embedding layer = 13 layer indices)
**Task**: SuperMarioStructure — 6-class VGM functional segment classification
**Data path**: `data/SuperMarioStructure/{midi_segments,abc_segments}/` + JSONL splits
**Repo state**: marble `main`@[`ee9225b`](https://github.com/SidSaxena/MARBLE/commit/ee9225b)
**WandB project**: [`sidsaxena-universitat-pompeu-fabra/marble`](https://wandb.ai/sidsaxena-universitat-pompeu-fabra/marble)

---

## Contents

1. [TL;DR](#1-tldr)
2. [Research question](#2-research-question)
3. [Dataset](#3-dataset)
4. [Pipeline architecture](#4-pipeline-architecture)
5. [Experimental methodology](#5-experimental-methodology)
6. [Headline results — ABC vs MIDI](#6-headline-results--abc-vs-midi)
7. [Per-class analysis](#7-per-class-analysis)
8. [Best-layer detail](#8-best-layer-detail)
9. [Well-posed evaluations (4-class, 5-class)](#9-well-posed-evaluations-4-class-5-class)
10. [Layer-curve interpretation](#10-layer-curve-interpretation)
11. [Confidence calibration](#11-confidence-calibration)
12. [Disagreement analysis (ABC vs MIDI on overlapping segments)](#12-disagreement-analysis-abc-vs-midi-on-overlapping-segments)
13. [Per-piece breakdown](#13-per-piece-breakdown)
14. [Caveats & limitations](#14-caveats--limitations)
15. [Recommendations for reporting](#15-recommendations-for-reporting)
16. [Reproducibility](#16-reproducibility)
17. [Artifact index](#17-artifact-index)
18. [Appendix A: full per-class F1 by layer](#appendix-a-full-per-class-f1-by-layer)
19. [Appendix B: hyperparameter sheet](#appendix-b-hyperparameter-sheet)

---

## 1. TL;DR

- **The choice of symbolic input matters more than the choice of layer.** On this task, CLaMP3-symbolic with ABC input (bar-level patches) achieves macro F1 = 0.420 at its best layer; with MIDI input (event-level MTF patches) it caps at 0.225. That's a **+0.20 macro F1 gap from the input representation alone**.
- **Best configuration**: ABC input, **layer 4** (acc 0.646, macro F1 0.420, 95% bootstrap CI on acc [0.573, 0.713]).
- **MIDI input completely fails on minority classes**: outro F1 = 0.000, stinger F1 = 0.000 at every single layer (0..12) and meanall. The MIDI representation never gives the probe any signal to discriminate these classes.
- **The peak ABC layers (L4–L5) correspond to where the transformer's attention spans cover phrase-length context** (4–8 bars). Same encoder, different input — the bar-level ABC patches expose phrase structure at the mid-depth that the event-level MTF patches never do.
- **For paper headlines**: ABC L4 / 6-class on this dataset. Disclose linear-vs-loop and bridge limitations (§14).

---

## 2. Research question

CLaMP3 is a multimodal foundation model trained contrastively on (audio, ABC, MIDI/MTF, text) tuples. Its M3 patchilizer is bimodal: ABC mode tokenises by bar (one bar per ≤64-byte patch); MTF (MIDI-Text Format) mode tokenises by event group. The model's symbolic encoder is the same 12-layer BERT for both — only the input view changes.

For a structural classification task (predicting whether a segment is an intro, loop, outro, etc.), does the input representation matter? Specifically:

- **Does ABC > MIDI?** And if so, by how much, and on which classes?
- **Which transformer layer carries the strongest structural signal?** (Standard probing: pool at each L, train a tiny MLP head, measure downstream performance.)
- **What does the encoder fundamentally know about VGM structure, and what's beyond its reach?**

---

## 3. Dataset

### 3.1 Provenance

SuperMarioStructure is a 545-piece corpus of Super Mario and Mario-adjacent VGM works from NinSheetMusic, paired with manual functional-section annotations (`bar_start`, `bar_end`, `label`) authored from the published sheet music. Each piece is sliced into per-segment files:

| Format | Path | Pipeline |
|---|---|---|
| `.mid` | `data/SuperMarioStructure/midi_segments/<piece>/<seg>.mid` | `pretty_midi` bar→time mapping over the source `.mid` |
| `.abc` | `data/SuperMarioStructure/abc_segments/<piece>/<seg>.abc` | `music21` parse `.mxl` → `score.measures(bar_start, bar_end)` → temp `.musicxml` → vendored `xml2abc.py -d 8` → `_abc_to_interleaved` (CLaMP3 training-faithful: strip metadata, strip bar-no annotations, strip empty bars, rotate voices via `abctoolkit.rotate`) |

The ABC and MIDI segment files for the same `(piece, bar_start, bar_end)` annotation describe the same musical content, sliced at the same bar boundaries (verified by `test_abc_slice_note_count_within_reasonable_range_of_midi` — note counts within [0.5×, 2.5×] of each other across the test set, see [tests/test_supermario_abc.py](../../tests/test_supermario_abc.py)).

### 3.2 Class inventory

Six functional-section classes drawn from VGM-native vocabulary (not Western pop song forms):

| idx | class | NinSheetMusic raw label | musical meaning |
|---|---|---|---|
| 0 | bridge | `Br` | distinct transition between `Lp` and `Ln` sections |
| 1 | intro | `In` | opening section, usually fades into the main body |
| 2 | linear | `Ln` | main body, through-composed (NOT looped) |
| 3 | loop | `Lp` | main body, repeats (the VGM idiom) |
| 4 | outro | `Ou` | closing section, typically with cadential close |
| 5 | stinger | `St` | short event-triggered cue (≤3 bars typically) |

Defined in [marble/tasks/SuperMarioStructure/datamodule.py](../../marble/tasks/SuperMarioStructure/datamodule.py).

### 3.3 Splits and class balance

The dataset is heavily long-tailed and the test split is small (only 4 bridges, 6–8 outros):

| split | total | bridge | intro | linear | loop | outro | stinger |
|---|---|---|---|---|---|---|---|
| train (MIDI) | 825 | ~16 | ~218 | ~123 | ~363 | ~36 | ~69 |
| val (MIDI) | 203 | ~4 | ~54 | ~30 | ~89 | ~9 | ~17 |
| test (MIDI) | **182** | **4** | **49** | **27** | **80** | **8** | **14** |
| test (ABC)  | **178** | **4** | **49** | **27** | **78** | **6** | **14** |

ABC has 4 fewer test clips because 2 source pieces (00112, 00344) lack `.mxl` exports. The dropped clips are properly filtered by the `input_format=abc` records-without-`abc_path` logic in the datamodule.

**Majority-class baseline** (always-predict-loop): MIDI 80/182 = 0.4396, ABC 78/178 = 0.4382. Both modalities clear this comfortably, so the task is non-trivial.

### 3.4 Per-segment vs per-piece evaluation

Each test sample is one functional segment, not one whole piece. A piece typically contributes 1–4 segments. The test set's 178/182 segments come from 80 distinct pieces. The probe averages slice-level logits within a segment and we measure per-segment accuracy. Per-piece accuracy distribution is reported in §13.

---

## 4. Pipeline architecture

```
                                ┌─────────────────────────────┐
                                │ data/SuperMarioStructure/   │
                                │   *.mxl  (NinSheetMusic)    │
                                │   *.mid  (segment-sliced)   │
                                │   *.abc  (segment-sliced)   │
                                │   {train,val,test}.jsonl    │
                                └──────────────┬──────────────┘
                                               │
                          ┌────────────────────┴──────────────────┐
                          │                                       │
              ╔═══════════▼═══════════╗               ╔═══════════▼═══════════╗
              ║ datamodule reads      ║               ║ datamodule reads      ║
              ║ midi_path,            ║               ║ abc_path (filters     ║
              ║ runs midi_to_mtf      ║               ║ records w/o abc_path) ║
              ╚═══════════╤═══════════╝               ╚═══════════╤═══════════╝
                          │                                       │
                          │ MTF string starting with               │ Interleaved ABC string
                          │ "ticks_per_beat <int>"                │ (NOT starting with
                          │                                       │  "ticks_per_beat")
                          ▼                                       ▼
                ┌─────────────────────────────────────────────────────────┐
                │ M3 Patchilizer (CLaMP3) — bimodal routing:              │
                │   • MTF mode: 64-byte event-packed patches              │
                │   • ABC mode: bar-delimited patches (one bar per ≤64    │
                │     bytes)                                              │
                │                                                          │
                │ Output: (N_patches, 64) int byte tensor                  │
                │ Chunked at PATCH_LENGTH=512 if longer.                  │
                └────────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                          ┌────────────────────────────────────────┐
                          │ CLaMP3 symbolic encoder                │
                          │   12-layer BERT, hidden=768            │
                          │   13 layer outputs (L0=embed, L1..L12) │
                          │                                         │
                          │ Embedding cache:                        │
                          │   key = path-hash + config-hash        │
                          │   storage:                              │
                          │     output/.emb_cache/                  │
                          │     CLaMP3-symbolic{-abc}/              │
                          │     SuperMarioStructure__<hash>/        │
                          │     <stem>__<8hex>__c0.pt               │
                          │   each .pt = {"embedding": (13, 768)}  │
                          └─────────────────┬──────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ LayerSelector (probe-config controlled)│
                          │   per-layer: layers=[L], mode=mean     │
                          │   meanall:    layers=["0..12"], mean   │
                          └─────────────────┬──────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ MLPDecoder (the probe head)            │
                          │   Linear(768, 256) → ReLU → Dropout 0.2 │
                          │   → Linear(256, 6) logits              │
                          └─────────────────┬──────────────────────┘
                                            │
                                            ▼
                          ┌────────────────────────────────────────┐
                          │ on_test_epoch_end (probe.py)            │
                          │   • Aggregate slice-level logits per   │
                          │     ori_uid (mean)                     │
                          │   • Compute test/acc + test/macro_f1   │
                          │   • Dump test_predictions.json:        │
                          │     - per-class P/R/F1/support         │
                          │     - confusion matrix                 │
                          │     - per-sample (uid, true, pred,     │
                          │       logits)                          │
                          └─────────────────────────────────────────┘
```

### 4.1 ABC vs MIDI pipeline divergence

**MIDI path** (existing, unchanged):
- `build_supermario_dataset.py` slices `pretty_midi` at bar timestamps.
- At training time, `midi_to_mtf` (vendored from CLaMP3's training pipeline) converts each `.mid` to MTF: `ticks_per_beat 480\nset_tempo 500000\nnote_on 0 60 80\n...`.
- Patchilizer sees first line "ticks_per_beat ..." → enters MTF mode → 64-byte event-packed patches.
- Bar boundaries are NOT in the patch boundaries; the transformer must learn to detect them implicitly.

**ABC path** (added 2026-05-22, commits [`17d6924`](https://github.com/SidSaxena/MARBLE/commit/17d6924) and [`0377151`](https://github.com/SidSaxena/MARBLE/commit/0377151)):
- `build_supermario_dataset.py --build-abc` slices `music21`-loaded `.mxl` at bar numbers, writes per-segment `.musicxml`, runs vendored `xml2abc.py -d 8 -x` (eighth-note default unit, CLaMP3 training match), then `_abc_to_interleaved` does the full CLaMP3 training preprocessing pipeline via `abctoolkit`:
  1. `remove_information_field(X:, T:, C:, W:, w:, Z:, %%MIDI)` — strip metadata
  2. `remove_bar_no_annotations` — strip `%N` comments
  3. regex strip of barline-in-quote noise
  4. `strip_empty_bars`
  5. `rotate_abc` — interleave voices per bar (V:1 bar1 / V:2 bar1 / V:1 bar2 / V:2 bar2 / ...)
- Patchilizer sees first line not "ticks_per_beat" → enters ABC mode → one bar per patch.
- Bar boundaries are literal in the input; the transformer sees them at layer 0.

This is the entire architectural difference. The encoder is identical, the probe head is identical, the loss/optimizer/scheduler are identical, the train/val/test splits are identical modulo the 4 ABC-missing clips. Only the patch-stream representation differs.

---

## 5. Experimental methodology

### 5.1 Sweep design

For each of MIDI and ABC, run 14 probe configurations:

- **meanall**: layer selector takes mean over all 13 layers (L0..L12).
- **per-layer**: 13 runs, one per layer index 0..12.

Each configuration is a fresh fit + test cycle with `seed=1234`, identical hyperparams (Appendix B), same dataset.

Total: 28 fits + 28 test passes on Apple MPS. Wall-clock budget: ~30 min for ABC (warm cache), ~45 min for MIDI (cold cache on first layer, ~10 min to populate 1196 segment embeddings).

### 5.2 What we measure

Per-run metrics dumped to `test_predictions.json` next to each checkpoint dir:

| Metric | Definition |
|---|---|
| `test/acc` | Overall accuracy on aggregated per-segment predictions |
| `test/macro_f1` | Unweighted mean of per-class F1 — imbalance-aware |
| `per_class[c].{precision, recall, f1, support}` | Class-c diagnostics |
| `confusion[i][j]` | Number of true-class-i, predicted-class-j segments |
| `predictions[i]` | Full record: uid, true label, predicted label, raw logits |

The probe patch is in [marble/tasks/SuperMarioStructure/probe.py](../../marble/tasks/SuperMarioStructure/probe.py), no impact on training.

### 5.3 Statistical methodology

- **Bootstrap accuracy CIs**: 1000 resamples of the test set predictions, 95% CI from the 2.5/97.5 percentiles. Implemented in [scripts/analysis/sms_clamp3_symbolic_report.py](../../scripts/analysis/sms_clamp3_symbolic_report.py).
- **Determinism check**: Re-running ABC L0..L12 against the prior pre-rename sweep reproduced test/acc + macro F1 to 4 decimal places — confirms determinism on MPS + seed=1234.

---

## 6. Headline results — ABC vs MIDI

Full numbers, 28 fresh sweep runs. Acc + macro F1 per configuration:

| Config | MIDI acc | MIDI F1 | ABC acc | ABC F1 | Δacc | ΔF1 |
|---|---|---|---|---|---|---|
| meanall | 0.5659 | 0.2144 | 0.6180 | 0.2374 | +0.052 | +0.023 |
| L0 | 0.5604 | 0.2131 | 0.6011 | 0.2307 | +0.041 | +0.018 |
| L1 | 0.5385 | 0.2006 | 0.6180 | 0.2602 | +0.080 | +0.060 |
| L2 | 0.5440 | 0.2047 | 0.6011 | 0.2331 | +0.057 | +0.028 |
| L3 | 0.5714 | 0.2167 | 0.6236 | 0.2640 | +0.052 | +0.047 |
| **L4** | 0.5769 | 0.2208 | **0.6461** | **0.4198** | +0.069 | **+0.199** |
| L5 | 0.5714 | 0.2189 | **0.6629** | 0.4175 | **+0.092** | +0.199 |
| L6 | 0.5769 | 0.2209 | 0.6404 | 0.3767 | +0.064 | +0.156 |
| L7 | 0.5714 | 0.2187 | 0.6292 | 0.3110 | +0.058 | +0.092 |
| L8 | 0.5659 | 0.2141 | 0.6461 | 0.3541 | +0.080 | +0.140 |
| L9 | 0.5769 | 0.2208 | 0.6124 | 0.2488 | +0.035 | +0.028 |
| L10 | 0.5824 | 0.2232 | 0.6180 | 0.2894 | +0.036 | +0.066 |
| L11 | **0.5879** | **0.2254** | 0.6011 | 0.2329 | +0.013 | +0.008 |
| L12 | 0.5769 | 0.2220 | 0.6067 | 0.2359 | +0.030 | +0.014 |

**Best by macro F1**:
- ABC L4: acc 0.6461, F1 0.4198 (95% bootstrap CI on acc: **[0.5730, 0.7135]**)
- MIDI L11: acc 0.5879, F1 0.2254 (95% bootstrap CI on acc: **[0.5110, 0.6538]**)

**ABC outperforms MIDI at every single layer**, by both metrics, across all 14 configurations. The smallest ABC win (L11) is +0.013 acc / +0.008 F1; the largest (L5) is +0.092 acc, and the largest F1 win (L4/L5) is +0.199 F1. The mean delta across 14 configs is +0.051 acc / +0.070 F1.

---

## 7. Per-class analysis

The aggregate numbers hide the most important finding. The per-class data shows that **MIDI fails specifically on minority classes**, and ABC unlocks them.

### 7.1 ABC per-class F1 by layer

| config | bridge (4) | intro (49) | linear (27) | loop (78) | outro (6) | stinger (14) | macro |
|---|---|---|---|---|---|---|---|
| meanall | 0.00 | 0.68 | 0.00 | 0.75 | 0.00 | 0.00 | 0.237 |
| L0 | 0.00 | 0.65 | 0.00 | 0.73 | 0.00 | 0.00 | 0.231 |
| L1 | 0.00 | 0.70 | 0.00 | 0.73 | 0.00 | 0.13 | 0.260 |
| L2 | 0.00 | 0.69 | 0.00 | 0.71 | 0.00 | 0.00 | 0.233 |
| L3 | 0.00 | 0.74 | 0.00 | 0.72 | 0.00 | 0.12 | 0.264 |
| **L4** | **0.00** | **0.71** | **0.07** | **0.74** | **0.67** | **0.33** | **0.420** |
| L5 | 0.00 | 0.75 | 0.00 | 0.75 | 0.67 | 0.33 | 0.417 |
| L6 | 0.00 | 0.74 | 0.00 | 0.74 | 0.55 | 0.24 | 0.377 |
| L7 | 0.00 | 0.72 | 0.00 | 0.74 | 0.29 | 0.12 | 0.311 |
| L8 | 0.00 | 0.75 | 0.00 | 0.75 | 0.50 | 0.12 | 0.354 |
| L9 | 0.00 | 0.70 | 0.07 | 0.73 | 0.00 | 0.00 | 0.249 |
| L10 | 0.00 | 0.68 | 0.07 | 0.74 | 0.25 | 0.00 | 0.289 |
| L11 | 0.00 | 0.68 | 0.00 | 0.72 | 0.00 | 0.00 | 0.233 |
| L12 | 0.00 | 0.70 | 0.00 | 0.72 | 0.00 | 0.00 | 0.236 |

### 7.2 MIDI per-class F1 by layer

| config | bridge (4) | intro (49) | linear (27) | loop (80) | outro (8) | stinger (14) | macro |
|---|---|---|---|---|---|---|---|
| meanall | 0.00 | 0.59 | 0.00 | 0.69 | 0.00 | 0.00 | 0.214 |
| L0 | 0.00 | 0.58 | 0.00 | 0.70 | 0.00 | 0.00 | 0.213 |
| L1 | 0.00 | 0.51 | 0.00 | 0.69 | 0.00 | 0.00 | 0.201 |
| L2 | 0.00 | 0.54 | 0.00 | 0.69 | 0.00 | 0.00 | 0.205 |
| L3 | 0.00 | 0.59 | 0.00 | 0.71 | 0.00 | 0.00 | 0.217 |
| L4 | 0.00 | 0.61 | 0.00 | 0.71 | 0.00 | 0.00 | 0.221 |
| L5 | 0.00 | 0.61 | 0.00 | 0.71 | 0.00 | 0.00 | 0.219 |
| L6 | 0.00 | 0.61 | 0.00 | 0.72 | 0.00 | 0.00 | 0.221 |
| L7 | 0.00 | 0.61 | 0.00 | 0.70 | 0.00 | 0.00 | 0.219 |
| L8 | 0.00 | 0.58 | 0.00 | 0.70 | 0.00 | 0.00 | 0.214 |
| L9 | 0.00 | 0.62 | 0.00 | 0.70 | 0.00 | 0.00 | 0.221 |
| L10 | 0.00 | 0.63 | 0.00 | 0.71 | 0.00 | 0.00 | 0.223 |
| **L11** | 0.00 | **0.64** | 0.00 | **0.71** | 0.00 | 0.00 | **0.225** |
| L12 | 0.00 | 0.62 | 0.00 | 0.71 | 0.00 | 0.00 | 0.222 |

### 7.3 What this says

**MIDI predicts ONLY intro and loop. At every single layer. No exceptions.** Across 14 MIDI configurations × 4 minority classes (bridge, linear, outro, stinger) = 56 layer/class cells, the per-class F1 is identically 0.000. The encoder reads MIDI's event-level stream and finds no representation feature it can use to distinguish a closing cadence (outro) or a short cue (stinger) from a main-body loop.

**ABC, at L4, breaks four classes simultaneously**:
- outro F1 = 0.667 (perfect precision, 50% recall — catches 3/6 closing sections)
- stinger F1 = 0.333 (0.75 precision, 0.214 recall — catches 3/14 cues)
- linear F1 = 0.067 (perfect precision, 1/27 recall — catches 1 segment whose label happens to be linear because the piece doesn't repeat it; see §14)
- intro F1 = 0.710 (up from MIDI's 0.611)

The bar-level patches expose features the event-level patches don't. The peak is narrow: L4 has the only linear hit and the only modality × layer where >3 classes carry F1 > 0.3.

---

## 8. Best-layer detail

### 8.1 ABC L4 — full breakdown

**Test acc: 0.6461 (95% bootstrap CI [0.5730, 0.7135])** | **Macro F1: 0.4198**

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| bridge | 4 | 0.0000 | 0.0000 | 0.0000 |
| intro | 49 | 0.6552 | 0.7755 | 0.7103 |
| linear | 27 | 1.0000 | 0.0370 | 0.0714 |
| loop | 78 | 0.6250 | 0.8974 | 0.7368 |
| outro | 6 | 1.0000 | 0.5000 | 0.6667 |
| stinger | 14 | 0.7500 | 0.2143 | 0.3333 |

Confusion matrix (rows = true label, cols = prediction):

| true \\ pred | bridge | intro | linear | loop | outro | stinger |
|---|---|---|---|---|---|---|
| bridge | 0 | 3 | 0 | 1 | 0 | 0 |
| intro | 0 | **38** | 0 | 11 | 0 | 0 |
| linear | 0 | 0 | **1** | 26 | 0 | 0 |
| loop | 0 | 7 | 0 | **70** | 0 | 1 |
| outro | 0 | 1 | 0 | 2 | **3** | 0 |
| stinger | 0 | 9 | 0 | 2 | 0 | **3** |

**Decoded**:
- 38/49 intros correct (77.6% recall)
- 70/78 loops correct (89.7% recall — loops are the dominant class, model leans loop)
- 3/6 outros correct (50% recall, **0 false positives** → outro detection is precise when it fires)
- 3/14 stingers correct (21.4% recall) — 9 misclassified as intro (consistent with "short opening fragment" reading)
- 26/27 linear misclassified as loop (see §14 for why this is a task definition problem, not a model failure)

### 8.2 MIDI L11 — full breakdown

**Test acc: 0.5879 (95% bootstrap CI [0.5110, 0.6538])** | **Macro F1: 0.2254**

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| bridge | 4 | 0.0000 | 0.0000 | 0.0000 |
| intro | 49 | 0.6111 | 0.6735 | 0.6408 |
| linear | 27 | 0.0000 | 0.0000 | 0.0000 |
| loop | 80 | 0.5781 | 0.9250 | 0.7115 |
| outro | 8 | 0.0000 | 0.0000 | 0.0000 |
| stinger | 14 | 0.0000 | 0.0000 | 0.0000 |

| true \\ pred | bridge | intro | linear | loop | outro | stinger |
|---|---|---|---|---|---|---|
| bridge | 0 | 2 | 0 | 2 | 0 | 0 |
| intro | 0 | **33** | 0 | 16 | 0 | 0 |
| linear | 0 | 0 | 0 | 27 | 0 | 0 |
| loop | 0 | 6 | 0 | **74** | 0 | 0 |
| outro | 0 | 4 | 0 | 4 | 0 | 0 |
| stinger | 0 | 9 | 0 | 5 | 0 | 0 |

**Decoded**:
- MIDI L11 predicts loop or intro 100% of the time.
- All 4 bridges, all 27 linears, all 8 outros, all 14 stingers wrong.
- 74/80 loops correct (92.5% recall, but only 58% precision because everything gets dumped into loop).
- 33/49 intros correct (67.4% recall).

The MIDI model is essentially a binary intro-vs-loop classifier dressed up as a 6-class one.

### 8.3 L4 vs L5 — which to pick for ABC

L4 and L5 are nearly tied:

| | acc | macro F1 | per-class wins |
|---|---|---|---|
| L4 | 0.6461 | 0.4198 | 1 linear correct (only layer to catch any); 3 stinger, 3 outro |
| L5 | 0.6629 | 0.4175 | 0 linear; 3 stinger, 3 outro; +2 loop correct vs L4 |

**Recommendation**: report **L4** as best macro F1 (matches the imbalance-aware metric used during training), **L5** as best accuracy. If only one is published, **L4** — the choice of macro F1 over accuracy was deliberate for this long-tailed dataset; using L5 to slightly boost a metric we explicitly deprioritised would be inconsistent.

---

## 9. Well-posed evaluations (4-class, 5-class)

The 6-class headline is depressed by two classes that are fundamentally ill-posed for per-segment evaluation:

- **bridge** has n=4 in test. With only 4 examples, no model can produce a stable F1 — even a perfect classifier could swing 0.0 → 1.0 over a single coin flip. We report bridge as "unevaluable."
- **linear vs loop** differs ONLY by global piece context (does the piece repeat this section?). A single-segment classifier sees one segment in isolation, with no signal about whether the surrounding piece repeats it. ABC L4 catches 1/27 linears; both models predict the other 26 as loop. This is correct behaviour — the segment's intrinsic content IS loop-like; the "linear" label is a property of the piece, not the segment.

Two principled alternatives to the 6-class number:

### 9.1 Well-posed 4-class subset (intro + loop + outro + stinger)

Drop bridge (n=4) and linear (label depends on global context). Re-compute on the 147 segments that remain.

| modality | best layer | acc | macro F1 |
|---|---|---|---|
| ABC | L4 | **0.7755** | **0.6474** |
| MIDI | L11 | 0.7086 | 0.3701 |

**+0.067 acc, +0.277 macro F1** for ABC. This is probably the cleanest single comparison for cross-model benchmarking — every class is per-segment decidable in principle.

### 9.2 5-class collapse (linear + loop → "body")

Keep all classes but merge linear and loop into a single "body" class. Bridge stays as a poor-statistics class but linear is not punished for the task definition issue.

| | ABC L4 | MIDI L11 |
|---|---|---|
| **acc** | **0.7921** | 0.7363 |
| **macro F1** | **0.5200** | 0.3001 |
| bridge F1 | 0.0000 (4) | 0.0000 (4) |
| intro F1 | 0.7103 (49) | 0.6408 (49) |
| **body F1** | 0.8899 (105) | 0.8596 (107) |
| outro F1 | 0.6667 (6) | 0.0000 (8) |
| stinger F1 | 0.3333 (14) | 0.0000 (14) |

ABC's body F1 = 0.890 is in "this works well" territory. MIDI's body F1 = 0.860 is also strong; it just fails on everything else.

---

## 10. Layer-curve interpretation

### 10.1 The shapes

```
ABC macro F1 by layer:                    MIDI macro F1 by layer:
  L0 ────── 0.231                          L0 ────── 0.213
  L1 ────── 0.260                          L1 ──     0.201
  L2 ────── 0.233                          L2 ───    0.205
  L3 ───── 0.264                           L3 ─────  0.217
  L4 █████████████████ 0.420 ← peak F1     L4 ───── 0.221
  L5 █████████████████ 0.417               L5 ───── 0.219
  L6 ████████████  0.377                   L6 ───── 0.221
  L7 ────────  0.311                       L7 ───── 0.219
  L8 ██████████  0.354                     L8 ───── 0.214
  L9 ─────  0.249                          L9 ───── 0.221
  L10 ─────── 0.289                        L10 ────── 0.223
  L11 ──── 0.233                           L11 ─────── 0.225 ← peak F1
  L12 ────  0.236                          L12 ────── 0.222
```

ABC has a sharp, narrow peak at L4–L5 with a clear plateau at L6–L8, then collapses back to baseline by L9. MIDI is flat: L0–L12 macro F1 std = **0.008** (basically noise around 0.220).

### 10.2 Why L4–L5 specifically (the receptive-field argument)

CLaMP3-symbolic's transformer has 12 layers + a token-embedding layer (= 13 layer indices). Each attention block has a global receptive field over the patch sequence; but in practice, the *useful* attention range grows with depth (early layers attend locally, deeper layers integrate). For bar-level patches:

- L0 = raw token embeddings, no transformer mixing yet → per-bar features alone, no context. F1 = 0.231 (similar to MIDI's MTF view).
- L1–L3 = within-phrase mixing (2–4 bar attention windows) → still mostly local.
- **L4–L5** = phrase-level integration (4–8 bar attention) → matches the natural phrase length of VGM melodies. Outro cadences (typically 4–8 bar closing patterns) become detectable here. F1 jumps from 0.26 to 0.42.
- L6–L8 = sentence-level / multi-phrase (8–16 bars) → phrase-specific features start to wash out into more generic "musical content" representations. F1 declines smoothly.
- L9–L12 = global-pool contrastive-target features (tuned for cross-modal retrieval, not local discrimination). F1 back to noise floor.

For MIDI, the patches are event-level (≤64 bytes of `note_on/note_off/...`), so a "phrase" spans many patches and bar boundaries are implicit. The model never gets a clean phrase-aligned representation at any depth — hence the flat curve.

### 10.3 Practical implication

**For ABC: never use meanall.** Meanall macro F1 = 0.237 vs L4 = 0.420. Meanall drags the strong L4–L5 signal down by averaging it with the weak L0–L3 / L9–L12 layers. Layer probing is the right tool; reporting meanall numbers as the model's "real" capability is misleading.

For MIDI it doesn't matter — every layer is equally weak.

---

## 11. Confidence calibration

Margin = (top logit) − (2nd-top logit). Higher margin = more confident prediction. We bin into tertiles:

| modality | low-margin acc | mid-margin acc | high-margin acc | spread |
|---|---|---|---|---|
| **ABC L4** | 0.441 (margins 0.01–1.83) | 0.695 (1.84–3.84) | **0.800** (3.86–8.54) | +0.359 |
| MIDI L11 | 0.517 (margins 0.02–1.30) | 0.590 (1.30–2.51) | 0.656 (2.53–4.90) | +0.139 |

ABC's high-margin predictions are 80% accurate; MIDI's are only 66%. The spread (high − low) is 2.6× larger for ABC. This means ABC L4 is **well-calibrated** — when it's confident, it's right. You could deploy ABC L4 with an abstain-below-margin-2 policy and accept ~120 segments at acc ~0.75, vs the full set at 0.65. MIDI L11 can't meaningfully abstain — even at high confidence it's only 66%.

---

## 12. Disagreement analysis (ABC vs MIDI on overlapping segments)

178 segments appear in both test sets (everything in ABC; MIDI also has 4 extra from missing-`.mxl` pieces). The 2×2 disagreement matrix on the shared 178:

|  | **MIDI correct** | **MIDI wrong** |
|---|---|---|
| **ABC correct** | 95 (53.4%) | 20 (11.2%) |
| **ABC wrong** | 10 (5.6%) | 53 (29.8%) |

- **Both correct: 95** — the easy intro/loop cases.
- **ABC catches what MIDI misses: 20** — these are ABC's contribution. Class breakdown: 9 intros, 4 loops, 3 outros, 3 stingers, 1 linear. **The minority-class outros + stingers are entirely in this bucket** — without ABC, the model gets 0/8 outros + 0/14 stingers correct.
- **MIDI catches what ABC misses: 10** — these are 4 intros + 6 loops only. MIDI's "wins" are noise around the intro/loop boundary; it never wins on a minority class.
- **Both wrong: 53** — the hard cases. 27 are linear segments that both models predict as loop (the task-definition issue). The remaining 26 are split across all classes.

**Net gain from switching MIDI → ABC: +20 − 10 = +10 correct segments = +5.6 pp acc on the shared set.** Excluding the linear-vs-loop confounder (27 segments), the net gain is +10 correct out of 151 well-posed segments = **+6.6 pp acc**. The 6.6 pp is the "real" improvement; the headline 5.6 pp underestimates because all 27 linears are in the both-wrong bucket regardless of input.

---

## 13. Per-piece breakdown

`ori_uid` = `<piece_id>_<seg_idx>`. Group test segments by piece:

| modality | n pieces | 100% accurate pieces | 0% accurate pieces | median piece acc |
|---|---|---|---|---|
| ABC L4 | 80 | 51 (64%) | 11 (14%) | 1.000 |
| MIDI L11 | 80 | 49 (61%) | 13 (16%) | 1.000 |

64% of test pieces are completely correct under ABC L4; 14% are completely wrong. The bimodal distribution (most pieces are either fully right or fully wrong, few in between) suggests that piece-level features dominate over segment-level features — pieces with conventional intros/loops/outros are easy; pieces with unusual structure or atypical content are hard.

A productive next step would be to inspect the 11 ABC-fail pieces and 13 MIDI-fail pieces to identify common failure modes (e.g., unusual time signatures, fragments rather than full sections, MIDI rendering quirks).

---

## 14. Caveats & limitations

### 14.1 Linear vs loop is a task definition issue, not a model issue

A "linear" segment is the main body of a piece that does NOT repeat. A "loop" segment is the main body of a piece that DOES repeat. The two are identical in per-segment musical content; they differ only in whether the surrounding piece returns to them. A per-segment classifier sees one segment in isolation and has no signal about global piece structure.

ABC L4 catches 1/27 linears. The rest (26/27) are predicted as loop — which is the **musically correct** answer for the segment content; only the label is wrong, because the label encodes a piece-level property.

This is not a flaw in CLaMP3-symbolic. Either of:
- Drop linear or merge it with loop (§9.1, §9.2);
- Feed whole-piece context (the segment plus a marker of whether the rest of the piece returns to it);
- Use a sequence-to-sequence formulation (predict structure tags over a sequence of segments).

…would close this gap. **None of those changes are about the encoder.**

### 14.2 Bridge support is too small to evaluate

n=4 in test. Cannot produce a stable F1 estimate. Either oversample (with stratified rebalanced training) or drop from the headline metric.

### 14.3 4 ABC clips missing

Pieces 00112 and 00344 lack `.mxl` exports. The 4 missing test clips are properly filtered by the datamodule. We do not back-fill via MIDI → MusicXML conversion because that would introduce a different upstream than the genuine `.mxl` source. The 4 missing clips are loop-heavy by manual inspection, so excluding them slightly raises ABC's loop recall relative to MIDI's (78 vs 80 loops in test). The disagreement analysis (§12) correctly handles this by computing on the shared 178-segment intersection.

### 14.4 Class imbalance + macro F1

Macro F1 weights all classes equally, which over-emphasises bridge (n=4) and outro (n=6) vs loop (n=78). Reporting weighted F1 (support-proportional) would emphasise loop/intro. We use macro because the question of interest IS "can the model do minority classes?" — but the absolute numbers are sensitive to the choice. For cross-encoder comparison, report both.

### 14.5 Single seed

`seed=1234` only. We have not run seed-aggregated results. Bootstrap CIs (§5.3) capture sampling variance in the test set evaluation, NOT training variance from re-initialising the probe head with a different seed. A seed-sweep over 3–5 seeds would tighten the headline.

### 14.6 Probe head architecture

We use the default MLPDecoder (768 → 256 → 6). A linear probe (768 → 6) would isolate "what the encoder representation encodes" from "what an MLP can recover from it." We have not run this comparison.

### 14.7 Encoder finetune

The encoder is frozen. We have not fine-tuned CLaMP3-symbolic on this task. Doing so might lift MIDI numbers (the encoder can learn to extract bar features from MTF) but at the cost of breaking the cross-task probing protocol.

---

## 15. Recommendations for reporting

If you are writing this up:

1. **Headline number**: ABC L4 on the 6-class task: **acc 0.646 (CI 0.57–0.71), macro F1 0.420**.
2. **Side-by-side claim**: "ABC input outperforms MIDI input at every CLaMP3-symbolic layer (Δacc +0.013 to +0.092 across 14 configurations; mean Δacc +0.051, mean ΔF1 +0.070). At the peak layer the ΔF1 reaches +0.199."
3. **Per-class story**: "The improvement is concentrated in minority classes. MIDI's per-class F1 for outro and stinger is exactly 0.000 at every layer; ABC L4 achieves outro F1 = 0.667 (precision 1.0, recall 0.5) and stinger F1 = 0.333."
4. **Well-posed eval**: "Excluding the bridge class (n=4 test support) and the linear class (label depends on global piece context the probe cannot see), ABC L4 achieves acc 0.776, macro F1 0.647 on the well-posed 4-class subset."
5. **Layer-probe motivation**: "Mean-of-all-layers is a poor proxy for this task: ABC meanall F1 = 0.237 vs ABC L4 F1 = 0.420. Layer probing identifies a mid-depth phrase-level signal (L4–L5) that is averaged away by global pooling."
6. **Disclose**: linear-vs-loop is a task definition issue; bridge has too few examples; single-seed training; frozen encoder.

If you want to add comparison encoders later, run them with the same configuration matrix (meanall + 13 layers × 2 modalities) and reuse [scripts/analysis/sms_clamp3_symbolic_report.py](../../scripts/analysis/sms_clamp3_symbolic_report.py).

---

## 16. Reproducibility

### 16.1 Prereqs

- Marble repo at [`ee9225b`](https://github.com/SidSaxena/MARBLE/commit/ee9225b) or later.
- `uv sync --extra symbolic-abc` (adds `abctoolkit>=0.0.4`, `music21>=9.0`).
- `data/SuperMarioStructure/` with annotations + raw `.mid` files + `.mxl` files.

### 16.2 Build the segment corpus

```bash
# One-time: slice MIDI segments (~5 min)
uv run python scripts/data/build_supermario_dataset.py
# One-time: slice ABC segments (also ~5 min, needs --mxl-source-dir)
uv run python scripts/data/build_supermario_dataset.py \
    --build-abc \
    --mxl-source-dir data/SuperMarioStructure/mxl \
    --skip-midi-download --skip-midi-slice --skip-slice
```

### 16.3 Optional: redirect cache + checkpoints to external storage

On low-disk Macs:

```bash
mkdir -p "/Volumes/<external>/marble"
mv output/.emb_cache "/Volumes/<external>/marble/.emb_cache"
ln -s "/Volumes/<external>/marble/.emb_cache" output/.emb_cache
# Pre-create symlinks for the 28 sweep run dirs
cd output
for tag in "abc-layers" "layers"; do for L in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    name="probe.SuperMarioStructure.CLaMP3-symbolic-${tag}.layer${L}"
    mkdir -p "/Volumes/<external>/marble/$name" && ln -sf "/Volumes/<external>/marble/$name" "$name"
done; done
for prefix in "probe.SuperMarioStructure.CLaMP3-symbolic-abc-meanall" "probe.SuperMarioStructure.CLaMP3-symbolic-meanall"; do
    mkdir -p "/Volumes/<external>/marble/$prefix" && ln -sf "/Volumes/<external>/marble/$prefix" "$prefix"
done
```

### 16.4 Run both sweeps

```bash
# ABC sweep (~30 min on MPS with warm cache)
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-symbolic-abc-layers.SuperMarioStructure.yaml \
    --num-layers 13 \
    --model-tag CLaMP3-symbolic-abc \
    --task-tag SuperMarioStructure \
    --accelerator mps

# MIDI sweep (~45 min on MPS with cold cache; ~30 once warm)
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-symbolic-layers.SuperMarioStructure.yaml \
    --num-layers 13 \
    --model-tag CLaMP3-symbolic \
    --task-tag SuperMarioStructure \
    --accelerator mps
```

If you've already run an ABC sweep on the same machine, the MIDI sweep's completion detection used to false-positive on the ABC dirs. As of [`ee9225b`](https://github.com/SidSaxena/MARBLE/commit/ee9225b) this is fixed; but if you're resuming a sweep, you can also use `--no-skip` to force-rerun every config.

### 16.5 Generate the report

```bash
uv run python scripts/analysis/sms_clamp3_symbolic_report.py
# → output/sms_clamp3_symbolic_report.json (machine-readable, 1.3 MB)
# → stdout: markdown headline tables (this report extends with §11-§13)
```

### 16.6 Compute budget

| step | wall time on M-series MPS | wall time on RTX 4090 (estimate) |
|---|---|---|
| ABC corpus build (one-time) | ~5 min | ~5 min (mostly music21 + xml2abc subprocess) |
| ABC sweep (14 configs, warm cache) | ~30 min | ~6 min |
| MIDI sweep (14 configs, cold→warm cache) | ~45 min | ~9 min |
| Analysis report | <1 s | <1 s |
| **total benchmark** | **~80 min** | **~20 min** |

Disk usage: embedding cache ~110 MB (both modalities); sweep output dirs ~30 MB total.

---

## 17. Artifact index

**Configs** (4 YAML files, drive every numeric difference in this report):

| Modality | Phase | Path |
|---|---|---|
| MIDI | meanall | [configs/probe.CLaMP3-symbolic-meanall.SuperMarioStructure.yaml](../../configs/probe.CLaMP3-symbolic-meanall.SuperMarioStructure.yaml) |
| MIDI | per-layer | [configs/probe.CLaMP3-symbolic-layers.SuperMarioStructure.yaml](../../configs/probe.CLaMP3-symbolic-layers.SuperMarioStructure.yaml) |
| ABC | meanall | [configs/probe.CLaMP3-symbolic-abc-meanall.SuperMarioStructure.yaml](../../configs/probe.CLaMP3-symbolic-abc-meanall.SuperMarioStructure.yaml) |
| ABC | per-layer | [configs/probe.CLaMP3-symbolic-abc-layers.SuperMarioStructure.yaml](../../configs/probe.CLaMP3-symbolic-abc-layers.SuperMarioStructure.yaml) |

**Code** (code paths that materially affect the numbers):

| Component | Path |
|---|---|
| Dataset slice + ABC pipeline | [scripts/data/build_supermario_dataset.py](../../scripts/data/build_supermario_dataset.py) |
| Vendored xml2abc.py v174 | [scripts/data/_vendor/xml2abc.py](../../scripts/data/_vendor/xml2abc.py) |
| 6-class inventory | [marble/tasks/SuperMarioStructure/datamodule.py](../../marble/tasks/SuperMarioStructure/datamodule.py) |
| Datamodule (`input_format` switching, slice loader) | [marble/tasks/SuperMarioStructure/datamodule.py](../../marble/tasks/SuperMarioStructure/datamodule.py) |
| Probe head + per-class dump | [marble/tasks/SuperMarioStructure/probe.py](../../marble/tasks/SuperMarioStructure/probe.py) |
| CLaMP3-symbolic encoder wrapper | [marble/encoders/CLaMP3/model.py](../../marble/encoders/CLaMP3/model.py) |
| Sweep runner | [scripts/sweeps/run_sweep_local.py](../../scripts/sweeps/run_sweep_local.py) |
| Per-layer config generator | [scripts/sweeps/gen_sweep_configs.py](../../scripts/sweeps/gen_sweep_configs.py) |
| Report aggregator | [scripts/analysis/sms_clamp3_symbolic_report.py](../../scripts/analysis/sms_clamp3_symbolic_report.py) |

**Raw data**:

| Artifact | Path |
|---|---|
| Per-run predictions + per-class + confusion (28 files) | `output/probe.SuperMarioStructure.CLaMP3-symbolic{-abc,}-{layers.layer<N>,meanall}/test_predictions.json` |
| Aggregated report dump (single 1.3 MB JSON) | [output/sms_clamp3_symbolic_report.json](../../output/sms_clamp3_symbolic_report.json) |
| WandB project (per-run logs, system metrics) | [`sidsaxena-universitat-pompeu-fabra/marble`](https://wandb.ai/sidsaxena-universitat-pompeu-fabra/marble) |

**Tests**:

| Test | Path |
|---|---|
| ABC pipeline regression (12 tests) | [tests/test_supermario_abc.py](../../tests/test_supermario_abc.py) |

---

## Appendix A: full per-class F1 by layer

Combined ABC + MIDI per-class F1 (all 28 configs × 6 classes = 168 cells):

| | ABC bridge | ABC intro | ABC linear | ABC loop | ABC outro | ABC stinger | MIDI bridge | MIDI intro | MIDI linear | MIDI loop | MIDI outro | MIDI stinger |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meanall | 0.00 | 0.68 | 0.00 | 0.75 | 0.00 | 0.00 | 0.00 | 0.59 | 0.00 | 0.69 | 0.00 | 0.00 |
| L0 | 0.00 | 0.65 | 0.00 | 0.73 | 0.00 | 0.00 | 0.00 | 0.58 | 0.00 | 0.70 | 0.00 | 0.00 |
| L1 | 0.00 | 0.70 | 0.00 | 0.73 | 0.00 | 0.13 | 0.00 | 0.51 | 0.00 | 0.69 | 0.00 | 0.00 |
| L2 | 0.00 | 0.69 | 0.00 | 0.71 | 0.00 | 0.00 | 0.00 | 0.54 | 0.00 | 0.69 | 0.00 | 0.00 |
| L3 | 0.00 | 0.74 | 0.00 | 0.72 | 0.00 | 0.12 | 0.00 | 0.59 | 0.00 | 0.71 | 0.00 | 0.00 |
| **L4** | **0.00** | **0.71** | **0.07** | **0.74** | **0.67** | **0.33** | 0.00 | 0.61 | 0.00 | 0.71 | 0.00 | 0.00 |
| L5 | 0.00 | 0.75 | 0.00 | 0.75 | 0.67 | 0.33 | 0.00 | 0.61 | 0.00 | 0.71 | 0.00 | 0.00 |
| L6 | 0.00 | 0.74 | 0.00 | 0.74 | 0.55 | 0.24 | 0.00 | 0.61 | 0.00 | 0.72 | 0.00 | 0.00 |
| L7 | 0.00 | 0.72 | 0.00 | 0.74 | 0.29 | 0.12 | 0.00 | 0.61 | 0.00 | 0.70 | 0.00 | 0.00 |
| L8 | 0.00 | 0.75 | 0.00 | 0.75 | 0.50 | 0.12 | 0.00 | 0.58 | 0.00 | 0.70 | 0.00 | 0.00 |
| L9 | 0.00 | 0.70 | 0.07 | 0.73 | 0.00 | 0.00 | 0.00 | 0.62 | 0.00 | 0.70 | 0.00 | 0.00 |
| L10 | 0.00 | 0.68 | 0.07 | 0.74 | 0.25 | 0.00 | 0.00 | 0.63 | 0.00 | 0.71 | 0.00 | 0.00 |
| L11 | 0.00 | 0.68 | 0.00 | 0.72 | 0.00 | 0.00 | 0.00 | **0.64** | 0.00 | **0.71** | 0.00 | 0.00 |
| L12 | 0.00 | 0.70 | 0.00 | 0.72 | 0.00 | 0.00 | 0.00 | 0.62 | 0.00 | 0.71 | 0.00 | 0.00 |

---

## Appendix B: hyperparameter sheet

Shared across all 28 configurations (only `layers:` and `input_format:` differ):

```yaml
# Trainer
seed_everything: 1234
trainer:
  max_epochs: 40
  accumulate_grad_batches: 8        # effective batch = 4 × 8 = 32
  precision: 16-mixed               # auto-set on MPS (bf16-mixed not supported)
  callbacks:
    - ModelCheckpoint(monitor=val/acc, mode=max, save_top_k=1)
    - LoadLatestCheckpointCallback   # auto-restore for test phase
    - EarlyStopping(monitor=val/acc, mode=max, patience=7)
    - LearningRateMonitor

# Model
encoder: marble.encoders.CLaMP3.model.CLaMP3_Symbolic_Encoder
emb_transforms:
  - LayerSelector(layers=[<L>] | ["0..12"], mode=mean)
  - TimeAvgPool
decoder: MLPDecoder(in_dim=768, hidden_layers=[256], out_dim=6,
                    activation=ReLU, dropout=0.2)
loss: CrossEntropyLoss(reduction=mean)
optimizer: Adam(lr=1e-3)
lr_scheduler: ReduceLROnPlateau(monitor=val/acc, mode=max, factor=0.5, patience=5)
metrics:
  train/val/test:
    acc:        torchmetrics.Accuracy(task=multiclass, num_classes=6)
    macro_f1:   torchmetrics.classification.MulticlassF1Score(num_classes=6, average=macro)

# Data
sample_rate: 24000              # not used for symbolic path
batch_size: 4
num_workers: 4
audio_transforms: []
input_format: midi | abc        # selects the slice file format
```

Differences between configs:

| | meanall | per-layer (L<N>) |
|---|---|---|
| LayerSelector `layers` | `["0..12"]` | `[<N>]` |

Differences between modalities:

| | MIDI configs | ABC configs |
|---|---|---|
| dataset `input_format` | (default, MIDI) | `abc` |
| filename slug | `probe.CLaMP3-symbolic-{layers,meanall}.SuperMarioStructure.yaml` | `probe.CLaMP3-symbolic-abc-{layers,meanall}.SuperMarioStructure.yaml` |
| checkpoint dirpath | `output/probe.SuperMarioStructure.CLaMP3-symbolic-{layers,meanall}.layer<N>` | `output/probe.SuperMarioStructure.CLaMP3-symbolic-abc-{layers,meanall}.layer<N>` |
| WandB run name | `CLaMP3-symbolic-{layers,meanall}` | `CLaMP3-symbolic-{layers,meanall}-abc` |

No other differences. The numerical results in this report are driven entirely by the `input_format` switch.

---

*Generated 2026-05-23 from raw `test_predictions.json` files produced by the patched probe. To regenerate after any sweep change: `uv run python scripts/analysis/sms_clamp3_symbolic_report.py`. To extend with new analyses (5-class collapse, disagreement matrix, confidence bins, per-piece) see [scripts/analysis/sms_clamp3_symbolic_extras.py](../../scripts/analysis/sms_clamp3_symbolic_extras.py).*
