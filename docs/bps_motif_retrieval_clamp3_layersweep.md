# BPS-Motif Retrieval — CLaMP3-symbolic layer sweep

Within-piece, within-letter motif **retrieval** on BPS-Motif (Hsiao, Hung, Chen,
Su — ISMIR 2023; *Beethoven_motif*: first movements of all 32 Beethoven piano
sonatas, 263 motifs / 4,944 occurrences). Zero-shot probe of CLaMP3-symbolic:
embed every motif-occurrence window, rank by cosine, score relevance as **same
`(piece_id, motif_letter)`** (letters are movement-local, so same-letter across
pieces is correctly *not* relevant).

This is the **Layer-1 representation diagnostic** for the leitmotifs
within-piece motif-discovery work: it measures whether CLaMP3's embedding space
*places motif occurrences close together at all* — the ceiling any
cosine-similarity discovery algorithm inherits.

## Run

- **5 folds × 13 layers = 65 runs**, all completed (rc=0), ~52 min wall on
  Apple Silicon **MPS** (~10.5 min/fold; encoder forward runs once per fold,
  the 13 layer-jobs read the `(L, H)` embedding cache).
- Zero-shot (`max_epochs: 0`) → **no checkpoints written**.
- Date: 2026-06-21. wandb: project `marble`, group
  `CLaMP3-symbolic / BPSMotifRetrieval`, runs tagged `fold0`…`fold4`,
  names `layer-0-test` … `layer-12-test`.
- Full metric suite enabled via `log_extended_retrieval_metrics: true`.

## Results — `test/map` by layer × fold

| layer | fold0 | fold1 | fold2 | fold3 | fold4 | **mean** |
|------:|------:|------:|------:|------:|------:|---------:|
| 0  | 0.3632 | 0.3758 | 0.3575 | 0.4325 | 0.4898 | 0.4037 |
| 1  | 0.3623 | 0.3773 | 0.3644 | 0.4165 | 0.4867 | 0.4014 |
| 2  | 0.3782 | 0.3954 | 0.3806 | 0.4301 | 0.4971 | 0.4163 |
| 3  | 0.3961 | 0.4167 | 0.4003 | 0.4575 | 0.5188 | 0.4379 |
| 4  | 0.4176 | 0.4309 | 0.4027 | 0.4744 | 0.5329 | 0.4517 |
| 5  | 0.4285 | 0.4387 | 0.4042 | 0.4842 | 0.5378 | 0.4587 |
| 6  | 0.4446 | 0.4467 | 0.4152 | 0.5039 | 0.5418 | 0.4705 |
| **7**  | **0.4477** | **0.4506** | **0.4149** | **0.5119** | **0.5436** | **0.4737** |
| 8  | 0.4504 | 0.4443 | 0.4066 | 0.4994 | 0.5348 | 0.4671 |
| 9  | 0.4431 | 0.4358 | 0.3988 | 0.4931 | 0.5240 | 0.4590 |
| 10 | 0.4434 | 0.4192 | 0.3957 | 0.4816 | 0.5209 | 0.4522 |
| 11 | 0.4272 | 0.3851 | 0.3694 | 0.4391 | 0.5005 | 0.4242 |
| 12 | 0.3892 | 0.3406 | 0.3288 | 0.3984 | 0.4471 | 0.3808 |

**Best layer: 7 (mean MAP 0.4737 raw, 0.4875 centered).**

## raw vs centered vs whitened (per-layer mean MAP)

| layer | raw | centered | whitened |
|------:|----:|---------:|---------:|
| 0  | 0.4037 | 0.4059 | 0.3910 |
| 1  | 0.4014 | 0.4048 | 0.3939 |
| 2  | 0.4163 | 0.4203 | 0.4094 |
| 3  | 0.4379 | 0.4468 | 0.4351 |
| 4  | 0.4517 | 0.4628 | 0.4482 |
| 5  | 0.4587 | 0.4714 | 0.4548 |
| 6  | 0.4705 | 0.4839 | 0.4619 |
| **7**  | 0.4737 | **0.4875** | 0.4591 |
| 8  | 0.4671 | 0.4841 | 0.4555 |
| 9  | 0.4590 | 0.4791 | 0.4476 |
| 10 | 0.4522 | 0.4738 | 0.4385 |
| 11 | 0.4242 | 0.4480 | 0.4154 |
| 12 | 0.3808 | 0.4103 | 0.3929 |

**Centering wins at *every* layer** (+~1.4 MAP points): a mild common-mean
(anisotropy) direction is worth removing. **Whitening consistently *hurts*** —
ZCA over the modest per-fold pools amplifies noise directions. Use **centered,
layer 7**.

## recall@K (best layer 7, mean across folds)

| K | raw | centered | whitened |
|---:|----:|---------:|---------:|
| 1   | 0.0549 | 0.0549 | 0.0556 |
| 5   | 0.1988 | 0.2019 | 0.2045 |
| 10  | 0.2910 | 0.2951 | 0.2979 |
| 50  | 0.5130 | 0.5311 | 0.5042 |
| 100 | 0.6141 | 0.6376 | 0.6001 |

Recall climbs steadily but **even the top-100 recovers only ~61%** of a motif's
occurrence set — the occurrence long tail is hard.

### Other metrics @ layer 7 (mean) — raw / centered / whitened

| metric | raw / centered / whitened | reading |
|---|---|---|
| `map` | 0.4737 / 0.4875 / 0.4591 | centered best |
| `r_precision` | 0.4431 / 0.4508 / 0.4329 | ~44% of relevant retrieved at the R cutoff |
| `median_rank` | 1.0 / 1.0 / 1.0 | first relevant hit almost always rank 1 |
| `mrr` | 0.9426 / 0.9438 / 0.9500 | first hit at avg rank ≈ 1.06 |
| `map@1` / `recall@1` | 0.0549 / 0.0549 / 0.0556 | top-1 ≈ 1/R of a large occurrence set |
| `hit_rate@10` | 0.9778 / 0.9810 / 0.9847 | 98% of queries have ≥1 relevant in top-10 |

## Findings

1. **Mid-layer peak (6–8), declining at both ends.** Surface layers (0–2) and
   the last layers (11–12, specialized for CLaMP3's *global* contrastive
   objective) are weaker. **`meanall` is suboptimal** — the motif-identity
   signal concentrates in the middle. Downstream discovery should use **layer 7**
   (or pool 6–8), not the all-layer mean.

2. **CLaMP3 finds *a* match well, but not *all* occurrences.** `mrr` 0.94 +
   `median_rank` 1.0 + `hit_rate@10` 0.98 say the nearest neighbor is almost
   always a correct same-motif hit. But `MAP` 0.47, `recall@10` 0.29,
   `recall@100` 0.61 and `r_precision` 0.44 say it pulls only ~half the full
   occurrence set together. This is exactly the **"finds the strong repeat,
   misses the rest"** pattern behind the low recall of within-piece discovery in
   the leitmotifs repo — now quantified.

3. **Embedding ceiling ≈ 0.47 raw / 0.49 centered MAP.** Even a perfect discovery
   algorithm on frozen CLaMP3-symbolic features is capped here. This is the
   empirical motivation for the later fine-tuning / PEFT stage (teach the encoder
   motif-occurrence invariance via self-supervised augmentation pairs; re-measure
   on this exact task).

## Caveats

- BPS-Motif is **Beethoven solo piano**, not the target VGM domain — valid for
  benchmarking the *encoder's* motif-similarity ability, but a domain gap to the
  Zelda OST remains.
- Per-fold means are reported because the BPS-Motif folds are movement-level CV
  splits (disjoint movements); for zero-shot retrieval the split only changes
  which movements are pooled, so the cross-fold mean is the headline.
- One clip-level vector per occurrence window (`TimeAvgPool` over patches, single
  layer).

## Reproduce

```bash
cd ~/developer/python/marble
# 5 folds × 13 layers on MPS (use --accelerator gpu on a CUDA box):
scripts/sweeps/run_bps_retrieval_folds.sh --accelerator mps
# aggregate the per-(fold,layer) MAP table + raw/centered/whitened + recall@K:
.venv/bin/python scripts/sweeps/bps_retrieval_summary.py
```

Requires the `symbolic-midi` extra (`uv pip install mido pretty_midi`) and the
built dataset under `data/BPS-Motif/` (see `docs/data/bps_motif_setup.md`).
