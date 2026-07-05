# Optimization roadmap — July 2026 research synthesis

Three-track research (git history / live code audit / web best-practices), 2026-07-06.
Hardware: Ryzen 9700X (8P/16L), 32 GB host (WSL sees ~25.4 GB via existing `.wslconfig`
memory=26GB), RTX 5060 Ti 16 GB Blackwell sm_120 (36 SMs), data+caches on WSL ext4 NVMe.

## Headline diagnosis (measured live during an OMAR-RQ extraction)

GPU at **98–100% SM, 96% of 180 W power cap — doing plain fp32 on CUDA cores**.
`extract.py` historically never set TF32 matmul precision and used no autocast (only
`cli.py` does, for Lightning runs). Tensor cores (fp16/bf16 ≈ 2× fp32: ~47 vs ~24
TFLOPS, 448 GB/s bus) were idle in every extraction ever run. Batch size is NOT the
lever (4.2/16 GB used, compute-saturated); CPU 93% idle. **Precision is the lever.**

Facts that corrected prior beliefs:
- WSL RAM = 25.4 GB (a `.wslconfig` exists; the "15.6 GB, no .wslconfig" note was stale).
- HookTheory WAVs are already 24 kHz mono PCM → waveform caching buys nothing; the
  encoder forward is ~100% of HTM's cost.
- MedleyDB configs had `cache_embeddings: false` → pending MERT/OMARRQ runs would
  re-encode every batch × 40 epochs × 13–24 layers × 5 folds. Frame caches are only
  13–30 GB fp16/encoder.
- "Not enough SMs" torch.compile warning is expected (Inductor wants ≥68 SMs for
  autotuned GEMMs; card has 36). Use default/reduce-overhead, never max-autotune.
- max-autotune can be *slower* than eager on small-SM GPUs.

## Ranked plan (status as of 2026-07-06)

| # | Optimization | Gain | Status |
|---|---|---|---|
| 1 | **Multi-head parallel probing** — one run trains all 13–24 layer heads off the shared frozen forward. Math exact: parameter-disjoint heads + per-param Adam ⇒ summed loss = independent runs (proven bitwise in tests/test_multihead_probe.py) | 10–20× on 195 pending MedleyDB runs; **only route to HookTheoryMelody** (frame cache is 1.9–4.3 TB fp16/encoder — infeasible; int8 still 1–2 TB) | Implemented on worktree commit `3b8e9dc` (branch worktree-agent-ad88cfd1a7871e733); adversarial review + GPU fold0 validation pending |
| 2 | **TF32 + bf16 autocast extraction** (`extract.py --precision {fp32,tf32,bf16}`, default bf16, inference_mode) + TF32 in `vgm_timbre_sweep_from_cache.py` sim matmuls | 2–4× extraction; 1.3–2× metric stage | Committed `7a6e825`; A/B guardrail staged (`/home/sid/ab_prepare_run.sh` + `ab_compare.py` on PC), gates CLaMP3 |
| 3 | **MedleyDB frame cache ON** (6 configs) + fold launcher NUM_WORKERS 0→6 (WSL-deadlock note was stale — 8 fork workers verified fine) | 4–8× per run, multiplies with #1 | Committed `7a6e825`; needs one fold0 cache-ON vs committed cache-OFF validation run |
| 4 | Per-job overhead: `--skip-fit-if-no-train` always for max_epochs=0; CachedStubEncoder for cache-complete jobs (skip HF load, 20–60 s + 2.5 GB/process) | ~4–10 h over OMARRQ×MedleyDB | TODO |
| 5 | Consolidated fp16 memmap per encoder×task (replace 350k tiny .pt; keep per-clip as write format, consolidate post-hoc; move cache reads into DataLoader workers — today get_batch blocks the main process) | 5–10× cold reads, ~10% disk | TODO |
| 6 | int8 frame caches (per-dim scale; published retrieval retention 97–100%, HF/mixedbread + FAISS SQ8) | 2× under fp16 | TODO — gate on one (layer,fold) A/B |
| 7 | MERT SDPA (MusicHubert hardcodes eager attention; MuQ already uses F.scaled_dot_product_attention) | 1.2–1.5× MERT forwards | TODO |
| 8 | CLaMP3/MusicFM compile_mode (0/28, 0/11 configs; recipe in docs/performance_optimizations.md) | ~1.2× | TODO before CLaMP3 extraction if cheap |
| 9 | Minor: zero-waveform placeholder on cache hits collated+shipped to GPU (~1–5%/cached epoch); `emb_cache.get()` harmless double-upcast; zstd on fp16 not worth it | small | noted |

## Decisions locked (user, 2026-07-06)

- Multi-head: build now; validate vs committed MuQ fold0 anchors (L1 RPA .638; L11 .557±.040; meanall .635±.044; ±0.01 tolerance) before MERT/OMARRQ/HTM use.
- Extraction: bf16 default with A/B guardrail (256 clips, min per-layer cos ≥ 0.999). OMAR finished pure-fp32 (no mixed-precision cache); CLaMP3 = first bf16 extraction.
- Metric stage: **TF32 everywhere** — re-run MuQ+MERT metric stages under TF32 and diff vs committed CSVs (uniformity record; committed numbers must not move at table precision, threshold 2e-3).
- MedleyDB config flips: applied + committed.
- Correctness bar (thesis): every change gated by explicit verification; fp32 archive on D: (`emb_cache_archive`, 64 GB) is the rollback until all audits pass.

## Verification matrix status

Done: 14 cache unit tests (Mac); **107/107 cache+retrieval tests on PC** with all new
code; fp16 migration integrity vs D: backup (max|Δ|=3.6e-3); fp16 ranking invariance
(top-50 overlap 0.9996–0.9999); OMAR live extraction writes fp16 `(24,1024)`;
multi-head bitwise equivalence + 13/13 tests + byte-identical config resolution.

Pending gates (scripts on PC at /home/sid/):
1. `ab_prepare_run.sh` + `ab_compare.py` — bf16 A/B (gates CLaMP3 bf16).
2. `muq_tf32_audit.sh` — MuQ layers {0,8,11,12,meanall} from migrated fp16 cache with
   TF32 vs committed summary_table.csv (gates OMAR sweep launch). Also locates the
   fp16 migration's `errors=1` file (one of 347,325 failed torch.load — must identify;
   re-extract that clip if damaged).
3. Post-OMAR-sweep: full MuQ (13+meanall) + full MERT TF32 re-runs vs both committed CSVs.
4. Multi-head: adversarial review verdict → merge → GPU fold0 validation.
5. MedleyDB cache-ON: one fold0 layer retrain vs committed number (seed-noise tolerance).

## Historical record (compiled from 406 commits, May–July 2026)

Phase 0 (May 7–13, no caching): parallel Modal (~175 h→~9 h), bf16-mixed+TF32 in
Lightning configs, skip-completed/resume. Phase 1 (May 14–20): (L,H) clip cache +
mean-commutativity (SHS100K ~13.5 h→~60 min), audio-I/O bypass, frame-level variant.
Phase 2–3 (May 22–24): auto-resume; frame batch 8→16; torch.compile rollout 142
configs (+20–23%, caught OMARRQ compile no-op via bit-identical metrics); MP3→WAV
(dataloader 84.7%→1.8% of runtime); JSONL metadata cache (20 min→<1 s); prefetch 2→4.
Phase 4 (May 25–28): FLAC render (312→145 GB); metric OOM arc: 84 GB argsort→1.7 GB,
grid 4 h→3 min, GPU streaming 7.9×. Phase 5–6 (Jun–Jul): RAM LRU (epoch 89→52 s);
autotune_concurrency; encoder-strip ckpts (2210 MB→0.87 MB, sweep 286 GB→113 MB);
fused single-pass GPU metrics + offline from-cache sweep (~50 min/layer→193 s);
fp16 on-disk cache (76→~57 GB done; supervised caches pending). Never landed:
int8 cache (TODO §3), CLaMP3/MusicFM compile, launcher silent-failure fix,
retrieval-datamodule unification, batch 16→32 (reverted for comparability — revisit
for extraction only).

## Key web-research references

PyTorch ≥2.7+cu128 required for sm_120 (prefer latest stable + cu13x). CUDA graphs
(reduce-overhead) worth more on WSL2 (launch latency is the main WSL overhead;
compute within ~1–10% of native). DataLoader: 6–8 workers, persistent, prefetch 2–4;
pin_memory must be A/B'd on WSL2 (cudaHostAlloc limits). SUPERB/s3prl = prior art for
one-forward-many-probes; Zaiem et al. (Interspeech 2023): probe architecture changes
model rankings → keep probes identical across layers/encoders (PerLayerHeads does).
Storage: monolithic fp16 memmap 3–10× I/O vs small files; int8 embeddings 97–100%
retrieval retention; binary too lossy; zstd on fp16 ~10–25% only. fstrim for vhdx
churn; `wsl --manage --resize` if ext4 needs >1 TB later.
