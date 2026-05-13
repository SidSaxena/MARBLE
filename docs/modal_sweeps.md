# Modal parallel layer sweeps

Runbook for executing MARBLE layer sweeps on Modal in parallel —
one container per layer — alongside sweeps running on the PC.

## Why Modal (and the cost reality)

Same total GPU-hours as running sequentially on one machine, but
wall-clock collapses from N × single-layer-time to ~single-layer-time
by spawning N containers in parallel via `run_one_layer.starmap()`.

Default GPU: **L4** (24 GB, Ada Lovelace, bf16 native), $0.80/hr —
chosen over A10G ($1.10/hr) on price/perf grounds. L4 is ~5% faster than
A10G for our workload (frozen encoder + small MLP probe, ~5–6 GB VRAM
per layer) and 27% cheaper. To override, edit the `gpu="L4"` line in
[modal_marble.py:`run_one_layer`](../modal_marble.py).

### Cost estimates (L4 spot, parallel)

Each 13-layer MERT/CLaMP3 sweep ≈ 1.5–2.5 h wall-clock at ~$10–15.
Each 24-layer OMARRQ sweep ≈ 1.5–2.5 h wall-clock at ~$18–28.

Priority-ordered plan (Tier 1 = MERT first, Tier 2 = CLaMP3, Tier 3 =
OMARRQ deferred; GS skipped):

| Tier | Sweep | Layers | Est. wall | Est. cost |
|---|---|---:|---:|---:|
| 1 | MERT × SHS100K | 13 | ~0.5 h | ~$5 (zero-shot) |
| 1 | MERT × HookTheoryMelody | 13 | ~2 h | ~$15 |
| 1 | MERT × HookTheoryStructure | 13 | ~1.5 h | ~$12 |
| 1 | MERT × HookTheoryKey | 13 | ~1.5 h | ~$12 |
| 1 | MERT × GTZANBeatTracking | 13 | ~2 h | ~$15 |
| 1 | MERT × Chords1217 | 13 | ~2.5 h | ~$20 |
| 1 | MERT × NSynth | 13 | ~2.5 h | ~$20 |
| 2 | CLaMP3 × SHS100K | 13 | ~0.5 h | ~$5 |
| 2 | CLaMP3 × HookTheoryKey | 13 | ~1.5 h | ~$12 |
| 2 | CLaMP3 × HookTheoryStructure | 13 | ~1.5 h | ~$12 |
| 2 | CLaMP3 × NSynth | 13 | ~2.5 h | ~$20 |
| 3 | OMARRQ × … (5 sweeps) | 24 ea. | ~12 h | ~$120 |

Tier 1+2 ≈ ~$150. Tier 3 adds ~$120. Full plan ≈ $270.

---

## Phase 0 — Secrets & volumes (one-time)

```bash
# Secrets — both must exist before any sweep
modal secret create wandb-secret WANDB_API_KEY=<your-key>
modal secret create huggingface HF_TOKEN=hf_...
modal secret list

# Volumes (auto-created on first use, verify state)
modal volume ls marble-data /
modal volume ls marble-output /
```

### Current Modal volume state (as of last audit)

| Dataset | Path | Status |
|---|---|---|
| SHS100K audio | `/SHS100K/audio/` | ✓ 6905 .m4a files |
| SHS100K JSONL | `/SHS100K/SHS100K.test.jsonl` | ✗ needs upload + rebuild (Phase 1.1) |
| HookTheory | `/HookTheory/` | ✗ not present (Phase 1.2) |
| NSynth | `/NSynth/` | ✗ not present (Phase 1.3) |
| GTZAN | `/GTZAN/` | ✗ not present (Phase 1.3) |
| Chords1217 | `/Chords1217/` | ✗ not present (Phase 1.3) |

---

## Phase 1 — Per-dataset data setup

### 1.1 SHS100K — rewrite JSONL on Modal (~15 min)

Audio is already on the volume. We need a clean JSONL that points at the
Modal mount path, with missing/corrupt entries dropped.

```bash
# 1. One-time: upload the local SHS100K.test.jsonl
modal volume put marble-data \
    data/SHS100K/SHS100K.test.jsonl \
    SHS100K/SHS100K.test.jsonl

# 2. Rewrite + verify on Modal (uses scripts/verify/verify_shs100k.py --rewrite
#    against the mounted audio dir)
modal run modal_marble.py::setup_shs100k_jsonl
```

What happens: the function loads the uploaded JSONL, rewrites each
`audio_path` to `data/SHS100K/audio/<ytid>.m4a` (Modal mount), runs
`torchaudio.info` on each file, drops failures, commits the cleaned
JSONL back to the volume.

### 1.2 HookTheory — full download + Melody JSONL build (~3–6 h, ~110 GB)

```bash
# Downloads m-a-p/HookTheory complete (clips + 104 GB full audio), extracts,
# then builds HookTheory.{train,val,test}.jsonl for the Melody task.
modal run modal_marble.py::setup_hooktheory_full
```

This is the heavy one. The function uses `huggingface_hub.snapshot_download`
which is restart-safe — if it dies mid-download, re-running picks up where
it left off.

Prerequisite: your HF account must have accepted the m-a-p/HookTheory terms
(check at https://huggingface.co/datasets/m-a-p/HookTheory).

### 1.3 Other datasets — download via existing scripts

For NSynth / GTZAN / Chords1217:

```bash
# Reuse the existing dataset-download function for the m-a-p/* repos
modal run modal_marble.py::download_gs_emo                       # (GS, EMO)
modal run modal_marble.py::download_gtzan_only                   # GTZAN, EMO, MTG, MTT, GS
modal run modal_marble.py::download                              # GTZAN + Chords1217 + GS + EMO + Covers80
# NSynth doesn't have an m-a-p/* repo — see download_nsynth.py for the path.
```

---

## Phase 2 — Smoke test (~$5, ~30 min)

After Phase 1.1, kick off SHS100K as the smoke test. It's zero-shot
retrieval (`max_epochs=0`) so test-only, fast, cheap:

```bash
modal run scripts/sweeps/modal/modal_sweep.py \
    --base-config configs/probe.MERT-v1-95M-layers.SHS100K.yaml \
    --num-layers 13 \
    --model-tag MERT-v1-95M \
    --task-tag SHS100K
```

Pass criteria:

- 13 L4 containers spin up in parallel (visible in Modal dashboard).
- WandB project `marble` shows 13 runs in group `MERT-v1-95M / SHS100K`,
  each with `test/MAP` and `test/MRR`.
- `marble-output` volume has
  `output/probe.SHS100K.MERT-v1-95M-layers/layer-{0..12}/wandb-summary.json`.

If smoke test passes, proceed to Phase 3.

---

## Phase 3 — Tier-1 + Tier-2 sweeps (MERT + CLaMP3, ~$150)

```bash
# All MERT sweeps (~7 sweeps, ~$100)
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --tier 1

# All CLaMP3 sweeps (~4 sweeps, ~$50)
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --tier 2

# Or one at a time
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --only MERT-NSynth
```

Resume-safe: re-running a finished sweep is free — `_layer_done`
detects completion via the wandb-summary on the output volume and
prints `skipped`.

---

## Phase 4 — Tier-3 OMARRQ (deferred decision point, ~$120)

After Tier 1+2 results are in, decide whether to spend on OMARRQ:

```bash
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --tier 3
```

Each OMARRQ sweep is 24 layers (vs 13 for MERT/CLaMP3), so cost
scales accordingly.

---

## Phase 5 — Result retrieval

WandB metrics flow online automatically (since `WANDB_API_KEY` is bound).
For artifacts:

```bash
# All output dirs
modal volume get marble-output output ./output --recursive

# Or per-sweep
modal volume get marble-output \
    output/probe.NSynth.MERT-v1-95M-layers \
    ./output/probe.NSynth.MERT-v1-95M-layers \
    --recursive
```

---

## Resume behaviour

Re-running a completed sweep is a no-op for finished layers — the
`_layer_done` check (mirrored from
[scripts/sweeps/run_sweep_local.py:97](../scripts/sweeps/run_sweep_local.py)) inspects
the WandB summary JSON on the `marble-output` volume. `output_vol.reload()`
runs at the top of each container so cross-container visibility is
consistent.

To force re-run: delete the per-layer output dir before re-submitting.

---

## Cost guardrails

- **Set a hard cap** in Modal dashboard → Settings → Usage → spending
  limit. Recommend $300 for the full Tier 1+2+3 plan.
- **Per-layer timeout**: 4 h, set in `run_one_layer`. Hung containers
  get terminated.
- **Retries**: `max_retries=2, backoff=2.0` for transient image-pull /
  OOM errors.

---

## Switching to an even cheaper SKU (optional)

If you want to trade speed for cost, change `gpu="L4"` → `gpu="T4"` in
[modal_marble.py:`run_one_layer`](../modal_marble.py). T4 (Turing) is
$0.59/hr (~26% cheaper than L4) but ~1.6× slower per layer and **does
not support bf16** — you'd need to inject
`--trainer.precision=16-mixed` into the fit/test commands.

Per-sweep cost roughly:
- L4: $10–15 (13 layers) / $18–28 (24 layers) — recommended default
- T4: $7–11 / $13–20 — cheaper but slower; needs fp16 patch
- A10G: $14–22 / $25–40 — original default; bf16 native; redundant given L4
- A100 40GB: $25–40 / $45–70 — overkill

---

## Risks & known gotchas

1. **HF gated terms** — Chords1217, HookTheory, NSynth require accepting
   the m-a-p org terms on HuggingFace before snapshot_download will work
   (you've confirmed this is done).

2. **HookTheory full audio download is long** — ~104 GB at Modal's HF
   bandwidth (~200 MB/s) ≈ 9 min just for transfer, plus extraction.
   Budget 30–60 min for `setup_hooktheory_full`.

3. **Volume eventual consistency** — `output_vol.reload()` at the start
   of `run_one_layer` handles cross-container visibility. Back-to-back
   re-submissions within seconds of a prior completion may still race —
   wait 30 s between submissions if you see false skip misses.

4. **PyTorch version skew** — Modal image pins `torch==2.6.0`;
   `pyproject.toml` (PC) pins `torch==2.7.0` for Blackwell support.
   For frozen-encoder probes the numerical drift is below 1e-4 — fine
   for layer-rank comparisons.

5. **L4 quota** — Modal supports L4 broadly but in tight regions it
   may queue. If you see "no capacity" errors, fall back to A10G.

---

## Critical files

| File | Role |
|---|---|
| [modal_marble.py](../modal_marble.py) | Modal app + `run_one_layer` (parallel unit) + `run_parallel_sweep` + data-setup functions |
| [scripts/sweeps/modal/modal_sweep.py](../scripts/sweeps/modal/modal_sweep.py) | CLI bridge — `modal run` wrapper around `run_parallel_sweep` |
| [scripts/sweeps/modal/modal_run_all_sweeps.py](../scripts/sweeps/modal/modal_run_all_sweeps.py) | Tier-prioritized orchestrator for the full migration |
| [scripts/data/build_hooktheory_melody_jsonl.py](../scripts/data/build_hooktheory_melody_jsonl.py) | Builds `HookTheory.{train,val,test}.jsonl` from upstream raw data |
| [scripts/verify/verify_shs100k.py](../scripts/verify/verify_shs100k.py) | SHS100K audit (`--rewrite` mode drops bad entries + repoints audio paths) |
| [download.py](../download.py) | HF dataset download — `download_dataset(..., with_full_audio=True)` for HookTheoryMelody |
