# Modal parallel layer sweeps

Runbook for executing the slow MARBLE layer sweeps on Modal in parallel —
one container per layer — while the PC handles the rest.

## Why Modal (and the cost reality)

Same total GPU-hours as running sequentially on one machine, but
wall-clock collapses from N × single-layer-time to ~single-layer-time
by spawning N containers in parallel via `run_one_layer.starmap()`.

A10G ≈ $1.10/hr. For a 24-layer OMARRQ sweep at ~1–2 h/layer:

- Sequential on PC: 24–48 h wall-clock
- Sequential on Modal: 24–48 h × $1.10 ≈ $30–55
- **Parallel on Modal: 1–2 h wall-clock, ~$30–55** (same cost, way less waiting)

The 5 migrated sweeps total ~9 h wall-clock and ~$190 spend.

---

## Phase 0 — One-time setup

```bash
# Secrets — both must exist before any Modal sweep can run
modal secret create wandb-secret WANDB_API_KEY=<your-key>
modal secret create huggingface HF_TOKEN=hf_...   # only if not already there
modal secret list

# Volumes (auto-created on first use, but verify)
modal volume ls marble-data /
modal volume ls marble-output /
```

If `wandb-secret` already exists with a different name, either rename it
or update the `Secret.from_name("wandb-secret")` references in
`modal_marble.py`.

### Verify datasets are on `marble-data`

The 5 migrated sweeps need:

| Dataset | Path on `marble-data` | Used by |
|---|---|---|
| NSynth | `NSynth/{train,valid,test}.jsonl` + audio | OMARRQ/MERT × NSynth |
| Chords1217 | `Chords1217/Chords1217.{train,val,test}.jsonl` + audio | OMARRQ/MERT × Chords1217 |
| GTZAN | `GTZAN/GTZANBeatTracking.{train,val,test}.jsonl` + audio | OMARRQ × GTZANBeatTracking |

To check:

```bash
modal volume ls marble-data NSynth | head
modal volume ls marble-data Chords1217 | head
modal volume ls marble-data GTZAN | head
```

If anything is missing, populate via:

```bash
modal run modal_marble.py::download   # GTZAN + Chords1217 + GS + EMO + Covers80
# NSynth has its own path — upload manually if needed:
modal volume put marble-data /path/to/local/NSynth NSynth
```

---

## Phase 1 — Smoke test (~$2, ~45 min)

Before kicking off the 5-sweep migration, validate the parallel path on
the cheapest sweep:

```bash
# 1. One layer to verify image + secrets + data path
modal run scripts/modal_sweep.py \
    --base-config configs/probe.CLaMP3-layers.GS.yaml \
    --num-layers 13 \
    --model-tag CLaMP3 \
    --task-tag GS \
    --layers 0

# Expected: ~30 min, ~$0.50, WandB shows a CLaMP3/GS run with `test/*` metrics.

# 2. Four layers in parallel — confirm 4 simultaneous A10G containers
modal run scripts/modal_sweep.py \
    --base-config configs/probe.CLaMP3-layers.GS.yaml \
    --num-layers 13 \
    --model-tag CLaMP3 \
    --task-tag GS \
    --layers 0,1,2,3

# Expected: 4 containers up at once in the Modal dashboard; total wall-clock
# ≈ one layer's time; ~$2 cost.

# 3. Resume test — every layer should report "skipped"
modal run scripts/modal_sweep.py \
    --base-config configs/probe.CLaMP3-layers.GS.yaml \
    --num-layers 13 --model-tag CLaMP3 --task-tag GS --layers 0,1,2,3
```

Pass criteria:

- WandB project `marble` shows the 4 runs with `test/weighted_score` and
  the group/tags from `gen_sweep_configs.py` (look for `CLaMP3 / GS` group).
- `marble-output` volume contains
  `output/probe.GS.CLaMP3-layers/layer-*/wandb/run-*/files/wandb-summary.json`.
- Step 3 prints all `skipped`, no fit/test was re-run.

---

## Phase 2 — Migration: 5 long sweeps

```bash
# All 5, sequential between sweeps, parallel within each
modal run scripts/modal_run_all_sweeps.py

# Or pick one
modal run scripts/modal_run_all_sweeps.py --only OMARRQ-NSynth

# Or skip ones already done elsewhere
modal run scripts/modal_run_all_sweeps.py --skip MERT-NSynth,MERT-Chords1217
```

Watch from the Modal dashboard: each sweep spawns 13 or 24 A10G containers
all at once, all finish within ~the slowest layer's time. With Modal's
default concurrency (≥1000), we never hit the cap.

Recommended order if running interactively (cheapest → most expensive so
you can stop and assess after each):

1. `MERT-Chords1217` (~$28, ~2 h)
2. `MERT-NSynth` (~$33, ~2.5 h)
3. `OMARRQ-GTZANBeatTracking` (~$33, ~1.5 h)
4. `OMARRQ-Chords1217` (~$44, ~2 h)
5. `OMARRQ-NSynth` (~$55, ~2.5 h)

---

## Phase 3 — Retrieve results

WandB metrics flow online automatically (since `WANDB_API_KEY` is bound).
For the offline artifacts (checkpoints, raw test outputs):

```bash
# Pull just the wandb summaries (small)
modal volume get marble-output \
    output/probe.NSynth.OMARRQ-multifeature25hz-layers \
    ./output/probe.NSynth.OMARRQ-multifeature25hz-layers \
    --recursive

# To download every result:
modal volume get marble-output output ./output --recursive
```

---

## Resume behaviour

Both `modal_sweep.py` and `modal_run_all_sweeps.py` re-running the same
sweep is a no-op for completed layers — the `_layer_done` check (mirrored
from `scripts/run_sweep_local.py:97`) looks at the WandB summary JSON on
the `marble-output` volume.

The volume is reloaded inside each container before the check, so:

- If the PC also ran some layers and synced outputs to the volume, Modal
  will skip them.
- Conversely, if Modal completed layers and you pull `output/` to the
  PC, `run_sweep_local.py` on the PC will also skip them.

To force re-run, delete the per-layer output dir before re-submitting.

---

## Cost guardrails

- **Set a hard cap** in Modal dashboard → Settings → Usage → spending
  limit. Recommend $250 for the 5-sweep migration ($190 expected +
  30% buffer).
- **Per-layer timeout**: 4 h, set in `run_one_layer`. Any hung container
  gets terminated; Modal stops billing once terminated.
- **Retries**: `max_retries=2, backoff=2.0` on `run_one_layer` for
  transient image-pull / OOM errors.

---

## Switching to a cheaper SKU (optional)

If $190 is more than you want to spend, edit `modal_marble.py`'s
`run_one_layer` decorator: change `gpu="A10G"` → `gpu="T4"`. T4 is
~40% cheaper but ~1.6× slower per layer. T4 doesn't support bf16, so
you'd also need to inject `--trainer.precision=16-mixed` into the fit
and test commands inside `run_one_layer`. Expected: ~$110 total,
~14 h wall-clock instead of ~9 h.

---

## Risks & known gotchas

1. **HF gated terms** — Chords1217 and HookTheory require accepting the
   m-a-p org terms on HuggingFace. Do this in your HF account *before*
   running OMARRQ × Chords1217 etc., or the dataset download silently
   fails inside the container.

2. **Volume eventual consistency** — `output_vol.reload()` at the start of
   `run_one_layer` handles cross-container visibility, but back-to-back
   re-submissions within seconds of a prior completion may still race.
   Wait 30 s between submissions if you see false skip misses.

3. **PyTorch version skew** — Modal image pins `torch==2.6.0`;
   `pyproject.toml` (PC) pins `torch==2.7.0` for Blackwell support.
   For frozen-encoder probes the numerical drift is below 1e-4 — fine
   for layer-rank comparisons. If exact-match is required, bump the
   Modal image to 2.7.0 + cu124 wheels.

4. **`omar-rq` floating ref** — `modal_marble.py` installs
   `git+https://github.com/MTG/omar-rq.git` without a commit pin.
   Modal caches the image so a rebuild is rare, but if you bump the
   image for any reason the omar-rq install could pick up an
   incompatible upstream commit. Pin a sha when this becomes a
   problem.

---

## Critical files

| File | Role |
|---|---|
| `modal_marble.py` | Modal app definition + `run_one_layer` (parallel unit) + `run_parallel_sweep` (orchestrator) |
| `scripts/modal_sweep.py` | CLI bridge — `modal run` wrapper around `run_parallel_sweep` |
| `scripts/modal_run_all_sweeps.py` | Orchestrator for the 5 migrated sweeps |
| `scripts/run_sweep_local.py` | PC equivalent — runs the other 17 sweeps |
| `scripts/run_all_sweeps.py` | PC orchestrator for all 22 (use `--skip` for the Modal 5) |
