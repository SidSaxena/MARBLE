# MARBLE docs index

Operational and design documentation for running MARBLE layer-probe
sweeps locally and on Modal. Code lives in
[`scripts/`](../scripts/), Modal entry point in
[`modal_marble.py`](../modal_marble.py).

## Start here

| Doc | What it covers |
|---|---|
| [workflow.md](workflow.md) | **End-to-end runbook** — phase 0 setup → phase 5 result retrieval. Copy-pasteable commands. Read this first. |
| [local_sweeps.md](local_sweeps.md) | PC workflow: `--concurrency`, MPS, the ffmpeg-on-Windows trap, per-layer logs |
| [modal_sweeps.md](modal_sweeps.md) | Modal workflow: data setup, smoke test, tier-prioritized batch runs, cost guardrails |

## Per-dataset acquisition

| Doc | Dataset |
|---|---|
| [data/shs100k.md](data/shs100k.md) | SHS100K cover-retrieval (.m4a → .flac conversion path) |
| [data/hooktheory.md](data/hooktheory.md) | HookTheory clips (Key/Structure) + full audio (Melody) |
| [data/vgmiditvar_setup.md](data/vgmiditvar_setup.md) | VGMIDI-TVar render-from-MIDI pipeline |

## Research / design notes

| Doc | What it covers |
|---|---|
| [leitmotif_clamp3.md](leitmotif_clamp3.md) | CLaMP3-based leitmotif detection — design, datasets, metrics, sweep wiring |
| [leitmotif_swtc.md](leitmotif_swtc.md) | SWTC standalone evaluation pipeline |

## Quick reference — common commands

```bash
# See completed sweeps + best layer per group
uv run python scripts/analysis/best_layer.py

# Drill into one group's full per-layer ranking
uv run python scripts/analysis/best_layer.py --group "MERT-v1-95M / HookTheoryKey"

# Run a local sweep (sequential, default)
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey

# Run a local sweep with 2× concurrency (16 GB GPU sweet spot)
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey \
    --concurrency 2
# Live output prefixed with [L0], [L1]; per-layer logs at output/logs/...

# Run a Modal sweep (parallel containers, one per layer)
modal run scripts/sweeps/modal/modal_sweep.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey
```
