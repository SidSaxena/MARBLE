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
| [data/supermario_setup.md](data/supermario_setup.md) | Super Mario VGM structure dataset (554 NSM MIDIs + 6-class annotations) |
| [data/hooktheory_melody_setup.md](data/hooktheory_melody_setup.md) | HookTheoryMelody full-song audio (HF download + optional yt-dlp recovery of missing ytids + Modal upload + sweep) |

## Research / design notes

| Doc | What it covers |
|---|---|
| [layer_analysis.md](layer_analysis.md) | **Cross-encoder, cross-task layer-selection guide** — the canonical reference for which layer of which encoder to use for which task |
| [vgm_encoder_selection.md](vgm_encoder_selection.md) | **Deployment-decision guide for VGM** — which encoder + layer to pick when you have audio only / MIDI only / both, plus workarounds (basic-pitch transcription, MIDI rendering) and the tests needed to firm up open recommendations |
| [leitmotif_findings.md](leitmotif_findings.md) | VGMIDITVar-leitmotif cross-instrument retrieval results — primary motif-analysis study |
| [supermario_findings.md](supermario_findings.md) | SuperMarioStructure CLaMP3-symbolic sweep — first symbolic-classification result, two-peak L4/L11, L12 cross-modality discussion |
| [leitmotif_clamp3.md](leitmotif_clamp3.md) | CLaMP3-based leitmotif detection — design, datasets, metrics, sweep wiring |
| [leitmotif_swtc.md](leitmotif_swtc.md) | SWTC standalone evaluation pipeline |
| [structure_datasets_survey.md](structure_datasets_survey.md) | Survey of music structure datasets MARBLE supports / could add |
| [symbolic_encoder_landscape.md](symbolic_encoder_landscape.md) | Survey of symbolic / MIDI-native music encoders relevant to this benchmark |
| [benchmarking_methodology.md](benchmarking_methodology.md) | How we evaluate probes — MLP-probe vs zero-shot retrieval, metric choices, fairness considerations |

## Infrastructure / performance

| Doc | What it covers |
|---|---|
| [embedding_cache.md](embedding_cache.md) | Per-clip embedding cache architecture, when to enable, cache-key derivation |
| [embedding_cache_correctness.md](embedding_cache_correctness.md) | Correctness checks + invariants for the cache (hit/miss semantics, hashing) |
| [performance_optimizations.md](performance_optimizations.md) | Repo-wide perf tricks — compile modes, data-pipeline scripts, diagnostics |
| [optimization_uniformity_audit.md](optimization_uniformity_audit.md) | Audit of per-encoder/per-task optimization parity for cross-encoder fairness |
| [nsynth_optimization_plan.md](nsynth_optimization_plan.md) | NSynth-specific data + training optimization plan |
| [TODO.md](TODO.md) | Open follow-ups — deferred experiments and design questions |

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
