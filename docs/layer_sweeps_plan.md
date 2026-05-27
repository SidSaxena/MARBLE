# Plan: full layer sweeps — SHS100K + VGMIDITVar-timbre × 4 encoders

## Goal

Run per-layer probes for the four production audio encoders on the two
retrieval tasks that have warm caches:

- **Encoders:** CLaMP3, MERT-v1-95M, MuQ, OMARRQ-multifeature-25hz
- **Tasks:** SHS100K, VGMIDITVar-timbre

This is the canonical layer-comparison sweep we'd report in a paper —
which transformer layer of each encoder best captures cover-song
identity (SHS100K) and cross-timbre theme retrieval
(VGMIDITVar-timbre).

## Pre-conditions

### ✅ Already satisfied

- All retrieval-task probe code on `main` (commit `999cf1c`):
  - OOM fix + row-batched bundle (`9b050d6`, `044c912`, `aab5aa1`)
  - UTF-8 JSONL load fix (`c0082ed`)
  - Per-cell condition-grid logging (`f3f3097`, `ca7e634`)
  - Methodology doc + SHS100K skew note (`999cf1c`)
- 49 retrieval-related tests pass (`tests/test_retrieval_metrics.py`
  + `tests/test_cover_retrieval_integration.py` +
  `tests/test_compute_map_self_exclusion.py` +
  `tests/test_emb_cache_slugs.py` +
  `tests/test_verify_retrieval_jsonl.py` +
  `tests/test_reconstruct_condition_grid.py`)
- SHS100K JSONL clean: 6,821 records, 100% at 24 kHz, dedup'd
- VGMIDITVar-timbre JSONL: 102,960 records, untouched since Phase 1
- Embedding caches warm for **all 8 (encoder × task) combinations** from
  the 2026-05-27 meanall pass — layer sweeps reuse the same
  `output/.emb_cache/<encoder>/<task>__<hash>/` directories because the
  config_hash depends only on `(encoder_model_id, sample_rate,
  clip_seconds, pipeline_signature, pool_time)` — the LayerSelector
  mode change (`mean` ↔ `single, layer=N`) is in `emb_transforms`, NOT
  the cache-key inputs. Verified by inspecting the cache directory hash
  names per encoder.
- Disk: ~33 GB free on my-pc; sweep adds nothing to the cache.

### 🔄 Open question — CLaMP3 × VGMIDITVar-timbre wandb-core failure

The 2026-05-27 meanall pass hit a persistent
`wandb-core exited with code 3221225794` (Windows
`STATUS_DLL_INIT_FAILED` desktop-heap exhaustion) on every attempt at
CLaMP3 × VGMIDITVar-timbre. Same class of bug as the earlier
ffmpeg-spawn issue. The other 7 combinations passed.

**Mitigation options** (ordered by recommendation):
1. **`WANDB_MODE=offline` for the failing combo.** Skips the wandb
   live-sync network service. Local wandb dir still captures all
   metrics; sync to wandb cloud after the sweep with `wandb sync`.
2. Reboot my-pc before launching the sweep — resets desktop heap.
   Brute but reliable.
3. Wrap launchers in a retry loop that sleeps 60 s and retries on rc
   3221225794. Empirical recovery rate unclear; may waste hours.

**Recommendation:** Reboot before launch (option 2). Cheap insurance
against ALL 8 runs hitting the same wall partway through; covers the
CLaMP3 × VGMIDITVar-timbre case automatically.

## Sweep matrix

| Encoder | Layers | SHS100K | VGMIDITVar-timbre |
|---|---|---|---|
| CLaMP3 | 13 (0–12) | `configs/probe.CLaMP3-layers.SHS100K.yaml` | `configs/probe.CLaMP3-layers.VGMIDITVar-timbre.yaml` |
| MERT-v1-95M | 13 (0–12) | `configs/probe.MERT-v1-95M-layers.SHS100K.yaml` | `configs/probe.MERT-v1-95M-layers.VGMIDITVar-timbre.yaml` |
| MuQ | 13 (0–12) | `configs/probe.MuQ-layers.SHS100K.yaml` | `configs/probe.MuQ-layers.VGMIDITVar-timbre.yaml` |
| OMARRQ-25hz | 25 (0–24) | `configs/probe.OMARRQ-multifeature-25hz-layers.SHS100K.yaml` | `configs/probe.OMARRQ-multifeature-25hz-layers.VGMIDITVar-timbre.yaml` |

**Total sweep count:** 13 + 13 + 13 + 25 = **64 layer-runs per task**;
**128 layer-runs across both tasks**.

## Cost estimate

All runs operate on **warm** caches (from the just-finished meanall
pass).

### VGMIDITVar-timbre per layer (N=102,960)

- DataLoader pass with cache hits: ~5 min
- Bundle pass (raw): ~3 min (per the perf-optimisation benchmark)
- Bundle pass (centered): ~3 min
- `compute_perpair_map_all` grid: ~3 min
- Per-cell condition grid logging + CSV/PNG write: ~5 s
- Anisotropy: ~5 s
- **Per layer: ~12 min**

| Encoder | Layers | Time |
|---|---|---|
| CLaMP3 | 13 | ~2.6 h |
| MERT-v1-95M | 13 | ~2.6 h |
| MuQ | 13 | ~2.6 h |
| OMARRQ-25hz | 25 | ~5 h |
| **VGMIDITVar-timbre subtotal** | | **~12.8 h** |

### SHS100K per layer (N=6,821)

- DataLoader pass with cache hits: ~1 min
- Bundle pass (raw): ~5 s (N² is tiny)
- Bundle pass (centered): ~5 s
- No condition grid (SHS100K has no conditions)
- Anisotropy: <1 s
- **Per layer: ~1.5 min**

| Encoder | Layers | Time |
|---|---|---|
| CLaMP3 | 13 | ~20 min |
| MERT-v1-95M | 13 | ~20 min |
| MuQ | 13 | ~20 min |
| OMARRQ-25hz | 25 | ~40 min |
| **SHS100K subtotal** | | **~1.7 h** |

### Total wall-time

**~14.5 hours.** Overnight + half a day. Realistic: budget 20 h
end-to-end for the inevitable retry of any wandb-core glitch and
launcher overhead.

## Launch design

A single bash launcher mirroring the earlier meanall pattern:

```bash
# my-pc:~/Developer/Python/marble/run_layer_sweeps.sh
#
# Sweep order: SHS100K first (faster, validates the launcher), then
# VGMIDITVar-timbre. Within each task, OMARRQ last so its 25-layer run
# can hit a fresh process state.
#
# Per-run command pattern (uses run_sweep_local.py because per-layer
# runs need the layer index threaded through):
#
#   uv run python scripts/sweeps/run_sweep_local.py \
#       --base-config configs/probe.<encoder>-layers.<task>.yaml \
#       --num-layers <N> \
#       --model-tag <encoder> \
#       --task-tag <task> \
#       --extra-tag layer-sweep-2026-05-27 \
#       --accelerator gpu \
#       --precision bf16-mixed
```

The launcher iterates 8 (encoder × task) pairs sequentially. Each
pair invokes `run_sweep_local.py` which internally iterates layers
0..N-1, emitting one wandb run per layer with `name="layer-N-test"`
and `job_type="test"`.

### `WANDB_MODE=offline` wrapper for CLaMP3 × VGMIDITVar-timbre

If reboot-before-launch (option 2 above) is chosen, no wrapper
needed. If we prefer the soft approach, the launcher sets
`WANDB_MODE=offline` for that one config and emits a one-liner at
the end:

```bash
wandb sync output/probe.VGMIDITVar-timbre.CLaMP3-layers/wandb/offline-run-*
```

after the rest of the sweep completes.

### Sequencing

```
Order  Encoder         Task                Layers  ETA      Cumulative
-----  ------------    ------------------  ------  ------   ----------
  1    CLaMP3          SHS100K             13       20 m     0:20
  2    MERT-v1-95M     SHS100K             13       20 m     0:40
  3    MuQ             SHS100K             13       20 m     1:00
  4    OMARRQ-25hz     SHS100K             25       40 m     1:40
  5    CLaMP3          VGMIDITVar-timbre   13     2:36 h     4:16   ← wandb-core risk
  6    MERT-v1-95M     VGMIDITVar-timbre   13     2:36 h     6:52
  7    MuQ             VGMIDITVar-timbre   13     2:36 h     9:28
  8    OMARRQ-25hz     VGMIDITVar-timbre   25     5:00 h    14:28
```

SHS100K first front-loads the cheap runs (1.7 h) → if any launcher
bug exists we find it before the 12-h VGMIDITVar-timbre block.

## Pre-flight checklist

Before invoking the launcher:

1. **`git pull origin main` on my-pc** — pick up the latest grid-logging
   + verify scripts.
2. **`bash` reboot Windows.** Resets desktop heap; reduces wandb-core
   failure probability across the 14 h run.
3. **Run JSONL verifier on both datasets**:
   ```
   python scripts/data/verify_retrieval_jsonl.py \
     --jsonl data/SHS100K/SHS100K.test.jsonl --target-sr 24000
   python scripts/data/verify_retrieval_jsonl.py \
     --jsonl data/VGMIDITVar-timbre/VGMIDITVar.jsonl
   ```
   Both should exit 0. Stop if either fails.
4. **Disk check:** `df -h /c` shows ≥ 30 GB free. Sweep doesn't add
   significant disk usage (cache is per-encoder-task, not per-layer),
   but leave headroom.
5. **Confirm cache directories present** for all 8 combinations:
   ```
   ls output/.emb_cache/{CLaMP3,MERT-v1-95M,MuQ,OMARRQ-multifeature-25hz}/{SHS100K.test,VGMIDITVar-timbre}__*
   ```

## Post-run artefacts (per-layer × per-task)

For each run, the probe emits:

- WandB run with all `test/*` keys (map, map_centered, recall@10,
  r_precision, median_rank, anisotropy/{mean_vec_norm,effective_rank}).
- VGMIDITVar-timbre additionally emits the **8×8 condition grid**:
  - 64 wandb scalars under `test/map_grid/<q>_to_<t>` + `..._n`.
  - `output/probe.VGMIDITVar-timbre.<encoder>-layers/wandb/run-*/files/condition_grid.csv`
  - `output/probe.VGMIDITVar-timbre.<encoder>-layers/wandb/run-*/files/condition_grid.png`
- Aggregate condition stats: `test/map_same_condition`,
  `test/map_cross_condition`, `test/condition_gap`.

## Analysis outputs (post-sweep)

Not part of the sweep itself, but the immediate follow-up:

1. **Per-encoder layer profile plot** — best-layer-for-this-encoder
   curve. One figure per task. matplotlib + wandb API.
2. **Cross-encoder layer-aligned comparison** — same axes, all four
   encoders overlaid. Reveals whether the optimal layer is
   stable across encoders.
3. **Per-encoder condition heatmap** at the best layer — the 8×8 GM
   program grid for VGMIDITVar-timbre. The PNGs the probe writes
   already cover this; just collect the four best-layer ones.
4. **MAP same-vs-cross condition curve** across layers — does any
   encoder show a layer at which timbre-invariance peaks (cross
   close to same), as opposed to the bottom-layer "spectral-only"
   solution where same > cross?

## Out of scope

- CLaMP3-symbolic on either task (symbolic encoder, audio-task
  comparison not the focus).
- MusicFM / Qwen2Audio (not in the priority encoder set).
- VGMIDITVar (base, non-timbre) — already complete per `docs/TODO.md`
  Tier B.
- Covers80 — already complete.
- LeitmotifDetection — separate task; needs the cache-integration
  follow-up (see `docs/TODO.md`).

## What needs your call

1. **Mitigation choice for CLaMP3 × VGMIDITVar-timbre wandb-core
   failure:** reboot (recommended) or `WANDB_MODE=offline` wrapper?
2. **Launch timing:** start now / overnight / wait for some other
   gate?
3. **Sweep tag in wandb:** `layer-sweep-2026-05-27` proposed —
   confirm or rename.

Once those three are decided I'll write the launcher script, run the
pre-flight checks, and brief you again before pressing go.
