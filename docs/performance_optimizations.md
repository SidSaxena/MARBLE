# Performance optimizations — current state

Canonical reference for the perf/optimisation work landed in 2026-05.
Read this before assuming a config has (or lacks) a given fix.

## TL;DR — what's deployed

| Optimisation | Scope | Status | Typical impact |
|---|---|---|---|
| `prefetch_factor=4` in DataLoader | **All tasks** (global) | ✓ shipped | 1.5-2× dataloader throughput on warm cache |
| Vectorised label compute + cached interp1d | HookTheoryMelody only | ✓ shipped | ~5-10× faster label compute (was per-getitem scipy calls) |
| `precompute_labels` flag (contiguous tensor) | HookTheoryMelody only | ✓ shipped | Minor — eliminates ~1 ms/item of label work, fork-safe |
| `audio_ext` parameter | HookTheoryMelody only | ✓ shipped | Lets configs swap MP3↔WAV without code change |
| MP3→WAV conversion pipeline | HookTheory + HookTheory clips | ✓ shipped (Modal) | Removes MP3 decode + resample CPU work; **31× faster dataloader on Modal volumes** |
| `torch.compile` on frozen encoder | MERT, OMARRQ, MuQ | ✓ shipped (142 configs) | **+20-25 % per-batch compute** (post warmup) |
| Modal volume warmup helpers | HookTheory MP3/WAV + clips WAV + NSynth | ✓ shipped | Pulls files into container local FS cache before training |
| Smoke A/B harnesses | local + Modal | ✓ shipped | Validate any encoder/config change in 5-15 min before broad rollout |

## Per-task quick reference

Tasks listed by what optimisations affect their throughput when run on Modal.
**Local on a 5060 Ti or similar workstation GPU**: most optimisations still help proportionally, except the Modal-volume warmup (no effect locally — OS page cache handles it for free).

| Task | Audio source | precompute_labels | WAV-converted | Modal sweep entrypoint | Compile-capable encoders |
|---|---|---|---|---|---|
| HookTheoryMelody | full HookTheory MP3s (~108 GB) | ✓ on all 10 configs | ✓ (`audio_wav/`) | 5 entrypoints | MERT, MuQ, MusicFM, OMARRQ |
| HookTheoryKey | hooktheory_clips MP3s | N/A (trivial labels) | ✓ (`hooktheory_clips_wav/`) | none (use generic `run_probe`) | MERT, MuQ, OMARRQ, CLaMP3, Qwen2Audio |
| HookTheoryStructure | hooktheory_clips MP3s | N/A | ✓ | none | MERT, MuQ, OMARRQ, CLaMP3 |
| NSynth | Magenta GCS WAVs | N/A | N/A (already WAV) | 4 entrypoints | MERT, OMARRQ, CLaMP3, MuQ |
| GTZAN{Genre,BeatTracking} | local WAVs | N/A | N/A | 2 entrypoints (Genre only) | MERT, MuQ, OMARRQ |
| GS, EMO | small WAV corpora | N/A | N/A | 2 entrypoints each | MERT, MuQ, OMARRQ, CLaMP3 |
| MTT, MTG{Genre,Inst,Mood,Top50} | huge .mp3/.flac corpora (~467 k files) | N/A | not done (no MP3 source change) | none | MERT (95M), MuQ |
| Chords1217 | FLAC, 16 k files | could benefit (deferred) | N/A | 1 (OMARRQ only) | MERT, MuQ, OMARRQ |
| Covers80, SHS100K | MP3, retrieval tasks | N/A | not done | none | MERT, MuQ, OMARRQ, CLaMP3 |
| BPSMotif, VGMIDITVar, SuperMarioStructure, HXMSA | symbolic / small audio | N/A | not done | none | MERT, MuQ, OMARRQ |

## Per-encoder compile_mode status

| Encoder | `compile_mode` support in code | Configs setting `default` | Notes |
|---|---|---|---|
| **MERT-v1-95M, MERT-v1-330M** | ✓ | 68 / 68 | Verified +20% per-batch on 5060 Ti × HookTheoryMelody |
| **OMARRQ-multifeature-25hz** | ✓ (wraps `self.model.net` — see `model.py:178`) | 38 / 38 | Verified +22% per-batch. **DO NOT use `reduce-overhead`** — errors fatally on the RoPE `einops.repeat` op due to CUDA Graphs storage aliasing |
| **MuQ** | ✓ (only when `train_mode='freeze'`) | 36 / 36 | Verified +23% (`default`) / +25% (`reduce-overhead`) per-batch. Defaults set to `default` for safety; opt into `reduce-overhead` per-config if you want the extra 2 % |
| **MuQMuLan** | ✗ no `compile_mode` param | 0 / 5 | Different encoder class; deferred |
| **CLaMP3** | ✗ no `compile_mode` param | 0 / 28 | Hybrid audio + symbolic forward; high implementation risk; deferred |
| **MusicFM** | ✗ no `compile_mode` param | 0 / 11 | HF transformers stack often has compile-unfriendly ops; deferred |
| **Qwen2AudioInstructEncoder** | ✗ no `compile_mode` param | 0 / 1 | Full LLM; would need vLLM-style compile path; deferred |

**Capability gate**: every encoder that supports `compile_mode` checks for Triton + CUDA at init. On macOS, Windows (no Triton), or CPU-only boxes, it falls back to eager with a one-line warning. So setting `compile_mode: default` in a config is safe to ship — it does no harm on incompatible hardware.

## Files of interest

### Code

| File | What it provides |
|---|---|
| `marble/core/base_datamodule.py` | `_PREFETCH_FACTOR = 4` constant — affects every datamodule that inherits |
| `marble/tasks/HookTheoryMelody/datamodule.py` | `audio_ext`, `precompute_labels`, vectorised `_compute_labels` |
| `marble/encoders/MERT/model.py` | Original `compile_mode` reference implementation |
| `marble/encoders/OMAR_RQ/model.py` | `compile_mode` (wraps `self.model.net` — important subtlety) |
| `marble/encoders/MuQ/model.py` | `compile_mode` (freeze-only) |

### Data pipeline scripts (under `scripts/data/`)

| Script | Purpose |
|---|---|
| `cache_audio_info_in_jsonl.py` | Pre-populate `num_samples`/`sample_rate` per record so datamodule init doesn't hit `torchaudio.info()` per-file |
| `convert_audio_format.py` | Universal ffmpeg-driven audio format converter (any→wav/flac/mp3/ogg, optional resample/downmix). Supports `--from-jsonl` for smoke subsets and `--jsonl` rewriting |
| `rewrite_jsonl_audio_paths.py` | Swap `audio_path` strings in JSONLs (e.g. `.mp3`→`.wav`, dir swap). Used for HookTheoryKey/Structure WAV migration |
| `build_smoke_jsonl.py` | Deterministic first-N record JSONL subsetter |
| `download_nsynth.py` | Magenta GCS NSynth download + JSONL build (not m-a-p) |

### Diagnostic / smoke harnesses (under `scripts/diagnostics/`)

| Script | Purpose |
|---|---|
| `smoke_test_wav.py` | Local A/B: MP3 baseline vs WAV experiment on a small subset |
| `smoke_test_compile.py` | Local A/B for `compile_mode`: eager vs default vs reduce-overhead, N-way comparison via `--modes` |

### Modal entrypoints (in `modal_marble.py`)

- **Setup**: `setup_hooktheory_full`, `setup_nsynth`, `setup_supermario_structure`, etc.
- **Audio metadata caching**: `setup_hooktheory_audio_info`
- **MP3→WAV conversion**: `convert_hooktheory_to_wav`, `convert_hooktheory_clips_to_wav`
- **Warmup**: `warmup_hooktheory_audio`, `warmup_hooktheory_audio_wav`, `warmup_hooktheory_clips_wav`, `warmup_nsynth_audio`
- **Smoke**: `smoke_test_wav` (Modal version of the local one)
- **Sweep convenience**: `sweep_<encoder>_hooktheorymelody` × 5, `sweep_<encoder>_nsynth` × 4

### Tests (`tests/`)

| Test | Coverage |
|---|---|
| `test_hooktheorymelody_changes.py` | 36 tests — vectorised label compute byte-equivalence, precompute_labels equivalence, audio_ext normalisation, edge cases |
| `test_rewrite_jsonl_audio_paths.py` | 7 tests — JSONL path-swap correctness, idempotency, multi-dot filename handling |

All 43 tests pass via `uv run python -m pytest tests/test_hooktheorymelody_changes.py tests/test_rewrite_jsonl_audio_paths.py`.

## Common workflows

### "I want to run task X on Modal with a sweep entrypoint"

```bash
MARBLE_IMAGE=audio MARBLE_GPU=A100-40GB modal run modal_marble.py::sweep_omarrq_hooktheorymelody
# Other entrypoints: sweep_mert95m_hooktheorymelody, sweep_mert330m_hooktheorymelody,
# sweep_muq_hooktheorymelody, sweep_musicfm_hooktheorymelody,
# sweep_omarrq_nsynth, sweep_mert95m_nsynth, sweep_mert330m_nsynth, sweep_clamp3_nsynth
```

Each one wires `warmup_audio_dir`, `cli_overrides=["--data.init_args.num_workers=16"]`, and the right base config + num_layers.

### "I want to A/B test a config change locally before committing"

For WAV vs MP3:
```bash
uv run python scripts/diagnostics/smoke_test_wav.py
```

For compile_mode (any encoder that supports it):
```bash
uv run python scripts/diagnostics/smoke_test_compile.py \
    --config configs/probe.<encoder>-meanall.<task>.yaml \
    --modes default reduce-overhead \
    --batches 500 \
    --jsonl-smoke
```

`--jsonl-smoke` rewrites `data.init_args` jsonl paths to their `.smoke.wav.jsonl` variants — for when only a subset of audio is on disk locally.

### "I want to set up HookTheoryMelody / NSynth on Modal from scratch"

```bash
# HookTheory full + audio info caching
modal run modal_marble.py::setup_hooktheory_full
modal run modal_marble.py::setup_hooktheory_audio_info

# WAV conversion (one-time, idempotent)
modal run modal_marble.py::convert_hooktheory_to_wav
modal run modal_marble.py::convert_hooktheory_clips_to_wav   # for Key/Structure

# NSynth
modal run modal_marble.py::setup_nsynth
```

### "I want to run a task without compile_mode (debugging numerics)"

Override on the CLI:
```bash
uv run python cli.py fit -c configs/probe.MERT-v1-95M-meanall.HookTheoryMelody.yaml \
    --model.init_args.encoder.init_args.compile_mode=null
```

## Gotchas worth knowing

### 1. Auto-resume from stale `last.ckpt` carries `EarlyStopping.best_score`

`modal_marble.py::_resume_args_for_config` auto-injects `--ckpt_path output/.../last.ckpt` when it finds one. That ckpt restores the EarlyStopping callback's `best_score` AND `wait_count`. If you switched data (e.g. MP3 → WAV) between runs, the metric on the new data might never beat the cached `best_score` → patience exhausts in ~7 epochs → run finishes "early" while val/acc was visibly improving.

**Workaround**: delete `last.ckpt` before a fresh run after a data change:
```bash
modal volume rm marble-output /probe.<task>.<encoder>.../checkpoints/last.ckpt
# or locally
rm output/probe.<task>.<encoder>.../checkpoints/last.ckpt
```

### 2. OMARRQ `compile_mode='reduce-overhead'` fatally errors

The RoPE in `omar_rq/nets/rope.py:305` does `einops.repeat(freqs, "... n -> ... (n r)", r=2)`. With CUDA Graphs, the `repeat` tensor's storage gets overwritten across calls → `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. Configs ship with `default` and the comments warn against switching.

### 3. `compile_mode` on small runs

First-batch compile takes 30-90 s for MERT / OMARRQ, similar for MuQ. On runs shorter than ~500 batches, wall-clock can look like a regression (compile warmup not amortised). The smoke harness's `run_training_batch` profiler row measures **post-warmup** per-batch compute — that's the cleanest signal.

### 4. `num_workers` is split between local and Modal

Configs default to `num_workers: 8` for local sanity (typical workstation has 8-16 cores). Modal sweep entrypoints inject `--data.init_args.num_workers=16` via `cli_overrides`. If you run a config directly via `cli.py fit -c ...` on a beefy machine and want more workers, override on the CLI: `--data.init_args.num_workers=16`.

### 5. MuQ `reduce-overhead` is a small extra gain available

Per the smoke, MuQ specifically benefits ~2 % more from `reduce-overhead` vs `default` (and unlike OMARRQ, doesn't error). Configs ship with `default` for safety. To opt in for a specific config, change `compile_mode: default` to `compile_mode: reduce-overhead` and run the smoke A/B to confirm on your hardware.

### 6. WAV `num_samples` differs from MP3 `num_samples`

The HookTheoryMelody datamodule's fast-path reads `num_samples` from the JSONL to build its index_map. After converting MP3→WAV, `num_samples` changes (different sample rate). The Modal converter (`convert_hooktheory_to_wav`) handles this by re-running `cache_audio_info_in_jsonl.py --force` against the WAV dir. If you do the conversion outside that function, remember this step.

## Encoders that don't yet have `compile_mode`

Tracked as follow-ups, in rough priority order (easier → harder):

1. **MusicFM** — HF transformers stack. Forward likely needs care around `output_hidden_states` collection but otherwise standard.
2. **CLaMP3** — Hybrid audio + symbolic forward paths. Highest implementation risk because of dual-mode branching.
3. **Qwen2AudioInstruct** — Full LLM. Needs a vLLM-style approach; out of scope for normal compile.
4. **MuQMuLan** — Cross-modal (audio + text/CLAP-style). Different forward shape than MuQ.

Each one would need:
- Add `compile_mode: str | None = None` to `__init__`
- Capability gate (Triton + CUDA), wrap the right attribute (look for what `forward()` actually calls — see OMARRQ gotcha)
- Add `compile_mode: default` to one HTM (or representative) config as smoke target
- Run `scripts/diagnostics/smoke_test_compile.py` for the per-encoder A/B
- Roll out to all configs if smoke shows ≥ 5 % per-batch speedup with val/acc drift < 0.005

## Commit ledger

Chronological for traceability. Search any of these with `git log --oneline`.

| Commit | Scope |
|---|---|
| `5aea2af` | HTM datamodule perf (interp1d cache, vectorise, precompute_labels, audio_ext) + 36 tests + pytest dev dep |
| `d8ff5f1` | MP3→WAV pipeline + smoke validator + Modal infra |
| `095d4b0` | 10 HTM configs → WAV |
| `b03cadd` | num_workers split: configs=8, Modal sweeps=16 via cli_overrides |
| `40d3f0f` | Key/Structure WAV migration infra |
| `3ee8657` | 24 Key/Structure configs → WAV |
| `4217f3e` | NSynth Modal infra (setup + warmup + 4 sweep entrypoints) |
| `3eb2499` | 68 MERT configs gained compile_mode |
| `7dca7d8` | OMARRQ compile_mode bug fix — wrap `self.model.net` not `self.model` |
| `8c7b4c7` | 70 OMARRQ + MuQ configs gained compile_mode |
