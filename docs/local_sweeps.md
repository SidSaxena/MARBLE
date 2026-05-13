# Local sweeps — PC workflow

Run MARBLE layer-probe sweeps on a workstation. Covers the things
specific to running locally (vs. Modal): parallelism on one GPU,
MPS on Apple Silicon, and the ffmpeg-on-Windows trap.

For the end-to-end runbook, see [workflow.md](workflow.md).

---

## --concurrency: parallel layers on one GPU

Each layer probe needs ~5–6 GB VRAM. On a 16 GB GPU two can run
simultaneously with headroom:

```bash
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey \
    --concurrency 2
```

What happens:

- N=2 fit+test pairs run in flight via `ThreadPoolExecutor` over
  `subprocess.Popen`.
- Auto-injects `--data.init_args.num_workers=4` per subprocess so total
  dataloader CPU workers stay at 8 (vs the config default of 8/proc =
  16 with N=2). Override with `--num-workers-per-proc`.
- Live output: each line prefixed with `[L{layer}]`, lock-serialized
  to the console.
- Per-layer logs at `output/logs/{model}.{task}/layer-{N}.log` —
  `tail -f` to follow one layer without prefix noise.

### Recommended values

| GPU | VRAM | `--concurrency` | num_workers_per_proc (auto) |
|---|---:|---:|---:|
| RTX 5060 Ti | 16 GB | **2** | 4 |
| RTX 4090 / 5090 | 24/32 GB | 3 | 3 |
| RTX 3090 | 24 GB | 3 | 3 |
| RTX 4070 | 12 GB | 1 (only one layer fits) | — |

Don't push it: at concurrency=3 on a 16 GB card you'll OOM partway
through fit.

### Recovery if a layer fails

The sweep continues past test errors (logged but not raised). Fit
errors raise — re-run with the same args; `_layer_done()` skips
completed layers via the WandB summary marker.

---

## MPS on Apple Silicon

The `--accelerator mps` flag is plumbed through. When set, the runner
auto-injects `--trainer.precision=16-mixed` (MPS doesn't support
bf16-mixed which most configs ship with).

```bash
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-layers.GS.yaml \
    --num-layers 13 --model-tag CLaMP3 --task-tag GS \
    --accelerator mps
```

Smoke-test MPS before committing to a long sweep:

```bash
uv run python scripts/diagnostics/test_mps_compat.py
```

This runs CLaMP3 × SHS100K layer 0 on MPS, validates
`torchaudio.list_audio_backends()`, audio reachability, and that the
test phase emits `test/*` keys. Use `--keep-output` to preserve the
output dir for inspection.

### What works on MPS

- **MERT** (Wav2Vec2 base): fine
- **CLaMP3** (BERT + audio encoder): fine
- **OMARRQ**: unknown — may fall back to CPU on some custom ops; test before committing

### What doesn't

- Configs with `precision: bf16-mixed` need the auto-override (handled).
- `MuQ`, `MusicFM`, `MuQMuLan`, `Qwen2AudioInstructEncoder`: hard-coded
  `.cuda()` calls in their model code — would need patching.

---

## ffmpeg on Windows

If your SHS100K (or any AAC/M4A) sweep dies with
`LibsndfileError: <unprintable>`, the cause is torchaudio's ffmpeg
backend not loading — libsndfile doesn't support AAC.

### Diagnostic

```powershell
uv run python -c "import torchaudio; print(torchaudio.list_audio_backends())"
```

If output is `['soundfile']` only (no `'ffmpeg'`):

```powershell
dir C:\ffmpeg\bin\*.dll
```

| Output | Cause | Fix |
|---|---|---|
| `ffmpeg.exe` only, no DLLs | Static `essentials_build` — torchaudio can't dynamically link | Download `_build-shared` variant (below) |
| `avcodec-*.dll`, `avformat-*.dll` etc. present | Shared libs installed but Python can't find them | Close + reopen all shells / VS Code (Windows DLL search is per-process) |
| No `C:\ffmpeg\` at all | ffmpeg not installed | Install (below) |

### Fix path A — install the shared variant (recommended)

1. Download `ffmpeg-7.1.1-essentials_build-shared.7z` from
   https://www.gyan.dev/ffmpeg/builds/ — the **"release essentials"
   shared** variant (NOT the plain `essentials_build`).
2. Extract to `C:\ffmpeg` (overwrite previous install).
3. `C:\ffmpeg\bin` should already be on PATH — no PATH change needed.
4. **Close every Python shell and VS Code**. Windows DLL search is
   per-process; re-opening picks up the new DLLs.
5. Verify:
   ```powershell
   uv run python -c "import torchaudio; print(torchaudio.list_audio_backends())"
   # Expect: ['soundfile', 'ffmpeg']
   ```

### Fix path B — convert audio to FLAC (universal)

```bash
uv run python scripts/data/convert_shs100k_to_flac.py
```

~30–60 min for SHS100K (6905 files), produces ~30 GB FLAC. After this,
soundfile decodes everything natively — no ffmpeg version dance ever
again on any machine.

See [data/shs100k.md](data/shs100k.md) for details.

### Why this matters

- **SHS100K**: only dataset shipped as .m4a; needs ffmpeg OR FLAC.
- **All other MARBLE datasets**: WAV / MP3 / FLAC, libsndfile-native,
  no ffmpeg needed.
