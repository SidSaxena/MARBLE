# Local sweeps — PC workflow

Run MARBLE layer-probe sweeps on a workstation. Covers the things
specific to running locally (vs. Modal): parallelism on one GPU,
MPS on Apple Silicon, and the ffmpeg-on-Windows trap.

For the end-to-end runbook, see [workflow.md](workflow.md).

---

## Embedding cache (read this before tuning concurrency)

For all cache-safe tasks (retrieval + clip-level supervised) the
encoder forward — historically ~33 min per layer for OMARRQ × SHS100K —
runs **once per (encoder, task) pair**, not once per layer. Subsequent
per-layer (and meanall) jobs load tensors from
`output/.emb_cache/<encoder>/<task>__<hash>/` and skip the encoder
entirely.

This changes the calculus on `--concurrency`. With cold cache the GPU
is saturated and concurrency > 1 doesn't help. With warm cache there's
no GPU work at all, so concurrency mostly affects how fast the cache
files get read from disk (typically you don't need >1).

See [`embedding_cache.md`](embedding_cache.md) for the full design,
disk-math, and CLI reference (`extract.py`, `manage.py`). The cache is
enabled by default via `cache_embeddings: true` in every cache-safe
config.

---

## Console output on Windows (`WANDB_CONSOLE`)

The sweep runner sets `WANDB_CONSOLE=wrap` in every subprocess by
default. This makes the WandB SDK **tee** stdout/stderr to BOTH the
terminal AND its own log file — so tqdm progress bars, `[emb_cache]
HIT/MISS` lines, etc. all show up live as the run progresses.

Without this, on Windows the WandB SDK defaults to `redirect` (a
legacy safe-default for pre-Python-3.7 console quirks), which hijacks
`sys.stdout` inside the subprocess and shows nothing in the terminal
until the run exits — the WandB Logs panel ends up being the only
place to watch progress in real time. On Linux/macOS the SDK already
defaults to `wrap`, so this just makes Windows match.

User override is honored via `setdefault`. Two useful values:

```bash
# Pure terminal output, no WandB stdout capture (slightly less
# overhead; WandB Logs panel will be empty but metrics still log).
WANDB_CONSOLE=off uv run python scripts/sweeps/run_sweep_local.py ...

# Force WandB to capture and NOT mirror to terminal (the old
# Windows default behavior).
WANDB_CONSOLE=redirect uv run python scripts/sweeps/run_sweep_local.py ...
```

**Cosmetic note**: with `wrap` on Windows, each tqdm progress update
may render on its own line (instead of overwriting via `\r` like on a
real Linux TTY). Slightly verbose but readable, and a strict
improvement over silence.

Per-layer log files are written at
`output/logs/{model}.{task}/layer-{N}.log` regardless of console
mode — `tail -f` works on either platform. In sequential mode the log
holds the test phase only; in parallel mode it holds fit + test.

---

## Launch sweeps from the **Console session** on Windows, not from SSH

> **Headline**: on Windows, do not launch long sweeps via SSH. Every
> `wandb.init()` call after the first few will fail with
> `wandb-core exited with code 3221225794` (`STATUS_DLL_INIT_FAILED`).
> Launch from a terminal on the machine's interactive desktop instead.

### Symptom

You launch the sweep over SSH. The first encoder might complete, but
partway through a long run — or even immediately on a freshly-rebooted
machine if the sweep is large enough — every `cli.py test` invocation
fails within a few seconds. Layer logs show:

```
OMAR-RQ: 792 weights loaded for `net`
OMAR-RQ: 2 weights loaded for `embedding_layer`
[OMARRQ-...] torch.compile(...) requested but skipped — Triton not installed
LayerSelector initialized with layers: [0]
Traceback (most recent call last):
  ...
  File "...wandb_init.py", line 900, in init
    service = self._wl.ensure_service()
  ...
wandb.sdk.lib.service.service_port_file.ServicePollForTokenError:
  wandb-core exited with code 3221225794
```

The launcher then dutifully marches through every remaining layer
producing `(no test metrics parsed)` for each one and exits with
`pass=N fail=0` — currently the launcher only checks the inner script's
exit code, not whether layers actually produced output. See
"silent-failure follow-up" in `docs/TODO.md`.

### Diagnosis

Run this on Windows to inspect which session your launched processes
are attached to:

```powershell
tasklist | findstr /I python uv wandb
```

You'll see a column that reads either **Console** or **Services**:

```
python.exe   26168 Console     1   3,754,608 K     ← launched from desktop terminal, OK
python.exe   30272 Services    0     552,104 K     ← launched via SSH, will fail
```

If your sweep processes show `Services`, the wandb-core failure is
imminent or already in progress.

### Cause

Windows allocates a small fixed-size **desktop heap** per
[window station](https://learn.microsoft.com/en-us/windows/win32/winstation/window-stations).
The interactive Console session gets ~3 MB; the non-interactive
Services session gets ~512 KB. Every GUI process consumes ~10 KB of
that heap on launch — including the `wandb-core` Go binary that
`wandb.init()` spawns under the hood. On the Console session the heap
holds dozens of process launches without issue. On the Services
session it depletes after a handful of nested-subprocess wandb
launches and `wandb-core` then exits at DLL initialization time.

Win32-OpenSSH ships with `sshd` configured to launch child processes
on the **Services** session (technically: Session 0, the non-interactive
service session). Every command you run over SSH inherits that —
including nested subprocesses N levels deep. The Lightning sweep
launches a 4-deep tree (uv → run_sweep_local → cli.py test → wandb-core),
so the heap pressure compounds.

The deeper-but-related "8 workers deadlock at LOCAL_RANK" symptom is
the same heap exhaustion — the worker spawn step needs heap to fork
each DataLoader worker, and silently blocks when the allocation fails.

### Fix

Launch from a terminal on the interactive Windows desktop:

```bash
# Open Git Bash / PowerShell / Windows Terminal on my-pc's desktop.
# DO NOT use ssh.
cd ~/developer/python/MARBLE

nohup uv run python scripts/sweeps/run_sweep_local.py \
  --base-config configs/probe.<encoder>.<task>.yaml \
  --num-layers 24 --model-tag <encoder> --task-tag <task> \
  --no-skip --skip-meanall --skip-fit-if-no-train \
  --num-workers 8 --accelerator gpu --precision bf16-mixed \
  > tmp/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

After ~30 seconds, confirm:

```powershell
tasklist | findstr /I python   # should show Console, not Services
```

The sweep can then run for hours without wandb-core failures. The 4
overnight-sweep encoders (CLaMP3 / MERT / MuQ / OMARRQ) all ran cleanly
this way; OMARRQ specifically failed three times in a row when
launched via SSH and succeeded immediately when launched from
Console.

### Workarounds if you must launch via SSH

Only two real options once the Services session is the entry point:

1. **`WANDB_MODE=disabled`** in the env. Bypasses `wandb-core` spawn
   entirely. Cost: no cloud sync, no `wandb-summary.json` per layer.
   The probe's `condition_grid.csv` and the local stdout-captured
   `[CoverRetrieval] MAP …` lines still land — analyse them by parsing
   the local sweep log instead of wandb's summary JSON.

2. **PsExec to launch into Session 1** (the interactive desktop)
   from an SSH-launched script. Untested; `psexec -accepteula -i 1 -d <cmd>`
   in principle works but requires PsExec installed and an unlocked
   interactive session. Not the recommended path — just launch from
   the desktop.

### What did NOT help in practice

- Killing all `python.exe`/`uv.exe`/`wandb-core.exe` processes and
  waiting 90 s. The Services-session heap doesn't recover meaningfully
  on that timescale — it persists across a fresh launch.
- Setting `--num-workers 0` to bypass the DataLoader deadlock. That
  fixes the worker-spawn variant of the heap issue but not the
  `wandb-core` spawn variant — those use distinct heap allocations.
- A full reboot fixes it for a short window (~the first few hours of
  a sweep), but the heap drains again as the sweep accumulates
  process-launch pressure. Reboot is treatment, not cure.

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
uv run python scripts/data/convert_audio_format.py \
    --src data/SHS100K/audio --in-place \
    --input-ext .m4a --to flac \
    --jsonl data/SHS100K/SHS100K.train.jsonl,data/SHS100K/SHS100K.val.jsonl,data/SHS100K/SHS100K.test.jsonl
```

~30–60 min for SHS100K (6905 files), produces ~30 GB FLAC. After this,
soundfile decodes everything natively — no ffmpeg version dance ever
again on any machine.

See [data/shs100k.md](data/shs100k.md) for details.

### Why this matters

- **SHS100K**: only dataset shipped as .m4a; needs ffmpeg OR FLAC.
- **All other MARBLE datasets**: WAV / MP3 / FLAC, libsndfile-native,
  no ffmpeg needed.
