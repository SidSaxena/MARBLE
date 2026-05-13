# MARBLE workflow — end-to-end runbook

Single page, ordered top-to-bottom. Each step is copy-pasteable and
documents the prerequisites it depends on. Skip a section if it
doesn't apply to your machine (e.g. Modal-only or PC-only sections).

---

## Phase 0 — One-time setup

### 0.1 Python environment

```bash
git clone https://github.com/SidSaxena/MARBLE.git
cd MARBLE
uv sync       # ~3–5 min first time; pulls torch/cu128, lightning, transformers, etc.
```

### 0.2 ffmpeg

torchaudio 2.7 needs ffmpeg 4–7 shared libraries to decode AAC/M4A.
ffmpeg 8 (current default in some package managers) is **incompatible**.

| OS | Command | Notes |
|---|---|---|
| macOS | `brew install ffmpeg@7` + `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib` | brew's default `ffmpeg` is currently 8.x |
| Windows | Download `ffmpeg-7.1.1-essentials_build-shared.7z` from https://www.gyan.dev/ffmpeg/builds/ and extract to `C:\ffmpeg`; add `C:\ffmpeg\bin` to PATH. **Must be the `_build-shared` variant** — the plain `essentials_build` is static (no DLLs) and torchaudio won't see it. | See [local_sweeps.md](local_sweeps.md#ffmpeg-on-windows) for the diagnostic |
| Linux | `sudo apt install ffmpeg` (works as-is, apt ships 4–6) | |

Verify:

```bash
uv run python -c "import torchaudio; print(torchaudio.list_audio_backends())"
# Should print: ['soundfile', 'ffmpeg']
```

If `'ffmpeg'` is missing, see [local_sweeps.md](local_sweeps.md#ffmpeg-on-windows).

### 0.3 Secrets

```bash
# WandB — used by both local + Modal sweeps
wandb login

# HuggingFace — required for gated datasets (Chords1217, HookTheory)
huggingface-cli login

# Modal — one-time
modal token new
modal secret create wandb-secret WANDB_API_KEY=$(grep -A1 api.wandb.ai ~/.netrc | tail -1 | awk '{print $NF}')
modal secret create huggingface HF_TOKEN=hf_...
```

---

## Phase 1 — Data acquisition

Most sweeps need a dataset present on disk. Run only what you need.

### 1.1 SHS100K (cover retrieval)

Requires YouTube cookies in a browser (for yt-dlp).

```bash
# Local
uv run python scripts/data/download_shs100k.py --browser firefox
# → data/SHS100K/audio/<ytid>.m4a + data/SHS100K/SHS100K.test.jsonl

# Convert to FLAC (recommended on Mac/Windows where torchaudio's ffmpeg
# backend may be missing — libsndfile handles FLAC natively)
uv run python scripts/data/convert_shs100k_to_flac.py
```

See [data/shs100k.md](data/shs100k.md) for details on the FLAC
conversion and Modal volume layout.

### 1.2 HookTheory Key + Structure (clip-based, ~4 GB)

```bash
# Local
uv run python scripts/data/download_hooktheory.py
# → data/HookTheory/{HookTheoryKey,HookTheoryStructure}.{train,val,test}.jsonl
#   + data/HookTheory/hooktheory_clips/*.mp3
```

### 1.3 HookTheory Melody (full audio, ~104 GB — Modal-only recommended)

```bash
# Modal — downloads zips/audio/*.tar, extracts, builds the rich JSONL
modal run modal_marble.py::setup_hooktheory_full
```

See [data/hooktheory.md](data/hooktheory.md) for the schema and how
the rich JSONL differs from Key/Structure.

### 1.4 NSynth / GTZAN / Chords1217 / GS / EMO / Covers80

```bash
# Modal — most are m-a-p HF repos
modal run modal_marble.py::download                          # GTZAN, Chords1217, GS, EMO, Covers80
modal run modal_marble.py::download_gtzan_only               # subset

# Local
uv run python scripts/data/download_covers80.py
uv run python scripts/data/download_nsynth.py
```

---

## Phase 2 — Data verification (optional but recommended)

```bash
# Audit SHS100K JSONL ↔ disk audio (drop bad entries optionally)
uv run python scripts/verify/verify_shs100k.py \
    --jsonl data/SHS100K/SHS100K.test.jsonl --rewrite

# Audit HookTheory clip JSONLs
uv run python scripts/verify/verify_hooktheory.py
```

On Modal:

```bash
# Rewrite SHS100K JSONL to use Modal mount paths + drop unreachable entries
modal volume put marble-data data/SHS100K/SHS100K.test.jsonl SHS100K/SHS100K.test.jsonl
modal run modal_marble.py::setup_shs100k_jsonl
```

---

## Phase 3 — Running sweeps

### 3.1 Local sequential (default)

```bash
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey
```

Live fit/test output streams to console; test metrics printed at the end.

### 3.2 Local parallel (concurrency)

```bash
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey \
    --concurrency 2
```

- Each layer runs in its own subprocess; ~1.7× speedup on a 16 GB GPU.
- Console shows `[L0]`, `[L1]` prefixed live output.
- Per-layer logs at `output/logs/{model}.{task}/layer-{N}.log` — `tail -f` them.

See [local_sweeps.md](local_sweeps.md) for the PC tuning details.

### 3.3 Run all sweeps

```bash
uv run python scripts/sweeps/run_all_sweeps.py --dry-run        # see plan
uv run python scripts/sweeps/run_all_sweeps.py --tasks GS       # filter
uv run python scripts/sweeps/run_all_sweeps.py --concurrency 2
```

### 3.4 Modal — single sweep

```bash
modal run scripts/sweeps/modal/modal_sweep.py \
    --base-config configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml \
    --num-layers 13 --model-tag MERT-v1-95M --task-tag HookTheoryKey
```

### 3.5 Modal — multi-sweep batch

```bash
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --tier 1     # MERT
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --tier 2     # CLaMP3
modal run scripts/sweeps/modal/modal_run_all_sweeps.py --only MERT-HookTheoryKey
```

See [modal_sweeps.md](modal_sweeps.md) for tier definitions + cost estimates.

---

## Phase 4 — Results

### 4.1 Best-layer summary

```bash
# Top-layer-per-group across the whole project
uv run python scripts/analysis/best_layer.py

# Drill into one
uv run python scripts/analysis/best_layer.py --group "MERT-v1-95M / SHS100K"

# Substring filter / metric override
uv run python scripts/analysis/best_layer.py --filter HookTheory --metric test/weighted_score
```

### 4.2 Retrieve Modal artifacts

```bash
modal volume get marble-output output ./output --recursive
```

### 4.3 WandB run housekeeping

If past runs end up with broken group/name conventions:

```bash
uv run python scripts/analysis/fix_wandb_runs.py            # dry-run
uv run python scripts/analysis/fix_wandb_runs.py --apply
```

---

## Phase 5 — Diagnostics

```bash
# Validate MPS on Apple Silicon
uv run python scripts/diagnostics/test_mps_compat.py

# Validate CLaMP3 cross-modal embedding semantic alignment
uv run python scripts/diagnostics/test_clamp3_crossmodal_semantic.py \
    --jsonl data/VGMIDITVar/VGMIDITVar.jsonl \
    --midi-dir data/VGMIDITVar/midi \
    --num-pairs 5
```
