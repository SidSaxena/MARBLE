# NSynth optimization plan (HookTheoryMelody-style)

## Status

NSynth has 8 probe configs and is **NOT yet wired for Modal**. The dataset
is native WAV at 16 kHz mono, 4 s fixed-duration clips, ~289 k training
files (~50 GB). Each `__getitem__` reads a small WAV and computes a
trivial label (`note - MIDI_OFFSET`). Configs cap to `max_samples: 50000`
for cost control.

## What carries over from the HookTheoryMelody work

| Optimisation | NSynth applicability |
|---|---|
| `BaseDataModule._PREFETCH_FACTOR=4` | ✓ Already applied globally |
| MP3 → WAV conversion | ✗ Source is already WAV, no decode bottleneck |
| `audio_ext` parameter in datamodule | ✗ JSONL stores full `audio_path`, no construction |
| `precompute_labels` | ✗ Label is `note - MIDI_OFFSET`, ~1 ns/call |
| Modal `_warmup_audio_dir` | ✓ ~50 GB corpus on a Modal volume warrants warmup |
| `num_workers=8` local default + Modal override to 16 | ✓ Same pattern as HookTheoryMelody |

So the work is narrower than HTM: **mostly Modal infrastructure + a small
config audit**, no datamodule or dataset format changes.

## Concrete deliverables

### 1. `modal_marble.py::setup_nsynth` (Modal function)

Mirrors `setup_hooktheory_full` — downloads `m-a-p/NSynth` from
HuggingFace into `data/NSynth/` on the marble-data volume, runs whatever
post-extraction step the dataset needs, then `data_vol.commit()`. Size
budget: ~50 GB on the volume.

```python
@app.function(image=image, volumes=VOL, timeout=4 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def setup_nsynth():
    """Download m-a-p/NSynth into data/NSynth/ on marble-data."""
    _chdir(); data_vol.reload()
    _download_marble_datasets.local(datasets=["NSynth"])  # or inline
    data_vol.commit()
```

The existing `_download_marble_datasets` helper already does the
HF-snapshot-download dance — passing `"NSynth"` to it should Just Work
(verify it follows the m-a-p naming convention).

### 2. `modal_marble.py::warmup_nsynth_audio` (Modal function)

Mirrors `warmup_hooktheory_audio_wav`:

```python
@app.function(image=base_image, volumes=VOL, timeout=60 * 60)
def warmup_nsynth_audio(workers: int = 32):
    """Pre-fetch every NSynth WAV into the container's local FS cache.

    Corpus is ~50 GB and the WAVs are small (~256 KB each at 16 kHz mono
    4 s), so warmup takes ~3-5 min at typical Modal volume bandwidth.
    """
    _chdir(); data_vol.reload()
    _warmup_audio_dir(f"{WORK_DIR}/data/NSynth/nsynth-train/audio", workers=workers)
```

Worth running standalone before a sweep so all containers in a parallel
sweep don't each pay the cold-start tax.

### 3. `modal_marble.py::sweep_*_nsynth` convenience entrypoints

Five entrypoints, one per encoder, mirroring the HookTheoryMelody
pattern at `modal_marble.py:1897` onwards. Each calls `run_sweep.remote(...)`
with `warmup_audio_dir=_NSYNTH_AUDIO` and the existing
`_HOOKTHEORY_CLI_OVERRIDES`-style num_workers bump:

```python
_NSYNTH_AUDIO = f"{WORK_DIR}/data/NSynth/nsynth-train/audio"
_NSYNTH_CLI_OVERRIDES = ["--data.init_args.num_workers=16"]

@app.local_entrypoint()
def sweep_omarrq_nsynth():
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature-25hz-meanall.NSynth.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature-25hz",
        task_tag="NSynth",
        warmup_audio_dir=_NSYNTH_AUDIO,
        cli_overrides=_NSYNTH_CLI_OVERRIDES,
    )

@app.local_entrypoint()
def sweep_mert95m_nsynth():
    run_sweep.remote(
        base_config="configs/probe.MERT-v1-95M-layers.NSynth.yaml",
        num_layers=13,
        ...
    )
# … sweep_mert330m_nsynth, sweep_clamp3_nsynth
```

### 4. Config audit (small)

All 8 NSynth configs already have `num_workers: 8` — no normalisation
needed. (Verified at audit time.) Confirm again after each commit:

```bash
grep "num_workers:" configs/probe.*NSynth*.yaml
```

### 5. GPU recommendation for NSynth sweeps

NSynth clips are 4 s × 16 kHz = 64 k samples per item (tiny vs HTM's
360 k). Per-batch compute should be lower than HookTheoryMelody at the
same batch size; the encoders will saturate more easily. Suggested:

| GPU | Expected throughput | Notes |
|---|---|---|
| Local 5060 Ti | ~5-8 it/s | Likely sufficient; corpus fits in 16 GB VRAM with batch=32. |
| Modal A10G | ~5-8 it/s | Similar to 5060 Ti, no clear advantage. |
| Modal A100-40GB | ~15-20 it/s | Big over-spec given clip size; only worth it for fastest wall-clock. |
| Modal H100 | ~30+ it/s if data keeps up | Likely dataloader-bound at this rate. |

**Default suggestion: run NSynth sweeps locally** on the 5060 Ti unless
wall-clock is critical. The 50 k subsampled corpus is small enough that
even a 24-layer OMARRQ sweep should finish overnight on local.

## Risks / unknowns to verify before execution

1. **NSynth dataset name on HuggingFace.** Confirm `m-a-p/NSynth` exists.
   If the dataset is `m-a-p/NSynthOctave` or split differently, adjust the
   download.
2. **Subsampling determinism.** The configs use `max_samples: 50000` with
   stratified subsampling logic in the datamodule. Ensure resumes pick
   the same subset (seed handling).
3. **No precompute_labels needed** — but if the per-sample label compute
   in the datamodule grows (e.g., adds per-sample augmentation), revisit.

## Effort

- Code: ~50 lines (3 Modal functions + 5 sweep entrypoints).
- Compute: one-time `setup_nsynth.remote()` for ~$2-5 to populate the
  volume. Then sweeps as needed.
- Disk: ~50 GB on the marble-data volume.

Defer until you actually plan to run NSynth sweeps; the code is the
mechanical part, the data setup is the long pole.
