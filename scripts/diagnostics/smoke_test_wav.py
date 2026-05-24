#!/usr/bin/env python3
"""Local smoke-test for the MP3 → WAV conversion hypothesis.

Mirrors ``modal_marble.py::smoke_test_wav`` but runs on the local
filesystem so you can validate the speedup on a workstation GPU (5060 Ti,
3090, etc.) before committing Modal time to the full corpus conversion.

What it does
────────────
1. Build smoke JSONLs (first N records of each split).
2. Convert just those N source MP3s → 24 kHz mono int16 WAV.
3. Cache num_samples/sample_rate in copied smoke JSONLs for the WAV files.
4. Run an MP3-baseline training (1 epoch, `--batches` train batches,
   profiler enabled, callbacks/logger disabled).
5. Run a WAV-experiment training with the same shape.
6. Parse each run's simple profiler output and print a comparison.

Idempotency
───────────
Every step is idempotent — re-running picks up where a prior run left
off (smoke JSONLs are rewritten cheaply; convert skips existing WAVs;
cache uses --force; training writes go to a throwaway dir). Use the
``--skip-*`` flags to manually bypass stages you've already completed.

Prerequisites
─────────────
- ``data/HookTheory/audio/<youtube_id>.mp3`` corpus available locally
  (i.e., you've run the MARBLE HookTheory download at some point).
- ``data/HookTheory/HookTheory.{train,val,test}.jsonl`` already have
  num_samples/sample_rate cached (run cache_audio_info_in_jsonl.py
  beforehand if not — otherwise the datamodule falls back to per-file
  torchaudio.info() which is slow but works).
- ffmpeg on PATH. (Note: on macOS with ffmpeg 8.x, torchaudio falls back
  to libsndfile for reads, which CAN'T read MP3 — the MP3 baseline run
  will fail unless you ``brew install ffmpeg@7`` + set
  DYLD_FALLBACK_LIBRARY_PATH per pyproject.toml. Linux apt's ffmpeg
  works as-is.)
- CUDA-capable GPU. 5060 Ti (Blackwell) needs torch 2.7 + cu128 wheels —
  already pinned in pyproject.toml.

Usage
─────
    # From the marble repo root
    uv run python scripts/diagnostics/smoke_test_wav.py \\
        --config configs/probe.OMARRQ-multifeature-25hz-layers.HookTheoryMelody.yaml

    # Just the WAV run (e.g. if MP3 baseline is broken on your ffmpeg setup)
    uv run python scripts/diagnostics/smoke_test_wav.py --skip-mp3

    # Cheaper test
    uv run python scripts/diagnostics/smoke_test_wav.py --n-songs 100 --batches 200
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins on leaves, dicts merged.

    Same impl as ``modal_marble.py::smoke_test_wav._deep_merge``. Lightning
    CLI's multi-``-c`` stacking does NOT deep-merge ``class_path`` +
    ``init_args`` blocks (a partial override of one inner key clobbers
    the surrounding ``class_path``), so we merge in Python and pass a
    single ``-c`` to ``cli.py``.
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _write_smoke_yaml(
    base_config: Path,
    out_path: Path,
    audio_dir: Path,
    audio_ext: str,
    jsonls: dict,
    batches: int,
) -> None:
    """Deep-merge a smoke override into the base config, write to out_path.

    Trainer knobs (max_epochs=1, limit_*_batches, profiler=simple,
    enable_checkpointing=False, callbacks=[], logger=False) bound the run
    and prevent it from polluting the production output dir or the wandb
    dashboard. ``data.init_args.{train,val,test}.init_args`` overrides
    just the dataset's audio paths and extension — everything else
    (sample_rate, clip_seconds, precompute_labels, channel_mode, etc.)
    is inherited from the base config.
    """
    splits = ("train", "val", "test")
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    overrides = {
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": batches,
            "limit_val_batches": 1,
            "profiler": "simple",
            "enable_checkpointing": False,
            "callbacks": [],
            "logger": False,
        },
        "data": {
            "init_args": {
                s: {
                    "init_args": {
                        "jsonl": str(jsonls[s]),
                        "audio_dir": str(audio_dir),
                        "audio_ext": audio_ext,
                    }
                }
                for s in splits
            },
        },
    }
    _deep_merge(cfg, overrides)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_training(label: str, smoke_yaml: Path) -> tuple[float, str, int]:
    """Stream-capture ``cli.py fit -c smoke_yaml``, return (wall, stdout, rc)."""
    print(f"\n━━━ {label} ━━━", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "cli.py", "fit", "-c", str(smoke_yaml)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        print(line, end="", flush=True)
    proc.wait()
    return time.time() - t0, "".join(captured), proc.returncode


def _parse_profiler(stdout: str) -> dict[str, float | None]:
    """Pull totals for the two rows we care about out of simple-profiler output.

    Lightning's simple profiler prints a table like:
      Action                                       |  Mean ...  |  Num calls  |  Total time  |  Percentage %
    We grab the 3rd numeric column (Total time) for the two rows that
    decide MP3 vs WAV: ``train_dataloader_next`` (the bottleneck row)
    and ``run_training_batch`` (compute side). Robust to whitespace and
    sci-notation variance.
    """

    def _row(name_pat: str) -> float | None:
        m = re.search(
            rf"{name_pat}\s*\|\s*[\d.eE+-]+\s*\|\s*[\d.eE+-]+\s*\|\s*([\d.eE+-]+)",
            stdout,
        )
        return float(m.group(1)) if m else None

    return {
        "train_dataloader_next": _row(r"\[_TrainingEpochLoop\]\.train_dataloader_next"),
        "run_training_batch": _row(r"run_training_batch"),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-songs", type=int, default=500, help="Songs in the smoke subset.")
    ap.add_argument(
        "--batches",
        type=int,
        default=500,
        help="limit_train_batches for each training run.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/probe.OMARRQ-multifeature-25hz-layers.HookTheoryMelody.yaml"),
        help="Base HookTheoryMelody config. Default: OMARRQ-multifeature-25hz layers.",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/HookTheory"),
        help="Dir containing HookTheory.{train,val,test}.jsonl (default: data/HookTheory).",
    )
    ap.add_argument(
        "--audio-src-dir",
        type=Path,
        default=Path("data/HookTheory/audio"),
        help="Source MP3 dir (default: data/HookTheory/audio).",
    )
    ap.add_argument(
        "--audio-wav-dir",
        type=Path,
        default=Path("data/HookTheory/audio_wav_smoke"),
        help="Output WAV dir for the smoke subset (default: data/HookTheory/audio_wav_smoke).",
    )
    ap.add_argument(
        "--ffmpeg-workers",
        type=int,
        default=8,
        help="Parallel ffmpeg workers for conversion (default: 8). Match your CPU cores.",
    )
    ap.add_argument(
        "--skip-build-jsonl",
        action="store_true",
        help="Skip the build_smoke_jsonl step (assume *.smoke.jsonl already in --data-dir).",
    )
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the MP3→WAV conversion (assume --audio-wav-dir already populated).",
    )
    ap.add_argument(
        "--skip-cache-info",
        action="store_true",
        help="Skip cache_audio_info on WAV JSONLs (assume *.smoke.wav.jsonl already correct).",
    )
    ap.add_argument("--skip-mp3", action="store_true", help="Skip the MP3 baseline training run.")
    ap.add_argument("--skip-wav", action="store_true", help="Skip the WAV experiment training run.")
    args = ap.parse_args()

    splits = ("train", "val", "test")
    src_jsonls = {s: args.data_dir / f"HookTheory.{s}.jsonl" for s in splits}
    smoke_jsonls_mp3 = {s: args.data_dir / f"HookTheory.{s}.smoke.jsonl" for s in splits}
    smoke_jsonls_wav = {s: args.data_dir / f"HookTheory.{s}.smoke.wav.jsonl" for s in splits}

    # Pre-flight
    for p in src_jsonls.values():
        if not p.exists() and not args.skip_build_jsonl:
            sys.exit(
                f"missing source JSONL: {p}\n(pass --skip-build-jsonl if reusing existing smoke files)"
            )
    if not args.audio_src_dir.exists() and not (args.skip_convert and args.skip_mp3):
        sys.exit(f"missing audio dir: {args.audio_src_dir}")
    if not shutil.which("ffmpeg") and not args.skip_convert:
        sys.exit("ffmpeg not found on PATH (apt install ffmpeg / brew install ffmpeg)")
    if not args.config.exists():
        sys.exit(f"missing config: {args.config}")

    print(
        f"\n━━━ smoke_test_wav (local): n_songs={args.n_songs} batches={args.batches} ━━━\n"
        f"  config:        {args.config}\n"
        f"  audio_src:     {args.audio_src_dir}\n"
        f"  audio_wav:     {args.audio_wav_dir}",
        flush=True,
    )

    # 1. Build smoke MP3 JSONLs
    if not args.skip_build_jsonl:
        _run(
            [
                sys.executable,
                "scripts/data/build_smoke_jsonl.py",
                *sum((["--jsonl", str(src_jsonls[s])] for s in splits), []),
                "--n",
                str(args.n_songs),
            ]
        )
    else:
        print("[skip] build_smoke_jsonl")

    # 2. Convert referenced MP3s → WAV (idempotent; existing WAVs are skipped)
    if not args.skip_convert:
        args.audio_wav_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "scripts/data/convert_audio_to_wav.py",
                "--src-dir",
                str(args.audio_src_dir),
                "--dst-dir",
                str(args.audio_wav_dir),
                "--src-ext",
                ".mp3",
                "--target-sr",
                "24000",
                "--channels",
                "1",
                "--workers",
                str(args.ffmpeg_workers),
                "--id-key",
                "youtube.id",
                *sum((["--from-jsonl", str(smoke_jsonls_mp3[s])] for s in splits), []),
            ]
        )
    else:
        print("[skip] convert_audio_to_wav")

    # 3. Copy smoke JSONLs → *.smoke.wav.jsonl and refresh num_samples/sample_rate
    if not args.skip_cache_info:
        for s in splits:
            shutil.copyfile(smoke_jsonls_mp3[s], smoke_jsonls_wav[s])
        _run(
            [
                sys.executable,
                "scripts/data/cache_audio_info_in_jsonl.py",
                *sum((["--jsonl", str(smoke_jsonls_wav[s])] for s in splits), []),
                "--audio-dir",
                str(args.audio_wav_dir),
                "--id-key",
                "youtube.id",
                "--audio-suffix",
                ".wav",
                "--workers",
                "16",
                "--force",
            ]
        )
    else:
        print("[skip] cache_audio_info_in_jsonl")

    # 4. Build merged YAMLs (MP3 + WAV variants)
    smoke_dir = Path("output/.smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    mp3_yaml = smoke_dir / "smoke_mp3.yaml"
    wav_yaml = smoke_dir / "smoke_wav.yaml"
    _write_smoke_yaml(
        args.config, mp3_yaml, args.audio_src_dir, ".mp3", smoke_jsonls_mp3, args.batches
    )
    _write_smoke_yaml(
        args.config, wav_yaml, args.audio_wav_dir, ".wav", smoke_jsonls_wav, args.batches
    )
    print(f"\nwrote smoke configs: {mp3_yaml} and {wav_yaml}")

    # 5. Run the two trainings.
    os.environ["WANDB_MODE"] = "disabled"

    mp3_time, mp3_out, mp3_rc = 0.0, "", -1
    wav_time, wav_out, wav_rc = 0.0, "", -1
    if not args.skip_mp3:
        mp3_time, mp3_out, mp3_rc = _run_training("MP3 baseline", mp3_yaml)
    else:
        print("[skip] MP3 baseline training")
    if not args.skip_wav:
        wav_time, wav_out, wav_rc = _run_training("WAV experiment", wav_yaml)
    else:
        print("[skip] WAV experiment training")

    # 6. Parse + print summary.
    mp3_p = (
        _parse_profiler(mp3_out)
        if mp3_out
        else {"train_dataloader_next": None, "run_training_batch": None}
    )
    wav_p = (
        _parse_profiler(wav_out)
        if wav_out
        else {"train_dataloader_next": None, "run_training_batch": None}
    )
    mp3_its = args.batches / mp3_time if mp3_time > 0 else 0.0
    wav_its = args.batches / wav_time if wav_time > 0 else 0.0

    print(f"\n━━━ Smoke test result (n_songs={args.n_songs}, batches={args.batches}) ━━━")
    if not args.skip_mp3:
        print(
            f"  MP3 baseline   : {mp3_time:6.1f}s wall  ≈ {mp3_its:.2f} it/s  "
            f"(train_dataloader_next={mp3_p['train_dataloader_next']}s, "
            f"run_training_batch={mp3_p['run_training_batch']}s, rc={mp3_rc})"
        )
    else:
        print("  MP3 baseline   : [skipped]")
    if not args.skip_wav:
        print(
            f"  WAV experiment : {wav_time:6.1f}s wall  ≈ {wav_its:.2f} it/s  "
            f"(train_dataloader_next={wav_p['train_dataloader_next']}s, "
            f"run_training_batch={wav_p['run_training_batch']}s, rc={wav_rc})"
        )
    else:
        print("  WAV experiment : [skipped]")

    if not args.skip_mp3 and not args.skip_wav and mp3_its > 0:
        speedup = wav_its / mp3_its
        print(f"  Speedup (it/s) : {speedup:.2f}×")
        if speedup >= 1.5:
            verdict = "GREEN — proceed with full conversion"
        elif speedup >= 1.0:
            verdict = "YELLOW — marginal; inspect dataloader vs compute split"
        else:
            verdict = "RED — WAV slower than MP3; debug before rolling out"
        print(f"  Decision       : {verdict}\n")
    else:
        print("  Speedup (it/s) : [one or both runs skipped — compare profiler rows above]\n")


if __name__ == "__main__":
    main()
