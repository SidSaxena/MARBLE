#!/usr/bin/env python3
"""
scripts/data/postprocess_vgmiditvar.py
──────────────────────────────────────
Reverb + EBU R128 loudness normalization for VGMIDITVar-timbre renders.

Why
───
Raw ``fluidsynth + SoundFont`` output is anechoic (no room acoustics)
and per-program volumes vary by ±12 dB depending on SoundFont sample
levels. Both are confounds for the cross-instrument timbre retrieval
test on the VGMIDITVar-timbre variant: pre-trained audio encoders
(MERT / MuQ / CLaMP3) were trained on commercial music that's recorded
in rooms and mastered to typical streaming loudness, so dry-and-quiet
SoundFont output is well outside their training distribution.

This script applies, in a single ffmpeg pass per file:
  1. **Convolution reverb** via ``afir`` with the bundled small-hall IR
     (see ``scripts/data/ir/small_hall_1.2s.wav``) — adds realistic ambience.
  2. **EBU R128 loudness normalization** via ``loudnorm`` to a uniform
     perceptual loudness (default ``-16 LUFS`` integrated, ``-1.5 dBFS``
     true peak).

The same reverb is applied to every program so reverb-character is held
constant across the 8 GM programs — it adds a constant acoustic context
without confounding the cross-instrument comparison.

Atomic + idempotent
───────────────────
Each output is written to ``<dst>.tmp.<pid>`` then renamed. A
``.postprocessed`` sentinel sidecar is dropped next to each processed
file so reruns skip-instantly without invoking ffmpeg. Use ``--force``
to reprocess.

Backups for A/B comparison
──────────────────────────
``--backup-sample N`` copies the first N source files (sorted by name)
to ``<src-dir>/../audio_preprocess_backup/`` BEFORE any processing.
Useful for manually comparing pre- and post-processed audio quality.

Usage
─────
Default — in-place processing of VGMIDITVar-timbre with 8 backup
samples for inspection::

    uv run python scripts/data/postprocess_vgmiditvar.py \\
        --src-dir data/VGMIDITVar-timbre/audio \\
        --backup-sample 8

Dry-run (print plan, no ffmpeg)::

    uv run python scripts/data/postprocess_vgmiditvar.py \\
        --src-dir data/VGMIDITVar-timbre/audio --dry-run

Apply to a different variant or alternative IR::

    uv run python scripts/data/postprocess_vgmiditvar.py \\
        --src-dir data/VGMIDITVar-leitmotif/audio \\
        --ir data/ir/medium_hall_1.8s.wav --target-lufs -18
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

# Sentinel sidecar marks a file as already processed so reruns are O(N)
# file-stats instead of N ffmpeg invocations. Cheap to set, cheap to
# check, easy to ``rm`` to force a re-run on a subset.
_SENTINEL_SUFFIX = ".postprocessed"


def _is_already_processed(path: Path) -> bool:
    return path.with_suffix(path.suffix + _SENTINEL_SUFFIX).exists()


def _mark_processed(path: Path) -> None:
    path.with_suffix(path.suffix + _SENTINEL_SUFFIX).touch()


# ── ffmpeg pipeline ───────────────────────────────────────────────────────────


def _build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    *,
    ir_path: Path,
    wet_db: float,
    dry_db: float,
    target_lufs: float,
    true_peak: float,
    lra: float,
) -> list[str]:
    """Single-pass ffmpeg: convolution reverb → loudness normalize.

    The ``afir`` filter takes the IR as a SECOND input stream (not as
    a filter option), so the IR file is passed via ``-i`` and the dry
    signal + IR are wired through ``-filter_complex``. dry/wet are
    afir's gain options in dB; we bias toward dry (10 dB vs 3 dB
    ≈ ~15% wet mix). ``loudnorm`` then maps the wet output to a
    uniform integrated loudness, true-peak ceiling, and loudness range
    — the EBU R128 broadcast triplet.

    Note: ``loudnorm`` runs single-pass here (no measure-then-apply
    two-pass). Single-pass is ~2× faster and within ±1 LU of two-pass
    for our use case (uniform-target retrieval, not broadcast delivery).
    """
    filter_complex = (
        f"[0:a][1:a]afir=dry={dry_db}:wet={wet_db}[wet];"
        f"[wet]loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}[out]"
    )
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-i",
        str(ir_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        # Preserve container (FLAC in → FLAC out) and 44.1 kHz / mono.
        "-c:a",
        "flac",
        "-ar",
        "44100",
        "-ac",
        "1",
        "-f",
        "flac",
        str(dst),
    ]


def _process_one(
    src: Path,
    *,
    ir_path: Path,
    wet_db: float,
    dry_db: float,
    target_lufs: float,
    true_peak: float,
    lra: float,
    force: bool,
) -> tuple[Path, bool, str]:
    """Convert in-place. Returns (src, ok, message)."""
    if not force and _is_already_processed(src):
        return src, True, "already"

    tmp = src.with_name(f"{src.name}.tmp.{os.getpid()}")
    cmd = _build_ffmpeg_cmd(
        src,
        tmp,
        ir_path=ir_path,
        wet_db=wet_db,
        dry_db=dry_db,
        target_lufs=target_lufs,
        true_peak=true_peak,
        lra=lra,
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return src, False, "timeout"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return src, False, f"exception: {e}"

    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        msg = (r.stderr or "")[:160].strip() or f"rc={r.returncode}"
        return src, False, msg

    try:
        os.replace(tmp, src)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return src, False, f"rename: {e}"

    _mark_processed(src)
    return src, True, "ok"


# ── backup ────────────────────────────────────────────────────────────────────


def _backup_sample(src_dir: Path, n: int) -> Path:
    """Copy the first ``n`` audio files (sorted by name) to a sibling
    backup directory. Used for manual A/B comparison post-processing.
    Returns the backup directory path."""
    files = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in (".flac", ".wav")])
    if not files:
        log.warning("No audio files found under %s — skipping backup", src_dir)
        return src_dir.parent / "audio_preprocess_backup"
    sample = files[:n]
    backup_dir = src_dir.parent / "audio_preprocess_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in sample:
        rel = f.relative_to(src_dir)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        shutil.copy2(f, dst)
    log.info("Backed up %d sample file(s) to %s", len(sample), backup_dir)
    return backup_dir


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/VGMIDITVar-timbre/audio"),
        help="Audio directory (FLAC/WAV files, in-place processed). Default: %(default)s",
    )
    ap.add_argument(
        "--ir",
        type=Path,
        default=Path("scripts/data/ir/small_hall_1.2s.wav"),
        help="Convolution-reverb impulse response. Default: %(default)s",
    )
    ap.add_argument(
        "--wet-db",
        type=float,
        default=3.0,
        help="afir wet gain in dB (default 3 — ~15%% wet vs 10 dB dry).",
    )
    ap.add_argument(
        "--dry-db",
        type=float,
        default=10.0,
        help="afir dry gain in dB (default 10).",
    )
    ap.add_argument(
        "--target-lufs",
        type=float,
        default=-16.0,
        help="EBU R128 integrated loudness target. Default -16 LUFS (typical streaming).",
    )
    ap.add_argument(
        "--true-peak",
        type=float,
        default=-1.5,
        help="True-peak ceiling in dBFS. Default -1.5.",
    )
    ap.add_argument(
        "--lra",
        type=float,
        default=11.0,
        help="Loudness range in LU. Default 11.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel ffmpeg workers. Default 8.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if .postprocessed sentinel exists.",
    )
    ap.add_argument(
        "--backup-sample",
        type=int,
        default=0,
        help="Copy first N source files (sorted) to <src>/../audio_preprocess_backup/ "
        "before processing. Useful for A/B comparison.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many files (smoke / pilot mode).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + ffmpeg command preview, do not invoke ffmpeg.",
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None and not args.dry_run:
        log.error("ffmpeg not on PATH.")
        return 1
    if not args.ir.exists():
        log.error("--ir file not found: %s", args.ir)
        return 1
    if not args.src_dir.is_dir():
        log.error("--src-dir is not a directory: %s", args.src_dir)
        return 1

    # Discover audio files
    files = sorted([p for p in args.src_dir.rglob("*") if p.suffix.lower() in (".flac", ".wav")])
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        log.warning("No FLAC/WAV files found under %s", args.src_dir)
        return 0

    n_already = sum(1 for f in files if _is_already_processed(f) and not args.force)
    n_todo = len(files) - n_already

    # Backup BEFORE processing
    if args.backup_sample > 0:
        _backup_sample(args.src_dir, args.backup_sample)

    log.info("─" * 60)
    log.info("VGMIDITVar-timbre post-processing plan")
    log.info("─" * 60)
    log.info("  --src-dir       : %s", args.src_dir)
    log.info("  --ir            : %s", args.ir)
    log.info("  dry/wet         : %.1f / %.1f dB", args.dry_db, args.wet_db)
    log.info("  target LUFS     : %.1f", args.target_lufs)
    log.info("  true peak       : %.1f dBFS", args.true_peak)
    log.info("  loudness range  : %.1f LU", args.lra)
    log.info("  --workers       : %d", args.workers)
    log.info("  files found     : %d", len(files))
    log.info("  already done    : %d", n_already)
    log.info("  to process      : %d", n_todo)
    log.info("─" * 60)

    if args.dry_run:
        if files:
            example = _build_ffmpeg_cmd(
                files[0],
                files[0].with_name(files[0].name + ".tmp"),
                ir_path=args.ir,
                wet_db=args.wet_db,
                dry_db=args.dry_db,
                target_lufs=args.target_lufs,
                true_peak=args.true_peak,
                lra=args.lra,
            )
            log.info("Example ffmpeg cmd:\n  %s", " ".join(example))
        return 0

    # Parallel process
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                _process_one,
                f,
                ir_path=args.ir,
                wet_db=args.wet_db,
                dry_db=args.dry_db,
                target_lufs=args.target_lufs,
                true_peak=args.true_peak,
                lra=args.lra,
                force=args.force,
            ): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            _, ok, msg = fut.result()
            if ok:
                if msg == "already":
                    n_skipped += 1
                else:
                    n_ok += 1
            else:
                n_failed += 1
                failures.append((futs[fut], msg))
            if i % 200 == 0 or i == len(files):
                log.info(
                    "  [%6d/%6d] ok=%d skipped=%d failed=%d",
                    i,
                    len(files),
                    n_ok,
                    n_skipped,
                    n_failed,
                )

    log.info("=" * 60)
    log.info(" Done: %d processed, %d skipped, %d failed", n_ok, n_skipped, n_failed)
    if failures:
        log.warning(" First few failures:")
        for f, msg in failures[:5]:
            log.warning("   - %s: %s", f.name, msg)
    log.info("=" * 60)
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
