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
  1. **Convolution reverb** via ``afir`` with a real-room IR you supply
     via ``--ir`` (recommend Samplicity Bricasti M7 Small Hall, free
     download — downmix the stereo IR to mono first via
     ``ffmpeg -i in.wav -ac 1 -ar 44100 out.wav``).
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
        --src-dir data/VGMIDITVar/audio \\
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
    ir_path: Path | None,
    wet: float,
    dry: float,
    target_lufs: float,
    true_peak: float,
    lra: float,
) -> list[str]:
    """Single-pass ffmpeg: (optional) convolution reverb → loudness normalize.

    When ``ir_path`` is None, the reverb stage is skipped — only the
    ``loudnorm`` filter runs. This is the recommended config for game
    music sources (chiptune / synth / JRPG-style scoring) which are
    typically dry-mixed in their training distribution.

    When ``ir_path`` is supplied, the ``afir`` filter takes the IR as
    SECOND input stream. ``dry`` and ``wet`` are LINEAR gains
    in [0, 10] (NOT dB) applied to the dry and reverberated signals
    respectively. Mix percentage = wet / (dry + wet). E.g.:
        dry=10 wet=0.3  → ~3% wet — very subtle
        dry=10 wet=0.5  → ~5% wet — subtle
        dry=10 wet=1.0  → ~9% wet — modest
        dry=10 wet=1.5  → ~13% wet — present
        dry=1  wet=1    → 50% wet — drenched (afir default)

    Note: ``loudnorm`` runs single-pass here (no measure-then-apply
    two-pass). Single-pass is ~2× faster and within ±1 LU of two-pass
    for our use case (uniform-target retrieval, not broadcast delivery).
    """
    common = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
    ]
    if ir_path is None:
        # Loudness-only path: simple -af chain, single input.
        return common + [
            "-af",
            f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}",
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
    # Reverb + loudness path: afir takes IR as second input.
    filter_complex = (
        f"[0:a][1:a]afir=dry={dry}:wet={wet}[wet];"
        f"[wet]loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}[out]"
    )
    return common + [
        "-i",
        str(ir_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
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
    dst: Path,
    *,
    ir_path: Path | None,
    wet: float,
    dry: float,
    target_lufs: float,
    true_peak: float,
    lra: float,
    force: bool,
) -> tuple[Path, bool, str]:
    """Process ``src`` → ``dst`` (atomic, idempotent). Returns (src, ok, msg).

    The ``.postprocessed`` sentinel is dropped next to ``dst`` (not
    ``src``) so the source tree stays untouched in non-in-place mode.
    """
    if not force and _is_already_processed(dst):
        return src, True, "already"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    cmd = _build_ffmpeg_cmd(
        src,
        tmp,
        ir_path=ir_path,
        wet=wet,
        dry=dry,
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
        os.replace(tmp, dst)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return src, False, f"rename: {e}"

    _mark_processed(dst)
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
        help="Source audio directory (FLAC/WAV). Default: %(default)s",
    )
    ap.add_argument(
        "--dst-dir",
        type=Path,
        default=None,
        help="Destination audio directory (mirrors --src-dir layout). "
        "Default: ``<src-dir>/../audio_processed/``. The source tree is "
        "NEVER modified — pass --in-place to override.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Process in-place, overwriting source files. WARNING: not "
        "iteration-safe — once applied, you cannot undo the reverb without "
        "re-rendering. Default is to write to a separate --dst-dir.",
    )
    ap.add_argument(
        "--ir",
        type=Path,
        default=None,
        help="Convolution-reverb impulse response (mono WAV recommended). "
        "Omit to skip reverb entirely (loudness-norm only). "
        "Source: a real-room IR like Samplicity Bricasti M7 Vocal Plate "
        "or Small Room (free download, downmix to mono via "
        "`ffmpeg -i in.wav -ac 1 -ar 44100 out.wav`).",
    )
    ap.add_argument(
        "--wet",
        type=float,
        default=0.5,
        help="afir wet linear gain in [0, 10]. NOT dB. Default 0.5 "
        "(~5%% wet vs --dry 10). Examples: 0.3 (~3%% wet, very subtle), "
        "0.5 (~5%%, subtle), 1.0 (~9%%, modest), 1.5 (~13%%, present).",
    )
    ap.add_argument(
        "--dry",
        type=float,
        default=10.0,
        help="afir dry linear gain in [0, 10]. Default 10 (keep dry "
        "near ceiling; loudnorm rescales overall level after).",
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
    if args.ir is not None and not args.ir.exists():
        log.error("--ir file not found: %s", args.ir)
        return 1
    if not args.src_dir.is_dir():
        log.error("--src-dir is not a directory: %s", args.src_dir)
        return 1
    if args.in_place and args.dst_dir is not None:
        log.error("--in-place and --dst-dir are mutually exclusive.")
        return 1

    # Resolve dst dir
    if args.in_place:
        dst_root = args.src_dir
    elif args.dst_dir is not None:
        dst_root = args.dst_dir
    else:
        dst_root = args.src_dir.parent / "audio_processed"

    # Discover audio files
    files = sorted([p for p in args.src_dir.rglob("*") if p.suffix.lower() in (".flac", ".wav")])
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        log.warning("No FLAC/WAV files found under %s", args.src_dir)
        return 0

    # Build (src, dst) pairs preserving relative layout.
    pairs = [(src, dst_root / src.relative_to(args.src_dir)) for src in files]

    n_already = sum(1 for _, dst in pairs if _is_already_processed(dst) and not args.force)
    n_todo = len(pairs) - n_already

    # Backup BEFORE processing (only meaningful in --in-place mode; in
    # separate-dst mode the source IS the backup).
    if args.backup_sample > 0 and args.in_place:
        _backup_sample(args.src_dir, args.backup_sample)

    if args.ir is not None:
        total = args.dry + args.wet
        wet_pct = (args.wet / total) * 100 if total > 0 else 0.0
        reverb_desc = f"{args.ir.name} (dry={args.dry:.2f} wet={args.wet:.2f} → {wet_pct:.1f}% wet)"
    else:
        reverb_desc = "<none — loudness-only>"
    log.info("─" * 60)
    log.info("VGMIDITVar post-processing plan")
    log.info("─" * 60)
    log.info("  --src-dir       : %s", args.src_dir)
    log.info(
        "  --dst-dir       : %s%s",
        dst_root,
        " (in-place)" if args.in_place else "",
    )
    log.info("  reverb          : %s", reverb_desc)
    log.info("  target LUFS     : %.1f", args.target_lufs)
    log.info("  true peak       : %.1f dBFS", args.true_peak)
    log.info("  loudness range  : %.1f LU", args.lra)
    log.info("  --workers       : %d", args.workers)
    log.info("  files found     : %d", len(pairs))
    log.info("  already done    : %d", n_already)
    log.info("  to process      : %d", n_todo)
    log.info("─" * 60)

    if args.dry_run:
        if pairs:
            src0, dst0 = pairs[0]
            example = _build_ffmpeg_cmd(
                src0,
                dst0.with_name(dst0.name + ".tmp"),
                ir_path=args.ir,
                wet=args.wet,
                dry=args.dry,
                target_lufs=args.target_lufs,
                true_peak=args.true_peak,
                lra=args.lra,
            )
            log.info("Example ffmpeg cmd:\n  %s", " ".join(example))
        return 0

    # Hard Ctrl-C handler — ThreadPoolExecutor's default waits for in-
    # flight workers, which with ffmpeg subprocesses can hang the terminal
    # for tens of seconds. SIGINT → immediate exit so the user always gets
    # the prompt back after one Ctrl-C.
    import signal as _signal

    def _bail(sig, frame):  # noqa: ARG001
        log.warning("Ctrl-C — terminating worker pool immediately")
        os._exit(130)

    _signal.signal(_signal.SIGINT, _bail)

    # Parallel process
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                _process_one,
                src,
                dst,
                ir_path=args.ir,
                wet=args.wet,
                dry=args.dry,
                target_lufs=args.target_lufs,
                true_peak=args.true_peak,
                lra=args.lra,
                force=args.force,
            ): src
            for src, dst in pairs
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
