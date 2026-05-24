#!/usr/bin/env python3
"""Convert an audio corpus to flat 24 kHz mono int16 WAV.

Why
───
HookTheory MP3s are full-length songs at native sample rate (~44.1 kHz
stereo, ~10 MB/file). The DataLoader's hot path does, per __getitem__:
  1. `torchaudio.load(path, frame_offset=…, num_frames=…)` — MP3 decode + seek
  2. channel downmix / pick
  3. resample to encoder target (typically 24 kHz mono)
  4. pad/truncate

Steps (1) + (3) are CPU work. On Modal we measured them dominating
~85 % of training runtime (the dataloader wait, not GPU compute). If the
files are already 24 kHz mono PCM WAVs, (1) becomes a zero-cost byte
seek + memcpy and (3) is a no-op.

Tradeoff
────────
WAV at 24 kHz mono int16 ≈ 48 kB/s ≈ 8.6 MB per 3-minute clip — slightly
smaller than the native MP3 in this corpus (~10 MB/file) because we drop
the second channel and downsample. So for HookTheory you actually save
disk space. (At higher target sample rates or stereo, WAV gets bigger
fast.)

Idempotency
───────────
Skips any destination file that already exists with non-zero size and
matches the target sr/channels (verified via torchaudio.info). Use
``--force`` to overwrite.

Usage
─────
    # Local
    uv run python scripts/data/convert_audio_to_wav.py \
        --src-dir data/HookTheory/audio \
        --dst-dir data/HookTheory/audio_wav \
        --target-sr 24000 --channels 1 --workers 16

    # Modal: see modal_marble.py::convert_hooktheory_to_wav
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _get_nested(d: dict, dotted_key: str) -> Any:
    """Resolve a dotted key like ``youtube.id`` from a nested dict.

    Copied from ``scripts/data/cache_audio_info_in_jsonl.py:72`` to keep
    this script self-contained — the two scripts intentionally don't
    share a module so either can be vendored elsewhere without
    cross-imports.
    """
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


def _convert_one(
    src: Path, dst: Path, target_sr: int, channels: int, force: bool
) -> tuple[Path, str, int]:
    """Convert one file via ffmpeg. Returns (src, status, dst_bytes).

    status ∈ {"ok", "skipped", "missing-src", "ffmpeg-failed"}.

    ffmpeg flags chosen for deterministic output:
      -y           overwrite (we guard separately with `force`)
      -i SRC       source
      -ac CHANNELS target channel count (1 = mono — ffmpeg averages all input channels)
      -ar SR       target sample rate (linear-phase resample)
      -sample_fmt s16  16-bit PCM
      -map_metadata -1  strip all metadata (smaller files, deterministic)
      -loglevel error  quiet on success, complain on error
    """
    if not src.exists():
        return src, "missing-src", 0
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return src, "skipped", dst.stat().st_size

    # Atomic: write to a sibling .tmp then rename. Avoids half-written WAVs
    # confusing future idempotency checks.
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-ac",
                str(channels),
                "-ar",
                str(target_sr),
                "-sample_fmt",
                "s16",
                "-map_metadata",
                "-1",
                # Explicit container: tmp filename ends in ".tmp" so ffmpeg
                # can't guess the format. We *want* the tmp suffix for the
                # atomic-write pattern (so a Ctrl-C / OOM partial leaves
                # ``<id>.wav.tmp`` rather than a half-written ``<id>.wav``).
                "-f",
                "wav",
                "-loglevel",
                "error",
                str(tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        msg = e.stderr.decode("utf-8", errors="replace")[:200] if e.stderr else "(no stderr)"
        print(f"  ! ffmpeg failed: {src.name}: {msg}", file=sys.stderr)
        return src, "ffmpeg-failed", 0
    tmp.replace(dst)
    return src, "ok", dst.stat().st_size


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src-dir", type=Path, required=True, help="Source audio dir.")
    ap.add_argument("--dst-dir", type=Path, required=True, help="Destination WAV dir.")
    ap.add_argument(
        "--src-ext",
        default=".mp3",
        help="Source file extension (default: .mp3). Only files matching this are scanned.",
    )
    ap.add_argument(
        "--from-jsonl",
        type=Path,
        action="append",
        default=None,
        help=(
            "Repeatable. Instead of globbing --src-dir, build the work list "
            "from audio IDs found in each JSONL (resolved via --id-key + "
            "--src-ext). Use this for smoke-testing: pass smoke JSONLs and "
            "only the ~N referenced source files are converted."
        ),
    )
    ap.add_argument(
        "--id-key",
        default="youtube.id",
        help=(
            "Dotted JSONL key whose value is the audio file's stem "
            "(default: youtube.id for HookTheory). Only used with --from-jsonl."
        ),
    )
    ap.add_argument("--target-sr", type=int, default=24000, help="Target sample rate (Hz).")
    ap.add_argument("--channels", type=int, default=1, help="Target channel count (1 = mono mix).")
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel ffmpeg workers (default: 16). Each one spawns a separate ffmpeg.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing WAVs.")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (apt install ffmpeg / brew install ffmpeg)."
        )
    if not args.src_dir.exists():
        raise SystemExit(f"--src-dir does not exist: {args.src_dir}")
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    src_ext = args.src_ext if args.src_ext.startswith(".") else f".{args.src_ext}"
    if args.from_jsonl:
        # JSONL-driven work list: collect every <id_key> value across the
        # given JSONLs, dedupe (multiple splits can share songs in some
        # corpora), resolve each to <src_dir>/<id><src_ext>. Records with
        # a missing id_key are skipped silently (the JSONL is allowed to
        # contain malformed rows — the user's full convert path won't see
        # those either).
        seen: set[str] = set()
        for jsonl in args.from_jsonl:
            if not jsonl.exists():
                raise SystemExit(f"--from-jsonl path does not exist: {jsonl}")
            with jsonl.open() as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    audio_id = _get_nested(json.loads(line), args.id_key)
                    if audio_id is None:
                        continue
                    seen.add(str(audio_id))
        src_files = sorted(args.src_dir / f"{aid}{src_ext}" for aid in seen)
        print(
            f"\nconvert_audio_to_wav: {len(src_files):,} {src_ext} files referenced by "
            f"{len(args.from_jsonl)} JSONL(s) in {args.src_dir}\n"
            f"  → {args.dst_dir} @ {args.target_sr} Hz × {args.channels} ch (int16)\n"
            f"  workers={args.workers} force={args.force} id_key={args.id_key!r}",
            flush=True,
        )
    else:
        src_files = sorted(args.src_dir.glob(f"*{src_ext}"))
        print(
            f"\nconvert_audio_to_wav: {len(src_files):,} {src_ext} files in {args.src_dir}\n"
            f"  → {args.dst_dir} @ {args.target_sr} Hz × {args.channels} ch (int16)\n"
            f"  workers={args.workers} force={args.force}",
            flush=True,
        )
    if not src_files:
        return

    work = []
    for src in src_files:
        dst = args.dst_dir / (src.stem + ".wav")
        work.append((src, dst, args.target_sr, args.channels, args.force))

    # ProcessPoolExecutor (not Thread) because ffmpeg is a subprocess —
    # threads buy us nothing over processes here, and processes avoid
    # any GIL surprises during the subprocess.wait() phase.
    PRINT_EVERY = max(50, len(work) // 50)
    PRINT_INTERVAL = 10.0
    ok = skipped = failed = missing = 0
    total_bytes = 0
    start = time.time()
    last_print = start
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_convert_one, *w): w[0] for w in work}
        done = 0
        for fut in as_completed(futures):
            _, status, dst_bytes = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                total_bytes += dst_bytes
            elif status == "skipped":
                skipped += 1
                total_bytes += dst_bytes
            elif status == "missing-src":
                missing += 1
            else:
                failed += 1

            now = time.time()
            if done % PRINT_EVERY == 0 or (now - last_print) >= PRINT_INTERVAL:
                elapsed = now - start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = len(work) - done
                eta = remaining / rate if rate > 0 else float("inf")
                pct = 100.0 * done / len(work)
                bar_width = 30
                filled = int(bar_width * done / len(work))
                bar = "█" * filled + "░" * (bar_width - filled)
                print(
                    f"  [{bar}] {pct:5.1f}%  {done:>6,}/{len(work):,}  "
                    f"{rate:5.1f} files/s  eta {_fmt_eta(eta):>5}  "
                    f"(ok={ok} skip={skipped} fail={failed} miss={missing})  "
                    f"{total_bytes / 1e9:.1f} GB",
                    flush=True,
                )
                last_print = now

    elapsed = time.time() - start
    print(
        f"\n━━━ Done in {_fmt_eta(elapsed)} ━━━\n"
        f"  converted:        {ok:,}\n"
        f"  already present:  {skipped:,}\n"
        f"  ffmpeg-failed:    {failed:,}\n"
        f"  missing source:   {missing:,}\n"
        f"  total WAV bytes:  {total_bytes / 1e9:.2f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
