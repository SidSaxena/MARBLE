#!/usr/bin/env python3
"""scripts/convert_shs100k_to_flac.py
─────────────────────────────────────
Convert SHS100K .m4a → .flac so torchaudio's soundfile backend can decode
them on any platform. SHS100K is the only MARBLE dataset that ships as
AAC/M4A, which libsndfile (the cross-platform default) can't read.

After running this once, the audio files are FLAC (lossless) and decode
via `soundfile` on Mac/Linux/Windows without needing a compatible
ffmpeg shared library installed. The JSONL is rewritten in-place to
point at the new files.

Disk cost: ~30 GB FLAC vs ~21 GB M4A (lossless is bigger).
Time cost: ~2–4 sec/file × 6905 files ÷ workers. At --workers 8 on a
modern CPU: ~30–60 min total.

Usage
-----
    # Convert in-place: writes .flac alongside .m4a, rewrites JSONL,
    # optionally deletes .m4a files on success.
    uv run python scripts/convert_shs100k_to_flac.py

    # Override paths (when audio lives somewhere other than the JSONL
    # references — e.g. moved between machines)
    uv run python scripts/convert_shs100k_to_flac.py \\
        --jsonl data/SHS100K/SHS100K.test.jsonl \\
        --audio-dir D:/datasets/SHS100K \\
        --out-dir D:/datasets/SHS100K

    # Keep originals (default deletes .m4a after successful conversion)
    uv run python scripts/convert_shs100k_to_flac.py --keep-originals

Prerequisites
-------------
Requires the `ffmpeg` CLI on PATH. (Even if torchaudio's Python
bindings can't see ffmpeg's shared libs, the CLI itself works fine —
that's the whole point of this conversion: decouple the inference-time
audio decode from the system ffmpeg.)
"""

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _resolve_input(rec: dict, audio_dir: Path | None) -> Path | None:
    """Find the source audio for a JSONL record, trying multiple paths."""
    candidates: list[Path] = []
    original = Path(rec["audio_path"])
    candidates.append(original)
    if audio_dir is not None:
        ytid = rec.get("youtube_id") or original.stem
        for ext in (".m4a", ".mp4", ".aac"):
            candidates.append(audio_dir / f"{ytid}{ext}")
    for c in candidates:
        if c.exists():
            return c
    return None


def _convert_one(rec: dict, audio_dir: Path | None, out_dir: Path,
                 keep_originals: bool) -> tuple[dict, str]:
    """Convert one record's audio to FLAC. Returns (updated_rec, status)."""
    src = _resolve_input(rec, audio_dir)
    if src is None:
        return rec, "missing"
    ytid = rec.get("youtube_id") or src.stem
    dst = out_dir / f"{ytid}.flac"

    if dst.exists() and dst.stat().st_size > 4096:
        # Already converted — just update the path and refresh metadata
        rec["audio_path"] = str(dst)
        return rec, "already"

    # ffmpeg: keep audio stream, encode to FLAC (lossless), no re-encode
    # of any video stream (-vn drops video — SHS100K m4a often has art tracks)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-i", str(src), "-vn", "-c:a", "flac", str(dst)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 4096:
        return rec, f"failed: {r.stderr.strip()[:120]}"

    rec["audio_path"] = str(dst)
    # Refresh metadata via ffprobe (path now points at the new FLAC)
    try:
        import json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(dst)],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            info = _json.loads(probe.stdout)
            astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
            if astreams:
                s = astreams[0]
                rec["sample_rate"] = int(s.get("sample_rate", rec["sample_rate"]))
                rec["channels"] = int(s.get("channels", rec["channels"]))
                duration = float(info.get("format", {}).get("duration", rec["duration"]))
                rec["duration"] = duration
                rec["num_samples"] = int(round(duration * rec["sample_rate"]))
    except Exception:
        pass  # metadata refresh is best-effort

    if not keep_originals and src.suffix.lower() != ".flac":
        try:
            src.unlink()
        except OSError:
            pass

    return rec, "converted"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--jsonl", type=Path, default=Path("data/SHS100K/SHS100K.test.jsonl"),
                    help="JSONL to convert/rewrite (default: %(default)s)")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="Fallback directory to search for <ytid>.m4a if the "
                         "JSONL's audio_path isn't reachable (e.g. files moved "
                         "to a different drive)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to write .flac files (default: same dir "
                         "as each source file)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel ffmpeg processes (default: 8)")
    ap.add_argument("--keep-originals", action="store_true",
                    help="Don't delete .m4a after successful conversion")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ERROR: ffmpeg / ffprobe not found on PATH.", file=sys.stderr)
        print("  macOS:   brew install ffmpeg", file=sys.stderr)
        print("  Windows: winget install Gyan.FFmpeg  (or scoop install ffmpeg)",
              file=sys.stderr)
        print("  Linux:   apt install ffmpeg", file=sys.stderr)
        sys.exit(1)

    if not args.jsonl.exists():
        print(f"ERROR: {args.jsonl} not found", file=sys.stderr)
        sys.exit(1)

    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records):,} records from {args.jsonl}")

    out_dir = args.out_dir
    if out_dir is None:
        # Default: write FLAC alongside each source file
        first_src = _resolve_input(records[0], args.audio_dir)
        if first_src is None:
            print("ERROR: can't resolve first record's audio_path; pass --out-dir",
                  file=sys.stderr)
            sys.exit(1)
        out_dir = first_src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing FLAC to: {out_dir}")

    t0 = time.time()
    counters = {"converted": 0, "already": 0, "missing": 0, "failed": 0}
    updated: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_convert_one, rec, args.audio_dir, out_dir, args.keep_originals): i
            for i, rec in enumerate(records)
        }
        for n, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            rec, status = fut.result()
            updated.append(rec)
            if status in counters:
                counters[status] += 1
            else:
                counters["failed"] += 1
            if n % 200 == 0 or n == len(records):
                elapsed = time.time() - t0
                rate = n / elapsed if elapsed else 0
                print(f"  [{n:>5}/{len(records)}]  {rate:.1f} files/s   "
                      f"converted={counters['converted']}  "
                      f"already={counters['already']}  "
                      f"missing={counters['missing']}  "
                      f"failed={counters['failed']}")

    print()
    print("=" * 64)
    print(" SHS100K → FLAC conversion summary")
    print("=" * 64)
    for k, v in counters.items():
        print(f"  {k:<10} {v:>6,}")
    print(f"  total      {len(records):>6,}")
    print(f"  elapsed    {(time.time()-t0)/60:.1f} min")
    print("=" * 64)

    if counters["failed"] == len(records):
        print("All conversions failed — JSONL not modified.", file=sys.stderr)
        sys.exit(2)

    # Rewrite JSONL with new paths (only entries we successfully resolved)
    with open(args.jsonl, "w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nRewrote {args.jsonl} with FLAC paths.")
    if counters["missing"] > 0:
        print(f"Note: {counters['missing']} records had no resolvable source "
              f"audio — they still have their original audio_path in the JSONL "
              f"and will fail at sweep time. Consider --audio-dir or "
              f"verify_shs100k.py --rewrite to drop them.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
