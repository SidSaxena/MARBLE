#!/usr/bin/env python3
"""
scripts/verify_shs100k.py
─────────────────────────
Audit the downloaded SHS-100K audio dataset and the JSONL that points at it.

Checks performed
----------------
1. Every JSONL entry's ``audio_path`` exists on disk and is non-empty.
2. Optional: ffprobe metadata matches the values stored in the JSONL
   (sample_rate, num_samples, channels, duration).
3. Optional: torchaudio can actually open and read the file.
4. The audio directory has no orphans (files present but not referenced
   by the JSONL).
5. AppleDouble sidecar files (``._<name>.m4a`` created by macOS) are
   reported and optionally deleted.

Conversion / cleanup
--------------------
yt-dlp sometimes falls back to format 18 (mp4 360p with an AAC track) when
no audio-only stream is offered.  These ``.mp4`` files work with torchaudio,
but if you prefer a clean ``.m4a``-only dataset, use ``--convert-mp4`` to
remux them in-place (audio stream copied losslessly, no re-encode).

Usage
-----
  # Quick check (file existence + size only)
  python scripts/verify_shs100k.py

  # Custom paths
  python scripts/verify_shs100k.py \\
      --jsonl data/SHS100K/SHS100K.test.jsonl \\
      --audio-dir "/Volumes/WD Black/datasets/MARBLE"

  # Thorough: also run ffprobe on every file and compare to JSONL metadata
  python scripts/verify_shs100k.py --ffprobe

  # Most thorough: also try opening with torchaudio (slowest)
  python scripts/verify_shs100k.py --torchaudio

  # Clean up AppleDouble sidecars (macOS metadata files) and convert .mp4 → .m4a
  python scripts/verify_shs100k.py --clean-appledouble --convert-mp4
"""

import argparse
import concurrent.futures
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_AUDIO_EXTS = {".m4a", ".mp4", ".webm", ".mp3", ".ogg", ".opus", ".flac", ".wav"}

# Minimum file size for a "real" audio file — anything smaller is almost
# certainly a partial download.
_MIN_BYTES = 10_000


# ──────────────────────────────────────────────────────────────────────────────

def _ffprobe(path: Path) -> dict | None:
    """Return ffprobe metadata, or None on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    sr, channels, dur = 0, 1, 0.0
    stream_dur = None
    for s in data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        try:
            sr = int(s["sample_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        channels = int(s.get("channels") or 1)
        stream_dur = s.get("duration")
        break
    if sr == 0:
        return None
    for raw in [stream_dur, data.get("format", {}).get("duration")]:
        if raw is None:
            continue
        try:
            dur = float(raw)
            if dur > 0:
                break
        except (ValueError, TypeError):
            pass
    if dur <= 0:
        return None
    return {"sample_rate": sr, "channels": channels,
            "duration": dur, "num_samples": int(sr * dur)}


def _torchaudio_load(path: Path) -> bool:
    """Return True if torchaudio can open and read at least one frame."""
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.num_frames > 0 and info.sample_rate > 0
    except Exception:
        return False


def _check_entry(
    rec: dict,
    use_ffprobe: bool,
    use_torchaudio: bool,
    tolerance_sec: float,
) -> dict:
    """Run all checks against one JSONL record. Returns a result dict."""
    out = {
        "rec": rec,
        "status": "ok",          # ok | missing | empty | corrupt | mismatch
        "issue": "",
    }
    path = Path(rec["audio_path"])
    if not path.exists():
        out["status"] = "missing"
        out["issue"] = "file not found"
        return out
    size = path.stat().st_size
    if size < _MIN_BYTES:
        out["status"] = "empty"
        out["issue"] = f"only {size} bytes"
        return out

    if use_ffprobe:
        meta = _ffprobe(path)
        if meta is None:
            out["status"] = "corrupt"
            out["issue"] = "ffprobe failed"
            return out
        diffs = []
        if meta["sample_rate"] != rec["sample_rate"]:
            diffs.append(f"sr {rec['sample_rate']}≠{meta['sample_rate']}")
        if meta["channels"] != rec["channels"]:
            diffs.append(f"ch {rec['channels']}≠{meta['channels']}")
        if abs(meta["duration"] - rec["duration"]) > tolerance_sec:
            diffs.append(
                f"dur {rec['duration']:.2f}≠{meta['duration']:.2f}"
            )
        if diffs:
            out["status"] = "mismatch"
            out["issue"] = "; ".join(diffs)
            return out

    if use_torchaudio:
        if not _torchaudio_load(path):
            out["status"] = "corrupt"
            out["issue"] = "torchaudio.info failed"
            return out

    return out


def _format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Verify SHS-100K downloaded dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--jsonl", default="data/SHS100K/SHS100K.test.jsonl",
                    help="Path to JSONL file (default: %(default)s)")
    ap.add_argument("--audio-dir", default=None,
                    help="Directory containing the audio files. "
                         "If omitted, inferred from the first JSONL entry.")
    ap.add_argument("--ffprobe", action="store_true",
                    help="Run ffprobe on each file and compare to JSONL metadata.")
    ap.add_argument("--torchaudio", action="store_true",
                    help="Verify every file opens with torchaudio (slowest).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel workers for ffprobe / torchaudio (default: 8)")
    ap.add_argument("--tolerance-sec", type=float, default=0.5,
                    help="Duration mismatch tolerance in seconds (default: 0.5)")
    ap.add_argument("--clean-appledouble", action="store_true",
                    help="Delete macOS AppleDouble sidecar files (._*).")
    ap.add_argument("--convert-mp4", action="store_true",
                    help="Remux .mp4 files to .m4a (audio stream copied, no re-encode). "
                         "Updates the JSONL audio_path entries to point at the new files.")
    ap.add_argument("--show-first", type=int, default=20,
                    help="Print at most this many problem entries (default: 20)")
    args = ap.parse_args()

    # ── Load JSONL ────────────────────────────────────────────────────────────
    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        log.error(f"JSONL not found: {jsonl}")
        sys.exit(1)
    with open(jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info(f"JSONL: {jsonl}  ({len(records):,} entries)")

    if not records:
        log.error(
            f"JSONL is empty.  Most likely the downloader was re-run with "
            f"--skip-audio pointing at the wrong audio dir, overwriting it.\n"
            f"  Fix: regenerate the JSONL from the actual audio dir:\n"
            f"    python scripts/download_shs100k.py --skip-audio "
            f"--audio-dir <your-audio-dir>"
        )
        sys.exit(1)

    # Infer audio_dir from first record if not given
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
    else:
        audio_dir = Path(records[0]["audio_path"]).parent
    log.info(f"Audio dir: {audio_dir}")

    # ── Tool availability ────────────────────────────────────────────────────
    if args.ffprobe or args.convert_mp4:
        if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
            log.error("ffmpeg / ffprobe not found on PATH.  Install ffmpeg first.")
            sys.exit(1)

    # ── AppleDouble cleanup (macOS metadata sidecar files) ───────────────────
    sidecars = sorted(audio_dir.glob("._*"))
    if sidecars:
        if args.clean_appledouble:
            n = 0
            for f in sidecars:
                try:
                    f.unlink()
                    n += 1
                except OSError:
                    pass
            log.info(f"Deleted {n:,} AppleDouble sidecar files")
        else:
            log.warning(
                f"Found {len(sidecars):,} AppleDouble sidecar files (._*) — "
                "macOS metadata junk.  Re-run with --clean-appledouble to remove."
            )

    # ── mp4 → m4a remux (in-place audio copy) ────────────────────────────────
    if args.convert_mp4:
        mp4_paths = [Path(r["audio_path"]) for r in records
                     if r["audio_path"].endswith(".mp4")]
        if mp4_paths:
            log.info(f"Remuxing {len(mp4_paths):,} .mp4 → .m4a ...")
            ytid_to_newpath: dict[str, str] = {}
            for src in mp4_paths:
                dst = src.with_suffix(".m4a")
                if dst.exists() and dst.stat().st_size > _MIN_BYTES:
                    src.unlink(missing_ok=True)
                    ytid_to_newpath[src.stem] = str(dst)
                    continue
                r = subprocess.run(
                    ["ffmpeg", "-v", "quiet", "-y",
                     "-i", str(src), "-vn", "-c:a", "copy", str(dst)],
                    capture_output=True,
                )
                if r.returncode == 0 and dst.exists() and dst.stat().st_size > _MIN_BYTES:
                    src.unlink(missing_ok=True)
                    ytid_to_newpath[src.stem] = str(dst)
                else:
                    log.warning(f"  remux failed: {src.name}")
            # Update JSONL paths
            for rec in records:
                p = Path(rec["audio_path"])
                if p.stem in ytid_to_newpath:
                    rec["audio_path"] = ytid_to_newpath[p.stem]
            with open(jsonl, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.info(f"Remuxed {len(ytid_to_newpath):,} files; JSONL paths updated.")
        else:
            log.info("No .mp4 files to remux.")

    # ── Verify each JSONL entry ──────────────────────────────────────────────
    log.info(
        f"Verifying entries  "
        f"(ffprobe={'on' if args.ffprobe else 'off'}  "
        f"torchaudio={'on' if args.torchaudio else 'off'}) ..."
    )

    by_status: dict[str, list[dict]] = {
        "ok": [], "missing": [], "empty": [], "corrupt": [], "mismatch": [],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_check_entry, rec, args.ffprobe, args.torchaudio,
                        args.tolerance_sec): rec
            for rec in records
        }
        n_done = 0
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            by_status[res["status"]].append(res)
            n_done += 1
            if n_done % 500 == 0:
                pct = 100 * n_done // len(records)
                log.info(f"  [{pct:3d}%] {n_done}/{len(records)}")

    # ── Orphan detection ─────────────────────────────────────────────────────
    referenced = {Path(r["audio_path"]).resolve() for r in records}
    on_disk = {
        p.resolve()
        for p in audio_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _AUDIO_EXTS
        and not p.name.startswith("._")
    }
    orphans = sorted(on_disk - referenced)

    # ── Disk usage of all audio referenced ───────────────────────────────────
    total_bytes = 0
    for r in records:
        try:
            total_bytes += Path(r["audio_path"]).stat().st_size
        except OSError:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    n_ok        = len(by_status["ok"])
    n_missing   = len(by_status["missing"])
    n_empty     = len(by_status["empty"])
    n_corrupt   = len(by_status["corrupt"])
    n_mismatch  = len(by_status["mismatch"])
    total       = len(records)

    print()
    print("=" * 64)
    print(f" SHS-100K verification summary")
    print("=" * 64)
    print(f"  JSONL entries     {total:>6,}")
    print(f"    ✓ ok            {n_ok:>6,}  ({100*n_ok/total:.1f}%)")
    if n_missing:  print(f"    ✗ missing       {n_missing:>6,}")
    if n_empty:    print(f"    ✗ empty         {n_empty:>6,}")
    if n_corrupt:  print(f"    ✗ corrupt       {n_corrupt:>6,}")
    if n_mismatch: print(f"    ⚠ mismatch      {n_mismatch:>6,}  (metadata diverges)")
    print()
    print(f"  Files on disk     {len(on_disk):>6,}  ({_format_bytes(total_bytes)})")
    print(f"  Orphans on disk   {len(orphans):>6,}  (not in JSONL)")
    print(f"  AppleDouble files {len(sidecars):>6,}  "
          + ("(remaining)" if not args.clean_appledouble else "(deleted)"))
    print()
    print(f"  Unique works      {len({r['work_id'] for r in records}):>6,}")
    durations = [r["duration"] for r in records]
    if durations:
        print(f"  Audio duration    "
              f"total={sum(durations)/3600:.1f}h  "
              f"avg={sum(durations)/len(durations):.1f}s  "
              f"min={min(durations):.1f}s  max={max(durations):.1f}s")
    print("=" * 64)

    # ── Detail listings ──────────────────────────────────────────────────────
    def _list_problems(label: str, items: list[dict]):
        if not items: return
        print(f"\n── {label} ({len(items)}) — first {args.show_first} ──")
        for it in items[:args.show_first]:
            print(f"  {it['rec']['youtube_id']}  {it['issue']}  "
                  f"→ {it['rec']['audio_path']}")

    _list_problems("MISSING",  by_status["missing"])
    _list_problems("EMPTY",    by_status["empty"])
    _list_problems("CORRUPT",  by_status["corrupt"])
    _list_problems("MISMATCH", by_status["mismatch"])

    if orphans:
        print(f"\n── ORPHANS on disk ({len(orphans)}) — first {args.show_first} ──")
        for p in orphans[:args.show_first]:
            print(f"  {p.name}")

    # ── Exit code: 0 if clean, 1 if anything failed ──────────────────────────
    sys.exit(0 if (n_missing + n_empty + n_corrupt) == 0 else 1)


if __name__ == "__main__":
    main()
