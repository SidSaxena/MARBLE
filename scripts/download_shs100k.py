#!/usr/bin/env python3
"""
scripts/download_shs100k.py
───────────────────────────
Download SHS-100K (Second Hand Songs 2025) and build MARBLE JSONL metadata.

Dataset
-------
  Source : https://github.com/second-hand-songs/shs-100k  (branch: 2025)
  Scale  : 10,000 works, ~110,000 cover performances split into:
    test.csv      500 works,  ~7,100 tracks  ← community benchmark (default)
    validate.csv  500 works,  ~5,000 tracks
    train.csv   9,000 works, ~100,000 tracks

  CSV format — no header row, 5 columns:
    performance_id, work_id, title, artist, youtube_id
  Artists with commas are quoted (standard CSV quoting).

Audio
-----
  Downloaded via yt-dlp. Format: m4a (AAC) where available, webm/opus
  otherwise.  Audio metadata (sample_rate, duration, channels) is read with
  ffprobe — torchaudio is NOT used here because it can't decode m4a/webm on
  Windows without the FFmpeg backend.

Prerequisites
-------------
  yt-dlp  — pip install yt-dlp  (or: uv sync, it's in project deps)
  ffmpeg  — required for ffprobe:
    Windows : winget install Gyan.FFmpeg   (restart terminal after)
    macOS   : brew install ffmpeg
    Linux   : sudo apt install ffmpeg

Usage
-----
  # Test split (community benchmark, ~7,100 tracks)
  python scripts/download_shs100k.py

  # With browser cookies to bypass YouTube bot-detection
  python scripts/download_shs100k.py --browser firefox
  python scripts/download_shs100k.py --browser edge   # close Edge first

  # With a pre-exported cookie file (safe for parallel workers)
  #   Export once:
  #   yt-dlp --cookies-from-browser firefox --cookies cookies.txt \\
  #           --skip-download "https://youtube.com/watch?v=rblt2EtFfC4"
  python scripts/download_shs100k.py --cookies-file cookies.txt --workers 4

  # Rebuild JSONL from already-downloaded files (no new downloads)
  python scripts/download_shs100k.py --skip-audio

  # Quick smoke-test (10 tracks)
  python scripts/download_shs100k.py --max-entries 10 --browser firefox
"""

import argparse
import csv
import io
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_CSV_BASE = "https://raw.githubusercontent.com/second-hand-songs/shs-100k/2025"
_CSV_URLS = {
    "test":  f"{_CSV_BASE}/test.csv",
    "val":   f"{_CSV_BASE}/validate.csv",
    "train": f"{_CSV_BASE}/train.csv",
}
_AUDIO_EXTS = {".m4a", ".webm", ".mp3", ".ogg", ".opus", ".flac", ".wav"}


# ── CSV ───────────────────────────────────────────────────────────────────────

def _fetch_csv(split: str) -> list[dict]:
    """Download and parse one SHS-100K CSV split from GitHub."""
    url = _CSV_URLS[split]
    log.info(f"Fetching {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    rows = []
    for i, row in enumerate(csv.reader(io.StringIO(text))):
        if len(row) < 5:
            continue
        try:
            rows.append({
                "perf_id":    int(row[0]),
                "work_id":    int(row[1]),
                "title":      row[2].strip(),
                "artist":     row[3].strip(),
                "youtube_id": row[4].strip(),
            })
        except (ValueError, IndexError) as e:
            log.debug(f"Skipping malformed CSV row {i}: {row!r} ({e})")
    log.info(f"  {len(rows):,} entries loaded from {split}")
    return rows


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _find_audio(ytid: str, audio_dir: Path) -> Optional[Path]:
    """Return an existing audio file for this YouTube ID, or None."""
    for path in audio_dir.glob(f"{ytid}.*"):
        if path.suffix.lower() in _AUDIO_EXTS and path.stat().st_size > 10_000:
            return path
    return None


def _ffprobe_info(path: Path) -> tuple[int, int, int]:
    """
    Return (sample_rate, num_frames, channels) via ffprobe.
    Returns (0, 0, 1) on any failure.

    Uses both -show_streams and -show_format so duration is found even for
    formats (e.g. some webm) where the stream-level duration field is absent.
    """
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        log.error("ffprobe not found. Install ffmpeg and restart your terminal.")
        return 0, 0, 1
    except subprocess.TimeoutExpired:
        log.warning(f"ffprobe timed out: {path.name}")
        return 0, 0, 1

    if r.returncode != 0:
        log.debug(f"ffprobe rc={r.returncode} for {path.name}: {r.stderr.strip()[:120]}")
        return 0, 0, 1

    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        log.debug(f"ffprobe returned non-JSON for {path.name}")
        return 0, 0, 1

    # Find the first audio stream
    sr = 0
    channels = 1
    stream_dur: Optional[str] = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        try:
            sr = int(stream["sample_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        channels = int(stream.get("channels") or 1)
        stream_dur = stream.get("duration")
        break

    if sr == 0:
        log.debug(f"No audio stream found by ffprobe: {path.name}")
        return 0, 0, 1

    # Duration: stream level → format level (both are in fractional seconds)
    dur = 0.0
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
        log.debug(f"ffprobe returned zero/missing duration: {path.name}")
        return 0, 0, 1

    return sr, int(sr * dur), channels


def _download(ytid: str, audio_dir: Path, cookie_args: list[str]) -> Optional[Path]:
    """
    Download audio for one YouTube video via yt-dlp.

    Format selection: m4a (AAC) first; any audio-only stream as fallback.
    Returns the Path of the downloaded file, or None on failure.
    Expected failures (private/deleted/geo-blocked) are logged at DEBUG.
    Actionable failures (bot-check, network errors) are logged at WARNING.
    """
    existing = _find_audio(ytid, audio_dir)
    if existing:
        return existing

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings",
        "--no-playlist",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-o", str(audio_dir / f"{ytid}.%(ext)s"),
    ] + cookie_args + [f"https://www.youtube.com/watch?v={ytid}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        log.warning(f"[timeout] {ytid}: yt-dlp exceeded 180 s")
        return None
    except Exception as e:
        log.warning(f"[error] {ytid}: {e}")
        return None

    if result.returncode == 0:
        return _find_audio(ytid, audio_dir)

    err = (result.stderr or result.stdout or "").strip()
    if "Sign in" in err or "bot" in err.lower():
        log.warning(
            f"[bot-check] {ytid}: YouTube requires auth — "
            "add --browser firefox (or export cookies with --cookies-file)."
        )
    elif any(p in err for p in [
        "unavailable", "private", "removed",
        "not available", "does not exist", "account associated",
    ]):
        log.debug(f"[skip] {ytid}: private/deleted/geo-blocked")
    else:
        log.warning(f"[yt-dlp rc={result.returncode}] {ytid}: {err[:200]}")
    return None


def _process(
    row: dict,
    audio_dir: Path,
    skip_audio: bool,
    cookie_args: list[str],
) -> Optional[dict]:
    """Download (or locate) one track and return its JSONL record, or None."""
    ytid = row["youtube_id"]

    if skip_audio:
        path = _find_audio(ytid, audio_dir)
        if path is None:
            return None
    else:
        path = _download(ytid, audio_dir, cookie_args)
        if path is None:
            return None

    sr, n_frames, channels = _ffprobe_info(path)
    if sr == 0 or n_frames == 0:
        log.warning(f"Could not read audio info for {path.name} — skipping")
        return None

    return {
        "audio_path":     str(path),
        "work_id":        row["work_id"],
        "performance_id": row["perf_id"],
        "title":          row["title"],
        "artist":         row["artist"],
        "youtube_id":     ytid,
        "sample_rate":    sr,
        "num_samples":    n_frames,
        "channels":       channels,
        "duration":       round(n_frames / sr, 3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Download SHS-100K and generate MARBLE JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--data-dir", default="data/SHS100K",
                    help="Root data directory (default: data/SHS100K).")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"],
                    help="Which splits to download (default: test).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel yt-dlp workers (default: 4).")
    ap.add_argument("--skip-audio", action="store_true",
                    help="Rebuild JSONL only; do not download anything.")
    ap.add_argument("--browser", default=None, metavar="BROWSER",
                    help="Browser to read cookies from (firefox/chrome/edge/safari).")
    ap.add_argument("--cookies-file", default=None, metavar="FILE",
                    help="Path to a Netscape-format cookie file.")
    ap.add_argument("--max-entries", type=int, default=None,
                    help="Process at most N entries per split (smoke-test).")
    args = ap.parse_args()

    # ── Verify ffprobe is reachable before doing any work ─────────────────────
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        log.error(
            "ffprobe not found on PATH. Install ffmpeg first:\n"
            "  Windows : winget install Gyan.FFmpeg   (restart terminal after)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )
        sys.exit(1)
    log.info(f"ffprobe : {ffprobe_path}")

    # ── Cookie args ───────────────────────────────────────────────────────────
    cookie_args: list[str] = []
    if args.cookies_file and args.browser:
        log.error("Specify only one of --cookies-file or --browser, not both.")
        sys.exit(1)
    if args.cookies_file:
        cfile = Path(args.cookies_file)
        if not cfile.exists():
            log.error(f"--cookies-file not found: {cfile}")
            sys.exit(1)
        cookie_args = ["--cookies", str(cfile)]
        log.info(f"Auth   : cookies file → {cfile}")
    elif args.browser:
        cookie_args = ["--cookies-from-browser", args.browser]
        log.info(f"Auth   : cookies from {args.browser}")
    else:
        log.info("Auth   : none  (add --browser firefox if YouTube requires sign-in)")

    # ── Per-split work ────────────────────────────────────────────────────────
    data_dir  = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        log.info(f"\n{'='*60}\nSplit : {split}\n{'='*60}")

        rows = _fetch_csv(split)
        if args.max_entries:
            rows = rows[: args.max_entries]

        records: list[dict] = []
        n_failed = 0
        n_done = 0
        total = len(rows)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process, row, audio_dir, args.skip_audio, cookie_args): row
                for row in rows
            }
            for fut in as_completed(futs):
                n_done += 1
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                else:
                    n_failed += 1
                if n_done % 100 == 0 or n_done == total:
                    pct = 100 * n_done // total
                    log.info(
                        f"  [{pct:3d}%] {n_done:>5}/{total}"
                        f"  ok={len(records):>5}  failed={n_failed}"
                    )

        # Sort by work_id then performance_id for deterministic JSONL order
        records.sort(key=lambda r: (r["work_id"], r["performance_id"]))

        out = data_dir / f"SHS100K.{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_works = len({r["work_id"] for r in records})
        log.info(f"\n  ✓  {len(records):>6,} tracks  →  {out}")
        log.info(f"  ✗  {n_failed:>6,} unavailable / failed")
        if n_works:
            log.info(f"  ⊕  {n_works:>6,} works  (avg {len(records)/n_works:.1f} versions)")

    log.info("\nAll splits done.")
    log.info("Next: python scripts/run_all_sweeps.py")


if __name__ == "__main__":
    main()
