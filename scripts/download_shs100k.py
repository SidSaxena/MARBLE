#!/usr/bin/env python3
"""
scripts/download_shs100k.py
───────────────────────────
Download SHS-100K (Second Hand Songs, 2025 edition) and generate MARBLE
JSONL metadata files for cover-song retrieval probing.

Dataset
-------
  Source : https://github.com/second-hand-songs/shs-100k  (branch: 2025)
  Scale  : 10 000 works × ~110 000 performances
    test.csv       500 works,   ~7 100 tracks  ← default (community benchmark)
    validate.csv   500 works,   ~5 000 tracks
    train.csv    9 000 works, ~100 000 tracks  (very large)

  CSV columns (no header): perf_id, work_id, title, artist, youtube_id

Audio
-----
  yt-dlp downloads in the native YouTube container — m4a (AAC) if available,
  webm/opus otherwise.  No conversion is done at download time.
  torchaudio reads both formats at inference time via the system ffmpeg backend.
  The datamodule clips files into 30-second windows on the fly.

  A small fraction of SHS-100K videos are private, geo-blocked, deleted, or
  premium-only.  These produce "Requested format is not available" or
  "Video unavailable" errors and are counted as expected data gaps.

Prerequisites
-------------
  uv sync         — installs yt-dlp and all Python deps
  ffmpeg on PATH  — required so torchaudio can decode webm/m4a files:
    Windows : winget install Gyan.FFmpeg   (restart terminal after)
    macOS   : brew install ffmpeg
    Linux   : sudo apt install ffmpeg

Usage
-----
  # Download test split (~7 100 tracks, the community benchmark)
  uv run python scripts/download_shs100k.py

  # Authenticate to bypass YouTube bot-detection (needed after ~100 downloads)
  # Firefox: DB is not locked, parallel workers are safe
  uv run python scripts/download_shs100k.py --browser firefox --workers 4
  # Chrome/Edge: close the browser first (Chromium locks the cookie DB)
  uv run python scripts/download_shs100k.py --browser edge --workers 4

  # If you need Chrome/Edge open while downloading, export cookies once first:
  #   yt-dlp --cookies-from-browser edge --cookies cookies.txt --skip-download ^
  #           "https://www.youtube.com/watch?v=jNQXAC9IVRw"
  uv run python scripts/download_shs100k.py --cookies-file cookies.txt --workers 4

  # Rebuild JSONL from already-downloaded files (no new downloads)
  uv run python scripts/download_shs100k.py --skip-audio

  # Smoke-test with a small subset
  uv run python scripts/download_shs100k.py --browser firefox --max-entries 20
"""

import argparse
import csv
import io
import json
import logging
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# All extensions yt-dlp might produce.  New downloads are m4a or webm.
_AUDIO_EXTS = {".mp3", ".m4a", ".webm", ".ogg", ".opus", ".flac", ".wav"}

_BASE = "https://raw.githubusercontent.com/second-hand-songs/shs-100k/2025"
CSV_URLS = {
    "train": f"{_BASE}/train.csv",
    "val":   f"{_BASE}/validate.csv",
    "test":  f"{_BASE}/test.csv",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(split: str) -> list[dict]:
    """Fetch and parse one SHS-100K CSV file."""
    url = CSV_URLS[split]
    log.info(f"Fetching {url} …")
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 5:
            continue
        rows.append({
            "perf_id":    int(row[0]),
            "work_id":    int(row[1]),
            "title":      row[2].strip(),
            "artist":     row[3].strip(),
            "youtube_id": row[4].strip(),
        })
    log.info(f"  {len(rows):,} entries loaded from {split}.csv")
    return rows


def _find_existing(ytid: str, audio_dir: Path) -> Optional[Path]:
    """Return any already-downloaded audio file for this YouTube ID, or None."""
    for f in audio_dir.glob(f"{ytid}.*"):
        if f.suffix.lower() in _AUDIO_EXTS and f.stat().st_size > 10_000:
            return f
    return None


def _audio_info(path: Path) -> tuple[int, int, int]:
    """
    Return (sample_rate, num_frames, num_channels).

    Tries torchaudio first.  If that fails (e.g. torchaudio's bundled ffmpeg
    lacks a codec on Windows), falls back to ffprobe from the system ffmpeg.
    Returns (0, 0, 1) only if both methods fail.
    """
    # ── Method 1: torchaudio ─────────────────────────────────────────────────
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.sample_rate, info.num_frames, info.num_channels
    except Exception:
        pass

    # ── Method 2: ffprobe (system ffmpeg) ────────────────────────────────────
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sr = int(stream.get("sample_rate", 0))
                    channels = int(stream.get("channels", 1))
                    # duration may be in the stream or in a tag
                    dur_str = (stream.get("duration")
                               or stream.get("tags", {}).get("DURATION", "0"))
                    try:
                        dur = float(dur_str)
                    except (ValueError, TypeError):
                        dur = 0.0
                    if sr > 0 and dur > 0:
                        return sr, int(sr * dur), channels
    except FileNotFoundError:
        log.warning("ffprobe not found — install system ffmpeg so torchaudio "
                    "can decode webm/m4a files (winget install Gyan.FFmpeg)")
    except Exception as e:
        log.debug(f"ffprobe failed for {path.name}: {e}")

    log.warning(f"Could not read audio info for {path.name} "
                "(torchaudio and ffprobe both failed)")
    return 0, 0, 1


def _download(
    ytid: str,
    audio_dir: Path,
    cookie_args: list[str],
) -> Optional[Path]:
    """
    Download audio for one YouTube video in its native container.

    Format preference: m4a (AAC) > any audio-only stream (webm/opus).
    m4a is preferred because torchaudio's bundled ffmpeg reliably decodes
    AAC on all platforms, including Windows.  webm/opus works too as long
    as system ffmpeg is on PATH (needed for the ffprobe fallback in
    _audio_info and for torchaudio.load at inference time).

    cookie_args: zero or more yt-dlp flags for authentication, e.g.
        ["--cookies-from-browser", "firefox"]  or
        ["--cookies", "/path/to/cookies.txt"]

    Returns the downloaded Path on success, None on failure.
    Failures fall into two categories:
      • Expected data gaps: private / deleted / geo-blocked / premium videos.
        These log at DEBUG — they are not bugs.
      • Unexpected: bot-detection (add --browser), network errors, etc.
        These log at WARNING.
    """
    existing = _find_existing(ytid, audio_dir)
    if existing:
        return existing

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings",
        "--no-playlist",
        # Prefer m4a (AAC).  If unavailable, take whatever audio-only stream
        # YouTube offers (usually webm/opus).  Do NOT fall back to "best"
        # (merged video+audio) — it is slow and the video data is discarded.
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-o", str(audio_dir / f"{ytid}.%(ext)s"),
    ] + cookie_args + [f"https://www.youtube.com/watch?v={ytid}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            return _find_existing(ytid, audio_dir)

        err = (result.stderr or result.stdout or "").strip()
        if "Sign in to confirm" in err or "bot" in err.lower():
            log.warning(
                f"[bot-check] {ytid}: YouTube requires authentication. "
                "Run with --browser firefox (or --browser edge with browser closed)."
            )
        elif any(p in err for p in ("unavailable", "not available",
                                    "private", "removed", "does not exist")):
            log.debug(f"[unavailable] {ytid}: video is private/deleted/geo-blocked")
        else:
            log.warning(f"[yt-dlp rc={result.returncode}] {ytid}: {err[:250]}")
        return None

    except subprocess.TimeoutExpired:
        log.warning(f"[timeout] {ytid}: yt-dlp exceeded 180 s")
        return None
    except Exception as e:
        log.warning(f"[error] {ytid}: {e}")
        return None


def _process_row(
    row: dict,
    audio_dir: Path,
    skip_audio: bool,
    cookie_args: list[str],
) -> Optional[dict]:
    """Download (or locate) audio for one CSV row and build a JSONL record."""
    ytid = row["youtube_id"]

    if skip_audio:
        path = _find_existing(ytid, audio_dir)
    else:
        path = _download(ytid, audio_dir, cookie_args)

    if path is None or not path.exists() or path.stat().st_size < 10_000:
        return None

    sr, n_frames, channels = _audio_info(path)
    if sr == 0 or n_frames == 0:
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

    parser = argparse.ArgumentParser(
        description="Download SHS-100K and generate MARBLE JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data/SHS100K",
        help="Root directory for SHS100K data (default: data/SHS100K).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        choices=["train", "val", "test"],
        help="Splits to download (default: test only).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel yt-dlp workers (default: 4).",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Skip downloads; rebuild JSONL from already-downloaded files.",
    )
    parser.add_argument(
        "--browser", default=None, metavar="BROWSER",
        help=(
            "Read cookies from this browser to bypass bot-detection "
            "(firefox / chrome / edge / safari). "
            "Firefox is safest for parallel workers. "
            "For Chrome/Edge: close the browser first."
        ),
    )
    parser.add_argument(
        "--cookies-file", default=None, metavar="FILE",
        help=(
            "Path to a Netscape-format cookies file. "
            "Use when the browser must stay open during the download. "
            "Export once with: yt-dlp --cookies-from-browser edge "
            "--cookies cookies.txt --skip-download "
            "\"https://www.youtube.com/watch?v=jNQXAC9IVRw\""
        ),
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Process at most N entries per split (for smoke-testing).",
    )
    args = parser.parse_args()

    # ── Build cookie args (passed verbatim to every yt-dlp call) ──────────────
    cookie_args: list[str] = []
    if args.cookies_file:
        cfile = Path(args.cookies_file)
        if not cfile.exists():
            log.error(f"--cookies-file not found: {cfile}")
            sys.exit(1)
        cookie_args = ["--cookies", str(cfile)]
        log.info(f"Auth: cookies file → {cfile}")
    elif args.browser:
        cookie_args = ["--cookies-from-browser", args.browser]
        log.info(f"Auth: cookies from {args.browser}")
    else:
        log.info("Auth: none (if YouTube asks you to sign in, add --browser firefox)")

    # ── Process each split ────────────────────────────────────────────────────
    data_dir  = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        log.info(f"\n{'='*60}\n  Split: {split}\n{'='*60}")

        rows = _load_csv(split)
        if args.max_entries:
            rows = rows[:args.max_entries]

        records: list[dict] = []
        failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process_row, row, audio_dir,
                            args.skip_audio, cookie_args): row
                for row in rows
            }
            n_done = 0
            for fut in as_completed(futs):
                n_done += 1
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                else:
                    failed += 1
                if n_done % 200 == 0:
                    log.info(f"  {n_done:>5}/{len(rows)}  "
                             f"ok={len(records)}  fail={failed}")

        out = data_dir / f"SHS100K.{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        n_works = len({r["work_id"] for r in records})
        log.info(f"\n  ✓ {len(records):>5} tracks  →  {out}")
        log.info(f"  ✗ {failed:>5} failed / unavailable")
        log.info(f"  Works covered : {n_works}")
        if n_works:
            log.info(f"  Avg versions  : {len(records)/n_works:.1f}")

    log.info("\nAll splits done.")
    log.info("Run sweeps with:")
    log.info("  uv run python scripts/run_all_sweeps.py --tasks SHS100K")


if __name__ == "__main__":
    main()
