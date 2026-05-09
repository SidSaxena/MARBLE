#!/usr/bin/env python3
"""
scripts/download_shs100k.py
───────────────────────────
Download SHS-100K (Second Hand Songs, 2025 official edition) and generate
MARBLE JSONL metadata files for cover-song retrieval probing.

Dataset
-------
  Source:  https://github.com/second-hand-songs/shs-100k  (branch: 2025)
  Scale:   10,000 works × ~110,000 performances total
    test.csv          500 works,   5,000 tracks  ← default (community eval benchmark)
    validate.csv      500 works,   5,000 tracks
    train.csv       9,000 works, 100,000 tracks  (very large — download separately)

  CSV format (no header row):
    perf_id, work_id, title, artist, youtube_id

Audio
-----
  Full songs are downloaded from YouTube and extracted to MP3 via yt-dlp +
  system ffmpeg.  MP3 is universally readable by torchaudio on all platforms.
  The datamodule handles splitting into 30-second clips at inference time.

Prerequisites
-------------
  yt-dlp is installed automatically by `uv sync`.
  System ffmpeg must be on PATH (used by yt-dlp for audio extraction):
    Windows: winget install Gyan.FFmpeg   (restart terminal after)
    macOS:   brew install ffmpeg
    Linux:   sudo apt install ffmpeg

Output
------
  data/SHS100K/audio/            full-song MP3 files, keyed by youtube_id
  data/SHS100K/SHS100K.test.jsonl
  data/SHS100K/SHS100K.val.jsonl   (only with --splits val)
  data/SHS100K/SHS100K.train.jsonl (only with --splits train)

Usage
-----
  # Recommended: test split only (~5 K tracks, ~20 GB, the eval benchmark)
  uv run python scripts/download_shs100k.py

  # Add val for completeness
  uv run python scripts/download_shs100k.py --splits test val

  # Rebuild JSONL without re-downloading (audio already on disk)
  uv run python scripts/download_shs100k.py --skip-audio

  # Authenticate via browser cookies (recommended; avoids bot detection).
  # Export cookies once at startup → reused by all parallel workers.
  # Close Edge/Chrome first on Windows (Chromium locks the cookie DB).
  uv run python scripts/download_shs100k.py --browser edge --workers 4

  # Alternative: pre-export cookies yourself and pass the file directly.
  # (Useful if the browser stays open while the download runs.)
  #   yt-dlp --cookies-from-browser edge --cookies cookies.txt --skip-download https://www.youtube.com/
  uv run python scripts/download_shs100k.py --cookies-file cookies.txt --workers 4

  # Smoke-test with a small subset
  uv run python scripts/download_shs100k.py --browser edge --max-entries 20
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

# Audio file extensions accepted as valid downloads.
# New downloads are MP3 (via yt-dlp --extract-audio --audio-format mp3).
# We also accept other formats so that files downloaded before ffmpeg was
# available (e.g. .webm, .m4a) are still recognised and reused.
_AUDIO_EXTS = {".mp3", ".m4a", ".webm", ".ogg", ".opus", ".flac", ".wav"}

# ── CSV sources (branch 2025, not main) ──────────────────────────────────────
_BASE = "https://raw.githubusercontent.com/second-hand-songs/shs-100k/2025"
CSV_URLS = {
    "train": f"{_BASE}/train.csv",
    "val":   f"{_BASE}/validate.csv",
    "test":  f"{_BASE}/test.csv",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_csv(split: str) -> list[dict]:
    """Fetch and parse one SHS-100K CSV file (no header row)."""
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


def _audio_info(path: Path) -> tuple[int, int, int]:
    """Return (sample_rate, num_frames, num_channels) via torchaudio."""
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.sample_rate, info.num_frames, info.num_channels
    except Exception as e:
        log.warning(f"torchaudio.info failed for {path.name}: {e}")
        return 0, 0, 1


def _find_existing(ytid: str, audio_dir: Path) -> Optional[Path]:
    """Return an already-downloaded audio file for this ytid, or None.

    Checks MP3 first (the standard output format), then any other recognised
    audio extension so that files from earlier download attempts are reused.
    """
    mp3 = audio_dir / f"{ytid}.mp3"
    if mp3.exists() and mp3.stat().st_size > 10_000:
        return mp3
    for f in audio_dir.glob(f"{ytid}.*"):
        if f.suffix.lower() in _AUDIO_EXTS and f.stat().st_size > 10_000:
            return f
    return None


def _export_cookies(browser: str, cookies_file: Path) -> bool:
    """Export browser cookies to a Netscape-format file for yt-dlp.

    Done ONCE at startup so all parallel workers share the same file.
    This avoids concurrent reads of the browser's SQLite cookie database,
    which is locked on Windows when a Chromium-based browser (Edge, Chrome)
    is open.  Workers then use --cookies <file> instead of
    --cookies-from-browser <browser>.

    Returns True on success.
    """
    log.info(f"Exporting cookies from {browser} → {cookies_file} …")
    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--quiet", "--no-warnings",
            "--cookies-from-browser", browser,
            "--cookies", str(cookies_file),
            "--skip-download",
            # Use a single known video, not the homepage.
            # The homepage extractor enumerates the feed and can take minutes;
            # any single video URL exports the same domain-scoped cookies.
            "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not cookies_file.exists():
        log.warning(
            f"Cookie export failed (rc={result.returncode}). "
            f"Ensure {browser} is installed and you are signed in to YouTube. "
            f"On Windows: close the browser first if you see a DB-lock error.\n"
            f"  {(result.stderr or result.stdout or '').strip()[:300]}"
        )
        return False
    log.info(f"Cookies exported ({cookies_file.stat().st_size:,} bytes).")
    return True


def _download_audio(
    ytid: str,
    audio_dir: Path,
    skip: bool,
    cookies_file: Optional[Path] = None,
) -> Optional[Path]:
    """Download and extract audio from YouTube as MP3.

    Format selector: ``bestaudio*``
      • Matches audio-only streams first (preferred: opus, aac — no video data)
      • Falls back to any stream that contains audio (e.g. muxed video+audio)
      Unlike ``bestaudio/best``, this NEVER fails on modern YouTube.
      ``best`` (without *) requires a pre-merged stream, which YouTube no
      longer provides for most videos — causing "format not available" errors.

    Audio is re-encoded to MP3 at highest VBR quality via system ffmpeg.
    cookies_file: Netscape cookies file exported at startup; if provided,
                  passed as --cookies to avoid bot-detection failures.
    """
    existing = _find_existing(ytid, audio_dir)
    if existing:
        return existing
    if skip:
        return None

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings",
        "--no-playlist",
        "-f", "bestaudio*",          # best audio stream; prefers audio-only,
                                     # falls back to muxed if needed
        "--extract-audio",           # strip video track if present
        "--audio-format", "mp3",     # re-encode to MP3 (requires ffmpeg)
        "--audio-quality", "0",      # highest VBR quality (~320 kbps)
        "-o", str(audio_dir / f"{ytid}.%(ext)s"),
    ]
    if cookies_file and cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    cmd.append(f"https://www.youtube.com/watch?v={ytid}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            log.warning(
                f"yt-dlp failed for {ytid} (rc={result.returncode}): "
                f"{(result.stderr or result.stdout or '(no output)').strip()[:300]}"
            )
            return None
        return _find_existing(ytid, audio_dir)
    except subprocess.TimeoutExpired:
        log.warning(f"yt-dlp timeout for {ytid}")
        return None
    except Exception as e:
        log.warning(f"yt-dlp error for {ytid}: {e}")
        return None


def _process_row(
    row: dict,
    audio_dir: Path,
    skip_audio: bool,
    cookies_file: Optional[Path] = None,
) -> Optional[dict]:
    """Download audio for one row and return a JSONL record, or None on failure."""
    ytid = row["youtube_id"]
    audio_path = _download_audio(ytid, audio_dir, skip=skip_audio, cookies_file=cookies_file)
    if audio_path is None or not audio_path.exists() or audio_path.stat().st_size < 10_000:
        return None

    sr, n_samples, channels = _audio_info(audio_path)
    if sr == 0 or n_samples == 0:
        return None

    return {
        "audio_path":     str(audio_path),
        "work_id":        row["work_id"],
        "performance_id": row["perf_id"],
        "title":          row["title"],
        "artist":         row["artist"],
        "youtube_id":     ytid,
        "sample_rate":    sr,
        "num_samples":    n_samples,
        "channels":       channels,
        "duration":       round(n_samples / sr, 3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download SHS-100K and generate MARBLE JSONL files."
    )
    parser.add_argument(
        "--data-dir", default="data/SHS100K",
        help="Root directory for SHS100K data (default: data/SHS100K).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        choices=["train", "val", "test"],
        help="Splits to download (default: test only — the community eval benchmark).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel yt-dlp workers (default: 4). Increase carefully — YouTube may throttle.",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Skip yt-dlp downloads; rebuild JSONL from already-downloaded audio.",
    )
    parser.add_argument(
        "--browser", default=None,
        metavar="BROWSER",
        help=(
            "Export cookies from this browser at startup and use them for all "
            "downloads (e.g. --browser edge, --browser chrome, --browser firefox). "
            "You must be signed in to YouTube in the browser. "
            "On Windows: close the browser first to avoid cookie DB-lock errors, "
            "or use --cookies-file with pre-exported cookies instead."
        ),
    )
    parser.add_argument(
        "--cookies-file", default=None,
        metavar="FILE",
        help=(
            "Path to a pre-exported Netscape-format cookies file. "
            "Use instead of --browser when the browser must stay open. "
            "Export once with: "
            "yt-dlp --cookies-from-browser edge --cookies cookies.txt "
            "--skip-download https://www.youtube.com/"
        ),
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Cap entries per split for smoke-testing (e.g. --max-entries 50).",
    )
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve cookies (done once; shared across all parallel workers) ────────
    cookies_file: Optional[Path] = None
    if args.cookies_file:
        cookies_file = Path(args.cookies_file)
        if not cookies_file.exists():
            log.error(f"--cookies-file not found: {cookies_file}")
            sys.exit(1)
        log.info(f"Using cookies file: {cookies_file}")
    elif args.browser:
        cookies_file = data_dir / ".yt-dlp-cookies.txt"
        if not _export_cookies(args.browser, cookies_file):
            log.warning("Continuing without authentication — bot-detection errors are likely.")

    for split in args.splits:
        log.info(f"\n{'='*60}")
        log.info(f"  Split: {split}")
        log.info(f"{'='*60}")

        rows = _load_csv(split)
        if args.max_entries:
            rows = rows[:args.max_entries]

        records: list[dict] = []
        failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_process_row, row, audio_dir, args.skip_audio,
                            cookies_file): row
                for row in rows
            }
            n_done = 0
            for fut in as_completed(futs):
                n_done += 1
                if n_done % 200 == 0:
                    log.info(f"  {n_done:>5}/{len(rows)}  ok={len(records)}  fail={failed}")
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                else:
                    failed += 1

        out = data_dir / f"SHS100K.{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        n_works = len(set(r["work_id"] for r in records))
        log.info(f"\n  ✓  {len(records):>5} tracks → {out}")
        log.info(f"  ✗  {failed:>5} failed (video unavailable / private)")
        log.info(f"  Works represented: {n_works}")
        log.info(f"  Avg versions/work: {len(records)/n_works:.1f}" if n_works else "")

    log.info("\nAll splits done.")
    log.info("Run the sweep with:")
    log.info("  uv run python scripts/run_all_sweeps.py --tasks SHS100K")


if __name__ == "__main__":
    main()
