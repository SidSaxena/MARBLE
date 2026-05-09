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
  Full songs are downloaded from YouTube in their native best-audio format
  (usually .m4a or .webm/opus) via yt-dlp — no ffmpeg conversion needed at
  download time.  torchaudio reads both formats at load time (uses its own
  ffmpeg backend).  The datamodule handles splitting into 30-second clips.

Prerequisites
-------------
  yt-dlp is installed automatically by `uv sync`.
  No ffmpeg required for downloading (native format is kept as-is).

Output
------
  data/SHS100K/audio/            full-song audio files (m4a or webm), keyed by youtube_id
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

  # Smoke-test with a small subset
  uv run python scripts/download_shs100k.py --max-entries 20
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

# Audio file extensions that torchaudio / ffmpeg can read.
# yt-dlp downloads in the native YouTube format (m4a or webm/opus) — we
# accept any of these rather than forcing an mp3 conversion via ffmpeg.
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
    """Return an already-downloaded audio file for this ytid, or None."""
    for f in audio_dir.glob(f"{ytid}.*"):
        if f.suffix.lower() in _AUDIO_EXTS and f.stat().st_size > 10_000:
            return f
    return None


def _download_audio(ytid: str, audio_dir: Path, skip: bool) -> Optional[Path]:
    """
    Download best-quality audio-only stream from YouTube.

    Keeps the native container (m4a or webm/opus) — no ffmpeg conversion.
    torchaudio.load() reads both formats via its built-in ffmpeg backend.
    Returns the downloaded Path, or None on failure / unavailability.
    """
    existing = _find_existing(ytid, audio_dir)
    if existing:
        return existing
    if skip:
        return None

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "yt_dlp",
                "--quiet", "--no-warnings",
                "-f", "bestaudio/best",        # best audio-only; native container
                "-o", str(audio_dir / f"{ytid}.%(ext)s"),
                f"https://www.youtube.com/watch?v={ytid}",
            ],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            log.debug(f"yt-dlp failed for {ytid}: {result.stderr[:200]}")
            return None
        return _find_existing(ytid, audio_dir)   # find whatever ext was used
    except subprocess.TimeoutExpired:
        log.warning(f"yt-dlp timeout for {ytid}")
        return None
    except Exception as e:
        log.warning(f"yt-dlp error for {ytid}: {e}")
        return None


def _process_row(row: dict, audio_dir: Path, skip_audio: bool) -> Optional[dict]:
    """Download audio for one row and return a JSONL record, or None on failure."""
    ytid = row["youtube_id"]
    audio_path = _download_audio(ytid, audio_dir, skip=skip_audio)
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
        "--max-entries", type=int, default=None,
        help="Cap entries per split for smoke-testing (e.g. --max-entries 50).",
    )
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

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
                pool.submit(_process_row, row, audio_dir, args.skip_audio): row
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
