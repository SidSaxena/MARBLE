#!/usr/bin/env python3
"""
scripts/download_hooktheory.py
──────────────────────────────
Download the HookTheory dataset from the SheetSage open-source release
(no HuggingFace account needed) and generate MARBLE JSONL files for:

  • HookTheoryKey       (24-class key classification, maj/min only)
  • HookTheoryStructure (7-class section-type classification)

Data source
-----------
  JSON metadata:  https://github.com/chrisdonahue/sheetsage-data
  Audio:          YouTube (downloaded via yt-dlp, extracted with ffmpeg)

Prerequisites
-------------
  # yt-dlp is installed automatically via `uv sync`.
  # ffmpeg must be on PATH (system package, not managed by uv):
  #   macOS:   brew install ffmpeg
  #   Linux:   sudo apt install ffmpeg
  #   Windows: winget install ffmpeg
  #            (or download from https://ffmpeg.org/download.html and add to PATH)

Usage
-----
  # Full download + JSONL generation
  python scripts/download_hooktheory.py

  # Only regenerate JSONL from already-downloaded clips (skip yt-dlp)
  python scripts/download_hooktheory.py --skip-audio

  # Limit parallel YouTube workers (default: 4)
  python scripts/download_hooktheory.py --workers 8

  # Custom data directory
  python scripts/download_hooktheory.py --data-dir /mnt/data/HookTheory

  # Only specific tasks
  python scripts/download_hooktheory.py --tasks key structure

Output
------
  data/HookTheory/
    hooktheory_clips/          ← extracted MP3 segments  ({uid}.mp3)
    audio/                     ← full downloaded YouTube audio ({ytid}.mp3)
    HookTheoryKey.train.jsonl
    HookTheoryKey.val.jsonl
    HookTheoryKey.test.jsonl
    HookTheoryStructure.train.jsonl
    HookTheoryStructure.val.jsonl
    HookTheoryStructure.test.jsonl
"""

import argparse
import gzip
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Dataset URLs (SheetSage public release) ──────────────────────────────────
SHEETSAGE_BASE = (
    "https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory"
)
HOOKTHEORY_JSON_URL = f"{SHEETSAGE_BASE}/Hooktheory.json.gz"


# ─── Label mappings (must match marble/tasks/HookTheory*/datamodule.py) ───────

TONIC_TO_NAME = {          # pitch class → flat-notation note name
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

KEY_LABEL2IDX = {          # only major + minor (maj_minor_only=True)
    "C major": 0,  "Db major": 1,  "D major": 2,  "Eb major": 3,
    "E major": 4,  "F major": 5,   "Gb major": 6,  "G major": 7,
    "Ab major": 8, "A major": 9,  "Bb major": 10, "B major": 11,
    "C minor": 12, "Db minor": 13, "D minor": 14,  "Eb minor": 15,
    "E minor": 16, "F minor": 17,  "Gb minor": 18, "G minor": 19,
    "Ab minor": 20,"A minor": 21, "Bb minor": 22, "B minor": 23,
}

STRUCTURE_LABEL2IDX = {
    "intro": 0, "verse": 1, "pre-chorus": 2, "chorus": 3,
    "bridge": 4, "outro": 5, "instrumental": 6, "solo": 6,
    "pre-chorus_chorus": 3, "verse_pre-chorus": 1,
    "intro_verse": 1, "intro_chorus": 3,
}
# Canonical section names we accept from the URL fragment
VALID_SECTIONS = {
    "intro", "verse", "pre-chorus", "chorus", "bridge",
    "outro", "instrumental", "solo",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _download_url(url: str, dest: Path) -> None:
    log.info(f"Downloading {url} …")
    urllib.request.urlretrieve(url, dest)


def _load_hooktheory_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _beat_to_time(beat: float, beats: list, times: list) -> float:
    """Linear interpolation with linear extrapolation (matches SheetSage align.py)."""
    b = np.asarray(beats, dtype=float)
    t = np.asarray(times, dtype=float)
    if beat <= b[0]:
        if len(b) >= 2:
            slope = (t[1] - t[0]) / (b[1] - b[0]) if b[1] != b[0] else 0.0
            return float(t[0] + slope * (beat - b[0]))
        return float(t[0])
    if beat >= b[-1]:
        if len(b) >= 2:
            slope = (t[-1] - t[-2]) / (b[-1] - b[-2]) if b[-1] != b[-2] else 0.0
            return float(t[-1] + slope * (beat - b[-1]))
        return float(t[-1])
    return float(np.interp(beat, b, t))


def _key_label(entry: dict) -> Optional[str]:
    """Return 'D major' / 'Eb minor' etc. from annotation, or None if unsupported."""
    keys = entry.get("annotations", {}).get("keys", [])
    if not keys:
        return None
    k = keys[0]
    tonic_name = TONIC_TO_NAME.get(k.get("tonic"), None)
    scale = k.get("scale", "")
    if tonic_name is None or scale not in ("major", "minor"):
        return None
    label = f"{tonic_name} {scale}"
    return label if label in KEY_LABEL2IDX else None


def _structure_label(entry: dict) -> Optional[str]:
    """
    Parse section type from the TheoryTab URL fragment.
    e.g. '…/billie-jean#chorus-2' → 'chorus'
    Returns None if no recognisable section type found.
    """
    urls = entry.get("hooktheory", {}).get("urls", [])
    for url in urls:
        fragment = url.split("#", 1)[-1].lower().strip() if "#" in url else ""
        # strip trailing digit(s) and hyphens: "chorus-2" → "chorus"
        base = re.sub(r"[-_]\d+$", "", fragment)
        if base in VALID_SECTIONS:
            return base
        # also try the raw fragment without stripping
        if fragment in VALID_SECTIONS:
            return fragment
    return None


def _get_alignment(entry: dict) -> Optional[tuple]:
    """Return (beats, times) from refined alignment, falling back to user."""
    align = entry.get("alignment", {})
    for key in ("refined", "user"):
        a = align.get(key, {})
        beats = a.get("beats", [])
        times = a.get("times", [])
        if beats and times and len(beats) == len(times):
            return beats, times
    return None


def _audio_info(path: Path) -> Optional[tuple]:
    """Return (sample_rate, num_samples, channels) via torchaudio."""
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.sample_rate, info.num_frames, info.num_channels
    except Exception as e:
        log.warning(f"torchaudio.info failed for {path.name}: {e}")
        return None


# ─── Audio download / extraction ──────────────────────────────────────────────

_AUDIO_EXTS = {".mp3", ".m4a", ".webm", ".ogg", ".opus", ".flac", ".wav"}


def _find_existing_audio(ytid: str, audio_dir: Path) -> Optional[Path]:
    """Return an already-downloaded audio file for this ytid, or None."""
    for f in audio_dir.glob(f"{ytid}.*"):
        if f.suffix.lower() in _AUDIO_EXTS and f.stat().st_size > 10_000:
            return f
    return None


def _download_youtube_audio(
    ytid: str,
    audio_dir: Path,
    cookies_file: Optional[Path] = None,
) -> Optional[Path]:
    """Download full YouTube audio in native format (m4a/webm).

    Format selector: ``bestaudio*``
      • Matches audio-only streams first (preferred: opus, aac)
      • Falls back to any stream with audio (muxed video+audio)
      Unlike ``bestaudio/best``, this never fails on modern YouTube because
      ``best`` requires a pre-merged stream that YouTube rarely serves.

    The native container is kept here; _extract_segment() uses ffmpeg to
    cut and re-encode the clip to MP3, so the container format is irrelevant.

    cookies_file: Netscape cookies file exported once at startup; avoids
                  concurrent SQLite DB reads on Windows (Chromium DB lock).
    """
    existing = _find_existing_audio(ytid, audio_dir)
    if existing:
        return existing

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings",
        "--no-playlist",
        "-f", "bestaudio*",          # best audio stream; never fails on modern YouTube
        "--no-cache-dir",
        "-o", str(audio_dir / f"{ytid}.%(ext)s"),
    ]
    if cookies_file and cookies_file.exists():
        cmd += ["--cookies", str(cookies_file)]
    cmd.append(f"https://www.youtube.com/watch?v={ytid}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log.warning(f"yt-dlp failed for {ytid} (rc={result.returncode}): "
                        f"{(result.stderr or result.stdout or '(no output)').strip()[:300]}")
            return None
        return _find_existing_audio(ytid, audio_dir)
    except subprocess.TimeoutExpired:
        log.warning(f"yt-dlp timeout for {ytid}")
        return None
    except Exception as e:
        log.error(f"yt-dlp error for {ytid}: {e}")
        return None


def _extract_segment(
    audio_path: Path,
    clip_path: Path,
    start: float,
    end: float,
) -> bool:
    """
    Extract [start, end] seconds from audio_path → clip_path via ffmpeg.
    Returns True on success.
    """
    if clip_path.exists() and clip_path.stat().st_size > 1000:
        return True   # already extracted

    duration = end - start
    if duration <= 0:
        log.warning(f"Non-positive segment duration ({start:.3f}→{end:.3f}) for {clip_path.name}")
        return False

    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.6f}",
        "-t",  f"{duration:.6f}",
        "-i",  str(audio_path),
        "-ar", "44100",   # standardise to 44100 Hz
        "-ac", "2",       # stereo
        "-y",             # overwrite
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.warning(f"ffmpeg failed for {clip_path.name}: {result.stderr[:200]}")
            return False
        return clip_path.exists() and clip_path.stat().st_size > 1000
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning(f"ffmpeg error for {clip_path.name}: {e}")
        return False


# ─── Per-entry processing ──────────────────────────────────────────────────────

def process_entry(
    uid: str,
    entry: dict,
    audio_dir: Path,
    clips_dir: Path,
    skip_audio: bool,
    tasks: set[str],
    cookies_file: Optional[Path] = None,
) -> Optional[dict]:
    """
    Download audio + extract segment for one HookTheory entry.
    Returns a record dict with fields needed by all requested tasks, or None.
    """
    # ── Basic checks ──────────────────────────────────────────────────────────
    tags = set(entry.get("tags", []))
    if "AUDIO_AVAILABLE" not in tags:
        return None

    ytid = entry.get("youtube", {}).get("id")
    if not ytid:
        return None

    align = _get_alignment(entry)
    if align is None:
        return None
    beats, times = align

    num_beats = entry.get("annotations", {}).get("num_beats")
    if not num_beats:
        return None

    segment_start = _beat_to_time(0, beats, times)
    segment_end   = _beat_to_time(num_beats, beats, times)
    if segment_end - segment_start < 1.0:
        return None   # degenerate segment

    # ── Key label ─────────────────────────────────────────────────────────────
    key_label = _key_label(entry) if "key" in tasks else None

    # ── Structure label ───────────────────────────────────────────────────────
    struct_label = _structure_label(entry) if "structure" in tasks else None

    # If neither task can use this entry, skip
    if "key" in tasks and key_label is None and "structure" not in tasks:
        return None
    if "structure" in tasks and struct_label is None and "key" not in tasks:
        return None

    # ── Audio paths ───────────────────────────────────────────────────────────
    # Clips are always stored as mp3 (ffmpeg re-encodes during extraction).
    # The source full-song file may be m4a or webm — we find it by glob.
    clip_path = clips_dir / f"{uid}.mp3"

    if not skip_audio:
        # Download full YouTube audio if needed
        audio_path = _download_youtube_audio(ytid, audio_dir, cookies_file=cookies_file)
        if audio_path is None:
            log.warning(f"  Skip {uid}: YouTube download failed ({ytid})")
            return None
        # Extract segment (ffmpeg converts to mp3 regardless of source format)
        if not _extract_segment(audio_path, clip_path, segment_start, segment_end):
            log.warning(f"  Skip {uid}: segment extraction failed")
            return None
    else:
        if not clip_path.exists():
            return None   # audio not present, skip silently

    # ── Audio metadata ────────────────────────────────────────────────────────
    info = _audio_info(clip_path)
    if info is None:
        return None
    sr, n_samples, channels = info

    return {
        "uid":            uid,
        "ytid":           ytid,
        "split":          entry.get("split", "TRAIN").lower(),   # train/valid/test
        "audio_path":     str(clip_path),
        "ori_audio_path": str(audio_path),
        "segment_start":  segment_start,
        "segment_end":    segment_end,
        "key_label":      key_label,
        "struct_label":   struct_label,
        "sample_rate":    sr,
        "num_samples":    n_samples,
        "channels":       channels,
        "duration":       round(n_samples / sr, 6) if sr > 0 else 0.0,
    }


# ─── JSONL writing ─────────────────────────────────────────────────────────────

def write_hooktheory_key_jsonl(records: list[dict], data_dir: Path) -> dict[str, int]:
    """Write HookTheoryKey.{train,val,test}.jsonl. Returns counts per split."""
    split_map = {"train": [], "valid": [], "test": []}
    for r in records:
        if r.get("key_label") is None:
            continue
        row = {
            "audio_path": r["audio_path"],
            "ori_uid":    r["ytid"],
            "label":      r["key_label"],
            "duration":   r["duration"],
            "sample_rate": r["sample_rate"],
            "num_samples": r["num_samples"],
            "channels":    r["channels"],
        }
        split_map[r["split"]].append(row)

    # "valid" → "val" for MARBLE naming convention
    counts = {}
    for split_raw, rows in split_map.items():
        marble_split = "val" if split_raw == "valid" else split_raw
        out = data_dir / f"HookTheoryKey.{marble_split}.jsonl"
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts[marble_split] = len(rows)
        log.info(f"  HookTheoryKey.{marble_split}.jsonl: {len(rows)} entries")
    return counts


def write_hooktheory_structure_jsonl(records: list[dict], data_dir: Path) -> dict[str, int]:
    """Write HookTheoryStructure.{train,val,test}.jsonl."""
    split_map = {"train": [], "valid": [], "test": []}
    for r in records:
        if r.get("struct_label") is None:
            continue
        row = {
            "audio_path":     r["audio_path"],
            "ori_audio_path": r["ori_audio_path"],
            "ori_uid":        r["ytid"],
            "label":          [r["struct_label"]],   # datamodule expects a list
            "duration":       r["duration"],
            "segment_start":  r["segment_start"],
            "segment_end":    r["segment_end"],
            "sample_rate":    r["sample_rate"],
            "num_samples":    r["num_samples"],
            "channels":       r["channels"],
        }
        split_map[r["split"]].append(row)

    counts = {}
    for split_raw, rows in split_map.items():
        marble_split = "val" if split_raw == "valid" else split_raw
        out = data_dir / f"HookTheoryStructure.{marble_split}.jsonl"
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        counts[marble_split] = len(rows)
        log.info(f"  HookTheoryStructure.{marble_split}.jsonl: {len(rows)} entries")
    return counts


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download HookTheory data (SheetSage release) and generate MARBLE JSONLs."
    )
    parser.add_argument(
        "--data-dir", default="data/HookTheory",
        help="Root directory for HookTheory data (default: data/HookTheory).",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["key", "structure"],
        choices=["key", "structure"],
        help="Which MARBLE tasks to generate JSONL for (default: key structure).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel YouTube download workers (default: 4). Higher = faster but riskier.",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Skip yt-dlp downloads and only (re-)generate JSONL from existing clips.",
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
        help="Process at most N entries (for testing).",
    )
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    clips_dir = data_dir / "hooktheory_clips"
    for d in (data_dir, audio_dir, clips_dir):
        d.mkdir(parents=True, exist_ok=True)

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
        log.info(f"Exporting cookies from {args.browser} → {cookies_file} …")
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "--cookies-from-browser", args.browser,
             "--cookies", str(cookies_file),
             "--skip-download", "https://www.youtube.com/"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not cookies_file.exists():
            log.warning(
                f"Cookie export failed. Ensure {args.browser} is installed and "
                f"you are signed in to YouTube. On Windows: close the browser first.\n"
                f"  {(result.stderr or result.stdout or '').strip()[:300]}"
            )
            log.warning("Continuing without authentication — bot-detection errors are likely.")
            cookies_file = None
        else:
            log.info(f"Cookies exported ({cookies_file.stat().st_size:,} bytes).")

    tasks = set(args.tasks)

    # ── 1. Download Hooktheory.json.gz ────────────────────────────────────────
    json_gz = data_dir / "Hooktheory.json.gz"
    if not json_gz.exists():
        _download_url(HOOKTHEORY_JSON_URL, json_gz)
    else:
        log.info(f"Hooktheory.json.gz already present ({json_gz.stat().st_size/1e6:.1f} MB)")

    log.info("Parsing Hooktheory.json.gz …")
    hooktheory = _load_hooktheory_json(json_gz)
    entries = list(hooktheory.items())
    if args.max_entries:
        entries = entries[:args.max_entries]
    log.info(f"Total entries: {len(entries)}")

    # ── 2. Filter to entries that are usable ──────────────────────────────────
    # Count by tag availability
    available = sum(1 for _, e in entries if "AUDIO_AVAILABLE" in e.get("tags", []))
    log.info(f"  AUDIO_AVAILABLE: {available}/{len(entries)}")

    # ── 3. Process entries (download + extract) ───────────────────────────────
    log.info(f"\nProcessing with {args.workers} worker(s) "
             f"({'JSONL-only' if args.skip_audio else 'download+extract'}) …")

    records = []
    failed  = 0

    if args.workers <= 1 or args.skip_audio:
        for uid, entry in entries:
            r = process_entry(uid, entry, audio_dir, clips_dir, args.skip_audio,
                              tasks, cookies_file)
            if r:
                records.append(r)
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_entry, uid, entry, audio_dir, clips_dir,
                    args.skip_audio, tasks, cookies_file
                ): uid
                for uid, entry in entries
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                r = future.result()
                if r:
                    records.append(r)
                else:
                    failed += 1
                if done % 100 == 0:
                    log.info(f"  {done}/{len(entries)} processed ({len(records)} ok, {failed} failed)")

    log.info(f"\nProcessed: {len(records)} ok, {failed} failed/skipped")

    # ── 4. Write JSONL files ──────────────────────────────────────────────────
    log.info("\nWriting JSONL files …")
    if "key" in tasks:
        counts = write_hooktheory_key_jsonl(records, data_dir)
        log.info(f"  Key splits: {counts}")

    if "structure" in tasks:
        counts = write_hooktheory_structure_jsonl(records, data_dir)
        log.info(f"  Structure splits: {counts}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    log.info(f"""
Done.  Files written to {data_dir.resolve()}/

Next steps:
  python scripts/run_sweep_local.py \\
      --base-config configs/probe.OMARRQ-multifeature25hz.HookTheoryKey.yaml \\
      --num-layers 24 --model-tag OMARRQ-multifeature25hz --task-tag HookTheoryKey

Note: YouTube availability varies — some songs may be geo-blocked or removed.
      The script can be re-run with --skip-audio to regenerate JSONL
      from any clips already on disk.
""")


if __name__ == "__main__":
    main()
