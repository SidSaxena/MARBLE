#!/usr/bin/env python3
"""
scripts/data/build_hxmsa_dataset.py
───────────────────────────────────
Build the HXMSA (Harmonix Set Music Structure Analysis) dataset for MARBLE.

Source: https://github.com/urinieto/harmonixset  (Nieto et al. ISMIR 2019)
  912 Western pop tracks, hand-annotated by professional musicians with
  functional segment labels (intro / verse / chorus / bridge / outro / ...).
  Audio is NOT distributed by the upstream repo — only YouTube URLs.

Pipeline
--------
  1. Fetch metadata.csv, youtube_urls.csv, and dataset/segments/*.txt from
     the upstream repo (lightweight git clone, ~10 MB).
  2. Download audio for each track via yt-dlp (reuses helpers from
     download_shs100k.py, which has been tested at scale).
  3. Parse each track's segment annotation file. Map raw labels → 13-class
     canonical inventory. Drop the "end" sentinel.
  4. Slice audio per segment via ffmpeg → 24 kHz mono FLAC.
  5. Build per-segment JSONL records.
  6. Split 80/10/10 by track_id (seed 1234) so segments from one track
     never leak across splits.
  7. Emit data/HXMSA/HXMSA.{train,val,test}.jsonl.

Label inventory (13 classes)
---------------------------
After dropping the "end" terminator and merging "instrumental" → "inst"
(both per the paper's own §3.3 note about repeated labels), the native
Harmonix vocabulary collapses to 13 distinct functional labels. See the
RAW_TO_CANONICAL map below for the exact rules.

Prerequisites
-------------
  yt-dlp   — must be up to date  (python -m yt_dlp -U)
  ffmpeg   — for segment slicing  (also needed by ffprobe)
  git      — for cloning the upstream annotation repo

  Cookies (one-time, recommended for fewer bot-checks):
    python scripts/data/export_youtube_cookies.py --browser firefox

  (NOTE: do NOT use the older `yt-dlp --cookies-from-browser firefox
  --cookies cookies.txt --skip-download URL` pattern — yt-dlp's
  --cookies FILE flag is bidirectional and aborts on a stale/malformed
  file with "'cookies.txt' does not look like a Netscape format
  cookies file". The helper above deletes any stale file first +
  uses yt-dlp's Python API to write a guaranteed-valid Netscape file.)

Usage
-----
  # Pilot (5 tracks, ~5 min) — recommended first run
  uv run python scripts/data/build_hxmsa_dataset.py --max-tracks 5

  # Full run (~3–6 h, rate-limited by yt-dlp)
  uv run python scripts/data/build_hxmsa_dataset.py --cookies-file cookies.txt

  # Rebuild JSONL from already-downloaded audio + segments (no new downloads)
  uv run python scripts/data/build_hxmsa_dataset.py --skip-download --skip-slice

  # Override the harmonixset repo location (e.g. if you already cloned it)
  uv run python scripts/data/build_hxmsa_dataset.py \\
      --harmonixset-dir /path/to/harmonixset

Disk budget
-----------
  ~4.6 GB  — full-track FLACs (912 × ~5 MB)
  ~0.9 GB  — per-segment FLACs (~9000 × ~100 KB)
  ~10 MB   — annotations + metadata (cloned repo)
  Total:   ~5.5 GB
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the battle-tested yt-dlp + ffprobe + SIGINT helpers from SHS100K.
# Underscore-prefixed but stable; importing keeps both scripts in sync if
# yt-dlp's player-client surface changes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shs100k import (  # noqa: E402
    _download as _ytdlp_download,
)
from download_shs100k import (
    _ffprobe_info,
    _find_audio,
    _install_sigint_handler,
    _shutdown,
)

log = logging.getLogger(__name__)

# ── Label inventory ────────────────────────────────────────────────────────────

# Native Harmonix labels (15) → canonical 13-class inventory.
# Two adjustments vs raw native labels (rationale documented in the plan file):
#   1. "end" is a terminator timestamp, not a section label — dropped.
#   2. "inst" and "instrumental" are the same word per the paper's §3.3 —
#      merged into a single "inst" class.
# Defensive aliases (hyphenated / underscore variants) cover annotation typos
# that may exist in the wild even if the paper's canonical form is unhyphenated.
RAW_TO_CANONICAL: dict[str, str | None] = {
    # ── Identity mappings (12 keep-as-is + "inst") ─────────────────────────
    "intro": "intro",
    "verse": "verse",
    "prechorus": "prechorus",
    "chorus": "chorus",
    "postchorus": "postchorus",
    "bridge": "bridge",
    "outro": "outro",
    "inst": "inst",
    "transition": "transition",
    "break": "break",
    "solo": "solo",
    "silence": "silence",
    "other": "other",
    # ── Merge: instrumental → inst ─────────────────────────────────────────
    "instrumental": "inst",
    # ── Drop: terminator sentinel ──────────────────────────────────────────
    "end": None,
    # ── Defensive aliases for common annotation typos ──────────────────────
    "pre-chorus": "prechorus",
    "post-chorus": "postchorus",
    "pre_chorus": "prechorus",
    "post_chorus": "postchorus",
}

# Stable, alphabetical class order — matches the order used in the datamodule's
# LABEL2IDX. Keeping it explicit here means class index assignments are
# deterministic across runs.
CANONICAL_LABELS: list[str] = [
    "break",
    "bridge",
    "chorus",
    "inst",
    "intro",
    "other",
    "outro",
    "postchorus",
    "prechorus",
    "silence",
    "solo",
    "transition",
    "verse",
]
assert len(CANONICAL_LABELS) == 13


# ── Upstream repo fetch ────────────────────────────────────────────────────────

_HARMONIXSET_REPO = "https://github.com/urinieto/harmonixset.git"


def _clone_or_update_harmonixset(dest: Path) -> Path:
    """Clone the harmonixset annotation repo (~10 MB) or pull if present.

    Returns the path to the cloned dataset directory.
    """
    repo_dir = dest / "harmonixset"
    if repo_dir.exists() and (repo_dir / ".git").exists():
        log.info(f"  found existing clone: {repo_dir} (pulling)")
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "pull", "--ff-only", "--quiet"],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            log.warning("  git pull timed out — using existing checkout as-is")
    else:
        log.info(f"  cloning {_HARMONIXSET_REPO} → {repo_dir}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", _HARMONIXSET_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error(
                "git clone failed: %s. Install git and verify network access. "
                "Alternatively, clone manually and pass --harmonixset-dir.",
                e,
            )
            sys.exit(1)
    return repo_dir


# ── Metadata + URL parsing ─────────────────────────────────────────────────────


def _ytid_from_url(url: str) -> str | None:
    """Extract a YouTube video ID from a full URL (defensive against trailing
    notes like ' (speed)' after the URL that the upstream CSV sometimes has)."""
    if not url:
        return None
    # Split on any whitespace; first token should be the URL itself.
    tok = url.strip().split()[0]
    # Handle both http and https, www. and bare youtube.com
    for marker in ("watch?v=", "youtu.be/"):
        if marker in tok:
            tail = tok.split(marker, 1)[1]
            return tail.split("&")[0].split("?")[0].split("/")[0][:32] or None
    return None


def _parse_urls_csv(path: Path) -> dict[str, str]:
    """Parse youtube_urls.csv into {file_id: ytid}. Skips rows missing URLs."""
    if not path.exists():
        log.error(f"youtube_urls.csv not found at {path}")
        sys.exit(1)
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("File") or "").strip()
            url = (row.get("URL") or "").strip()
            if not fid:
                continue
            ytid = _ytid_from_url(url)
            if not ytid:
                log.debug(f"  no parseable ytid for {fid}: {url!r}")
                continue
            out[fid] = ytid
    log.info(f"  parsed {len(out):,} (file_id, ytid) pairs from youtube_urls.csv")
    return out


def _parse_metadata_csv(path: Path) -> dict[str, dict]:
    """Parse metadata.csv into {file_id: metadata_dict}."""
    if not path.exists():
        log.error(f"metadata.csv not found at {path}")
        sys.exit(1)
    out: dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = (row.get("File") or "").strip()
            if not fid:
                continue
            out[fid] = {
                "title": row.get("Title", "").strip(),
                "artist": row.get("Artist", "").strip(),
                "genre": row.get("Genre", "").strip(),
                # Duration field is the authors' annotation duration; we'll
                # also probe the actual downloaded audio.
                "annotated_duration": float(row.get("Duration") or 0.0),
            }
    log.info(f"  parsed {len(out):,} tracks from metadata.csv")
    return out


# ── Segment annotation parsing ─────────────────────────────────────────────────


def _parse_segments_file(path: Path) -> list[tuple[float, float, str]]:
    """Parse one Harmonix segment file.

    Format (space-separated, one segment per line):
        <timestamp_sec> <label>
        ...
        <end_timestamp_sec> end

    Returns list of (start_sec, end_sec, canonical_label) tuples.
    Drops the "end" terminator. Filters out unknown labels (logs warning).
    """
    if not path.exists():
        return []
    raw_rows: list[tuple[float, str]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                log.debug(f"  {path.name}:{line_no} malformed: {line!r}")
                continue
            try:
                ts = float(parts[0])
            except ValueError:
                log.debug(f"  {path.name}:{line_no} bad timestamp: {parts[0]!r}")
                continue
            label = " ".join(parts[1:]).strip().lower()
            raw_rows.append((ts, label))

    # Convert (start_only) rows into (start, end, label) by pairing consecutive rows.
    segments: list[tuple[float, float, str]] = []
    for i in range(len(raw_rows) - 1):
        start, label_raw = raw_rows[i]
        end, _ = raw_rows[i + 1]
        if label_raw not in RAW_TO_CANONICAL:
            log.warning(f"  {path.name}: unknown label {label_raw!r} at t={start:.2f}; skipping")
            continue
        canonical = RAW_TO_CANONICAL[label_raw]
        if canonical is None:
            # "end" sentinel as a segment label — shouldn't really happen mid-file
            continue
        if end <= start:
            log.debug(f"  {path.name}: zero-length segment at t={start:.2f}; skipping")
            continue
        segments.append((start, end, canonical))
    return segments


# ── ffmpeg segment slicing ─────────────────────────────────────────────────────


def _slice_segment(
    src_audio: Path,
    dst_audio: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int = 24000,
) -> bool:
    """Extract one segment from src_audio → dst_audio (mono FLAC @ target_sr).

    Returns True on success, False on failure. Idempotent — skips if dst exists.
    """
    if dst_audio.exists() and dst_audio.stat().st_size > 1024:
        return True
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return False
    # -ss before -i is fast (input-side seek with keyframe accuracy).  For
    # short segments and lossy source formats this is accurate enough — we're
    # downsampling to 24 kHz mono anyway, sub-frame precision doesn't matter
    # for the encoder which sees 15-s clips.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src_audio),
        "-ar",
        str(target_sr),
        "-ac",
        "1",  # mono
        "-c:a",
        "flac",
        str(dst_audio),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning(f"  ffmpeg slice failed: {dst_audio.name}: {e}")
        return False
    if r.returncode != 0:
        log.warning(
            f"  ffmpeg slice failed: {dst_audio.name}: rc={r.returncode} {r.stderr.strip()[:120]}"
        )
        return False
    return dst_audio.exists() and dst_audio.stat().st_size > 1024


# ── Train/val/test split ───────────────────────────────────────────────────────


def _assign_splits(
    track_ids: list[str], seed: int = 1234, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, str]:
    """Deterministic 80/10/10 split by track_id.

    Splits at the TRACK level (not segment level) so segments from a single
    track never appear in different splits.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    ids = sorted(track_ids)  # determinism: sort before shuffling
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    out = {}
    for i, tid in enumerate(ids):
        if i < n_train:
            out[tid] = "train"
        elif i < n_train + n_val:
            out[tid] = "val"
        else:
            out[tid] = "test"
    return out


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
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
        "--data-dir",
        default="data/HXMSA",
        help="Output root dir (JSONL + segments live under here). Default: %(default)s",
    )
    ap.add_argument(
        "--audio-dir",
        default=None,
        help="Where full-track audio goes. Default: <data-dir>/full_tracks",
    )
    ap.add_argument(
        "--segments-dir",
        default=None,
        help="Where per-segment FLACs go. Default: <data-dir>/segments",
    )
    ap.add_argument(
        "--harmonixset-dir",
        default=None,
        help="Path to an already-cloned harmonixset repo. "
        "Default: clone fresh into <data-dir>/_upstream/.",
    )
    ap.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Pilot mode: process at most this many tracks total.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="yt-dlp download workers (default: 2). Higher = faster but more 429s.",
    )
    ap.add_argument(
        "--cookies-file",
        default=None,
        help="Path to a cookies.txt for yt-dlp (recommended — see header docstring).",
    )
    ap.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser to extract cookies from (firefox/chrome/edge/etc). "
        "Alternative to --cookies-file.",
    )
    ap.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Sample rate of the sliced segment FLACs (default: 24000, matches encoders).",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip yt-dlp downloads; assume audio is already present.",
    )
    ap.add_argument(
        "--skip-slice",
        action="store_true",
        help="Skip ffmpeg segment slicing; assume segments are already present.",
    )
    ap.add_argument(
        "--cleanup-full-tracks",
        action="store_true",
        help="Delete full-track audio after slicing (saves ~5 GB).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for the 80/10/10 train/val/test split (default: 1234).",
    )
    ap.add_argument(
        "--min-segment-sec",
        type=float,
        default=2.0,
        help="Drop segments shorter than this (default: 2.0 s).",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    audio_dir = Path(args.audio_dir) if args.audio_dir else data_dir / "full_tracks"
    segments_dir = Path(args.segments_dir) if args.segments_dir else data_dir / "segments"
    for d in (data_dir, audio_dir, segments_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Validate dependencies ──────────────────────────────────────────────
    for tool, hint in [
        ("ffmpeg", "needed for segment slicing"),
        ("ffprobe", "comes with ffmpeg"),
        ("git", "needed to clone the annotation repo"),
    ]:
        if shutil.which(tool) is None:
            log.error(f"{tool} not found on PATH — {hint}. See script header.")
            sys.exit(1)

    # ── Fetch upstream repo ─────────────────────────────────────────────────
    if args.harmonixset_dir:
        repo_dir = Path(args.harmonixset_dir)
        if not (repo_dir / "dataset").exists():
            log.error(f"--harmonixset-dir {repo_dir} doesn't look like a harmonixset checkout")
            sys.exit(1)
    else:
        log.info("Fetching upstream annotations …")
        repo_dir = _clone_or_update_harmonixset(data_dir / "_upstream")

    dataset_root = repo_dir / "dataset"
    metadata = _parse_metadata_csv(dataset_root / "metadata.csv")
    ytids = _parse_urls_csv(dataset_root / "youtube_urls.csv")
    segments_root = dataset_root / "segments"
    if not segments_root.exists():
        log.error(f"segments dir missing: {segments_root}")
        sys.exit(1)

    # ── Build canonical track list (intersection of metadata + URLs + annotations) ──
    all_segment_files = sorted(segments_root.glob("*.txt"))
    candidates: list[str] = []
    for sf in all_segment_files:
        fid = sf.stem
        if fid in metadata and fid in ytids:
            candidates.append(fid)
        else:
            log.debug(f"  {fid}: missing from metadata or youtube_urls; skipping")

    if args.max_tracks:
        candidates = sorted(candidates)[: args.max_tracks]

    # ── Render-plan preamble (fail-loud BEFORE the slow loop) ──────────────
    n_audio_present = sum(1 for fid in candidates if _find_audio(ytids[fid], audio_dir))
    n_to_download = 0 if args.skip_download else len(candidates) - n_audio_present
    log.info("")
    log.info("─" * 60)
    log.info("Build plan")
    log.info("─" * 60)
    log.info(f"  data-dir         : {data_dir}")
    log.info(f"  audio-dir        : {audio_dir}")
    log.info(f"  segments-dir     : {segments_dir}")
    log.info(
        f"  candidate tracks : {len(candidates):,} "
        f"({len(all_segment_files):,} have annotations, "
        f"{len(ytids):,} have URLs, {len(metadata):,} have metadata)"
    )
    log.info(f"  audio present    : {n_audio_present:,}")
    log.info(
        f"  audio to fetch   : {n_to_download:,}{'  (SKIPPED via --skip-download)' if args.skip_download else ''}"
    )
    log.info(f"  workers          : {args.workers}")
    log.info(
        f"  cookies          : "
        f"{'file=' + args.cookies_file if args.cookies_file else ('browser=' + args.cookies_from_browser if args.cookies_from_browser else 'none')}"
    )
    log.info(f"  segment slicing  : {'SKIPPED via --skip-slice' if args.skip_slice else 'enabled'}")
    log.info(f"  target sr        : {args.target_sr} Hz (mono FLAC)")
    log.info("─" * 60)

    _install_sigint_handler()

    cookie_args: list[str] = []
    if args.cookies_file:
        cookie_args = ["--cookies", args.cookies_file]
    elif args.cookies_from_browser:
        cookie_args = ["--cookies-from-browser", args.cookies_from_browser]

    # ── Step 1: Download audio ─────────────────────────────────────────────
    track_audio: dict[str, Path | None] = {}
    if args.skip_download:
        for fid in candidates:
            track_audio[fid] = _find_audio(ytids[fid], audio_dir)
        n_ok = sum(1 for p in track_audio.values() if p)
        log.info(f"  --skip-download: {n_ok}/{len(candidates)} tracks have audio")
    else:
        log.info(f"Downloading audio for {len(candidates):,} tracks (workers={args.workers}) …")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(_ytdlp_download, ytids[fid], audio_dir, cookie_args, False): fid
                for fid in candidates
            }
            for i, fut in enumerate(as_completed(futs), 1):
                if _shutdown.is_set():
                    break
                fid = futs[fut]
                try:
                    track_audio[fid] = fut.result()
                except Exception as e:
                    log.warning(f"  [{fid}] download error: {e}")
                    track_audio[fid] = None
                if i % 25 == 0 or i == len(candidates):
                    n_ok = sum(1 for p in track_audio.values() if p)
                    pct = 100 * i // max(1, len(candidates))
                    log.info(f"  [{pct:3d}%] {i:>4}/{len(candidates)}  ok={n_ok:>4}")

    n_audio_ok = sum(1 for p in track_audio.values() if p)
    log.info(f"  audio available: {n_audio_ok}/{len(candidates)} tracks")
    if n_audio_ok == 0:
        log.error("No audio successfully downloaded — aborting.")
        sys.exit(2)

    # ── Step 2: Slice segments + build per-segment records ─────────────────
    records: list[dict] = []
    raw_label_counter: Counter[str] = Counter()
    canonical_label_counter: Counter[str] = Counter()
    dropped_short = 0
    failed_slice = 0

    for fid in candidates:
        audio_path = track_audio.get(fid)
        if audio_path is None:
            continue
        sr, n_samples, channels = _ffprobe_info(audio_path)
        if sr == 0:
            log.debug(f"  {fid}: ffprobe failed on {audio_path}; skipping")
            continue
        track_duration = n_samples / sr

        seg_file = segments_root / f"{fid}.txt"
        segments = _parse_segments_file(seg_file)
        if not segments:
            log.debug(f"  {fid}: no parseable segments; skipping")
            continue

        out_track_dir = segments_dir / fid

        for seg_idx, (start, end, label_canonical) in enumerate(segments):
            # Filter: end must be within actual track duration
            if start >= track_duration:
                log.debug(
                    f"  {fid}:{seg_idx} start ({start:.1f}s) beyond duration ({track_duration:.1f}s); skipping"
                )
                continue
            end_clamped = min(end, track_duration)
            seg_duration = end_clamped - start
            if seg_duration < args.min_segment_sec:
                dropped_short += 1
                continue

            raw_label_counter[label_canonical] += 1  # we already mapped to canonical
            canonical_label_counter[label_canonical] += 1

            seg_path = out_track_dir / f"{seg_idx:03d}_{label_canonical}.flac"
            if not args.skip_slice:
                ok = _slice_segment(audio_path, seg_path, start, end_clamped, args.target_sr)
                if not ok:
                    failed_slice += 1
                    continue
            elif not seg_path.exists():
                continue

            seg_info = _ffprobe_info(seg_path)
            if seg_info[0] == 0:
                failed_slice += 1
                continue
            seg_sr, seg_nsamp, seg_chan = seg_info

            md = metadata.get(fid, {})
            records.append(
                {
                    "audio_path": str(seg_path.as_posix()),
                    "ori_uid": f"{fid}_{seg_idx:03d}",  # per-segment uid for probe aggregation
                    "work_id": fid,  # track-level grouping
                    "label": label_canonical,
                    "seg_idx": seg_idx,
                    "seg_start": round(start, 3),
                    "seg_end": round(end_clamped, 3),
                    "duration": round(seg_nsamp / seg_sr, 3),
                    "sample_rate": seg_sr,
                    "num_samples": seg_nsamp,
                    "channels": seg_chan,
                    "bit_depth": 16,  # FLAC default we used
                    "title": md.get("title", ""),
                    "artist": md.get("artist", ""),
                    "genre": md.get("genre", ""),
                }
            )

    log.info("")
    log.info(
        f"  built {len(records):,} segment records "
        f"(dropped {dropped_short} short, {failed_slice} ffmpeg failures)"
    )
    if not records:
        log.error("No segment records — aborting.")
        sys.exit(2)

    # ── Step 3: Split by track_id ──────────────────────────────────────────
    track_ids_in_use = sorted({r["work_id"] for r in records})
    split_assign = _assign_splits(track_ids_in_use, seed=args.seed)
    split_counts = Counter(split_assign.values())
    log.info(
        f"  train/val/test tracks: "
        f"{split_counts.get('train', 0)} / "
        f"{split_counts.get('val', 0)} / "
        f"{split_counts.get('test', 0)}"
    )

    # Distribute records into split JSONLs
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[split_assign[r["work_id"]]].append(r)

    for split, recs in by_split.items():
        # Sort for deterministic JSONL output
        recs.sort(key=lambda r: (r["work_id"], r["seg_idx"]))
        out_path = data_dir / f"HXMSA.{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"  wrote {len(recs):>5} records → {out_path}")

    # ── Final stats ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(
        f" HXMSA build complete — {len(records):,} segments from {len(track_ids_in_use)} tracks"
    )
    log.info(f"   train: {len(by_split.get('train', []))}")
    log.info(f"   val:   {len(by_split.get('val', []))}")
    log.info(f"   test:  {len(by_split.get('test', []))}")
    log.info("")
    log.info(" Canonical label distribution:")
    for lbl in CANONICAL_LABELS:
        c = canonical_label_counter.get(lbl, 0)
        log.info(f"   {lbl:>12s}  {c:>5,}")
    log.info("=" * 60)

    # ── Optional cleanup ───────────────────────────────────────────────────
    if args.cleanup_full_tracks and not args.skip_download:
        n_deleted = 0
        for p in audio_dir.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    n_deleted += 1
                except OSError:
                    pass
        log.info(f"  --cleanup-full-tracks: removed {n_deleted} files from {audio_dir}")


if __name__ == "__main__":
    main()
