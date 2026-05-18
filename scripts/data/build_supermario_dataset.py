#!/usr/bin/env python3
"""
scripts/data/build_supermario_dataset.py
────────────────────────────────────────
Build the SuperMario Structure dataset for MARBLE.

Source: https://github.com/ShxLuo-Saxon/supermario-structure-annotation
  554 Super Mario video game music pieces, each annotated with
  function-level structural segments (intro / loop / transition /
  bridge / outro / stinger). Annotations are BAR-RANGE-based, not
  time-based — we use pretty_midi to convert bar numbers to seconds
  via the source MIDI's tempo + time-signature events.

Pipeline
--------
  1. Clone the upstream annotation repo (annotations + pieces.csv +
     pairs.csv). Lightweight, ~5 MB.
  2. Download per-piece source MIDIs from NinSheetMusic
     (url_mid in pieces.csv) — used purely as the bar→time clock.
  3. For each piece:
     a. Locate the user-supplied audio file in --audio-dir (matched
        by piece_id stem with multiple extension fallbacks).
     b. Parse annotations/<piece_id>.json. Use the "Function" entries
        (coarse 6-class labels). Skip the "Section" entries (those
        are for the v2 section-similarity task).
     c. Use pretty_midi.get_downbeats() to map BarRange[start, end]
        to (start_sec, end_sec) in the source MIDI's tempo.
     d. Slice user audio via ffmpeg → 24 kHz mono FLAC per segment.
  4. Split by piece_id 70/15/15 with seed 1234 (matches the upstream
     paper's split convention).
  5. Emit data/SuperMarioStructure/SuperMarioStructure.{train,val,test}.jsonl.

Critical assumption: user audio is tempo-aligned with the MIDI score.
The bar→time mapping comes from the MIDI; if the user's recording is
performed at a different tempo (e.g., live human performance), the
segment boundaries will drift. For MIDI-rendered audio (the expected
v1 case), this is exact.

Label inventory (6 classes — game-music native)
-----------------------------------------------
  In → intro       (opening; not part of the main loop)
  Lp → loop        (main repeating section, the bulk of most VGM)
  Tr → transition  (connecting passage)
  Br → bridge      (contrasting middle section)
  Ou → outro       (closing)
  St → stinger     (short punctuation cue)

Prerequisites
-------------
  ffmpeg   — for segment slicing  (also needed by ffprobe)
  git      — for cloning the upstream annotation repo
  pretty_midi (already a project dep) — bar→time via tempo events

Usage
-----
  # Pilot (5 pieces, ~2 min) — recommended first run
  uv run python scripts/data/build_supermario_dataset.py \\
      --audio-dir /path/to/your/audio --max-pieces 5

  # Full build
  uv run python scripts/data/build_supermario_dataset.py \\
      --audio-dir /path/to/your/audio

  # Rebuild JSONL from already-sliced segments (no new slicing)
  uv run python scripts/data/build_supermario_dataset.py \\
      --audio-dir /path/to/your/audio --skip-slice

Disk budget
-----------
  ~5 MB   — annotations + metadata (cloned repo)
  ~5 MB   — source MIDIs (auto-downloaded)
  ~0.4 GB — per-segment FLACs (~3500 segments × ~100 KB)
  User audio is read-only; we don't copy or modify it.
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
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the ffprobe + SIGINT helpers from SHS100K (already battle-tested).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shs100k import _ffprobe_info, _install_sigint_handler, _shutdown  # noqa: E402

log = logging.getLogger(__name__)

# ── Label inventory ──────────────────────────────────────────────────────────

# Raw 2-letter Function code in the JSON → canonical lowercase label.
RAW_TO_CANONICAL: dict[str, str] = {
    "In": "intro",
    "Lp": "loop",
    "Tr": "transition",
    "Br": "bridge",
    "Ou": "outro",
    "St": "stinger",
}

# Alphabetical canonical order (matches the datamodule's LABEL2IDX).
CANONICAL_LABELS: list[str] = [
    "bridge",
    "intro",
    "loop",
    "outro",
    "stinger",
    "transition",
]
assert len(CANONICAL_LABELS) == 6
assert set(CANONICAL_LABELS) == set(RAW_TO_CANONICAL.values())

# Audio file extensions to probe when looking up user audio per piece_id.
_AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus")

_UPSTREAM_REPO = "https://github.com/ShxLuo-Saxon/supermario-structure-annotation.git"


# ── Upstream repo fetch ──────────────────────────────────────────────────────


def _clone_or_update_upstream(dest: Path) -> Path:
    """Clone the supermario-structure-annotation repo or pull if present."""
    repo_dir = dest / "supermario-structure-annotation"
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
        log.info(f"  cloning {_UPSTREAM_REPO} → {repo_dir}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", _UPSTREAM_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error(
                "git clone failed: %s. Install git and verify network access. "
                "Alternatively, clone manually and pass --upstream-dir.",
                e,
            )
            sys.exit(1)
    return repo_dir


# ── Metadata parsing ─────────────────────────────────────────────────────────


def _parse_pieces_csv(path: Path) -> dict[str, dict]:
    """Parse pieces.csv → {piece_id: row_dict}. piece_id is the 5-digit string."""
    if not path.exists():
        log.error(f"pieces.csv not found at {path}")
        sys.exit(1)
    out: dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("piece_id") or "").strip()
            if not pid:
                continue
            out[pid] = {
                "title": (row.get("title") or "").strip(),
                "ninsheetmusic_id": (row.get("ninsheetmusic_id") or "").strip(),
                "url_mid": (row.get("url_mid") or "").strip(),
            }
    log.info(f"  parsed {len(out):,} pieces from pieces.csv")
    return out


def _parse_pairs_csv_for_splits(path: Path) -> dict[str, str]:
    """Parse pairs.csv to extract per-piece splits.

    pairs.csv uses NUMERIC piece_id (1, 2, 3, ...); pieces.csv uses
    zero-padded 5-digit ("00001", "00002"). We normalise to the 5-digit
    form to match annotations/<piece_id>.json file names.

    Returns {piece_id_padded: "train"|"val"|"test"}.
    """
    if not path.exists():
        log.warning(f"pairs.csv not found at {path} — will assign all splits ourselves.")
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid_num = (row.get("piece_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not pid_num or split not in ("train", "val", "test"):
                continue
            # Normalise NUMERIC piece_id → zero-padded 5-digit
            try:
                pid_padded = f"{int(pid_num):05d}"
            except ValueError:
                continue
            out[pid_padded] = split
    log.info(f"  loaded upstream splits for {len(out):,} pieces from pairs.csv")
    return out


# ── Annotation parsing ───────────────────────────────────────────────────────


def _parse_annotation_json(path: Path) -> list[tuple[int, int, str]]:
    """Parse one SuperMario annotation file.

    Returns list of (bar_start, bar_end, canonical_label) tuples from the
    "Function" array (coarse, 6-class). The "Section" array is skipped —
    it's reserved for the v2 section-similarity task.

    Bar numbers are 1-indexed in the annotations.
    """
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"  {path.name}: could not load JSON: {e}")
        return []

    out: list[tuple[int, int, str]] = []
    for entry in data.get("Function", []):
        raw_label = entry.get("Function")
        bar_range = entry.get("BarRange")
        if raw_label not in RAW_TO_CANONICAL:
            log.warning(f"  {path.name}: unknown Function code {raw_label!r}; skipping entry")
            continue
        if not isinstance(bar_range, list) or len(bar_range) != 2:
            log.warning(f"  {path.name}: bad BarRange {bar_range!r}; skipping entry")
            continue
        try:
            start_bar = int(bar_range[0])
            end_bar = int(bar_range[1])
        except (TypeError, ValueError):
            log.warning(f"  {path.name}: non-integer BarRange {bar_range!r}")
            continue
        if end_bar <= start_bar:
            log.debug(f"  {path.name}: zero/negative bar range {bar_range!r}")
            continue
        out.append((start_bar, end_bar, RAW_TO_CANONICAL[raw_label]))
    return out


# ── Bar → time mapping via pretty_midi ────────────────────────────────────────


def _midi_bar_times(midi_path: Path) -> list[float] | None:
    """Return the time (seconds) of the START of each bar in the MIDI.

    Uses pretty_midi.get_downbeats() which handles tempo + time-signature
    changes correctly. Bar 1 in the annotation = first downbeat = index 0.
    Bar N's start time = result[N-1]; Bar N's end time = result[N] if it
    exists, else the MIDI's end time.

    Returns None if the MIDI is unparseable.
    """
    try:
        import pretty_midi
    except ImportError:
        log.error("pretty_midi not installed — `uv sync` to install.")
        sys.exit(1)
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        log.warning(f"  pretty_midi could not load {midi_path.name}: {e}")
        return None
    try:
        downbeats = pm.get_downbeats()
    except Exception as e:
        log.warning(f"  pretty_midi.get_downbeats() failed for {midi_path.name}: {e}")
        return None
    if len(downbeats) == 0:
        log.warning(f"  {midi_path.name}: pretty_midi returned 0 downbeats")
        return None
    # Append the MIDI's end time so we can compute the END of the last bar.
    end_time = pm.get_end_time()
    return list(downbeats) + [end_time]


def _bar_range_to_time(
    bar_start: int, bar_end: int, bar_times: list[float]
) -> tuple[float, float] | None:
    """Convert 1-indexed (bar_start, bar_end) → (start_sec, end_sec).

    Returns None if the bar numbers are out of range for this MIDI.
    Annotation BarRange is inclusive on start, exclusive on end.

    Convention: bar N's start time = bar_times[N-1]. End time of a
    BarRange [a, b] = bar_times[b-1] (start of bar b, since b is exclusive
    in the annotator's convention... or is it inclusive? We treat it as
    exclusive, meaning the segment covers bars [a, b-1], which matches
    common music-analysis conventions for span notation).
    """
    # Defensive: bar 0 is not used in the annotations (1-indexed)
    if bar_start < 1 or bar_end < 1:
        return None
    # bar_times has N+1 entries (N bars + end-of-file sentinel)
    if bar_start - 1 >= len(bar_times) or bar_end - 1 >= len(bar_times):
        return None
    start_sec = float(bar_times[bar_start - 1])
    end_sec = float(bar_times[bar_end - 1])
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


# ── MIDI download ─────────────────────────────────────────────────────────────


def _download_midi(url: str, dst: Path, timeout: int = 30) -> bool:
    """Download a MIDI file via urllib. Returns True on success.

    Idempotent: skips if dst already exists with size > 100 bytes.
    """
    if dst.exists() and dst.stat().st_size > 100:
        return True
    if not url:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "marble-build/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 100:
            log.warning(f"  MIDI download too small ({len(data)} bytes): {url}")
            return False
        dst.write_bytes(data)
        return True
    except Exception as e:
        log.warning(f"  MIDI download failed for {url}: {e}")
        return False


# ── User audio lookup ─────────────────────────────────────────────────────────


def _find_user_audio(piece_id: str, audio_dir: Path) -> Path | None:
    """Look up user-supplied audio for a piece.

    Tries multiple file extensions. Returns the first match, or None.
    """
    for ext in _AUDIO_EXTS:
        p = audio_dir / f"{piece_id}{ext}"
        if p.exists() and p.stat().st_size > 1024:
            return p
    return None


# ── ffmpeg segment slicing ────────────────────────────────────────────────────


def _slice_segment(
    src_audio: Path,
    dst_audio: Path,
    start_sec: float,
    end_sec: float,
    target_sr: int = 24000,
) -> bool:
    """Extract one segment from src_audio → dst_audio (mono FLAC @ target_sr).

    Same pattern as build_hxmsa_dataset.py. Idempotent. Input-side seek
    for speed (precision sufficient for 15-s encoder windows downstream).
    """
    if dst_audio.exists() and dst_audio.stat().st_size > 1024:
        return True
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return False
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
        "1",
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


# ── Split assignment ──────────────────────────────────────────────────────────


def _assign_splits(
    piece_ids: list[str],
    upstream_splits: dict[str, str],
    seed: int = 1234,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, str]:
    """Assign splits per piece.

    For pieces in upstream_splits (those covered by pairs.csv), honour
    the upstream assignment. For the remainder, deterministic random
    70/15/15 with the given seed — matches the upstream paper's ratio.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    out: dict[str, str] = {}
    unassigned: list[str] = []
    for pid in sorted(piece_ids):
        if pid in upstream_splits:
            out[pid] = upstream_splits[pid]
        else:
            unassigned.append(pid)
    rng = random.Random(seed)
    rng.shuffle(unassigned)
    n = len(unassigned)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    for i, pid in enumerate(unassigned):
        if i < n_train:
            out[pid] = "train"
        elif i < n_train + n_val:
            out[pid] = "val"
        else:
            out[pid] = "test"
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


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
        "--audio-dir",
        required=True,
        help="Directory containing user audio files. Matched by piece_id stem "
        "(e.g. 00001.flac, 00001.wav, 00001.mp3, ...).",
    )
    ap.add_argument(
        "--data-dir",
        default="data/SuperMarioStructure",
        help="Output root dir (JSONL + segments live under here). Default: %(default)s",
    )
    ap.add_argument(
        "--segments-dir",
        default=None,
        help="Where per-segment FLACs go. Default: <data-dir>/segments",
    )
    ap.add_argument(
        "--midi-dir",
        default=None,
        help="Where source MIDIs are cached. Default: <data-dir>/midi",
    )
    ap.add_argument(
        "--upstream-dir",
        default=None,
        help="Path to an already-cloned supermario-structure-annotation repo. "
        "Default: clone fresh into <data-dir>/_upstream/.",
    )
    ap.add_argument(
        "--max-pieces",
        type=int,
        default=None,
        help="Pilot mode: process at most this many pieces total.",
    )
    ap.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Sample rate of the sliced segment FLACs (default: 24000, matches encoders).",
    )
    ap.add_argument(
        "--skip-slice",
        action="store_true",
        help="Skip ffmpeg segment slicing; assume segments are already present.",
    )
    ap.add_argument(
        "--skip-midi-download",
        action="store_true",
        help="Skip MIDI downloads; assume source MIDIs are already cached.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for assigning splits to pieces not in pairs.csv (default: 1234).",
    )
    ap.add_argument(
        "--min-segment-sec",
        type=float,
        default=2.0,
        help="Drop segments shorter than this (default: 2.0 s).",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    segments_dir = Path(args.segments_dir) if args.segments_dir else data_dir / "segments"
    midi_dir = Path(args.midi_dir) if args.midi_dir else data_dir / "midi"
    audio_dir = Path(args.audio_dir)
    for d in (data_dir, segments_dir, midi_dir):
        d.mkdir(parents=True, exist_ok=True)
    if not audio_dir.exists():
        log.error(f"--audio-dir does not exist: {audio_dir}")
        sys.exit(1)

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
    if args.upstream_dir:
        repo_dir = Path(args.upstream_dir)
        if not (repo_dir / "annotations").exists():
            log.error(
                f"--upstream-dir {repo_dir} doesn't look like a "
                "supermario-structure-annotation checkout"
            )
            sys.exit(1)
    else:
        log.info("Fetching upstream annotations …")
        repo_dir = _clone_or_update_upstream(data_dir / "_upstream")

    annotations_root = repo_dir / "annotations"
    pieces_meta = _parse_pieces_csv(repo_dir / "metadata" / "pieces.csv")
    upstream_splits = _parse_pairs_csv_for_splits(repo_dir / "metadata" / "pairs.csv")

    # ── Build candidate piece list (intersection of pieces.csv + annotations + audio) ──
    all_annotation_files = sorted(annotations_root.glob("*.json"))
    candidates: list[str] = []
    no_audio: list[str] = []
    for af in all_annotation_files:
        pid = af.stem
        if pid not in pieces_meta:
            log.debug(f"  {pid}: missing from pieces.csv; skipping")
            continue
        if _find_user_audio(pid, audio_dir) is None:
            no_audio.append(pid)
            continue
        candidates.append(pid)

    if args.max_pieces:
        candidates = sorted(candidates)[: args.max_pieces]

    # ── Render-plan preamble (fail-loud BEFORE the slow loop) ──────────────
    n_midi_present = sum(
        1
        for pid in candidates
        if (midi_dir / f"{pid}.mid").exists() and (midi_dir / f"{pid}.mid").stat().st_size > 100
    )
    n_to_download_midi = 0 if args.skip_midi_download else len(candidates) - n_midi_present
    log.info("")
    log.info("─" * 60)
    log.info("Build plan")
    log.info("─" * 60)
    log.info(f"  audio-dir         : {audio_dir}")
    log.info(f"  data-dir          : {data_dir}")
    log.info(f"  segments-dir      : {segments_dir}")
    log.info(f"  midi-dir          : {midi_dir}")
    log.info(
        f"  candidate pieces  : {len(candidates):,} "
        f"({len(all_annotation_files):,} annotations available, "
        f"{len(no_audio):,} have no audio in --audio-dir)"
    )
    if no_audio and not args.max_pieces:
        sample = ", ".join(no_audio[:5])
        log.info(
            f"  missing audio for : {len(no_audio)} pieces — first 5: {sample}{'...' if len(no_audio) > 5 else ''}"
        )
    log.info(f"  MIDI already cached  : {n_midi_present:,}")
    log.info(
        f"  MIDI to download now : {n_to_download_midi:,}"
        f"{'  (SKIPPED via --skip-midi-download)' if args.skip_midi_download else ''}"
    )
    log.info(
        f"  segment slicing   : {'SKIPPED via --skip-slice' if args.skip_slice else 'enabled'}"
    )
    log.info(f"  target sr         : {args.target_sr} Hz (mono FLAC)")
    log.info(
        f"  upstream splits   : honoured for {len(upstream_splits):,} pieces; "
        f"seed-{args.seed} 70/15/15 for the rest"
    )
    log.info("─" * 60)

    if not candidates:
        log.error("No candidate pieces — check --audio-dir naming convention.")
        sys.exit(2)

    _install_sigint_handler()

    # ── Step 1: Download source MIDIs ──────────────────────────────────────
    if not args.skip_midi_download:
        log.info("Downloading source MIDIs from NinSheetMusic …")
        for i, pid in enumerate(candidates, 1):
            if _shutdown.is_set():
                break
            url = pieces_meta[pid].get("url_mid", "")
            dst = midi_dir / f"{pid}.mid"
            _download_midi(url, dst)
            if i % 50 == 0 or i == len(candidates):
                log.info(f"  [{i:>4}/{len(candidates)}]")

    # ── Step 2: For each piece, parse → bar-time-map → slice ───────────────
    records: list[dict] = []
    canonical_label_counter: Counter[str] = Counter()
    dropped_short = 0
    dropped_oor = 0
    dropped_no_midi = 0
    failed_slice = 0

    for pid in candidates:
        audio_path = _find_user_audio(pid, audio_dir)
        if audio_path is None:
            continue
        midi_path = midi_dir / f"{pid}.mid"
        if not midi_path.exists():
            log.debug(f"  {pid}: no MIDI; skipping (need bar→time mapping)")
            dropped_no_midi += 1
            continue
        bar_times = _midi_bar_times(midi_path)
        if bar_times is None:
            dropped_no_midi += 1
            continue

        ann_path = annotations_root / f"{pid}.json"
        segments = _parse_annotation_json(ann_path)
        if not segments:
            log.debug(f"  {pid}: no parseable Function segments; skipping")
            continue

        out_piece_dir = segments_dir / pid
        meta = pieces_meta[pid]

        for seg_idx, (bar_start, bar_end, label) in enumerate(segments):
            time_range = _bar_range_to_time(bar_start, bar_end, bar_times)
            if time_range is None:
                log.debug(
                    f"  {pid}:{seg_idx} BarRange [{bar_start}, {bar_end}] out of "
                    f"range for MIDI (max bar = {len(bar_times) - 1})"
                )
                dropped_oor += 1
                continue
            start_sec, end_sec = time_range
            if end_sec - start_sec < args.min_segment_sec:
                dropped_short += 1
                continue

            seg_path = out_piece_dir / f"{seg_idx:03d}_{label}.flac"
            if not args.skip_slice:
                ok = _slice_segment(audio_path, seg_path, start_sec, end_sec, args.target_sr)
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

            canonical_label_counter[label] += 1
            records.append(
                {
                    "audio_path": str(seg_path.as_posix()),
                    "ori_uid": f"{pid}_{seg_idx:03d}",  # per-segment uid
                    "work_id": pid,  # track-level grouping
                    "label": label,
                    "seg_idx": seg_idx,
                    "bar_start": bar_start,
                    "bar_end": bar_end,
                    "seg_start": round(start_sec, 3),
                    "seg_end": round(end_sec, 3),
                    "duration": round(seg_nsamp / seg_sr, 3),
                    "sample_rate": seg_sr,
                    "num_samples": seg_nsamp,
                    "channels": seg_chan,
                    "bit_depth": 16,
                    "title": meta.get("title", ""),
                    "ninsheetmusic_id": meta.get("ninsheetmusic_id", ""),
                }
            )

    log.info("")
    log.info(
        f"  built {len(records):,} segment records "
        f"(dropped: {dropped_short} short, {dropped_oor} bar-out-of-range, "
        f"{dropped_no_midi} missing/unparseable MIDI, {failed_slice} ffmpeg failures)"
    )
    if not records:
        log.error("No segment records — aborting.")
        sys.exit(2)

    # ── Step 3: Split assignment + JSONL emit ──────────────────────────────
    piece_ids_in_use = sorted({r["work_id"] for r in records})
    split_assign = _assign_splits(piece_ids_in_use, upstream_splits, seed=args.seed)
    split_counts = Counter(split_assign.values())
    log.info(
        f"  train/val/test pieces: "
        f"{split_counts.get('train', 0)} / "
        f"{split_counts.get('val', 0)} / "
        f"{split_counts.get('test', 0)}"
    )

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_split[split_assign[r["work_id"]]].append(r)

    for split, recs in by_split.items():
        recs.sort(key=lambda r: (r["work_id"], r["seg_idx"]))
        out_path = data_dir / f"SuperMarioStructure.{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"  wrote {len(recs):>5} records → {out_path}")

    # ── Final stats ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(
        f" SuperMarioStructure build complete — {len(records):,} segments "
        f"from {len(piece_ids_in_use)} pieces"
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


if __name__ == "__main__":
    main()
