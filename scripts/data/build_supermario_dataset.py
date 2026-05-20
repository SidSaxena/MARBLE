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
  2. Source per-piece MIDIs. NinSheetMusic blocks scrapers (HTTP 403
     regardless of headers), so the preferred path is
     ``--midi-source-dir <dir>`` pointing at MIDIs you fetched
     manually (or via the `ohsheet` Rust CLI, or any other method
     that respects NSM's terms). The script falls back to attempted
     auto-download from url_mid in pieces.csv, but that will fail
     for NSM in practice.
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
  # Symbolic-only build with user-supplied MIDIs (the common case)
  uv run python scripts/data/build_supermario_dataset.py \\
      --midi-source-dir /path/to/your/midis

  # Pilot first — 5 pieces, ~30 s
  uv run python scripts/data/build_supermario_dataset.py \\
      --midi-source-dir /path/to/your/midis --max-pieces 5

  # Full build with both symbolic + audio
  uv run python scripts/data/build_supermario_dataset.py \\
      --midi-source-dir /path/to/your/midis \\
      --audio-dir /path/to/your/audio

  # Rebuild JSONL from already-sliced segments (no new slicing)
  uv run python scripts/data/build_supermario_dataset.py \\
      --skip-midi-slice --skip-slice --skip-midi-download

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

    Annotation BarRange is **INCLUSIVE on both ends**. Verified against
    upstream supermario-structure-annotation/annotations/00001.json:
    intro [1, 10] is followed by loop [11, 39] — contiguous, so bar 10
    is the last bar of intro and bar 11 is the first bar of loop. If
    bar_end were exclusive, bars 10 and 39 would belong to no segment.

    bar_times has N+1 entries: bar_times[i] = start time of bar (i+1),
    with bar_times[N] = end-of-MIDI sentinel appended in
    _midi_bar_times(). Thus the END of bar b is bar_times[b]
    (= start of bar b+1, or the EOF sentinel when b == N).
    """
    # Defensive: bar 0 is not used in the annotations (1-indexed)
    if bar_start < 1 or bar_end < 1:
        return None
    # bar_times has N+1 entries (N bars + end-of-file sentinel). Index
    # bar_end (= end of bar bar_end) must be valid.
    if bar_start - 1 >= len(bar_times) or bar_end >= len(bar_times):
        return None
    start_sec = float(bar_times[bar_start - 1])
    end_sec = float(bar_times[bar_end])
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec


# ── MIDI download ─────────────────────────────────────────────────────────────


def _download_midi(url: str, dst: Path, timeout: int = 30) -> bool:
    """Download a MIDI file via urllib. Returns True on success.

    Idempotent: skips if dst already exists with size > 100 bytes.

    Note: NinSheetMusic actively blocks all automated access (HTTP 403
    regardless of User-Agent / Referer / cookies — verified against
    multiple browser-mimicking headers). This function is provided as a
    best-effort fallback, but in practice you should populate the MIDI
    cache via ``--midi-source-dir`` instead. See the script header for
    the three documented sourcing options.
    """
    if dst.exists() and dst.stat().st_size > 100:
        return True
    if not url:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Browser-ish headers; doesn't actually help against NSM but kept
        # for any other hosts that might surface in future variants.
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Referer": "https://www.ninsheetmusic.org/",
            },
        )
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


def _import_user_midi(piece_id: str, source_dir: Path, dst: Path) -> bool:
    """Copy a user-supplied MIDI into the cache dir.

    Returns True if a matching file was found and successfully placed
    at ``dst``. Tries common MIDI extensions (.mid, .midi, .smf) in
    two naming conventions:

      1. ``<piece_id>.<ext>`` (legacy plain naming)
      2. ``<piece_id>_<title-slug>.<ext>`` (download_ninsheetmusic.py
         default since the title-naming change — see that script's
         docstring for the slug format)

    The slug fallback handles files saved by the canonical NSM
    downloader without requiring a separate rename step.
    """
    if dst.exists() and dst.stat().st_size > 100:
        return True
    # 1. Plain <piece_id>.<ext>
    for ext in (".mid", ".midi", ".smf"):
        candidate = source_dir / f"{piece_id}{ext}"
        if candidate.exists() and candidate.stat().st_size > 100:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate, dst)
            return True
    # 2. <piece_id>_<slug>.<ext> — sorted() pins to deterministic match
    #    if more than one slug exists for the same piece_id.
    for ext in (".mid", ".midi", ".smf"):
        matches = sorted(source_dir.glob(f"{piece_id}_*{ext}"))
        if matches and matches[0].stat().st_size > 100:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(matches[0], dst)
            return True
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


# ── MIDI segment slicing ──────────────────────────────────────────────────────


def _slice_midi(
    src_path: Path,
    dst_path: Path,
    start_sec: float,
    end_sec: float,
) -> bool:
    """Slice MIDI to keep notes in [start_sec, end_sec] and rebase to t=0.

    Uses pretty_midi: drops tempo / control-change events but preserves
    note pitches, velocities, durations, and program-change info per
    instrument — adequate for CLaMP3 M3 tokenisation, which encodes
    note + program semantics. The new MIDI's initial tempo is taken
    from the source MIDI's local tempo at ``start_sec`` so absolute
    durations are preserved.

    Idempotent: skips if dst exists with size > 100 bytes.
    """
    if dst_path.exists() and dst_path.stat().st_size > 100:
        return True
    if end_sec <= start_sec:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pretty_midi
    except ImportError:
        log.error("pretty_midi not installed — `uv sync` to install.")
        sys.exit(1)
    try:
        pm_src = pretty_midi.PrettyMIDI(str(src_path))
    except Exception as e:
        log.warning(f"  pretty_midi failed to load {src_path.name}: {e}")
        return False

    # Use the source MIDI's local tempo at start_sec for the new MIDI's
    # initial tempo. Falls back to 120 BPM if no tempo events.
    initial_tempo = 120.0
    try:
        import numpy as np

        times, tempos = pm_src.get_tempo_changes()
        if len(tempos):
            idx = max(0, int(np.searchsorted(times, start_sec, side="right")) - 1)
            initial_tempo = float(tempos[idx])
    except Exception:
        pass

    pm_new = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
    for inst in pm_src.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for note in inst.notes:
            # Keep notes that overlap [start_sec, end_sec]; clip to the
            # boundary and rebase to t=0 relative to start_sec.
            if note.end <= start_sec or note.start >= end_sec:
                continue
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0.0, note.start - start_sec),
                    end=min(end_sec, note.end) - start_sec,
                )
            )
        if new_inst.notes:
            pm_new.instruments.append(new_inst)

    if not pm_new.instruments:
        # Empty segment — no notes overlap the window. Skip cleanly.
        return False

    try:
        pm_new.write(str(dst_path))
    except Exception as e:
        log.warning(f"  pretty_midi write failed for {dst_path}: {e}")
        return False
    return dst_path.exists() and dst_path.stat().st_size > 100


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
        default=None,
        help="Directory containing user audio files. Matched by piece_id stem "
        "(e.g. 00001.flac, 00001.wav, 00001.mp3, ...). "
        "OPTIONAL: omit to run the symbolic-only build (MIDI segments only). "
        "The annotations are bar-based, derived from the score; MIDI is the "
        "exact-match input domain, audio is a derivation.",
    )
    ap.add_argument(
        "--midi-segments-dir",
        default=None,
        help="Where per-segment MIDIs go. Default: <data-dir>/midi_segments. "
        "Always populated regardless of --audio-dir (symbolic is the primary path).",
    )
    ap.add_argument(
        "--skip-midi-slice",
        action="store_true",
        help="Skip MIDI segment slicing; assume segment MIDIs are already present.",
    )
    ap.add_argument(
        "--midi-source-dir",
        default=None,
        help="Directory containing user-supplied source MIDIs matched by "
        "piece_id stem (e.g. 00001.mid, 00002.mid, ...). Strongly preferred "
        "over auto-download — NinSheetMusic actively blocks scrapers "
        "(returns HTTP 403 for any non-browser request). The upstream repo "
        "README explicitly says to download manually via the url_mid links. "
        "Three viable ways to populate this dir: "
        "(1) manually click the url_mid links from metadata/pieces.csv in a "
        "browser, (2) use the `ohsheet` Rust CLI "
        "(https://crates.io/crates/ohsheet) — runs separately, requires "
        "`cargo install ohsheet`, (3) any other scraper that respects NSM's "
        "terms. The build script does NOT bundle any of these.",
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
        default=1.0,
        help="Drop segments shorter than this (default: 1.0 s). VGM "
        "stingers (`St` class) can be 1-2 s at fast tempos; the previous "
        "2.0 s default systematically dropped them. Lower further at "
        "your own risk — sub-second segments yield very few samples "
        "for an SSL encoder to embed.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    segments_dir = Path(args.segments_dir) if args.segments_dir else data_dir / "segments"
    midi_segments_dir = (
        Path(args.midi_segments_dir) if args.midi_segments_dir else data_dir / "midi_segments"
    )
    midi_dir = Path(args.midi_dir) if args.midi_dir else data_dir / "midi"
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    for d in (data_dir, segments_dir, midi_segments_dir, midi_dir):
        d.mkdir(parents=True, exist_ok=True)
    if audio_dir is not None and not audio_dir.exists():
        log.error(f"--audio-dir does not exist: {audio_dir}")
        sys.exit(1)
    audio_enabled = audio_dir is not None

    # ── Validate dependencies ──────────────────────────────────────────────
    needed_tools = [
        ("git", "needed to clone the annotation repo"),
    ]
    if audio_enabled:
        needed_tools += [
            ("ffmpeg", "needed for audio segment slicing"),
            ("ffprobe", "comes with ffmpeg"),
        ]
    for tool, hint in needed_tools:
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

    # ── Build candidate piece list ──
    # Symbolic (MIDI) is the primary path — annotations are bar-based, derived
    # from the score, so MIDI is the exact-match domain. Audio is secondary.
    # Candidates need: (1) annotation JSON, (2) entry in pieces.csv. Audio is
    # OPTIONAL — if --audio-dir is not given, only MIDI segments are produced.
    all_annotation_files = sorted(annotations_root.glob("*.json"))
    candidates: list[str] = []
    no_audio: list[str] = []
    for af in all_annotation_files:
        pid = af.stem
        if pid not in pieces_meta:
            log.debug(f"  {pid}: missing from pieces.csv; skipping")
            continue
        if audio_enabled and _find_user_audio(pid, audio_dir) is None:
            no_audio.append(pid)
            # Audio missing is NOT fatal — we still build the symbolic
            # records for this piece. Audio records just get skipped.
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
    log.info(f"  data-dir          : {data_dir}")
    log.info(f"  midi-dir          : {midi_dir}   (source MIDIs from NinSheetMusic)")
    log.info(
        f"  midi-segments-dir : {midi_segments_dir}   "
        f"(per-segment sliced MIDIs — primary symbolic path)"
    )
    log.info(
        f"  audio-dir         : {audio_dir if audio_enabled else '(none — symbolic-only mode)'}"
    )
    log.info(
        f"  segments-dir      : "
        f"{segments_dir if audio_enabled else '(skipped)'}   "
        f"(per-segment sliced audio FLACs — only if --audio-dir given)"
    )
    log.info(
        f"  candidate pieces  : {len(candidates):,} "
        f"({len(all_annotation_files):,} annotations available; "
        f"audio missing for {len(no_audio):,} of these)"
    )
    if no_audio and audio_enabled and not args.max_pieces:
        sample = ", ".join(no_audio[:5])
        log.info(
            f"  audio gaps        : {len(no_audio)} pieces — first 5: "
            f"{sample}{'...' if len(no_audio) > 5 else ''}  "
            f"(these still get symbolic records)"
        )
    log.info(f"  MIDI already cached  : {n_midi_present:,}")
    midi_source_str = args.midi_source_dir if args.midi_source_dir else "(none)"
    log.info(f"  MIDI source dir   : {midi_source_str}   (preferred over download)")
    log.info(
        f"  MIDI to fetch now : {n_to_download_midi:,}  "
        f"(source-dir first, download fallback"
        f"{' SKIPPED via --skip-midi-download' if args.skip_midi_download else ''}; "
        f"NSM auto-download typically 403s — see runbook)"
    )
    log.info(
        f"  MIDI segment slice : {'SKIPPED via --skip-midi-slice' if args.skip_midi_slice else 'enabled'}"
    )
    if audio_enabled:
        log.info(
            f"  audio slicing     : "
            f"{'SKIPPED via --skip-slice' if args.skip_slice else 'enabled'} "
            f"({args.target_sr} Hz mono FLAC)"
        )
    log.info(
        f"  upstream splits   : honoured for {len(upstream_splits):,} pieces; "
        f"seed-{args.seed} 70/15/15 for the rest"
    )
    log.info("─" * 60)

    if not candidates:
        log.error("No candidate pieces — check --audio-dir naming convention.")
        sys.exit(2)

    _install_sigint_handler()

    # ── Step 1: Source MIDIs ─────────────────────────────────────────────────
    # Priority order: (1) already cached at midi-dir/<pid>.mid, (2) copy from
    # --midi-source-dir if provided, (3) fallback auto-download (works only
    # for non-NSM hosts; NSM returns 403 for all automated requests).
    midi_source_dir = Path(args.midi_source_dir) if args.midi_source_dir else None
    if midi_source_dir is not None and not midi_source_dir.exists():
        log.error(f"--midi-source-dir does not exist: {midi_source_dir}")
        sys.exit(1)

    log.info("Sourcing per-piece MIDIs …")
    n_from_cache = 0
    n_from_source = 0
    n_from_download = 0
    n_download_failed = 0
    for i, pid in enumerate(candidates, 1):
        if _shutdown.is_set():
            break
        dst = midi_dir / f"{pid}.mid"
        if dst.exists() and dst.stat().st_size > 100:
            n_from_cache += 1
        elif midi_source_dir is not None and _import_user_midi(pid, midi_source_dir, dst):
            n_from_source += 1
        elif not args.skip_midi_download:
            url = pieces_meta[pid].get("url_mid", "")
            if _download_midi(url, dst):
                n_from_download += 1
            else:
                n_download_failed += 1
        if i % 50 == 0 or i == len(candidates):
            log.info(
                f"  [{i:>4}/{len(candidates)}]  cached={n_from_cache} "
                f"from-source={n_from_source} downloaded={n_from_download} "
                f"failed={n_download_failed}"
            )
    log.info(
        f"  MIDI sourcing complete: cached={n_from_cache}, "
        f"copied-from-source={n_from_source}, downloaded={n_from_download}, "
        f"failed={n_download_failed}"
    )
    if n_download_failed > 0 and n_from_source == 0 and midi_source_dir is None:
        log.warning(
            "NinSheetMusic returned 403 for %d pieces — this is expected; the "
            "site blocks scrapers. Either re-run with `--midi-source-dir "
            "<dir>` pointing at MIDIs you sourced manually (or via the "
            "`ohsheet` Rust CLI), or accept the partial coverage.",
            n_download_failed,
        )

    # ── Step 2: For each piece, parse → bar-time-map → slice MIDI (+ audio) ─
    # Symbolic is primary: every record gets a per-segment MIDI. Audio
    # records (with audio_path + audio metadata) are emitted alongside
    # when --audio-dir was given AND the audio slice succeeded.
    records: list[dict] = []
    canonical_label_counter: Counter[str] = Counter()
    dropped_short = 0
    dropped_oor = 0
    dropped_no_midi = 0
    failed_midi_slice = 0
    failed_audio_slice = 0
    audio_segments_written = 0
    midi_segments_written = 0

    for pid in candidates:
        midi_path = midi_dir / f"{pid}.mid"
        if not midi_path.exists():
            log.debug(f"  {pid}: no source MIDI; skipping (need bar→time mapping)")
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

        meta = pieces_meta[pid]
        audio_path = _find_user_audio(pid, audio_dir) if audio_enabled else None
        out_midi_dir = midi_segments_dir / pid
        out_audio_dir = segments_dir / pid

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

            # ── MIDI segment (always — primary symbolic path) ──────────
            midi_seg_path = out_midi_dir / f"{seg_idx:03d}_{label}.mid"
            if not args.skip_midi_slice:
                if not _slice_midi(midi_path, midi_seg_path, start_sec, end_sec):
                    failed_midi_slice += 1
                    continue
                midi_segments_written += 1
            elif not midi_seg_path.exists():
                continue

            # ── Audio segment (only if --audio-dir given) ──────────────
            audio_record_fields: dict = {}
            if audio_enabled and audio_path is not None:
                audio_seg_path = out_audio_dir / f"{seg_idx:03d}_{label}.flac"
                if not args.skip_slice:
                    ok = _slice_segment(
                        audio_path, audio_seg_path, start_sec, end_sec, args.target_sr
                    )
                    if not ok:
                        failed_audio_slice += 1
                        # Continue with MIDI-only record — symbolic still works
                        audio_seg_path = None
                if audio_seg_path is not None and audio_seg_path.exists():
                    seg_info = _ffprobe_info(audio_seg_path)
                    if seg_info[0] != 0:
                        seg_sr, seg_nsamp, seg_chan = seg_info
                        audio_record_fields = {
                            "audio_path": str(audio_seg_path.as_posix()),
                            "audio_sample_rate": seg_sr,
                            "audio_num_samples": seg_nsamp,
                            "audio_channels": seg_chan,
                            "audio_duration": round(seg_nsamp / seg_sr, 3),
                        }
                        audio_segments_written += 1

            canonical_label_counter[label] += 1
            records.append(
                {
                    # ── symbolic-primary fields ────────────────────────
                    "midi_path": str(midi_seg_path.as_posix()),
                    "ori_uid": f"{pid}_{seg_idx:03d}",  # per-segment uid
                    "work_id": pid,  # track-level grouping
                    "label": label,
                    "seg_idx": seg_idx,
                    "bar_start": bar_start,
                    "bar_end": bar_end,
                    "seg_start": round(start_sec, 3),
                    "seg_end": round(end_sec, 3),
                    # ── audio-derived fields (subset, only when slicing succeeded) ──
                    # The audio_path / audio_* fields are absent on
                    # symbolic-only records. The audio datamodule reads
                    # `audio_path` (top-level) and the standard
                    # sample_rate/num_samples/channels — for backwards
                    # compatibility with the HXMSA datamodule pattern we
                    # mirror those names below when audio is present.
                    **(
                        {
                            "audio_path": audio_record_fields["audio_path"],
                            "duration": audio_record_fields["audio_duration"],
                            "sample_rate": audio_record_fields["audio_sample_rate"],
                            "num_samples": audio_record_fields["audio_num_samples"],
                            "channels": audio_record_fields["audio_channels"],
                            "bit_depth": 16,
                        }
                        if audio_record_fields
                        else {}
                    ),
                    "title": meta.get("title", ""),
                    "ninsheetmusic_id": meta.get("ninsheetmusic_id", ""),
                }
            )

    log.info("")
    log.info(
        f"  built {len(records):,} segment records "
        f"(MIDI segments written: {midi_segments_written:,}, "
        f"audio segments written: {audio_segments_written:,})"
    )
    log.info(
        f"  dropped: {dropped_short} short, {dropped_oor} bar-out-of-range, "
        f"{dropped_no_midi} missing/unparseable MIDI, "
        f"{failed_midi_slice} MIDI-slice failures, "
        f"{failed_audio_slice} audio-slice failures"
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
