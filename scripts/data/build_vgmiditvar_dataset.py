#!/usr/bin/env python3
"""
scripts/data/build_vgmiditvar_dataset.py
───────────────────────────────────
Render the Variation-Transformer VGMIDI-TVar (and optionally POP909-TVar)
MIDI dataset to audio and build a MARBLE-compatible JSONL.

Source
------
https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model

Filename convention
-------------------
    {piece_id}_{section}_{idx}.mid
        idx == 0     →  theme
        idx >= 1     →  variation idx of that theme
The (piece_id, section) pair identifies a "work" for retrieval purposes —
the theme and all its variations share the same work_id.

Rendering pipeline
------------------
MIDI → fluidsynth + SoundFont → 44.1 kHz mono WAV → ffprobe metadata.

To avoid the encoder latching onto a single SoundFont's timbre signature
(a real concern when probing on synthesised audio), the script rotates
through a list of SoundFonts deterministically per piece — multiple
renders per theme give the encoder variety without breaking reproducibility.

Prerequisites
-------------
  fluidsynth   - audio synthesis from MIDI
    macOS    : brew install fluid-synth
    Linux    : sudo apt install fluidsynth
    Windows  : winget install FluidSynth.FluidSynth

  ffprobe      - metadata extraction (comes with ffmpeg, already required by
                 the SHS-100K downloader)

  One or more SoundFont (.sf2) files.  Defaults to a single SoundFont
  searched at $FLUIDR3_SF2 if no explicit list provided.  Recommended:

    - SGM-V2.01 (https://archive.org/details/SGM-V2.01)
    - GeneralUser GS (https://schristiancollins.com/generaluser.php)
    - FluidR3_GM (often packaged with fluidsynth)

Usage
-----
    # Render the test split of VGMIDI-TVar (default)
    python scripts/data/build_vgmiditvar_dataset.py \\
        --midi-zip data/source/VGMIDI-TVar.zip \\
        --audio-dir data/VGMIDITVar/audio \\
        --data-dir  data/VGMIDITVar \\
        --soundfont /path/to/SGM-V2.01.sf2

    # Use multiple SoundFonts rotated per-piece (recommended)
    python scripts/data/build_vgmiditvar_dataset.py \\
        --soundfont sf2/SGM-V2.01.sf2 \\
        --soundfont sf2/GeneralUser-GS.sf2 \\
        --soundfont sf2/FluidR3_GM.sf2

    # Rebuild JSONL from already-rendered audio (no re-rendering)
    python scripts/data/build_vgmiditvar_dataset.py --skip-render

    # Render POP909-TVar instead of VGMIDI-TVar
    python scripts/data/build_vgmiditvar_dataset.py \\
        --midi-zip data/source/POP909-TVar.zip \\
        --audio-dir data/POP909TVar/audio --data-dir data/POP909TVar
"""

import argparse
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Filename pattern handles both POP909-TVar and VGMIDI-TVar conventions:
#   POP909-TVar:  052_A_0.mid                   piece=052, section=A, idx=0
#   VGMIDI-TVar:  e0_real_Other games_NES_Monster Party_Title Screen_A_1.mid
#                 piece="e0_real_Other games_NES_Monster Party_Title Screen", section=A, idx=1
# The trailing  _<section>_<idx>.mid  is what identifies the variation; everything
# before that (including underscores, spaces, slashes) is the piece identifier.
_FILENAME_RE = re.compile(r"^(?P<piece>.+)_(?P<section>[A-Z]+)_(?P<idx>\d+)\.mid$")


# ── MIDI extraction ──────────────────────────────────────────────────────────

def _extract_midi_zip(zip_path: Path, dest: Path) -> dict[str, list[Path]]:
    """Extract MIDI files from the source zip, return {split: [paths]}.

    The zip's internal layout is::

        29thSep2023_theme_var_extracted_for_training/
            train/*.mid
            test/*.mid
    """
    dest.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[Path]] = {"train": [], "test": []}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".mid"):
                continue
            if "__MACOSX" in name or "/._" in name:
                continue
            parts = Path(name).parts
            split = next((p for p in parts if p in ("train", "test")), None)
            if split is None:
                continue
            out_path = dest / split / Path(name).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            splits[split].append(out_path)
    log.info(f"  extracted: train={len(splits['train']):,}  test={len(splits['test']):,}  MIDI files")
    return splits


# ── parsing filenames → work IDs ─────────────────────────────────────────────

def _parse_filename(stem: str) -> Optional[dict]:
    """Parse  '052_A_0'  →  {'piece': '052', 'section': 'A', 'idx': 0}."""
    m = _FILENAME_RE.match(stem + ".mid")
    if not m:
        return None
    return {
        "piece":   m.group("piece"),
        "section": m.group("section"),
        "idx":     int(m.group("idx")),
    }


def _work_id_for(piece: str, section: str) -> int:
    """Stable 9-digit-ish integer derived from (piece, section).

    We avoid relying on Python's hash() because it's salted per process.
    md5 is plenty for the small label space (a few hundred works).
    """
    h = md5(f"{piece}_{section}".encode("utf-8")).hexdigest()
    # Take first 8 hex chars → fits in int32
    return int(h[:8], 16)


# ── audio rendering ──────────────────────────────────────────────────────────

def _render_midi(
    midi_path: Path,
    audio_path: Path,
    soundfont: Path,
    sample_rate: int = 44100,
) -> bool:
    """Render one MIDI to mono 44.1 kHz WAV via fluidsynth.  Returns True on success."""
    if audio_path.exists() and audio_path.stat().st_size > 1024:
        return True
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fluidsynth",
        "-ni",                          # no interactive, immediate exit
        "-g", "1.0",                    # gain
        "-r", str(sample_rate),
        "-T", "wav",                    # write WAV
        "-F", str(audio_path),
        str(soundfont),
        str(midi_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        log.warning(f"  render failed: {midi_path.name}: {e}")
        return False
    if r.returncode != 0 or not audio_path.exists():
        log.warning(f"  render failed: {midi_path.name}: rc={r.returncode}  {r.stderr[:120]}")
        return False
    return True


def _pick_soundfont(stem: str, soundfonts: list[Path]) -> Path:
    """Deterministic rotation: index = hash(stem) mod N."""
    if len(soundfonts) == 1:
        return soundfonts[0]
    return soundfonts[int(md5(stem.encode("utf-8")).hexdigest()[:8], 16) % len(soundfonts)]


# ── metadata via ffprobe ─────────────────────────────────────────────────────

def _ffprobe_info(path: Path) -> Optional[tuple[int, int, int]]:
    """Return (sample_rate, num_samples, channels), or None on failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=20,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    sr, channels, stream_dur = 0, 1, None
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
        return None
    return sr, int(sr * dur), channels


# ── main ──────────────────────────────────────────────────────────────────────

def _process_one(
    midi_path: Path,
    split: str,
    audio_dir: Path,
    soundfonts: list[Path],
    skip_render: bool,
) -> Optional[dict]:
    parsed = _parse_filename(midi_path.stem)
    if parsed is None:
        log.debug(f"  unexpected filename: {midi_path.name}")
        return None

    audio_path = audio_dir / f"{midi_path.stem}.wav"
    if not skip_render:
        sf = _pick_soundfont(midi_path.stem, soundfonts)
        ok = _render_midi(midi_path, audio_path, sf)
        if not ok:
            return None
    elif not audio_path.exists():
        return None

    info = _ffprobe_info(audio_path)
    if info is None:
        log.warning(f"  ffprobe failed: {audio_path.name}")
        return None
    sr, n_samples, channels = info

    return {
        "audio_path":  str(audio_path),
        "work_id":     _work_id_for(parsed["piece"], parsed["section"]),
        "variation":   parsed["idx"],
        "piece_id":    parsed["piece"],
        "section":     parsed["section"],
        "split":       split,
        "sample_rate": sr,
        "num_samples": n_samples,
        "channels":    channels,
        "duration":    round(n_samples / sr, 3),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Render VGMIDI-TVar MIDI to audio and build JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--midi-zip", default="data/source/VGMIDI-TVar.zip",
                    help="Path to the source zip from the Variation-Transformer "
                         "repo (default: %(default)s)")
    ap.add_argument("--audio-dir", default="data/VGMIDITVar/audio",
                    help="Where rendered WAVs go (default: %(default)s)")
    ap.add_argument("--data-dir", default="data/VGMIDITVar",
                    help="Where the JSONL is written (default: %(default)s)")
    ap.add_argument("--soundfont", action="append", default=[],
                    help="Path to a .sf2 SoundFont.  Pass multiple --soundfont "
                         "flags for SoundFont rotation (recommended).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel rendering workers (default: 4)")
    ap.add_argument("--skip-render", action="store_true",
                    help="Don't render; rebuild JSONL from existing WAVs only.")
    ap.add_argument("--midi-extract-dir", default=None,
                    help="Where to extract the MIDI files (default: <data-dir>/midi)")
    args = ap.parse_args()

    # ── Validate dependencies ────────────────────────────────────────────────
    if not args.skip_render:
        for tool in ("fluidsynth", "ffprobe"):
            if shutil.which(tool) is None:
                log.error(f"{tool} not found on PATH.  See the script header for "
                          f"install instructions.")
                sys.exit(1)
        soundfonts = [Path(p) for p in args.soundfont]
        if not soundfonts:
            default_sf = Path(__import__("os").environ.get("FLUIDR3_SF2", ""))
            if default_sf.exists():
                soundfonts = [default_sf]
            else:
                log.error("No --soundfont provided and $FLUIDR3_SF2 is unset.  "
                          "Download SGM-V2.01 (free) and pass --soundfont path/to/SGM-V2.01.sf2")
                sys.exit(1)
        for sf in soundfonts:
            if not sf.exists():
                log.error(f"SoundFont not found: {sf}")
                sys.exit(1)
        log.info(f"SoundFonts ({len(soundfonts)}): {[s.name for s in soundfonts]}")
    else:
        soundfonts = []

    # ── Extract MIDI from zip ────────────────────────────────────────────────
    midi_extract_dir = Path(args.midi_extract_dir or Path(args.data_dir) / "midi")
    midi_zip = Path(args.midi_zip)
    if not args.skip_render:
        if not midi_zip.exists():
            log.error(f"MIDI zip not found: {midi_zip}")
            sys.exit(1)
        log.info(f"Extracting {midi_zip} → {midi_extract_dir}")
        splits = _extract_midi_zip(midi_zip, midi_extract_dir)
    else:
        # Discover by walking the extract dir
        splits = {"train": [], "test": []}
        for split in splits:
            d = midi_extract_dir / split
            if d.exists():
                splits[split] = sorted(d.glob("*.mid"))
        log.info(f"  found extracted MIDI: "
                 f"train={len(splits['train']):,}  test={len(splits['test']):,}")

    audio_dir = Path(args.audio_dir)
    data_dir  = Path(args.data_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Render + parse ───────────────────────────────────────────────────────
    records: list[dict] = []
    total = sum(len(v) for v in splits.values())
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for split, paths in splits.items():
            for midi_path in paths:
                futs[pool.submit(
                    _process_one, midi_path, split, audio_dir,
                    soundfonts, args.skip_render
                )] = (midi_path, split)
        for fut in as_completed(futs):
            n_done += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if n_done % 100 == 0 or n_done == total:
                pct = 100 * n_done // total
                log.info(f"  [{pct:3d}%] {n_done:>5}/{total}  ok={len(records):>5}")

    if not records:
        log.error("No records — nothing to write")
        sys.exit(1)

    # Sort for deterministic JSONL output
    records.sort(key=lambda r: (r["split"], r["piece_id"], r["section"], r["variation"]))

    # ── Write JSONL (single file with split field) ───────────────────────────
    out = data_dir / "VGMIDITVar.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_works = len({(r["piece_id"], r["section"]) for r in records})
    n_train = sum(1 for r in records if r["split"] == "train")
    n_test  = sum(1 for r in records if r["split"] == "test")
    log.info("")
    log.info("=" * 60)
    log.info(f" Wrote {len(records):,} entries → {out}")
    log.info(f"   train: {n_train:,}  test: {n_test:,}")
    log.info(f"   unique works (piece+section): {n_works:,}")
    log.info(f"   avg variations per work: {len(records) / n_works:.1f}")
    log.info("=" * 60)
    log.info("Next: configure a layer sweep with this JSONL")


if __name__ == "__main__":
    main()
