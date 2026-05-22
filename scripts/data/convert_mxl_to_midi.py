#!/usr/bin/env python3
"""
scripts/data/convert_mxl_to_midi.py
───────────────────────────────────
Bulk-convert MusicXML (.mxl / .musicxml / .xml) → MIDI (.mid).

Why this exists
---------------
The SuperMarioStructure build pipeline (and any future MARBLE pipeline
that consumes MIDI) expects `<piece_id>.mid` under `--midi-source-dir`.
NinSheetMusic's original score files are .mus (Finale's binary format)
which is hostile to open tooling — most of our path was blocked on
"how do you read .mus without owning Finale?". MusicXML (.mxl) is the
open interchange format Finale (and MuseScore, Sibelius, ...) all
export to without losing the score-level information that .mid would
discard (dynamics, articulation, slurs, voicings, repeat structure).

Once you have .mxl files, this helper does the one remaining step —
.mxl → .mid via music21 — so the existing build script can consume them
through `--midi-source-dir` without any other changes.

What this does (and does not) do
--------------------------------
- DOES convert .mxl / .musicxml / .xml → .mid via music21's standard
  `converter.parse(...).write('midi', fp=...)` pipeline. music21
  preserves tempo, time signature, key signature, voicing, ties, and
  expands `volta`/repeat structure correctly — strictly better than
  any .mus → .mid path Finale would give.
- DOES preserve the original stem (e.g. `00042.mxl` → `00042.mid`,
  `00042_Goombas_March.mxl` → `00042_Goombas_March.mid`) so the
  SuperMarioStructure build's existing `<piece_id>(_<slug>)?.mid` name
  patterns match without renaming.
- DOES NOT touch .mus binary files — music21 can't read those directly.
  If you only have .mus, open them in MuseScore or Finale and Save As
  → MusicXML first, then point this script at the .mxl dir.
- DOES NOT re-emit MusicXML's dynamics / articulation into the MIDI
  (those don't have a lossless MIDI equivalent). For tasks that NEED
  that information, use a future encoder that consumes MusicXML / ABC
  directly (e.g. NotaGen) rather than going through MIDI.

Prerequisites
-------------
  music21 (one-time install — not a core MARBLE dep because most users
  who use MARBLE never go through this path):

    uv pip install music21

Usage
-----
  # Convert every .mxl in a dir → .mid alongside (default)
  uv run python scripts/data/convert_mxl_to_midi.py \\
      --in-dir  /path/to/your/mxl_files \\
      --out-dir data/SuperMarioStructure/midi_user

  # Pilot — first 5 files, verbose
  uv run python scripts/data/convert_mxl_to_midi.py \\
      --in-dir /path/to/your/mxl_files \\
      --out-dir /tmp/midi_pilot \\
      --max-files 5 --verbose

Then feed the output directory to the SuperMarioStructure build:

  uv run python scripts/data/build_supermario_dataset.py \\
      --midi-source-dir data/SuperMarioStructure/midi_user
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# music21 will accept any of these as input.
_MUSICXML_EXTS = (".mxl", ".musicxml", ".xml")


def _convert_one(src: Path, dst: Path) -> tuple[bool, str]:
    """Convert one MusicXML file to MIDI. Returns (ok, message)."""
    import music21  # lazy — heavy import (~1.5 s); only pay once

    try:
        score = music21.converter.parse(str(src))
    except Exception as e:
        return False, f"parse failed: {type(e).__name__}: {e}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        score.write("midi", fp=str(dst))
    except Exception as e:
        return False, f"write failed: {type(e).__name__}: {e}"

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--in-dir", required=True, help="Directory containing .mxl/.musicxml/.xml files"
    )
    ap.add_argument("--out-dir", required=True, help="Directory to write .mid files into")
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Pilot: convert only the first N files (default: all)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files where the .mid output already exists (default: overwrite)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        log.error(f"--in-dir {in_dir} does not exist")
        sys.exit(1)

    sources = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in _MUSICXML_EXTS)
    if not sources:
        log.error(f"No MusicXML files in {in_dir}. Looking for: {_MUSICXML_EXTS}")
        sys.exit(1)
    if args.max_files:
        sources = sources[: args.max_files]
    log.info(f"Found {len(sources)} MusicXML files to convert.")

    ok = skipped = failed = 0
    for src in sources:
        dst = out_dir / (src.stem + ".mid")
        if args.skip_existing and dst.exists():
            log.debug(f"  skip (exists): {src.name}")
            skipped += 1
            continue
        success, msg = _convert_one(src, dst)
        if success:
            log.debug(f"  ok: {src.name} → {dst.name}")
            ok += 1
        else:
            log.warning(f"  ! {src.name}: {msg}")
            failed += 1

    log.info(
        f"\nDone. {ok} converted, {skipped} skipped, {failed} failed "
        f"(out of {len(sources)}). Output dir: {out_dir}/"
    )


if __name__ == "__main__":
    main()
