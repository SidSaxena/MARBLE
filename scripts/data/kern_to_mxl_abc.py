#!/usr/bin/env python3
"""Convert Humdrum ``**kern`` file(s) to **MusicXML** and **ABC** — general purpose.

Useful beyond CLaMP3: given ``**kern``, writes both score-native formats —
  ``<stem>.musicxml``  via converter21 (humlib-grade **kern reader) + music21
  ``<stem>.abc``       standard ABC via W.G. Vree's ``xml2abc`` (the MusicXML→ABC step)
and, optionally, CLaMP3's voice-interleaved ABC (``--interleaved``).

Provenance note: ``xml2abc.py`` is **W.G. Vree's** standard converter (the same one
CLaMP3 vendors); we reuse MARBLE's vendored copy. The ``--interleaved`` variant adds
CLaMP3's bar-aligned voice interleaving on top (only needed to *feed CLaMP3*); plain
``.abc`` is the portable, abc2*-compatible output for everything else.

Usage:
  python scripts/data/kern_to_mxl_abc.py INPUT [INPUT ...] -o OUTDIR \\
      [--glob '*.krn'] [--interleaved]
INPUT is a ``.krn`` file or a directory (recursed with ``--glob``; default ``*.krn``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marble.encoders.CLaMP3.abc_util import (  # noqa: E402
    _abc_to_interleaved,
    _register_converter21,
    _run_xml2abc,
)


def _kern_files(inputs: list[str], glob: str) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.rglob(glob)))
        elif p.is_file():
            out.append(p)
        else:
            print(f"  ! not found: {p}", file=sys.stderr)
    return out


def convert_one(krn: Path, outdir: Path, *, interleaved: bool = False) -> list[str]:
    """Write ``<stem>.musicxml`` + ``<stem>.abc`` (+ interleaved). Returns filenames."""
    import music21

    _register_converter21()
    score = music21.converter.parse(str(krn))
    stem = krn.stem
    xml_path = outdir / f"{stem}.musicxml"
    score.write("musicxml", fp=str(xml_path))
    abc = _run_xml2abc(xml_path)
    (outdir / f"{stem}.abc").write_text(abc)
    written = [xml_path.name, f"{stem}.abc"]
    if interleaved:
        (outdir / f"{stem}.interleaved.abc").write_text(_abc_to_interleaved(abc))
        written.append(f"{stem}.interleaved.abc")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="**kern file(s) or directory(ies)")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*.krn", help="glob for directory inputs (default *.krn)")
    ap.add_argument(
        "--interleaved", action="store_true", help="also write CLaMP3 voice-interleaved ABC"
    )
    args = ap.parse_args()

    files = _kern_files(args.inputs, args.glob)
    if not files:
        print("no **kern files found", file=sys.stderr)
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"converting {len(files)} kern file(s) -> {args.out_dir}")
    ok = fail = 0
    for k in files:
        try:
            written = convert_one(k, args.out_dir, interleaved=args.interleaved)
            ok += 1
            print(f"  ok   {k.name} -> {', '.join(written)}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  FAIL {k.name}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"done: {ok} ok, {fail} failed")
    if fail and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
