#!/usr/bin/env python3
"""
scripts/data/convert_mxl_to_abc.py
──────────────────────────────────
Bulk-convert MusicXML (.mxl / .musicxml / .xml) → ABC notation (.abc).

Why this exists
---------------
CLaMP3's M3 patchilizer ([marble/encoders/CLaMP3/clamp3_util.py:196](../../marble/encoders/CLaMP3/clamp3_util.py))
is bimodal: it accepts MTF (MIDI text format, event-level) OR ABC
(bar-level). MARBLE has historically fed it MTF via ``midi_to_mtf``
because we only had MIDI inputs. ABC is what CLaMP3 was primarily
trained on — bar-level tokenisation aligns naturally with the M3 patch
contract (1 patch = 1 bar) and is plausibly the higher-fidelity path
for the downstream probes (motif retrieval, structure segmentation).

This helper produces the .abc files. The SuperMarioStructure build
script's ``--build-abc`` mode does per-segment slicing on top; this
standalone tool produces full-piece ABCs (one .abc per .mxl) for
smoke-checking and any non-SMS use cases.

Pipeline
--------
.mxl → xml2abc (vendored at scripts/data/_vendor/xml2abc.py) → .abc

xml2abc is the canonical Wim-Vree MusicXML→ABC converter. We chose
it over music21's built-in ABC writer because music21's writer was
broken in our tests (it emitted ``<music21.stream.Score 0x...>``
repr instead of valid ABC).

Prerequisites
-------------
xml2abc.py (vendored, no install needed) handles the bulk of files
on its own using only stdlib. About 5-10% of SuperMario .mxl files
trip xml2abc on unsupported `<direction-type>` elements; for those we
fall back to ``music21`` to re-emit a cleaner .musicxml and retry. The
fallback is OPT-IN via ``--use-music21-fallback`` (default on);
disable to fail-fast.

  # If you want the fallback (typical):
  uv pip install music21
  uv run python scripts/data/convert_mxl_to_abc.py --in-dir ... --out-dir ...

  # Skip the fallback (faster on a clean corpus, fails ~5–10% on SMS):
  uv run python scripts/data/convert_mxl_to_abc.py --in-dir ... --out-dir ... \
      --no-music21-fallback

Usage
-----
  # Convert every .mxl in a dir → .abc alongside (out-dir flat layout)
  uv run python scripts/data/convert_mxl_to_abc.py \\
      --in-dir  data/SuperMarioStructure/mxl \\
      --out-dir data/SuperMarioStructure/abc_full

  # Pilot — first 5 files, verbose
  uv run python scripts/data/convert_mxl_to_abc.py \\
      --in-dir  data/SuperMarioStructure/mxl \\
      --out-dir /tmp/abc_pilot \\
      --max-files 5 --verbose
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_VENDORED_XML2ABC = Path(__file__).resolve().parent / "_vendor" / "xml2abc.py"
_MUSICXML_EXTS = (".mxl", ".musicxml", ".xml")


def _check_xml2abc() -> None:
    if not _VENDORED_XML2ABC.exists():
        log.error(
            f"Vendored xml2abc not found at {_VENDORED_XML2ABC}.\n"
            "  See scripts/data/_vendor/README.md for what to do."
        )
        sys.exit(1)


def _run_xml2abc(src: Path, out_dir: Path) -> tuple[bool, str]:
    """Direct xml2abc on the .mxl. Fails on the ~5-10% of SMS files with
    unsupported `<direction-type>` elements."""
    try:
        result = subprocess.run(
            [sys.executable, str(_VENDORED_XML2ABC), "-o", str(out_dir), str(src)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "xml2abc timed out (>120 s)"
    if result.returncode != 0:
        return False, f"xml2abc exit={result.returncode}: {result.stderr.strip()[:200]}"
    target = out_dir / (src.stem + ".abc")
    if target.exists() and target.stat().st_size > 0:
        return True, "ok"
    return False, f"no .abc produced; stderr: {result.stderr.strip()[:200]}"


def _run_xml2abc_via_music21(src: Path, out_dir: Path) -> tuple[bool, str]:
    """Fallback: re-emit the .mxl through music21 (which silently drops
    constructs xml2abc can't parse), write a temp .musicxml, then run
    xml2abc on the cleaned file. Recovers the ~5-10% of SMS files that
    fail the direct path."""
    try:
        import music21
    except ImportError:
        return False, "music21 not installed; pip install music21 to enable the fallback"
    import tempfile

    try:
        score = music21.converter.parse(str(src))
    except Exception as e:
        return False, f"music21 parse failed: {type(e).__name__}: {e}"
    # delete=False + manual cleanup; the context-manager form would unlink
    # the file before xml2abc can read it.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".musicxml",
        delete=False,
        dir=str(out_dir),
    )
    tmp.close()
    try:
        score.write("musicxml", fp=tmp.name)
        # xml2abc names the .abc after the temp file's stem; rename after.
        target = out_dir / (src.stem + ".abc")
        try:
            result = subprocess.run(
                [sys.executable, str(_VENDORED_XML2ABC), "-o", str(out_dir), tmp.name],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "xml2abc (m21-cleaned) timed out (>120 s)"
        if result.returncode != 0:
            return (
                False,
                f"xml2abc (m21-cleaned) exit={result.returncode}: {result.stderr.strip()[:200]}",
            )
        # xml2abc writes <tmp_stem>.abc; rename to <src_stem>.abc.
        tmp_abc = Path(tmp.name).with_suffix(".abc")
        if not tmp_abc.exists() or tmp_abc.stat().st_size == 0:
            return False, "xml2abc (m21-cleaned) produced no output"
        if target.exists():
            target.unlink()
        tmp_abc.rename(target)
        return True, "ok (via music21 fallback)"
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _convert_one(src: Path, out_dir: Path, use_music21_fallback: bool) -> tuple[bool, str]:
    """Run xml2abc on a single file with optional music21 fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, msg = _run_xml2abc(src, out_dir)
    if ok:
        return True, msg
    if not use_music21_fallback:
        return False, msg
    # Try the fallback. If it also fails, return both error messages so the
    # caller knows what was attempted.
    ok2, msg2 = _run_xml2abc_via_music21(src, out_dir)
    if ok2:
        return True, msg2
    return False, f"both paths failed; direct: {msg}; m21: {msg2}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dir", required=True, help="Directory containing .mxl/.musicxml/.xml")
    ap.add_argument("--out-dir", required=True, help="Where to write .abc files")
    ap.add_argument("--max-files", type=int, default=None, help="Pilot: only convert first N files")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files where the .abc already exists (default: overwrite)",
    )
    ap.add_argument(
        "--no-music21-fallback",
        dest="use_music21_fallback",
        action="store_false",
        help="Don't fall back to music21 → re-emit-musicxml → xml2abc for "
        "files xml2abc can't parse directly. Faster but ~5-10%% failure rate on SMS.",
    )
    ap.set_defaults(use_music21_fallback=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    _check_xml2abc()

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
        dst = out_dir / (src.stem + ".abc")
        if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
            log.debug(f"  skip (exists): {src.name}")
            skipped += 1
            continue
        success, msg = _convert_one(src, out_dir, args.use_music21_fallback)
        if success:
            log.debug(f"  ok: {src.name} → {dst.name}  ({msg})")
            ok += 1
        else:
            log.warning(f"  ! {src.name}: {msg}")
            failed += 1

    log.info(f"\nDone. {ok} converted, {skipped} skipped, {failed} failed (of {len(sources)}).")
    log.info(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
