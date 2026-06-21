# marble/encoders/CLaMP3/abc_util.py
"""Score-native symbolic input for CLaMP3: ``**kern`` / MusicXML → interleaved ABC.

CLaMP3's M3 symbolic encoder was trained on two text views of music:

* **MTF** (MIDI Text Format) — a lossless serialisation of MIDI *performance*
  (exact ticks/velocities, message-segmented). This is what ``midi_util.py``
  produces and what the symbolic datamodules currently feed.
* **interleaved ABC** — bar-segmented *notation* (key, pitch spelling, meter,
  beaming, slurs) with multiple voices interleaved bar-by-bar.

For motif/structure retrieval over score-native datasets (MTC-ANN, JKUPDD,
BPS-Motif) the ABC path is preferable: ABC patches are bar-aligned (motifs
live in bars) and carry notation semantics that ``**kern → MIDI → MTF``
discards (key, pitch spelling, meter, phrasing). See
``docs/symbolic_kern_to_abc_conversion.md`` for the rationale.

This module is the **shared converter** all three datasets reuse. It does NOT
do per-dataset motif-window slicing — that lives in each dataset's code (each
supplies its own note-spans and feeds a music21 fragment to ``score_to_abc``).

Pipeline
--------
::

    **kern  --converter21 (humlib-derived, registered into music21)-->  MusicXML
            --(vendored xml2abc v174)-->  standard ABC
            --(abctoolkit interleave: strip metadata + voice-rotate)-->  interleaved ABC
            -->  M3Patchilizer.encode  -->  CLaMP3 symbolic encoder

The interleave step (``_abc_to_interleaved``) is **bit-faithful** to CLaMP3's
training preprocessing (``preprocessing/abc/batch_interleaved_abc.py``). It was
ported from the leitmotifs adapter
(``scripts/utils/clamp3_adapter.py::_abc_to_interleaved``), which is the
reference implementation in this codebase family; a sibling copy lives in
``scripts/data/build_supermario_dataset.py``. Keep the three in sync when the
interleave / strip rules change.

Dependencies (install via ``uv sync --extra symbolic-abc``):
``converter21`` (kern→MusicXML), ``music21`` (slicing + MusicXML write),
``abctoolkit`` (interleave). The MusicXML→ABC step shells out to the vendored
``scripts/data/_vendor/xml2abc.py`` (W.G. Vree, v174), exactly as the
leitmotifs and SuperMario paths do.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Vendored W.G. Vree MusicXML → ABC converter (v174), already in the repo for
# the SuperMario ABC path. Reused here so we don't carry a second copy.
# This file is marble/marble/encoders/CLaMP3/abc_util.py → parents[3] is the
# repo root (parents[2] is the `marble` package dir).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_XML2ABC = _REPO_ROOT / "scripts" / "data" / "_vendor" / "xml2abc.py"

# xml2abc is mostly fast (<1 s) but a pathological MusicXML can loop on element
# traversal; bound each subprocess. Matches the leitmotifs adapter's timeout.
_XML2ABC_TIMEOUT_SEC = 120

# xml2abc flags, matching the proven leitmotifs invocation
# (clamp3_adapter.py): -d 8 sets the ABC default note length to L:1/8 (the
# value CLaMP3's preprocessing used); -x disables line breaks so each voice's
# tune body is a single line (required by abctoolkit's bar-aligned rotation).
_XML2ABC_ARGS = ["-d", "8", "-x"]

_CONVERTER21_REGISTERED = False


# ─── MusicXML → standard ABC (vendored xml2abc subprocess) ───────────────────


def _run_xml2abc(xml_path: Path) -> str:
    """Run the vendored xml2abc on a MusicXML file, returning ABC from stdout.

    Mirrors ``batch_xml2abc.convert_xml2abc`` (and the leitmotifs adapter):
    xml2abc is a self-contained CLI that prints ABC to stdout, so we invoke it
    as a subprocess rather than importing its ``__main__``-gated logic.

    Raises:
        RuntimeError: if the vendored script is missing, times out, exits
            non-zero, or produces empty output.
    """
    if not _VENDORED_XML2ABC.exists():
        raise RuntimeError(
            f"Vendored xml2abc.py not found at {_VENDORED_XML2ABC}. "
            "Re-checkout scripts/data/_vendor/xml2abc.py."
        )
    try:
        result = subprocess.run(
            [sys.executable, str(_VENDORED_XML2ABC), *_XML2ABC_ARGS, str(xml_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=_XML2ABC_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"xml2abc.py timed out (>{_XML2ABC_TIMEOUT_SEC}s) on {xml_path.name}"
        ) from e
    # xml2abc can return rc=0 yet emit nothing for unsupported constructs.
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"xml2abc.py failed on {xml_path.name}: rc={result.returncode}; "
            f"stderr={result.stderr.strip()[-300:]}"
        )
    return result.stdout


# ─── standard ABC → interleaved ABC (abctoolkit) ─────────────────────────────


def _abc_to_interleaved(abc_text: str) -> str:
    """Convert standard (multi-voice) ABC into CLaMP3's INTERLEAVED ABC.

    This is the post-xml2abc cleanup + bar-by-bar voice rotation that CLaMP3's
    symbolic-branch training preprocessing applied
    (``preprocessing/abc/batch_interleaved_abc.py::abc_pipeline``). Without it,
    multi-voice scores serialise as ``V:1 [all bars]\\nV:2 [all bars]`` — which
    is out-of-distribution for the M3 patchiliser. With it, bodies become
    ``[V:1]bar1|[V:2]bar1|\\n[V:1]bar2|[V:2]bar2|...`` — voices interleaved by
    bar. It also strips the non-musical metadata fields (X:, T:, C:, W:, w:, Z:,
    %%MIDI) and xml2abc's ``%N`` bar-number comments, matching training.

    Ported bit-faithfully from leitmotifs
    (``scripts/utils/clamp3_adapter.py::_abc_to_interleaved``), which mirrors
    the upstream ``abc_pipeline``. The ``abctoolkit`` primitives are pulled at
    call time so importing this module is cheap and doesn't hard-require the
    ``symbolic-abc`` extra.

    For a single-voice melody (MTC-ANN) the "interleave" is a no-op rotation
    over one voice — the strip/clean steps still apply, which is what we want.

    Raises:
        ImportError: if ``abctoolkit`` is not installed (``uv sync --extra
            symbolic-abc``).
        RuntimeError: if abctoolkit's ``strip_empty_bars`` / ``rotate_abc``
            return ``None`` (degenerate / unalignable ABC).
    """
    from abctoolkit.rotate import rotate_abc
    from abctoolkit.utils import (
        Barlines,
        Quote_re,
        remove_bar_no_annotations,
        remove_information_field,
        strip_empty_bars,
    )

    abc_lines = [line + "\n" for line in abc_text.splitlines() if line.strip()]
    abc_lines = remove_information_field(
        abc_lines=abc_lines,
        info_fields=["X:", "T:", "C:", "W:", "w:", "Z:", "%%MIDI"],
    )
    abc_lines = remove_bar_no_annotations(abc_lines)

    # Strip escaped quotes + annotation text that contains barline chars (rare
    # but breaks rotate_abc's bar splitting).
    for i, line in enumerate(abc_lines):
        if not (re.search(r"^[A-Za-z]:", line) or line.startswith("%")):
            abc_lines[i] = line.replace(r"\"", "")
            for quote_content in re.findall(Quote_re, line):
                for barline in Barlines:
                    if barline in quote_content:
                        abc_lines[i] = abc_lines[i].replace(quote_content, "")

    stripped, _bar_counts = strip_empty_bars(abc_lines)
    if stripped is None:
        raise RuntimeError("strip_empty_bars returned None (degenerate ABC).")

    rotated = rotate_abc(stripped)
    if rotated is None:
        raise RuntimeError("rotate_abc returned None (unalignable ABC).")

    return "".join(rotated)


# ─── Public API ──────────────────────────────────────────────────────────────


def musicxml_to_interleaved_abc(xml: str | Path) -> str:
    """MusicXML file → CLaMP3 interleaved ABC.

    Args:
        xml: path to a ``.musicxml`` / ``.xml`` / ``.mxl`` file.

    Returns:
        Interleaved-ABC string, ready for ``M3Patchilizer.encode`` and the
        CLaMP3 symbolic encoder.

    Raises:
        FileNotFoundError: if ``xml`` does not exist.
        RuntimeError / ImportError: see ``_run_xml2abc`` / ``_abc_to_interleaved``.
    """
    xml_path = Path(xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"MusicXML not found: {xml_path}")
    abc = _run_xml2abc(xml_path)
    return _abc_to_interleaved(abc)


def _register_converter21() -> None:
    """Register converter21's humlib-derived Humdrum importer into music21.

    Idempotent — converter21's ``register()`` swaps its high-fidelity
    ``**kern → MusicXML`` reader in for music21's lossy built-in one, so every
    subsequent ``music21.converter.parse(<.krn>)`` uses it. We register once.
    """
    global _CONVERTER21_REGISTERED
    if _CONVERTER21_REGISTERED:
        return
    import converter21

    converter21.register()
    _CONVERTER21_REGISTERED = True


def kern_to_abc(krn_path: str | Path) -> str:
    """``**kern`` (Humdrum) file → CLaMP3 interleaved ABC.

    Pipeline: ``converter21`` parses the ``**kern`` (humlib-grade fidelity,
    registered into music21) → music21 writes MusicXML to a temp file →
    ``musicxml_to_interleaved_abc``. converter21's reader preserves key/spelling
    /meter/phrasing that the ``**kern → MIDI`` path drops.

    Args:
        krn_path: path to a ``.krn`` Humdrum file.

    Returns:
        Interleaved-ABC string.

    Raises:
        FileNotFoundError: if ``krn_path`` does not exist.
        RuntimeError: if music21 cannot parse the kern or write MusicXML.
        ImportError: if ``converter21`` / ``music21`` are not installed.
    """
    krn = Path(krn_path)
    if not krn.exists():
        raise FileNotFoundError(f"**kern file not found: {krn}")

    import music21

    _register_converter21()
    try:
        score = music21.converter.parse(str(krn))
    except Exception as e:  # noqa: BLE001 — surface a clear, actionable error
        raise RuntimeError(
            f"converter21/music21 failed to parse {krn.name}: {type(e).__name__}: {e}"
        ) from e
    return score_to_abc(score)


def score_to_abc(m21_score) -> str:
    """A music21 stream (e.g. a sliced motif fragment) → CLaMP3 interleaved ABC.

    Writes the stream to a temp MusicXML and runs the MusicXML→interleaved-ABC
    path. This is the entry point per-dataset code uses for motif WINDOWS:
    parse the source with converter21 registered, slice the note/measure range
    (e.g. ``score.measures(start, end)``), then pass the fragment here.

    Fragment caveats (the caller owns these — music21 carries most of it
    automatically when you slice via ``.measures()``):

    * **Key / clef / meter context.** ``Stream.measures(a, b)`` copies the
      governing ``KeySignature``/``Clef``/``TimeSignature`` into the first
      measure of the slice by default (``collect=...``), so the ABC ``K:``/
      ``M:`` headers reflect the fragment's true context rather than C-major
      defaults. If you build a fragment some other way (raw ``insert`` of
      notes), set those context objects yourself before calling this.
    * **Partial bars.** A motif that starts/ends mid-bar yields an anacrusis /
      truncated final bar. xml2abc + abctoolkit handle incomplete bars (ABC has
      no fixed-width-bar requirement); ``strip_empty_bars`` only drops *empty*
      bars, not short ones. The bar-aligned patching still places the partial
      bar in its own patch.

    Args:
        m21_score: a ``music21.stream.Score``/``Part``/``Stream``.

    Returns:
        Interleaved-ABC string for the fragment.

    Raises:
        RuntimeError: if music21 cannot write the stream to MusicXML.
    """
    # delete=False + manual cleanup: xml2abc reads the file after we close it,
    # and on Windows an open NamedTemporaryFile can't be reopened by another
    # process. Mirrors the leitmotifs / SuperMario temp-file handling.
    tmp = tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False)  # noqa: SIM115
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        try:
            m21_score.write("musicxml", fp=str(tmp_path))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"music21 failed to write MusicXML for the fragment: {type(e).__name__}: {e}"
            ) from e
        return musicxml_to_interleaved_abc(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
