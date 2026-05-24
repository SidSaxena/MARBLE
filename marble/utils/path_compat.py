"""
marble/utils/path_compat.py

Cross-platform path compatibility for JSONL ``audio_path`` strings.

Background
----------
MARBLE datasets store audio file locations as plain strings inside JSONL
metadata files. Those JSONLs are commonly generated on one OS (e.g.
Windows when running ``scripts/data/download_nsynth.py`` locally) and
consumed on another (e.g. Linux on Modal / RunPod / Azure for actual
training). On Windows, ``str(pathlib.Path("data/NSynth/foo.wav"))``
yields ``"data\\NSynth\\foo.wav"`` (backslash-separated). On Linux,
``\\`` is a literal filename character, not a path separator, so
``torchaudio.load("data\\NSynth\\foo.wav")`` fails with
``"No such file or directory"``.

Forward slashes work natively on Linux/macOS AND on Windows (the
Win32 file APIs accept both separators), so emitting forward slashes
universally is the safe lingua franca.

Helpers
-------
- ``posix_path(p)``  : normalize ``\\`` → ``/`` in a single string.
  Use on the *reader* side, applied to ``info["audio_path"]`` right
  after JSONL load.

- ``as_posix_str(p)``: shorthand for ``Path(p).as_posix()``. Use on
  the *writer* side, when serialising a ``Path`` to JSONL. The
  ``str()`` builtin is OS-flavoured (backslashes on Windows);
  ``.as_posix()`` is always forward-slash.

Two-layer defence
-----------------
Writers use ``as_posix_str`` so NEW JSONLs are forward-slash everywhere.
Readers use ``posix_path`` so EXISTING Windows-generated JSONLs work
without regeneration. Either layer alone fixes the bug; together they
make the failure mode unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath


def posix_path(p: str) -> str:
    """Normalise a stored path string to forward slashes.

    Idempotent: a path that's already POSIX-style is returned unchanged.
    Does not touch the filesystem — purely a string transform.

    >>> posix_path("data\\NSynth\\nsynth-valid\\audio\\foo.wav")
    'data/NSynth/nsynth-valid/audio/foo.wav'
    >>> posix_path("data/NSynth/foo.wav")
    'data/NSynth/foo.wav'
    """
    return p.replace("\\", "/")


def as_posix_str(p: str | PurePath) -> str:
    """Serialise a path-like to a forward-slash string for JSONL output.

    Prefer this over plain ``str(path)`` in any code that writes
    ``audio_path`` (or any other path) into JSONL metadata, so the file
    is portable across operating systems.

    Accepts:
      - ``pathlib.Path`` (concrete OS path)        → ``.as_posix()``
      - ``pathlib.PureWindowsPath`` / ``PurePosixPath`` → ``.as_posix()``
      - ``str`` (possibly Windows-style)           → backslash normalisation

    For string input we cannot round-trip through ``Path(p).as_posix()``
    because on POSIX hosts ``pathlib.Path`` treats ``\\`` as a literal
    filename character (not a separator), so the backslashes survive
    unchanged. We normalise the string directly instead.

    >>> from pathlib import PureWindowsPath
    >>> as_posix_str(PureWindowsPath("data") / "foo.wav")
    'data/foo.wav'
    >>> as_posix_str("data\\NSynth\\foo.wav")
    'data/NSynth/foo.wav'
    """
    if isinstance(p, PurePath):
        return p.as_posix()
    return posix_path(p)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file as a list of dicts, normalising ``audio_path``.

    Single chokepoint for every MARBLE datamodule that reads metadata
    from JSONL. Any record's ``audio_path`` field (if present) is
    coerced to forward slashes via :func:`posix_path`, so a JSONL
    generated on Windows is transparently usable on Linux/macOS and vice
    versa.

    Records without an ``audio_path`` field are passed through
    unchanged — the helper is safe for non-audio JSONL too.
    """
    with open(path) as f:
        records: list[dict] = [json.loads(line) for line in f]
    for r in records:
        if "audio_path" in r:
            r["audio_path"] = posix_path(r["audio_path"])
    return records
