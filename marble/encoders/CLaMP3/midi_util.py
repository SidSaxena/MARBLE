# marble/encoders/CLaMP3/midi_util.py
"""
MIDI ↔ MTF (Music Text Format) conversion utilities for CLaMP3's
symbolic-music encoder.

MTF is the text serialisation used by the CLaMP3 M3 model.  Each line is
either::

    ticks_per_beat <N>
    <msg_type> <arg1> <arg2> ...

where messages are produced by serialising every `mido` MIDI event with
its ``dict()`` representation.

Adapted from the upstream CLaMP3 preprocessing pipeline:
  https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_midi2mtf.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import mido


_M3_SKIP_META = {
    "text", "copyright", "track_name", "instrument_name",
    "lyrics", "marker", "cue_marker", "device_name",
}


def _msg_to_str(msg: mido.Message) -> str:
    """Serialise a single mido message dict to a whitespace-joined string."""
    parts = []
    for _, value in msg.dict().items():
        parts.append(str(value))
    # Match upstream's unicode_escape behaviour so non-ASCII metadata can't
    # break the patch encoder (which dispatches byte-by-byte).
    return " ".join(parts).encode("unicode_escape").decode("utf-8")


def midi_to_mtf(midi_path: Union[str, Path], m3_compatible: bool = True) -> str:
    """Read a MIDI file and return its MTF-format text.

    Parameters
    ----------
    midi_path : str | Path
        Source MIDI file.
    m3_compatible : bool, default True
        When True (the default used for CLaMP3 inference), strip metadata
        events that the M3 model was trained to ignore: ``text``,
        ``copyright``, ``track_name``, ``instrument_name``, ``lyrics``,
        ``marker``, ``cue_marker``, ``device_name``.

    Returns
    -------
    str
        Newline-separated MTF text starting with ``ticks_per_beat <N>``.
    """
    mid = mido.MidiFile(str(midi_path))
    lines = [f"ticks_per_beat {mid.ticks_per_beat}"]
    for msg in mid.merged_track:
        if m3_compatible and msg.is_meta and msg.type in _M3_SKIP_META:
            continue
        lines.append(_msg_to_str(msg))
    return "\n".join(lines)
