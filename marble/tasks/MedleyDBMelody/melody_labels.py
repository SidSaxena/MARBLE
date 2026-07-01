"""Pure frame-label core for the MedleyDB melody-extraction probe.

MedleyDB melody annotations are per-frame f0 in Hz on a uniform grid of
hop = 256 / 44100 s (~172.27 fps), unvoiced frames encoded as 0.0 Hz
(see Bittner et al., ISMIR 2014; confirmed against the released CSVs).

This module converts that into the frame-level MIDI-pitch labels the probe
consumes, at the encoder's token rate (``label_freq``). Kept free of torch /
torchaudio / disk so it is trivially unit-testable.
"""

from __future__ import annotations

import numpy as np

# MedleyDB f0 annotations are sampled every 256 samples at 44.1 kHz.
MEDLEYDB_NATIVE_RATE = 44100 / 256  # ≈ 172.265625 fps


def f0_to_midi(freqs: np.ndarray) -> np.ndarray:
    """Convert an array of f0 values (Hz) to integer MIDI pitch.

    Unvoiced frames (f0 == 0) map to the sentinel ``-1`` (masked by the loss
    and metrics). Voiced frames are rounded to the nearest semitone and
    clamped to the valid MIDI range [0, 127].
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    midi = np.full(freqs.shape, -1, dtype=np.int64)
    voiced = freqs > 0
    midi[voiced] = np.clip(np.rint(69.0 + 12.0 * np.log2(freqs[voiced] / 440.0)), 0, 127).astype(
        np.int64
    )
    return midi


def validate_native_grid(times: np.ndarray, *, track: str = "?") -> None:
    """Assert a MedleyDB melody CSV's time column is the canonical native grid.

    The label loader (``clip_frame_labels``) assumes row ``i`` is the frame at
    ``i / MEDLEYDB_NATIVE_RATE`` seconds — it never reads the timestamp column.
    If an annotation starts at t≠0, uses a different hop, or has a dropped/
    duplicated row, every subsequent label silently shifts against the audio.
    This turns that silent failure into a loud one at dataset construction.

    No-op for arrays too short to validate (<2 rows).
    """
    times = np.asarray(times, dtype=np.float64)
    n = times.shape[0]
    if n < 2:
        return
    hop = 1.0 / MEDLEYDB_NATIVE_RATE
    if abs(times[0]) > 0.5 * hop:
        raise ValueError(
            f"MedleyDB melody CSV '{track}': time grid must start at t=0, but the "
            f"first timestamp is {times[0]:.5f}s. Labels would be shifted."
        )
    max_dev = float(np.max(np.abs(np.diff(times) - hop)))
    if max_dev > 0.25 * hop:
        raise ValueError(
            f"MedleyDB melody CSV '{track}': non-uniform or wrong-hop time grid "
            f"(max |Δt − {hop * 1000:.3f}ms| = {max_dev * 1000:.3f}ms). Expected the "
            f"canonical 256/44100s ({MEDLEYDB_NATIVE_RATE:.3f} fps) grid; a dropped/"
            f"duplicated row or a resampled annotation would shift labels silently."
        )


def clip_frame_labels(
    track_midi: np.ndarray,
    clip_start_time: float,
    label_freq: int,
    label_len: int,
) -> np.ndarray:
    """Frame-level MIDI labels for one clip, at ``label_freq`` Hz.

    For each output frame ``k`` (center time ``clip_start_time + (k+0.5)/label_freq``)
    we nearest-sample the precomputed per-native-frame ``track_midi`` array
    (MIDI ints, -1 for unvoiced). Frames whose center falls before/after the
    annotation grid map to ``-1`` (treated as silence).
    """
    frame_times = clip_start_time + (np.arange(label_len) + 0.5) / label_freq
    idx = np.rint(frame_times * MEDLEYDB_NATIVE_RATE).astype(np.int64)
    out = np.full(label_len, -1, dtype=np.int64)
    in_range = (idx >= 0) & (idx < track_midi.shape[0])
    out[in_range] = track_midi[idx[in_range]]
    return out
