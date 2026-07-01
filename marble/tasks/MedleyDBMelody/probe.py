# File: marble/tasks/MedleyDBMelody/probe.py
"""MedleyDB melody-extraction probe.

The frame-level melody machinery — ``ProbeAudioTask`` (MLPDecoderKeepTime head,
test_step that forwards clip_ids), the masked ``MelodyCrossEntropyLoss``, and
the ``RawPitchAccuracy`` / ``RawChromaAccuracy`` metrics — is generic and
identical to HookTheoryMelody's. We re-export it here (single source of truth)
so MedleyDB and HookTheory layer curves stay strictly comparable: a change to
the probe necessarily applies to both. Configs reference
``marble.tasks.MedleyDBMelody.probe.<symbol>``.
"""

from marble.tasks.HookTheoryMelody.probe import (  # noqa: F401
    MelodyCrossEntropyLoss,
    ProbeAudioTask,
    RawChromaAccuracy,
    RawPitchAccuracy,
)

__all__ = [
    "ProbeAudioTask",
    "MelodyCrossEntropyLoss",
    "RawPitchAccuracy",
    "RawChromaAccuracy",
]
