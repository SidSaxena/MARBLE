# File: marble/tasks/MedleyDBMelody/datamodule.py
"""DataModule for the MedleyDB melody-extraction probe.

Frame-level MIDI-pitch classification, mirroring HookTheoryMelody, but the
labels come from MedleyDB's per-frame f0 annotations (Hz on a 256/44100 s grid,
0 Hz = unvoiced) instead of beat-aligned symbolic notes. The audio/clip
machinery is inherited from ``BaseAudioDataset``; only target construction
differs (``get_targets`` → nearest-sample f0→MIDI onto the encoder token grid).

5-fold CV (BPS-Motif convention): each split dataset takes ``fold_idx`` (0–4)
and a ``jsonl_template`` with ``{fold}``/``{split}`` placeholders, and exposes
``self.fold_idx`` so ``LogSweepCoordsCallback`` can stamp ``sweep/fold``. Each
JSONL record:
    {
        "audio_path":  ".../<Track>_MIX.wav",
        "melody_csv":  ".../MELODY2/<Track>_MELODY2.csv",
        "sample_rate": 44100,
        "num_samples": <int>,
        "track":       "<Track>",
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from marble.core.base_datamodule import BaseAudioDataset, BaseDataModule

from .melody_labels import clip_frame_labels, f0_to_midi, validate_native_grid


class _MedleyDBMelodyDataset(BaseAudioDataset):
    """MedleyDB melody dataset: full-mix audio clips → frame-level MIDI labels."""

    JSONL_TEMPLATE = "data/MedleyDB/MedleyDBMelody.fold{fold}.{split}.jsonl"

    def __init__(
        self,
        split: str,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        label_freq: int,
        fold_idx: int = 0,
        jsonl_template: str | None = None,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
        backend: str | None = None,
    ):
        # Frame-grid sanity (fail fast, before any I/O): label_freq × clip_seconds
        # must be integer, else per-frame label/token alignment drifts silently.
        label_len_float = label_freq * clip_seconds
        if abs(label_len_float - round(label_len_float)) > 1e-6:
            raise ValueError(
                f"label_freq ({label_freq}) × clip_seconds ({clip_seconds}) "
                f"= {label_len_float} must be integer; otherwise per-frame label "
                f"indices drift relative to encoder-token indices. Pick clip_seconds "
                f"that divides 1/label_freq cleanly (e.g. 15.0 @ 25 Hz = 375 frames)."
            )

        jsonl = (jsonl_template or self.JSONL_TEMPLATE).format(fold=fold_idx, split=split)
        super().__init__(
            jsonl=jsonl,
            sample_rate=sample_rate,
            channels=channels,
            clip_seconds=clip_seconds,
            label_freq=label_freq,
            channel_mode=channel_mode,
            min_clip_ratio=min_clip_ratio,
            backend=backend,
        )
        # Authoritative CV-fold source for LogSweepCoordsCallback (sweep/fold).
        self.split = split
        self.fold_idx = fold_idx

        # Load each track's f0 CSV once and convert to a per-native-frame MIDI
        # array (parallel to self.meta). One small int64 array per track
        # (~50k ints), held once and COW-shared across forked DataLoader
        # workers — get_targets does only a cheap gather, no per-call parse.
        # We read BOTH columns so the time grid can be validated (a dropped row
        # or wrong hop would otherwise shift every label silently).
        self._track_midi: list[np.ndarray] = []
        for info in self.meta:
            csv_path = info["melody_csv"]
            track = info.get("track", csv_path)
            try:
                arr = pd.read_csv(csv_path, header=None, dtype=np.float64).to_numpy()
            except pd.errors.EmptyDataError as e:
                raise ValueError(f"MedleyDB melody CSV '{track}' is empty ({csv_path}).") from e
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"MedleyDB melody CSV '{track}': expected 2 columns (time_sec, freq_hz), "
                    f"got shape {arr.shape} ({csv_path})."
                )
            validate_native_grid(arr[:, 0], track=track)
            self._track_midi.append(f0_to_midi(arr[:, 1]))

    def get_targets(self, file_idx: int, slice_idx: int, orig_sr: int, orig_clip_frames: int):
        clip_start_time = slice_idx * (orig_clip_frames / orig_sr)
        label_len = int(round(self.label_freq * self.clip_seconds))
        labels = clip_frame_labels(
            self._track_midi[file_idx],
            clip_start_time=clip_start_time,
            label_freq=self.label_freq,
            label_len=label_len,
        )
        return torch.from_numpy(labels)


class MedleyDBMelodyTrain(_MedleyDBMelodyDataset):
    """Training split: shuffle in DataLoader."""

    def __init__(self, **kwargs):
        super().__init__(split="train", **kwargs)


class MedleyDBMelodyVal(_MedleyDBMelodyDataset):
    """Validation split: no shuffling."""

    def __init__(self, **kwargs):
        super().__init__(split="val", **kwargs)


class MedleyDBMelodyTest(_MedleyDBMelodyDataset):
    """Test split: same behavior as validation."""

    def __init__(self, **kwargs):
        super().__init__(split="test", **kwargs)


class MedleyDBMelodyDataModule(BaseDataModule):
    """
    DataModule for the MedleyDB Melody task (5-fold CV).

    Configuration example:
        datamodule:
            _target_: marble.tasks.MedleyDBMelody.datamodule.MedleyDBMelodyDataModule
            batch_size: 16
            num_workers: 8
            train: { _target_: ...MedleyDBMelodyTrain, fold_idx: 0, sample_rate: 24000, ... }
            val:   { _target_: ...MedleyDBMelodyVal,   fold_idx: 0, ... }
            test:  { _target_: ...MedleyDBMelodyTest,  fold_idx: 0, ... }
    """

    pass
