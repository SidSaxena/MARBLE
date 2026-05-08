# marble/tasks/NSynth/datamodule.py
"""
NSynth pitch classification datamodule.

Dataset format — one JSON line per audio clip:
  {
    "audio_path": "data/NSynth/nsynth-train/audio/bass_acoustic_000-021-075.wav",
    "note":       21,           # MIDI note number (21–108 inclusive)
    "velocity":   75,           # MIDI velocity (25 | 50 | 75 | 100 | 127)
    "instrument_family": "bass",
    "instrument_source": "acoustic",
    "sample_rate": 16000,
    "num_samples": 64000,       # always 4 s × 16 kHz = 64 000
    "channels":    1,
    "duration":    4.0
  }

Label mapping:  class_idx = note - 21   →  88 classes  (A0=0 … C8=87)

Download:  python scripts/download_nsynth.py
"""

import json
from typing import List

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule


# ─── constants ────────────────────────────────────────────────────────────────
NUM_PITCH_CLASSES = 88          # MIDI pitches 21–108
MIDI_OFFSET       = 21          # class_idx = note - MIDI_OFFSET
NSYNTH_SR         = 16_000      # original sample rate of all NSynth WAVs
NSYNTH_DURATION   = 4.0         # all NSynth clips are exactly 4 seconds


class _NSynthAudioBase(Dataset):
    """
    Base NSynth dataset.  Each clip is exactly 4 s — no slicing needed.

    Args:
        jsonl        : path to train / val / test JSONL file.
        sample_rate  : target sample rate (resampling applied if ≠ 16 kHz).
        channels     : target channel count (1 = mono).
        clip_seconds : clip length in seconds (default 4.0).
        channel_mode : how to downmix stereo → mono ("first" | "mix").
    """

    #: Map  note (str) → class index.  Not used for lookup (note is an int),
    #  but kept for documentation / export.
    IDX2NOTE: dict[int, int] = {i: MIDI_OFFSET + i for i in range(NUM_PITCH_CLASSES)}
    NOTE2IDX: dict[int, int] = {v: k for k, v in IDX2NOTE.items()}

    EXAMPLE_JSONL = {
        "audio_path": "data/NSynth/nsynth-train/audio/bass_acoustic_000-021-075.wav",
        "note": 21,
        "velocity": 75,
        "instrument_family": "bass",
        "instrument_source": "acoustic",
        "sample_rate": 16000,
        "num_samples": 64000,
        "channels": 1,
        "duration": 4.0,
    }

    def __init__(
        self,
        jsonl: str,
        sample_rate: int = NSYNTH_SR,
        channels: int = 1,
        clip_seconds: float = NSYNTH_DURATION,
        channel_mode: str = "first",
        max_samples: int | None = None,
    ):
        """
        Args:
            max_samples : if set, randomly subsample this many entries from the
                          JSONL (stratified by pitch class).  Useful for capping
                          the large NSynth train split (~289 K clips) to speed
                          up the sweep without losing label coverage.
                          Default (None) uses the full split.
        """
        self.sample_rate      = int(sample_rate)
        self.channels         = channels
        self.channel_mode     = channel_mode
        self.clip_len_target  = int(clip_seconds * self.sample_rate)

        with open(jsonl) as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Optional stratified subsample ──────────────────────────────────────
        if max_samples is not None and max_samples < len(self.meta):
            import random
            # Group by pitch class
            from collections import defaultdict
            buckets: dict[int, List[dict]] = defaultdict(list)
            for entry in self.meta:
                buckets[int(entry["note"]) - MIDI_OFFSET].append(entry)
            # Per-class quota (floor); remainder goes to the largest classes
            per_class = max_samples // NUM_PITCH_CLASSES
            subsampled: List[dict] = []
            for cls_entries in buckets.values():
                random.shuffle(cls_entries)
                subsampled.extend(cls_entries[:per_class])
            # Fill remainder
            all_rest = [e for cls in buckets.values() for e in cls[per_class:]]
            random.shuffle(all_rest)
            subsampled.extend(all_rest[: max_samples - len(subsampled)])
            random.shuffle(subsampled)
            self.meta = subsampled
            print(f"NSynth: subsampled {len(self.meta)} / {max_samples} clips "
                  f"(requested cap={max_samples}).")

        # Pre-build resamplers for any non-target SR (NSynth is always 16 kHz,
        # but guard for edge cases or mixed sources).
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        waveform : Tensor  (channels, clip_len_target)
        label    : int     class index 0–87
        path     : str     audio_path (used as uid for test aggregation)
        """
        info    = self.meta[idx]
        path    = info["audio_path"]
        note    = int(info["note"])
        label   = note - MIDI_OFFSET      # 0–87
        orig_sr = int(info["sample_rate"])

        waveform, _ = torchaudio.load(path)   # (C, T)

        # ── channel handling ──────────────────────────────────────────────────
        C = waveform.size(0)
        if C >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(0, keepdim=True)
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[: self.channels]
        else:
            deficit = self.channels - C
            waveform = torch.cat([waveform, waveform[-1:].repeat(deficit, 1)], 0)

        # ── resample ──────────────────────────────────────────────────────────
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # ── pad / truncate ────────────────────────────────────────────────────
        T = waveform.size(1)
        if T < self.clip_len_target:
            waveform = F.pad(waveform, (0, self.clip_len_target - T))
        elif T > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform, label, path


class NSynthAudioTrain(_NSynthAudioBase):
    """Training split — DataModule sets shuffle=True."""
    pass


class NSynthAudioVal(_NSynthAudioBase):
    """Validation split — DataModule sets shuffle=False."""
    pass


class NSynthAudioTest(_NSynthAudioBase):
    """Test split — same logic as validation."""
    pass


class NSynthDataModule(BaseDataModule):
    """Thin wrapper: all logic lives in BaseDataModule."""
    pass
