# marble/tasks/SHS100K/datamodule.py
"""
SHS-100K cover-song retrieval datamodule.

SHS-100K (Second Hand Songs, 2025 edition) contains 10,000 musical works
with ~110,000 total cover performances split into train / val / test.
The test split (500 works, ~5,000 tracks) is the community-standard
evaluation benchmark for cover song retrieval.

Evaluation is zero-shot retrieval — no probe training, just:
  embed all test tracks → mean-pool clips per file → cosine-similarity MAP.

JSONL format (one line per track):
  {
    "audio_path":     "data/SHS100K/audio/dQw4w9WgXcQ.mp3",
    "work_id":        8855,          # Second Hand Songs work ID (groups covers)
    "performance_id": 8855,
    "title":          "1999",
    "artist":         "Prince",
    "youtube_id":     "dQw4w9WgXcQ",
    "sample_rate":    44100,
    "num_samples":    9234567,
    "channels":       2,
    "duration":       209.4
  }

Download:  python scripts/download_shs100k.py
"""

import json
import random
from typing import List, Optional, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule


DEFAULT_CLIP_SECONDS  = 30.0
DEFAULT_MIN_CLIP_RATIO = 0.5


class _SHS100KAudioBase(Dataset):
    """
    Base dataset for SHS-100K cover-song retrieval.

    Long tracks are split into fixed-length non-overlapping clips.
    Each item returns:  (waveform, work_id, audio_path)

    ``work_id`` is the Second Hand Songs work ID — it groups all cover
    versions of the same musical work and is used to compute MAP.

    Args:
        jsonl          : path to JSONL metadata file.
        sample_rate    : target sample rate for the downstream encoder.
        channels       : output channels (1 = mono).
        clip_seconds   : clip length in seconds.
        min_clip_ratio : minimum tail fraction to keep as a clip.
        channel_mode   : "first" | "mix" | "random".
    """

    def __init__(
        self,
        jsonl: str,
        sample_rate: int,
        channels: int = 1,
        clip_seconds: float = DEFAULT_CLIP_SECONDS,
        min_clip_ratio: float = DEFAULT_MIN_CLIP_RATIO,
        channel_mode: str = "mix",
        backend: Optional[str] = None,
    ):
        self.sample_rate     = int(sample_rate)
        self.channels        = channels
        self.clip_seconds    = clip_seconds
        self.clip_len_target = int(clip_seconds * self.sample_rate)
        self.min_clip_ratio  = min_clip_ratio
        self.channel_mode    = channel_mode
        self.backend         = backend

        with open(jsonl, encoding="utf-8") as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Build resamplers for any non-target sample rates in the dataset
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

        # index_map: list of (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr          = int(info["sample_rate"])
            total_samples    = int(info["num_samples"])
            orig_clip_frames = int(clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue
            n_full   = total_samples // orig_clip_frames
            rem      = total_samples - n_full * orig_clip_frames
            n_slices = n_full + (1 if rem / orig_clip_frames >= min_clip_ratio else 0)
            for s in range(n_slices):
                self.index_map.append((file_idx, s, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info    = self.meta[file_idx]
        path    = info["audio_path"]
        work_id = int(info["work_id"])

        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames,
            backend=self.backend,
        )  # (C, T)

        # ── channel handling ──────────────────────────────────────────────────
        C = waveform.size(0)
        if C >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(0, keepdim=True)
                elif self.channel_mode == "random":
                    if torch.rand(()) < 0.5:
                        waveform = waveform.mean(0, keepdim=True)
                    else:
                        ch = torch.randint(0, C, ()).item()
                        waveform = waveform[ch : ch + 1]
                else:
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[: self.channels]
        else:
            deficit  = self.channels - C
            waveform = torch.cat([waveform, waveform[-1:].repeat(deficit, 1)], 0)

        # ── resample ──────────────────────────────────────────────────────────
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # ── pad / truncate to exact clip length ───────────────────────────────
        T = waveform.size(1)
        if T < self.clip_len_target:
            waveform = F.pad(waveform, (0, self.clip_len_target - T))
        elif T > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform, work_id, path


class SHS100KAudioAll(_SHS100KAudioBase):
    """Full split — used for zero-shot retrieval evaluation."""
    pass


class SHS100KAudioDummy(_SHS100KAudioBase):
    """
    Placeholder dataset for the train / val dataloaders.
    Points at the test JSONL so LightningCLI setup doesn't require separate
    files; these loaders are never iterated (max_epochs=0).
    """
    pass


class SHS100KDataModule(BaseDataModule):
    """Thin wrapper — all logic is in BaseDataModule."""
    pass
