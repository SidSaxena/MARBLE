# marble/tasks/Covers80/datamodule.py
"""
Covers80 cover-song retrieval datamodule.

Covers80 has 80 musical works, each recorded by 2 different artists
("original" in list1, "cover" in list2).  There is no train/val split —
evaluation is purely retrieval: embed all 160 tracks, rank by cosine
similarity, report MAP.

JSONL format (one line per track):
  {
    "audio_path": "data/Covers80/covers32k/list1/song_name/version.mp3",
    "work_id":    42,     # integer 0–79 (same for both versions of a work)
    "version":    0,      # 0 = list1 (original), 1 = list2 (cover)
    "sample_rate": 32000,
    "num_samples": 9600000,
    "channels":   2,
    "duration":   300.0
  }

Download:  python scripts/data/download_covers80.py
"""

import json
import random
from typing import List, Optional, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule


# Maximum clip length in seconds when splitting long songs into chunks.
# 30 s gives a good balance between context and batch efficiency.
DEFAULT_CLIP_SECONDS = 30.0
DEFAULT_MIN_CLIP_RATIO = 0.5


class _Covers80AudioBase(Dataset):
    """
    Base dataset for Covers80.

    Long tracks are split into fixed-length clips (non-overlapping).
    Each clip returns:  (waveform, work_id, audio_path)

    ``work_id`` is used at evaluation time to identify positive pairs.

    Args:
        jsonl          : path to JSONL metadata file.
        sample_rate    : target sample rate for the downstream encoder.
        channels       : number of output channels (1 = mono).
        clip_seconds   : clip duration in seconds.
        min_clip_ratio : fraction of clip_seconds needed to keep a short tail.
        channel_mode   : "first" | "mix" | "random".
    """

    EXAMPLE_JSONL = {
        "audio_path": "data/Covers80/covers32k/list1/billie_jean/original.mp3",
        "work_id":    0,
        "version":    0,
        "sample_rate": 32000,
        "num_samples": 9600000,
        "channels":   1,
        "duration":   300.0,
    }

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

        with open(jsonl) as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Build resamplers
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: List[Tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr          = int(info["sample_rate"])
            total_samples    = int(info["num_samples"])
            orig_clip_frames = int(clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem    = total_samples - n_full * orig_clip_frames
            n_slices = n_full + (1 if rem / orig_clip_frames >= min_clip_ratio else 0)

            for s in range(n_slices):
                self.index_map.append((file_idx, s, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        waveform : Tensor  (channels, clip_len_target)
        work_id  : int
        path     : str   audio_path (used as uid during retrieval aggregation)
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info    = self.meta[file_idx]
        path    = info["audio_path"]
        work_id = int(info["work_id"])

        offset   = slice_idx * orig_clip_frames
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

        # ── pad / truncate ────────────────────────────────────────────────────
        T = waveform.size(1)
        if T < self.clip_len_target:
            waveform = F.pad(waveform, (0, self.clip_len_target - T))
        elif T > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform, work_id, path


class Covers80AudioAll(_Covers80AudioBase):
    """Full dataset (all 160 tracks) — used for test/retrieval evaluation."""
    pass


class Covers80AudioDummy(_Covers80AudioBase):
    """
    Dummy split pointing to the same data as Covers80AudioAll.
    Used as the 'train' and 'val' dataloaders when max_epochs=0 so that
    LightningCLI / DataModule setup doesn't need special-casing.
    """
    pass


class Covers80DataModule(BaseDataModule):
    """Thin wrapper — all logic in BaseDataModule."""
    pass
