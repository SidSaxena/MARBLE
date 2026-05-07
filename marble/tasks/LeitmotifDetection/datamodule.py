# marble/tasks/LeitmotifDetection/datamodule.py
"""
Flexible audio classification dataset for leitmotif / theme detection.

JSONL format (one JSON object per line):
    {
        "audio_path": "/path/to/clip.wav",
        "label":      "theme_name",
        "sample_rate": 44100,
        "num_samples": 220500
    }

String labels are resolved via a ``labels: List[str]`` parameter supplied
through the YAML config, so this module works for any label vocabulary.

Audio clips are split into non-overlapping windows of ``clip_seconds``
seconds (last window is zero-padded to full length).
"""

import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule


class _LeitmotifAudioBase(Dataset):
    """
    Base dataset for leitmotif detection audio.

    Splits each audio file into non-overlapping clips of ``clip_seconds``
    seconds.  The last (potentially shorter) clip is kept when its length
    is at least ``min_clip_ratio * clip_seconds``.

    Parameters
    ----------
    sample_rate : int
        Target sampling rate.  Audio is resampled on-the-fly if the file's
        native rate differs.
    channels : int
        Number of output channels (1 = mono).
    clip_seconds : float
        Duration of each output clip in seconds.
    jsonl : str
        Path to the JSONL metadata file.
    labels : List[str]
        Ordered list of label strings.  ``labels[i]`` maps to class index *i*.
    channel_mode : str
        One of ``"first"``, ``"mix"``, or ``"random"``.
    min_clip_ratio : float
        Minimum fraction of ``clip_seconds`` required to keep the last clip.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        labels: List[str],
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(clip_seconds * sample_rate)
        self.channel_mode = channel_mode
        self.min_clip_ratio = min_clip_ratio

        if channel_mode not in ("first", "mix", "random"):
            raise ValueError(f"Unknown channel_mode: {channel_mode!r}")

        # Build label → index mapping from the ordered list
        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

        # Load metadata
        with open(jsonl, "r") as fh:
            self.meta: List[dict] = [json.loads(line) for line in fh]

        # Build a flat index_map over all (file, slice) pairs
        self.index_map: List[Tuple[int, int, int, int, int]] = []
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        for file_idx, info in enumerate(self.meta):
            orig_sr: int = info["sample_rate"]
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

            orig_clip_frames = int(clip_seconds * orig_sr)
            orig_channels: int = info.get("channels", 1)
            total_samples: int = info["num_samples"]

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames

            n_slices = n_full + (
                1 if rem > 0 and rem / orig_clip_frames >= min_clip_ratio else 0
            )

            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns
        -------
        waveform : torch.Tensor, shape (channels, clip_len_target)
        label    : int
        path     : str
        """
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path: str = info["audio_path"]
        label: int = self.label2idx[info["label"]]

        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path, frame_offset=offset, num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # ---- Channel handling ----
        if orig_channels >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    choice = torch.randint(0, orig_channels + 1, (1,)).item()
                    if choice == orig_channels:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice : choice + 1]
            else:
                waveform = waveform[: self.channels]
        else:
            # Repeat last channel to pad
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # ---- Resample if needed ----
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # ---- Zero-pad short tail clip ----
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

        return waveform, label, path


class LeitmotifAudioTrain(_LeitmotifAudioBase):
    """Training split — DataModule sets ``shuffle=True``."""
    pass


class LeitmotifAudioVal(_LeitmotifAudioBase):
    """Validation split — DataModule sets ``shuffle=False``."""
    pass


class LeitmotifAudioTest(_LeitmotifAudioBase):
    """Test split — same logic as validation."""
    pass


class LeitmotifDetectionDataModule(BaseDataModule):
    """
    LightningDataModule for the LeitmotifDetection task.

    Inherits all boilerplate (DataLoader creation, transforms, etc.) from
    ``marble.core.base_datamodule.BaseDataModule``.  No additional logic
    is needed here — the datasets above already handle everything.
    """
    pass
