# marble/tasks/VGMIDITVar/datamodule.py
"""
VGMIDI-TVar theme-and-variation retrieval datamodule (audio version).

Source
------
Variation Transformer dataset (Gao et al., ISMIR 2024).
https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model

Each piece in the dataset has one or more "theme + variations" groups.
Filenames encode the relationship:

    {piece_id}_{section}_0.mid      ← the theme
    {piece_id}_{section}_1.mid      ← variation 1 of that theme
    {piece_id}_{section}_2.mid      ← variation 2 of that theme
    ...

The audio version of this task synthesises every MIDI to audio (see
``scripts/build_vgmiditvar_dataset.py``) and uses the work-group identity
(``piece_id + section``) as the retrieval label — analogous to
``work_id`` in SHS-100K / Covers80.

Task
----
Zero-shot retrieval (no probe training).  Given the audio rendering of
every theme + variation, embed each track, mean-pool clips per file, and
report MAP — same scheme as SHS-100K.

JSONL format (one line per audio file):

    {
      "audio_path": "data/VGMIDITVar/audio/052_A_0.wav",
      "work_id":    52001,        ← integer encoding piece_id + section
      "variation":  0,             ← 0 = theme, ≥1 = variation index
      "piece_id":   "052",
      "section":    "A",
      "split":      "train" | "test",
      "sample_rate": 44100,
      "num_samples": 220500,
      "channels":   1,
      "duration":   5.0
    }

The JSONL is split-aware: the dataset ships with explicit train/test splits.
``VGMIDITVarAudioAll`` ignores the split field and evaluates over the union
(zero-shot retrieval doesn't use the train set for fitting); the split is
preserved so future supervised probes can use it.
"""

import json
from typing import List, Optional, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule


# Default clip length when splitting long renders into chunks.  Themes are
# usually short (≤30 s) so we use a smaller default than Covers80.
DEFAULT_CLIP_SECONDS  = 15.0
DEFAULT_MIN_CLIP_RATIO = 0.5


class _VGMIDITVarAudioBase(Dataset):
    """
    Base dataset for VGMIDI-TVar audio retrieval.

    Long renders are split into fixed-length non-overlapping clips.
    Each clip returns: ``(waveform, work_id, audio_path)`` — identical
    contract to ``_Covers80AudioBase`` / ``_SHS100KAudioBase`` so the
    ``CoverRetrievalTask`` from Covers80 can drive evaluation unchanged.

    Args
    ----
    jsonl          : path to JSONL metadata file (see module docstring).
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
        split: Optional[str] = None,
    ):
        self.sample_rate     = int(sample_rate)
        self.channels        = channels
        self.clip_seconds    = clip_seconds
        self.clip_len_target = int(clip_seconds * self.sample_rate)
        self.min_clip_ratio  = min_clip_ratio
        self.channel_mode    = channel_mode
        self.backend         = backend
        self.split           = split

        with open(jsonl, encoding="utf-8") as f:
            self.meta: List[dict] = [json.loads(line) for line in f]

        # Optionally filter to one split
        if split is not None:
            self.meta = [m for m in self.meta if m.get("split") == split]

        # Resamplers for any non-target sample rates in the dataset
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_sr, self.sample_rate
                )

        # Flat index: (file_idx, slice_idx, orig_sr, orig_clip_frames)
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
            n_slices = max(n_slices, 1)   # always keep at least 1 clip even if short
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


class VGMIDITVarAudioAll(_VGMIDITVarAudioBase):
    """All renders, used for zero-shot retrieval evaluation."""
    pass


class VGMIDITVarAudioTest(_VGMIDITVarAudioBase):
    """Test split only — filters JSONL by ``split == 'test'``."""

    def __init__(self, *args, **kwargs):
        kwargs["split"] = "test"
        super().__init__(*args, **kwargs)


class VGMIDITVarAudioDummy(_VGMIDITVarAudioBase):
    """
    Placeholder dataset for the train / val dataloaders so that
    LightningCLI doesn't need special-casing.  Points at the same JSONL
    as the test dataset; never iterated when ``max_epochs=0``.
    """
    pass


class VGMIDITVarDataModule(BaseDataModule):
    """Thin wrapper — all logic is in BaseDataModule."""
    pass
