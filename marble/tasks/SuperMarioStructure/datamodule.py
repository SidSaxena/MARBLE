# marble/tasks/SuperMarioStructure/datamodule.py
"""
SuperMarioStructure datamodule — Super Mario VGM functional structure.

Adapted from marble/tasks/HXMSA/datamodule.py with two substantive
changes:

  1. LABEL2IDX uses the 6-class SuperMario inventory (VGM-native
     labels including `loop` and `stinger`, which don't exist in
     pop-music structure datasets like HXMSA / HookTheoryStructure).
  2. Class names: ``_SuperMarioStructureAudioBase`` /
     ``SuperMarioStructureAudioTrain`` etc.

Everything else is verbatim from HXMSA / HookTheoryStructure: 4-tuple
emit with clip_id for cache integration, defensive
getattr(self, "cache_check_fn", None) per the standardised
Windows-spawn-pickle pattern, clip slicing via index_map, channel-mode
handling, on-the-fly resampling, zero-pad for short clips. The
per-segment ``ori_uid`` returned as the third tuple element drives
the probe's segment-level aggregation.

The 6-class inventory must match
scripts/data/build_supermario_dataset.py CANONICAL_LABELS exactly.
Order matters — index 0 is the first alphabetical class.
"""

import json

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id


class _SuperMarioStructureAudioBase(Dataset):
    """
    Base dataset for SuperMarioStructure audio:
    - Each JSONL record is one pre-extracted segment FLAC (typically
      8–60 s, sliced via ffmpeg from user audio using MIDI-derived
      bar→time mapping).
    - Splits each segment into non-overlapping ``clip_seconds`` windows;
      last window zero-padded. Same logic as HXMSA / HookTheoryStructure.
    - 6-class multi-class classification with VGM-native labels.

    The LABEL2IDX inventory must stay in sync with
    ``scripts/data/build_supermario_dataset.py:CANONICAL_LABELS`` —
    order matters because the index = the class id used by the loss /
    metrics.
    """

    # ── 6-class canonical SuperMario functional-segment inventory ─────────
    # Sorted alphabetically (matches build script CANONICAL_LABELS).
    # Labels derived from the upstream JSON's 2-letter codes via
    # RAW_TO_CANONICAL in the build script.
    LABEL2IDX = {
        "bridge": 0,  # raw "Br" — contrasting middle section
        "intro": 1,  # raw "In" — opening, not part of main loop
        "loop": 2,  # raw "Lp" — main repeating section (VGM-native)
        "outro": 3,  # raw "Ou" — closing
        "stinger": 4,  # raw "St" — short punctuation cue (VGM-native)
        "transition": 5,  # raw "Tr" — connecting passage
    }

    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}
    NUM_CLASSES = 6

    EXAMPLE_JSONL = {
        "audio_path": "data/SuperMarioStructure/segments/00001/000_intro.flac",
        "ori_uid": "00001_000",  # per-segment uid for probe aggregation
        "work_id": "00001",  # piece-level grouping
        "label": "intro",
        "seg_idx": 0,
        "bar_start": 1,
        "bar_end": 10,
        "seg_start": 0.0,
        "seg_end": 18.75,
        "duration": 18.75,
        "sample_rate": 24000,
        "num_samples": 450000,
        "channels": 1,
        "bit_depth": 16,
        "title": "Captain Toad Treasure Tracker - Retro RampUp",
        "ninsheetmusic_id": "4405",
    }

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio

        # Read metadata
        with open(jsonl) as f:
            self.meta = [json.loads(line) for line in f]

        # Validate labels up-front (fail-loud)
        for info in self.meta:
            lbl = info["label"]
            if isinstance(lbl, list):
                lbl = lbl[0] if lbl else None
                info["label"] = lbl
            if lbl not in self.LABEL2IDX:
                raise ValueError(
                    f"Unknown label {lbl!r} for {info.get('audio_path', '?')}; "
                    f"valid labels: {sorted(set(self.LABEL2IDX))}"
                )

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: list[tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        # Set by the task at setup() time when the per-clip embedding cache
        # is active. See marble.utils.emb_cache.EmbeddingCacheMixin.
        self.cache_check_fn = None
        for file_idx, info in enumerate(self.meta):
            orig_sr = info["sample_rate"]
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info["channels"]
            total_samples = info["num_samples"]

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full
            # Guarantee at least one slice per segment — VGM stingers can be
            # very short (1-2 bars at fast tempo) and would otherwise be
            # silently dropped.
            if n_slices == 0:
                n_slices = 1

            for slice_idx in range(n_slices):
                self.index_map.append(
                    (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
                )

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Load and return one audio clip and its label.

        Returns:
            waveform: torch.Tensor, shape (self.channels, self.clip_len_target)
            label:    int
            ori_uid:  str  — per-segment id; the probe aggregates slices per uid
            clip_id:  str  — per-slice cache key
        """
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        ori_uid = info["ori_uid"]
        label = self.LABEL2IDX[info["label"]]
        clip_id = make_clip_id(path, slice_idx)

        # Cache hit — skip audio I/O entirely. Defensive getattr against
        # Windows-spawn pickle/state-mismatch (commit 23f8e36 standardisation).
        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            waveform = torch.zeros(self.channels, self.clip_len_target)
            return waveform, label, ori_uid, clip_id

        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path, frame_offset=offset, num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # Defensive: handle 0-sample loads from stale num_samples
        if waveform.size(1) == 0:
            waveform = torch.zeros(orig_channels, orig_clip, dtype=waveform.dtype)

        # Channel alignment / downmixing
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
                    raise ValueError(f"Unknown channel_mode: {self.channel_mode}")
            else:
                waveform = waveform[: self.channels]
        else:
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

        return waveform, label, ori_uid, clip_id


class SuperMarioStructureAudioTrain(_SuperMarioStructureAudioBase):
    """Training split — DataModule sets shuffle=True."""

    pass


class SuperMarioStructureAudioVal(_SuperMarioStructureAudioBase):
    """Validation split — DataModule sets shuffle=False."""

    pass


class SuperMarioStructureAudioTest(SuperMarioStructureAudioVal):
    """Test split — same logic as val."""

    pass


class SuperMarioStructureDataModule(BaseDataModule):
    pass
