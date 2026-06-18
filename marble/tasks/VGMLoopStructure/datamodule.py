# marble/tasks/VGMLoopStructure/datamodule.py

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id


class _VGMLoopStructureAudioBase(Dataset):
    """
    Base dataset for VGMLoopStructure audio:
    - Splits each audio file into non-overlapping clips of length `clip_seconds` (last clip zero-padded).
    - This is a 3-class loop-type classification task:
        through_composed / loop_from_start / intro_loop
    """

    LABEL2IDX = {
        "through_composed": 0,
        "loop_from_start": 1,
        "intro_loop": 2,
    }

    IDX2LABEL = {
        0: "through_composed",
        1: "loop_from_start",
        2: "intro_loop",
    }

    EXAMPLE_JSONL = {
        "audio_path": "data/VGM/audio/track_001.wav",
        "sample_rate": 24000,
        "num_samples": 480000,
        "channels": 1,
        "bit_depth": 16,
        "label": "loop_from_start",
        "duration": 20.0,
    }

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ["first", "mix", "random"]:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio

        # Cross-OS JSONL load (Windows backslash audio_paths → POSIX).
        # See marble/utils/path_compat.py.
        from marble.utils.path_compat import load_jsonl

        self.meta = load_jsonl(jsonl)

        # Map label: accept either a string or a list (take the first element).
        for info in self.meta:
            lbl = info["label"]
            if isinstance(lbl, list):
                lbl = lbl[0]
            if lbl not in self.LABEL2IDX:
                raise ValueError(f"Unknown label: {lbl!r}")
            info["label"] = lbl

        # Build index map: (file_idx, slice_idx, orig_sr, orig_clip_frames, orig_channels)
        self.index_map: list[tuple[int, int, int, int, int]] = []
        self.resamplers = {}
        # Set by the task at setup() time when the per-clip embedding cache
        # is active. See marble.utils.emb_cache.EmbeddingCacheMixin.
        self.cache_check_fn = None
        for file_idx, info in enumerate(self.meta):
            orig_sr = info["sample_rate"]
            # Prepare resampler if needed
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

            orig_clip_frames = int(self.clip_seconds * orig_sr)
            orig_channels = info["channels"]
            total_samples = info["num_samples"]

            # Number of full clips and remainder
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            # Decide whether to keep the last shorter clip
            if rem / orig_clip_frames >= self.min_clip_ratio:
                n_slices = n_full + 1
            else:
                n_slices = n_full

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
            label: int
            path: str
            clip_id: str
        """
        # Unpack mapping info
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        label = self.LABEL2IDX[info["label"]]
        clip_id = make_clip_id(path, slice_idx)

        # Cache hit — skip audio I/O entirely. The task's forward() ignores
        # `x` on cache hits and uses the cached (L, H) tensor instead.
        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            waveform = torch.zeros(self.channels, self.clip_len_target)
            return waveform, label, path, clip_id

        # Compute frame offset and load clip
        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path, frame_offset=offset, num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # Defensive: if the JSONL's num_samples is stale (file got
        # re-encoded / truncated since metadata was probed) the load may
        # return a (orig_channels, 0) tensor. Fall back to silent audio so the
        # batch still flows through.
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
            # Repeat last channel to pad to desired channels
            last = waveform[-1:].repeat(self.channels - orig_channels, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad to target length if short
        if waveform.size(1) < self.clip_len_target:
            pad = self.clip_len_target - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

        # Final shape: (self.channels, self.clip_len_target)
        return waveform, label, path, clip_id


class VGMLoopStructureAudioTrain(_VGMLoopStructureAudioBase):
    """Training split; DataModule sets shuffle=True."""

    pass


class VGMLoopStructureAudioVal(_VGMLoopStructureAudioBase):
    """Validation split; DataModule sets shuffle=False."""

    pass


class VGMLoopStructureAudioTest(VGMLoopStructureAudioVal):
    """Test split; same logic as val."""

    pass


class VGMLoopStructureDataModule(BaseDataModule):
    pass
