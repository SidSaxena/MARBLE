# marble/tasks/VGMLoopFrames/datamodule.py
#
# Frame-level VGM loop-structure dataset.
#
# JSONL schema per row:
#   {
#     "audio_path": str,
#     "sample_rate": 24000,
#     "num_samples": int,
#     "channels": 1,
#     "label": {
#       "intro_end_sec":  float | null,
#       "loop_seam_sec":  float | null,
#       "loop_type":      str,      # "intro_loop" | "loop_from_start" | "through_composed"
#       "total_sec":      float
#     }
#   }
#
# __getitem__ returns a 3-tuple (waveform, targets, path) — NO clip_id — so
# the datamodule is cache-unsafe (frame tasks keep the time axis anyway).
#
# targets is a dict:
#   "boundary"  : (L,) float32  Gaussian heatmap at intro_end and loop_seam
#   "function"  : (L,) int64    per-frame class  0=intro  1=loop  2=through


import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule

# ── label constants ────────────────────────────────────────────────────────────

LOOP_TYPE_FUNCTION = {
    # through_composed → all frames = 2
    "through_composed": 2,
    # loop_from_start  → all frames = 1 (loop, no intro)
    "loop_from_start": 1,
    # intro_loop       → transitions from 0 (intro) to 1 (loop) at intro_end_sec
    "intro_loop": None,
}

# ── Gaussian bump helper ───────────────────────────────────────────────────────


def _gaussian_bump(heatmap: np.ndarray, frame_idx: int, sigma: float) -> None:
    """Add a unit-peak Gaussian centred at *frame_idx* into *heatmap* (in-place)."""
    L = len(heatmap)
    frames = np.arange(L, dtype=np.float32)
    heatmap += np.exp(-0.5 * ((frames - frame_idx) / sigma) ** 2)


# ── base dataset ──────────────────────────────────────────────────────────────


class _VGMLoopFramesAudioBase(Dataset):
    """
    Base dataset for VGMLoopFrames frame-level tasks.

    For each clip the __getitem__ returns::

        waveform  : (channels, clip_len_target)  float32
        targets   : {
            "boundary" : (L,)  float32  — Gaussian heatmap of structure boundaries
            "function" : (L,)  int64    — per-frame class (0=intro,1=loop,2=through)
        }
        path      : str

    No ``clip_id`` is emitted so the base-task embedding cache cannot be
    accidentally activated for this module.

    Parameters
    ----------
    sample_rate : int
    channels : int
    clip_seconds : float
    jsonl : str
        Path to the JSONL manifest file.
    label_freq : int
        Frame rate for the target tensors (default 25 Hz).
    boundary_sigma : float
        Gaussian σ in frames for the boundary heatmap (default 1.5).
    channel_mode : str
        One of ``"first"``, ``"mix"``, ``"random"``.
    min_clip_ratio : float
        Minimum fraction of a full clip required to keep the last partial clip.
    """

    EXAMPLE_JSONL = {
        "audio_path": "data/VGM/audio/track_001.wav",
        "sample_rate": 24000,
        "num_samples": 480000,
        "channels": 1,
        "label": {
            "intro_end_sec": 12.0,
            "loop_seam_sec": None,
            "loop_type": "intro_loop",
            "total_sec": 120.0,
        },
    }

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        clip_seconds: float,
        jsonl: str,
        label_freq: int = 25,
        boundary_sigma: float = 1.5,
        channel_mode: str = "first",
        min_clip_ratio: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.channel_mode = channel_mode
        if channel_mode not in ("first", "mix", "random"):
            raise ValueError(f"Unknown channel_mode: {channel_mode!r}")
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.label_freq = label_freq
        self.boundary_sigma = boundary_sigma
        self.min_clip_ratio = min_clip_ratio

        from marble.utils.path_compat import load_jsonl

        self.meta: list[dict] = load_jsonl(jsonl)

        # Validate loop_type values
        for info in self.meta:
            lt = info["label"]["loop_type"]
            if lt not in LOOP_TYPE_FUNCTION:
                raise ValueError(f"Unknown loop_type: {lt!r}")

        # Build resamplers for any source SR ≠ target SR
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # index_map: (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: list[tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(self.clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue

            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            n_slices = n_full + (1 if rem / orig_clip_frames >= self.min_clip_ratio else 0)
            for slice_idx in range(n_slices):
                self.index_map.append((file_idx, slice_idx, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        waveform : Tensor (channels, clip_len_target)
        targets  : dict
            ``"boundary"`` (L,) float32 — Gaussian heatmap
            ``"function"`` (L,) int64   — per-frame class index
        path : str
        """
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        label = info["label"]

        # ── 1. load & preprocess waveform ──────────────────────────────────────
        waveform = self._load_and_preprocess(
            path=path,
            slice_idx=slice_idx,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames,
        )

        # ── 2. clip window in seconds (based on original SR) ───────────────────
        clip_start = slice_idx * (orig_clip_frames / orig_sr)
        clip_end = clip_start + self.clip_seconds
        label_len = int(self.label_freq * self.clip_seconds)

        # ── 3. boundary heatmap ───────────────────────────────────────────────
        boundary = np.zeros(label_len, dtype=np.float32)
        for key in ("intro_end_sec", "loop_seam_sec"):
            t = label.get(key)
            if t is None:
                continue
            t = float(t)
            if clip_start <= t < clip_end:
                frame_idx = int(round((t - clip_start) * self.label_freq))
                frame_idx = max(0, min(frame_idx, label_len - 1))
                _gaussian_bump(boundary, frame_idx, self.boundary_sigma)
        # Clamp to [0, 1] in case two bumps overlap
        np.clip(boundary, 0.0, 1.0, out=boundary)
        boundary_tensor = torch.from_numpy(boundary)

        # ── 4. per-frame function class ──────────────────────────────────────
        loop_type = label["loop_type"]
        function = np.zeros(label_len, dtype=np.int64)

        if loop_type == "through_composed":
            function[:] = 2

        elif loop_type == "loop_from_start":
            function[:] = 1

        elif loop_type == "intro_loop":
            intro_end = label.get("intro_end_sec")
            if intro_end is None:
                # Degenerate: treat as all loop (shouldn't happen in clean data)
                function[:] = 1
            else:
                intro_end = float(intro_end)
                # Frame index of the boundary within the clip
                # (relative to clip_start)
                for fi in range(label_len):
                    t_frame = clip_start + fi / self.label_freq
                    function[fi] = 0 if t_frame < intro_end else 1

        function_tensor = torch.from_numpy(function)

        targets = {
            "boundary": boundary_tensor,  # (L,) float32
            "function": function_tensor,  # (L,) int64
        }

        return waveform, targets, path

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_and_preprocess(
        self, path: str, slice_idx: int, orig_sr: int, orig_clip_frames: int
    ) -> torch.Tensor:
        """Load one clip slice, channel-align, resample, pad/truncate."""
        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(path, frame_offset=offset, num_frames=orig_clip_frames)

        if waveform.size(1) == 0:
            waveform = torch.zeros(waveform.size(0), orig_clip_frames, dtype=waveform.dtype)

        orig_ch = waveform.size(0)

        # Channel alignment
        if orig_ch >= self.channels:
            if self.channels == 1:
                if self.channel_mode == "first":
                    waveform = waveform[0:1]
                elif self.channel_mode == "mix":
                    waveform = waveform.mean(dim=0, keepdim=True)
                elif self.channel_mode == "random":
                    choice = torch.randint(0, orig_ch + 1, (1,)).item()
                    if choice == orig_ch:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    else:
                        waveform = waveform[choice : choice + 1]
            else:
                waveform = waveform[: self.channels]
        else:
            last = waveform[-1:].repeat(self.channels - orig_ch, 1)
            waveform = torch.cat([waveform, last], dim=0)

        # Resample
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # Pad / truncate
        cur_len = waveform.size(1)
        if cur_len < self.clip_len_target:
            waveform = F.pad(waveform, (0, self.clip_len_target - cur_len))
        elif cur_len > self.clip_len_target:
            waveform = waveform[:, : self.clip_len_target]

        return waveform


# ── split classes ─────────────────────────────────────────────────────────────


class VGMLoopFramesAudioTrain(_VGMLoopFramesAudioBase):
    """Training split; DataModule sets shuffle=True."""

    pass


class VGMLoopFramesAudioVal(_VGMLoopFramesAudioBase):
    """Validation split; no shuffling."""

    pass


class VGMLoopFramesAudioTest(VGMLoopFramesAudioVal):
    """Test split; same logic as val."""

    pass


class VGMLoopFramesDataModule(BaseDataModule):
    pass
