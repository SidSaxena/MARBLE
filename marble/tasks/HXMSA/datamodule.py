# marble/tasks/HXMSA/datamodule.py
"""
HXMSA datamodule — Harmonix Set Music Structure Analysis.

Adapted from marble/tasks/HookTheoryStructure/datamodule.py with three
substantive changes:

  1. LABEL2IDX uses the 13-class Harmonix canonical inventory (vs the
     7-class HookTheory one). See scripts/data/build_hxmsa_dataset.py
     for the raw→canonical normalisation map and the rationale.
  2. ``__getitem__`` returns the per-segment ``ori_uid`` (the
     ``{file_id}_{seg_idx:03d}`` string set by the build script) as the
     third element rather than the raw audio_path. The probe's per-uid
     aggregation then averages slices within a segment, never across
     segments.
  3. Class names: ``_HXMSAAudioBase`` / ``HXMSAAudioTrain`` etc.

Everything else — 4-tuple emit with clip_id for cache integration,
defensive getattr(self, "cache_check_fn", None) per the standardised
Windows-spawn-pickle pattern, clip slicing via index_map, channel-mode
handling, on-the-fly resampling, zero-pad for short clips — is
verbatim from HookTheoryStructure.

The 13-class inventory must match build_hxmsa_dataset.py CANONICAL_LABELS
exactly. Order matters — index 0 is the first alphabetical class.
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id


class _HXMSAAudioBase(Dataset):
    """
    Base dataset for HXMSA audio:
    - Each JSONL record is one pre-extracted segment FLAC (8–32 s typical).
    - Splits each segment into non-overlapping ``clip_seconds`` windows;
      last window zero-padded.
    - 13-class multi-class classification (functional segments of pop tracks).

    The LABEL2IDX inventory must stay in sync with
    ``scripts/data/build_hxmsa_dataset.py:CANONICAL_LABELS`` — order matters
    because the index = the class id used by the loss / metrics.
    """

    # ── 13-class canonical Harmonix functional-segment inventory ─────────────
    # Sorted alphabetically (matches build_hxmsa_dataset.py CANONICAL_LABELS).
    # 13 classes after dropping the "end" terminator and merging
    # "instrumental" → "inst" per the paper's own §3.3 note.
    LABEL2IDX = {
        "break": 0,
        "bridge": 1,
        "chorus": 2,
        "inst": 3,
        "intro": 4,
        "other": 5,
        "outro": 6,
        "postchorus": 7,
        "prechorus": 8,
        "silence": 9,
        "solo": 10,
        "transition": 11,
        "verse": 12,
        # Defensive aliases — accept hyphenated forms in case a hand-edited
        # JSONL slips through. The build script normalises these already,
        # so under normal conditions these are never hit.
        "pre-chorus": 8,
        "post-chorus": 7,
        "instrumental": 3,
    }

    IDX2LABEL = {
        0: "break",
        1: "bridge",
        2: "chorus",
        3: "inst",
        4: "intro",
        5: "other",
        6: "outro",
        7: "postchorus",
        8: "prechorus",
        9: "silence",
        10: "solo",
        11: "transition",
        12: "verse",
    }

    NUM_CLASSES = 13

    EXAMPLE_JSONL = {
        "audio_path": "data/HXMSA/segments/0001_12step/000_intro.flac",
        "ori_uid": "0001_12step_000",  # per-segment uid for probe aggregation
        "work_id": "0001_12step",  # track-level grouping
        "label": "intro",  # canonical label string (mapped to int via LABEL2IDX)
        "seg_idx": 0,
        "seg_start": 0.0,
        "seg_end": 8.5,
        "duration": 8.5,
        "sample_rate": 24000,
        "num_samples": 204000,
        "channels": 1,
        "bit_depth": 16,
        "title": "1, 2 Step",
        "artist": "Ciara",
        "genre": "R&B",
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
        # Cross-OS JSONL load (Windows backslash audio_paths → POSIX).
        # See marble/utils/path_compat.py.
        from marble.utils.path_compat import load_jsonl

        self.meta = load_jsonl(jsonl)

        # Validate labels up-front (fail-loud)
        for info in self.meta:
            lbl = info["label"]
            if isinstance(lbl, list):
                # Defensive: in case future variants emit multi-label lists,
                # take the first canonical hit.
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
            # Guarantee at least one slice per segment (some segments may be
            # shorter than clip_seconds — those get zero-padded to clip_len_target).
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

        Inputs:
            idx: int - index of the clip

        Returns:
            waveform: torch.Tensor, shape (self.channels, self.clip_len_target)
            label:    int
            ori_uid:  str  — per-segment id; the probe aggregates slices per uid
            clip_id:  str  — per-slice cache key (encoder is independent of uid)
        """
        # Unpack mapping info
        file_idx, slice_idx, orig_sr, orig_clip, orig_channels = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        ori_uid = info["ori_uid"]
        label = self.LABEL2IDX[info["label"]]
        clip_id = make_clip_id(path, slice_idx)

        # Cache hit — skip audio I/O entirely. The task's forward() ignores
        # `x` on cache hits and uses the cached (L, H) tensor instead.
        # Defensive getattr against the Windows-spawn pickle/state-mismatch
        # bug fixed in commit 23f8e36 — falls back to "no bypass" cleanly
        # rather than AttributeError when a worker unpickles a Dataset whose
        # __init__ predates the cache integration.
        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            waveform = torch.zeros(self.channels, self.clip_len_target)
            return waveform, label, ori_uid, clip_id

        # Compute frame offset and load clip
        offset = slice_idx * orig_clip
        waveform, _ = torchaudio.load(
            path, frame_offset=offset, num_frames=orig_clip
        )  # (orig_channels, orig_clip)

        # Defensive: if the JSONL's num_samples is stale (file got re-encoded
        # / truncated since metadata was probed) the load may return a
        # (orig_channels, 0) tensor. Downstream resample then blows up on
        # the empty dim with "cannot reshape tensor of 0 elements into
        # shape [-1, 0]". Fall back to silent audio so the batch still
        # flows through.
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
        return waveform, label, ori_uid, clip_id


class HXMSAAudioTrain(_HXMSAAudioBase):
    """Training split — DataModule sets shuffle=True."""

    pass


class HXMSAAudioVal(_HXMSAAudioBase):
    """Validation split — DataModule sets shuffle=False."""

    pass


class HXMSAAudioTest(HXMSAAudioVal):
    """Test split — same logic as val."""

    pass


class HXMSADataModule(BaseDataModule):
    pass
