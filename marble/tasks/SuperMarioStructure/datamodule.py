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
        "bridge": 0,  # raw "Br" — distinct transition section between Lp/Ln
        "intro": 1,  # raw "In" — opening section
        "linear": 2,  # raw "Ln" — main through-composed body (NOT looped)
        "loop": 3,  # raw "Lp" — main repeating section (VGM-native)
        "outro": 4,  # raw "Ou" — closing section
        "stinger": 5,  # raw "St" — short event-triggered cue (VGM-native)
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


# ────────────────────────────────────────────────────────────────────────────
# Symbolic path (CLaMP3 native MIDI encoder) — PRIMARY representation
# ────────────────────────────────────────────────────────────────────────────
#
# The SuperMario annotations are *bar-based*, derived from the source MUS/MXL
# scores. MIDI is therefore the exact-match input domain (audio is a
# derivation). For classification we slice the source MIDI per segment via
# scripts/data/build_supermario_dataset.py (using pretty_midi), then feed
# the per-segment MIDI fragments through CLaMP3's M3 tokeniser.
#
# Contract per item — 4-tuple matching the audio path:
#   patches : LongTensor (MAX_PATCHES, PATCH_SIZE) — padded with pad_token_id
#   label   : int — 6-class canonical inventory (shares LABEL2IDX with audio)
#   ori_uid : str — per-segment id; probe averages slices per uid
#   clip_id : str — per-segment cache key (only one "slice" per MIDI segment;
#                   the patchilizer is deterministic so caching the encoder
#                   output gives the full speedup)
#
# Cache integration mirrors the audio path's BaseTask flow — same
# EmbeddingCacheMixin handles hit/miss routing. M3 tokenisation is cheap
# (~10 ms / segment) but the CLaMP3 BERT forward pass is the expensive
# step; caching its (L, H) output is the speedup.


class _SuperMarioStructureSymbolicBase(Dataset):
    """Symbolic dataset: returns CLaMP3 M3 patches per pre-sliced MIDI segment.

    Reads ``midi_path`` from each JSONL record (the per-segment MIDI file
    produced by ``scripts/data/build_supermario_dataset.py``). M3
    patchilization happens lazily in ``__getitem__`` so cold-start cost
    is bounded; warm-cache hits skip the patchilizer entirely via
    ``cache_check_fn``.
    """

    # ── Reuse the 6-class inventory from the audio path ───────────────────
    # Same dict object — kept in sync automatically. The classification head
    # is shared, so symbolic and audio configs use the same out_dim=6.
    LABEL2IDX = _SuperMarioStructureAudioBase.LABEL2IDX
    IDX2LABEL = _SuperMarioStructureAudioBase.IDX2LABEL
    NUM_CLASSES = _SuperMarioStructureAudioBase.NUM_CLASSES

    EXAMPLE_JSONL = {
        "midi_path": "data/SuperMarioStructure/midi_segments/00001/000_intro.mid",
        "ori_uid": "00001_000",
        "work_id": "00001",
        "label": "intro",
        "seg_idx": 0,
        "bar_start": 1,
        "bar_end": 10,
        "seg_start": 0.0,
        "seg_end": 18.75,
        "title": "Captain Toad Treasure Tracker - Retro RampUp",
        "ninsheetmusic_id": "4405",
    }

    def __init__(
        self,
        jsonl: str,
        max_patches: int | None = None,
        input_format: str = "midi",
    ):
        """
        Parameters
        ----------
        jsonl         : per-split JSONL produced by build_supermario_dataset.py.
        max_patches   : sequence-length cap (default = CLaMP3Config.PATCH_LENGTH).
        input_format  : "midi" (default, MTF mode through `midi_to_mtf`) or
                        "abc" (bar-level mode, requires `--build-abc` at build
                        time so every record has an `abc_path`). ABC is what
                        CLaMP3 was primarily trained on and gives bar-aligned
                        patches; MIDI/MTF still works but is the secondary
                        mode. See docs/data/supermario_setup.md § Option E.
        """
        # Intentional lazy import — matches the VGMIDITVar symbolic pattern.
        # Eagerly importing CLaMP3 at module top level would force the heavy
        # CLaMP3 weight download on machines that never run the symbolic
        # config (e.g. audio-only sweeps over MuQ / MERT / OMARRQ).
        from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer
        from marble.encoders.CLaMP3.midi_util import midi_to_mtf
        from marble.encoders.CLaMP3.model import CLaMP3Config

        self._midi_to_mtf = midi_to_mtf
        self.patchilizer = M3Patchilizer()
        self.max_patches = max_patches or CLaMP3Config.PATCH_LENGTH
        self.patch_size = CLaMP3Config.PATCH_SIZE
        self.pad_token_id = self.patchilizer.pad_token_id

        if input_format not in ("midi", "abc"):
            raise ValueError(f"input_format must be 'midi' or 'abc', got {input_format!r}")
        self.input_format = input_format

        with open(jsonl, encoding="utf-8") as f:
            self.meta: list[dict] = [json.loads(line) for line in f]

        # Fail-loud validation. For MIDI: every record needs `midi_path`. For
        # ABC: filter to records that have `abc_path` (those without are
        # silently skipped — the build script doesn't guarantee ABC for
        # every record if --mxl-source-dir was missing some pieces).
        if input_format == "midi":
            for info in self.meta:
                if "midi_path" not in info:
                    raise ValueError(
                        f"JSONL record missing `midi_path`: {info.get('ori_uid', '?')} — "
                        f"re-run scripts/data/build_supermario_dataset.py to regenerate"
                    )
        else:  # abc
            n_before = len(self.meta)
            self.meta = [m for m in self.meta if m.get("abc_path")]
            n_dropped = n_before - len(self.meta)
            if n_dropped:
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "  dropped %d/%d records lacking `abc_path` (input_format='abc'); "
                    "re-run build_supermario_dataset.py with --build-abc + "
                    "--mxl-source-dir to populate.",
                    n_dropped,
                    n_before,
                )
            if not self.meta:
                raise ValueError(
                    f"input_format='abc' but no records in {jsonl} have abc_path. "
                    f"Re-build with: scripts/data/build_supermario_dataset.py "
                    f"--build-abc --mxl-source-dir <dir>"
                )

        # Common label validation.
        for info in self.meta:
            lbl = info["label"]
            if isinstance(lbl, list):
                lbl = lbl[0] if lbl else None
                info["label"] = lbl
            if lbl not in self.LABEL2IDX:
                raise ValueError(
                    f"Unknown label {lbl!r} for {info.get('midi_path', '?')}; "
                    f"valid labels: {sorted(set(self.LABEL2IDX))}"
                )

        # Set by the task at setup() time when the per-clip embedding cache
        # is active. See marble.utils.emb_cache.EmbeddingCacheMixin.
        self.cache_check_fn = None

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        """Return ``(patches, label, ori_uid, clip_id)`` 4-tuple."""
        info = self.meta[idx]
        # input_format='abc': use abc_path; the patchilizer's ABC mode kicks
        # in automatically because the input string doesn't start with
        # 'ticks_per_beat'. Otherwise route through midi_to_mtf.
        if self.input_format == "abc":
            symbolic_path = info["abc_path"]
        else:
            symbolic_path = info["midi_path"]
        label = self.LABEL2IDX[info["label"]]
        ori_uid = info["ori_uid"]
        # One "slice" per segment. Cache key encodes the source path so
        # ABC and MIDI variants of the same segment get separate cache
        # entries (different post-encoder embeddings).
        clip_id = make_clip_id(symbolic_path, 0)

        # Cache hit — skip patchilization entirely.
        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            patches = torch.zeros((self.max_patches, self.patch_size), dtype=torch.long)
            return patches, label, ori_uid, clip_id

        if self.input_format == "abc":
            # Read ABC text directly — the patchilizer's bar-mode handles it.
            with open(symbolic_path, encoding="utf-8") as f:
                tokenizer_input = f.read()
        else:
            # MIDI → MTF (event-level packing).
            tokenizer_input = self._midi_to_mtf(symbolic_path)

        patches_list = self.patchilizer.encode(
            tokenizer_input, patch_size=self.patch_size, add_special_patches=True
        )
        patches_list = patches_list[: self.max_patches]
        patches = torch.tensor(patches_list, dtype=torch.long)  # (P, patch_size)

        # Pad to (max_patches, patch_size) so default_collate can batch.
        if patches.size(0) < self.max_patches:
            pad_rows = self.max_patches - patches.size(0)
            pad_block = torch.full((pad_rows, self.patch_size), self.pad_token_id, dtype=torch.long)
            patches = torch.cat([patches, pad_block], dim=0)

        return patches, label, ori_uid, clip_id


class SuperMarioStructureSymbolicTrain(_SuperMarioStructureSymbolicBase):
    """Training split — DataModule sets shuffle=True."""

    pass


class SuperMarioStructureSymbolicVal(_SuperMarioStructureSymbolicBase):
    """Validation split — DataModule sets shuffle=False."""

    pass


class SuperMarioStructureSymbolicTest(SuperMarioStructureSymbolicVal):
    """Test split — same logic as val."""

    pass
