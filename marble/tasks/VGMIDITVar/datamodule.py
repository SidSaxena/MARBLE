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
``scripts/data/build_vgmiditvar_dataset.py``) and uses the work-group identity
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

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id

# Default clip length when splitting long renders into chunks.  Themes are
# usually short (≤30 s) so we use a smaller default than Covers80.
DEFAULT_CLIP_SECONDS = 15.0
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
        backend: str | None = None,
        split: str | None = None,
    ):
        self.sample_rate = int(sample_rate)
        self.channels = channels
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio
        self.channel_mode = channel_mode
        self.backend = backend
        # Set by the task at setup() time when the per-clip embedding
        # cache is active. See marble.utils.emb_cache.EmbeddingCacheMixin.
        self.cache_check_fn = None
        self.split = split

        with open(jsonl, encoding="utf-8") as f:
            self.meta: list[dict] = [json.loads(line) for line in f]

        # Optionally filter to one split
        if split is not None:
            self.meta = [m for m in self.meta if m.get("split") == split]

        # Resamplers for any non-target sample rates in the dataset
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        for info in self.meta:
            orig_sr = int(info["sample_rate"])
            if orig_sr != self.sample_rate and orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.sample_rate)

        # Flat index: (file_idx, slice_idx, orig_sr, orig_clip_frames)
        self.index_map: list[tuple[int, int, int, int]] = []
        for file_idx, info in enumerate(self.meta):
            orig_sr = int(info["sample_rate"])
            total_samples = int(info["num_samples"])
            orig_clip_frames = int(clip_seconds * orig_sr)
            if orig_clip_frames <= 0:
                continue
            n_full = total_samples // orig_clip_frames
            rem = total_samples - n_full * orig_clip_frames
            n_slices = n_full + (1 if rem / orig_clip_frames >= min_clip_ratio else 0)
            # VGMIDI-TVar themes are short (often 4–8 s).  Many renders end
            # up shorter than clip_seconds=15, which would give n_slices=0
            # via the formula above.  Force at least one slice — the
            # __getitem__ zero-pad path handles the short length cleanly,
            # so we'd rather include a short theme than drop it.
            n_slices = max(n_slices, 1)
            for s in range(n_slices):
                self.index_map.append((file_idx, s, orig_sr, orig_clip_frames))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, slice_idx, orig_sr, orig_clip_frames = self.index_map[idx]
        info = self.meta[file_idx]
        path = info["audio_path"]
        work_id = int(info["work_id"])
        clip_id = make_clip_id(path, slice_idx)

        cache_check = getattr(self, "cache_check_fn", None)
        if cache_check is not None and cache_check(clip_id):
            # Cache hit — skip audio I/O entirely. The task's forward()
            # ignores ``x`` on cache hits and uses the cached (L, H) tensor.
            target_len = int(self.clip_seconds * self.sample_rate)
            waveform = torch.zeros(self.channels, target_len)
            return waveform, work_id, path, clip_id

        offset = slice_idx * orig_clip_frames
        waveform, _ = torchaudio.load(
            path,
            frame_offset=offset,
            num_frames=orig_clip_frames,
            backend=self.backend,
        )  # (C, T)

        # ── channel handling ──────────────────────────────────────────────────
        C = waveform.size(0)
        if self.channels <= C:
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
            deficit = self.channels - C
            waveform = torch.cat([waveform, waveform[-1:].repeat(deficit, 1)], 0)

        # ── resample ──────────────────────────────────────────────────────────
        if orig_sr != self.sample_rate:
            waveform = self.resamplers[orig_sr](waveform)

        # ── pad / truncate to exact clip length ───────────────────────────────
        T = waveform.size(1)
        if self.clip_len_target > T:
            waveform = F.pad(waveform, (0, self.clip_len_target - T))
        elif self.clip_len_target < T:
            waveform = waveform[:, : self.clip_len_target]

        return waveform, work_id, path, clip_id


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


# ──────────────────────────────────────────────────────────────────────────────
# Symbolic path (CLaMP3 native MIDI encoder)
# ──────────────────────────────────────────────────────────────────────────────
#
# Returns pre-tokenised CLaMP3 M3 patches instead of waveforms.  Same
# (input, work_id, path) contract so the existing CoverRetrievalTask works
# unchanged — only the encoder differs (CLaMP3_Symbolic_Encoder).
#
# Each item:
#   patches : LongTensor (MAX_PATCHES, PATCH_SIZE)   — padded with pad_token_id
#   work_id : int
#   midi_path : str
#
# Padding to a fixed MAX_PATCHES lets PyTorch's default_collate batch items
# without a custom collate_fn.  PATCH_LENGTH=512 from CLaMP3Config — for
# theme-length MIDI (a few seconds each) this is comfortably enough.


class _VGMIDITVarSymbolicBase(Dataset):
    """Dataset returning CLaMP3 M3 patches for each MIDI file.

    Args
    ----
    jsonl       : MARBLE JSONL.  Each row must include either
                  ``midi_path`` directly OR ``audio_path`` from which the
                  MIDI path is derived by replacing the audio extension
                  with ``.mid``.  In the VGMIDI-TVar build, MIDIs sit
                  alongside the audio renders under  ``<data-dir>/midi/<split>/``,
                  so the dataset will look there if ``midi_path`` is absent.
    split       : optional split filter (``"train"`` / ``"test"``).
    max_patches : sequence-length cap.  Defaults to CLaMP3Config.PATCH_LENGTH (512).
    """

    def __init__(
        self,
        jsonl: str,
        split: str | None = None,
        max_patches: int | None = None,
        midi_dir: str | None = None,
    ):
        # Intentional lazy imports.  The audio path of VGMIDITVar runs
        # with MERT / OMARRQ encoders that should NOT pull in CLaMP3 at
        # all — eagerly importing CLaMP3 at module top-level would slow
        # down those other sweeps and force the (heavy) CLaMP3 weight
        # download on machines that never run the symbolic config.
        # Do not move these to the file head.
        from pathlib import Path as _Path

        from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer
        from marble.encoders.CLaMP3.midi_util import midi_to_mtf
        from marble.encoders.CLaMP3.model import CLaMP3Config

        self._midi_to_mtf = midi_to_mtf
        self.patchilizer = M3Patchilizer()
        self.max_patches = max_patches or CLaMP3Config.PATCH_LENGTH
        self.patch_size = CLaMP3Config.PATCH_SIZE
        self.pad_token_id = self.patchilizer.pad_token_id
        self.midi_dir = _Path(midi_dir) if midi_dir else None

        with open(jsonl, encoding="utf-8") as f:
            self.meta: list[dict] = [json.loads(line) for line in f]
        if split is not None:
            self.meta = [m for m in self.meta if m.get("split") == split]

    def _resolve_midi_path(self, entry: dict) -> str:
        if "midi_path" in entry:
            return entry["midi_path"]
        # Fall back: replace audio extension with .mid; if a midi_dir is
        # given, also try <midi_dir>/<split>/<stem>.mid (matches the layout
        # build_vgmiditvar_dataset.py produces).
        from pathlib import Path as _Path

        audio_path = _Path(entry["audio_path"])
        candidate = audio_path.with_suffix(".mid")
        if candidate.exists():
            return str(candidate)
        if self.midi_dir is not None:
            cand = self.midi_dir / entry.get("split", "") / (audio_path.stem + ".mid")
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(
            f"Could not find MIDI for {audio_path.stem}; tried {candidate} and {self.midi_dir}"
        )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        midi_path = self._resolve_midi_path(entry)
        mtf = self._midi_to_mtf(midi_path)

        # Encode → list of patches; each patch is a list of int tokens.
        patches_list = self.patchilizer.encode(
            mtf, patch_size=self.patch_size, add_special_patches=True
        )
        # Truncate to max_patches and stack to a LongTensor.
        patches_list = patches_list[: self.max_patches]
        patches_t = torch.tensor(patches_list, dtype=torch.long)  # (P, patch_size)

        # Pad to (max_patches, patch_size) so default_collate can batch.
        if patches_t.size(0) < self.max_patches:
            pad_rows = self.max_patches - patches_t.size(0)
            pad_block = torch.full(
                (pad_rows, self.patch_size),
                self.pad_token_id,
                dtype=torch.long,
            )
            patches_t = torch.cat([patches_t, pad_block], dim=0)

        work_id = int(entry["work_id"])
        return patches_t, work_id, midi_path


class VGMIDITVarSymbolicAll(_VGMIDITVarSymbolicBase):
    """All MIDI files — zero-shot symbolic retrieval evaluation."""

    pass


class VGMIDITVarSymbolicTest(_VGMIDITVarSymbolicBase):
    """Test-split MIDI files only."""

    def __init__(self, *args, **kwargs):
        kwargs["split"] = "test"
        super().__init__(*args, **kwargs)


class VGMIDITVarSymbolicDummy(_VGMIDITVarSymbolicBase):
    """Placeholder dataset for the train / val loaders under max_epochs=0."""

    pass
