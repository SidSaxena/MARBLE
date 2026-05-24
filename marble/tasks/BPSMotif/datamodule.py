# marble/tasks/BPSMotif/datamodule.py
"""
BPS-Motif datamodules — symbolic, CLaMP3-M3-patch tokenised.

Source dataset: Hsiao, Hung, Chen, Su, ISMIR 2023.
  https://github.com/Wiilly07/Beethoven_motif

This module ships TWO parallel tasks:

* **MNID** (Motif-Note Identification) — clip-level binary classification.
  Each clip is a sliced sub-MIDI of a Beethoven sonata first movement,
  labelled 1 if the window is a motif occurrence and 0 if it's a sampled
  same-length non-motif window. Designed to feed straight into ``BaseTask``
  with a 2-class CrossEntropy head + TimeAvgPool.

* **Retrieval** — within-piece within-letter motif retrieval. Each clip
  is a positive motif window. Each clip's ``work_id`` encodes
  ``(piece_id, motif_letter)`` jointly so the standard
  ``CoverRetrievalTask`` scores "same piece + same motif letter" as
  relevant — which is exactly the within-movement motif identity we want
  because motif letters are movement-local in this dataset (A in Op.2 No.1
  has nothing to do with A in Op.2 No.2).

Both datasets read per-fold JSONLs produced by
``scripts/data/build_bps_motif_dataset.py`` and tokenise MIDIs on the fly
through CLaMP3's M3 patchilizer — no on-disk patch cache (tokenisation is
sub-millisecond per file; the heavy work is the encoder forward).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from marble.core.base_datamodule import BaseDataModule
from marble.utils.emb_cache import make_clip_id

# ──────────────────────────────────────────────────────────────────────────────
# Shared base — MIDI → M3 patches at runtime
# ──────────────────────────────────────────────────────────────────────────────


class _BPSMotifSymbolicBase(Dataset):
    """Common machinery for both MNID and Retrieval datasets.

    Loads a per-fold JSONL (produced by build_bps_motif_dataset.py),
    filters by split, and tokenises each clip's MIDI on the fly via the
    M3 patchiliser. Both subclasses share the patches/mask/clip_id
    construction; what changes is the per-item *label* tuple.

    Args
    ----
    jsonl_template : path with a ``{fold}`` placeholder, e.g.
        ``"data/BPS-Motif/BPSMotif.MNID.fold{fold}.{split}.jsonl"``. The
        ``{split}`` placeholder is filled in per subclass.
    split          : "train" | "val" | "test".
    fold_idx       : 0..4 (5-fold CV).
    max_patches    : sequence-length cap (default CLaMP3 PATCH_LENGTH = 512).
    """

    def __init__(
        self,
        jsonl_template: str,
        split: str,
        fold_idx: int = 0,
        max_patches: int | None = None,
    ):
        # Lazy imports — CLaMP3 is heavy and we don't want to pay the
        # import cost when running other (audio-only) tasks.
        from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer
        from marble.encoders.CLaMP3.midi_util import midi_to_mtf
        from marble.encoders.CLaMP3.model import CLaMP3Config

        self._midi_to_mtf = midi_to_mtf
        self.patchilizer = M3Patchilizer()
        self.max_patches = max_patches or CLaMP3Config.PATCH_LENGTH
        self.patch_size = CLaMP3Config.PATCH_SIZE
        self.pad_token_id = self.patchilizer.pad_token_id

        jsonl_path = jsonl_template.format(fold=fold_idx, split=split)
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(
                f"BPS-Motif JSONL not found: {jsonl_path}\n"
                "  Build the dataset first:\n"
                "    uv run python scripts/data/build_bps_motif_dataset.py"
            )
        # Cross-OS JSONL load (Windows backslash audio_paths → POSIX).
        # See marble/utils/path_compat.py.
        from marble.utils.path_compat import load_jsonl

        self.meta: list[dict] = load_jsonl(jsonl_path)
        self.split = split
        self.fold_idx = fold_idx
        # Set by EmbeddingCacheMixin._inject_cache_check_into_datasets when
        # the parent task uses cache_embeddings=True. We don't use it for
        # audio-I/O bypass here (no audio decode to skip) but we honour the
        # 4-tuple contract by emitting clip_id.
        self.cache_check_fn = None

    def __len__(self) -> int:
        return len(self.meta)

    def _tokenise(self, midi_path: str) -> torch.Tensor:
        """MIDI → padded (max_patches, patch_size) LongTensor."""
        mtf = self._midi_to_mtf(midi_path)
        patches_list = self.patchilizer.encode(
            mtf, patch_size=self.patch_size, add_special_patches=True
        )
        patches_list = patches_list[: self.max_patches]
        patches_t = torch.tensor(patches_list, dtype=torch.long)
        if patches_t.size(0) < self.max_patches:
            pad_rows = self.max_patches - patches_t.size(0)
            pad_block = torch.full(
                (pad_rows, self.patch_size),
                self.pad_token_id,
                dtype=torch.long,
            )
            patches_t = torch.cat([patches_t, pad_block], dim=0)
        return patches_t


# ──────────────────────────────────────────────────────────────────────────────
# MNID — clip-level binary classification
# ──────────────────────────────────────────────────────────────────────────────


class _BPSMotifMNIDBase(_BPSMotifSymbolicBase):
    """Returns ``(patches, label, midi_path, clip_id)`` — BaseTask 4-tuple.

    ``label`` is 0 (non-motif window) or 1 (motif window). The midi_path
    slot is required so BaseTask's test_step can aggregate per-file (it's
    a no-op here since each window is already its own file, but the
    contract has to hold).
    """

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotif.MNID.fold{fold}.{split}.jsonl"

    def __init__(
        self,
        fold_idx: int = 0,
        split: str = "train",
        max_patches: int | None = None,
        jsonl_template: str | None = None,
    ):
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=fold_idx,
            max_patches=max_patches,
        )

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        midi_path = entry["midi_path"]
        patches = self._tokenise(midi_path)
        label = torch.tensor(int(entry["is_motif"]), dtype=torch.long)
        clip_id = make_clip_id(midi_path, 0)
        return patches, label, midi_path, clip_id


class BPSMotifMNIDTrain(_BPSMotifMNIDBase):
    """MNID train split."""

    def __init__(self, **kwargs):
        kwargs["split"] = "train"
        super().__init__(**kwargs)


class BPSMotifMNIDVal(_BPSMotifMNIDBase):
    """MNID val split."""

    def __init__(self, **kwargs):
        kwargs["split"] = "val"
        super().__init__(**kwargs)


class BPSMotifMNIDTest(_BPSMotifMNIDBase):
    """MNID test split."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval — within-piece within-letter motif retrieval
# ──────────────────────────────────────────────────────────────────────────────


def _encode_work_id(piece_id: str, motif_letter: str) -> int:
    """Encode (piece_id, motif_letter) → unique int for CoverRetrievalTask.

    piece_id format from upstream is "<NN>-<M>" (zero-padded sonata-
    movement, e.g. "01-1"). We pack:
      sonata * 100000 + movement * 1000 + ord(letter)
    which is unique for sonata ∈ [1..32], movement ∈ [1..9], and any
    lowercase ASCII letter. Decode is unnecessary — only equality
    comparison is used by ``CoverRetrievalTask._compute_map``.
    """
    sonata_str, movement_str = piece_id.split("-")
    sonata = int(sonata_str)
    movement = int(movement_str)
    letter_ord = ord(motif_letter.lower()) if motif_letter else 0
    return sonata * 100_000 + movement * 1_000 + letter_ord


class _BPSMotifRetrievalBase(_BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, midi_path, clip_id)`` — CoverRetrievalTask
    4-tuple contract.

    ``work_id`` jointly encodes ``(piece_id, motif_letter)`` so the
    standard MAP scoring counts "same piece + same letter" occurrences as
    relevant — which is the within-movement motif identity we want
    because motif letters in this dataset are movement-local.
    """

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotif.Retrieval.fold{fold}.{split}.jsonl"

    def __init__(
        self,
        fold_idx: int = 0,
        split: str = "test",
        max_patches: int | None = None,
        jsonl_template: str | None = None,
    ):
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=fold_idx,
            max_patches=max_patches,
        )

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        midi_path = entry["midi_path"]
        patches = self._tokenise(midi_path)
        work_id = _encode_work_id(entry["piece_id"], entry["motif_letter"])
        clip_id = make_clip_id(midi_path, 0)
        # Retrieval contract: (input, work_id, path, clip_id). path is used
        # for per-file aggregation in CoverRetrievalTask; since each window
        # is already a separate file here, aggregation is a no-op.
        return patches, work_id, midi_path, clip_id


class BPSMotifRetrievalTest(_BPSMotifRetrievalBase):
    """Test-split retrieval evaluation."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class BPSMotifRetrievalDummy(_BPSMotifRetrievalBase):
    """Placeholder for max_epochs=0 train/val dataloaders.

    Points at the test JSONL but is never iterated when ``max_epochs=0``
    (CoverRetrievalTask runs zero-shot — fit is a no-op).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# DataModule wrappers
# ──────────────────────────────────────────────────────────────────────────────


class BPSMotifMNIDDataModule(BaseDataModule):
    """Thin wrapper — BaseDataModule handles dataloaders + transforms."""

    pass


class BPSMotifRetrievalDataModule(BaseDataModule):
    """Thin wrapper — BaseDataModule handles dataloaders + transforms."""

    pass
