"""JKUPDD within-piece motif RETRIEVAL datamodule — symbolic, CLaMP3-M3.

JKUPDD (JKU Patterns Development Database, Collins 2013) ground truth as a
zero-shot retrieval task: each annotated pattern-occurrence MIDI is a query,
and relevance = same ``(piece, annotator, pattern)`` group. Reuses BPS-Motif's
MIDI→MTF→M3 tokenisation base; only the per-item label (``work_id``) differs.

Build the JSONL first:
    uv run python scripts/data/build_jkupdd_retrieval.py --jkupdd-root <path>
"""

from __future__ import annotations

import hashlib

import torch

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.BPSMotif.datamodule import _BPSMotifSymbolicBase
from marble.utils.emb_cache import make_clip_id


def _work_id(group: str) -> int:
    """Stable int id for a ``piece|annotator|pattern`` group (relevance key).

    CoverRetrievalTask counts two clips as relevant iff their ``work_id`` is
    equal, so any collision-free encoding works. We take the first 8 hex of a
    SHA-1 (≤ 2^32, fits int64); 32 groups → no collisions.
    """
    return int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16)


class _JKUPDDRetrievalDataset(_BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, midi_path, clip_id)`` — CoverRetrievalTask
    contract. ``work_id`` groups occurrences of the same annotated pattern."""

    JSONL_TEMPLATE = "data/JKUPDD/JKUPDDRetrieval.{split}.jsonl"

    def __init__(self, split: str = "test", max_patches=None, jsonl_template=None):
        # JKUPDD has no CV folds — the template has no {fold} placeholder, so
        # the inherited fold_idx is unused.
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=0,
            max_patches=max_patches,
        )

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        midi_path = entry["midi_path"]
        patches = self._tokenise(midi_path)
        work_id = _work_id(entry["group"])
        clip_id = make_clip_id(midi_path, 0)
        return patches, torch.tensor(work_id, dtype=torch.long), midi_path, clip_id


class JKUPDDRetrievalTest(_JKUPDDRetrievalDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalDummy(_JKUPDDRetrievalDataset):
    """Stand-in for the train/val splits of a zero-shot (max_epochs=0) probe —
    points at the test JSONL since fit is skipped anyway."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalDataModule(BaseDataModule):
    """Thin DataModule — all logic lives in the dataset classes / BaseDataModule."""


# ──────────────────────────────────────────────────────────────────────────────
# Matched MTF variant — the SAME (lossy-MIDI→MTF) tokenisation as the base task,
# but restricted to the occurrence pool that survived ABC alignment, so the
# ABC-vs-MTF A/B is on an identical occurrence set. Reads the matched subset
# JSONL emitted by ``scripts/data/build_jkupdd_abc.py`` alongside the ABC JSONL.
# ──────────────────────────────────────────────────────────────────────────────


class _JKUPDDRetrievalMatchedDataset(_JKUPDDRetrievalDataset):
    """Identical to :class:`_JKUPDDRetrievalDataset` (MIDI→MTF tokenisation,
    same ``work_id`` / relevance) but points at the matched-subset JSONL."""

    JSONL_TEMPLATE = "data/JKUPDD/JKUPDDRetrieval.matched.{split}.jsonl"


class JKUPDDRetrievalMatchedTest(_JKUPDDRetrievalMatchedDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalMatchedDummy(_JKUPDDRetrievalMatchedDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalMatchedDataModule(BaseDataModule):
    """Thin DataModule for the matched-MTF subset variant."""


# ──────────────────────────────────────────────────────────────────────────────
# ABC variant — score-native interleaved-ABC instead of MIDI→MTF
# ──────────────────────────────────────────────────────────────────────────────
#
# The MTF datasets above tokenise each occurrence's lossy MIDI window via
# ``midi_to_mtf`` → ``M3Patchilizer.encode``. The ABC variant below instead reads
# a pre-built **interleaved-ABC** string per occurrence (produced offline by
# ``scripts/data/build_jkupdd_abc.py`` from the piece ``**kern`` + the JKUPDD
# point-set, so it preserves key / pitch-spelling / meter / bar structure that
# the MIDI round-trip discards) and feeds it through the *same* M3 patchiliser.
#
# ``M3Patchilizer.encode`` auto-detects the input format: MTF starts with a
# ``ticks_per_beat`` header line, ABC does not — so the identical encode call
# bar-segments the ABC. Everything else (work_id, relevance, clip_id, the
# CoverRetrievalTask scoring) is byte-identical to the MTF task, so the per-layer
# MAP is directly comparable. The JSONL carries the ABC inline under an ``abc``
# field; ``occurrence_id`` / ``group`` are the same canonical 78/20 dedup'd set.


class _JKUPDDRetrievalABCDataset(_BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, clip_key, clip_id)`` — same CoverRetrievalTask
    contract as the MTF dataset, but the patches come from each occurrence's
    pre-built interleaved-ABC string (JSONL ``abc`` field) rather than its MIDI.

    ``work_id`` (and therefore relevance) is computed identically to the MTF
    dataset via :func:`_work_id`, so MAP is apples-to-apples."""

    JSONL_TEMPLATE = "data/JKUPDD/JKUPDDRetrievalABC.{split}.jsonl"

    def __init__(self, split: str = "test", max_patches=None, jsonl_template=None):
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=0,
            max_patches=max_patches,
        )

    def _tokenise_abc(self, abc: str) -> torch.Tensor:
        """Interleaved-ABC string → padded (max_patches, patch_size) LongTensor.

        Mirrors ``_BPSMotifSymbolicBase._tokenise`` exactly but skips the
        ``midi_to_mtf`` step — the input is already the text CLaMP3 tokenises.
        ``M3Patchilizer.encode`` takes the ABC branch (no ``ticks_per_beat``
        header), so this is the score-native counterpart of the MTF path.
        """
        patches_list = self.patchilizer.encode(
            abc, patch_size=self.patch_size, add_special_patches=True
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

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        abc = entry["abc"]
        patches = self._tokenise_abc(abc)
        work_id = _work_id(entry["group"])
        # clip_key uniquely identifies this occurrence for per-file aggregation
        # + emb-cache keys; there is no midi_path here, so use occurrence_id.
        clip_key = entry["occurrence_id"]
        clip_id = make_clip_id(clip_key, 0)
        return patches, torch.tensor(work_id, dtype=torch.long), clip_key, clip_id


class JKUPDDRetrievalABCTest(_JKUPDDRetrievalABCDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalABCDummy(_JKUPDDRetrievalABCDataset):
    """Stand-in for the train/val splits of a zero-shot (max_epochs=0) probe —
    points at the test JSONL since fit is skipped anyway."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class JKUPDDRetrievalABCDataModule(BaseDataModule):
    """Thin DataModule for the ABC variant — logic lives in the dataset classes."""
