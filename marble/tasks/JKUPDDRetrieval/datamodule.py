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
