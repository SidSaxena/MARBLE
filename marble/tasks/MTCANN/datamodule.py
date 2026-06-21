"""MTC-ANN folk-melody RETRIEVAL datamodules — symbolic, CLaMP3-M3.

MTC-ANN (Meertens Tune Collections, Annotated Corpus; van Kranenburg et al.)
ground truth as a zero-shot retrieval task, mirroring the JKUPDDRetrieval
pipeline exactly. Two parallel tasks:

* **MTCANNTuneFamily** — cross-melody tune-family retrieval. Each melody is a
  query; relevance = same ``tunefamily`` group. The canonical MTC-ANN benchmark
  (folk melodies cluster into oral-transmission tune families).

* **MTCANNMotif** — within-corpus motif retrieval. Each annotated motif
  occurrence is a query; relevance = same ``(tunefamily, motif)`` group.

Each task ships **two dataset variants** so the same occurrence pool can be
embedded two ways for a clean A/B:

* **ABC** — reads the score-native interleaved-ABC string (JSONL ``abc`` field),
  preserving key / pitch-spelling / meter / bar structure.
* **MTF** — reads the lossy ``midi_path`` field, tokenised via MIDI→MTF.

Both variants compute ``work_id`` identically (stable hash of the ``group``
string), so MAP is apples-to-apples. Zero-shot (``max_epochs=0``) leave-one-out,
**no CV folds** — same structure as JKUPDDRetrieval.

The JSONL is produced by the parallel data-build agent in the SAME field schema
as JKUPDD's retrieval JSONL (``group`` / ``occurrence_id`` / ``abc`` / ``midi_path``
/ ``split``). Paths (one zero-shot test pool per task x arm; no ``.test.``
infix — every row already carries ``"split": "test"``)::

    data/MTC-ANN/MTCANN.TuneFamily.MTF.jsonl   (midi_path field)
    data/MTC-ANN/MTCANN.TuneFamily.ABC.jsonl   (abc field)
    data/MTC-ANN/MTCANN.Motif.MTF.jsonl
    data/MTC-ANN/MTCANN.Motif.ABC.jsonl
"""

from __future__ import annotations

import hashlib

import torch

from marble.core.base_datamodule import BaseDataModule
from marble.tasks.BPSMotif.datamodule import _BPSMotifSymbolicBase
from marble.utils.emb_cache import make_clip_id


def _work_id(group: str) -> int:
    """Stable int id for a relevance group string (the relevance key).

    CoverRetrievalTask counts two clips as relevant iff their ``work_id`` is
    equal, so any collision-free encoding works. We take the first 8 hex of a
    SHA-1 (<= 2^32, fits int64) — identical to JKUPDDRetrieval, so the two
    benchmarks share the exact same relevance machinery. With ~few-hundred
    groups the birthday-collision probability over a 32-bit space is negligible.
    """
    return int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16)


def _family_id(group: str) -> int:
    """Stable int id for the *tune-family* of a Motif relevance group.

    The Motif ``group`` is ``"<family>|<motifclass>"`` so the tune family is
    the prefix before the first ``|``. We hash it the same way as
    :func:`_work_id` (8-hex SHA-1) so the metric step can build a same-family
    hard-distractor mask by integer equality, exactly mirroring the work_id
    self-mask path. For the TuneFamily task ``group == family`` so this would
    equal ``_work_id`` (degenerate) — TuneFamily datasets deliberately don't
    emit it (4-tuple), so the same-family metric is skipped there.
    """
    return int(hashlib.sha1(group.split("|", 1)[0].encode("utf-8")).hexdigest()[:8], 16)


def _note_count(entry: dict) -> int:
    """Per-fragment motif note-count for length-stratified MAP.

    The build writes ``n_src_notes`` (the count of source kern notes in the
    fragment — the ground-truth length, independent of the lossy MTF/ABC
    re-encode). Fall back to ``n_abc_notes`` / ``n_mtf_notes`` then 0 if a
    legacy JSONL lacks it (the metric just buckets unknown-length as 0 → long
    bucket is unaffected, short bucket would only ever shrink).
    """
    for key in ("n_src_notes", "n_abc_notes", "n_mtf_notes"):
        v = entry.get(key)
        if v is not None:
            return int(v)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# MTF variant — lossy MIDI → MTF tokenisation (reads midi_path)
# ──────────────────────────────────────────────────────────────────────────────


class _MTCANNRetrievalMTFDataset(_BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, midi_path, clip_id)`` — CoverRetrievalTask
    contract. ``work_id`` groups occurrences of the same relevance group.

    Subclasses set :attr:`JSONL_TEMPLATE` per task (TuneFamily / Motif). The
    template has no ``{fold}`` placeholder (MTC-ANN has no CV folds), so the
    inherited ``fold_idx`` is unused.
    """

    JSONL_TEMPLATE: str = ""  # set by subclass
    # When True (Motif task only) emit a 6-tuple carrying the per-fragment
    # tune-family id + motif note-count for the same-family hard-distractor
    # MAP and the length-stratified MAP. TuneFamily leaves this False → plain
    # 4-tuple → CoverRetrievalTask's new Motif-only metrics are skipped.
    EMIT_MOTIF_META: bool = False

    def __init__(self, split: str = "test", max_patches=None, jsonl_template=None):
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
        if self.EMIT_MOTIF_META:
            return (
                patches,
                torch.tensor(work_id, dtype=torch.long),
                midi_path,
                clip_id,
                torch.tensor(_family_id(entry["group"]), dtype=torch.long),
                torch.tensor(_note_count(entry), dtype=torch.long),
            )
        return patches, torch.tensor(work_id, dtype=torch.long), midi_path, clip_id


# ──────────────────────────────────────────────────────────────────────────────
# ABC variant — score-native interleaved-ABC (reads abc field)
# ──────────────────────────────────────────────────────────────────────────────


class _MTCANNRetrievalABCDataset(_BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, clip_key, clip_id)`` — same
    CoverRetrievalTask contract as the MTF dataset, but the patches come from
    each occurrence's pre-built interleaved-ABC string (JSONL ``abc`` field)
    rather than its MIDI.

    ``work_id`` (and therefore relevance) is computed identically to the MTF
    dataset via :func:`_work_id`, so MAP is apples-to-apples.
    ``M3Patchilizer.encode`` auto-detects ABC vs MTF (MTF starts with a
    ``ticks_per_beat`` header; ABC does not), so the same encode call
    bar-segments the ABC — the score-native counterpart of the MTF path.
    """

    JSONL_TEMPLATE: str = ""  # set by subclass
    EMIT_MOTIF_META: bool = False  # see _MTCANNRetrievalMTFDataset

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
        if self.EMIT_MOTIF_META:
            return (
                patches,
                torch.tensor(work_id, dtype=torch.long),
                clip_key,
                clip_id,
                torch.tensor(_family_id(entry["group"]), dtype=torch.long),
                torch.tensor(_note_count(entry), dtype=torch.long),
            )
        return patches, torch.tensor(work_id, dtype=torch.long), clip_key, clip_id


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — TuneFamily retrieval (cross-melody, same tune family)
# ──────────────────────────────────────────────────────────────────────────────


class _MTCANNTuneFamilyMTFDataset(_MTCANNRetrievalMTFDataset):
    # NB: the build agent emits a single zero-shot test pool per (task, arm)
    # with no per-split file, so the filename has no {split} infix (every row
    # already carries "split": "test"). The inherited base does not filter on
    # ``split`` — it loads every row of whatever file the template resolves to.
    JSONL_TEMPLATE = "data/MTC-ANN/MTCANN.TuneFamily.MTF.jsonl"


class MTCANNTuneFamilyMTFTest(_MTCANNTuneFamilyMTFDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNTuneFamilyMTFDummy(_MTCANNTuneFamilyMTFDataset):
    """Stand-in for the train/val splits of a zero-shot (max_epochs=0) probe —
    points at the test JSONL since fit is skipped anyway."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class _MTCANNTuneFamilyABCDataset(_MTCANNRetrievalABCDataset):
    JSONL_TEMPLATE = "data/MTC-ANN/MTCANN.TuneFamily.ABC.jsonl"


class MTCANNTuneFamilyABCTest(_MTCANNTuneFamilyABCDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNTuneFamilyABCDummy(_MTCANNTuneFamilyABCDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNTuneFamilyMTFDataModule(BaseDataModule):
    """Thin DataModule — all logic lives in the dataset classes / BaseDataModule."""


class MTCANNTuneFamilyABCDataModule(BaseDataModule):
    """Thin DataModule for the TuneFamily ABC variant."""


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Motif retrieval (within-corpus, same annotated motif)
# ──────────────────────────────────────────────────────────────────────────────


class _MTCANNMotifMTFDataset(_MTCANNRetrievalMTFDataset):
    JSONL_TEMPLATE = "data/MTC-ANN/MTCANN.Motif.MTF.jsonl"
    EMIT_MOTIF_META = True  # 6-tuple: + (family_id, note_count) for the new metrics


class MTCANNMotifMTFTest(_MTCANNMotifMTFDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNMotifMTFDummy(_MTCANNMotifMTFDataset):
    """Stand-in for the train/val splits of a zero-shot (max_epochs=0) probe."""

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class _MTCANNMotifABCDataset(_MTCANNRetrievalABCDataset):
    JSONL_TEMPLATE = "data/MTC-ANN/MTCANN.Motif.ABC.jsonl"
    EMIT_MOTIF_META = True  # 6-tuple: + (family_id, note_count) for the new metrics


class MTCANNMotifABCTest(_MTCANNMotifABCDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNMotifABCDummy(_MTCANNMotifABCDataset):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class MTCANNMotifMTFDataModule(BaseDataModule):
    """Thin DataModule for the Motif MTF variant."""


class MTCANNMotifABCDataModule(BaseDataModule):
    """Thin DataModule for the Motif ABC variant."""
