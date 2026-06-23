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


# ──────────────────────────────────────────────────────────────────────────────
# ABC variant — score-native interleaved-ABC instead of MIDI → MTF
# ──────────────────────────────────────────────────────────────────────────────
#
# The MTF datasets above tokenise each window's lossy MIDI slice via
# ``midi_to_mtf`` → ``M3Patchilizer.encode``. The ABC variant below instead reads
# a pre-built **interleaved-ABC** string per window (produced offline by
# ``scripts/data/build_bps_motif_abc.py`` — Option B: the same notes the MTF
# window contains, reconstructed directly from ``csv_notes`` as a single-voice,
# re-zeroed line, so it carries pitch spelling / meter / bar structure the MIDI
# round-trip discards) and feeds it through the *same* M3 patchiliser.
#
# ``M3Patchilizer.encode`` auto-detects the input format (MTF starts with a
# ``ticks_per_beat`` header; ABC does not), so the identical encode call
# bar-segments the ABC. Everything else (label / work_id / relevance / clip_id /
# the BaseTask + CoverRetrievalTask scoring) is byte-identical to the MTF task,
# so the per-layer numbers are directly comparable. The JSONLs carry the ABC
# inline under an ``abc`` field and preserve every MTF field (piece_id, fold,
# split, is_motif, motif_letter, occurrence_id, start_sec, end_sec).


class _BPSMotifABCMixin:
    """Shared ABC tokenisation: interleaved-ABC string → padded patch tensor.

    Mirrors ``_BPSMotifSymbolicBase._tokenise`` but skips ``midi_to_mtf`` — the
    JSONL ``abc`` field is already the text CLaMP3 tokenises. Used by both the
    MNID and Retrieval ABC datasets so the encode path is identical to the MTF
    one minus the MIDI→MTF step.
    """

    def _tokenise_abc(self, abc: str) -> torch.Tensor:
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


# ── MNID (binary classification) — ABC input ─────────────────────────────────


class _BPSMotifMNIDABCBase(_BPSMotifABCMixin, _BPSMotifSymbolicBase):
    """Returns ``(patches, label, clip_key, clip_id)`` — same BaseTask 4-tuple
    as the MTF MNID dataset, but patches come from the JSONL ``abc`` field."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifABC.MNID.fold{fold}.{split}.jsonl"

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
        patches = self._tokenise_abc(entry["abc"])
        label = torch.tensor(int(entry["is_motif"]), dtype=torch.long)
        # No midi_path for the ABC arm; use occurrence_id as the per-window key
        # for per-file aggregation + emb-cache keys.
        clip_key = entry["occurrence_id"]
        clip_id = make_clip_id(clip_key, 0)
        return patches, label, clip_key, clip_id


class BPSMotifMNIDABCTrain(_BPSMotifMNIDABCBase):
    def __init__(self, **kwargs):
        kwargs["split"] = "train"
        super().__init__(**kwargs)


class BPSMotifMNIDABCVal(_BPSMotifMNIDABCBase):
    def __init__(self, **kwargs):
        kwargs["split"] = "val"
        super().__init__(**kwargs)


class BPSMotifMNIDABCTest(_BPSMotifMNIDABCBase):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


# ── Retrieval (zero-shot MAP) — ABC input ────────────────────────────────────


class _BPSMotifRetrievalABCBase(_BPSMotifABCMixin, _BPSMotifSymbolicBase):
    """Returns ``(patches, work_id, clip_key, clip_id)`` — same
    CoverRetrievalTask contract / ``work_id`` encoding as the MTF Retrieval
    dataset, but patches come from the JSONL ``abc`` field."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifABC.Retrieval.fold{fold}.{split}.jsonl"

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
        patches = self._tokenise_abc(entry["abc"])
        work_id = _encode_work_id(entry["piece_id"], entry["motif_letter"])
        clip_key = entry["occurrence_id"]
        clip_id = make_clip_id(clip_key, 0)
        return patches, work_id, clip_key, clip_id


class BPSMotifRetrievalABCTest(_BPSMotifRetrievalABCBase):
    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class BPSMotifRetrievalABCDummy(_BPSMotifRetrievalABCBase):
    """Placeholder for max_epochs=0 train/val dataloaders (fit is a no-op)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)


class BPSMotifMNIDABCDataModule(BaseDataModule):
    """Thin wrapper for the ABC MNID variant."""

    pass


class BPSMotifRetrievalABCDataModule(BaseDataModule):
    """Thin wrapper for the ABC Retrieval variant."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Within-piece phrase-window — ABC input (within-movement same-motif retrieval)
# ──────────────────────────────────────────────────────────────────────────────
#
# The Retrieval task above slices ONE ABC fragment per labelled motif occurrence
# and scores cross-piece within-letter retrieval (same piece + same letter via
# the packed work_id). The WITHIN-PIECE task below instead slides N-bar phrase
# windows (stride 1) over each WHOLE movement (dataset built offline by
# ``scripts/data/build_bps_motif_within_piece.py``) and measures, PER MOVEMENT,
# whether two windows that share a motif letter retrieve each other — genuine
# within-movement recurrence. The semantics are the shuffle-control-validated
# leitmotifs prototype (scripts/eval/bps_within_piece_metric.py::within_movement_map).
#
# The inherited single-label full-gallery CoverRetrievalTask MAP can't express
# this (it's multi-label + per-movement-gallery + same-occurrence-excluded), so
# the dataset emits a 6-tuple and BPSMotifWithinPieceTask runs its own metric.
# The 6-tuple mirrors MTCANN's 6-tuple shape (datamodule.py:186-204): the two
# extra slots ride through default collation as a list-of-str (like ``paths``).


import hashlib  # noqa: E402 — local to the within-piece block


def _movement_group_id(movement_id: str) -> int:
    """Stable int gallery-group id for a movement (the within-piece group key).

    The within-piece metric restricts each query's gallery to its OWN movement,
    keyed by integer equality on this id. We take the first 8 hex of a SHA-1
    (<= 2^32, fits int64) — identical discipline to MTCANN's ``_work_id`` so the
    relevance machinery is shared. Collision probability over 32 movements in a
    32-bit space is negligible.
    """
    return int(hashlib.sha1(movement_id.encode("utf-8")).hexdigest()[:8], 16)


class _BPSMotifWithinPieceABCBase(_BPSMotifABCMixin, _BPSMotifSymbolicBase):
    """Returns the within-piece 6-tuple
    ``(patches, movement_id_int, occ_ids_str, letters_str, clip_key, clip_id)``.

    * ``patches`` — tokenised from the JSONL ``abc`` field (the phrase-window
      fragment), identical encode path to every other ABC dataset here.
    * ``movement_id_int`` — stable sha1-hash int of ``movement_id`` (the gallery
      group: the metric restricts each query's gallery to its own movement).
    * ``occ_ids_str`` / ``letters_str`` — ``'|'``-joined strings of the window's
      occurrence ids / motif letters. Emitting them as plain ``str`` lets them
      ride through PyTorch's default collation as a list-of-str (exactly like
      the ``paths`` slot in the other datasets) without a custom collate_fn; the
      task splits on ``'|'`` to recover the sets. Empty → ``""``.
    * ``clip_key`` — the per-window key (``window_id``) for per-file aggregation
      + emb-cache keys (there is no midi_path here).
    * ``clip_id`` — ``make_clip_id(window_id, 0)``.
    """

    JSONL_TEMPLATE: str = ""  # set by subclass

    def __init__(self, split: str = "test", max_patches=None, jsonl_template=None):
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=0,
            max_patches=max_patches,
        )

    def __getitem__(self, idx: int):
        entry = self.meta[idx]
        patches = self._tokenise_abc(entry["abc"])
        movement_id_int = _movement_group_id(entry["movement_id"])
        # '|'-joined so the two variable-length label sets survive default
        # collation as a list-of-str; "" when empty. The task splits on '|'.
        occ_ids_str = "|".join(entry.get("occurrence_ids", []))
        letters_str = "|".join(entry.get("letters", []))
        clip_key = entry["window_id"]
        clip_id = make_clip_id(clip_key, 0)
        return (
            patches,
            torch.tensor(movement_id_int, dtype=torch.long),
            occ_ids_str,
            letters_str,
            clip_key,
            clip_id,
        )


class BPSMotifWithinPieceN4ABCTest(_BPSMotifWithinPieceABCBase):
    """Within-piece N=4 phrase-window test split (zero-shot)."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifWithinPiece.N4.ABC.jsonl"

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class BPSMotifWithinPieceN4ABCDummy(_BPSMotifWithinPieceABCBase):
    """Placeholder for the max_epochs=0 train/val dataloaders (fit is a no-op)."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifWithinPiece.N4.ABC.jsonl"

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)


class BPSMotifWithinPieceABCDataModule(BaseDataModule):
    """Thin wrapper for the within-piece phrase-window ABC variant."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# Within-piece WHOLE-PIECE-CONTEXT — ABC input
# ──────────────────────────────────────────────────────────────────────────────
#
# The clip-isolated within-piece task above tokenises each 4-bar window's OWN ABC
# slice (≈4 patches) — the encoder never sees the rest of the movement. The
# WHOLE-PIECE task below instead yields the WHOLE-movement ABC patches (NO 512
# truncation) plus the per-window specs, so BPSMotifWithinPieceWholeTask can
# encode each movement ONCE (per-patch, segmenting >512), map patches→physical
# bars, and pool each window's bar-patches IN MOVEMENT CONTEXT. The metric +
# windows + labels are byte-identical to the clip-isolated task — ONLY the
# encoding context differs — so the two layer curves are directly comparable.
#
# Built offline by `build_bps_motif_within_piece.py --whole`. Each JSONL row is
# one MOVEMENT: {movement_id, abc(whole 2-voice movement), max_bar,
# windows:[{window_id, bar_start, bar_end, letters, occurrence_ids, n_bars}]}.
#
# CRITICAL: the per-item patches must NOT be truncated to 512 (movements run
# 144–1250 patches; truncating would silently drop windows). So this dataset
# does its OWN un-capped tokenisation (NO `max_patches` slice, NO padding) and
# yields a variable-length patch tensor + a JSON window-spec string. A custom
# collate keeps items as a python list (no stacking of ragged movements).


def _whole_collate(batch: list) -> list:
    """Identity collate — keep ragged whole-movement items as a python list.

    Whole movements have wildly different patch counts (144–1250), so the default
    stacking collate would fail. The probe iterates the list and encodes each
    movement separately. batch_size is therefore effectively 1-at-a-time inside
    the probe regardless of the loader's batch_size.
    """
    return batch


class _BPSMotifWithinPieceWholeABCBase(_BPSMotifABCMixin, _BPSMotifSymbolicBase):
    """Yields one WHOLE-movement item for the whole-piece-context probe.

    Returns ``(patches_full, movement_id_int, windows_json, movement_id_str)``:

    * ``patches_full`` — the WHOLE movement's patches, shape ``(P, PATCH_SIZE)``,
      UN-truncated and UN-padded (``P`` may exceed 512). Tokenised with
      ``add_special_patches=True`` then the BOS/EOS bookends stripped so the row
      index space matches the bar map (identical to the leitmotifs adapter's
      ``symbolic_patches_with_text``).
    * ``movement_id_int`` — stable sha1-hash int (the gallery group key, shared
      with the clip-isolated task via ``_movement_group_id``).
    * ``windows_json`` — JSON string of the row's ``windows`` list (rides default
      collation as a plain str; the probe parses it). Each spec carries
      ``bar_start``/``bar_end``/``letters``/``occurrence_ids``/``window_id``.
    * ``movement_id_str`` — the raw movement id (for logging / clip keys).
    """

    JSONL_TEMPLATE: str = ""  # set by subclass

    def __init__(self, split: str = "test", max_patches=None, jsonl_template=None):
        super().__init__(
            jsonl_template=jsonl_template or self.JSONL_TEMPLATE,
            split=split,
            fold_idx=0,
            max_patches=max_patches,
        )

    def _tokenise_abc_full(self, abc: str) -> torch.Tensor:
        """Whole-movement ABC → UN-truncated ``(P, PATCH_SIZE)`` patch tensor.

        Unlike ``_tokenise_abc`` this does NOT cap at ``max_patches`` and does NOT
        pad — the probe segments >512 itself. BOS/EOS bookends added by
        ``add_special_patches=True`` are stripped so row 0 == first content patch,
        matching ``_bar_of_patch_from_texts``' patch index space.
        """
        patches_list = self.patchilizer.encode(
            abc, patch_size=self.patch_size, add_special_patches=True
        )
        if len(patches_list) < 3:
            raise ValueError("Whole-movement ABC produced no non-special patches.")
        patches_list = patches_list[1:-1]  # drop BOS/EOS
        return torch.tensor(patches_list, dtype=torch.long)

    def __getitem__(self, idx: int):
        import json

        entry = self.meta[idx]
        patches_full = self._tokenise_abc_full(entry["abc"])
        movement_id_int = _movement_group_id(entry["movement_id"])
        windows_json = json.dumps(entry["windows"])
        return (
            patches_full,
            torch.tensor(movement_id_int, dtype=torch.long),
            windows_json,
            entry["movement_id"],
        )


class BPSMotifWithinPieceWholeN4ABCTest(_BPSMotifWithinPieceWholeABCBase):
    """Whole-piece-context N=4 phrase-window test split (zero-shot)."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifWithinPieceWhole.N4.ABC.jsonl"

    def __init__(self, **kwargs):
        kwargs["split"] = "test"
        super().__init__(**kwargs)


class BPSMotifWithinPieceWholeN4ABCDummy(_BPSMotifWithinPieceWholeABCBase):
    """Placeholder for the max_epochs=0 train/val dataloaders (fit is a no-op)."""

    JSONL_TEMPLATE = "data/BPS-Motif/BPSMotifWithinPieceWhole.N4.ABC.jsonl"

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)


class BPSMotifWithinPieceWholeABCDataModule(BaseDataModule):
    """Thin wrapper for the whole-piece-context within-piece ABC variant.

    Overrides the dataloaders to use the ragged-list identity collate at
    ``batch_size=1``: whole movements have wildly different patch counts
    (144–1250) so the default stacking collate would fail, and the probe encodes
    each movement separately anyway. ``num_workers=0`` keeps the CLaMP3
    patchilizer init (lazy-imported in ``_BPSMotifSymbolicBase.__init__``) in the
    main process and avoids re-pickling the big patch tensors across workers.
    """

    def _whole_loader(self, dataset):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_whole_collate,
        )

    def train_dataloader(self):
        return self._whole_loader(self.train_dataset)

    def val_dataloader(self):
        return self._whole_loader(self.val_dataset)

    def test_dataloader(self):
        return self._whole_loader(self.test_dataset)
