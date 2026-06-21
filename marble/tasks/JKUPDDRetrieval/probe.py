"""JKUPDD within-piece motif retrieval task.

Trivial subclass of :class:`CoverRetrievalTask` (same MAP / centering /
whitening / anisotropy machinery): the datamodule encodes each occurrence's
``(piece, annotator, pattern)`` group into ``work_id`` so the standard scoring
counts occurrences of the same annotated pattern as relevant. Exists only to
give the WandB run a distinct ``task`` tag and a stable LightningCLI import path.
"""

from __future__ import annotations

from marble.tasks.Covers80.probe import CoverRetrievalTask


class JKUPDDRetrievalTask(CoverRetrievalTask):
    pass
