"""MTC-ANN folk-melody retrieval tasks.

Two trivial subclasses of :class:`CoverRetrievalTask` (same MAP / centering /
whitening / anisotropy machinery): the datamodule encodes each occurrence's
relevance ``group`` into ``work_id`` so the standard scoring counts members of
the same group as relevant. They exist only to give each WandB run a distinct
``task`` tag and a stable LightningCLI import path — exactly like
:class:`marble.tasks.JKUPDDRetrieval.probe.JKUPDDRetrievalTask`.

* :class:`MTCANNTuneFamilyTask` — relevance = same tune family.
* :class:`MTCANNMotifTask`      — relevance = same annotated motif.
"""

from __future__ import annotations

from marble.tasks.Covers80.probe import CoverRetrievalTask


class MTCANNTuneFamilyTask(CoverRetrievalTask):
    pass


class MTCANNMotifTask(CoverRetrievalTask):
    pass
