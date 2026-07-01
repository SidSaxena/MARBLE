"""Per-cell cosine-score distributions for retrieval tasks.

``CoverRetrievalTask`` scores retrieval by cosine *ranking* (MAP) and discards
the raw similarities. This module accumulates, per (query_condition,
target_condition) cell, the distribution of candidate cosine scores split into
RELEVANT (same ``work_id``) vs DISTRACTOR (different ``work_id``), self-excluded —
so one can inspect the score distribution and the relevant-vs-distractor
separation, not only the aggregate MAP.

Streaming: :meth:`RetrievalScoreAccumulator.update` takes a batch of query rows,
so it works at N≈100k without materialising the full N×N similarity matrix. Only
fixed-size histograms + running moments are kept.
"""

from __future__ import annotations

import numpy as np
import torch


def _new_pool(n_bins: int) -> dict:
    return {
        "n": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "max": float("-inf"),
        "min": float("inf"),
        "hist": np.zeros(n_bins, dtype=np.int64),
    }


class RetrievalScoreAccumulator:
    """Accumulate relevant/distractor cosine-score histograms + moments per cell."""

    def __init__(
        self,
        work_ids,
        conditions=None,
        *,
        n_bins: int = 50,
        lo: float = -1.0,
        hi: float = 1.0,
    ):
        self.work = torch.as_tensor(work_ids)
        self.cond = None if conditions is None else torch.as_tensor(conditions)
        self.n_bins = int(n_bins)
        self.edges = np.linspace(lo, hi, self.n_bins + 1)
        self._unique_conds = (
            [int(c) for c in torch.unique(self.cond).tolist()] if self.cond is not None else []
        )
        # key -> {"rel": pool, "distr": pool}; key is "overall" or (qc, tc).
        self._cells: dict = {}

    def _cell(self, key) -> dict:
        if key not in self._cells:
            self._cells[key] = {"rel": _new_pool(self.n_bins), "distr": _new_pool(self.n_bins)}
        return self._cells[key]

    def _add(self, pool: dict, scores: torch.Tensor) -> None:
        if scores.numel() == 0:
            return
        s = scores.detach().to(torch.float64).cpu().numpy()
        pool["n"] += int(s.size)
        pool["sum"] += float(s.sum())
        pool["sumsq"] += float((s * s).sum())
        pool["max"] = max(pool["max"], float(s.max()))
        pool["min"] = min(pool["min"], float(s.min()))
        pool["hist"] += np.histogram(s, bins=self.edges)[0]

    def update(self, query_idx, sim_rows) -> None:
        """Process a batch of query rows. ``query_idx`` = the global row indices
        (length B); ``sim_rows`` = ``(B, N)`` similarities for those queries."""
        sim_rows = torch.as_tensor(sim_rows)
        n = self.work.shape[0]
        for bi, qi in enumerate(query_idx):
            qi = int(qi)
            row = sim_rows[bi]
            not_self = torch.ones(n, dtype=torch.bool)
            not_self[qi] = False
            wq = self.work[qi]
            same_work = self.work == wq
            ov = self._cell("overall")
            self._add(ov["rel"], row[same_work & not_self])
            self._add(ov["distr"], row[(~same_work) & not_self])
            if self.cond is not None:
                qc = int(self.cond[qi])
                for tc in self._unique_conds:
                    tc_mask = (self.cond == tc) & not_self
                    cell = self._cell((qc, tc))
                    self._add(cell["rel"], row[same_work & tc_mask])
                    self._add(cell["distr"], row[(~same_work) & tc_mask])

    def _finalize(self, pool: dict) -> dict:
        n = pool["n"]
        mean = pool["sum"] / n if n else 0.0
        var = (pool["sumsq"] / n - mean * mean) if n else 0.0
        return {
            "n": n,
            "mean": float(mean),
            "std": float(np.sqrt(max(var, 0.0))),
            "max": float(pool["max"]) if n else 0.0,
            "min": float(pool["min"]) if n else 0.0,
            "hist": pool["hist"].tolist(),
            "edges": self.edges.tolist(),
        }

    def result(self) -> dict:
        finalized = {}
        for key, cell in self._cells.items():
            rel = self._finalize(cell["rel"])
            distr = self._finalize(cell["distr"])
            finalized[key] = {
                "relevant": rel,
                "distractor": distr,
                "separation": rel["mean"] - distr["mean"],
            }
        empty = {
            "relevant": self._finalize(_new_pool(self.n_bins)),
            "distractor": self._finalize(_new_pool(self.n_bins)),
            "separation": 0.0,
        }
        return {
            "overall": finalized.get("overall", empty),
            "cells": {k: v for k, v in finalized.items() if k != "overall"},
        }


def score_distributions(
    sim, work_ids, conditions=None, *, n_bins: int = 50, lo=-1.0, hi=1.0
) -> dict:
    """Whole-matrix convenience wrapper (feeds every row through the accumulator).

    For large N use :class:`RetrievalScoreAccumulator` with batched rows instead.
    """
    acc = RetrievalScoreAccumulator(work_ids, conditions, n_bins=n_bins, lo=lo, hi=hi)
    sim = torch.as_tensor(sim)
    acc.update(list(range(sim.shape[0])), sim)
    return acc.result()
