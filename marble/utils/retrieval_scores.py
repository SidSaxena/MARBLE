"""Per-cell cosine-score distributions for retrieval tasks.

``CoverRetrievalTask`` scores retrieval by cosine *ranking* (MAP) and discards
the raw similarities. This module accumulates, per (query_condition,
target_condition) cell, the distribution of candidate cosine scores split into
RELEVANT (same ``work_id``) vs DISTRACTOR (different ``work_id``), self-excluded —
so one can inspect the score distribution and the relevant-vs-distractor
separation, not only the aggregate MAP.

Variation control (VGMIDITVar twin confound): when ``variations`` is supplied,
the RELEVANT pool is split into

  * ``relevant``          — all same-work candidates (the *confounded* pool, which
                            in a cross-condition cell contains the query's own
                            composition re-rendered in the target timbre — an
                            audio near-duplicate "twin"), and
  * ``relevant_diffvar``  — same-work candidates with a *different* variation id,
                            i.e. the twins removed.

so ``separation`` (confounded) and ``separation_varctl`` (twins removed) can be
compared directly — mirroring the ``condition_gap`` vs ``condition_gap_varctl``
split of the MAP grid. Without ``variations`` the two coincide.

Empty pools: a cell whose RELEVANT pool is empty (e.g. a work with no same-work
peer at that condition) reports ``separation = None`` rather than a bogus
``0.0 - distractor_mean``, so callers cannot silently average garbage in.

Sentinel conditions: condition ``-1`` (the "unparsed" sentinel used across the
probe) is excluded from cell enumeration, matching the MAP grid's ``c != -1``
filter, so the two diagnostics score the same cell universe.

Streaming: :meth:`RetrievalScoreAccumulator.update` takes a batch of query rows
and processes them with vectorised tensor masks (no per-query Python loop), so it
works at N≈100k without materialising the full N×N similarity matrix. Only
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


def _merge_pools(a: dict, b: dict) -> dict:
    """Additive combine of two moment/histogram pools (RELEVANT = diffvar ∪ twin)."""
    return {
        "n": a["n"] + b["n"],
        "sum": a["sum"] + b["sum"],
        "sumsq": a["sumsq"] + b["sumsq"],
        "max": max(a["max"], b["max"]),
        "min": min(a["min"], b["min"]),
        "hist": a["hist"] + b["hist"],
    }


class RetrievalScoreAccumulator:
    """Accumulate relevant/distractor cosine-score histograms + moments per cell.

    The RELEVANT side is tracked as two sub-pools — ``rel_diff`` (different
    variation) and ``rel_twin`` (same-(work, variation) near-duplicate) — so both
    the confounded and the variation-controlled separation are recoverable in a
    single pass. ``rel_twin`` is only ever populated when ``variations`` is given.
    """

    def __init__(
        self,
        work_ids,
        conditions=None,
        variations=None,
        *,
        n_bins: int = 50,
        lo: float = -1.0,
        hi: float = 1.0,
        exclude_condition: int | None = -1,
    ):
        self.work = torch.as_tensor(work_ids)
        self.cond = None if conditions is None else torch.as_tensor(conditions)
        self.var = None if variations is None else torch.as_tensor(variations)
        self.n_bins = int(n_bins)
        self.edges = np.linspace(lo, hi, self.n_bins + 1)
        # Cell enumeration excludes the ``exclude_condition`` sentinel (default -1)
        # so cells match the MAP grid, which filters ``c != -1``.
        if self.cond is not None:
            uc = [int(c) for c in torch.unique(self.cond).tolist()]
            self._unique_conds = [
                c for c in uc if exclude_condition is None or c != exclude_condition
            ]
        else:
            self._unique_conds = []
        # key -> {"rel_diff": pool, "rel_twin": pool, "distr": pool}; key "overall" or (qc, tc).
        self._cells: dict = {}

    def _cell(self, key) -> dict:
        if key not in self._cells:
            self._cells[key] = {
                "rel_diff": _new_pool(self.n_bins),
                "rel_twin": _new_pool(self.n_bins),
                "distr": _new_pool(self.n_bins),
            }
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
        # Clip into [lo, hi] so out-of-range scores land in the edge bins instead of
        # being dropped by np.histogram — keeps sum(hist) == n. Moments above use the
        # true (unclipped) values.
        clipped = np.clip(s, self.edges[0], self.edges[-1])
        pool["hist"] += np.histogram(clipped, bins=self.edges)[0]

    def update(self, query_idx, sim_rows) -> None:
        """Process a batch of query rows with vectorised masks.

        ``query_idx`` = the global row indices (length B); ``sim_rows`` = ``(B, N)``
        similarities for those queries. No per-query Python loop, so this scales to
        N≈100k when fed in row-chunks.
        """
        sim = torch.as_tensor(sim_rows)
        if sim.ndim == 1:
            sim = sim.unsqueeze(0)
        qidx = torch.as_tensor([int(x) for x in query_idx])
        b, n = sim.shape
        rows = torch.arange(b)

        wq = self.work[qidx].unsqueeze(1)  # (B, 1)
        same_work = self.work.unsqueeze(0) == wq  # (B, N)
        not_self = torch.ones(b, n, dtype=torch.bool)
        not_self[rows, qidx] = False
        same_work = same_work & not_self
        if self.var is not None:
            vq = self.var[qidx].unsqueeze(1)  # (B, 1)
            twin = same_work & (self.var.unsqueeze(0) == vq)  # (B, N)
        else:
            twin = torch.zeros(b, n, dtype=torch.bool)
        rel_diff = same_work & ~twin
        distr = (~same_work) & not_self

        ov = self._cell("overall")
        self._add(ov["rel_diff"], sim[rel_diff])
        self._add(ov["rel_twin"], sim[twin])
        self._add(ov["distr"], sim[distr])

        if self.cond is not None and self._unique_conds:
            qc = self.cond[qidx]  # (B,)
            for tc in self._unique_conds:
                col = (self.cond == tc).unsqueeze(0)  # (1, N)
                for qcv in self._unique_conds:
                    rowm = (qc == qcv).unsqueeze(1)  # (B, 1)
                    if not bool(rowm.any()):
                        continue
                    cell = self._cell((qcv, tc))
                    sel = rowm & col
                    self._add(cell["rel_diff"], sim[rel_diff & sel])
                    self._add(cell["rel_twin"], sim[twin & sel])
                    self._add(cell["distr"], sim[distr & sel])

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

    def _finalize_cell(self, cell: dict) -> dict:
        rel_all = self._finalize(_merge_pools(cell["rel_diff"], cell["rel_twin"]))
        rel_diff = self._finalize(cell["rel_diff"])
        distr = self._finalize(cell["distr"])
        # separation is None for an empty RELEVANT (or DISTRACTOR) pool so callers
        # can't average a bogus ``0.0 - distractor_mean`` in.
        sep = rel_all["mean"] - distr["mean"] if rel_all["n"] > 0 and distr["n"] > 0 else None
        sep_ctl = rel_diff["mean"] - distr["mean"] if rel_diff["n"] > 0 and distr["n"] > 0 else None
        return {
            "relevant": rel_all,
            "distractor": distr,
            "separation": sep,
            "relevant_diffvar": rel_diff,
            "separation_varctl": sep_ctl,
        }

    def result(self) -> dict:
        finalized = {key: self._finalize_cell(cell) for key, cell in self._cells.items()}
        empty = self._finalize_cell(
            {
                "rel_diff": _new_pool(self.n_bins),
                "rel_twin": _new_pool(self.n_bins),
                "distr": _new_pool(self.n_bins),
            }
        )
        return {
            "overall": finalized.get("overall", empty),
            "cells": {k: v for k, v in finalized.items() if k != "overall"},
        }


def score_distributions(
    sim, work_ids, conditions=None, variations=None, *, n_bins: int = 50, lo=-1.0, hi=1.0
) -> dict:
    """Whole-matrix convenience wrapper (feeds every row through the accumulator).

    For large N use :class:`RetrievalScoreAccumulator` with batched rows instead.
    """
    acc = RetrievalScoreAccumulator(work_ids, conditions, variations, n_bins=n_bins, lo=lo, hi=hi)
    sim = torch.as_tensor(sim)
    acc.update(list(range(sim.shape[0])), sim)
    return acc.result()
