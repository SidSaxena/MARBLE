# marble/tasks/SHS100K/probe.py
"""
SHS-100K cover-song retrieval probe.

Zero-shot evaluation only (no probe training, max_epochs=0).
Reuses CoverRetrievalTask from Covers80 — the MAP computation is identical.

The test split has ~5,000 tracks across 500 works (~10 versions per work).
Evaluation:
  1. Embed every 30-second clip of every track.
  2. Mean-pool clip embeddings per file → L2-normalise → one vector per track.
  3. Build a (N × N) cosine-similarity matrix.
  4. For each query, rank the N-1 other tracks; AP = sum(P@k × rel_k) / n_relevant.
  5. MAP = mean(AP) over all N queries.
"""

from marble.tasks.Covers80.probe import CoverRetrievalTask  # noqa: F401 — re-export

__all__ = ["CoverRetrievalTask"]
