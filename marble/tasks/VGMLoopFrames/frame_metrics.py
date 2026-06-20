# marble/tasks/VGMLoopFrames/frame_metrics.py
#
# Pure, importable frame-level scoring helpers for the VGMLoopFrames probes.
#
# These functions take the SAME tensors the probe already has in its test step:
#   - boundary heatmap : (L,) float  per-clip, label_freq frames/sec
#   - function classes  : (L,) int    per-clip, 0=intro 1=loop 2=through
#
# and convert them to the seconds-based, mir_eval-backed spec metrics defined in
# msa-vgm/src/msa_compare/vgm/eval.py (boundary_metrics / seam_recall /
# segment_label_metrics).  No torch, no I/O, no Lightning — so they unit-test on
# synthetic numpy arrays.
#
# Frame -> seconds mapping (from VGMLoopFrames/datamodule.py):
#   within-clip time of frame i  =  i / label_freq   seconds
#   label_freq defaults to 25 Hz; L = label_freq * clip_seconds.

from __future__ import annotations

import numpy as np

try:
    import mir_eval  # noqa: F401

    _HAVE_MIR_EVAL = True
except Exception:  # pragma: no cover - mir_eval should be present
    _HAVE_MIR_EVAL = False

WINDOWS = (0.5, 3.0)


def _tag(w: float) -> str:
    return str(w).replace(".", "_")


# ---------------------------------------------------------------------------
# Peak picking: boundary heatmap (L,) -> boundary frame indices -> seconds
# ---------------------------------------------------------------------------


def peak_pick_boundaries(
    heatmap,
    threshold: float = 0.5,
    min_distance: int = 3,
) -> list[int]:
    """Pick peak frame indices from a 1-D boundary heatmap.

    A frame ``i`` is a peak when it is a strict local maximum over its
    immediate neighbours (>= on both sides, > on at least one) AND its value is
    >= ``threshold``.  Peaks closer than ``min_distance`` frames are de-duplicated
    keeping the larger one (simple greedy non-maximum suppression).

    Parameters
    ----------
    heatmap : array_like, shape (L,)
        Per-frame boundary activation in roughly [0, 1].
    threshold : float
        Minimum activation for a frame to qualify as a boundary.
    min_distance : int
        Minimum spacing (in frames) between accepted peaks.

    Returns
    -------
    list[int]
        Sorted frame indices of accepted peaks.
    """
    h = np.asarray(heatmap, dtype=np.float64).ravel()
    L = h.shape[0]
    if L == 0:
        return []

    candidates = []
    for i in range(L):
        left = h[i - 1] if i > 0 else -np.inf
        right = h[i + 1] if i < L - 1 else -np.inf
        if h[i] < threshold:
            continue
        # local maximum: not exceeded by either neighbour, and strictly greater
        # than at least one neighbour (so flat plateaus do not all fire).
        if h[i] >= left and h[i] >= right and (h[i] > left or h[i] > right):
            candidates.append(i)
        # isolated single-frame edge peak (L==1 handled too)
        elif L == 1:
            candidates.append(i)

    if not candidates:
        return []

    # Greedy NMS by descending activation, enforce min_distance.
    candidates.sort(key=lambda i: h[i], reverse=True)
    kept: list[int] = []
    for i in candidates:
        if all(abs(i - j) >= min_distance for j in kept):
            kept.append(i)
    return sorted(kept)


def frames_to_seconds(frame_indices, label_freq: float) -> list[float]:
    """Convert frame indices to within-clip seconds: ``i / label_freq``."""
    if label_freq <= 0:
        raise ValueError(f"label_freq must be > 0, got {label_freq!r}")
    return [float(i) / float(label_freq) for i in frame_indices]


# ---------------------------------------------------------------------------
# T1: boundary-detection P/R/F at tolerance windows  (ported from eval.py)
# ---------------------------------------------------------------------------


def _median_gap(ref: np.ndarray, est: np.ndarray):
    if len(ref) == 0 or len(est) == 0:
        return None
    return float(np.median([np.min(np.abs(est - r)) for r in ref]))


def boundary_metrics(ref_times, est_times, windows=WINDOWS) -> dict:
    """Boundary-detection P/R/F at each tolerance window, plus median gap.

    Ported verbatim (semantics) from msa_compare.vgm.eval.boundary_metrics.
    Empty ref AND empty est = perfect (1.0); exactly one empty = 0.0.
    """
    ref = np.array(sorted(float(t) for t in ref_times))
    est = np.array(sorted(float(t) for t in est_times))
    out: dict = {}
    for w in windows:
        if len(ref) == 0 and len(est) == 0:
            f = p = r = 1.0
        elif len(ref) == 0 or len(est) == 0:
            f = p = r = 0.0
        else:
            import mir_eval

            f, p, r = mir_eval.onset.f_measure(ref, est, window=w)
        out[f"f_{_tag(w)}"] = float(f)
        out[f"p_{_tag(w)}"] = float(p)
        out[f"r_{_tag(w)}"] = float(r)
    out["median_gap"] = _median_gap(ref, est)
    return out


def seam_recall(ref_seam, est_times, windows=WINDOWS) -> dict:
    """Did any predicted boundary land within ``w`` seconds of the loop seam?

    Ported from msa_compare.vgm.eval.seam_recall.
    """
    est = np.array([float(t) for t in est_times])
    out: dict = {}
    for w in windows:
        hit = len(est) > 0 and bool(np.min(np.abs(est - float(ref_seam))) <= w)
        out[f"recall_{_tag(w)}"] = float(hit)
    return out


# ---------------------------------------------------------------------------
# T3: per-frame function-class metrics (pairwise-F + label agreement)
# ---------------------------------------------------------------------------


def function_frame_metrics(ref_classes, est_classes) -> dict:
    """Frame-level label agreement + pairwise clustering F1 for the function task.

    This is the frame-array analogue of msa_compare.vgm.eval.segment_label_metrics:
    the upstream function samples (start,end,label) segments onto a fixed grid and
    then runs the exact same boolean co-cluster F1.  Here the probe already holds
    per-frame class arrays, so we skip the segment->grid sampling and compute the
    identical pairwise-F directly on the frame classes.

    Parameters
    ----------
    ref_classes, est_classes : array_like (L,) int
        Ground-truth / predicted per-frame class indices.

    Returns
    -------
    dict with keys
        ``frame_label_agreement`` : fraction of frames with equal labels
        ``pairwise_f``            : pairwise frame-clustering F1 (label-agnostic)
    """
    ref = np.asarray(ref_classes).ravel()
    est = np.asarray(est_classes).ravel()
    if ref.shape[0] != est.shape[0]:
        raise ValueError(f"ref/est frame-length mismatch: {ref.shape[0]} vs {est.shape[0]}")

    n = ref.shape[0]
    if n == 0:
        return {"frame_label_agreement": 1.0, "pairwise_f": 1.0}

    agreement = float(np.sum(ref == est) / n)

    ref_same = ref[:, None] == ref[None, :]  # (n,n) bool
    est_same = est[:, None] == est[None, :]  # (n,n) bool

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    ref_flat = ref_same[mask]
    est_flat = est_same[mask]

    tp = int(np.sum(ref_flat & est_flat))
    fp = int(np.sum(~ref_flat & est_flat))
    fn = int(np.sum(ref_flat & ~est_flat))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pf = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Edge case mirrored from eval.py: no positive pairs in EITHER clustering
    # (e.g. all frames distinct in both — cannot happen with finite classes — or
    # the degenerate empty grid) => trivially identical.
    if tp == 0 and fp == 0 and fn == 0:
        pf = 1.0

    return {"frame_label_agreement": float(agreement), "pairwise_f": float(pf)}


def is_single_segment(classes) -> bool:
    """True if the per-frame class array is a single contiguous label (no change)."""
    c = np.asarray(classes).ravel()
    if c.shape[0] == 0:
        return True
    return bool(np.all(c == c[0]))
