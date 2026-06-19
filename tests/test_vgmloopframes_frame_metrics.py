"""tests/test_vgmloopframes_frame_metrics.py

Unit tests for the pure frame-level spec metrics ported into
marble/tasks/VGMLoopFrames/frame_metrics.py.

All tests run on SYNTHETIC numpy frame data — no audio, no encoder, no
Lightning.  They verify:
  * peak-pick + frame->seconds conversion
  * boundary-F1 = 1.0 when est lands on GT within tolerance, 0.0 when off-by-many
  * seam recall hit/miss within tolerance
  * function pairwise-F = 1.0 for identical class maps, < 1.0 / 0.x for wrong ones
"""

import numpy as np
import pytest

from marble.tasks.VGMLoopFrames.frame_metrics import (
    boundary_metrics,
    frames_to_seconds,
    function_frame_metrics,
    is_single_segment,
    peak_pick_boundaries,
    seam_recall,
)

LABEL_FREQ = 25  # Hz, matches datamodule default


def _gaussian_heatmap(length, peak_frames, sigma=1.5):
    h = np.zeros(length, dtype=np.float32)
    frames = np.arange(length, dtype=np.float32)
    for pf in peak_frames:
        h += np.exp(-0.5 * ((frames - pf) / sigma) ** 2)
    return np.clip(h, 0.0, 1.0)


# ── peak picking ──────────────────────────────────────────────────────────────


def test_peak_pick_single_bump():
    h = _gaussian_heatmap(375, [100])
    peaks = peak_pick_boundaries(h, threshold=0.5)
    assert peaks == [100]


def test_peak_pick_two_well_separated_bumps():
    h = _gaussian_heatmap(375, [50, 250])
    peaks = peak_pick_boundaries(h, threshold=0.5)
    assert peaks == [50, 250]


def test_peak_pick_empty_when_below_threshold():
    h = _gaussian_heatmap(375, [100], sigma=0.1) * 0.1  # tiny activation
    peaks = peak_pick_boundaries(h, threshold=0.5)
    assert peaks == []


def test_frames_to_seconds():
    assert frames_to_seconds([100], LABEL_FREQ) == [4.0]
    assert frames_to_seconds([0, 25, 50], LABEL_FREQ) == [0.0, 1.0, 2.0]


def test_frames_to_seconds_rejects_bad_freq():
    with pytest.raises(ValueError):
        frames_to_seconds([1], 0)


# ── T1: boundary metrics ──────────────────────────────────────────────────────


def test_boundary_f1_perfect_on_exact_match():
    # GT boundary at frame 100 -> 4.0 s.  Predicted heatmap peaks at frame 100.
    gt_frame = 100
    gt_sec = gt_frame / LABEL_FREQ  # 4.0
    h = _gaussian_heatmap(375, [gt_frame])
    est_sec = frames_to_seconds(peak_pick_boundaries(h, threshold=0.5), LABEL_FREQ)
    m = boundary_metrics([gt_sec], est_sec)
    assert m["f_0_5"] == 1.0
    assert m["f_3_0"] == 1.0


def test_boundary_f1_within_half_second_tolerance():
    # GT at 4.0 s, prediction 0.4 s away -> inside +-0.5 s window -> F1 = 1.0
    gt_sec = 4.0
    est_frame = int(round((gt_sec + 0.4) * LABEL_FREQ))
    h = _gaussian_heatmap(375, [est_frame])
    est_sec = frames_to_seconds(peak_pick_boundaries(h, threshold=0.5), LABEL_FREQ)
    m = boundary_metrics([gt_sec], est_sec)
    assert m["f_0_5"] == 1.0


def test_boundary_f1_zero_when_off_by_many():
    # GT at frame 100 (4.0 s); prediction at frame 300 (12.0 s) -> 8 s away.
    gt_sec = 100 / LABEL_FREQ
    h = _gaussian_heatmap(375, [300])
    est_sec = frames_to_seconds(peak_pick_boundaries(h, threshold=0.5), LABEL_FREQ)
    m = boundary_metrics([gt_sec], est_sec)
    assert m["f_0_5"] == 0.0
    assert m["f_3_0"] == 0.0  # 8 s gap >> 3 s tolerance


def test_boundary_f1_off_by_2s_fails_half_passes_three():
    # 2.0 s gap: fails +-0.5 s, passes +-3 s  (CLaMP3 coarse-encoder caveat)
    gt_sec = 100 / LABEL_FREQ  # 4.0
    est_frame = int(round((gt_sec + 2.0) * LABEL_FREQ))
    h = _gaussian_heatmap(375, [est_frame])
    est_sec = frames_to_seconds(peak_pick_boundaries(h, threshold=0.5), LABEL_FREQ)
    m = boundary_metrics([gt_sec], est_sec)
    assert m["f_0_5"] == 0.0
    assert m["f_3_0"] == 1.0


def test_boundary_metrics_both_empty_is_perfect():
    m = boundary_metrics([], [])
    assert m["f_0_5"] == 1.0 and m["f_3_0"] == 1.0


def test_boundary_metrics_one_empty_is_zero():
    m = boundary_metrics([4.0], [])
    assert m["f_0_5"] == 0.0 and m["f_3_0"] == 0.0


# ── T1: seam recall ───────────────────────────────────────────────────────────


def test_seam_recall_hit():
    seam_sec = 5.0
    est = [5.2]  # 0.2 s away
    m = seam_recall(seam_sec, est)
    assert m["recall_0_5"] == 1.0
    assert m["recall_3_0"] == 1.0


def test_seam_recall_miss_tight_hit_loose():
    seam_sec = 5.0
    est = [7.0]  # 2.0 s away
    m = seam_recall(seam_sec, est)
    assert m["recall_0_5"] == 0.0
    assert m["recall_3_0"] == 1.0


def test_seam_recall_no_predictions():
    m = seam_recall(5.0, [])
    assert m["recall_0_5"] == 0.0 and m["recall_3_0"] == 0.0


# ── T3: function-class metrics ────────────────────────────────────────────────


def test_function_pairwise_f_perfect_on_identical():
    ref = np.array([0] * 100 + [1] * 275)  # intro_loop with boundary at frame 100
    est = ref.copy()
    m = function_frame_metrics(ref, est)
    assert m["pairwise_f"] == 1.0
    assert m["frame_label_agreement"] == 1.0


def test_function_pairwise_f_perfect_on_all_same():
    ref = np.full(375, 1, dtype=np.int64)  # loop_from_start
    est = np.full(375, 1, dtype=np.int64)
    m = function_frame_metrics(ref, est)
    assert m["pairwise_f"] == 1.0
    assert m["frame_label_agreement"] == 1.0


def test_function_pairwise_f_label_agnostic_same_partition():
    # Same partition (boundary at 100) but est uses different class id for loop.
    ref = np.array([0] * 100 + [1] * 275)
    est = np.array([0] * 100 + [2] * 275)  # same clusters, relabeled
    m = function_frame_metrics(ref, est)
    # pairwise-F is label-agnostic -> partition identical -> 1.0
    assert m["pairwise_f"] == 1.0
    # but frame-label agreement drops (loop frames mislabeled)
    assert m["frame_label_agreement"] == pytest.approx(100 / 375)


def test_function_pairwise_f_degrades_with_wrong_boundary():
    ref = np.array([0] * 100 + [1] * 275)
    est = np.array([0] * 250 + [1] * 125)  # boundary far off
    m = function_frame_metrics(ref, est)
    assert m["pairwise_f"] < 1.0
    assert m["frame_label_agreement"] < 1.0


def test_function_pairwise_f_low_for_scrambled():
    rng = np.random.default_rng(0)
    ref = np.array([0] * 100 + [1] * 275)
    est = rng.integers(0, 3, size=375)
    m = function_frame_metrics(ref, est)
    assert m["pairwise_f"] < 0.9


def test_function_length_mismatch_raises():
    with pytest.raises(ValueError):
        function_frame_metrics(np.zeros(10), np.zeros(11))


def test_is_single_segment():
    assert is_single_segment(np.full(375, 2)) is True
    assert is_single_segment(np.array([0] * 100 + [1] * 275)) is False
    assert is_single_segment(np.array([])) is True
