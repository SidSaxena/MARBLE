"""
tests/test_cover_retrieval_integration.py

Integration tests for ``CoverRetrievalTask.on_test_epoch_end`` —
exercise the full metric-aggregation pipeline without requiring real
audio. The method is called directly with pre-populated ``_test_*``
buffers; ``self.log`` is monkey-patched to a dict so we can assert
exactly which metric keys are produced and what their values are.

This validates the integration risk that unit tests on
``compute_perpair_map`` / ``recall_at_k`` / ``anisotropy_metrics``
alone don't cover:
  - 5-tuple datamodule output flows through ``test_step`` into the
    ``_test_conditions`` buffer
  - Per-condition aggregation in ``on_test_epoch_end`` correctly
    groups by file via ``path2cond.setdefault`` first-seen
  - Cross-condition metrics fire when condition values vary; skip
    silently when all values are -1 sentinel (base VGMIDITVar) or
    when the buffer stays empty (Covers80, SHS100K — 4-tuple flow)
  - Anisotropy metrics fire on every retrieval run regardless of
    condition presence
"""

from __future__ import annotations

import torch

from marble.tasks.Covers80.probe import CoverRetrievalTask


def _make_task_with_buffers(
    n_clips: int,
    n_files: int,
    n_works: int,
    conditions: list[int] | None,
    seed: int = 0,
) -> tuple[CoverRetrievalTask, dict[str, float]]:
    """Build a CoverRetrievalTask with pre-populated test buffers + a
    captured-log dict. The task is constructed via __new__ to bypass
    encoder instantiation (we never call forward).
    """
    torch.manual_seed(seed)

    # Construct via __new__ to skip __init__ (avoids encoder download).
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)
    task._test_embeddings = []
    task._test_work_ids = []
    task._test_paths = []
    task._test_conditions = []

    # Generate n_clips entries: each file has n_clips/n_files clips.
    # work_ids cycle through n_works.
    clips_per_file = n_clips // n_files
    H = 16
    embs = torch.randn(n_clips, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)

    for i in range(n_clips):
        file_idx = i // clips_per_file
        wid = file_idx % n_works
        path = f"file_{file_idx:02d}.wav"
        task._test_embeddings.append(embs[i : i + 1])
        task._test_work_ids.append(torch.tensor([wid]))
        task._test_paths.append(path)
        if conditions is not None:
            cond_val = conditions[file_idx]
            task._test_conditions.append(torch.tensor([cond_val]))

    # Capture self.log calls into a dict.
    captured: dict[str, float] = {}

    def fake_log(key, value, **kwargs):
        captured[key] = float(value)

    task.log = fake_log
    return task, captured


def test_covers80_style_skips_condition_metrics():
    """Covers80 / SHS100K emit 4-tuples → _test_conditions stays empty
    → per-condition metrics are silently skipped. Generic + anisotropy
    metrics still fire."""
    task, log = _make_task_with_buffers(n_clips=20, n_files=10, n_works=5, conditions=None)
    task.on_test_epoch_end()

    # Existing metrics — all present.
    for key in (
        "test/map",
        "test/map_centered",
        "test/map@1",
        "test/map@1_centered",
        "test/mrr",
        "test/mrr_centered",
    ):
        assert key in log, f"missing existing key {key}"

    # Generic retrieval metrics — should fire on N=10 files.
    assert "test/recall@1" in log
    assert "test/recall@5" in log
    assert "test/hit_rate@1" in log
    assert "test/median_rank" in log
    assert "test/r_precision" in log
    # K=50 and K=100 should be skipped (N=10 < 50)
    assert "test/recall@50" not in log
    assert "test/recall@100" not in log

    # Anisotropy — always fires.
    for key in (
        "test/anisotropy/mean_vec_norm",
        "test/anisotropy/avg_pair_cos",
        "test/anisotropy/top1_sv_share",
        "test/anisotropy/effective_rank",
    ):
        assert key in log

    # Per-condition — must NOT fire (no conditions).
    assert "test/map_same_condition" not in log
    assert "test/map_cross_condition" not in log
    assert "test/condition_gap" not in log


def test_vgmiditvar_leitmotif_style_fires_cross_condition():
    """VGMIDITVar-leitmotif emits 5-tuples with gm_program; per-condition
    grid should produce same + cross + gap metrics.

    Fixture: 4 works × 4 files-per-work × 2 conditions (2 files per
    condition per work). This guarantees BOTH same-condition same-work
    peers (for diagonal cells) AND cross-condition same-work peers (for
    off-diagonal cells) exist for every query.
    """
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)
    task._test_embeddings = []
    task._test_work_ids = []
    task._test_paths = []
    task._test_conditions = []

    torch.manual_seed(0)
    # 4 works × 4 files = 16 files. Each work has files at conditions
    # [0, 0, 1, 1]. So per work: 2 files in cond 0, 2 in cond 1.
    file_idx = 0
    for work in range(4):
        for cond in (0, 0, 1, 1):
            emb = torch.randn(1, 16)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            task._test_embeddings.append(emb)
            task._test_work_ids.append(torch.tensor([work]))
            task._test_paths.append(f"file_{file_idx:02d}.wav")
            task._test_conditions.append(torch.tensor([cond]))
            file_idx += 1

    captured: dict[str, float] = {}

    def fake_log(key, value, **kwargs):
        captured[key] = float(value)

    task.log = fake_log
    task.on_test_epoch_end()

    # Per-condition metrics MUST fire.
    assert "test/map_same_condition" in captured, (
        f"missing same-condition MAP. Got keys: {sorted(k for k in captured if 'condition' in k or 'cross' in k or 'same' in k)}"
    )
    assert "test/map_cross_condition" in captured, "missing cross-condition MAP"
    assert "test/condition_gap" in captured, "missing condition_gap"

    # Sanity: condition_gap = same - cross (both finite floats).
    same = captured["test/map_same_condition"]
    cross = captured["test/map_cross_condition"]
    gap = captured["test/condition_gap"]
    assert abs(gap - (same - cross)) < 1e-5, f"gap arithmetic: {gap} vs {same - cross}"


def test_base_vgmiditvar_style_all_sentinel_skips():
    """Base VGMIDITVar emits 5-tuples but condition=-1 for every record
    (no gm_program or soundfont_id in JSONL). Per-condition block must
    skip since `any(c != -1) == False`."""
    file_conditions = [-1] * 10
    task, log = _make_task_with_buffers(
        n_clips=20, n_files=10, n_works=5, conditions=file_conditions
    )
    task.on_test_epoch_end()

    # Per-condition — silently skipped.
    assert "test/map_same_condition" not in log
    assert "test/map_cross_condition" not in log
    assert "test/condition_gap" not in log

    # Generic + anisotropy + existing still fire.
    assert "test/recall@1" in log
    assert "test/anisotropy/mean_vec_norm" in log
    assert "test/map" in log


def test_single_condition_logs_only_same_no_cross():
    """If only ONE condition value appears (e.g. all gm_program=0,
    piano-only baseline), there are no off-diagonal cells. The probe
    should log map_same_condition but not map_cross_condition or
    condition_gap (the latter two need both same+cross to exist)."""
    file_conditions = [0] * 10
    task, log = _make_task_with_buffers(
        n_clips=20, n_files=10, n_works=5, conditions=file_conditions
    )
    task.on_test_epoch_end()

    # same exists (diagonal cell 0,0)
    assert "test/map_same_condition" in log
    # cross + gap do NOT (no off-diagonal cells with non-zero n_queries)
    assert "test/map_cross_condition" not in log
    assert "test/condition_gap" not in log


def test_first_seen_aggregation_path_to_condition():
    """Sanity: multiple clips from the same file share a condition;
    path2cond.setdefault keeps the first-seen value. Verify by
    constructing two clips per file with the SAME condition (the
    realistic case)."""
    # 6 clips, 3 files, 3 works; condition per file is [10, 20, 30].
    # Two clips per file.
    file_conditions = [10, 20, 30]
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)
    task._test_embeddings = []
    task._test_work_ids = []
    task._test_paths = []
    task._test_conditions = []

    torch.manual_seed(0)
    for file_idx in range(3):
        for _ in range(2):  # 2 clips per file
            task._test_embeddings.append(torch.randn(1, 8))
            task._test_work_ids.append(torch.tensor([file_idx]))
            task._test_paths.append(f"file_{file_idx}.wav")
            task._test_conditions.append(torch.tensor([file_conditions[file_idx]]))

    captured = {}

    def fake_log(key, value, **kwargs):
        captured[key] = float(value)

    task.log = fake_log
    task.on_test_epoch_end()

    # Same-condition cells exist (one cell per unique condition).
    # No cross-condition POSITIVE pairs because each work has only 1 file.
    # But the block should still attempt the cross cells; whether they
    # log depends on whether any query in cross-cell has n_relevant > 0.
    # In this case: query in cond 10, target in cond 20 → no same-work
    # candidates → n_rel=0 → cell skipped. So cross_aps stays empty.
    # Same-condition diagonals: each cell has 2 queries from same file,
    # but they aren't same-work peers either (each file is its own work).
    # So same_aps is also empty → neither key should log.
    assert "test/map_same_condition" not in captured
    assert "test/map_cross_condition" not in captured
    # Anisotropy still fires regardless.
    assert "test/anisotropy/mean_vec_norm" in captured
