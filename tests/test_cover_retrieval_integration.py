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
    task.log_extended_retrieval_metrics = False

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


def test_extended_flag_logs_full_metric_set():
    """log_extended_retrieval_metrics=True restores the pre-trim metric
    set (map@1, mrr, full K range, hit_rate, centered duplicates, all 4
    anisotropy keys). Verifies the flag actually toggles behavior."""
    task, log = _make_task_with_buffers(n_clips=40, n_files=20, n_works=5, conditions=None)
    task.log_extended_retrieval_metrics = True
    task.on_test_epoch_end()

    for key in (
        "test/map@1",
        "test/map@1_centered",
        "test/mrr",
        "test/mrr_centered",
        "test/recall@1",
        "test/recall@5",
        "test/recall@10",
        "test/recall@10_centered",
        "test/hit_rate@1",
        "test/hit_rate@5",
        "test/hit_rate@10",
        "test/median_rank_centered",
        "test/r_precision_centered",
        "test/anisotropy/avg_pair_cos",
        "test/anisotropy/top1_sv_share",
    ):
        assert key in log, f"missing extended key {key}"


def test_covers80_style_skips_condition_metrics():
    """Covers80 / SHS100K emit 4-tuples → _test_conditions stays empty
    → per-condition metrics are silently skipped. Generic + anisotropy
    metrics still fire."""
    task, log = _make_task_with_buffers(n_clips=40, n_files=20, n_works=5, conditions=None)
    task.on_test_epoch_end()

    # Headline trim set (default — log_extended_retrieval_metrics=False).
    for key in ("test/map", "test/map_centered"):
        assert key in log, f"missing headline key {key}"

    # Headline secondary metrics — raw only, no _centered duplicates.
    assert "test/recall@10" in log
    assert "test/median_rank" in log
    assert "test/r_precision" in log

    # Extended set MUST NOT fire under default flag.
    for absent in (
        "test/map@1",
        "test/map@1_centered",
        "test/mrr",
        "test/mrr_centered",
        "test/recall@1",
        "test/recall@5",
        "test/recall@50",
        "test/recall@100",
        "test/recall@10_centered",
        "test/hit_rate@1",
        "test/hit_rate@5",
        "test/hit_rate@10",
        "test/median_rank_centered",
        "test/r_precision_centered",
    ):
        assert absent not in log, f"unexpected extended key {absent}"

    # All four anisotropy metrics fire by default. (Previously
    # avg_pair_cos + top1_sv_share were gated behind the extended flag
    # — moved out 2026-05-28 after an audit showed they're cheap and
    # serve as independent cross-checks on mean_vec_norm.)
    for key in (
        "test/anisotropy/mean_vec_norm",
        "test/anisotropy/effective_rank",
        "test/anisotropy/avg_pair_cos",
        "test/anisotropy/top1_sv_share",
    ):
        assert key in log, f"missing anisotropy key {key}"

    # Per-condition — must NOT fire (no conditions).
    assert "test/map_same_condition" not in log
    assert "test/map_cross_condition" not in log
    assert "test/condition_gap" not in log


def test_vgmiditvar_timbre_style_fires_cross_condition():
    """VGMIDITVar-timbre emits 5-tuples with gm_program; per-condition
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
    task.log_extended_retrieval_metrics = False

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

    # Per-cell logging: 2 conditions → 4 cells, each yielding map_grid/q_to_t
    # plus map_grid/q_to_t_n (n_queries). 8 keys total when both conditions
    # contribute to every cell (this fixture guarantees that).
    grid_keys = sorted(k for k in captured if k.startswith("test/map_grid/"))
    assert "test/map_grid/0_to_0" in grid_keys
    assert "test/map_grid/0_to_1" in grid_keys
    assert "test/map_grid/1_to_0" in grid_keys
    assert "test/map_grid/1_to_1" in grid_keys
    # Cells in the grid match the aggregate same/cross stats they feed.
    diag = (captured["test/map_grid/0_to_0"] + captured["test/map_grid/1_to_1"]) / 2
    off = (captured["test/map_grid/0_to_1"] + captured["test/map_grid/1_to_0"]) / 2
    assert abs(diag - same) < 1e-5, f"diag-mean {diag} != same {same}"
    assert abs(off - cross) < 1e-5, f"off-mean {off} != cross {cross}"


def test_dump_condition_grid_artifacts_uses_ascii_only_runtime_prints():
    """On Windows, Python's default stdout encoding is cp1252 and wandb's
    console-capture wrapper re-encodes through it. A single non-ASCII
    character in a runtime print() (e.g. U+2192 right-arrow) raises
    UnicodeEncodeError that propagates up through Lightning's
    on_test_epoch_end hook and kills the run AFTER metrics have already
    logged.

    This test scans probe.py for non-ASCII characters in runtime
    print/log statements (NOT in comments or docstrings) inside
    ``_dump_condition_grid_artifacts*`` and the on_test_epoch_end body.
    Comments/docstrings can use whatever Unicode they want; only actual
    runtime byte streams must stay ASCII.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "marble" / "tasks" / "Covers80" / "probe.py"
    text = src.read_text(encoding="utf-8")

    # Find every `print(...)` call (single-line; multi-line prints
    # don't exist in this file at time of writing). For each, extract
    # the format string and assert it's ASCII.
    print_calls = re.findall(r"print\(([^)]*)\)", text)
    bad = []
    for call in print_calls:
        for ch in call:
            if ord(ch) > 127:
                bad.append((ch, call.strip()[:80]))
                break
    assert not bad, (
        f"Non-ASCII character(s) in print() runtime calls (would crash on "
        f"Windows cp1252 stdout): {bad[:3]}"
    )


def test_dump_condition_grid_artifacts_never_propagates():
    """The outer wrapper must swallow ANY exception from the inner
    write path -- artefact failures are side-channel, not load-bearing.
    Force the inner method to raise and confirm the outer returns
    cleanly without re-raising.
    """
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)

    def boom(*a, **kw):
        raise RuntimeError("simulated artefact-writer crash")

    task._dump_condition_grid_artifacts_inner = boom  # type: ignore[method-assign]
    # Should NOT raise. If it does, the assertion in the test framework
    # fails and we know the defensive wrapper regressed.
    task._dump_condition_grid_artifacts([0, 1], {(0, 0): (0.5, 10), (0, 1): (0.3, 10)})


def test_base_vgmiditvar_style_all_sentinel_skips():
    """Base VGMIDITVar emits 5-tuples but condition=-1 for every record
    (no gm_program or soundfont_id in JSONL). Per-condition block must
    skip since `any(c != -1) == False`."""
    file_conditions = [-1] * 20
    task, log = _make_task_with_buffers(
        n_clips=40, n_files=20, n_works=5, conditions=file_conditions
    )
    task.on_test_epoch_end()

    # Per-condition — silently skipped.
    assert "test/map_same_condition" not in log
    assert "test/map_cross_condition" not in log
    assert "test/condition_gap" not in log

    # Trim-default headline still fires.
    assert "test/recall@10" in log
    assert "test/anisotropy/mean_vec_norm" in log
    assert "test/map" in log
    # Extended-only key absent under default trim.
    assert "test/recall@1" not in log


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
    task.log_extended_retrieval_metrics = False

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
