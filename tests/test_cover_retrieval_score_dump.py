"""Probe-level integration test for the opt-in score-dump + variation-controlled
grid on CoverRetrievalTask (VGMIDITVar-style paths encoding a variation index).

Uses the same __new__-fixture pattern as test_cover_retrieval_integration.py.
"""

from __future__ import annotations

import torch

from marble.tasks.Covers80.probe import CoverRetrievalTask

# 2 works x 2 variations x 2 GM programs. Filename: {piece}_{SECTION}_{idx}_p{prog}.
FILES = [
    ("P0_A_0_p0.wav", 0, 0),
    ("P0_A_0_p1.wav", 0, 1),
    ("P0_A_1_p0.wav", 0, 0),
    ("P0_A_1_p1.wav", 0, 1),
    ("P1_B_0_p0.wav", 1, 0),
    ("P1_B_0_p1.wav", 1, 1),
    ("P1_B_1_p0.wav", 1, 0),
    ("P1_B_1_p1.wav", 1, 1),
]


def _build_task(files):
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)
    task._test_embeddings = []
    task._test_work_ids = []
    task._test_paths = []
    task._test_conditions = []
    task.log_extended_retrieval_metrics = False
    task.metric_device = "cpu"  # force non-streaming so the varctl grid runs
    task.dump_retrieval_scores = True
    task.dump_scores_n_bins = 20
    task.variation_id_regex = r"_(?P<section>[A-Z]+)_(?P<idx>\d+)(?:_p\d+)?$"
    task.require_different_variation = True

    torch.manual_seed(0)
    embs = torch.randn(len(files), 16)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    for i, (path, wid, cond) in enumerate(files):
        task._test_embeddings.append(embs[i : i + 1])
        task._test_work_ids.append(torch.tensor([wid]))
        task._test_paths.append(path)
        task._test_conditions.append(torch.tensor([cond]))

    captured: dict[str, float] = {}
    task.log = lambda k, v, **kw: captured.__setitem__(k, float(v))
    return task, captured


def test_score_dump_and_variation_controlled_metrics_logged():
    task, log = _build_task(FILES)
    task.on_test_epoch_end()
    # Score-distribution summary scalars.
    assert "test/score_sep_overall" in log
    assert "test/score_sep_within" in log
    assert "test/score_sep_cross" in log
    # Variation-controlled condition grid.
    assert "test/map_cross_condition_varctl" in log
    assert "test/condition_gap_varctl" in log
    # The normal (confounded) condition grid is still logged for comparison.
    assert "test/map_cross_condition" in log
    assert "test/condition_gap" in log


def test_features_off_by_default_no_new_keys():
    # Same data, but the new flags default-off (not set) → none of the new keys.
    task = CoverRetrievalTask.__new__(CoverRetrievalTask)
    task._test_embeddings = []
    task._test_work_ids = []
    task._test_paths = []
    task._test_conditions = []
    task.log_extended_retrieval_metrics = False
    task.metric_device = "cpu"
    torch.manual_seed(0)
    embs = torch.randn(len(FILES), 16)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    for i, (path, wid, cond) in enumerate(FILES):
        task._test_embeddings.append(embs[i : i + 1])
        task._test_work_ids.append(torch.tensor([wid]))
        task._test_paths.append(path)
        task._test_conditions.append(torch.tensor([cond]))
    log: dict[str, float] = {}
    task.log = lambda k, v, **kw: log.__setitem__(k, float(v))
    task.on_test_epoch_end()
    assert "test/map_cross_condition" in log  # normal grid still runs
    assert "test/score_sep_overall" not in log
    assert "test/map_cross_condition_varctl" not in log
