"""Probe-level integration test for the opt-in score-dump + variation-controlled
grid on CoverRetrievalTask (VGMIDITVar-style paths encoding a variation index).

Uses the same __new__-fixture pattern as test_cover_retrieval_integration.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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
    # Twin-controlled score separation (fix: the score dump must NOT silently
    # re-include the same-(work, variation) twin the varctl grid removes).
    assert "test/score_sep_cross_varctl" in log
    # Variation-controlled condition grid.
    assert "test/map_cross_condition_varctl" in log
    assert "test/condition_gap_varctl" in log
    # Surviving-query disclosure for the varctl grid (selection-bias visibility).
    assert "test/map_cross_condition_varctl_n" in log
    # The normal (confounded) condition grid is still logged for comparison.
    assert "test/map_cross_condition" in log
    assert "test/condition_gap" in log


def test_score_dump_json_and_csv_written_and_parseable(tmp_path):
    # Regression for the tuple-key json.dump TypeError: with conditions present,
    # res["cells"] is keyed by (q, t) tuples. json.dump must not choke, and BOTH
    # artifacts must land on disk. Needs a trainer.logger.save_dir so the method
    # reaches the write block (the __new__ fixture otherwise returns early).
    task, _ = _build_task(FILES)
    # ``trainer`` is a read-only LightningModule property backed by ``_trainer``;
    # set the backing field (the __new__ fixture skips nn.Module.__init__).
    task._trainer = SimpleNamespace(logger=SimpleNamespace(save_dir=str(tmp_path)))
    task._fabric = None
    task._jit_is_scripting = False
    task.on_test_epoch_end()
    jpath = tmp_path / "retrieval_score_distributions.json"
    cpath = tmp_path / "retrieval_score_summary.csv"
    assert jpath.exists(), "score-distribution JSON was not written (tuple-key TypeError?)"
    assert cpath.exists(), "score summary CSV was not written"
    obj = json.loads(jpath.read_text())  # must be valid JSON
    assert "overall" in obj and "cells" in obj
    # Cells present (conditions given) and keyed by strings, not tuples.
    assert obj["cells"], "expected per-condition cells"
    assert all(isinstance(k, str) and "_to_" in k for k in obj["cells"])


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
