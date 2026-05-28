"""
tests/test_reconstruct_condition_grid.py

End-to-end smoke test for
``scripts/analysis/reconstruct_condition_grid_from_cache.py``.

Generates a synthetic cache directory + matching JSONL, runs the script
as a subprocess, and asserts the CSV / JSON outputs look right. The
heatmap PNG is optional (matplotlib may not be available); we just
check it's produced when matplotlib is importable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "reconstruct_condition_grid_from_cache.py"


def _sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _make_synthetic_cache(
    cache_dir: Path,
    n_works: int = 4,
    versions_per_cell: int = 2,
    conditions: tuple[int, ...] = (0, 24, 48, 60),
    H: int = 16,
) -> list[dict]:
    """Build a synthetic cache + JSONL records.

    Each ``(work, condition)`` cell carries ``versions_per_cell`` files,
    so every diagonal cell has at least one same-work-same-condition
    peer for every query (other than self). Without this, the
    self-exclusion in ``compute_perpair_map_all`` would leave diagonal
    cells with ``n_rel == 0`` and the test would assert against empty
    cells.

    Embeddings lean on a per-work base vector plus a small
    condition-specific perturbation, so within-work peers have higher
    similarity than cross-work pairs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    torch.manual_seed(0)
    for w in range(n_works):
        work_vec = torch.randn(H)
        for c in conditions:
            cond_vec = torch.randn(H) * 0.05
            for v in range(versions_per_cell):
                file_idx = len(records)
                audio_path = f"data/fake/{file_idx:03d}.flac"
                # Tiny per-version jitter so identical-cell files aren't
                # cosine-collinear (which would make argsort ordering
                # non-deterministic on ties).
                file_vec = work_vec + cond_vec + torch.randn(H) * 0.005
                # Cache stores (L, H) — give it 2 layers so the
                # mean-over-L path is exercised.
                stacked = torch.stack([file_vec, file_vec + torch.randn(H) * 0.001], dim=0)
                stem = Path(audio_path).stem
                h = _sha1_8(audio_path)  # mirrors make_clip_id
                clip_id = f"{stem}__{h}__c0"
                # Production cache writes a ``{"embedding": tensor}`` dict
                # (see EmbeddingCache.put). Mirror that format here.
                torch.save({"embedding": stacked}, cache_dir / f"{clip_id}.pt")
                records.append(
                    {
                        "audio_path": audio_path,
                        "work_id": w,
                        "gm_program": c,
                        "sample_rate": 24000,
                        "num_samples": 24000,
                        "channels": 1,
                    }
                )
    return records


def test_reconstruct_grid_produces_csv_and_json(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    records = _make_synthetic_cache(cache_dir)
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr

    csv_path = out_dir / "condition_grid.csv"
    json_path = out_dir / "condition_grid.json"
    assert csv_path.exists(), "CSV not written"
    assert json_path.exists(), "JSON not written"

    # CSV: 4 conditions → 16 cells + header.
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16, f"expected 16 cells, got {len(rows)}"
    # With 2 versions per (work, condition) cell, every cell has at
    # least one same-work peer (other than self) in the target condition.
    for row in rows:
        assert int(row["n_queries"]) > 0, f"cell {row} has no queries"
        ap = float(row["map"])
        assert 0.0 <= ap <= 1.0, f"map out of range: {row}"

    # JSON: mirrors the CSV plus metadata.
    blob = json.loads(json_path.read_text())
    assert blob["encoder"] == "FakeEncoder"
    # 4 works × 4 conditions × 2 versions = 32 files.
    assert blob["n_files"] == 32, blob["n_files"]
    assert blob["unique_conditions"] == [0, 24, 48, 60]
    assert len(blob["cells"]) == 16

    # Heatmap PNG only if matplotlib is available; smoke check.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out_dir / "condition_grid.png").exists()


def test_reconstruct_multi_clip_aggregation_matches_probe(tmp_path: Path):
    """Regression test for the per-clip-L2-then-mean fix (2026-05-28).

    Pre-fix the script computed ``F.normalize(mean(per_slice))`` — i.e.
    L2-normed AFTER the per-file clip mean. The probe does
    ``F.normalize(mean(F.normalize(per_clip)))`` — per-clip L2 first.

    For a multi-clip file with clips that differ noticeably in magnitude,
    the two orderings produce different per-file vectors. Test reproduces
    the probe's logic by hand on a synthetic 2-clip file and asserts the
    script's aggregated embedding matches.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    audio_path = "data/fake/multiclip_0.flac"
    h = _sha1_8(audio_path)
    stem = Path(audio_path).stem
    # Cache writes (L, H) per clip. The reconstruct script averages over
    # L (meanall convention). Build TWO clips whose post-L-mean vectors
    # have noticeably different magnitudes — the bug only surfaces here.
    H = 16
    L = 2
    torch.manual_seed(0)
    # Clip 0: small magnitude, will normalise to roughly e_0.
    clip0_stack = torch.zeros(L, H)
    clip0_stack[:, 0] = 0.1
    clip0_stack += torch.randn(L, H) * 0.001
    # Clip 1: larger magnitude, normalises to roughly e_1.
    clip1_stack = torch.zeros(L, H)
    clip1_stack[:, 1] = 10.0
    clip1_stack += torch.randn(L, H) * 0.01
    torch.save({"embedding": clip0_stack}, cache_dir / f"{stem}__{h}__c0.pt")
    torch.save({"embedding": clip1_stack}, cache_dir / f"{stem}__{h}__c1.pt")

    # Hand-compute the probe-correct per-file embedding:
    #   per-clip layer-mean → L2-normalise → average → re-normalise
    clip0_layer_mean = clip0_stack.mean(dim=0)
    clip1_layer_mean = clip1_stack.mean(dim=0)
    clip0_unit = F.normalize(clip0_layer_mean, dim=-1)
    clip1_unit = F.normalize(clip1_layer_mean, dim=-1)
    expected_emb = F.normalize(torch.stack([clip0_unit, clip1_unit]).mean(dim=0), dim=-1)
    # Pre-fix (buggy) per-file embedding for contrast — should NOT match.
    buggy_emb = F.normalize(torch.stack([clip0_layer_mean, clip1_layer_mean]).mean(dim=0), dim=-1)
    # Sanity: the two recipes really do diverge on this fixture.
    divergence = (expected_emb - buggy_emb).abs().max().item()
    assert divergence > 0.05, (
        f"fixture failed to expose the bug — buggy and correct outputs "
        f"diverge by only {divergence}; pick more unbalanced clips"
    )

    # Now drive the script and read the produced JSON to recover the
    # implied per-file embedding direction. Reconstruct script needs a
    # JSONL with at least one extra peer to compute any AP, so we add a
    # filler file that has a single clip and a different work_id.
    filler_path = "data/fake/filler_0.flac"
    filler_stack = torch.randn(L, H)
    h2 = _sha1_8(filler_path)
    torch.save(
        {"embedding": filler_stack},
        cache_dir / f"{Path(filler_path).stem}__{h2}__c0.pt",
    )

    records = [
        {
            "audio_path": audio_path,
            "work_id": 1,
            "gm_program": 0,
            "sample_rate": 24000,
            "num_samples": 24000,
            "channels": 1,
        },
        {
            "audio_path": filler_path,
            "work_id": 2,
            "gm_program": 0,
            "sample_rate": 24000,
            "num_samples": 24000,
            "channels": 1,
        },
    ]
    jsonl = tmp_path / "rec.jsonl"
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Import the script's loader directly to inspect the produced (N, H)
    # matrix without parsing the heatmap.
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "_reconstruct_condition_grid_from_cache",
        REPO_ROOT / "scripts" / "analysis" / "reconstruct_condition_grid_from_cache.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    embs, wids, conds = mod._collect_per_file_embeddings(records, cache_dir)
    # Multi-clip file is records[0]; its row in `embs` is row 0.
    actual = embs[0]
    max_err = (actual - expected_emb).abs().max().item()
    max_err_vs_buggy = (actual - buggy_emb).abs().max().item()
    assert max_err < 1e-5, (
        f"reconstruct aggregation diverges from probe convention; "
        f"max err vs expected = {max_err:.4e}, vs buggy old behaviour = "
        f"{max_err_vs_buggy:.4e}"
    )


def test_reconstruct_skips_records_missing_cache(tmp_path: Path):
    """Records whose cached .pt file is missing should be skipped with a
    warning, not crash the run."""
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    records = _make_synthetic_cache(cache_dir, n_works=2, conditions=(0, 24))

    # Add 2 extra records whose cache files don't exist.
    for ghost in range(2):
        records.append(
            {
                "audio_path": f"data/fake/ghost_{ghost}.flac",
                "work_id": 99,
                "gm_program": 0,
                "sample_rate": 24000,
                "num_samples": 24000,
                "channels": 1,
            }
        )
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    # Warning about missing records lands on stderr.
    assert "no cached embedding" in p.stderr, p.stderr
