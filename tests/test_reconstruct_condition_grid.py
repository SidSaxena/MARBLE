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
                torch.save(stacked, cache_dir / f"{clip_id}.pt")
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
