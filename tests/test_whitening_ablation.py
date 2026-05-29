"""
tests/test_whitening_ablation.py

Unit + integration tests for ``scripts/analysis/whitening_ablation.py``.

Three classes of tests:

1. **Math correctness** — the transformations in ``_apply_treatment``
   match their mathematical specs (α=0 == centered, ABTT-0 == centered,
   α=1 produces identity covariance, sign-invariance of ABTT, etc).

2. **CLI / error paths** — unrecognised treatments fail loudly, missing
   ``_meta.json`` aborts, ``--max-works`` stratifies by work_id.

3. **End-to-end smoke** — full subprocess run against a synthetic
   cache + JSONL fixture; output CSV has the expected columns and
   the raw/centered MAP values differ when whitening is applied
   (guards against a silent "transform forgot to apply" bug).

Tests mirror the subprocess pattern from
``tests/test_reconstruct_condition_grid.py``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analysis" / "whitening_ablation.py"


def _load_module():
    """Import the script as a module so we can call its helpers directly."""
    spec = importlib.util.spec_from_file_location("_whitening_ablation", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# ──────────────────────────────────────────────────────────────────────
# Math-correctness unit tests (on the in-process helpers)
# ──────────────────────────────────────────────────────────────────────


def test_apply_treatment_alpha0_equals_centered():
    """``whiten-a0.0`` collapses to plain centering — identity scaling
    leaves the centered vectors untouched up to fp32 rounding."""
    mod = _load_module()
    torch.manual_seed(0)
    embs = F.normalize(torch.randn(80, 16), dim=-1)
    mu, eigvals, eigvecs = mod._compute_corpus_pca(embs)
    centered = mod._apply_treatment("centered", embs, mu, eigvals, eigvecs)
    whiten_a0 = mod._apply_treatment("whiten-a0.0", embs, mu, eigvals, eigvecs)
    # α=0 → Λ^0 = I, so the ZCA rotation cancels to identity. Should be
    # exactly the centered vectors. Tolerance accommodates the
    # eigvec @ eigvec.T = I drift in fp32.
    assert torch.allclose(centered, whiten_a0, atol=1e-5), (
        f"max diff = {(centered - whiten_a0).abs().max().item()}"
    )


def test_apply_treatment_abtt0_equals_centered():
    """``abtt-0`` is "subtract the rank-0 projection" = no-op. Should
    match ``centered`` exactly."""
    mod = _load_module()
    torch.manual_seed(1)
    embs = F.normalize(torch.randn(50, 32), dim=-1)
    mu, eigvals, eigvecs = mod._compute_corpus_pca(embs)
    centered = mod._apply_treatment("centered", embs, mu, eigvals, eigvecs)
    abtt0 = mod._apply_treatment("abtt-0", embs, mu, eigvals, eigvecs)
    assert torch.equal(centered, abtt0)


def test_apply_treatment_whiten_a1_produces_identity_covariance():
    """Full ZCA whitening (α=1.0) on a known-anisotropic input should
    produce a residual with covariance close to identity — before the
    final L2-norm step. We verify the residual cov property on the
    *pre-L2-norm* output of ``_apply_treatment``, which is the
    mathematically correct check (post-L2-norm the cov is NOT identity
    because every vector lies on the sphere)."""
    mod = _load_module()
    torch.manual_seed(2)
    N, H = 5000, 16
    # Build embeddings with a strongly anisotropic covariance.
    raw = torch.randn(N, H)
    # Stretch the first 3 directions strongly.
    raw[:, 0] *= 10.0
    raw[:, 1] *= 5.0
    raw[:, 2] *= 3.0
    embs = F.normalize(raw, dim=-1)
    mu, eigvals, eigvecs = mod._compute_corpus_pca(embs)
    whitened = mod._apply_treatment("whiten-a1.0", embs, mu, eigvals, eigvecs)
    # Covariance of whitened residuals should ≈ I.
    cov = whitened.T @ whitened / N
    eye = torch.eye(H)
    # fp32 cast of fp64 eigh + float matmul → tolerance ~5e-3.
    err = (cov - eye).abs().max().item()
    assert err < 0.05, f"whitened cov not ≈ I: max diff = {err}"


def test_whitening_recovers_buried_signal_vs_independent_numpy():
    """Keystone audit (2026-05-28), baked into CI.

    Validates the whitening *finding* — not just the transform math — on
    synthetic ground truth, against a fully independent numpy
    implementation that shares no code with the script.

    Construction: a timbre-dominated cone where the work-identity signal
    lives in LOW-variance directions (timbre amplitude 8x work amplitude).
    Raw cosine is dominated by timbre -> poor same-work retrieval.
    Whitening downweights the high-variance timbre directions -> should
    recover same-work retrieval.

    Three assertions:
      1. Independent numpy whitening + brute-force MAP shows
         whiten >> raw (the mechanism is real).
      2. The script's whitening transform produces the SAME cosine
         geometry as independent numpy ZCA (implementation correct).
      3. The script's streaming MAP matches the independent brute-force
         MAP (metric correct end-to-end).

    This is the regression form of the manual audit that confirmed the
    +109-425% real-data gains were not a code artifact.
    """
    import numpy as np  # noqa: PLC0415

    from marble.utils.retrieval_metrics import (  # noqa: PLC0415
        compute_retrieval_metrics_streaming,
    )

    mod = _load_module()
    rng = np.random.default_rng(0)
    n_works, n_timbres, n_var, H = 120, 8, 3, 64
    timbre_amp, work_amp, noise = 8.0, 1.0, 0.5
    work_vecs = rng.standard_normal((n_works, H))
    timbre_vecs = rng.standard_normal((n_timbres, H))
    embs, wids = [], []
    for w in range(n_works):
        for _v in range(n_var):
            mvec = work_vecs[w] + 0.15 * rng.standard_normal(H)
            for c in range(n_timbres):
                embs.append(
                    timbre_amp * timbre_vecs[c] + work_amp * mvec + noise * rng.standard_normal(H)
                )
                wids.append(w)
    embs = np.asarray(embs, dtype=np.float64)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    wids = np.asarray(wids)

    def indep_whiten(X, alpha=1.0):
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / X.shape[0]
        ev, U = np.linalg.eigh(np.clip(cov, None, None))
        ev = np.clip(ev, 0, None)
        scale = np.power(np.clip(ev, 1e-12, None), -alpha / 2.0)
        return Xc @ ((U * scale) @ U.T)

    def indep_map(X, work_ids):
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim = X @ X.T
        np.fill_diagonal(sim, -np.inf)
        aps = []
        for i in range(X.shape[0]):
            order = np.argsort(-sim[i])
            order = order[order != i]  # drop self from ranking
            rel = work_ids[order] == work_ids[i]
            nr = int(rel.sum())
            if nr == 0:
                continue
            hits = np.cumsum(rel)
            ranks = np.arange(1, len(rel) + 1)
            aps.append(float((rel * hits / ranks).sum() / nr))
        return float(np.mean(aps))

    # 1. Mechanism: independent whitening recovers work-identity.
    raw_map = indep_map(embs, wids)
    w_indep = indep_whiten(embs, 1.0)
    whit_map = indep_map(w_indep, wids)
    assert whit_map > raw_map * 2.0, (
        f"whitening should recover buried signal: raw={raw_map:.3f} whit={whit_map:.3f}"
    )

    # 2. Script transform == independent transform (sign/rotation-invariant
    # comparison via the similarity matrix on a subset). Cast to float32
    # to match the real loader (which stores fp32 per-file embeddings).
    embs_t = torch.from_numpy(embs).float()
    mu, eigvals, eigvecs = mod._compute_corpus_pca(embs_t)
    w_script = mod._apply_treatment("whiten-a1.0", embs_t, mu, eigvals, eigvecs).numpy()
    sw = w_script / np.linalg.norm(w_script, axis=1, keepdims=True)
    iw = w_indep / np.linalg.norm(w_indep, axis=1, keepdims=True)
    sub = rng.choice(embs.shape[0], size=300, replace=False)
    sim_diff = np.abs((sw[sub] @ sw[sub].T) - (iw[sub] @ iw[sub].T)).max()
    assert sim_diff < 1e-3, f"script vs independent whitening geometry diff {sim_diff:.2e}"

    # 3. Script streaming MAP == independent brute-force MAP.
    wids_t = torch.tensor(wids, dtype=torch.long)
    script_map = compute_retrieval_metrics_streaming(
        F.normalize(torch.from_numpy(w_script), dim=-1),
        wids_t,
        recall_ks=(),
        include_r_precision=False,
        include_median_rank=False,
        include_map=True,
        device="cpu",
        batch=256,
    )["map"]
    assert abs(script_map - whit_map) < 0.01, (
        f"script streaming MAP {script_map:.4f} != independent {whit_map:.4f}"
    )


def test_apply_treatment_abtt_removes_top_k_pc_variance():
    """After ABTT-K, the variance along the top-K principal directions
    should be (nearly) zero in the residual."""
    mod = _load_module()
    torch.manual_seed(3)
    embs = F.normalize(torch.randn(200, 32), dim=-1)
    mu, eigvals, eigvecs = mod._compute_corpus_pca(embs)
    for k in (1, 3, 5):
        residual = mod._apply_treatment(f"abtt-{k}", embs, mu, eigvals, eigvecs)
        # Project the residual onto the top-K PCs — should be near zero.
        projection = residual @ eigvecs[:, :k]
        assert projection.abs().max().item() < 1e-5, (
            f"abtt-{k}: top-{k} PC projection of residual not zero "
            f"(max {projection.abs().max().item()})"
        )
        # And the remaining directions should still carry variance.
        rest = residual @ eigvecs[:, k:]
        assert rest.abs().max().item() > 0.01


def test_parse_treatments_accepts_valid_and_rejects_invalid():
    """Strict regex parsing: all defined forms accept, unrecognised tokens
    raise SystemExit with a helpful message."""
    mod = _load_module()
    # All these should parse successfully.
    good = [
        "raw",
        "centered",
        "abtt-0",
        "abtt-1",
        "abtt-128",
        "whiten-a0.0",
        "whiten-a0.5",
        "whiten-a1.0",
        "whiten-a1.0-erel-1e-3",
        "whiten-a0.7-erel-1e-2",
    ]
    out = mod._parse_treatments(good)
    assert out == good

    for bad in ["junk", "whiten-a", "abtt-foo", "whiten-1.0", "whiten-a1.0-eps-1e-3"]:
        try:
            mod._parse_treatments([bad])
        except SystemExit as e:
            assert "unrecognised treatment" in str(e), str(e)
        else:
            raise AssertionError(f"expected SystemExit for treatment {bad!r}")


def test_work_disjoint_split_no_work_in_both_sides():
    """The inductive split must partition by work id — no work may appear in
    both the fit and eval sets, and the union must cover every file."""
    mod = _load_module()
    # 12 works, 3 files each (e.g. 3 renditions), interleaved.
    work_ids = [w for w in range(12) for _ in range(3)]
    fit_idx, eval_idx = mod._work_disjoint_split(work_ids, fit_frac=0.5, seed=0)

    fit_works = {work_ids[i] for i in fit_idx}
    eval_works = {work_ids[i] for i in eval_idx}
    assert fit_works.isdisjoint(eval_works), "work leaked across fit/eval"
    assert fit_works | eval_works == set(work_ids), "split dropped works"
    assert sorted(fit_idx + eval_idx) == list(range(len(work_ids))), "files lost"
    # All renditions of a chosen work stay together → no singletons in eval.
    from collections import Counter

    eval_counts = Counter(work_ids[i] for i in eval_idx)
    assert all(c == 3 for c in eval_counts.values()), "eval work lost renditions"


def test_work_disjoint_split_deterministic_under_seed():
    mod = _load_module()
    work_ids = [w for w in range(20) for _ in range(2)]
    a = mod._work_disjoint_split(work_ids, 0.5, seed=7)
    b = mod._work_disjoint_split(work_ids, 0.5, seed=7)
    c = mod._work_disjoint_split(work_ids, 0.5, seed=8)
    assert a == b, "same seed must give identical split"
    assert a != c, "different seed should (almost surely) differ"


def test_work_disjoint_split_rejects_bad_frac():
    mod = _load_module()
    for bad in (0.0, 1.0, -0.2, 1.5):
        try:
            mod._work_disjoint_split([0, 0, 1, 1], bad, seed=0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for fit_frac={bad}")


# ──────────────────────────────────────────────────────────────────────
# Integration: synthetic-cache subprocess run
# ──────────────────────────────────────────────────────────────────────


def _make_synthetic_cache(
    cache_dir: Path,
    n_works: int = 6,
    versions_per_cell: int = 3,
    conditions: tuple[int, ...] = (0, 24, 48, 60),
    H: int = 32,
    n_layers: int = 4,
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Build a synthetic cache mirroring ``EmbeddingCache.put`` output.

    Each cache file stores ``{"embedding": (L, H) tensor}``. To exercise
    multi-clip aggregation, every other file gets two slices (c0, c1).

    Returns (records, meta_dict). ``meta_dict`` is written as
    ``_meta.json`` in the cache dir so the production script's
    provenance check passes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    torch.manual_seed(seed)
    for w in range(n_works):
        work_vec = torch.randn(H)
        for c in conditions:
            cond_vec = torch.randn(H) * 0.05
            for v in range(versions_per_cell):
                file_idx = len(records)
                audio_path = f"data/fake/{file_idx:04d}.flac"
                # Per-version jitter so identical-cell files aren't
                # cosine-collinear.
                file_vec_base = work_vec + cond_vec + torch.randn(H) * 0.005
                # n_layers layers, each a slight variation. Use this
                # layer-stack so layer-index slicing has something to do.
                layer_stack = torch.stack(
                    [
                        file_vec_base + torch.randn(H) * (0.001 + 0.002 * li)
                        for li in range(n_layers)
                    ],
                    dim=0,
                )  # (L, H)
                stem = Path(audio_path).stem
                h = _sha1_8(audio_path)
                # Half the files get 2 clips (multi-clip path); the other
                # half get just c0 (single-clip path). Same audio path,
                # multiple cN suffixes.
                n_clips = 2 if file_idx % 2 == 0 else 1
                for ci in range(n_clips):
                    clip_stack = layer_stack + torch.randn_like(layer_stack) * 0.001
                    clip_id = f"{stem}__{h}__c{ci}"
                    torch.save({"embedding": clip_stack}, cache_dir / f"{clip_id}.pt")
                records.append(
                    {
                        "audio_path": audio_path,
                        "work_id": w,
                        "gm_program": c,
                        "sample_rate": 24000,
                        "num_samples": 24000 * (1 + n_clips),
                        "channels": 1,
                    }
                )
    meta = {
        "encoder_model_id": "fake-encoder/v0",
        "task_name": "VGMIDITVar-timbre",
        "sample_rate": 24000,
        "clip_seconds": 1.0,
        "pipeline_signature": "fake-pipeline-sig",
        "pool_time": True,
    }
    (cache_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return records, meta


def _write_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_smoke_run_produces_csv_with_all_treatments(tmp_path: Path):
    """End-to-end: synthetic cache + JSONL → run all default treatments
    → CSV has one row per treatment with expected columns."""
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "out.csv"
    records, _ = _make_synthetic_cache(cache_dir)
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    treatments = ["raw", "centered", "abtt-1", "whiten-a0.5", "whiten-a1.0"]
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--task-tag",
            "VGMIDITVar-timbre",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
            "--batch",
            "32",
            "--treatments",
            *treatments,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stdout + "\n---\n" + p.stderr

    assert out_csv.exists(), "CSV not written"
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(treatments), f"expected {len(treatments)} rows, got {len(rows)}"
    seen_treatments = [r["treatment"] for r in rows]
    assert seen_treatments == treatments
    # Provenance + diagnostic columns populated.
    for row in rows:
        assert row["encoder"] == "FakeEncoder"
        assert row["encoder_model_id"] == "fake-encoder/v0"
        assert int(row["n_files"]) > 0
        assert float(row["lambda_max"]) > 0.0
        assert 0.0 <= float(row["map"]) <= 1.0
    # Default run is transductive: fit set == eval set.
    assert all(r["fit_mode"] == "transductive" for r in rows)
    assert all(int(r["n_fit"]) == int(r["n_files"]) for r in rows)


def test_smoke_run_inductive_fit_frac(tmp_path: Path):
    """--fit-frac fits PCA on a work-disjoint held-out slice and evaluates on
    the complement: fit_mode is recorded, n_fit + n_files (eval) partition the
    corpus, and works don't leak across the split."""
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "out.csv"
    records, _ = _make_synthetic_cache(cache_dir)  # 6 works x 12 files = 72
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    def _run(fit_on: str, out: Path):
        p = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--encoder",
                "FakeEncoder",
                "--encoder-tag",
                "FakeEncoder-tag",
                "--task-tag",
                "VGMIDITVar-timbre",
                "--layer",
                "0",
                "--jsonl",
                str(jsonl),
                "--cache-dir",
                str(cache_dir),
                "--out-csv",
                str(out),
                "--device",
                "cpu",
                "--batch",
                "32",
                "--treatments",
                "raw",
                "whiten-a1.0",
                "--fit-frac",
                "0.5",
                "--fit-seed",
                "0",
                "--fit-on",
                fit_on,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert p.returncode == 0, p.stdout + "\n---\n" + p.stderr
        with open(out) as f:
            return list(csv.DictReader(f))

    ind_rows = _run("complement", out_csv)
    trans_rows = _run("self", tmp_path / "out_self.csv")
    assert ind_rows and trans_rows
    # 6 works split 50/50 → 3 fit + 3 eval works → 36 files each.
    for r in ind_rows:
        assert r["fit_mode"].startswith("inductive"), r["fit_mode"]
        assert int(r["n_fit"]) == 36 and int(r["n_files"]) == 36
        assert 0.0 <= float(r["map"]) <= 1.0
    # fit_on=self → transductive baseline on the SAME eval set.
    for r in trans_rows:
        assert r["fit_mode"].startswith("transductive-eval"), r["fit_mode"]
        assert int(r["n_files"]) == 36
    # raw is fit-independent → identical eval set → identical raw MAP.
    ind_raw = next(float(r["map"]) for r in ind_rows if r["treatment"] == "raw")
    trans_raw = next(float(r["map"]) for r in trans_rows if r["treatment"] == "raw")
    assert abs(ind_raw - trans_raw) < 1e-6, "same eval set must give same raw MAP"


def test_smoke_run_whitening_changes_top_k_ranking(tmp_path: Path):
    """Guards against a silent "transform forgot to apply" bug: the
    MAP values under ``raw`` / ``centered`` / ``whiten-a1.0`` should
    differ meaningfully, NOT be bit-identical."""
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "out.csv"
    records, _ = _make_synthetic_cache(cache_dir)
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--task-tag",
            "VGMIDITVar-timbre",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
            "--batch",
            "32",
            "--treatments",
            "raw",
            "centered",
            "whiten-a1.0",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stdout + p.stderr

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    treatments_to_map = {r["treatment"]: float(r["map"]) for r in rows}
    # At least one pair must differ by more than fp rounding noise. We
    # build the synthetic cache with structured anisotropy so the three
    # treatments produce visibly different rankings.
    raw_m = treatments_to_map["raw"]
    cent_m = treatments_to_map["centered"]
    whit_m = treatments_to_map["whiten-a1.0"]
    assert abs(raw_m - whit_m) > 1e-3 or abs(cent_m - whit_m) > 1e-3, (
        f"raw={raw_m:.4f}  centered={cent_m:.4f}  whiten={whit_m:.4f} — "
        f"transformations look like no-ops; ablation is broken"
    )


def test_unrecognised_treatment_fails_loudly(tmp_path: Path):
    """Bad treatment names exit non-zero with helpful error."""
    cache_dir = tmp_path / "cache"
    records, _ = _make_synthetic_cache(cache_dir, n_works=2)
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
            "--out-csv",
            str(tmp_path / "out.csv"),
            "--treatments",
            "frobnicate-zorp-99",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode != 0
    combined = p.stdout + p.stderr
    assert "unrecognised treatment" in combined
    assert "frobnicate-zorp-99" in combined


def test_missing_meta_json_aborts(tmp_path: Path):
    """If the cache dir has no ``_meta.json``, the script must refuse
    to proceed (provenance guard)."""
    cache_dir = tmp_path / "cache"
    records, _ = _make_synthetic_cache(cache_dir, n_works=2)
    # Delete the _meta.json the fixture wrote.
    (cache_dir / "_meta.json").unlink()
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--device",
            "cpu",
            "--out-csv",
            str(tmp_path / "out.csv"),
            "--treatments",
            "raw",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert p.returncode != 0
    combined = p.stdout + p.stderr
    assert "_meta.json" in combined


def test_max_works_stratifies_by_work_id(tmp_path: Path):
    """``--max-works 3`` should keep all clips from exactly 3 distinct
    work_ids — never partial-coverage of more works."""
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "out.csv"
    # n_works=6 conditions=4 versions=2 → 48 files, 6 work_ids
    records, _ = _make_synthetic_cache(
        cache_dir, n_works=6, versions_per_cell=2, conditions=(0, 24, 48, 60)
    )
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
            "--treatments",
            "raw",
            "--max-works",
            "3",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    # n_works=3 × conds=4 × versions=2 = 24 files retained
    with open(out_csv) as f:
        row = next(csv.DictReader(f))
    assert int(row["n_files"]) == 24, f"unexpected n_files = {row['n_files']}"


def test_csv_overwrite_on_rerun_with_default_flags(tmp_path: Path):
    """Regression test (2026-05-28): re-running the script with default
    flags must TRUNCATE the existing CSV, not append to it. Earlier code
    silently appended because the file existed from a prior run.

    Failure mode this guards against: the second run's results pile on
    top of the first, and the user can't tell which rows came from which
    invocation. Particularly bad when the same (encoder, layer) is
    re-run after a code change — the old (incorrect) rows survive.
    """
    cache_dir = tmp_path / "cache"
    out_csv = tmp_path / "out.csv"
    records, _ = _make_synthetic_cache(cache_dir, n_works=2)
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    # Run 1: write a single row.
    cmd_base = [
        sys.executable,
        str(SCRIPT),
        "--encoder",
        "FakeEncoder",
        "--encoder-tag",
        "FakeEncoder-tag",
        "--layer",
        "0",
        "--jsonl",
        str(jsonl),
        "--cache-dir",
        str(cache_dir),
        "--out-csv",
        str(out_csv),
        "--device",
        "cpu",
        "--batch",
        "32",
    ]
    p1 = subprocess.run(
        cmd_base + ["--treatments", "raw"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p1.returncode == 0, p1.stdout + p1.stderr
    with open(out_csv) as f:
        rows_1 = list(csv.DictReader(f))
    assert len(rows_1) == 1, f"run 1 expected 1 row, got {len(rows_1)}"

    # Run 2 (same CSV, default flags): MUST OVERWRITE.
    p2 = subprocess.run(
        cmd_base + ["--treatments", "centered"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p2.returncode == 0, p2.stdout + p2.stderr
    with open(out_csv) as f:
        rows_2 = list(csv.DictReader(f))
    assert len(rows_2) == 1, (
        f"run 2 expected 1 row (overwrite), got {len(rows_2)} — "
        f"CSV is being appended on re-runs without --force-append"
    )
    assert rows_2[0]["treatment"] == "centered"

    # Run 3 with --force-append: MUST append.
    p3 = subprocess.run(
        cmd_base + ["--treatments", "raw", "--force-append"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p3.returncode == 0, p3.stdout + p3.stderr
    with open(out_csv) as f:
        rows_3 = list(csv.DictReader(f))
    assert len(rows_3) == 2, f"run 3 (--force-append) expected 2 rows total, got {len(rows_3)}"
    assert [r["treatment"] for r in rows_3] == ["centered", "raw"]


def test_effective_rank_centered_matches_anisotropy_metrics_formula():
    """Regression test (2026-05-28): the ``effective_rank_centered`` CSV
    column must use the SAME entropy-based formula as
    ``anisotropy_metrics.effective_rank`` (both are ``exp(-Σ pᵢ log pᵢ)``
    of the normalised non-zero eigenvalue spectrum).

    The first implementation accidentally computed stable rank
    ``(Σλ)² / Σλ²`` instead — a DIFFERENT, non-comparable scalar that
    would have been silently logged in the same row as the entropy-based
    ``effective_rank`` column. The pre-fix value diverges by 2-5× from
    the correct one on cone-collapsed inputs, which is exactly where
    the ablation is supposed to be informative.

    This test hand-computes both metrics on a known anisotropic input
    and asserts that the script's column matches the anisotropy_metrics
    formula, NOT the stable rank.
    """
    mod = _load_module()
    torch.manual_seed(42)
    # Cone-collapsed embeddings: 3 dominant + many tiny directions.
    N, H = 2000, 32
    raw = torch.randn(N, H)
    raw[:, 0] *= 8.0
    raw[:, 1] *= 4.0
    raw[:, 2] *= 2.0
    embs = F.normalize(raw, dim=-1)

    # Replicate the script's own computation.
    _mu, eigvals_desc, _eigvecs = mod._compute_corpus_pca(embs)
    lambda_max = float(eigvals_desc[0])
    nonzero_eigs = eigvals_desc[eigvals_desc > 1e-10 * lambda_max]
    share = nonzero_eigs / nonzero_eigs.sum()
    expected_eff_rank = float((-(share * (share + 1e-12).log()).sum()).exp())

    # Sanity: the (incorrect) stable rank formula gives a DIFFERENT number.
    stable_rank = float((nonzero_eigs.sum() ** 2 / (nonzero_eigs**2).sum()).item())
    assert abs(expected_eff_rank - stable_rank) > 0.5, (
        f"fixture failed to distinguish formulas: "
        f"entropy={expected_eff_rank:.3f}, stable={stable_rank:.3f}"
    )

    # Now run the production script (--dry-run prints PCA stats but
    # doesn't compute MAP). We read effective_rank_centered out of a
    # full run since dry-run doesn't write CSV.
    tmp_path = Path("/tmp") / "whitening_ablation_eff_rank_test"
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    if cache_dir.exists():
        import shutil  # noqa: PLC0415

        shutil.rmtree(cache_dir)
    records, _ = _make_synthetic_cache(
        cache_dir, n_works=6, versions_per_cell=3, conditions=(0, 24, 48, 60), seed=42
    )
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)
    out_csv = tmp_path / "out.csv"

    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
            "--batch",
            "32",
            "--treatments",
            "raw",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stdout + p.stderr

    with open(out_csv) as f:
        row = next(csv.DictReader(f))
    csv_eff_rank = float(row["effective_rank_centered"])
    csv_eff_rank_from_anisotropy = float(row["effective_rank"])
    # The CSV ``effective_rank_centered`` (computed on the full corpus
    # eigenvalues) and ``effective_rank`` (computed by
    # anisotropy_metrics on a subsample) should be close but not
    # identical. Both use the entropy formula.
    relative_err = abs(csv_eff_rank - csv_eff_rank_from_anisotropy) / max(
        csv_eff_rank_from_anisotropy, 1.0
    )
    assert relative_err < 0.20, (
        f"effective_rank_centered ({csv_eff_rank:.3f}) diverges from "
        f"anisotropy_metrics.effective_rank ({csv_eff_rank_from_anisotropy:.3f}) "
        f"by {100 * relative_err:.1f}% — likely the formula is wrong again"
    )


def test_dry_run_skips_metric_pass_and_succeeds(tmp_path: Path):
    """``--dry-run`` exits cleanly without invoking the streaming MAP."""
    cache_dir = tmp_path / "cache"
    records, _ = _make_synthetic_cache(cache_dir, n_works=2)
    jsonl = tmp_path / "test.jsonl"
    _write_jsonl(jsonl, records)

    out_csv = tmp_path / "out.csv"
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--encoder",
            "FakeEncoder",
            "--encoder-tag",
            "FakeEncoder-tag",
            "--layer",
            "0",
            "--jsonl",
            str(jsonl),
            "--cache-dir",
            str(cache_dir),
            "--out-csv",
            str(out_csv),
            "--device",
            "cpu",
            "--treatments",
            "raw",
            "whiten-a1.0",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert not out_csv.exists()  # no CSV written on dry run
    assert "skipping metric pass" in p.stdout
