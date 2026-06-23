# VGMLoop Audit Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three audit findings in the `vgmloopstructure-task` branch: CE loss shape crash in VGMLoopFrames probe, SR guard in the VGM corpus converter, and a frame-mode JSONL conversion path.

**Architecture:** All fixes are surgical — no new files for Tasks 1-2; Task 3 extends the existing converter with a `--mode` flag and a sibling `convert_frame` function. Tests use the existing subprocess-based fixture pattern already in `test_convert_vgm_corpus_to_jsonl.py`.

**Tech Stack:** PyTorch (CrossEntropyLoss), soundfile, pytest, Python stdlib argparse.

## Global Constraints

- Branch: `vgmloopstructure-task` — do NOT switch branches.
- Never add `Co-Authored-By` or Claude trailers to commits.
- All file paths must be absolute when referenced in commands.
- Tests run via `pytest` from repo root; use `uv run pytest` if plain `pytest` is unavailable.
- FIXED_SAMPLE_RATE = 24000 is the project constant; do not change it.
- Frame JSONL `label` must be a dict (not a string); clip JSONL `label` must remain a string.

---

### Task 1: Fix CE loss shape crash in VGMLoopFrames probe

**Files:**
- Modify: `/Users/sid/Developer/Python/marble/marble/tasks/VGMLoopFrames/probe.py:101`

**Interfaces:**
- Consumes: `self.loss_fns` list, `logits` `(B, L, 3)` or `(B, L)`, `y` `(B, L)` int64 or float32.
- Produces: fixed `_shared_step` that flattens before CE loss call; no signature change.

**Root cause:** `CrossEntropyLoss` expects `(N, C, ...)` input — i.e. the class dim must be at index 1. When `logits` is `(B, L, 3)` and `targets` is `(B, L)`, PyTorch interprets shape as `(N=B, C=L, d1=3)` and target `(B, L)` fails because it expects target shape `(N, d1)=(B, 3)`, not `(B, L)`. The fix: when logits is 3D (function/CE path), flatten to `(B*L, C)` / `(B*L,)` before the loss call, exactly mirroring what the metrics path already does.

- [ ] **Step 1: Write the failing unit test**

Add a new test file `/Users/sid/Developer/Python/marble/tests/test_vgmloopframes_probe_loss.py`:

```python
"""tests/test_vgmloopframes_probe_loss.py

Focused regression test: _shared_step loss call must not crash for the
VGMLoopFunctionProbe when logits=(B,L,3) and targets=(B,L) int64.

We test the loss computation in isolation (no encoder, no Lightning trainer)
by calling CrossEntropyLoss directly with the shapes that _shared_step
produces — both before and after the flatten fix.
"""
import pytest
import torch
import torch.nn as nn


def _ce_loss_flat(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The fixed path: flatten (B,L,C)→(B*L,C) and (B,L)→(B*L,) before CE."""
    B, L, C = logits.shape
    return nn.CrossEntropyLoss()(logits.reshape(B * L, C), targets.reshape(B * L))


def test_ce_loss_with_blc_logits_crashes_without_flatten():
    """Confirm the un-fixed call raises RuntimeError (documents the original bug)."""
    logits = torch.randn(2, 50, 3)
    targets = torch.randint(0, 3, (2, 50))
    with pytest.raises(RuntimeError):
        nn.CrossEntropyLoss()(logits, targets)


def test_ce_loss_flat_does_not_crash():
    """The flattened path must produce a finite scalar without raising."""
    logits = torch.randn(2, 50, 3)
    targets = torch.randint(0, 3, (2, 50))
    loss = _ce_loss_flat(logits, targets)
    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"


def test_ce_loss_flat_single_batch():
    """Works for batch size 1 (edge case)."""
    logits = torch.randn(1, 25, 3)
    targets = torch.randint(0, 3, (1, 25))
    loss = _ce_loss_flat(logits, targets)
    assert torch.isfinite(loss)


def test_bce_path_unchanged():
    """
    Boundary BCEWithLogitsLoss path: logits (B,L) and targets (B,L) float32.
    This path must NOT be flattened in the loss call — verify it works as-is.
    """
    logits = torch.randn(2, 50)
    targets = torch.rand(2, 50)
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run the new tests to verify the bug test passes and the fix test FAILS (expected)**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_vgmloopframes_probe_loss.py -v
```

Expected: `test_ce_loss_with_blc_logits_crashes_without_flatten` PASSES (the bug is real).
`test_ce_loss_flat_does_not_crash` PASSES (the helper already uses the fix).
All 4 tests should PASS because they test the standalone helper, not the probe code yet.

- [ ] **Step 3: Apply the fix in probe.py**

In `/Users/sid/Developer/Python/marble/marble/tasks/VGMLoopFrames/probe.py`, replace the loss computation block (lines ~99-101):

**Old:**
```python
        bs = x.size(0)

        losses = [fn(logits, y) for fn in self.loss_fns]
```

**New:**
```python
        bs = x.size(0)

        # CrossEntropyLoss requires the class dim at index 1: (N, C, ...).
        # When logits is (B, L, C), flatten to (B*L, C) and y to (B*L,) before
        # the CE call.  The boundary BCE path keeps (B, L)/(B, L) as-is.
        if logits.dim() == 3:
            B, L, C = logits.shape
            loss_logits = logits.reshape(B * L, C)
            loss_y = y.reshape(B * L)
        else:
            loss_logits = logits
            loss_y = y

        losses = [fn(loss_logits, loss_y) for fn in self.loss_fns]
```

- [ ] **Step 4: Run all probe loss tests**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_vgmloopframes_probe_loss.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/sid/Developer/Python/marble && git add marble/tasks/VGMLoopFrames/probe.py tests/test_vgmloopframes_probe_loss.py
git commit -m "fix(VGMLoopFrames): flatten (B,L,C)→(B*L,C) before CrossEntropyLoss in _shared_step"
```

---

### Task 2: SR guard in convert_vgm_corpus_to_jsonl.py

**Files:**
- Modify: `/Users/sid/Developer/Python/marble/scripts/data/convert_vgm_corpus_to_jsonl.py:149-157`
- Modify: `/Users/sid/Developer/Python/marble/tests/test_convert_vgm_corpus_to_jsonl.py` (add test)

**Interfaces:**
- Consumes: `_probe_wav` returns `(num_samples, samplerate, channels)`; `FIXED_SAMPLE_RATE = 24000`.
- Produces: rows at non-24k SR are skipped with a stderr warning; `n_missing` counter incremented.

**Root cause:** The converter probes the actual file samplerate but never checks it against `FIXED_SAMPLE_RATE`. If a WAV is at e.g. 44100 Hz, `num_samples` from `sf.info` is the actual frame count at 44100, but `duration = num_samples / 24000` computes a wrong duration, and the downstream datamodule's `num_samples` will be silently wrong.

- [ ] **Step 1: Write the failing test first**

Append this test to `/Users/sid/Developer/Python/marble/tests/test_convert_vgm_corpus_to_jsonl.py`:

```python
def test_non_24k_wav_is_skipped(tmp_path):
    """A WAV at a sample rate other than 24000 must be skipped with a warning."""
    corpus = tmp_path / "corpus"
    manifest, audio_root = _build_mini_corpus(corpus)

    # Overwrite the train WAV with a 44100 Hz file (same path, different SR)
    bad_wav = corpus / "audio" / "train_tc.wav"
    _write_wav(bad_wav, n_samples=48000, sr=44100)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--manifest", str(manifest),
            "--audio-root", str(audio_root),
            "--out-dir", str(out_dir),
            "--name", "VGMLoopStructure",
        ],
        capture_output=True,
        text=True,
    )
    # Must not crash
    assert result.returncode == 0, f"Converter crashed:\n{result.stderr}"

    # train split must be empty (skipped)
    train_rows = _read_jsonl(out_dir / "VGMLoopStructure.train.wav.jsonl")
    assert len(train_rows) == 0, f"Expected 0 train rows (44100 Hz skipped), got {len(train_rows)}"

    # val and test must still have 1 row each
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.val.wav.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.test.wav.jsonl")) == 1

    # Warning must appear on stderr
    assert "44100" in result.stderr or "sample_rate" in result.stderr.lower() or "24000" in result.stderr, (
        f"Expected SR warning on stderr, got:\n{result.stderr}"
    )
```

- [ ] **Step 2: Run the new test to confirm it FAILS (no guard yet)**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_convert_vgm_corpus_to_jsonl.py::test_non_24k_wav_is_skipped -v
```

Expected: FAIL — the 44100 Hz WAV is currently accepted and the train split has 1 row.

- [ ] **Step 3: Apply the SR guard in the converter**

In `/Users/sid/Developer/Python/marble/scripts/data/convert_vgm_corpus_to_jsonl.py`, replace the block after `_probe_wav` call (lines ~149-157):

**Old:**
```python
        # Probe audio file for actual num_samples / sample_rate / channels
        try:
            num_samples, samplerate, channels = _probe_wav(wav_path)
        except Exception as exc:
            print(
                f"WARNING: could not probe {wav_path}: {exc} — skipping",
                file=sys.stderr,
            )
            n_missing += 1
            continue
```

**New:**
```python
        # Probe audio file for actual num_samples / sample_rate / channels
        try:
            num_samples, samplerate, channels = _probe_wav(wav_path)
        except Exception as exc:
            print(
                f"WARNING: could not probe {wav_path}: {exc} — skipping",
                file=sys.stderr,
            )
            n_missing += 1
            continue

        if samplerate != FIXED_SAMPLE_RATE:
            print(
                f"WARNING: row id={row.get('id')!r} has sample_rate={samplerate} "
                f"(expected {FIXED_SAMPLE_RATE}) — skipping",
                file=sys.stderr,
            )
            n_missing += 1
            continue
```

- [ ] **Step 4: Run all converter tests**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_convert_vgm_corpus_to_jsonl.py -v
```

Expected: all tests PASS including the new `test_non_24k_wav_is_skipped`.

- [ ] **Step 5: Commit**

```bash
cd /Users/sid/Developer/Python/marble && git add scripts/data/convert_vgm_corpus_to_jsonl.py tests/test_convert_vgm_corpus_to_jsonl.py
git commit -m "fix(converter): skip WAVs with sample_rate != 24000, warn to stderr"
```

---

### Task 3: Frame-mode JSONL converter with --mode flag

**Files:**
- Modify: `/Users/sid/Developer/Python/marble/scripts/data/convert_vgm_corpus_to_jsonl.py` (add `--mode` flag, `convert_frame` function)
- Modify: `/Users/sid/Developer/Python/marble/tests/test_convert_vgm_corpus_to_jsonl.py` (add frame-mode tests using existing fixture pattern)

**Interfaces:**
- Consumes: manifest rows with `intro_end_sec` (float|null), `loop_seam_sec` (float|null), `loop_type` (str), `total_sec` (float) fields; `split` field shared with clip mode.
- Produces: per-split JSONL `<name>.{train,val,test}.wav.jsonl` where `label` is a dict:
  ```json
  {"intro_end_sec": float|null, "loop_seam_sec": float|null, "loop_type": "...", "total_sec": float}
  ```
- `--mode clip` (default): existing behaviour unchanged, `label` = string.
- `--mode frame`: new behaviour, `label` = dict as above.

**Manifest rows for frame mode** must contain: `intro_end_sec`, `loop_seam_sec`, `total_sec` (in addition to the existing fields). Missing optional keys (`intro_end_sec`, `loop_seam_sec`) default to `null`.

**Split assignment:** frame mode reads `split` from the manifest, same as clip mode. Both modes share the same split values — frame + clip tasks can be aligned by `audio_path`.

- [ ] **Step 1: Extend `_build_mini_corpus` in tests to support frame manifest fields**

Append a new helper to `/Users/sid/Developer/Python/marble/tests/test_convert_vgm_corpus_to_jsonl.py` (do NOT modify `_build_mini_corpus`; add a sibling):

```python
def _build_frame_corpus(root: Path) -> tuple[Path, Path]:
    """
    Synthetic corpus for frame-mode conversion.

    Rows include intro_end_sec / loop_seam_sec / total_sec fields.
    Splits: train=intro_loop, val=loop_from_start, test=through_composed.
    """
    audio_dir = root / "audio"

    rows = [
        {
            "id": "il_001",
            "split": "train",
            "loop_type": "intro_loop",
            "audio_path": "audio/train_il.wav",
            "intro_end_sec": 4.0,
            "loop_seam_sec": None,
            "total_sec": 8.0,
        },
        {
            "id": "lfs_001",
            "split": "val",
            "loop_type": "loop_from_start",
            "audio_path": "audio/val_lfs.wav",
            "intro_end_sec": None,
            "loop_seam_sec": 6.0,
            "total_sec": 12.0,
        },
        {
            "id": "tc_001",
            "split": "test",
            "loop_type": "through_composed",
            "audio_path": "audio/test_tc.wav",
            "intro_end_sec": None,
            "loop_seam_sec": None,
            "total_sec": 10.0,
        },
    ]

    _write_wav(root / "audio/train_il.wav", n_samples=48000)
    _write_wav(root / "audio/val_lfs.wav", n_samples=72000)
    _write_wav(root / "audio/test_tc.wav", n_samples=96000)

    manifest_path = root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(rows, f)

    return manifest_path, root


def _run_converter_mode(
    manifest: Path,
    audio_root: Path,
    out_dir: Path,
    name: str = "TestVGMFrame",
    mode: str = "clip",
) -> subprocess.CompletedProcess:
    """Run converter with --mode argument; returns CompletedProcess (does not raise)."""
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--manifest", str(manifest),
            "--audio-root", str(audio_root),
            "--out-dir", str(out_dir),
            "--name", name,
            "--mode", mode,
        ],
        capture_output=True,
        text=True,
    )
    return result
```

- [ ] **Step 2: Write the frame-mode tests (they will FAIL until the converter is extended)**

Append to `/Users/sid/Developer/Python/marble/tests/test_convert_vgm_corpus_to_jsonl.py`:

```python
# ─────────────────────────────────────────────────────────────────────────────
# Frame-mode tests
# ─────────────────────────────────────────────────────────────────────────────


def test_frame_mode_files_exist(tmp_path):
    """--mode frame writes one JSONL per split."""
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    result = _run_converter_mode(manifest, audio_root, out_dir, name="VGMLoopFrames", mode="frame")
    assert result.returncode == 0, f"Converter crashed:\n{result.stderr}"
    for split in ("train", "val", "test"):
        assert (out_dir / f"VGMLoopFrames.{split}.wav.jsonl").exists(), f"Missing {split} JSONL"


def test_frame_mode_label_is_dict(tmp_path):
    """In frame mode, label must be a dict with the four required keys."""
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter_mode(manifest, audio_root, out_dir, name="VGMLoopFrames", mode="frame")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGMLoopFrames.{split}.wav.jsonl")
        assert len(rows) == 1, f"Expected 1 row in {split}"
        label = rows[0]["label"]
        assert isinstance(label, dict), f"{split}: label must be a dict, got {type(label)}"
        for key in ("intro_end_sec", "loop_seam_sec", "loop_type", "total_sec"):
            assert key in label, f"{split}: missing key {key!r} in label"


def test_frame_mode_label_values(tmp_path):
    """Frame-mode label values match the manifest fields."""
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter_mode(manifest, audio_root, out_dir, name="VGMLoopFrames", mode="frame")

    # train: intro_loop with intro_end_sec=4.0, loop_seam_sec=None, total_sec=8.0
    train_rows = _read_jsonl(out_dir / "VGMLoopFrames.train.wav.jsonl")
    lbl = train_rows[0]["label"]
    assert lbl["loop_type"] == "intro_loop"
    assert lbl["intro_end_sec"] == pytest.approx(4.0)
    assert lbl["loop_seam_sec"] is None
    assert lbl["total_sec"] == pytest.approx(8.0)

    # val: loop_from_start with loop_seam_sec=6.0
    val_rows = _read_jsonl(out_dir / "VGMLoopFrames.val.wav.jsonl")
    lbl = val_rows[0]["label"]
    assert lbl["loop_type"] == "loop_from_start"
    assert lbl["intro_end_sec"] is None
    assert lbl["loop_seam_sec"] == pytest.approx(6.0)

    # test: through_composed, both None
    test_rows = _read_jsonl(out_dir / "VGMLoopFrames.test.wav.jsonl")
    lbl = test_rows[0]["label"]
    assert lbl["loop_type"] == "through_composed"
    assert lbl["intro_end_sec"] is None
    assert lbl["loop_seam_sec"] is None


def test_frame_mode_shares_splits_with_clip_mode(tmp_path):
    """
    Frame and clip modes must produce the same split assignment for the same manifest.
    Both modes use the manifest's 'split' field — verify train/val/test row counts agree.
    """
    # Use the frame corpus (has all frame fields); clip mode only needs loop_type + split
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir_clip = tmp_path / "clip"
    out_dir_frame = tmp_path / "frame"

    _run_converter_mode(manifest, audio_root, out_dir_clip, name="VGM", mode="clip")
    _run_converter_mode(manifest, audio_root, out_dir_frame, name="VGM", mode="frame")

    for split in ("train", "val", "test"):
        clip_rows = _read_jsonl(out_dir_clip / f"VGM.{split}.wav.jsonl")
        frame_rows = _read_jsonl(out_dir_frame / f"VGM.{split}.wav.jsonl")
        assert len(clip_rows) == len(frame_rows), (
            f"{split}: clip has {len(clip_rows)} rows, frame has {len(frame_rows)}"
        )


def test_clip_mode_still_emits_string_label(tmp_path):
    """Regression: --mode clip (default) must still emit label as a string."""
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter_mode(manifest, audio_root, out_dir, name="VGM", mode="clip")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGM.{split}.wav.jsonl")
        assert isinstance(rows[0]["label"], str), (
            f"{split}: clip mode label must be str, got {type(rows[0]['label'])}"
        )


def test_frame_mode_default_is_clip(tmp_path):
    """Invoking converter WITHOUT --mode must behave as --mode clip."""
    manifest, audio_root = _build_frame_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    # _run_converter (original helper) passes no --mode arg
    _run_converter(manifest, audio_root, out_dir, name="VGM")
    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGM.{split}.wav.jsonl")
        assert isinstance(rows[0]["label"], str), f"{split}: default mode label must be str"
```

- [ ] **Step 3: Run frame-mode tests to confirm they FAIL (expected)**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_convert_vgm_corpus_to_jsonl.py -k "frame" -v
```

Expected: all frame-mode tests FAIL (unknown argument `--mode`).

- [ ] **Step 4: Implement frame mode in the converter**

In `/Users/sid/Developer/Python/marble/scripts/data/convert_vgm_corpus_to_jsonl.py`:

**4a. Add `--mode` argument** to the argparse block (after `--name`):

```python
    ap.add_argument(
        "--mode",
        choices=["clip", "frame"],
        default="clip",
        metavar="MODE",
        help="Output label schema: 'clip' emits label=loop_type string (default); "
             "'frame' emits label=dict with intro_end_sec/loop_seam_sec/loop_type/total_sec.",
    )
```

**4b. Pass `mode` to a new `convert` function.** Replace `main()`'s inner loop with calls that select the label builder:

Replace the `out_row: dict = { ... }` block (lines ~159-167) with:

```python
        if args.mode == "frame":
            out_row: dict = {
                "audio_path": str(wav_path.resolve()),
                "sample_rate": FIXED_SAMPLE_RATE,
                "num_samples": num_samples,
                "channels": 1,
                "bit_depth": 16,
                "label": {
                    "intro_end_sec": row.get("intro_end_sec"),
                    "loop_seam_sec": row.get("loop_seam_sec"),
                    "loop_type": loop_type,
                    "total_sec": float(row.get("total_sec", 0.0)),
                },
                "duration": num_samples / FIXED_SAMPLE_RATE,
            }
        else:
            out_row = {
                "audio_path": str(wav_path.resolve()),
                "sample_rate": FIXED_SAMPLE_RATE,
                "num_samples": num_samples,
                "channels": 1,
                "bit_depth": 16,
                "label": loop_type,
                "duration": num_samples / FIXED_SAMPLE_RATE,
            }
```

- [ ] **Step 5: Run all converter tests**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_convert_vgm_corpus_to_jsonl.py -v
```

Expected: all tests PASS (original clip tests + new SR guard test + new frame-mode tests).

- [ ] **Step 6: Commit**

```bash
cd /Users/sid/Developer/Python/marble && git add scripts/data/convert_vgm_corpus_to_jsonl.py tests/test_convert_vgm_corpus_to_jsonl.py
git commit -m "feat(converter): add --mode frame for dict-label JSONL (VGMLoopFrames task)"
```

---

### Task 4: Push and verify

- [ ] **Step 1: Run the full relevant test suite**

```bash
cd /Users/sid/Developer/Python/marble && python -m pytest tests/test_convert_vgm_corpus_to_jsonl.py tests/test_vgmloopframes_probe_loss.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Push**

```bash
git push origin vgmloopstructure-task
```

- [ ] **Step 3: Report findings**

Per finding:
- T1 (CE shape crash): **REAL** — `CrossEntropyLoss(logits(B,L,3), targets(B,L))` misidentifies class dim. Fixed by flatten before loss call.
- T2 (SR guard): **REAL** — non-24k file accepted, duration computed wrong. Fixed by skip+warn after probe.
- T3 (frame-mode converter): **REAL gap** — no `--mode` flag existed. Fixed by adding `--mode {clip,frame}` with dict-label path.
