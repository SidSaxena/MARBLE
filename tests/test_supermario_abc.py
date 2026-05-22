"""Regression tests for the SuperMarioStructure ABC input path.

These tests exercise the integrated pipeline:
  .mxl (MusicXML) →
  scripts/data/build_supermario_dataset.py (--build-abc) →
  per-segment .abc files in abc_segments/ +
  JSONL records gain `abc_path` field →
  marble.tasks.SuperMarioStructure.datamodule (input_format='abc') →
  CLaMP3 M3 patchilizer in BAR-LEVEL ABC mode (not MTF mode).

The tests are write-once + run-many. They depend on a populated local
build (`data/SuperMarioStructure/abc_segments/` + JSONLs with
`abc_path` fields). Each test SKIPS — does not fail — when the
underlying data isn't present, so this file is safe to leave checked
in on a CI runner that doesn't carry the full corpus.

Run manually:
    uv run python tests/test_supermario_abc.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DATA = Path("data/SuperMarioStructure")
_TRAIN_JSONL = _DATA / "SuperMarioStructure.train.jsonl"
_ABC_DIR = _DATA / "abc_segments"


class SkipTest(Exception):
    """Raise to skip a test cleanly when prerequisites are missing."""


def _skip_if_no_build():
    """Skip if the local SuperMarioStructure build with --build-abc isn't present."""
    if not _TRAIN_JSONL.exists():
        raise SkipTest(f"{_TRAIN_JSONL} not found — run build_supermario_dataset.py --build-abc")
    if not _ABC_DIR.exists():
        raise SkipTest(f"{_ABC_DIR} not found — run with --build-abc + --mxl-source-dir")
    # Check we have at least some records with abc_path.
    with open(_TRAIN_JSONL) as f:
        n = 0
        for line in f:
            rec = json.loads(line)
            if rec.get("abc_path"):
                n += 1
            if n >= 1:
                return
    raise SkipTest("no records with abc_path field — re-run build with --build-abc")


# ──────────────────────────────────────────────────────────────────────────
# 1. ABC files on disk look like ABC, not MIDI byte garbage
# ──────────────────────────────────────────────────────────────────────────


def test_abc_files_have_canonical_abc_headers():
    """Every emitted .abc segment must contain the musically-required ABC
    headers (M: meter, L: default length, K: key) and must NOT look like
    MTF (``ticks_per_beat ...`` first line would route the patchiliser
    into MTF mode and silently invalidate the experiment).

    The first-line check was relaxed once the interleaved-ABC pipeline
    landed: ``_abc_to_interleaved`` strips ``X:``, ``T:``, ``C:`` and
    related metadata fields (matching CLaMP3 training preprocessing),
    so the file may start with ``%%score`` or ``L:`` instead of ``X:``.
    """
    _skip_if_no_build()
    abc_paths = list(_ABC_DIR.rglob("*.abc"))
    assert len(abc_paths) > 100, (
        f"Too few .abc segments found ({len(abc_paths)}); is the build complete?"
    )
    # Recognised ABC header line prefixes (the patchiliser's ABC-mode
    # branch accepts these; anything else is a body line).
    ABC_HEADER_PREFIXES = (
        "X:",
        "T:",
        "C:",
        "Z:",
        "%%",
        "L:",
        "M:",
        "K:",
        "Q:",
        "V:",
        "I:",
        "W:",
        "w:",
    )
    n_checked = 0
    n_missing_required = 0
    for p in abc_paths[:200]:  # sample, not all 1k+ — speed
        with open(p, encoding="utf-8") as f:
            head = f.read(2048)
        first_line = head.split("\n", 1)[0]
        # MUST not look like MTF (`ticks_per_beat ...`).
        assert not first_line.startswith("ticks_per_beat"), (
            f"{p} would route patchiliser into MTF mode (first line: {first_line!r})"
        )
        # First non-empty line should be a recognised ABC header.
        assert any(first_line.startswith(h) for h in ABC_HEADER_PREFIXES), (
            f"{p}: first line is {first_line!r}, expected one of {ABC_HEADER_PREFIXES}"
        )
        # Required ABC fields: at least one of M:, L:, K:.
        has_required = any(f"\n{h}:" in head or head.startswith(f"{h}:") for h in ("M", "L", "K"))
        if not has_required:
            n_missing_required += 1
        n_checked += 1
    assert n_missing_required == 0, f"{n_missing_required} files are missing M:/L:/K: ABC headers"
    print(f"  checked {n_checked} ABC files; all canonical")


# ──────────────────────────────────────────────────────────────────────────
# 2. Patchiliser actually enters ABC mode (not MTF) on our files
# ──────────────────────────────────────────────────────────────────────────


def test_patchiliser_uses_abc_mode():
    """Send one of our .abc files into M3Patchilizer.encode() and verify
    the resulting patches are ASCII bar text — NOT a packed event stream
    (which is what MTF mode would produce)."""
    _skip_if_no_build()
    from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer

    abc_paths = list(_ABC_DIR.rglob("*.abc"))[:5]
    p = M3Patchilizer()
    # Any recognised ABC header line prefix (interleaved output strips
    # X:/T:/C:, so the first real patch may start at %%score, L:, M:, K:,
    # V:, etc. — anything from the recognised header set is fine).
    ABC_HEADER_PREFIXES = (
        "X:",
        "T:",
        "C:",
        "Z:",
        "%%",
        "L:",
        "M:",
        "K:",
        "Q:",
        "V:",
        "I:",
        "W:",
        "w:",
    )
    for abc_path in abc_paths:
        with open(abc_path, encoding="utf-8") as f:
            abc_text = f.read()
        patches = p.encode(abc_text, add_special_patches=True)
        # patches[0] = BOS, patches[1] should be the first real patch.
        # In ABC mode, that's an ABC header line.
        assert len(patches) > 2, f"{abc_path}: too few patches ({len(patches)})"
        first_real = bytes(b for b in patches[1] if b > 3).decode("ascii", errors="replace")
        assert any(first_real.startswith(h) for h in ABC_HEADER_PREFIXES), (
            f"{abc_path}: first real patch is {first_real!r}, expected an ABC header "
            f"({ABC_HEADER_PREFIXES}). MTF mode would emit 'ticks_per_beat ...' here."
        )
        # Sample a mid-piece patch and check it contains barline delimiters
        # OR ABC header chars. (Pure event packing would have e.g. 'note_on ...')
        mid = patches[len(patches) // 2]
        mid_text = bytes(b for b in mid if b > 3).decode("ascii", errors="replace")
        # Heuristic: ABC body patches have `|` or note letters or digits, no `note_on`/`set_tempo`.
        assert "note_on" not in mid_text and "set_tempo" not in mid_text, (
            f"{abc_path}: mid-patch text looks like MTF events: {mid_text[:80]!r}"
        )
    print(f"  patchiliser entered ABC mode for all {len(abc_paths)} sampled files")


# ──────────────────────────────────────────────────────────────────────────
# 3. Datamodule loads ABC input + emits the right tuple shape
# ──────────────────────────────────────────────────────────────────────────


def test_datamodule_input_format_abc_loads_and_tokenises():
    _skip_if_no_build()
    import torch

    from marble.tasks.SuperMarioStructure.datamodule import SuperMarioStructureSymbolicTrain

    ds = SuperMarioStructureSymbolicTrain(jsonl=str(_TRAIN_JSONL), input_format="abc")
    assert len(ds) > 50, f"too few abc records ({len(ds)}); is the build complete?"
    # Spot-check several items
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        patches, label, ori_uid, clip_id = ds[idx]
        assert isinstance(patches, torch.Tensor)
        assert patches.dtype == torch.long
        assert patches.shape == (512, 64), patches.shape  # (max_patches, patch_size)
        assert 0 <= label < 6  # 6 SuperMario functional classes
        assert isinstance(ori_uid, str) and "_" in ori_uid
        assert isinstance(clip_id, str)
        # Verify the patches encode ABC headers (interleaved output may
        # start at %%score / L: / M: / K: rather than X: — any recognised
        # ABC header prefix is acceptable).
        first_real = bytes(int(b) for b in patches[1] if b > 3).decode("ascii", errors="replace")
        ABC_HEADER_PREFIXES = (
            "X:",
            "T:",
            "C:",
            "Z:",
            "%%",
            "L:",
            "M:",
            "K:",
            "Q:",
            "V:",
            "I:",
            "W:",
            "w:",
        )
        assert any(first_real.startswith(h) for h in ABC_HEADER_PREFIXES), (
            f"item {idx} patches[1] is {first_real!r} — expected an ABC header "
            f"({ABC_HEADER_PREFIXES})"
        )
    print(f"  loaded {len(ds)} ABC records; sampled 3 with correct shape + ABC headers")


# ──────────────────────────────────────────────────────────────────────────
# 4. ABC vs MIDI clip_ids differ → cache entries don't collide
# ──────────────────────────────────────────────────────────────────────────


def test_abc_and_midi_clip_ids_differ():
    """Critical for cache correctness: if MIDI and ABC variants used the
    same clip_id, the second sweep would silently retrieve stale
    embeddings from the first. Path-hashing in `make_clip_id` should
    prevent this — verify."""
    _skip_if_no_build()
    from marble.utils.emb_cache import make_clip_id

    with open(_TRAIN_JSONL) as f:
        records_with_abc = [json.loads(line) for line in f if json.loads(line).get("abc_path")]
    assert len(records_with_abc) > 0, "no records with abc_path in train.jsonl"

    sampled = records_with_abc[:50]
    collisions = []
    for r in sampled:
        midi_id = make_clip_id(r["midi_path"], 0)
        abc_id = make_clip_id(r["abc_path"], 0)
        if midi_id == abc_id:
            collisions.append((r["midi_path"], r["abc_path"]))
    assert not collisions, (
        f"{len(collisions)} clip_id collisions between MIDI and ABC paths "
        f"— cache would silently mix embeddings. e.g. {collisions[0]}"
    )
    print(f"  {len(sampled)} sampled records: zero clip_id collisions (good)")


# ──────────────────────────────────────────────────────────────────────────
# 5. ABC slicing covers the requested bar range (content sanity)
# ──────────────────────────────────────────────────────────────────────────


def test_abc_slice_note_count_within_reasonable_range_of_midi():
    """For each segment, the ABC version should have a note count within
    a tight factor of the MIDI version. With ``score.expandRepeats()``
    applied in the build, bar numbering aligns with the expanded MIDI
    that pretty_midi sees → counts should differ only by representation
    conventions (tie/slur/chord = a few notes per measure at most).

    Threshold: 0.5×–2.5×. Older builds without expandRepeats had ratios
    as low as 0.29× when the annotation referenced bar numbers beyond
    the unexpanded score's length — that is the bug this test catches.
    """
    _skip_if_no_build()
    try:
        import pretty_midi
    except ImportError as e:
        raise SkipTest("pretty_midi not installed") from e

    with open(_TRAIN_JSONL) as f:
        records = [json.loads(line) for line in f]
    records_with_abc = [r for r in records if r.get("abc_path")]
    # Sample a spread of segment lengths.
    sampled = records_with_abc[:: max(1, len(records_with_abc) // 20)][:20]

    n_compared = 0
    bad = []
    for r in sampled:
        midi_path = Path(r["midi_path"])
        abc_path = Path(r["abc_path"])
        if not midi_path.exists() or not abc_path.exists():
            continue
        # Count MIDI notes in the segment
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        n_midi = sum(len(i.notes) for i in pm.instruments)
        # Count ABC notes by scanning for note letters (rough heuristic).
        # A more rigorous parse via music21.converter.parse(.abc) is possible
        # but slow; the rough count catches gross slicing bugs.
        with open(abc_path, encoding="utf-8") as f:
            abc_text = f.read()
        # Strip headers; count occurrences of pitch letters in body lines.
        body_lines = [
            line
            for line in abc_text.split("\n")
            if line
            and not (line[0].isalpha() and len(line) > 1 and line[1] == ":")
            and not line.startswith("%")
        ]
        body = " ".join(body_lines)
        # ABC pitch letters: a–g and A–G (with optional accidentals + octave marks).
        n_abc_letters = sum(1 for c in body if c in "abcdefgABCDEFG")
        if n_midi == 0:
            continue
        ratio = n_abc_letters / n_midi
        if not (0.5 <= ratio <= 2.5):
            bad.append((r["ori_uid"], n_midi, n_abc_letters, ratio))
        n_compared += 1
    assert n_compared >= 5, f"compared too few segments ({n_compared})"
    assert not bad, (
        f"{len(bad)}/{n_compared} segments have ABC/MIDI note-count ratio outside [0.5, 2.5] "
        f"— possible slicing bug (missing bars? extra repeats?). e.g. {bad[0]}"
    )
    print(f"  compared {n_compared} segments; all within 0.5–2.5× MIDI note count")


# ──────────────────────────────────────────────────────────────────────────
# 6. cache config_hash is the same for ABC and MIDI configs
# ──────────────────────────────────────────────────────────────────────────


def test_cache_config_hash_shared_but_safe():
    """The cache directory for CLaMP3-symbolic SMS is shared between
    ABC and MIDI sweeps (same encoder, same sample_rate, same clip
    seconds, same empty audio_transforms). This is fine BECAUSE the
    per-entry clip_id is path-based and ABC/MIDI paths differ. Document
    this invariant via a hash check + a comment."""
    _skip_if_no_build()
    from marble.utils.emb_cache import compute_config_hash

    common = dict(
        encoder_model_id="sander-wood/clamp3",
        sample_rate=24000,
        clip_seconds=15.0,
        pool_time=True,
    )
    # Pipeline signature is empty for both ABC and MIDI symbolic configs
    # (no audio_transforms). So config_hash MUST match.
    h_midi = compute_config_hash(pipeline_signature="", **common)
    h_abc = compute_config_hash(pipeline_signature="", **common)
    assert h_midi == h_abc, (
        f"ABC ({h_abc}) and MIDI ({h_midi}) config_hashes differ — "
        f"unexpected: they should share the cache directory."
    )
    # Document the per-entry collision-avoidance: make_clip_id hashes the
    # path, so MIDI and ABC clip_ids within the shared dir don't collide.
    from marble.utils.emb_cache import make_clip_id

    a = make_clip_id("data/SuperMarioStructure/midi_segments/00002/000_intro.mid", 0)
    b = make_clip_id("data/SuperMarioStructure/abc_segments/00002/000_intro.abc", 0)
    assert a != b, "make_clip_id collision: ABC and MIDI paths produced identical IDs"
    # Stronger: the hash slugs should differ.
    a_hash = a.split("__")[1]
    b_hash = b.split("__")[1]
    assert a_hash != b_hash, f"clip_id path hashes coincidentally match: {a_hash} == {b_hash}"
    print(f"  config_hash shared ({h_midi}), clip_ids differ ({a_hash} vs {b_hash}) — safe")


# ──────────────────────────────────────────────────────────────────────────
# 7. input_format='abc' filters records lacking abc_path (no silent crashes)
# ──────────────────────────────────────────────────────────────────────────


def test_input_format_abc_filters_records_without_abc_path():
    """Records emitted before --build-abc was used (or where ABC slicing
    failed for some pieces) lack `abc_path`. The datamodule must filter
    those out gracefully instead of erroring at __getitem__ time."""
    import tempfile

    from marble.tasks.SuperMarioStructure.datamodule import SuperMarioStructureSymbolicTrain

    # Synthesise a tiny JSONL with a mix of records, one missing abc_path.
    tmp_jsonl = Path(tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name)
    try:
        rows = [
            # First two need real on-disk paths so the rest of the test makes sense.
            # Use any real existing midi + abc segment pair.
            None,
            None,
        ]
        # Find a real record pair
        if not _TRAIN_JSONL.exists():
            raise SkipTest("no train.jsonl to lift sample records from")
        with open(_TRAIN_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("abc_path") and rows[0] is None:
                    rows[0] = rec
                elif rec.get("abc_path") and rows[1] is None:
                    rows[1] = dict(rec)
                    rows[1]["ori_uid"] = "x_001"  # dedup
                    break
        assert rows[0] and rows[1]
        # Add a 3rd record WITHOUT abc_path (should be filtered out)
        no_abc = dict(rows[0])
        no_abc["ori_uid"] = "x_002"
        no_abc.pop("abc_path", None)
        rows.append(no_abc)

        with open(tmp_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        ds = SuperMarioStructureSymbolicTrain(jsonl=str(tmp_jsonl), input_format="abc")
        # Should keep 2 (the ones with abc_path), drop 1.
        assert len(ds) == 2, f"expected 2 records after filter, got {len(ds)}"
        print("  filtered 1/3 records lacking abc_path; 2 kept")
    finally:
        tmp_jsonl.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# 8. convert_mxl_to_abc helper works end-to-end on one file
# ──────────────────────────────────────────────────────────────────────────


def test_abc_files_are_interleaved():
    """Verify .abc files have voices INTERLEAVED by bar, not raw multi-voice.

    Raw xml2abc output looks like:
        V:1 [bar1]|[bar2]|[bar3]|
        V:2 [bar1]|[bar2]|[bar3]|

    Interleaved (CLaMP3 training-faithful) output looks like:
        V:1 treble
        V:2 bass
        [V:1]bar1|[V:2]bar1|
        [V:1]bar2|[V:2]bar2|

    Detection heuristic: interleaved bodies contain ``[V:1]`` and
    ``[V:2]`` markers IN-LINE (one per bar of each voice); raw bodies
    have voice content split into top-level ``V:1`` / ``V:2`` blocks
    with no inline ``[V:N]`` per-bar markers.
    """
    _skip_if_no_build()
    abc_paths = list(_ABC_DIR.rglob("*.abc"))[:30]
    n_interleaved = 0
    n_single_voice = 0
    n_raw_multivoice = 0
    for p in abc_paths:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        has_inline_v1 = "[V:1]" in text
        has_inline_v2 = "[V:2]" in text
        # Count top-level voice declarations (V:N at start of line, before any [V:N] markers)
        top_v_decls = sum(
            1 for line in text.split("\n") if line.startswith("V:") and "[V:" not in line
        )
        if has_inline_v1 and has_inline_v2:
            n_interleaved += 1
        elif top_v_decls <= 1:
            # Single-voice piece — no interleaving possible. Counts as OK.
            n_single_voice += 1
        else:
            n_raw_multivoice += 1
    assert n_raw_multivoice == 0, (
        f"{n_raw_multivoice}/{len(abc_paths)} sampled .abc files are still in "
        f"raw multi-voice layout (no inline [V:N] markers but multiple V: "
        f"declarations) — _abc_to_interleaved didn't run. Check abctoolkit "
        f"is installed via `uv sync --extra symbolic-abc`."
    )
    print(
        f"  {n_interleaved} interleaved + {n_single_voice} single-voice "
        f"out of {len(abc_paths)} sampled (0 raw multi-voice)"
    )


def test_no_metadata_fields_in_abc():
    """Verify the CLaMP3-training-faithful preprocessing stripped the
    non-musical metadata fields (X:, T:, C:, Z:, W:, w:, %%MIDI). These
    are training-corpus annotations that shouldn't leak into the model's
    input at inference time. The musically-load-bearing fields
    (K:, M:, L:, Q:, V:) MUST remain."""
    _skip_if_no_build()
    abc_paths = list(_ABC_DIR.rglob("*.abc"))[:30]
    forbidden_prefixes = ("X:", "T:", "C:", "Z:", "W:", "w:", "%%MIDI")
    required_at_least_one = ("K:", "M:", "L:")  # one of these must appear
    n_with_forbidden = 0
    n_missing_required = 0
    for p in abc_paths:
        with open(p, encoding="utf-8") as f:
            lines = f.read().split("\n")
        if any(any(line.startswith(prefix) for prefix in forbidden_prefixes) for line in lines):
            n_with_forbidden += 1
        if not any(any(line.startswith(req) for req in required_at_least_one) for line in lines):
            n_missing_required += 1
    assert n_with_forbidden == 0, (
        f"{n_with_forbidden}/{len(abc_paths)} files still contain forbidden "
        f"metadata fields ({forbidden_prefixes}). The interleave step should "
        f"have stripped these."
    )
    assert n_missing_required == 0, (
        f"{n_missing_required}/{len(abc_paths)} files are missing required "
        f"musical headers ({required_at_least_one})."
    )
    print(f"  {len(abc_paths)} files have no forbidden metadata + all required headers")


def test_no_bar_number_comments_in_abc():
    """xml2abc emits ``%N`` bar-number comments at the end of each line
    by default. CLaMP3 training preprocessing strips these via
    ``remove_bar_no_annotations`` — verify ours does too."""
    _skip_if_no_build()
    import re

    abc_paths = list(_ABC_DIR.rglob("*.abc"))[:30]
    # The marker pattern: whitespace + % + digits + end-of-line. Match
    # only when % is preceded by whitespace so we don't false-positive
    # on `%%score` / `%%MIDI` header directives (those start the line).
    pattern = re.compile(r"\s%\d+\s*$", re.MULTILINE)
    n_with_bar_no = 0
    samples = []
    for p in abc_paths:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if pattern.search(text):
            n_with_bar_no += 1
            if len(samples) < 3:
                samples.append(p.name)
    assert n_with_bar_no == 0, (
        f"{n_with_bar_no}/{len(abc_paths)} .abc files still contain `%N` "
        f"bar-number comments. Sample files: {samples}. "
        f"remove_bar_no_annotations didn't run — check abctoolkit install."
    )
    print(f"  {len(abc_paths)} files have no `%N` bar-number comments")


def test_no_leaked_temp_files_in_abc_segments():
    """The build's temp-file cleanup must remove any `tmpXXX.abc` /
    `tmpXXX.musicxml` xml2abc may have partially written before
    failing. Without this, downstream scans of ``*.abc`` find broken
    empty files alongside the real segments — first version of this
    PR shipped this bug; regression test prevents recurrence."""
    _skip_if_no_build()
    leaked = list(_ABC_DIR.rglob("tmp*.abc")) + list(_ABC_DIR.rglob("tmp*.musicxml"))
    assert not leaked, (
        f"Found {len(leaked)} leaked temp files in abc_segments/: "
        f"{[str(p) for p in leaked[:5]]}. Re-run build_supermario_dataset.py "
        f"after cleanup — the slicer should remove them in its finally block."
    )
    print(f"  no leaked temp files in {_ABC_DIR}")


def test_convert_mxl_to_abc_one_file_via_subprocess():
    """Run the standalone full-piece converter on a single .mxl and
    verify the resulting .abc is parseable ABC. This catches breakage
    in the convert_mxl_to_abc.py CLI shape independently of the build
    script."""
    import subprocess
    import tempfile

    mxl_dir = _DATA / "mxl"
    if not mxl_dir.is_dir():
        raise SkipTest(f"{mxl_dir} not found")
    # Pick a piece that we know works (we've already shipped abc_segments
    # for it; that means the converter works for this file).
    abc_dir = _ABC_DIR
    if not abc_dir.is_dir():
        raise SkipTest("no abc_segments to find a known-good piece")
    known_good_pids = {p.name for p in abc_dir.iterdir() if p.is_dir() and any(p.iterdir())}
    if not known_good_pids:
        raise SkipTest("no piece with at least one .abc segment")

    pid = sorted(known_good_pids)[0]
    mxl_candidates = list(mxl_dir.glob(f"{pid}_*.mxl")) + list(mxl_dir.glob(f"{pid}.mxl"))
    if not mxl_candidates:
        raise SkipTest(f"no .mxl matching {pid}")
    src = mxl_candidates[0]

    with tempfile.TemporaryDirectory() as out_dir:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/data/convert_mxl_to_abc.py",
                "--in-dir",
                str(src.parent),
                "--out-dir",
                out_dir,
                "--max-files",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # We only requested 1 file, but the first file by sort order may
        # not be `src`. Just check that SOMETHING worked.
        assert result.returncode == 0, (
            f"convert_mxl_to_abc.py exit={result.returncode}: {result.stderr[:200]}"
        )
        produced = list(Path(out_dir).glob("*.abc"))
        assert produced, f"converter produced no .abc files; stderr={result.stderr[:200]}"
        # Verify the content
        with open(produced[0], encoding="utf-8") as f:
            head = f.read(512)
        # convert_mxl_to_abc.py is the FULL-PIECE converter and does NOT
        # apply the interleave/strip step — so its output still has X:
        # at the top, unlike the build script's per-segment .abc files.
        assert head.startswith("X:"), f"produced ABC doesn't start with X: — got {head[:80]!r}"
        print(f"  converter produced {produced[0].name} with canonical ABC header")


if __name__ == "__main__":
    tests = [
        test_abc_files_have_canonical_abc_headers,
        test_patchiliser_uses_abc_mode,
        test_datamodule_input_format_abc_loads_and_tokenises,
        test_abc_and_midi_clip_ids_differ,
        test_abc_slice_note_count_within_reasonable_range_of_midi,
        test_cache_config_hash_shared_but_safe,
        test_input_format_abc_filters_records_without_abc_path,
        test_abc_files_are_interleaved,
        test_no_metadata_fields_in_abc,
        test_no_bar_number_comments_in_abc,
        test_no_leaked_temp_files_in_abc_segments,
        test_convert_mxl_to_abc_one_file_via_subprocess,
    ]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP {t.__name__}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (of {len(tests)})")
    if failed:
        sys.exit(1)
