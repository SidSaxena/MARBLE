#!/usr/bin/env python3
"""Rewrite the audio_path field in JSONL records.

Why
───
Tasks like HookTheoryKey / HookTheoryStructure store ``audio_path`` as a
full string in each JSONL record (unlike HookTheoryMelody, which derives
``<audio_dir>/<youtube_id>.<ext>`` from a ``youtube.id`` key + the
datamodule's ``audio_ext`` parameter). To point those tasks at a
converted WAV corpus, we need to rewrite the ``audio_path`` strings —
swap the directory and/or the extension — and write to a new JSONL.

This script does that transformation. Generic: any task whose JSONL uses
an ``audio_path`` field can use it.

Properties
──────────
- **Atomic**: writes ``<out>.tmp`` then renames, so a Ctrl-C mid-write
  doesn't leave a half-written JSONL.
- **Idempotent**: ``--out-suffix`` defaults to ``.wav.jsonl`` so the
  original JSONL is left intact; re-running overwrites the WAV JSONL.
- **Schema-preserving**: every other field in each record is copied
  byte-for-byte; only ``audio_path`` is rewritten.
- **Validate-only mode**: ``--dry-run`` prints the first 5 rewrites
  without touching disk, useful for confirming the regex pattern works.

Usage
─────
    # HookTheoryKey: swap hooktheory_clips/ → hooktheory_clips_wav/ and .mp3 → .wav
    uv run python scripts/data/rewrite_jsonl_audio_paths.py \\
        --jsonl data/HookTheory/HookTheoryKey.train.jsonl \\
        --jsonl data/HookTheory/HookTheoryKey.val.jsonl \\
        --jsonl data/HookTheory/HookTheoryKey.test.jsonl \\
        --from-dir hooktheory_clips --to-dir hooktheory_clips_wav \\
        --from-ext .mp3 --to-ext .wav
    # → writes HookTheoryKey.{train,val,test}.wav.jsonl alongside originals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _out_path(jsonl: Path, out_suffix: str) -> Path:
    """``foo.train.jsonl`` → ``foo.train.wav.jsonl`` (or whatever ``out_suffix``).

    Built on ``stem`` so multi-dot names work — ``with_suffix`` would
    clobber ``.jsonl`` instead of appending.
    """
    return jsonl.with_name(jsonl.stem + out_suffix)


def _rewrite(audio_path: str, from_dir: str, to_dir: str, from_ext: str, to_ext: str) -> str:
    """Apply the dir + ext swap on a single audio_path string.

    The swaps are independent — pass an empty string for either to skip
    that side of the swap. Dir replacement is a literal substring match
    (not regex) to keep behaviour predictable.

    Output is always POSIX (forward slashes) so the rewritten JSONL is
    portable across Windows / Linux / macOS, regardless of what the
    input looked like or what OS this rewriter ran on.
    See marble/utils/path_compat.py.
    """
    # Import inside the function to keep this script importable from
    # contexts where marble is not on sys.path (e.g. a Modal sandbox
    # that only mounts scripts/data/).
    try:
        from marble.utils.path_compat import posix_path
    except ImportError:
        # Fallback: inline the same one-line transform so we never silently
        # drift from path_compat.posix_path. Tested against the canonical
        # helper in tests/test_path_compat.py.
        def posix_path(s: str) -> str:
            return s.replace("\\", "/")

    p = audio_path
    if from_dir and from_dir in p:
        # First-occurrence replace, so the same dir name appearing twice in
        # the path (rare but possible) only swaps the first instance.
        p = p.replace(from_dir, to_dir, 1)
    if from_ext and p.endswith(from_ext):
        p = p[: -len(from_ext)] + to_ext
    return posix_path(p)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="JSONL to rewrite. Repeat for multiple splits.",
    )
    ap.add_argument(
        "--from-dir",
        default="",
        help="Substring in audio_path to replace. Empty = skip dir swap.",
    )
    ap.add_argument(
        "--to-dir",
        default="",
        help="Replacement for --from-dir.",
    )
    ap.add_argument(
        "--from-ext",
        default=".mp3",
        help="File extension to replace (default: .mp3).",
    )
    ap.add_argument(
        "--to-ext",
        default=".wav",
        help="Replacement extension (default: .wav).",
    )
    ap.add_argument(
        "--out-suffix",
        default=".wav.jsonl",
        help="Suffix for the output file (replaces the input's .jsonl). Default: .wav.jsonl",
    )
    ap.add_argument(
        "--audio-path-key",
        default="audio_path",
        help="Top-level JSONL key that holds the audio path (default: audio_path).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first 5 rewrites without touching disk.",
    )
    args = ap.parse_args()

    if not args.from_dir and not (args.from_ext != args.to_ext):
        raise SystemExit("Nothing to rewrite — pass --from-dir and/or change --from-ext/--to-ext.")

    print(
        f"\nrewrite_jsonl_audio_paths: {len(args.jsonl)} JSONL(s)\n"
        f"  dir: {args.from_dir!r} → {args.to_dir!r}\n"
        f"  ext: {args.from_ext!r} → {args.to_ext!r}\n"
        f"  key: {args.audio_path_key!r}   out_suffix: {args.out_suffix!r}\n"
        f"  dry_run: {args.dry_run}",
        flush=True,
    )

    for jsonl in args.jsonl:
        if not jsonl.exists():
            print(f"  ! missing JSONL: {jsonl}, skipping")
            continue
        out = _out_path(jsonl, args.out_suffix)
        tmp = out.with_suffix(out.suffix + ".tmp")
        n_records = n_rewritten = n_missing = 0
        previews: list[tuple[str, str]] = []
        if args.dry_run:
            sink = None
        else:
            sink = tmp.open("w")
        try:
            with jsonl.open() as src:
                for raw in src:
                    line = raw.strip()
                    if not line:
                        continue
                    n_records += 1
                    rec = json.loads(line)
                    orig = rec.get(args.audio_path_key)
                    if not isinstance(orig, str):
                        n_missing += 1
                        if sink:
                            sink.write(line + "\n")
                        continue
                    new = _rewrite(
                        orig,
                        args.from_dir,
                        args.to_dir,
                        args.from_ext,
                        args.to_ext,
                    )
                    if new != orig:
                        n_rewritten += 1
                        rec[args.audio_path_key] = new
                        if len(previews) < 5:
                            previews.append((orig, new))
                    if sink:
                        sink.write(json.dumps(rec) + "\n")
        finally:
            if sink:
                sink.close()
        if previews:
            print(f"\n  ── {jsonl.name} ──")
            for o, n in previews:
                print(f"    {o}")
                print(f"      → {n}")
        if args.dry_run:
            print(
                f"  [dry-run] {jsonl.name}: {n_records:,} records, "
                f"{n_rewritten:,} rewritten, {n_missing:,} missing key"
            )
        else:
            tmp.replace(out)
            print(
                f"  ✓ {jsonl.name} → {out.name}  "
                f"({n_records:,} records, {n_rewritten:,} rewritten, {n_missing:,} missing key)"
            )


if __name__ == "__main__":
    main()
