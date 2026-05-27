#!/usr/bin/env python3
"""
scripts/data/verify_retrieval_jsonl.py
──────────────────────────────────────
Audit a MARBLE retrieval-task JSONL against the audio files on disk.

Use this BEFORE launching any retrieval sweep (Covers80, SHS100K,
VGMIDITVar, VGMIDITVar-timbre) to catch silent failures from a
half-finished conversion, a stale Modal pull, or a mismatched
``sample_rate`` field. The probe trusts the JSONL's
``sample_rate``/``num_samples`` to slice clips — if they disagree with
the file on disk, the slicing math goes off the rails and every
embedding is wrong.

Checks per record:
  1. ``audio_path`` resolves on disk (relative to repo root).
  2. ``ffprobe`` succeeds and returns ``sample_rate``,
     ``num_samples``, ``channels``.
  3. JSONL ``sample_rate`` matches ffprobe (no tolerance — it's an int).
  4. JSONL ``num_samples`` is within ``--num-samples-tolerance`` of
     ffprobe (default 1% — ffprobe's duration→samples conversion
     rounds, so absolute equality is too strict).
  5. If ``--target-sr`` is given, ``sample_rate`` matches it exactly
     for >= ``--target-sr-min-frac`` of records.
  6. ``work_id`` parses as int.

Exit code 0 on full pass, 1 on any failure.

Usage
-----
  python scripts/data/verify_retrieval_jsonl.py \\
      --jsonl data/SHS100K/SHS100K.test.jsonl \\
      --audio-dir data/SHS100K/audio \\
      --target-sr 24000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ffprobe(path: Path) -> tuple[int, int, int] | None:
    """Return ``(sample_rate, num_samples, channels)`` or None on failure."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    sr = 0
    ch = 1
    dur_s: str | None = None
    for s in data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        try:
            sr = int(s["sample_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        ch = int(s.get("channels") or 1)
        dur_s = s.get("duration")
        break
    if sr == 0:
        return None
    dur = 0.0
    for raw in [dur_s, data.get("format", {}).get("duration")]:
        if raw is None:
            continue
        try:
            dur = float(raw)
            if dur > 0:
                break
        except (ValueError, TypeError):
            continue
    if dur <= 0:
        return None
    return sr, int(round(sr * dur)), ch


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jsonl", required=True, help="Retrieval-task JSONL path")
    ap.add_argument(
        "--audio-dir",
        default=None,
        help="Resolve relative audio_path entries against this dir. "
        "Default: parent of --jsonl (most production JSONLs use "
        "relative paths already, this is a no-op).",
    )
    ap.add_argument(
        "--target-sr",
        type=int,
        default=None,
        help="Pin every record's sample_rate to this value (strict match).",
    )
    ap.add_argument(
        "--target-sr-min-frac",
        type=float,
        default=1.0,
        help="Minimum fraction of records that must have sample_rate == "
        "--target-sr (default 1.0 = all). Soften for partial-conversion "
        "scenarios.",
    )
    ap.add_argument(
        "--num-samples-tolerance",
        type=float,
        default=0.01,
        help="Allowable |JSONL.num_samples - ffprobe.num_samples| / "
        "ffprobe.num_samples (default 1%).",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Audit at most this many records (smoke-test).",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-failure prints; just emit counters + exit code.",
    )
    args = ap.parse_args()

    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not on PATH. Install ffmpeg.", file=sys.stderr)
        return 1

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"ERROR: jsonl not found: {jsonl_path}", file=sys.stderr)
        return 1

    # ``audio_path`` entries in retrieval JSONLs are usually repo-relative
    # POSIX paths. We resolve relative-to-CWD by default; ``--audio-dir``
    # only matters if the JSONL stores bare filenames.
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    n_total = 0
    n_ok = 0
    n_missing = 0
    n_sr_mismatch = 0
    n_samples_mismatch = 0
    n_bad_work_id = 0
    n_ffprobe_fail = 0
    n_target_sr_hit = 0

    failures: list[str] = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if args.max_records is not None and n_total >= args.max_records:
                break
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                failures.append(f"line {line_no}: not valid JSON ({e})")
                continue

            ap_str = rec.get("audio_path")
            if not isinstance(ap_str, str):
                failures.append(f"line {line_no}: audio_path missing or non-string")
                continue
            path = Path(ap_str)
            if not path.is_absolute() and audio_dir is not None and not path.exists():
                path = audio_dir / Path(ap_str).name
            if not path.exists():
                n_missing += 1
                failures.append(f"line {line_no}: missing on disk: {ap_str}")
                continue

            wid_raw = rec.get("work_id")
            try:
                int(wid_raw)
            except (TypeError, ValueError):
                n_bad_work_id += 1
                failures.append(f"line {line_no}: work_id not int: {wid_raw!r}")
                continue

            info = ffprobe(path)
            if info is None:
                n_ffprobe_fail += 1
                failures.append(f"line {line_no}: ffprobe failed on {ap_str}")
                continue
            sr_probe, n_probe, _ = info
            sr_json = int(rec.get("sample_rate", 0))
            n_json = int(rec.get("num_samples", 0))

            if sr_json != sr_probe:
                n_sr_mismatch += 1
                failures.append(
                    f"line {line_no}: sample_rate JSONL={sr_json} vs ffprobe={sr_probe} on {ap_str}"
                )
                continue

            if n_probe > 0:
                rel_err = abs(n_json - n_probe) / n_probe
                if rel_err > args.num_samples_tolerance:
                    n_samples_mismatch += 1
                    failures.append(
                        f"line {line_no}: num_samples JSONL={n_json} vs ffprobe={n_probe} "
                        f"(rel err={rel_err:.4f}) on {ap_str}"
                    )
                    continue

            if args.target_sr is not None and sr_probe == args.target_sr:
                n_target_sr_hit += 1

            n_ok += 1

    # ── Report ───────────────────────────────────────────────────────────
    if not args.summary_only:
        for f_str in failures[:30]:
            print(f_str)
        if len(failures) > 30:
            print(f"... and {len(failures) - 30} more failures")

    print()
    print(f"Audited {n_total} records from {jsonl_path}")
    print(f"  ok                 : {n_ok}")
    print(f"  missing on disk    : {n_missing}")
    print(f"  ffprobe failed     : {n_ffprobe_fail}")
    print(f"  sample_rate mismatch: {n_sr_mismatch}")
    print(f"  num_samples mismatch: {n_samples_mismatch}")
    print(f"  bad work_id        : {n_bad_work_id}")

    rc = 0
    if args.target_sr is not None and n_total > 0:
        frac = n_target_sr_hit / n_total
        print(f"  sample_rate == {args.target_sr}: {n_target_sr_hit} ({frac:.2%})")
        if frac < args.target_sr_min_frac:
            print(
                f"FAIL: target-sr coverage {frac:.2%} < --target-sr-min-frac {args.target_sr_min_frac:.2%}"
            )
            rc = 1

    if n_total == 0:
        print("FAIL: zero records in JSONL")
        return 1
    if n_ok != n_total:
        print(f"FAIL: {n_total - n_ok} record(s) had issues")
        return 1
    print("PASS")
    return rc


if __name__ == "__main__":
    sys.exit(main())
