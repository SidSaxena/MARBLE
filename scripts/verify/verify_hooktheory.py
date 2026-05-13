#!/usr/bin/env python3
"""
scripts/verify/verify_hooktheory.py
────────────────────────────
Audit the HookTheory pre-segmented clips against the shipped JSONL.

Why this exists
---------------
The m-a-p/HookTheory release ships ``hooktheory_clips/*.mp3`` plus JSONL
files that include ``num_samples`` and ``sample_rate`` per clip.  Those
fields were computed by the dataset authors using a particular MP3
decoder (most likely ffmpeg/ffprobe on Linux).

On other platforms — especially Windows with torchaudio routed through
the soundfile backend's mpg123 — the *decoded* frame count can be a few
thousand samples shorter than the JSONL's ``num_samples`` value because:

  • LAME tail padding is interpreted differently
  • Frame-boundary handling differs at the file edges
  • Some decoders count ID3 header bytes; some don't

When the mismatch crosses a ``clip_seconds`` boundary, the index_map ends
up generating a slice whose ``frame_offset`` lies past the file's actual
decodable end.  ``torchaudio.load`` then returns ``(channels, 0)`` and
the downstream resample call dies with::

    cannot reshape tensor of 0 elements into shape [-1, 0]

The datamodule now has a defensive guard that replaces empty waveforms
with silent audio.  This script diagnoses how widespread the mismatch is
and optionally writes a corrected JSONL.

Usage
-----
  # Just diagnose — no writes
  python scripts/verify/verify_hooktheory.py \\
      --jsonl data/HookTheory/HookTheoryKey.train.jsonl

  # Diagnose all six standard splits in one go
  python scripts/verify/verify_hooktheory.py \\
      --jsonl-dir data/HookTheory

  # Rewrite each JSONL with torchaudio-derived num_samples (one-shot fix)
  python scripts/verify/verify_hooktheory.py \\
      --jsonl-dir data/HookTheory --rewrite
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torchaudio

log = logging.getLogger(__name__)


def _probe(path: Path) -> Optional[tuple[int, int, int]]:
    """Return (num_frames, sample_rate, channels) via torchaudio.info."""
    try:
        info = torchaudio.info(str(path))
    except Exception as e:
        log.debug(f"  torchaudio.info failed: {path.name}: {e}")
        return None
    return info.num_frames, info.sample_rate, info.num_channels


def _audit_one(rec: dict) -> dict:
    """Return a diff record comparing JSONL num_samples vs torchaudio.info."""
    # IMPORTANT: store the original string from the JSONL as the audit key,
    # NOT str(Path(...)).  On Windows the Path round-trip flips '/' to '\\'
    # which breaks the dict lookup in _rewrite_jsonl that uses the raw
    # JSONL value as the key.
    raw_path = rec["audio_path"]
    path = Path(raw_path)
    out = {"path": raw_path, "jsonl_ns": rec.get("num_samples"), "found": None}
    if not path.exists():
        out["status"] = "missing"
        return out
    probed = _probe(path)
    if probed is None:
        out["status"] = "unreadable"
        return out
    n_frames, sr, channels = probed
    out["found"] = n_frames
    out["found_sr"] = sr
    out["found_channels"] = channels
    jsonl_ns = rec.get("num_samples", 0)
    if jsonl_ns == 0:
        out["status"] = "no-jsonl-ns"
        return out
    diff = jsonl_ns - n_frames
    out["diff_samples"] = diff
    out["diff_seconds"] = diff / sr
    # Categorise: tiny mismatches are normal MP3-decoder differences;
    # large ones might indicate a different file entirely.
    if abs(diff) <= sr * 0.05:                     # ≤50 ms drift — harmless
        out["status"] = "ok"
    elif abs(diff) <= sr * 1.0:                    # ≤1 s drift — typical MP3 quirk
        out["status"] = "minor-drift"
    else:
        out["status"] = "large-drift"
    return out


def _audit_jsonl(jsonl: Path, workers: int) -> list[dict]:
    """Read the JSONL and audit every entry."""
    with open(jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info(f"  {jsonl.name}: {len(records):,} entries")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_audit_one, r) for r in records]
        for i, fut in enumerate(as_completed(futs), start=1):
            results.append(fut.result())
            if i % 500 == 0:
                log.info(f"    audited {i}/{len(records)}")
    return results


def _summarise(results: list[dict]) -> dict:
    """Aggregate status counts and worst-case offenders."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    drifts = [r for r in results if "diff_seconds" in r and r["status"] != "ok"]
    drifts.sort(key=lambda r: abs(r["diff_seconds"]), reverse=True)
    return {"counts": counts, "worst": drifts[:10]}


def _rewrite_jsonl(jsonl: Path, results: list[dict], keep_drifted: bool):
    """Overwrite the JSONL with torchaudio-derived num_samples values.

    Entries are filtered out when they're unsalvageable:
      - status == 'missing'        → file not on disk
      - status == 'unreadable'     → torchaudio.info() failed
      - status == 'large-drift'    → file content doesn't match JSONL
                                     (kept only when --keep-drifted is set)

    Entries with status 'ok' / 'minor-drift' / 'no-jsonl-ns' are kept and
    their num_samples / sample_rate / channels / duration are refreshed
    from torchaudio.info() so the index_map agrees with the decoder.
    """
    by_path = {r["path"]: r for r in results}
    out_lines: list[str] = []
    n_kept = n_dropped = 0
    drop_counts: dict[str, int] = {}

    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audit = by_path.get(rec["audio_path"])
            if audit is None:
                # Shouldn't happen, but if it does, keep the entry unchanged
                out_lines.append(json.dumps(rec, ensure_ascii=False))
                n_kept += 1
                continue

            status = audit["status"]
            unsalvageable = status in ("missing", "unreadable") or (
                status == "large-drift" and not keep_drifted
            )
            if unsalvageable:
                drop_counts[status] = drop_counts.get(status, 0) + 1
                n_dropped += 1
                continue

            if audit.get("found") is not None:
                rec["num_samples"] = audit["found"]
                rec["sample_rate"] = audit["found_sr"]
                rec["channels"]    = audit["found_channels"]
                rec["duration"]    = round(audit["found"] / audit["found_sr"], 3)
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            n_kept += 1

    backup = jsonl.with_suffix(jsonl.suffix + ".bak")
    if not backup.exists():
        jsonl.rename(backup)
        log.info(f"    backed up original to {backup.name}")
    with open(jsonl, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")
    msg = f"    rewrote {jsonl.name}: kept {n_kept:,}, dropped {n_dropped:,}"
    if drop_counts:
        msg += "  (" + ", ".join(f"{k}={v}" for k, v in drop_counts.items()) + ")"
    log.info(msg)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description="Audit HookTheory JSONLs vs torchaudio frame counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--jsonl",     type=Path, help="audit a single JSONL")
    g.add_argument("--jsonl-dir", type=Path, help="audit every *.jsonl in this dir")
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--rewrite",  action="store_true",
                    help="Overwrite JSONLs with torchaudio frame counts "
                         "(creates a .bak first).  Drops 'missing', "
                         "'unreadable', and 'large-drift' entries — the "
                         "audio either isn't on disk or doesn't match "
                         "the JSONL's stated content.")
    ap.add_argument("--keep-drifted", action="store_true",
                    help="With --rewrite, keep 'large-drift' entries "
                         "instead of dropping them.  Their content "
                         "doesn't match the JSONL but they may still be "
                         "usable if the label still applies to the "
                         "actual audio.")
    args = ap.parse_args()

    if args.jsonl_dir:
        jsonls = sorted(args.jsonl_dir.glob("HookTheory*.jsonl"))
    else:
        jsonls = [args.jsonl]
    if not jsonls:
        log.error("No matching JSONL files found")
        sys.exit(1)

    for jp in jsonls:
        log.info(f"── auditing {jp} ──")
        results = _audit_jsonl(jp, args.workers)
        summary = _summarise(results)
        print()
        print(f"  {jp.name} — {len(results):,} entries")
        for status, n in sorted(summary["counts"].items()):
            print(f"    {status:>15s}: {n:>6,}")
        if summary["worst"]:
            print(f"    Worst offenders (largest |jsonl - actual| in seconds):")
            for w in summary["worst"][:5]:
                print(f"      {Path(w['path']).name:>40s}  "
                      f"jsonl={w['jsonl_ns']:>10,}  actual={w['found']:>10,}  "
                      f"diff={w['diff_seconds']:+.3f}s")
        print()

        if args.rewrite:
            _rewrite_jsonl(jp, results, keep_drifted=args.keep_drifted)
            print()


if __name__ == "__main__":
    main()
