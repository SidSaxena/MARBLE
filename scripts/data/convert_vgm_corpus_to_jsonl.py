#!/usr/bin/env python3
"""scripts/data/convert_vgm_corpus_to_jsonl.py
────────────────────────────────────────────────
Convert a VGM rendered-corpus ``manifest.json`` into per-split JSONL files
suitable for the ``VGMLoopStructure`` MARBLE task.

Background
----------
The corpus renderer produces a ``manifest.json`` (a JSON list) whose rows
carry at minimum:

    {
        "id":          "<unique track id>",
        "split":       "train" | "val" | "test",
        "loop_type":   "through_composed" | "loop_from_start" | "intro_loop",
        "audio_path":  "<relative path under audio-root, e.g. audio/track_001.wav>",
        "total_sec":   <float: loop-body duration in seconds>
    }

The rendered audio file is ``intro + 2×loop``, so its actual length is
**longer** than ``total_sec``.  ``num_samples`` is therefore probed directly
from each WAV via ``soundfile.info`` — never computed from ``total_sec``.

Output schema (per row)
-----------------------
    {
        "audio_path":  "<absolute path to WAV>",
        "sample_rate": 24000,
        "num_samples": <int, from file>,
        "channels":    1,
        "bit_depth":   16,
        "label":       "<loop_type string>",
        "duration":    <float, num_samples / sample_rate>
    }

One file per split is written:
    <out-dir>/<name>.train.wav.jsonl
    <out-dir>/<name>.val.wav.jsonl
    <out-dir>/<name>.test.wav.jsonl

Usage
-----
    uv run python scripts/data/convert_vgm_corpus_to_jsonl.py \\
        --manifest /data/vgm/manifest.json \\
        --audio-root /data/vgm \\
        --out-dir data/VGM \\
        --name VGMLoopStructure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {"through_composed", "loop_from_start", "intro_loop"}
FIXED_SAMPLE_RATE = 24000


def _probe_wav(wav_path: Path) -> tuple[int, int, int]:
    """Return (num_samples, sample_rate, channels) by reading WAV metadata.

    Uses soundfile.info — fast header read, no audio decode.
    """
    import soundfile as sf

    info = sf.info(str(wav_path))
    return info.frames, info.samplerate, info.channels


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--manifest",
        required=True,
        metavar="PATH",
        help="Path to corpus manifest.json (list of rows).",
    )
    ap.add_argument(
        "--audio-root",
        required=True,
        metavar="DIR",
        help="Directory that contains the audio/ tree referenced in manifest rows.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="Directory where the output JSONL files will be written.",
    )
    ap.add_argument(
        "--name",
        default="VGMLoopStructure",
        metavar="NAME",
        help="Stem used in output filenames: <name>.{split}.wav.jsonl (default: VGMLoopStructure).",
    )
    ap.add_argument(
        "--mode",
        choices=["clip", "frame"],
        default="clip",
        metavar="MODE",
        help=(
            "Output label schema: 'clip' emits label=loop_type string (default); "
            "'frame' emits label=dict with intro_end_sec/loop_seam_sec/loop_type/total_sec."
        ),
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    audio_root = Path(args.audio_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)

    # Accumulate rows per split
    split_rows: dict[str, list[dict]] = {s: [] for s in VALID_SPLITS}
    n_missing = 0
    n_written = 0

    for row in rows:
        split = row.get("split", "").lower()
        loop_type = row.get("loop_type", "")
        audio_rel = row.get("audio_path", "")

        # Validate split
        if split not in VALID_SPLITS:
            print(
                f"WARNING: row id={row.get('id')!r} has unknown split {split!r} — skipping",
                file=sys.stderr,
            )
            continue

        # Validate label
        if loop_type not in VALID_LABELS:
            print(
                f"WARNING: row id={row.get('id')!r} has unknown loop_type {loop_type!r} — skipping",
                file=sys.stderr,
            )
            continue

        wav_path = audio_root / audio_rel
        if not wav_path.exists():
            print(
                f"WARNING: WAV missing for id={row.get('id')!r}: {wav_path} — skipping",
                file=sys.stderr,
            )
            n_missing += 1
            continue

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
        split_rows[split].append(out_row)
        n_written += 1

    # Write one JSONL per split
    for split, out_rows in split_rows.items():
        out_file = out_dir / f"{args.name}.{split}.wav.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for out_row in out_rows:
                f.write(json.dumps(out_row) + "\n")
        print(f"Wrote {len(out_rows):4d} rows → {out_file}")

    print(
        f"\nDone: {n_written} rows written, {n_missing} skipped (missing/unreadable).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
