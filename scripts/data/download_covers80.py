#!/usr/bin/env python3
"""
scripts/data/download_covers80.py
────────────────────────────
Download the Covers80 dataset and generate a single JSONL file suitable for
MARBLE's zero-shot cover-song retrieval evaluation.

Covers80
  80 musical works, each with exactly 2 recordings:
    • list1/  — "original" version (one subdirectory per work)
    • list2/  — "cover"    version (same subdirectory names, matched by sort order)

We assign a work_id 0–79 to each work (sorted alphabetically) and write one
JSONL line per audio file:
  {
    "audio_path": "data/Covers80/covers32k/list1/<work>/song.mp3",
    "work_id":    42,
    "version":    0,      # 0 = list1, 1 = list2
    "sample_rate": 32000,
    "num_samples": 9600000,
    "channels":   1,
    "duration":   300.0
  }

All 160 tracks go into one file:  data/Covers80/Covers80.test.jsonl
(There is no train/val split — retrieval evaluation is unsupervised.)

Source archive
  Columbia LABROSA:
    http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz

If the download fails (the Columbia server can be slow), place the archive
manually at  data/Covers80/covers80.tgz  and re-run.

Usage
-----
python scripts/data/download_covers80.py
python scripts/data/download_covers80.py --data-dir /mnt/data
python scripts/data/download_covers80.py --no-download   # if archive already exists
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

COVERS80_URL = "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz"


# ─── helpers ─────────────────────────────────────────────────────────────────


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r  {pct:3d}%  ({downloaded / 1e6:.0f} / {total_size / 1e6:.0f} MB)")
    else:
        sys.stdout.write(f"\r  {downloaded / 1e6:.0f} MB downloaded …")
    sys.stdout.flush()


def _audio_files(d: Path) -> list[Path]:
    return sorted(list(d.rglob("*.mp3")) + list(d.rglob("*.wav")) + list(d.rglob("*.flac")))


def _audio_info(path: Path) -> tuple[int, int, int]:
    """Return (sample_rate, num_samples, channels) via torchaudio."""
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        return info.sample_rate, info.num_frames, info.num_channels
    except Exception as e:
        print(f"\n    ⚠ torchaudio.info failed for {path.name}: {e}")
        return 0, 0, 1  # safe fallback; duration = 0 flags broken files


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download Covers80 and generate MARBLE JSONL.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/Covers80",
        help="Root directory for Covers80 data (default: data/Covers80).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading — regenerate JSONL from already-extracted data.",
    )
    args = parser.parse_args()

    dest = Path(args.data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "covers80.tgz"

    # ── 1. Download ───────────────────────────────────────────────────────────
    if not args.no_download:
        if archive.exists():
            print(f"Archive already present: {archive}")
        else:
            print("Downloading Covers80 (~60 MB) …")
            print(f"  URL: {COVERS80_URL}")
            try:
                urllib.request.urlretrieve(COVERS80_URL, archive, reporthook=_progress_hook)
                print()
            except Exception as e:
                print(f"\n\nDownload failed: {e}")
                print("Manually place covers80.tgz in the data directory and re-run:")
                print(f"  python scripts/data/download_covers80.py --data-dir {dest} --no-download")
                sys.exit(1)

    # ── 2. Extract ────────────────────────────────────────────────────────────
    # The archive usually expands to coversongs/covers32k/ inside the current dir.
    covers_root = next(dest.rglob("covers32k"), None)
    if not covers_root:
        if not archive.exists():
            print(f"ERROR: archive not found at {archive}")
            sys.exit(1)
        print("Extracting archive …")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
        print("Extraction complete.")
        covers_root = next(dest.rglob("covers32k"), None)
        if not covers_root:
            print("ERROR: could not find covers32k directory after extraction")
            sys.exit(1)

    # ── 3. Locate works ──────────────────────────────────────────────────────
    # All work subdirectories are directly under covers32k/
    works = sorted([p for p in covers_root.iterdir() if p.is_dir()])
    n_works = len(works)
    print(f"Found {n_works} works under {covers_root}")

    # ── 4. Build JSONL ────────────────────────────────────────────────────────
    records: list[dict] = []
    skipped: list[str] = []

    for work_id, work_dir in enumerate(works):
        files = sorted(_audio_files(work_dir))
        if len(files) < 2:
            skipped.append(
                f"work {work_id}: expected at least 2 audio files, found {len(files)} in {work_dir}"
            )
            continue

        # Take the first 2 files (sorted alphabetically)
        if len(files) > 2:
            print(f"WARNING: work {work_id} has {len(files)} files, using first 2: {files[:2]}")

        for version, audio_path in enumerate(files[:2]):
            sr, n_samples, channels = _audio_info(audio_path)
            duration = n_samples / sr if sr > 0 else 0.0

            records.append(
                {
                    "audio_path": str(audio_path),
                    "work_id": work_id,
                    "version": version,
                    "work_name": work_dir.name,  # human-readable label
                    "sample_rate": sr,
                    "num_samples": n_samples,
                    "channels": channels,
                    "duration": round(duration, 3),
                }
            )

    if skipped:
        print(f"\nSkipped {len(skipped)} entries:")
        for s in skipped:
            print(f"  {s}")

    # ── 6. Write JSONL ────────────────────────────────────────────────────────
    out_path = dest / "Covers80.test.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Also write a labels.json for reference
    labels_path = dest / "labels.json"
    unique_works = sorted({r["work_name"] for r in records})
    labels_path.write_text(json.dumps(unique_works, indent=2))

    print(
        f"\n✓ {len(records)} tracks ({len(set(r['work_id'] for r in records))} works × 2 versions)"
    )
    print(f"  → {out_path}")
    print(f"  → {labels_path}")

    work_ids_seen = {r["work_id"] for r in records}
    print(
        f"\nWork IDs in JSONL: {min(work_ids_seen)}–{max(work_ids_seen)} "
        f"({len(work_ids_seen)} unique)"
    )

    print("\nNext steps — run the Covers80 layer sweep:")
    print("  python scripts/sweeps/run_sweep_local.py \\")
    print("      --base-config configs/probe.OMARRQ-multifeature25hz.Covers80.yaml \\")
    print("      --num-layers 24 --model-tag OMARRQ-multifeature25hz --task-tag Covers80")


if __name__ == "__main__":
    main()
