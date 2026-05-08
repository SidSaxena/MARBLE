#!/usr/bin/env python3
"""
scripts/download_covers80.py
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
    http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tar.bz2

If the download fails (the Columbia server can be slow), place the archive
manually at  data/Covers80/covers80.tar.bz2  and re-run.

Usage
-----
python scripts/download_covers80.py
python scripts/download_covers80.py --data-dir /mnt/data
python scripts/download_covers80.py --no-download   # if archive already exists
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path


COVERS80_URL = (
    "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tar.bz2"
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r  {pct:3d}%  ({downloaded/1e6:.0f} / {total_size/1e6:.0f} MB)")
    else:
        sys.stdout.write(f"\r  {downloaded/1e6:.0f} MB downloaded …")
    sys.stdout.flush()


def _audio_files(d: Path) -> list[Path]:
    return sorted(
        list(d.rglob("*.mp3")) +
        list(d.rglob("*.wav")) +
        list(d.rglob("*.flac"))
    )


def _audio_info(path: Path) -> tuple[int, int, int]:
    """Return (sample_rate, num_samples, channels) via torchaudio."""
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        return info.sample_rate, info.num_frames, info.num_channels
    except Exception as e:
        print(f"\n    ⚠ torchaudio.info failed for {path.name}: {e}")
        return 0, 0, 1   # safe fallback; duration = 0 flags broken files


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Covers80 and generate MARBLE JSONL.",
    )
    parser.add_argument(
        "--data-dir", default="data/Covers80",
        help="Root directory for Covers80 data (default: data/Covers80).",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip downloading — regenerate JSONL from already-extracted data.",
    )
    args = parser.parse_args()

    dest = Path(args.data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "covers80.tar.bz2"

    # ── 1. Download ───────────────────────────────────────────────────────────
    if not args.no_download:
        if archive.exists():
            print(f"Archive already present: {archive}")
        else:
            print(f"Downloading Covers80 (~60 MB) …")
            print(f"  URL: {COVERS80_URL}")
            try:
                urllib.request.urlretrieve(COVERS80_URL, archive, reporthook=_progress_hook)
                print()
            except Exception as e:
                print(f"\n\nDownload failed: {e}")
                print("Manually place covers80.tar.bz2 in the data directory and re-run:")
                print(f"  python scripts/download_covers80.py --data-dir {dest} --no-download")
                sys.exit(1)

    # ── 2. Extract ────────────────────────────────────────────────────────────
    # The archive usually expands to covers32k/ inside the current dir.
    covers_root = dest / "covers32k"
    if not covers_root.exists():
        if not archive.exists():
            print(f"ERROR: archive not found at {archive}")
            sys.exit(1)
        print("Extracting archive …")
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(dest)
        print("Extraction complete.")

    # ── 3. Locate list1 / list2 ───────────────────────────────────────────────
    # Try common directory names used in various versions of the archive.
    list1 = None
    list2 = None
    for candidate in (covers_root, dest):
        for name1, name2 in [("list1", "list2"), ("covers1", "covers2")]:
            d1 = next(candidate.glob(f"**/{name1}"), None)
            d2 = next(candidate.glob(f"**/{name2}"), None)
            if d1 and d2:
                list1, list2 = d1, d2
                break
        if list1:
            break

    if not list1 or not list2:
        print(f"ERROR: could not find list1/list2 directories under {dest}")
        print(f"Found: {list(dest.rglob('*'))[:20]}")
        sys.exit(1)

    print(f"list1 → {list1}")
    print(f"list2 → {list2}")

    # ── 4. Match works across both lists ─────────────────────────────────────
    # Each subdirectory of list1/list2 corresponds to one work.
    # They should have the same names (alphabetically sorted).

    def works(d: Path) -> list[Path]:
        """Return sorted list of subdirectories (one per work)."""
        return sorted([p for p in d.iterdir() if p.is_dir()])

    works1 = works(list1)
    works2 = works(list2)

    if len(works1) != len(works2):
        print(
            f"WARNING: list1 has {len(works1)} works, list2 has {len(works2)}. "
            f"Matching by sorted order."
        )

    n_works = min(len(works1), len(works2))

    # ── 5. Build JSONL ────────────────────────────────────────────────────────
    records: list[dict] = []
    skipped: list[str]  = []

    for work_id, (w1_dir, w2_dir) in enumerate(
        zip(works1[:n_works], works2[:n_works])
    ):
        for version, work_dir in enumerate([w1_dir, w2_dir]):
            files = _audio_files(work_dir)
            if not files:
                skipped.append(f"work {work_id} v{version}: no audio in {work_dir}")
                continue

            # Take the first (and usually only) audio file in the work dir.
            audio_path = files[0]
            sr, n_samples, channels = _audio_info(audio_path)
            duration = n_samples / sr if sr > 0 else 0.0

            records.append({
                "audio_path":  str(audio_path),
                "work_id":     work_id,
                "version":     version,
                "work_name":   w1_dir.name,          # human-readable label
                "sample_rate": sr,
                "num_samples": n_samples,
                "channels":    channels,
                "duration":    round(duration, 3),
            })

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

    print(f"\n✓ {len(records)} tracks ({n_works} works × 2 versions)")
    print(f"  → {out_path}")
    print(f"  → {labels_path}")

    work_ids_seen = {r["work_id"] for r in records}
    print(f"\nWork IDs in JSONL: {min(work_ids_seen)}–{max(work_ids_seen)} "
          f"({len(work_ids_seen)} unique)")

    print(f"\nNext steps — run the Covers80 layer sweep:")
    print(f"  python scripts/run_sweep_local.py \\")
    print(f"      --base-config configs/probe.OMARRQ-multifeature25hz.Covers80.yaml \\")
    print(f"      --num-layers 24 --model-tag OMARRQ-multifeature25hz --task-tag Covers80")


if __name__ == "__main__":
    main()
