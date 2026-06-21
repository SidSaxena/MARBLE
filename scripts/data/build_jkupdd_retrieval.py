"""Build a JKUPDD within-piece motif RETRIEVAL dataset for MARBLE.

JKUPDD (JKU Patterns Development Database, Collins 2013 — the MIREX "Discovery
of Repeated Themes & Sections" dev set) annotates, for each of 5 pieces, a set
of repeated patterns and every occurrence of each. It ships a ready-made MIDI
per occurrence, so this builder just enumerates them and writes a single
zero-shot retrieval JSONL in the same shape as the BPS-Motif retrieval task.

Relevance for retrieval = same ``(piece, annotator, pattern)`` group: two
occurrences are relevant iff they are occurrences of the *same* annotated
pattern. We keep the annotator in the key because JKUPDD has several
overlapping annotation sources per piece (barlowAndMorgenstern, schoenberg,
tomCollins, …); mixing them into one class would conflate distinct analyses.

This is a SMALL benchmark (~165 occurrence windows across 5 pieces) — a
cross-composer (Bach/Beethoven/Chopin/Gibbons/Mozart) Layer-1 sanity check
complementing the Beethoven-only BPS-Motif retrieval, not a large eval.

Usage:
  uv run python scripts/data/build_jkupdd_retrieval.py \
      --jkupdd-root /path/to/JKUPDD-noAudio-Aug2013 \
      [--texture polyphonic|monophonic] [--annotators a,b,..]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "JKUPDD"
MIDI_OUT = OUT_DIR / "midi_windows"
JSONL_OUT = OUT_DIR / "JKUPDDRetrieval.test.jsonl"


def build(jkupdd_root: Path, texture: str, annot_filter: set[str] | None) -> dict:
    gt = jkupdd_root / "groundTruth"
    if not gt.is_dir():
        raise FileNotFoundError(f"groundTruth/ not found under {jkupdd_root}")
    MIDI_OUT.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    n_pieces = n_patterns = 0
    for piece_dir in sorted(p for p in gt.iterdir() if p.is_dir()):
        piece = piece_dir.name
        rp = piece_dir / texture / "repeatedPatterns"
        if not rp.is_dir():
            continue
        n_pieces += 1
        for annot_dir in sorted(a for a in rp.iterdir() if a.is_dir()):
            annot = annot_dir.name
            if annot_filter and annot not in annot_filter:
                continue
            for pat_dir in sorted(p for p in annot_dir.iterdir() if p.is_dir()):
                letter = pat_dir.name
                occ_midi_dir = pat_dir / "occurrences" / "midi"
                if not occ_midi_dir.is_dir():
                    continue
                occ_midis = sorted(occ_midi_dir.glob("*.mid*"))
                if not occ_midis:
                    continue
                n_patterns += 1
                for occ in occ_midis:
                    stem = f"{piece}__{annot}__{letter}__{occ.stem}"
                    dest = MIDI_OUT / f"{stem}.mid"
                    shutil.copyfile(occ, dest)
                    records.append(
                        {
                            "midi_path": str(dest.relative_to(REPO)),
                            "piece_id": piece,
                            "annotator": annot,
                            "pattern": letter,
                            "group": f"{piece}|{annot}|{letter}",
                            "occurrence_id": stem,
                            "split": "test",
                        }
                    )

    with open(JSONL_OUT, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # how many singleton groups (no relevant partner — excluded from MAP)
    from collections import Counter

    gc = Counter(r["group"] for r in records)
    singletons = sum(1 for g, c in gc.items() if c < 2)
    return {
        "pieces": n_pieces,
        "patterns": n_patterns,
        "occurrences": len(records),
        "groups": len(gc),
        "singleton_groups": singletons,
        "jsonl": str(JSONL_OUT),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--jkupdd-root",
        type=Path,
        required=True,
        help="path to JKUPDD-noAudio-Aug2013/ (contains groundTruth/)",
    )
    ap.add_argument("--texture", choices=["polyphonic", "monophonic"], default="polyphonic")
    ap.add_argument(
        "--annotators", default="", help="comma-separated annotator allowlist (default: all)"
    )
    args = ap.parse_args()
    annot_filter = {a.strip() for a in args.annotators.split(",") if a.strip()} or None
    stats = build(args.jkupdd_root, args.texture, annot_filter)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
