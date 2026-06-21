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

Within-group byte-dedup (default ON, ``--no-dedup-within-group`` to disable):
JKUPDD ships every occurrence MIDI **time-zeroed to t=0**, so a literal
pitch+rhythm phrase-repeat collapses to a *byte-identical* file. Inside a
``(piece, annotator, pattern)`` group those identical-content occurrences are a
**trivial-duplicate artifact**: CLaMP3 maps identical bytes to identical
embeddings (cosine = 1.000), guaranteeing rank-1 self-relevant hits that inflate
MAP independent of encoder quality (of the raw 165 occurrences only 88 are
byte-distinct; 71% of queries have a byte-identical same-group twin; 12 of 32
groups are entirely byte-identical). We therefore keep **one representative per
distinct byte-content per group**. We dedup **BYTE-IDENTICAL ONLY** — transposed
or rhythmically-varied repeats are *legitimate* motif variation the encoder must
be tested on and are kept. A retrieval group needs ≥2 distinct contents to be a
valid query pool, so groups that **collapse to a single distinct content are
dropped** (they can only produce trivial self-relevant queries).

Caveat (keeping all annotators): a few occurrence MIDIs are byte-identical
across *different* annotator passes (e.g. Mozart K282 ``barlowAndMorgenstern|A``
≡ ``barlowAndMorgensternRevised|C``). Those are scored non-relevant (different
group) despite identical content → a small, *conservative* MAP deflation (it
under-states, never inflates). Pass ``--annotators <one-per-piece>`` for a
single-source pool if you want to avoid it. (Dedup is *within*-group only and
does not touch this cross-annotator case.)

This is a SMALL benchmark (165 raw / 78 byte-dedup'd occurrence windows across 5
pieces) — a cross-composer (Bach/Beethoven/Chopin/Gibbons/Mozart) Layer-1 sanity
check complementing the Beethoven-only BPS-Motif retrieval, not a large eval.

Usage:
  uv run python scripts/data/build_jkupdd_retrieval.py \
      --jkupdd-root /path/to/JKUPDD-noAudio-Aug2013 \
      [--texture polyphonic|monophonic] [--annotators a,b,..] \
      [--no-dedup-within-group]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "data" / "JKUPDD"
MIDI_OUT = OUT_DIR / "midi_windows"
JSONL_OUT = OUT_DIR / "JKUPDDRetrieval.test.jsonl"


def build(
    jkupdd_root: Path,
    texture: str,
    annot_filter: set[str] | None,
    dedup_within_group: bool = True,
) -> dict:
    gt = jkupdd_root / "groundTruth"
    if not gt.is_dir():
        raise FileNotFoundError(f"groundTruth/ not found under {jkupdd_root}")
    if MIDI_OUT.is_dir():
        # Wipe stale windows so a dedup re-build never leaves orphan MIDIs that
        # disagree with the regenerated JSONL.
        shutil.rmtree(MIDI_OUT)
    MIDI_OUT.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: enumerate every annotated occurrence into its group. ---
    # ``occ_by_group[group] = [(stem, src_path, sha256), ...]`` in stable order.
    occ_by_group: dict[str, list[tuple[str, Path, str]]] = {}
    group_meta: dict[str, tuple[str, str, str]] = {}  # group -> (piece, annot, letter)
    n_pieces = n_patterns = 0
    n_occ_seen = 0
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
                group = f"{piece}|{annot}|{letter}"
                group_meta[group] = (piece, annot, letter)
                bucket = occ_by_group.setdefault(group, [])
                for occ in occ_midis:
                    n_occ_seen += 1
                    stem = f"{piece}__{annot}__{letter}__{occ.stem}"
                    sha = hashlib.sha256(occ.read_bytes()).hexdigest()
                    bucket.append((stem, occ, sha))

    groups_before = len(occ_by_group)
    occ_before = n_occ_seen

    # --- Pass 2: within-group BYTE-IDENTICAL dedup + drop singleton groups. ---
    # Keep one representative per distinct byte-content per group (first by
    # stable stem order). A group that collapses to a single distinct content
    # cannot form a valid query pool (only trivial self-relevant queries), so
    # it is dropped entirely. We dedup byte-identical ONLY — transposed /
    # rhythmically-varied repeats hash differently and are kept on purpose.
    records: list[dict] = []
    groups_dropped: list[str] = []
    for group, occs in occ_by_group.items():
        piece, annot, letter = group_meta[group]
        if dedup_within_group:
            seen: set[str] = set()
            kept: list[tuple[str, Path, str]] = []
            for stem, src, sha in occs:
                if sha in seen:
                    continue
                seen.add(sha)
                kept.append((stem, src, sha))
        else:
            kept = list(occs)
        if dedup_within_group and len(kept) < 2:
            # Collapsed to a single distinct content -> no valid relevant pair.
            groups_dropped.append(group)
            continue
        for stem, src, _sha in kept:
            dest = MIDI_OUT / f"{stem}.mid"
            shutil.copyfile(src, dest)
            records.append(
                {
                    "midi_path": str(dest.relative_to(REPO)),
                    "piece_id": piece,
                    "annotator": annot,
                    "pattern": letter,
                    "group": group,
                    "occurrence_id": stem,
                    "split": "test",
                }
            )

    with open(JSONL_OUT, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    # how many surviving groups are singletons (none expected when dedup is on,
    # since collapsed groups are dropped above; informative when dedup is off).
    from collections import Counter

    gc = Counter(r["group"] for r in records)
    singletons = sum(1 for _g, c in gc.items() if c < 2)
    return {
        "pieces": n_pieces,
        "patterns": n_patterns,
        "dedup_within_group": dedup_within_group,
        "occurrences_before": occ_before,
        "occurrences": len(records),
        "groups_before": groups_before,
        "groups": len(gc),
        "groups_dropped": len(groups_dropped),
        "dropped_groups": sorted(groups_dropped),
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
    ap.add_argument(
        "--dedup-within-group",
        dest="dedup_within_group",
        action="store_true",
        default=True,
        help="keep one representative per distinct byte-content per group "
        "(default ON; removes the time-zeroing trivial-duplicate artifact)",
    )
    ap.add_argument(
        "--no-dedup-within-group",
        dest="dedup_within_group",
        action="store_false",
        help="disable within-group byte-dedup (reproduces the inflated raw build)",
    )
    args = ap.parse_args()
    annot_filter = {a.strip() for a in args.annotators.split(",") if a.strip()} or None
    stats = build(args.jkupdd_root, args.texture, annot_filter, args.dedup_within_group)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
