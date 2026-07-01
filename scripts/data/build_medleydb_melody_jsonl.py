#!/usr/bin/env python
"""Build MedleyDBMelody 5-fold CV JSONLs from a local MedleyDB copy.

Output: ``data/MedleyDB/MedleyDBMelody.fold{F}.{split}.jsonl`` for F=0..4,
split ∈ {train,val,test}. Each record:
    {"audio_path", "melody_csv", "sample_rate", "num_samples", "track", "artist"}

Audio (``<Track>_MIX.wav``) comes from the Zenodo audio download; melody
annotations (``<Track>_MELODY2.csv``) from the marl/medleydb GitHub repo (or the
bundled sample). The two ship in different sub-layouts, so we discover both by
recursive glob and join on track name — only tracks present in *both* are kept.

Splitting (default): artist-conditional 5-fold CV. Artist = track-name prefix
(``split('_')[0]`` — MedleyDB ids are ``<Artist>_<Title>``). ``MusicDelta_*``
genre demos are one leak-safe group (the medleydb package is not used — it doesn't
import on modern PyYAML and its metadata lumps MusicDelta inconsistently anyway).
Pass ``--split-json`` to pin an exact single split, or ``--smoke`` to put every
discovered track into fold0's train/val/test (quick end-to-end pipeline smoke on
the 2-track sample).

Example (full data on the PC):
    uv run python scripts/data/build_medleydb_melody_jsonl.py \
        --audio-root /mnt/d/datasets/medleydb/Audio \
        --annotation-root /mnt/d/datasets/medleydb/Annotations \
        --out-dir data/MedleyDB
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from marble.tasks.MedleyDBMelody.split import fold_split  # noqa: E402

MIX_SUFFIX = "_MIX.wav"
MELODY_SUFFIX = "_MELODY2.csv"  # MARBLE uses MELODY2 (predominant f0)


def _warn(msg: str) -> None:
    print(f"  [warn] {msg}", file=sys.stderr)


def discover(audio_root: str, annotation_root: str) -> list[dict]:
    def _index(root: str, suffix: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in glob(os.path.join(root, "**", f"*{suffix}"), recursive=True):
            key = os.path.basename(p)[: -len(suffix)]
            if key in out and out[key] != p:
                _warn(f"duplicate '{key}{suffix}' — keeping {out[key]}, ignoring {p}")
                continue
            out[key] = p
        return out

    mixes = _index(audio_root, MIX_SUFFIX)
    melodies = _index(annotation_root, MELODY_SUFFIX)
    tracks = sorted(set(mixes) & set(melodies))
    if set(melodies) - set(mixes):
        _warn(f"{len(set(melodies) - set(mixes))} tracks have MELODY2 but no MIX audio (skipped)")
    if set(mixes) - set(melodies):
        _warn(f"{len(set(mixes) - set(melodies))} tracks have MIX audio but no MELODY2 (skipped)")
    recs = []
    for t in tracks:
        try:
            info = torchaudio.info(mixes[t])
        except Exception as e:  # corrupt/truncated wav → skip, don't kill the build
            _warn(f"{t}: torchaudio.info failed ({e}); skipping")
            continue
        if info.num_frames <= 0 or info.sample_rate <= 0:
            _warn(f"{t}: zero-length/invalid audio (frames={info.num_frames}); skipping")
            continue
        recs.append(
            {
                "audio_path": mixes[t],
                "melody_csv": melodies[t],
                "sample_rate": int(info.sample_rate),
                "num_samples": int(info.num_frames),
                "track": t,
                "artist": t.split("_")[0],  # display only; real grouping below
            }
        )
    return recs


def _write_fold(
    out_dir: str, fold: int, by_track: dict[str, dict], split_names: dict[str, list[str]]
):
    for split, names in split_names.items():
        out = os.path.join(out_dir, f"MedleyDBMelody.fold{fold}.{split}.jsonl")
        with open(out, "w") as fh:
            for t in names:
                fh.write(json.dumps(by_track[t]) + "\n")
        print(f"  wrote {os.path.basename(out)}: {len(names)} tracks")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--audio-root", required=True, help="dir containing <Track>_MIX.wav (recursive)"
    )
    ap.add_argument(
        "--annotation-root", required=True, help="dir containing <Track>_MELODY2.csv (recursive)"
    )
    ap.add_argument("--out-dir", default="data/MedleyDB")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split-json",
        default=None,
        help='JSON {"train":[track,...],"val":[...],"test":[...]} to pin ONE exact split (written as fold0)',
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="put every discovered track in fold0 train+val+test (pipeline/runtime smoke)",
    )
    args = ap.parse_args()

    print(f"Discovering tracks under\n  audio: {args.audio_root}\n  annot: {args.annotation_root}")
    recs = discover(args.audio_root, args.annotation_root)
    print(f"  found {len(recs)} tracks with both MIX audio and MELODY2 annotation")
    if not recs:
        sys.exit("No tracks found — check --audio-root / --annotation-root.")
    by_track = {r["track"]: r for r in recs}
    os.makedirs(args.out_dir, exist_ok=True)

    if args.smoke:
        names = sorted(by_track)
        _write_fold(args.out_dir, 0, by_track, {"train": names, "val": names, "test": names})
        print("  [smoke] every track placed in fold0 train+val+test")
        return

    if args.split_json:
        with open(args.split_json) as fh:
            spec = json.load(fh)
        names = {k: list(spec.get(k, [])) for k in ("train", "val", "test")}
        # disjointness + missing checks (silent leakage/drops defeat the purpose)
        sets = {k: set(v) for k, v in names.items()}
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                sys.exit(
                    f"--split-json: {len(overlap)} track(s) appear in both {a} and {b}: {sorted(overlap)[:5]}"
                )
        missing = (sets["train"] | sets["val"] | sets["test"]) - set(by_track)
        if missing:
            _warn(
                f"--split-json lists {len(missing)} track(s) not discovered (dropped): {sorted(missing)[:5]}"
            )
            names = {k: [t for t in v if t in by_track] for k, v in names.items()}
        for k in ("train", "val", "test"):
            if not names[k]:
                sys.exit(
                    f"--split-json: '{k}' split is empty (omitted, or all its tracks "
                    f"undiscovered) — refusing to write a 0-line JSONL."
                )
        _write_fold(args.out_dir, 0, by_track, names)
        print(f"  [split-json] {args.split_json} (written as fold0)")
        return

    # Default: artist-conditional 5-fold CV. Artist = track-name prefix (MedleyDB
    # ids are <Artist>_<Title>). MusicDelta_* genre demos share the "MusicDelta"
    # source → treated as one leak-safe group (concentrates them in one fold,
    # mitigated by 5-fold averaging). The medleydb package's artist index is not
    # used (doesn't import on modern PyYAML; its metadata lumps MusicDelta anyway).
    def artist_of(t: str) -> str:
        return t.split("_")[0]

    folds = fold_split(
        sorted(by_track),
        artist_of=artist_of,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    for f, split_names in enumerate(folds):
        sizes = {k: len(v) for k, v in split_names.items()}
        print(f"  fold {f}: train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}")
        _write_fold(args.out_dir, f, by_track, split_names)


if __name__ == "__main__":
    main()
