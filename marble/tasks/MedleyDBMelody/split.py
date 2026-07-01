"""Artist-conditional 5-fold CV split for MedleyDB melody.

`fold_split` is a pure GroupKFold-style partition: artists are bin-packed into
``n_folds`` balanced test folds (no artist spans two splits within a fold), and
each fold's validation set is carved from its train portion artist-conditionally.

Artists are keyed by the track-name prefix (``split('_')[0]`` — MedleyDB track
ids are ``<Artist>_<Title>``). The ``medleydb`` package is deliberately NOT used:
it doesn't import on modern PyYAML (unmaintained), pulls conflicting deps, and its
per-track ``artist`` metadata lumps the ``MusicDelta_*`` genre demos inconsistently
("Music Delta" vs "Music Delta Multitracks") — no better than the prefix. So we
treat ``MusicDelta_*`` as one leak-safe group (a known fold concentration,
mitigated by averaging across the 5 folds).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable


def fold_split(
    tracks: list[str],
    artist_of: Callable[[str], str],
    n_folds: int = 5,
    val_frac: float = 0.15,
    seed: int = 0,
) -> list[dict[str, list[str]]]:
    """Return ``n_folds`` artist-conditional {train,val,test} splits.

    - Test folds partition all tracks (each track tested exactly once).
    - No artist appears in more than one split within a fold (leak-free).
    - Raises ValueError if there are fewer artists than folds, or if any fold
      would have an empty split (skewed group sizes / too-small corpus).
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for t in tracks:
        groups[artist_of(t)].append(t)
    artists = sorted(groups)
    if len(artists) < n_folds:
        raise ValueError(
            f"artist-conditional {n_folds}-fold split needs ≥ {n_folds} distinct "
            f"artists for non-empty test folds; got {len(artists)}."
        )

    # Assign artists to test folds: largest groups first into the smallest fold
    # (greedy bin-packing → balanced fold sizes). Seed breaks ties among
    # equal-sized groups deterministically.
    order = list(artists)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda a: -len(groups[a]))
    fold_of: dict[str, int] = {}
    fold_track_counts = [0] * n_folds
    for a in order:
        f = min(range(n_folds), key=lambda i: (fold_track_counts[i], i))
        fold_of[a] = f
        fold_track_counts[f] += len(groups[a])

    target_val = max(1, round(val_frac * len(tracks)))
    folds: list[dict[str, list[str]]] = []
    for f in range(n_folds):
        test_artists = [a for a in artists if fold_of[a] == f]
        trainval_artists = [a for a in artists if fold_of[a] != f]

        # Carve validation from trainval artist-conditionally (~val_frac of all
        # tracks), seeded per fold so val composition is reproducible.
        val_order = list(trainval_artists)
        random.Random(seed * 1000 + f).shuffle(val_order)
        val_artists: list[str] = []
        val_count = 0
        for a in val_order:
            if val_count >= target_val:
                break
            val_artists.append(a)
            val_count += len(groups[a])
        val_set = set(val_artists)
        train_artists = [a for a in trainval_artists if a not in val_set]

        train = sorted(t for a in train_artists for t in groups[a])
        val = sorted(t for a in val_artists for t in groups[a])
        test = sorted(t for a in test_artists for t in groups[a])
        if not (train and val and test):
            raise ValueError(
                f"fold {f}: empty split (train={len(train)}, val={len(val)}, "
                f"test={len(test)}). Too few artists or skewed group sizes for "
                f"n_folds={n_folds}, val_frac={val_frac}."
            )
        folds.append({"train": train, "val": val, "test": test})
    return folds
