"""Tests for the MedleyDB artist-conditional 5-fold split (leak-free CV)."""

from __future__ import annotations

import pytest

from marble.tasks.MedleyDBMelody.split import fold_split


def _tracks_with_artists(spec: dict[str, int]) -> tuple[list[str], dict[str, str]]:
    """spec: artist -> n_tracks. Returns (track_list, artist_of_map)."""
    tracks, artist_of = [], {}
    for artist, n in spec.items():
        for i in range(n):
            t = f"{artist}_song{i}"
            tracks.append(t)
            artist_of[t] = artist
    return tracks, artist_of


def _make(spec, **kw):
    tracks, amap = _tracks_with_artists(spec)
    return fold_split(tracks, artist_of=amap.get, **kw), amap


def test_returns_n_folds():
    folds, _ = _make({f"a{i}": 2 for i in range(20)}, n_folds=5, seed=0)
    assert len(folds) == 5
    for f in folds:
        assert set(f) == {"train", "val", "test"}


def test_no_artist_leaks_across_splits_within_a_fold():
    folds, amap = _make({f"a{i}": 3 for i in range(20)}, n_folds=5, seed=0)
    for f in folds:
        artists = {k: {amap[t] for t in f[k]} for k in ("train", "val", "test")}
        assert artists["train"].isdisjoint(artists["test"])
        assert artists["train"].isdisjoint(artists["val"])
        assert artists["val"].isdisjoint(artists["test"])


def test_test_folds_partition_all_tracks():
    spec = {f"a{i}": (i % 3) + 1 for i in range(20)}
    folds, _ = _make(spec, n_folds=5, seed=0)
    all_tracks = {t for artist, n in spec.items() for t in [f"{artist}_song{j}" for j in range(n)]}
    test_union, seen = set(), 0
    for f in folds:
        test_union |= set(f["test"])
        seen += len(f["test"])
    assert test_union == all_tracks  # every track tested exactly once
    assert seen == len(all_tracks)  # disjoint across folds


def test_every_split_nonempty():
    folds, _ = _make({f"a{i}": 2 for i in range(20)}, n_folds=5, seed=0)
    for f in folds:
        assert len(f["train"]) > 0 and len(f["val"]) > 0 and len(f["test"]) > 0


def test_deterministic_given_seed():
    f1, _ = _make({f"a{i}": 2 for i in range(20)}, n_folds=5, seed=7)
    f2, _ = _make({f"a{i}": 2 for i in range(20)}, n_folds=5, seed=7)
    assert f1 == f2


def test_raises_when_too_few_artists_for_folds():
    # 3 artists cannot form 5 non-empty test folds.
    with pytest.raises(ValueError, match="artist|fold"):
        _make({"a": 2, "b": 2, "c": 2}, n_folds=5, seed=0)


def test_musicdelta_style_one_big_artist_does_not_leak():
    # One 32-track 'artist' (the MusicDelta failure mode) must stay within a
    # single fold's test (or wholly in train across other folds), never split.
    spec = {"MusicDelta": 32}
    spec.update({f"a{i}": 2 for i in range(20)})
    folds, amap = _make(spec, n_folds=5, seed=0)
    for f in folds:
        in_test = any(amap[t] == "MusicDelta" for t in f["test"])
        in_train = any(amap[t] == "MusicDelta" for t in f["train"])
        in_val = any(amap[t] == "MusicDelta" for t in f["val"])
        # MusicDelta tracks are all in exactly one of the three splits, never split across.
        assert sum([in_test, in_train, in_val]) == 1
