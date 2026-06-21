"""The BPS summary aggregators must parse both dir orderings:
  - layer-primary (new):  ...-layers.layer6.fold3
  - fold-primary (old):   ...-layers.fold3.layer6
Returns (fold, layer) so it matches the (f, l) cell key, or None.
"""

from scripts.sweeps.bps_mnid_summary import parse_fold_layer


def test_layer_primary():
    assert parse_fold_layer("probe.BPSMotifMNID.CLaMP3-symbolic-layers.layer6.fold3") == (3, 6)


def test_fold_primary_backcompat():
    assert parse_fold_layer("probe.BPSMotifMNID.CLaMP3-symbolic-layers.fold3.layer6") == (3, 6)


def test_double_digit_layer_primary():
    assert parse_fold_layer("x.CLaMP3-symbolic-layers.layer12.fold0") == (0, 12)


def test_meanall_or_other_returns_none():
    assert parse_fold_layer("probe.BPSMotifMNID.CLaMP3-symbolic-meanall.fold3") is None
    assert parse_fold_layer("nonsense") is None
