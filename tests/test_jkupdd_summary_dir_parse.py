"""The JKUPDD retrieval aggregator must parse the no-fold per-layer dir naming
and reject BPS-style (foldful) or meanall dirs.

JKUPDD has no CV folds → exactly one run per layer, dirs end with ``.layer{N}``
(no ``.fold{F}`` suffix). ``parse_layer`` returns the layer int, or None for
anything that isn't a JKUPDD per-layer dir.
"""

from scripts.sweeps.jkupdd_retrieval_summary import parse_layer


def test_single_digit_layer():
    assert parse_layer("probe.JKUPDDRetrieval.CLaMP3-symbolic-layers.layer6") == 6


def test_double_digit_layer():
    assert parse_layer("x.CLaMP3-symbolic-layers.layer12") == 12


def test_layer_zero():
    assert parse_layer("probe.JKUPDDRetrieval.CLaMP3-symbolic-layers.layer0") == 0


def test_foldful_bps_dir_returns_none():
    # A BPS-style fold-suffixed dir must NOT be picked up as a JKUPDD cell,
    # even if both sweeps share the output/ root.
    assert parse_layer("probe.BPSMotifRetrieval.CLaMP3-symbolic-layers.layer6.fold3") is None


def test_meanall_or_other_returns_none():
    assert parse_layer("probe.JKUPDDRetrieval.CLaMP3-symbolic-meanall") is None
    assert parse_layer("nonsense") is None
