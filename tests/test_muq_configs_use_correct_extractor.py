"""Regression guard for audit issue #1.

MuQ expects a raw waveform tensor in [-1, 1]. The MERT-specific feature
extractor (``Wav2Vec2FeatureExtractor`` with ``do_normalize=True``)
applies zero-mean unit-variance normalization to the waveform, producing
values with std=1.0 (up to ±1.4 peak) — an out-of-distribution input
for MuQ. Several MuQ configs were originally cloned from MERT templates
and inadvertently kept the MERT feature extractor, silently corrupting
the MuQ × {HXMSA, HookTheoryKey, HookTheoryStructure, SuperMarioStructure}
sweeps.

This test prevents that regression: every ``configs/probe.MuQ-*.yaml``
must use the encoder-agnostic ``OMARRQ_FeatureExtractor`` (which only
mono-squeezes), not MERT's. See ``marble/encoders/MERT/model.py`` and
``marble/encoders/OMAR_RQ/model.py`` for the two extractors' actual
behaviors.
"""

from __future__ import annotations

import glob
from pathlib import Path


def test_muq_configs_never_use_mert_feature_extractor():
    """No ``configs/probe.MuQ-*.yaml`` may reference
    ``MERT_v1_95M_FeatureExtractor`` — MuQ needs raw waveform input,
    not MERT's normalized output. Use ``OMARRQ_FeatureExtractor`` for
    mono-squeeze behavior instead.
    """
    repo_root = Path(__file__).resolve().parent.parent
    pattern = str(repo_root / "configs" / "probe.MuQ-*.yaml")
    cfgs = glob.glob(pattern)
    assert cfgs, f"No MuQ configs found under {pattern} — check working dir"

    offenders: list[str] = []
    for cfg in cfgs:
        with open(cfg) as f:
            text = f.read()
        if "MERT_v1_95M_FeatureExtractor" in text:
            offenders.append(cfg)

    assert not offenders, (
        "The following MuQ configs use MERT's feature extractor, which "
        "applies waveform normalization MuQ wasn't trained on. Replace "
        "with marble.encoders.OMAR_RQ.model.OMARRQ_FeatureExtractor:\n"
        + "\n".join(f"  - {p}" for p in offenders)
    )
