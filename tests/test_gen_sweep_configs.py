"""Tests for gen_sweep_configs path patching.

patch_output_dirs appends .layer{N} (and an optional dir_suffix that lands
AFTER the layer) to the dirpath + save_dir output paths. The suffix is how a
fold sweep gets layer-primary dirs: ...-layers.layer6.fold3 (so `ls` groups
all folds of a layer together).
"""

from scripts.sweeps.gen_sweep_configs import patch_output_dirs


def test_appends_layer_to_save_dir():
    t = 'save_dir: "./output/probe.X.CLaMP3-symbolic-layers/"'
    assert patch_output_dirs(t, 6) == 'save_dir: "./output/probe.X.CLaMP3-symbolic-layers.layer6/"'


def test_appends_layer_to_dirpath_preserving_tail():
    t = 'dirpath: "./output/probe.X.CLaMP3-symbolic-layers/checkpoints/"'
    got = patch_output_dirs(t, 2)
    assert got == 'dirpath: "./output/probe.X.CLaMP3-symbolic-layers.layer2/checkpoints/"'


def test_dir_suffix_is_layer_primary():
    t = 'save_dir: "./output/probe.X.CLaMP3-symbolic-layers/"'
    got = patch_output_dirs(t, 6, ".fold3")
    assert got == 'save_dir: "./output/probe.X.CLaMP3-symbolic-layers.layer6.fold3/"'


def test_dir_suffix_on_dirpath():
    t = 'dirpath: "./output/probe.X.CLaMP3-symbolic-layers/checkpoints/"'
    got = patch_output_dirs(t, 12, ".fold0")
    assert got == 'dirpath: "./output/probe.X.CLaMP3-symbolic-layers.layer12.fold0/checkpoints/"'


def test_patches_both_dirpath_and_save_dir_together():
    t = (
        'dirpath: "./output/base.CLaMP3-symbolic-layers/checkpoints/"\n'
        'save_dir: "./output/base.CLaMP3-symbolic-layers/"'
    )
    got = patch_output_dirs(t, 7, ".fold1")
    assert "base.CLaMP3-symbolic-layers.layer7.fold1/checkpoints/" in got
    assert "base.CLaMP3-symbolic-layers.layer7.fold1/" in got


def test_empty_suffix_default_unchanged_behavior():
    # default (no fold sweep) must match the historical naming exactly
    t = 'save_dir: "./output/probe.GS.OMARRQ-layers/"'
    assert patch_output_dirs(t, 0) == 'save_dir: "./output/probe.GS.OMARRQ-layers.layer0/"'
