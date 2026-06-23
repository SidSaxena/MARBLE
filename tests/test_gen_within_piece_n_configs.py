"""Tests for the per-N within-piece config generator.

The window-size breaking-point sweep transforms the canonical N4 templates into
per-N (layers + meanall) configs that (a) retag wandb group/tags/save_dir to
``BPSMotifWithinPieceN{N}`` and (b) rewrite the train/val/test datasets to the
GENERIC ``BPSMotifWithinPieceABC{Test,Dummy}`` class carrying a per-N
``jsonl_template`` init arg. The dataset rewrite MUST run before the retag (else
the N4 class-name regex no longer matches) — this is the bug these tests pin.
"""

import yaml

from scripts.sweeps.gen_within_piece_n_configs import (
    _patch_window_comment,
    _retag,
    _rewrite_datasets,
)

LAYERS_TEMPLATE = """\
trainer:
  logger:
    init_args:
      group: "CLaMP3-symbolic-abc / BPSMotifWithinPieceN4"
      tags: ["CLaMP3-symbolic-abc", "BPSMotifWithinPieceN4", "within-piece"]
      save_dir: "./output/probe.BPSMotifWithinPieceN4.CLaMP3-symbolic-abc-layers/"
data:
  init_args:
    train:
      class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceN4ABCDummy
    val:
      class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceN4ABCDummy
    test:
      class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceN4ABCTest
"""


def _transform(text: str, n: int) -> str:
    # Same order as gen_for_n: rewrite datasets FIRST, then retag.
    text = _rewrite_datasets(text, n)
    text = _retag(text, n)
    text = _patch_window_comment(text, n)
    return text


def test_dataset_rewrite_uses_generic_class_with_jsonl():
    out = _transform(LAYERS_TEMPLATE, 8)
    d = yaml.safe_load(out)
    test = d["data"]["init_args"]["test"]
    assert test["class_path"].endswith("BPSMotifWithinPieceABCTest")
    assert test["init_args"]["jsonl_template"] == "data/BPS-Motif/BPSMotifWithinPiece.N8.ABC.jsonl"
    for k in ("train", "val"):
        ds = d["data"]["init_args"][k]
        assert ds["class_path"].endswith("BPSMotifWithinPieceABCDummy")
        assert (
            ds["init_args"]["jsonl_template"] == "data/BPS-Motif/BPSMotifWithinPiece.N8.ABC.jsonl"
        )


def test_no_n4_dataset_classes_remain():
    out = _transform(LAYERS_TEMPLATE, 16)
    assert "BPSMotifWithinPieceN4ABCDummy" not in out
    assert "BPSMotifWithinPieceN4ABCTest" not in out
    # the generic classes must NOT have an N glued on
    assert "BPSMotifWithinPieceN16ABCTest" not in out


def test_retag_updates_group_tags_savedir():
    out = _transform(LAYERS_TEMPLATE, 24)
    d = yaml.safe_load(out)
    ia = d["trainer"]["logger"]["init_args"]
    assert ia["group"] == "CLaMP3-symbolic-abc / BPSMotifWithinPieceN24"
    assert "BPSMotifWithinPieceN24" in ia["tags"]
    assert "BPSMotifWithinPieceN24" in ia["save_dir"]
    assert "BPSMotifWithinPieceN4" not in out


def test_output_is_valid_yaml_for_all_windows():
    for n in (1, 2, 6, 8, 12, 16, 24, 32):
        out = _transform(LAYERS_TEMPLATE, n)
        d = yaml.safe_load(out)  # raises on malformed YAML
        assert (
            d["data"]["init_args"]["test"]["init_args"]["jsonl_template"]
            == f"data/BPS-Motif/BPSMotifWithinPiece.N{n}.ABC.jsonl"
        )
