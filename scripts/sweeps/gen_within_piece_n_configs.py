#!/usr/bin/env python3
"""
scripts/sweeps/gen_within_piece_n_configs.py
────────────────────────────────────────────
Generate per-N BPS-Motif WITHIN-PIECE (clip-isolated) layers + meanall configs
for the window-size breaking-point sweep, from the canonical N4 templates.

The N4 task (``BPSMotifWithinPieceN4``) hardcoded the JSONL path in its dataset
class so it needed no init args. The window-size sweep scores MANY window sizes
N (in bars), each at its OWN best layer (bigger N may prefer a deeper layer), so
we need one generic dataset class + per-N config files that differ ONLY in:

  * the test/dummy dataset JSONL path     → the generic
    ``BPSMotifWithinPieceABC{Test,Dummy}`` class's ``jsonl_template`` init arg,
  * the wandb group / tags / run save_dir → ``BPSMotifWithinPieceN{N}``,

so the ``run_sweep_local.py`` meanall-sibling resolver
(``probe.<enc>-meanall.<task>.yaml``) and per-layer config-naming all keep
working unchanged with ``--task-tag BPSMotifWithinPieceN{N}``.

We TRANSFORM the two N4 templates (layers + meanall) per N rather than emit YAML
from scratch so the generated configs stay byte-for-byte identical to the proven
N4 configs except for the four N-specific substitutions — no drift in encoder /
transforms / trainer / metric wiring.

Per N it writes (alongside the N4 templates, in ``configs/``):

    probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPieceN{N}.yaml
    probe.CLaMP3-symbolic-abc-meanall.BPSMotifWithinPieceN{N}.yaml

each with the test/dummy dataset rewritten to the generic
``BPSMotifWithinPieceABC{Test,Dummy}`` class carrying::

    init_args:
      jsonl_template: data/BPS-Motif/BPSMotifWithinPiece.N{N}.ABC.jsonl

Build each window's dataset first::

    uv run python scripts/data/build_bps_motif_within_piece.py --window {N} --workers 8

Usage::

    python scripts/sweeps/gen_within_piece_n_configs.py --windows 1 2 4 6 8 12 16 24 32
    # N4 is skipped by default (its canonical templates are the source); pass
    # --include-n4 to also emit generic N4 configs (not needed — the originals work).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"

LAYERS_TEMPLATE = "probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPieceN4.yaml"
MEANALL_TEMPLATE = "probe.CLaMP3-symbolic-abc-meanall.BPSMotifWithinPieceN4.yaml"


def _retag(text: str, n: int) -> str:
    """Replace every ``BPSMotifWithinPieceN4`` token with ``...N{n}``.

    Covers the wandb group, tags, save_dir, and the dataset-build comment — all
    the places the N4 string appears. Done as a plain string replace because the
    token is unambiguous (no ``BPSMotifWithinPieceWholeN4`` in these two
    clip-isolated templates).
    """
    return text.replace("BPSMotifWithinPieceN4", f"BPSMotifWithinPieceN{n}")


def _rewrite_datasets(text: str, n: int) -> str:
    """Point train/val/test at the GENERIC class + this N's JSONL.

    The N4 templates name the hardcoded-path classes
    ``BPSMotifWithinPieceN4ABC{Dummy,Test}`` with NO init_args. We swap them for
    the generic ``BPSMotifWithinPieceABC{Dummy,Test}`` and attach a
    ``jsonl_template`` init arg pointing at this N's JSONL. The generic class
    takes ``jsonl_template`` as a direct (leaf) init arg, so LightningCLI
    instantiates it natively from the nested YAML ``init_args`` block.
    """
    jsonl = f"data/BPS-Motif/BPSMotifWithinPiece.N{n}.ABC.jsonl"

    # train/val → generic Dummy with init_args. The N4 templates write these as
    #     train:
    #       class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceN4ABCDummy
    # (likewise val). Rewrite each to the generic class + init_args, preserving
    # the original indentation of the class_path line.
    def _dummy_repl(m: re.Match) -> str:
        key = m.group("key")
        key_indent = m.group("kindent")
        cp_indent = m.group("cindent")
        ia_indent = cp_indent  # init_args at same indent as class_path
        return (
            f"{key_indent}{key}:\n"
            f"{cp_indent}class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceABCDummy\n"
            f"{ia_indent}init_args:\n"
            f"{ia_indent}  jsonl_template: {jsonl}"
        )

    text = re.sub(
        r"(?P<kindent>[ \t]*)(?P<key>train|val):\n"
        r"(?P<cindent>[ \t]+)class_path: [^\n]*BPSMotifWithinPieceN4ABCDummy",
        _dummy_repl,
        text,
    )

    # test → generic Test with init_args.
    def _test_repl(m: re.Match) -> str:
        key_indent = m.group("kindent")
        cp_indent = m.group("cindent")
        return (
            f"{key_indent}test:\n"
            f"{cp_indent}class_path: marble.tasks.BPSMotif.datamodule.BPSMotifWithinPieceABCTest\n"
            f"{cp_indent}init_args:\n"
            f"{cp_indent}  jsonl_template: {jsonl}"
        )

    text = re.sub(
        r"(?P<kindent>[ \t]*)test:\n"
        r"(?P<cindent>[ \t]+)class_path: [^\n]*BPSMotifWithinPieceN4ABCTest",
        _test_repl,
        text,
    )
    return text


def _patch_window_comment(text: str, n: int) -> str:
    """Fix the ``--window 4`` dataset-build hint in the header comment to N."""
    return text.replace(
        "build_bps_motif_within_piece.py --window 4",
        f"build_bps_motif_within_piece.py --window {n}",
    )


def gen_for_n(n: int) -> list[Path]:
    written: list[Path] = []
    for template_name in (LAYERS_TEMPLATE, MEANALL_TEMPLATE):
        src = CONFIGS / template_name
        text = src.read_text()
        # Rewrite the dataset classes FIRST (matches the N4 class names in the
        # template), then retag — otherwise _retag would rename the N4 class
        # tokens to N{n} and the dataset regex would no longer match.
        text = _rewrite_datasets(text, n)
        text = _retag(text, n)
        text = _patch_window_comment(text, n)
        out_name = template_name.replace("BPSMotifWithinPieceN4", f"BPSMotifWithinPieceN{n}")
        out_path = CONFIGS / out_name
        out_path.write_text(text)
        written.append(out_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--windows",
        type=int,
        nargs="+",
        required=True,
        help="Window sizes N (bars) to generate configs for, e.g. 1 2 6 8 12 16 24 32.",
    )
    ap.add_argument(
        "--include-n4",
        action="store_true",
        help="Also (re)generate generic N4 configs. Off by default — the canonical "
        "hardcoded-path N4 templates are the source and already work.",
    )
    args = ap.parse_args()

    for tmpl in (LAYERS_TEMPLATE, MEANALL_TEMPLATE):
        if not (CONFIGS / tmpl).exists():
            raise SystemExit(f"Template not found: {CONFIGS / tmpl}")

    for n in sorted(set(args.windows)):
        if n == 4 and not args.include_n4:
            print(
                f"  N={n:>2}  skipped (canonical N4 templates are the source; --include-n4 to force)"
            )
            continue
        paths = gen_for_n(n)
        for p in paths:
            print(f"  N={n:>2}  ->  {p.relative_to(REPO)}")

    print("\nGenerated per-N within-piece configs. Build each window's dataset, then sweep:")
    print("  uv run python scripts/data/build_bps_motif_within_piece.py --window N --workers 8")
    print("  scripts/sweeps/run_bps_within_piece_n.sh --windows ... --concurrency 4")


if __name__ == "__main__":
    main()
