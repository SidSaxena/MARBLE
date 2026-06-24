#!/usr/bin/env python3
"""
scripts/sweeps/gen_within_piece_n_configs.py
────────────────────────────────────────────
Generate per-N BPS-Motif WITHIN-PIECE layers + meanall configs for the
window-size breaking-point sweep, from the canonical N4 templates — for BOTH
encoding arms:

  * clip-isolated  (``BPSMotifWithinPiece``)      — each N-bar window's OWN ABC
  * whole-piece    (``BPSMotifWithinPieceWhole``)  — whole movement, pool N bars

The N4 tasks hardcoded the JSONL path in their dataset class so they needed no
init args. The window-size sweep scores MANY window sizes N (in bars), each at
its OWN best layer, so we need one generic dataset class + per-N config files
that differ ONLY in:

  * the test/dummy dataset JSONL path     → the generic
    ``BPSMotifWithinPiece[Whole]ABC{Test,Dummy}`` class's ``jsonl_template``,
  * the wandb group / tags / run save_dir → ``BPSMotifWithinPiece[Whole]N{N}``.

We TRANSFORM the two N4 templates per N (and per arm) rather than emit YAML from
scratch so the generated configs stay byte-for-byte identical to the proven N4
configs except for the N-specific substitutions — no drift in encoder /
transforms / trainer / metric wiring.

Per N per arm it writes (alongside the N4 templates, in ``configs/``)::

    probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPiece[Whole]N{N}.yaml
    probe.CLaMP3-symbolic-abc-meanall.BPSMotifWithinPiece[Whole]N{N}.yaml

Build each window's dataset first::

    uv run python scripts/data/build_bps_motif_within_piece.py --window {N} [--whole]

Usage::

    # both arms (default):
    python scripts/sweeps/gen_within_piece_n_configs.py --windows 1 2 3 6 8 12 16 24 32
    # one arm:
    python scripts/sweeps/gen_within_piece_n_configs.py --windows 8 --arms clip
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"

# Per-arm config knobs. ``token`` is the unambiguous N4 string in the template
# (wandb group / save_dir / tags / dataset class); ``ds_class`` is the generic
# class the per-N config points at; ``jsonl`` is that arm's per-N JSONL stem.
ARMS = {
    "clip": {
        "layers_tmpl": "probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPieceN4.yaml",
        "meanall_tmpl": "probe.CLaMP3-symbolic-abc-meanall.BPSMotifWithinPieceN4.yaml",
        "token": "BPSMotifWithinPieceN4",  # not a substring of ...WholeN4
        "n4_test_class": "BPSMotifWithinPieceN4ABCTest",
        "n4_dummy_class": "BPSMotifWithinPieceN4ABCDummy",
        "test_class": "BPSMotifWithinPieceABCTest",
        "dummy_class": "BPSMotifWithinPieceABCDummy",
        "jsonl": "data/BPS-Motif/BPSMotifWithinPiece.N{n}.ABC.jsonl",
    },
    "whole": {
        "layers_tmpl": "probe.CLaMP3-symbolic-abc-layers.BPSMotifWithinPieceWholeN4.yaml",
        "meanall_tmpl": "probe.CLaMP3-symbolic-abc-meanall.BPSMotifWithinPieceWholeN4.yaml",
        "token": "BPSMotifWithinPieceWholeN4",
        "n4_test_class": "BPSMotifWithinPieceWholeN4ABCTest",
        "n4_dummy_class": "BPSMotifWithinPieceWholeN4ABCDummy",
        "test_class": "BPSMotifWithinPieceWholeABCTest",
        "dummy_class": "BPSMotifWithinPieceWholeABCDummy",
        "jsonl": "data/BPS-Motif/BPSMotifWithinPieceWhole.N{n}.ABC.jsonl",
    },
}


def _rewrite_datasets(text: str, arm: dict, n: int) -> str:
    """Point train/val/test at the GENERIC class + this N's JSONL.

    The N4 templates name hardcoded-path classes with NO init_args. Swap them for
    the arm's generic class and attach a ``jsonl_template`` init arg. Done BEFORE
    retag so the N4 class tokens still match (retag would rename them otherwise),
    and so the generic class names we write (no ``N4``) survive retag untouched.
    """
    jsonl = arm["jsonl"].format(n=n)

    def _dummy_repl(m: re.Match) -> str:
        key = m.group("key")
        key_indent = m.group("kindent")
        cp_indent = m.group("cindent")
        return (
            f"{key_indent}{key}:\n"
            f"{cp_indent}class_path: marble.tasks.BPSMotif.datamodule.{arm['dummy_class']}\n"
            f"{cp_indent}init_args:\n"
            f"{cp_indent}  jsonl_template: {jsonl}"
        )

    text = re.sub(
        r"(?P<kindent>[ \t]*)(?P<key>train|val):\n"
        r"(?P<cindent>[ \t]+)class_path: [^\n]*" + re.escape(arm["n4_dummy_class"]),
        _dummy_repl,
        text,
    )

    def _test_repl(m: re.Match) -> str:
        key_indent = m.group("kindent")
        cp_indent = m.group("cindent")
        return (
            f"{key_indent}test:\n"
            f"{cp_indent}class_path: marble.tasks.BPSMotif.datamodule.{arm['test_class']}\n"
            f"{cp_indent}init_args:\n"
            f"{cp_indent}  jsonl_template: {jsonl}"
        )

    text = re.sub(
        r"(?P<kindent>[ \t]*)test:\n"
        r"(?P<cindent>[ \t]+)class_path: [^\n]*" + re.escape(arm["n4_test_class"]),
        _test_repl,
        text,
    )

    # Robustness / idempotency (BUGFIX): whatever class form the template used,
    # FORCE every jsonl_template to THIS window's data. The class-name regexes
    # above only match the CANONICAL N4 template; if a template was already
    # rewritten to the generic class (e.g. --include-n4 overwrites the N4 template
    # in place, then later windows read THAT file), those regexes no-op and the
    # jsonl would silently stay at N4 — the bug that made windows 6..32 all encode
    # N4 data. This catch-all replaces any ``<stem>.N<d>.ABC.jsonl`` for THIS
    # arm's stem with the correct window, regardless of class form.
    stem = arm["jsonl"].split(".N")[0]  # data/.../BPSMotifWithinPiece[Whole]
    text = re.sub(re.escape(stem) + r"\.N\d+\.ABC\.jsonl", jsonl, text)
    return text


def _retag(text: str, arm: dict, n: int) -> str:
    """Replace the arm's ``...N4`` group/save_dir/tag token with ``...N{n}``.

    Run AFTER ``_rewrite_datasets`` so it no longer hits dataset class_paths
    (those were swapped to the generic, N4-free class names).
    """
    return text.replace(arm["token"], arm["token"].replace("N4", f"N{n}"))


def _neutralize_group(text: str, arm: dict, n: int) -> str:
    """Drop the ``N{n}`` from the wandb ``group:`` line only — fold-style.

    The window-size sweep is the window analog of the fold sweep: ALL windows of
    an arm live in ONE wandb group (``CLaMP3-symbolic-abc / BPSMotifWithinPiece``
    [or ``…Whole``]) and are distinguished by the ``sweep/window`` coord (stamped
    by LogSweepCoordsCallback from the JSONL), exactly as folds share a group and
    differ by ``sweep/fold``. save_dir / tags / task-tag KEEP the ``N{n}`` so
    per-window outputs and resume-skip stay isolated; only the group is neutral.
    """
    neutral = arm["token"].replace("N4", "")  # BPSMotifWithinPiece[Whole]
    nn = arm["token"].replace("N4", f"N{n}")
    return re.sub(
        r'(group:\s*"[^"\n]*?)' + re.escape(nn) + r'(")',
        lambda m: m.group(1) + neutral + m.group(2),
        text,
    )


def _patch_window_comment(text: str, n: int) -> str:
    """Fix the ``--window 4`` dataset-build hint in the header comment to N."""
    return text.replace(
        "build_bps_motif_within_piece.py --window 4",
        f"build_bps_motif_within_piece.py --window {n}",
    )


def gen_for_n(arm_name: str, n: int) -> list[Path]:
    arm = ARMS[arm_name]
    written: list[Path] = []
    expected_jsonl = arm["jsonl"].format(n=n)
    wrong_jsonl = arm["jsonl"].format(n=4)  # the silent-failure value
    for tmpl_key in ("layers_tmpl", "meanall_tmpl"):
        src = CONFIGS / arm[tmpl_key]
        text = src.read_text()
        text = _rewrite_datasets(text, arm, n)
        text = _retag(text, arm, n)
        text = _neutralize_group(text, arm, n)
        text = _patch_window_comment(text, n)
        # FAIL LOUD: every split must point at THIS window's JSONL, and (for n!=4)
        # NONE may still point at N4. This catches the regression where windows
        # >4 silently inherited N4's jsonl and encoded the wrong data.
        if expected_jsonl not in text or (n != 4 and wrong_jsonl in text):
            raise SystemExit(
                f"[gen] {arm_name} N={n}: jsonl rewrite FAILED for {arm[tmpl_key]} "
                f"— expected '{expected_jsonl}' present and '{wrong_jsonl}' absent."
            )
        out_name = arm[tmpl_key].replace("N4", f"N{n}")
        out_path = CONFIGS / out_name
        out_path.write_text(text)
        written.append(out_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--windows", type=int, nargs="+", required=True)
    ap.add_argument(
        "--arms",
        nargs="+",
        choices=["clip", "whole"],
        default=["clip", "whole"],
        help="Which encoding arm(s) to generate (default: both).",
    )
    ap.add_argument(
        "--include-n4",
        action="store_true",
        help="Also (re)generate generic N4 configs (off by default — the "
        "canonical hardcoded-path N4 templates are the source and already work).",
    )
    args = ap.parse_args()

    for arm_name in args.arms:
        for key in ("layers_tmpl", "meanall_tmpl"):
            tmpl = CONFIGS / ARMS[arm_name][key]
            if not tmpl.exists():
                raise SystemExit(f"Template not found: {tmpl}")

    for arm_name in args.arms:
        for n in sorted(set(args.windows)):
            if n == 4 and not args.include_n4:
                print(f"  [{arm_name:5}] N={n:>2}  skipped (canonical N4 is the source)")
                continue
            for p in gen_for_n(arm_name, n):
                print(f"  [{arm_name:5}] N={n:>2}  ->  {p.relative_to(REPO)}")

    print("\nGenerated per-N within-piece configs. Build each window's dataset, then sweep:")
    print("  uv run python scripts/data/build_bps_motif_within_piece.py --window N [--whole]")
    print("  scripts/sweeps/run_bps_within_piece_window_sweep.sh --windows ... --concurrency 4")


if __name__ == "__main__":
    main()
