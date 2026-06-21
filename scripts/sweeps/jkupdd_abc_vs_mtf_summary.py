"""Aggregate the JKUPDD **ABC-vs-MTF** A/B layer sweeps, side by side.

This compares two zero-shot CLaMP3-symbolic layer sweeps run on the *identical*
occurrence pool (the 66 occurrences / 15 groups that survived score-native ABC
alignment — see ``scripts/data/build_jkupdd_abc.py`` + ``docs/jkupdd_abc_vs_mtf.md``):

  * **ABC**  — task tag ``JKUPDDRetrievalABC``: each occurrence tokenised from a
    notation-preserving interleaved-ABC string sliced from the piece ``**kern``.
  * **MTF**  — task tag ``JKUPDDRetrievalMatched``: the same occurrences tokenised
    from the lossy ``**kern → MIDI → MTF`` path (the existing pipeline), restricted
    to the same 66 so the A/B is apples-to-apples.

Both have **no CV folds** (one test run per layer) and a ``-meanall`` baseline.
Output dirs (globbed under ``<base>/output``)::

    probe.JKUPDDRetrievalABC.CLaMP3-symbolic-layers.layer{N}/...
    probe.JKUPDDRetrievalABC.CLaMP3-symbolic-meanall/...
    probe.JKUPDDRetrievalMatched.CLaMP3-symbolic-layers.layer{N}/...
    probe.JKUPDDRetrievalMatched.CLaMP3-symbolic-meanall/...

Reports a per-layer ``test/map`` table (ABC | MTF | Δ) raw + centered, flags each
arm's best layer, and writes a leaderboard CSV. Pure stdlib.

Usage::

  python3 scripts/sweeps/jkupdd_abc_vs_mtf_summary.py \
      [--base DIR] [--out-csv PATH] [--out-md PATH]
"""

import argparse
import csv
import glob
import json
import os
import re

_LAYER_RX = re.compile(r"-layers\.layer(\d+)(?!\.fold)(?:$|[^0-9])")
RECALL_KS = (1, 5, 10, 50)


def parse_layer(name: str) -> int | None:
    m = _LAYER_RX.search(name)
    return int(m.group(1)) if m else None


def _latest_test_summary(run_glob: str) -> dict | None:
    picked = None
    for p in sorted(glob.glob(run_glob)):  # ascending timestamp -> last wins
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if any(k.startswith("test/") for k in d):
            picked = d
    return picked


def load_arm(base: str, task_tag: str) -> tuple[dict[int, dict], dict | None]:
    """Return ``({layer: summary}, meanall_summary)`` for one A/B arm."""
    cells: dict[int, dict] = {}
    root = os.path.join(base, "output", f"probe.{task_tag}.CLaMP3-symbolic-layers.*")
    for d in sorted(glob.glob(root)):
        layer = parse_layer(d)
        if layer is None:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[layer] = summ
    mroot = os.path.join(base, "output", f"probe.{task_tag}.CLaMP3-symbolic-meanall")
    meanall = _latest_test_summary(
        os.path.join(mroot, "wandb", "run-*", "files", "wandb-summary.json")
    )
    return cells, meanall


def _f(x, p=4):
    return f"{x:.{p}f}" if x is not None else "  --  "


def _delta(a, b):
    return (a - b) if (a is not None and b is not None) else None


def _best(cells: dict[int, dict], key="test/map") -> int | None:
    scored = {l: c.get(key) for l, c in cells.items() if c.get(key) is not None}
    return max(scored, key=scored.__getitem__) if scored else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/developer/python/marble"))
    ap.add_argument("--abc-tag", default="JKUPDDRetrievalABC")
    ap.add_argument("--mtf-tag", default="JKUPDDRetrievalMatched")
    ap.add_argument("--out-json", default="/tmp/jkupdd_abc_vs_mtf.json")
    ap.add_argument("--out-md", default="/tmp/jkupdd_abc_vs_mtf.md")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    abc, abc_mean = load_arm(args.base, args.abc_tag)
    mtf, mtf_mean = load_arm(args.base, args.mtf_tag)
    if not abc or not mtf:
        print(f"missing runs: abc layers={len(abc)} mtf layers={len(mtf)} under {args.base}")
        return

    layer_ids = sorted(set(abc) | set(mtf))
    abc_best, mtf_best = _best(abc), _best(mtf)

    def ac(l, k):
        return abc.get(l, {}).get(k)

    def mc(l, k):
        return mtf.get(l, {}).get(k)

    out = []
    out.append("# JKUPDD Retrieval — ABC vs MTF (per-layer MAP, identical pool)")
    out.append("")
    out.append(
        "Same 66 occurrences / 15 groups (Bach, Chopin, Gibbons, Mozart — Beethoven "
        "dropped at alignment). Zero-shot CLaMP3-symbolic, no CV folds. "
        "Δ = ABC − MTF (positive ⇒ notation-preserving ABC helps)."
    )
    out.append("")
    out.append("## raw test/map")
    out.append("")
    out.append("| layer | ABC | MTF | Δ (ABC−MTF) |")
    out.append("|------:|----:|----:|------------:|")
    for l in layer_ids:
        a, m = ac(l, "test/map"), mc(l, "test/map")
        flags = ""
        if l == abc_best:
            flags += " ⭐ABC"
        if l == mtf_best:
            flags += " ⭐MTF"
        out.append(f"| {l}{flags} | {_f(a)} | {_f(m)} | {_f(_delta(a, m))} |")
    if abc_mean or mtf_mean:
        am = abc_mean.get("test/map") if abc_mean else None
        mm = mtf_mean.get("test/map") if mtf_mean else None
        out.append(f"| **meanall** | {_f(am)} | {_f(mm)} | {_f(_delta(am, mm))} |")
    out.append("")
    out.append("## centered test/map")
    out.append("")
    out.append("| layer | ABC | MTF | Δ (ABC−MTF) |")
    out.append("|------:|----:|----:|------------:|")
    for l in layer_ids:
        a, m = ac(l, "test/map_centered"), mc(l, "test/map_centered")
        out.append(f"| {l} | {_f(a)} | {_f(m)} | {_f(_delta(a, m))} |")
    out.append("")

    # peak summary
    a_peak = ac(abc_best, "test/map") if abc_best is not None else None
    m_peak = mc(mtf_best, "test/map") if mtf_best is not None else None
    out.append("## Verdict (raw MAP)")
    out.append("")
    out.append(f"- **ABC best layer = {abc_best}** (MAP {_f(a_peak)})")
    out.append(f"- **MTF best layer = {mtf_best}** (MAP {_f(m_peak)})")
    out.append(f"- peak Δ (ABC_best − MTF_best) = {_f(_delta(a_peak, m_peak))}")
    if abc_best is not None:
        out.append(
            f"- same-layer Δ at ABC's peak (L{abc_best}): "
            f"{_f(_delta(ac(abc_best, 'test/map'), mc(abc_best, 'test/map')))}"
        )
    out.append("")
    out.append("## recall@K at each arm's best layer (raw)")
    out.append("")
    out.append(f"| K | ABC@L{abc_best} | MTF@L{mtf_best} |")
    out.append("|---:|----:|----:|")
    for k in RECALL_KS:
        out.append(
            f"| {k} | {_f(ac(abc_best, f'test/recall@{k}'))} | "
            f"{_f(mc(mtf_best, f'test/recall@{k}'))} |"
        )
    out.append("")

    md = "\n".join(out)
    print(md)
    with open(args.out_md, "w") as fh:
        fh.write(md)

    payload = {
        "abc_best_layer": abc_best,
        "mtf_best_layer": mtf_best,
        "abc": {str(l): {k: v for k, v in abc[l].items() if k.startswith("test/")} for l in abc},
        "mtf": {str(l): {k: v for k, v in mtf[l].items() if k.startswith("test/")} for l in mtf},
        "abc_meanall": (
            {k: v for k, v in abc_mean.items() if k.startswith("test/")} if abc_mean else None
        ),
        "mtf_meanall": (
            {k: v for k, v in mtf_mean.items() if k.startswith("test/")} if mtf_mean else None
        ),
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[wrote {args.out_json} and {args.out_md}]")

    if args.out_csv:

        def rnd(x):
            return round(x, 4) if x is not None else ""

        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["layer", "abc_map", "mtf_map", "delta_map", "abc_map_centered", "mtf_map_centered"]
            )
            for l in layer_ids:
                w.writerow(
                    [
                        l,
                        rnd(ac(l, "test/map")),
                        rnd(mc(l, "test/map")),
                        rnd(_delta(ac(l, "test/map"), mc(l, "test/map"))),
                        rnd(ac(l, "test/map_centered")),
                        rnd(mc(l, "test/map_centered")),
                    ]
                )
            am = abc_mean.get("test/map") if abc_mean else None
            mm = mtf_mean.get("test/map") if mtf_mean else None
            w.writerow(
                [
                    "meanall",
                    rnd(am),
                    rnd(mm),
                    rnd(_delta(am, mm)),
                    rnd(abc_mean.get("test/map_centered") if abc_mean else None),
                    rnd(mtf_mean.get("test/map_centered") if mtf_mean else None),
                ]
            )
        print(f"[wrote {args.out_csv}]")


if __name__ == "__main__":
    main()
