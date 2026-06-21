"""Aggregate the MTC-ANN **ABC-vs-MTF** A/B layer sweeps, side by side.

Compares two zero-shot CLaMP3-symbolic layer sweeps run on the *identical*
MTC-ANN occurrence pool (same melodies / occurrences, same ``work_id`` /
relevance) but different symbolic encodings, for BOTH tasks:

  * **TuneFamily** — cross-melody tune-family retrieval (relevance = same
    tunefamily). Tasks tags ``MTCANNTuneFamily``.
  * **Motif**      — within-corpus motif retrieval (relevance = same
    (tunefamily, motif)). Task tag ``MTCANNMotif``.

Each task has two arms:

  * **ABC** — model tag ``CLaMP3-symbolic-abc``: each occurrence tokenised from
    a notation-preserving interleaved-ABC string (key / spelling / meter / bars).
  * **MTF** — model tag ``CLaMP3-symbolic``: the same occurrences tokenised from
    the lossy ``MIDI -> MTF`` path.

Both arms have **no CV folds** (one test run per layer) and a ``-meanall``
baseline — exactly like the JKUPDD ABC-vs-MTF sweep. Output dirs (globbed under
``<base>/output``)::

    probe.MTCANN{Task}.CLaMP3-symbolic-abc-layers.layer{N}/...
    probe.MTCANN{Task}.CLaMP3-symbolic-abc-meanall/...
    probe.MTCANN{Task}.CLaMP3-symbolic-layers.layer{N}/...
    probe.MTCANN{Task}.CLaMP3-symbolic-meanall/...

Reports a per-layer ``test/map`` table (ABC | MTF | Δ) raw + centered for each
task, flags each arm's best layer + the depth peak, and writes a leaderboard CSV
+ a JSON. Supports ``--wandb-fallback`` (like the BPS aggregator) so an arm whose
local ``output/`` dirs were cleaned up can still be pulled from W&B by group via
the ``sweep/{layer,stage,repr}`` coords stamped by ``LogSweepCoordsCallback``.
Pure stdlib (W&B import is lazy, only under ``--wandb-fallback``).

Usage::

  python3 scripts/sweeps/mtc_ann_abc_vs_mtf_summary.py \
      [--base DIR] [--task TuneFamily|Motif|both] [--out-csv PATH] [--out-md PATH] \
      [--wandb-fallback]
"""

import argparse
import csv
import glob
import json
import os
import re

# Per-layer dirs: probe.MTCANN{Task}.{model_tag}-layers.layer{N}  (no .fold).
_LAYER_RX = re.compile(r"-layers\.layer(\d+)(?!\.fold)(?:$|[^0-9])")
RECALL_KS = (1, 5, 10, 50)
RETR_METRICS = ["map", "map_centered", "map_whitened", "recall@1", "recall@5", "mrr"]
TASKS = ("TuneFamily", "Motif")

# Motif-only confound-free metrics (logged only by the MTC-ANN Motif task's
# 6-tuple datamodule). ``map`` is the realistic full-gallery number;
# ``map_samefamily`` is the CONFOUND-FREE discriminative number (gallery
# restricted to the query's own tune family, so it measures motif identity not
# tune-family similarity); the ``map_len_*`` pair splits queries by motif
# note-count (<=3 vs >3 notes).
MOTIF_DISCRIM_METRICS = [
    "map",
    "map_samefamily",
    "map_samefamily_centered",
    "map_len_le3",
    "map_len_gt3",
    "map_samefamily_len_le3",
    "map_samefamily_len_gt3",
]


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


def load_arm(base: str, task: str, model_tag: str) -> tuple[dict[int, dict], dict | None]:
    """Return ``({layer: summary}, meanall_summary)`` for one (task, arm).

    ``model_tag`` is ``CLaMP3-symbolic`` (MTF) or ``CLaMP3-symbolic-abc`` (ABC).
    The save_dir prefix is ``probe.MTCANN{task}.{model_tag}-layers`` — note the
    ``.`` boundary before ``{model_tag}-layers`` prevents the MTF glob from
    also matching the ABC dirs (``...-abc-layers``), since the MTF model_tag is
    a prefix of the ABC one.
    """
    cells: dict[int, dict] = {}
    prefix = f"probe.MTCANN{task}.{model_tag}-layers"
    root = os.path.join(base, "output", f"{prefix}.*")
    for d in sorted(glob.glob(root)):
        layer = parse_layer(d)
        if layer is None:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[layer] = summ
    mroot = os.path.join(base, "output", f"probe.MTCANN{task}.{model_tag}-meanall")
    meanall = _latest_test_summary(
        os.path.join(mroot, "wandb", "run-*", "files", "wandb-summary.json")
    )
    return cells, meanall


def load_arm_wandb(
    task: str, model_tag: str, entity: str, project: str
) -> tuple[dict[int, dict], dict | None]:
    """Same as :func:`load_arm` but pulls from W&B by group.

    Used when an arm's local ``output/`` dirs were cleaned up but the runs
    persist on W&B (group ``{model_tag} / MTCANN{task}``). Reads the per-layer
    test runs + the meanall run via the ``sweep/{layer,repr,stage}`` coords
    stamped by ``LogSweepCoordsCallback``. Returns the same
    ``({layer: summary}, meanall_summary)``.
    """
    import wandb

    api = wandb.Api()
    group = f"{model_tag} / MTCANN{task}"
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    cells: dict[int, dict] = {}
    meanall: dict | None = None
    for r in runs:
        cfg = r.config
        if cfg.get("sweep/stage") != "test":
            continue
        summ = {k: v for k, v in r.summary.items() if isinstance(k, str) and k.startswith("test/")}
        if not summ:
            continue
        layer = cfg.get("sweep/layer")
        if cfg.get("sweep/repr") == "meanall" or layer == -1:
            meanall = summ
        elif layer is not None:
            cells[int(layer)] = summ
    return cells, meanall


def _f(x, p=4):
    return f"{x:.{p}f}" if x is not None else "  --  "


def _delta(a, b):
    return (a - b) if (a is not None and b is not None) else None


def _best(cells: dict[int, dict], key="test/map") -> int | None:
    scored = {l: c.get(key) for l, c in cells.items() if c.get(key) is not None}
    return max(scored, key=scored.__getitem__) if scored else None


def summarise_task(args, task: str, out: list[str], payload: dict, csv_rows: list[list]) -> None:
    abc, abc_mean = load_arm(args.base, task, "CLaMP3-symbolic-abc")
    mtf, mtf_mean = load_arm(args.base, task, "CLaMP3-symbolic")
    if not abc and args.wandb_fallback:
        abc, abc_mean = load_arm_wandb(
            task, "CLaMP3-symbolic-abc", args.wandb_entity, args.wandb_project
        )
    if not mtf and args.wandb_fallback:
        mtf, mtf_mean = load_arm_wandb(
            task, "CLaMP3-symbolic", args.wandb_entity, args.wandb_project
        )
    if not abc or not mtf:
        out.append(f"## MTCANN{task}: missing runs (abc layers={len(abc)} mtf layers={len(mtf)})")
        out.append("")
        return

    layer_ids = sorted(set(abc) | set(mtf))
    abc_best, mtf_best = _best(abc), _best(mtf)

    def ac(l, k):
        return abc.get(l, {}).get(k)

    def mc(l, k):
        return mtf.get(l, {}).get(k)

    out.append(f"# MTC-ANN {task} — ABC vs MTF (per-layer MAP, identical pool)")
    out.append("")
    out.append(
        "Identical occurrence pool (same melodies, same work_id / relevance). "
        "Zero-shot CLaMP3-symbolic, no CV folds. "
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
        csv_rows.append([task, l, _f(a), _f(m), _f(_delta(a, m))])
    if abc_mean or mtf_mean:
        am = abc_mean.get("test/map") if abc_mean else None
        mm = mtf_mean.get("test/map") if mtf_mean else None
        out.append(f"| **meanall** | {_f(am)} | {_f(mm)} | {_f(_delta(am, mm))} |")
        csv_rows.append([task, "meanall", _f(am), _f(mm), _f(_delta(am, mm))])
    out.append("")
    out.append("## centered test/map")
    out.append("")
    out.append("| layer | ABC | MTF | Δ (ABC−MTF) |")
    out.append("|------:|----:|----:|------------:|")
    for l in layer_ids:
        a, m = ac(l, "test/map_centered"), mc(l, "test/map_centered")
        out.append(f"| {l} | {_f(a)} | {_f(m)} | {_f(_delta(a, m))} |")
    out.append("")

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

    # ── Motif-only: same-family hard-distractor + length-stratified MAP ──────
    # For the Motif task, the full-gallery ``test/map`` is CONFOUNDED by
    # tune-family similarity (every relevant peer is same-family; ~605/698
    # gallery items per query are other-family easy negatives). The
    # confound-free discriminative number is ``test/map_samefamily`` (gallery
    # restricted to the query's own family). Surface both per layer so the
    # sweep verdict can be read off the right metric.
    if task == "Motif":
        any_sf = any(
            c.get("test/map_samefamily") is not None
            for c in list(abc.values()) + list(mtf.values())
        )
        out.append("## Confound-free Motif metrics (per layer)")
        out.append("")
        out.append(
            "- `test/map` = realistic full-gallery MAP (CONFOUNDED by tune-family "
            "similarity for the Motif task).\n"
            "- **`test/map_samefamily` = the CONFOUND-FREE discriminative metric** "
            "(gallery restricted to the query's own tune family → measures motif "
            "identity, not tune-family identity). Read the verdict off THIS column.\n"
            "- `test/map_len_le3` / `test/map_len_gt3` = MAP for short (≤3-note) vs "
            "long (>3-note) motif queries."
        )
        out.append("")
        if not any_sf:
            out.append(
                "_(No `map_samefamily` found in these runs — they predate the "
                "two-MAP harness. Re-run the sweep with the updated probe.)_"
            )
            out.append("")
        else:
            cols = [
                ("map", "full"),
                ("map_samefamily", "same-fam"),
                ("map_len_le3", "≤3"),
                ("map_len_gt3", ">3"),
            ]
            hdr = "| layer | arm | " + " | ".join(lbl for _, lbl in cols) + " |"
            sep = "|------:|:----|" + "----:|" * len(cols)
            out.append(hdr)
            out.append(sep)
            for l in layer_ids:
                for arm, cells in (("ABC", abc), ("MTF", mtf)):
                    row = cells.get(l, {})
                    vals = " | ".join(_f(row.get(f"test/{m}")) for m, _ in cols)
                    out.append(f"| {l} | {arm} | {vals} |")
            out.append("")
            # Verdict on the confound-free metric.
            abc_sf_best = _best(abc, key="test/map_samefamily")
            mtf_sf_best = _best(mtf, key="test/map_samefamily")
            a_sf = ac(abc_sf_best, "test/map_samefamily") if abc_sf_best is not None else None
            m_sf = mc(mtf_sf_best, "test/map_samefamily") if mtf_sf_best is not None else None
            out.append("### Verdict on the CONFOUND-FREE metric (map_samefamily)")
            out.append("")
            out.append(f"- **ABC best layer = {abc_sf_best}** (map_samefamily {_f(a_sf)})")
            out.append(f"- **MTF best layer = {mtf_sf_best}** (map_samefamily {_f(m_sf)})")
            out.append(f"- peak Δ (ABC_best − MTF_best) = {_f(_delta(a_sf, m_sf))}")
            out.append("")

    payload[task] = {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/developer/marble"))
    ap.add_argument("--task", default="both", choices=["TuneFamily", "Motif", "both"])
    ap.add_argument("--out-json", default="/tmp/mtc_ann_abc_vs_mtf.json")
    ap.add_argument("--out-md", default="/tmp/mtc_ann_abc_vs_mtf.md")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument(
        "--wandb-fallback",
        action="store_true",
        help="If an arm has no local output dirs, pull it from W&B by group "
        "(group `CLaMP3-symbolic[-abc] / MTCANN<task>`).",
    )
    ap.add_argument("--wandb-entity", default="sidsaxena-universitat-pompeu-fabra")
    ap.add_argument("--wandb-project", default="marble")
    args = ap.parse_args()

    tasks = list(TASKS) if args.task == "both" else [args.task]
    out: list[str] = []
    payload: dict = {}
    csv_rows: list[list] = []
    for task in tasks:
        summarise_task(args, task, out, payload, csv_rows)

    md = "\n".join(out)
    print(md)
    with open(args.out_md, "w") as fh:
        fh.write(md)
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[wrote {args.out_json} and {args.out_md}]")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "layer", "abc_map", "mtf_map", "delta_abc_minus_mtf"])
            for r in csv_rows:
                w.writerow(r)
        print(f"[wrote {args.out_csv}]")


if __name__ == "__main__":
    main()
