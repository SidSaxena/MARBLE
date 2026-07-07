#!/usr/bin/env python3
"""Generate the top-k layer-subset weighted-probe matrix (MedleyDB, fold 0).

Tests whether the layer subsets identified by the HookTheory-learned gates
("the right layers", estimated on ~30x more data) transfer to MedleyDB, per
encoder. Five arms per encoder, all single-head ProbeAudioTask runs over the
EXISTING MedleyDB frame cache (the cache stores the raw pre-transform layer
stack keyed by encoder config only — see compute_config_hash — so every arm
is a config-only run, no re-extraction):

  htm-top3        LayerSelector(HTM top-3)  + LayerSoftmaxSum(3, learnable)
  htm-top5        LayerSelector(HTM top-5)  + LayerSoftmaxSum(5, learnable)
  own-top3        MedleyDB's own top-3 gates (control: native vs transferred)
  bottom3         HTM bottom-3 gates (negative control: does layer choice
                  matter at all under a trained head?)
  htm-top3-frozen HTM top-3 with the HTM gate VALUES frozen (renormalised,
                  learnable=False) — the pure gate-transfer arm; only the
                  MLP head trains.

Layer sets and gate values are READ from the committed campaign CSVs
(docs/figures/{hooktheory,medleydb}_melody_multihead/) — no hand-typed
numbers. Encoders present in the HookTheory gates CSV are generated; others
are skipped with a note (OMAR-RQ joins once its HTM run lands).

Usage (repo root):
    uv run python scripts/sweeps/gen_topk_configs.py --encoder MuQ
    uv run python scripts/sweeps/gen_topk_configs.py --encoder MERT-v1-95M

Writes ``configs/_topk_<encoder>_<arm>.yaml`` (underscore prefix = generated,
regenerable, not committed).
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import yaml

HTM_GATES = Path("docs/figures/hooktheory_melody_multihead/layer_gates.csv")
MED_GATES = Path("docs/figures/medleydb_melody_multihead/layer_gates_fold0.csv")


def load_gates(path: Path, encoder: str) -> dict[int, float]:
    g = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["encoder"] == encoder:
                g[int(r["layer"])] = float(r["gate"])
    if not g:
        raise SystemExit(f"[skip] no gates for {encoder} in {path} — run its campaign first")
    return g


def topk(gates: dict[int, float], k: int, reverse: bool = True) -> list[int]:
    return sorted(gates, key=gates.get, reverse=reverse)[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True)
    ap.add_argument(
        "--layers-config",
        default=None,
        help="base single-layer config (default: configs/probe.<encoder>-layers.MedleyDBMelody.yaml)",
    )
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    htm = load_gates(HTM_GATES, args.encoder)
    med = load_gates(MED_GATES, args.encoder)
    t3, t5 = topk(htm, 3), topk(htm, 5)
    arms = {
        "htm-top3": {"layers": t3, "learnable": True, "weights": None},
        "htm-top5": {"layers": t5, "learnable": True, "weights": None},
        "own-top3": {"layers": topk(med, 3), "learnable": True, "weights": None},
        "bottom3": {"layers": topk(htm, 3, reverse=False), "learnable": True, "weights": None},
        "htm-top3-frozen": {
            "layers": t3,
            "learnable": False,
            # renormalised over the subset; LayerSoftmaxSum re-normalises
            # defensively, but we pass the exact renormalised values so the
            # config documents them.
            "weights": [round(htm[L] / sum(htm[x] for x in t3), 6) for L in t3],
        },
    }

    src_path = args.layers_config or f"configs/probe.{args.encoder}-layers.MedleyDBMelody.yaml"
    src = yaml.safe_load(Path(src_path).read_text())

    for arm, spec in arms.items():
        c = copy.deepcopy(src)
        mi = c["model"]["init_args"]
        # transforms: LayerSelector -> arm layers; insert LayerSoftmaxSum after it
        tfs = mi["emb_transforms"]
        sel_idx = next(i for i, t in enumerate(tfs) if t["class_path"].endswith("LayerSelector"))
        tfs[sel_idx]["init_args"]["layers"] = spec["layers"]
        lss = {
            "class_path": "marble.modules.transforms.LayerSoftmaxSum",
            "init_args": {
                "num_layers": len(spec["layers"]),
                "normalize": True,
                "learnable": spec["learnable"],
            },
        }
        if spec["weights"] is not None:
            lss["init_args"]["init_weights"] = spec["weights"]
        tfs.insert(sel_idx + 1, lss)

        run_dir = f"./output/probe.MedleyDBMelody.{args.encoder}-topk-{arm}.fold{args.fold}/"
        for cb in c["trainer"]["callbacks"]:
            ia = cb.get("init_args", {})
            if "dirpath" in ia:
                ia["dirpath"] = run_dir + "checkpoints/"
        lg = c["trainer"]["logger"]["init_args"]
        lg["name"] = f"topk-{arm}"
        lg["save_dir"] = run_dir
        lg["tags"] = sorted(set(lg.get("tags", []) + ["topk"]))
        for split in ("train", "val", "test"):
            ia = c["data"]["init_args"][split]["init_args"]
            if "fold_idx" in ia:
                ia["fold_idx"] = args.fold

        p = Path("configs") / f"_topk_{args.encoder}_{arm}.yaml"
        with p.open("w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        w = spec["weights"] if spec["weights"] else "learned"
        print(f"wrote {p}  layers={spec['layers']} weights={w}")


if __name__ == "__main__":
    main()
