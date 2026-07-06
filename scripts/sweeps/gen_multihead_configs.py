#!/usr/bin/env python3
"""Generate multi-head MedleyDBMelody fold-configs for any encoder.

Takes the encoder's existing single-layer sweep config (the source of truth
for encoder / feature-extractor / datamodule settings) and applies the
VALIDATED multi-head delta from ``configs/probe.MuQ-multihead.MedleyDBMelody.yaml``
(fold-0 anchor validation 2026-07-06: worst |dRPA| = 0.0091, shape corr
0.996 — see docs/multihead_probe_validation.md):

  * task class -> ProbeAudioTaskMultiHead (+ its primary_metric arg)
  * LayerSelector -> ALL layers; decoder -> PerLayerHeads (same head
    architecture as the single-layer decoder, + meanall head)
  * callbacks -> multihead set (PerHeadBestCheckpoint after
    LoadLatestCheckpointCallback; NO EarlyStopping)
  * lr_scheduler -> ReduceLROnPlateau on val/acc_rpa_best
  * per-fold fold_idx / checkpoint dir / wandb run name

Usage (from the repo root):
    uv run python scripts/sweeps/gen_multihead_configs.py \\
        --encoder MERT-v1-95M --num-layers 13 --folds 0 1 2 3 4
    uv run python scripts/sweeps/gen_multihead_configs.py \\
        --encoder OMARRQ-multifeature-25hz --num-layers 24 --folds 0 1 2 3 4

Writes ``configs/_multihead_<encoder>_fold<F>.yaml`` (underscore prefix =
generated, regenerable, not committed).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

TEMPLATE = "configs/probe.MuQ-multihead.MedleyDBMelody.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", required=True, help="encoder slug, e.g. MERT-v1-95M")
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--weighted",
        action="store_true",
        help="add the SUPERB-style weighted-sum head (include_weighted) + "
        "LogLayerWeightsCallback; run dirs/names get a -weighted suffix. "
        "Run AFTER the plain protocol: the weighted head can drive the "
        "shared LR schedule (see docs/multihead_probe_validation.md).",
    )
    ap.add_argument(
        "--layers-config",
        default=None,
        help="source single-layer config (default: configs/probe.<encoder>-layers.MedleyDBMelody.yaml)",
    )
    args = ap.parse_args()

    src_path = args.layers_config or f"configs/probe.{args.encoder}-layers.MedleyDBMelody.yaml"
    src = yaml.safe_load(Path(src_path).read_text())
    tpl = yaml.safe_load(Path(TEMPLATE).read_text())
    L = args.num_layers

    cfg = copy.deepcopy(src)
    mi, ti = cfg["model"]["init_args"], tpl["model"]["init_args"]

    # task class + multihead-only init args from the template
    cfg["model"]["class_path"] = tpl["model"]["class_path"]
    if "primary_metric" in ti:
        mi["primary_metric"] = ti["primary_metric"]

    # LayerSelector -> all layers
    for t in mi["emb_transforms"]:
        if t["class_path"].endswith("LayerSelector"):
            t["init_args"]["layers"] = [f"0..{L - 1}"]

    # decoder -> PerLayerHeads, same head architecture as the single-layer one
    head = mi["decoders"][0]["init_args"]
    mi["decoders"] = [
        {
            "class_path": "marble.modules.decoders.PerLayerHeads",
            "init_args": {
                "in_dim": head["in_dim"],
                "out_dim": head["out_dim"],
                "hidden_layers": head.get("hidden_layers", [256]),
                "activation_fn": head.get("activation_fn"),
                "dropout": head.get("dropout", 0.2),
                "num_layers": L,
                "include_meanall": True,
                "include_weighted": bool(args.weighted),
            },
        }
    ]
    if args.weighted:
        cfg["trainer"]["callbacks"] = cfg["trainer"]["callbacks"] + [
            {"class_path": "marble.modules.callbacks.LogLayerWeightsCallback"}
        ]

    # callbacks + scheduler from the validated template
    cfg["trainer"]["callbacks"] = copy.deepcopy(tpl["trainer"]["callbacks"])
    if "lr_scheduler" in tpl:
        cfg["lr_scheduler"] = copy.deepcopy(tpl["lr_scheduler"])
    if "optimizer" in tpl and "optimizer" not in cfg:
        cfg["optimizer"] = copy.deepcopy(tpl["optimizer"])

    out_dir = Path("configs")
    for F in args.folds:
        c = copy.deepcopy(cfg)
        suffix = "-weighted" if args.weighted else ""
        run_dir = f"./output/probe.MedleyDBMelody.{args.encoder}-multihead{suffix}.fold{F}/"
        for cb in c["trainer"]["callbacks"]:
            ia = cb.get("init_args", {})
            if "dirpath" in ia:
                ia["dirpath"] = run_dir + "checkpoints/"
        lg = c["trainer"]["logger"]["init_args"]
        lg["name"] = f"multihead{suffix}-fold{F}"
        lg["save_dir"] = run_dir
        lg["tags"] = sorted(set(lg.get("tags", []) + ["multihead"]))
        for split in ("train", "val", "test"):
            c["data"]["init_args"][split]["init_args"]["fold_idx"] = F
        p = out_dir / f"_multihead{suffix}_{args.encoder}_fold{F}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
