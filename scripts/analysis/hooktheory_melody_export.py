#!/usr/bin/env python3
"""Export HookTheoryMelody multi-head+weighted results to committed CSVs for
the thesis figure generator. Per encoder pulls per-layer RPA+RCA (+meanall,
weighted) from the *fixed* test run (name 'multihead-weighted-test-fix', post
compute_groups fix 4e24707) and the learned softmax gates from the weighted
head checkpoint. Skips encoders whose -fix run or checkpoint is absent, so it
can be re-run as each encoder's retest lands.

Writes docs/figures/hooktheory_melody_multihead/{layer_curves.csv,layer_gates.csv}.
Run on the PC (needs wandb creds + checkpoints). Idempotent / append-safe:
rewrites both CSVs from whatever encoders are currently available.
"""

import csv
import os
from pathlib import Path

import torch
import wandb

ENT = "sidsaxena-universitat-pompeu-fabra"
ENCS = [("MuQ", 13), ("MERT-v1-95M", 13), ("OMARRQ-multifeature-25hz", 24)]
OUT = Path("docs/figures/hooktheory_melody_multihead")
OUT.mkdir(parents=True, exist_ok=True)
api = wandb.Api()

curves_rows, gate_rows = [], []
for enc, nl in ENCS:
    runs = api.runs(f"{ENT}/marble", filters={"group": f"{enc} / HookTheoryMelody"})
    fix = sorted(
        [r for r in runs if (r.name or "") == "multihead-weighted-test-fix"],
        key=lambda r: r.created_at,
    )
    ck = f"output/probe.HookTheoryMelody.{enc}-multihead-weighted.fold0/checkpoints/best.ckpt"
    if not fix or not os.path.exists(ck):
        print(f"[skip] {enc}: fix_run={bool(fix)} ckpt={os.path.exists(ck)}")
        continue
    s = dict(fix[-1].summary)
    for L in list(range(nl)) + ["meanall", "weighted"]:
        rp, rc = (
            s.get(f"test/acc_rpa_l{L}" if isinstance(L, int) else f"test/acc_rpa_{L}"),
            s.get(f"test/acc_rca_l{L}" if isinstance(L, int) else f"test/acc_rca_{L}"),
        )
        if rp is None:
            continue
        curves_rows.append([enc, L, f"{rp:.5f}", f"{rc:.5f}"])
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    gk = [k for k in sd if k.endswith("layer_gate")][0]
    g = torch.softmax(sd[gk].float(), dim=0).numpy()
    for L, v in enumerate(g):
        gate_rows.append([enc, L, f"{v:.5f}"])
    print(f"[ok]   {enc}: {nl} layers + meanall/weighted + gates")

with open(OUT / "layer_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoder", "layer", "rpa", "rca"])
    w.writerows(curves_rows)
with open(OUT / "layer_gates.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["encoder", "layer", "gate"])
    w.writerows(gate_rows)
encs = sorted({r[0] for r in curves_rows})
print(f"wrote CSVs for encoders: {encs}")
