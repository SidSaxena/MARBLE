"""SuperMarioStructure CLaMP3-symbolic ABC vs MIDI — full benchmark report.

Reads `test_predictions.json` (per-class P/R/F1 + confusion matrix + raw
predictions) from every sweep run dir and aggregates into a single report:
- Side-by-side acc + macro F1 table per layer (MIDI vs ABC)
- Per-class F1 matrix per layer × modality
- Full confusion matrices for best layer of each modality
- 4-class evaluation (excluding bridge + linear, which are not
  per-segment decidable) — see § 6 of the prior analysis for the rationale
- Bootstrap CIs on the best-layer test acc (95%, 1000 resamples)

Outputs:
- Markdown report on stdout
- JSON dump at output/sms_clamp3_symbolic_report.json (machine-readable)
"""

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "output"
IDX2LABEL = ["bridge", "intro", "linear", "loop", "outro", "stinger"]
LABEL2IDX = {l: i for i, l in enumerate(IDX2LABEL)}

# Classes that are per-segment-decidable. linear/loop differ only by global
# context (does the piece repeat the section?), which a single-segment
# classifier cannot see. bridge has support=4 in test — too few for a
# stable F1 estimate. Restricting to the well-posed subset gives a fairer
# headline.
WELL_POSED = {"intro", "loop", "outro", "stinger"}


def load_run(dir_path: Path) -> dict | None:
    p = dir_path / "test_predictions.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def macro_f1(per_class):
    f1s = [c["f1"] for c in per_class]
    return sum(f1s) / len(f1s)


def acc_from_preds(preds):
    n = len(preds)
    correct = sum(1 for p in preds if p["true"] == p["pred"])
    return correct / n if n else 0.0


def well_posed_acc_f1(preds):
    """Compute acc + macro F1 over the WELL_POSED subset only."""
    sub = [p for p in preds if p["true"] in WELL_POSED]
    if not sub:
        return None, None
    acc = sum(1 for p in sub if p["true"] == p["pred"]) / len(sub)
    classes = sorted(WELL_POSED)
    f1s = []
    for c in classes:
        tp = sum(1 for p in sub if p["true"] == c and p["pred"] == c)
        fp = sum(1 for p in sub if p["true"] != c and p["pred"] == c)
        fn = sum(1 for p in sub if p["true"] == c and p["pred"] != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return acc, sum(f1s) / len(f1s)


def bootstrap_acc_ci(preds, n_resamples=1000, alpha=0.05, seed=42):
    rng = random.Random(seed)
    n = len(preds)
    accs = []
    for _ in range(n_resamples):
        sample = [preds[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(1 for p in sample if p["true"] == p["pred"]) / n)
    accs.sort()
    lo = accs[int(n_resamples * alpha / 2)]
    hi = accs[int(n_resamples * (1 - alpha / 2))]
    return lo, hi


def collect(modality_tag: str) -> dict:
    """modality_tag = 'abc' for ABC sweep, '' for MIDI sweep."""
    suffix = f"-{modality_tag}" if modality_tag else ""
    out = {}
    # meanall
    d = OUTPUT / f"probe.SuperMarioStructure.CLaMP3-symbolic{suffix}-meanall"
    if d.exists():
        r = load_run(d)
        if r:
            out["meanall"] = r
    for L in range(13):
        d = OUTPUT / f"probe.SuperMarioStructure.CLaMP3-symbolic{suffix}-layers.layer{L}"
        r = load_run(d)
        if r:
            out[f"L{L}"] = r
    return out


def main():
    abc = collect("abc")
    mid = collect("")
    if not abc and not mid:
        print("No test_predictions.json found in any sweep dir yet — runs still in progress?")
        return

    keys = ["meanall"] + [f"L{i}" for i in range(13)]

    # ── headline table ────────────────────────────────────────────────────
    print("# SuperMarioStructure × CLaMP3-symbolic — ABC vs MIDI benchmark report\n")
    print("Test set: 178 ABC clips / 182 MIDI clips. 6-class VGM functional segments.")
    print("Patched probe logs per-class metrics + confusion matrix on every test run.")
    print(f"Sweeps available: ABC ({len(abc)}/14 runs), MIDI ({len(mid)}/14 runs).\n")

    print("## 1. Headline acc + macro F1, per layer\n")
    print("| Config | MIDI acc | MIDI F1 | ABC acc | ABC F1 | Δacc | ΔF1 |")
    print("|---|---|---|---|---|---|---|")
    for k in keys:
        ma = mf = aa = af = None
        if k in mid:
            r = mid[k]
            ma = acc_from_preds(r["predictions"])
            mf = macro_f1(r["per_class"])
        if k in abc:
            r = abc[k]
            aa = acc_from_preds(r["predictions"])
            af = macro_f1(r["per_class"])
        cells = [
            k,
            f"{ma:.4f}" if ma is not None else "—",
            f"{mf:.4f}" if mf is not None else "—",
            f"{aa:.4f}" if aa is not None else "—",
            f"{af:.4f}" if af is not None else "—",
            f"{aa - ma:+.4f}" if (aa is not None and ma is not None) else "—",
            f"{af - mf:+.4f}" if (af is not None and mf is not None) else "—",
        ]
        print("| " + " | ".join(cells) + " |")

    # ── per-class F1 matrix, ABC ──────────────────────────────────────────
    print("\n## 2. ABC — per-class F1 by layer\n")
    header = ["config", "support→"] + IDX2LABEL + ["macro"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for k in keys:
        if k not in abc:
            continue
        r = abc[k]
        pc = {row["label"]: row for row in r["per_class"]}
        row_cells = (
            [k, ""]
            + [f"{pc[l]['f1']:.2f}" for l in IDX2LABEL]
            + [f"{macro_f1(r['per_class']):.3f}"]
        )
        print("| " + " | ".join(row_cells) + " |")
    if abc:
        supports = {row["label"]: row["support"] for row in next(iter(abc.values()))["per_class"]}
        print("\nSupport (ABC test set):  " + ", ".join(f"{l}={supports[l]}" for l in IDX2LABEL))

    # ── per-class F1 matrix, MIDI ─────────────────────────────────────────
    print("\n## 3. MIDI — per-class F1 by layer\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for k in keys:
        if k not in mid:
            continue
        r = mid[k]
        pc = {row["label"]: row for row in r["per_class"]}
        row_cells = (
            [k, ""]
            + [f"{pc[l]['f1']:.2f}" for l in IDX2LABEL]
            + [f"{macro_f1(r['per_class']):.3f}"]
        )
        print("| " + " | ".join(row_cells) + " |")
    if mid:
        supports = {row["label"]: row["support"] for row in next(iter(mid.values()))["per_class"]}
        print("\nSupport (MIDI test set): " + ", ".join(f"{l}={supports[l]}" for l in IDX2LABEL))

    # ── best layer per modality, full detail ──────────────────────────────
    for modality, runs in (("ABC", abc), ("MIDI", mid)):
        if not runs:
            continue
        best_key = max(runs.keys(), key=lambda k: macro_f1(runs[k]["per_class"]))
        r = runs[best_key]
        acc = acc_from_preds(r["predictions"])
        f1 = macro_f1(r["per_class"])
        lo, hi = bootstrap_acc_ci(r["predictions"])
        print(f"\n## 4-{modality}. Best {modality} layer: {best_key}")
        print(f"\n- Test acc: **{acc:.4f}** (95 % bootstrap CI [{lo:.4f}, {hi:.4f}])")
        print(f"- Macro F1: **{f1:.4f}**")
        wp_acc, wp_f1 = well_posed_acc_f1(r["predictions"])
        if wp_acc is not None:
            print("- Well-posed 4-class subset (intro+loop+outro+stinger only):")
            print(f"  - acc: **{wp_acc:.4f}** | macro F1: **{wp_f1:.4f}**")

        print(f"\nPer-class breakdown ({best_key}):\n")
        print("| class | support | precision | recall | F1 |")
        print("|---|---|---|---|---|")
        for c in r["per_class"]:
            print(
                f"| {c['label']} | {c['support']} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1']:.4f} |"
            )

        print(f"\nConfusion matrix ({best_key}, rows = true, cols = pred):\n")
        print("| true \\ pred | " + " | ".join(IDX2LABEL) + " |")
        print("|" + "|".join("---" for _ in range(len(IDX2LABEL) + 1)) + "|")
        for i, row in enumerate(r["confusion"]):
            cells = [IDX2LABEL[i]] + [str(v) for v in row]
            print("| " + " | ".join(cells) + " |")

    # ── machine-readable dump ─────────────────────────────────────────────
    out_json = OUTPUT / "sms_clamp3_symbolic_report.json"
    with open(out_json, "w") as f:
        json.dump({"abc": abc, "midi": mid}, f, indent=2)
    print(f"\n---\nMachine-readable dump: {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
