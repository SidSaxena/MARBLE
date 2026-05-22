"""Additional analyses on top of sms_clamp3_symbolic_report.json.

- 5-class collapse (treat linear+loop as a single body class)
- Per-piece accuracy distribution
- Disagreement matrix: ABC L4 vs MIDI L11 on the segments both modalities tested
- Confidence (logit margin) analysis: are high-confidence predictions more accurate?
- Per-class accuracy of "well-posed only" eval
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
with open(REPO / "output/sms_clamp3_symbolic_report.json") as f:
    report = json.load(f)

IDX2LABEL = ["bridge", "intro", "linear", "loop", "outro", "stinger"]

abc_best = report["abc"]["L4"]
mid_best = report["midi"]["L11"]


def macro_f1(per_class):
    return sum(c["f1"] for c in per_class) / len(per_class)


def acc(preds):
    return sum(1 for p in preds if p["true"] == p["pred"]) / len(preds)


def per_class_f1(preds, classes):
    """Compute per-class P/R/F1 manually."""
    out = []
    for c in classes:
        tp = sum(1 for p in preds if p["true"] == c and p["pred"] == c)
        fp = sum(1 for p in preds if p["true"] != c and p["pred"] == c)
        fn = sum(1 for p in preds if p["true"] == c and p["pred"] != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = sum(1 for p in preds if p["true"] == c)
        out.append({"label": c, "support": support, "precision": prec, "recall": rec, "f1": f1})
    return out


print("=" * 78)
print("§1. 5-CLASS COLLAPSE: treat linear+loop as a single 'body' class")
print("=" * 78)
print("Rationale: linear vs loop differs only by global piece-level context")
print("(does the surrounding music repeat?) — a per-segment classifier cannot")
print("distinguish them. Collapsing this distinction gives a fairer 5-class eval.\n")


def collapse_to_5(preds):
    out = []
    for p in preds:
        p2 = dict(p)
        if p2["true"] in ("linear", "loop"):
            p2["true"] = "body"
        if p2["pred"] in ("linear", "loop"):
            p2["pred"] = "body"
        out.append(p2)
    return out


for label, runs_dict, best_key in [
    ("ABC L4", report["abc"], "L4"),
    ("MIDI L11", report["midi"], "L11"),
]:
    preds = runs_dict[best_key]["predictions"]
    p5 = collapse_to_5(preds)
    classes = ["bridge", "intro", "body", "outro", "stinger"]
    pc = per_class_f1(p5, classes)
    a = acc(p5)
    mf = sum(c["f1"] for c in pc) / len(pc)
    print(f"  {label}: acc={a:.4f}, macro F1={mf:.4f}")
    print(f"    {'class':<10} {'support':<10} {'prec':<8} {'rec':<8} {'F1':<8}")
    for c in pc:
        print(
            f"    {c['label']:<10} {c['support']:<10} {c['precision']:<8.4f} {c['recall']:<8.4f} {c['f1']:<8.4f}"
        )
    print()

print("=" * 78)
print("§2. CONFIDENCE (LOGIT MARGIN) ANALYSIS")
print("=" * 78)
print("Margin = (top logit) - (2nd-top logit). High margin = confident prediction.\n")


def margin(p):
    s = sorted(p["logits"], reverse=True)
    return s[0] - s[1]


for label, runs_dict, best_key in [
    ("ABC L4", report["abc"], "L4"),
    ("MIDI L11", report["midi"], "L11"),
]:
    preds = runs_dict[best_key]["predictions"]
    # Bin by margin tertile
    margins = sorted([(margin(p), p["true"] == p["pred"]) for p in preds])
    n = len(margins)
    lo = margins[: n // 3]
    mid = margins[n // 3 : 2 * n // 3]
    hi = margins[2 * n // 3 :]

    def acc_of(bucket):
        return sum(1 for _, ok in bucket if ok) / len(bucket)

    print(f"  {label}: n={n}")
    print(
        f"    low-margin tertile  ({len(lo):>3}): margin {lo[0][0]:.2f}-{lo[-1][0]:.2f}  → acc {acc_of(lo):.4f}"
    )
    print(
        f"    mid-margin tertile  ({len(mid):>3}): margin {mid[0][0]:.2f}-{mid[-1][0]:.2f}  → acc {acc_of(mid):.4f}"
    )
    print(
        f"    high-margin tertile ({len(hi):>3}): margin {hi[0][0]:.2f}-{hi[-1][0]:.2f}  → acc {acc_of(hi):.4f}"
    )
    print()

print("=" * 78)
print("§3. ABC L4 vs MIDI L11 — DISAGREEMENT MATRIX on overlapping uids")
print("=" * 78)
print("How often does ABC L4 catch a segment that MIDI L11 misses, and vice versa?\n")

abc_by_uid = {p["uid"]: p for p in abc_best["predictions"]}
mid_by_uid = {p["uid"]: p for p in mid_best["predictions"]}
shared = set(abc_by_uid) & set(mid_by_uid)

# 2x2 matrix: ABC correct? × MIDI correct?
matrix = {("✓abc", "✓mid"): 0, ("✓abc", "✗mid"): 0, ("✗abc", "✓mid"): 0, ("✗abc", "✗mid"): 0}
abc_only_uids = []
mid_only_uids = []
for u in shared:
    a = abc_by_uid[u]
    m = mid_by_uid[u]
    a_ok = a["true"] == a["pred"]
    m_ok = m["true"] == m["pred"]
    key = (f"{'✓abc' if a_ok else '✗abc'}", f"{'✓mid' if m_ok else '✗mid'}")
    matrix[key] += 1
    if a_ok and not m_ok:
        abc_only_uids.append((u, a["true"], m["pred"]))
    if m_ok and not a_ok:
        mid_only_uids.append((u, m["true"], a["pred"]))

print(f"  Shared uids (in both ABC and MIDI test sets): {len(shared)}\n")
print(f"  {'':<14} {'MIDI correct':<14} {'MIDI wrong':<14}")
print(f"  {'ABC correct':<14} {matrix[('✓abc', '✓mid')]:<14} {matrix[('✓abc', '✗mid')]:<14}")
print(f"  {'ABC wrong':<14}   {matrix[('✗abc', '✓mid')]:<14} {matrix[('✗abc', '✗mid')]:<14}")
total = sum(matrix.values())
print(
    f"\n  Both correct:       {matrix[('✓abc', '✓mid')]}/{total} ({matrix[('✓abc', '✓mid')] / total * 100:.1f}%)"
)
print(
    f"  ABC only catches:   {matrix[('✓abc', '✗mid')]}/{total} ({matrix[('✓abc', '✗mid')] / total * 100:.1f}%) — ABC's contribution"
)
print(
    f"  MIDI only catches:  {matrix[('✗abc', '✓mid')]}/{total} ({matrix[('✗abc', '✓mid')] / total * 100:.1f}%) — MIDI's contribution"
)
print(
    f"  Both wrong:         {matrix[('✗abc', '✗mid')]}/{total} ({matrix[('✗abc', '✗mid')] / total * 100:.1f}%)"
)

print("\nClass distribution of 'ABC only catches' (segments ABC gets right, MIDI doesn't):")
abc_only_classes = Counter(t for _, t, _ in abc_only_uids)
for c in IDX2LABEL:
    if abc_only_classes.get(c):
        print(f"  {c:<10}: {abc_only_classes[c]} segments")

print("\nClass distribution of 'MIDI only catches' (segments MIDI gets right, ABC doesn't):")
mid_only_classes = Counter(t for _, t, _ in mid_only_uids)
for c in IDX2LABEL:
    if mid_only_classes.get(c):
        print(f"  {c:<10}: {mid_only_classes[c]} segments")

print("\n" + "=" * 78)
print("§4. PER-PIECE ACCURACY DISTRIBUTION")
print("=" * 78)
print("ori_uid format is <piece_id>_<seg_idx>. Group by piece, count hits.\n")

for label, preds in [("ABC L4", abc_best["predictions"]), ("MIDI L11", mid_best["predictions"])]:
    per_piece = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in preds:
        piece_id = p["uid"].rsplit("_", 1)[0]
        per_piece[piece_id]["total"] += 1
        if p["true"] == p["pred"]:
            per_piece[piece_id]["correct"] += 1
    pieces = list(per_piece.values())
    acc_dist = [pp["correct"] / pp["total"] for pp in pieces]
    perfect = sum(1 for a in acc_dist if a == 1.0)
    zero = sum(1 for a in acc_dist if a == 0.0)
    print(f"  {label}: {len(pieces)} pieces with test segments")
    print(
        f"    pieces with 100% accuracy: {perfect}/{len(pieces)} ({perfect / len(pieces) * 100:.1f}%)"
    )
    print(f"    pieces with 0% accuracy:   {zero}/{len(pieces)} ({zero / len(pieces) * 100:.1f}%)")
    print(f"    median per-piece acc:      {sorted(acc_dist)[len(acc_dist) // 2]:.4f}")
    print()
