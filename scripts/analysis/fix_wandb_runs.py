#!/usr/bin/env python3
"""scripts/analysis/fix_wandb_runs.py
──────────────────────────────────────
Retroactively normalize WandB runs that were created before the unified
naming convention landed. Dry-run by default — pass --apply to actually
write changes via the WandB API.

What it fixes
-------------
1. `GTZANBeatTracking` group (missing model prefix) → renamed to
   `MERT-v1-95M / GTZANBeatTracking` (model inferred from existing tags).

2. Modal SHS100K runs in `CLaMP3 / SHS100K` and `MERT-v1-95M / SHS100K`
   that have the bare name `layer-N` (no -fit/-test suffix) — renamed to
   `layer-N-fit` (no test/* keys in summary) or `layer-N-test` (has
   `test/*` keys), with matching `job_type` set via tags and the
   added `modal` source tag.

Skips runs that already follow the convention — idempotent re-run
prints "no change".

Usage
-----
    # Dry-run: print everything that would change, write nothing
    uv run python scripts/analysis/fix_wandb_runs.py

    # Apply changes
    uv run python scripts/analysis/fix_wandb_runs.py --apply

    # Operate on a specific WandB project / entity
    uv run python scripts/analysis/fix_wandb_runs.py \\
        --project marble --entity sidsaxena-universitat-pompeu-fabra --apply
"""

import argparse
import re
import sys
import time


# Groups we will touch — anything else is left strictly alone.
# Value (when set) overrides per-run model-from-tags inference; None means
# infer from each run's existing tags (whichever model tag is present).
BROKEN_GROUPS_NO_MODEL_PREFIX = {
    "GTZANBeatTracking": None,   # actually OMARRQ-multifeature25hz per tags
}
MODAL_SHS100K_GROUPS = {
    "CLaMP3 / SHS100K",
    "MERT-v1-95M / SHS100K",
}

# Known model tag values (used to pick the right one out of a run's tags).
KNOWN_MODEL_TAGS = {
    "MERT-v1-95M",
    "CLaMP3",
    "OMARRQ-multifeature25hz",
    "MuQ",
    "MusicFM",
    "DaSheng",
}


def _infer_model(run) -> str | None:
    """Pick the model tag from a run's tag list, if any."""
    for t in run.tags or []:
        if t in KNOWN_MODEL_TAGS:
            return t
    return None


def _has_test_metric(run) -> bool:
    return any(str(k).startswith("test/") for k in run.summary.keys())


def _layer_index(run) -> int | None:
    # Prefer the name; fall back to tags ("layer-N").
    m = re.search(r"layer-?(\d+)", run.name or "")
    if m:
        return int(m.group(1))
    for t in run.tags or []:
        m = re.search(r"^layer-?(\d+)$", t)
        if m:
            return int(m.group(1))
    return None


def _ensure_tag(tags: list[str], wanted: str) -> list[str]:
    if wanted in tags:
        return tags
    return tags + [wanted]


def _remove_tags(tags: list[str], unwanted: set[str]) -> list[str]:
    return [t for t in tags if t not in unwanted]


def _apply(run, *, new_name=None, new_group=None, tag_add=(), tag_remove=()):
    changed_fields = []
    if new_name and run.name != new_name:
        run.name = new_name
        changed_fields.append(f"name={new_name}")
    if new_group and run.group != new_group:
        run.group = new_group
        changed_fields.append(f"group={new_group}")
    cur_tags = list(run.tags or [])
    new_tags = _remove_tags(cur_tags, set(tag_remove))
    for t in tag_add:
        new_tags = _ensure_tag(new_tags, t)
    if new_tags != cur_tags:
        run.tags = new_tags
        changed_fields.append(f"tags+={list(tag_add)}")
    return changed_fields


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="marble")
    ap.add_argument("--entity", default=None,
                    help="WandB entity (default: authenticated user's default)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write changes. Without this, prints intended changes only.")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="Sleep (s) between API writes to avoid rate limits (default: 0.3)")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed.", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    proj = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(proj, per_page=500))
    print(f"Loaded {len(runs)} runs from {proj}")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}\n")

    n_touched = 0
    n_skipped = 0

    for r in runs:
        group = r.group or ""
        name = r.name or ""
        intent: dict = {}

        # ── Fix #1: orphan groups missing the model prefix ──────────────────
        if group in BROKEN_GROUPS_NO_MODEL_PREFIX:
            forced = BROKEN_GROUPS_NO_MODEL_PREFIX[group]
            model = forced or _infer_model(r)
            if model is None:
                print(f"  ! {r.id} in {group} — no model tag, skipping")
                continue
            intent["new_group"] = f"{model} / {group}"
            intent.setdefault("tag_add", []).append(model)

        # ── Fix #2: Modal SHS100K runs with bare `layer-N` name ─────────────
        if group in MODAL_SHS100K_GROUPS:
            layer = _layer_index(r)
            if layer is None:
                continue
            # If already has -fit/-test, this run is already fine.
            if not (name.endswith("-fit") or name.endswith("-test")):
                kind = "test" if _has_test_metric(r) else "fit"
                intent["new_name"] = f"layer-{layer}-{kind}"
                intent.setdefault("tag_add", []).append(kind)
                intent["tag_add"].append("modal")
                # Drop the conflicting opposite if present (defensive)
                intent["tag_remove"] = {"fit", "test"} - {kind}

        if not intent:
            n_skipped += 1
            continue

        # Print the planned change
        print(f"{r.id}  cur: group={r.group!r}  name={r.name!r}  tags={r.tags}")
        bits = []
        if "new_group" in intent: bits.append(f"group→{intent['new_group']!r}")
        if "new_name"  in intent: bits.append(f"name→{intent['new_name']!r}")
        if intent.get("tag_add"): bits.append(f"+tags={intent['tag_add']}")
        if intent.get("tag_remove"): bits.append(f"-tags={list(intent['tag_remove'])}")
        print(f"           {'   |   '.join(bits)}")

        if args.apply:
            changed = _apply(
                r,
                new_name=intent.get("new_name"),
                new_group=intent.get("new_group"),
                tag_add=intent.get("tag_add", ()),
                tag_remove=intent.get("tag_remove", set()),
            )
            if changed:
                try:
                    r.update()
                    print(f"           ✓ wrote: {changed}")
                    n_touched += 1
                except Exception as e:
                    print(f"           ✗ write failed: {e}")
                time.sleep(args.sleep)
        else:
            n_touched += 1   # counted as "would touch"
        print()

    print(f"\n── Summary ──")
    print(f"  {'Would touch' if not args.apply else 'Touched'}:  {n_touched}")
    print(f"  Skipped (already correct):  {n_skipped}")
    if not args.apply:
        print(f"\nRe-run with --apply to actually write the changes.")


if __name__ == "__main__":
    main()
