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

# Per the 2026-05-14 variant audit, all historical OMARRQ-multifeature25hz
# runs used the -fsq variant (the original default). Rename retroactively
# so the variant is explicit in the group name; new runs going forward use
# OMARRQ-multifeature-25hz (hyphen, non-fsq is the new default).
LEGACY_OMARRQ_GROUP_RENAMES: dict[str, str] = {
    # Original sweep groups → explicit -fsq label
    # Group rewrites: replace OMARRQ-multifeature25hz (no hyphen) →
    # OMARRQ-multifeature-25hz-fsq (hyphenated + fsq-explicit).
    # Plus the meanall variants.
}
# Built programmatically below — covers all tasks dynamically.

# Known model tag values (used to pick the right one out of a run's tags).
KNOWN_MODEL_TAGS = {
    "MERT-v1-95M",
    "CLaMP3",
    "OMARRQ-multifeature25hz",       # legacy spelling (no hyphen)
    "OMARRQ-multifeature-25hz",      # new convention
    "OMARRQ-multifeature-25hz-fsq",  # explicit-fsq retro label
    "MuQ",
    "MusicFM",
    "DaSheng",
}


def _is_legacy_omarrq_group(group: str) -> str | None:
    """Detect legacy OMARRQ groups (un-hyphenated 25hz). Returns the new
    group name with hyphens AND -fsq variant suffix, or None if no rewrite."""
    # OMARRQ-multifeature25hz-meanall / X → OMARRQ-multifeature-25hz-fsq-meanall / X
    m = re.match(r"^OMARRQ-multifeature25hz-meanall / (.+)$", group)
    if m:
        return f"OMARRQ-multifeature-25hz-fsq-meanall / {m.group(1)}"
    # OMARRQ-multifeature25hz / X → OMARRQ-multifeature-25hz-fsq / X
    m = re.match(r"^OMARRQ-multifeature25hz / (.+)$", group)
    if m:
        return f"OMARRQ-multifeature-25hz-fsq / {m.group(1)}"
    return None


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

        # ── Fix #3: legacy OMARRQ-multifeature25hz → variant-explicit name ──
        legacy_rename = _is_legacy_omarrq_group(group)
        if legacy_rename is not None:
            intent["new_group"] = legacy_rename
            # Tag both the new variant slug AND fsq for filterability
            intent.setdefault("tag_add", []).extend([
                "OMARRQ-multifeature-25hz-fsq",
                "fsq",
            ])
            # If the new group has -meanall, also tag the aggregation
            if "meanall" in legacy_rename:
                intent["tag_add"].append("mean-all")
            else:
                intent["tag_add"].append("single-layer")

        # ── Fix #5: detect meanall runs and tag/rename them ─────────────────
        # Lightning's WandbLogger doesn't expose the nested emb_transforms
        # config, so we detect "meanall" runs by several signals:
        #   1. tag "mean-agg" or "mean-all"  (config-driven new convention)
        #   2. name contains "meanall"        (e.g. "meanall-fit", "meanall-test")
        #   3. tag "variant-swap"             (early single-shot meanall tests
        #      we ran during the OMAR-RQ variant audit — these were all
        #      mean-of-all-layers per their YAML)
        #   4. name in known legacy patterns: nonfsq-test, base-test,
        #      25hz-nonfsq-test (same OMAR-RQ variant audit)
        # For matches, ensure the group ends in "-meanall" to disambiguate
        # from the per-layer sweep group with the same encoder.
        cur_tags = set(r.tags or [])
        legacy_meanall_names = {
            "nonfsq-test", "base-test", "25hz-nonfsq-test", "variant-swap",
        }
        is_meanall = (
            ("mean-agg" in cur_tags) or ("mean-all" in cur_tags)
            or "meanall" in (name or "")
            or ("variant-swap" in cur_tags)
            or (name in legacy_meanall_names)
        )

        effective_group = intent.get("new_group", group)
        if is_meanall and " / " in effective_group:
            enc_part, task_part = effective_group.split(" / ", 1)
            if not enc_part.endswith("-meanall"):
                intent["new_group"] = f"{enc_part}-meanall / {task_part}"
            intent.setdefault("tag_add", []).append("mean-all")
            existing_remove = intent.get("tag_remove", set())
            if not isinstance(existing_remove, set):
                existing_remove = set(existing_remove)
            # Drop a stale "single-layer" tag if present (it'd contradict)
            intent["tag_remove"] = existing_remove | {"single-layer"}

        # ── Fix #4: ensure encoder-family tag is present on every run ───────
        # (cross-cutting filter for "show me all OMARRQ runs" etc.)
        family = None
        if "OMARRQ" in group:
            family = "OMARRQ"
        elif group.startswith("MERT"):
            family = "MERT"
        elif group.startswith("CLaMP3-symbolic"):
            family = "CLaMP3-symbolic"
        elif group.startswith("CLaMP3"):
            family = "CLaMP3"
        if family is not None and family not in (r.tags or []):
            intent.setdefault("tag_add", []).append(family)

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
