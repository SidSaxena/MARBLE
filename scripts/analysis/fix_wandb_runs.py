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

3. Legacy OMARRQ-multifeature25hz (un-hyphenated) groups → explicit
   `OMARRQ-multifeature-25hz-fsq / <task>` (the audit found that all
   historical OMARRQ-multifeature25hz runs were the -fsq variant).

4. Cross-cutting encoder-family tag (`OMARRQ`, `MERT`, `CLaMP3`,
   `CLaMP3-symbolic`, `MuQ`, `MusicFM`, `DaSheng`) added to every run
   for filterability, and `layer-sweep` added to every per-sweep run
   that was missing it (some early base configs forgot to include it).

5. Meanall runs live in the per-layer sweep group, NOT a sibling
   `<encoder>-meanall / <task>` group. Detected by tags/name signals
   (mean-all, mean-agg, layer-meanall, "meanall" in name, variant-swap,
   legacy variant-audit run names). For each match:
     - strip any `-meanall` suffix from the group  →  parent group
     - rename to `layer-meanall-<fit|test>` (kind from test/* keys)
     - ensure tag `mean-all`; drop the now-redundant `layer-meanall`
       (the run name already conveys layer-position) and legacy
       `mean-agg` + contradictory `single-layer`.
   The earlier convention put meanall in its own group; this fix
   migrates those runs back to live alongside layer-0..N-1.

6. The non-FSQ variant became the default for every OMARRQ family
   (FSQ keeps its explicit `-fsq` suffix), so any `-nonfsq` in the
   group/tag is redundant:
     - `OMARRQ-multifeature-25hz-nonfsq` → `OMARRQ-multifeature-25hz`
     - `OMARRQ-multifeature-nonfsq`      → `OMARRQ-multifeature`

7. Tag cleanup:
     - Drop legacy un-hyphenated `OMARRQ-multifeature25hz` slug from
       runs that already have the new hyphenated tag — having both is
       just sidebar noise.
     - Drop `single-layer` everywhere — the `layer-N` tag plus the
       absence of `mean-all` already conveys "single-layer".

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
    "GTZANBeatTracking": None,  # actually OMARRQ-multifeature25hz per tags
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
    "OMARRQ-multifeature25hz",  # legacy spelling (no hyphen)
    "OMARRQ-multifeature-25hz",  # new convention
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
    # NOTE: wandb's `Summary` object iterates over integer indices when
    # you `for k in run.summary` — you must call `.keys()` explicitly to
    # get string metric names. (ruff's SIM118 autofix gets this wrong.)
    return any(str(k).startswith("test/") for k in run.summary.keys())  # noqa: SIM118


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


def _planned_changes(run, *, new_name=None, new_group=None, tag_add=(), tag_remove=()):
    """Return [(field, value), ...] for changes that would actually flip
    a run's state. Used by both dry-run (to skip no-op intents) and
    _apply (so re-runs are idempotent end-to-end)."""
    changes = []
    if new_name and run.name != new_name:
        changes.append(("name", new_name))
    if new_group and run.group != new_group:
        changes.append(("group", new_group))
    cur_tags = list(run.tags or [])
    new_tags = _remove_tags(cur_tags, set(tag_remove))
    for t in tag_add:
        new_tags = _ensure_tag(new_tags, t)
    if new_tags != cur_tags:
        changes.append(("tags", new_tags))
    return changes


def _apply(run, *, new_name=None, new_group=None, tag_add=(), tag_remove=()):
    changes = _planned_changes(
        run,
        new_name=new_name,
        new_group=new_group,
        tag_add=tag_add,
        tag_remove=tag_remove,
    )
    fields = []
    for field, value in changes:
        setattr(run, field, value)
        fields.append(f"{field}={value}")
    return fields


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", default="marble")
    ap.add_argument(
        "--entity", default=None, help="WandB entity (default: authenticated user's default)"
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this, prints intended changes only.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep (s) between API writes to avoid rate limits (default: 0.3)",
    )
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
            # Tag both the new variant slug AND fsq for filterability.
            # Drop the legacy un-hyphenated slug `OMARRQ-multifeature25hz`
            # since the new hyphenated slug supersedes it (otherwise the
            # filter sidebar shows two entries for the same encoder).
            intent.setdefault("tag_add", []).extend(
                [
                    "OMARRQ-multifeature-25hz-fsq",
                    "fsq",
                ]
            )
            intent.setdefault("tag_remove", set())
            if not isinstance(intent["tag_remove"], set):
                intent["tag_remove"] = set(intent["tag_remove"])
            intent["tag_remove"].add("OMARRQ-multifeature25hz")
            # If the new group has -meanall, also tag the aggregation
            if "meanall" in legacy_rename:
                intent["tag_add"].append("mean-all")
            # Note: we no longer add `single-layer` to per-layer runs —
            # the `layer-N` tag + absence of `mean-all` already conveys
            # that this is a single-layer run.

        # ── Fix #6: drop redundant `-nonfsq` suffix from OMARRQ slugs ───────
        # The non-FSQ variant became the default for every OMARRQ family
        # (the FSQ variant retains its explicit `-fsq` suffix), so any
        # `-nonfsq` in the group name or encoder tag is redundant.
        NONFSQ_RENAMES = {
            "OMARRQ-multifeature-25hz-nonfsq": "OMARRQ-multifeature-25hz",
            "OMARRQ-multifeature-nonfsq": "OMARRQ-multifeature",
        }
        effective_group = intent.get("new_group", group)
        if " / " in effective_group:
            enc, task = effective_group.split(" / ", 1)
            if enc in NONFSQ_RENAMES:
                new_enc = NONFSQ_RENAMES[enc]
                intent["new_group"] = f"{new_enc} / {task}"
                # Replace the encoder tag if the old slug is on the run.
                cur_tags_set = set(r.tags or [])
                if enc in cur_tags_set:
                    intent.setdefault("tag_remove", set())
                    if not isinstance(intent["tag_remove"], set):
                        intent["tag_remove"] = set(intent["tag_remove"])
                    intent["tag_remove"].add(enc)
                if new_enc not in cur_tags_set:
                    intent.setdefault("tag_add", []).append(new_enc)

        # ── Fix #7: drop redundant / superseded tags everywhere ────────────
        # - `single-layer`: originally added as a complement to `mean-all`
        #   but the `layer-N` tag + absence of `mean-all` already conveys
        #   it. Pure noise.
        # - `OMARRQ-multifeature25hz` (un-hyphenated): legacy slug from
        #   before the variant audit. Runs that ALSO have the new
        #   hyphenated slug (`OMARRQ-multifeature-25hz` or
        #   `OMARRQ-multifeature-25hz-fsq`) shouldn't keep both — having
        #   two encoder tags pointing at the same run pollutes the
        #   sidebar filter list. Only drop the legacy tag if the new one
        #   is present, so we never strip without an unambiguous
        #   replacement.
        cur_tags_set = set(r.tags or [])
        new_hyphenated_present = (
            "OMARRQ-multifeature-25hz" in cur_tags_set
            or "OMARRQ-multifeature-25hz-fsq" in cur_tags_set
        )
        for stale in ("single-layer",):
            if stale in cur_tags_set:
                intent.setdefault("tag_remove", set())
                if not isinstance(intent["tag_remove"], set):
                    intent["tag_remove"] = set(intent["tag_remove"])
                intent["tag_remove"].add(stale)
        if "OMARRQ-multifeature25hz" in cur_tags_set and new_hyphenated_present:
            intent.setdefault("tag_remove", set())
            if not isinstance(intent["tag_remove"], set):
                intent["tag_remove"] = set(intent["tag_remove"])
            intent["tag_remove"].add("OMARRQ-multifeature25hz")

        # ── Fix #5: meanall runs live in the per-layer sweep group ──────────
        # New convention (2026-05-14): mean-of-all-layers is just another
        # aggregation choice for the same (encoder, task), so it belongs in
        # the SAME WandB group as the per-layer sweep — named
        # `layer-meanall-<fit|test>` alongside `layer-N-<fit|test>`. This
        # lets WandB's group view show the full comparison in one panel.
        #
        # Detection (signal-based; Lightning's WandbLogger doesn't expose
        # the nested `emb_transforms.init_args.mode` config):
        #   1. tag "mean-all" / "mean-agg" / "layer-meanall"
        #   2. name contains "meanall"           (e.g. meanall-fit, layer-meanall-test)
        #   3. tag "variant-swap"                (legacy single-shot meanall tests
        #      from the OMAR-RQ variant audit — all mean-of-all-layers per YAML)
        #   4. legacy bare names: nonfsq-test, base-test, 25hz-nonfsq-test
        #
        # Action (idempotent):
        #   - strip any trailing "-meanall" from group  →  parent group
        #   - rename to `layer-meanall-<fit|test>` (kind inferred from test/* keys)
        #   - add `mean-all` + `layer-meanall` tags; drop legacy `mean-agg`
        cur_tags = set(r.tags or [])
        legacy_meanall_names = {
            "nonfsq-test",
            "base-test",
            "25hz-nonfsq-test",
            "variant-swap",
        }
        is_meanall = (
            ("mean-agg" in cur_tags)
            or ("mean-all" in cur_tags)
            or ("layer-meanall" in cur_tags)
            or "meanall" in (name or "")
            or ("variant-swap" in cur_tags)
            or (name in legacy_meanall_names)
        )

        effective_group = intent.get("new_group", group)
        if is_meanall and " / " in effective_group:
            enc_part, task_part = effective_group.split(" / ", 1)
            # Move out of any `-meanall` group back to the parent encoder
            # group so meanall sits alongside the per-layer runs.
            enc_canonical = enc_part.removesuffix("-meanall")
            if enc_canonical != enc_part:
                intent["new_group"] = f"{enc_canonical} / {task_part}"

            # Normalize the run name to `layer-meanall-<fit|test>`.
            kind = "test" if _has_test_metric(r) else "fit"
            target_name = f"layer-meanall-{kind}"
            current_name = intent.get("new_name", name)
            if current_name != target_name:
                intent["new_name"] = target_name

            intent.setdefault("tag_add", []).append("mean-all")
            existing_remove = intent.get("tag_remove", set())
            if not isinstance(existing_remove, set):
                existing_remove = set(existing_remove)
            # Drop conflicting / superseded tags.
            intent["tag_remove"] = existing_remove | {
                "single-layer",  # contradicts mean-all
                "mean-agg",  # superseded by mean-all
                "layer-meanall",  # superseded by mean-all + the run name
            }

        # ── Fix #4: ensure encoder-family tag + `layer-sweep` are present ───
        # Family is the cross-cutting filter ("show me all OMARRQ runs"
        # across every variant). For single-variant encoders (MuQ, MusicFM,
        # DaSheng) family == encoder, so adding it is a no-op duplicate
        # rejected by the no-op filter downstream.
        sp = group.split(" / ", 1)
        enc_part = sp[0] if len(sp) == 2 else group
        family = None
        if enc_part.startswith("OMARRQ"):
            family = "OMARRQ"
        elif enc_part.startswith("MERT"):
            family = "MERT"
        elif enc_part.startswith("CLaMP3-symbolic"):
            family = "CLaMP3-symbolic"
        elif enc_part.startswith("CLaMP3"):
            family = "CLaMP3"
        elif enc_part.startswith("MuQ"):
            family = "MuQ"
        elif enc_part.startswith("MusicFM"):
            family = "MusicFM"
        elif enc_part.startswith("DaSheng"):
            family = "DaSheng"
        cur_tags_list = r.tags or []
        if family is not None and family not in cur_tags_list:
            intent.setdefault("tag_add", []).append(family)
        # `layer-sweep` was missing from some recent base configs; ensure it
        # ends up on every run that belongs to a per-layer or meanall sweep.
        if "layer-sweep" not in cur_tags_list and " / " in group:
            intent.setdefault("tag_add", []).append("layer-sweep")

        if not intent:
            n_skipped += 1
            continue

        # Filter out no-op intents (e.g. tag_add that's already on the run,
        # tag_remove of tags that aren't there). Keeps re-runs idempotent.
        planned = _planned_changes(
            r,
            new_name=intent.get("new_name"),
            new_group=intent.get("new_group"),
            tag_add=intent.get("tag_add", ()),
            tag_remove=intent.get("tag_remove", set()),
        )
        if not planned:
            n_skipped += 1
            continue

        # Print the planned change
        print(f"{r.id}  cur: group={r.group!r}  name={r.name!r}  tags={r.tags}")
        bits = []
        if "new_group" in intent:
            bits.append(f"group→{intent['new_group']!r}")
        if "new_name" in intent:
            bits.append(f"name→{intent['new_name']!r}")
        if intent.get("tag_add"):
            bits.append(f"+tags={intent['tag_add']}")
        if intent.get("tag_remove"):
            bits.append(f"-tags={list(intent['tag_remove'])}")
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
            n_touched += 1  # counted as "would touch"
        print()

    print("\n── Summary ──")
    print(f"  {'Would touch' if not args.apply else 'Touched'}:  {n_touched}")
    print(f"  Skipped (already correct):  {n_skipped}")
    if not args.apply:
        print("\nRe-run with --apply to actually write the changes.")


if __name__ == "__main__":
    main()
