#!/usr/bin/env python3
"""scripts/analysis/fix_wandb_runs.py — first-class W&B run operations.

A reusable CLI for the routine W&B run-management tasks this project needs.
Dry-run by default; pass ``--apply`` to write. Nothing is ever deleted.

Subcommands
-----------
  list      Show runs matching a selection (id, name, group, job_type, state,
            tags, whether they carry test/* metrics).
  set       Bulk-set name / group / job_type / tags on the selected runs
            (via ``run.update()`` — the proven, persistent path).
  archive   Mark the selected runs' group with a ``[archive]`` suffix and move
            them to the ``marble-archive`` project — ONE run at a time via the
            SCALAR moveRuns filter (see safety note). Supersedes ad-hoc archival.
  normalize Legacy one-off naming normalizer (historical fixes #1-#7); kept for
            reproducibility. See ``normalize --help``.

Selection flags (shared by list/set/archive)
--------------------------------------------
  --group GROUP            exact group match (e.g. "MuQ / VGMLoopStructure")
  --group-regex RE         regex over the group name
  --name-regex RE          regex over the run name (e.g. "^layer-(8|9)-")
  --state STATE            finished | running | crashed | failed | killed
  --layers 8 9 10          restrict to these layer indices (parsed from name/tags)
  --ids ID [ID ...]        operate on these exact run ids (skips the scan)

moveRuns SAFETY (read before touching archive)
----------------------------------------------
W&B officially supports moving runs only via the UI. The ``moveRuns`` GraphQL
mutation is internal and its ``filters`` JSONString does NOT behave like
``api.runs(filters=...)``. A ``{"$or": [...]}`` filter once over-matched ~1041
runs (ignored scope); ``{}`` drains the whole project. The ONLY safe form is a
scalar ``{"name": "<single_id>"}`` that moves exactly one run. This tool ALWAYS
moves one id at a time with that scalar form and refuses anything else, then
verifies that the source project lost exactly the moved ids and the archive
gained no extras. Moves are reversible (archive→marble); deletes are not — and
this tool never deletes.

Usage
-----
    # what runs match?
    uv run python scripts/analysis/fix_wandb_runs.py list \
        --group "MuQ / VGMLoopStructure" --layers 8 9 10 11 12 --state finished

    # set job_type on some runs
    uv run python scripts/analysis/fix_wandb_runs.py set \
        --ids abc123 def456 --job-type test --apply

    # archive the superseded MuQ 8-12 runs (dry-run first, then --apply)
    uv run python scripts/analysis/fix_wandb_runs.py archive \
        --ids abc123 def456 ... --apply

    # legacy normalizer
    uv run python scripts/analysis/fix_wandb_runs.py normalize --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

DEFAULT_ENTITY = "sidsaxena-universitat-pompeu-fabra"
DEFAULT_PROJECT = "marble"
ARCHIVE_PROJECT = "marble-archive"
ARCHIVE_SUFFIX = "[archive]"

KNOWN_MODEL_TAGS = {
    "MERT-v1-95M", "CLaMP3", "OMARRQ-multifeature25hz", "OMARRQ-multifeature-25hz",
    "OMARRQ-multifeature-25hz-fsq", "MuQ", "MusicFM", "DaSheng",
}


# ── shared helpers ──────────────────────────────────────────────────────────

def _has_test_metric(run) -> bool:
    # wandb's Summary iterates integer indices under `for k in summary`; use
    # .keys() to get string metric names. (ruff SIM118 autofix is wrong here.)
    return any(str(k).startswith("test/") for k in run.summary.keys())  # noqa: SIM118


def _layer_index(run) -> int | None:
    m = re.search(r"layer-?(\d+)", run.name or "")
    if m:
        return int(m.group(1))
    for t in run.tags or []:
        m = re.search(r"^layer-?(\d+)$", t)
        if m:
            return int(m.group(1))
    return None


def _infer_model(run) -> str | None:
    for t in run.tags or []:
        if t in KNOWN_MODEL_TAGS:
            return t
    return None


def _ensure_tag(tags: list[str], wanted: str) -> list[str]:
    return tags if wanted in tags else tags + [wanted]


def _remove_tags(tags: list[str], unwanted: set[str]) -> list[str]:
    return [t for t in tags if t not in unwanted]


def _planned_changes(run, *, new_name=None, new_group=None, new_job_type=None,
                     tag_add=(), tag_remove=()):
    """Changes that would actually flip the run's state (keeps re-runs idempotent)."""
    changes = []
    if new_name and run.name != new_name:
        changes.append(("name", new_name))
    if new_group is not None and run.group != new_group:
        changes.append(("group", new_group))
    if new_job_type and getattr(run, "job_type", None) != new_job_type:
        changes.append(("job_type", new_job_type))
    cur = list(run.tags or [])
    nt = _remove_tags(cur, set(tag_remove))
    for t in tag_add:
        nt = _ensure_tag(nt, t)
    if nt != cur:
        changes.append(("tags", nt))
    return changes


def _apply(run, *, new_name=None, new_group=None, new_job_type=None,
           tag_add=(), tag_remove=()):
    changes = _planned_changes(run, new_name=new_name, new_group=new_group,
                               new_job_type=new_job_type, tag_add=tag_add,
                               tag_remove=tag_remove)
    fields = []
    for field, value in changes:
        setattr(run, field, value)
        fields.append(f"{field}={value}")
    return fields


# ── run selection ───────────────────────────────────────────────────────────

def select_runs(api, proj, args):
    """Return the runs matching the shared selection flags on ``args``."""
    if getattr(args, "ids", None):
        out = []
        for rid in args.ids:
            try:
                out.append(api.run(f"{proj}/{rid}"))
            except Exception as e:  # noqa: BLE001
                print(f"  ! id {rid} not found: {e}", file=sys.stderr)
        return out
    runs = list(api.runs(proj, per_page=500))
    layers = set(args.layers) if getattr(args, "layers", None) else None
    sel = []
    for r in runs:
        if args.group and (r.group or "") != args.group:
            continue
        if args.group_regex and not re.search(args.group_regex, r.group or ""):
            continue
        if args.name_regex and not re.search(args.name_regex, r.name or ""):
            continue
        if args.state and r.state != args.state:
            continue
        if layers is not None and _layer_index(r) not in layers:
            continue
        sel.append(r)
    return sel


def _describe(r) -> str:
    jt = getattr(r, "job_type", None)
    return (f"{r.id}  state={r.state:<8} group={r.group!r}  name={r.name!r}  "
            f"job_type={jt!r}  test={_has_test_metric(r)}  tags={r.tags}")


# ── moveRuns (DANGEROUS — scalar, one id at a time, guarded) ─────────────────

def _gql():
    try:
        from wandb_gql import gql  # vendored by wandb
    except Exception:  # noqa: BLE001
        from gql import gql
    return gql


_MOVE_MUTATION = """
mutation MoveRuns($input: MoveRunsInput!) {
  moveRuns(input: $input) { clientMutationId }
}
"""


def _count_runs(api, proj) -> int:
    try:
        return len(list(api.runs(proj, per_page=500)))
    except Exception:  # noqa: BLE001
        return -1


def _move_one(api, entity, src_project, run_id, *, dest_project=ARCHIVE_PROJECT):
    """Move EXACTLY ONE run to ``dest_project`` via the scalar moveRuns filter.

    Hard-refuses any non-string / empty id so the filter can never become a
    bulk ``$or``/``$in``/``{}`` selector.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"refusing non-scalar run id: {run_id!r}")
    filters = json.dumps({"name": run_id})  # SCALAR — matches exactly one run
    inp = {
        "sourceEntityName": entity,
        "sourceProjectName": src_project,
        "destinationEntityName": entity,
        "destinationProjectName": dest_project,
        "filters": filters,
    }
    api.client.execute(_gql()(_MOVE_MUTATION), variable_values={"input": inp})


# ── subcommands ─────────────────────────────────────────────────────────────

def cmd_list(api, proj, args):
    runs = select_runs(api, proj, args)
    print(f"{len(runs)} run(s) in {proj}:\n")
    for r in sorted(runs, key=lambda r: (r.group or "", r.name or "")):
        print("  " + _describe(r))


def cmd_set(api, proj, args):
    runs = select_runs(api, proj, args)
    if not runs:
        print("No runs matched the selection.")
        return
    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {len(runs)} run(s) selected\n")
    n = 0
    for r in runs:
        planned = _planned_changes(
            r, new_name=args.name, new_group=args.group_to,
            new_job_type=args.job_type, tag_add=tuple(args.add_tag or ()),
            tag_remove=set(args.remove_tag or ()))
        if not planned:
            continue
        print("  " + _describe(r))
        print(f"      → {planned}")
        if args.apply:
            _apply(r, new_name=args.name, new_group=args.group_to,
                   new_job_type=args.job_type, tag_add=tuple(args.add_tag or ()),
                   tag_remove=set(args.remove_tag or ()))
            try:
                r.update()
                print("      ✓ written")
                n += 1
            except Exception as e:  # noqa: BLE001
                print(f"      ✗ write failed: {e}")
            time.sleep(args.sleep)
    print(f"\n{'Wrote' if args.apply else 'Would write'}: {n if args.apply else '—'}")


def cmd_archive(api, proj, args, entity, project):
    runs = select_runs(api, proj, args)
    if not runs:
        print("No runs matched the selection — nothing to archive.")
        return
    # Safety: archive needs an explicit, scoped selection.
    if not (args.ids or args.group or args.group_regex or args.name_regex
            or args.layers):
        print("Refusing to archive without a scoped selection "
              "(use --ids / --group / --name-regex / --layers).", file=sys.stderr)
        sys.exit(2)
    ids = [r.id for r in runs]
    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: archive {len(runs)} run(s) "
          f"({proj} → {entity}/{ARCHIVE_PROJECT})\n")
    for r in runs:
        print("  " + _describe(r))
    if not args.apply:
        print("\nRe-run with --apply to (1) suffix the group with "
              f"'{ARCHIVE_SUFFIX}' and (2) move each run individually.")
        return

    archive_proj = f"{entity}/{ARCHIVE_PROJECT}"
    arch_before = _count_runs(api, archive_proj)
    print(f"\nmarble-archive run count before: {arch_before}")

    # Step 1: mark group [archive] (reliable run.update()).
    print("\n[1/3] tagging groups with " + ARCHIVE_SUFFIX)
    for r in runs:
        g = (r.group or "").strip()
        if g.endswith(ARCHIVE_SUFFIX):
            continue
        newg = f"{g} {ARCHIVE_SUFFIX}".strip()
        r.group = newg
        try:
            r.update()
            print(f"  ✓ {r.id} group → {newg!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {r.id} group update failed: {e}")
        time.sleep(args.sleep)

    # Step 2: move each run ONE AT A TIME via the scalar filter.
    print("\n[2/3] moving runs to marble-archive (one id at a time)")
    moved = []
    for rid in ids:
        try:
            _move_one(api, entity, project, rid)
            moved.append(rid)
            print(f"  ✓ requested move: {rid}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ move failed for {rid}: {e}")
        time.sleep(max(args.sleep, 0.5))

    # Step 3: verify (moveRuns is async — poll the source for disappearance).
    print("\n[3/3] verifying (moveRuns is async; polling source project)…")
    deadline = args.verify_timeout
    waited = 0.0
    src_ids = set(ids)
    while waited < deadline:
        remaining = []
        for rid in moved:
            try:
                api.run(f"{proj}/{rid}")
                remaining.append(rid)
            except Exception:  # noqa: BLE001
                pass  # gone from source = moved
        if not remaining:
            break
        time.sleep(5)
        waited += 5
    arch_after = _count_runs(api, archive_proj)
    gained = arch_after - arch_before if arch_before >= 0 and arch_after >= 0 else None
    print(f"\nmarble-archive run count after: {arch_after}"
          + (f"  (gained {gained}, expected {len(moved)})" if gained is not None else ""))
    if gained is not None and gained > len(moved):
        print("  ⚠️  archive gained MORE runs than moved — investigate immediately.",
              file=sys.stderr)
    still = [rid for rid in moved if _run_exists(api, proj, rid)]
    if still:
        print(f"  (still resolving in source, async lag is normal): {still}")
    print("\nDone.")


def _run_exists(api, proj, rid) -> bool:
    try:
        api.run(f"{proj}/{rid}")
        return True
    except Exception:  # noqa: BLE001
        return False


def cmd_normalize(api, proj, args):
    """Legacy one-off naming normalizer (historical fixes #1-#7).

    Preserved verbatim in intent from the original fix_wandb_runs.py so a
    re-run is reproducible. Idempotent: prints "no change" for already-correct
    runs. See the module history for the full rationale of each fix.
    """
    runs = list(api.runs(proj, per_page=500))
    print(f"Loaded {len(runs)} runs from {proj}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}\n")
    MODAL_SHS100K_GROUPS = {"CLaMP3 / SHS100K", "MERT-v1-95M / SHS100K"}
    BROKEN_GROUPS = {"GTZANBeatTracking": None}
    NONFSQ = {"OMARRQ-multifeature-25hz-nonfsq": "OMARRQ-multifeature-25hz",
              "OMARRQ-multifeature-nonfsq": "OMARRQ-multifeature"}
    legacy_meanall_names = {"nonfsq-test", "base-test", "25hz-nonfsq-test", "variant-swap"}
    n_touched = n_skipped = 0
    for r in runs:
        group = r.group or ""
        name = r.name or ""
        intent: dict = {}
        if group in BROKEN_GROUPS:
            model = BROKEN_GROUPS[group] or _infer_model(r)
            if model is None:
                continue
            intent["new_group"] = f"{model} / {group}"
            intent.setdefault("tag_add", []).append(model)
        if group in MODAL_SHS100K_GROUPS:
            layer = _layer_index(r)
            if layer is not None and not (name.endswith("-fit") or name.endswith("-test")):
                kind = "test" if _has_test_metric(r) else "fit"
                intent["new_name"] = f"layer-{layer}-{kind}"
                intent.setdefault("tag_add", []).extend([kind, "modal"])
                intent["tag_remove"] = {"fit", "test"} - {kind}
        m = re.match(r"^OMARRQ-multifeature25hz(-meanall)? / (.+)$", group)
        if m:
            suffix = "-fsq-meanall" if m.group(1) else "-fsq"
            intent["new_group"] = f"OMARRQ-multifeature-25hz{suffix} / {m.group(2)}"
            intent.setdefault("tag_add", []).extend(["OMARRQ-multifeature-25hz-fsq", "fsq"])
            tr = intent.get("tag_remove", set())
            tr = tr if isinstance(tr, set) else set(tr)
            tr.add("OMARRQ-multifeature25hz")
            intent["tag_remove"] = tr
            if m.group(1):
                intent["tag_add"].append("mean-all")
        eff = intent.get("new_group", group)
        if " / " in eff:
            enc, task = eff.split(" / ", 1)
            if enc in NONFSQ:
                intent["new_group"] = f"{NONFSQ[enc]} / {task}"
                cur = set(r.tags or [])
                tr = intent.get("tag_remove", set())
                tr = tr if isinstance(tr, set) else set(tr)
                if enc in cur:
                    tr.add(enc)
                intent["tag_remove"] = tr
                if NONFSQ[enc] not in cur:
                    intent.setdefault("tag_add", []).append(NONFSQ[enc])
        cur = set(r.tags or [])
        is_meanall = bool(cur & {"mean-agg", "mean-all", "layer-meanall", "variant-swap"}) \
            or "meanall" in name or name in legacy_meanall_names
        eff = intent.get("new_group", group)
        if is_meanall and " / " in eff:
            enc, task = eff.split(" / ", 1)
            enc2 = enc.removesuffix("-meanall")
            if enc2 != enc:
                intent["new_group"] = f"{enc2} / {task}"
            kind = "test" if _has_test_metric(r) else "fit"
            if intent.get("new_name", name) != f"layer-meanall-{kind}":
                intent["new_name"] = f"layer-meanall-{kind}"
            intent.setdefault("tag_add", []).append("mean-all")
            tr = intent.get("tag_remove", set())
            tr = tr if isinstance(tr, set) else set(tr)
            intent["tag_remove"] = tr | {"single-layer", "mean-agg", "layer-meanall"}
        sp = group.split(" / ", 1)
        enc_part = sp[0] if len(sp) == 2 else group
        fam = next((f for f in ("OMARRQ", "MERT", "CLaMP3-symbolic", "CLaMP3",
                                "MuQ", "MusicFM", "DaSheng") if enc_part.startswith(f)), None)
        if fam and fam not in (r.tags or []):
            intent.setdefault("tag_add", []).append(fam)
        if "layer-sweep" not in (r.tags or []) and " / " in group:
            intent.setdefault("tag_add", []).append("layer-sweep")
        if not intent:
            n_skipped += 1
            continue
        planned = _planned_changes(
            r, new_name=intent.get("new_name"), new_group=intent.get("new_group"),
            tag_add=intent.get("tag_add", ()), tag_remove=intent.get("tag_remove", set()))
        if not planned:
            n_skipped += 1
            continue
        print(f"{r.id}  group={r.group!r} name={r.name!r} → {planned}")
        if args.apply:
            _apply(r, new_name=intent.get("new_name"), new_group=intent.get("new_group"),
                   tag_add=intent.get("tag_add", ()), tag_remove=intent.get("tag_remove", set()))
            try:
                r.update()
                n_touched += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ write failed: {e}")
            time.sleep(args.sleep)
    print(f"\n{'Touched' if args.apply else 'Would touch'}: {n_touched}  "
          f"Skipped: {n_skipped}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _add_selection(p):
    p.add_argument("--group")
    p.add_argument("--group-regex")
    p.add_argument("--name-regex")
    p.add_argument("--state")
    p.add_argument("--layers", type=int, nargs="*")
    p.add_argument("--ids", nargs="*")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--apply", action="store_true",
                    help="Write changes (default is dry-run).")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="Seconds between API writes (default 0.3).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="show matching runs")
    _add_selection(pl)

    ps = sub.add_parser("set", help="bulk-set name/group/job_type/tags")
    _add_selection(ps)
    ps.add_argument("--name")
    ps.add_argument("--group-to", help="set the group to this value")
    ps.add_argument("--job-type")
    ps.add_argument("--add-tag", nargs="*")
    ps.add_argument("--remove-tag", nargs="*")

    pa = sub.add_parser("archive", help="suffix group [archive] + move to marble-archive")
    _add_selection(pa)
    pa.add_argument("--verify-timeout", type=float, default=120.0,
                    help="Seconds to poll for async move completion (default 120).")

    sub.add_parser("normalize", help="legacy one-off naming normalizer")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    import wandb
    api = wandb.Api()
    proj = f"{args.entity}/{args.project}"
    if args.cmd == "list":
        cmd_list(api, proj, args)
    elif args.cmd == "set":
        cmd_set(api, proj, args)
    elif args.cmd == "archive":
        cmd_archive(api, proj, args, args.entity, args.project)
    elif args.cmd == "normalize":
        cmd_normalize(api, proj, args)


if __name__ == "__main__":
    main()
