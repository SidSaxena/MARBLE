#!/usr/bin/env python3
"""scripts/embeddings/audit_cache_integration.py
─────────────────────────────────────────────────
Static audit of every config with ``cache_embeddings: true``.
Verifies the entire cache plumbing chain — task class, dataset
class, and integration markers — instead of relying on indirect
wall-clock signals.

Why this exists
---------------
The HookTheoryStructure / HookTheoryKey datasets were silently
cache-no-op'd for two weeks because:
  - their custom Dataset subclasses didn't emit the 4-tuple
    (waveform, label, path, clip_id)
  - the task probe accepted ``cache_embeddings: true`` and built
    the cache directory + ``_meta.json`` at setup time
  - but ``_cached_forward_layer_tuple`` short-circuited to
    ``use_cache=False`` at runtime because ``clip_ids`` was None
The bug only manifested as "the cache dir has 0 .pt files" — easily
mistaken for "the sweep is in progress, give it time."

This script catches that class of failure pre-flight. Run it any
time you add a new cache-enabled config OR refactor the cache
plumbing.

What it checks (per cache-enabled config)
-----------------------------------------
1. Task class:
     - Resolves cleanly from model.class_path
     - Accepts ``cache_embeddings`` in __init__ (or inherits a class
       that does)
     - Inherits ``EmbeddingCacheMixin`` (directly or transitively)
2. Dataset classes (train / val / test):
     - Resolve cleanly from data.init_args.{train,val,test}.class_path
     - EITHER inherits ``BaseAudioDataset`` (4-tuple emit comes for free)
     - OR has all three explicit-integration markers in the source:
         a. ``from marble.utils.emb_cache import make_clip_id``
         b. ``clip_id`` in at least one return statement
         c. ``cache_check_fn`` attribute set in __init__
3. Lightweight runtime probe (best-effort, no audio loading):
     - Instantiate the dataset with `__init__` if all required JSONL
       files exist on disk
     - Pull one item, verify it's a 4-tuple, verify the last
       element looks like a clip_id (regex match)
     - Skipped silently if data isn't present (so this script runs
       on dev machines without datasets)

Output
------
- ✓ for fully verified configs
- ✗ for failures, with the specific marker that's missing

Exit code 0 if all configs pass, 1 if any fail.

Usage
-----
::

    uv run python scripts/embeddings/audit_cache_integration.py

    # Verbose mode — show what was checked for every config
    uv run python scripts/embeddings/audit_cache_integration.py -v

    # Only audit specific configs
    uv run python scripts/embeddings/audit_cache_integration.py \\
        --filter 'HookTheory'
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import sys
import textwrap
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import yaml

# Required markers in dataset source for explicit (non-inherited) integration:
MARKER_IMPORT = "from marble.utils.emb_cache import make_clip_id"
MARKER_CLIP_ID_RETURN_RE = re.compile(
    r"return\s+\(?[^,()]+,\s*[^,()]+,\s*[^,()]+,\s*clip_id\b"
)
MARKER_CACHE_CHECK_FN_RE = re.compile(r"self\.cache_check_fn\s*=\s*None")

CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_.+\-]+__[0-9a-f]{8}__c\d+$")


@dataclass
class CheckResult:
    config: Path
    task_class: str | None = None
    dataset_classes: list[str] = field(default_factory=list)
    task_ok: bool = False
    task_reason: str = ""
    datasets_ok: bool = False
    datasets_reason: str = ""
    runtime_ok: bool | None = None  # None = skipped (no data)
    runtime_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.task_ok and self.datasets_ok and (self.runtime_ok is not False)


# ──────────────────────────────────────────────────────────────────────────
# Static checks
# ──────────────────────────────────────────────────────────────────────────


def _resolve_class(class_path: str):
    """Resolve a 'module.path.ClassName' string to the actual class."""
    if not class_path or "." not in class_path:
        return None
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except Exception:
        return None


def _check_task_class(class_path: str) -> tuple[bool, str]:
    """Verify the task class is cache-aware end-to-end:

    1. Resolves
    2. __init__ accepts ``cache_embeddings`` (in own class or via super)
    3. Inherits ``EmbeddingCacheMixin``
    4. If it overrides ``forward()``, the override calls
       ``_cached_forward_layer_tuple`` — otherwise the cache hit/miss
       branches are bypassed even if everything else is wired up.
    5. If it overrides ``_shared_step``/``training_step``/``validation_step``/
       ``test_step``, those methods pass ``clip_ids`` to ``self(...)``.

    Items 4 and 5 are exactly the failure modes I missed for
    HookTheoryStructure (a custom dataset broke 5; if I'd later
    overridden forward(), it would have broken 4 too).
    """
    cls = _resolve_class(class_path)
    if cls is None:
        return False, f"task class {class_path!r} did not resolve"

    # (1)–(3) constructor + inheritance
    for c in cls.__mro__:
        if "cache_embeddings" in inspect.signature(c.__init__).parameters:
            break
    else:
        return False, f"no `cache_embeddings` kwarg in {class_path}.__init__ chain"

    try:
        from marble.utils.emb_cache import EmbeddingCacheMixin
    except ImportError:
        return False, "couldn't import EmbeddingCacheMixin (broken install?)"

    if not issubclass(cls, EmbeddingCacheMixin):
        return False, f"{class_path} does not inherit EmbeddingCacheMixin"

    # (4) If forward is overridden, it must actually CALL
    # `_cached_forward_layer_tuple` (not just mention it in a docstring/
    # comment). AST analysis avoids the substring-match false-positive.
    own_forward = cls.__dict__.get("forward")
    if own_forward is not None and not _calls_method(own_forward, "_cached_forward_layer_tuple"):
        return False, (
            f"{class_path}.forward is overridden but doesn't CALL "
            "`_cached_forward_layer_tuple` — cache hit/miss branches are "
            "bypassed at runtime"
        )

    # (5) Custom step methods that actually consume `batch` must also
    # propagate `clip_ids`. No-op overrides (e.g. retrieval task's
    # `training_step: return None`) are intentionally exempted — they're
    # the documented "max_epochs=0, this never runs" pattern.
    for step_name in ("_shared_step", "training_step", "validation_step", "test_step"):
        own_step = cls.__dict__.get(step_name)
        if own_step is None:
            continue  # inherits BaseTask's, which is correct
        if not _references_name(own_step, "batch"):
            continue  # no-op override; doesn't read the batch, so cache is irrelevant
        if not _references_name(own_step, "clip_ids"):
            return False, (
                f"{class_path}.{step_name} unpacks `batch` but doesn't "
                "reference `clip_ids` — cache would silently no-op when "
                "this step runs"
            )

    return True, ""


# ──────────────────────────────────────────────────────────────────────────
# AST helpers — substring matches in source give false positives when the
# marker appears in docstrings/comments, so we walk the AST instead.
# ──────────────────────────────────────────────────────────────────────────


def _parse_method(method) -> ast.AST | None:
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        return None
    # The source includes the def's indentation; dedent so ast.parse accepts it.
    try:
        return ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None


def _calls_method(method, attr_name: str) -> bool:
    """True iff the method's AST contains a Call whose function ends in
    `.<attr_name>` (e.g. `self._cached_forward_layer_tuple(...)`)."""
    tree = _parse_method(method)
    if tree is None:
        return False
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == attr_name:
                found = True
                break
    return found


def _references_name(method, name: str) -> bool:
    """True iff the method's BODY (not signature) references the given
    identifier. Walks only the function body so that ``def step(self, batch,
    batch_idx): return None`` doesn't count as "uses batch" — that's the
    canonical no-op override pattern for retrieval tasks where validation
    / training never run.

    Catches: bare identifiers (``batch``, ``clip_ids``), kwargs
    (``self(x, clip_ids=...)``), attribute access (``batch[0]``).
    """
    tree = _parse_method(method)
    if tree is None:
        return False
    # `tree` is a Module with one FunctionDef (the dedented method). Walk
    # just the body's statements, skipping the args list entirely.
    funcdefs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcdefs:
        return False
    fn = funcdefs[0]
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == name:
                return True
            if isinstance(node, ast.keyword) and node.arg == name:
                return True
    return False


def _check_dataset_class(class_path: str) -> tuple[bool, str]:
    """Verify the dataset class emits a 4-tuple with clip_id."""
    cls = _resolve_class(class_path)
    if cls is None:
        return False, f"dataset class {class_path!r} did not resolve"

    # Path A: inherits BaseAudioDataset (which we've fixed once for the whole
    # tree)
    try:
        from marble.core.base_datamodule import BaseAudioDataset
    except ImportError:
        return False, "couldn't import BaseAudioDataset"

    if issubclass(cls, BaseAudioDataset):
        return True, "(via BaseAudioDataset)"

    # Path B: explicit integration — check source for all three markers
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return False, f"couldn't read source of {class_path}"

    # Walk parent classes too — markers might be in a base class within the
    # same file (e.g. `_HookTheoryStructureAudioBase` is the base; its
    # subclasses just `pass`).
    sources = [src]
    for base in cls.__mro__[1:]:
        if base.__module__ == cls.__module__:
            try:
                sources.append(inspect.getsource(base))
            except (OSError, TypeError):
                pass
    full_source = "\n".join(sources)
    # Also pull the entire module file as a last resort (some bases are
    # defined at module level under unusual names):
    try:
        module_src = inspect.getsource(sys.modules[cls.__module__])
        full_source = module_src
    except (OSError, TypeError, KeyError):
        pass

    missing = []
    if MARKER_IMPORT not in full_source:
        missing.append("missing `from marble.utils.emb_cache import make_clip_id`")
    if not MARKER_CLIP_ID_RETURN_RE.search(full_source):
        missing.append("no `return ..., clip_id` 4-tuple in __getitem__")
    if not MARKER_CACHE_CHECK_FN_RE.search(full_source):
        missing.append("no `self.cache_check_fn = None` in __init__")

    if missing:
        return False, "; ".join(missing)
    return True, "(explicit integration)"


# ──────────────────────────────────────────────────────────────────────────
# Lightweight runtime probe
# ──────────────────────────────────────────────────────────────────────────


def _runtime_probe(cfg: dict, split: str = "test") -> tuple[bool | None, str]:
    """Best-effort: instantiate the dataset for ``split`` and pull one item.
    Returns (None, reason) if data isn't present so we can run on dev
    machines without datasets."""
    try:
        ds_cfg = cfg["data"]["init_args"][split]
    except KeyError:
        return None, f"no data.init_args.{split} in config"

    init_args = ds_cfg.get("init_args", {})
    jsonl_path = init_args.get("jsonl")
    if jsonl_path is None:
        return None, "no jsonl path in init_args (skipping runtime probe)"
    if not Path(jsonl_path).exists():
        return None, f"jsonl not present: {jsonl_path}"

    cls = _resolve_class(ds_cfg.get("class_path", ""))
    if cls is None:
        return False, f"class {ds_cfg.get('class_path')!r} didn't resolve"

    try:
        ds = cls(**init_args)
    except Exception as e:
        return False, f"instantiation raised {type(e).__name__}: {e}"

    if len(ds) == 0:
        return None, "dataset is empty (no clips in jsonl?)"

    try:
        item = ds[0]
    except Exception as e:
        return False, f"ds[0] raised {type(e).__name__}: {e}"

    if not isinstance(item, (tuple, list)):
        return False, f"ds[0] returned {type(item).__name__}, expected tuple/list"
    if len(item) != 4:
        return False, f"ds[0] returned {len(item)}-tuple, expected 4-tuple"

    clip_id = item[3]
    if not isinstance(clip_id, str):
        return False, f"clip_id (4th element) is {type(clip_id).__name__}, expected str"
    if not CLIP_ID_RE.match(clip_id):
        return False, f"clip_id {clip_id!r} doesn't match expected format"

    return True, f"ds[0] returns 4-tuple, clip_id={clip_id!r}"


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────


def _has_cache_enabled(cfg: dict) -> bool:
    try:
        return bool(cfg["model"]["init_args"]["cache_embeddings"])
    except (KeyError, TypeError):
        return False


def audit_one(config_path: Path, runtime: bool = False) -> CheckResult:
    cfg = yaml.safe_load(config_path.read_text())
    result = CheckResult(config=config_path)

    if not _has_cache_enabled(cfg):
        return result  # Not opted-in; not in audit scope

    task_class = cfg.get("model", {}).get("class_path", "")
    result.task_class = task_class

    # Task class check
    result.task_ok, result.task_reason = _check_task_class(task_class)

    # Dataset class checks (train + val + test)
    dataset_paths = []
    for split in ("train", "val", "test"):
        try:
            cp = cfg["data"]["init_args"][split]["class_path"]
            dataset_paths.append(cp)
        except (KeyError, TypeError):
            continue
    result.dataset_classes = sorted(set(dataset_paths))

    if not result.dataset_classes:
        result.datasets_ok = False
        result.datasets_reason = "no dataset class_paths found in config"
    else:
        problems = []
        for cp in result.dataset_classes:
            ok, reason = _check_dataset_class(cp)
            if not ok:
                problems.append(f"{cp}: {reason}")
        if problems:
            result.datasets_ok = False
            result.datasets_reason = " | ".join(problems)
        else:
            result.datasets_ok = True
            result.datasets_reason = "(all dataset classes pass)"

    # Optional runtime probe
    if runtime:
        result.runtime_ok, result.runtime_reason = _runtime_probe(cfg, "test")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--filter", default="", help="Substring filter on config filename."
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show one line per audited config (default: only failures + summary)."
    )
    ap.add_argument(
        "--runtime", action="store_true",
        help="Also try to instantiate the test dataset and pull one item "
        "(requires data on disk; falls back gracefully if missing).",
    )
    args = ap.parse_args()

    cache_enabled_configs = []
    for p in sorted(Path("configs").glob("probe.*.yaml")):
        try:
            cfg = yaml.safe_load(p.read_text())
            if _has_cache_enabled(cfg):
                if not args.filter or args.filter.lower() in p.name.lower():
                    cache_enabled_configs.append(p)
        except Exception:
            pass

    print(f"Auditing {len(cache_enabled_configs)} cache-enabled configs...")
    if args.filter:
        print(f"  (filter: {args.filter!r})")
    print()

    failures: list[CheckResult] = []
    by_task: dict[str, list[CheckResult]] = {}

    for cfg_path in cache_enabled_configs:
        result = audit_one(cfg_path, runtime=args.runtime)
        task = result.task_class or "(unknown)"
        by_task.setdefault(task, []).append(result)
        if not result.ok:
            failures.append(result)
        if args.verbose:
            mark = "✓" if result.ok else "✗"
            print(f"  {mark}  {cfg_path.name}")
            if not result.ok:
                if not result.task_ok:
                    print(f"       TASK: {result.task_reason}")
                if not result.datasets_ok:
                    print(f"       DATA: {result.datasets_reason}")
                if result.runtime_ok is False:
                    print(f"       RUNTIME: {result.runtime_reason}")

    # ── Failure summary ─────────────────────────────────────────────────
    if failures:
        print("\n── FAILURES ──")
        # Group failures by (task_class, root cause) so we don't print 11x
        # the same message for the same task with 11 encoder variants
        seen: set[tuple[str, str]] = set()
        for r in failures:
            key = (r.task_class or "", r.datasets_reason or r.task_reason or "")
            if key in seen:
                continue
            seen.add(key)
            print(f"\n  Task: {r.task_class}")
            print(f"  Example config: {r.config.name}")
            if not r.task_ok:
                print(f"    TASK CLASS: {r.task_reason}")
            if not r.datasets_ok:
                print(f"    DATASETS:   {r.datasets_reason}")
            if r.runtime_ok is False:
                print(f"    RUNTIME:    {r.runtime_reason}")
            # Show how many configs share this failure
            same_root_count = sum(
                1 for x in failures if (x.task_class, x.datasets_reason or x.task_reason) == key
            )
            print(f"    (affects {same_root_count} config(s))")

    # ── Task-level summary table ─────────────────────────────────────────
    print("\n── PER-TASK SUMMARY ──")
    print(f"  {'Task':<55} {'configs':>8} {'pass':>6} {'fail':>6}")
    print("  " + "-" * 79)
    for task in sorted(by_task):
        results = by_task[task]
        n_pass = sum(1 for r in results if r.ok)
        n_fail = len(results) - n_pass
        mark = "✓" if n_fail == 0 else "✗"
        print(f"  {mark} {task:<53} {len(results):>8} {n_pass:>6} {n_fail:>6}")

    print(f"\nTotal: {len(cache_enabled_configs)} configs, "
          f"{len(failures)} failures, "
          f"{len(cache_enabled_configs) - len(failures)} passes.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
