"""Canonicalize a sweep run's identity into queryable coordinates.

A MARBLE layer-sweep produces many wandb runs whose layer/fold/stage are
encoded only in the run *name* / *tags* / *job_type* (e.g. ``layer-6-test-fold0``).
That makes "group by layer, average across folds" impossible in the wandb UI,
because layer and fold are not first-class scalar fields.

``parse_sweep_coords`` turns ``(name, job_type, tags)`` into
``{layer, fold, stage, repr}``:

  - ``layer``: int layer index, ``-1`` for the mean-over-all-layers baseline,
    or ``None`` if undetermined.
  - ``fold``: int CV fold, or ``None`` (e.g. fit runs that don't carry it).
  - ``stage``: ``"fit"`` / ``"test"`` / ``None``.
  - ``repr``: ``"single"`` (one layer) or ``"meanall"``.

Both the live ``LogSweepCoordsCallback`` (stamps new runs) and the one-time
wandb backfill use this so new and historical runs are grouped identically.
Pure stdlib; no wandb import.
"""

import re

_LAYER_RX = re.compile(r"layer-(\d+)")
_FOLD_RX = re.compile(r"fold[-_]?(\d+)")


def parse_sweep_coords(
    name: str | None,
    job_type: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    name = name or ""
    job_type = job_type or ""
    tags = tags or []

    is_meanall = (
        ("meanall" in name)
        or ("mean-all" in name)
        or any("mean-all" in t or "meanall" in t for t in tags)
    )

    # layer: name first (e.g. "layer-6-..."), then a "layer-N" tag; meanall -> -1
    if is_meanall:
        layer: int | None = -1
    else:
        m = _LAYER_RX.search(name)
        if not m:
            for t in tags:
                m = _LAYER_RX.search(t)
                if m:
                    break
        layer = int(m.group(1)) if m else None

    # fold: from name, else job_type (e.g. "test-fold2")
    fm = _FOLD_RX.search(name) or _FOLD_RX.search(job_type)
    fold = int(fm.group(1)) if fm else None

    # stage: prefer job_type's leading token, fall back to name content
    if job_type.startswith("test") or "-test" in name or name.endswith("test"):
        stage: str | None = "test"
    elif job_type.startswith("fit") or "-fit" in name or name.endswith("fit"):
        stage = "fit"
    else:
        stage = None

    return {
        "layer": layer,
        "fold": fold,
        "stage": stage,
        "repr": "meanall" if is_meanall else "single",
    }


def resolve_coords(
    name: str | None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    *,
    fold_idx: int | None = None,
    stage: str | None = None,
) -> dict:
    """``parse_sweep_coords`` plus authoritative overrides.

    The live callback knows the fold from ``trainer.datamodule.fold_idx`` and
    the stage from the Lightning hook — both more reliable than name parsing
    (fit-run names carry no fold). These win when provided. ``fold_idx`` is
    checked with ``is not None`` so fold 0 is honored.
    """
    coords = parse_sweep_coords(name, job_type, tags)
    if fold_idx is not None:
        coords["fold"] = int(fold_idx)
    if stage is not None:
        coords["stage"] = stage
    return coords
