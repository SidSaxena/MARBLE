# marble/modules/callbacks.py
import contextlib
import glob
import os

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from marble.utils.sweep_coords import resolve_coords


class LoadLatestCheckpointCallback(Callback):
    """At test start, load the BEST checkpoint into pl_module.

    Picks ``best.ckpt`` when present in the ModelCheckpoint dirpath
    (that's the val-metric-tracked checkpoint written by
    ``ModelCheckpoint(monitor=..., save_top_k=1, filename="best")``).
    Falls back to "newest by mtime" only if ``best.ckpt`` isn't there
    (e.g. user used a non-"best" filename pattern).

    History — why this is non-trivial: the original implementation
    just picked the newest .ckpt by mtime. That was correct as long
    as ``save_last`` was off, because then the only file ever written
    was best.ckpt. Once ``save_last: true`` was added to the configs
    (so a killed run can resume from last.ckpt), the "newest by mtime"
    pick became wrong: ``last.ckpt`` is updated every epoch, so it's
    always newer than ``best.ckpt`` (which only updates on val-metric
    improvement). Testing would then load last.ckpt = whatever-was-last,
    not the validated best — a silent quality regression.
    """

    def on_test_start(self, trainer, pl_module):
        ckpt_cb: ModelCheckpoint | None = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if ckpt_cb is None:
            raise RuntimeError(
                "LoadLatestCheckpointCallback: no ModelCheckpoint callback found; "
                "cannot locate the checkpoint directory."
            )

        ckpt_dir = ckpt_cb.dirpath
        if not os.path.isdir(ckpt_dir):
            raise RuntimeError(f"LoadLatestCheckpointCallback: ckpt dir not found: {ckpt_dir}")

        # Prefer best.ckpt over last.ckpt and any per-epoch named checkpoints.
        # If the ModelCheckpoint config uses a non-"best" filename, fall back
        # to "newest .ckpt that is not last.ckpt".
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        if not candidates:
            raise RuntimeError(f"LoadLatestCheckpointCallback: no .ckpt found in {ckpt_dir}")

        best_path = os.path.join(ckpt_dir, "best.ckpt")
        if os.path.exists(best_path):
            chosen = best_path
            why = "best.ckpt (validated checkpoint)"
        else:
            non_last = [p for p in candidates if os.path.basename(p) != "last.ckpt"]
            if non_last:
                chosen = max(non_last, key=os.path.getmtime)
                why = "newest non-last .ckpt"
            else:
                # Only last.ckpt exists — best wasn't written (maybe val never
                # improved enough to trigger save). Use last.ckpt as a last
                # resort and warn.
                chosen = candidates[0]
                why = "last.ckpt fallback (no best.ckpt found — val metric may not have improved)"

        map_loc = {"cpu": "cpu"}
        if pl_module.device.type == "cuda":
            map_loc = {"cuda:0": f"cuda:{pl_module.device.index or 0}"}
        checkpoint = torch.load(chosen, map_location=map_loc)
        # Let the module re-inject anything it strips on save (e.g. a frozen
        # encoder dropped by BaseTask.on_save_checkpoint) so the strict load
        # below still matches. No-op for modules that don't strip / full ckpts.
        if hasattr(pl_module, "on_load_checkpoint"):
            pl_module.on_load_checkpoint(checkpoint)
        state_dict = checkpoint.get("state_dict", checkpoint)
        pl_module.load_state_dict(state_dict)

        if trainer.logger is not None:
            trainer.logger.log_metrics({"loaded_ckpt": os.path.basename(chosen)})
        print(f"[LoadLatestCheckpoint] loaded {chosen}  ({why})")


def _find_wandb_run(trainer):
    """Return the active wandb Run object if a WandbLogger is attached, else None.

    Isolated from the callback so it can be monkeypatched in tests (no real
    wandb session needed). Imports WandbLogger lazily so non-wandb setups
    don't pay the import.
    """
    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception:
        return None
    for lg in getattr(trainer, "loggers", None) or []:
        if isinstance(lg, WandbLogger):
            return lg.experiment
    return None


class LogSweepCoordsCallback(Callback):
    """Stamp each wandb run with queryable sweep coordinates.

    A layer sweep encodes layer/fold/stage only in the run name + tags
    (e.g. ``layer-6-test-fold0``), so wandb cannot "group by layer" or average
    across folds — those aren't scalar fields. This callback writes
    ``sweep/layer``, ``sweep/fold``, ``sweep/stage``, ``sweep/repr`` into the
    run config (derived from the run name/tags and, authoritatively, the
    datamodule's ``fold_idx``). Then the dashboard can **group by
    ``sweep/layer``** filtered to ``sweep/stage = test`` and read the per-layer
    mean across folds directly — no post-hoc script.

    No-op when no WandbLogger is attached; never raises (logging metadata must
    not break a training/test run).
    """

    def on_fit_start(self, trainer, pl_module):
        self._stamp(trainer, "fit")

    def on_test_start(self, trainer, pl_module):
        self._stamp(trainer, "test")

    def _stamp(self, trainer, stage):
        run = _find_wandb_run(trainer)
        if run is None:
            return
        fold_idx = getattr(getattr(trainer, "datamodule", None), "fold_idx", None)
        coords = resolve_coords(
            getattr(run, "name", "") or "",
            getattr(run, "job_type", None),
            list(getattr(run, "tags", []) or []),
            fold_idx=fold_idx,
            stage=stage,
        )
        # "sweep/<k>" namespaced keys: the wandb UI groups/filters on these
        # fine (the slash is display-nesting, not a blocker). logging metadata
        # must never break a training/test run.
        with contextlib.suppress(Exception):
            run.config.update(
                {f"sweep/{k}": v for k, v in coords.items()},
                allow_val_change=True,
            )
