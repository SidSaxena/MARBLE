# marble/modules/callbacks.py
import contextlib
import glob
import os
import re

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

    @staticmethod
    def _resolve_fold_idx(datamodule):
        """Best-effort extraction of the CV fold from the datamodule.

        Authoritative source for ``sweep/fold`` so it is stamped on **fit** runs
        too (whose name carries no ``foldN`` token — without this they used to
        land with ``sweep/fold = None`` and were unrecoverable, the exact bug the
        per-fold sweeps hit). The ``fold_idx`` does NOT live on the
        ``BaseDataModule`` itself — it sits on the per-split *datasets* (set from
        the YAML ``init_args.fold_idx``), which only exist after ``setup()``, and
        in the split *config dicts*, which exist before it. We probe, in order:

          1. ``datamodule.fold_idx``                         (if a subclass adds it)
          2. the instantiated split datasets' ``fold_idx``   (post-setup)
          3. the split config dicts' ``init_args.fold_idx``  (pre-setup; the
             value always present at on_fit_start / on_test_start)

        Returns an int fold or None (tasks without folds — JKUPDD etc.).
        """
        if datamodule is None:
            return None
        direct = getattr(datamodule, "fold_idx", None)
        if direct is not None:
            return direct
        for attr in ("test_dataset", "train_dataset", "val_dataset"):
            ds = getattr(datamodule, attr, None)
            # AudioTransformDataset wraps the real dataset under `.dataset`.
            for cand in (ds, getattr(ds, "dataset", None)):
                fi = getattr(cand, "fold_idx", None)
                if fi is not None:
                    return fi
        for attr in ("test_config", "train_config", "val_config"):
            cfg = getattr(datamodule, attr, None)
            if isinstance(cfg, dict):
                fi = (cfg.get("init_args") or {}).get("fold_idx")
                if fi is not None:
                    return fi
        return None

    @staticmethod
    def _resolve_window(datamodule):
        """Best-effort within-piece window size (bars) from the JSONL path.

        The within-piece window-size sweep points each run at a per-N JSONL
        (e.g. ``data/BPS-Motif/BPSMotifWithinPiece.N8.ABC.jsonl``). Parsing ``N``
        from the test split's ``jsonl_template`` stamps ``sweep/window``
        authoritatively — independent of run naming — exactly as
        ``_resolve_fold_idx`` does for the CV fold, so window is recoverable on
        BOTH fit and test runs. Returns an int window or None (tasks without a
        per-N JSONL: every non-within-piece task).
        """
        if datamodule is None:
            return None
        rx = re.compile(r"\.N(\d+)\.")
        for attr in ("test_config", "val_config", "train_config"):
            cfg = getattr(datamodule, attr, None)
            if isinstance(cfg, dict):
                tmpl = (cfg.get("init_args") or {}).get("jsonl_template")
                if isinstance(tmpl, str):
                    m = rx.search(tmpl)
                    if m:
                        return int(m.group(1))
        return None

    def _stamp(self, trainer, stage):
        run = _find_wandb_run(trainer)
        if run is None:
            return
        dm = getattr(trainer, "datamodule", None)
        fold_idx = self._resolve_fold_idx(dm)
        window = self._resolve_window(dm)
        coords = resolve_coords(
            getattr(run, "name", "") or "",
            getattr(run, "job_type", None),
            list(getattr(run, "tags", []) or []),
            fold_idx=fold_idx,
            window=window,
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
