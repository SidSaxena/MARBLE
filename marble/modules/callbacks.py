# marble/modules/callbacks.py
import glob
import os

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


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
        state_dict = checkpoint.get("state_dict", checkpoint)
        pl_module.load_state_dict(state_dict)

        if trainer.logger is not None:
            trainer.logger.log_metrics({"loaded_ckpt": os.path.basename(chosen)})
        print(f"[LoadLatestCheckpoint] loaded {chosen}  ({why})")
