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


class PerHeadBestCheckpoint(Callback):
    """Per-head best-weight tracking for multi-head parallel layer probes.

    Why ``ModelCheckpoint`` alone is not enough: in a multi-head run,
    ``best.ckpt`` (monitored on ``val/<primary>_best``) freezes ALL heads at
    the single epoch where the best-performing head peaked — but head k's own
    best epoch is generally a different one (early layers tend to peak
    earlier than late layers). The single-head protocol tests each layer at
    its own validated-best checkpoint; to reproduce that per layer within one
    run, this callback snapshots any head whose monitored val metric improved
    (heads are ~1 MB — 14 snapshots cost less than one full ckpt) and
    restores every head to its own best weights at test start.

    Files: ``<dirpath>/head_<name>_best.pt`` with payload
    ``{"state_dict": ..., "epoch": int, "metric": float, "monitor": str}``,
    written atomically (tmp + os.replace, same dance as the embedding cache).
    ``<name>`` comes from the decoder's ``head_names`` ("l0".."l12",
    "meanall").

    Wiring expectations:
      * ``pl_module.decoders[0]`` exposes ``.heads`` (ModuleList) and
        ``.head_names`` — i.e. a ``PerLayerHeads`` decoder driven by
        ``ProbeAudioTaskMultiHead``, which logs ``val/<monitor_base>_<name>``
        per head.
      * Config placement: list this AFTER ``LoadLatestCheckpointCallback`` —
        callbacks fire in list order at test start, so the per-head restore
        must run after (and overwrite) the whole-model best.ckpt load.
      * ``dirpath=None`` reuses the ModelCheckpoint dirpath, keeping all
        checkpoint artifacts of a run in one directory.

    Resume-safety: the incumbent best value per head is (re-)seeded from the
    on-disk payloads on first use, so a killed-and-resumed fit keeps
    improving on the earlier epochs' snapshots instead of overwriting them
    with a worse post-resume value. Test runs in a fresh process need no
    callback state at all — they just read the files.
    """

    def __init__(
        self,
        dirpath: str | None = None,
        monitor_base: str = "acc_rpa",
        mode: str = "max",
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"PerHeadBestCheckpoint mode must be 'max' or 'min', got {mode!r}")
        self.dirpath = dirpath
        self.monitor_base = monitor_base
        self.mode = mode
        # name → best metric value so far. None until first use — seeded
        # lazily from disk so resumed fits inherit pre-kill bests.
        self._best: dict[str, float] | None = None
        self._warned_missing: set[str] = set()

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_dir(self, trainer) -> str:
        if self.dirpath:
            return self.dirpath
        ckpt_cb: ModelCheckpoint | None = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if ckpt_cb is None or not ckpt_cb.dirpath:
            raise RuntimeError(
                "PerHeadBestCheckpoint: no dirpath given and no ModelCheckpoint "
                "callback to borrow one from."
            )
        return ckpt_cb.dirpath

    @staticmethod
    def _decoder(pl_module):
        decoders = getattr(pl_module, "decoders", None)
        dec = decoders[0] if decoders is not None and len(decoders) else None
        if dec is None or not (hasattr(dec, "heads") and hasattr(dec, "head_names")):
            raise RuntimeError(
                "PerHeadBestCheckpoint needs pl_module.decoders[0] to expose "
                ".heads/.head_names (a PerLayerHeads decoder); got "
                f"{type(dec).__name__}."
            )
        return dec

    @staticmethod
    def _path_for(dirpath: str, name: str) -> str:
        return os.path.join(dirpath, f"head_{name}_best.pt")

    def _seed_best_from_disk(self, dirpath: str, head_names) -> None:
        if self._best is not None:
            return
        self._best = {}
        for name in head_names:
            path = self._path_for(dirpath, name)
            if os.path.exists(path):
                payload = torch.load(path, map_location="cpu", weights_only=True)
                metric = payload.get("metric")
                if metric is not None:
                    self._best[name] = float(metric)

    # ── fit: snapshot improved heads each val epoch ──────────────────────

    def on_validation_end(self, trainer, pl_module):
        # on_validation_end (NOT on_validation_epoch_end): by this hook the
        # logger connector has computed the epoch's object-logged metrics
        # into trainer.callback_metrics — the same timing ModelCheckpoint
        # relies on for its monitor.
        if getattr(trainer, "sanity_checking", False):
            return
        dirpath = self._resolve_dir(trainer)
        os.makedirs(dirpath, exist_ok=True)
        dec = self._decoder(pl_module)
        self._seed_best_from_disk(dirpath, dec.head_names)
        for k, name in enumerate(dec.head_names):
            key = f"val/{self.monitor_base}_{name}"
            value = trainer.callback_metrics.get(key)
            if value is None:
                # Warn once per key — a missing per-head metric means the
                # task's per-head logging and this callback's monitor_base
                # disagree; every head silently never snapshotting would be
                # a nasty way to discover that at test time.
                if key not in self._warned_missing:
                    self._warned_missing.add(key)
                    print(
                        f"[PerHeadBestCheckpoint] WARNING: {key!r} not in "
                        f"callback_metrics — head {name!r} will never snapshot. "
                        f"Check monitor_base vs the task's primary_metric."
                    )
                continue
            value = float(value)
            incumbent = self._best.get(name)
            improved = incumbent is None or (
                value > incumbent if self.mode == "max" else value < incumbent
            )
            if not improved:
                continue
            self._best[name] = value
            payload = {
                # detach+cpu+clone: snapshot must not alias live (possibly
                # GPU) training weights that keep changing after this hook.
                "state_dict": {
                    sk: sv.detach().cpu().clone() for sk, sv in dec.heads[k].state_dict().items()
                },
                "epoch": int(getattr(trainer, "current_epoch", -1)),
                "metric": value,
                "monitor": key,
            }
            target = self._path_for(dirpath, name)
            tmp = f"{target}.{os.getpid()}.tmp"
            torch.save(payload, tmp)
            os.replace(tmp, target)

    # ── test: restore every head to its own best weights ────────────────

    def on_test_start(self, trainer, pl_module):
        dirpath = self._resolve_dir(trainer)
        dec = self._decoder(pl_module)
        restored, missing = [], []
        for k, name in enumerate(dec.head_names):
            path = self._path_for(dirpath, name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            # load_state_dict copies INTO the existing parameters, so device
            # placement (set by the Trainer) is preserved.
            dec.heads[k].load_state_dict(payload["state_dict"])
            restored.append((name, payload.get("epoch"), payload.get("metric")))
        if restored:
            summary = ", ".join(f"{n}(ep{e}, {m:.4f})" for n, e, m in restored)
            print(f"[PerHeadBestCheckpoint] restored {len(restored)} head(s): {summary}")
        if missing:
            print(
                f"[PerHeadBestCheckpoint] WARNING: no best snapshot for head(s) "
                f"{missing} in {dirpath} — keeping the weights already loaded "
                f"(e.g. from best.ckpt)."
            )


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
            # AudioTransformDataset wraps the real dataset under `.base`
            # (symbolic/other wrappers may use `.dataset`) — probe both.
            for cand in (ds, getattr(ds, "base", None), getattr(ds, "dataset", None)):
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


class LogLayerWeightsCallback(Callback):
    """Log the softmax layer-contribution gates of any weighted-sum module.

    Walks the LightningModule for sub-modules exposing ``layer_weights()``
    (``PerLayerHeads(include_weighted=True)`` and ``LayerSoftmaxSum``) and
    logs ``layer_weight/l{k}`` each validation epoch — the per-epoch
    trajectory IS the SUPERB-style contribution chart (watch mass migrate
    toward the winning layers), and the final epoch's values are the bar
    chart. Also prints the final weights at fit end for the console record.

    Interpretation guardrails (thesis-facing; see PerLayerHeads docstring):
    the gates are only readable as contributions because the weighted head
    LayerNorms each layer before mixing (Feng et al., TASLP 2024); even so,
    learned gates correlate only weakly with true per-layer probe accuracy
    (Spearman ρ≈0.37–0.49 ibid.) — the per-layer heads' own metric curves
    remain the ground-truth contribution measure.
    """

    @staticmethod
    def _weighted_modules(pl_module):
        for name, mod in pl_module.named_modules():
            fn = getattr(mod, "layer_weights", None)
            include_weighted = getattr(mod, "include_weighted", True)
            if callable(fn) and include_weighted:
                yield name, mod

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        payload = {}
        for _name, mod in self._weighted_modules(pl_module):
            w = mod.layer_weights()
            payload.update({f"layer_weight/l{k}": float(w[k]) for k in range(w.numel())})
        if payload:
            pl_module.log_dict(payload, on_step=False, on_epoch=True, sync_dist=False)

    def on_fit_end(self, trainer, pl_module):
        for name, mod in self._weighted_modules(pl_module):
            w = mod.layer_weights()
            top = ", ".join(f"l{i}={w[i]:.3f}" for i in w.argsort(descending=True)[:5].tolist())
            print(f"[LogLayerWeights] {name or type(mod).__name__} final gates (top-5): {top}")
