"""LogSweepCoordsCallback writes sweep/* coords into the wandb run config.

The wandb-run lookup is isolated in ``_find_wandb_run`` so these tests can
inject a fake run (no real wandb session) and assert the stamped config.
"""

import types

from marble.modules import callbacks as cb


class _FakeConfig:
    def __init__(self):
        self.updated = {}

    def update(self, d, allow_val_change=False):
        self.updated.update(d)


class _FakeRun:
    def __init__(self, name, job_type, tags):
        self.name = name
        self.job_type = job_type
        self.tags = tags
        self.config = _FakeConfig()


def _trainer(fold_idx):
    dm = types.SimpleNamespace(fold_idx=fold_idx) if fold_idx is not None else None
    return types.SimpleNamespace(datamodule=dm, loggers=[])


def test_callback_stamps_test_run(monkeypatch):
    run = _FakeRun("layer-6-test-fold0", "test-fold0", ["layer-6", "layer-sweep"])
    monkeypatch.setattr(cb, "_find_wandb_run", lambda trainer: run)
    cb.LogSweepCoordsCallback().on_test_start(_trainer(0), None)
    assert run.config.updated == {
        "sweep/layer": 6,
        "sweep/fold": 0,
        "sweep/stage": "test",
        "sweep/repr": "single",
    }


def test_callback_recovers_fold_for_fit_run_from_datamodule(monkeypatch):
    # fit-run name carries no fold; datamodule.fold_idx=3 supplies it
    run = _FakeRun("layer-6-fit", "fit", ["layer-6"])
    monkeypatch.setattr(cb, "_find_wandb_run", lambda trainer: run)
    cb.LogSweepCoordsCallback().on_fit_start(_trainer(3), None)
    assert run.config.updated["sweep/fold"] == 3
    assert run.config.updated["sweep/stage"] == "fit"
    assert run.config.updated["sweep/layer"] == 6


def test_callback_noop_without_wandb(monkeypatch):
    monkeypatch.setattr(cb, "_find_wandb_run", lambda trainer: None)
    # must not raise even with no datamodule
    cb.LogSweepCoordsCallback().on_fit_start(_trainer(None), None)


def test_callback_never_raises_on_config_error(monkeypatch):
    class _BadConfig:
        def update(self, *a, **k):
            raise RuntimeError("wandb down")

    run = _FakeRun("layer-1-test-fold1", "test-fold1", ["layer-1"])
    run.config = _BadConfig()
    monkeypatch.setattr(cb, "_find_wandb_run", lambda trainer: run)
    # swallowed — logging metadata must never break a run
    cb.LogSweepCoordsCallback().on_test_start(_trainer(1), None)
