"""Unit tests for MULTI-HEAD parallel layer probing.

Pieces under test (all additive — the single-head path is untouched):
  1. ``PerLayerHeads`` (marble/modules/decoders.py) — K per-layer heads, each
     structurally identical to ``MLPDecoderKeepTime``, + optional meanall
     head; stacked (B, K, T, C) output.
  2. The UPDATE-EQUIVALENCE invariant: because heads are parameter-disjoint
     and Adam state is per-parameter, training with the summed multi-head
     loss produces EXACTLY the parameter trajectories independent single-head
     runs would produce at the same LR. Proven here bitwise on synthetic data.
  3. ``ProbeAudioTaskMultiHead`` — per-head summed loss, per-head metric
     collections ({split}/acc_rpa_l{k} / _meanall), aggregate _best.
  4. ``PerHeadBestCheckpoint`` — snapshots a head only when its monitored
     val metric improves; restores each head's own best weights at test
     start; seeds the incumbent best from disk (resume-safety).

All tests are CPU-only and use synthetic tensors — no audio, no encoder
downloads (<5 s total).

Run with:
    uv run pytest tests/test_multihead_probe.py -v
or as a plain script:
    uv run python tests/test_multihead_probe.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marble.modules.callbacks import PerHeadBestCheckpoint  # noqa: E402
from marble.modules.decoders import MLPDecoderKeepTime, PerLayerHeads  # noqa: E402
from marble.modules.transforms import LayerSelector  # noqa: E402
from marble.tasks.HookTheoryMelody.probe import (  # noqa: E402
    MelodyCrossEntropyLoss,
    ProbeAudioTaskMultiHead,
    RawChromaAccuracy,
    RawPitchAccuracy,
)

RELU_CFG = {"class_path": "torch.nn.ReLU"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_heads(L=3, H=8, C=6, hidden=None, dropout=0.0, include_meanall=True):
    return PerLayerHeads(
        in_dim=H,
        out_dim=C,
        num_layers=L,
        hidden_layers=[5] if hidden is None else hidden,
        activation_fn=RELU_CFG,
        dropout=dropout,
        include_meanall=include_meanall,
    )


def _make_single(H=8, C=6, hidden=None, dropout=0.0):
    return MLPDecoderKeepTime(
        in_dim=H,
        out_dim=C,
        hidden_layers=[5] if hidden is None else hidden,
        activation_fn=RELU_CFG,
        dropout=dropout,
    )


class _StubEncoder(nn.Module):
    """Frozen 'encoder' emitting a deterministic tuple of L (B, T, H) layer
    tensors — the same layer-tuple contract a real MARBLE encoder (or a
    frame-level cache hit) presents to the emb_transforms."""

    def __init__(self, num_layers: int, t_out: int, hidden: int):
        super().__init__()
        self.num_layers = num_layers
        self.t_out = t_out
        self.hidden = hidden

    def forward(self, x):
        b = x.size(0)
        g = torch.Generator().manual_seed(1234)
        return tuple(
            torch.randn(b, self.t_out, self.hidden, generator=g) for _ in range(self.num_layers)
        )


def _make_task(L=3, T=6, H=8, C=12, include_meanall=True):
    metrics = {
        split: {
            "acc_rpa": RawPitchAccuracy(time_dim_mismatch_tol=5, ignore_index=-1),
            "acc_rca": RawChromaAccuracy(time_dim_mismatch_tol=5, ignore_index=-1),
        }
        for split in ("train", "val", "test")
    }
    return ProbeAudioTaskMultiHead(
        encoder=_StubEncoder(L, T, H),
        emb_transforms=[LayerSelector(layers=[f"0..{L - 1}"])],
        decoders=[_make_heads(L=L, H=H, C=C, include_meanall=include_meanall)],
        losses=[MelodyCrossEntropyLoss(time_dim_mismatch_tol=5, ignore_index=-1)],
        metrics=metrics,
        sample_rate=24000,
    )


def _make_batch(B=4, T=6, C=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 100, generator=g)  # waveform-ish; the stub ignores it
    y = torch.randint(-1, C, (B, T), generator=g)  # includes -1 (unvoiced) frames
    paths = [f"clip{i}.wav" for i in range(B)]
    return x, y, paths


# ─────────────────────────────────────────────────────────────────────────────
# 1. PerLayerHeads shapes + per-head equivalence to a single-layer run
# ─────────────────────────────────────────────────────────────────────────────


def test_perlayerheads_shapes_and_names():
    torch.manual_seed(0)
    B, L, T, H, C = 2, 3, 5, 8, 6
    dec = _make_heads(L=L, H=H, C=C, include_meanall=True)
    out = dec(torch.randn(B, L, T, H))
    assert out.shape == (B, L + 1, T, C), f"got {tuple(out.shape)}"
    assert dec.head_names == ["l0", "l1", "l2", "meanall"]

    dec_no_mean = _make_heads(L=L, H=H, C=C, include_meanall=False)
    out2 = dec_no_mean(torch.randn(B, L, T, H))
    assert out2.shape == (B, L, T, C), f"got {tuple(out2.shape)}"
    assert dec_no_mean.head_names == ["l0", "l1", "l2"]


def test_perlayerheads_head_matches_single_layer_run():
    """Head k with weights θ must produce bitwise the same logits a
    single-layer run's MLPDecoderKeepTime with weights θ produces on the
    LayerSelector(layers=[k]) slice — probe math identical per layer."""
    torch.manual_seed(3)
    B, L, T, H, C = 2, 3, 5, 8, 6
    dec = _make_heads(L=L, H=H, C=C, include_meanall=True)
    dec.eval()
    emb = torch.randn(B, L, T, H)
    out = dec(emb)
    for k in range(L):
        single = _make_single(H=H, C=C)
        single.load_state_dict(dec.heads[k].state_dict())
        single.eval()
        expected = single(emb[:, k : k + 1, :, :])
        torch.testing.assert_close(out[:, k], expected, rtol=0.0, atol=0.0)
    # meanall head == a dedicated meanall run: plain MLPDecoderKeepTime over
    # ALL layers (its internal mean over L is the meanall pooling).
    single = _make_single(H=H, C=C)
    single.load_state_dict(dec.heads[-1].state_dict())
    single.eval()
    torch.testing.assert_close(out[:, -1], single(emb), rtol=0.0, atol=0.0)


def test_perlayerheads_rejects_layer_mismatch():
    dec = _make_heads(L=3)
    try:
        dec(torch.randn(2, 2, 5, 8))  # L=2 != num_layers=3
    except ValueError as e:
        assert "LayerSelector" in str(e), f"error should point at the config fix: {e}"
        return
    raise AssertionError("Expected ValueError for L mismatch")


def test_perlayerheads_does_not_share_activation_instances():
    """A pre-instantiated parameterised activation (PReLU) must be
    deep-copied per head — instantiate_from_config passes instances through
    as-is, which would otherwise weight-tie the activation across heads."""
    act = nn.PReLU()
    dec = PerLayerHeads(
        in_dim=4, out_dim=3, num_layers=2, hidden_layers=[5], activation_fn=act, dropout=0.0
    )
    prelus = [[m for m in dec.heads[k].modules() if isinstance(m, nn.PReLU)] for k in range(2)]
    assert prelus[0] and prelus[1], "PReLU missing from heads"
    assert prelus[0][0] is not prelus[1][0], "activation instance shared across heads"
    assert prelus[0][0].weight is not prelus[1][0].weight, "PReLU weight tied across heads"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Update-equivalence proof
# ─────────────────────────────────────────────────────────────────────────────


def test_update_equivalence():
    """THE invariant that makes multi-head probing a drop-in replacement for
    K independent runs: heads are parameter-disjoint and Adam state is
    per-parameter, so training on the SUMMED loss yields exactly the updates
    each single-head run would produce at the same LR. Verified bitwise
    (rtol=atol=0) over several optimizer steps.

    dropout=0.0 here: with dropout on, multi-head and solo runs draw
    different masks from the global RNG (seed-level noise, not bias), which
    would break bitwise comparison without invalidating the invariant."""
    torch.manual_seed(7)
    B, L, T, H, C = 4, 2, 6, 8, 5
    lr, steps = 1e-3, 6

    mh = _make_heads(L=L, H=H, C=C, hidden=[7], include_meanall=False)
    singles = []
    for k in range(L):
        s = _make_single(H=H, C=C, hidden=[7])
        s.load_state_dict(mh.heads[k].state_dict())  # identical init per head
        singles.append(s)

    loss_fn = MelodyCrossEntropyLoss()
    opt_mh = torch.optim.Adam(mh.parameters(), lr=lr)
    opts = [torch.optim.Adam(s.parameters(), lr=lr) for s in singles]

    g = torch.Generator().manual_seed(99)
    batches = [
        (torch.randn(B, L, T, H, generator=g), torch.randint(-1, C, (B, T), generator=g))
        for _ in range(steps)
    ]

    for step, (emb, y) in enumerate(batches):
        # multi-head step: ONE optimizer over all heads, summed loss
        logits = mh(emb)
        loss = sum(loss_fn(logits[:, k], y) for k in range(L))
        opt_mh.zero_grad()
        loss.backward()
        opt_mh.step()

        # K independent single-head steps on the same data
        for k in range(L):
            single_loss = loss_fn(singles[k](emb[:, k : k + 1, :, :]), y)
            opts[k].zero_grad()
            single_loss.backward()
            opts[k].step()

        # parameter trajectories must coincide EXACTLY after every step
        for k in range(L):
            for (name_a, p_a), (name_b, p_b) in zip(
                mh.heads[k].named_parameters(), singles[k].named_parameters(), strict=True
            ):
                assert name_a == name_b
                torch.testing.assert_close(
                    p_a,
                    p_b,
                    rtol=0.0,
                    atol=0.0,
                    msg=f"step {step}, head {k}, param {name_a}: trajectories diverged",
                )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ProbeAudioTaskMultiHead — loss wiring, metrics, smoke forward
# ─────────────────────────────────────────────────────────────────────────────


def test_task_init_validation():
    """Config errors must fail loudly at __init__, not 40 epochs later."""
    L, T, H, C = 3, 6, 8, 12
    common = dict(
        encoder=_StubEncoder(L, T, H),
        emb_transforms=[LayerSelector(layers=[f"0..{L - 1}"])],
        sample_rate=24000,
    )
    # decoder is not a PerLayerHeads
    try:
        ProbeAudioTaskMultiHead(
            **common, decoders=[_make_single(H=H, C=C)], losses=[MelodyCrossEntropyLoss()]
        )
        raise AssertionError("expected ValueError for non-PerLayerHeads decoder")
    except ValueError as e:
        assert "head" in str(e).lower()
    # two losses
    try:
        ProbeAudioTaskMultiHead(
            **common,
            decoders=[_make_heads(L=L, H=H, C=C)],
            losses=[MelodyCrossEntropyLoss(), MelodyCrossEntropyLoss()],
        )
        raise AssertionError("expected ValueError for 2 losses")
    except ValueError as e:
        assert "ONE loss" in str(e)
    # primary metric missing from a configured split
    try:
        ProbeAudioTaskMultiHead(
            **common,
            decoders=[_make_heads(L=L, H=H, C=C)],
            losses=[MelodyCrossEntropyLoss()],
            metrics={"val": {"acc_rca": RawChromaAccuracy()}},
            primary_metric="acc_rpa",
        )
        raise AssertionError("expected ValueError for missing primary metric")
    except ValueError as e:
        assert "primary_metric" in str(e)


def test_multihead_loss_is_sum_of_per_head_losses():
    """_shared_step's loss must equal Σ_k loss_fn(logits[:, k], y) — no
    hidden averaging (a 1/K factor would rescale every head's effective LR
    vs the single-head runs)."""
    torch.manual_seed(11)
    L, T, C = 3, 6, 12
    task = _make_task(L=L, T=T, C=C)
    x, y, paths = _make_batch(T=T, C=C)
    # dropout=0 + deterministic stub encoder → the recomputed forward inside
    # _shared_step matches this reference forward exactly.
    logits = task(x)
    expected = sum(task.loss_fns[0](logits[:, k], y) for k in range(logits.size(1)))
    loss = task._shared_step((x, y, paths), 0, "train")
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
    assert loss.grad_fn is not None, "loss must stay attached to the graph"


def test_per_head_metrics_are_independent_and_named():
    """Each head gets its OWN metric accumulators (clone = deep copy) with
    the head name baked into the key."""
    torch.manual_seed(13)
    L, T, C = 3, 6, 12
    task = _make_task(L=L, T=T, C=C, include_meanall=True)
    x, y, paths = _make_batch(T=T, C=C)
    task._shared_step((x, y, paths), 0, "val")

    heads_mc = task.val_head_metrics
    assert len(heads_mc) == L + 1
    keys0 = set(heads_mc[0].compute())
    assert keys0 == {"val/acc_rpa_l0", "val/acc_rca_l0"}, f"got {keys0}"
    keys_last = set(heads_mc[-1].compute())
    assert keys_last == {"val/acc_rpa_meanall", "val/acc_rca_meanall"}, f"got {keys_last}"
    # independence: updating head 0 again must not move head 1's state
    before = heads_mc[1].compute()["val/acc_rpa_l1"].clone()
    heads_mc[0].update(torch.randn(4, T, C), y)
    after = heads_mc[1].compute()["val/acc_rpa_l1"]
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_task_smoke_forward_and_epoch_end():
    """End-to-end smoke over the synthetic (B, L, T, H) pipeline: forward
    shape, train/val steps, test_step, and the per-head primary values the
    _best aggregate is built from."""
    torch.manual_seed(17)
    L, T, C = 3, 6, 12
    task = _make_task(L=L, T=T, C=C, include_meanall=True)
    x, y, paths = _make_batch(T=T, C=C)

    logits = task(x)
    assert logits.shape == (4, L + 1, T, C), f"got {tuple(logits.shape)}"

    for split in ("train", "val"):
        loss = task._shared_step((x, y, paths), 0, split)
        assert torch.isfinite(loss), f"{split} loss not finite"

    task.test_step((x, y, paths), 0)

    for split in ("train", "val", "test"):
        values = task._head_primary_values(split)
        assert values is not None and len(values) == L + 1
        assert all(torch.isfinite(v) for v in values)
        best = torch.stack([v.float() for v in values]).max()
        assert 0.0 <= float(best) <= 1.0

    # 4-tuple batches (with clip_ids) must also unpack cleanly
    loss = task._shared_step((x, y, paths, [f"id{i}" for i in range(4)]), 0, "train")
    assert torch.isfinite(loss)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PerHeadBestCheckpoint
# ─────────────────────────────────────────────────────────────────────────────


def _make_cb_fixture(td):
    torch.manual_seed(23)
    dec = PerLayerHeads(
        in_dim=4, out_dim=3, num_layers=2, hidden_layers=[5], activation_fn=None, dropout=0.0
    )
    pl_module = SimpleNamespace(decoders=[dec])
    cb = PerHeadBestCheckpoint(dirpath=td, monitor_base="acc_rpa", mode="max")
    return dec, pl_module, cb


def _trainer_stub(metrics: dict[str, float], epoch: int, sanity: bool = False):
    return SimpleNamespace(
        callback_metrics={k: torch.tensor(v) for k, v in metrics.items()},
        sanity_checking=sanity,
        current_epoch=epoch,
        callbacks=[],
    )


def _snapshot(head: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def _perturb(dec: nn.Module) -> None:
    with torch.no_grad():
        for p in dec.parameters():
            p.add_(1.0)


def test_perhead_best_checkpoint_tracks_improvement_only():
    with tempfile.TemporaryDirectory() as td:
        dec, pl_module, cb = _make_cb_fixture(td)
        w_ep0 = [_snapshot(h) for h in dec.heads]

        # epoch 0: both heads snapshot (first value is always an improvement)
        cb.on_validation_end(
            _trainer_stub({"val/acc_rpa_l0": 0.5, "val/acc_rpa_l1": 0.3}, epoch=0), pl_module
        )
        p0 = Path(td) / "head_l0_best.pt"
        p1 = Path(td) / "head_l1_best.pt"
        assert p0.exists() and p1.exists()

        # epoch 1 (weights moved on): l0 got WORSE → keeps epoch-0 snapshot;
        # l1 improved → snapshots the new weights.
        _perturb(dec)
        w_ep1 = [_snapshot(h) for h in dec.heads]
        cb.on_validation_end(
            _trainer_stub({"val/acc_rpa_l0": 0.4, "val/acc_rpa_l1": 0.6}, epoch=1), pl_module
        )
        pay0 = torch.load(p0, map_location="cpu", weights_only=True)
        pay1 = torch.load(p1, map_location="cpu", weights_only=True)
        assert pay0["epoch"] == 0 and abs(pay0["metric"] - 0.5) < 1e-6
        assert pay1["epoch"] == 1 and abs(pay1["metric"] - 0.6) < 1e-6
        for k, v in pay0["state_dict"].items():
            torch.testing.assert_close(v, w_ep0[0][k], rtol=0.0, atol=0.0)
        for k, v in pay1["state_dict"].items():
            torch.testing.assert_close(v, w_ep1[1][k], rtol=0.0, atol=0.0)

        # test start (weights moved on again): each head restored to ITS OWN
        # best epoch — l0 from epoch 0, l1 from epoch 1.
        _perturb(dec)
        cb.on_test_start(_trainer_stub({}, epoch=2), pl_module)
        for k, v in dec.heads[0].state_dict().items():
            torch.testing.assert_close(v, w_ep0[0][k], rtol=0.0, atol=0.0)
        for k, v in dec.heads[1].state_dict().items():
            torch.testing.assert_close(v, w_ep1[1][k], rtol=0.0, atol=0.0)


def test_perhead_best_checkpoint_skips_sanity_and_missing_metrics():
    with tempfile.TemporaryDirectory() as td:
        dec, pl_module, cb = _make_cb_fixture(td)
        # sanity-check epoch: nothing written even though values are present
        cb.on_validation_end(
            _trainer_stub({"val/acc_rpa_l0": 0.9, "val/acc_rpa_l1": 0.9}, epoch=0, sanity=True),
            pl_module,
        )
        assert not any(Path(td).glob("head_*_best.pt"))
        # metric key missing for l1: l0 written, l1 skipped, no crash
        cb.on_validation_end(_trainer_stub({"val/acc_rpa_l0": 0.5}, epoch=0), pl_module)
        assert (Path(td) / "head_l0_best.pt").exists()
        assert not (Path(td) / "head_l1_best.pt").exists()


def test_perhead_best_checkpoint_seeds_incumbent_from_disk():
    """A FRESH callback instance (resumed fit / separate process) must not
    overwrite a better on-disk snapshot with a worse value."""
    with tempfile.TemporaryDirectory() as td:
        dec, pl_module, cb = _make_cb_fixture(td)
        cb.on_validation_end(
            _trainer_stub({"val/acc_rpa_l0": 0.5, "val/acc_rpa_l1": 0.5}, epoch=0), pl_module
        )
        # new instance, worse value → the epoch-0 payload must survive
        cb2 = PerHeadBestCheckpoint(dirpath=td, monitor_base="acc_rpa", mode="max")
        _perturb(dec)
        cb2.on_validation_end(
            _trainer_stub({"val/acc_rpa_l0": 0.45, "val/acc_rpa_l1": 0.55}, epoch=1), pl_module
        )
        pay0 = torch.load(Path(td) / "head_l0_best.pt", map_location="cpu", weights_only=True)
        pay1 = torch.load(Path(td) / "head_l1_best.pt", map_location="cpu", weights_only=True)
        assert pay0["epoch"] == 0 and abs(pay0["metric"] - 0.5) < 1e-6, "worse value overwrote best"
        assert pay1["epoch"] == 1 and abs(pay1["metric"] - 0.55) < 1e-6, "better value not saved"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Lightning Trainer integration (hook ordering)
# ─────────────────────────────────────────────────────────────────────────────


def test_trainer_integration_end_to_end():
    """Real Trainer fit+test on CPU — verifies the hook-ordering contracts
    the unit tests can't: (a) the object-logged per-head metrics and the
    module-hook-logged ``val/acc_rpa_best`` land in ``callback_metrics``
    before ModelCheckpoint / PerHeadBestCheckpoint fire at
    ``on_validation_end``; (b) the test run reports every per-head key; (c)
    the whole-model best.ckpt load + per-head restore chain runs in order."""
    import os

    import lightning.pytorch as lightning_pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader

    from marble.modules.callbacks import LoadLatestCheckpointCallback

    torch.manual_seed(31)
    L, T, C = 3, 6, 12

    def _loader(seed, n=8):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 100, generator=g)
        y = torch.randint(-1, C, (n, T), generator=g)
        ds = list(zip(x, y, [f"c{i}.wav" for i in range(n)]))
        return DataLoader(ds, batch_size=4)  # default collate → [x, y, list[str]]

    with tempfile.TemporaryDirectory() as td:
        task = _make_task(L=L, T=T, C=C)
        # LightningCLI wires the optimizer from YAML; here we attach it directly.
        task.configure_optimizers = lambda: torch.optim.Adam(task.parameters(), lr=1e-3)

        ckpt_dir = os.path.join(td, "checkpoints")
        trainer = lightning_pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=td,
            callbacks=[
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="best",
                    save_top_k=1,
                    save_last=True,
                    monitor="val/acc_rpa_best",
                    mode="max",
                ),
                LoadLatestCheckpointCallback(),
                # AFTER LoadLatestCheckpointCallback — same order as the config
                PerHeadBestCheckpoint(dirpath=ckpt_dir, monitor_base="acc_rpa", mode="max"),
            ],
        )
        trainer.fit(task, train_dataloaders=_loader(seed=1), val_dataloaders=_loader(seed=2))

        # (a) monitor + per-head keys reached callback_metrics; ModelCheckpoint
        # found its monitor (a missing monitor raises inside fit); per-head
        # snapshots written for all L+1 heads.
        assert "val/acc_rpa_best" in trainer.callback_metrics
        assert "val/acc_rpa_l0" in trainer.callback_metrics
        assert os.path.exists(os.path.join(ckpt_dir, "best.ckpt"))
        snaps = sorted(p.name for p in Path(ckpt_dir).glob("head_*_best.pt"))
        assert len(snaps) == L + 1, f"expected {L + 1} head snapshots, got {snaps}"

        # (b)+(c) test: best.ckpt load, per-head restore, per-head test keys
        results = trainer.test(task, dataloaders=_loader(seed=3))
        keys = set(results[0])
        expected = {f"test/acc_rpa_l{k}" for k in range(L)} | {
            "test/acc_rpa_meanall",
            "test/acc_rpa_best",
            "test/acc_rca_l0",
        }
        assert expected <= keys, f"missing test keys: {expected - keys}"
        best = results[0]["test/acc_rpa_best"]
        per_head = [results[0][f"test/acc_rpa_l{k}"] for k in range(L)] + [
            results[0]["test/acc_rpa_meanall"]
        ]
        assert abs(best - max(per_head)) < 1e-6, "best aggregate != max over heads"


if __name__ == "__main__":
    tests = [
        test_perlayerheads_shapes_and_names,
        test_perlayerheads_head_matches_single_layer_run,
        test_perlayerheads_rejects_layer_mismatch,
        test_perlayerheads_does_not_share_activation_instances,
        test_update_equivalence,
        test_task_init_validation,
        test_multihead_loss_is_sum_of_per_head_losses,
        test_per_head_metrics_are_independent_and_named,
        test_task_smoke_forward_and_epoch_end,
        test_perhead_best_checkpoint_tracks_improvement_only,
        test_perhead_best_checkpoint_skips_sanity_and_missing_metrics,
        test_perhead_best_checkpoint_seeds_incumbent_from_disk,
        test_trainer_integration_end_to_end,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")
