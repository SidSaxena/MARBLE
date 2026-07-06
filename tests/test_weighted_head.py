"""SUPERB-style weighted-sum head + LayerSoftmaxSum transform + weight logging.

Verification for the layer-contribution machinery added per the July 2026
layer-aggregation research (Feng et al. TASLP 2024 normalized benchmarking;
SUPERB featurizer; see PerLayerHeads / LayerSoftmaxSum docstrings):

1. LayerSoftmaxSum: shapes, uniform-at-init (== meanall of LayerNormed
   layers), gates sum to 1, normalize toggle actually changes the output.
2. PerLayerHeads(include_weighted=True): head ordering/naming (weighted
   LAST), gate gradient flow, layer_weights() contract.
3. NON-INTERFERENCE (the load-bearing one): adding the weighted head leaves
   every per-layer head's training trajectory BITWISE unchanged (constant
   LR, dropout 0) — extends the update-equivalence invariant of
   tests/test_multihead_probe.py to the weighted variant.
4. LogLayerWeightsCallback module discovery + payload keys.
"""

import torch
import torch.nn.functional as F

from marble.modules.callbacks import LogLayerWeightsCallback
from marble.modules.decoders import PerLayerHeads
from marble.modules.transforms import LayerSoftmaxSum

B, L, T, H, C = 3, 5, 7, 16, 11


def _emb(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, T, H, generator=g)


# ── 1. LayerSoftmaxSum ─────────────────────────────────────────────────────


def test_softmaxsum_shape_and_uniform_init():
    m = LayerSoftmaxSum(num_layers=L)
    x = _emb()
    y = m(x)
    assert y.shape == (B, 1, T, H)
    # zeros init → softmax uniform → output == mean over LayerNormed layers
    expected = F.layer_norm(x, (H,)).mean(dim=1, keepdim=True)
    torch.testing.assert_close(y, expected)
    w = m.layer_weights()
    torch.testing.assert_close(w, torch.full((L,), 1.0 / L))
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_softmaxsum_normalize_toggle_changes_output():
    x = _emb()
    y_norm = LayerSoftmaxSum(num_layers=L, normalize=True)(x)
    y_raw = LayerSoftmaxSum(num_layers=L, normalize=False)(x)
    assert not torch.allclose(y_norm, y_raw)
    # raw path at uniform init == plain meanall
    torch.testing.assert_close(y_raw, x.mean(dim=1, keepdim=True))


def test_softmaxsum_rejects_wrong_layer_count():
    m = LayerSoftmaxSum(num_layers=L)
    try:
        m(torch.randn(B, L + 1, T, H))
    except ValueError as e:
        assert "built for" in str(e)
        return
    raise AssertionError("expected ValueError on wrong L")


# ── 2. PerLayerHeads(include_weighted=True) ────────────────────────────────


def _heads(include_meanall=True, include_weighted=True, seed=0):
    torch.manual_seed(seed)
    return PerLayerHeads(
        in_dim=H,
        out_dim=C,
        num_layers=L,
        hidden_layers=[8],
        dropout=0.0,
        include_meanall=include_meanall,
        include_weighted=include_weighted,
    )


def test_weighted_head_ordering_and_names():
    dec = _heads()
    assert dec.head_names == [f"l{k}" for k in range(L)] + ["meanall", "weighted"]
    out = dec(_emb())
    assert out.shape == (B, L + 2, T, C)
    # per-layer + meanall logits are unaffected by the weighted head's
    # existence: compare against a decoder built without it, same seed.
    dec2 = _heads(include_weighted=False, seed=0)
    out2 = dec2(_emb())
    torch.testing.assert_close(out[:, : L + 1], out2)


def test_gate_gradient_flows_and_weights_contract():
    dec = _heads()
    out = dec(_emb())
    loss = F.cross_entropy(out[:, -1].reshape(-1, C), torch.zeros(B * T, dtype=torch.long))
    loss.backward()
    assert dec.layer_gate.grad is not None
    assert dec.layer_gate.grad.abs().sum() > 0
    w = dec.layer_weights()
    assert w.shape == (L,)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    # contract: raises when the head isn't configured
    try:
        _heads(include_weighted=False).layer_weights()
    except RuntimeError:
        pass
    else:
        raise AssertionError("layer_weights() must raise without include_weighted")


# ── 3. Non-interference: weighted head leaves other heads bitwise alone ───


def test_weighted_head_does_not_perturb_other_heads():
    """Train two decoders — with and without the weighted head — from
    identical seeds under constant-LR Adam on the SUMMED loss. Every
    per-layer + meanall head parameter must stay BITWISE identical: the
    gate/weighted-head params are disjoint, so their gradients never touch
    the other heads (same invariant as the multi-head equivalence proof)."""
    dec_a = _heads(include_weighted=True, seed=42)
    dec_b = _heads(include_weighted=False, seed=42)
    # seed=42 builds A's heads [0..L+1] and B's heads [0..L]; shared prefix
    # heads start identical because construction order matches until the
    # extra head. Verify precondition, then train.
    for pa, pb in zip(
        [p for k in range(L + 1) for p in dec_a.heads[k].parameters()],
        [p for k in range(L + 1) for p in dec_b.heads[k].parameters()],
        strict=True,
    ):
        assert torch.equal(pa, pb), "precondition: shared heads must start identical"

    opt_a = torch.optim.Adam(dec_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(dec_b.parameters(), lr=1e-3)
    y = torch.randint(0, C, (B * T,), generator=torch.Generator().manual_seed(7))
    for step in range(4):
        x = _emb(seed=100 + step)
        for dec, opt in ((dec_a, opt_a), (dec_b, opt_b)):
            opt.zero_grad()
            out = dec(x)
            loss = sum(F.cross_entropy(out[:, k].reshape(-1, C), y) for k in range(out.size(1)))
            loss.backward()
            opt.step()
    for k in range(L + 1):  # every per-layer head + meanall
        for pa, pb in zip(dec_a.heads[k].parameters(), dec_b.heads[k].parameters(), strict=True):
            assert torch.equal(pa, pb), (
                f"head {k} diverged when the weighted head was added — "
                "parameter-disjointness violated"
            )


# ── 4. LogLayerWeightsCallback ─────────────────────────────────────────────


def test_callback_discovers_and_formats_payload():
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = _heads()
            self.plain = _heads(include_weighted=False, seed=1)

    holder = Holder()
    found = list(LogLayerWeightsCallback._weighted_modules(holder))
    # only the weighted decoder is discovered; the plain one is skipped
    assert len(found) == 1 and found[0][1] is holder.dec
    w = found[0][1].layer_weights()
    payload = {f"layer_weight/l{k}": float(w[k]) for k in range(w.numel())}
    assert set(payload) == {f"layer_weight/l{k}" for k in range(L)}
    assert abs(sum(payload.values()) - 1.0) < 1e-6
