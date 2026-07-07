"""LayerSoftmaxSum frozen / warm-start gates (the gate-transfer mechanism)."""

import pytest
import torch

from marble.modules.transforms import LayerSoftmaxSum


def test_frozen_gates_reproduce_given_weights_exactly():
    w = [0.6038, 0.2287, 0.1676]  # HTM MuQ top-3, renormalised
    m = LayerSoftmaxSum(num_layers=3, normalize=False, learnable=False, init_weights=w)
    got = m.layer_weights()
    expect = torch.tensor(w) / sum(w)
    assert torch.allclose(got, expect, atol=1e-6)
    # forward == the explicit weighted sum
    x = torch.randn(2, 3, 7, 5)
    out = m(x)
    ref = (x * expect.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    assert out.shape == (2, 1, 7, 5)
    assert torch.allclose(out, ref, atol=1e-6)


def test_frozen_gates_have_no_trainable_params_and_no_grad():
    m = LayerSoftmaxSum(num_layers=3, learnable=False, init_weights=[1.0, 2.0, 3.0])
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 0
    out = m(torch.randn(1, 3, 4, 5, requires_grad=True))
    out.sum().backward()  # input grad flows; gate stays a buffer
    assert not isinstance(m.layer_gate, torch.nn.Parameter)


def test_warm_start_learnable_starts_at_given_weights():
    w = [0.5, 0.3, 0.2]
    m = LayerSoftmaxSum(num_layers=3, learnable=True, init_weights=w)
    assert torch.allclose(m.layer_weights(), torch.tensor(w), atol=1e-6)
    assert isinstance(m.layer_gate, torch.nn.Parameter)
    assert m.layer_gate.requires_grad


def test_default_is_uniform_learnable_unchanged():
    m = LayerSoftmaxSum(num_layers=4)
    assert torch.allclose(m.layer_weights(), torch.full((4,), 0.25))
    assert isinstance(m.layer_gate, torch.nn.Parameter)


def test_frozen_without_weights_rejected():
    with pytest.raises(ValueError, match="learnable=False without init_weights"):
        LayerSoftmaxSum(num_layers=3, learnable=False)


def test_bad_weights_rejected():
    with pytest.raises(ValueError, match="entries but num_layers"):
        LayerSoftmaxSum(num_layers=3, init_weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="strictly positive"):
        LayerSoftmaxSum(num_layers=2, init_weights=[1.0, 0.0])
