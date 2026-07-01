"""Tests for marble.core.utils.instantiate_recursive.

The non-recursive ``instantiate_from_config`` passes nested ``class_path`` dicts
through verbatim, which is fine for the datamodule (it stores split configs raw)
but breaks model instantiation — ``BaseTask.__init__`` wraps ``emb_transforms`` /
``decoders`` / ``metrics`` in ``nn.ModuleList`` and expects real Modules, so a raw
dict raises "dict is not a Module subclass" (the bug that killed extract.py on
frame-level probes). ``instantiate_recursive`` resolves nested class_path configs
(in lists, and in metrics-style dict-of-dicts) into real objects first.
"""

from __future__ import annotations

import torch

from marble.core.utils import instantiate_from_config, instantiate_recursive


class _Holder(torch.nn.Module):
    """Toy module that takes another module as a kwarg (like a decoder taking
    an ``activation_fn`` class_path)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner


def test_leaf_class_path_instantiates():
    obj = instantiate_recursive({"class_path": "torch.nn.ReLU"})
    assert isinstance(obj, torch.nn.ReLU)


def test_scalar_init_args():
    obj = instantiate_recursive(
        {"class_path": "torch.nn.Linear", "init_args": {"in_features": 4, "out_features": 2}}
    )
    assert obj.in_features == 4 and obj.out_features == 2


def test_list_of_configs_each_instantiated():
    out = instantiate_recursive(
        [{"class_path": "torch.nn.ReLU"}, {"class_path": "torch.nn.Sigmoid"}]
    )
    assert isinstance(out[0], torch.nn.ReLU) and isinstance(out[1], torch.nn.Sigmoid)


def test_nested_class_path_inside_init_args():
    obj = instantiate_recursive(
        {
            "class_path": "tests.test_instantiate_recursive._Holder",
            "init_args": {"inner": {"class_path": "torch.nn.ReLU"}},
        }
    )
    # type-name check avoids pytest's duplicate-module identity quirk; the
    # load-bearing assertion is that the nested class_path became a real module.
    assert type(obj).__name__ == "_Holder" and isinstance(obj.inner, torch.nn.ReLU)


def test_metrics_style_dict_of_dicts():
    cfg = {
        "train": {"a": {"class_path": "torch.nn.ReLU"}},
        "val": {"b": {"class_path": "torch.nn.Sigmoid"}},
    }
    out = instantiate_recursive(cfg)
    assert isinstance(out["train"]["a"], torch.nn.ReLU)
    assert isinstance(out["val"]["b"], torch.nn.Sigmoid)


def test_scalars_and_plain_values_passthrough():
    assert instantiate_recursive(5) == 5
    assert instantiate_recursive("hello") == "hello"
    assert instantiate_recursive([1, 2, 3]) == [1, 2, 3]
    assert instantiate_recursive({"x": 1, "y": 2}) == {"x": 1, "y": 2}


def test_instantiate_from_config_is_idempotent_on_instances():
    # Some classes (e.g. MLPDecoderKeepTime) self-instantiate a nested config
    # (activation_fn) via instantiate_from_config. When instantiate_recursive has
    # already resolved it to a real object, instantiate_from_config must pass it
    # through rather than treating the instance as a config dict.
    relu = torch.nn.ReLU()
    assert instantiate_from_config(relu) is relu
