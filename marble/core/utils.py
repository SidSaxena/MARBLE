# marble/core/utils.py
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    # Idempotent: an already-instantiated object (not a config dict) is returned
    # as-is. Lets classes that self-instantiate a nested config (e.g. a decoder's
    # activation_fn) also accept one pre-resolved by ``instantiate_recursive``.
    if not isinstance(config, dict):
        return config
    if "class_path" not in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


def instantiate_recursive(node):
    """Recursively instantiate a jsonargparse/LightningCLI-style config tree.

    Unlike :func:`instantiate_from_config` (which passes nested ``class_path``
    dicts through verbatim), this resolves them into real objects — needed when
    the target ``__init__`` expects instantiated submodules (e.g. ``BaseTask``
    wraps ``emb_transforms``/``decoders``/``metrics`` in ``nn.ModuleList`` and
    would otherwise choke on a raw dict). Rules:

      * ``{"class_path": ..., "init_args": {...}}`` → instantiate, recursing into
        each ``init_args`` value first (so nested module configs like a decoder's
        ``activation_fn`` are resolved).
      * ``dict`` without ``class_path`` → recurse each value (e.g. the
        ``metrics: {train: {name: {class_path}}}`` dict-of-dicts).
      * ``list`` → recurse each element.
      * anything else → returned unchanged.
    """
    if isinstance(node, dict):
        if "class_path" in node:
            cls = get_obj_from_str(node["class_path"])
            init_args = node.get("init_args") or {}
            return cls(**{k: instantiate_recursive(v) for k, v in init_args.items()})
        return {k: instantiate_recursive(v) for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate_recursive(v) for v in node]
    return node
