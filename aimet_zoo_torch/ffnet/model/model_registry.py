# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import sys

_model_entrypoints = {}


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    return fn


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name"""
    if model_name in _model_entrypoints:
        return _model_entrypoints[model_name]
    else:
        raise RuntimeError(
            f"Unknown model ({model_name}); known models are: "
            f"{_model_entrypoints.keys()}"
        )
