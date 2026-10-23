"""Defer unused TUM neural-network imports on CPUs without AVX.

The pinned TensorFlow remains installed. Accessing any TensorFlow API loads the
real package; this does not implement or replace any TensorFlow operation.
Only the deterministic policies in this capstone can run on this machine.
"""
import importlib
import importlib.util
import sys
import types


def defer_tensorflow():
    if "tensorflow" in sys.modules:
        return
    spec = importlib.util.find_spec("tensorflow")
    if spec is None:
        raise ImportError("Install the original TUM requirements first")
    deferred = types.ModuleType("tensorflow", "Deferred import of installed TensorFlow")
    deferred.__spec__ = spec
    deferred.__path__ = list(spec.submodule_search_locations or [])
    loaded = []

    def load_attribute(name):
        if not loaded:
            del sys.modules["tensorflow"]
            loaded.append(importlib.import_module("tensorflow"))
        return getattr(loaded[0], name)

    deferred.__getattr__ = load_attribute
    sys.modules["tensorflow"] = deferred
