"""Canonical/legacy artifact removal and loaded backend state invalidation."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends.native_cpu_backend import NativeCPUBackend
from aether.core.exceptions import ModelNotFoundError
from aether.runtime.runtime import Runtime
from aether.utils.file_io import aeg_cache_candidates, cached_model_paths, delete_model


@pytest.mark.parametrize("layout", [0, 1, 2, 3])
def test_delete_model_removes_every_existing_cache_layout(tmp_path, layout):
    path = aeg_cache_candidates("org/model", tmp_path)[layout]
    path.mkdir(parents=True)
    (path / "weights.bin").write_bytes(b"old")
    assert delete_model("org/model", tmp_path)
    assert not path.exists()
    assert not delete_model("org/model", tmp_path)


def test_cache_list_name_is_a_valid_removal_alias(tmp_path):
    path = aeg_cache_candidates("org/model", tmp_path)[0]
    path.mkdir(parents=True)
    assert delete_model(path.name, tmp_path)
    assert not path.exists()


def test_cache_removal_is_bounded_to_models_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    assert not cached_model_paths("../outside", tmp_path)
    assert not delete_model("../outside", tmp_path)
    assert (outside / "keep.txt").exists()


def test_runtime_removal_invalidates_all_aliases_and_unloads_before_deleting(tmp_path):
    path = aeg_cache_candidates("org/model", tmp_path)[0]
    path.mkdir(parents=True)
    backend = Mock()
    backend.unload_model.side_effect = lambda _: pytest.fail("artifact deleted before backend unload") if not path.exists() else None
    runtime = object.__new__(Runtime)
    runtime.config = SimpleNamespace(model_cache_dir=str(tmp_path))
    runtime._lock = threading.RLock()
    runtime._execution_features = Mock()
    runtime._loaded_models = {"org/model": object(), "alias": object()}
    runtime._loaded_backends = {"org/model": backend, "alias": backend}
    runtime._model_targets = {"org/model": "cpu_generic", "alias": "cpu_generic"}
    runtime._aeg_paths = {"org/model": (None, str(path)), "alias": (None, str(path))}
    runtime.remove("org/model")
    assert not path.exists()
    assert backend.unload_model.call_count == 2
    assert runtime._execution_features.unload.call_count == 2
    assert not runtime._loaded_models and not runtime._loaded_backends
    assert not runtime._model_targets and not runtime._aeg_paths
    with pytest.raises(ModelNotFoundError):
        runtime.remove("org/model")


def test_native_cpu_unload_discards_model_tokenizer_and_batcher():
    backend = NativeCPUBackend()
    backend._models["model"] = object()
    backend._tokenizers["model"] = object()
    backend._batchers["model"] = object()
    backend.unload_model("model")
    assert not backend._models and not backend._tokenizers and not backend._batchers

