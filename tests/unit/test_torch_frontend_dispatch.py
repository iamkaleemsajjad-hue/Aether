"""Placement and model-format contracts; mocked devices are not execution evidence."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends.torch_backend import TorchBackend
from aether.core.exceptions import BackendError


def frontend(monkeypatch, devices, actual_map=None):
    backend = object.__new__(TorchBackend)
    backend._models = {}
    backend._tokenizers = {}
    backend._device = devices[0]
    backend._devices = devices
    backend._runtime_family = "cpu" if devices == ["cpu"] else "metal" if devices == ["mps"] else "cuda"
    backend._configure_devices = lambda _: None
    backend._try_load_compiled_aeg = lambda *_: None
    backend.info = SimpleNamespace(name="pytorch")
    model = SimpleNamespace(hf_device_map=actual_map or {"": devices[0]})
    load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float16="fp16", float32="fp32",
        cuda=SimpleNamespace(mem_get_info=lambda _: (1024, 2048)),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
    ))
    return backend, load


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:2", "mps"])
def test_single_device_mesh_controls_frontend_placement(monkeypatch, device):
    backend, load = frontend(monkeypatch, [device])
    backend.load_model("organization/model", execution_devices=[device], offline=True)
    assert load.call_args.kwargs["device_map"] == {"": device}
    assert load.call_args.kwargs["torch_dtype"] == ("fp16" if device.startswith("cuda:") else "fp32")
    assert "execution_devices" not in load.call_args.kwargs
    assert load.call_args.kwargs["local_files_only"]


@pytest.mark.parametrize("mapping", ["auto", "cpu", {"": "cuda:1"}, {"": "disk"}])
def test_explicit_gpu_mesh_rejects_conflicting_frontend_map(monkeypatch, mapping):
    backend, load = frontend(monkeypatch, ["cuda:0"])
    with pytest.raises(BackendError, match="device_map.*execution_devices"):
        backend.load_model("organization/model", execution_devices=["cuda:0"], device_map=mapping)
    load.assert_not_called()


def test_multi_gpu_frontend_partitioning_excludes_unrequested_devices(monkeypatch):
    backend, load = frontend(monkeypatch, ["cuda:1", "cuda:3"], {"a": 1, "b": 3})
    backend.load_model("organization/model", execution_devices=["cuda:1", "cuda:3"])
    assert load.call_args.kwargs["device_map"] == "auto"
    assert load.call_args.kwargs["max_memory"] == {1: 1024, 3: 1024}


def test_frontend_placement_outside_requested_mesh_is_not_cached_as_success(monkeypatch):
    backend, _ = frontend(monkeypatch, ["cuda:0"], {"a": "cuda:0", "b": "cpu"})
    with pytest.raises(BackendError, match="outside execution_devices"):
        backend.load_model("organization/model", execution_devices=["cuda:0"])
    assert not backend._models
    assert not backend._tokenizers


def test_raw_sources_need_transformers_but_aegs_do_not(monkeypatch):
    backend = object.__new__(TorchBackend)
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert backend.available_for_model("compiled", aeg_path="compiled.aeg")
    assert not backend.available_for_model("organization/model")
    assert not backend.available_for_model("model.gguf")
    assert not backend.available_for_model("model.onnx")


def test_cached_model_mesh_is_independent_of_later_global_load_state(monkeypatch):
    backend, load = frontend(monkeypatch, ["cpu"])
    model = backend.load_model("cpu-source", execution_devices=["cpu"])
    backend._device = "cuda:3"
    backend._devices = ["cuda:3"]
    backend._runtime_family = "cuda"
    assert backend.load_model("cpu-source", execution_devices=["cpu:0"]) is model
    assert backend._input_device("cpu-source") == "cpu"
    assert backend._execution_family("cpu-source") == "cpu"
    with pytest.raises(BackendError, match="another execution mesh"):
        backend.load_model("cpu-source", execution_devices=["cuda:3"])
    load.assert_called_once()
    backend.unload_model("cpu-source")
    assert not backend._model_execution


def test_source_prefers_embedding_execution_hook_over_global_or_meta_device(monkeypatch):
    backend, load = frontend(monkeypatch, ["cuda:2"])
    model = load.return_value
    model.get_input_embeddings = lambda: SimpleNamespace(
        _hf_hook=SimpleNamespace(execution_device="cuda:2"),
        weight=SimpleNamespace(device="meta"),
    )
    backend.load_model("offloaded-source", execution_devices=["cuda:2"])
    backend._device = "cpu"
    assert backend._input_device("offloaded-source") == "cuda:2"


def test_numpy_aeg_device_does_not_inherit_a_later_accelerator_load(monkeypatch, tmp_path):
    from aether.backends.compiled_handle import CompiledAEGHandle

    backend, _ = frontend(monkeypatch, ["cuda:2"])
    handle = CompiledAEGHandle("cpu-aeg", tmp_path, {}, engine=object())
    backend._models["cpu-aeg"] = handle
    assert backend._input_device("cpu-aeg") == "cpu"
    assert backend._execution_family("cpu-aeg") == "cpu"


def test_cached_automatic_cpu_offload_cannot_satisfy_a_later_explicit_gpu_mesh(monkeypatch):
    backend, load = frontend(monkeypatch, ["cuda:0", "cuda:1"], {"embedding": "cuda:0", "layer": "cpu"})
    backend.load_model("auto-offloaded-source")
    with pytest.raises(BackendError, match="another execution mesh"):
        backend.load_model("auto-offloaded-source", execution_devices=["cuda:0", "cuda:1"])
    load.assert_called_once()
