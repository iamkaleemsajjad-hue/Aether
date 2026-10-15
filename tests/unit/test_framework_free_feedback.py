"""CPU and ingestion contracts hold without optional frameworks and SDKs."""

from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from aether.core.exceptions import IngestionError


def test_numpy_array_with_device_property_uses_numpy_linear_dispatch(monkeypatch):
    from aether.kernels import quantized_linear as linear

    class ArrayWithDevice(np.ndarray):
        @property
        def device(self):
            return "cpu"

    monkeypatch.setattr(linear, "_torch", lambda: None)
    x = np.asarray([[1, 2]], dtype=np.float32).view(ArrayWithDevice)
    weights = np.asarray([[2, 3]], dtype=np.int8)
    result, metadata = linear.int8_linear(x, weights, np.asarray([0.5], dtype=np.float32))
    np.testing.assert_allclose(result, [[4.0]])
    assert metadata.backend == "numpy_matmul"
    assert metadata.execution_mode == "dequant_fallback"


def test_mlx_bf16_safetensors_use_framework_free_decoder(tmp_path):
    from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader

    header = json.dumps(
        {
            "weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            }
        }
    ).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "model.safetensors"
    path.write_bytes(
        struct.pack("<Q", len(header))
        + header
        + np.asarray([0x3FC0, 0xC020], dtype="<u2").tobytes()
    )
    result = MLXLoader(path).load()
    np.testing.assert_array_equal(result["weights"]["weight"], [1.5, -2.5])


def test_mlx_corrupt_shard_cannot_silently_drop_weights(tmp_path):
    from aether.compiler.stage1_ingestion.mlx_loader import MLXLoader
    from safetensors.numpy import save_file

    save_file({"weight": np.ones((2, 2), dtype=np.float32)}, str(tmp_path / "a.safetensors"))
    (tmp_path / "b.safetensors").write_bytes(b"corrupt")
    with pytest.raises(IngestionError, match="Could not load MLX shard"):
        MLXLoader(tmp_path).load()


def test_gguf_tokenizer_export_uses_packaged_tokenizer_without_transformers(tmp_path):
    from aether.backends.native_cpu_backend import PackagedTokenizer
    from aether.compiler.stage1_ingestion.gguf_loader import export_gguf_tokenizer_metadata

    export_gguf_tokenizer_metadata(
        {
            "tokenizer.ggml.tokens": ["<unk>", "▁hello", "▁world"],
            "tokenizer.ggml.scores": [0.0, -0.1, -0.2],
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.chat_template": "{{ messages[0]['content'] }}",
        },
        tmp_path,
    )
    tokenizer = PackagedTokenizer(tmp_path / "tokenizer.json")
    np.testing.assert_array_equal(tokenizer("hello world")["input_ids"], [[1, 2]])
    assert tokenizer.decode([1, 2]) == "hello world"
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}]) == "hello"


def test_directml_os_dlls_are_not_evidence_of_an_executable_device(monkeypatch):
    import ctypes

    from aether.backends.hardware_detector import detect_directml

    # DLL presence is the only evidence this fixture supplies. The independent
    # OS probe may otherwise return a real device cached by an earlier test.
    inventory = Mock(return_value={"devices": [], "complete": False})
    monkeypatch.setattr("aether.backends.hardware_inventory.probe_gpu_inventory", inventory)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    load_dll = Mock(return_value=object())
    monkeypatch.setattr(ctypes, "WinDLL", load_dll, raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    capabilities = detect_directml()
    inventory.assert_called_once_with()
    assert [call.args[0] for call in load_dll.call_args_list] == ["d3d12", "dxgi"]
    assert capabilities.available is False
    assert capabilities.implemented is True  # ONNX adapter code is independent of installed runtime
    assert capabilities.backend_available is False
    assert capabilities.hardware_present is None
    assert capabilities.execution_tested is False
    assert capabilities.supports_fp16 is False
    assert capabilities.extra["detection_method"] == "d3d12_dxgi_dll_probe"


@pytest.mark.parametrize("providers", [None, ["CPUExecutionProvider"], ["DmlExecutionProvider", "CPUExecutionProvider"]],
                         ids=["runtime-missing", "cpu-provider-only", "dml-registered"])
@pytest.mark.parametrize("presence", ["unknown", "absent", "present"])
def test_directml_dependency_and_inventory_evidence_remain_independent(monkeypatch, providers, presence):
    """Controlled vendor inputs verify reporting; they never execute a GPU."""
    import ctypes

    from aether.backends.hardware_detector import detect_directml

    devices = [{"device": "controlled graphics adapter", "device_id": "test:gpu"}] if presence == "present" else []
    inventory = Mock(return_value={"devices": devices, "complete": presence != "unknown"})
    load_dll = Mock(return_value=object())
    monkeypatch.setattr("aether.backends.hardware_inventory.probe_gpu_inventory", inventory)
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "WinDLL", load_dll, raising=False)
    runtime = None if providers is None else SimpleNamespace(
        __version__="controlled-test-runtime", get_available_providers=Mock(return_value=providers),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    cap = detect_directml()

    inventory.assert_called_once_with()
    assert cap.hardware_present is {"unknown": None, "absent": False, "present": True}[presence]
    assert cap.extra["physical_adapters"] == devices
    registered = providers is not None and "DmlExecutionProvider" in providers
    assert cap.backend_dependencies == {"onnxruntime": providers is not None, "DmlExecutionProvider": registered}
    assert cap.runtime_available is (None if registered else False)
    assert cap.backend_available is (None if registered else False)
    assert cap.implemented and not cap.available
    assert not cap.execution_tested and not cap.compile_tested and not cap.production_validated
    assert not any(cap.to_dict()["precision_summary"].values())
    assert cap.memory_bytes == 0 and not cap.to_dict()["memory_bytes_known"]
    if runtime is None:
        assert [call.args[0] for call in load_dll.call_args_list] == ["d3d12", "dxgi"]
        assert cap.extra["os_dlls_present"]
    else:
        runtime.get_available_providers.assert_called_once_with()
        load_dll.assert_not_called()
        assert cap.extra["ort_providers"] == providers


def test_torch_availability_is_independent_of_the_previous_models_mesh(monkeypatch):
    from aether.backends.torch_backend import TorchBackend

    backend = object.__new__(TorchBackend)
    backend._runtime_family = "cpu"
    backend._devices = ["cpu"]
    monkeypatch.setattr(backend, "is_available", lambda: True)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        version=SimpleNamespace(hip=None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert backend.available_for_target("cuda_sm80")
    assert backend.available_for_target("cpu_avx2")
    assert not backend.available_for_target("rocm_cdna3")
    detected = []
    monkeypatch.setattr(backend, "_try_detect_device", lambda: detected.append(True))
    monkeypatch.delenv("AETHER_EXECUTION_DEVICES", raising=False)
    backend._configure_devices(None)
    assert detected == [True]


def test_benchmark_provenance_does_not_import_an_optional_framework(monkeypatch):
    import builtins
    import sys

    from aether.observability.benchmark_runner import BenchmarkProvenance

    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("CPU provenance initialized PyTorch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    provenance = BenchmarkProvenance(model_id="any-model")
    assert provenance.hardware_target == "cpu"
    assert "torch" not in sys.modules


def test_cpu_backend_lookup_does_not_initialize_optional_backends(monkeypatch):
    from aether.backends.registry import BackendRegistry

    registry = BackendRegistry()
    assert registry._backends == {}

    def forbidden():
        raise AssertionError("CPU lookup initialized an optional backend")

    for name in registry._factories:
        if name != "aether_cpu":
            registry._factories[name] = forbidden
    assert registry.get_backend("aether_cpu").name == "aether_cpu"
    assert set(registry._backends) == {"aether_cpu"}


def test_backend_factory_is_initialized_once_under_concurrent_lookup():
    from concurrent.futures import ThreadPoolExecutor

    from aether.backends.registry import BackendRegistry

    registry = BackendRegistry()
    calls = []

    def factory():
        calls.append(True)
        return SimpleNamespace(name="example")

    registry._factories["example"] = factory
    with ThreadPoolExecutor(max_workers=4) as pool:
        backends = list(pool.map(lambda _: registry.get_backend("example"), range(16)))
    assert calls == [True]
    assert all(backend is backends[0] for backend in backends)
