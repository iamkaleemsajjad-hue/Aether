"""CPU and ingestion contracts hold without optional frameworks and SDKs."""

from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace

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

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: object(), raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    capabilities = detect_directml()
    assert capabilities.available is False
    assert capabilities.implemented is True  # ONNX adapter code is independent of installed runtime
    assert capabilities.backend_available is False
    assert capabilities.hardware_present is None
    assert capabilities.execution_tested is False
    assert capabilities.supports_fp16 is False
    assert capabilities.extra["detection_method"] == "d3d12_dxgi_dll_probe"


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
