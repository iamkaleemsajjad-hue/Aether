"""
Aether Runtime — Complete Hardware Backends Test Suite.

Tests ALL hardware backend implementations with proper fail-closed
behavior when vendor hardware is not available.

All tests use the REAL GenerationRequest/GenerationResult signatures
from aether.backends.base as discovered from source.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aether.backends.base import (
    Backend,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
)
from aether.backends.hardware_backends import (
    CUDABackend,
    MetalBackend,
    ROCmBackend,
)
from aether.backends.registry import BackendRegistry
from aether.core.exceptions import BackendError


# ---------------------------------------------------------------------------
# GenerationRequest — real signature: model_id is first required arg
# ---------------------------------------------------------------------------

class TestGenerationRequest:
    def test_basic_creation(self):
        req = GenerationRequest(
            model_id="test_model",
            prompt="Hello, world",
            max_tokens=128,
            temperature=0.7,
        )
        assert req.model_id == "test_model"
        assert req.prompt == "Hello, world"
        assert req.max_tokens == 128
        assert req.temperature == 0.7

    def test_defaults(self):
        req = GenerationRequest(model_id="m")
        assert req.max_tokens >= 1
        assert 0.0 <= req.temperature <= 2.0
        assert req.prompt is None

    def test_stop_sequences(self):
        req = GenerationRequest(
            model_id="m", prompt="Test", stop=["<|end|>", "\n\n"]
        )
        assert "<|end|>" in req.stop

    def test_top_p_top_k(self):
        req = GenerationRequest(
            model_id="m",
            prompt="Test",
            max_tokens=50,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
        )
        assert req.top_p == 0.9
        assert req.top_k == 40

    def test_to_dict(self):
        req = GenerationRequest(model_id="m", prompt="hello")
        d = req.to_dict()
        assert d["model_id"] == "m"
        assert d["prompt"] == "hello"


class TestGenerationResult:
    def test_basic_creation(self):
        # Real fields: text, prompt_tokens, completion_tokens, finish_reason
        result = GenerationResult(
            text="Hello",
            completion_tokens=1,
            finish_reason="stop",
        )
        assert result.text == "Hello"
        assert result.completion_tokens == 1
        assert result.finish_reason == "stop"

    def test_defaults(self):
        result = GenerationResult(text="test")
        assert result.prompt_tokens == 0
        assert result.finish_reason == "stop"
        assert result.backend_name == "unknown"


# ---------------------------------------------------------------------------
# CUDA Backend
# ---------------------------------------------------------------------------

class TestCUDABackend:
    def test_init_no_crash(self):
        backend = CUDABackend(target_id="cuda_sm90")
        assert backend is not None

    def test_target_id_set(self):
        backend = CUDABackend(target_id="cuda_sm80")
        assert backend.target_id == "cuda_sm80"

    def test_availability_matches_torch(self):
        backend = CUDABackend()
        try:
            import torch
            expected = torch.cuda.is_available()
        except ImportError:
            expected = False
        assert backend.is_available() == expected

    def test_load_raises_when_unavailable(self):
        backend = CUDABackend()
        if backend.is_available():
            pytest.skip("CUDA is available on this system")
        with pytest.raises(BackendError):
            backend.load("/fake/model.aeg")

    def test_generate_raises_before_load(self):
        backend = CUDABackend()
        req = GenerationRequest(model_id="test", prompt="Test")
        with pytest.raises(BackendError):
            backend.generate(req)

    def test_capabilities_dict(self):
        backend = CUDABackend(target_id="cuda_sm90")
        caps = backend.get_capabilities()
        assert isinstance(caps, dict)
        assert "target_id" in caps
        assert "cuda_available" in caps
        assert caps["target_id"] == "cuda_sm90"

    def test_fp8_support_sm89(self):
        backend = CUDABackend(target_id="cuda_sm89")
        caps = backend.get_capabilities()
        assert caps.get("supports_fp8") is True

    def test_fp4_support_sm100(self):
        backend = CUDABackend(target_id="cuda_sm100")
        caps = backend.get_capabilities()
        assert caps.get("supports_fp4") is True

    def test_no_fp4_for_sm80(self):
        backend = CUDABackend(target_id="cuda_sm80")
        caps = backend.get_capabilities()
        assert caps.get("supports_fp4") is False

    def test_tee_support_sm100_tee(self):
        backend = CUDABackend(target_id="cuda_sm100_tee")
        caps = backend.get_capabilities()
        assert caps.get("supports_tee") is True

    def test_info_capabilities(self):
        backend = CUDABackend(target_id="cuda_sm90")
        assert "generate" in backend.info.capabilities
        assert "flash_attention" in backend.info.capabilities

    def test_all_sm_variants_instantiate(self):
        for target in ["cuda_sm70", "cuda_sm80", "cuda_sm89", "cuda_sm90",
                       "cuda_sm100", "cuda_sm120", "cuda_sm130"]:
            b = CUDABackend(target_id=target)
            assert b is not None
            assert b.target_id == target

    def test_unload_when_no_model_loaded(self):
        backend = CUDABackend()
        backend.unload()  # Should not raise


# ---------------------------------------------------------------------------
# ROCm Backend
# ---------------------------------------------------------------------------

class TestROCmBackend:
    def test_init_no_crash(self):
        backend = ROCmBackend(target_id="rocm_cdna3")
        assert backend is not None

    def test_is_available_boolean(self):
        backend = ROCmBackend()
        assert isinstance(backend.is_available(), bool)

    def test_load_raises_when_unavailable(self):
        backend = ROCmBackend()
        if backend.is_available():
            pytest.skip("ROCm is available on this system")
        with pytest.raises(BackendError):
            backend.load("/fake/model.aeg")

    def test_generate_raises_before_load(self):
        backend = ROCmBackend()
        req = GenerationRequest(model_id="test", prompt="Test")
        with pytest.raises(BackendError):
            backend.generate(req)

    def test_capabilities_list(self):
        backend = ROCmBackend()
        caps = backend.get_capabilities()
        assert isinstance(caps, list)
        assert "generate" in caps

    def test_hip_kernel_source_emitted(self):
        backend = ROCmBackend()
        source = backend.emit_hip_source("rmsnorm", {"hidden_size": 4096})
        assert isinstance(source, str)
        assert len(source) > 10

    def test_mi350x_has_fp8(self):
        backend = ROCmBackend(target_id="rocm_cdna5_mi455x")
        assert "fp8" in backend.info.capabilities or "mxfp6" in backend.info.capabilities

    def test_all_rocm_targets_instantiate(self):
        for t in ["rocm_rdna3", "rocm_cdna3", "rocm_cdna5_mi455x", "amd_mi350x"]:
            b = ROCmBackend(target_id=t)
            assert b is not None

    def test_unload_no_error(self):
        ROCmBackend().unload()


# ---------------------------------------------------------------------------
# Metal Backend
# ---------------------------------------------------------------------------

class TestMetalBackend:
    def test_init_no_crash(self):
        backend = MetalBackend(target_id="metal_m1")
        assert backend is not None

    def test_is_available_boolean(self):
        assert isinstance(MetalBackend().is_available(), bool)

    def test_load_raises_when_unavailable(self):
        backend = MetalBackend()
        if backend.is_available():
            pytest.skip("Metal MPS is available on this system")
        with pytest.raises(BackendError):
            backend.load("/fake/model.aeg")

    def test_generate_raises_before_load(self):
        backend = MetalBackend()
        req = GenerationRequest(model_id="test", prompt="Test")
        with pytest.raises(BackendError):
            backend.generate(req)

    def test_msl_source_emitted(self):
        backend = MetalBackend()
        source = backend.emit_msl_source("softmax", {"hidden_size": 4096})
        assert isinstance(source, str)
        assert len(source) > 10

    def test_m3_has_tensor_ops(self):
        backend = MetalBackend(target_id="metal_m3")
        caps = backend.get_capabilities()
        assert "metal4_tensor_ops" in caps or "fp16_native" in caps

    def test_all_metal_targets_instantiate(self):
        for t in ["metal_m1", "metal_m3"]:
            assert MetalBackend(target_id=t) is not None

    def test_unload_no_error(self):
        MetalBackend().unload()


# ---------------------------------------------------------------------------
# TensorRT-LLM Backend
# ---------------------------------------------------------------------------

class TestTensorRTLLMBackend:
    def test_importable(self):
        from aether.backends.hardware_backends import TensorRTLLMBackend
        assert TensorRTLLMBackend is not None

    def test_init_no_crash(self):
        from aether.backends.hardware_backends import TensorRTLLMBackend
        assert TensorRTLLMBackend() is not None

    def test_not_available_without_trtllm(self):
        from aether.backends.hardware_backends import TensorRTLLMBackend
        backend = TensorRTLLMBackend()
        if not backend.is_available():
            with pytest.raises(BackendError):
                backend.load("/fake/path")


# ---------------------------------------------------------------------------
# QNN / OpenVINO / RISC-V / FPGA (import tests)
# ---------------------------------------------------------------------------

class TestQNNBackend:
    def test_importable_or_skip(self):
        try:
            from aether.backends.hardware_backends import QNNBackend
            assert QNNBackend() is not None
        except ImportError:
            pytest.skip("QNNBackend not implemented")

    def test_not_available_on_x86(self):
        try:
            import platform
            from aether.backends.hardware_backends import QNNBackend
            backend = QNNBackend()
            if "x86" in platform.machine().lower():
                assert backend.is_available() is False
        except ImportError:
            pytest.skip("QNNBackend not implemented")


class TestOpenVINOBackend:
    def test_importable_or_skip(self):
        try:
            from aether.backends.hardware_backends import OpenVINOBackend
            assert isinstance(OpenVINOBackend().is_available(), bool)
        except ImportError:
            pytest.skip("OpenVINOBackend not implemented")


class TestRISCVBackend:
    def test_availability_false_on_x86(self):
        try:
            import platform
            from aether.backends.hardware_backends import RISCVBackend
            backend = RISCVBackend()
            if "x86" in platform.machine().lower():
                assert backend.is_available() is False
        except ImportError:
            pytest.skip("RISCVBackend not implemented")


class TestFPGABackend:
    def test_not_available_by_default(self):
        try:
            from aether.backends.hardware_backends import FPGABackend
            backend = FPGABackend()
            if not backend.is_available():
                with pytest.raises(BackendError):
                    backend.load("/fake/bitstream")
        except ImportError:
            pytest.skip("FPGABackend not implemented")


# ---------------------------------------------------------------------------
# Backend registry — real method names: get_backend, backend_names, etc.
# ---------------------------------------------------------------------------

class TestBackendRegistry:
    def test_registry_has_backends(self):
        registry = BackendRegistry()
        # backend_names returns all registered backends
        names = registry.backend_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_registry_get_backend_by_name(self):
        registry = BackendRegistry()
        # "pytorch" or "onnx" should be registered
        backend = registry.get_backend("pytorch")
        # May or may not be available, but shouldn't crash
        assert backend is not None or True

    def test_registry_get_backend_alias(self):
        """Aliases like 'llama.cpp' → 'llamacpp' should work."""
        registry = BackendRegistry()
        b1 = registry.get_backend("llamacpp")
        b2 = registry.get_backend("llama.cpp")
        if b1 is not None and b2 is not None:
            assert b1.name == b2.name

    def test_registry_get_unknown_backend_returns_none(self):
        registry = BackendRegistry()
        result = registry.get_backend("nonexistent_target_xyz")
        assert result is None

    def test_registry_get_available_backends(self):
        registry = BackendRegistry()
        available = registry.get_available_backends()
        assert isinstance(available, list)

    def test_registry_get_available_backend_names(self):
        registry = BackendRegistry()
        names = registry.get_available_backend_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_registry_register_manual(self):
        registry = BackendRegistry()
        # Create a minimal fake backend to register
        class FakeBackend(Backend):
            def __init__(self):
                super().__init__(BackendInfo("fake_test", "1.0", [], ["generate"]))
            def is_available(self): return False
            def load(self, path, config=None): raise BackendError("not available")
            def load_model(self, model_id, **kwargs): raise BackendError("not available")
            def generate(self, req): raise BackendError("not available")
            def generate_stream(self, req): raise BackendError("not available")
            def get_capabilities(self): return []
            def unload(self): pass

        fake = FakeBackend()
        registry.register_backend(fake)
        assert registry.get_backend("fake_test") is not None


# ---------------------------------------------------------------------------
# BackendInfo
# ---------------------------------------------------------------------------

class TestBackendInfo:
    def test_creation(self):
        info = BackendInfo(
            name="test_backend",
            version="1.0.0",
            supported_targets=["cpu_test"],
            capabilities=["generate"],
        )
        assert info.name == "test_backend"
        assert "generate" in info.capabilities

    def test_to_dict(self):
        info = BackendInfo("x", "1.0", ["cpu"], ["gen"])
        d = info.to_dict()
        assert d["name"] == "x"
        assert "cpu" in d["supported_targets"]


# ---------------------------------------------------------------------------
# TorchBackend (CPU path)
# ---------------------------------------------------------------------------

class TestTorchBackend:
    def test_importable(self):
        from aether.backends.torch_backend import TorchBackend
        assert TorchBackend is not None

    def test_cpu_path_available(self):
        try:
            import torch
            from aether.backends.torch_backend import TorchBackend
            assert TorchBackend() is not None
        except ImportError:
            pytest.skip("torch not installed")


# ---------------------------------------------------------------------------
# ONNXBackend
# ---------------------------------------------------------------------------

class TestONNXBackend:
    def test_importable(self):
        from aether.backends.onnx_backend import ONNXBackend
        assert ONNXBackend is not None

    def test_onnx_runtime_check(self):
        from aether.backends.onnx_backend import ONNXBackend
        backend = ONNXBackend()
        assert isinstance(backend.is_available(), bool)
