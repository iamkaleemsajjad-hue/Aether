"""SDK and compiler presence must not advertise unfinished inference adapters."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aether.backends.hardware_backends import (
    FPGABackend,
    OpenVINOBackend,
    QualcommBackend,
    RISCVNPUBackend,
    TensorRTLLMBackend as HardwareTensorRTLLMBackend,
    create_backend,
)
from aether.backends.trtllm_backend import TensorRTLLMBackend
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError


def _not_inference_ready(backend):
    assert backend.is_available() is False
    assert backend.available_for_target(backend.info.supported_targets[0]) is False
    assert "generate" not in backend.get_capabilities()
    assert "chat" not in backend.get_capabilities()
    assert "stream" not in backend.get_capabilities()
    assert "generate" not in backend.info.capabilities
    assert backend.info.version == AETHER_VERSION
    with pytest.raises(BackendError) as error:
        backend.load_model("model", engine_path="prebuilt-engine")
    assert backend.unavailable_reason in str(error.value)


@pytest.mark.parametrize("target", ["qualcomm_qnn", "qualcomm_cloud_ai100"])
def test_existing_qnn_sdk_directory_does_not_supply_an_execution_adapter(
    monkeypatch, tmp_path, target
):
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    backend = create_backend(target)
    assert isinstance(backend, QualcommBackend)
    assert backend._qnn_available is True
    assert "directory found" in backend.unavailable_reason
    _not_inference_ready(backend)


@pytest.mark.parametrize(
    "target", ["riscv_mips_s8200", "riscv_sifive_x160", "riscv_xuantie_c930", "riscv_cervell"]
)
def test_riscv_cpu_and_onnxruntime_do_not_prove_an_npu_adapter(monkeypatch, target):
    monkeypatch.setattr("aether.backends.hardware_backends.platform.machine", lambda: "riscv64")
    monkeypatch.setattr(RISCVNPUBackend, "_detect_ort", lambda _: True)
    backend = create_backend(target)
    assert backend._ort_available is True
    _not_inference_ready(backend)
    with pytest.raises(BackendError, match="vendor NPU execution adapter"):
        backend.load("model")


@pytest.mark.parametrize("target", ["openvino_npu", "openvino_gpu"])
def test_openvino_vendor_plugin_is_not_an_aether_generation_runner(monkeypatch, target):
    compile_model = Mock(return_value=object())
    core = SimpleNamespace(available_devices=["NPU", "GPU"], compile_model=compile_model)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    backend = create_backend(target)
    assert isinstance(backend, OpenVINOBackend)
    assert backend._available is True
    _not_inference_ready(backend)
    compile_model.assert_not_called()


def test_openvino_low_level_graph_compilation_is_preserved(monkeypatch, tmp_path):
    compiled = object()
    compile_model = Mock(return_value=compiled)
    core = SimpleNamespace(available_devices=["NPU"], compile_model=compile_model)
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    model = tmp_path / "emitted.onnx"
    model.write_bytes(b"fake model: this unit test exercises dispatch, not real compilation")
    backend = OpenVINOBackend()
    assert backend.load(str(model)) is True
    compile_model.assert_called_once_with(str(model), "NPU")
    assert backend._compiled_model is compiled
    assert backend.is_available() is False


def test_riscv_ir_compiler_dispatch_is_preserved(monkeypatch):
    graph = object()
    result = {"target": "riscv_mips_s8200", "instructions": []}
    compiler = SimpleNamespace(compile=Mock(return_value=result))
    constructor = Mock(return_value=compiler)
    monkeypatch.setattr(
        "aether.compiler.stage3_targeting.riscv_npu_ir.RISCVNPUCompiler", constructor
    )
    monkeypatch.setattr(RISCVNPUBackend, "_detect_ort", lambda _: False)
    backend = RISCVNPUBackend()
    assert backend.compile_to_riscv_ir(graph) is result
    constructor.assert_called_once_with(target_id="riscv_mips_s8200")
    compiler.compile.assert_called_once_with(graph)


@pytest.mark.parametrize("backend_class", [HardwareTensorRTLLMBackend, TensorRTLLMBackend])
def test_tensorrt_sdk_and_engine_cache_do_not_supply_a_runner(monkeypatch, backend_class):
    monkeypatch.setitem(sys.modules, "tensorrt_llm", SimpleNamespace(__version__="test-sdk"))
    backend = backend_class()
    if hasattr(backend, "_engines"):
        backend._engines["model"] = object()
    _not_inference_ready(backend)


def test_discovery_does_not_advertise_an_installed_tensorrt_stub(monkeypatch):
    from aether.backends.registry import BackendRegistry

    monkeypatch.setitem(sys.modules, "tensorrt_llm", SimpleNamespace(__version__="test-sdk"))
    registry = BackendRegistry()
    registry._factories = {"trtllm": TensorRTLLMBackend}
    assert registry.get_available_backend_names() == []
    assert registry.get_backend("trtllm") is not None


@pytest.mark.parametrize("target", ["fpga_xilinx_vu9p", "fpga_ternary"])
def test_fpga_profiles_do_not_advertise_unimplemented_execution_or_power(target):
    backend = create_backend(target)
    assert isinstance(backend, FPGABackend)
    _not_inference_ready(backend)
    assert backend.get_capabilities() == []
    for feature in ("low_power", "decode_only", "ternary_arithmetic", "sub2bit"):
        assert backend.supports(feature) is False
