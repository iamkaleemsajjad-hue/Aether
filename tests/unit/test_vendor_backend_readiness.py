"""Vendor readiness contracts; mocked discovery is never execution evidence."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from aether.backends.base import GenerationRequest, GenerationResult
from aether.backends.capabilities import HardwareCapabilities
from aether.backends.compiled_handle import CompiledAEGHandle
from aether.backends.hardware_backends import CUDABackend, MetalBackend, ROCmBackend
from aether.backends.native_cuda_backend import NativeCUDABackend
from aether.backends.torch_backend import TorchBackend
from aether.compiler.stage3_targeting.backend_selector import BackendSelector
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.exceptions import BackendError, BackendNotAvailableError
from aether.kernels.native_cuda import (
    CUDACompilerToolchain,
    NativeCUDAKernels,
    detect_cuda_toolchain,
)


VENDORS = [
    (CUDABackend, "cuda_sm89", "NVIDIA", "sm89", "cuda:2", "detect_cuda_devices"),
    (ROCmBackend, "rocm_rdna3", "AMD", "gfx1100", "cuda:2", "detect_rocm_devices"),
    (MetalBackend, "metal_m3", "Apple", "apple_silicon", "mps", "detect_metal"),
]


def adapter(monkeypatch, specification, *, runtime_available=True):
    cls, target, vendor, architecture, device, detector_name = specification
    cap = HardwareCapabilities(
        vendor=vendor,
        device="diagnostic device",
        architecture=architecture,
        target_id=target,
        implemented=True,
        available=True,
        extra={"device_index": 2},
    )
    detected = cap if detector_name == "detect_metal" else [cap]
    monkeypatch.setattr(f"aether.backends.hardware_detector.{detector_name}", lambda: detected)
    runtime = SimpleNamespace(
        available_for_target=Mock(return_value=runtime_available),
        load_model=Mock(return_value=object()),
        unload_model=Mock(),
        generate=Mock(return_value=GenerationResult(text="diagnostic response")),
    )
    monkeypatch.setattr("aether.backends.torch_backend.TorchBackend", lambda: runtime)
    return cls(target), cap, runtime, device


@pytest.mark.parametrize("specification", VENDORS)
def test_vendor_device_presence_does_not_satisfy_backend_dependencies(monkeypatch, specification):
    backend, cap, runtime, _ = adapter(monkeypatch, specification, runtime_available=False)
    assert cap.available
    assert not backend.is_available()
    assert "PyTorch" in backend.unavailable_reason
    with pytest.raises(BackendError, match="PyTorch"):
        backend.load_model("logical-id", aeg_path="model.aeg")
    runtime.load_model.assert_not_called()
    assert not cap.execution_tested


@pytest.mark.parametrize("specification", VENDORS)
def test_vendor_load_preserves_model_id_and_forces_its_physical_device(monkeypatch, specification):
    backend, cap, runtime, device = adapter(monkeypatch, specification)
    model = backend.load_model("logical-id", aeg_path="model.aeg", offline=True)
    assert model is runtime.load_model.return_value
    runtime.load_model.assert_called_once_with(
        "logical-id",
        aeg_path="model.aeg",
        offline=True,
        execution_devices=[device],
        device_map={"": device},
    )
    request = GenerationRequest(model_id="logical-id", prompt="hello")
    assert backend.generate(request) is runtime.generate.return_value
    runtime.generate.assert_called_once_with(request)
    assert not cap.execution_tested


@pytest.mark.parametrize("specification", VENDORS)
@pytest.mark.parametrize("mesh", [["cpu"], ["cpu", "cuda:2"], ["cuda:9"]])
def test_vendor_adapter_rejects_conflicting_or_cpu_mesh(monkeypatch, specification, mesh):
    backend, _, runtime, _ = adapter(monkeypatch, specification)
    with pytest.raises(BackendError, match="execution_devices"):
        backend.load_model("model", execution_devices=mesh)
    runtime.load_model.assert_not_called()


@pytest.mark.parametrize("specification", VENDORS)
@pytest.mark.parametrize("device_map", ["auto", "cpu", {"": "cpu"}])
def test_vendor_frontend_cannot_silently_map_to_cpu(monkeypatch, specification, device_map):
    backend, _, runtime, _ = adapter(monkeypatch, specification)
    with pytest.raises(BackendError, match="device_map"):
        backend.load_model("model", device_map=device_map)
    runtime.load_model.assert_not_called()


@pytest.mark.parametrize("specification", VENDORS)
def test_vendor_loading_failures_are_not_converted_into_success_or_hidden(
    monkeypatch, specification
):
    backend, _, runtime, _ = adapter(monkeypatch, specification)
    runtime.load_model.side_effect = BackendError("actual loader failure")
    with pytest.raises(BackendError, match="actual loader failure"):
        backend.load("model")
    assert not hasattr(backend, "_torch_backend")


@pytest.mark.parametrize("specification", VENDORS)
def test_vendor_unload_uses_real_backend_api_for_the_logical_model_id(monkeypatch, specification):
    backend, _, runtime, _ = adapter(monkeypatch, specification)
    backend.unload()
    assert backend.load("model", {"offline": True})
    backend.unload_model("model")
    runtime.unload_model.assert_called_once_with("model")
    backend.unload()
    assert not hasattr(backend, "_torch_backend")


def test_cuda_capabilities_report_the_device_separately_from_target_profile(monkeypatch):
    backend, cap, _, _ = adapter(monkeypatch, VENDORS[0])
    assert not cap.supports_fp8
    capabilities = backend.get_capabilities()
    assert capabilities["supports_fp8"] is False
    assert capabilities["target_profile_capabilities"]["fp8"] is True
    assert capabilities["architecture"] == cap.architecture
    assert capabilities["backend_available"] is True
    assert not cap.execution_tested


def test_cuda_without_hardware_does_not_confirm_requested_precision_or_tee(monkeypatch):
    monkeypatch.setattr("aether.backends.hardware_detector.detect_cuda_devices", lambda: [])
    capabilities = CUDABackend("cuda_sm100_tee").get_capabilities()
    assert not capabilities["supports_fp8"]
    assert not capabilities["supports_fp4"]
    assert not capabilities["supports_tee"]
    assert not capabilities["backend_available"]
    assert capabilities["target_profile_capabilities"]["tee"]


@pytest.mark.parametrize(
    ("architecture", "requested", "compatible"),
    [
        ("sm89", "cuda_sm80", True),
        ("sm80", "cuda_sm89", False),
        ("sm90", "cuda_sm80", False),
        ("sm100", "cuda_sm90", False),
    ],
)
def test_cuda_baseline_variant_compatibility_does_not_cross_major_generations(
    monkeypatch, architecture, requested, compatible
):
    cap = HardwareCapabilities(
        vendor="NVIDIA",
        device="diagnostic",
        architecture=architecture,
        target_id="cuda_" + architecture,
        available=True,
    )
    monkeypatch.setattr("aether.backends.hardware_detector.detect_cuda_devices", lambda: [cap])
    backend = CUDABackend(requested)
    assert backend._cuda_available is compatible
    assert cap.execution_tested is False


def test_adapter_caches_optional_runtime_and_device_discovery(monkeypatch):
    backend, _, runtime, _ = adapter(monkeypatch, VENDORS[0], runtime_available=False)
    construction = Mock(return_value=runtime)
    monkeypatch.setattr("aether.backends.torch_backend.TorchBackend", construction)
    assert not backend.is_available()
    assert not backend.is_available()
    with pytest.raises(BackendError):
        backend.load_model("model")
    construction.assert_called_once()


def test_torch_unload_releases_model_and_session_state_without_importing_a_runtime(tmp_path):
    backend = object.__new__(TorchBackend)
    handle = CompiledAEGHandle("logical-id", tmp_path, {})
    handle.session_caches["session"] = (object(), object())
    handle.batched_engine = object()
    backend._models = {"logical-id": handle}
    backend._tokenizers = {"logical-id": object()}
    backend._placement = object()
    backend._placement_planner = object()
    backend._placement_devices = ["cuda:2"]
    backend._placement_failed = True
    backend._placement_forced = True
    backend.unload_model("logical-id")
    assert not backend._models
    assert not backend._tokenizers
    assert not handle.session_caches
    assert handle.batched_engine is None
    assert backend._placement is None
    assert not backend._placement_failed
    backend.unload()


@pytest.mark.parametrize(
    ("hip", "has_gpu", "target", "available"),
    [
        ("6.2", True, "amd_mi350x", True),
        ("6.2", True, "rocm_cdna5_mi455x", True),
        (None, True, "amd_mi350x", False),
        (None, True, "rocm_cdna5_mi455x", False),
        ("6.2", False, "amd_mi350x", False),
        ("6.2", False, "rocm_cdna5_mi455x", False),
        ("6.2", True, "cuda_sm89", False),
    ],
)
def test_torch_canonical_amd_aliases_require_hip_and_device_availability(
    monkeypatch, hip, has_gpu, target, available
):
    runtime = object.__new__(TorchBackend)
    module = SimpleNamespace(
        version=SimpleNamespace(hip=hip), cuda=SimpleNamespace(is_available=lambda: has_gpu)
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    assert runtime.available_for_target(target) is available


def test_native_cuda_incomplete_interface_is_explicit_and_skips_unusable_compilation(monkeypatch):
    toolchain = Mock(side_effect=AssertionError("must not compile an incomplete inference adapter"))
    monkeypatch.setattr("aether.kernels.native_cuda.detect_cuda_toolchain", toolchain)
    backend = NativeCUDABackend()
    assert not backend.is_available()
    assert "incomplete" in backend.unavailable_reason
    assert "upload_tensor" in backend.missing_execution_methods
    assert "native_cuda" not in backend.get_capabilities()
    assert "jit_compiled" not in backend.get_capabilities()
    assert backend.info.name == "aether_cuda"
    toolchain.assert_not_called()
    with pytest.raises(BackendError, match="incomplete"):
        backend.load_model("model", aeg_path="model.aeg")


def test_native_cuda_compile_failure_cannot_advertise_a_ready_engine(monkeypatch):
    class CompleteInterface:
        is_available = True  # property semantics, not a function
        build_error = "diagnostic compiler failure"

        def compile(self):
            return False

    for name in (
        "upload_tensor",
        "download_tensor",
        "rmsnorm",
        "sgemm",
        "rope",
        "flash_attn",
        "add",
        "swiglu",
    ):
        setattr(CompleteInterface, name, lambda *_args: None)
    monkeypatch.setattr("aether.kernels.native_cuda.NativeCUDAKernels", CompleteInterface)
    monkeypatch.setattr("aether.kernels.native_cuda.detect_cuda_toolchain", lambda: object())
    backend = NativeCUDABackend()
    assert not backend.is_available()
    assert "diagnostic compiler failure" in backend.unavailable_reason


def test_native_cuda_device_probe_does_not_import_or_misclassify_a_hip_framework(monkeypatch):
    monkeypatch.setattr("aether.kernels.native_cuda.detect_cuda_toolchain", lambda _target: None)
    monkeypatch.setattr("aether.backends.hardware_detector.detect_cuda_devices", lambda: [])
    kernels = NativeCUDAKernels("cuda_sm89")
    assert not kernels.is_available
    assert not kernels.compile()
    assert not kernels.compiled
    assert "nvcc" in kernels.build_error


def test_nvcc_discovery_skips_directories_and_uses_cuda_home_for_the_requested_sm(
    monkeypatch, tmp_path
):
    stale_root = tmp_path / "missing compiler"
    stale_root.mkdir()
    sdk_root = tmp_path / "toolkit with spaces"
    executable = sdk_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    executable.parent.mkdir(parents=True)
    executable.write_text("diagnostic executable; subprocess is mocked")
    monkeypatch.setenv("CUDA_PATH", str(stale_root))
    monkeypatch.setenv("CUDA_HOME", str(sdk_root))
    monkeypatch.setattr("aether.kernels.native_cuda.shutil.which", lambda _name: None)
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("aether.kernels.native_cuda.subprocess.run", run)
    toolchain = detect_cuda_toolchain("cuda_sm89")
    assert toolchain.executable == str(executable)
    assert toolchain.arch_flags == ("-arch=sm_89",)
    run.assert_called_once_with(
        [str(executable), "--version"], capture_output=True, text=True, timeout=5
    )


def test_nvcc_discovery_continues_after_a_broken_path_candidate(monkeypatch, tmp_path):
    broken = tmp_path / "broken-nvcc"
    broken.write_text("diagnostic")
    root = tmp_path / "working"
    executable = root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    executable.parent.mkdir(parents=True)
    executable.write_text("diagnostic")
    monkeypatch.setenv("CUDA_PATH", str(root))
    monkeypatch.setattr("aether.kernels.native_cuda.shutil.which", lambda _name: str(broken))
    run = Mock(side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)])
    monkeypatch.setattr("aether.kernels.native_cuda.subprocess.run", run)
    assert detect_cuda_toolchain("cuda_sm90").executable == str(executable)
    assert run.call_count == 2


def test_cuda_build_command_never_has_an_orphan_windows_host_compiler_flag(tmp_path):
    toolchain = CUDACompilerToolchain("nvcc", "nvcc")
    command = toolchain.build_command(tmp_path / "source.cu", tmp_path / "kernel.dll")
    if os.name == "nt":
        assert "-Xcompiler" not in command
    else:
        assert command[command.index("-Xcompiler") + 1] == "-fPIC"
    assert "" not in command


def test_native_cuda_unknown_target_is_not_silently_compiled_as_sm80():
    with pytest.raises(ValueError, match="unsupported native CUDA target"):
        detect_cuda_toolchain("cuda_made_up")


def package(tmp_path, *, targets, portable=(), statuses=None, plans=None):
    # Import-time isolation tests can reload aeg_format between collection
    # and execution. Construct the package with the current public class.
    from aether.core.aeg_format import AEGPackage, KernelSetMetadata

    artifact = AEGPackage(tmp_path / "logical.aeg")
    artifact.manifest = SimpleNamespace(
        kernels=KernelSetMetadata(
            targets=list(targets),
            portable_backends=list(portable),
            variant_status=statuses or {target: "executable" for target in targets},
            backend_plans=plans or {},
        )
    )
    return artifact


def selector(backends):
    instance = object.__new__(BackendSelector)
    instance.registry = Mock()
    instance.registry.get_backend.side_effect = lambda name: backends.get(name)
    return instance


def ready_backend(name):
    return SimpleNamespace(name=name, available_for_target=Mock(return_value=True))


def test_selector_rejects_gpu_for_a_cpu_artifact_before_initializing_backends(tmp_path):
    selection = selector({"pytorch": ready_backend("pytorch")})
    artifact = package(tmp_path, targets=["cpu_avx2"])
    with pytest.raises(BackendNotAvailableError, match="no compatible execution contract"):
        selection.select_for_target_id("cuda_sm89", artifact)
    selection.registry.get_backend.assert_not_called()


def test_selector_rejects_plan_only_gpu_targets(tmp_path):
    selection = selector({"pytorch": ready_backend("pytorch")})
    artifact = package(tmp_path, targets=["cuda_sm89"], statuses={"cuda_sm89": "plan_only"})
    with pytest.raises(BackendNotAvailableError, match="no compatible execution contract"):
        selection.select_for_target_id("cuda_sm89", artifact)


def test_selector_only_materializes_an_aeg_through_its_portable_backend_contract(tmp_path):
    torch = ready_backend("pytorch")
    selection = selector({"vllm": ready_backend("vllm"), "pytorch": torch})
    artifact = package(tmp_path, targets=["cpu_avx2"], portable=["aether_cpu", "pytorch"])
    assert selection.select_for_target_id("cuda_sm89", artifact) is torch
    assert all(call.args[0] != "vllm" for call in selection.registry.get_backend.call_args_list)


def test_selector_prefers_cpu_artifact_contract_over_installed_frontend_plugins(tmp_path):
    native = ready_backend("aether_cpu")
    selection = selector({"llama.cpp": ready_backend("llamacpp"), "aether_cpu": native})
    assert (
        selection.select_for_target_id("cpu_avx2", package(tmp_path, targets=["cpu_avx2"]))
        is native
    )


def test_selector_honors_an_explicit_executable_variant_backend_plan(tmp_path):
    native = ready_backend("custom_cuda")
    selection = selector({"custom_cuda": native, "pytorch": ready_backend("pytorch")})
    artifact = package(tmp_path, targets=["cuda_sm89"], plans={"cuda_sm89": "custom_cuda"})
    assert selection.select_for_target_id("cuda_sm89", artifact) is native


def test_selector_without_an_artifact_preserves_profile_selection():
    torch = ready_backend("pytorch")
    selection = selector({"pytorch": torch})
    assert selection.select(HardwareProfile.from_target_id("cuda_sm89")) is torch
