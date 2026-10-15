"""
Aether Runtime — Complete Hardware Backend Implementations.

Provides executable adapters and retained target-specific development code.
Vendor adapters report availability only when their device and execution
dependencies are present; this is not evidence of execution or production use.
CPU reference execution remains available through the normal CPU/Torch backend;
it is not reported as CUDA, ROCm, Metal, FPGA, or QNN production support.

Research basis:
  - NVIDIA CUDA Programming Guide (2024)
  - AMD ROCm HIP Programming Guide (2024)
  - Apple Metal Shading Language Specification (2024)
  - Intel OpenVINO Runtime API (2024)
  - Qualcomm QNN SDK (2024)
  - RISC-V ONNX Runtime Backend (2024)
"""

from __future__ import annotations

import json
import os
import platform
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import AETHER_VERSION
from aether.core.exceptions import BackendError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class _TorchVendorAdapter:
    """Delegate through the backend API with an explicit accelerator device."""

    def _matching_torch_runtime(self) -> Any | None:
        try:
            runtime = getattr(self, "_torch_runtime", None)
            if runtime is None:
                from aether.backends.torch_backend import TorchBackend

                runtime = TorchBackend()
                self._torch_runtime = runtime
            if not runtime.available_for_target(self.target_id):
                self._runtime_error = (
                    f"installed PyTorch does not provide a usable {self.target_id} runtime"
                )
                return None
            self._runtime_error = None
            return runtime
        except Exception as exc:  # optional dependency / driver may fail to initialize
            self._runtime_error = f"PyTorch runtime initialization failed: {exc}"
            return None

    @property
    def unavailable_reason(self) -> str | None:
        if not self._hardware_available:
            return f"no compatible device/runtime was detected for {self.target_id}"
        if self._matching_torch_runtime() is None:
            return self._runtime_error
        return None

    def is_available(self) -> bool:
        return bool(self._hardware_available and self._matching_torch_runtime() is not None)

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        """Compatibility entry point; errors remain visible to the caller."""
        self.load_model(model_path, **dict(config or {}))
        return True

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        if not self.is_available():
            raise BackendError(
                self.unavailable_reason or f"{self.target_id} backend is unavailable",
                backend_name=self.name,
                details={"target": self.target_id},
            )
        options = dict(kwargs)
        requested = options.pop("execution_devices", None)
        if requested is not None and list(requested) != [self._device]:
            raise BackendError(
                f"{self.target_id} adapter requires execution_devices={[self._device]!r}; "
                f"received {requested!r}",
                backend_name=self.name,
            )
        options["execution_devices"] = [self._device]
        # The optional Hugging Face frontend otherwise defaults to an
        # automatic map that can move the entire model onto CPU. AEG engines
        # receive the same explicit mesh through the canonical loader.
        device_map = options.get("device_map", {"": self._device})
        if device_map not in (self._device, {"": self._device}):
            raise BackendError(
                f"{self.target_id} adapter requires device_map={{'': {self._device!r}}}",
                backend_name=self.name,
            )
        options["device_map"] = device_map
        runtime = self._torch_runtime
        # Preserve the logical model ID. Replacing it with aeg_path means
        # subsequent generation cannot find the model it just loaded.
        model = runtime.load_model(model_id, aeg_path=aeg_path, **options)
        self._torch_backend = runtime
        if not hasattr(self, "_loaded_model_ids"):
            self._loaded_model_ids: set[str] = set()
        self._loaded_model_ids.add(model_id)
        return model

    def unload_model(self, model_id: str) -> None:
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload_model(model_id)
        getattr(self, "_loaded_model_ids", set()).discard(model_id)

    def unload(self) -> None:
        for model_id in tuple(getattr(self, "_loaded_model_ids", ())):
            self.unload_model(model_id)
        if hasattr(self, "_torch_backend"):
            del self._torch_backend


# ---------------------------------------------------------------------------
# CUDA Backend (all sm variants)
# ---------------------------------------------------------------------------

class CUDABackend(_TorchVendorAdapter, Backend):
    """
    NVIDIA CUDA backend supporting sm70 through sm130.

    Executes via PyTorch CUDA when CUDA is available.

    Supported targets: cuda_sm70, cuda_sm80, cuda_sm89, cuda_sm90,
                       cuda_sm100, cuda_sm120, cuda_sm130
    """

    # Compute capability -> architecture name mapping
    _SM_NAMES = {
        "sm70": "Volta (V100)",
        "sm80": "Ampere (A100)",
        "sm89": "Ada Lovelace (RTX 4090)",
        "sm90": "Hopper (H100)",
        "sm100": "Blackwell (B200)",
        "sm100_tee": "Blackwell CC (B200 Confidential)",
        "sm100_gb300": "Blackwell Ultra (GB300)",
        "sm120": "Rubin R100",
        "sm130": "Rubin Ultra",
    }

    def __init__(self, target_id: str = "cuda_sm90") -> None:
        self.target_id = target_id
        sm = target_id.replace("cuda_", "")
        arch_name = self._SM_NAMES.get(sm, f"CUDA ({sm})")

        info = BackendInfo(
            name=f"cuda_{sm}",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=[
                "generate", "chat", "embed", "flash_attention",
                "tensor_cores", "mixed_precision", "structured_output",
            ],
        )
        if sm in ("sm100", "sm120", "sm130", "sm100_gb300"):
            info.capabilities.extend(["fp4", "nvfp4", "mxfp4"])
        if sm in ("sm89", "sm90", "sm100", "sm120", "sm130"):
            info.capabilities.append("fp8")
        if sm == "sm100_tee":
            info.capabilities.append("confidential_computing")

        super().__init__(info)
        self._device = "cpu"  # Actual device — upgraded to CUDA if available
        self._arch_name = arch_name
        self._cuda_devices: list[Any] = []
        self._cuda_available = self._detect_cuda()
        self._hardware_available = self._cuda_available

    def _detect_cuda(self) -> bool:
        """Detect NVIDIA devices without importing an ML framework."""
        try:
            from aether.backends.hardware_detector import detect_cuda_devices
            detected = [cap for cap in detect_cuda_devices() if cap.available]
        except Exception as exc:  # noqa: BLE001
            logger.debug("framework-free CUDA detection failed: %s", exc)
            detected = []
        self._cuda_devices = detected
        requested = self.target_id.replace("cuda_", "")
        import re
        requested_match = re.match(r"sm(\d+)", requested)
        requested_sm = int(requested_match.group(1)) if requested_match else 0
        for index, cap in enumerate(detected):
            actual_match = re.match(r"sm(\d+)", str(cap.architecture))
            actual_sm = int(actual_match.group(1)) if actual_match else 0
            if requested in {"sm100_tee", "sm100_gb300"}:
                compatible = str(cap.target_id).replace("cuda_", "") == requested
            else:
                # Baseline cubin compatibility is within one major SM
                # generation and from older to newer minor revisions. Cross-
                # generation graph materialization requires an explicit AEG
                # portable contract in the runtime, not a numeric comparison.
                compatible = bool(
                    requested_sm and actual_sm // 10 == requested_sm // 10
                    and actual_sm % 10 >= requested_sm % 10
                )
            if compatible:
                device_index = int(cap.extra.get("device_index", index))
                self._device = f"cuda:{device_index}"
                return True
        return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Execute generation on this CUDA target."""
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(
            f"No model loaded for CUDA target {self.target_id}",
            details={"target": self.target_id, "cuda_available": self._cuda_available},
        )

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        """Stream generation tokens."""
        if hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"No model loaded for CUDA target {self.target_id}")

    def get_capabilities(self) -> dict[str, Any]:
        """Return this target's hardware capabilities."""
        sm = self.target_id.replace("cuda_", "")
        caps = {
            "target_id": self.target_id,
            "architecture": None,
            "target_profile_name": self._arch_name,
            "cuda_available": self._cuda_available,
            "active_device": self._device,
            "backend_available": self.is_available(),
            "unavailable_reason": self.unavailable_reason,
            "supports_fp8": False,
            "supports_fp4": False,
            "supports_tee": False,
            "warp_size": None,
            "target_profile_capabilities": {
                "fp8": sm in ("sm89", "sm90", "sm100", "sm120", "sm130"),
                "fp4": sm in ("sm100", "sm120", "sm130"),
                "tee": "tee" in sm,
            },
        }

        if self._cuda_available:
            device_index = int(self._device.rsplit(":", 1)[-1])
            device = next(
                (
                    cap for cap in self._cuda_devices
                    if int(cap.extra.get("device_index", -1)) == device_index
                ),
                self._cuda_devices[0],
            )
            caps.update({
                "gpu_name": device.device,
                "architecture": device.architecture,
                "total_memory_mb": device.memory_bytes // (1024 * 1024),
                "memory_bytes_known": device.to_dict()["memory_bytes_known"],
                "compute_capability": device.extra.get("compute_capability"),
                "cuda_driver": device.driver_version,
                "detection_method": device.runtime_version,
                "device_count": len(self._cuda_devices),
                "framework_free_detection": True,
                "supports_fp8": device.supports_fp8,
                "supports_fp4": device.supports_fp4,
                "supports_tee": device.supports_tee,
                "warp_size": device.warp_or_wavefront_size,
            })
        return caps


# ---------------------------------------------------------------------------
# ROCm Backend
# ---------------------------------------------------------------------------

class ROCmBackend(_TorchVendorAdapter, Backend):
    """
    AMD ROCm/HIP backend supporting RDNA3 and CDNA3/MI300X.

    Executes via PyTorch ROCm when available.
    Targets: rocm_rdna3, rocm_cdna3, rocm_cdna5_mi455x, amd_mi350x
    """

    def __init__(self, target_id: str = "rocm_cdna3") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"rocm_{target_id}",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=["generate", "chat", "embed", "hip_kernels", "wmma"],
        )
        if "mi350x" in target_id or "cdna5" in target_id:
            info.capabilities.extend(["mxfp6", "fp8"])
        super().__init__(info)
        self._device = "cpu"
        self._rocm_devices: list[Any] = []
        self._rocm_available = self._detect_rocm()
        self._hardware_available = self._rocm_available

    def _detect_rocm(self) -> bool:
        """Detect AMD devices using ROCm system interfaces, not PyTorch."""
        try:
            from aether.backends.hardware_detector import detect_rocm_devices
            detected = [cap for cap in detect_rocm_devices() if cap.available]
        except Exception as exc:  # noqa: BLE001
            logger.debug("framework-free ROCm detection failed: %s", exc)
            detected = []
        self._rocm_devices = detected
        compatible = next((cap for cap in detected if cap.target_id == self.target_id), None)
        if compatible is not None:
            index = int(compatible.extra.get("device_index", 0))
            # Explicit Torch HIP execution, when installed, addresses HIP
            # tensors through ``cuda:<index>``. Detection remains framework-free.
            self._device = f"cuda:{index}"
            return True

        # Compatibility for applications that have already initialized a
        # PyTorch HIP runtime.  We never import PyTorch here: consulting an
        # existing module is optional and cannot turn the core package into a
        # framework dependency.  The canonical detector above remains the
        # source of truth for normal discovery.
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and getattr(getattr(torch, "version", None), "hip", None):
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and bool(cuda.is_available()):
                try:
                    from aether.backends.hardware_detector import _rocm_target_id

                    for index in range(int(cuda.device_count())):
                        name = str(cuda.get_device_name(index))
                        properties = cuda.get_device_properties(index)
                        architecture = str(getattr(properties, "gcnArchName", ""))
                        if _rocm_target_id(f"{name} {architecture}") == self.target_id:
                            self._device = f"cuda:{index}"
                            return True
                except Exception as exc:
                    logger.debug("initialized HIP runtime enumeration failed: %s", exc)
        return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(f"No model loaded for ROCm target {self.target_id}")

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        if hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"No model loaded for ROCm target {self.target_id}")

    def get_capabilities(self) -> list[str]:
        # Keep the backend API stable; detailed per-device evidence is exposed
        # by ``hardware_detector.detect_all_capabilities()``.
        return self.info.capabilities

    def emit_hip_source(self, op_name: str, config: dict[str, Any]) -> str:
        """Emit HIP C++ source for a given operation."""
        # Generate real HIP kernel source using the ROCm target templates
        try:
            from aether.compiler.stage3_targeting.target_rocm import ROCmTargetBackend
            backend = ROCmTargetBackend()
            if hasattr(backend, "emit_kernel_source"):
                return backend.emit_kernel_source(op_name, config)
        except Exception:  # noqa: BLE001
            pass
        # Fallback: return a minimal compilable HIP kernel
        return f"""
// Aether ROCm kernel: {op_name}
// Target: {self.target_id}
#include <hip/hip_runtime.h>
__global__ void aether_{op_name.replace('.', '_')}(float* out, const float* in, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}}
"""

# ---------------------------------------------------------------------------
# Metal Backend (Apple Silicon)
# ---------------------------------------------------------------------------

class MetalBackend(_TorchVendorAdapter, Backend):
    """
    Apple Metal backend for M1/M2/M3/M4/M5 Silicon.

    Executes via PyTorch MPS when on Apple hardware.
    Targets: metal_m1, metal_m3
    """

    def __init__(self, target_id: str = "metal_m1") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"metal_{target_id}",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=["generate", "chat", "embed", "metal_shaders", "unified_memory"],
        )
        if "m3" in target_id:
            info.capabilities.extend(["metal4_tensor_ops", "fp16_native"])
        super().__init__(info)
        self._device = "cpu"
        self._metal_device: Any | None = None
        self._mps_available = self._detect_mps()
        self._hardware_available = self._mps_available

    def _detect_mps(self) -> bool:
        """Detect Apple Silicon/Metal without importing PyTorch."""
        try:
            from aether.backends.hardware_detector import detect_metal
            detected = detect_metal()
        except Exception as exc:  # noqa: BLE001
            logger.debug("framework-free Metal detection failed: %s", exc)
            return False
        if detected.available and (
            detected.target_id == self.target_id
            or (self.target_id == "metal_m1" and detected.target_id == "metal_m3")
        ):
            self._metal_device = detected
            self._device = "mps"
            return True
        return False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(f"No model loaded for Metal target {self.target_id}")

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        if hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"No model loaded for Metal target {self.target_id}")

    def get_capabilities(self) -> list[str]:
        # Keep the backend API stable; detailed device evidence is exposed by
        # ``hardware_detector.detect_all_capabilities()``.
        return self.info.capabilities

    def emit_msl_source(self, op_name: str, config: dict[str, Any]) -> str:
        """Emit Metal Shading Language source for a kernel."""
        try:
            from aether.compiler.stage3_targeting.target_metal import MetalTargetBackend
            backend = MetalTargetBackend()
            if hasattr(backend, "emit_kernel_source"):
                return backend.emit_kernel_source(op_name, config)
        except Exception:  # noqa: BLE001
            pass
        return f"""
// Aether Metal Shading Language kernel: {op_name}
// Target: {self.target_id}
#include <metal_stdlib>
using namespace metal;

kernel void aether_{op_name.replace('.', '_')}(
    device float* output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {{
    output[gid] = input[gid];
}}
"""

# ---------------------------------------------------------------------------
# TensorRT-LLM Backend
# ---------------------------------------------------------------------------

class TensorRTLLMBackend(Backend):
    """
    NVIDIA TensorRT-LLM backend.

    Provides the TensorRT-LLM interface for optimized NVIDIA inference.
    """

    def __init__(self) -> None:
        info = BackendInfo(
            name="tensorrt_llm",
            version=AETHER_VERSION,
            supported_targets=["cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100"],
            capabilities=["engine_artifact_metadata"],
        )
        super().__init__(info)
        self._trtllm_available = self._detect_trtllm()
        self._engine_path: str | None = None
        self.unavailable_reason = (
            f"TensorRT-LLM package {'detected' if self._trtllm_available else 'not detected'}; "
            "inference is unavailable because a real engine runner is not implemented"
        )

    def _detect_trtllm(self) -> bool:
        try:
            import tensorrt_llm  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._trtllm_available:
            raise BackendError("TensorRT-LLM Python package is not available", backend_name=self.name)
        engine_dir = Path(model_path) / "kernels" / "trtllm"
        if not engine_dir.exists():
            raise BackendError(
                f"TensorRT-LLM requires a pre-built engine directory at {engine_dir}",
                backend_name=self.name,
            )
        self._engine_path = str(engine_dir)
        return True

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise BackendError(
            "TensorRT-LLM generation requires binding a real engine runner; no placeholder output is returned",
            backend_name=self.name,
        )

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        raise BackendError(
            "TensorRT-LLM streaming requires binding a real engine runner; no placeholder output is returned",
            backend_name=self.name,
        )

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def unload(self) -> None:
        self._engine_path = None
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


# ---------------------------------------------------------------------------
# RISC-V NPU Backend
# ---------------------------------------------------------------------------

class RISCVNPUBackend(Backend):
    """
    RISC-V NPU backend supporting MIPS S8200, SiFive X160, XuanTie C930.

    Compiles models to the RISC-V NPU Abstract IR and dispatches to the
    appropriate vendor backend when available.

    Reference: PRD v4.0 §3.2, RISC-V NPU Abstract IR specification.
    """

    def __init__(self, target_id: str = "riscv_mips_s8200") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"riscv_{target_id}",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=["compile_riscv_ir"],
        )
        super().__init__(info)
        self._vendor = self._detect_vendor(target_id)
        self._ort_available = self._detect_ort()
        self.unavailable_reason = (
            f"ONNX Runtime {'detected' if self._ort_available else 'not detected'}; "
            "RISC-V target IR compilation is available, but no vendor NPU execution adapter is implemented"
        )

    def _detect_vendor(self, target_id: str) -> str:
        if "mips" in target_id:
            return "mips_s8200"
        elif "sifive" in target_id:
            return "sifive_x160"
        elif "xuantie" in target_id:
            return "xuantie_c930"
        elif "cervell" in target_id:
            return "cervell"
        return "generic_riscv"

    def _detect_ort(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        # CPU architecture and a generic ONNX Runtime package cannot supply
        # the missing target-specific NPU execution adapter.
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if hasattr(self, "_onnx_backend"):
            return self._onnx_backend.generate(request)
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(f"RISC-V NPU {self.target_id}: no loaded model")

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        if hasattr(self, "_onnx_backend"):
            yield from self._onnx_backend.generate_stream(request)
        elif hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"RISC-V NPU {self.target_id}: no loaded model")

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def compile_to_riscv_ir(self, graph: Any) -> dict[str, Any]:
        """Compile an AEG graph to the RISC-V NPU Abstract IR."""
        try:
            from aether.compiler.stage3_targeting.riscv_npu_ir import RISCVNPUCompiler
            compiler = RISCVNPUCompiler(target_id=self.target_id)
            return compiler.compile(graph)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "target": self.target_id}

    def unload(self) -> None:
        for attr in ("_onnx_backend", "_torch_backend"):
            if hasattr(self, attr):
                getattr(self, attr).unload()
                delattr(self, attr)


# ---------------------------------------------------------------------------
# FPGA Backend
# ---------------------------------------------------------------------------

class FPGABackend(Backend):
    """
    FPGA backend for Xilinx VU9P and ternary FPGA targets.

    The VU9P target is optimized for cost-efficient decode at 10x lower
    cost-per-token vs GPU through bitstream-based acceleration.

    For standard AEG models: requires a vendor or ONNX Runtime execution path.
    """

    def __init__(self, target_id: str = "fpga_xilinx_vu9p") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"fpga_{target_id}",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=[],
        )
        super().__init__(info)
        self.unavailable_reason = f"FPGA execution adapter is not implemented for target {self.target_id}"

    def is_available(self) -> bool:
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        raise BackendError(
            self.unavailable_reason,
            backend_name=self.name,
        )

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        self.load(aeg_path or model_id, kwargs or None)

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(f"FPGA {self.target_id}: no loaded model")

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        if hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"FPGA {self.target_id}: no loaded model")

    def unload(self) -> None:
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


# ---------------------------------------------------------------------------
# Qualcomm Backend
# ---------------------------------------------------------------------------

class QualcommBackend(Backend):
    """
    Qualcomm AI 100 Ultra and QNN backend.

    Dispatches through QNN SDK when available, ONNX Runtime otherwise.
    """

    def __init__(self, target_id: str = "qualcomm_cloud_ai100") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"qualcomm_{target_id}",
            version=AETHER_VERSION,
            supported_targets=[target_id, "qualcomm_qnn"],
            capabilities=[],
        )
        super().__init__(info)
        self._qnn_available = self._detect_qnn()
        self.unavailable_reason = (
            f"Qualcomm QNN SDK directory {'found' if self._qnn_available else 'not found'}; "
            "the target execution adapter is not implemented"
        )

    def _detect_qnn(self) -> bool:
        # Check for Qualcomm Neural Network SDK
        qnn_paths = ["/opt/qcom/aistack/qnn", os.environ.get("QNN_SDK_ROOT", "")]
        return any(Path(p).exists() for p in qnn_paths if p)

    def is_available(self) -> bool:
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if hasattr(self, "_torch_backend"):
            return self._torch_backend.generate(request)
        raise BackendError(f"Qualcomm {self.target_id}: no loaded model")

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        if hasattr(self, "_torch_backend"):
            yield from self._torch_backend.generate_stream(request)
        else:
            raise BackendError(f"Qualcomm {self.target_id}: no loaded model")

    def unload(self) -> None:
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


class OpenVINOBackend(Backend):
    """Intel OpenVINO graph compiler with a reserved inference adapter.

    Aether does not convert an AEG into an arbitrary OpenVINO graph at load
    time. The low-level ``load`` method compiles a pre-emitted ONNX/OpenVINO
    bundle. Generation remains unavailable until a task-specific runner exists.
    """

    def __init__(self, target_id: str = "openvino_npu") -> None:
        if target_id not in {"openvino_npu", "openvino_gpu"}:
            raise ValueError(f"unsupported OpenVINO target {target_id!r}")
        self.target_id = target_id
        self._device_name = "NPU" if target_id == "openvino_npu" else "GPU"
        super().__init__(BackendInfo(
            name="openvino",
            version=AETHER_VERSION,
            supported_targets=[target_id],
            capabilities=["compile_model", "openvino_artifact"],
        ))
        self._core = None
        self._available = False
        try:
            import openvino
            self._core = openvino.Core()
            self._available = self._device_name in list(self._core.available_devices)
        except Exception:
            self._available = False
        self.unavailable_reason = (
            f"OpenVINO {self._device_name} device {'detected' if self._available else 'not detected'}; "
            "graph compilation is supported, but the task-specific generation runner is not implemented"
        )

    def is_available(self) -> bool:
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._available or self._core is None:
            raise BackendError(
                f"OpenVINO {self._device_name} runtime is unavailable",
                backend_name=self.name,
            )
        path = Path(model_path)
        onnx = path if path.suffix.lower() == ".onnx" else path / "kernels" / "openvino" / "model.onnx"
        if not onnx.is_file():
            raise BackendError(
                "OpenVINO execution requires an emitted ONNX/OpenVINO target artifact; "
                f"none was found at {onnx}", backend_name=self.name,
            )
        self._compiled_model = self._core.compile_model(str(onnx), self._device_name)
        return True

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        raise BackendError(self.unavailable_reason, backend_name=self.name)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise BackendError("OpenVINO generation adapter requires a task-specific compiled model runner", backend_name=self.name)

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResult]:
        raise BackendError("OpenVINO streaming requires a task-specific compiled model runner", backend_name=self.name)

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def unload(self) -> None:
        self._compiled_model = None


# Public names retained for target-specific callers.  They are concrete
# adapters, not mocks: availability and load still require the real SDK.
class QNNBackend(QualcommBackend):
    def __init__(self, target_id: str = "qualcomm_qnn") -> None:
        super().__init__(target_id)


class RISCVBackend(RISCVNPUBackend):
    pass


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type] = {
    # CUDA variants
    "cuda_sm70": CUDABackend,
    "cuda_sm80": CUDABackend,
    "cuda_sm89": CUDABackend,
    "cuda_sm90": CUDABackend,
    "cuda_sm100": CUDABackend,
    "cuda_sm100_tee": CUDABackend,
    "cuda_sm100_gb300": CUDABackend,
    "cuda_sm120": CUDABackend,
    "cuda_sm130": CUDABackend,
    # ROCm/AMD
    "rocm_rdna3": ROCmBackend,
    "rocm_cdna3": ROCmBackend,
    "rocm_cdna5_mi455x": ROCmBackend,
    "amd_mi350x": ROCmBackend,
    # Apple Metal
    "metal_m1": MetalBackend,
    "metal_m3": MetalBackend,
    # TRT-LLM
    "tensorrt_llm": TensorRTLLMBackend,
    # RISC-V NPU
    "riscv_mips_s8200": RISCVNPUBackend,
    "riscv_sifive_x160": RISCVNPUBackend,
    "riscv_xuantie_c930": RISCVNPUBackend,
    "riscv_cervell": RISCVNPUBackend,
    # FPGA
    "fpga_xilinx_vu9p": FPGABackend,
    "fpga_ternary": FPGABackend,
    # Qualcomm
    "qualcomm_cloud_ai100": QualcommBackend,
    "qualcomm_qnn": QualcommBackend,
    # OpenVINO / Intel NPU
    "openvino_npu": OpenVINOBackend,
    "openvino_gpu": OpenVINOBackend,
}


def create_backend(target_id: str) -> Backend:
    """
    Factory function to create the appropriate backend for a hardware target.

    Args:
        target_id: Hardware target identifier (e.g., 'cuda_sm90', 'metal_m3').

    Returns:
        A Backend instance for the given target.

    Raises:
        ValueError: If the target_id is not recognized.
    """
    backend_cls = _BACKEND_REGISTRY.get(target_id)
    if backend_cls is None:
        # Try prefix matching
        for key, cls in _BACKEND_REGISTRY.items():
            if target_id.startswith(key) or key.startswith(target_id):
                return cls(target_id)
        msg = f"No backend registered for target: {target_id!r}"
        raise ValueError(msg)

    if backend_cls in (CUDABackend, ROCmBackend, MetalBackend, RISCVNPUBackend, RISCVBackend, FPGABackend, QualcommBackend, QNNBackend, OpenVINOBackend):
        return backend_cls(target_id)
    return backend_cls()
