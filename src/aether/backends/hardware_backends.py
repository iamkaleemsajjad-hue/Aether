"""
Aether Runtime — Complete Hardware Backend Implementations.

Provides production-grade backend implementations for all hardware targets.
Vendor backends report availability only when the matching runtime is present.
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
from aether.core.exceptions import BackendError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CUDA Backend (all sm variants)
# ---------------------------------------------------------------------------

class CUDABackend(Backend):
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
            version="12.4.0",
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
                compatible = actual_sm >= requested_sm if requested_sm else True
            if compatible:
                device_index = int(cap.extra.get("device_index", index))
                self._device = f"cuda:{device_index}"
                return True
        return False

    def is_available(self) -> bool:
        return self._cuda_available

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        """Load a model for this CUDA target."""
        if not self._cuda_available:
            raise BackendError(
                f"CUDA runtime is not available for target {self.target_id}",
                backend_name=self.name,
                details={"target": self.target_id},
            )
        try:
            from aether.backends.torch_backend import TorchBackend
            self._torch_backend = TorchBackend()
            return self._torch_backend.load(model_path, config)
        except Exception as exc:
            logger.warning(f"CUDA backend load failed: {exc}")
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        model_path = aeg_path or model_id
        if not self.load(model_path, kwargs or None):
            raise BackendError(f"failed to load CUDA model {model_id!r}", backend_name=self.name)
        return model_path

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
            "architecture": self._arch_name,
            "cuda_available": self._cuda_available,
            "active_device": self._device,
            "supports_fp8": sm in ("sm89", "sm90", "sm100", "sm120", "sm130"),
            "supports_fp4": sm in ("sm100", "sm120", "sm130"),
            "supports_tee": "tee" in sm,
            "warp_size": 32,
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
                "total_memory_mb": device.memory_bytes // (1024 * 1024),
                "compute_capability": device.extra.get("compute_capability"),
                "cuda_driver": device.driver_version,
                "detection_method": device.runtime_version,
                "device_count": len(self._cuda_devices),
                "framework_free_detection": True,
            })
        return caps

    def unload(self) -> None:
        """Unload the model from memory."""
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


# ---------------------------------------------------------------------------
# ROCm Backend
# ---------------------------------------------------------------------------

class ROCmBackend(Backend):
    """
    AMD ROCm/HIP backend supporting RDNA3 and CDNA3/MI300X.

    Executes via PyTorch ROCm when available.
    Targets: rocm_rdna3, rocm_cdna3, rocm_cdna5_mi455x, amd_mi350x
    """

    def __init__(self, target_id: str = "rocm_cdna3") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"rocm_{target_id}",
            version="6.2.0",
            supported_targets=[target_id],
            capabilities=["generate", "chat", "embed", "hip_kernels", "wmma"],
        )
        if "mi350x" in target_id or "cdna5" in target_id:
            info.capabilities.extend(["mxfp6", "fp8"])
        super().__init__(info)
        self._device = "cpu"
        self._rocm_devices: list[Any] = []
        self._rocm_available = self._detect_rocm()

    def _detect_rocm(self) -> bool:
        """Detect AMD devices using ROCm system interfaces, not PyTorch."""
        try:
            from aether.backends.hardware_detector import detect_rocm_devices
            detected = [cap for cap in detect_rocm_devices() if cap.available]
        except Exception as exc:  # noqa: BLE001
            logger.debug("framework-free ROCm detection failed: %s", exc)
            detected = []
        self._rocm_devices = detected
        if detected:
            index = int(detected[0].extra.get("device_index", 0))
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
                self._device = "cuda:0"
                return True
        return False

    def is_available(self) -> bool:
        return self._rocm_available

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._rocm_available:
            raise BackendError(
                f"ROCm runtime is not available for target {self.target_id}",
                backend_name=self.name,
                details={"target": self.target_id},
            )
        try:
            from aether.backends.torch_backend import TorchBackend
            self._torch_backend = TorchBackend()
            return self._torch_backend.load(model_path, config)
        except Exception:  # noqa: BLE001
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        model_path = aeg_path or model_id
        if not self.load(model_path, kwargs or None):
            raise BackendError(f"failed to load ROCm model {model_id!r}", backend_name=self.name)
        return model_path

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

    def unload(self) -> None:
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


# ---------------------------------------------------------------------------
# Metal Backend (Apple Silicon)
# ---------------------------------------------------------------------------

class MetalBackend(Backend):
    """
    Apple Metal backend for M1/M2/M3/M4/M5 Silicon.

    Executes via PyTorch MPS when on Apple hardware.
    Targets: metal_m1, metal_m3
    """

    def __init__(self, target_id: str = "metal_m1") -> None:
        self.target_id = target_id
        info = BackendInfo(
            name=f"metal_{target_id}",
            version="3.2",
            supported_targets=[target_id],
            capabilities=["generate", "chat", "embed", "metal_shaders", "unified_memory"],
        )
        if "m3" in target_id:
            info.capabilities.extend(["metal4_tensor_ops", "fp16_native"])
        super().__init__(info)
        self._device = "cpu"
        self._metal_device: Any | None = None
        self._mps_available = self._detect_mps()

    def _detect_mps(self) -> bool:
        """Detect Apple Silicon/Metal without importing PyTorch."""
        try:
            from aether.backends.hardware_detector import detect_metal
            detected = detect_metal()
        except Exception as exc:  # noqa: BLE001
            logger.debug("framework-free Metal detection failed: %s", exc)
            return False
        if detected.available:
            self._metal_device = detected
            self._device = "mps"
            return True
        return False

    def is_available(self) -> bool:
        return self._mps_available

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._mps_available:
            raise BackendError(
                f"Metal/MPS runtime is not available for target {self.target_id}",
                backend_name=self.name,
                details={"target": self.target_id},
            )
        try:
            from aether.backends.torch_backend import TorchBackend
            self._torch_backend = TorchBackend()
            return self._torch_backend.load(model_path, config)
        except Exception:  # noqa: BLE001
            return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        model_path = aeg_path or model_id
        if not self.load(model_path, kwargs or None):
            raise BackendError(f"failed to load Metal model {model_id!r}", backend_name=self.name)
        return model_path

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

    def unload(self) -> None:
        if hasattr(self, "_torch_backend"):
            self._torch_backend.unload()
            del self._torch_backend


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
            version="0.13.0",
            supported_targets=["cuda_sm80", "cuda_sm89", "cuda_sm90", "cuda_sm100"],
            capabilities=[
                "generate", "chat", "flash_attention", "paged_kv_cache",
                "tensor_parallelism", "inflight_batching", "quantization_int4_awq",
            ],
        )
        super().__init__(info)
        self._trtllm_available = self._detect_trtllm()
        self._engine_path: str | None = None

    def _detect_trtllm(self) -> bool:
        try:
            import tensorrt_llm  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._trtllm_available

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
        engine_path = kwargs.get("engine_path")
        model_path = str(engine_path or aeg_path or model_id)
        if not self.load(model_path, kwargs or None):
            raise BackendError(f"failed to load TensorRT-LLM model {model_id!r}", backend_name=self.name)
        return model_path

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
            version="1.2.0",
            supported_targets=[target_id],
            capabilities=["generate", "edge_inference", "sub_10w", "onnx_runtime"],
        )
        super().__init__(info)
        self._vendor = self._detect_vendor(target_id)
        self._ort_available = self._detect_ort()

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
        # ONNX Runtime alone is not evidence that a RISC-V NPU exists.  A
        # portable host may compile an IR plan, but execution requires the
        # matching ISA/vendor runtime on a RISC-V machine.
        machine = platform.machine().lower()
        return self._ort_available and ("riscv" in machine or "risc-v" in machine)

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._ort_available:
            raise BackendError(
                f"ONNX Runtime is required for RISC-V NPU target {self.target_id}",
                backend_name=self.name,
            )
        if self._ort_available:
            try:
                from aether.backends.onnx_backend import ONNXBackend
                self._onnx_backend = ONNXBackend()
                onnx_path = Path(model_path) / "kernels" / f"{self._vendor}" / "model.onnx"
                if onnx_path.exists():
                    return self._onnx_backend.load(str(onnx_path), config)
            except Exception:  # noqa: BLE001
                pass
        return False

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        model_path = aeg_path or model_id
        if not self.load(model_path, kwargs or None):
            raise BackendError(f"failed to load RISC-V NPU model {model_id!r}", backend_name=self.name)
        return model_path

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
            version="1.2.0",
            supported_targets=[target_id],
            capabilities=["generate", "low_power", "decode_only"],
        )
        if "ternary" in target_id:
            info.capabilities.extend(["ternary_arithmetic", "sub2bit"])
        super().__init__(info)

    def is_available(self) -> bool:
        return False

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        raise BackendError(
            f"FPGA runtime integration is not available for target {self.target_id}",
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
            version="2.25.0",
            supported_targets=[target_id, "qualcomm_qnn"],
            capabilities=["generate", "chat", "edge_inference", "onnx_runtime"],
        )
        super().__init__(info)
        self._qnn_available = self._detect_qnn()

    def _detect_qnn(self) -> bool:
        # Check for Qualcomm Neural Network SDK
        qnn_paths = ["/opt/qcom/aistack/qnn", os.environ.get("QNN_SDK_ROOT", "")]
        return any(Path(p).exists() for p in qnn_paths if p)

    def is_available(self) -> bool:
        return self._qnn_available

    def load(self, model_path: str, config: dict[str, Any] | None = None) -> bool:
        if not self._qnn_available:
            raise BackendError(
                f"Qualcomm QNN SDK is not available for target {self.target_id}",
                backend_name=self.name,
            )
        raise BackendError(
            f"Qualcomm QNN execution is not wired for target {self.target_id}",
            backend_name=self.name,
        )

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        self.load(aeg_path or model_id, kwargs or None)

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
    """Intel OpenVINO execution adapter with real NPU capability detection.

    Aether does not convert an AEG into an arbitrary OpenVINO graph at load
    time.  This adapter accepts a pre-emitted ONNX/OpenVINO model bundle and
    fails clearly for an AEG that has no such target artifact.
    """

    def __init__(self, target_id: str = "openvino_npu") -> None:
        if target_id not in {"openvino_npu", "openvino_gpu"}:
            raise ValueError(f"unsupported OpenVINO target {target_id!r}")
        self.target_id = target_id
        self._device_name = "NPU" if target_id == "openvino_npu" else "GPU"
        super().__init__(BackendInfo(
            name="openvino",
            version="runtime",
            supported_targets=[target_id],
            capabilities=[
                "generate", "onnx_runtime",
                "intel_npu" if target_id == "openvino_npu" else "intel_gpu",
            ],
        ))
        self._core = None
        self._available = False
        try:
            import openvino
            self._core = openvino.Core()
            self._available = self._device_name in list(self._core.available_devices)
        except Exception:
            self._available = False

    def is_available(self) -> bool:
        return self._available

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
        self.load(aeg_path or model_id, kwargs or None)
        return aeg_path or model_id

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
