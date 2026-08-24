"""
Aether core type definitions — DType, precision, hardware profiles, tensor shapes.

This module defines the foundational type classes and enums that are used
throughout the Aether compiler and runtime. These types establish a shared
vocabulary for tensor shapes, data types, precision tiers, hardware targets,
memory layouts, and sharding plans.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


class DType(enum.Enum):
    """Supported tensor data types in AEG-IR.

    AEG-IR models are typically represented in BF16 or FP16 at runtime, but
    individual operations or layers may use FP8, INT4, INT8, or other formats
    specified by the precision map.
    """

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8 = "fp8"
    FP4 = "fp4"
    """4-bit floating point (NVFP4 / MXFP4 family)."""
    MXFP6 = "mxfp6"
    """6-bit floating point — AMD MI455X CDNA5 format (between FP8 and FP4)."""
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    BOOL = "bool"
    FLOAT = "float32"
    # Sub-2-bit formats — BitNet b1.58 / BTC-LLM / NanoQuant (PRD v5.0 Pass 19)
    TERNARY = "ternary"
    """1.58-bit ternary weights {-1, 0, +1} — BitNet b1.58 format.  Stored packed
    as 2 bits per weight: 00=-1, 01=0, 10=+1.  Requires addition-only kernels."""
    BINARY = "binary"
    """~0.8-1.11-bit binary codebook format — BTC-LLM / NanoQuant.  Weights stored
    as indices into a learned binary codebook."""

    def byte_size(self) -> int:
        """Return the number of bytes per element of this dtype.

        For sub-2-bit types (TERNARY, BINARY) this returns the packed storage
        byte size.  Actual effective bits per weight are fractional; callers
        that need fractional precision should use `bits_per_element()` instead.
        """
        mapping = {
            DType.BF16: 2,
            DType.FP16: 2,
            DType.FP32: 4,
            DType.FP8: 1,
            DType.FP4: 1,      # two FP4 values packed per byte
            DType.MXFP6: 1,    # ceiling; actual: 0.75 bytes per element
            DType.INT4: 1,
            DType.INT8: 1,
            DType.INT16: 2,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.UINT8: 1,
            DType.UINT16: 2,
            DType.BOOL: 1,
            DType.FLOAT: 4,
            DType.TERNARY: 1,  # 4 ternary values packed per byte (2 bits each)
            DType.BINARY: 1,   # codebook index; size depends on codebook
        }
        return mapping[self]

    def is_floating_point(self) -> bool:
        """Return True if this dtype is a floating-point type."""
        return self in (DType.BF16, DType.FP16, DType.FP32, DType.FP8, DType.FP4, DType.MXFP6, DType.FLOAT)

    def is_integer(self) -> bool:
        """Return True if this dtype is an integer type."""
        return self in (DType.INT4, DType.INT8, DType.INT16, DType.INT32, DType.INT64, DType.UINT8, DType.UINT16)

    def is_quantized(self) -> bool:
        """Return True if this dtype represents a quantized format."""
        return self in (DType.INT4, DType.INT8, DType.TERNARY, DType.BINARY)

    def is_sub2bit(self) -> bool:
        """Return True if this dtype is a sub-2-bit quantized format (PRD v5.0)."""
        return self in (DType.TERNARY, DType.BINARY)

    def bits_per_element(self) -> float:
        """Return the effective number of bits per element for this dtype.

        For sub-2-bit types this returns the true fractional bit count rather
        than the packed storage size returned by `byte_size()`.
        """
        mapping: dict[DType, float] = {
            DType.BF16: 16.0,
            DType.FP16: 16.0,
            DType.FP32: 32.0,
            DType.FP8: 8.0,
            DType.FP4: 4.0,
            DType.MXFP6: 6.0,
            DType.INT4: 4.0,
            DType.INT8: 8.0,
            DType.INT16: 16.0,
            DType.INT32: 32.0,
            DType.INT64: 64.0,
            DType.UINT8: 8.0,
            DType.UINT16: 16.0,
            DType.BOOL: 1.0,
            DType.FLOAT: 32.0,
            DType.TERNARY: 1.58,  # log2(3) — information content of ternary symbol
            DType.BINARY: 1.0,    # 1 bit per codebook-selected weight
        }
        return mapping[self]

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(value: str) -> DType:
        """Parse a dtype from its string representation (case-insensitive)."""
        normalized = value.lower().strip().replace("-", "").replace("_", "")
        mapping: dict[str, DType] = {
            "bf16": DType.BF16,
            "bfloat16": DType.BF16,
            "fp16": DType.FP16,
            "float16": DType.FP16,
            "fp32": DType.FP32,
            "float32": DType.FP32,
            "float": DType.FP32,
            "fp8": DType.FP8,
            "float8": DType.FP8,
            "fp4": DType.FP4,
            "float4": DType.FP4,
            "mxfp6": DType.MXFP6,
            "int4": DType.INT4,
            "int8": DType.INT8,
            "int16": DType.INT16,
            "int32": DType.INT32,
            "int64": DType.INT64,
            "uint8": DType.UINT8,
            "uint16": DType.UINT16,
            "bool": DType.BOOL,
            "ternary": DType.TERNARY,
            "bitnet": DType.TERNARY,
            "int158": DType.TERNARY,
            "binary": DType.BINARY,
            "btc": DType.BINARY,
        }
        result = mapping.get(normalized)
        if result is None:
            msg = f"Unknown dtype: {value!r}. Valid values: {sorted(mapping)}"
            raise ValueError(msg)
        return result


class Precision(enum.Enum):
    """Precision format identifiers for mixed-precision quantization.

    These correspond to the precision formats used in Aether's sensitivity-guided
    mixed-precision assignment and in the final AEG weight blob.
    """

    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"
    FP8 = "FP8"
    FP8_E4M3 = "FP8_E4M3"
    FP8_E5M2 = "FP8_E5M2"
    FP4 = "FP4"
    NVFP4 = "NVFP4"
    MXFP4 = "MXFP4"
    MXFP6 = "MXFP6"
    """6-bit floating point — AMD MI455X CDNA5 (PRD v5.0 Pass 3/19)."""
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"
    Q3_K = "Q3_K"
    Q3_K_S = "Q3_K_S"
    IQ3_XS = "IQ3_XS"
    Q2_K = "Q2_K"
    Q2_K_S = "Q2_K_S"
    IQ2_XXS = "IQ2_XXS"
    INT4 = "INT4"
    INT8 = "INT8"
    # Sub-2-bit formats (PRD v5.0 Pass 19)
    TERNARY = "TERNARY"
    """BitNet b1.58 ternary {-1, 0, +1} — 1.58 bits per weight."""
    BINARY = "BINARY"
    """BTC-LLM binary codebook — 0.8-1.11 bits per weight."""
    NANOQ = "NANOQ"
    """NanoQuant trellis codebook — sub-1-bit per weight."""

    @property
    def bit_width(self) -> float:
        """Return the effective bit width of this precision format.

        For sub-2-bit formats (TERNARY, BINARY, NANOQ) this is a fractional
        value representing the information-theoretic bit cost per weight.
        """
        mapping: dict[Precision, float] = {
            Precision.BF16: 16.0,
            Precision.FP16: 16.0,
            Precision.FP32: 32.0,
            Precision.FP8: 8.0,
            Precision.FP8_E4M3: 8.0,
            Precision.FP8_E5M2: 8.0,
            Precision.FP4: 4.0,
            Precision.NVFP4: 4.0,
            Precision.MXFP4: 4.0,
            Precision.MXFP6: 6.0,
            Precision.Q8_0: 8.0,
            Precision.Q6_K: 6.0,
            Precision.Q4_K_M: 4.0,
            Precision.Q4_0: 4.0,
            Precision.Q3_K: 3.0,
            Precision.Q3_K_S: 3.0,
            Precision.IQ3_XS: 3.0,
            Precision.Q2_K: 2.0,
            Precision.Q2_K_S: 2.0,
            Precision.IQ2_XXS: 2.0,
            Precision.INT4: 4.0,
            Precision.INT8: 8.0,
            Precision.TERNARY: 1.58,   # log2(3)
            Precision.BINARY: 1.0,
            Precision.NANOQ: 0.9,      # approximate; depends on codebook size
        }
        return mapping[self]

    @property
    def byte_size(self) -> float:
        """Return the number of bytes per element."""
        return self.bit_width / 8.0

    def is_quantized(self) -> bool:
        """Return True if this is a quantized precision format."""
        return (
            self.value.startswith("Q")
            or self.value.startswith("I")
            or self in (Precision.TERNARY, Precision.BINARY, Precision.NANOQ)
        )

    def is_sub2bit(self) -> bool:
        """Return True if this is a sub-2-bit precision (PRD v5.0 Pass 19)."""
        return self in (Precision.TERNARY, Precision.BINARY, Precision.NANOQ)

    def is_floating_point(self) -> bool:
        """Return True if this is a floating-point precision."""
        return self in (
            Precision.BF16, Precision.FP16, Precision.FP32,
            Precision.FP8, Precision.FP8_E4M3, Precision.FP8_E5M2,
            Precision.FP4, Precision.NVFP4, Precision.MXFP4, Precision.MXFP6,
        )

    def to_dtype(self) -> DType:
        """Convert this precision to an approximate AEG-IR DType."""
        mapping = {
            Precision.BF16: DType.BF16,
            Precision.FP16: DType.FP16,
            Precision.FP32: DType.FP32,
            Precision.FP8: DType.FP8,
            Precision.FP8_E4M3: DType.FP8,
            Precision.FP8_E5M2: DType.FP8,
            Precision.INT4: DType.INT4,
            Precision.INT8: DType.INT8,
        }
        return mapping.get(self, DType.BF16)

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(value: str) -> Precision:
        """Parse a precision from its string representation (case-insensitive)."""
        normalized = value.upper().strip().replace("-", "_")
        for member in Precision:
            if member.value == normalized:
                return member
        msg = f"Unknown precision format: {value}"
        raise ValueError(msg)


class PrecisionTier(enum.IntEnum):
    """Tiered precision levels for sensitivity-guided mixed precision."""

    FULL = 4
    """Full BF16 precision — no quality loss."""

    HIGH = 3
    """High precision — FP8 or Q6_K — minimal quality loss."""

    MEDIUM = 2
    """Medium precision — Q4_K_M — moderate quality loss."""

    LOW = 1
    """Low precision — Q3_K or IQ3_XS — aggressive quantization."""

    MINIMAL = 0
    """Minimal precision — Q2_K — maximum compression."""

    @property
    def label(self) -> str:
        """Return a human-readable label for this tier."""
        return {
            PrecisionTier.FULL: "Full (BF16)",
            PrecisionTier.HIGH: "High (FP8/Q6_K)",
            PrecisionTier.MEDIUM: "Medium (Q4_K_M)",
            PrecisionTier.LOW: "Low (Q3_K/IQ3_XS)",
            PrecisionTier.MINIMAL: "Minimal (Q2_K)",
        }[self]

    @property
    def recommended_precisions(self) -> list[Precision]:
        """Return the precision formats typically used at this tier."""
        mapping = {
            PrecisionTier.FULL: [Precision.BF16],
            PrecisionTier.HIGH: [Precision.FP8, Precision.Q6_K],
            PrecisionTier.MEDIUM: [Precision.FP4, Precision.NVFP4, Precision.MXFP4, Precision.Q4_K_M, Precision.Q4_0],
            PrecisionTier.LOW: [Precision.Q3_K, Precision.Q3_K_S, Precision.IQ3_XS],
            PrecisionTier.MINIMAL: [Precision.Q2_K, Precision.Q2_K_S],
        }
        return mapping[self]


class MemoryLayout(enum.Enum):
    """Memory layout strategies for tensors in AEG-IR."""

    DENSE = "dense"
    """Standard dense row-major tensor layout."""

    NCHW = "nchw"
    """Channels-first layout for convolution operations."""

    NHWC = "nhwc"
    """Channels-last layout typical in GPU-optimized models."""

    BLOCKED = "blocked"
    """Blocked layout for efficient GPU GEMM and attention."""

    STRIDED = "strided"
    """Strided layout for sub-tensors and slices."""

    SPARSE_CSR = "sparse_csr"
    """Compressed sparse row format for sparse operations."""

    SPARSE_COO = "sparse_coo"
    """Coordinate list sparse format."""

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(value: str) -> MemoryLayout:
        """Parse a memory layout from its string representation."""
        normalized = value.lower().strip()
        for member in MemoryLayout:
            if member.value == normalized:
                return member
        msg = f"Unknown memory layout: {value}"
        raise ValueError(msg)


class HardwareTarget(enum.Enum):
    """Identifiers for supported hardware targets.

    Values follow the naming scheme: ``{backend}_{architecture}``.
    Targets added for PRD v4.0 are marked (v4.0); PRD v5.0 targets (v5.0).
    """

    # ── v3.1 targets (already implemented) ────────────────────────────────────
    CUDA_SM70 = "cuda_sm70"
    CUDA_SM80 = "cuda_sm80"
    CUDA_SM89 = "cuda_sm89"
    CUDA_SM90 = "cuda_sm90"
    CUDA_SM100 = "cuda_sm100"
    CUDA_SM120 = "cuda_sm120"
    METAL_M1 = "metal_m1"
    METAL_M3 = "metal_m3"
    ROCM_RDNA3 = "rocm_rdna3"
    ROCM_CDNA3 = "rocm_cdna3"
    OPENVINO_NPU = "openvino_npu"
    OPENVINO_GPU = "openvino_gpu"
    QUALCOMM_QNN = "qualcomm_qnn"
    CPU_AVX512 = "cpu_avx512"
    CPU_AVX2 = "cpu_avx2"
    CPU_NEON = "cpu_neon"

    # ── v4.0 new hardware targets ──────────────────────────────────────────────
    CUDA_SM130 = "cuda_sm130"
    """NVIDIA Rubin Ultra 2027 — dual Rubin cores, ~100 PFLOPS FP4 (future)."""
    CUDA_SM100_TEE = "cuda_sm100_tee"
    """NVIDIA B200 Confidential Computing mode — encrypted weights + activations."""
    RISCV_MIPS_S8200 = "riscv_mips_s8200"
    """MIPS S8200 agentic NPU — RISC-V, sub-10W, battery-powered edge."""
    RISCV_SIFIVE_X160 = "riscv_sifive_x160"
    """SiFive Intelligence X160 — unified scalar+vector+matrix RISC-V ISA."""
    RISCV_XUANTIE_C930 = "riscv_xuantie_c930"
    """Alibaba XuanTie C930 — high-perf RISC-V + integrated NPU, robotics."""
    FPGA_XILINX_VU9P = "fpga_xilinx_vu9p"
    """Xilinx VU9P FPGA — decode-only target, 10x lower cost-per-token vs GPU."""
    AMD_MI350X = "amd_mi350x"
    """AMD MI350X CDNA4 — HBM3e, successor to MI300X."""
    QUALCOMM_CLOUD_AI100 = "qualcomm_cloud_ai100"
    """Qualcomm Cloud AI 100 Ultra — data center NPU."""

    # ── v5.0 new hardware targets ──────────────────────────────────────────────
    CUDA_SM100_GB300 = "cuda_sm100_gb300"
    """NVIDIA GB300 Blackwell Ultra — 1.5x B200 FP4 throughput, HBM3e+."""
    ROCM_CDNA5_MI455X = "rocm_cdna5_mi455x"
    """AMD MI455X CDNA5 — 432 GB HBM4, 23.3 TB/s, MXFP6 precision format."""
    CPU_AVX512_TERNARY = "cpu_avx512_ternary"
    """x86 AVX2 addition-only ternary kernels — BitNet b1.58, no multiply."""
    CPU_NEON_TERNARY = "cpu_neon_ternary"
    """ARM NEON addition-only ternary kernels — mobile, Apple M-series."""
    FPGA_TERNARY = "fpga_ternary"
    """Generic FPGA BTC-LLM purpose-built addition circuits, 0.8-1.58 bit."""
    RISCV_CERVELL = "riscv_cervell"
    """Semidynamics Cervell — unified scalar/vector/tensor RISC-V NPU."""

    @property
    def vendor(self) -> str:
        """Return the vendor name for this target."""
        _vendors: dict[HardwareTarget, str] = {
            HardwareTarget.CUDA_SM70: "NVIDIA",
            HardwareTarget.CUDA_SM80: "NVIDIA",
            HardwareTarget.CUDA_SM89: "NVIDIA",
            HardwareTarget.CUDA_SM90: "NVIDIA",
            HardwareTarget.CUDA_SM100: "NVIDIA",
            HardwareTarget.CUDA_SM120: "NVIDIA",
            HardwareTarget.CUDA_SM130: "NVIDIA",
            HardwareTarget.CUDA_SM100_TEE: "NVIDIA",
            HardwareTarget.CUDA_SM100_GB300: "NVIDIA",
            HardwareTarget.METAL_M1: "Apple",
            HardwareTarget.METAL_M3: "Apple",
            HardwareTarget.ROCM_RDNA3: "AMD",
            HardwareTarget.ROCM_CDNA3: "AMD",
            HardwareTarget.AMD_MI350X: "AMD",
            HardwareTarget.ROCM_CDNA5_MI455X: "AMD",
            HardwareTarget.OPENVINO_NPU: "Intel",
            HardwareTarget.OPENVINO_GPU: "Intel",
            HardwareTarget.QUALCOMM_QNN: "Qualcomm",
            HardwareTarget.QUALCOMM_CLOUD_AI100: "Qualcomm",
            HardwareTarget.CPU_AVX512: "CPU",
            HardwareTarget.CPU_AVX2: "CPU",
            HardwareTarget.CPU_NEON: "CPU",
            HardwareTarget.CPU_AVX512_TERNARY: "CPU",
            HardwareTarget.CPU_NEON_TERNARY: "CPU",
            HardwareTarget.RISCV_MIPS_S8200: "MIPS",
            HardwareTarget.RISCV_SIFIVE_X160: "SiFive",
            HardwareTarget.RISCV_XUANTIE_C930: "Alibaba",
            HardwareTarget.RISCV_CERVELL: "Semidynamics",
            HardwareTarget.FPGA_XILINX_VU9P: "Xilinx",
            HardwareTarget.FPGA_TERNARY: "FPGA",
        }
        return _vendors[self]

    @property
    def display_name(self) -> str:
        """Return a human-readable display name for this target."""
        _names: dict[HardwareTarget, str] = {
            HardwareTarget.CUDA_SM70: "NVIDIA V100 (Volta)",
            HardwareTarget.CUDA_SM80: "NVIDIA A100 (Ampere)",
            HardwareTarget.CUDA_SM89: "NVIDIA RTX 4090 (Ada)",
            HardwareTarget.CUDA_SM90: "NVIDIA H100 (Hopper)",
            HardwareTarget.CUDA_SM100: "NVIDIA B200 (Blackwell)",
            HardwareTarget.CUDA_SM120: "NVIDIA Rubin R100 (sm_120)",
            HardwareTarget.CUDA_SM130: "NVIDIA Rubin Ultra (sm_130)",
            HardwareTarget.CUDA_SM100_TEE: "NVIDIA B200 Confidential Computing",
            HardwareTarget.CUDA_SM100_GB300: "NVIDIA GB300 Blackwell Ultra",
            HardwareTarget.METAL_M1: "Apple M1/M2",
            HardwareTarget.METAL_M3: "Apple M3/M4/M5",
            HardwareTarget.ROCM_RDNA3: "AMD RX 7000 Series (RDNA3)",
            HardwareTarget.ROCM_CDNA3: "AMD MI300X (CDNA3)",
            HardwareTarget.AMD_MI350X: "AMD MI350X (CDNA4)",
            HardwareTarget.ROCM_CDNA5_MI455X: "AMD MI455X (CDNA5, 432 GB HBM4)",
            HardwareTarget.OPENVINO_NPU: "Intel Arc NPU (OpenVINO)",
            HardwareTarget.OPENVINO_GPU: "Intel GPU (OpenVINO)",
            HardwareTarget.QUALCOMM_QNN: "Qualcomm Snapdragon NPU (QNN)",
            HardwareTarget.QUALCOMM_CLOUD_AI100: "Qualcomm Cloud AI 100 Ultra",
            HardwareTarget.CPU_AVX512: "x86_64 (AVX-512)",
            HardwareTarget.CPU_AVX2: "x86_64 (AVX2)",
            HardwareTarget.CPU_NEON: "ARM (NEON SIMD)",
            HardwareTarget.CPU_AVX512_TERNARY: "x86_64 AVX2 Ternary (BitNet b1.58)",
            HardwareTarget.CPU_NEON_TERNARY: "ARM NEON Ternary (BitNet b1.58)",
            HardwareTarget.RISCV_MIPS_S8200: "MIPS S8200 NPU (RISC-V, sub-10W)",
            HardwareTarget.RISCV_SIFIVE_X160: "SiFive Intelligence X160",
            HardwareTarget.RISCV_XUANTIE_C930: "Alibaba XuanTie C930 RISC-V NPU",
            HardwareTarget.RISCV_CERVELL: "Semidynamics Cervell RISC-V NPU",
            HardwareTarget.FPGA_XILINX_VU9P: "Xilinx VU9P FPGA (decode-only)",
            HardwareTarget.FPGA_TERNARY: "FPGA BTC-LLM Ternary (sub-2-bit)",
        }
        return _names[self]

    @property
    def backend_candidates(self) -> list[str]:
        """Return priority-ordered candidate backend names for this target."""
        _candidates: dict[HardwareTarget, list[str]] = {
            HardwareTarget.CUDA_SM70: ["pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM80: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM89: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM90: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM100: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM120: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM130: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.CUDA_SM100_TEE: ["pytorch"],  # TEE requires bespoke backend
            HardwareTarget.CUDA_SM100_GB300: ["vllm", "pytorch", "tensorrt-llm"],
            HardwareTarget.METAL_M1: ["mlx", "llama.cpp", "pytorch"],
            HardwareTarget.METAL_M3: ["mlx", "llama.cpp", "pytorch"],
            HardwareTarget.ROCM_RDNA3: ["pytorch", "llama.cpp"],
            HardwareTarget.ROCM_CDNA3: ["vllm", "pytorch"],
            HardwareTarget.AMD_MI350X: ["vllm", "pytorch"],
            HardwareTarget.ROCM_CDNA5_MI455X: ["vllm", "pytorch"],
            HardwareTarget.OPENVINO_NPU: ["onnxruntime", "pytorch"],
            HardwareTarget.OPENVINO_GPU: ["onnxruntime", "pytorch"],
            HardwareTarget.QUALCOMM_QNN: ["onnxruntime", "pytorch"],
            HardwareTarget.QUALCOMM_CLOUD_AI100: ["onnxruntime", "pytorch"],
            HardwareTarget.CPU_AVX512: ["llama.cpp", "onnxruntime", "aether_cpu"],
            HardwareTarget.CPU_AVX2: ["llama.cpp", "onnxruntime", "aether_cpu"],
            HardwareTarget.CPU_NEON: ["llama.cpp", "onnxruntime", "aether_cpu"],
            HardwareTarget.CPU_AVX512_TERNARY: ["bitnet.cpp", "llama.cpp", "aether_cpu"],
            HardwareTarget.CPU_NEON_TERNARY: ["bitnet.cpp", "llama.cpp", "aether_cpu"],
            HardwareTarget.RISCV_MIPS_S8200: ["onnxruntime"],
            HardwareTarget.RISCV_SIFIVE_X160: ["onnxruntime"],
            HardwareTarget.RISCV_XUANTIE_C930: ["onnxruntime", "pytorch"],
            HardwareTarget.RISCV_CERVELL: ["onnxruntime"],
            HardwareTarget.FPGA_XILINX_VU9P: ["onnxruntime"],
            HardwareTarget.FPGA_TERNARY: ["bitnet.cpp"],
        }
        return _candidates[self]

    @property
    def supports_ternary(self) -> bool:
        """Return True if this target supports BitNet b1.58 ternary kernels."""
        return self in (
            HardwareTarget.CPU_AVX512_TERNARY,
            HardwareTarget.CPU_NEON_TERNARY,
            HardwareTarget.FPGA_TERNARY,
        )

    @property
    def supports_mxfp6(self) -> bool:
        """Return True if this target natively supports MXFP6 precision."""
        return self == HardwareTarget.ROCM_CDNA5_MI455X

    @property
    def is_confidential_compute(self) -> bool:
        """Return True if this target is a Confidential Computing (TEE) target."""
        return self == HardwareTarget.CUDA_SM100_TEE

    @property
    def is_riscv(self) -> bool:
        """Return True if this is a RISC-V based target."""
        return self in (
            HardwareTarget.RISCV_MIPS_S8200,
            HardwareTarget.RISCV_SIFIVE_X160,
            HardwareTarget.RISCV_XUANTIE_C930,
            HardwareTarget.RISCV_CERVELL,
        )

    @property
    def is_fpga(self) -> bool:
        """Return True if this is an FPGA target."""
        return self in (
            HardwareTarget.FPGA_XILINX_VU9P,
            HardwareTarget.FPGA_TERNARY,
        )

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_string(value: str) -> HardwareTarget:
        """Parse a hardware target from its string representation."""
        normalized = value.lower().strip()
        for member in HardwareTarget:
            if member.value == normalized:
                return member
        msg = f"Unknown hardware target: {value}"
        raise ValueError(msg)

    @staticmethod
    def auto() -> HardwareTarget:
        """Detect the current hardware platform and return the best target.

        Uses native vendor APIs (NVML for CUDA, platform APIs for Metal/ROCm)
        rather than PyTorch. This method does NOT require torch to be installed.

        Detection order:
          1. CUDA via pynvml (preferred, already installed as transitive dep)
          2. CUDA via CUDA Driver API ctypes (fallback, no pynvml)
          3. ROCm via /dev/kfd on Linux
          4. Metal/MPS via platform + system_profiler on macOS
          5. MLX on Apple Silicon
          6. CPU fallback (always succeeds)
        """
        import sys

        # ── 1. CUDA via pynvml ────────────────────────────────────────────────
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                pynvml.nvmlShutdown()
                if (major, minor) >= (10, 0):
                    return HardwareTarget.CUDA_SM100
                if (major, minor) >= (9, 0):
                    return HardwareTarget.CUDA_SM90
                if (major, minor) >= (8, 9):
                    return HardwareTarget.CUDA_SM89
                if (major, minor) >= (8, 0):
                    return HardwareTarget.CUDA_SM80
                return HardwareTarget.CUDA_SM70
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass

        # ── 2. CUDA Driver API ctypes fallback ────────────────────────────────
        try:
            import ctypes  # noqa: PLC0415
            _lib_name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
            _cuda = ctypes.CDLL(_lib_name)
            _init_result = _cuda.cuInit(0)
            if _init_result == 0:  # CUDA_SUCCESS
                count = ctypes.c_int(0)
                _cuda.cuDeviceGetCount(ctypes.byref(count))
                if count.value > 0:
                    major = ctypes.c_int(0)
                    minor = ctypes.c_int(0)
                    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
                    _cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, 0)
                    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
                    _cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, 0)
                    cap = (major.value, minor.value)
                    if cap >= (10, 0):
                        return HardwareTarget.CUDA_SM100
                    if cap >= (9, 0):
                        return HardwareTarget.CUDA_SM90
                    if cap >= (8, 9):
                        return HardwareTarget.CUDA_SM89
                    if cap >= (8, 0):
                        return HardwareTarget.CUDA_SM80
                    return HardwareTarget.CUDA_SM70
        except Exception:  # noqa: BLE001
            pass

        # ── 3. ROCm/HIP via /dev/kfd on Linux ────────────────────────────────
        if sys.platform == "linux":
            try:
                import pathlib  # noqa: PLC0415
                if pathlib.Path("/dev/kfd").exists():
                    return HardwareTarget.ROCM_CDNA3
            except Exception:  # noqa: BLE001
                pass

        # ── 4. Apple Metal via platform APIs ─────────────────────────────────
        # OpenVINO is the framework-independent probe for Intel integrated
        # and discrete GPUs. Require the plugin to enumerate the device;
        # package presence alone is not evidence of usable hardware.
        try:
            import openvino  # type: ignore[import]  # noqa: PLC0415
            devices = list(openvino.Core().available_devices)
            if "GPU" in devices:
                return HardwareTarget.OPENVINO_GPU
            if "NPU" in devices:
                return HardwareTarget.OPENVINO_NPU
        except Exception:  # noqa: BLE001
            pass

        if sys.platform == "darwin":
            try:
                import subprocess  # noqa: PLC0415
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                if "Metal" in result.stdout or "Apple" in result.stdout:
                    return HardwareTarget.METAL_M3
            except Exception:  # noqa: BLE001
                pass
            # MLX as secondary Metal indicator
            try:
                import mlx.core  # noqa: F401,PLC0415
                return HardwareTarget.METAL_M3
            except ImportError:
                pass

        # ── 5. CPU fallback — always available ────────────────────────────────
        # CPU fallback: report the actual instruction-set floor rather than
        # labeling every x86 host as AVX-512.
        import platform

        machine = platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return HardwareTarget.CPU_NEON
        try:
            from numpy.core import _multiarray_umath

            features = getattr(_multiarray_umath, "__cpu_features__", {})
            if bool(features.get("AVX512F")):
                return HardwareTarget.CPU_AVX512
        except (ImportError, AttributeError):
            pass
        return HardwareTarget.CPU_AVX2


@dataclass(frozen=True)
class TensorShape:
    """A tensor shape with named dimensions.

    Provides symbolic dimension support (None for dynamic) and arithmetic
    operations commonly needed in graph modifications.
    """

    dims: tuple[int | None, ...]
    """Tuple of dimension sizes. None = dynamic/unknown."""

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.dims)

    @property
    def num_elements(self) -> int | None:
        """Return the total number of elements, or None if dynamic."""
        total = 1
        for d in self.dims:
            if d is None:
                return None
            total *= d
        return total

    @property
    def is_fully_known(self) -> bool:
        """Return True if all dimensions are known."""
        return all(d is not None for d in self.dims)

    def is_compatible_with(self, other: TensorShape) -> bool:
        """Check if this shape is broadcast-compatible with another."""
        if self.ndim != other.ndim:
            return False
        for a, b in zip(self.dims, other.dims):
            if a is not None and b is not None and a != b:
                return False
        return True

    def replace_dim(self, index: int, value: int | None) -> TensorShape:
        """Return a new shape with dimension at index replaced."""
        dims = list(self.dims)
        dims[index] = value
        return TensorShape(tuple(dims))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {"dims": list(self.dims)}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TensorShape:
        """Deserialize from a dictionary."""
        return TensorShape(tuple(data["dims"]))

    @staticmethod
    def scalar() -> TensorShape:
        """Create a zero-dimensional (scalar) shape."""
        return TensorShape(())

    @staticmethod
    def from_list(dims: list[int | None]) -> TensorShape:
        """Create a shape from a list of dimension sizes."""
        return TensorShape(tuple(dims))

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, index: int) -> int | None:
        return self.dims[index]

    def __iter__(self) -> Iterator[int | None]:
        return iter(self.dims)

    def __repr__(self) -> str:
        dims_str = "x".join(str(d) if d is not None else "?" for d in self.dims)
        return f"TensorShape({dims_str})"


@dataclass(frozen=True)
class TensorLayout:
    """Describes the memory layout and stride information of a tensor."""

    shape: TensorShape
    """The logical shape of the tensor."""

    dtype: DType
    """The data type of each element."""

    layout: MemoryLayout = MemoryLayout.DENSE
    """How the elements are arranged in memory."""

    strides: tuple[int, ...] | None = None
    """Stride in elements for each dimension. None = contiguous row-major."""

    alignment: int = 64
    """Memory alignment constraint in bytes (default 64 = cache line)."""

    @property
    def byte_size(self) -> int:
        """Return the total size of this tensor in bytes."""
        if self.shape.num_elements is None:
            return 0
        return self.shape.num_elements * self.dtype.byte_size()

    @property
    def is_contiguous(self) -> bool:
        """Return True if the tensor is contiguous in memory."""
        if self.strides is None:
            return True
        if self.shape.ndim == 0:
            return True
        expected = 1
        for d in reversed(self.shape.dims):
            if d is not None and expected != self.strides[self.shape.ndim - len(self.strides) + self.shape.dims.index(d)]:
                return False
            if d is not None:
                expected *= d
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "shape": self.shape.to_dict(),
            "dtype": self.dtype.value,
            "layout": self.layout.value,
            "strides": list(self.strides) if self.strides else None,
            "alignment": self.alignment,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TensorLayout:
        """Deserialize from a dictionary."""
        strides = tuple(data["strides"]) if data.get("strides") else None
        return TensorLayout(
            shape=TensorShape.from_dict(data["shape"]),
            dtype=DType.from_string(data["dtype"]),
            layout=MemoryLayout.from_string(data.get("layout", "dense")),
            strides=strides,
            alignment=data.get("alignment", 64),
        )

    def __repr__(self) -> str:
        return f"TensorLayout({self.shape}, {self.dtype.value})"


@dataclass(frozen=True)
class AEGVersion:
    """Version descriptor for an AEG format or file."""

    major: int
    minor: int

    @property
    def version_string(self) -> str:
        """Return the canonical version string (e.g., 'AEG/1.0')."""
        return f"AEG/{self.major}.{self.minor}"

    def is_compatible_with(self, minimum: AEGVersion) -> bool:
        """Check if this version is compatible with a minimum required version.

        Compatibility requires:
        - Same major version (format lock).
        - Minor version >= minimum minor version.
        """
        if self.major != minimum.major:
            return False
        return self.minor >= minimum.minor

    def __str__(self) -> str:
        return self.version_string

    @staticmethod
    def parse(version_str: str) -> AEGVersion:
        """Parse a version string like 'AEG/1.0' into an AEGVersion."""
        normalized = version_str.strip().upper()
        if normalized.startswith("AEG/"):
            normalized = normalized[4:]
        parts = normalized.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return AEGVersion(major=major, minor=minor)

    @staticmethod
    def current() -> AEGVersion:
        """Return the current AEG format version."""
        from aether.core.constants import AEG_FORMAT_VERSION

        return AEGVersion.parse(AEG_FORMAT_VERSION)

    def to_dict(self) -> dict[str, int]:
        return {"major": self.major, "minor": self.minor}


@dataclass(frozen=True)
class ShardingPlan:
    """A parallelism sharding plan for a specific GPU count and phase."""

    num_gpus: int
    """Number of GPUs in this plan."""

    phase: str
    """Phase of execution: 'prefill' or 'decode'."""

    tensor_parallel_degree: int = 1
    """Degree of tensor parallelism."""

    pipeline_stages: int = 1
    """Number of pipeline parallelism stages."""

    expert_parallel_degree: int = 1
    """Degree of expert parallelism (MoE only)."""

    context_parallel_degree: int = 1
    """Degree of context parallelism (long sequences)."""

    memory_per_gpu_gb: float = 0.0
    """Estimated per-GPU memory usage in GB."""

    def __post_init__(self) -> None:
        """Validate the plan parameters."""
        if self.tensor_parallel_degree < 1:
            msg = "tensor_parallel_degree must be >= 1"
            raise ValueError(msg)
        if self.pipeline_stages < 1:
            msg = "pipeline_stages must be >= 1"
            raise ValueError(msg)
        if self.expert_parallel_degree < 1:
            msg = "expert_parallel_degree must be >= 1"
            raise ValueError(msg)
        if self.context_parallel_degree < 1:
            msg = "context_parallel_degree must be >= 1"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "num_gpus": self.num_gpus,
            "phase": self.phase,
            "tensor_parallel_degree": self.tensor_parallel_degree,
            "pipeline_stages": self.pipeline_stages,
            "expert_parallel_degree": self.expert_parallel_degree,
            "context_parallel_degree": self.context_parallel_degree,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShardingPlan:
        """Deserialize from a dictionary."""
        return ShardingPlan(
            num_gpus=data["num_gpus"],
            phase=data["phase"],
            tensor_parallel_degree=data.get("tensor_parallel_degree", 1),
            pipeline_stages=data.get("pipeline_stages", 1),
            expert_parallel_degree=data.get("expert_parallel_degree", 1),
            context_parallel_degree=data.get("context_parallel_degree", 1),
            memory_per_gpu_gb=data.get("memory_per_gpu_gb", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"ShardingPlan({self.num_gpus}gpu[{self.phase}]: "
            f"TP{self.tensor_parallel_degree} PP{self.pipeline_stages}"
            f"EP{self.expert_parallel_degree} CP{self.context_parallel_degree})"
        )


class MemoryTier(enum.IntEnum):
    """Memory tier identifiers for the global KV cache."""

    L1_GPU_HBM = 1
    """GPU high-bandwidth memory — active request KV blocks."""

    L2_CPU_DRAM = 2
    """CPU system RAM — prefix cache and recently evicted blocks."""

    L3_NVME = 3
    """NVMe SSD storage — long system prompts and RAG KV."""

    L4_HUB = 4
    """Aether Hub CDN — globally shared system prompts."""

    @property
    def label(self) -> str:
        return {
            MemoryTier.L1_GPU_HBM: "GPU HBM",
            MemoryTier.L2_CPU_DRAM: "CPU DRAM",
            MemoryTier.L3_NVME: "NVMe SSD",
            MemoryTier.L4_HUB: "Aether Hub",
        }[self]

    @property
    def approximate_bandwidth_gb_s(self) -> float:
        return {
            MemoryTier.L1_GPU_HBM: 2000.0,
            MemoryTier.L2_CPU_DRAM: 50.0,
            MemoryTier.L3_NVME: 5.0,
            MemoryTier.L4_HUB: 0.5,
        }[self]

    def __str__(self) -> str:
        return self.label


def _optional_float(value: Any) -> float | None:
    """Coerce a manifest field to ``float`` while preserving an absent value.

    A missing execution-numerics field means "use the architecture default",
    which is semantically different from an explicit ``0.0``.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class ModelArchitecture:
    """Describes the architecture of a model detected during ingestion."""

    family: str
    """Architecture family name (e.g., 'llama_family', 'qwen_family')."""

    params_billion: float
    """Model parameter count in billions."""

    layers: int
    """Number of transformer layers."""

    hidden_size: int
    """Hidden dimension size."""

    num_attention_heads: int
    """Number of query attention heads."""

    num_kv_heads: int | None = None
    """Number of key/value heads. None for non-GQA models."""

    head_dim: int | None = None
    """Dimension per attention head. Computed if not provided."""

    intermediate_size: int | None = None
    """FFN intermediate dimension."""

    context_length: int = 4096
    """Maximum context length in tokens."""

    vocab_size: int = 32000
    """Vocabulary size."""

    norm_eps: float = 1e-5
    """Normalization epsilon."""

    rope_theta: float = 10000.0
    """RoPE theta base frequency."""

    is_moe: bool = False
    """Whether this is a Mixture-of-Experts model."""

    is_encoder: bool = False
    """Whether this is an encoder-only architecture (e.g. BERT, RoBERTa)."""

    is_encoder_decoder: bool = False
    """Whether this architecture requires separate encoder and decoder graphs."""

    is_multimodal: bool = False
    """Whether the checkpoint includes a vision/audio/video modality branch."""

    num_experts: int = 0
    """Total number of experts (MoE only)."""

    num_activated_experts: int = 0
    """Number of experts activated per token (top-K) (MoE only)."""

    moe_layer_indices: list[int] | None = None
    """Indices of decoder layers containing routed experts.

    ``None`` means all decoder layers.  The explicit pattern is needed for
    hybrid checkpoints that retain dense layers before or between MoE layers.
    """

    mtp_heads: int = 0
    """Number of native multi-token-prediction heads, when declared by the checkpoint."""

    attention_type: str = "GQA"
    """Attention type: GQA, MLA, or MHA."""

    # Multi-head latent attention (DeepSeek/Kimi/GLM-compatible) geometry.
    # These are optional because ordinary attention does not define them; when
    # present they are copied from config.json and become part of the AEG
    # execution contract rather than being inferred from a model name.
    mla_kv_lora_rank: int | None = None
    mla_q_lora_rank: int | None = None
    mla_qk_nope_head_dim: int | None = None
    mla_qk_rope_head_dim: int | None = None
    mla_v_head_dim: int | None = None

    # Stateful sequence-model geometry (Mamba selective scan/RWKV adapters).
    # ``None`` means this is an ordinary transformer decoder.
    ssm_variant: str | None = None
    ssm_state_size: int | None = None
    ssm_inner_size: int | None = None
    ssm_dt_rank: int | None = None
    ssm_conv_kernel: int | None = None
    ssm_num_heads: int | None = None
    ssm_num_groups: int | None = None
    ssm_head_dim: int | None = None
    hybrid_layer_types: list[str] | None = None

    ffn_type: str = "SwiGLU"
    """FFN activation/type: SwiGLU, GeGLU, GELU."""

    norm_type: str = "RMSNorm"
    """Normalization used by the decoder blocks (RMSNorm or LayerNorm)."""

    position_type: str = "RoPE"
    """Position encoding used by the decoder (RoPE, ALiBi, or none)."""

    attention_layers: list[str] | None = None
    """Per-layer attention pattern, e.g. GPT-Neo's ``global``/``local`` mix."""

    attention_window: int | None = None
    """Local causal attention window in tokens when the source declares one."""

    embedding_norm: bool = False
    """Whether the token embedding output has a learned normalization."""

    qk_norm: bool = False
    """Whether attention applies separate RMSNorm weights to Q and K heads."""

    parallel_residual: bool = False
    """Whether attention and FFN branches share the same pre-norm input.

    GPT-J uses a parallel residual block rather than the sequential
    attention-then-FFN block used by most decoder families.  This is a
    checkpoint-declared execution capability, not a model-name shortcut.
    """

    encoder_layers: int | None = None
    """Encoder depth for encoder-decoder architectures."""

    decoder_layers: int | None = None
    """Decoder depth for encoder-decoder architectures."""

    tie_word_embeddings: bool = True
    """Whether the decoder LM head shares the input embedding matrix."""

    relative_attention_num_buckets: int = 32
    """Number of T5 relative-position buckets when declared by the source."""

    # ── Execution numerics ──────────────────────────────────────────────────
    # Decoder families agree on the *shape* of the transformer block but differ
    # in a small set of scalar constants and structural placements.  Those
    # differences are not stylistic: omitting one changes every logit.  They are
    # read from the source configuration at ingestion and become part of the AEG
    # execution contract, so no runtime code needs to branch on a family name.

    attention_scale: float | None = None
    """Softmax scale applied to ``QKᵀ``.  ``None`` means ``1/sqrt(head_dim)``.

    GPT-Neo publishes unscaled attention (scale 1.0); Gemma-2/3 derive the
    scale from ``query_pre_attn_scalar`` rather than the head width; Granite
    uses an explicit ``attention_multiplier``.
    """

    attention_scale_by_layer_index: bool = False
    """Whether to additionally divide scores by ``layer_index + 1``.

    GPT-2's ``scale_attn_by_inverse_layer_idx`` training trick, which must be
    reproduced at inference for checkpoints that enabled it.
    """

    embedding_scale: float | None = None
    """Multiplier applied to the token embedding output.

    Gemma scales embeddings by ``sqrt(hidden_size)``; Granite uses an explicit
    ``embedding_multiplier``.
    """

    residual_scale: float | None = None
    """Multiplier applied to each sublayer output before the residual add.

    Granite's ``residual_multiplier``.
    """

    logit_scale: float | None = None
    """Multiplier applied to the final logits (Cohere's ``logit_scale``)."""

    attn_logit_softcap: float | None = None
    """Gemma-2 attention logit soft cap: ``cap * tanh(scores / cap)``."""

    final_logit_softcap: float | None = None
    """Gemma-2 output logit soft cap: ``cap * tanh(logits / cap)``."""

    norm_offset_one: bool = False
    """Whether RMSNorm uses ``x * (1 + weight)`` instead of ``x * weight``.

    Gemma stores its normalization weights as offsets from unity.
    """

    rope_partial_dim: int | None = None
    """Number of leading head dimensions that RoPE rotates.

    GPT-NeoX (``rotary_pct``), GPT-J (``rotary_dim``), StableLM and Phi
    (``partial_rotary_factor``) rotate only a prefix of each head and pass the
    remaining channels through unchanged.  ``None`` rotates the full head.
    """

    rope_interleaved: bool = False
    """Whether RoPE pairs adjacent channels instead of half-split halves.

    GPT-J's published ``rotate_every_two`` convention pairs ``(x0,x1)``,
    ``(x2,x3)``, ...; the GPT-NeoX/Llama convention pairs ``(x_i, x_{i+d/2})``.
    The two are not interchangeable.
    """

    rope_local_theta: float | None = None
    """Separate RoPE base for sliding-window layers (Gemma-3's local base)."""

    norm_placement: str = "pre"
    """Where block normalization is applied: ``pre``, ``post``, or ``sandwich``.

    ``pre`` is the standard pre-norm block.  ``post`` normalizes each sublayer
    *output* before the residual add (OLMo-2).  ``sandwich`` normalizes both the
    input and the output of each sublayer (Gemma-2/3, EXAONE-4).
    """

    qk_norm_scope: str = "head"
    """Whether Q/K normalization spans one head (``head``) or the whole
    projection (``full``).  OLMo-2 normalizes the full projection."""

    fused_qkv_layout: str = "contiguous"
    """Memory layout of a fused QKV projection in the source checkpoint.

    ``contiguous`` stacks ``[all Q | all K | all V]`` (Phi-3, MPT, GPT-2).
    ``head_interleaved`` stacks per head ``[q,k,v][q,k,v]...`` (GPT-NeoX,
    BLOOM).  ``group_interleaved`` stacks per KV group
    ``[q...q, k, v][q...q, k, v]...`` (Falcon's new decoder architecture).
    Splitting the wrong way silently permutes every attention head.
    """

    no_rope_layers: list[int] | None = None
    """Indices of layers that apply no positional rotation at all (NoPE).

    SmolLM3 interleaves NoPE layers among RoPE layers.
    """

    gelu_approximate: bool = True
    """Whether a GELU activation uses the tanh approximation.

    ``gelu_new``/``gelu_pytorch_tanh``/``gelu_fast`` (GPT-2, GPT-Neo, GPT-J,
    Gemma, Starcoder2, BLOOM) use the tanh form; a plain ``gelu``
    (GPT-NeoX, Falcon, MPT) uses the exact error function.  The two differ by
    up to ~1e-3 per activation, which is small per layer and visible in the
    logits after accumulating over depth.
    """

    def __post_init__(self) -> None:
        """Auto-compute derived fields."""
        if self.head_dim is None:
            self.head_dim = (self.hidden_size // max(self.num_attention_heads, 1)) if self.num_attention_heads else 1
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
        if self.decoder_layers is None:
            self.decoder_layers = self.layers
        if self.encoder_layers is None:
            self.encoder_layers = self.layers
        if self.params_billion == 0.0 and self.layers > 0 and self.hidden_size > 0:
            self.params_billion = self._estimate_params()
        # A positive expert count *is* what makes a model MoE. Keeping the flag
        # independent let `num_experts=32, is_moe=False` silently disable Pass 5.
        if self.num_experts > 0:
            self.is_moe = True

    def _estimate_params(self) -> float:
        """Rough estimate of parameter count in billions."""
        h = self.hidden_size
        i = self.intermediate_size or h * 4
        v = self.vocab_size
        L = self.layers  # noqa: N806
        # Embedding
        emb = v * h
        # Per transformer layer
        per_layer = (
            4 * h * h +  # QKV + O projections (rough)
            3 * h * i +  # FFN gate/up/down
            2 * h  # RMS norms
        )
        total = emb + L * per_layer
        return total / 1e9

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "family": self.family,
            "params_billion": self.params_billion,
            "layers": self.layers,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "tie_word_embeddings": self.tie_word_embeddings,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "norm_eps": self.norm_eps,
            "rope_theta": self.rope_theta,
            "is_moe": self.is_moe,
            "is_encoder": self.is_encoder,
            "is_encoder_decoder": self.is_encoder_decoder,
            "is_multimodal": self.is_multimodal,
            "num_experts": self.num_experts,
            "num_activated_experts": self.num_activated_experts,
            "moe_layer_indices": self.moe_layer_indices,
            "mtp_heads": self.mtp_heads,
            "attention_type": self.attention_type,
            "mla_kv_lora_rank": self.mla_kv_lora_rank,
            "mla_q_lora_rank": self.mla_q_lora_rank,
            "mla_qk_nope_head_dim": self.mla_qk_nope_head_dim,
            "mla_qk_rope_head_dim": self.mla_qk_rope_head_dim,
            "mla_v_head_dim": self.mla_v_head_dim,
            "ssm_variant": self.ssm_variant,
            "ssm_state_size": self.ssm_state_size,
            "ssm_inner_size": self.ssm_inner_size,
            "ssm_dt_rank": self.ssm_dt_rank,
            "ssm_conv_kernel": self.ssm_conv_kernel,
            "ssm_num_heads": self.ssm_num_heads,
            "ssm_num_groups": self.ssm_num_groups,
            "ssm_head_dim": self.ssm_head_dim,
            "hybrid_layer_types": self.hybrid_layer_types,
            "ffn_type": self.ffn_type,
            "norm_type": self.norm_type,
            "position_type": self.position_type,
            "attention_layers": self.attention_layers,
            "attention_window": self.attention_window,
            "embedding_norm": self.embedding_norm,
            "qk_norm": self.qk_norm,
            "parallel_residual": self.parallel_residual,
            "attention_scale": self.attention_scale,
            "attention_scale_by_layer_index": self.attention_scale_by_layer_index,
            "embedding_scale": self.embedding_scale,
            "residual_scale": self.residual_scale,
            "logit_scale": self.logit_scale,
            "attn_logit_softcap": self.attn_logit_softcap,
            "final_logit_softcap": self.final_logit_softcap,
            "norm_offset_one": self.norm_offset_one,
            "rope_partial_dim": self.rope_partial_dim,
            "rope_interleaved": self.rope_interleaved,
            "rope_local_theta": self.rope_local_theta,
            "norm_placement": self.norm_placement,
            "qk_norm_scope": self.qk_norm_scope,
            "fused_qkv_layout": self.fused_qkv_layout,
            "no_rope_layers": self.no_rope_layers,
            "gelu_approximate": self.gelu_approximate,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ModelArchitecture:
        """Deserialize from a dictionary."""
        return ModelArchitecture(
            family=data["family"],
            params_billion=data.get("params_billion", 0.0),
            layers=data["layers"],
            encoder_layers=data.get("encoder_layers"),
            decoder_layers=data.get("decoder_layers"),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
            relative_attention_num_buckets=int(data.get("relative_attention_num_buckets", 32)),
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_kv_heads=data.get("num_kv_heads"),
            head_dim=data.get("head_dim"),
            intermediate_size=data.get("intermediate_size"),
            context_length=data.get("context_length", 4096),
            vocab_size=data.get("vocab_size", 32000),
            norm_eps=data.get("norm_eps", 1e-5),
            rope_theta=data.get("rope_theta", 10000.0),
            is_moe=data.get("is_moe", False),
            is_encoder=data.get("is_encoder", False),
            is_encoder_decoder=data.get("is_encoder_decoder", False),
            is_multimodal=bool(data.get("is_multimodal", False)),
            num_experts=data.get("num_experts", 0),
            num_activated_experts=data.get("num_activated_experts", 0),
            moe_layer_indices=(
                [int(index) for index in data["moe_layer_indices"]]
                if data.get("moe_layer_indices") is not None else None
            ),
            mtp_heads=data.get("mtp_heads", 0),
            attention_type=data.get("attention_type", "GQA"),
            mla_kv_lora_rank=data.get("mla_kv_lora_rank"),
            mla_q_lora_rank=data.get("mla_q_lora_rank"),
            mla_qk_nope_head_dim=data.get("mla_qk_nope_head_dim"),
            mla_qk_rope_head_dim=data.get("mla_qk_rope_head_dim"),
            mla_v_head_dim=data.get("mla_v_head_dim"),
            ssm_variant=data.get("ssm_variant"),
            ssm_state_size=data.get("ssm_state_size"),
            ssm_inner_size=data.get("ssm_inner_size"),
            ssm_dt_rank=data.get("ssm_dt_rank"),
            ssm_conv_kernel=data.get("ssm_conv_kernel"),
            ssm_num_heads=data.get("ssm_num_heads"),
            ssm_num_groups=data.get("ssm_num_groups"),
            ssm_head_dim=data.get("ssm_head_dim"),
            hybrid_layer_types=(
                [str(value) for value in data["hybrid_layer_types"]]
                if data.get("hybrid_layer_types") is not None else None
            ),
            ffn_type=data.get("ffn_type", "SwiGLU"),
            norm_type=data.get("norm_type", "RMSNorm"),
            position_type=data.get("position_type", "RoPE"),
            attention_layers=(
                [str(value) for value in data["attention_layers"]]
                if isinstance(data.get("attention_layers"), list) else None
            ),
            attention_window=(
                int(data["attention_window"])
                if data.get("attention_window") is not None else None
            ),
            embedding_norm=bool(data.get("embedding_norm", False)),
            qk_norm=bool(data.get("qk_norm", False)),
            parallel_residual=bool(data.get("parallel_residual", False)),
            attention_scale=_optional_float(data.get("attention_scale")),
            attention_scale_by_layer_index=bool(
                data.get("attention_scale_by_layer_index", False)
            ),
            embedding_scale=_optional_float(data.get("embedding_scale")),
            residual_scale=_optional_float(data.get("residual_scale")),
            logit_scale=_optional_float(data.get("logit_scale")),
            attn_logit_softcap=_optional_float(data.get("attn_logit_softcap")),
            final_logit_softcap=_optional_float(data.get("final_logit_softcap")),
            norm_offset_one=bool(data.get("norm_offset_one", False)),
            rope_partial_dim=(
                int(data["rope_partial_dim"])
                if data.get("rope_partial_dim") is not None else None
            ),
            rope_interleaved=bool(data.get("rope_interleaved", False)),
            rope_local_theta=_optional_float(data.get("rope_local_theta")),
            norm_placement=str(data.get("norm_placement", "pre") or "pre"),
            qk_norm_scope=str(data.get("qk_norm_scope", "head") or "head"),
            fused_qkv_layout=str(data.get("fused_qkv_layout", "contiguous") or "contiguous"),
            no_rope_layers=(
                [int(value) for value in data["no_rope_layers"]]
                if data.get("no_rope_layers") is not None else None
            ),
            gelu_approximate=bool(data.get("gelu_approximate", True)),
        )

    def __repr__(self) -> str:
        return (
            f"ModelArchitecture({self.family}, {self.params_billion}B, "
            f"L={self.layers}, H={self.hidden_size})"
        )
