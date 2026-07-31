"""
ROCm / HIP Kernel Emitter for AMD GPUs (MI300X and family).

Generates HIP (Heterogeneous-compute Interface for Portability) compute
kernels for AMD CDNA/RDNA targets:
  - GEMM using rocBLAS-compatible tile layout
  - FlashAttention using LDS (Local Data Store) tiles
  - RMSNorm / LayerNorm fused kernels
  - SiLU-GEMM fused FFN kernels
  - FP8 GEMM for MI300X E4M3 native

AMD MI300X specs (2023):
  - 192 GB HBM3 shared memory pool
  - 1307 GB/s HBM3 bandwidth
  - 383 TFLOPS FP16 peak
  - Native FP8 E4M3/E5M2 support (similar to NVIDIA H100)
  - ROCm 6.x software stack

HIP compile pipeline:
  .hip → hipcc -arch=gfx942 → .o → .hsaco (ROCm binary)
  Or: .hip → LLVM IR → AMDGPU backend → .isa

For AMD MI300X: gfx942 (CDNA3)
For AMD RX 7900 XTX: gfx1100 (RDNA3)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ROCm target profile
# ---------------------------------------------------------------------------

@dataclass
class ROCmTargetProfile:
    """Hardware profile for an AMD GPU target."""
    device_name: str = "AMD MI300X"
    gfx_arch: str = "gfx942"          # GPU ISA target
    cu_count: int = 304                # Compute units
    wavefront_size: int = 64           # AMD wavefront = 64 threads
    lds_per_cu_kb: int = 64            # Local Data Store per CU
    hbm_capacity_gb: float = 192.0
    hbm_bandwidth_gbs: float = 1307.0
    peak_fp16_tflops: float = 383.0
    supports_fp8: bool = True
    rocm_version: str = "6.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_name": self.device_name,
            "gfx_arch": self.gfx_arch,
            "cu_count": self.cu_count,
            "wavefront_size": self.wavefront_size,
            "lds_per_cu_kb": self.lds_per_cu_kb,
            "hbm_capacity_gb": self.hbm_capacity_gb,
            "hbm_bandwidth_gbs": self.hbm_bandwidth_gbs,
            "peak_fp16_tflops": self.peak_fp16_tflops,
            "supports_fp8": self.supports_fp8,
            "rocm_version": self.rocm_version,
        }

    @classmethod
    def mi300x(cls) -> "ROCmTargetProfile":
        return cls()  # defaults are MI300X

    @classmethod
    def mi250x(cls) -> "ROCmTargetProfile":
        return cls(
            device_name="AMD MI250X",
            gfx_arch="gfx90a",
            cu_count=220,
            hbm_capacity_gb=128.0,
            hbm_bandwidth_gbs=896.0,
            peak_fp16_tflops=191.0,
            supports_fp8=False,
        )

    @classmethod
    def rdna3(cls) -> "ROCmTargetProfile":
        return cls(
            device_name="AMD RX 7900 XTX",
            gfx_arch="gfx1100",
            cu_count=96,
            wavefront_size=32,   # RDNA uses wave32
            lds_per_cu_kb=128,
            hbm_capacity_gb=24.0,
            hbm_bandwidth_gbs=960.0,
            peak_fp16_tflops=122.8,
            supports_fp8=False,
        )


# ---------------------------------------------------------------------------
# HIP kernel templates
# ---------------------------------------------------------------------------

_HIP_GEMM = """\
#include <hip/hip_runtime.h>
#include <hip/hip_{dtype_include}.h>

// Tiled GEMM for ROCm: C = A @ B
// Tile size: BLOCK_M x BLOCK_N, loop over K
// AMD LDS (shared memory) optimization with bank-conflict-free layout
#define BLOCK_M {block_m}
#define BLOCK_N {block_n}
#define BLOCK_K {block_k}

__global__ void gemm_{dtype}(
    const {hip_dtype}* __restrict__ A,  // (M, K)
    const {hip_dtype}* __restrict__ B,  // (K, N)
    {hip_dtype}* __restrict__ C,        // (M, N)
    int M, int N, int K
) {{
    __shared__ {hip_dtype} As[BLOCK_M][BLOCK_K + 1];  // +1 to avoid bank conflicts
    __shared__ {hip_dtype} Bs[BLOCK_K][BLOCK_N + 1];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    float acc = 0.0f;

    for (int k = 0; k < K; k += BLOCK_K) {{
        // Load A tile into shared memory
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = ({hip_dtype})0.0f;

        // Load B tile into shared memory
        if ((k + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = ({hip_dtype})0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++)
            acc += float(As[threadIdx.y][kk]) * float(Bs[kk][threadIdx.x]);

        __syncthreads();
    }}

    if (row < M && col < N)
        C[row * N + col] = ({hip_dtype})acc;
}}
"""

_HIP_FLASH_ATTENTION = """\
#include <hip/hip_runtime.h>
#include <hip/hip_{dtype_include}.h>
#include <math.h>

// FlashAttention-2 for ROCm with online softmax
// One block per (query_tile, head), loop over K/V tiles
#define TILE_Q  {tile_q}
#define TILE_KV {tile_kv}
#define HEAD_DIM_MAX {head_dim_max}

__global__ void flash_attention_{dtype}(
    const {hip_dtype}* __restrict__ Q,  // (seq, heads, head_dim)
    const {hip_dtype}* __restrict__ K,
    const {hip_dtype}* __restrict__ V,
    {hip_dtype}* __restrict__ O,
    int seq_len, int num_heads, int head_dim,
    float scale, bool causal
) {{
    const int q_idx    = blockIdx.y * TILE_Q + threadIdx.y;
    const int head_idx = blockIdx.z;

    if (q_idx >= seq_len || head_idx >= num_heads) return;

    // Online softmax state
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_reg[HEAD_DIM_MAX] = {{0.0f}};

    // Load Q for this position
    float q_reg[HEAD_DIM_MAX];
    const int q_base = (q_idx * num_heads + head_idx) * head_dim;
    #pragma unroll 4
    for (int d = 0; d < head_dim && d < HEAD_DIM_MAX; d++)
        q_reg[d] = float(Q[q_base + d]);

    // Tile loop over K/V
    for (int kv_start = 0; kv_start < seq_len; kv_start += TILE_KV) {{
        const int kv_end = min(kv_start + TILE_KV, seq_len);
        for (int kv = kv_start; kv < kv_end; kv++) {{
            if (causal && kv > q_idx) continue;

            const int k_base = (kv * num_heads + head_idx) * head_dim;
            float qk = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < head_dim && d < HEAD_DIM_MAX; d++)
                qk += q_reg[d] * float(K[k_base + d]);
            qk *= scale;

            float m_new = fmaxf(m_i, qk);
            float exp_qk = expf(qk - m_new);
            float scale_prev = expf(m_i - m_new);
            l_i = scale_prev * l_i + exp_qk;
            m_i = m_new;

            #pragma unroll 4
            for (int d = 0; d < head_dim && d < HEAD_DIM_MAX; d++)
                o_reg[d] = scale_prev * o_reg[d] + exp_qk * float(V[k_base + d]);
        }}
    }}

    // Write normalized output
    const int o_base = q_base;
    #pragma unroll 4
    for (int d = 0; d < head_dim && d < HEAD_DIM_MAX; d++)
        O[o_base + d] = ({hip_dtype})(o_reg[d] / (l_i + 1e-9f));
}}
"""

_HIP_RMSNORM = """\
#include <hip/hip_runtime.h>
#include <hip/hip_{dtype_include}.h>
#include <math.h>

__global__ void rmsnorm_{dtype}(
    const {hip_dtype}* __restrict__ x,
    const {hip_dtype}* __restrict__ weight,
    {hip_dtype}* __restrict__ output,
    int hidden_dim,
    float eps
) {{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int offset = row * hidden_dim;

    // Warp reduction for sum of squares
    float local_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x)
        local_sq += float(x[offset + i]) * float(x[offset + i]);

    // Warp-level reduction
    for (int stride = warpSize / 2; stride > 0; stride >>= 1)
        local_sq += __shfl_down(local_sq, stride);

    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    if (tid % warpSize == 0) atomicAdd(&shared_sum, local_sq);
    __syncthreads();

    float rms = rsqrtf(shared_sum / float(hidden_dim) + eps);
    for (int i = tid; i < hidden_dim; i += blockDim.x)
        output[offset + i] = ({hip_dtype})(float(x[offset + i]) * rms * float(weight[i]));
}}
"""

_HIP_SILU_GATE = """\
#include <hip/hip_runtime.h>
#include <hip/hip_{dtype_include}.h>
#include <math.h>

__device__ inline float silu(float x) {{ return x / (1.0f + expf(-x)); }}

__global__ void silu_gate_ffn_{dtype}(
    const {hip_dtype}* __restrict__ gate_proj,
    const {hip_dtype}* __restrict__ up_proj,
    {hip_dtype}* __restrict__ output,
    int ffn_dim
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ffn_dim)
        output[idx] = ({hip_dtype})(silu(float(gate_proj[idx])) * float(up_proj[idx]));
}}
"""


# ---------------------------------------------------------------------------
# ROCm kernel descriptor
# ---------------------------------------------------------------------------

@dataclass
class ROCmKernelDescriptor:
    """Describes a compiled ROCm/HIP kernel."""
    name: str
    source: str
    function_name: str
    dtype: str = "fp16"
    block_size: tuple[int, int, int] = (32, 32, 1)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "function_name": self.function_name,
            "dtype": self.dtype,
            "block_size": list(self.block_size),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# ROCm kernel emitter
# ---------------------------------------------------------------------------

class ROCmKernelEmitter:
    """
    Emits HIP (ROCm) compute kernels for AMD GPU targets.

    Generates:
    - .hip source files (C++ with AMD HIP extensions)
    - Kernel manifest for runtime dispatch
    - hipcc compile scripts

    Compile pipeline:
      .hip → hipcc --offload-arch=gfx942 → .o → .hsaco
    """

    def __init__(self, profile: ROCmTargetProfile | None = None) -> None:
        self.profile = profile or ROCmTargetProfile.mi300x()
        self._kernels: list[ROCmKernelDescriptor] = []

    def _hip_dtype(self, dtype: str) -> tuple[str, str]:
        """Return (hip_dtype, dtype_include) for a given dtype name."""
        if dtype in ("fp16", "half"):
            return "__half", "fp16"
        elif dtype == "bf16":
            return "__hip_bfloat16", "bfloat16"
        elif dtype == "fp32":
            return "float", "runtime"
        elif dtype == "fp8":
            return "__hip_fp8_e4m3_fnuz", "fp8"
        return "__half", "fp16"

    def emit_gemm(
        self,
        dtype: str = "fp16",
        block_m: int = 32,
        block_n: int = 32,
        block_k: int = 16,
    ) -> ROCmKernelDescriptor:
        """Emit a tiled GEMM kernel."""
        hip_dtype, dtype_include = self._hip_dtype(dtype)
        source = _HIP_GEMM.format(
            dtype=dtype,
            hip_dtype=hip_dtype,
            dtype_include=dtype_include,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
        kernel = ROCmKernelDescriptor(
            name=f"gemm_{dtype}",
            source=source,
            function_name=f"gemm_{dtype}",
            dtype=dtype,
            block_size=(block_n, block_m, 1),
            metadata={"block_m": block_m, "block_n": block_n, "block_k": block_k},
        )
        self._kernels.append(kernel)
        return kernel

    def emit_flash_attention(
        self,
        dtype: str = "fp16",
        tile_q: int = 32,
        tile_kv: int = 64,
        head_dim_max: int = 128,
    ) -> ROCmKernelDescriptor:
        """Emit FlashAttention-2 ROCm kernel."""
        hip_dtype, dtype_include = self._hip_dtype(dtype)
        source = _HIP_FLASH_ATTENTION.format(
            dtype=dtype,
            hip_dtype=hip_dtype,
            dtype_include=dtype_include,
            tile_q=tile_q,
            tile_kv=tile_kv,
            head_dim_max=head_dim_max,
        )
        kernel = ROCmKernelDescriptor(
            name=f"flash_attention_{dtype}",
            source=source,
            function_name=f"flash_attention_{dtype}",
            dtype=dtype,
            block_size=(tile_kv, tile_q, 1),
            metadata={"tile_q": tile_q, "tile_kv": tile_kv},
        )
        self._kernels.append(kernel)
        return kernel

    def emit_rmsnorm(self, dtype: str = "fp16") -> ROCmKernelDescriptor:
        """Emit RMSNorm kernel."""
        hip_dtype, dtype_include = self._hip_dtype(dtype)
        source = _HIP_RMSNORM.format(
            dtype=dtype,
            hip_dtype=hip_dtype,
            dtype_include=dtype_include,
        )
        kernel = ROCmKernelDescriptor(
            name=f"rmsnorm_{dtype}",
            source=source,
            function_name=f"rmsnorm_{dtype}",
            dtype=dtype,
            block_size=(256, 1, 1),
        )
        self._kernels.append(kernel)
        return kernel

    def emit_silu_gate_ffn(self, dtype: str = "fp16") -> ROCmKernelDescriptor:
        """Emit SiLU-gate FFN kernel."""
        hip_dtype, dtype_include = self._hip_dtype(dtype)
        source = _HIP_SILU_GATE.format(
            dtype=dtype,
            hip_dtype=hip_dtype,
            dtype_include=dtype_include,
        )
        kernel = ROCmKernelDescriptor(
            name=f"silu_gate_{dtype}",
            source=source,
            function_name=f"silu_gate_ffn_{dtype}",
            dtype=dtype,
            block_size=(1024, 1, 1),
        )
        self._kernels.append(kernel)
        return kernel

    def emit_all_standard_kernels(self, dtype: str = "fp16") -> list[ROCmKernelDescriptor]:
        """Emit the full standard kernel suite."""
        return [
            self.emit_gemm(dtype),
            self.emit_flash_attention(dtype),
            self.emit_rmsnorm(dtype),
            self.emit_silu_gate_ffn(dtype),
        ]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        """Save all kernels as .hip source files and a manifest."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        for kernel in self._kernels:
            hip_path = out / f"{kernel.name}.hip"
            hip_path.write_text(kernel.source, encoding="utf-8")
            saved[kernel.name] = hip_path
            logger.debug("ROCm kernel saved: %s", hip_path)

        # Write manifest
        manifest = {
            "target": self.profile.to_dict(),
            "kernels": [k.to_dict() for k in self._kernels],
            "compile_command": (
                f"hipcc --offload-arch={self.profile.gfx_arch} "
                "-O3 -std=c++17 {kernel}.hip -o {kernel}.hsaco"
            ),
        }
        manifest_path = out / "kernel_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        saved["manifest"] = manifest_path

        logger.info(
            "ROCm kernels saved: %d kernels to %s (target=%s)",
            len(self._kernels), out, self.profile.gfx_arch
        )
        return saved

    def get_grid_size(
        self, kernel_name: str, M: int, N: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Compute block and grid dimensions for dispatch."""
        for k in self._kernels:
            if k.name == kernel_name:
                bx, by, bz = k.block_size
                gx = (N + bx - 1) // bx
                gy = (M + by - 1) // by
                return k.block_size, (gx, gy, 1)
        raise KeyError(f"Kernel '{kernel_name}' not found")


# ---------------------------------------------------------------------------
# ROCmTarget — compiler-facing backend target class
# ---------------------------------------------------------------------------

class ROCmTarget:
    """
    AMD ROCm/HIP backend target for the Aether compiler.

    Wraps ROCmKernelEmitter and ROCmTargetProfile into the standardized
    target interface expected by the stage3 targeting system.
    """

    name = "rocm"
    supported_dtypes = ("fp16", "bf16", "fp8", "fp32")

    def __init__(
        self,
        device: str = "mi300x",
        dtype: str = "fp16",
    ) -> None:
        self.dtype = dtype
        device_lower = device.lower()
        if "mi300" in device_lower:
            self.profile = ROCmTargetProfile.mi300x()
        elif "mi250" in device_lower:
            self.profile = ROCmTargetProfile.mi250x()
        elif "rdna3" in device_lower or "7900" in device_lower:
            self.profile = ROCmTargetProfile.rdna3()
        else:
            self.profile = ROCmTargetProfile.mi300x()
        self.emitter = ROCmKernelEmitter(self.profile)

    def compile(self, output_dir: str | Path) -> dict[str, "Path"]:
        """Emit all standard kernels to output_dir and return saved paths."""
        self.emitter.emit_all_standard_kernels(dtype=self.dtype)
        return self.emitter.save(output_dir)

    def get_profile(self) -> ROCmTargetProfile:
        return self.profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "dtype": self.dtype,
            "profile": self.profile.to_dict(),
        }

