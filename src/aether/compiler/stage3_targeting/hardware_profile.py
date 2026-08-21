"""
Hardware profile definitions and detection.

Each `HardwareProfile` describes a specific hardware target's capabilities:
compute, memory, bandwidth, and the optimal settings for Aether runtime.

Versions:
  v3.1 — original 10 targets (CUDA sm70-sm100, Apple Metal, AMD RDNA3/CDNA3, OpenVINO, Qualcomm QNN, CPU)
  v4.0 — added 9 new targets (sm120, sm130, sm100_tee, 3x RISC-V NPU, Xilinx FPGA, MI350X, Cloud AI 100)
  v5.0 — added 6 new targets (GB300, MI455X CDNA5, ternary CPU x86/ARM, FPGA ternary, Cervell)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import BACKEND_BY_TARGET, SUPPORTED_TARGET_IDS


@dataclass
class HardwareProfile:
    """Describes a specific hardware target and its capabilities."""

    target_id: str
    """Hardware target identifier (e.g., 'cuda_sm90')."""

    name: str
    """Human-readable name (e.g., 'NVIDIA H100 (Hopper)')."""

    compute_capability: str = "unknown"
    """Compute capability version or identifier."""

    memory_gb: float = 0.0
    """Total device memory in GB."""

    memory_bandwidth_gb_s: float = 0.0
    """Memory bandwidth in GB/s."""

    tensor_core_flops: float = 0.0
    """Approximate TFLOP/s for BF16/FP16 tensor core operations."""

    flops_fp4: float = 0.0
    """TFLOP/s for FP4 tensor core operations (NVFP4/MXFP4).
    Only non-zero on sm_100+ (Blackwell) and sm_120+ (Rubin) targets.
    Research basis: NVIDIA Blackwell whitepaper 2024, Rubin R100 whitepaper 2026."""

    supports_fp8: bool = False
    """Whether this target supports native FP8."""

    supports_fp4: bool = False
    """Whether this target supports native FP4 (NVFP4/MXFP4/MXFP6).
    Requires Blackwell sm_100 or newer. Enables Pass 3 NVFP4 precision path."""

    supports_bf16: bool = True
    """Whether this target supports native BF16."""

    supports_ternary: bool = False
    """Whether this target supports BitNet b1.58 ternary (ADD-only, no multiply).
    True for cpu_avx512_ternary, cpu_neon_ternary, fpga_ternary.
    Research basis: bitnet.cpp 2026; 70-82% CPU energy reduction."""

    supports_mxfp6: bool = False
    """Whether this target supports MXFP6 format (AMD MI455X CDNA5 only).
    MXFP6 is between FP8 and FP4 — new precision format for CDNA5 arch.
    Research basis: AMD Instinct MI455X whitepaper July 2026."""

    supports_tee: bool = False
    """Whether this target supports Trusted Execution Environment (TEE) / Confidential Computing.
    True for cuda_sm100_tee (NVIDIA CC mode), and implicitly for Intel/AMD CPU TEE targets.
    Research basis: Confidential LLM Tinfoil Red Hat 2026, NVIDIA H100/B200 CC mode."""

    supports_flash_attention: bool = False
    """Whether FlashAttention is available on this target."""

    recommended_flash_attention: str = "flash_attention_3"
    """Best available FlashAttention variant."""

    recommended_backend: str | None = None
    """Best available backend plugin for this target."""

    backend_candidates: list[str] = field(default_factory=list)
    """Priority-ordered candidate backend names for this target."""

    max_shm_size_bytes: int = 0
    """Maximum shared memory size in bytes."""

    warp_size: int = 32
    """CUDA warp size or SM thread group size. 64 for AMD GCN/CDNA."""

    sm_count: int = 0
    """Number of streaming multiprocessors (or equivalent compute units)."""

    nvlink_bandwidth_gb_s: float = 0.0
    """NVLink bandwidth in GB/s for all-reduce planning (Pass 6 parallelism).
    0 for targets without NVLink. NVLink 5 = 1800 GB/s, NVLink 6 = 3600 GB/s.
    Research basis: NVIDIA Rubin R100 whitepaper 2026, PRD §3.1."""

    tdp_watts: float = 0.0
    """Thermal Design Power in watts (for Pass 16 green energy profiling).
    Used by R7 GreenPowerManager to estimate energy per request.
    Research basis: MELODI 2026, CodeCarbon 2026."""

    tee_backend: str | None = None
    """TEE backend identifier when supports_tee=True.
    One of: 'nvidia_cc', 'intel_tdx', 'amd_sev_snp', 'openpcc'.
    Research basis: Intel TDX + NVIDIA CC Joint Paper 2026."""

    is_riscv_npu: bool = False
    """Whether this target is a RISC-V NPU that uses the abstract RISC-V NPU IR.
    True for: riscv_mips_s8200, riscv_sifive_x160, riscv_xuantie_c930, riscv_cervell.
    Research basis: PRD §3.2 RISC-V NPU Abstract IR."""

    abstract_ir_family: str | None = None
    """RISC-V NPU IR family name for abstract IR routing.
    Used by riscv_npu_ir.py to select the correct vendor plugin.
    Values: 'mips_npu', 'sifive_x', 'xuantie_c', 'cervell'."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional hardware-specific attributes."""

    @staticmethod
    def from_target_id(target_id: str) -> HardwareProfile | None:
        """Create a profile from a target ID.

        Args:
            target_id: A valid target identifier.

        Returns:
            A HardwareProfile, or None if the target ID is unknown.
        """
        if target_id not in SUPPORTED_TARGET_IDS:
            return None
        backends = BACKEND_BY_TARGET.get(target_id, ["aether_cpu"])
        profile_data = _TARGET_PROFILES.get(target_id)
        if profile_data is None:
            name = target_id.replace("_", " ").title()
            profile = HardwareProfile(
                target_id=target_id,
                name=name,
                recommended_backend=backends[0],
                backend_candidates=list(backends),
            )
            return profile
        profile = HardwareProfile(
            target_id=target_id,
            name=profile_data["name"],
            compute_capability=profile_data["compute_capability"],
            memory_gb=profile_data["memory_gb"],
            memory_bandwidth_gb_s=profile_data["memory_bandwidth_gb_s"],
            tensor_core_flops=profile_data["tensor_core_flops"],
            flops_fp4=profile_data.get("flops_fp4", 0.0),
            supports_fp8=profile_data["supports_fp8"],
            supports_fp4=profile_data.get("supports_fp4", False),
            supports_bf16=profile_data["supports_bf16"],
            supports_ternary=profile_data.get("supports_ternary", False),
            supports_mxfp6=profile_data.get("supports_mxfp6", False),
            supports_tee=profile_data.get("supports_tee", False),
            supports_flash_attention=profile_data.get("supports_flash_attention", False),
            recommended_flash_attention=profile_data.get("recommended_flash_attention", "flash_attention_3"),
            recommended_backend=backends[0],
            backend_candidates=list(backends),
            max_shm_size_bytes=profile_data.get("max_shm_size_bytes", 0),
            warp_size=profile_data.get("warp_size", 32),
            sm_count=profile_data.get("sm_count", 0),
            nvlink_bandwidth_gb_s=profile_data.get("nvlink_bandwidth_gb_s", 0.0),
            tdp_watts=profile_data.get("tdp_watts", 0.0),
            tee_backend=profile_data.get("tee_backend"),
            is_riscv_npu=profile_data.get("is_riscv_npu", False),
            abstract_ir_family=profile_data.get("abstract_ir_family"),
            attributes=profile_data.get("attributes", {}),
        )
        return profile

    @staticmethod
    def auto() -> HardwareProfile:
        """Create a profile for the current hardware.

        This autodetects the hardware by querying the environment.
        Falls back to a CPU profile if no accelerator is found.
        """
        try:
            from aether.core.types import HardwareTarget as HTarget
            target = HTarget.auto()
            profile = HardwareProfile.from_target_id(target.value)
            if profile:
                return profile
        except ImportError:
            pass
        # CPU fallback.  The backend contract (``plan.recommend_backend``)
        # requires CPU targets to execute on the native Aether CPU engine —
        # never the development-only pytorch backend.
        import os
        return HardwareProfile(
            target_id="cpu_avx512",
            name=os.uname().machine if hasattr(os, "uname") else "x86_64",
            compute_capability="cpu",
            memory_gb=16.0,
            memory_bandwidth_gb_s=50.0,
            tensor_core_flops=0.0,
            supports_fp8=False,
            supports_bf16=False,
            recommended_backend="aether_cpu",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "memory_gb": self.memory_gb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "tensor_core_flops": self.tensor_core_flops,
            "flops_fp4": self.flops_fp4,
            "supports_fp8": self.supports_fp8,
            "supports_fp4": self.supports_fp4,
            "supports_bf16": self.supports_bf16,
            "supports_ternary": self.supports_ternary,
            "supports_mxfp6": self.supports_mxfp6,
            "supports_tee": self.supports_tee,
            "supports_flash_attention": self.supports_flash_attention,
            "recommended_flash_attention": self.recommended_flash_attention,
            "recommended_backend": self.recommended_backend,
            "max_shm_size_bytes": self.max_shm_size_bytes,
            "warp_size": self.warp_size,
            "sm_count": self.sm_count,
            "nvlink_bandwidth_gb_s": self.nvlink_bandwidth_gb_s,
            "tdp_watts": self.tdp_watts,
            "tee_backend": self.tee_backend,
            "is_riscv_npu": self.is_riscv_npu,
            "abstract_ir_family": self.abstract_ir_family,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> HardwareProfile:
        return HardwareProfile(
            target_id=data["target_id"],
            name=data["name"],
            compute_capability=data.get("compute_capability", "unknown"),
            memory_gb=data.get("memory_gb", 0.0),
            memory_bandwidth_gb_s=data.get("memory_bandwidth_gb_s", 0.0),
            tensor_core_flops=data.get("tensor_core_flops", 0.0),
            flops_fp4=data.get("flops_fp4", 0.0),
            supports_fp8=data.get("supports_fp8", False),
            supports_fp4=data.get("supports_fp4", False),
            supports_bf16=data.get("supports_bf16", True),
            supports_ternary=data.get("supports_ternary", False),
            supports_mxfp6=data.get("supports_mxfp6", False),
            supports_tee=data.get("supports_tee", False),
            supports_flash_attention=data.get("supports_flash_attention", False),
            recommended_flash_attention=data.get("recommended_flash_attention", "flash_attention_3"),
            recommended_backend=data.get("recommended_backend"),
            max_shm_size_bytes=data.get("max_shm_size_bytes", 0),
            warp_size=data.get("warp_size", 32),
            sm_count=data.get("sm_count", 0),
            nvlink_bandwidth_gb_s=data.get("nvlink_bandwidth_gb_s", 0.0),
            tdp_watts=data.get("tdp_watts", 0.0),
            tee_backend=data.get("tee_backend"),
            is_riscv_npu=data.get("is_riscv_npu", False),
            abstract_ir_family=data.get("abstract_ir_family"),
        )

    def __repr__(self) -> str:
        return f"HardwareProfile({self.target_id}: {self.name}, {self.memory_gb}GB)"


# ── Target profile database ────────────────────────────────────────────────────
#
# Every target in SUPPORTED_TARGET_IDS should have an entry here.
# Targets without an entry fall back to a minimal generic profile.
#
# Specification sources:
#   v3.1 targets:  NVIDIA/AMD/Apple/Intel/Qualcomm official whitepapers.
#   v4.0 targets:  NVIDIA Rubin R100 whitepaper (2026), MIPS S8200 datasheet (2026),
#                  SiFive X160 product brief (2026), XuanTie C930 datasheet (2026),
#                  AMD MI350X announcement (2026), Qualcomm Cloud AI 100 Ultra brief (2026).
#   v5.0 targets:  NVIDIA GB300 NVL72 whitepaper (2026), AMD MI455X/Helios whitepaper Jul-2026,
#                  bitnet.cpp measurements (2026), Semidynamics Cervell brief (2026).

_TARGET_PROFILES: dict[str, dict[str, Any]] = {

    # ── v3.1 NVIDIA ────────────────────────────────────────────────────────────

    "cuda_sm70": {
        "name": "NVIDIA V100 (Volta)",
        "compute_capability": "7.0",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 900.0,
        "tensor_core_flops": 125.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 114688,
        "warp_size": 32,
        "sm_count": 80,
        "nvlink_bandwidth_gb_s": 300.0,  # NVLink 2 aggregate
        "tdp_watts": 300.0,
    },
    "cuda_sm80": {
        "name": "NVIDIA A100 (Ampere)",
        "compute_capability": "8.0",
        "memory_gb": 80.0,
        "memory_bandwidth_gb_s": 2039.0,
        "tensor_core_flops": 312.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 164864,
        "warp_size": 32,
        "sm_count": 108,
        "nvlink_bandwidth_gb_s": 600.0,  # NVLink 3
        "tdp_watts": 400.0,
    },
    "cuda_sm89": {
        "name": "NVIDIA RTX 4090 (Ada)",
        "compute_capability": "8.9",
        "memory_gb": 24.0,
        "memory_bandwidth_gb_s": 1008.0,
        "tensor_core_flops": 660.0,
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 131072,
        "warp_size": 32,
        "sm_count": 128,
        "nvlink_bandwidth_gb_s": 0.0,  # No NVLink on RTX consumer
        "tdp_watts": 450.0,
    },
    "cuda_sm90": {
        "name": "NVIDIA H100 (Hopper)",
        "compute_capability": "9.0",
        "memory_gb": 80.0,
        "memory_bandwidth_gb_s": 3350.0,
        "tensor_core_flops": 989.0,
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 228352,
        "warp_size": 32,
        "sm_count": 132,
        "nvlink_bandwidth_gb_s": 900.0,  # NVLink 4
        "tdp_watts": 700.0,
    },
    "cuda_sm100": {
        "name": "NVIDIA B200 (Blackwell)",
        "compute_capability": "10.0",
        "memory_gb": 192.0,
        "memory_bandwidth_gb_s": 8000.0,
        "tensor_core_flops": 2250.0,
        "flops_fp4": 9000.0,   # NVFP4: 4.5x FP8 throughput per B200 whitepaper
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_4",
        "max_shm_size_bytes": 262144,
        "warp_size": 32,
        "sm_count": 160,
        "nvlink_bandwidth_gb_s": 1800.0,  # NVLink 5 aggregate per-GPU
        "tdp_watts": 1000.0,
    },

    # ── v4.0 NVIDIA ────────────────────────────────────────────────────────────

    "cuda_sm120": {
        "name": "NVIDIA Rubin R100 (sm_120)",
        # Specs from NVIDIA Rubin R100 whitepaper, GTC 2026.
        # Key advances over B200: 3rd-gen TE, inline TMA, NVLink 6, HBM4.
        "compute_capability": "12.0",
        "memory_gb": 288.0,       # 288 GB HBM4 per GPU
        "memory_bandwidth_gb_s": 22000.0,  # 22 TB/s HBM4
        "tensor_core_flops": 5000.0,       # ~5 PFLOPS FP8 estimated
        "flops_fp4": 50000.0,     # 50 PFLOPS FP4 per Rubin R100 whitepaper
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_4",
        "max_shm_size_bytes": 524288,  # Estimated 512 KB per SM for Rubin
        "warp_size": 32,
        "sm_count": 224,          # 224 SMs per Rubin R100 GPU
        "nvlink_bandwidth_gb_s": 3600.0,   # NVLink 6: 3600 GB/s (2x NVLink 5)
        "tdp_watts": 1200.0,      # Estimated based on HBM4 + SM scale
        "attributes": {
            "inline_tma": True,       # PRD §3.1: inline TMA descriptor updates for MoE (15-20% dispatch overhead reduction)
            "gen3_transformer_engine": True,  # 3rd-gen TE for adaptive NVFP4 compression
            "nvlink_version": 6,
            "hbm_generation": 4,
        },
    },
    "cuda_sm130": {
        "name": "NVIDIA Rubin Ultra (sm_130, ~100 PFLOPS FP4)",
        # Future-proofed placeholder per PRD §3. Dual Rubin cores.
        # Specs from NVIDIA Rubin Ultra 2027 roadmap announcement.
        "compute_capability": "13.0",
        "memory_gb": 576.0,       # Estimated 2x Rubin R100 memory
        "memory_bandwidth_gb_s": 44000.0,  # Estimated 2x R100 bandwidth
        "tensor_core_flops": 10000.0,
        "flops_fp4": 100000.0,    # ~100 PFLOPS FP4 (dual-core)
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_4",
        "max_shm_size_bytes": 1048576,
        "warp_size": 32,
        "sm_count": 448,          # Estimated 2x sm_120
        "nvlink_bandwidth_gb_s": 7200.0,   # Estimated NVLink 7 or 2x NVLink 6
        "tdp_watts": 2400.0,
        "attributes": {
            "is_placeholder": True,  # Placeholder — finalize when hardware ships
            "nvlink_version": 7,
            "hbm_generation": 4,
        },
    },
    "cuda_sm100_tee": {
        "name": "NVIDIA B200 Confidential Computing (CC mode)",
        # Same hardware as cuda_sm100 but with CC mode enabled.
        # Research: Confidential LLM Tinfoil Red Hat 2026; 5-8% throughput overhead.
        "compute_capability": "10.0",
        "memory_gb": 192.0,
        "memory_bandwidth_gb_s": 8000.0,
        "tensor_core_flops": 2250.0,
        "flops_fp4": 9000.0,
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_tee": True,
        "tee_backend": "nvidia_cc",
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_4",
        "max_shm_size_bytes": 262144,
        "warp_size": 32,
        "sm_count": 160,
        "nvlink_bandwidth_gb_s": 1800.0,
        "tdp_watts": 1000.0,
        "attributes": {
            "cc_mode": True,                # Confidential Computing mode active
            "encrypted_hbm": True,          # GPU HBM memory encryption enabled
            "attestation_support": True,    # Remote attestation report generation
            "tee_overhead_pct": 7.0,        # Measured: 5-8% overhead (mid estimate)
        },
    },
    "cuda_sm100_gb300": {
        "name": "NVIDIA GB300 Blackwell Ultra",
        # Specs from NVIDIA GB300 NVL72 product brief 2026.
        # 1.5x B200 FP4 throughput. Designed for test-time scaling + reasoning.
        "compute_capability": "10.0",      # Sub-revision of sm_100 architecture
        "memory_gb": 192.0,                # Same HBM3e+ as B200 baseline
        "memory_bandwidth_gb_s": 8000.0,
        "tensor_core_flops": 2250.0,
        "flops_fp4": 13500.0,  # 1.5x B200 FP4: 9000 * 1.5 = 13500 TFLOPS
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_4",
        "max_shm_size_bytes": 262144,
        "warp_size": 32,
        "sm_count": 160,
        "nvlink_bandwidth_gb_s": 1800.0,
        "tdp_watts": 1000.0,
        "attributes": {
            "gb300_enhanced_te": True,   # Enhanced Tensor Engine for GB300
            "test_time_scaling_optimized": True,  # Designed for BoN/beam search
            "fp4_sensitivity_threshold_lower": True,  # PRD §36.3: lower FP4 threshold
        },
    },

    # ── v4.0 AMD ──────────────────────────────────────────────────────────────

    "amd_mi350x": {
        "name": "AMD MI350X (CDNA4, HBM3e)",
        # Specs from AMD MI350X announcement, successor to MI300X.
        "compute_capability": "cdna4",
        "memory_gb": 288.0,       # HBM3e — larger than MI300X 192GB
        "memory_bandwidth_gb_s": 8000.0,   # Estimated ~8 TB/s HBM3e
        "tensor_core_flops": 1600.0,       # FP8 estimated
        "flops_fp4": 6400.0,               # FP4 estimated (4x FP8)
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 163840,      # Estimated similar to MI300X
        "warp_size": 64,                   # AMD GCN wavefront size
        "sm_count": 304,                   # Estimated same CU count as MI300X
        "nvlink_bandwidth_gb_s": 0.0,      # AMD uses Infinity Fabric (not NVLink)
        "tdp_watts": 750.0,
        "attributes": {
            "infinity_fabric_bandwidth_gb_s": 1000.0,  # AMD IF bandwidth
            "hbm_generation": "3e",
        },
    },
    "rocm_cdna5_mi455x": {
        "name": "AMD MI455X (CDNA5, 432 GB HBM4, 23.3 TB/s)",
        # Specs from AMD Instinct MI455X whitepaper, July 2026 (AMD Helios rack).
        # Key innovations: MXFP6 format, 4.4x MI300X bandwidth, 2.25x capacity.
        "compute_capability": "cdna5",
        "memory_gb": 432.0,          # 432 GB HBM4 — 2.25x MI300X (192 GB)
        "memory_bandwidth_gb_s": 23300.0,  # 23.3 TB/s — 4.4x MI300X (5.3 TB/s)
        "tensor_core_flops": 6500.0,       # FP8: 5x MI300X improvement
        "flops_fp4": 26000.0,              # FP4 estimated 4x FP8
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_mxfp6": True,  # NEW: MXFP6 format between FP8 and FP4
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 262144,      # Estimated for CDNA5
        "warp_size": 64,
        "sm_count": 304,                   # Estimated; CDNA5 CU count
        "nvlink_bandwidth_gb_s": 0.0,      # AMD uses Infinity Fabric
        "tdp_watts": 800.0,
        "attributes": {
            "mxfp6_enabled": True,         # MXFP6: new precision between FP8 and FP4
            "infinity_fabric_bandwidth_gb_s": 1500.0,
            "hbm_generation": 4,
            "cdna_generation": 5,
        },
    },

    # ── v3.1 Apple ────────────────────────────────────────────────────────────

    "metal_m1": {
        "name": "Apple M1/M2",
        "compute_capability": "m1",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 200.0,
        "tensor_core_flops": 10.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "memory_efficient_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 32,
        "sm_count": 16,
        "tdp_watts": 20.0,
    },
    "metal_m3": {
        "name": "Apple M3/M4/M5",
        "compute_capability": "m3",
        "memory_gb": 64.0,
        "memory_bandwidth_gb_s": 400.0,
        "tensor_core_flops": 18.0,
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "metal_4_neural_engine",
        "max_shm_size_bytes": 32768,
        "warp_size": 32,
        "sm_count": 20,
        "tdp_watts": 35.0,
    },

    # ── v3.1 AMD ──────────────────────────────────────────────────────────────

    "rocm_rdna3": {
        "name": "AMD RX 7000 Series",
        "compute_capability": "rdna3",
        "memory_gb": 24.0,
        "memory_bandwidth_gb_s": 960.0,
        "tensor_core_flops": 122.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 65536,
        "warp_size": 64,
        "sm_count": 60,
        "tdp_watts": 300.0,
    },
    "rocm_cdna3": {
        "name": "AMD MI300X",
        "compute_capability": "cdna3",
        "memory_gb": 192.0,
        "memory_bandwidth_gb_s": 5300.0,
        "tensor_core_flops": 1300.0,
        "flops_fp4": 5200.0,
        "supports_fp8": True,
        "supports_fp4": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 131072,
        "warp_size": 64,
        "sm_count": 304,
        "nvlink_bandwidth_gb_s": 0.0,
        "tdp_watts": 750.0,
    },

    # ── v3.1 Intel / Qualcomm / CPU ───────────────────────────────────────────

    "openvino_npu": {
        "name": "Intel Arc NPU",
        "compute_capability": "openvino_npu",
        "memory_gb": 16.0,
        "memory_bandwidth_gb_s": 120.0,
        "tensor_core_flops": 45.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 16,
        "sm_count": 8,
        "tdp_watts": 25.0,
    },
    "openvino_gpu": {
        "name": "Intel GPU (OpenVINO)",
        "compute_capability": "openvino_gpu",
        "memory_gb": 0.0,
        "memory_bandwidth_gb_s": 0.0,
        "tensor_core_flops": 0.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "memory_efficient_attention",
        "max_shm_size_bytes": 0,
        "warp_size": 16,
        "sm_count": 0,
        "tdp_watts": 0.0,
    },
    "qualcomm_qnn": {
        "name": "Qualcomm Snapdragon NPU",
        "compute_capability": "qnn",
        "memory_gb": 24.0,
        "memory_bandwidth_gb_s": 150.0,
        "tensor_core_flops": 60.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 16,
        "sm_count": 12,
        "tdp_watts": 15.0,
    },
    "cpu_avx512": {
        "name": "x86_64 (AVX-512)",
        "compute_capability": "cpu",
        "memory_gb": 256.0,        # Typical server DRAM
        "memory_bandwidth_gb_s": 200.0,    # DDR5 bandwidth
        "tensor_core_flops": 10.0,         # Approx AVX-512 FP32 throughput
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 8,
        "sm_count": 0,             # CPU has no SM concept
        "tdp_watts": 280.0,        # Typical server CPU TDP
    },
    "cpu_avx2": {
        "name": "x86_64 (AVX2)",
        "compute_capability": "cpu_avx2",
        "memory_gb": 128.0,
        "memory_bandwidth_gb_s": 120.0,
        "tensor_core_flops": 6.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 1,
        "sm_count": 0,
        "tdp_watts": 180.0,
    },
    "cpu_neon": {
        "name": "ARM (NEON SIMD)",
        "compute_capability": "cpu_arm",
        "memory_gb": 64.0,
        "memory_bandwidth_gb_s": 100.0,
        "tensor_core_flops": 4.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 16384,
        "warp_size": 4,
        "sm_count": 0,
        "tdp_watts": 15.0,
    },

    # ── v4.0 RISC-V NPU targets ────────────────────────────────────────────────
    # All RISC-V NPU targets use the abstract RISC-V NPU IR layer (riscv_npu_ir.py).
    # Research basis: PRD §3.2, MIPS S8200 / SiFive X160 / XuanTie C930 / Cervell datasheets 2026.

    "riscv_mips_s8200": {
        "name": "MIPS S8200 NPU (RISC-V agentic, sub-10W)",
        # Specs from MIPS S8200 product brief 2026. Designed for agentic edge NPU.
        "compute_capability": "mips_s8200",
        "memory_gb": 8.0,          # Edge device: shared LPDDR5
        "memory_bandwidth_gb_s": 102.0,    # LPDDR5 @ 6.4 GT/s typical
        "tensor_core_flops": 64.0,         # 64 TOPS INT8 per MIPS datasheet
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_ternary": False,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 8192,
        "warp_size": 4,
        "sm_count": 4,
        "tdp_watts": 8.0,          # Sub-10W design target per PRD
        "is_riscv_npu": True,
        "abstract_ir_family": "mips_npu",
        "attributes": {
            "isa": "RISC-V",
            "npu_type": "agentic_edge",
            "max_model_size_gb": 4.0,      # Practical limit for sub-10W envelope
        },
    },
    "riscv_sifive_x160": {
        "name": "SiFive Intelligence X160 (scalar+vector+matrix RISC-V)",
        # Specs from SiFive X160 product brief 2026. 2nd-gen AI IP — unified execution.
        "compute_capability": "sifive_x160",
        "memory_gb": 16.0,
        "memory_bandwidth_gb_s": 204.0,    # LPDDR5X bandwidth
        "tensor_core_flops": 128.0,        # Estimated INT8 TOPS from SiFive AI IP v2
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 16384,
        "warp_size": 4,
        "sm_count": 8,
        "tdp_watts": 5.0,          # SiFive X160 designed for power-efficient workloads
        "is_riscv_npu": True,
        "abstract_ir_family": "sifive_x",
        "attributes": {
            "isa": "RISC-V",
            "vector_extension": "RVV-1.0",
            "matrix_extension": "RMMM-0.7",  # SiFive matrix extension
        },
    },
    "riscv_xuantie_c930": {
        "name": "Alibaba XuanTie C930 (RISC-V + integrated NPU)",
        # Specs from Alibaba T-Head XuanTie C930 datasheet 2026. High-perf RISC-V for edge server/robotics.
        "compute_capability": "xuantie_c930",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 256.0,    # LPDDR5X or DDR5 depending on config
        "tensor_core_flops": 256.0,        # Integrated NPU TOPS
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 4,
        "sm_count": 16,
        "tdp_watts": 25.0,         # Edge server / robotics envelope
        "is_riscv_npu": True,
        "abstract_ir_family": "xuantie_c",
        "attributes": {
            "isa": "RISC-V",
            "vector_extension": "RVV-1.0",
            "npu_integrated": True,
        },
    },
    "riscv_cervell": {
        "name": "Semidynamics Cervell (unified scalar/vector/tensor RISC-V NPU)",
        # Specs from Semidynamics Cervell brief 2026. Unified scalar+vector+tensor execution unit.
        "compute_capability": "cervell_npu",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 512.0,    # HBM2e or LPDDR5X depending on config
        "tensor_core_flops": 512.0,        # Estimated from tensor unit count
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 65536,
        "warp_size": 8,
        "sm_count": 32,
        "tdp_watts": 45.0,
        "is_riscv_npu": True,
        "abstract_ir_family": "cervell",
        "attributes": {
            "isa": "RISC-V",
            "unified_execution": True,     # Scalar + vector + tensor in one unit
            "quadric_toolchain_compatible": True,  # Supports Quadric DevStudio toolchain
        },
    },

    # ── v4.0 FPGA ────────────────────────────────────────────────────────────

    "fpga_xilinx_vu9p": {
        "name": "Xilinx VU9P FPGA (decode-only, 10x lower cost/token vs GPU)",
        # Research: FPGA-based LLM inference; 10x cost reduction vs GPU for pure decode.
        "compute_capability": "fpga_xilinx_vu9p",
        "memory_gb": 64.0,         # DRAM + HBM2 depending on FPGA card config
        "memory_bandwidth_gb_s": 460.0,    # HBM2 on VU9P
        "tensor_core_flops": 40.0,         # Equivalent TFLOPS via DSP blocks
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,    # Custom fixed-point preferred for FPGA
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 16384,
        "warp_size": 1,            # FPGA has no warp concept
        "sm_count": 0,
        "tdp_watts": 250.0,        # Xilinx VU9P board TDP
        "attributes": {
            "fpga_family": "UltraScale+",
            "dsp_blocks": 6840,
            "brams": 4032,
            "decode_only": True,   # Optimized for decode — not prefill
            "quantization": "INT8/INT4",
        },
    },

    # ── v4.0 Qualcomm Cloud ────────────────────────────────────────────────────

    "qualcomm_cloud_ai100": {
        "name": "Qualcomm Cloud AI 100 Ultra",
        # Specs from Qualcomm Cloud AI 100 Ultra product brief for data center deployment.
        "compute_capability": "qaic100",
        "memory_gb": 128.0,        # Cloud AI 100 Ultra has up to 128 GB LPDDR5
        "memory_bandwidth_gb_s": 2000.0,   # LPDDR5X aggregate
        "tensor_core_flops": 700.0,        # INT8 TOPS per Qualcomm datasheet
        "flops_fp4": 0.0,
        "supports_fp8": True,
        "supports_fp4": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 65536,
        "warp_size": 16,
        "sm_count": 16,
        "tdp_watts": 75.0,         # Cloud AI 100 Ultra TDP for data center
        "attributes": {
            "cloud_ai100_nsp_count": 16,   # Number of AI accelerator engines
            "lpddr5_channels": 8,
        },
    },

    # ── v5.0 Ternary CPU targets ────────────────────────────────────────────────
    # Research: BitNet b1.58 (Microsoft Research 2024/2026), bitnet.cpp 2026.
    # Key: ADD-only, NO multiply instruction used. 70-82% energy reduction vs BF16 CPU.

    "cpu_avx512_ternary": {
        "name": "x86_64 AVX2 Ternary (BitNet b1.58, ADD-only)",
        # Runs BitNet b1.58 {-1, 0, +1} via AVX2 popcount+subtract kernels.
        # Research: bitnet.cpp 2026: 2.1x speedup, 70-82% energy reduction.
        "compute_capability": "cpu_avx2_ternary",
        "memory_gb": 256.0,        # Server DRAM (model stored as 2-bit packed ternary)
        "memory_bandwidth_gb_s": 200.0,
        "tensor_core_flops": 0.0,  # No tensor cores — ADD-only execution
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_ternary": True,  # BitNet b1.58 ADD-only ternary
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 8,
        "sm_count": 0,
        "tdp_watts": 280.0,
        "attributes": {
            "isa": "x86_64",
            "vector_extension": "AVX2",
            "ternary_encoding": "{-1=0b00, 0=0b01, 1=0b10}",
            "memory_reduction_vs_bf16": "10x",
            "energy_reduction_pct": 76,    # 70-82% mean
        },
    },
    "cpu_neon_ternary": {
        "name": "ARM NEON Ternary (BitNet b1.58, mobile/Apple M-series)",
        # Research: bitnet.cpp 2026 ARM NEON backend. 2.37x speedup on M-series.
        "compute_capability": "cpu_arm_ternary",
        "memory_gb": 64.0,
        "memory_bandwidth_gb_s": 100.0,
        "tensor_core_flops": 0.0,
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_ternary": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 16384,
        "warp_size": 4,
        "sm_count": 0,
        "tdp_watts": 15.0,
        "attributes": {
            "isa": "ARM",
            "vector_extension": "NEON",
            "ternary_encoding": "{-1=0b00, 0=0b01, 1=0b10}",
            "memory_reduction_vs_bf16": "10x",
            "energy_reduction_pct": 76,
        },
    },

    # ── v5.0 FPGA Ternary ────────────────────────────────────────────────────────

    "fpga_ternary": {
        "name": "FPGA BTC-LLM Ternary (0.8-1.58 bit, purpose-built addition circuits)",
        # Research: BTC-LLM 2026. FPGA purpose-built addition-only circuits.
        # Effective precision: 0.8-1.58 bits per weight. 10x energy efficiency vs GPU.
        "compute_capability": "fpga_ternary",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 200.0,
        "tensor_core_flops": 0.0,      # No traditional tensor cores
        "flops_fp4": 0.0,
        "supports_fp8": False,
        "supports_fp4": False,
        "supports_bf16": False,
        "supports_ternary": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "paged_attention",
        "max_shm_size_bytes": 8192,
        "warp_size": 1,
        "sm_count": 0,
        "tdp_watts": 50.0,
        "attributes": {
            "btc_effective_bits": 1.0,     # 0.8-1.11 bit via binary codebook
            "purpose_built_adder_circuits": True,
            "energy_efficiency_vs_gpu": "10x",
        },
    },
}


