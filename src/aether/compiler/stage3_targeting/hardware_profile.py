"""
Hardware profile definitions and detection.

Each `HardwareProfile` describes a specific hardware target's capabilities:
compute, memory, bandwidth, and the optimal settings for Aether runtime.
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
    """Approximate TFLOP/s for tensor core operations."""

    supports_fp8: bool = False
    """Whether this target supports native FP8."""

    supports_bf16: bool = True
    """Whether this target supports native BF16."""

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
    """CUDA warp size or SM thread group size."""

    sm_count: int = 0
    """Number of streaming multiprocessors."""

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
        backends = BACKEND_BY_TARGET.get(target_id, ["pytorch"])
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
            supports_fp8=profile_data["supports_fp8"],
            supports_bf16=profile_data["supports_bf16"],
            supports_flash_attention=profile_data.get("supports_flash_attention", False),
            recommended_flash_attention=profile_data.get("recommended_flash_attention", "flash_attention_3"),
            recommended_backend=backends[0],
            backend_candidates=list(backends),
            max_shm_size_bytes=profile_data.get("max_shm_size_bytes", 0),
            warp_size=profile_data.get("warp_size", 32),
            sm_count=profile_data.get("sm_count", 0),
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
        # CPU fallback
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
            recommended_backend="pytorch",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "memory_gb": self.memory_gb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "tensor_core_flops": self.tensor_core_flops,
            "supports_fp8": self.supports_fp8,
            "supports_bf16": self.supports_bf16,
            "supports_flash_attention": self.supports_flash_attention,
            "recommended_flash_attention": self.recommended_flash_attention,
            "recommended_backend": self.recommended_backend,
            "max_shm_size_bytes": self.max_shm_size_bytes,
            "warp_size": self.warp_size,
            "sm_count": self.sm_count,
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
            supports_fp8=data.get("supports_fp8", False),
            supports_bf16=data.get("supports_bf16", True),
            supports_flash_attention=data.get("supports_flash_attention", False),
            recommended_flash_attention=data.get("recommended_flash_attention", "flash_attention_3"),
            recommended_backend=data.get("recommended_backend"),
            max_shm_size_bytes=data.get("max_shm_size_bytes", 0),
            warp_size=data.get("warp_size", 32),
            sm_count=data.get("sm_count", 0),
        )

    def __repr__(self) -> str:
        return f"HardwareProfile({self.target_id}: {self.name}, {self.memory_gb}GB)"


_TARGET_PROFILES: dict[str, dict[str, Any]] = {
    "cuda_sm70": {
        "name": "NVIDIA V100 (Volta)",
        "compute_capability": "7.0",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 900.0,
        "tensor_core_flops": 125.0,
        "supports_fp8": False,
        "supports_bf16": False,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 114688,
        "warp_size": 32,
        "sm_count": 80,
    },
    "cuda_sm80": {
        "name": "NVIDIA A100 (Ampere)",
        "compute_capability": "8.0",
        "memory_gb": 80.0,
        "memory_bandwidth_gb_s": 2039.0,
        "tensor_core_flops": 312.0,
        "supports_fp8": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 164864,
        "warp_size": 32,
        "sm_count": 108,
    },
    "cuda_sm89": {
        "name": "NVIDIA RTX 4090 (Ada)",
        "compute_capability": "8.9",
        "memory_gb": 24.0,
        "memory_bandwidth_gb_s": 1008.0,
        "tensor_core_flops": 660.0,
        "supports_fp8": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 131072,
        "warp_size": 32,
        "sm_count": 128,
    },
    "cuda_sm90": {
        "name": "NVIDIA H100 (Hopper)",
        "compute_capability": "9.0",
        "memory_gb": 80.0,
        "memory_bandwidth_gb_s": 3350.0,
        "tensor_core_flops": 989.0,
        "supports_fp8": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 228352,
        "warp_size": 32,
        "sm_count": 132,
    },
    "cuda_sm100": {
        "name": "NVIDIA B200 (Blackwell)",
        "compute_capability": "10.0",
        "memory_gb": 192.0,
        "memory_bandwidth_gb_s": 8000.0,
        "tensor_core_flops": 2250.0,
        "supports_fp8": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 262144,
        "warp_size": 32,
        "sm_count": 160,
    },
    "metal_m1": {
        "name": "Apple M1/M2",
        "compute_capability": "m1",
        "memory_gb": 32.0,
        "memory_bandwidth_gb_s": 200.0,
        "tensor_core_flops": 10.0,
        "supports_fp8": False,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "memory_efficient_attention",
        "max_shm_size_bytes": 32768,
        "warp_size": 32,
        "sm_count": 16,
    },
    "metal_m3": {
        "name": "Apple M3/M4/M5",
        "compute_capability": "m3",
        "memory_gb": 64.0,
        "memory_bandwidth_gb_s": 400.0,
        "tensor_core_flops": 18.0,
        "supports_fp8": True,
        "supports_bf16": True,
        "supports_flash_attention": False,
        "recommended_flash_attention": "metal_4_neural_engine",
        "max_shm_size_bytes": 32768,
        "warp_size": 32,
        "sm_count": 20,
    },
    "rocm_rdna3": {
        "name": "AMD RX 7000 Series",
        "compute_capability": "rdna3",
        "memory_gb": 24.0,
        "memory_bandwidth_gb_s": 960.0,
        "tensor_core_flops": 122.0,
        "supports_fp8": False,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_2",
        "max_shm_size_bytes": 65536,
        "warp_size": 64,
        "sm_count": 60,
    },
    "rocm_cdna3": {
        "name": "AMD MI300X",
        "compute_capability": "cdna3",
        "memory_gb": 192.0,
        "memory_bandwidth_gb_s": 5300.0,
        "tensor_core_flops": 1300.0,
        "supports_fp8": True,
        "supports_bf16": True,
        "supports_flash_attention": True,
        "recommended_flash_attention": "flash_attention_3",
        "max_shm_size_bytes": 131072,
        "warp_size": 64,
        "sm_count": 304,
    },
}
