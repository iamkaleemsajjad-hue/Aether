"""
Tests for hardware detection and profiles.
"""

from __future__ import annotations

import pytest

from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.constants import SUPPORTED_TARGET_IDS
from aether.core.types import HardwareTarget


class TestHardwareProfile:
    """Tests for hardware profile creation."""

    def test_from_target_id_cuda_sm90(self) -> None:
        profile = HardwareProfile.from_target_id("cuda_sm90")
        assert profile is not None
        assert profile.target_id == "cuda_sm90"
        assert "H100" in profile.name

    def test_from_target_id_unknown(self) -> None:
        profile = HardwareProfile.from_target_id("invalid")
        assert profile is None

    def test_auto(self) -> None:
        profile = HardwareProfile.auto()
        assert profile is not None
        assert profile.target_id in SUPPORTED_TARGET_IDS

    def test_to_dict(self) -> None:
        profile = HardwareProfile.from_target_id("cuda_sm90")
        data = profile.to_dict()
        assert data["target_id"] == "cuda_sm90"
        assert data["name"] == profile.name

    def test_from_dict(self) -> None:
        data = {
            "target_id": "cpu_avx512",
            "name": "x86_64 (AVX-512)",
            "compute_capability": "cpu",
            "memory_gb": 16.0,
            "recommended_backend": "pytorch",
        }
        profile = HardwareProfile.from_dict(data)
        assert profile.target_id == "cpu_avx512"


class TestHardwareTarget:
    """Tests for hardware target enum."""

    def test_from_string(self) -> None:
        target = HardwareTarget.from_string("cuda_sm90")
        assert target == HardwareTarget.CUDA_SM90

    def test_from_string_case_insensitive(self) -> None:
        target = HardwareTarget.from_string("CUDA_SM90")
        assert target == HardwareTarget.CUDA_SM90

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            HardwareTarget.from_string("invalid_target")

    def test_auto(self) -> None:
        target = HardwareTarget.auto()
        assert target in HardwareTarget

    def test_vendor(self) -> None:
        assert HardwareTarget.CUDA_SM90.vendor == "NVIDIA"
        assert HardwareTarget.METAL_M3.vendor == "Apple"
        assert HardwareTarget.ROCM_RDNA3.vendor == "AMD"

    def test_backend_candidates(self) -> None:
        assert HardwareTarget.CUDA_SM90.backend_candidates == ["vllm", "pytorch", "tensorrt-llm"]
        assert HardwareTarget.METAL_M3.backend_candidates == ["mlx", "llama.cpp", "pytorch"]


class TestTargetRegistry:
    """Tests for the target registry."""

    def test_supported_targets(self) -> None:
        registry = TargetRegistry()
        assert "cuda_sm90" in registry.supported_targets
        assert "cpu_avx512" in registry.supported_targets

    def test_is_supported(self) -> None:
        registry = TargetRegistry()
        assert registry.is_supported("cuda_sm90")
        assert not registry.is_supported("invalid")

    def test_get_profile(self) -> None:
        registry = TargetRegistry()
        profile = registry.get_profile("cuda_sm90")
        assert profile.target_id == "cuda_sm90"

    def test_get_profile_unknown(self) -> None:
        registry = TargetRegistry()
        with pytest.raises(Exception):
            registry.get_profile("invalid")

    def test_recommend_targets_small(self) -> None:
        from aether.core.types import ModelArchitecture
        arch = ModelArchitecture(
            family="llama_family",
            params_billion=1.0,
            layers=16,
            hidden_size=2048,
            num_attention_heads=32,
        )
        registry = TargetRegistry()
        targets = registry.recommend_targets(arch)
        assert "cuda_sm89" in targets or "metal_m3" in targets

    def test_recommend_targets_moe(self) -> None:
        from aether.core.types import ModelArchitecture
        arch = ModelArchitecture(
            family="deepseek_family",
            params_billion=671.0,
            layers=80,
            hidden_size=7168,
            num_attention_heads=64,
            is_moe=True,
            num_experts=256,
        )
        registry = TargetRegistry()
        targets = registry.recommend_targets(arch)
        assert "cuda_sm90" in targets
