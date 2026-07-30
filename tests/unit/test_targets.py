"""Tests for the targets package."""

from __future__ import annotations

from aether.targets import (
    CUDATargetKernels,
    MetalTargetKernels,
    ROCmTargetKernels,
    TargetRegistry,
    TemplateLibrary,
)


class TestTargetRegistry:
    def test_builtin_targets(self) -> None:
        registry = TargetRegistry()
        targets = registry.list_targets()
        assert "cuda_sm90" in targets
        assert "metal_m3" in targets
        assert "cpu_avx512" in targets

    def test_get_target(self) -> None:
        registry = TargetRegistry()
        info = registry.get("cuda_sm90")
        assert info is not None
        assert info.vendor == "NVIDIA"
        assert info.display_name != ""

    def test_get_unknown(self) -> None:
        registry = TargetRegistry()
        assert registry.get("nonexistent") is None


class TestCUDATargetKernels:
    def test_sm80_preferences(self) -> None:
        kernels = CUDATargetKernels("cuda_sm80")
        assert kernels.preferred_attention == "flash_attention_2"
        assert not kernels.supports_fp8
        assert kernels.supports_int4_gemm

    def test_sm70_preferences(self) -> None:
        kernels = CUDATargetKernels("cuda_sm70")
        assert kernels.preferred_attention == "vanilla"

    def test_sm90_flags(self) -> None:
        kernels = CUDATargetKernels("cuda_sm90")
        flags = kernels.recommended_flags()
        assert flags["use_fp8"] is True
        assert flags["target_sm"] == 90


class TestMetalTargetKernels:
    def test_m1(self) -> None:
        kernels = MetalTargetKernels("metal_m1")
        assert kernels.generation == 1

    def test_recommended_flags(self) -> None:
        kernels = MetalTargetKernels("metal_m3")
        flags = kernels.recommended_flags()
        assert "use_metal_performance_shaders" in flags
        assert flags["use_mlx"] is True

    def test_supports_int8(self) -> None:
        kernels = MetalTargetKernels("metal_m3")
        assert kernels.supports_int8


class TestROCMTargetKernels:
    def test_rdna3(self) -> None:
        kernels = ROCmTargetKernels("rocm_rdna3")
        assert kernels.arch == "rdna3"

    def test_cdna3_preferred_attention(self) -> None:
        kernels = ROCmTargetKernels("rocm_cdna3")
        assert kernels.preferred_attention == "flash_attention_2"


class TestTemplateLibrary:
    def test_list_templates(self) -> None:
        templates = TemplateLibrary.list_templates()
        assert "fused_attention" in templates

    def test_render(self) -> None:
        rendered = TemplateLibrary.render("fused_attention", target_id="cuda_sm90", inputs="q,k,v", output="out")
        assert "cuda_sm90" in rendered
