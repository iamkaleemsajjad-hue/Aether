"""Tests for the KernelFalcon autonomous kernel generation subsystem.

Covers the CPU-testable requirements from the v5 PRD: candidate generation,
safety screening, correctness validation against a trusted reference,
benchmarking, selection, and caching. GPU execution is explicitly
NOT_TESTABLE_ON_CURRENT_MACHINE and the tests assert KernelFalcon reports
that honestly rather than fabricating results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether.kernels.kernel_falcon import (
    KernelCandidate,
    KernelFalcon,
    KernelFalconResult,
    _generate_candidates,
)


@pytest.fixture()
def falcon(tmp_path: Path) -> KernelFalcon:
    return KernelFalcon(cache_dir=tmp_path / "falcon_cache")


class TestCandidateGeneration:
    def test_rmsnorm_candidates_generated(self) -> None:
        candidates = _generate_candidates("rmsnorm")
        c_names = [c.name for c in candidates if c.language == "c"]
        assert any("unroll" in n for n in c_names)
        assert any(c.language == "numpy" for c in candidates)

    def test_unknown_op_rejected(self) -> None:
        with pytest.raises(ValueError, match="no generator"):
            _generate_candidates("warp_drive")

    def test_candidates_have_source_hashes(self) -> None:
        for candidate in _generate_candidates("silu"):
            assert len(candidate.source_hash) == 16


class TestSafetyScreening:
    def test_unsafe_source_rejected_before_compilation(self, falcon: KernelFalcon) -> None:
        evil = KernelCandidate(
            name="evil",
            language="c",
            source="#include <stdio.h>\nvoid evil(void) { system(\"rm -rf /\"); }",
            symbol="evil",
            signature="elementwise3",
        )
        reason = falcon._screen_safety(evil)
        assert reason is not None
        assert "unsafe construct" in reason

    def test_inline_asm_rejected(self, falcon: KernelFalcon) -> None:
        evil = KernelCandidate(
            name="asm", language="c",
            source="void f(void) { __asm__(\"int3\"); }",
            symbol="f", signature="elementwise3",
        )
        assert falcon._screen_safety(evil) is not None

    def test_clean_source_passes(self, falcon: KernelFalcon) -> None:
        clean = KernelCandidate(
            name="ok", language="c",
            source="#include <math.h>\nvoid ok(float* x) { x[0] = 1.0f; }",
            symbol="ok", signature="elementwise3",
        )
        assert falcon._screen_safety(clean) is None


class TestValidationAndSelection:
    def test_rmsnorm_selects_a_validated_kernel(self, falcon: KernelFalcon) -> None:
        result = falcon.optimize("rmsnorm")
        assert isinstance(result, KernelFalconResult)
        assert result.selected is not None
        assert result.selected in result.validated
        # The selected kernel must have a real, measured benchmark.
        assert result.selected in result.benchmark_ms
        assert result.benchmark_ms[result.selected] >= 0.0

    def test_numpy_only_ops_validate(self, falcon: KernelFalcon) -> None:
        for op in ("sgemm", "softmax"):
            result = falcon.optimize(op)
            assert result.selected is not None
            assert result.selected in result.validated

    def test_selection_is_reused_from_cache(self, falcon: KernelFalcon) -> None:
        first = falcon.optimize("silu")
        assert first.cached is False
        second = falcon.optimize("silu")
        assert second.cached is True
        assert second.selected == first.selected

    def test_unknown_reference_op_rejects_all(self, falcon: KernelFalcon) -> None:
        result = falcon.optimize("nonexistent_op_xyz")
        assert result.selected is None
        assert "no reference implementation" in result.rejected.get("nonexistent_op_xyz", "")


class TestHonestStatus:
    def test_status_does_not_claim_gpu(self, falcon: KernelFalcon) -> None:
        status = falcon.status()
        assert status["implementation"] == "IMPLEMENTED"
        assert status["gpu_execution"] == "NOT_TESTABLE_ON_CURRENT_MACHINE"
        assert "no CUDA" in status["gpu_execution_reason"]

    def test_benchmarks_are_measured_not_fabricated(self, falcon: KernelFalcon) -> None:
        result = falcon.optimize("rmsnorm")
        for name, ms in result.benchmark_ms.items():
            assert isinstance(ms, float)
            assert ms > 0.0, f"benchmark for {name} must be a real measurement"
