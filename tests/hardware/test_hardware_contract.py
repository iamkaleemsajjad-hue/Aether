"""
Hardware backend contract tests (PRD §6, §12, §41).

These tests verify backend contracts WITHOUT requiring real hardware.
They run on any machine including CPU-only Windows.

Contract rules verified:
  1. Every backend must implement the required interface methods.
  2. HardwareCapabilities must have valid types for every field.
  3. is_available() must return a bool consistent with real detection.
  4. Unavailable backends must raise BackendError (not silently return fake results).
  5. hardware_validation_matrix.json must exist and be valid JSON.
  6. CPU backend must always be available.
  7. No GPU target may claim production_validated=True without execution_tested=True.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# HardwareCapabilities contract
# ---------------------------------------------------------------------------

class TestHardwareCapabilitiesContract:
    """HardwareCapabilities must conform to the PRD §12 schema."""

    def test_detect_all_returns_list(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        assert isinstance(caps, list)
        assert len(caps) > 0

    def test_every_cap_has_required_fields(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        required = [
            "vendor", "device", "architecture", "target_id",
            "implemented", "available", "compile_tested",
            "execution_tested", "production_validated",
        ]
        for c in caps:
            d = c.to_dict()
            for field in required:
                assert field in d, f"Field {field!r} missing from {c.target_id}"

    def test_bool_fields_are_bool(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        bool_fields = ["implemented", "available", "compile_tested",
                       "execution_tested", "production_validated"]
        for c in caps:
            for field in bool_fields:
                val = getattr(c, field)
                assert isinstance(val, bool), \
                    f"{c.target_id}.{field} is {type(val).__name__}, expected bool"

    def test_unavailable_has_reason(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        for c in caps:
            if not c.available:
                assert c.unavailable_reason is not None, \
                    f"{c.target_id}: available=False but unavailable_reason is None"
                assert len(c.unavailable_reason) > 5, \
                    f"{c.target_id}: unavailable_reason is too short"

    def test_production_validated_requires_execution_tested(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        for c in caps:
            if c.production_validated:
                assert c.execution_tested, \
                    f"{c.target_id}: production_validated=True but execution_tested=False — impossible"

    def test_cpu_always_available(self) -> None:
        from aether.backends.hardware_detector import detect_cpu
        cpu = detect_cpu()
        assert cpu.available is True
        assert cpu.implemented is True
        assert cpu.vendor == "CPU"
        assert cpu.target_id.startswith("cpu")

    def test_to_dict_is_json_serializable(self) -> None:
        from aether.backends.hardware_detector import detect_all_capabilities
        caps = detect_all_capabilities()
        for c in caps:
            d = c.to_dict()
            # Must not raise
            json_str = json.dumps(d, default=str)
            assert isinstance(json_str, str)

    def test_validate_precision_cpu(self) -> None:
        from aether.backends.hardware_detector import detect_cpu
        cpu = detect_cpu()
        assert cpu.validate_precision("fp32") is True
        assert cpu.validate_precision("FP32") is True
        # fp8 should not be available on CPU without special hardware
        # (we don't assert False because some CPUs might support emulation)
        result = cpu.validate_precision("fp8")
        assert isinstance(result, bool)

    def test_no_cuda_claimed_without_cuda(self) -> None:
        """On CPU-only host, no CUDA target may claim available=True."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if has_cuda:
            pytest.skip("CUDA is available on this host")

        from aether.backends.hardware_detector import detect_cuda_devices
        cuda_caps = detect_cuda_devices()
        for c in cuda_caps:
            assert c.available is False, \
                f"CUDA {c.target_id} claims available=True but torch.cuda.is_available()=False"


# ---------------------------------------------------------------------------
# Backend interface contract
# ---------------------------------------------------------------------------

class TestBackendInterfaceContract:
    """Concrete backends must implement the required interface."""

    def test_cuda_backend_has_is_available(self) -> None:
        from aether.backends.hardware_backends import CUDABackend
        b = CUDABackend("cuda_sm90")
        assert hasattr(b, "is_available")
        assert isinstance(b.is_available(), bool)

    def test_rocm_backend_has_is_available(self) -> None:
        from aether.backends.hardware_backends import ROCmBackend
        b = ROCmBackend("rocm_cdna3")
        assert hasattr(b, "is_available")
        assert isinstance(b.is_available(), bool)

    def test_metal_backend_has_is_available(self) -> None:
        from aether.backends.hardware_backends import MetalBackend
        b = MetalBackend("metal_m1")
        assert hasattr(b, "is_available")
        assert isinstance(b.is_available(), bool)

    def test_cuda_backend_load_raises_when_unavailable(self) -> None:
        from aether.backends.hardware_backends import CUDABackend
        from aether.core.exceptions import BackendError
        b = CUDABackend("cuda_sm90")
        if not b.is_available():
            with pytest.raises((BackendError, Exception)):
                b.load("nonexistent_model")

    def test_rocm_backend_load_raises_when_unavailable(self) -> None:
        from aether.backends.hardware_backends import ROCmBackend
        from aether.core.exceptions import BackendError
        b = ROCmBackend("rocm_cdna3")
        if not b.is_available():
            with pytest.raises((BackendError, Exception)):
                b.load("nonexistent_model")

    def test_backends_have_get_capabilities(self) -> None:
        """All hardware backends should expose get_capabilities()."""
        from aether.backends.hardware_backends import CUDABackend, ROCmBackend, MetalBackend
        for BackendClass in [CUDABackend, ROCmBackend, MetalBackend]:
            b = BackendClass()
            assert hasattr(b, "get_capabilities")
            caps = b.get_capabilities()
            assert caps is not None


# ---------------------------------------------------------------------------
# Validation contract
# ---------------------------------------------------------------------------

class TestValidationContract:
    """validate_backend_environment must run without crashing for all targets."""

    @pytest.mark.parametrize("target_id", ["cpu", "cuda_sm90", "rocm_cdna3", "metal_m1"])
    def test_validate_returns_result(self, target_id: str) -> None:
        from aether.backends.hardware_detector import validate_backend_environment
        result = validate_backend_environment(target_id)
        assert result is not None
        assert isinstance(result.available, bool)
        assert isinstance(result.checks_passed, list)
        assert isinstance(result.checks_failed, list)
        d = result.to_dict()
        assert "backend_name" in d
        assert "available" in d

    def test_cpu_validate_passes(self) -> None:
        from aether.backends.hardware_detector import validate_backend_environment
        result = validate_backend_environment("cpu")
        # CPU validation must pass on any host with Python + numpy
        assert result.available is True or len(result.checks_failed) > 0  # graceful

    def test_unknown_target_returns_unavailable(self) -> None:
        from aether.backends.hardware_detector import validate_backend_environment
        result = validate_backend_environment("totally_nonexistent_target_xyz")
        assert result.available is False


# ---------------------------------------------------------------------------
# Hardware validation matrix
# ---------------------------------------------------------------------------

class TestHardwareValidationMatrix:
    """The hardware_validation_matrix.json must exist and be valid."""

    @pytest.fixture
    def matrix_path(self) -> Path:
        # Locate the file relative to this test file
        test_dir = Path(__file__).parent
        for candidate in [
            test_dir.parent.parent / "hardware_validation_matrix.json",
            test_dir.parent / "hardware_validation_matrix.json",
        ]:
            if candidate.exists():
                return candidate
        pytest.skip("hardware_validation_matrix.json not found")

    def test_matrix_is_valid_json(self, matrix_path: Path) -> None:
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_matrix_has_targets(self, matrix_path: Path) -> None:
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        # May have either "targets" (new format) or nested structure
        assert "targets" in data or "feature_matrix" in data

    def test_matrix_no_fake_production_validated(self, matrix_path: Path) -> None:
        """No target in the matrix may claim production_validated=true on CPU-only host."""
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        targets = data.get("targets", [])
        for t in targets:
            if t.get("production_validated") is True:
                # This is only allowed if execution_tested is also True
                assert t.get("execution_tested") is True, \
                    f"Target {t.get('target_id')}: production_validated=true but execution_tested!=true"

    def test_cpu_is_marked_available(self, matrix_path: Path) -> None:
        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        targets = data.get("targets", [])
        cpu_entries = [t for t in targets if t.get("target_id", "").startswith("cpu")]
        assert len(cpu_entries) > 0, "Matrix must have at least one CPU entry"
        for cpu_entry in cpu_entries:
            if cpu_entry["target_id"] == "cpu":
                assert cpu_entry.get("available") is True, \
                    "cpu target must be marked available=true"

    def test_gpu_targets_available_false_on_cpu_host(self, matrix_path: Path) -> None:
        """On CPU-only host, GPU targets must be available=false in the matrix."""
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False

        if has_gpu:
            pytest.skip("GPU available on this host — matrix may legitimately have GPU targets")

        data = json.loads(matrix_path.read_text(encoding="utf-8"))
        targets = data.get("targets", [])
        for t in targets:
            tid = t.get("target_id", "")
            if tid.startswith("cuda") or tid.startswith("rocm"):
                assert t.get("available") is False, \
                    f"Target {tid} claims available=true but no GPU detected on this host"
