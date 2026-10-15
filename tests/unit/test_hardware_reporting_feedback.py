"""Hardware reports distinguish discovery, readiness, and actual execution."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from aether.backends.capabilities import HardwareCapabilities, ValidationResult
from aether.cli import cli
from aether.core.types import HardwareTarget
from aether.runtime.hardware import HardwareDetector


def _cpu(**kwargs):
    return HardwareCapabilities(
        vendor="CPU",
        device="Actual CPU",
        architecture="AMD64",
        target_id="cpu_avx2",
        available=True,
        implemented=True,
        hardware_present=True,
        runtime_available=True,
        **kwargs,
    )


def _device(target, vendor, device, **kwargs):
    return HardwareCapabilities(
        vendor=vendor,
        device=device,
        architecture=target,
        target_id=target,
        available=True,
        implemented=True,
        hardware_present=True,
        runtime_available=True,
        **kwargs,
    )


def _patch_detection(monkeypatch, caps):
    probe = Mock(return_value=caps)
    monkeypatch.setattr("aether.backends.hardware_detector.detect_all_capabilities", probe)
    return probe


def test_fingerprint_uses_actual_rtx4070_identity_and_one_probe(monkeypatch):
    gpu = _device(
        "cuda_sm89",
        "NVIDIA",
        "NVIDIA GeForce RTX 4070 Laptop GPU",
        memory_bytes=8 * 1024**3,
        driver_version="580.97",
        extra={"compute_capability": "8.9"},
    )
    probe = _patch_detection(monkeypatch, [_cpu(), gpu])
    monkeypatch.setattr(HardwareTarget, "auto", Mock(side_effect=AssertionError("duplicate probe")))
    detector = HardwareDetector()
    probe.assert_not_called()
    fingerprint = detector.detect()
    probe.assert_called_once()
    assert fingerprint.target_id == "cuda_sm89"
    assert fingerprint.target_name == gpu.device
    assert fingerprint.gpu_name == gpu.device
    assert fingerprint.gpu_memory_gb == 8.0
    assert fingerprint.compute_capability == "8.9"
    assert fingerprint.gpu_count == 1
    assert "RTX 4090" not in fingerprint.target_name


def test_fingerprint_counts_intel_gpu_and_excludes_npu_and_ane(monkeypatch):
    gpu = _device("openvino_gpu", "Intel", "Actual Intel GPU", memory_bytes=1024**3)
    npu = _device("openvino_npu", "Intel", "Actual Intel NPU")
    ane = _device("apple_ane", "Apple", "Apple Neural Engine")
    _patch_detection(monkeypatch, [_cpu(), npu, ane, gpu])
    fingerprint = HardwareDetector().detect()
    assert fingerprint.target_id == "openvino_gpu"
    assert fingerprint.gpu_name == "Actual Intel GPU"
    assert fingerprint.gpu_count == 1
    assert fingerprint.attributes["available_accelerator_count"] == 1
    assert "apple_ane" not in fingerprint.attributes["available_targets"]
    assert "openvino_npu" in fingerprint.attributes["available_targets"]


def test_fingerprint_unknown_memory_is_not_invented(monkeypatch):
    gpu = _device("cuda_sm89", "NVIDIA", "Actual GPU", driver_version="unknown (ctypes)")
    _patch_detection(monkeypatch, [_cpu(), gpu])
    monkeypatch.setattr("aether.backends.capabilities._host_memory_bytes", lambda: 0)
    fingerprint = HardwareDetector().detect()
    assert fingerprint.total_ram_gb is None
    assert fingerprint.gpu_memory_gb is None
    assert fingerprint.driver_version is None
    assert fingerprint.to_dict()["gpu_memory_gb"] is None
    assert fingerprint.attributes["ram_memory_known"] is False


def test_fingerprint_reports_unimplemented_gpu_without_selecting_it(monkeypatch):
    gpu = HardwareCapabilities(
        vendor="Intel",
        device="Intel GPU with vendor runtime only",
        architecture="xe",
        target_id="openvino_gpu",
        available=True,
        implemented=False,
        hardware_present=True,
        runtime_available=True,
    )
    _patch_detection(monkeypatch, [_cpu(), gpu])
    fingerprint = HardwareDetector().detect()
    assert fingerprint.target_id == "cpu_avx2"
    assert fingerprint.gpu_count == 1
    assert fingerprint.gpu_name == gpu.device
    assert "openvino_gpu" in fingerprint.attributes["available_targets"]


def test_fingerprint_preserves_unknown_gpu_evidence_without_inventing_target(monkeypatch):
    gpu = _device("rocm_unknown", "AMD", "Unknown actual AMD GPU")
    _patch_detection(monkeypatch, [_cpu(), gpu])
    fingerprint = HardwareDetector().detect()
    assert fingerprint.target_id == "cpu_avx2"
    assert fingerprint.gpu_count == 1
    assert "rocm_unknown" not in fingerprint.attributes["available_targets"]
    assert fingerprint.capabilities[1]["target_id"] == "rocm_unknown"


@pytest.mark.parametrize("argv", [["hardware"], ["hardware", "--json"], ["hw"]])
def test_bare_hardware_reports_actual_fingerprint(monkeypatch, argv):
    _patch_detection(monkeypatch, [_cpu(memory_bytes=4 * 1024**3)])
    result = CliRunner().invoke(cli, argv, terminal_width=20)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["target_name"] == "Actual CPU"
    assert data["total_ram_gb"] == 4.0


@pytest.mark.parametrize(
    "argv",
    [
        ["hardware", "detect", "--json"],
        ["hardware", "capabilities", "--json"],
        ["backend", "list", "--json"],
    ],
)
def test_hardware_json_is_not_wrapped_at_terminal_width(monkeypatch, argv):
    cap = _cpu(extra={"evidence": "long runtime dependency evidence " * 50})
    _patch_detection(monkeypatch, [cap])
    result = CliRunner().invoke(cli, argv, terminal_width=20)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    targets = data["targets"] if isinstance(data, dict) else data
    assert targets[0]["extra"]["evidence"] == cap.extra["evidence"]


def test_detection_saves_only_explicit_path_and_keeps_json_clean(monkeypatch, tmp_path):
    _patch_detection(monkeypatch, [_cpu()])
    destination = tmp_path / "laptop-hardware.json"
    # Click 8.1 merges stderr into stdout by default; 8.2+ preserves separate
    # streams and no longer accepts the mix_stderr constructor argument.
    try:
        runner = CliRunner(mix_stderr=False)
    except TypeError:
        runner = CliRunner()
    result = runner.invoke(cli, ["hardware", "detect", "--json", "--save", str(destination)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads(destination.read_text(encoding="utf-8"))
    assert "Saved to" in result.stderr
    missing_path = CliRunner().invoke(cli, ["hardware", "detect", "--save"])
    assert missing_path.exit_code == 2


def test_capabilities_show_unknown_measurements_and_separate_statuses(monkeypatch):
    _patch_detection(
        monkeypatch, [_cpu(extra={"precision_execution": {"bf16": "fp32 conversion"}})]
    )
    result = CliRunner().invoke(cli, ["hardware", "capabilities", "cpu_avx2"], terminal_width=160)
    assert result.exit_code == 0, result.output
    assert "unknown / not measured" in result.output
    assert "backend_available" in result.output
    assert "NOT CHECKED" in result.output
    assert "NOT TESTED" in result.output
    assert "NOT VALIDATED" in result.output
    assert "native_bf16" in result.output
    assert "not confirmed" in result.output
    assert "fp32 conversion" in result.output


@pytest.mark.parametrize(
    ("scope", "label", "other_label", "semantics"),
    [
        ("runtime_formats", "runtime_format", "native", "queried vendor runtime formats"),
        ("native_arithmetic", "native", "runtime_format", "confirmed device arithmetic"),
    ],
)
def test_capability_precision_labels_preserve_their_evidence_scope(
    monkeypatch, scope, label, other_label, semantics,
):
    evidence = {"precision_scope": scope, "precision_evidence": "queried format evidence"}
    cap = _device("openvino_gpu", "Intel", "Queried device", supports_fp16=True, extra=evidence)
    _patch_detection(monkeypatch, [cap])
    runner = CliRunner()
    text = runner.invoke(cli, ["hardware", "capabilities", cap.target_id], terminal_width=220)
    assert text.exit_code == 0, text.output
    assert f"{label}_fp16" in text.output
    assert f"{label}_bf16" in text.output
    assert f"{other_label}_fp16" not in text.output
    assert semantics in text.output
    structured = runner.invoke(cli, ["hardware", "capabilities", cap.target_id, "--json"])
    assert structured.exit_code == 0, structured.output
    data, = json.loads(structured.stdout)
    assert data["supports_fp16"] is True and data["supports_bf16"] is False
    assert data["extra"] == evidence
    assert data["precision_semantics"].startswith(semantics)


def test_validate_forwards_real_execution_constraints_and_outputs_raw_json(monkeypatch):
    validation = ValidationResult(
        backend_name="pytorch",
        available=True,
        backend_available=True,
        execution_tested=True,
        execution_scope="backend_workload",
        checks_passed=["real workload evidence " * 50],
    )
    validator = Mock(return_value=validation)
    monkeypatch.setattr("aether.backends.hardware_detector.validate_backend_environment", validator)
    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "validate",
            "cuda_sm89",
            "--execute",
            "--backend",
            "pytorch",
            "--device-index",
            "1",
            "--json",
        ],
        terminal_width=20,
    )
    assert result.exit_code == 0, result.output
    validator.assert_called_once_with(
        "cuda_sm89", execute=True, backend_name="pytorch", device_index=1
    )
    data = json.loads(result.stdout)
    assert data["execution_tested"] is True
    assert data["production_validated"] is False


def test_failed_json_validation_has_nonzero_exit_without_corrupting_json(monkeypatch):
    validation = ValidationResult(
        backend_name="cuda_sm89",
        available=False,
        checks_failed=["dependency missing " * 50],
    )
    monkeypatch.setattr(
        "aether.backends.hardware_detector.validate_backend_environment", lambda *a, **k: validation
    )
    result = CliRunner().invoke(cli, ["hardware", "validate", "cuda_sm89", "--execute", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert data["all_passed"] is False
    assert data["execution_tested"] is False


def test_environment_only_validation_is_not_labeled_executed(monkeypatch):
    validation = ValidationResult(backend_name="cpu", available=True)
    validator = Mock(return_value=validation)
    monkeypatch.setattr("aether.backends.hardware_detector.validate_backend_environment", validator)
    result = CliRunner().invoke(cli, ["hardware", "validate"])
    assert result.exit_code == 0, result.output
    validator.assert_called_once_with("cpu", execute=False, backend_name=None, device_index=0)
    assert "Execution: NOT TESTED" in result.output
    assert "Production validation: NOT VALIDATED" in result.output


def test_validate_rejects_negative_device_index_before_probing(monkeypatch):
    validator = Mock()
    monkeypatch.setattr("aether.backends.hardware_detector.validate_backend_environment", validator)
    result = CliRunner().invoke(cli, ["hardware", "validate", "--device-index", "-1"])
    assert result.exit_code == 2
    validator.assert_not_called()


def test_doctor_reports_native_compiler_failure_as_warning(monkeypatch):
    _patch_detection(monkeypatch, [_cpu()])
    validation = ValidationResult(
        backend_name="cpu",
        available=True,
        warnings=["Native compilation failed: no host C++ compiler found; using NumPy reference"],
    )
    monkeypatch.setattr(
        "aether.backends.hardware_detector.validate_backend_environment", lambda *a, **k: validation
    )
    monkeypatch.setattr("aether.Runtime", lambda *a, **k: SimpleNamespace())
    kernels = SimpleNamespace(
        ensure_compiled=lambda: False, build_error="no host C++ compiler found"
    )
    monkeypatch.setattr("aether.kernels.native_cpu.get_native_kernels", lambda: kernels)
    result = CliRunner().invoke(cli, ["doctor", "--json"], terminal_width=20)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    checks = {check["name"]: check for check in data["checks"]}
    assert checks["kernels:native_cpu"]["status"] == "warn"
    assert checks["hardware:cpu"]["status"] == "warn"
    assert checks["hardware:cpu"]["detail"]["execution_tested"] is False
    assert "optional" in checks["kernels:native_cpu"]["detail"]["fallback"]
    assert data["summary"]["warnings"] == 2
