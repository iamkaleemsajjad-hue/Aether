"""Failed quality gates cannot be bypassed or hidden by device selection."""

import json

import pytest
from test_runtime_artifact_selection import _package, _runtime

from aether.backends.artifact_validation import (
    ArtifactEvaluationError,
    validate_aeg_evaluation,
)


def _failed_package(tmp_path):
    package = _package(tmp_path, ["cpu_avx512"], portable=False)
    package.metadata["evaluation_status"] = "rejected"
    package.metadata["eval_report"] = {
        "gate": {"passed": False, "failing_benchmarks": ["measured_quality"]}
    }
    package.save()
    return package


@pytest.mark.parametrize("host", ["cpu_avx2", "cuda_sm89", "rocm_cdna3", "metal_m3", "directml"])
def test_failed_gate_precedes_target_selection_without_initializing_any_backend(tmp_path, host):
    """A permanent quality rejection stays the same across incompatible hosts."""
    package = _failed_package(tmp_path)
    runtime, backend = _runtime(tmp_path, host)
    runtime._resolve_aeg_path = lambda _: str(package.root)
    with pytest.raises(
        ArtifactEvaluationError, match="persisted evaluation gate: measured_quality"
    ):
        runtime._load_model("model")
    backend.load_model.assert_not_called()
    runtime.backend_registry.get_available_backends.assert_not_called()


def test_failed_report_cannot_be_edited_to_pass_without_updating_its_declared_hash(tmp_path):
    """Admission verifies the policy payload before treating it as a passing gate."""
    package = _failed_package(tmp_path)
    report = package.root / "observability" / "eval_report.json"
    report.write_text(json.dumps({"gate": {"passed": True}}), encoding="utf-8")
    with pytest.raises(ArtifactEvaluationError, match="checksum"):
        validate_aeg_evaluation(package)


def test_declared_gate_cannot_disappear_to_become_uncertified(tmp_path):
    """Missing recorded policy fails closed; only diagnostics can waive absent evidence."""
    package = _failed_package(tmp_path)
    (package.root / "observability" / "eval_report.json").unlink()
    with pytest.raises(ArtifactEvaluationError, match="declared evaluation report is missing"):
        validate_aeg_evaluation(package)
    untouched = _package(tmp_path / "uncertified", ["cpu_avx2"])
    validate_aeg_evaluation(untouched, allow_uncertified=True)


def test_quality_report_boolean_string_cannot_be_treated_as_success(tmp_path):
    """Malformed quality evidence is rejected independently of backend readiness."""
    package = _failed_package(tmp_path)
    package.metadata["eval_report"]["gate"]["passed"] = "false"
    package.save()
    with pytest.raises(ArtifactEvaluationError, match="must be boolean"):
        validate_aeg_evaluation(package)


def test_benchmark_report_replacement_cannot_erase_its_recorded_rejection(tmp_path, monkeypatch):
    """Hash authentication and policy parsing must operate on identical bytes."""
    from aether.core import hash_utils

    package = _package(tmp_path, ["cpu_avx2"])
    package.metadata["eval_report"] = {"gate": {"passed": False}}
    package.save()
    report = package.root / "observability" / "eval_report.json"
    original_hash = hash_utils.compute_file_hash

    def replace_after_hash(path):
        digest = original_hash(path)
        if str(path) == str(report):
            report.write_text('{"gate":{"passed":true}}', encoding="utf-8")
        return digest

    monkeypatch.setattr(hash_utils, "compute_file_hash", replace_after_hash)
    with pytest.raises(ArtifactEvaluationError):
        validate_aeg_evaluation(package, allow_uncertified=True)


def test_benchmark_report_read_is_bounded_before_parsing(tmp_path):
    """A declared optional report cannot force an unbounded deployment read."""
    from aether.core.quality_gate import MAX_REPORT_BYTES

    package = _package(tmp_path, ["cpu_avx2"])
    report = package.root / "observability" / "eval_report.json"
    report.parent.mkdir(exist_ok=True)
    report.write_bytes(b" " * MAX_REPORT_BYTES + b'{"gate":{"passed":true}}')
    package.save()
    with pytest.raises(ArtifactEvaluationError, match="too large"):
        validate_aeg_evaluation(package, allow_uncertified=True)
