"""Exercise gate production/validation with actual HF, CPU and Torch arithmetic.

These unit tests invoke worker bodies in-process to isolate metric/binding
contracts. Process ownership and resource admission have separate subprocess
tests; this fixture is not evidence of subprocess isolation or hardware speed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from aether import Compiler, CompilerConfig
from aether.core.quality_gate import GATE_PATH, NumericalQualityError, read_gate

if TYPE_CHECKING:
    from pathlib import Path

    from aether.core.aeg_format import AEGPackage


@pytest.fixture
def measured_package(
    tmp_path: Path, tiny_local_safetensors_model: Path, monkeypatch: pytest.MonkeyPatch
) -> AEGPackage:
    """Run real reference/candidate math without replacing it with passing DTOs."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from aether.compiler import numerical_admission
    from aether.evaluation.admission_worker import _candidate, _hf, write_json

    monkeypatch.setenv("AETHER_TORCH_DTYPE", "float32")
    calls = []

    def actual_worker(
        work: Path, name: str, job: dict[str, Any], config: CompilerConfig, reserve: int
    ) -> dict[str, Any]:
        calls.append(name)
        if name == "hf":
            result = _hf(job, work)
            write_json(work / "hf-result.json", result)
            return result
        return _candidate(job, work)

    monkeypatch.setattr(numerical_admission, "run_worker", actual_worker)
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        json.dumps({"id": "fixed", "prompt_ids": [1, 2, 3], "continuation_ids": [4, 5]}) + "\n",
        encoding="utf-8",
    )
    package = Compiler(
        CompilerConfig(
            targets=["cpu_generic"],
            precision="FP32",
            skip_download=True,
            cache_dir=str(tmp_path / "cache"),
            quality_dataset=str(probes),
            quality_greedy_tokens=4,
            quality_budget=0.001,
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "measured.aeg")
    assert calls == ["hf", "cpu", "torch_cpu"]
    return package


def test_actual_metrics_publish_and_survive_fresh_strict_load(measured_package: AEGPackage) -> None:
    """The saved bound report must authorize ordinary execution of the same bytes."""
    import numpy as np

    from aether.core.aeg_format import AEGPackage
    from aether.runtime.aeg_loader import load_engine_from_package

    reloaded = AEGPackage(measured_package.root).load()
    reloaded.verify_integrity()
    report = read_gate(reloaded)
    assert report is not None and report["status"] == "passed"
    for samples in report["measurements"].values():
        metrics = samples["fixed"]
        assert metrics["logits"]["max_kl_divergence"] <= 0.01
        assert metrics["logits"]["min_cosine_similarity"] >= 0.999
        assert metrics["tokens"]["agreement_rate"] >= 0.95
        assert metrics["perplexity"]["threshold_relative_delta"] == 0.001
    logits, _ = load_engine_from_package(reloaded).forward_step([1, 2, 3])
    assert logits.shape == (32,) and np.isfinite(logits).all()


def test_undeclared_execution_payload_cannot_inherit_measured_admission(
    measured_package: AEGPackage,
) -> None:
    """An added runtime feature file was not part of the measured candidate."""
    path = measured_package.root / "ttt" / "fast_weight_config.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text('{"enabled": true}', encoding="utf-8")
    # The existing format verifies declared hashes, so admission must also
    # require a closed set of measured payloads before runtime feature loading.
    measured_package.verify_integrity()
    with pytest.raises(NumericalQualityError, match="undeclared"):
        read_gate(measured_package)


def test_weight_transform_invalidates_measured_admission(measured_package: AEGPackage) -> None:
    """Changed weights require fresh evidence and remain explicitly diagnostic."""
    from aether.compiler.aeg_merging import invalidate_weight_dependent_evidence

    invalidate_weight_dependent_evidence(measured_package)
    measured_package.save()
    measured_package.verify_integrity()
    assert not (measured_package.root / GATE_PATH).exists()
    assert read_gate(measured_package, allow_uncertified=True) is None
    with pytest.raises(NumericalQualityError, match="no measured"):
        read_gate(measured_package)


def test_strict_runtime_cannot_inherit_base_admission_after_precision_changes(
    measured_package: AEGPackage, tmp_path: Path
) -> None:
    """Base evidence cannot authorize unmeasured automatic or explicit mutations."""
    import numpy as np

    from aether import Runtime, RuntimeConfig

    config = RuntimeConfig(
        backend_name="aether_cpu",
        execution_devices=["cpu"],
        hf_offline=True,
        model_cache_dir=str(tmp_path / "runtime"),
        dynamic_precision=True,
        precision_calibration_tokens=[[1, 2]],
        enable_semantic_cache=False,
    )
    model = str(measured_package.root)
    with Runtime(config) as runtime:
        runtime._load_model(model)
        handle = runtime._loaded_models[model]
        before = handle.engine.weights.layers[0].q_proj.copy()
        with pytest.raises(ValueError, match="diagnostic"):
            runtime.adjust_precision(model, [[1, 2]])
        runtime._execution_features.automatic_precision(model, handle)
        assert not runtime._execution_features._controllers
        np.testing.assert_array_equal(handle.engine.weights.layers[0].q_proj, before)
        assert handle.numerical_quality_admitted


def test_diagnostic_precision_change_clears_cached_base_admission(
    measured_package: AEGPackage, tmp_path: Path
) -> None:
    """Real changed resident tensors cannot retain their former admission flag."""
    from aether import Runtime, RuntimeConfig

    config = RuntimeConfig(
        allow_uncertified=True,
        backend_name="aether_cpu",
        execution_devices=["cpu"],
        hf_offline=True,
        model_cache_dir=str(tmp_path / "runtime"),
        dynamic_precision=True,
        precision_allow_lossy=True,
        precision_max_logit_error=0.1,
        precision_max_slowdown=1000,
        enable_semantic_cache=False,
    )
    model = str(measured_package.root)
    with Runtime(config) as runtime:
        runtime._load_model(model)
        handle = runtime._loaded_models[model]
        assert handle.numerical_quality_admitted
        report = runtime.adjust_precision(model, [[1, 2]])
        assert report["execution_changed"], report
        assert not handle.numerical_quality_admitted


def test_candidate_worker_rejects_another_candidate_binding_before_measurement(
    measured_package: AEGPackage,
    tmp_path: Path,
) -> None:
    """A worker must compare the candidate it loaded with its parent's identity."""
    from aether.evaluation.admission_worker import _candidate

    with pytest.raises(NumericalQualityError, match="candidate binding"):
        _candidate(
            {
                "artifact": str(measured_package.root),
                "executor": "cpu",
                "artifact_binding": {},
                "max_perplexity_delta": 0.001,
            },
            tmp_path,
        )


def test_report_replacement_between_hashing_and_parsing_cannot_erase_rejection(
    measured_package: AEGPackage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes parsed must be the same bounded bytes authenticated by the hash."""
    from aether.core import hash_utils

    path = measured_package.root / GATE_PATH
    passing_bytes = path.read_bytes()
    rejected = json.loads(passing_bytes)
    rejected["status"] = "rejected"
    path.write_text(json.dumps(rejected), encoding="utf-8")
    measured_package.save()
    original_hash = hash_utils.compute_file_hash

    def replace_after_hash(filename: str | Path) -> str:
        digest = original_hash(filename)
        if str(filename) == str(path):
            path.write_bytes(passing_bytes)
        return digest

    monkeypatch.setattr(hash_utils, "compute_file_hash", replace_after_hash)
    with pytest.raises(NumericalQualityError):
        read_gate(measured_package)


@pytest.mark.parametrize("change", ["executor", "threshold", "count", "greedy", "source", "header"])
def test_actual_report_cannot_survive_inconsistent_evidence_or_candidate_changes(
    measured_package: AEGPackage, change: str
) -> None:
    """Rehashing a changed report must not waive its mathematical/binding rules."""
    package = measured_package
    path = package.root / GATE_PATH
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report["measurements"]["cpu"]["fixed"]
    if change == "executor":
        del report["measurements"]["torch_cpu"]
    elif change == "threshold":
        metrics["logits"]["threshold_kl"] = 0.1
    elif change == "count":
        metrics["logits"]["num_positions"] += 1
    elif change == "greedy":
        metrics["tokens"]["agreement_rate"] = 0.5
    elif change == "source":
        report["source"]["digest"] = "sha256:" + "0" * 64
    else:
        assert package.manifest is not None
        package.manifest.model_id = "changed-header"
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    package.save()
    package.verify_integrity()
    for diagnostic in (False, True):
        with pytest.raises(NumericalQualityError):
            read_gate(package, allow_uncertified=diagnostic)
