"""Fail closed on unsupported Mamba-2 parameters and authenticate execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba2_normalization_fixtures import write_checkpoint

from aether import Compiler, CompilerConfig
from aether.core.aeg_format import load_aeg_package
from aether.core.exceptions import AEGIntegrityError, CompilationError
from aether.core.mamba2_parameters import Mamba2Parameters
from aether.core.types import ModelArchitecture
from aether.runtime.aeg_loader import load_engine_from_path
from aether.runtime.disaggregated_execution import clone_cpu_engine_for_worker

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("source_parameters", [
    {"hidden_act": "gelu"},
    {"hidden_act": None},
    {"time_step_limit": None},
    {"time_step_limit": [0]},
    {"time_step_limit": [0, 1, 2]},
    {"time_step_limit": [False, 1]},
    {"time_step_limit": ["0", 1]},
    {"time_step_limit": [None, None]},
    {"time_step_limit": [0, float("nan")]},
    {"time_step_limit": [float("inf"), 1]},
    {"time_step_limit": [0, 1e100]},
])
@pytest.mark.parametrize("existing", [False, True])
def test_source_parameters_refuse_publication_and_preserve_prior_destination(
    tmp_path: Path, source_parameters: dict[str, Any], existing: bool,
) -> None:
    """Even diagnostic compilation must not silently change the source function."""
    source = tmp_path / "source"
    config = write_checkpoint(source)
    config.update(source_parameters)
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    artifact = tmp_path / "model.aeg"
    if existing:
        artifact.mkdir()
        (artifact / "prior").write_bytes(b"prior published artifact")
    compiler = Compiler(CompilerConfig(
        quality_mode="diagnostic", targets=["cpu_generic"], skip_download=True,
        precision_assignment_mode="uniform", precision="FP32", calibration_tokens=4,
        overwrite=existing, cache_dir=str(tmp_path / "cache"),
    ))
    with pytest.raises(CompilationError, match=r"Mamba-2.*(hidden_act|time_step_limit)"):
        compiler.compile(str(source), output_path=artifact)
    if existing:
        assert sorted(path.name for path in artifact.iterdir()) == ["prior"]
        assert (artifact / "prior").read_bytes() == b"prior published artifact"
    else:
        assert not artifact.exists()


@pytest.mark.parametrize("envelope", [
    None, [], "silu", {"unexpected": True}, {"conv_activation": "gelu"},
    {"time_step_limit": [0]}, {"time_step_limit": [False, 1]},
    {"time_step_limit": [0, float("inf")]},
    {"time_step_limit": [float("-inf"), 1]},
    {"time_step_limit": [0, float("nan")]},
    {"time_step_limit": [0, 1e100]},
])
def test_aeg_parameters_reject_unsupported_or_non_json_limits(envelope: Any) -> None:
    """Persisted parameters must have supported fields and finite JSON values."""
    with pytest.raises(ValueError, match="Mamba-2"):
        Mamba2Parameters.from_dict(envelope)


@pytest.mark.parametrize("limits", [(0, float("inf")), (float("-inf"), float("inf"))])
def test_source_unbounded_limits_normalize_to_legacy_execution(limits: tuple[float, float]) -> None:
    """Clamping positive softplus to an unbounded interval changes no arithmetic."""
    assert Mamba2Parameters.from_source_config({"time_step_limit": limits}) == Mamba2Parameters()


def test_required_parameters_roundtrip_without_changing_legacy_architecture() -> None:
    """Legacy dictionaries retain their exact fields; new semantics survive decoding."""
    architecture = ModelArchitecture(
        family="mamba_family", params_billion=0.0, layers=1, hidden_size=16,
        num_attention_heads=4,
    )
    legacy = architecture.to_dict()
    assert "ssm_runtime_parameters" not in legacy
    assert ModelArchitecture.from_dict(legacy).to_dict() == legacy
    parameters = Mamba2Parameters("relu", (0.1, 0.2)).to_dict()
    architecture.ssm_runtime_parameters = parameters
    encoded = architecture.to_dict()
    assert encoded["ssm_runtime_parameters"] == parameters
    assert ModelArchitecture.from_dict(encoded).to_dict() == encoded
    encoded["ssm_runtime_parameters"] = []
    with pytest.raises(ValueError, match=r"SSM runtime parameters.*JSON object"):
        ModelArchitecture.from_dict(encoded)


def test_packaged_parameters_are_authenticated_and_survive_worker_clone(tmp_path: Path) -> None:
    """The real artifact reader and serving clone consume the declared function."""
    source = tmp_path / "source"
    write_checkpoint(source, activation="relu", time_step_limit=(0.1, 0.2))
    artifact = tmp_path / "model.aeg"
    Compiler(CompilerConfig(
        quality_mode="diagnostic", targets=["cpu_generic"], skip_download=True,
        precision_assignment_mode="uniform", precision="FP32", calibration_tokens=4,
        cache_dir=str(tmp_path / "cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest is not None
    parameters = Mamba2Parameters("relu", (0.1, 0.2))
    assert package.manifest.architecture.ssm_runtime_parameters == parameters.to_dict()
    engine: Any = load_engine_from_path(
        artifact, allow_uncertified=True, prepare_native_kernels=False,
    )
    worker = clone_cpu_engine_for_worker(engine)
    assert worker is not engine and worker.weights is engine.weights
    assert worker.weights.parameters == parameters
    ids = np.asarray([1, 2, 3, 4, 5], np.int64)
    expected, _ = engine.forward(ids)
    actual, _ = worker.forward(ids)
    np.testing.assert_array_equal(actual, expected)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["architecture"]["ssm_runtime_parameters"]["time_step_limit"][1] = 0.3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AEGIntegrityError, match="manifest integrity check failed"):
        load_aeg_package(artifact)
