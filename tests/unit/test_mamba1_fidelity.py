"""Exercise the Mamba-1 HF weight contract through actual AEG/engine paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba1_fidelity_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.core.exceptions import CompilationError, IngestionError
from aether.runtime.aeg_loader import AEGLoadError, load_engine_from_path
from aether.runtime.spec_decode import clone_cache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("rms_norm", [False, True])
def test_hf_mamba1_uses_rmsnorm_despite_layer_norm_epsilon(rms_norm: bool) -> None:
    """HF 4.57.3 constructs MambaRMSNorm, independent of its unused rms_norm flag."""
    config, _ = checkpoint_data(rms_norm=rms_norm)
    assert ArchitectureDetector()._from_config(config).norm_type == "RMSNorm"


@pytest.mark.parametrize("expansion", [2, 4])
def test_mamba1_minimal_hf_config_preserves_expansion(expansion: int) -> None:
    """HF derives intermediate_size from expand; the raw config need not repeat it."""
    config, _ = checkpoint_data(expansion=expansion)
    assert "intermediate_size" not in config
    architecture = ArchitectureDetector()._from_config(config)
    assert architecture.ssm_inner_size == expansion * config["hidden_size"]


def _compiled(tmp_path: Path, backend: str, **options: Any) -> tuple[Any, Any, Any]:
    if backend == "torch_cpu":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source, **options)
    _, tensors = checkpoint_data(**options)
    artifact = tmp_path / "model.aeg"
    Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            targets=["cpu_generic"],
            skip_download=True,
            precision_assignment_mode="uniform",
            precision="FP32",
            calibration_tokens=4,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)
    engine: Any = load_engine_from_path(artifact, allow_uncertified=True)
    if backend == "torch_cpu":
        from aether.runtime.torch_state_engine import TorchMambaAEGEngine

        engine = TorchMambaAEGEngine(engine, "cpu")
    return engine, config, tensors


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
@pytest.mark.parametrize("expansion", [2, 4])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packaged_mamba1_forward_and_owned_continuation_match_matrix_oracle(
    tmp_path: Path,
    backend: str,
    expansion: int,
    width: int,
    with_bias: bool,
) -> None:
    """Keep real nonzero B/C, learned normalizers and biases through packaging."""
    engine, config, tensors = _compiled(
        tmp_path,
        backend,
        expansion=expansion,
        width=width,
        with_bias=with_bias,
    )
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], np.int64)
    expected = matrix_logits(config, tensors, ids)
    actual, _ = engine.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    _, prefix = engine.forward(ids[:3])
    saved_logits = np.asarray(prefix.last_logits).copy()
    suffix, continued = engine.forward(ids[3:], clone_cache(prefix))
    assert prefix.length == 3 and continued.length == len(ids)
    np.testing.assert_array_equal(np.asarray(prefix.last_logits), saved_logits)
    np.testing.assert_allclose(suffix, expected[3:], rtol=3e-5, atol=3e-6)
    assert all(history.shape[-1] == width - 1 for history in continued.conv_history)
    all_rows, _ = engine.forward_step(ids.tolist(), return_all_logits=True)
    np.testing.assert_allclose(all_rows, expected, rtol=3e-5, atol=3e-6)
    last, _ = engine.forward_step(ids.tolist())
    assert last.shape == (config["vocab_size"],)
    np.testing.assert_allclose(last, expected[-1], rtol=3e-5, atol=3e-6)
    if with_bias:
        np.testing.assert_array_equal(
            engine.weights.layers[0].in_proj_bias, tensors["backbone.layers.0.mixer.in_proj.bias"]
        )
        np.testing.assert_array_equal(
            engine.weights.layers[0].out_proj_bias, tensors["backbone.layers.0.mixer.out_proj.bias"]
        )


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
def test_large_finite_mamba1_time_step_bias_matches_finite_matrix_output(
    tmp_path: Path,
    backend: str,
) -> None:
    """Softplus(1000) is finite; no exp overflow is required by the checkpoint math."""
    engine, config, tensors = _compiled(tmp_path, backend, large_delta=True)
    ids = np.asarray([1, 2, 3, 4], np.int64)
    with np.errstate(over="raise", invalid="raise"):
        actual, _ = engine.forward(ids)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, matrix_logits(config, tensors, ids), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
def test_zero_epsilon_is_preserved_from_checkpoint_to_runtime(tmp_path: Path, backend: str) -> None:
    """Zero is a valid declared epsilon, not an absent configuration value."""
    engine, config, tensors = _compiled(tmp_path, backend, epsilon=0.0)
    assert engine.weights.norm_eps == 0.0
    ids = np.asarray([1, 2, 3, 4], np.int64)
    actual, _ = engine.forward(ids)
    np.testing.assert_allclose(actual, matrix_logits(config, tensors, ids), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
@pytest.mark.parametrize("site", ["gate", "conv"])
def test_silu_extremes_follow_the_sigmoid_definition(
    tmp_path: Path,
    backend: str,
    site: str,
) -> None:
    """A clipped sigmoid can amplify tiny false outputs through valid large weights."""
    engine, config, tensors = _compiled(tmp_path, backend, with_bias=True, large_silu=site)
    ids = np.asarray([1, 2, 3, 4], np.int64)
    with np.errstate(over="raise", invalid="raise"):
        actual, _ = engine.forward(ids)
    np.testing.assert_allclose(actual, matrix_logits(config, tensors, ids), rtol=3e-5, atol=3e-6)


def test_malformed_learned_norm_is_refused_instead_of_padded(tmp_path: Path) -> None:
    """Diagnostic execution cannot invent missing learned scale values."""
    with pytest.raises((AEGLoadError, CompilationError), match=r"[Nn]orm|shape"):
        _compiled(tmp_path, "cpu", norm_width=15)


@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
def test_unsupported_mixer_activation_is_refused_before_publication(
    tmp_path: Path,
    activation: str,
) -> None:
    """A recognized family name cannot turn an unsupported source into SiLU."""
    source = tmp_path / "mamba"
    write_checkpoint(source, activation=activation)
    artifact = tmp_path / "model.aeg"
    with pytest.raises(CompilationError, match=r"[Aa]ctivation"):
        Compiler(
            CompilerConfig(
                quality_mode="diagnostic",
                targets=["cpu_generic"],
                skip_download=True,
                precision_assignment_mode="uniform",
                precision="FP32",
                calibration_tokens=4,
                cache_dir=str(tmp_path / "cache"),
            )
        ).compile(str(source), output_path=artifact)
    assert not artifact.exists()


@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
@pytest.mark.parametrize(
    "options",
    [
        {"activation": "swish"},
        {"omit_activation": True},
        {"layers": 2, "with_bias": True, "expansion": 3},
    ],
)
def test_supported_activation_and_multilayer_controls_match_full_matrix(
    tmp_path: Path,
    backend: str,
    options: dict[str, Any],
) -> None:
    """Allow the true SiLU aliases/default, and compose distinct recurrent blocks."""
    engine, config, tensors = _compiled(tmp_path, backend, **options)
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], np.int64)
    expected = matrix_logits(config, tensors, ids)
    actual, _ = engine.forward(ids)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    _, prefix = engine.forward(ids[:2])
    continued, cache = engine.forward(ids[2:], clone_cache(prefix))
    assert cache.length == len(ids) and prefix.length == 2
    np.testing.assert_allclose(continued, expected[2:], rtol=3e-5, atol=3e-6)


def test_specialized_loader_failure_cannot_bypass_activation_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real generic fallback still reaches the common source-contract check."""
    from aether.compiler.stage1_ingestion.ssm_loader import SSMLoader

    source = tmp_path / "source"
    config = write_checkpoint(source, activation="relu")
    calls = []

    def unavailable(self: SSMLoader, *args: Any, **kwargs: Any) -> None:
        calls.append(args[0])
        raise RuntimeError("specialized loader unavailable")

    monkeypatch.setattr(SSMLoader, "load", unavailable)
    with pytest.raises(IngestionError, match="Unsupported Mamba-1 hidden_act"):
        IngestionPipeline().ingest(str(source), ArchitectureDetector()._from_config(config))
    assert calls == [source]


def test_materialized_snapshot_cannot_bypass_activation_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read the actual materialized config even when ingestion began with a Hub ID."""
    source = tmp_path / "snapshot"
    config = write_checkpoint(source, activation="gelu")

    def local_snapshot(self: IngestionPipeline, model: str) -> str:
        assert model == "offline-test/mamba-fidelity"
        return str(source)

    monkeypatch.setattr(IngestionPipeline, "_download_hf_snapshot", local_snapshot)
    pipeline = IngestionPipeline(CompilerConfig(skip_download=False))
    with pytest.raises(IngestionError, match="Unsupported Mamba-1 hidden_act"):
        pipeline.ingest("offline-test/mamba-fidelity", ArchitectureDetector()._from_config(config))


@pytest.mark.parametrize("existing_destination", [False, True])
def test_cli_activation_refusal_never_publishes_or_replaces_destination(
    tmp_path: Path,
    existing_destination: bool,
) -> None:
    """Exercise actual Click/compiler failure and preserve any prior destination."""
    from click.testing import CliRunner

    from aether.cli import cli

    source = tmp_path / "source"
    write_checkpoint(source, activation="tanh")
    artifact = tmp_path / "model.aeg"
    if existing_destination:
        artifact.mkdir()
        (artifact / "prior.txt").write_bytes(b"prior destination")
    result = CliRunner().invoke(
        cli,
        [
            "--diagnostic",
            "--cache-dir",
            str(tmp_path / "cache"),
            "compile",
            str(source),
            "--target",
            "cpu_generic",
            "--precision",
            "FP32",
            "--output",
            str(artifact),
            "--overwrite",
        ],
    )
    assert result.exit_code != 0
    assert "Unsupported Mamba-1 hidden_act" in result.output
    assert "Traceback" not in result.output
    if existing_destination:
        assert [path.name for path in artifact.iterdir()] == ["prior.txt"]
        assert (artifact / "prior.txt").read_bytes() == b"prior destination"
    else:
        assert not artifact.exists()
