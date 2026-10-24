"""Finite-input Mamba-2 math through actual source checkpoints and executors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba2_normalization_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig
from aether.runtime.aeg_loader import load_engine_from_path
from aether.runtime.mamba2_engine import Mamba2ExecutionEngine
from aether.runtime.spec_decode import clone_cache

if TYPE_CHECKING:
    from pathlib import Path

CASES = [
    pytest.param({}, id="default"),
    pytest.param({"large_delta": True}, id="large-delta"),
    pytest.param({"with_bias": True, "large_silu": "gate"}, id="negative-gate"),
    pytest.param({"large_silu": "conv"}, id="negative-conv"),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mamba2_silu_preserves_finite_input_function_and_ownership(dtype: Any) -> None:
    """Compare x*sigmoid(x) with independent FP64 log-sigmoid arithmetic.

    Dao & Gu (2024), section 7.1, uses SiLU for convolution and gating:
    https://arxiv.org/html/2405.21060v1
    Official v2.2.4 Mamba2Simple uses nn.SiLU(), not a clipped input.
    """
    limit = np.finfo(dtype).max
    inputs = np.asarray([
        -limit, -1e20, -1000, -700, -100, -80, -61, -60, -30, -1, 0,
        1, 30, 60, 61, 100, 1000, 1e20, limit,
    ], dtype=dtype)
    before = inputs.copy()
    reference = inputs.astype(np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        expected = reference * np.exp(-np.logaddexp(0, -reference))
        actual = Mamba2ExecutionEngine._silu(inputs)
    assert actual.dtype == inputs.dtype and actual.shape == inputs.shape
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(inputs, before)
    # Below the dtype's normal range, intermediate exponent rounding has an
    # absolute precision floor; it must not produce a normal-sized false value.
    np.testing.assert_allclose(
        actual, expected, rtol=float(8 * np.finfo(dtype).eps), atol=float(np.finfo(dtype).tiny),
    )


@pytest.mark.parametrize("options", CASES)
def test_mamba2_matrix_oracle_matches_original_hf(tmp_path: Path, options: dict[str, Any]) -> None:
    """Validate the full semiseparable equation independently of either AEG engine."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "source"
    config = write_checkpoint(source, **options)
    _, tensors = checkpoint_data(**options)
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
    ).eval()
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    with torch.inference_mode():
        actual = reference(torch.tensor(ids[None, :]), use_cache=False).logits[0].numpy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        expected = matrix_logits(config, tensors, ids)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("options", CASES)
@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
def test_packaged_mamba2_stability_and_owned_cache_match_matrix_oracle(
    tmp_path: Path, options: dict[str, Any], backend: str,
) -> None:
    """Keep real B/C, learned norms and cached state when finite inputs are extreme."""
    if backend == "torch_cpu":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source, **options)
    _, tensors = checkpoint_data(**options)
    inner = config["expand"] * config["hidden_size"]
    assert np.any(tensors["backbone.layers.0.mixer.in_proj.weight"][2 * inner:])
    assert np.any(tensors["backbone.layers.0.mixer.conv1d.weight"][inner:])
    artifact = tmp_path / "model.aeg"
    Compiler(CompilerConfig(
        quality_mode="diagnostic", targets=["cpu_generic"], skip_download=True,
        precision_assignment_mode="uniform", precision="FP32", calibration_tokens=4,
        cache_dir=str(tmp_path / "cache"),
    )).compile(str(source), output_path=artifact)
    engine: Any = load_engine_from_path(
        artifact, allow_uncertified=True, prepare_native_kernels=False,
    )
    if backend == "torch_cpu":
        from aether.runtime.torch_state_engine import TorchMamba2AEGEngine

        engine = TorchMamba2AEGEngine(engine, "cpu")
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        expected = matrix_logits(config, tensors, ids)
        actual, _ = engine.forward(ids)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    _, prefix = engine.forward(ids[:3])
    prefix_logits = np.asarray(prefix.last_logits).copy()
    suffix, updated = engine.forward(ids[3:], clone_cache(prefix))
    assert prefix.length == 3 and updated.length == len(ids)
    np.testing.assert_array_equal(np.asarray(prefix.last_logits), prefix_logits)
    np.testing.assert_allclose(suffix, expected[3:], rtol=3e-5, atol=3e-6)
    assert all(history.shape[-1] == config["conv_kernel"] - 1 for history in updated.conv_history)
    all_rows, _ = engine.forward_step(ids.tolist(), return_all_logits=True)
    np.testing.assert_allclose(all_rows, expected, rtol=3e-5, atol=3e-6)
    last, _ = engine.forward_step(ids.tolist())
    assert last.shape == (config["vocab_size"],)
    np.testing.assert_allclose(last, expected[-1], rtol=3e-5, atol=3e-6)
