"""Preserve source Mamba-2 convolution activation and post-softplus bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from mamba2_normalization_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig
from aether.runtime.aeg_loader import load_engine_from_path
from aether.runtime.spec_decode import clone_cache

if TYPE_CHECKING:
    from pathlib import Path


CASES = [
    pytest.param({}, id="default"),
    pytest.param({"time_step_limit": (0.0, 0.001)}, id="upper-limit"),
    pytest.param({"time_step_limit": (1.0, 2.0)}, id="lower-limit"),
    pytest.param({"time_step_limit": (0.03, 0.03)}, id="equal-limits"),
    pytest.param({"time_step_limit": (0.0, 0.0)}, id="zero-limit"),
    pytest.param({"time_step_limit": (0.0, float("inf"))}, id="unbounded"),
    pytest.param({"time_step_limit": (1.0, None)}, id="open-upper-limit"),
    pytest.param({"time_step_limit": (None, 0.001)}, id="open-lower-limit"),
    pytest.param({"time_step_limit": (0.2, 0.1)}, id="inverted-limits"),
    pytest.param({"activation": "relu"}, id="relu-convolution"),
    pytest.param({"activation": "relu", "time_step_limit": (0.1, 0.2)}, id="relu-bounded"),
]


@pytest.mark.parametrize("options", CASES)
def test_mamba2_source_parameter_matrix_matches_original_hf(
    tmp_path: Path, options: dict[str, Any],
) -> None:
    """Validate matrix math against the source framework before testing AEG.

    Transformers v4.57.3 Mamba2Mixer.torch_forward applies ACT2FN[hidden_act]
    to convolution and clamps softplus(dt+bias) before the SSD recurrence:
    https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mamba2/modeling_mamba2.py
    The post-convolution output gate remains SiLU (paper section 7.1).
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "source"
    config = write_checkpoint(source, **options)
    _, tensors = checkpoint_data(**options)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
    ).eval()
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    with torch.inference_mode():
        actual = model(torch.tensor(ids[None]), use_cache=False).logits[0].numpy()
    expected = matrix_logits(config, tensors, ids)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("options", CASES)
@pytest.mark.parametrize("backend", ["cpu", "torch_cpu"])
def test_packaged_mamba2_source_parameters_and_cached_continuation(
    tmp_path: Path, options: dict[str, Any], backend: str,
) -> None:
    """Compile real source parameters; check full rows and an owned cache fork."""
    if backend == "torch_cpu":
        pytest.importorskip("torch")
    source = tmp_path / "source"
    config = write_checkpoint(source, **options)
    _, tensors = checkpoint_data(**options)
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
    expected = matrix_logits(config, tensors, ids)
    actual, _ = engine.forward_step(ids, return_all_logits=True)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    _, cache = engine.forward_step(ids[:3])
    prefix_logits = np.asarray(cache.last_logits).copy()
    continued, updated = engine.forward_step(ids[3:], clone_cache(cache), return_all_logits=True)
    np.testing.assert_allclose(continued, expected[3:], rtol=3e-5, atol=3e-6)
    assert cache.length == 3 and updated.length == len(ids)
    np.testing.assert_array_equal(np.asarray(cache.last_logits), prefix_logits)
    assert all(h.shape[-1] == config["conv_kernel"] - 1 for h in updated.conv_history)
