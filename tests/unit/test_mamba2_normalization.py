"""Keep distinct learned Mamba-2 norms through packaging and real execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from mamba2_normalization_fixtures import checkpoint_data, matrix_logits, write_checkpoint

from aether import Compiler, CompilerConfig
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.runtime.aeg_loader import load_engine_from_path

if TYPE_CHECKING:
    from pathlib import Path


def test_recurrent_reference_accepts_finite_probes_without_inventing_source_context() -> None:
    """A recurrent HF model has no positional table but still has bounded work."""
    from aether.evaluation.admission_worker import tokenize_samples

    probe = {"id": "ssm", "prompt_ids": [1, 2], "continuation_ids": [3, 4]}
    samples = tokenize_samples([probe], None, 17, None, 8)
    assert samples[0]["prompt_ids"] == [1, 2]
    assert samples[0]["continuation_ids"] == [3, 4]
    assert samples[0]["greedy_tokens"] == 8


def test_recurrent_reference_window_is_bounded_separately_from_model_capability() -> None:
    """Source-context absence must not authorize unbounded reference work."""
    from aether.evaluation.admission_worker import tokenize_samples

    probe = {"id": "too-long", "prompt_ids": [1] * 4096, "continuation_ids": [2]}
    with pytest.raises(ValueError, match=r"recurrent reference probes.*4096"):
        tokenize_samples([probe], None, 17, None, 8)


@pytest.mark.parametrize("rms_norm", [True, False])
def test_hf_mamba2_uses_rmsnorm_even_with_layer_norm_epsilon(rms_norm: bool) -> None:
    """HF 4.57.3 constructs Mamba2RMSNorm regardless of its unused rms_norm flag."""
    config, _ = checkpoint_data(rms_norm=rms_norm)
    architecture = ArchitectureDetector()._from_config(config)
    assert architecture.norm_type == "RMSNorm"


def test_mamba2_norms_have_distinct_source_and_canonical_keys() -> None:
    """Input and post-gate normalization are different learned functions."""
    normalise = IngestionPipeline._normalise_weight_name
    assert normalise("backbone.layers.0.norm.weight") == (0, "ssm_norm")
    assert normalise("backbone.layers.0.mixer.norm.weight") == (0, "ssm_output_norm")
    assert normalise("layer_0_ssm_output_norm") == (0, "ssm_output_norm")
    assert normalise("model.layers.0.mamba.norm.weight") == (0, "ssm_norm")


@pytest.mark.parametrize("expansion", [2, 4])
def test_mamba2_preserves_hf_expanded_head_geometry(expansion: int) -> None:
    """The expanded channel count is expand*hidden, matching the stored projections."""
    config, _ = checkpoint_data(expansion=expansion)
    architecture = ArchitectureDetector()._from_config(config)
    assert architecture.ssm_inner_size == config["expand"] * config["hidden_size"]
    assert architecture.ssm_head_dim == config["head_dim"]


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("expansion", [2, 4])
def test_mamba2_packaged_norms_and_cached_logits_match_matrix_oracle(
    tmp_path: Path,
    expansion: int,
    with_bias: bool,
) -> None:
    """Consume every learned normalizer, comparing independent equation-3 outputs."""
    source = tmp_path / "source"
    config = write_checkpoint(source, expansion=expansion, with_bias=with_bias)
    _, tensors = checkpoint_data(expansion=expansion, with_bias=with_bias)
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
    engine = load_engine_from_path(artifact, allow_uncertified=True)
    np.testing.assert_array_equal(
        engine.weights.layers[0].norm, tensors["backbone.layers.0.norm.weight"]
    )
    np.testing.assert_array_equal(
        engine.weights.layers[0].output_norm, tensors["backbone.layers.0.mixer.norm.weight"]
    )
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], np.int64)
    expected = matrix_logits(config, tensors, ids)
    logits, _ = engine.forward_step(ids, return_all_logits=True)
    np.testing.assert_allclose(logits, expected, atol=3e-6, rtol=3e-5)
    _, cache = engine.forward_step(ids[:3])
    last, cache = engine.forward_step(ids[3:], cache)
    assert last.shape == (config["vocab_size"],)
    np.testing.assert_allclose(last, expected[-1], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("with_bias", [False, True])
def test_mamba2_torch_cpu_uses_the_same_learned_norm_and_bias(with_bias: bool) -> None:
    """Run actual Torch CPU forwards against the independent full-matrix oracle."""
    pytest.importorskip("torch")
    from aether.runtime.mamba2_engine import (
        Mamba2ExecutionEngine,
        Mamba2LayerWeights,
        Mamba2ModelWeights,
    )
    from aether.runtime.torch_state_engine import TorchMamba2AEGEngine

    config, tensors = checkpoint_data(with_bias=with_bias)
    prefix = "backbone.layers.0."
    layer = Mamba2LayerWeights(
        norm=tensors[prefix + "norm.weight"],
        in_proj=tensors[prefix + "mixer.in_proj.weight"],
        conv1d=tensors[prefix + "mixer.conv1d.weight"],
        a_log=tensors[prefix + "mixer.A_log"],
        d=tensors[prefix + "mixer.D"],
        dt=tensors[prefix + "mixer.dt_bias"],
        out_proj=tensors[prefix + "mixer.out_proj.weight"],
        conv_bias=tensors[prefix + "mixer.conv1d.bias"],
        in_proj_bias=tensors.get(prefix + "mixer.in_proj.bias"),
        output_norm=tensors[prefix + "mixer.norm.weight"],
        out_proj_bias=tensors.get(prefix + "mixer.out_proj.bias"),
    )
    source = Mamba2ExecutionEngine(
        Mamba2ModelWeights(
            tensors["backbone.embeddings.weight"],
            [layer],
            tensors["backbone.norm_f.weight"],
            tensors["lm_head.weight"],
        ),
        state_size=4,
        inner_size=32,
        num_heads=4,
        num_groups=2,
        conv_kernel=3,
        head_dim=8,
    )
    engine = TorchMamba2AEGEngine(source, "cpu")
    ids = np.asarray([1, 2, 3, 4, 5, 6, 7], np.int64)
    logits, _ = engine.forward_step(ids, return_all_logits=True)
    np.testing.assert_allclose(logits, matrix_logits(config, tensors, ids), atol=3e-6, rtol=3e-5)
    _, cache = engine.forward_step(ids[:3])
    last, _ = engine.forward_step(ids[3:], cache)
    assert last.shape == (config["vocab_size"],)
    np.testing.assert_allclose(last, logits[-1], atol=3e-6, rtol=3e-5)
