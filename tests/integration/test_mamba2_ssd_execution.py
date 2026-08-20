"""Real SafeTensors compile/run coverage for the Mamba-2 SSD contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import load_aeg_package
from aether.runtime.aeg_loader import load_engine_from_path


def _write_tokenizer(path: Path, vocab_size: int) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "hello": 1}
    vocab.update({f"tok{i}": i + 2 for i in range(vocab_size - 2)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>").save_pretrained(str(path))


@pytest.mark.integration
def test_mamba2_ssd_compiles_serializes_and_generates(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    source = tmp_path / "mamba2"
    source.mkdir()
    vocab, hidden, state, conv, groups, heads, head_dim = 16, 8, 2, 3, 2, 4, 4
    inner = heads * head_dim
    channels = inner + 2 * groups * state
    rng = np.random.default_rng(31)

    def matrix(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.normal(size=shape) * 0.05).astype("float32")

    def vector(size: int, value: float = 1.0) -> np.ndarray:
        return np.full(size, value, dtype="float32")

    config = {
        "architectures": ["Mamba2ForCausalLM"], "model_type": "mamba2",
        "num_hidden_layers": 1, "d_model": hidden, "hidden_size": hidden,
        "d_state": state, "d_conv": conv, "n_heads": heads,
        "headdim": head_dim, "n_groups": groups, "vocab_size": vocab,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": False,
    }
    tensors = {
        "backbone.embeddings.weight": matrix((vocab, hidden)),
        "backbone.norm_f.weight": vector(hidden),
        "lm_head.weight": matrix((vocab, hidden)),
        "backbone.layers.0.norm.weight": vector(hidden),
        "backbone.layers.0.mixer.in_proj.weight": matrix((2 * inner + 2 * groups * state + heads, hidden)),
        "backbone.layers.0.mixer.in_proj.bias": np.zeros(2 * inner + 2 * groups * state + heads, dtype="float32"),
        "backbone.layers.0.mixer.conv1d.weight": matrix((channels, 1, conv)),
        "backbone.layers.0.mixer.conv1d.bias": np.zeros(channels, dtype="float32"),
        "backbone.layers.0.mixer.A_log": np.zeros(heads, dtype="float32"),
        "backbone.layers.0.mixer.D": vector(heads),
        "backbone.layers.0.mixer.dt_bias": np.zeros(heads, dtype="float32"),
        "backbone.layers.0.mixer.out_proj.weight": matrix((hidden, inner)),
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors.save_file(tensors, str(source / "model.safetensors"))
    _write_tokenizer(source, vocab)

    artifact = tmp_path / "mamba2.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=4, cache_dir=str(tmp_path / "cache"),
    )).compile(
        str(source), output_path=artifact
    )
    package = load_aeg_package(artifact)
    assert package.manifest is not None
    assert package.manifest.architecture.ssm_variant == "ssd"
    assert package.supports_runtime_target("cpu_avx2")
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    names = set(package.weight_store().entries)
    assert "layer_0_ssm_in_proj" in names
    assert "layer_0_ssm_dt" in names
    result = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert result.text

    torch = pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchMamba2AEGEngine

    cpu_engine = load_engine_from_path(artifact)
    cpu_logits, _ = cpu_engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    portable = TorchMamba2AEGEngine(cpu_engine, "cpu")
    portable_logits, _ = portable.forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)
