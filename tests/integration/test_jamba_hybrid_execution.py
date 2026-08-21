"""Real mixed attention/selective-scan compile and runtime coverage."""

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
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


@pytest.mark.integration
def test_jamba_hybrid_compiles_reloads_and_reuses_cache(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    source = tmp_path / "jamba"
    source.mkdir()
    vocab, hidden, inner, state, dt_rank, kernel = 16, 8, 16, 2, 1, 3
    heads = kv_heads = 2
    head_dim = hidden // heads
    intermediate = 16
    rng = np.random.default_rng(101)

    def matrix(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.normal(size=shape) * 0.05).astype("float32")

    def vector(size: int) -> np.ndarray:
        return np.ones(size, dtype="float32")

    config = {
        "architectures": ["JambaForCausalLM"], "model_type": "jamba",
        "num_hidden_layers": 2, "hidden_size": hidden, "d_model": hidden,
        "intermediate_size": intermediate, "num_attention_heads": heads,
        "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "d_inner": inner, "d_state": state, "dt_rank": dt_rank,
        "d_conv": kernel, "vocab_size": vocab, "rms_norm_eps": 1e-5,
        "layers_block_type": ["mamba", "attention"],
        "tie_word_embeddings": False, "torch_dtype": "float32",
    }
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": matrix((vocab, hidden)),
        "model.norm.weight": vector(hidden),
        "lm_head.weight": matrix((vocab, hidden)),
        # Jamba/Mamba block 0.
        "model.layers.0.mamba.norm.weight": vector(hidden),
        "model.layers.0.mamba.in_proj.weight": matrix((2 * inner, hidden)),
        "model.layers.0.mamba.conv1d.weight": matrix((inner, kernel)),
        "model.layers.0.mamba.conv1d.bias": np.zeros(inner, dtype="float32"),
        "model.layers.0.mamba.x_proj.weight": matrix((dt_rank + 2 * state, inner)),
        "model.layers.0.mamba.dt_proj.weight": matrix((inner, dt_rank)),
        "model.layers.0.mamba.dt_proj.bias": np.zeros(inner, dtype="float32"),
        "model.layers.0.mamba.A_log": matrix((inner, state)),
        "model.layers.0.mamba.D": vector(inner),
        "model.layers.0.mamba.out_proj.weight": matrix((hidden, inner)),
        # Transformer block 1.
        "model.layers.1.input_layernorm.weight": vector(hidden),
        "model.layers.1.self_attn.q_proj.weight": matrix((heads * head_dim, hidden)),
        "model.layers.1.self_attn.k_proj.weight": matrix((kv_heads * head_dim, hidden)),
        "model.layers.1.self_attn.v_proj.weight": matrix((kv_heads * head_dim, hidden)),
        "model.layers.1.self_attn.o_proj.weight": matrix((hidden, heads * head_dim)),
        "model.layers.1.post_attention_layernorm.weight": vector(hidden),
        "model.layers.1.mlp.gate_proj.weight": matrix((intermediate, hidden)),
        "model.layers.1.mlp.up_proj.weight": matrix((intermediate, hidden)),
        "model.layers.1.mlp.down_proj.weight": matrix((hidden, intermediate)),
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors.save_file(tensors, str(source / "model.safetensors"))
    _write_tokenizer(source, vocab)

    artifact = tmp_path / "jamba.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=4,
        cache_dir=str(tmp_path / "cache"),
    )).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    assert package.manifest is not None
    assert package.manifest.architecture.ssm_variant == "hybrid_selective_scan"
    assert package.manifest.architecture.hybrid_layer_types == ["ssm", "attention"]
    names = set(package.weight_store().entries)
    assert "layer_0_ssm_in_proj" in names
    assert "layer_1_q_proj" in names
    assert "layer_1_gate_proj" in names
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert package.supports_runtime_target("cuda_sm90")
    assert package.supports_runtime_target("rocm_cdna3")
    assert package.supports_runtime_target("metal_m3")
    assert "pytorch" in package.manifest.kernels.portable_backends

    engine = load_engine_from_path(artifact)
    full_logits, full_cache = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    step_logits, step_cache = engine.forward(np.asarray([1, 2], dtype=np.int64))
    step_logits_2, step_cache = engine.forward(np.asarray([3], dtype=np.int64), step_cache)
    assert full_logits.shape == (3, vocab)
    np.testing.assert_allclose(full_logits[-1], step_logits_2[-1], rtol=1e-5, atol=1e-5)
    assert full_cache.length == step_cache.length == 3

    torch = pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchHybridAEGEngine

    portable = TorchHybridAEGEngine(engine, "cpu")
    portable_logits, _ = portable.forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(full_logits, portable_logits, rtol=2e-5, atol=2e-5)
    mesh = TorchHybridAEGEngine(engine, "cpu", devices=["cpu:0", "cpu:1"])
    mesh_logits, _ = mesh.forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(full_logits, mesh_logits, rtol=2e-5, atol=2e-5)

    result = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert result.usage["completion_tokens"] == 2
    assert result.text


@pytest.mark.integration
def test_jamba_hybrid_rejects_incomplete_schedule(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    source = tmp_path / "invalid-jamba"
    source.mkdir()
    config = {
        "architectures": ["JambaForCausalLM"], "model_type": "jamba",
        "num_hidden_layers": 2, "hidden_size": 8, "d_model": 8,
        "num_attention_heads": 2, "num_key_value_heads": 2, "vocab_size": 8,
        "d_inner": 16, "d_state": 2, "dt_rank": 1, "d_conv": 3,
        "layers_block_type": ["mamba"],
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors.save_file({"model.embed_tokens.weight": np.zeros((8, 8), dtype="float32")}, str(source / "model.safetensors"))
    with pytest.raises(Exception, match="hybrid|schedule|layer"):
        Compiler(CompilerConfig(targets=["cpu_avx2"], overwrite=True)).compile(
            str(source), output_path=tmp_path / "invalid.aeg"
        )
