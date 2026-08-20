"""Offline end-to-end test for a real local SafeTensors model.

This deliberately uses a tiny Llama checkpoint written with the real
SafeTensors and Transformers tokenizer formats.  It verifies the contract
that matters to users: compile, close/reload the persisted AEG, and generate
through the public Runtime API without a network connection.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.runtime.aeg_loader import load_engine_from_path


def _write_tiny_llama(
    path: Path,
    weights_format: str = "safetensors",
    max_position_embeddings: int | None = None,
    num_layers: int = 1,
    architecture_name: str = "LlamaForCausalLM",
    model_type: str = "llama",
    mtp_heads: int = 0,
) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    vocab_size = 32
    hidden = 16
    intermediate = 32
    config = {
                "architectures": [architecture_name],
                "model_type": model_type,
                "num_hidden_layers": num_layers,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": vocab_size,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "torch_dtype": "float32",
            }
    if max_position_embeddings is not None:
        config["max_position_embeddings"] = max_position_embeddings
    if mtp_heads:
        config["num_nextn_predict_layers"] = mtp_heads
    (path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    tensors = {
        "model.embed_tokens.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.norm.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
    }
    for layer_index in range(num_layers):
        tensors.update({
            f"model.layers.{layer_index}.input_layernorm.weight": np.ones(hidden, dtype="float32"),
            f"model.layers.{layer_index}.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
            f"model.layers.{layer_index}.self_attn.q_proj.weight": rng.normal(size=(16, hidden)).astype("float32"),
            f"model.layers.{layer_index}.self_attn.k_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
            f"model.layers.{layer_index}.self_attn.v_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
            f"model.layers.{layer_index}.self_attn.o_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
            f"model.layers.{layer_index}.mlp.gate_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
            f"model.layers.{layer_index}.mlp.up_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
            f"model.layers.{layer_index}.mlp.down_proj.weight": rng.normal(size=(hidden, intermediate)).astype("float32"),
        })
    for head_index in range(mtp_heads):
        tensors[f"model.mtp_heads.{head_index}.weight"] = rng.normal(
            size=(vocab_size, hidden)
        ).astype("float32")
    if weights_format == "safetensors":
        safetensors.save_file(tensors, str(path / "model.safetensors"))
    elif weights_format == "pytorch":
        torch = pytest.importorskip("torch")
        torch.save(
            {name: torch.from_numpy(value) for name, value in tensors.items()},
            str(path / "pytorch_model.bin"),
        )
    else:
        raise ValueError(f"unsupported test weight format: {weights_format}")

    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


def _write_tiny_gpt2(path: Path) -> None:
    """Write a real GPT-2-style checkpoint, including Conv1D layouts/biases."""
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab_size, hidden, intermediate, positions = 32, 16, 32, 32
    (path / "config.json").write_text(json.dumps({
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_layer": 1,
        "n_embd": hidden,
        "n_head": 2,
        "n_inner": intermediate,
        "n_positions": positions,
        "vocab_size": vocab_size,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
        "torch_dtype": "float32",
    }), encoding="utf-8")
    rng = np.random.default_rng(19)
    tensors = {
        "transformer.wte.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "transformer.wpe.weight": rng.normal(size=(positions, hidden)).astype("float32"),
        "transformer.ln_f.weight": np.ones(hidden, dtype="float32"),
        "transformer.ln_f.bias": np.zeros(hidden, dtype="float32"),
        "lm_head.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "transformer.h.0.ln_1.weight": np.ones(hidden, dtype="float32"),
        "transformer.h.0.ln_1.bias": np.zeros(hidden, dtype="float32"),
        "transformer.h.0.ln_2.weight": np.ones(hidden, dtype="float32"),
        "transformer.h.0.ln_2.bias": np.zeros(hidden, dtype="float32"),
        # GPT-2 Conv1D stores matrices as (in_features, out_features).
        "transformer.h.0.attn.c_attn.weight": rng.normal(size=(hidden, 3 * hidden)).astype("float32"),
        "transformer.h.0.attn.c_attn.bias": rng.normal(size=(3 * hidden,)).astype("float32"),
        "transformer.h.0.attn.c_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "transformer.h.0.attn.c_proj.bias": rng.normal(size=(hidden,)).astype("float32"),
        "transformer.h.0.mlp.c_fc.weight": rng.normal(size=(hidden, intermediate)).astype("float32"),
        "transformer.h.0.mlp.c_fc.bias": rng.normal(size=(intermediate,)).astype("float32"),
        "transformer.h.0.mlp.c_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        "transformer.h.0.mlp.c_proj.bias": rng.normal(size=(hidden,)).astype("float32"),
    }
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


def _write_tiny_t5(path: Path) -> None:
    """Write a minimal, real T5-style encoder-decoder SafeTensors checkpoint."""
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab_size, hidden, heads, head_dim, intermediate = 32, 8, 2, 4, 16
    (path / "config.json").write_text(json.dumps({
        "architectures": ["T5ForConditionalGeneration"],
        "model_type": "t5",
        "vocab_size": vocab_size,
        "d_model": hidden,
        "d_kv": head_dim,
        "d_ff": intermediate,
        "num_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": heads,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 32,
        "layer_norm_epsilon": 1e-6,
        "feed_forward_proj": "relu",
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
    }), encoding="utf-8")
    rng = np.random.default_rng(23)
    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(size=(rows, cols)).astype("float32")
    def norm() -> np.ndarray:
        return np.ones(hidden, dtype="float32")
    tensors = {
        "shared.weight": matrix(vocab_size, hidden),
        "encoder.final_layer_norm.weight": norm(),
        "decoder.final_layer_norm.weight": norm(),
        "encoder.block.0.layer.0.SelfAttention.q.weight": matrix(hidden, hidden),
        "encoder.block.0.layer.0.SelfAttention.k.weight": matrix(hidden, hidden),
        "encoder.block.0.layer.0.SelfAttention.v.weight": matrix(hidden, hidden),
        "encoder.block.0.layer.0.SelfAttention.o.weight": matrix(hidden, hidden),
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": matrix(8, heads),
        "encoder.block.0.layer.0.layer_norm.weight": norm(),
        "encoder.block.0.layer.1.DenseReluDense.wi.weight": matrix(intermediate, hidden),
        "encoder.block.0.layer.1.DenseReluDense.wo.weight": matrix(hidden, intermediate),
        "encoder.block.0.layer.1.layer_norm.weight": norm(),
        "decoder.block.0.layer.0.SelfAttention.q.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.0.SelfAttention.k.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.0.SelfAttention.v.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.0.SelfAttention.o.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": matrix(8, heads),
        "decoder.block.0.layer.0.layer_norm.weight": norm(),
        "decoder.block.0.layer.1.EncDecAttention.q.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.1.EncDecAttention.k.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.1.EncDecAttention.v.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.1.EncDecAttention.o.weight": matrix(hidden, hidden),
        "decoder.block.0.layer.1.layer_norm.weight": norm(),
        "decoder.block.0.layer.2.DenseReluDense.wi.weight": matrix(intermediate, hidden),
        "decoder.block.0.layer.2.DenseReluDense.wo.weight": matrix(hidden, intermediate),
        "decoder.block.0.layer.2.layer_norm.weight": norm(),
    }
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4}
    vocab.update({f"tok{i}": i + 5 for i in range(vocab_size - 5)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>", pad_token="<pad>", eos_token="</s>"
    ).save_pretrained(str(path))


def _write_tiny_mixtral(path: Path) -> None:
    """Write a minimal Mixtral-style sparse MoE checkpoint.

    The tensor names intentionally use the canonical Hugging Face Mixtral
    layout (w1/w2/w3 expert projections plus block_sparse_moe.gate) so this
    test exercises model-generic normalization rather than an Aether-only
    naming convention.
    """
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab_size, hidden, heads, intermediate, experts = 32, 8, 2, 12, 2
    (path / "config.json").write_text(json.dumps({
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_hidden_layers": 1,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "vocab_size": vocab_size,
        "num_local_experts": experts,
        "num_experts_per_tok": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }), encoding="utf-8")
    rng = np.random.default_rng(37)
    tensors = {
        "model.embed_tokens.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.norm.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.self_attn.k_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.self_attn.v_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.block_sparse_moe.gate.weight": rng.normal(size=(experts, hidden)).astype("float32"),
    }
    for expert in range(experts):
        tensors.update({
            f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
            f"model.layers.0.block_sparse_moe.experts.{expert}.w2.weight": rng.normal(size=(hidden, intermediate)).astype("float32"),
            f"model.layers.0.block_sparse_moe.experts.{expert}.w3.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        })
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


def _write_tiny_mla(path: Path, *, moe: bool = False) -> None:
    """Write a dense or routed DeepSeek-style MLA checkpoint."""
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab, hidden, heads = 32, 8, 2
    q_rank, kv_rank, nope, rope, value, intermediate = 4, 3, 2, 2, 2, 12
    config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "num_hidden_layers": 1,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "vocab_size": vocab,
        "kv_lora_rank": kv_rank,
        "q_lora_rank": q_rank,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope,
        "v_head_dim": value,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
    }
    if moe:
        config.update({"n_routed_experts": 2, "num_experts_per_tok": 1, "first_k_dense_replace": 0})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rng = np.random.default_rng(43)
    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(size=(rows, cols)).astype("float32")
    tensors = {
        "model.embed_tokens.weight": matrix(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": matrix(vocab, hidden),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.self_attn.q_a_proj.weight": matrix(q_rank, hidden),
        "model.layers.0.self_attn.q_a_layernorm.weight": np.ones(q_rank, dtype="float32"),
        "model.layers.0.self_attn.q_b_proj.weight": matrix(heads * (nope + rope), q_rank),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": matrix(kv_rank, hidden),
        "model.layers.0.self_attn.kv_a_layernorm.weight": np.ones(kv_rank, dtype="float32"),
        "model.layers.0.self_attn.kv_b_proj.weight": matrix(heads * (nope + value), kv_rank),
        "model.layers.0.self_attn.k_rope_proj.weight": matrix(heads * rope, hidden),
        "model.layers.0.self_attn.o_proj.weight": matrix(hidden, heads * value),
    }
    if moe:
        tensors["model.layers.0.mlp.gate.weight"] = matrix(2, hidden)
        for expert in range(2):
            tensors.update({
                f"model.layers.0.mlp.experts.{expert}.gate_proj.weight": matrix(intermediate, hidden),
                f"model.layers.0.mlp.experts.{expert}.up_proj.weight": matrix(intermediate, hidden),
                f"model.layers.0.mlp.experts.{expert}.down_proj.weight": matrix(hidden, intermediate),
            })
    else:
        tensors.update({
            "model.layers.0.mlp.gate_proj.weight": matrix(intermediate, hidden),
            "model.layers.0.mlp.up_proj.weight": matrix(intermediate, hidden),
            "model.layers.0.mlp.down_proj.weight": matrix(hidden, intermediate),
        })
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab_map = {"<unk>": 0, "hello": 1, "world": 2}
    vocab_map.update({f"tok{i}": i + 3 for i in range(vocab - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab_map, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


def _write_tiny_mamba(path: Path) -> None:
    """Write a minimal Mamba-1 selective-scan checkpoint."""
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab, hidden, inner, state, dt_rank, kernel = 32, 8, 16, 2, 1, 3
    (path / "config.json").write_text(json.dumps({
        "architectures": ["MambaForCausalLM"], "model_type": "mamba",
        "num_hidden_layers": 1, "hidden_size": hidden, "d_model": hidden,
        "d_inner": inner, "d_state": state, "dt_rank": dt_rank,
        "d_conv": kernel, "num_attention_heads": 1, "vocab_size": vocab,
        "rms_norm_eps": 1e-5, "torch_dtype": "float32",
    }), encoding="utf-8")
    rng = np.random.default_rng(47)
    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(size=(rows, cols)).astype("float32")
    tensors = {
        "backbone.embeddings.weight": matrix(vocab, hidden),
        "backbone.norm_f.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": matrix(vocab, hidden),
        "backbone.layers.0.norm.weight": np.ones(hidden, dtype="float32"),
        "backbone.layers.0.mixer.in_proj.weight": matrix(2 * inner, hidden),
        "backbone.layers.0.mixer.conv1d.weight": matrix(inner, kernel),
        "backbone.layers.0.mixer.conv1d.bias": np.zeros(inner, dtype="float32"),
        "backbone.layers.0.mixer.x_proj.weight": matrix(dt_rank + 2 * state, inner),
        "backbone.layers.0.mixer.dt_proj.weight": matrix(inner, dt_rank),
        "backbone.layers.0.mixer.dt_proj.bias": np.zeros(inner, dtype="float32"),
        "backbone.layers.0.mixer.A_log": matrix(inner, state),
        "backbone.layers.0.mixer.D": np.ones(inner, dtype="float32"),
        "backbone.layers.0.mixer.out_proj.weight": matrix(hidden, inner),
    }
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    vocab_map = {"<unk>": 0, "hello": 1, "world": 2}
    vocab_map.update({f"tok{i}": i + 3 for i in range(vocab - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab_map, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


@pytest.mark.integration
def test_local_safetensors_compile_reload_runtime(tmp_path: Path) -> None:
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "tiny.aeg"

    compiler = Compiler(
        CompilerConfig(
            # Request one AEG build covering the portable CPU contract and
            # accelerator destination profiles.  Accelerator entries must be
            # reported as portable/plan-only unless real device code exists.
            targets=["cpu_avx512", "cuda_sm90", "rocm_cdna3", "metal_m3", "openvino_npu"],
            overwrite=True,
            calibration_tokens=16,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    )
    compiler.compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.has_weights
    assert package.manifest is not None
    assert package.manifest.kernels.variant_status["cpu_avx512"] == "executable"
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert package.manifest.kernels.variant_status["openvino_npu"] == "plan_only"
    assert "pytorch" in package.manifest.kernels.portable_backends
    assert package.supports_runtime_target("cpu_avx512")
    assert package.supports_runtime_target("cuda_sm90")
    assert package.supports_runtime_target("rocm_cdna3")
    assert package.supports_runtime_target("metal_m3")
    assert not package.supports_runtime_target("openvino_npu")

    archive = tmp_path / "tiny.aeg.tar.gz"
    package.save_as_archive(archive)
    archived = AEGPackage.load_from_archive(archive)
    archived.verify_integrity()
    assert archived.has_weights
    archived_logits, _ = load_engine_from_path(archived.root).forward(
        np.asarray([1], dtype=np.int64)
    )
    assert archived_logits.shape == (1, 32)

    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    packaged_kernels = package.metadata.get("kernel_artifacts", [])
    if packaged_kernels:
        packaged_path = artifact / packaged_kernels[0]["path"]
        assert packaged_path.is_file()
        assert Path(engine.kernels.library_path).resolve() == packaged_path.resolve()

    runtime = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=3))
    response = runtime.generate(str(artifact), prompt="hello world", max_tokens=3, temperature=0.0)
    assert response.usage["completion_tokens"] == 3
    assert response.text
    cached = runtime.generate(str(artifact), prompt="hello world", max_tokens=3, temperature=0.0)
    assert cached.text == response.text
    assert cached.metrics.extra.get("cache_hit") is True

    # Compile-once/use-everywhere contract: disabling the response cache must
    # still reuse the authenticated executable engine for repeated requests,
    # while a fresh Runtime process can reload the same persisted AEG.
    reusable = Runtime(
        RuntimeConfig(
            hf_offline=True,
            default_max_tokens=1,
            enable_semantic_cache=False,
        )
    )
    first_reuse = reusable.generate(str(artifact), prompt="hello", max_tokens=1, temperature=0.0)
    loaded_handle = reusable._loaded_models[str(artifact)]
    second_reuse = reusable.generate(str(artifact), prompt="world", max_tokens=1, temperature=0.0)
    assert first_reuse.text and second_reuse.text
    assert reusable._loaded_models[str(artifact)] is loaded_handle
    reloaded = Runtime(
        RuntimeConfig(hf_offline=True, default_max_tokens=1, enable_semantic_cache=False)
    )
    reloaded_response = reloaded.generate(str(artifact), prompt="reload", max_tokens=1, temperature=0.0)
    assert reloaded_response.text
    slo_runtime = Runtime(
        RuntimeConfig(hf_offline=True, default_max_tokens=2, scheduler="slo_aware")
    )
    slo_response = slo_runtime.generate(
        str(artifact), prompt="hello world", max_tokens=2, temperature=0.0, slo_tier="latency"
    )
    assert slo_response.metrics.extra["slo_tier"] == "latency"
    assert slo_runtime.slo_scheduler.stats.total_processed == 1
    chat_response = slo_runtime.chat(
        str(artifact),
        [{"role": "user", "content": "hello world"}],
        max_tokens=1,
        temperature=0.0,
        slo_deadline_ms=25.0,
    )
    assert chat_response.text
    assert chat_response.metrics.extra["slo_deadline_s"] == pytest.approx(0.025)
    stream_chunks = list(
        slo_runtime.generate_stream(
            str(artifact), "hello", max_tokens=1, temperature=0.0, slo_deadline_ms=25.0
        )
    )
    assert "".join(stream_chunks)
    quant_report = runtime.quantization_report(str(artifact))
    assert quant_report["status"] == "measured"
    assert quant_report["weight_bytes"] > 0
    assert quant_report["memory_mb"] > 0
    assert quant_report["vs_bf16_reduction"].endswith("x")

    from fastapi.testclient import TestClient
    from aether.server.app import create_app

    app = create_app(RuntimeConfig(hf_offline=True, default_max_tokens=2))
    api_response = TestClient(app).post(
        "/v1/generate",
        json={"model": str(artifact), "prompt": "hello world", "max_tokens": 2, "temperature": 0.0},
    )
    assert api_response.status_code == 200, api_response.text
    body = api_response.json()
    assert body["usage"]["completion_tokens"] == 2
    assert body["text"]

    with TestClient(app).stream(
        "POST",
        "/v1/generate",
        json={
            "model": str(artifact),
            "prompt": "hello world",
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": True,
        },
    ) as streamed:
        assert streamed.status_code == 200
        events = [line for line in streamed.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    stream_text = "".join(
        json.loads(line[6:])["text"] for line in events[:-1] if line[6:] != "[DONE]"
    )
    assert stream_text

    with TestClient(app).stream(
        "POST",
        "/v1/chat",
        json={
            "model": str(artifact),
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": True,
        },
    ) as chat_streamed:
        assert chat_streamed.status_code == 200
        chat_events = [line for line in chat_streamed.iter_lines() if line.startswith("data: ")]
    assert chat_events[-1] == "data: [DONE]"
    assert "".join(
        json.loads(line[6:])["text"] for line in chat_events[:-1] if line[6:] != "[DONE]"
    )

    from aether.core.exceptions import BackendError

    weight_blob = package.weight_store().blob_path
    weight_blob.write_bytes(weight_blob.read_bytes() + b"tampered")
    with pytest.raises(BackendError, match="integrity"):
        Runtime(RuntimeConfig(hf_offline=True)).generate(
            str(artifact), prompt="hello", max_tokens=1, temperature=0.0
        )


@pytest.mark.integration
def test_non_qwen_gemma_family_uses_generic_decoder_path(tmp_path: Path) -> None:
    """A non-Qwen family must compile and execute through the same public path."""
    source = tmp_path / "tiny-gemma"
    source.mkdir()
    _write_tiny_llama(
        source,
        architecture_name="GemmaForCausalLM",
        model_type="gemma2",
    )
    artifact = tmp_path / "tiny-gemma.aeg"
    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_avx2"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    )
    compiler.compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest.architecture.family == "gemma_family"
    assert package.manifest.architecture.ffn_type == "GeGLU"
    assert package.manifest.architecture.family != "qwen_family"
    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello world", max_tokens=2, temperature=0.0
    )
    assert response.text


@pytest.mark.integration
def test_gpt2_conv1d_absolute_position_path_is_generic(tmp_path: Path) -> None:
    """GPT-2 layout, LayerNorm, GELU and learned positions must be executable."""
    source = tmp_path / "tiny-gpt2"
    source.mkdir()
    _write_tiny_gpt2(source)
    artifact = tmp_path / "tiny-gpt2.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    architecture = package.manifest.architecture
    assert architecture.family == "gpt_family"
    assert architecture.norm_type == "LayerNorm"
    assert architecture.ffn_type == "GELU"
    assert architecture.position_type == "absolute"
    logits, _ = load_engine_from_path(artifact).forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)


@pytest.mark.integration
def test_t5_encoder_decoder_compiles_and_generates(tmp_path: Path) -> None:
    """T5-family checkpoints use a real encoder-decoder runtime, not decoder fallback."""
    source = tmp_path / "tiny-t5"
    source.mkdir()
    _write_tiny_t5(source)
    artifact = tmp_path / "tiny-t5.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest is not None
    architecture = package.manifest.architecture
    assert architecture.is_encoder_decoder
    assert architecture.encoder_layers == 1
    assert architecture.decoder_layers == 1
    assert architecture.ffn_type == "ReLU"
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([3, 4], dtype=np.int64))
    assert logits.shape == (1, 32)
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello world", max_tokens=2, temperature=0.0
    )
    assert response.text
    pytest.importorskip("torch")
    from aether.runtime.torch_transformer_engine import TorchSeq2SeqAEGEngine

    cpu_logits, _ = load_engine_from_path(artifact).forward(np.asarray([3, 4], dtype=np.int64))
    portable_logits, _ = TorchSeq2SeqAEGEngine(engine, "cpu").forward(np.asarray([3, 4], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.integration
def test_mixtral_moe_compiles_serializes_experts_and_generates(tmp_path: Path) -> None:
    """Mixtral routing must execute from authenticated expert tensors."""
    source = tmp_path / "tiny-mixtral"
    source.mkdir()
    _write_tiny_mixtral(source)
    artifact = tmp_path / "tiny-mixtral.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest is not None
    assert package.manifest.architecture.is_moe
    assert package.manifest.architecture.num_experts == 2
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    store_names = set(package.weight_store().entries)
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert f"layer_0_expert_{expert}_{projection}" in store_names
    assert "layer_0_moe_router" in store_names
    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    assert np.isfinite(logits).all()
    pytest.importorskip("torch")
    from aether.runtime.torch_engine import TorchAEGEngine

    cpu_logits, _ = load_engine_from_path(artifact).forward(np.asarray([1, 2, 3], dtype=np.int64))
    portable_logits, _ = TorchAEGEngine(engine, "cpu").forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello world", max_tokens=2, temperature=0.0
    )
    assert response.text


@pytest.mark.integration
def test_deepseek_style_mla_compiles_and_runs_without_standard_qkv(tmp_path: Path) -> None:
    """MLA must use its compressed projections, not a Qwen/Llama QKV fallback."""
    source = tmp_path / "tiny-mla"
    source.mkdir()
    _write_tiny_mla(source)
    artifact = tmp_path / "tiny-mla.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    architecture = package.manifest.architecture
    assert architecture.attention_type == "MLA"
    assert package.manifest.kernels.variant_status["cpu_avx2"] == "executable"
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    entries = set(package.weight_store().entries)
    assert "layer_0_kv_a_proj" in entries
    assert "layer_0_kv_b_proj" in entries
    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    assert np.isfinite(logits).all()
    generated = engine.generate(np.asarray([1], dtype=np.int64), max_tokens=2, temperature=0.0)
    assert len(generated) == 2
    pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchMLAAEGEngine

    cpu_logits, _ = load_engine_from_path(artifact).forward(np.asarray([1, 2, 3], dtype=np.int64))
    portable_logits, _ = TorchMLAAEGEngine(engine, "cpu").forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.integration
def test_deepseek_style_mla_moe_compiles_and_runs_with_routed_ffn(tmp_path: Path) -> None:
    """The MLA contract also carries routed expert FFNs without dense fallback."""
    source = tmp_path / "tiny-mla-moe"
    source.mkdir()
    _write_tiny_mla(source, moe=True)
    artifact = tmp_path / "tiny-mla-moe.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest.architecture.is_moe
    assert package.manifest.architecture.num_experts == 2
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    names = set(package.weight_store().entries)
    assert "layer_0_moe_router" in names
    assert "layer_0_expert_0_gate_proj" in names
    assert "layer_0_expert_1_down_proj" in names
    engine = load_engine_from_path(artifact)
    cpu_logits, _ = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    assert cpu_logits.shape == (3, 32)
    pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchMLAAEGEngine

    portable_logits, _ = TorchMLAAEGEngine(engine, "cpu").forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.integration
def test_mamba_selective_scan_compiles_and_runs_with_recurrent_state(tmp_path: Path) -> None:
    """Mamba uses the selective-scan state contract, not transformer QKV."""
    source = tmp_path / "tiny-mamba"
    source.mkdir()
    _write_tiny_mamba(source)
    artifact = tmp_path / "tiny-mamba.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest.architecture.ssm_variant == "selective_scan"
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    entries = set(package.weight_store().entries)
    assert "layer_0_ssm_a_log" in entries
    assert "layer_0_ssm_x_proj" in entries
    engine = load_engine_from_path(artifact)
    logits, cache = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    assert cache.length == 2
    generated = engine.generate(np.asarray([], dtype=np.int64), max_tokens=2, cache=cache, temperature=0.0)
    assert len(generated) == 2
    pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchMambaAEGEngine

    cpu_logits, _ = load_engine_from_path(artifact).forward(np.asarray([1, 2, 3], dtype=np.int64))
    portable_logits, _ = TorchMambaAEGEngine(engine, "cpu").forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.integration
def test_generic_registry_decoder_family_compiles_and_runs(tmp_path: Path) -> None:
    """An OLMo-style registry family uses the same capability-driven decoder."""
    source = tmp_path / "tiny-olmo"
    source.mkdir()
    _write_tiny_llama(
        source,
        architecture_name="OlmoForCausalLM",
        model_type="olmo",
    )
    artifact = tmp_path / "tiny-olmo.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest.architecture.family == "generic_decoder_family"
    logits, _ = load_engine_from_path(artifact).forward(np.asarray([1], dtype=np.int64))
    assert logits.shape == (1, 32)


@pytest.mark.integration
def test_unknown_standard_decoder_contract_compiles_and_runs(tmp_path: Path) -> None:
    """A new unregistered causal-LM family must use the generic contract."""
    source = tmp_path / "tiny-acme"
    source.mkdir()
    _write_tiny_llama(
        source,
        architecture_name="AcmeTransformerForCausalLM",
        model_type="acme_transformer",
    )
    artifact = tmp_path / "tiny-acme.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "compiler-cache"),
    )).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    assert package.manifest.architecture.family == "generic_decoder_family"
    logits, _ = load_engine_from_path(artifact).forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)


@pytest.mark.integration
def test_local_bitnet_sub2bit_aeg_roundtrip(tmp_path: Path) -> None:
    """Pass 19 BitNet weights must be packed, reloaded, and executed on CPU."""
    source = tmp_path / "tiny-llama-bitnet"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "tiny-bitnet.aeg"

    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
            enable_sub2bit=True,
            sub2bit_mode="ternary",
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.manifest.format_version == "AEG/3.0"
    assert "sub2bit_quantization" in package.metadata["optimizer_passes"]
    manifest = json.loads(
        (artifact / "quantization" / "sub2bit_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["runtime_codec"] == "TERNARY"
    assert manifest["weight_reconstruction"]["elements"] > 0
    assert any(value == "TERNARY" for value in package.get_precision_map().values())

    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert response.text
    assert response.usage["completion_tokens"] == 2


@pytest.mark.integration
def test_local_aeg_compiled_lora_is_consumed_by_runtime(tmp_path: Path) -> None:
    """Pass 21 artifacts must affect the real CPU transformer on reload."""
    source = tmp_path / "tiny-llama-lora"
    source.mkdir()
    _write_tiny_llama(source)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 2}), encoding="utf-8"
    )
    safetensors = pytest.importorskip("safetensors.numpy")
    safetensors.save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((2, 16), dtype=np.float32),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones((16, 2), dtype=np.float32),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    artifact = tmp_path / "tiny-lora.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
            enable_advanced_peft=True,
            peft_adapter_paths=[str(adapter)],
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert "advanced_peft_compilation" in package.metadata.get("optimizer_passes", [])
    base = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    adapted = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0, adapter_id="adapter"
    )
    assert base.text
    assert adapted.text
    assert adapted.metrics.extra["lora_adapter"]["adapter_id"] == "adapter"


@pytest.mark.integration
def test_local_aeg_consumes_persisted_sparse_attention_plan(tmp_path: Path) -> None:
    """An enabled Pass 8 plan must reach the executable CPU attention path."""
    source = tmp_path / "tiny-long-context"
    source.mkdir()
    _write_tiny_llama(source, max_position_embeddings=65536)
    artifact = tmp_path / "tiny-sparse.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    ).compile(str(source), output_path=artifact)
    package = load_aeg_package(artifact)
    plan = package.metadata["attention_head_patterns"]
    assert plan["enabled"] is True
    engine = load_engine_from_path(artifact)
    assert engine.sparse_attention_plan == plan
    logits, _ = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    assert logits.shape == (3, 32)


@pytest.mark.integration
def test_local_pytorch_checkpoint_compile_reload_runtime(tmp_path: Path) -> None:
    """A torch.save state dict must follow the same real AEG path as SafeTensors."""
    source = tmp_path / "tiny-llama-pytorch"
    source.mkdir()
    _write_tiny_llama(source, weights_format="pytorch")
    artifact = tmp_path / "tiny-pytorch.aeg"

    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.has_weights
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert response.text
    assert response.usage["completion_tokens"] == 2


@pytest.mark.integration
def test_local_transformer_family_aliases_compile_reload_and_generate(tmp_path: Path) -> None:
    """Common HF text-family configs must use the real generic CPU path."""
    families = (
        ("qwen2", "Qwen2ForCausalLM", "qwen_family"),
        ("gemma", "GemmaForCausalLM", "gemma_family"),
        ("mistral", "MistralForCausalLM", "mistral_family"),
    )
    for model_type, architecture_name, expected_family in families:
        source = tmp_path / model_type
        source.mkdir()
        _write_tiny_llama(
            source,
            architecture_name=architecture_name,
            model_type=model_type,
        )
        artifact = tmp_path / f"{model_type}.aeg"
        Compiler(
            CompilerConfig(
                targets=["cpu_avx512"],
                overwrite=True,
                calibration_tokens=8,
                cache_dir=str(tmp_path / f"cache-{model_type}"),
            )
        ).compile(str(source), output_path=artifact)
        package = load_aeg_package(artifact)
        assert package.manifest is not None
        assert package.manifest.architecture.family == expected_family
        response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
            str(artifact), prompt="hello", max_tokens=2, temperature=0.0
        )
        assert response.text


@pytest.mark.integration
def test_local_declared_mtp_checkpoint_reaches_runtime_speculation(tmp_path: Path) -> None:
    """A declared DeepSeek-style MTP checkpoint must compile and execute R1."""
    source = tmp_path / "tiny-deepseek-mtp"
    source.mkdir()
    _write_tiny_llama(
        source,
        architecture_name="DeepseekForCausalLM",
        model_type="deepseek",
        mtp_heads=2,
    )
    artifact = tmp_path / "tiny-mtp.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_mtp_head=True,
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    assert package.manifest is not None
    assert package.manifest.architecture.mtp_heads == 2
    assert (package.root / "speculation" / "mtp_config.json").is_file()
    runtime = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=3))
    response = runtime.generate(str(artifact), prompt="hello", max_tokens=3, temperature=0.0)
    speculative = response.metrics.extra.get("speculative")
    assert isinstance(speculative, dict)
    assert int(speculative["draft_tokens"]) > 0


@pytest.mark.integration
def test_cli_serve_exposes_real_tcp_api_for_local_aeg(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """The documented ``aether serve`` process must serve a real local AEG."""
    import httpx

    artifact = tmp_path / "served.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=artifact)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    environment = os.environ.copy()
    environment["AETHER_HF_OFFLINE"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether.cli",
            "serve",
            str(artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20.0
        last_error = "server did not become ready"
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout is not None else ""
                    raise AssertionError(f"aether serve exited early: {output}")
                try:
                    health = client.get(f"{base_url}/health")
                    if health.status_code == 200:
                        break
                    last_error = f"health returned HTTP {health.status_code}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                time.sleep(0.1)
            else:
                output = process.stdout.read() if process.stdout is not None else ""
                raise AssertionError(f"{last_error}; server output: {output}")

            response = client.post(
                f"{base_url}/v1/generate",
                json={
                    "model": str(artifact),
                    "prompt": "hello",
                    "max_tokens": 2,
                    "temperature": 0.0,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["text"]
        assert body["usage"]["completion_tokens"] == 2
    finally:
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=5)
            except Exception:
                pass


@pytest.mark.integration
def test_agentic_session_reuses_real_cpu_aeg_kv_prefix(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """Agentic turns reuse only an exact token prefix from the CPU AEG cache."""
    artifact = tmp_path / "agentic.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=artifact)

    runtime = Runtime(
        RuntimeConfig(
            hf_offline=True,
            default_max_tokens=2,
            enable_semantic_cache=False,
        )
    )

    async def exercise() -> tuple[dict[str, object], dict[str, object]]:
        async with runtime.agentic_session(str(artifact), system="Be concise") as session:
            first = await session.generate("hello", temperature=0.0)
            second = await session.generate("world", temperature=0.0)
            return first.metrics.to_dict(), second.metrics.to_dict()

    first_metrics, second_metrics = asyncio.run(exercise())
    assert first_metrics.get("kv_reuse") is False
    assert second_metrics.get("kv_reuse") is True
    assert int(second_metrics.get("kv_reused_tokens", 0)) > 0

    backend = runtime._loaded_backends[str(artifact)]  # noqa: SLF001 - cleanup contract
    handle = backend._models[str(artifact)]  # noqa: SLF001 - cleanup contract
    assert handle.session_caches == {}


@pytest.mark.integration
def test_multi_agent_session_reuses_real_shared_cpu_prefix_kv(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """R2 publishes one real prefix cache and clones it for divergent agents."""
    artifact = tmp_path / "multi-agent.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=artifact)
    runtime = Runtime(
        RuntimeConfig(
            hf_offline=True,
            default_max_tokens=2,
            enable_semantic_cache=False,
        )
    )

    async def exercise() -> tuple[dict[str, object], dict[str, object]]:
        async with runtime.multi_agent_session(
            [str(artifact)],
            coordination="relay",
            shared_prefix="Shared system context",
        ) as session:
            first_agent = await session.spawn_agent(
                str(artifact), context="Shared system context"
            )
            second_agent = await session.spawn_agent(
                str(artifact), context="Shared system context"
            )
            first = await first_agent.generate("first branch", temperature=0.0)
            second = await second_agent.generate("second branch", temperature=0.0)
            return first.metrics.to_dict(), second.metrics.to_dict()

    first_metrics, second_metrics = asyncio.run(exercise())
    assert first_metrics["multi_agent_kv_reuse"] is False
    assert second_metrics["multi_agent_kv_reuse"] is True
    assert int(second_metrics["multi_agent_kv_reused_tokens"]) > 0


@pytest.mark.integration
def test_enabled_optimizer_artifacts_are_persisted(tmp_path: Path) -> None:
    """Enabled passes must leave files in the saved AEG, not only reports."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "features.aeg"
    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_grammar_constraint=True,
            grammar_schema='root ::= "hello"',
            enable_ttt=True,
            enable_green_energy=True,
            enable_tee=True,
            tee_backend="nvidia_cc",
            enable_mdlm_drafter=True,
        )
    )
    package = compiler.compile(str(source), output_path=artifact)
    assert package.manifest is not None
    # Pass 18 is deliberately fail-closed until a real MDLM drafter weight
    # bundle is supplied. Grammar/TTT/green/TEE are genuine v4 payloads, so
    # this artifact is AEG/2.0 rather than a misleading v5 claim.
    assert package.manifest.format_version == "AEG/2.0"
    expected = [
        "grammar/fsm.bin",
        "ttt/fast_weight_config.json",
        "metadata/green_profile.json",
    ]
    for relative in expected:
        assert (package.root / relative).is_file(), relative
    grammar_config = json.loads(
        (package.root / "grammar" / "fsm_config.json").read_text(encoding="utf-8")
    )
    assert grammar_config["tokenizer_aware"] is True
    assert grammar_config["tokenizer_fingerprint"]
    package.verify_integrity()
    reloaded = AEGPackage(package.root).load()
    assert reloaded.format_version == "AEG/2.0"

    runtime = Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False))
    runtime._load_model(str(package.root))  # noqa: SLF001 - verify persisted layer reachability
    assert runtime.grammar_engine is not None
    assert runtime.ttt_engine is not None
    assert runtime.green_power_manager is not None
    assert runtime.tee_manager is None
    constrained = runtime.generate_constrained(
        str(package.root),
        prompt="world",
        grammar='root ::= "hello"',
        max_tokens=1,
        temperature=0.0,
    )
    assert constrained.text == "hello"
    constrained_stream = "".join(
        runtime.generate_constrained_stream(
            str(package.root),
            prompt="world",
            grammar='root ::= "hello"',
            max_tokens=1,
            temperature=0.0,
        )
    )
    assert constrained_stream == "hello"
    chat_stream = "".join(
        runtime.generate_stream(
            str(package.root),
            messages=[{"role": "user", "content": "world"}],
            max_tokens=1,
            temperature=0.0,
        )
    )
    assert chat_stream
    adapted = runtime.generate(str(package.root), "world", max_tokens=1, temperature=0.0)
    assert "ttt_adaptation_loss" in adapted.metrics.extra
    assert adapted.metrics.extra["energy_mj"] > 0.0
    assert adapted.metrics.extra["carbon_gco2"] > 0.0
    assert adapted.metrics.extra["energy_source"] == "tdp_duration_estimate"
    assert runtime.green_power_manager.stats.total_requests == 2

    from fastapi.testclient import TestClient
    from aether.server.app import create_app

    client = TestClient(create_app(RuntimeConfig(hf_offline=True)))
    constrained_api = client.post(
        "/v1/generate",
        json={
            "model": str(package.root),
            "prompt": "world",
            "grammar": 'root ::= "hello"',
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    assert constrained_api.status_code == 200, constrained_api.text
    assert constrained_api.json()["text"] == "hello"
    with client.stream(
        "POST",
        "/v1/chat",
        json={
            "model": str(package.root),
            "messages": [{"role": "user", "content": "world"}],
            "grammar": 'root ::= "hello"',
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        },
    ) as constrained_chat:
        assert constrained_chat.status_code == 200, constrained_chat.text
        constrained_events = [
            line for line in constrained_chat.iter_lines() if line.startswith("data: ")
        ]
    assert constrained_events[-1] == "data: [DONE]"
    assert "hello" == "".join(
        json.loads(line[6:])["text"]
        for line in constrained_events[:-1]
        if line[6:] != "[DONE]"
    )


@pytest.mark.integration
def test_local_aeg_grpc_generate_and_stream(tmp_path: Path) -> None:
    """Exercise the gRPC transport against a real compiled CPU AEG."""
    grpc = pytest.importorskip("grpc")
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "grpc.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)

    from aether.server.grpc_service import AetherGrpcClient, start_grpc_server

    runtime = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2, enable_semantic_cache=False))
    server = start_grpc_server(runtime, port=0, auth_token="test-token")
    client = AetherGrpcClient(f"127.0.0.1:{server.aether_port}", auth_token="test-token")
    try:
        assert client.health()["status"] == "ok"
        request = {"model_id": str(artifact), "prompt": "hello", "max_tokens": 2, "temperature": 0.0}
        response = client.generate(request)
        assert response["text"]
        assert response["completion_tokens"] == 2
        assert isinstance(response["metrics"], dict)
        assert all(not value.__class__.__module__.startswith("google.protobuf") for value in response["metrics"].values())
        chunks = list(client.generate_stream(request))
        assert chunks and chunks[-1]["final"] is True
        assert any(not chunk["final"] for chunk in chunks[:-1])
        assert "".join(chunk["text"] for chunk in chunks) == response["text"]

        unauthorized = AetherGrpcClient(f"127.0.0.1:{server.aether_port}", auth_token="wrong-token")
        try:
            with pytest.raises(grpc.RpcError) as error:
                unauthorized.health()
            assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        finally:
            unauthorized.close()

    finally:
        client.close()
        server.stop(0)


@pytest.mark.integration
def test_local_aeg_evaluation_gate_measures_runtime_output(tmp_path: Path) -> None:
    """The configured evaluator must invoke the compiled model and gate it."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "eval.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)

    dataset = tmp_path / "mmlu.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "expected": "__this_answer_cannot_match__",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from aether.observability.ci_pipeline import JsonlBenchmarkEvaluator

    runtime = Runtime(
        RuntimeConfig(hf_offline=True, default_max_tokens=2, enable_semantic_cache=False)
    )
    evaluator = JsonlBenchmarkEvaluator(
        {"mmlu": dataset},
        lambda *, prompt, benchmark, max_tokens: runtime.generate(
            str(artifact), prompt=prompt, max_tokens=max_tokens, temperature=0.0
        ).text,
        max_tokens=2,
    )
    report = runtime.eval_gate(
        model=str(artifact),
        benchmarks=["mmlu"],
        max_regression=0.02,
        evaluator=evaluator,
        baselines={"mmlu": 1.0},
    )
    assert report.status == "failed"
    assert report.passed is False
    assert report["benchmarks"][0]["num_total"] == 1
    assert report["benchmarks"][0]["metadata"]["evaluator"] == "jsonl_exact_match"


@pytest.mark.integration
def test_compiler_rejects_and_runtime_blocks_failed_eval_artifact(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """A failed measured gate cannot be returned or loaded for deployment."""
    from aether.core.exceptions import BackendError, CompilationError

    artifact = tmp_path / "rejected.aeg"

    def evaluator(benchmark, spec):
        return {
            "benchmark": benchmark,
            "score": 0.0,
            "num_correct": 0,
            "num_total": 1,
            "latency_ms": 1.0,
        }

    with pytest.raises(CompilationError, match="Evaluation gate failed"):
        Compiler(
            CompilerConfig(
                targets=["cpu_avx512"],
                overwrite=True,
                calibration_tokens=8,
                cache_dir=str(tmp_path / "cache"),
            )
        ).compile(
            str(tiny_local_safetensors_model),
            output_path=artifact,
            evaluation_evaluator=evaluator,
            eval_benchmarks=["mmlu"],
            eval_baselines={"mmlu": 1.0},
        )

    report_path = artifact / "observability" / "eval_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    with pytest.raises(BackendError, match="rejected by its persisted evaluation gate"):
        Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False)).generate(
            str(artifact), "hello", max_tokens=1, temperature=0.0
        )


@pytest.mark.integration
def test_runtime_merge_writes_and_reloads_real_aeg(tmp_path: Path) -> None:
    """Pass 12 must produce a loadable artifact, not a recorded config."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    base_path = tmp_path / "base.aeg"
    task_path = tmp_path / "task.aeg"
    compiler_config = CompilerConfig(
        targets=["cpu_avx512"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "cache"),
    )
    Compiler(compiler_config).compile(str(source), output_path=base_path)
    Compiler(compiler_config).compile(str(source), output_path=task_path)

    # Make the source a genuine task vector while preserving its AEG format.
    task = AEGPackage(task_path).load()
    from aether.quantization.formats import quantize_tensor

    task.weights = {}
    for name, tensor in task.weight_store().dequantize_all().items():
        task.weights[name] = quantize_tensor((tensor + 0.01).astype("float32"), task.weight_store().entries[name].precision)
    task.save()

    runtime = Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False))
    result = runtime.merge(
        str(base_path),
        task_vectors=[{"name": "task", "path": str(task_path), "coefficient": 1.0}],
    )
    merged_path = Path(result["output_model"])
    assert result["status"] == "merged"
    assert merged_path.is_dir()
    merged = load_aeg_package(merged_path)
    merged.verify_integrity()
    assert merged.has_weights
    task_payload = merged.metadata["task_vectors"]
    assert task_payload["format"] == "aether_task_vectors_v1"
    assert task_payload["vectors"][0]["name"] == "task"
    assert (merged_path / task_payload["vectors"][0]["path"]).is_file()

    response = runtime.generate(str(merged_path), prompt="hello", max_tokens=2, temperature=0.0)
    assert response.text
    runtime.set_task_weights(str(merged_path), task=1.0)
    reweighted = runtime.generate(str(merged_path), prompt="hello", max_tokens=2, temperature=0.0)
    assert reweighted.text
    assert reweighted.metrics.extra["task_reweighting"]["vectors"] == ["task"]


@pytest.mark.integration
def test_semantic_kv_plan_survives_aeg_reload_and_changes_real_cache(tmp_path: Path) -> None:
    """Pass 14 must be consumed by the executable CPU KV cache after reload."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "semantic-kv.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_semantic_kv=True,
            semantic_kv_strategy="chunk",
            semantic_kv_compression_ratio=0.5,
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    plan = package.metadata.get("kv_compression_plan")
    assert isinstance(plan, dict)
    assert plan["format"] == "aether_kv_compression_v1"
    assert "semantic_kv_compression" in package.metadata["optimizer_passes"]

    engine = load_engine_from_path(artifact)
    _, cache = engine.forward(np.asarray([1, 2, 3, 4], dtype=np.int64))
    assert cache.length == 4
    assert cache.positions[0] is not None
    assert int(cache.positions[0][-1]) == 3
    logits, cache = engine.forward(np.asarray([5], dtype=np.int64), cache)
    assert logits.shape == (1, 32)
    assert cache.length == 5
    assert cache.positions[0] is not None
    assert int(cache.positions[0][-1]) == 4


@pytest.mark.integration
def test_cross_layer_kv_plan_survives_aeg_reload_and_aliases_cpu_cache(tmp_path: Path) -> None:
    """Pass 15 must materialize real pointer sharing after AEG reload."""
    source = tmp_path / "tiny-llama-2l"
    source.mkdir()
    _write_tiny_llama(source, num_layers=2)
    artifact = tmp_path / "cross-layer-kv.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_cross_layer_kv=True,
            cross_layer_kv_share_threshold=0.0,
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    plan = package.metadata.get("cross_layer_kv_plan")
    assert isinstance(plan, dict)
    assert plan["format"] == "aether_cross_layer_kv_v1"
    assert plan["sharing_groups"]
    engine = load_engine_from_path(artifact)
    _, cache = engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    assert cache.keys[0] is cache.keys[1]
    assert cache.values[0] is cache.values[1]
    assert cache.positions[0] is cache.positions[1]


@pytest.mark.integration
def test_mdlm_bundle_compiles_into_aeg_and_reloads_r9(tmp_path: Path) -> None:
    """Pass 18 must persist and execute a real trained-shaped CPU head."""
    source = tmp_path / "tiny-llama-mdlm"
    source.mkdir()
    _write_tiny_llama(source)
    rng = np.random.default_rng(31)
    weights = {
        "token_embedding": rng.normal(size=(32, 4)).astype("float32"),
        "context_projection": rng.normal(size=(16, 4)).astype("float32"),
        "output_projection": rng.normal(size=(4, 32)).astype("float32"),
        "output_bias": np.zeros(32, dtype="float32"),
        "time_embedding": rng.normal(size=(7, 4)).astype("float32"),
    }
    bundle = tmp_path / "mdlm-head.npz"
    np.savez(bundle, **weights)
    artifact = tmp_path / "mdlm.aeg"
    package = Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_mdlm_drafter=True,
            mdlm_drafter_weights_path=str(bundle),
            mdlm_drafter_steps=6,
            mdlm_draft_block_size=3,
        )
    ).compile(str(source), output_path=artifact)
    assert package.manifest.format_version == "AEG/3.0"
    assert "mdlm_drafter_compilation" in package.metadata["optimizer_passes"]
    package.verify_integrity()
    assert (artifact / "graph" / "mdlm_draft_head.npz").is_file()

    runtime = Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False))
    runtime._load_model(str(artifact))  # noqa: SLF001 - verify restart path
    engine = runtime._diffusion_engine  # noqa: SLF001 - inspect loaded R9 layer
    assert engine is not None and engine.is_ready()
    engine.mask_token_id = 99
    draft = engine.draft(np.ones((2, 16), dtype="float32"), position=0)
    assert len(draft.tokens) == 3
    assert all(0 <= token < 32 for token in draft.tokens)
