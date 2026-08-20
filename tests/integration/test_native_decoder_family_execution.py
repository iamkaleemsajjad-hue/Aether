"""Native checkpoint-name integration tests for non-Qwen decoder families.

The fixtures use real SafeTensors payloads and the public Compiler/Runtime
path. They are intentionally tiny, but preserve layouts used by GPT-2, GPT-J,
GPT-NeoX, Falcon, BLOOM and OPT so coverage tests binding rather than only
architecture-name detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import load_aeg_package


def _write_tokenizer(path: Path, vocab_size: int) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


def _write_family(path: Path, family: str) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    path.mkdir()
    vocab, hidden, intermediate, heads, kv_heads = 32, 16, 32, 2, 2
    rng = np.random.default_rng(abs(hash(family)) % (2**32))

    def matrix(shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(size=shape).astype("float32")

    def vector(size: int, value: float = 1.0) -> np.ndarray:
        return np.full(size, value, dtype="float32")

    common = {
        "num_hidden_layers": 1,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "vocab_size": vocab,
        "max_position_embeddings": 32,
        "torch_dtype": "float32",
    }
    tensors: dict[str, np.ndarray]
    if family == "gpt2":
        config = {
            **common,
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "n_layer": 1,
            "n_embd": hidden,
            "n_head": heads,
            "n_positions": 32,
            "norm_type": "LayerNorm",
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "position_type": "absolute",
        }
        tensors = {
            "transformer.wte.weight": matrix((vocab, hidden)),
            "transformer.wpe.weight": matrix((32, hidden)),
            "transformer.ln_f.weight": vector(hidden),
            "transformer.ln_f.bias": np.zeros(hidden, dtype="float32"),
            "lm_head.weight": matrix((vocab, hidden)),
            "transformer.h.0.ln_1.weight": vector(hidden),
            "transformer.h.0.ln_1.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.attn.c_attn.weight": matrix((hidden, 3 * hidden)),
            "transformer.h.0.attn.c_attn.bias": np.zeros(3 * hidden, dtype="float32"),
            "transformer.h.0.attn.c_proj.weight": matrix((hidden, hidden)),
            "transformer.h.0.attn.c_proj.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.ln_2.weight": vector(hidden),
            "transformer.h.0.ln_2.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.mlp.c_fc.weight": matrix((hidden, intermediate)),
            "transformer.h.0.mlp.c_fc.bias": np.zeros(intermediate, dtype="float32"),
            "transformer.h.0.mlp.c_proj.weight": matrix((intermediate, hidden)),
            "transformer.h.0.mlp.c_proj.bias": np.zeros(hidden, dtype="float32"),
        }
    elif family == "gpt_j":
        config = {
            **common,
            "architectures": ["GPTJForCausalLM"],
            "model_type": "gptj",
            "norm_type": "LayerNorm",
            "hidden_act": "gelu",
            "layer_norm_epsilon": 1e-5,
            "position_type": "RoPE",
        }
        tensors = {
            "transformer.wte.weight": matrix((vocab, hidden)),
            "transformer.ln_f.weight": vector(hidden),
            "transformer.ln_f.bias": np.zeros(hidden, dtype="float32"),
            "lm_head.weight": matrix((vocab, hidden)),
            "transformer.h.0.ln_1.weight": vector(hidden),
            "transformer.h.0.ln_1.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.attn.q_proj.weight": matrix((hidden, hidden)),
            "transformer.h.0.attn.k_proj.weight": matrix((hidden, hidden)),
            "transformer.h.0.attn.v_proj.weight": matrix((hidden, hidden)),
            "transformer.h.0.attn.out_proj.weight": matrix((hidden, hidden)),
            "transformer.h.0.mlp.fc_in.weight": matrix((intermediate, hidden)),
            "transformer.h.0.mlp.fc_out.weight": matrix((hidden, intermediate)),
        }
    elif family == "gpt_neox":
        config = {
            **common,
            "architectures": ["GPTNeoXForCausalLM"],
            "model_type": "gpt_neox",
            "layer_norm_eps": 1e-5,
        }
        tensors = {
            "gpt_neox.embed_in.weight": matrix((vocab, hidden)),
            "gpt_neox.final_layer_norm.weight": vector(hidden),
            "embed_out.weight": matrix((vocab, hidden)),
            "gpt_neox.layers.0.input_layernorm.weight": vector(hidden),
            "gpt_neox.layers.0.attention.query_key_value.weight": matrix((3 * hidden, hidden)),
            "gpt_neox.layers.0.attention.dense.weight": matrix((hidden, hidden)),
            "gpt_neox.layers.0.post_attention_layernorm.weight": vector(hidden),
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight": matrix((intermediate, hidden)),
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight": matrix((hidden, intermediate)),
        }
    elif family == "falcon":
        kv_heads = 1
        config = {
            **common,
            "architectures": ["FalconForCausalLM"],
            "model_type": "falcon",
            "num_kv_heads": kv_heads,
            "num_key_value_heads": kv_heads,
            "norm_type": "LayerNorm",
            "hidden_act": "gelu",
            "layer_norm_epsilon": 1e-5,
        }
        qkv_width = (heads + 2 * kv_heads) * (hidden // heads)
        tensors = {
            "transformer.word_embeddings.weight": matrix((vocab, hidden)),
            "transformer.ln_f.weight": vector(hidden),
            "lm_head.weight": matrix((vocab, hidden)),
            "transformer.h.0.input_layernorm.weight": vector(hidden),
            "transformer.h.0.self_attention.query_key_value.weight": matrix((qkv_width, hidden)),
            "transformer.h.0.self_attention.query_key_value.bias": np.zeros(qkv_width, dtype="float32"),
            "transformer.h.0.self_attention.dense.weight": matrix((hidden, hidden)),
            "transformer.h.0.ln_mlp.weight": vector(hidden),
            "transformer.h.0.mlp.dense_h_to_4h.weight": matrix((intermediate, hidden)),
            "transformer.h.0.mlp.dense_4h_to_h.weight": matrix((hidden, intermediate)),
        }
    elif family == "bloom":
        config = {
            **common,
            "architectures": ["BloomForCausalLM"],
            "model_type": "bloom",
            "alibi": True,
            "norm_type": "LayerNorm",
            "hidden_act": "gelu",
            "layer_norm_epsilon": 1e-5,
        }
        tensors = {
            "transformer.word_embeddings.weight": matrix((vocab, hidden)),
            "transformer.word_embeddings_layernorm.weight": vector(hidden),
            "transformer.word_embeddings_layernorm.bias": np.zeros(hidden, dtype="float32"),
            "transformer.ln_f.weight": vector(hidden),
            "transformer.ln_f.bias": np.zeros(hidden, dtype="float32"),
            "lm_head.weight": matrix((vocab, hidden)),
            "transformer.h.0.input_layernorm.weight": vector(hidden),
            "transformer.h.0.input_layernorm.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.self_attention.query_key_value.weight": matrix((3 * hidden, hidden)),
            "transformer.h.0.self_attention.query_key_value.bias": np.zeros(3 * hidden, dtype="float32"),
            "transformer.h.0.self_attention.dense.weight": matrix((hidden, hidden)),
            "transformer.h.0.post_attention_layernorm.weight": vector(hidden),
            "transformer.h.0.post_attention_layernorm.bias": np.zeros(hidden, dtype="float32"),
            "transformer.h.0.mlp.dense_h_to_4h.weight": matrix((intermediate, hidden)),
            "transformer.h.0.mlp.dense_4h_to_h.weight": matrix((hidden, intermediate)),
        }
    elif family == "opt":
        config = {
            **common,
            "architectures": ["OPTForCausalLM"],
            "model_type": "opt",
            "norm_type": "LayerNorm",
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
        }
        tensors = {
            "model.decoder.embed_tokens.weight": matrix((vocab, hidden)),
            "model.decoder.embed_positions.weight": matrix((32, hidden)),
            "model.decoder.final_layer_norm.weight": vector(hidden),
            "lm_head.weight": matrix((vocab, hidden)),
            "model.decoder.layers.0.self_attn_layer_norm.weight": vector(hidden),
            "model.decoder.layers.0.self_attn.q_proj.weight": matrix((hidden, hidden)),
            "model.decoder.layers.0.self_attn.k_proj.weight": matrix((hidden, hidden)),
            "model.decoder.layers.0.self_attn.v_proj.weight": matrix((hidden, hidden)),
            "model.decoder.layers.0.self_attn.out_proj.weight": matrix((hidden, hidden)),
            "model.decoder.layers.0.final_layer_norm.weight": vector(hidden),
            "model.decoder.layers.0.fc1.weight": matrix((intermediate, hidden)),
            "model.decoder.layers.0.fc2.weight": matrix((hidden, intermediate)),
        }
    elif family == "phi":
        config = {
            **common,
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
        }
        tensors = {
            "model.embed_tokens.weight": matrix((vocab, hidden)),
            "model.norm.weight": vector(hidden),
            "lm_head.weight": matrix((vocab, hidden)),
            "model.layers.0.input_layernorm.weight": vector(hidden),
            "model.layers.0.self_attn.qkv_proj.weight": matrix((3 * hidden, hidden)),
            "model.layers.0.self_attn.o_proj.weight": matrix((hidden, hidden)),
            "model.layers.0.post_attention_layernorm.weight": vector(hidden),
            "model.layers.0.mlp.gate_up_proj.weight": matrix((2 * intermediate, hidden)),
            "model.layers.0.mlp.down_proj.weight": matrix((hidden, intermediate)),
        }
    elif family == "internlm":
        config = {
            **common,
            "architectures": ["InternLM2ForCausalLM"],
            "model_type": "internlm2",
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
        }
        tensors = {
            "tok_embeddings.weight": matrix((vocab, hidden)),
            "norm.weight": vector(hidden),
            "output.weight": matrix((vocab, hidden)),
            "layers.0.attention_norm.weight": vector(hidden),
            "layers.0.attention.wqkv.weight": matrix((3 * hidden, hidden)),
            "layers.0.attention.wo.weight": matrix((hidden, hidden)),
            "layers.0.ffn_norm.weight": vector(hidden),
            "layers.0.feed_forward.w1.weight": matrix((intermediate, hidden)),
            "layers.0.feed_forward.w3.weight": matrix((intermediate, hidden)),
            "layers.0.feed_forward.w2.weight": matrix((hidden, intermediate)),
        }
    elif family == "mpt":
        config = {
            **common,
            "architectures": ["MptForCausalLM"],
            "model_type": "mpt",
            "norm_type": "LayerNorm",
            "hidden_act": "gelu",
            "layer_norm_epsilon": 1e-5,
        }
        tensors = {
            "transformer.wte.weight": matrix((vocab, hidden)),
            "transformer.norm_f.weight": vector(hidden),
            "lm_head.weight": matrix((vocab, hidden)),
            "transformer.blocks.0.norm_1.weight": vector(hidden),
            "transformer.blocks.0.attn.Wqkv.weight": matrix((3 * hidden, hidden)),
            "transformer.blocks.0.attn.out_proj.weight": matrix((hidden, hidden)),
            "transformer.blocks.0.norm_2.weight": vector(hidden),
            "transformer.blocks.0.ffn.up_proj.weight": matrix((intermediate, hidden)),
            "transformer.blocks.0.ffn.down_proj.weight": matrix((hidden, intermediate)),
        }
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(family)

    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors.save_file(tensors, str(path / "model.safetensors"))
    _write_tokenizer(path, vocab)


@pytest.mark.integration
@pytest.mark.parametrize(
    "family", ["gpt2", "gpt_j", "gpt_neox", "falcon", "bloom", "opt", "phi", "internlm", "mpt"]
)
def test_native_decoder_family_compiles_and_generates(tmp_path: Path, family: str) -> None:
    source = tmp_path / family
    _write_family(source, family)
    artifact = tmp_path / f"{family}.aeg"

    Compiler(
        CompilerConfig(
            targets=["cpu_avx2"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    assert package.has_weights
    assert package.manifest is not None
    assert package.manifest.architecture.family != "qwen_family"
    stored_names = set(package.weight_store().entries)
    if family in {"falcon", "bloom"}:
        assert "layer_0_q_proj_bias" in stored_names
        assert "layer_0_k_proj_bias" in stored_names
        assert "layer_0_v_proj_bias" in stored_names
    if family == "bloom":
        assert "embedding_norm_bias" in stored_names
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    # A valid decoder may select the tokenizer's unknown/special token for a
    # random tiny fixture; the execution contract is the generated token
    # count, not non-empty detokenized prose.
    assert response.usage["completion_tokens"] == 2
