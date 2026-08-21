"""Real SafeTensors compile/run coverage for the RWKV recurrent contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import load_aeg_package
from aether.runtime.aeg_loader import load_engine_from_path


@pytest.mark.integration
def test_rwkv_time_mix_compiles_serializes_and_generates(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    source = tmp_path / "rwkv"
    source.mkdir()
    vocab, hidden, intermediate = 16, 8, 32
    rng = np.random.default_rng(41)

    def matrix(shape: tuple[int, ...]) -> np.ndarray:
        return (rng.normal(size=shape) * 0.04).astype("float32")

    def vector(size: int, value: float = 0.5) -> np.ndarray:
        return np.full(size, value, dtype="float32")

    vocab_map = {"<unk>": 0, "hello": 1}
    vocab_map.update({f"tok{i}": i + 2 for i in range(vocab - 2)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab_map, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(source / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(tokenizer_file=str(source / "tokenizer.json"), unk_token="<unk>").save_pretrained(str(source))

    config = {
        "architectures": ["RwkvForCausalLM"], "model_type": "rwkv",
        "num_hidden_layers": 1, "hidden_size": hidden, "intermediate_size": intermediate,
        "num_attention_heads": 1, "vocab_size": vocab, "context_length": 64,
        "layer_norm_epsilon": 1e-5, "tie_word_embeddings": False,
    }
    tensors = {
        "emb.weight": matrix((vocab, hidden)), "ln_out.weight": vector(hidden),
        "head.weight": matrix((vocab, hidden)),
        "blocks.0.ln1.weight": vector(hidden), "blocks.0.ln2.weight": vector(hidden),
        "blocks.0.att.time_decay": np.full(hidden, -0.2, dtype="float32"),
        "blocks.0.att.time_first": np.zeros(hidden, dtype="float32"),
        "blocks.0.att.time_mix_k": vector(hidden), "blocks.0.att.time_mix_v": vector(hidden),
        "blocks.0.att.time_mix_r": vector(hidden),
        "blocks.0.att.key.weight": matrix((hidden, hidden)),
        "blocks.0.att.value.weight": matrix((hidden, hidden)),
        "blocks.0.att.receptance.weight": matrix((hidden, hidden)),
        "blocks.0.att.output.weight": matrix((hidden, hidden)),
        "blocks.0.ffn.time_mix_k": vector(hidden), "blocks.0.ffn.time_mix_r": vector(hidden),
        "blocks.0.ffn.key.weight": matrix((intermediate, hidden)),
        "blocks.0.ffn.value.weight": matrix((hidden, intermediate)),
        "blocks.0.ffn.receptance.weight": matrix((hidden, hidden)),
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    safetensors.save_file(tensors, str(source / "model.safetensors"))

    artifact = tmp_path / "rwkv.aeg"
    Compiler(CompilerConfig(
        targets=["cpu_avx2", "cuda_sm90", "rocm_cdna3", "metal_m3"],
        overwrite=True, calibration_tokens=4, cache_dir=str(tmp_path / "cache"),
    )).compile(
        str(source), output_path=artifact
    )
    package = load_aeg_package(artifact)
    assert package.manifest is not None
    assert package.manifest.architecture.ssm_variant == "rwkv_time_mix"
    assert package.manifest.kernels.variant_status["cuda_sm90"] == "portable"
    assert package.manifest.kernels.variant_status["rocm_cdna3"] == "portable"
    assert package.manifest.kernels.variant_status["metal_m3"] == "portable"
    assert "pytorch" in package.manifest.kernels.portable_backends
    names = set(package.weight_store().entries)
    assert "layer_0_ssm_time_decay" in names
    assert "layer_0_ssm_ffn_time_mix_k" in names
    result = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert result.text

    pytest.importorskip("torch")
    from aether.runtime.torch_state_engine import TorchRWKVAEGEngine

    cpu_engine = load_engine_from_path(artifact)
    cpu_logits, _ = cpu_engine.forward(np.asarray([1, 2, 3], dtype=np.int64))
    portable = TorchRWKVAEGEngine(cpu_engine, "cpu")
    portable_logits, _ = portable.forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, portable_logits, rtol=2e-5, atol=2e-5)
    mesh = TorchRWKVAEGEngine(cpu_engine, "cpu", devices=["cpu:0", "cpu:1"])
    mesh_logits, _ = mesh.forward(np.asarray([1, 2, 3], dtype=np.int64))
    np.testing.assert_allclose(cpu_logits, mesh_logits, rtol=2e-5, atol=2e-5)
