"""End-to-end coverage for a real BERT-style encoder AEG artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from aether import Compiler, CompilerConfig
from aether.backends.torch_backend import TorchBackend
from aether.cli import cli
from aether.core.aeg_format import load_aeg_package
from aether.runtime.aeg_loader import load_engine_from_path


def _write_tiny_bert(path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    hidden = 8
    rng = np.random.default_rng(91)
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["BertModel"],
                "model_type": "bert",
                "num_hidden_layers": 1,
                "hidden_size": hidden,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "vocab_size": 20,
                "max_position_embeddings": 8,
                "type_vocab_size": 2,
                "layer_norm_eps": 1e-12,
            }
        ),
        encoding="utf-8",
    )
    matrix = lambda shape: rng.normal(size=shape).astype("float32")
    vector = lambda size: np.zeros(size, dtype="float32")
    tensors: dict[str, np.ndarray] = {
        "embeddings.word_embeddings.weight": matrix((20, hidden)),
        "embeddings.position_embeddings.weight": matrix((8, hidden)),
        "embeddings.token_type_embeddings.weight": matrix((2, hidden)),
        "embeddings.LayerNorm.weight": np.ones(hidden, dtype="float32"),
        "embeddings.LayerNorm.bias": vector(hidden),
        "encoder.layer.0.attention.output.dense.weight": matrix((hidden, hidden)),
        "encoder.layer.0.attention.output.dense.bias": vector(hidden),
        "encoder.layer.0.attention.output.LayerNorm.weight": np.ones(hidden, dtype="float32"),
        "encoder.layer.0.attention.output.LayerNorm.bias": vector(hidden),
        "encoder.layer.0.intermediate.dense.weight": matrix((16, hidden)),
        "encoder.layer.0.intermediate.dense.bias": vector(16),
        "encoder.layer.0.output.dense.weight": matrix((hidden, 16)),
        "encoder.layer.0.output.dense.bias": vector(hidden),
        "encoder.layer.0.output.LayerNorm.weight": np.ones(hidden, dtype="float32"),
        "encoder.layer.0.output.LayerNorm.bias": vector(hidden),
        "pooler.dense.weight": matrix((hidden, hidden)),
        "pooler.dense.bias": vector(hidden),
    }
    for projection in ("query", "key", "value"):
        tensors[f"encoder.layer.0.attention.self.{projection}.weight"] = matrix((hidden, hidden))
        tensors[f"encoder.layer.0.attention.self.{projection}.bias"] = vector(hidden)
    safetensors.save_file(tensors, str(path / "model.safetensors"))

    vocabulary = {f"tok{i}": i for i in range(20)}
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="tok0")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="tok0"
    ).save_pretrained(str(path))


@pytest.mark.integration
def test_encoder_compile_reload_and_embedding(tmp_path: Path) -> None:
    source = tmp_path / "tiny-bert"
    _write_tiny_bert(source)
    artifact = tmp_path / "tiny-bert.aeg"

    package = Compiler(
        CompilerConfig(pruning_target_sparsity=0.0)
    ).compile(str(source), output_path=artifact)

    assert package.manifest.architecture.is_encoder is True
    assert "position_embedding" in package.weights
    assert "pooler" in package.weights
    engine = load_engine_from_path(artifact)
    embedding = np.asarray(engine.embed([[1, 2, 3]]), dtype=np.float32)
    assert embedding.shape == (1, 8)
    assert np.isfinite(embedding).all()
    assert not np.allclose(embedding, 0.0)

    backend = TorchBackend()
    backend.load_model("tiny-bert", str(artifact), offline=True)
    api_embedding = np.asarray(backend.embed("tiny-bert", ["tok1 tok2"]), dtype=np.float32)
    assert api_embedding.shape == (1, 8)
    assert np.isfinite(api_embedding).all()

    result = CliRunner().invoke(
        cli,
        ["green-profile", str(artifact), "--no-defer"],
    )
    assert result.exit_code == 0, result.output
    reloaded = load_aeg_package(artifact)
    assert "green_energy_compilation" in reloaded.metadata["optimizer_passes"]
    assert (artifact / "metadata" / "green_profile.json").is_file()
