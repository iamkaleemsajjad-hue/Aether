"""Tests for real SafeTensors weight ingestion.

``_ingest_safetensors`` previously ignored the model path entirely and built a
graph from architecture metadata alone, so ``SafeTensorsLoader`` was never called
and no weight ever reached a node. These tests write genuine ``.safetensors``
files to disk and assert the tensors arrive with their exact values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.core.types import ModelArchitecture

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file


@pytest.fixture
def architecture() -> ModelArchitecture:
    return ModelArchitecture(
        family="llama_family",
        params_billion=0.01,
        layers=2,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Write a real SafeTensors checkpoint using HuggingFace naming."""
    rs = np.random.RandomState(0)
    arrays: dict[str, np.ndarray] = {}
    for layer in range(2):
        arrays[f"model.layers.{layer}.self_attn.q_proj.weight"] = rs.randn(64, 64).astype(np.float32)
        arrays[f"model.layers.{layer}.self_attn.o_proj.weight"] = rs.randn(64, 64).astype(np.float32)
        arrays[f"model.layers.{layer}.mlp.gate_proj.weight"] = rs.randn(128, 64).astype(np.float32)
        arrays[f"model.layers.{layer}.mlp.down_proj.weight"] = rs.randn(64, 128).astype(np.float32)
    arrays["model.embed_tokens.weight"] = rs.randn(100, 64).astype(np.float32)
    arrays["lm_head.weight"] = rs.randn(100, 64).astype(np.float32)
    save_file({k: torch.tensor(v) for k, v in arrays.items()}, str(tmp_path / "model.safetensors"))
    return tmp_path, arrays


class TestWeightNameNormalisation:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.layers.0.self_attn.q_proj.weight", (0, "qkv")),
            ("model.layers.7.self_attn.k_proj.weight", (7, "qkv")),
            ("model.layers.1.self_attn.o_proj.weight", (1, "out_proj")),
            ("model.layers.2.mlp.gate_proj.weight", (2, "gate_proj")),
            ("model.layers.3.mlp.down_proj.weight", (3, "ffn")),
            ("model.layers.4.input_layernorm.weight", (4, "rmsnorm")),
            ("model.embed_tokens.weight", (None, "embedding")),
            ("lm_head.weight", (None, "lm_head")),
        ],
    )
    def test_checkpoint_names_resolve(self, name: str, expected: tuple) -> None:
        assert IngestionPipeline._normalise_weight_name(name) == expected

    @pytest.mark.parametrize(
        ("node_id", "expected"),
        [
            ("layer_0_qkv", (0, "qkv")),
            ("layer_3_out_proj", (3, "out_proj")),
            ("layer_1_gate_proj", (1, "gate_proj")),
            ("layer_2_ffn", (2, "ffn")),
            ("embedding", (None, "embedding")),
            ("lm_head", (None, "lm_head")),
        ],
    )
    def test_node_ids_resolve_to_same_keys(self, node_id: str, expected: tuple) -> None:
        """Both vocabularies must land on identical keys or nothing can match."""
        assert IngestionPipeline._normalise_weight_name(node_id) == expected

    @pytest.mark.parametrize("name", ["input", "output", ""])
    def test_unidentifiable_names_yield_empty_key(self, name: str) -> None:
        assert IngestionPipeline._normalise_weight_name(name) == (None, None)

    @pytest.mark.parametrize(
        "name",
        ["layer_0_residual_1", "layer_0_attention", "model.layers.0.self_attn.rotary_emb.inv_freq"],
    )
    def test_names_without_a_component_are_unusable(self, name: str) -> None:
        """A (layer, None) key identifies nothing and must never be matched on.

        Both ``layer_0_attention`` and an unrecognised ``layers.0.*`` tensor
        reduce to ``(0, None)``; treating that as a key would bind an arbitrary
        tensor to an unrelated node.
        """
        assert IngestionPipeline._normalise_weight_name(name)[1] is None

    def test_prefixes_are_stripped(self) -> None:
        bare = IngestionPipeline._normalise_weight_name("layers.0.self_attn.q_proj.weight")
        prefixed = IngestionPipeline._normalise_weight_name("model.layers.0.self_attn.q_proj.weight")
        assert bare == prefixed


class TestSafeTensorsIngestion:
    def test_weights_are_attached(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        path, arrays = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        assert graph.metadata["weights_attached"] == len(arrays)
        assert graph.metadata["weight_tensor_count"] == len(arrays)

    def test_attached_values_match_the_checkpoint_exactly(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        """Regression: the graph used to contain no weights at all."""
        path, arrays = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        np.testing.assert_array_equal(
            graph.get_node("layer_0_qkv").get_attribute("weight"),
            arrays["model.layers.0.self_attn.q_proj.weight"],
        )
        np.testing.assert_array_equal(
            graph.get_node("embedding").get_attribute("weight"),
            arrays["model.embed_tokens.weight"],
        )

    def test_weight_shapes_are_recorded(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        path, _ = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        assert graph.get_node("layer_0_gate_proj").get_attribute("weight_shape") == [128, 64]

    def test_source_tensor_name_is_recorded(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        path, _ = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        source = graph.get_node("layer_1_ffn").get_attribute("weight_source")
        assert source == "model.layers.1.mlp.down_proj.weight"

    def test_fused_nodes_record_all_contributing_tensors(
        self, tmp_path: Path, architecture: ModelArchitecture
    ) -> None:
        """A single qkv node covers separate q/k/v checkpoint tensors."""
        rs = np.random.RandomState(1)
        arrays = {
            f"model.layers.0.self_attn.{proj}.weight": rs.randn(64, 64).astype(np.float32)
            for proj in ("q_proj", "k_proj", "v_proj")
        }
        save_file({k: torch.tensor(v) for k, v in arrays.items()}, str(tmp_path / "m.safetensors"))
        graph = IngestionPipeline().ingest(str(tmp_path), architecture)
        fused = graph.get_node("layer_0_qkv").get_attribute("fused_weight_sources")
        assert fused is not None
        assert len(fused) == 3

    def test_single_file_path_is_accepted(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        path, arrays = checkpoint
        graph = IngestionPipeline().ingest(str(path / "model.safetensors"), architecture)
        assert graph.metadata["weights_attached"] == len(arrays)

    def test_graph_structure_is_still_built(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        path, _ = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        assert graph.get_node("input") is not None
        assert len(list(graph)) > len(_)


class TestIngestionFallbacks:
    def test_missing_path_still_produces_a_graph(self, architecture: ModelArchitecture) -> None:
        """A Hub id or absent path must not break ingestion."""
        graph = IngestionPipeline().ingest("meta-llama/Llama-3-8B", architecture)
        assert graph.metadata["weights_attached"] == 0
        assert len(list(graph)) > 0

    def test_empty_directory_produces_a_graph(
        self, tmp_path: Path, architecture: ModelArchitecture
    ) -> None:
        graph = IngestionPipeline().ingest(str(tmp_path), architecture)
        assert graph.metadata["weights_attached"] == 0

    def test_unrecognised_tensor_names_are_skipped(
        self, tmp_path: Path, architecture: ModelArchitecture
    ) -> None:
        arrays = {"some.unrelated.tensor": np.zeros((4, 4), dtype=np.float32)}
        save_file({k: torch.tensor(v) for k, v in arrays.items()}, str(tmp_path / "m.safetensors"))
        graph = IngestionPipeline().ingest(str(tmp_path), architecture)
        assert graph.metadata["weights_attached"] == 0


class TestIngestedWeightsFeedThePasses:
    def test_pruning_pass_computes_masks_on_ingested_weights(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        """The point of real ingestion: downstream passes get real tensors."""
        from aether.compiler.config import CompilerConfig
        from aether.compiler.stage2_optimizer.optimizer import PruningSparsityPass

        path, _ = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        _, report = PruningSparsityPass().run(graph, architecture, CompilerConfig())
        assert report.details["masks_computed"] > 0

    def test_quantizing_an_ingested_weight_roundtrips(
        self, checkpoint: tuple[Path, dict[str, np.ndarray]], architecture: ModelArchitecture
    ) -> None:
        from aether.quantization.formats import dequantize_tensor, quantize_tensor

        path, _ = checkpoint
        graph = IngestionPipeline().ingest(str(path), architecture)
        weight = graph.get_node("layer_0_qkv").get_attribute("weight")
        restored = dequantize_tensor(quantize_tensor(weight, "Q4_K_M", 32))
        assert restored.shape == weight.shape
        assert float(np.sqrt(np.mean((weight - restored) ** 2))) < 0.15
