"""Concrete encoder graph regression tests with no network dependency."""

from __future__ import annotations

from aether.compiler.config import CompilerConfig
from aether.compiler.stage1_ingestion.architecture_detector import ArchitectureDetector
from aether.compiler.stage1_ingestion.ingestion import IngestionPipeline
from aether.core.graph import AEGGraph


def test_bert_encoder_graph_contains_runtime_required_nodes() -> None:
    architecture = ArchitectureDetector()._from_config(
        {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "vocab_size": 64,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
        }
    )
    graph = AEGGraph(name="bert-regression", architecture=architecture)
    IngestionPipeline(CompilerConfig(skip_download=True))._build_encoder_graph(
        graph, architecture
    )

    node_ids = set(graph.nodes)
    required = {
        "input_ids",
        "token_embeddings",
        "position_embeddings",
        "token_type_embeddings",
        "pooler",
        "output",
        "layer_0_qkv",
        "layer_1_qkv",
    }
    assert required.issubset(node_ids)
    assert graph.get_metadata("is_encoder") is True
    assert graph.get_metadata("encoder_layers") == 2
