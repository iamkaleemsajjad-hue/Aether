"""
Tests for AEG computation graph.
"""

from __future__ import annotations

import pytest

from aether.core.graph import AEGGraph, AEGGraphEdge, AEGGraphNode, AEGGraphNodeType
from aether.core.types import ModelArchitecture


class TestAEGGraphCreation:
    """Tests for graph creation and basic operations."""

    def test_create_empty_graph(self) -> None:
        graph = AEGGraph(name="empty")
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self) -> None:
        graph = AEGGraph()
        node = AEGGraphNode(id="x", node_type=AEGGraphNodeType.INPUT, name="input")
        graph.add_node(node)
        assert graph.node_count == 1
        assert graph.get_node("x") is node

    def test_add_node_duplicate_id_raises(self) -> None:
        graph = AEGGraph()
        graph.add_node(AEGGraphNode(id="x", node_type=AEGGraphNodeType.INPUT, name="x"))
        with pytest.raises(ValueError):
            graph.add_node(AEGGraphNode(id="x", node_type=AEGGraphNodeType.INPUT, name="x2"))

    def test_add_edge(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b")
        graph.add_node(a)
        graph.add_node(b)
        edge = AEGGraphEdge(source="a", target="b")
        graph.add_edge(edge)
        assert graph.edge_count == 1
        assert graph.get_output_edges("a")[0].target == "b"
        assert graph.get_input_edges("b")[0].source == "a"

    def test_add_edge_missing_node_raises(self) -> None:
        graph = AEGGraph()
        graph.add_node(AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a"))
        with pytest.raises(ValueError):
            graph.add_edge(AEGGraphEdge(source="a", target="missing"))


class TestGraphTraversal:
    """Tests for graph traversal and ordering."""

    def test_topological_order(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b", inputs=["a"])
        c = AEGGraphNode(id="c", node_type=AEGGraphNodeType.OUTPUT, name="c", inputs=["b"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        graph.add_edge(AEGGraphEdge(source="b", target="c"))
        order = graph.topological_order()
        assert order == ["a", "b", "c"]

    def test_cycle_detected(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.OPERATION, name="a", inputs=["b"])
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b", inputs=["a"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        graph.add_edge(AEGGraphEdge(source="b", target="a"))
        with pytest.raises(ValueError):
            graph.topological_order()

    def test_predecessors_and_successors(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b", inputs=["a"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        assert graph.get_successors("a") == [b]
        assert graph.get_predecessors("b") == [a]


class TestGraphFusion:
    """Tests for graph fusion."""

    def test_fuse_subgraph(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b", inputs=["a"])
        c = AEGGraphNode(id="c", node_type=AEGGraphNodeType.OPERATION, name="c", inputs=["b"])
        d = AEGGraphNode(id="d", node_type=AEGGraphNodeType.OUTPUT, name="d", inputs=["c"])
        for n in [a, b, c, d]:
            graph.add_node(n)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        graph.add_edge(AEGGraphEdge(source="b", target="c"))
        graph.add_edge(AEGGraphEdge(source="c", target="d"))
        fused = graph.fuse_subgraph(["b", "c"], "fused", "aeg.fused")
        assert fused.node_type == AEGGraphNodeType.FUSED_OPERATION
        assert "a" in fused.inputs
        assert "d" in fused.outputs

    def test_fuse_empty_subgraph_raises(self) -> None:
        graph = AEGGraph()
        with pytest.raises(ValueError):
            graph.fuse_subgraph([], "fused", "aeg.fused")


class TestGraphValidation:
    """Tests for graph validation."""

    def test_valid_graph(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OUTPUT, name="b", inputs=["a"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        result = graph.validate()
        assert result.is_valid

    def test_missing_edge_endpoint(self) -> None:
        graph = AEGGraph()
        graph.add_node(AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a"))
        with pytest.raises(ValueError):
            graph.add_edge(AEGGraphEdge(source="a", target="missing"))

    def test_parameter_without_consumer(self) -> None:
        graph = AEGGraph()
        graph.add_node(AEGGraphNode(id="w", node_type=AEGGraphNodeType.PARAMETER, name="w"))
        result = graph.validate()
        assert result.is_valid
        assert any("w" in w for w in result.warnings)


class TestGraphSerialization:
    """Tests for graph serialization."""

    def test_json_roundtrip(self, small_architecture: ModelArchitecture) -> None:
        graph = AEGGraph(name="test", architecture=small_architecture)
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OUTPUT, name="b", inputs=["a"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        json_str = graph.to_json()
        loaded = AEGGraph.from_json(json_str)
        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        assert loaded.name == "test"

    def test_summary(self) -> None:
        graph = AEGGraph()
        graph.add_node(AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a"))
        graph.add_node(AEGGraphNode(id="b", node_type=AEGGraphNodeType.OUTPUT, name="b"))
        summary = graph.summary()
        assert summary["nodes"] == 2
        assert summary["inputs"] == 1
        assert summary["outputs"] == 1


class TestGraphIteration:
    """Tests for graph iteration."""

    def test_iter_topological(self) -> None:
        graph = AEGGraph()
        a = AEGGraphNode(id="a", node_type=AEGGraphNodeType.INPUT, name="a")
        b = AEGGraphNode(id="b", node_type=AEGGraphNodeType.OPERATION, name="b", inputs=["a"])
        c = AEGGraphNode(id="c", node_type=AEGGraphNodeType.OUTPUT, name="c", inputs=["b"])
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(c)
        graph.add_edge(AEGGraphEdge(source="a", target="b"))
        graph.add_edge(AEGGraphEdge(source="b", target="c"))
        nodes = list(graph)
        assert [n.id for n in nodes] == ["a", "b", "c"]


class TestNodeProperties:
    """Tests for node helper properties."""

    def test_node_type_helpers(self) -> None:
        n = AEGGraphNode(id="x", node_type=AEGGraphNodeType.PARAMETER, name="x")
        assert n.is_parameter
        assert not n.is_input
        n2 = AEGGraphNode(id="y", node_type=AEGGraphNodeType.INPUT, name="y")
        assert n2.is_input

    def test_set_precision(self) -> None:
        from aether.core.types import Precision
        n = AEGGraphNode(id="x", node_type=AEGGraphNodeType.OPERATION, name="x")
        n.set_precision(Precision.Q4_K_M)
        assert n.precision == Precision.Q4_K_M
        assert n.attributes["precision"] == "Q4_K_M"
