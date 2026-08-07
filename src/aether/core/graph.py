"""
AEG computation graph — nodes, edges, dependencies, and validation.

The AEG graph is a high-level, hardware-agnostic representation of a model's
computation. It is produced during ingestion and consumed by the optimizer
passes, the backend selector, and the runtime executor. Unlike AEG-IR (which is
a textual/serialized IR), the graph is an in-memory object graph optimized for
compiler transformations.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from aether.core.types import ModelArchitecture, Precision, TensorLayout, TensorShape


class AEGGraphNodeType(enum.Enum):
    """Types of nodes in the AEG computation graph."""

    TENSOR = "tensor"
    """A dense tensor input/output."""

    PARAMETER = "parameter"
    """A learned parameter (weight or bias)."""

    OPERATION = "operation"
    """A tensor operation such as matmul, attention, or activation."""

    FUSED_OPERATION = "fused_operation"
    """A fused sequence of operations treated as a single unit."""

    CONTROL = "control"
    """Control flow such as conditional routing or loops."""

    INPUT = "input"
    """External input to the graph."""

    OUTPUT = "output"
    """External output of the graph."""

    KV_CACHE = "kv_cache"
    """A KV cache read/write node."""

    EXPERT_ROUTER = "expert_router"
    """MoE expert routing/dispatch node."""

    EXPERT_BANK = "expert_bank"
    """A bank of expert FFNs in a MoE layer."""

    PREFIX_CACHE = "prefix_cache"
    """A prefix cache hint for the runtime."""

    SHARD_ANNOTATION = "shard_annotation"
    """A placement/sharding hint for distributed execution."""

    PRECISION_ANNOTATION = "precision_annotation"
    """A precision hint for a subgraph or weight."""

    def __str__(self) -> str:
        return self.value


class AEGGraphEdgeType(enum.Enum):
    """Types of edges in the AEG computation graph."""

    DATA = "data"
    """A data dependency (tensor flows from one node to another)."""

    CONTROL = "control"
    """A control dependency (e.g., must run before another node)."""

    PARAMETER = "parameter"
    """A parameter usage edge (operation reads a parameter)."""

    SHARDING = "sharding"
    """A sharding/parallelism relation."""

    FUSION = "fusion"
    """An edge indicating fused operations."""

    KV_LINK = "kv_link"
    """A link between KV cache nodes across layers."""


@dataclass
class AEGGraphNode:
    """A node in the AEG computation graph."""

    id: str
    """Unique identifier for the node."""

    node_type: AEGGraphNodeType
    """Type of the node."""

    name: str
    """Human-readable name."""

    op_type: str | None = None
    """Operation type (e.g., 'rmsnorm', 'gqa', 'swiglu_ffn')."""

    inputs: list[str] = field(default_factory=list)
    """IDs of input nodes."""

    outputs: list[str] = field(default_factory=list)
    """IDs of output nodes."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Arbitrary attributes: precision, fusion flags, sharding hints, etc."""

    layout: TensorLayout | None = None
    """Tensor layout for tensor/parameter nodes."""

    precision: Precision | None = None
    """Assigned precision for this node (for weights and ops)."""

    layer_index: int | None = None
    """Transformer layer index for structured models."""

    subgraph: list[str] | None = None
    """For fused/control nodes, the IDs of the enclosed subgraph."""

    def __post_init__(self) -> None:
        """Validate basic node properties."""
        if not self.id:
            msg = "Graph node ID cannot be empty"
            raise ValueError(msg)

    @property
    def is_fused(self) -> bool:
        """Return True if this node represents a fused operation."""
        return self.node_type == AEGGraphNodeType.FUSED_OPERATION

    @property
    def is_parameter(self) -> bool:
        """Return True if this node is a learned parameter."""
        return self.node_type == AEGGraphNodeType.PARAMETER

    @property
    def is_input(self) -> bool:
        """Return True if this node is an external input."""
        return self.node_type == AEGGraphNodeType.INPUT

    @property
    def is_output(self) -> bool:
        """Return True if this node is an external output."""
        return self.node_type == AEGGraphNodeType.OUTPUT

    @property
    def has_precision(self) -> bool:
        """Return True if the node has an assigned precision."""
        return self.precision is not None

    def set_precision(self, precision: Precision) -> None:
        """Assign a precision to this node."""
        self.precision = precision
        self.attributes.setdefault("precision", precision.value)

    def add_attribute(self, key: str, value: Any) -> None:
        """Add or update an attribute."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value with a default."""
        return self.attributes.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to a dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
            "layout": self.layout.to_dict() if self.layout else None,
            "precision": self.precision.value if self.precision else None,
            "layer_index": self.layer_index,
            "subgraph": self.subgraph,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGGraphNode:
        """Deserialize a node from a dictionary."""
        layout = data.get("layout")
        precision = data.get("precision")
        return AEGGraphNode(
            id=data["id"],
            node_type=AEGGraphNodeType(data["node_type"]),
            name=data["name"],
            op_type=data.get("op_type"),
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            attributes=dict(data.get("attributes", {})),
            layout=TensorLayout.from_dict(layout) if layout else None,
            precision=Precision.from_string(precision) if precision else None,
            layer_index=data.get("layer_index"),
            subgraph=list(data.get("subgraph", [])) if data.get("subgraph") else None,
        )

    def __repr__(self) -> str:
        return f"AEGGraphNode({self.id}, {self.node_type.value}, {self.name})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AEGGraphNode):
            return NotImplemented
        return self.id == other.id


@dataclass
class AEGGraphEdge:
    """An edge in the AEG computation graph."""

    source: str
    """Source node ID."""

    target: str
    """Target node ID."""

    edge_type: AEGGraphEdgeType = AEGGraphEdgeType.DATA
    """Type of dependency."""

    label: str | None = None
    """Optional edge label (e.g., argument name)."""

    attributes: dict[str, Any] = field(default_factory=dict)
    """Additional edge attributes."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this edge to a dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "label": self.label,
            "attributes": self.attributes,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGGraphEdge:
        """Deserialize an edge from a dictionary."""
        return AEGGraphEdge(
            source=data["source"],
            target=data["target"],
            edge_type=AEGGraphEdgeType(data.get("edge_type", "data")),
            label=data.get("label"),
            attributes=dict(data.get("attributes", {})),
        )

    def __repr__(self) -> str:
        return f"AEGGraphEdge({self.source} -> {self.target}, {self.edge_type.value})"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type.value, self.label))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AEGGraphEdge):
            return NotImplemented
        return (
            self.source == other.source
            and self.target == other.target
            and self.edge_type == other.edge_type
            and self.label == other.label
        )


@dataclass
class GraphValidationResult:
    """Result of validating an AEG graph."""

    is_valid: bool
    """True if the graph passes validation."""

    errors: list[str] = field(default_factory=list)
    """List of validation error messages."""

    warnings: list[str] = field(default_factory=list)
    """List of validation warnings."""

    def add_error(self, message: str) -> None:
        """Add a validation error and mark the graph invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GraphValidationResult:
        """Deserialize validation result from a dictionary."""
        result = GraphValidationResult(
            is_valid=data.get("is_valid", True),
            errors=list(data.get("errors", [])),
            warnings=list(data.get("warnings", [])),
        )
        return result

    def __repr__(self) -> str:
        status = "valid" if self.is_valid else "invalid"
        return (
            f"GraphValidationResult({status}, {len(self.errors)} errors, "
            f"{len(self.warnings)} warnings)"
        )


class AEGGraph:
    """An in-memory AEG computation graph.

    The graph stores nodes and edges, provides traversal helpers, and supports
    common compiler operations: inserting/removing nodes, fusing subgraphs,
    annotating precision and sharding, and validating structure.
    """

    def __init__(
        self,
        name: str = "aeg_graph",
        architecture: ModelArchitecture | None = None,
    ) -> None:
        """Create an empty AEG graph.

        Args:
            name: Human-readable graph name.
            architecture: Detected model architecture metadata.
        """
        self.name = name
        self.architecture = architecture
        self._nodes: dict[str, AEGGraphNode] = {}
        self._edges: list[AEGGraphEdge] = []
        self._node_input_edges: dict[str, list[AEGGraphEdge]] = {}
        self._node_output_edges: dict[str, list[AEGGraphEdge]] = {}
        self._metadata: dict[str, Any] = {}

    @property
    def nodes(self) -> dict[str, AEGGraphNode]:
        """Return the mapping of node IDs to nodes."""
        return self._nodes

    @property
    def edges(self) -> list[AEGGraphEdge]:
        """Return all edges in the graph."""
        return self._edges

    @property
    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of edges."""
        return len(self._edges)

    @property
    def input_nodes(self) -> list[AEGGraphNode]:
        """Return all external input nodes."""
        return [n for n in self._nodes.values() if n.is_input]

    @property
    def output_nodes(self) -> list[AEGGraphNode]:
        """Return all external output nodes."""
        return [n for n in self._nodes.values() if n.is_output]

    @property
    def parameter_nodes(self) -> list[AEGGraphNode]:
        """Return all parameter nodes."""
        return [n for n in self._nodes.values() if n.is_parameter]

    @property
    def operation_nodes(self) -> list[AEGGraphNode]:
        """Return all operation nodes."""
        return [n for n in self._nodes.values() if n.node_type in (AEGGraphNodeType.OPERATION, AEGGraphNodeType.FUSED_OPERATION)]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return graph metadata."""
        return self._metadata

    def add_node(self, node: AEGGraphNode) -> AEGGraphNode:
        """Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            The added node.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.id in self._nodes:
            msg = f"Node with ID '{node.id}' already exists"
            raise ValueError(msg)
        self._nodes[node.id] = node
        self._node_input_edges.setdefault(node.id, [])
        self._node_output_edges.setdefault(node.id, [])
        return node

    def add_edge(self, edge: AEGGraphEdge) -> AEGGraphEdge:
        """Add an edge to the graph.

        Args:
            edge: The edge to add.

        Returns:
            The added edge.

        Raises:
            ValueError: If source or target node does not exist.
        """
        if edge.source not in self._nodes:
            msg = f"Source node '{edge.source}' does not exist"
            raise ValueError(msg)
        if edge.target not in self._nodes:
            msg = f"Target node '{edge.target}' does not exist"
            raise ValueError(msg)
        self._edges.append(edge)
        self._node_output_edges.setdefault(edge.source, []).append(edge)
        self._node_input_edges.setdefault(edge.target, []).append(edge)
        return edge

    def add_node_with_edges(
        self,
        node: AEGGraphNode,
        input_edges: Sequence[AEGGraphEdge] | None = None,
        output_edges: Sequence[AEGGraphEdge] | None = None,
    ) -> AEGGraphNode:
        """Add a node and a set of connecting edges atomically.

        Args:
            node: The node to add.
            input_edges: Edges pointing into the new node.
            output_edges: Edges pointing out of the new node.

        Returns:
            The added node.
        """
        self.add_node(node)
        for edge in input_edges or []:
            self.add_edge(edge)
        for edge in output_edges or []:
            self.add_edge(edge)
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all incident edges from the graph.

        Args:
            node_id: The ID of the node to remove.
        """
        if node_id not in self._nodes:
            return
        # Remove incident edges first
        incident = list(self._node_input_edges.get(node_id, []) + self._node_output_edges.get(node_id, []))
        for edge in incident:
            self.remove_edge(edge)
        del self._nodes[node_id]
        self._node_input_edges.pop(node_id, None)
        self._node_output_edges.pop(node_id, None)

    def remove_edge(self, edge: AEGGraphEdge) -> None:
        """Remove a specific edge from the graph.

        Args:
            edge: The edge to remove.
        """
        if edge in self._edges:
            self._edges.remove(edge)
            self._node_output_edges.get(edge.source, []).remove(edge)
            self._node_input_edges.get(edge.target, []).remove(edge)

    def get_node(self, node_id: str) -> AEGGraphNode | None:
        """Look up a node by ID."""
        return self._nodes.get(node_id)

    def get_input_edges(self, node_id: str) -> list[AEGGraphEdge]:
        """Return all edges pointing into a node."""
        return list(self._node_input_edges.get(node_id, []))

    def get_output_edges(self, node_id: str) -> list[AEGGraphEdge]:
        """Return all edges pointing out of a node."""
        return list(self._node_output_edges.get(node_id, []))

    def get_predecessors(self, node_id: str) -> list[AEGGraphNode]:
        """Return all predecessor nodes of a given node."""
        return [self._nodes[e.source] for e in self.get_input_edges(node_id) if e.source in self._nodes]

    def get_successors(self, node_id: str) -> list[AEGGraphNode]:
        """Return all successor nodes of a given node."""
        return [self._nodes[e.target] for e in self.get_output_edges(node_id) if e.target in self._nodes]

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (data dependencies only).

        Nodes that have been fused into a megakernel (``is_fused_away=True``)
        are excluded from the ordering — they are retained in the node dict for
        inspection but are logically replaced by the enclosing fused node.

        Returns:
            A list of node IDs sorted such that all predecessors appear before
            their successors.
        """
        # Exclude nodes that were subsumed into a fused operation.
        live_nodes = {
            nid: node
            for nid, node in self._nodes.items()
            if not node.attributes.get("is_fused_away", False)
        }

        in_degree: dict[str, int] = {nid: 0 for nid in live_nodes}
        for edge in self._edges:
            if edge.edge_type != AEGGraphEdgeType.DATA:
                continue
            # Only count edges where both endpoints are live nodes.
            if edge.source in live_nodes and edge.target in live_nodes:
                in_degree[edge.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for edge in self.get_output_edges(current):
                if edge.edge_type == AEGGraphEdgeType.DATA and edge.target in in_degree:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)

        if len(order) != len(live_nodes):
            msg = "Graph contains a cycle; cannot produce topological order"
            raise ValueError(msg)
        return order

    def iter_layers(self) -> Iterator[list[AEGGraphNode]]:
        """Iterate over nodes grouped by transformer layer index.

        Yields:
            Lists of nodes for each layer index, sorted by index.
        """
        by_layer: dict[int, list[AEGGraphNode]] = {}
        for node in self._nodes.values():
            idx = node.layer_index if node.layer_index is not None else -1
            by_layer.setdefault(idx, []).append(node)
        for idx in sorted(by_layer):
            yield by_layer[idx]

    def find_nodes_by_op_type(self, op_type: str) -> list[AEGGraphNode]:
        """Return all nodes matching an operation type."""
        return [n for n in self._nodes.values() if n.op_type == op_type]

    def find_nodes_by_type(self, node_type: AEGGraphNodeType) -> list[AEGGraphNode]:
        """Return all nodes matching a node type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def fuse_subgraph(
        self,
        node_ids: Sequence[str],
        fused_name: str,
        fused_op_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> AEGGraphNode:
        """Replace a subgraph with a single fused operation node.

        The fused node keeps the external inputs and outputs of the original
        subgraph. The original nodes are retained as a subgraph annotation for
        inspection and debugging.

        Args:
            node_ids: IDs of nodes to fuse.
            fused_name: Name of the new fused node.
            fused_op_type: Operation type for the fused node.
            attributes: Additional attributes for the fused node.

        Returns:
            The new fused node.
        """
        node_ids = list(node_ids)
        if not node_ids:
            msg = "Cannot fuse empty subgraph"
            raise ValueError(msg)

        node_set = set(node_ids)
        fused_id = f"fused_{fused_name}_{'_'.join(node_ids[:3])}"
        if len(node_ids) > 3:
            fused_id += f"_and_{len(node_ids) - 3}_more"

        # Determine external inputs (from outside the subgraph)
        external_inputs: list[str] = []
        for nid in node_ids:
            node = self._nodes[nid]
            for in_id in node.inputs:
                if in_id not in node_set and in_id not in external_inputs:
                    external_inputs.append(in_id)

        # Determine external outputs (to outside the subgraph)
        external_outputs: list[str] = []
        for edge in self._edges:
            if edge.source in node_set and edge.target not in node_set:
                if edge.target not in external_outputs:
                    external_outputs.append(edge.target)

        # Build fused node
        fused_attrs = dict(attributes or {})
        fused_attrs["fused_node_count"] = len(node_ids)
        fused_attrs["fused_op_types"] = sorted({self._nodes[nid].op_type for nid in node_ids if self._nodes[nid].op_type})
        fused_node = AEGGraphNode(
            id=fused_id,
            node_type=AEGGraphNodeType.FUSED_OPERATION,
            name=fused_name,
            op_type=fused_op_type,
            inputs=external_inputs,
            outputs=external_outputs,
            attributes=fused_attrs,
            subgraph=node_ids,
        )
        self.add_node(fused_node)

        # Rewire edges
        for in_id in external_inputs:
            self.add_edge(AEGGraphEdge(source=in_id, target=fused_id, edge_type=AEGGraphEdgeType.DATA, label="input"))
        for out_id in external_outputs:
            self.add_edge(AEGGraphEdge(source=fused_id, target=out_id, edge_type=AEGGraphEdgeType.DATA, label="output"))

        # Mark original nodes as fused (but keep them for inspection)
        for nid in node_ids:
            self._nodes[nid].attributes["fused_into"] = fused_id
            self._nodes[nid].attributes["is_fused_away"] = True

        return fused_node

    def validate(self) -> GraphValidationResult:
        """Validate the graph structure and return a result object.

        Checks include:
        - All edge endpoints refer to existing nodes.
        - No duplicate node IDs.
        - No cycles in data dependencies.
        - Input and output nodes exist.
        - Fused nodes reference valid subgraphs.
        """
        result = GraphValidationResult(is_valid=True)

        # Check for duplicate IDs (defensive)
        seen_ids: set[str] = set()
        for nid in self._nodes:
            if nid in seen_ids:
                result.add_error(f"Duplicate node ID: {nid}")
            seen_ids.add(nid)

        # Check edge endpoints
        for edge in self._edges:
            if edge.source not in self._nodes:
                result.add_error(f"Edge references missing source node: {edge.source}")
            if edge.target not in self._nodes:
                result.add_error(f"Edge references missing target node: {edge.target}")

        # Check for cycles
        try:
            self.topological_order()
        except ValueError as exc:
            result.add_error(str(exc))

        # Check inputs/outputs
        if not self.input_nodes:
            result.add_warning("Graph has no input nodes")
        if not self.output_nodes:
            result.add_warning("Graph has no output nodes")

        # Check fused subgraphs
        for node in self._nodes.values():
            if node.is_fused and node.subgraph:
                for nid in node.subgraph:
                    if nid not in self._nodes:
                        result.add_error(
                            f"Fused node '{node.id}' references missing subgraph node '{nid}'"
                        )

        # Check parameter usage
        for param in self.parameter_nodes:
            if not self.get_output_edges(param.id):
                result.add_warning(f"Parameter '{param.id}' has no consumers")

        return result

    def set_metadata(self, key: str, value: Any) -> None:
        """Set a graph-level metadata entry."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a graph-level metadata entry."""
        return self._metadata.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire graph to a dictionary."""
        return {
            "name": self.name,
            "architecture": self.architecture.to_dict() if self.architecture else None,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
            "metadata": self._metadata,
        }

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the graph to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGGraph:
        """Deserialize an AEG graph from a dictionary."""
        architecture = data.get("architecture")
        graph = AEGGraph(
            name=data.get("name", "aeg_graph"),
            architecture=ModelArchitecture.from_dict(architecture) if architecture else None,
        )
        for node_data in data.get("nodes", []):
            graph.add_node(AEGGraphNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(AEGGraphEdge.from_dict(edge_data))
        graph._metadata = dict(data.get("metadata", {}))
        return graph

    @staticmethod
    def from_json(json_str: str) -> AEGGraph:
        """Deserialize an AEG graph from a JSON string."""
        data = json.loads(json_str)
        return AEGGraph.from_dict(data)

    def clone(self) -> AEGGraph:
        """Create a deep copy of this graph."""
        return AEGGraph.from_dict(self.to_dict())

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the graph."""
        return {
            "name": self.name,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "inputs": len(self.input_nodes),
            "outputs": len(self.output_nodes),
            "parameters": len(self.parameter_nodes),
            "operations": len(self.operation_nodes),
            "fused_operations": len(self.find_nodes_by_type(AEGGraphNodeType.FUSED_OPERATION)),
            "kv_cache_nodes": len(self.find_nodes_by_type(AEGGraphNodeType.KV_CACHE)),
            "expert_nodes": len(self.find_nodes_by_type(AEGGraphNodeType.EXPERT_BANK)),
            "architecture": self.architecture.to_dict() if self.architecture else None,
        }

    def __repr__(self) -> str:
        return f"AEGGraph({self.name}, {self.node_count} nodes, {self.edge_count} edges)"

    def __iter__(self) -> Iterator[AEGGraphNode]:
        """Iterate over nodes in topological order."""
        order = self.topological_order()
        return iter(self._nodes[nid] for nid in order)


def merge_graphs(graphs: Sequence[AEGGraph], merged_name: str = "merged_graph") -> AEGGraph:
    """Merge multiple disjoint graphs into a single graph with unique IDs.

    If node IDs collide, they are renamed with a graph prefix.

    Args:
        graphs: Graphs to merge.
        merged_name: Name of the resulting graph.

    Returns:
        A new merged graph.
    """
    merged = AEGGraph(name=merged_name)
    id_mapping: dict[tuple[int, str], str] = {}
    for g_idx, graph in enumerate(graphs):
        prefix = f"g{g_idx}_"
        for node in graph._nodes.values():
            new_id = f"{prefix}{node.id}"
            id_mapping[(g_idx, node.id)] = new_id
            new_node = AEGGraphNode(
                id=new_id,
                node_type=node.node_type,
                name=node.name,
                op_type=node.op_type,
                inputs=[],
                outputs=[],
                attributes=dict(node.attributes),
                layout=node.layout,
                precision=node.precision,
                layer_index=node.layer_index,
                subgraph=[id_mapping[(g_idx, nid)] for nid in node.subgraph] if node.subgraph else None,
            )
            merged.add_node(new_node)
        for edge in graph._edges:
            new_source = id_mapping[(g_idx, edge.source)]
            new_target = id_mapping[(g_idx, edge.target)]
            merged.add_edge(
                AEGGraphEdge(
                    source=new_source,
                    target=new_target,
                    edge_type=edge.edge_type,
                    label=edge.label,
                    attributes=dict(edge.attributes),
                )
            )
        # Reconnect node inputs/outputs
        for node in merged._nodes.values():
            if node.id.startswith(prefix):
                original_id = node.id[len(prefix) :]
                original_node = graph._nodes[original_id]
                node.inputs = [id_mapping[(g_idx, nid)] for nid in original_node.inputs]
                node.outputs = [id_mapping[(g_idx, nid)] for nid in original_node.outputs]
    return merged


def create_simple_transformer_graph(
    hidden_size: int,
    num_layers: int,
    name: str = "simple_transformer",
) -> AEGGraph:
    """Create a minimal transformer graph for testing and documentation.

    Args:
        hidden_size: Hidden dimension.
        num_layers: Number of layers.
        name: Graph name.

    Returns:
        A simple AEGGraph with placeholder input, layer, and output nodes.
    """
    graph = AEGGraph(name=name)
    input_node = AEGGraphNode(
        id="input",
        node_type=AEGGraphNodeType.INPUT,
        name="input_tokens",
        layout=TensorLayout(
            shape=TensorShape.from_list([None, hidden_size]),
            dtype=__import__("aether.core.types", fromlist=["DType"]).DType.INT64,
        ),
    )
    graph.add_node(input_node)
    prev = input_node.id
    for layer in range(num_layers):
        layer_node = AEGGraphNode(
            id=f"layer_{layer}",
            node_type=AEGGraphNodeType.OPERATION,
            name=f"transformer_layer_{layer}",
            op_type="transformer_layer",
            inputs=[prev],
            layer_index=layer,
            attributes={"hidden_size": hidden_size},
        )
        graph.add_node(layer_node)
        graph.add_edge(AEGGraphEdge(source=prev, target=layer_node.id))
        prev = layer_node.id
    output_node = AEGGraphNode(
        id="output",
        node_type=AEGGraphNodeType.OUTPUT,
        name="logits",
        inputs=[prev],
        layout=TensorLayout(
            shape=TensorShape.from_list([None, hidden_size]),
            dtype=__import__("aether.core.types", fromlist=["DType"]).DType.FP32,
        ),
    )
    graph.add_node(output_node)
    graph.add_edge(AEGGraphEdge(source=prev, target=output_node.id))
    return graph
