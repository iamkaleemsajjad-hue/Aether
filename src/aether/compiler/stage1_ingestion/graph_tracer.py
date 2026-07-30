"""
Symbolic graph tracer.

Traces a PyTorch model's forward pass to produce a symbolic computation graph.
The traced graph is used to seed the AEG-IR representation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.fx import symbolic_trace

from aether.core.aeg_ir import AEGIRModule, AEGInstruction, AEGOpCode, AEGOperand, Block, Function
from aether.core.exceptions import GraphTraceError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class GraphTracer:
    """Traces a PyTorch model and converts it to an AEG-IR module."""

    def __init__(self) -> None:
        self._module: AEGIRModule | None = None

    def trace(self, model: torch.nn.Module, example_input: torch.Tensor | None = None) -> AEGIRModule:
        """Trace a PyTorch model and convert to AEG-IR.

        Args:
            model: PyTorch model to trace.
            example_input: Optional example input for tracing.

        Returns:
            AEG-IR module.
        """
        try:
            traced = symbolic_trace(model, concrete_args=None) if example_input is None else torch.jit.trace(model, example_input)
        except Exception as exc:
            msg = f"Failed to trace model: {exc}"
            raise GraphTraceError(msg) from exc
        return self._convert_traced_graph(traced)

    def _convert_traced_graph(self, traced: Any) -> AEGIRModule:
        """Convert a traced graph to an AEG-IR module."""
        module = AEGIRModule(version="AEG-IR/1.0")
        func = Function(name="model")
        block = Block(name="entry")

        # Placeholder mapping from torch.fx node names to AEG-IR operands
        name_map: dict[str, str] = {}

        if hasattr(traced, "graph"):
            nodes = list(traced.graph.nodes)
        else:
            nodes = []

        for node in nodes:
            if node.op == "placeholder":
                name_map[node.name] = node.name
                block.add_instruction(
                    AEGInstruction(
                        results=[AEGOperand(name=node.name, type_str="tensor<*xbf16>")],
                        op_code=AEGOpCode.ARGUMENT,
                        inputs=[],
                        attributes={"name": node.name},
                    )
                )
            elif node.op == "call_function":
                op_code = self._map_op(node.target.__name__ if hasattr(node.target, "__name__") else str(node.target))
                output_name = node.name
                input_names = [name_map.get(str(a), str(a)) for a in node.args if isinstance(a, (str, type(node)))]
                block.add_instruction(
                    AEGInstruction(
                        results=[AEGOperand(name=output_name, type_str="tensor<*xbf16>")],
                        op_code=op_code,
                        inputs=[n for n in input_names if n],
                        attributes={"target": str(node.target)},
                    )
                )
                name_map[node.name] = output_name
            elif node.op == "output":
                continue

        func.add_block(block)
        module.add_function(func)
        self._module = module
        return module

    def _map_op(self, target_name: str) -> AEGOpCode:
        """Map a PyTorch function name to an AEG-IR opcode."""
        mapping = {
            "add": AEGOpCode.ADD,
            "mul": AEGOpCode.MUL,
            "matmul": AEGOpCode.MATMUL,
            "linear": AEGOpCode.LINEAR,
            "embedding": AEGOpCode.EMBEDDING,
            "softmax": AEGOpCode.SOFTMAX,
            "silu": AEGOpCode.SILU,
            "relu": AEGOpCode.RELU,
            "gelu": AEGOpCode.GELU,
        }
        return mapping.get(target_name.lower(), AEGOpCode.CUSTOM)

    def __repr__(self) -> str:
        return f"GraphTracer(module={self._module is not None})"
