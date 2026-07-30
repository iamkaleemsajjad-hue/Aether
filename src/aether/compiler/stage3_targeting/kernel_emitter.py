"""
Kernel emitter.

Produces target-specific kernel invocation plans from the optimized AEG-IR.
Rather than emitting hand-written native code, Aether emits a kernel plan that
the selected backend plugin can execute. This preserves backend independence and
allows each backend to use its own optimized kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.exceptions import KernelError
from aether.targets.registry import TargetRegistry
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KernelPlan:
    """A plan for executing a kernel on a target."""

    target_id: str
    kernel_name: str
    backend: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    launch_params: dict[str, Any] = field(default_factory=dict)
    precision: str = "BF16"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "kernel_name": self.kernel_name,
            "backend": self.backend,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "launch_params": self.launch_params,
            "precision": self.precision,
        }


class KernelEmitter:
    """Emits kernel execution plans for a target."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.registry = TargetRegistry()
        self.target = self.registry.get(target_id)
        if self.target is None:
            msg = f"Target '{target_id}' is not registered"
            raise KernelError(msg, target_id=target_id)

    def emit(self, op_name: str, inputs: list[str], outputs: list[str], attrs: dict[str, Any] | None = None) -> KernelPlan:
        """Emit a kernel plan for a single operation.

        Args:
            op_name: AEG-IR operation name (e.g., 'aeg.attention').
            inputs: Input tensor names.
            outputs: Output tensor names.
            attrs: Operation attributes.

        Returns:
            KernelPlan describing how to execute the operation.
        """
        backend = self.target.backend_candidates[0] if self.target.backend_candidates else "pytorch"
        return KernelPlan(
            target_id=self.target_id,
            kernel_name=op_name,
            backend=backend,
            inputs=inputs,
            outputs=outputs,
            launch_params=attrs or {},
        )

    def emit_graph(self, graph: Any, precision_map: dict[str, str] | None = None) -> list[KernelPlan]:
        """Emit kernel plans for all operations in an AEG-IR graph."""
        plans: list[KernelPlan] = []
        if not hasattr(graph, "functions"):
            return plans
        for func in graph.functions:
            for block in func.blocks:
                for instruction in block.instructions:
                    op_name = str(instruction.op_code.value) if hasattr(instruction.op_code, "value") else str(instruction.op_code)
                    input_names = [i for i in instruction.inputs]
                    output_names = [r.name for r in instruction.results]
                    plan = self.emit(op_name, input_names, output_names, instruction.attributes)
                    plans.append(plan)
        return plans

    def __repr__(self) -> str:
        return f"KernelEmitter({self.target_id})"
