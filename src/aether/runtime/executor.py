"""
Runtime executor — dispatches AEG-IR operations to the active backend.

The executor is the bridge between the portable AEG representation and the
concrete kernels provided by the selected backend. It manages execution
contexts, handles memory allocation, and collects per-op profiling signals.
"""

from __future__ import annotations

import contextlib
import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from aether.core.exceptions import RuntimeError as AetherRuntimeError
from aether.core.types import DType, HardwareTarget, TensorLayout, TensorShape
from aether.runtime.config import RuntimeConfig
from aether.runtime.hardware import HardwareDetector
from aether.utils.logging import get_logger
from aether.utils.profiling import OpProfile, Profiler

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """Context for a single execution invocation."""

    request_id: str
    model_id: str
    target: HardwareTarget
    precision: str
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    op_profiles: list[OpProfile] = field(default_factory=list)

    def add_profile(self, profile: OpProfile) -> None:
        """Record an operation profile."""
        self.op_profiles.append(profile)


@dataclass
class Allocation:
    """A tensor allocation managed by the executor."""

    name: str
    layout: TensorLayout
    device: str
    ptr: int | None = None
    value: Any | None = None

    @property
    def size_bytes(self) -> int:
        return self.layout.byte_size


class Executor:
    """Dispatches AEG-IR operations to the active backend.

    The executor maintains a registry of operation implementations and handles
    the lifecycle of tensor allocations. It is intentionally lightweight: the
    heavy lifting is done by the selected backend plugin.
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.hardware = HardwareDetector().detect()
        self.profiler = Profiler(enabled=self.config.enable_memory_profiling)
        self._ops: dict[str, Callable[[ExecutionContext, dict[str, Any], dict[str, Allocation]], Allocation]] = {}
        self._allocations: dict[str, Allocation] = {}
        self._register_default_ops()
        logger.info("Executor initialized", target=self.hardware.target_id)

    def _register_default_ops(self) -> None:
        """Register lightweight CPU-based reference implementations."""
        self._ops["aeg.embedding"] = self._op_embedding
        self._ops["aeg.linear"] = self._op_linear
        self._ops["aeg.rms_norm"] = self._op_rms_norm
        self._ops["aeg.attention"] = self._op_attention
        self._ops["aeg.matmul"] = self._op_matmul
        self._ops["aeg.add"] = self._op_add
        self._ops["aeg.mul"] = self._op_mul
        self._ops["aeg.softmax"] = self._op_softmax
        self._ops["aeg.silu"] = self._op_silu

    def _allocate(
        self,
        name: str,
        layout: TensorLayout,
        device: str = "cpu",
        value: Any | None = None,
    ) -> Allocation:
        """Create a managed allocation."""
        allocation = Allocation(name=name, layout=layout, device=device, value=value)
        self._allocations[name] = allocation
        return allocation

    def _lookup_tensor(self, name: str) -> Any:
        """Return a concrete tensor value for an allocation name.

        In the native runtime this would interface with backend buffers. The
        reference executor carries concrete CPU values when available and
        returns ``None`` for shape-only allocations instead of fabricating a
        zero tensor that could be mistaken for a real result.
        """
        allocation = self._allocations.get(name)
        if allocation is None:
            msg = f"Allocation '{name}' not found"
            raise AetherRuntimeError(msg)
        return allocation.value

    @staticmethod
    def _layout_for_value(value: Any) -> TensorLayout:
        """Build a concrete FP32 layout for a reference-interpreter value."""
        import numpy as np

        array = np.asarray(value)
        return TensorLayout(
            shape=TensorShape.from_list([int(dimension) for dimension in array.shape]),
            dtype=DType.FP32,
        )

    @staticmethod
    def _values(inputs: dict[str, Allocation]) -> list[Any]:
        """Return concrete input values, ignoring shape-only allocations."""
        return [allocation.value for allocation in inputs.values() if allocation.value is not None]

    def _op_embedding(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        import numpy as np

        start = time.perf_counter()
        vocab_size = attrs.get("vocab_size", 32000)
        hidden_size = attrs.get("hidden_size", 4096)
        layout = TensorLayout.from_dict(
            {
                "shape": {"dims": [1, hidden_size]},
                "dtype": "bf16",
                "layout": "dense",
            }
        )
        value = None
        concrete = self._values(inputs)
        weight = attrs.get("weight")
        if concrete and weight is not None:
            value = np.asarray(weight)[np.asarray(concrete[0], dtype=np.int64)]
            layout = self._layout_for_value(value)
        result = self._allocate("emb_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="embedding",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=layout.byte_size,
                attributes={"vocab_size": vocab_size, "hidden_size": hidden_size},
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_linear(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        import numpy as np

        start = time.perf_counter()
        in_features = attrs.get("in_features", 4096)
        out_features = attrs.get("out_features", 4096)
        layout = TensorLayout.from_dict(
            {
                "shape": {"dims": [1, out_features]},
                "dtype": "bf16",
                "layout": "dense",
            }
        )
        value = None
        concrete = self._values(inputs)
        weight = attrs.get("weight")
        bias = attrs.get("bias")
        if weight is None and len(concrete) >= 2:
            weight, concrete = concrete[1], concrete[:1]
        if concrete and weight is not None:
            value = np.matmul(np.asarray(concrete[0]), np.asarray(weight).T)
            if bias is not None:
                value = value + np.asarray(bias)
            layout = self._layout_for_value(value)
        result = self._allocate("linear_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="linear",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=layout.byte_size,
                attributes={"in_features": in_features, "out_features": out_features},
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_rms_norm(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        hidden_size = attrs.get("hidden_size", 4096)
        layout = TensorLayout.from_dict(
            {
                "shape": {"dims": [1, hidden_size]},
                "dtype": "bf16",
                "layout": "dense",
            }
        )
        value = None
        concrete = self._values(inputs)
        if concrete:
            import numpy as np

            tensor = np.asarray(concrete[0], dtype=np.float32)
            epsilon = float(attrs.get("epsilon", 1e-5))
            value = tensor / np.sqrt(np.mean(np.square(tensor), axis=-1, keepdims=True) + epsilon)
            layout = self._layout_for_value(value)
        result = self._allocate("norm_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="rms_norm",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_attention(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        hidden_size = attrs.get("hidden_size", 4096)
        layout = TensorLayout.from_dict(
            {
                "shape": {"dims": [1, hidden_size]},
                "dtype": "bf16",
                "layout": "dense",
            }
        )
        value = None
        concrete = self._values(inputs)
        if len(concrete) >= 3:
            import numpy as np

            query, key, value_tensor = (
                np.asarray(item, dtype=np.float32) for item in concrete[:3]
            )
            scale = float(attrs.get("scale", query.shape[-1] ** -0.5))
            scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
            scores -= np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            value = np.matmul(probabilities, value_tensor)
            layout = self._layout_for_value(value)
        result = self._allocate("attn_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="attention",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=layout.byte_size * 4,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_matmul(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        m = attrs.get("m", 1)
        n = attrs.get("n", 4096)
        layout = TensorLayout.from_dict(
            {
                "shape": {"dims": [m, n]},
                "dtype": "bf16",
                "layout": "dense",
            }
        )
        value = None
        concrete = self._values(inputs)
        if len(concrete) >= 2:
            import numpy as np

            value = np.matmul(np.asarray(concrete[0]), np.asarray(concrete[1]))
            layout = self._layout_for_value(value)
        result = self._allocate("matmul_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="matmul",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_add(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        if not inputs:
            msg = "Add op requires at least one input"
            raise AetherRuntimeError(msg)
        first = next(iter(inputs.values()))
        value = None
        concrete = self._values(inputs)
        if concrete:
            import numpy as np

            value = np.asarray(concrete[0])
            for item in concrete[1:]:
                value = np.add(value, item)
            layout = self._layout_for_value(value)
        result = self._allocate("add_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="add",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=first.layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_mul(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        if not inputs:
            msg = "Mul op requires at least one input"
            raise AetherRuntimeError(msg)
        first = next(iter(inputs.values()))
        value = None
        concrete = self._values(inputs)
        if concrete:
            import numpy as np

            value = np.asarray(concrete[0])
            for item in concrete[1:]:
                value = np.multiply(value, item)
            layout = self._layout_for_value(value)
        result = self._allocate("mul_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="mul",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=first.layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_softmax(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        if not inputs:
            msg = "Softmax op requires input"
            raise AetherRuntimeError(msg)
        first = next(iter(inputs.values()))
        value = None
        concrete = self._values(inputs)
        if concrete:
            import numpy as np

            tensor = np.asarray(concrete[0], dtype=np.float32)
            shifted = tensor - np.max(tensor, axis=-1, keepdims=True)
            exponent = np.exp(shifted)
            value = exponent / np.sum(exponent, axis=-1, keepdims=True)
            layout = self._layout_for_value(value)
        result = self._allocate("softmax_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="softmax",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=first.layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def _op_silu(self, ctx: ExecutionContext, attrs: dict[str, Any], inputs: dict[str, Allocation]) -> Allocation:
        start = time.perf_counter()
        if not inputs:
            msg = "SiLU op requires input"
            raise AetherRuntimeError(msg)
        first = next(iter(inputs.values()))
        value = None
        concrete = self._values(inputs)
        if concrete:
            import numpy as np

            tensor = np.asarray(concrete[0], dtype=np.float32)
            value = tensor / (1.0 + np.exp(-tensor))
            layout = self._layout_for_value(value)
        result = self._allocate("silu_out", layout, value=value)
        self.profiler.record(
            OpProfile(
                op_name="silu",
                duration_ms=(time.perf_counter() - start) * 1000,
                memory_bytes=first.layout.byte_size,
            )
        )
        ctx.add_profile(self.profiler.last)
        return result

    def dispatch(
        self,
        ctx: ExecutionContext,
        op_name: str,
        attrs: dict[str, Any],
        inputs: dict[str, Allocation],
    ) -> Allocation:
        """Dispatch a single operation to the registered handler or backend."""
        handler = self._ops.get(op_name)
        if handler is None:
            msg = f"Operation '{op_name}' is not supported by the executor"
            raise AetherRuntimeError(msg)
        return handler(ctx, attrs, inputs)

    def execute_graph(
        self,
        ctx: ExecutionContext,
        graph: Any,
        inputs: dict[str, Any],
    ) -> dict[str, Allocation]:
        """Execute a simple AEG-IR graph by iterating its instructions.

        This is a reference interpreter used for validation and smoke tests. Real
        inference flows through the active backend plugin.
        """
        outputs: dict[str, Allocation] = {}
        for name, value in inputs.items():
            if isinstance(value, Allocation):
                self._allocations[name] = value
                continue
            try:
                import numpy as np

                concrete = np.asarray(value)
                self._allocate(name, self._layout_for_value(concrete), value=concrete)
            except (TypeError, ValueError) as exc:
                raise AetherRuntimeError(
                    f"Graph input {name!r} is not a concrete tensor value"
                ) from exc
        if not hasattr(graph, "functions"):
            return outputs
        for func in graph.functions:
            for block in func.blocks:
                for instruction in block.instructions:
                    op_name = instruction.op_code.value if hasattr(instruction.op_code, "value") else str(instruction.op_code)
                    op_inputs = {name: outputs.get(name, self._allocations.get(name)) for name in instruction.inputs if name}
                    op_inputs = {k: v for k, v in op_inputs.items() if v is not None}
                    result = self.dispatch(ctx, op_name, instruction.attributes, op_inputs)
                    for res in instruction.results:
                        outputs[res.name] = result
                        self._allocations[res.name] = result
        return outputs

    @contextlib.contextmanager
    def session(self, request_id: str, model_id: str) -> Iterator[ExecutionContext]:
        """Create a scoped execution context."""
        ctx = ExecutionContext(
            request_id=request_id,
            model_id=model_id,
            target=HardwareTarget.from_string(self.hardware.target_id),
            precision="BF16",
        )
        try:
            yield ctx
        finally:
            self._allocations.clear()

    def profile_summary(self) -> dict[str, Any]:
        """Return a summary of recorded operation profiles."""
        return self.profiler.summary()

    def __repr__(self) -> str:
        return f"Executor(target={self.hardware.target_id}, ops={len(self._ops)})"
