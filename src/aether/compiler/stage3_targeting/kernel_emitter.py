"""
Kernel emitter.

Produces target-specific kernel invocation plans from the optimized AEG-IR.
Rather than emitting hand-written native code, Aether emits a kernel plan that
the selected backend plugin can execute. This preserves backend independence and
allows each backend to use its own optimized kernels.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class KernelArtifact:
    """A compiled, loadable target-kernel artifact."""

    target_id: str
    kernel_name: str
    artifact_path: Path
    sha256: str
    symbols: tuple[str, ...]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "kernel_name": self.kernel_name,
            "artifact_path": str(self.artifact_path),
            "sha256": self.sha256,
            "symbols": list(self.symbols),
            "backend": self.backend,
            "executable": True,
        }


class KernelEmitter:
    """Emits kernel execution plans for a target."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.registry = TargetRegistry()
        self.target = self.registry.get(target_id)
        if self.target is None:
            msg = f"Target '{target_id}' is not registered"
            raise KernelError(msg, details={"target_id": target_id})

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

    def emit_executable(
        self,
        op_name: str,
        output_path: str | Path | None = None,
    ) -> KernelArtifact:
        """Build a loadable kernel library for a supported host target.

        The CPU implementation is the repository's real native kernel backend.
        It exports a fixed, audited symbol set and is compiled through the same
        toolchain/cache used by runtime inference.  Other targets continue to
        require their vendor compiler/runtime and therefore fail closed here;
        returning a :class:`KernelPlan` is not equivalent to generating code.
        """
        if self.target_id.startswith("cuda_"):
            from aether.kernels.native_cuda import get_native_cuda_kernels
            cuda_native = get_native_cuda_kernels(self.target_id)
            if not cuda_native._toolchain:
                raise KernelError(
                    f"CUDA compiler (nvcc) not available for target {self.target_id!r}; "
                    "executable kernel generation is not implemented on this host",
                    details={"target_id": self.target_id},
                )
            if not cuda_native.compile() or cuda_native._library_path is None:
                raise KernelError(
                    f"CUDA kernel compilation failed for target {self.target_id!r}: {cuda_native._build_error or 'unknown error'}",
                    details={"target_id": self.target_id},
                )
            dest = cuda_native._library_path
            if output_path is not None:
                dest = Path(output_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.resolve() != cuda_native._library_path.resolve():
                    shutil.copy2(cuda_native._library_path, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            return KernelArtifact(
                target_id=self.target_id,
                kernel_name=op_name,
                artifact_path=dest,
                sha256=digest,
                symbols=("launch_" + op_name.replace("aeg.", ""),),
                backend="native_cuda",
            )

        if not self.target_id.startswith("cpu_"):
            raise KernelError(
                f"executable kernel generation is not implemented for target {self.target_id!r}; "
                "a registered target profile is not an executable backend",
            )

        from aether.kernels.native_cpu import get_native_kernels

        symbol = _cpu_symbol_for_op(op_name)
        native = get_native_kernels()
        if symbol not in native.available_kernels():
            raise KernelError(
                f"CPU native kernel {symbol!r} is not available",
            )
        if not native.ensure_compiled() or native.library_path is None:
            raise KernelError(
                f"could not compile the CPU native kernel library: {native.build_error or 'unknown error'}",
            )

        source_path = native.library_path
        destination = source_path
        if output_path is not None:
            destination = Path(output_path)
            if destination.exists() and destination.is_dir():
                raise KernelError(
                    f"kernel output must be a file path, not a directory: {destination}",
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.resolve() != source_path.resolve():
                shutil.copy2(source_path, destination)

        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return KernelArtifact(
            target_id=self.target_id,
            kernel_name=op_name,
            artifact_path=destination,
            sha256=digest,
            symbols=(symbol,),
            backend="native_cpu",
        )

    def __repr__(self) -> str:
        return f"KernelEmitter({self.target_id})"


_CPU_OP_SYMBOLS = {
    "rmsnorm": "aether_rmsnorm",
    "aeg.rmsnorm": "aether_rmsnorm",
    "silu": "aether_silu",
    "aeg.silu": "aether_silu",
    "swiglu": "aether_swiglu",
    "aeg.swiglu": "aether_swiglu",
    "softmax": "aether_softmax",
    "aeg.softmax": "aether_softmax",
    "gemm": "aether_sgemm",
    "linear": "aether_sgemm",
    "aeg.linear": "aether_sgemm",
    "rope": "aether_rope",
    "aeg.rope": "aether_rope",
    "argmax": "aether_argmax",
    "aeg.argmax": "aether_argmax",
    "qgemv_affine": "aether_qgemv_affine",
    "dequantize_affine": "aether_dequantize_affine",
    "dequantize_symmetric": "aether_dequantize_symmetric",
}


def _cpu_symbol_for_op(op_name: str) -> str:
    normalized = op_name.strip().lower()
    try:
        return _CPU_OP_SYMBOLS[normalized]
    except KeyError as exc:
        raise KernelError(
            f"no audited native CPU implementation exists for operation {op_name!r}",
        ) from exc
