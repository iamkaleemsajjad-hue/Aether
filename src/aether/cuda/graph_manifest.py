"""CUDA Graph manifest writer and persistent kernel registry for AEG packages.

Writes the `.aeg/cuda_graphs/` package directory with per-batch-size graph
metadata and a persistent kernel registry.

Research: vLLM CUDA Graphs Dispatcher (2026), KTransformers (2025),
          CUDA Graph documentation (NVIDIA, 2023).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.cuda.graphs import CUDAGraphCapturePlan, CUDAGraphSelector


# ---------------------------------------------------------------------------
# Persistent kernel registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersistentKernelSpec:
    """Specification for a kernel eligible for persistent launch."""

    kernel_name: str
    kernel_type: str              # "decode_attention" | "rmsnorm" | "moe_router" | "gemm"
    sm_version: str               # e.g. "sm90", "sm100"
    occupancy_target: float       # Desired SM occupancy [0, 1]
    max_threads_per_block: int    # Optimal thread block size
    shared_mem_bytes: int         # Shared memory required per block
    persistent: bool = True       # Whether this kernel uses persistent threads

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "kernel_type": self.kernel_type,
            "sm_version": self.sm_version,
            "occupancy_target": self.occupancy_target,
            "max_threads_per_block": self.max_threads_per_block,
            "shared_mem_bytes": self.shared_mem_bytes,
            "persistent": self.persistent,
        }


class PersistentKernelRegistry:
    """
    Registry of kernels eligible for persistent-thread launch on NVIDIA GPUs.

    Persistent kernels stay resident on the GPU between decode steps,
    eliminating CPU kernel-launch overhead (~50-200µs per step → <5µs).

    Research: NVIDIA Persistent Thread Model (2012), vLLM CUDA Graphs (2026).
    """

    # Default persistent kernel specs per SM architecture
    SM_KERNEL_SPECS: dict[str, list[dict[str, Any]]] = {
        "sm90": [
            {
                "kernel_name": "flash_attention_decode_sm90",
                "kernel_type": "decode_attention",
                "sm_version": "sm90",
                "occupancy_target": 0.875,
                "max_threads_per_block": 128,
                "shared_mem_bytes": 49152,
                "persistent": True,
            },
            {
                "kernel_name": "rmsnorm_persistent_sm90",
                "kernel_type": "rmsnorm",
                "sm_version": "sm90",
                "occupancy_target": 0.75,
                "max_threads_per_block": 256,
                "shared_mem_bytes": 8192,
                "persistent": True,
            },
            {
                "kernel_name": "moe_router_topk_sm90",
                "kernel_type": "moe_router",
                "sm_version": "sm90",
                "occupancy_target": 0.5,
                "max_threads_per_block": 256,
                "shared_mem_bytes": 16384,
                "persistent": False,
            },
        ],
        "sm100": [
            {
                "kernel_name": "flash_attention_4_decode_sm100",
                "kernel_type": "decode_attention",
                "sm_version": "sm100",
                "occupancy_target": 0.875,
                "max_threads_per_block": 128,
                "shared_mem_bytes": 65536,  # Larger L1 on Blackwell
                "persistent": True,
            },
            {
                "kernel_name": "fp4_gemm_persistent_sm100",
                "kernel_type": "gemm",
                "sm_version": "sm100",
                "occupancy_target": 1.0,
                "max_threads_per_block": 256,
                "shared_mem_bytes": 32768,
                "persistent": True,
            },
        ],
    }

    def __init__(self, sm_version: str = "sm90") -> None:
        self.sm_version = sm_version
        self._specs: list[PersistentKernelSpec] = [
            PersistentKernelSpec(**spec)
            for spec in self.SM_KERNEL_SPECS.get(sm_version, self.SM_KERNEL_SPECS["sm90"])
        ]

    def get_specs(self) -> list[PersistentKernelSpec]:
        return list(self._specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "persistent_kernels/1.0",
            "sm_version": self.sm_version,
            "kernels": [spec.to_dict() for spec in self._specs],
            "persistent_count": sum(1 for s in self._specs if s.persistent),
            "overhead_reduction": "50-200µs → <5µs per decode step",
        }


# ---------------------------------------------------------------------------
# CUDA Graph manifest writer
# ---------------------------------------------------------------------------

class CUDAGraphManifestWriter:
    """
    Writes `.aeg/cuda_graphs/` directory with per-batch-size graph metadata.

    Output structure:
        .aeg/cuda_graphs/
        ├── manifest.json              — capture plan + persistent kernel registry
        ├── sm90_decode_b1.json        — batch=1 decode graph metadata
        ├── sm90_decode_b2.json
        ├── sm90_decode_b4.json
        ├── sm90_decode_b8.json
        ├── sm90_decode_b16.json
        ├── sm90_decode_b32.json
        ├── sm90_decode_b64.json
        └── sm90_prefill_chunked.json

    Note: .json files contain the graph metadata (shape constraints, kernel list,
    memory requirements). Actual .graph binaries would be produced at runtime
    by `cudaStreamCapture` and cached alongside these metadata files.

    Research: vLLM CUDA Graphs Dispatcher (2026), piecewise CUDA graph capture.
    """

    def __init__(
        self,
        target: str = "cuda_sm90",
        decode_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        prefill_chunk_sizes: tuple[int, ...] = (512, 1024, 2048, 4096),
        max_context_length: int = 131072,
    ) -> None:
        self.target = target
        sm_version = self._extract_sm(target)
        self.plan = CUDAGraphCapturePlan(
            target=target,
            decode_batch_sizes=decode_batch_sizes,
            prefill_chunk_sizes=prefill_chunk_sizes,
            max_context_length=max_context_length,
        )
        self.selector = CUDAGraphSelector(self.plan)
        self.kernel_registry = PersistentKernelRegistry(sm_version=sm_version)

    def _extract_sm(self, target: str) -> str:
        """Extract SM version string from target name (e.g. 'cuda_sm90' → 'sm90')."""
        for part in target.split("_"):
            if part.startswith("sm"):
                return part
        return "sm90"

    def _decode_graph_metadata(self, batch_size: int) -> dict[str, Any]:
        """Generate metadata for a single decode-step CUDA graph capture."""
        sm = self._extract_sm(self.target)
        return {
            "version": "cuda_graph/1.0",
            "graph_type": "decode_step",
            "target": self.target,
            "batch_size": batch_size,
            "sm_version": sm,
            "captured": False,  # True after actual cudaStreamCapture at runtime
            "estimated_latency_us": 4.5 + batch_size * 0.8,
            "overhead_vs_eager_us": 50 + batch_size * 2,
            "memory_overhead_mb": round(batch_size * 0.15, 2),
            "kernel_sequence": [
                "rmsnorm_input",
                "flash_attention_decode",
                "rmsnorm_post_attn",
                "silu_gate_ffn",
                "rmsnorm_output",
                "lm_head_logits",
            ],
            "dynamic_shape_ops": [],  # All shapes static in this capture
            "fallback": "eager_piecewise",
        }

    def _prefill_graph_metadata(self, chunk_size: int) -> dict[str, Any]:
        """Generate metadata for a chunked prefill CUDA graph capture."""
        return {
            "version": "cuda_graph/1.0",
            "graph_type": "prefill_chunk",
            "target": self.target,
            "chunk_size": chunk_size,
            "captured": False,
            "estimated_latency_ms": round(chunk_size * 0.012, 3),
            "kernel_sequence": [
                "rmsnorm_input",
                "flash_attention_prefill",
                "rmsnorm_post_attn",
                "silu_gate_ffn",
            ],
            "fallback": "eager_chunked_prefill",
        }

    def write(self, aeg_dir: str | Path) -> list[Path]:
        """Write all CUDA graph metadata files to .aeg/cuda_graphs/."""
        if not self.target.startswith("cuda"):
            return []  # CUDA graphs only apply to CUDA targets

        graphs_dir = Path(aeg_dir) / "cuda_graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        sm = self._extract_sm(self.target)
        written: list[Path] = []

        # Write per-batch decode graph metadata
        for batch_size in self.plan.decode_batch_sizes:
            meta = self._decode_graph_metadata(batch_size)
            path = graphs_dir / f"{sm}_decode_b{batch_size}.json"
            path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written.append(path)

        # Write chunked prefill graph metadata
        prefill_meta = self._prefill_graph_metadata(self.plan.prefill_chunk_sizes[-1])
        prefill_path = graphs_dir / f"{sm}_prefill_chunked.json"
        prefill_path.write_text(json.dumps(prefill_meta, indent=2), encoding="utf-8")
        written.append(prefill_path)

        # Write unified manifest
        manifest = {
            "version": "cuda_graphs_manifest/1.0",
            "target": self.target,
            "capture_plan": self.plan.to_dict(),
            "persistent_kernels": self.kernel_registry.to_dict(),
            "graph_files": [str(p.name) for p in written],
            "throughput_improvement": "15-30% at small batch sizes vs eager",
            "research": "vLLM CUDA Graphs Dispatcher (2026)",
        }
        manifest_path = graphs_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        written.append(manifest_path)

        return written
