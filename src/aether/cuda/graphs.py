"""CUDA Graph capture and persistent-kernel selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CUDAGraphCapturePlan:
    """Piecewise CUDA graph capture manifest."""

    target: str
    decode_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    prefill_chunk_sizes: tuple[int, ...] = (512, 1024, 2048, 4096)
    max_context_length: int = 131072
    persistent_kernels: tuple[str, ...] = ("decode_attention", "rmsnorm", "moe_router")
    dynamic_shape_fallback: str = "eager_piecewise"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "cuda_graph_capture/1.0",
            "target": self.target,
            "enabled": self.target.startswith("cuda"),
            "piecewise": True,
            "decode_batch_sizes": list(self.decode_batch_sizes),
            "prefill_chunk_sizes": list(self.prefill_chunk_sizes),
            "max_context_length": self.max_context_length,
            "persistent_kernels": list(self.persistent_kernels),
            "dynamic_shape_fallback": self.dynamic_shape_fallback,
        }


class CUDAGraphSelector:
    """Select captured graph buckets by rounding dynamic shapes upward."""

    def __init__(self, plan: CUDAGraphCapturePlan) -> None:
        self.plan = plan
        self._decode_sizes = tuple(sorted(set(plan.decode_batch_sizes)))
        self._prefill_sizes = tuple(sorted(set(plan.prefill_chunk_sizes)))

    def select_decode_batch(self, batch_size: int) -> int | None:
        return self._round_up(batch_size, self._decode_sizes)

    def select_prefill_chunk(self, token_count: int) -> int | None:
        return self._round_up(token_count, self._prefill_sizes)

    def select_graph(self, phase: str, size: int) -> dict[str, Any]:
        if phase == "decode":
            bucket = self.select_decode_batch(size)
        elif phase == "prefill":
            bucket = self.select_prefill_chunk(size)
        else:
            raise ValueError("phase must be 'decode' or 'prefill'")
        if bucket is None:
            return {"phase": phase, "size": size, "mode": self.plan.dynamic_shape_fallback, "bucket": None}
        return {"phase": phase, "size": size, "mode": "cuda_graph_replay", "bucket": bucket}

    def manifest(self) -> dict[str, Any]:
        return self.plan.to_dict()

    def _round_up(self, value: int, buckets: tuple[int, ...]) -> int | None:
        if value <= 0:
            return None
        for bucket in buckets:
            if value <= bucket:
                return bucket
        return None
