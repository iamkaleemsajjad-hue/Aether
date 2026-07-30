"""LoRA runtime fusion and hot-swap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LoRAAdapter:
    """Low-rank adapter matrices for one target module."""

    adapter_id: str
    delta_a: np.ndarray
    delta_b: np.ndarray
    alpha: float = 1.0

    @property
    def rank(self) -> int:
        return int(self.delta_a.shape[1])

    def delta_weight(self) -> np.ndarray:
        scale = self.alpha / max(self.rank, 1)
        return scale * (self.delta_a @ self.delta_b)


class LoRAHotSwapEngine:
    """BGMV-style per-request LoRA adapter hot-swap engine."""

    def __init__(self, base_weight: np.ndarray) -> None:
        self.base_weight = np.asarray(base_weight, dtype=np.float32)
        self.adapter_pool: dict[str, LoRAAdapter] = {}

    def register(self, adapter: LoRAAdapter) -> None:
        if adapter.delta_a.shape[0] != self.base_weight.shape[0]:
            raise ValueError("LoRA A rows must match base output dimension")
        if adapter.delta_b.shape[1] != self.base_weight.shape[1]:
            raise ValueError("LoRA B columns must match base input dimension")
        self.adapter_pool[adapter.adapter_id] = adapter

    def forward(self, inputs: np.ndarray, adapter_ids: list[str | None]) -> np.ndarray:
        batch = np.asarray(inputs, dtype=np.float32)
        if batch.ndim != 2:
            raise ValueError("inputs must be a 2D batch matrix")
        if len(adapter_ids) != batch.shape[0]:
            raise ValueError("adapter_ids length must match batch size")
        outputs = batch @ self.base_weight.T
        for row_index, adapter_id in enumerate(adapter_ids):
            if adapter_id is None:
                continue
            adapter = self.adapter_pool[adapter_id]
            outputs[row_index] += batch[row_index] @ adapter.delta_weight().T
        return outputs

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": "multi_slot",
            "slots": len(self.adapter_pool),
            "adapters": [
                {"id": adapter.adapter_id, "rank": adapter.rank, "alpha": adapter.alpha}
                for adapter in self.adapter_pool.values()
            ],
        }
