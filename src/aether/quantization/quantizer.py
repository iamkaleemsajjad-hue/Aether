"""
Quantizer — applies quantization to model weights.

Orchestrates the quantization of model weights given a precision map. This is
used both during compilation (to produce quantized AEG weights) and at runtime
(for dynamic precision adjustment).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.core.constants import PRECISION_SIZES_BYTES
from aether.core.exceptions import QuantizationError
from aether.quantization.formats import QuantizedTensor, quantize_tensor
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class Quantizer:
    """Orchestrates quantization of model weights.

    Applies a precision map to a dictionary of named weight tensors and returns
    a dictionary of quantized tensors.
    """

    def __init__(self, block_size: int = 32) -> None:
        self.block_size = block_size

    def quantize(self, weights: dict[str, np.ndarray], precision_map: dict[str, str]) -> dict[str, QuantizedTensor]:
        """Quantize all weights according to a precision map.

        Args:
            weights: Named weight tensors (e.g., {"model.layers.0.attn.q_proj.weight": ndarray}).
            precision_map: Precision assignment keyed by layer or weight name.

        Returns:
            Dictionary mapping weight names to QuantizedTensors.
        """
        result: dict[str, QuantizedTensor] = {}
        for name, tensor_data in weights.items():
            precision = self._resolve_precision(name, precision_map)
            logger.debug("Quantizing %s -> %s", name, precision)
            try:
                qt = quantize_tensor(tensor_data, precision, self.block_size)
            except Exception as exc:
                msg = f"Failed to quantize '{name}': {exc}"
                raise QuantizationError(msg) from exc
            result[name] = qt
        return result

    def estimate_compression(self, weights: dict[str, np.ndarray], precision_map: dict[str, str]) -> dict[str, Any]:
        """Estimate compression without performing full quantization.

        Returns:
            Dictionary with estimated sizes and ratios.
        """
        original_bytes = sum(w.nbytes for w in weights.values())
        compressed_bytes = 0
        for name, tensor_data in weights.items():
            precision = self._resolve_precision(name, precision_map)
            bits_per_elem = PRECISION_SIZES_BYTES.get(precision.upper(), 2.0) * 8
            compressed_bytes += int(tensor_data.size * bits_per_elem / 8)
        return {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": original_bytes / max(compressed_bytes, 1),
        }

    def _resolve_precision(self, weight_name: str, precision_map: dict[str, str]) -> str:
        """Resolve the precision for a weight name from the precision map.

        Looks up the weight name directly, then by layer prefix, then falls back
        to a default of BF16.
        """
        if weight_name in precision_map:
            return precision_map[weight_name]
        parts = weight_name.split(".")
        for key in precision_map:
            if key in weight_name or weight_name.startswith(key):
                return precision_map[key]
        return "BF16"

    def __repr__(self) -> str:
        return f"Quantizer(block_size={self.block_size})"
