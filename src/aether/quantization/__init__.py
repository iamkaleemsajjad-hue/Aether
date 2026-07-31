"""
Quantization package initialization.
"""

from __future__ import annotations

from aether.quantization.codecs import (
    AffineIntCodec,
    Codec,
    FP4Codec,
    FP8Codec,
    NF4Codec,
    PassthroughCodec,
    SymmetricIntCodec,
    get_codec,
    supported_precisions,
)
from aether.quantization.formats import QuantizationFormat, QuantizedTensor, quantize_tensor, dequantize_tensor
from aether.quantization.quantizer import Quantizer
from aether.quantization.sensitivity import SensitivityScorer
from aether.quantization.assignment import PrecisionAssigner
from aether.quantization.packing import BitPacker
from aether.quantization.pruning import (
    PruningMask,
    apply_mask,
    build_mask,
    build_nm_mask,
    build_unstructured_mask,
    compute_importance,
    verify_nm_pattern,
)

__all__ = [
    "QuantizationFormat",
    "QuantizedTensor",
    "quantize_tensor",
    "dequantize_tensor",
    "Quantizer",
    "SensitivityScorer",
    "PrecisionAssigner",
    "BitPacker",
    # Codecs
    "Codec",
    "AffineIntCodec",
    "SymmetricIntCodec",
    "NF4Codec",
    "FP8Codec",
    "FP4Codec",
    "PassthroughCodec",
    "get_codec",
    "supported_precisions",
    # Pruning and sparsity
    "PruningMask",
    "compute_importance",
    "build_mask",
    "build_nm_mask",
    "build_unstructured_mask",
    "apply_mask",
    "verify_nm_pattern",
]
