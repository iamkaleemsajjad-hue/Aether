"""
Target kernel templates.

Contains platform-agnostic template strings that can be adapted to CUDA,
Metal, ROCm, and CPU targets. These are not hand-optimized kernels; they are
used by the backend selector to describe the expected kernel shape.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# Simple Triton-like template for a fused attention-like operation
FUSED_ATTENTION_TEMPLATE = """
# Fused attention kernel template
# Target: {target_id}
# Inputs: {inputs}
# Output: {output}

import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vk, stride_vn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Simplified matmul attention stub
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        q = tl.load(q_ptr + pid_m * BLOCK_M * stride_qm + pid_h * stride_qh)
        k = tl.load(k_ptr + start_n * stride_kn + pid_h * stride_kh)
        acc += tl.dot(q, k)
    tl.store(out_ptr + pid_m * BLOCK_M * stride_qm + pid_h * stride_qh, acc)
"""

# Linear kernel template for quantized GEMM
QUANTIZED_GEMM_TEMPLATE = """
# Quantized GEMM kernel template
# Target: {target_id}
# Precision: {precision}

def quantized_gemm_reference(input, weight, scales, zero_points):
    # Dequantize weight block-wise then multiply
    dequantized = weight.astype(float) * scales
    return input @ dequantized
"""


class TemplateLibrary:
    """Library of named kernel templates."""

    TEMPLATES: dict[str, str] = {
        "fused_attention": FUSED_ATTENTION_TEMPLATE,
        "quantized_gemm": QUANTIZED_GEMM_TEMPLATE,
    }

    @classmethod
    def render(cls, template_name: str, **kwargs: Any) -> str:
        """Render a template with substitutions."""
        template = cls.TEMPLATES.get(template_name, "")
        return template.format(**kwargs)

    @classmethod
    def list_templates(cls) -> list[str]:
        """Return all template names."""
        return sorted(cls.TEMPLATES.keys())
