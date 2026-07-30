"""
OpenVINO NPU targeting pass.

Selects OpenVINO/Intel-specific kernels and backend preferences for Intel NPU targets.
"""

from __future__ import annotations

from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class OpenVINOTarget:
    """OpenVINO-specific targeting configuration."""

    def __init__(self, target_id: str) -> None:
        self.target_id = target_id
        self.emitter = KernelEmitter(target_id)

    def flags(self) -> dict[str, object]:
        """Return OpenVINO-specific compiler flags."""
        return {
            "use_openvino": True,
            "preferred_backend": "onnxruntime",
            "precision": "FP16",
            "supports_int8": True,
        }

    def __repr__(self) -> str:
        return f"OpenVINOTarget({self.target_id})"
