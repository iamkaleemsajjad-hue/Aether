"""
Target registry for kernel templates and target-specific settings.

Maintains a registry of target identifiers and their associated kernel
templates, backend preferences, and compilation flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import BACKEND_BY_TARGET, SUPPORTED_TARGETS
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TargetInfo:
    """Information about a hardware target."""

    target_id: str
    display_name: str
    vendor: str
    backend_candidates: list[str] = field(default_factory=list)
    kernel_templates: list[str] = field(default_factory=list)
    compiler_flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "display_name": self.display_name,
            "vendor": self.vendor,
            "backend_candidates": self.backend_candidates,
            "kernel_templates": self.kernel_templates,
            "compiler_flags": self.compiler_flags,
        }


class TargetRegistry:
    """Registry of supported hardware targets."""

    def __init__(self) -> None:
        self._targets: dict[str, TargetInfo] = {}
        self._register_builtin_targets()

    def _register_builtin_targets(self) -> None:
        """Register all built-in targets from constants."""
        vendors = {
            "cuda": "NVIDIA",
            "metal": "Apple",
            "rocm": "AMD",
            "openvino": "Intel",
            "cpu": "CPU",
        }
        for target_id, display_name in SUPPORTED_TARGETS.items():
            vendor = next((v for prefix, v in vendors.items() if target_id.startswith(prefix)), "Unknown")
            self._targets[target_id] = TargetInfo(
                target_id=target_id,
                display_name=display_name,
                vendor=vendor,
                backend_candidates=BACKEND_BY_TARGET.get(target_id, ["pytorch"]),
                kernel_templates=[],
            )

    def register(self, info: TargetInfo) -> None:
        """Register a custom target."""
        self._targets[info.target_id] = info

    def get(self, target_id: str) -> TargetInfo | None:
        """Return target info by ID."""
        return self._targets.get(target_id)

    def list_targets(self) -> list[str]:
        """Return all registered target IDs."""
        return sorted(self._targets.keys())

    def __repr__(self) -> str:
        return f"TargetRegistry(targets={len(self._targets)})"
