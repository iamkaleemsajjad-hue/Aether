"""
Kernel base classes and contracts.

Defines the abstract base class for all kernel implementations and the kernel
dispatch table that maps AEG op codes to concrete kernel implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class Kernel:
    """Abstract base for all kernel implementations."""

    name: str = "kernel"
    supported_formats: list[str] = []

    def __init__(self) -> None:
        self._profiling_enabled: bool = False
        self._metrics: dict[str, float] = {}

    def enable_profiling(self) -> None:
        self._profiling_enabled = True

    def disable_profiling(self) -> None:
        self._profiling_enabled = False

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def compile(self, target: str, **kwargs: Any) -> bytes:
        """Compile this kernel for a specific target.

        Returns compiled kernel bytes.
        """
        msg = f"Kernel '{self.name}' compilation not implemented for target '{target}'"
        raise NotImplementedError(msg)

    def __repr__(self) -> str:
        return f"Kernel({self.name})"
