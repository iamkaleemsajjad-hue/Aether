"""
base_pass.py — Abstract base class for all Aether optimizer passes.

Kept in a separate module to avoid circular imports: passes import
BasePass from here, and optimizer.py imports passes after its own
BasePass definition (which is now just a re-export for compatibility).
"""

from __future__ import annotations

from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport


class BasePass:
    """Base class for all Stage 2 optimizer passes.

    Each pass must implement ``run()`` which takes a computation graph and
    returns an (optimized_graph, PassReport) tuple.

    Rules:
      - ``run()`` must NEVER raise an exception. Catch all errors internally
        and return ``status="failed"`` in the PassReport.
      - When a pass is disabled via config, return ``status="skipped"``.
      - Write all AEG artifacts inside ``run()`` only when ``status="ok"``.
    """

    name: str = "base"
    description: str = "Base optimizer pass."

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        """Execute the pass.

        Args:
            graph: Input computation graph or AEG-IR module.
            architecture: Model architecture metadata dict.
            config: Compiler configuration.

        Returns:
            Tuple of (optimized_graph, PassReport).
        """
        raise NotImplementedError(
            f"Pass {self.__class__.__name__} must implement run()."
        )
