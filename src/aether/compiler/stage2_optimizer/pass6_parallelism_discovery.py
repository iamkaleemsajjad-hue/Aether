"""
Pass 6: Automatic Parallelism Discovery.

Searches over the parallelism strategy space and produces separate prefill and
decode sharding plans. These plans are stored in the AEG artifact for zero-
configuration multi-GPU deployment.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import ParallelismDiscoveryPass
from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.core.exceptions import CompilerPassError
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ParallelismDiscoveryPass"]


class ParallelismDiscoveryPass(ParallelismDiscoveryPass):
    """Exported alias of the parallelism discovery pass."""
