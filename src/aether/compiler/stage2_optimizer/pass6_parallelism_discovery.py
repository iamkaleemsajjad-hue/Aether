"""
Pass 6: Automatic Parallelism Discovery.

Searches over the parallelism strategy space and produces separate prefill and
decode sharding plans. These plans are stored in the AEG artifact for zero-
configuration multi-GPU deployment.

The implementation lives in :mod:`aether.compiler.stage2_optimizer.optimizer`;
this module is the stable public entry point for the pass.
"""

from __future__ import annotations

from aether.compiler.stage2_optimizer.optimizer import ParallelismDiscoveryPass

__all__ = ["ParallelismDiscoveryPass"]


