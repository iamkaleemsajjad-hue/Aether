"""
Distributed communication primitives.

Provides lightweight wrappers around collective operations (all-reduce, all-
gather, reduce-scatter) and a device group abstraction. The real implementations
would delegate to PyTorch/NCRCCL/MLX; this module provides the interface.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CommunicationGroup:
    """Group of devices that participate in a collective operation."""

    def __init__(self, group_id: str, device_ids: list[int]) -> None:
        self.group_id = group_id
        self.device_ids = device_ids

    def __repr__(self) -> str:
        return f"CommunicationGroup({self.group_id}, devices={self.device_ids})"


class CollectiveBackend:
    """Base class for collective communication backends.

    Subclasses implement backend-specific collectives (NCRCCL, Gloo, MLX, etc.).
    The default implementation provides CPU-based reference collectives.
    """

    def __init__(self, device_ids: list[int]) -> None:
        self.device_ids = device_ids
        self._groups: dict[str, CommunicationGroup] = {}

    def register_group(self, name: str, device_ids: list[int]) -> CommunicationGroup:
        """Register a communication group."""
        group = CommunicationGroup(name, device_ids)
        self._groups[name] = group
        return group

    def all_reduce(self, tensor: np.ndarray, op: str = "sum", group: str | None = None) -> np.ndarray:
        """All-reduce a tensor."""
        logger.debug("all_reduce", op=op, shape=tensor.shape, group=group)
        return tensor * len(self.device_ids) if op == "sum" else tensor

    def all_gather(self, tensor: np.ndarray, axis: int = 0, group: str | None = None) -> np.ndarray:
        """All-gather tensors along an axis."""
        logger.debug("all_gather", axis=axis, shape=tensor.shape, group=group)
        return np.concatenate([tensor for _ in self.device_ids], axis=axis)

    def reduce_scatter(self, tensor: np.ndarray, axis: int = 0, group: str | None = None) -> np.ndarray:
        """Reduce-scatter a tensor along an axis."""
        logger.debug("reduce_scatter", axis=axis, shape=tensor.shape, group=group)
        split_size = tensor.shape[axis] // max(len(self.device_ids), 1)
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(0, split_size)
        return tensor[tuple(slices)]

    def broadcast(self, tensor: np.ndarray, src: int = 0, group: str | None = None) -> np.ndarray:
        """Broadcast a tensor from a source device."""
        return tensor

    def __repr__(self) -> str:
        return f"CollectiveBackend(devices={self.device_ids})"
