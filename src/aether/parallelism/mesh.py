"""
Device mesh and communication topology.

Defines a multi-dimensional device mesh used by tensor, pipeline, expert, and
context parallelism. Supports logical-to-physical device mapping and communication
group discovery.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class DeviceMesh:
    """Represents an N-dimensional logical device mesh.

    Attributes:
        shape: Tuple of mesh dimensions (e.g., (2, 4) for 2 pipeline stages and 4 TP devices).
        device_ids: Flat list of physical device IDs.
    """

    def __init__(self, shape: tuple[int, ...], device_ids: list[int] | None = None) -> None:
        self.shape = shape
        total = 1
        for d in shape:
            total *= d
        if device_ids is None:
            device_ids = list(range(total))
        if len(device_ids) != total:
            msg = f"Device count {len(device_ids)} does not match mesh shape {shape}"
            raise ValueError(msg)
        self.device_ids = device_ids
        self._coords: dict[int, tuple[int, ...]] = {}
        for idx, dev_id in enumerate(device_ids):
            self._coords[dev_id] = self._index_to_coords(idx)

    def _index_to_coords(self, index: int) -> tuple[int, ...]:
        """Convert a flat index to multi-dimensional mesh coordinates."""
        coords = []
        for dim in reversed(self.shape):
            coords.append(index % dim)
            index //= dim
        return tuple(reversed(coords))

    def _coords_to_index(self, coords: tuple[int, ...]) -> int:
        """Convert mesh coordinates to a flat index."""
        index = 0
        stride = 1
        for dim, coord in zip(reversed(self.shape), reversed(coords)):
            index += coord * stride
            stride *= dim
        return index

    def get_device_id(self, coords: tuple[int, ...]) -> int:
        """Return the physical device ID at given logical coordinates."""
        return self.device_ids[self._coords_to_index(coords)]

    def get_coords(self, device_id: int) -> tuple[int, ...]:
        """Return the logical coordinates of a physical device."""
        return self._coords[device_id]

    def get_group_along_axis(self, axis: int, fixed_coords: tuple[int, ...]) -> list[int]:
        """Return the communication group along an axis for fixed coordinates.

        For example, in a (2, 4) mesh with axis=1 (tensor parallel), the group
        includes all 4 devices sharing the same pipeline stage.
        """
        if axis >= len(self.shape):
            return []
        group: list[int] = []
        for pos in range(self.shape[axis]):
            coords = list(fixed_coords) if fixed_coords else [0] * len(self.shape)
            coords[axis] = pos
            group.append(self.get_device_id(tuple(coords)))
        return group

    def get_all_groups_along_axis(self, axis: int) -> list[list[int]]:
        """Return all communication groups along an axis."""
        if axis >= len(self.shape):
            return []
        groups: list[list[int]] = []
        other_dims = [d for i, d in enumerate(self.shape) if i != axis]
        if not other_dims:
            return [self.device_ids]
        total = 1
        for d in other_dims:
            total *= d
        for flat in range(total):
            coords = [0] * len(self.shape)
            remaining = flat
            for i, dim in enumerate(reversed([d for i, d in enumerate(self.shape) if i != axis])):
                idx = len(self.shape) - 1 - i
                if idx == axis:
                    idx -= 1
                coords[idx] = remaining % dim
                remaining //= dim
            groups.append(self.get_group_along_axis(axis, tuple(coords)))
        return groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "device_ids": self.device_ids,
            "total_devices": len(self.device_ids),
        }

    def __repr__(self) -> str:
        return f"DeviceMesh(shape={self.shape}, devices={self.device_ids})"
