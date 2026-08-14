"""
Aether Runtime — Collective Communication Backend Abstraction

Provides a clear type hierarchy that honestly labels what each collective
implementation can and cannot do on the current host:

  SocketCollectiveBackend     — CPU reference via stdlib sockets (always works)
  NCCLCollectiveBackend       — NVIDIA NCCL (requires CUDA multi-GPU)
  RCCLCollectiveBackend       — AMD RCCL (requires ROCm multi-GPU)
  PlaceholderCollectiveBackend— Fail-closed stub for unimplemented backends

Usage:
    from aether.parallelism.collective_backends import get_collective_backend
    backend = get_collective_backend("socket")   # always safe
    backend = get_collective_backend("nccl")     # raises if CUDA unavailable
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class CollectiveBackendError(RuntimeError):
    """Raised when a collective backend cannot be initialized on this host."""


class CollectiveBackend(ABC):
    """Abstract base for all collective communication backends."""

    #: Human-readable label for this backend's actual implementation mode.
    mode: str = "unknown"
    #: True if this backend can support multi-GPU/multi-node production workloads.
    production_capable: bool = False

    @abstractmethod
    def initialize(self, rank: int, world_size: int, **kwargs: Any) -> None:
        """Initialize the collective communication group."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources held by the collective."""

    @abstractmethod
    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        """All-reduce across all ranks."""

    @abstractmethod
    def broadcast(self, data: Any, src: int = 0) -> Any:
        """Broadcast from src rank to all ranks."""

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all ranks."""

    @property
    def description(self) -> str:
        """One-line description of this backend's capabilities and limitations."""
        return f"{self.__class__.__name__}(mode={self.mode!r}, production={self.production_capable})"


class SocketCollectiveBackend(CollectiveBackend):
    """CPU reference collective using stdlib sockets.

    Always available. Suitable for single-machine multi-process testing.
    NOT suitable for production multi-GPU inference due to bandwidth constraints.
    """

    mode = "cpu_socket_mp"
    production_capable = False

    def __init__(
        self,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> None:
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialized = False
        self._rank = 0
        self._world_size = 1

    def initialize(self, rank: int, world_size: int, **kwargs: Any) -> None:
        from aether.parallelism.distributed import SocketCollective
        self._rank = rank
        self._world_size = world_size
        if world_size > 1:
            self._collective = SocketCollective(
                rank=rank,
                world_size=world_size,
                master_addr=self.master_addr,
                master_port=self.master_port,
            )
            self._collective.initialize()
        self._initialized = True
        logger.info("SocketCollectiveBackend initialized", rank=rank, world_size=world_size)

    def shutdown(self) -> None:
        if self._initialized and hasattr(self, "_collective"):
            try:
                self._collective.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self._initialized = False

    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        if not self._initialized or self._world_size == 1:
            return data
        return self._collective.all_reduce(data, op=op)

    def broadcast(self, data: Any, src: int = 0) -> Any:
        if not self._initialized or self._world_size == 1:
            return data
        return self._collective.broadcast(data, src=src)

    def barrier(self) -> None:
        if self._initialized and self._world_size > 1:
            self._collective.barrier()


class NCCLCollectiveBackend(CollectiveBackend):
    """NVIDIA NCCL collective backend.

    Requires:
      - CUDA-capable GPU hardware
      - torch.distributed with NCCL build
      - Multiple GPU devices for meaningful parallelism

    Raises CollectiveBackendError at initialize() time if unavailable.
    """

    mode = "nccl_multi_gpu"
    production_capable = True

    def __init__(self) -> None:
        self._available = self._check_availability()
        self._process_group: Any = None

    @staticmethod
    def _check_availability() -> bool:
        try:
            import torch.distributed as _dist
            return _dist.is_nccl_available()
        except Exception:  # noqa: BLE001
            return False

    @property
    def available(self) -> bool:
        return self._available

    def initialize(self, rank: int, world_size: int, **kwargs: Any) -> None:
        if not self._available:
            raise CollectiveBackendError(
                "NCCL backend unavailable on this host. "
                "Requires CUDA GPU hardware and torch.distributed with NCCL. "
                "Use SocketCollectiveBackend for CPU-only environments."
            )
        import torch
        import torch.distributed as dist
        if not torch.cuda.is_available():
            raise CollectiveBackendError(
                "NCCL requires CUDA hardware; torch.cuda.is_available() = False."
            )
        init_method = kwargs.get("init_method", f"tcp://127.0.0.1:{kwargs.get('master_port', 29500)}")
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        logger.info("NCCLCollectiveBackend initialized", rank=rank, world_size=world_size)

    def shutdown(self) -> None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass

    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        import torch
        import torch.distributed as dist
        dist.all_reduce(data, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX)
        return data

    def broadcast(self, data: Any, src: int = 0) -> Any:
        import torch.distributed as dist
        dist.broadcast(data, src=src)
        return data

    def barrier(self) -> None:
        import torch.distributed as dist
        dist.barrier()


class RCCLCollectiveBackend(CollectiveBackend):
    """AMD RCCL collective backend.

    Requires:
      - AMD ROCm GPU hardware
      - torch.distributed with RCCL (via ROCm build of PyTorch)

    Raises CollectiveBackendError at initialize() time if unavailable.
    """

    mode = "rccl_multi_gpu"
    production_capable = True

    def __init__(self) -> None:
        self._available = self._check_availability()

    @staticmethod
    def _check_availability() -> bool:
        try:
            import torch
            return torch.version.hip is not None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return False

    @property
    def available(self) -> bool:
        return self._available

    def initialize(self, rank: int, world_size: int, **kwargs: Any) -> None:
        if not self._available:
            raise CollectiveBackendError(
                "RCCL backend unavailable on this host. "
                "Requires AMD ROCm GPU hardware and ROCm-build of PyTorch. "
                "Use SocketCollectiveBackend for CPU-only environments."
            )
        import torch.distributed as dist
        init_method = kwargs.get("init_method", f"tcp://127.0.0.1:{kwargs.get('master_port', 29500)}")
        dist.init_process_group(backend="nccl", init_method=init_method, rank=rank, world_size=world_size)
        logger.info("RCCLCollectiveBackend initialized", rank=rank, world_size=world_size)

    def shutdown(self) -> None:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass

    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        import torch.distributed as dist
        dist.all_reduce(data)
        return data

    def broadcast(self, data: Any, src: int = 0) -> Any:
        import torch.distributed as dist
        dist.broadcast(data, src=src)
        return data

    def barrier(self) -> None:
        import torch.distributed as dist
        dist.barrier()


class PlaceholderCollectiveBackend(CollectiveBackend):
    """Fail-closed stub for unimplemented or unsupported backends.

    Calling initialize() always raises CollectiveBackendError.
    Used when a backend name is recognized but has no implementation.
    """

    mode = "placeholder_unsupported"
    production_capable = False

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def initialize(self, rank: int, world_size: int, **kwargs: Any) -> None:
        raise CollectiveBackendError(
            f"Backend {self.name!r} is not implemented: {self.reason}"
        )

    def shutdown(self) -> None:
        pass

    def all_reduce(self, data: Any, op: str = "sum") -> Any:
        raise CollectiveBackendError(f"Backend {self.name!r} not initialized.")

    def broadcast(self, data: Any, src: int = 0) -> Any:
        raise CollectiveBackendError(f"Backend {self.name!r} not initialized.")

    def barrier(self) -> None:
        raise CollectiveBackendError(f"Backend {self.name!r} not initialized.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[CollectiveBackend]] = {
    "socket": SocketCollectiveBackend,
    "nccl": NCCLCollectiveBackend,
    "rccl": RCCLCollectiveBackend,
    "rocm": RCCLCollectiveBackend,
}


def get_collective_backend(name: str, **kwargs: Any) -> CollectiveBackend:
    """Return the collective backend for the given name.

    Args:
        name: One of "socket", "nccl", "rccl", "rocm".
        **kwargs: Passed to the backend constructor (e.g. master_addr, master_port).

    Returns:
        A CollectiveBackend instance. For NCCL/RCCL, call .initialize() before use —
        it will raise CollectiveBackendError if hardware is unavailable.

    Raises:
        CollectiveBackendError: If name is unknown.
    """
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise CollectiveBackendError(
            f"Unknown collective backend: {name!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    backend = cls(**{k: v for k, v in kwargs.items() if k in ("master_addr", "master_port", "name", "reason")})
    logger.debug("Collective backend created", backend=name, mode=backend.mode, production=backend.production_capable)
    return backend


__all__ = [
    "CollectiveBackend",
    "CollectiveBackendError",
    "NCCLCollectiveBackend",
    "PlaceholderCollectiveBackend",
    "RCCLCollectiveBackend",
    "SocketCollectiveBackend",
    "get_collective_backend",
]
