"""
Aether exception hierarchy.

All exceptions raised by Aether are rooted in `AetherError` (or `AEGError` for
AEG-specific problems). This module provides a stable, granular set of error
types so callers can catch exactly what they care about.
"""

from __future__ import annotations

from typing import Any


class AetherError(Exception):
    """Base exception for all Aether errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class AEGError(AetherError):
    """Base exception for AEG format and IR errors."""


class AEGFormatError(AEGError):
    """Raised when an AEG file or manifest is malformed."""


class AEGVersionError(AEGError):
    """Raised when an AEG version is not supported by the runtime."""

    def __init__(
        self,
        message: str,
        *,
        aeg_version: str | None = None,
        minimum_version: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.aeg_version = aeg_version
        self.minimum_version = minimum_version


class AEGIntegrityError(AEGError):
    """Raised when an AEG content hash verification fails."""

    def __init__(
        self,
        message: str,
        *,
        file_path: str | None = None,
        expected_hash: str | None = None,
        actual_hash: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.file_path = file_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class IngestionError(AetherError):
    """Raised when model ingestion fails."""


class UnsupportedFormatError(IngestionError):
    """Raised when a model format is not supported."""


class ArchitectureDetectionError(IngestionError):
    """Raised when architecture detection fails."""


class GraphTraceError(IngestionError):
    """Raised when symbolic graph tracing fails."""


class CompilationError(AetherError):
    """Raised when the compiler fails to produce a valid AEG."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.model_id = model_id
        self.stage = stage


class CompilerPassError(CompilationError):
    """Raised when a specific optimizer pass fails."""

    def __init__(
        self,
        message: str,
        *,
        pass_name: str | None = None,
        model_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, model_id=model_id, stage=pass_name, details=details)
        self.pass_name = pass_name


class CompilerConfigError(AetherError):
    """Raised when compiler configuration is invalid."""


class CalibrationError(CompilationError):
    """Raised when calibration or sensitivity evaluation fails."""


class TargetingError(AetherError):
    """Raised when hardware targeting or kernel emission fails."""


class UnsupportedTargetError(TargetingError):
    """Raised when a target ID is not supported."""


class KernelError(TargetingError):
    """Raised when kernel compilation, caching, or loading fails."""


class KernelCacheError(KernelError):
    """Raised when the local or remote kernel cache fails."""


class BackendError(AetherError):
    """Raised when a backend plugin encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        backend_name: str | None = None,
        target_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.backend_name = backend_name
        self.target_id = target_id


class BackendNotAvailableError(BackendError):
    """Raised when a requested backend is not installed."""


class RuntimeError(AetherError):
    """Raised when the Aether runtime fails."""


class RuntimeConfigError(AetherError):
    """Raised when runtime configuration is invalid."""


class ModelNotFoundError(RuntimeError):
    """Raised when a requested model is not found locally or on the Hub."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.model_id = model_id


class ModelAlreadyLoadedError(RuntimeError):
    """Raised when a model is already loaded and cannot be reloaded."""


class SchedulingError(RuntimeError):
    """Raised when the request scheduler cannot place a request."""


class KVCacheError(RuntimeError):
    """Raised when KV cache management fails."""


class KVCacheTierError(KVCacheError):
    """Raised when KV cache tiering/offload fails."""


class SpeculativeDecodingError(RuntimeError):
    """Raised when the speculative decoding engine fails."""


class PrecisionAdjustmentError(RuntimeError):
    """Raised when dynamic precision adjustment fails."""


class ModelNotFoundError(RuntimeError):
    """Raised when a requested model is not loaded in the registry."""

    def __init__(self, message: str, model_id: str | None = None) -> None:
        super().__init__(message)
        self.model_id = model_id


class QuantizationError(AetherError):
    """Raised when quantization or dequantization fails."""


class PrecisionAssignmentError(QuantizationError):
    """Raised when precision assignment cannot satisfy the quality budget."""


class HubError(AetherError):
    """Raised when the Aether Hub client fails."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.url = url
        self.status_code = status_code


class AuthenticationError(HubError):
    """Raised when Hub authentication fails."""


class ParallelismError(AetherError):
    """Raised when automatic parallelism discovery or runtime sharding fails."""


class DistributedError(ParallelismError):
    """Raised when multi-node communication fails."""


class ServerError(AetherError):
    """Raised when the REST server encounters an error."""


class ValidationError(AetherError):
    """Raised when input validation fails."""


class ConfigurationError(AetherError):
    """Raised when a configuration file or environment variable is invalid."""


class BenchmarkError(AetherError):
    """Raised when the benchmark harness fails."""


class TimeoutError(AetherError):
    """Raised when an operation exceeds its time budget."""


class ResourceExhaustedError(AetherError):
    """Raised when a resource (memory, disk, queue) is exhausted."""


class CancelledError(AetherError):
    """Raised when an operation is cancelled by the user or system."""
