"""
Backend selection logic.

Chooses the best available backend plugin for a given hardware profile and AEG
artifact. The backend is selected based on priority, availability, and model
characteristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether.backends.base import Backend
from aether.backends.registry import BackendRegistry
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.core.exceptions import BackendNotAvailableError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class BackendSelector:
    """Selects the best backend for a given hardware profile and model."""

    def __init__(self) -> None:
        self.registry = BackendRegistry()

    def select(
        self,
        profile: HardwareProfile,
        aeg: Any | None = None,
        model_characteristics: dict[str, Any] | None = None,
    ) -> Backend:
        """Select the best available backend for a hardware profile.

        Args:
            profile: Hardware profile.
            aeg: Optional loaded AEG package, manifest, or package path. Its
                target and backend contracts constrain execution selection.
            model_characteristics: Optional model characteristics (e.g., is_moe, size).

        Returns:
            The best available Backend instance.

        Raises:
            BackendNotAvailableError: If no suitable backend is available.
        """
        package = None
        if aeg is not None:
            from aether.core.aeg_format import AEGManifest, AEGPackage

            if isinstance(aeg, AEGPackage):
                package = aeg
            elif isinstance(aeg, AEGManifest):
                package = AEGPackage(Path("."))
                package.manifest = aeg
            elif isinstance(aeg, (str, Path)):
                package = AEGPackage(aeg).load()
            else:
                raise TypeError("aeg must be an AEGPackage, AEGManifest, or package path")
            if not package.supports_runtime_target(profile.target_id):
                targets = package.manifest.kernels.targets if package.manifest else []
                raise BackendNotAvailableError(
                    f"AEG artifact targets {targets} have no compatible execution contract "
                    f"for {profile.target_id}",
                    target_id=profile.target_id,
                )

        candidates = profile.recommended_backend or "pytorch"
        if not isinstance(candidates, list):
            candidates = [candidates]
        # Extend with target's candidate list from constants
        from aether.core.constants import BACKEND_BY_TARGET

        candidates = BACKEND_BY_TARGET.get(profile.target_id, candidates) + [
            c for c in candidates if c not in BACKEND_BY_TARGET.get(profile.target_id, [])
        ]
        allowed: set[str] | None = None
        if package is not None and package.manifest is not None:
            kernels = package.manifest.kernels
            allowed = {BackendRegistry._normalize_name(name) for name in kernels.portable_backends}
            if profile.target_id.startswith("cpu_") and package.supports_declared_runtime_target(
                profile.target_id
            ):
                allowed.add("aether_cpu")
            plan = package.get_backend_plan(profile.target_id)
            if plan and package.supports_declared_runtime_target(profile.target_id):
                allowed.add(BackendRegistry._normalize_name(plan))
                candidates = [plan, *candidates]
            # CPU reference execution is the packaged CPU contract. Frontend
            # plugins such as llama.cpp/vLLM cannot consume arbitrary AEGs
            # merely because their package happens to be installed.
            if profile.target_id.startswith("cpu_") and "aether_cpu" in allowed:
                candidates = ["aether_cpu", *candidates]

        # Remove duplicates preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        for backend_name in unique:
            if allowed is not None and BackendRegistry._normalize_name(backend_name) not in allowed:
                continue
            backend = self.registry.get_backend(backend_name)
            if backend is not None and backend.available_for_target(profile.target_id):
                logger.info(f"Selected backend '{backend_name}' for target {profile.target_id}")
                return backend

        msg = (
            f"No backend available for target {profile.target_id}. "
            f"Tried: {unique}. Install one of the backend packages."
        )
        raise BackendNotAvailableError(
            msg, backend_name=", ".join(unique), target_id=profile.target_id
        )

    def select_for_target_id(
        self,
        target_id: str,
        aeg: Any | None = None,
        model_characteristics: dict[str, Any] | None = None,
    ) -> Backend:
        """Select the best backend for a target ID string."""
        from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile

        profile = HardwareProfile.from_target_id(target_id)
        if profile is None:
            msg = f"Unknown target: {target_id}"
            raise ValueError(msg)
        return self.select(profile, aeg, model_characteristics)
