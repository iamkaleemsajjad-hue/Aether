"""Artifact-wide deployment policy, enforced before device/backend selection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from aether.core.exceptions import AEGError, BackendError
from aether.core.hash_utils import verify_file_hash

if TYPE_CHECKING:
    from aether.core.aeg_format import AEGPackage


class ArtifactValidationError(BackendError):
    """Permanent artifact invalidity; backend fallback cannot repair its bytes."""


class ArtifactEvaluationError(ArtifactValidationError):
    """A permanent artifact rejection that trying another backend cannot repair."""


def verify_aeg_integrity(package: AEGPackage, *, backend_name: str | None = None) -> None:
    """Preserve artifact integrity errors across automatic provider fallback."""
    try:
        package.verify_integrity()
    except (AEGError, OSError, ValueError, TypeError) as exc:
        raise ArtifactValidationError(
            f"AEG artifact failed integrity validation: {exc}", backend_name=backend_name
        ) from exc


def validate_aeg_evaluation(package: AEGPackage, *, backend_name: str | None = None) -> None:
    """Check persisted evaluation policy without loading weights or initializing devices.

    A missing evaluation is uncertified, not a failed evaluation. A recorded
    failure or malformed declared gate cannot become runnable through fallback.
    Full package integrity remains the loader's responsibility; this boundary
    verifies the gate's declared payload hash before interpreting its contents.
    """
    relative = "observability/eval_report.json"
    path = package.root / relative
    expected = package.manifest.artifacts.get(relative)
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(package.root.resolve()):
            raise ValueError("evaluation report path escapes the artifact or is symlinked")
        if not path.is_file():
            if expected is not None:
                raise ValueError("declared evaluation report is missing")
            gate = {}
        else:
            if expected is not None and not verify_file_hash(path, expected):
                raise ValueError("evaluation report failed its declared checksum")
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or not isinstance(report.get("gate", {}), dict):
                raise ValueError("evaluation report and gate must be objects")
            gate = report.get("gate", {})
            passed = gate.get("passed")
            if passed is not None and type(passed) is not bool:
                raise ValueError("evaluation gate passed must be boolean or null")
        if gate.get("passed") is False or package.metadata.get("evaluation_status") == "rejected":
            failing = gate.get("failing_benchmarks", [])
            if not isinstance(failing, list):
                raise ValueError("evaluation failing_benchmarks must be a list")
            raise ArtifactEvaluationError(
                "AEG artifact is rejected by its persisted evaluation gate: "
                + (", ".join(str(item) for item in failing) or "recorded rejection"),
                backend_name=backend_name,
            )
    except ArtifactEvaluationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactEvaluationError(
            f"AEG artifact has an invalid persisted evaluation gate: {exc}",
            backend_name=backend_name,
        ) from exc
