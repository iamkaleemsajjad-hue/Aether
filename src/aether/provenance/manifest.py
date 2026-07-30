"""AEG provenance manifest helpers."""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformationRecord:
    """A compiler transformation recorded for provenance."""

    pass_name: str
    version: str = "1.0"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"pass": self.pass_name, "version": self.version, "parameters": self.parameters}


@dataclass
class ProvenanceManifest:
    """Compliance-oriented provenance manifest for compiled AEG artifacts."""

    source_model_id: str
    compiler_version: str
    source_license: str = "unknown"
    transformations: list[TransformationRecord] = field(default_factory=list)
    risk_category: str = "unknown"
    eval_results: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def model_hash(self) -> str:
        payload = f"{self.source_model_id}:{self.compiler_version}:{self.source_license}"
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash(),
            "compiler_version": self.compiler_version,
            "source_model": {"id": self.source_model_id, "license": self.source_license},
            "transformations": [record.to_dict() for record in self.transformations],
            "eu_ai_act": {
                "risk_category": self.risk_category,
                "transparency_obligations_met": bool(self.eval_results),
            },
            "hardware_certification": {"eval_gate_passed": bool(self.eval_results), "eval_results": self.eval_results},
            "created_at": self.created_at,
        }
