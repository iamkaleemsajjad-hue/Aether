"""Complete AEG Package Format v3.1 (Section 39).

Extends the base AEG package with all v3.1 directories:
- cuda_graphs/  — pre-captured decode graphs
- parallelism/cp.json — context parallelism plans
- inference/    — compute profiles + PRM head
- watermark/    — SynthID-style output watermark config
- provenance/   — full EU AI Act provenance + fingerprint

Research: PRD Section 39 — Complete AEG Format v3.1
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _hash_model_source(model_weights_path: str | None) -> str | None:
    """Return a content hash for an existing model file or directory.

    A path string or model identifier is not provenance.  Legacy skeleton
    artifacts therefore leave the source hash unavailable unless the caller
    supplies readable source bytes.  Directory hashing is deterministic and
    rejects symlinked files so the recorded identity is not redirected after
    the manifest is written.
    """
    if not model_weights_path:
        return None
    source = Path(model_weights_path).expanduser()
    if not source.exists() or source.is_symlink():
        return None
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.name.encode("utf-8"))
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    if not source.is_dir():
        return None
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        return None
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ---------------------------------------------------------------------------
# AEG v3.1 manifest structure
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceManifest:
    """Full provenance manifest for EU AI Act compliance."""

    model_id: str
    model_hash: str | None
    source_hash_status: str = "unavailable"
    compiler_version: str = "aether/3.1.0"
    source_license: str = "Apache-2.0"
    transformations: list[dict[str, Any]] = field(default_factory=list)
    eu_ai_act_risk_category: str = "limited_risk"
    c2pa_binding: str = ""
    eval_gate_passed: bool = False
    eval_results: dict[str, float] = field(default_factory=dict)
    certified_targets: list[str] = field(default_factory=list)

    @classmethod
    def from_compile_run(
        cls,
        model_id: str,
        model_weights_path: str | None = None,
        transformations: list[dict[str, Any]] | None = None,
        eval_results: dict[str, float] | None = None,
        targets: list[str] | None = None,
    ) -> "ProvenanceManifest":
        model_hash = _hash_model_source(model_weights_path)

        return cls(
            model_id=model_id,
            model_hash=model_hash,
            source_hash_status="verified" if model_hash else "unavailable",
            transformations=list(transformations or []),
            eval_gate_passed=bool(eval_results),
            eval_results=dict(eval_results or {}),
            certified_targets=list(targets or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "provenance/1.0",
            "model_hash": self.model_hash,
            "source_hash_status": self.source_hash_status,
            "compiler_version": self.compiler_version,
            "source_model": {
                "id": self.model_id,
                "license": self.source_license,
            },
            "transformations": self.transformations,
            "c2pa_binding": self.c2pa_binding or "",
            "eu_ai_act": {
                "risk_category": self.eu_ai_act_risk_category,
                "transparency_obligations_met": True,
                "article_50_compliant": True,
                "binding_date": "2026-08-01",
            },
            "hardware_certification": {
                "certified_targets": self.certified_targets,
                "eval_gate_passed": self.eval_gate_passed,
                "eval_results": self.eval_results,
            },
        }


@dataclass
class WatermarkConfig:
    """SynthID-style output watermarking configuration."""

    enabled: bool = False
    method: str = "green_list_token"   # SynthID approach
    delta: float = 1.0                 # Logit boost for green-list tokens
    context_window: int = 16           # Tokens of context for green-list hash
    detection_z_threshold: float = 4.0  # z-score threshold (p < 0.00003 FPR)
    green_list_fraction: float = 0.25  # 25% of vocab is green-listed per context

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "watermark/1.0",
            "enabled": self.enabled,
            "method": self.method,
            "delta": self.delta,
            "context_window": self.context_window,
            "detection_z_threshold": self.detection_z_threshold,
            "green_list_fraction": self.green_list_fraction,
            "research": "SynthID-Text (Google DeepMind 2024)",
            "eu_ai_act": "Art. 50 — AI content disclosure obligation",
        }


@dataclass
class InferenceComputeConfig:
    """Inference-time compute scaling profiles (Section 30)."""

    strategies: dict[str, Any] = field(default_factory=lambda: {
        "greedy": {
            "max_tokens": 512,
            "temperature": 0.0,
            "description": "Fast greedy decode for simple queries",
        },
        "best_of_4": {
            "n_samples": 4,
            "selection": "prm_top1",
            "max_tokens": 2048,
            "temperature": 0.8,
            "parallel": True,
        },
        "best_of_8": {
            "n_samples": 8,
            "selection": "reward_model",
            "max_tokens": 4096,
            "temperature": 0.9,
            "parallel": True,
        },
        "beam_4": {
            "beam_width": 4,
            "length_penalty": 1.0,
            "max_tokens": 8192,
        },
        "mcts": {
            "simulations": 32,
            "ucb_constant": 1.4,
            "max_depth": 10,
            "max_tokens": 32768,
        },
        "adaptive": {
            "complexity_classifier": "embedded_classifier",
            "budget_map": {
                "simple":    {"strategy": "greedy",    "max_tokens": 512},
                "medium":    {"strategy": "best_of_4", "max_tokens": 2048},
                "hard":      {"strategy": "beam_4",    "max_tokens": 8192},
                "very_hard": {"strategy": "mcts",      "max_tokens": 32768},
            },
        },
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "compute_profiles/1.0",
            "strategies": self.strategies,
            "research": [
                "InferenceTimeScaling:Google2025",
                "ThreadWeaver:2026",
                "InferenceTimePessimism:2026",
            ],
        }


# ---------------------------------------------------------------------------
# AEG v3.1 package builder
# ---------------------------------------------------------------------------

class AEGPackageV31:
    """
    Complete AEG Format v3.1 package builder.

    Creates all directories and manifest files defined in PRD Section 39.

    Output structure:
        model.aeg/
        ├── FORMAT_VERSION
        ├── manifest.json         (top-level with all section hashes)
        ├── graph/
        ├── weights/
        ├── adapters/
        ├── kernels/
        ├── cuda_graphs/          [3.1 NEW]
        ├── parallelism/
        ├── inference/            [3.1 NEW]
        ├── safety/               [3.0]
        ├── provenance/           [3.1 NEW]
        └── watermark/            [3.1 NEW]
    """

    FORMAT_VERSION = "AEG/1.1"

    def __init__(
        self,
        model_id: str,
        target: str = "cuda_sm90",
        provenance: ProvenanceManifest | None = None,
        watermark: WatermarkConfig | None = None,
        compute: InferenceComputeConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.target = target
        self.provenance = provenance or ProvenanceManifest.from_compile_run(model_id)
        self.watermark = watermark or WatermarkConfig()
        self.compute = compute or InferenceComputeConfig()

    def build(self, output_dir: str | Path) -> Path:
        """
        Build a non-executable v3.1 metadata skeleton at ``output_dir``.

        This legacy helper is useful for schema/documentation tooling only.
        It never writes model weights, compiled kernels, or captured CUDA
        graphs, and the top-level manifest marks it as non-loadable.  Use the
        canonical compiler package writer for an executable AEG artifact.
        """
        aeg_dir = Path(output_dir)
        aeg_dir.mkdir(parents=True, exist_ok=True)

        written: dict[str, list[str]] = {}

        # FORMAT_VERSION
        (aeg_dir / "FORMAT_VERSION").write_text(self.FORMAT_VERSION, encoding="utf-8")

        # graph/
        graph_dir = aeg_dir / "graph"
        graph_dir.mkdir(exist_ok=True)
        (graph_dir / "attention_head_patterns.json").write_text(
            json.dumps({
                "version": "minference/1.0",
                "patterns": {},
                "note": "Populated by Pass 8 (MInference) during compilation",
            }, indent=2), encoding="utf-8"
        )
        written["graph"] = ["attention_head_patterns.json"]

        # weights/
        weights_dir = aeg_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        (weights_dir / "precision_map.json").write_text(
            json.dumps({
                "version": "precision_map/1.0",
                "default_precision": None,
                "per_layer": {},
                "note": "Populated by Pass 3 (Precision Assignment) during compilation",
            }, indent=2), encoding="utf-8"
        )
        (weights_dir / "sparsity_masks.json").write_text(
            json.dumps({
                "version": "sparsity/1.0",
                "method": None,
                "sparsity_ratio": 0.0,
                "note": "Populated by Pass 9 (Pruning) if enabled",
            }, indent=2), encoding="utf-8"
        )
        written["weights"] = ["precision_map.json", "sparsity_masks.json"]

        # adapters/
        adapters_dir = aeg_dir / "adapters"
        adapters_dir.mkdir(exist_ok=True)
        (adapters_dir / "manifest.json").write_text(
            json.dumps({
                "version": "lora_adapters/1.0",
                "adapters": [],
                "max_slots": 0,
                "bgmv_enabled": False,
                "status": "no adapter tensors are present in metadata skeleton",
            }, indent=2), encoding="utf-8"
        )
        written["adapters"] = ["manifest.json"]

        # kernels/ -- target profiles are not executable kernels.
        kernels_dir = aeg_dir / "kernels"
        kernels_dir.mkdir(exist_ok=True)
        (kernels_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": "kernels/1.0",
                    "compiled": False,
                    "targets": [],
                    "status": "no executable kernels in metadata skeleton",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["kernels"] = ["manifest.json"]

        # cuda_graphs/ [3.1 NEW]
        cuda_graph_dir = aeg_dir / "cuda_graphs"
        cuda_graph_dir.mkdir(exist_ok=True)
        (cuda_graph_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": "cuda_graphs/1.0",
                    "target": self.target,
                    "captured": False,
                    "graphs": [],
                    "status": "no CUDA graph binary is present in metadata skeleton",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["cuda_graphs"] = ["manifest.json"]

        # parallelism/ — write standard CP plans
        para_dir = aeg_dir / "parallelism"
        para_dir.mkdir(exist_ok=True)
        (para_dir / "prefill_decode_split.json").write_text(
            json.dumps({
                "version": "disaggregated/1.0",
                "enabled": False,
                "status": "no distributed runtime deployment is present in metadata skeleton",
            }, indent=2), encoding="utf-8"
        )
        written["parallelism"] = ["prefill_decode_split.json"]

        # inference/ [3.1 NEW]
        inference_dir = aeg_dir / "inference"
        inference_dir.mkdir(exist_ok=True)
        (inference_dir / "compute_profiles.json").write_text(
            json.dumps(self.compute.to_dict(), indent=2), encoding="utf-8"
        )
        (inference_dir / "prm_head.json").write_text(
            json.dumps({
                "version": "prm_head/1.0",
                "enabled": False,
                "type": None,
                "status": "no PRM checkpoint is present in metadata skeleton",
            }, indent=2), encoding="utf-8"
        )
        written["inference"] = ["compute_profiles.json", "prm_head.json"]

        # safety/ [3.0]
        safety_dir = aeg_dir / "safety"
        safety_dir.mkdir(exist_ok=True)
        from aether.safety.policy import SafetyManifestWriter
        smw = SafetyManifestWriter()
        smw.write(aeg_dir)
        written["safety"] = ["prompt_guard.json", "output_filter.json", "policy.json", "toxicity_config.json"]

        # provenance/ [3.1 NEW]
        provenance_dir = aeg_dir / "provenance"
        provenance_dir.mkdir(exist_ok=True)
        prov_path = provenance_dir / "manifest.json"
        prov_path.write_text(json.dumps(self.provenance.to_dict(), indent=2), encoding="utf-8")
        # Write fingerprint
        from aether.provenance.fingerprint import AEGModelFingerprint
        fp = AEGModelFingerprint()
        fp.write(aeg_dir, owner_id=f"owner:{self.model_id}", n_triggers=20)
        written["provenance"] = ["manifest.json", "fingerprint.json"]

        # watermark/ [3.1 NEW]
        watermark_dir = aeg_dir / "watermark"
        watermark_dir.mkdir(exist_ok=True)
        (watermark_dir / "config.json").write_text(
            json.dumps(self.watermark.to_dict(), indent=2), encoding="utf-8"
        )
        written["watermark"] = ["config.json"]

        # Top-level manifest.json
        manifest = self._build_top_level_manifest(written)
        (aeg_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return aeg_dir

    def _build_top_level_manifest(self, written: dict[str, list[str]]) -> dict[str, Any]:
        return {
            "version": self.FORMAT_VERSION,
            "artifact_kind": "metadata_skeleton",
            "executable": False,
            "status": "not_loadable_without_compiler_emitted_weights_and_kernels",
            "model_id": self.model_id,
            "aether_version": "3.1.0",
            "target": self.target,
            "sections": {
                section: files for section, files in written.items()
            },
            "features": {
                "operator_fusion": False,
                "sensitivity_quantization": False,
                "precision_assignment": False,
                "kv_cache_structuring": False,
                "moe_expert_routing": False,
                "parallelism_discovery": False,
                "reasoning_graph": False,
                "sparse_attention_minference": False,
                "pruning_sparsity": False,
                "lora_multi_slot": False,
                "ssm_hybrid": False,
                "rag_native": False,
                "long_context_1m": False,
                "cuda_graphs": False,
                "safety_guardrails": False,
                "provenance": bool(self.provenance.model_hash),
                "watermarking": False,
                "eu_ai_act_compliant": False,
            },
            "provenance_hash": hashlib.sha256(
                json.dumps(self.provenance.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
        }


# ---------------------------------------------------------------------------
# AEG Manifest v3.1 (read/write helper)
# ---------------------------------------------------------------------------

class AEGManifestV31:
    """Helper for reading and writing AEG v3.1 manifests."""

    def __init__(self, aeg_dir: str | Path) -> None:
        self.aeg_dir = Path(aeg_dir)
        self._manifest: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load manifest.json from the AEG directory."""
        path = self.aeg_dir / "manifest.json"
        if path.exists():
            self._manifest = json.loads(path.read_text(encoding="utf-8"))
        return self._manifest

    def get(self, key: str, default: Any = None) -> Any:
        """Get a field from the manifest."""
        if not self._manifest:
            self.load()
        return self._manifest.get(key, default)

    def update(self, updates: dict[str, Any]) -> None:
        """Update manifest fields and write back to disk."""
        if not self._manifest:
            self.load()
        self._manifest.update(updates)
        (self.aeg_dir / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2), encoding="utf-8"
        )

    def verify_format_version(self) -> bool:
        """Check that this AEG package has the correct format version."""
        version_file = self.aeg_dir / "FORMAT_VERSION"
        if version_file.exists():
            return version_file.read_text().strip().startswith("AEG/")
        return self.get("version", "").startswith("AEG/")

    def list_available_targets(self) -> list[str]:
        """List all hardware targets with compiled kernels in this package."""
        kernels_dir = self.aeg_dir / "kernels"
        if not kernels_dir.exists():
            return []
        targets: list[str] = []
        for descriptor in kernels_dir.glob("*/kernels.json"):
            try:
                payload = json.loads(descriptor.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("compiled") is True and isinstance(payload.get("target"), str):
                targets.append(payload["target"])
        return sorted(set(targets))

    def has_cuda_graphs(self) -> bool:
        """Check if this package includes pre-captured CUDA graphs."""
        path = self.aeg_dir / "cuda_graphs" / "manifest.json"
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(payload.get("captured"))

    def has_long_context_profile(self) -> bool:
        """Check if this package has a long-context profile."""
        return "long_context_profile" in self.get("", {})

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the AEG package."""
        manifest = self.load()
        return {
            "model_id": manifest.get("model_id", "unknown"),
            "format_version": manifest.get("version", "unknown"),
            "aether_version": manifest.get("aether_version", "unknown"),
            "target": manifest.get("target", "unknown"),
            "available_targets": self.list_available_targets(),
            "has_cuda_graphs": self.has_cuda_graphs(),
            "features": manifest.get("features", {}),
            "sections": list(manifest.get("sections", {}).keys()),
        }
