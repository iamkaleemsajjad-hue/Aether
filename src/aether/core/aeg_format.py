"""
AEG format serialization and package management.

This module defines the on-disk structure of AEG artifacts: the manifest,
graph, weights, kernels, parallelism plans, and version metadata. It also
provides loading/saving, integrity verification, and round-trip guarantees.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aether.core.aeg_ir import AEGIRModule
from aether.core.constants import (
    AEG_FORMAT_VERSION,
    AEG_MINIMUM_COMPATIBLE_VERSION,
    AEG_SUPPORTED_FORMAT_VERSIONS,
    AEG_GRAPH_FILENAME,
    AEG_IR_FILE_EXTENSION,
    AEG_MANIFEST_FILENAME,
    AEG_QUANT_FILE_EXTENSION,
    DEFAULT_MODEL_CACHE_SUBDIR,
    SUPPORTED_TARGET_IDS,
)
from aether.core.exceptions import AEGFormatError, AEGIntegrityError, AEGVersionError
from aether.core.hash_utils import compute_content_hash, compute_directory_hash, compute_file_hash, verify_file_hash
from aether.core.types import AEGVersion, ModelArchitecture, Precision, ShardingPlan


def _has_portable_target(target_id: str) -> bool:
    """Return whether a portable AEG backend is defined for ``target_id``."""
    return bool(
        target_id in SUPPORTED_TARGET_IDS
        and target_id != "cuda_sm100_tee"
        and target_id.startswith(("cuda_", "rocm_", "metal_"))
    )


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract an AEG archive without path traversal or link escapes."""
    root = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise AEGFormatError(f"link archive members are not permitted: {member.name!r}")
        candidate = (root / member.name.replace("\\", "/")).resolve()
        try:
            inside = candidate == root or candidate.is_relative_to(root)
        except AttributeError:  # Python 3.8 compatibility
            inside = str(candidate).startswith(str(root) + str(Path("/"))) or candidate == root
        if not inside or member.name.replace("\\", "/").startswith("/"):
            raise AEGFormatError(f"unsafe archive member path: {member.name!r}")
    archive.extractall(root)


@dataclass
class ProvenanceInfo:
    """Provenance information embedded in an AEG manifest.

    Loaded from .aeg/provenance/manifest.json when present, otherwise
    constructed from the manifest's own fields so the CLI safety command
    always has a non-None provenance object to inspect.
    """

    source_model_id: str = ""
    """Original source model identifier."""

    compiler_version: str = ""
    """Aether Runtime version that produced this AEG."""

    compile_timestamp: float = 0.0
    """Unix timestamp of compilation."""

    model_hash: str = ""
    """SHA-256 hash of the source model weights."""

    aeg_hash: str = ""
    """SHA-256 hash of the compiled AEG artifact."""

    transformations: list[str] = field(default_factory=list)
    """Names of optimizer passes applied during compilation."""

    provenance_chain_hash: str = ""
    """Hash of the full transformation chain."""

    c2pa_binding: str = ""
    """The ``urn:c2pa:<uuid>`` label of this artifact's signed C2PA manifest.

    Empty when the artifact has not been signed. It is a manifest label, not a
    resolvable URL: the manifest store itself lives at
    ``provenance/c2pa.manifest`` inside the package. See
    :mod:`aether.provenance.c2pa`.
    """

    watermark_enabled: bool = False
    """Whether the AEG carries an output watermark."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_model_id": self.source_model_id,
            "compiler_version": self.compiler_version,
            "compile_timestamp": self.compile_timestamp,
            "model_hash": self.model_hash,
            "aeg_hash": self.aeg_hash,
            "transformations": self.transformations,
            "provenance_chain_hash": self.provenance_chain_hash,
            "c2pa_binding": self.c2pa_binding,
            "watermark_enabled": self.watermark_enabled,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProvenanceInfo":
        return ProvenanceInfo(
            source_model_id=data.get("source_model_id", "") or data.get("source_model", {}).get("id", ""),
            compiler_version=data.get("compiler_version", ""),
            compile_timestamp=float(data.get("compile_timestamp", 0.0)),
            model_hash=data.get("model_hash", "").replace("sha256:", ""),
            aeg_hash=data.get("aeg_hash", "").replace("sha256:", ""),
            transformations=list(data.get("transformations", [])),
            provenance_chain_hash=data.get("provenance_chain_hash", ""),
            c2pa_binding=data.get("c2pa_binding", ""),
            watermark_enabled=bool(data.get("watermark", {}).get("enabled", data.get("watermark_enabled", False))),
        )

    @classmethod
    def load_from_aeg(cls, aeg_root: "Path") -> "ProvenanceInfo":
        """Load provenance from .aeg/provenance/manifest.json if it exists."""
        p = aeg_root / "provenance" / "manifest.json"
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return cls.from_dict(data)
            except Exception:  # noqa: BLE001
                pass
        return cls()


@dataclass
class OptimizationMetadata:
    """Metadata about the optimization applied during compilation."""

    fusion_passes_applied: list[str] = field(default_factory=list)
    """Names of fusion passes applied."""

    fused_ops_count: int = 0
    """Number of fused operations produced."""

    sensitivity_calibration_dataset: str = "wikitext-2"
    """Calibration dataset used for sensitivity analysis."""

    quality_budget_ppl_increase: float = 0.02
    """Configured maximum perplexity increase."""

    actual_ppl_increase: float | None = None
    """Measured perplexity increase after quantization."""

    precision_distribution: dict[str, str] = field(default_factory=dict)
    """Distribution of precisions by percentage (e.g., {'BF16': '12%', ...})."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusion_passes_applied": self.fusion_passes_applied,
            "fused_ops_count": self.fused_ops_count,
            "sensitivity_calibration_dataset": self.sensitivity_calibration_dataset,
            "quality_budget_ppl_increase": self.quality_budget_ppl_increase,
            "actual_ppl_increase": self.actual_ppl_increase,
            "precision_distribution": self.precision_distribution,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> OptimizationMetadata:
        return OptimizationMetadata(
            fusion_passes_applied=list(data.get("fusion_passes_applied", [])),
            fused_ops_count=data.get("fused_ops_count", 0),
            sensitivity_calibration_dataset=data.get("sensitivity_calibration_dataset", "wikitext-2"),
            quality_budget_ppl_increase=data.get("quality_budget_ppl_increase", 0.02),
            actual_ppl_increase=data.get("actual_ppl_increase"),
            precision_distribution=dict(data.get("precision_distribution", {})),
        )


@dataclass
class KernelSetMetadata:
    """Metadata about compiled kernel sets in the AEG."""

    targets: list[str] = field(default_factory=list)
    """Hardware targets covered by this kernel set."""

    flash_attention_variant: str = "flash_attention_3"
    """FlashAttention variant used for attention kernels."""

    backend_plans: dict[str, str] = field(default_factory=dict)
    """Map of target_id to backend plan descriptor."""

    portable_backends: list[str] = field(default_factory=list)
    """Framework-backed execution contracts that can materialize the AEG IR.

    A target entry is not, by itself, proof that executable code exists.  A
    portable backend entry means the artifact contains the canonical graph and
    weights required by that backend and the runtime may select it on a
    compatible destination (for example PyTorch CUDA, ROCm, or MPS).
    """

    variant_status: dict[str, str] = field(default_factory=dict)
    """Per-target status: ``executable``, ``portable``, or ``plan_only``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": self.targets,
            "flash_attention_variant": self.flash_attention_variant,
            "backend_plans": self.backend_plans,
            "portable_backends": self.portable_backends,
            "variant_status": self.variant_status,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> KernelSetMetadata:
        return KernelSetMetadata(
            targets=list(data.get("targets", [])),
            flash_attention_variant=data.get("flash_attention_variant", "flash_attention_3"),
            backend_plans=dict(data.get("backend_plans", {})),
            portable_backends=list(data.get("portable_backends", [])),
            variant_status=dict(data.get("variant_status", {})),
        )


@dataclass
class MemoryRequirements:
    """Memory requirements for different precisions and configs."""

    bf16_gb: float
    """Memory requirement at full BF16 precision (GB)."""

    compiled_min_gb: float
    """Minimum memory requirement with compiled precision (GB)."""

    recommended_gb: float
    """Recommended memory for optimal performance (GB)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bf16_gb": self.bf16_gb,
            "compiled_min_gb": self.compiled_min_gb,
            "recommended_gb": self.recommended_gb,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MemoryRequirements:
        return MemoryRequirements(
            bf16_gb=data["bf16_gb"],
            compiled_min_gb=data["compiled_min_gb"],
            recommended_gb=data["recommended_gb"],
        )


@dataclass
class AEGManifest:
    """Top-level manifest for an AEG artifact.

    The manifest contains version information, model architecture metadata,
    optimization metadata, kernel set metadata, and hashes for integrity
    verification.
    """

    model_id: str
    """Model identifier (e.g., 'Qwen/Qwen3-72B-Instruct')."""

    aether_version: str
    """Aether Runtime version that produced this AEG."""

    compiled_at: str
    """ISO 8601 timestamp of compilation."""

    graph_hash: str
    """Content-addressed hash of the AEG-IR graph."""

    architecture: ModelArchitecture
    """Detected model architecture."""

    optimization: OptimizationMetadata = field(default_factory=OptimizationMetadata)
    """Optimization metadata from the compiler."""

    kernels: KernelSetMetadata = field(default_factory=KernelSetMetadata)
    """Kernel set metadata."""

    memory_requirements: MemoryRequirements | None = None
    """Memory requirements at different precisions."""

    artifacts: dict[str, str] = field(default_factory=dict)
    """Content hashes for optional v3.x artifact files."""

    manifest_hash: str | None = None
    """Hash of the manifest itself (excluding this field)."""

    format_version: str = AEG_FORMAT_VERSION
    """AEG format version string."""

    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    """Provenance metadata for this AEG artifact.

    Populated from .aeg/provenance/manifest.json when the package is loaded.
    Always non-None: defaults to an empty ProvenanceInfo if no provenance
    file was recorded at compile time.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_id": self.model_id,
            "aether_version": self.aether_version,
            "compiled_at": self.compiled_at,
            "graph_hash": self.graph_hash,
            "architecture": self.architecture.to_dict(),
            "optimization": self.optimization.to_dict(),
            "kernels": self.kernels.to_dict(),
            "memory_requirements": self.memory_requirements.to_dict() if self.memory_requirements else None,
            "artifacts": self.artifacts,
            "manifest_hash": self.manifest_hash,
            # NOTE: provenance is intentionally excluded here.
            # It is stored in provenance/manifest.json and loaded at runtime;
            # it must NOT be written into manifest.json to keep the hash stable.
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AEGManifest:
        memory_data = data.get("memory_requirements")
        memory_requirements = MemoryRequirements.from_dict(memory_data) if memory_data else None
        provenance_data = data.get("provenance")
        provenance = ProvenanceInfo.from_dict(provenance_data) if provenance_data else ProvenanceInfo()
        return AEGManifest(
            model_id=data["model_id"],
            aether_version=data["aether_version"],
            compiled_at=data["compiled_at"],
            graph_hash=data["graph_hash"],
            architecture=ModelArchitecture.from_dict(data["architecture"]),
            optimization=OptimizationMetadata.from_dict(data.get("optimization", {})),
            kernels=KernelSetMetadata.from_dict(data.get("kernels", {})),
            memory_requirements=memory_requirements,
            artifacts=dict(data.get("artifacts", {})),
            manifest_hash=data.get("manifest_hash"),
            format_version=data.get("format_version", AEG_FORMAT_VERSION),
            provenance=provenance,
        )

    @staticmethod
    def from_json(json_str: str) -> AEGManifest:
        return AEGManifest.from_dict(json.loads(json_str))

    def compute_and_set_manifest_hash(self) -> None:
        """Compute the manifest hash and set it."""
        payload = self.to_dict()
        payload.pop("manifest_hash", None)
        self.manifest_hash = compute_content_hash(payload)

    def verify(self) -> None:
        """Verify the manifest's version and integrity.

        Raises:
            AEGVersionError: If the AEG format version is unsupported.
            AEGIntegrityError: If the manifest hash does not match.
        """
        current = AEGVersion.current()
        minimum = AEGVersion.parse(AEG_MINIMUM_COMPATIBLE_VERSION)
        if self.format_version not in AEG_SUPPORTED_FORMAT_VERSIONS:
            raise AEGVersionError(
                f"AEG format {self.format_version} is not supported by this runtime",
                aeg_version=self.format_version,
            )
        try:
            manifest_version = AEGVersion.parse(self.format_version)
        except (ValueError, IndexError) as exc:
            msg = f"Invalid AEG format version: {self.format_version}"
            raise AEGVersionError(msg, aeg_version=self.format_version) from exc
        if manifest_version.major == 1 and not manifest_version.is_compatible_with(minimum):
            msg = (
                f"AEG format {self.format_version} is not compatible with runtime "
                f"{current}. Minimum compatible: {minimum}"
            )
            raise AEGVersionError(
                msg,
                aeg_version=self.format_version,
                minimum_version=str(minimum),
            )
        if self.manifest_hash:
            payload = self.to_dict()
            payload.pop("manifest_hash", None)
            computed = compute_content_hash(payload)
            if computed != self.manifest_hash:
                msg = f"AEG manifest integrity check failed: expected {self.manifest_hash}, got {computed}"
                raise AEGIntegrityError(msg, expected_hash=self.manifest_hash, actual_hash=computed)

    def __repr__(self) -> str:
        return f"AEGManifest({self.model_id}, {self.format_version}, hash={self.graph_hash[:20]}...)"


class AEGPackage:
    """Represents a compiled AEG package on disk.

    An AEG package is a directory (or tar archive) containing:
    - FORMAT_VERSION
    - graph/computation_graph.aeg-ir
    - graph/metadata.json
    - weights/quantized/ (precision_map.json + model.aeg-quant)
    - kernels/ (per-target kernel metadata/plans)
    - parallelism/ (1/2/4/8 GPU sharding plans)
    - manifest.json
    """

    def __init__(self, root: Path | str) -> None:
        """Open an existing AEG package at the given root path.

        Args:
            root: Path to the AEG package directory.
        """
        self.root = Path(root).resolve()
        self.manifest: AEGManifest | None = None
        self.ir: AEGIRModule | None = None
        self.precision_map: dict[str, str] = {}
        self.sharding_plans: dict[int, ShardingPlan] = {}
        self.metadata: dict[str, Any] = {}
        #: Quantized weights staged for :meth:`save`, keyed by weight name. A
        #: package without these is graph-only and cannot run inference.
        self.weights: dict[str, Any] = {}
        self._weight_store: Any = None
        self._is_loaded = False
        # Retained only for packages loaded from an archive.  Keeping the
        # TemporaryDirectory object alive keeps the extracted package usable
        # for the lifetime of the returned AEGPackage.
        self._archive_tempdir: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return True if the package has been fully loaded into memory."""
        return self._is_loaded

    def weight_store(self) -> Any:
        """Return the package's :class:`~aether.core.weight_store.WeightStore`.

        The store is created lazily and its index read on first use, so opening a
        package does not pay the cost of touching weights that may never be read.
        """
        from aether.core.weight_store import WeightStore

        existing = getattr(self, "_weight_store", None)
        if existing is None:
            existing = WeightStore(self.root / "weights" / "quantized")
            existing.load_index()
            self._weight_store = existing
        return existing

    @property
    def has_weights(self) -> bool:
        """True when this package carries the weights needed for inference."""
        return self.weight_store().exists

    @property
    def model_id(self) -> str | None:
        """Return the model ID from the loaded manifest."""
        return self.manifest.model_id if self.manifest else None

    @property
    def format_version(self) -> str | None:
        """Return the AEG format version."""
        return self.manifest.format_version if self.manifest else None

    @staticmethod
    def create(root: Path | str, model_id: str, aether_version: str) -> AEGPackage:
        """Create a new AEG package directory structure.

        Args:
            root: Root directory where the package will be created.
            model_id: Model identifier.
            aether_version: Aether version string.

        Returns:
            A new, empty AEGPackage.
        """
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        package = AEGPackage(root_path)
        package.manifest = AEGManifest(
            model_id=model_id,
            aether_version=aether_version,
            compiled_at="2026-07-27T00:00:00Z",  # Will be updated at save time
            graph_hash="sha256:pending",
            architecture=ModelArchitecture(
                family="unknown",
                params_billion=0.0,
                layers=1,
                hidden_size=64,
                num_attention_heads=4,
            ),
            optimization=OptimizationMetadata(),
            kernels=KernelSetMetadata(),
            memory_requirements=MemoryRequirements(bf16_gb=0.0, compiled_min_gb=0.0, recommended_gb=0.0),
        )
        return package

    def set_graph(self, graph: Any) -> None:
        """Attach an AEGGraph to this package and convert it to AEGIRModule."""
        from aether.core.aeg_ir import AEGIRModule
        from aether.core.hash_utils import compute_graph_hash

        if isinstance(graph, AEGIRModule):
            self.ir = graph
        else:
            self.ir = AEGIRModule.from_graph(graph)
        self.manifest.graph_hash = compute_graph_hash(self.ir)
        if hasattr(graph, "architecture") and graph.architecture is not None:
            self.manifest.architecture = graph.architecture

    def set_ir(self, ir: AEGIRModule) -> None:
        """Attach an AEGIRModule and compute its graph hash."""
        self.ir = ir
        from aether.core.hash_utils import compute_graph_hash
        self.manifest.graph_hash = compute_graph_hash(ir)

    def load(self) -> AEGPackage:
        """Load the entire package from disk into memory.

        Returns:
            Self, for chaining.

        Raises:
            AEGFormatError: If the package is malformed.
            FileNotFoundError: If required files are missing.
        """
        if not self.root.exists():
            msg = f"AEG package not found: {self.root}"
            raise FileNotFoundError(msg)
        manifest_path = self.root / AEG_MANIFEST_FILENAME
        if not manifest_path.exists():
            msg = f"Missing AEG manifest: {manifest_path}"
            raise FileNotFoundError(msg)
        self.manifest = AEGManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        self.manifest.verify()
        format_sentinel = self.root / "FORMAT_VERSION"
        if format_sentinel.exists():
            declared = format_sentinel.read_text(encoding="utf-8").strip()
            if declared != self.manifest.format_version:
                raise AEGFormatError(
                    f"AEG FORMAT_VERSION disagrees with manifest: {declared!r} != {self.manifest.format_version!r}"
                )
        # Load graph
        graph_path = self.root / "graph" / AEG_GRAPH_FILENAME
        if graph_path.exists():
            self.ir = AEGIRModule.from_json(graph_path.read_text(encoding="utf-8"))
        # Load metadata
        metadata_path = self.root / "graph" / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # AEG/2.0 and AEG/3.0 are extensible, but an extension is only valid
        # when its compiler claim has a complete, parseable payload.  Validate
        # this after metadata is loaded so a stale or hand-edited artifact
        # cannot reach a runtime merely because its hashes are internally
        # consistent.
        feature_errors = self.validate_feature_claims()
        if feature_errors:
            raise AEGFormatError(
                "Invalid AEG optimizer feature claims",
                details={"errors": feature_errors, "format_version": self.manifest.format_version},
            )
        # Load precision map
        precision_path = self.root / "weights" / "quantized" / "precision_map.json"
        if precision_path.exists():
            self.precision_map = json.loads(precision_path.read_text(encoding="utf-8"))
        # Drop any cached weight index so it is re-read from this package's blob.
        # Tensors themselves stay lazy via weight_store(), keeping load() cheap
        # for large models.
        self._weight_store = None
        # Load provenance from the dedicated provenance/manifest.json file if
        # present.  The file has richer data than the lightweight summary stored
        # in manifest.json, so prefer it; fall back to what from_dict already
        # populated from the manifest when the file does not exist.
        richer_prov = ProvenanceInfo.load_from_aeg(self.root)
        if richer_prov.source_model_id or richer_prov.model_hash:
            self.manifest.provenance = richer_prov
        # Load sharding plans
        parallelism_dir = self.root / "parallelism"
        if parallelism_dir.exists():
            for plan_file in parallelism_dir.glob("*.json"):
                plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
                if "num_gpus" not in plan_data or "phase" not in plan_data:
                    self.metadata[plan_file.stem] = plan_data
                    continue
                plan = ShardingPlan.from_dict(plan_data)
                self.sharding_plans[plan.num_gpus] = plan
        self._is_loaded = True
        return self

    def save(self) -> None:
        """Save the package to disk.

        Raises:
            AEGFormatError: If the package is incomplete or malformed.
        """
        if not self.manifest:
            msg = "Cannot save AEG package without a manifest"
            raise AEGFormatError(msg)
        # ``Compiler`` sets compiled_at from the real build start (or the
        # reproducible SOURCE_DATE_EPOCH). Do not overwrite it on subsequent
        # saves such as evaluation-gate finalization; doing so made identical
        # builds differ merely because the package was saved twice.
        # Ensure directories
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "graph").mkdir(exist_ok=True)
        (self.root / "weights" / "quantized").mkdir(parents=True, exist_ok=True)
        (self.root / "kernels").mkdir(exist_ok=True)
        (self.root / "parallelism").mkdir(exist_ok=True)
        (self.root / "safety").mkdir(exist_ok=True)
        (self.root / "provenance").mkdir(exist_ok=True)
        (self.root / "watermark").mkdir(exist_ok=True)
        (self.root / "adapters").mkdir(exist_ok=True)
        (self.root / "cuda_graphs").mkdir(exist_ok=True)
        (self.root / "inference").mkdir(exist_ok=True)
        (self.root / "observability").mkdir(exist_ok=True)
        (self.root / "rollout").mkdir(exist_ok=True)
        (self.root / "fleet").mkdir(exist_ok=True)
        (self.root / "distillation").mkdir(exist_ok=True)
        (self.root / "agentic").mkdir(exist_ok=True)
        (self.root / "mla").mkdir(exist_ok=True)
        (self.root / "speculation").mkdir(exist_ok=True)
        (self.root / "multimodal").mkdir(exist_ok=True)
        if self.manifest.format_version in {"AEG/2.0", "AEG/3.0"}:
            for directory in (
                "structured_output", "merging", "ttt", "green", "tee",
                "multi_agent", "mcp", "semantic_cache", "training",
            ):
                (self.root / directory).mkdir(exist_ok=True)
        if self.manifest.format_version == "AEG/3.0":
            for directory in ("video", "cxl", "generated_kernels"):
                (self.root / directory).mkdir(exist_ok=True)
        # Write format version
        (self.root / "FORMAT_VERSION").write_text(self.manifest.format_version + "\n", encoding="utf-8")
        # Write graph
        if self.ir:
            (self.root / "graph" / AEG_GRAPH_FILENAME).write_text(self.ir.to_json(indent=2), encoding="utf-8")
            # Compute and set graph hash
            graph_hash = compute_file_hash(self.root / "graph" / AEG_GRAPH_FILENAME)
            self.manifest.graph_hash = graph_hash
            (self.root / "graph" / "graph.sha256").write_text(graph_hash + "\n", encoding="utf-8")
        # A runnable artifact (weights attached) must never be finalized with a
        # pending graph hash or the placeholder architecture — that is how the
        # 4-layer -> 1-layer bug shipped. Graph-only planning packages are
        # permitted but are rejected by the runtime loader.
        if self.weights:
            if self.ir is None or self.manifest.graph_hash == "sha256:pending":
                msg = (
                    "Refusing to save a runnable AEG without a computed graph hash "
                    "(graph_hash would remain 'sha256:pending')"
                )
                raise AEGFormatError(msg)
            arch = self.manifest.architecture
            if arch.family == "unknown" or arch.layers <= 0:
                msg = (
                    "Refusing to save a runnable AEG with a placeholder architecture "
                    f"(family={arch.family!r}, layers={arch.layers}); set the real "
                    f"source architecture before saving"
                )
                raise AEGFormatError(msg)
        # Write metadata
        (self.root / "graph" / "metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        # Write precision map
        precision_json = json.dumps(self.precision_map, indent=2)
        (self.root / "weights" / "quantized" / "precision_map.json").write_text(
            precision_json, encoding="utf-8"
        )
        (self.root / "weights" / "precision_map.json").write_text(precision_json, encoding="utf-8")
        # Write quantized weights, without which the package cannot run inference
        if self.weights:
            from aether.core.weight_store import WeightStore

            store = WeightStore(self.root / "weights" / "quantized")
            written = store.save(self.weights)
            self.metadata["weight_bytes"] = written
            self.metadata["weight_tensor_count"] = len(self.weights)
        # Write sharding plans
        for num_gpus, plan in self.sharding_plans.items():
            plan_path = self.root / "parallelism" / f"{num_gpus}gpu.json"
            plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        # Write kernel metadata
        kernel_meta_path = self.root / "kernels" / "kernel_sets.json"
        kernel_meta_path.write_text(json.dumps(self.manifest.kernels.to_dict(), indent=2), encoding="utf-8")
        self._write_extended_artifacts()
        # Compute and set manifest hash
        self.manifest.compute_and_set_manifest_hash()
        # Write manifest
        (self.root / AEG_MANIFEST_FILENAME).write_text(self.manifest.to_json(indent=2), encoding="utf-8")

    def save_as_archive(self, output_path: Path | str) -> Path:
        """Save the package as a tar archive.

        Args:
            output_path: Path to write the tar archive.

        Returns:
            Path to the created archive.
        """
        self.save()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as tar:
            tar.add(self.root, arcname=self.root.name)
        return output

    @staticmethod
    def load_from_archive(archive_path: Path | str) -> AEGPackage:
        """Load an AEG package from a tar archive.

        Args:
            archive_path: Path to the tar archive.

        Returns:
            Loaded AEGPackage.
        """
        archive = Path(archive_path)
        if not archive.exists():
            msg = f"Archive not found: {archive}"
            raise FileNotFoundError(msg)
        tmpdir = tempfile.TemporaryDirectory(prefix="aether-aeg-")
        try:
            with tarfile.open(archive, "r:gz") as tar:
                _safe_extract_tar(tar, Path(tmpdir.name))
            extracted_root = Path(tmpdir.name)
            # Find the AEG package root inside the archive
            candidates = list(extracted_root.glob("*"))
            package_root = candidates[0] if candidates else extracted_root
            package = AEGPackage(package_root)
            package._archive_tempdir = tmpdir  # noqa: SLF001 - lifetime ownership
            package.load()
            return package
        except Exception:
            tmpdir.cleanup()
            raise

    def verify_integrity(self) -> None:
        """Verify the integrity of all package files using stored hashes.

        Raises:
            AEGIntegrityError: If any file hash does not match.
        """
        if not self.manifest:
            msg = "Cannot verify package without manifest"
            raise AEGIntegrityError(msg)
        self.manifest.verify()
        feature_errors = self.validate_feature_claims()
        if feature_errors:
            raise AEGFormatError(
                "Invalid AEG optimizer feature claims",
                details={"errors": feature_errors, "format_version": self.manifest.format_version},
            )
        # Verify graph hash
        graph_path = self.root / "graph" / AEG_GRAPH_FILENAME
        graph_resolved = graph_path.resolve()
        if graph_path.is_symlink() or not graph_resolved.is_relative_to(self.root.resolve()):
            raise AEGIntegrityError(
                "AEG graph path is symlinked or escapes the package root",
                file_path=str(graph_path),
            )
        if graph_path.exists():
            expected = self.manifest.graph_hash
            if not verify_file_hash(graph_path, expected):
                actual = compute_file_hash(graph_path)
                msg = f"AEG graph hash mismatch: expected {expected}, got {actual}"
                raise AEGIntegrityError(msg, file_path=str(graph_path), expected_hash=expected, actual_hash=actual)
        # Verify overall package hash if available
        package_hash_path = self.root / "graph" / "graph.sha256"
        if package_hash_path.exists():
            stored = package_hash_path.read_text(encoding="utf-8").strip()
            computed = compute_file_hash(graph_path)
            if stored != computed:
                msg = f"AEG graph file hash mismatch: stored {stored}, computed {computed}"
                raise AEGIntegrityError(msg, file_path=str(package_hash_path), expected_hash=stored, actual_hash=computed)

        # Verify every payload declared by the manifest. Previously only the
        # graph was checked, so a tampered weight blob, kernel plan, safety
        # file, or provenance artifact could pass integrity verification.
        package_root = self.root.resolve()
        for relative_path, expected in self.manifest.artifacts.items():
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise AEGIntegrityError(
                    f"Unsafe AEG artifact path: {relative_path}",
                    file_path=str(self.root / relative_path),
                )
            payload = (package_root / relative).resolve()
            if not payload.is_relative_to(package_root):
                raise AEGIntegrityError(
                    f"AEG artifact escapes package root: {relative_path}",
                    file_path=str(payload),
                )
            original_payload = package_root / relative
            if original_payload.is_symlink() or payload.is_symlink():
                raise AEGIntegrityError(
                    f"Symlinked AEG artifacts are not allowed: {relative_path}",
                    file_path=str(original_payload),
                )
            if not payload.is_file():
                raise AEGIntegrityError(
                    f"Declared AEG artifact is missing: {relative_path}",
                    file_path=str(payload),
                    expected_hash=expected,
                )
            actual = compute_file_hash(payload)
            if actual != expected:
                raise AEGIntegrityError(
                    f"AEG artifact hash mismatch for {relative_path}: expected {expected}, got {actual}",
                    file_path=str(payload),
                    expected_hash=expected,
                    actual_hash=actual,
                )

        # Validate the self-describing weight index and blob bounds even when
        # a caller has not yet requested a tensor through WeightStore.
        store = self.weight_store()
        if store.exists:
            store.load_index()
            blob_size = store.blob_path.stat().st_size
            for entry in store.entries.values():
                for offset, size, label in (
                    (entry.codes_offset, entry.codes_bytes, "codes"),
                    (entry.scales_offset, entry.scales_bytes, "scales"),
                    (entry.zero_points_offset, entry.zero_points_bytes, "zero_points"),
                ):
                    if offset < 0 or size < 0 or offset + size > blob_size:
                        raise AEGIntegrityError(
                            f"Weight index entry {entry.name!r} has out-of-bounds {label} range",
                            file_path=str(store.index_path),
                        )

    def validate_feature_claims(self) -> list[str]:
        """Validate compiler-declared v4/v5 feature payloads.

        The canonical AEG writer stores the applied optimizer pass names in
        ``graph/metadata.json``.  That metadata is the compatibility bridge
        between the compiler and runtime; it must therefore be treated as a
        schema, not as an informational log.  This method validates the
        concrete payloads emitted by each pass and returns all errors so
        callers can report the complete artifact defect in one exception.

        AEG/1.0 and AEG/1.1 remain backward compatible and do not require the
        v4/v5 extension payloads.  AEG/2.0 requires v4 claims to be complete;
        AEG/3.0 is required for v5 claims.
        """
        if self.manifest is None:
            return ["AEG feature claims cannot be checked without a manifest"]

        version = self.manifest.format_version
        if version in {"AEG/1.0", "AEG/1.1"}:
            passes: Any = self.metadata.get("optimizer_passes", [])
            extension_passes = {
                "mtp_head_compilation", "grammar_constraint_compilation",
                "model_merging", "ttt_fast_weight_injection",
                "semantic_kv_compression", "cross_layer_kv_sharing",
                "green_energy_compilation", "tee_kernel_wrapping",
                "mdlm_drafter_compilation", "sub2bit_quantization",
                "video_token_compression", "advanced_peft_compilation",
                "rlvr_verifier_head_injection",
            }
            if isinstance(passes, list) and extension_passes.intersection(passes):
                return [
                    f"AEG {version} contains v4/v5 optimizer claims but is not an extension format"
                ]
            return []

        errors: list[str] = []
        passes = self.metadata.get("optimizer_passes", [])
        if passes is None:
            passes = []
        if not isinstance(passes, list) or not all(isinstance(item, str) for item in passes):
            return ["graph metadata optimizer_passes must be a list of strings"]

        v4_passes = {
            "mtp_head_compilation",
            "grammar_constraint_compilation",
            "model_merging",
            "ttt_fast_weight_injection",
            "semantic_kv_compression",
            "cross_layer_kv_sharing",
            "green_energy_compilation",
            "tee_kernel_wrapping",
        }
        v5_passes = {
            "mdlm_drafter_compilation",
            "sub2bit_quantization",
            "video_token_compression",
            "advanced_peft_compilation",
            "rlvr_verifier_head_injection",
        }
        if version == "AEG/2.0" and v5_passes.intersection(passes):
            errors.append("AEG/2.0 cannot contain v5 optimizer claims")
        if version not in {"AEG/2.0", "AEG/3.0"} and (v4_passes | v5_passes).intersection(passes):
            errors.append(f"unsupported extension format for optimizer claims: {version}")

        def safe_relative(relative: Any) -> Path | None:
            if not isinstance(relative, str):
                errors.append("feature payload reference must be a relative string")
                return None
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"unsafe feature payload path: {relative!r}")
                return None
            resolved = (self.root / path).resolve()
            if not resolved.is_relative_to(self.root.resolve()):
                errors.append(f"feature payload escapes package root: {relative!r}")
                return None
            return resolved

        def require_file(relative: str, *, nonempty: bool = False) -> Path | None:
            path = safe_relative(relative)
            if path is None:
                return None
            if not path.is_file():
                errors.append(f"optimizer claim requires missing payload: {relative}")
                return None
            if path.is_symlink():
                errors.append(f"optimizer payload may not be a symlink: {relative}")
                return None
            if nonempty and path.stat().st_size == 0:
                errors.append(f"optimizer payload is empty: {relative}")
                return None
            return path

        def read_object(relative: str) -> dict[str, Any] | None:
            path = require_file(relative, nonempty=True)
            if path is None:
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid optimizer JSON payload {relative}: {exc}")
                return None
            if not isinstance(payload, dict):
                errors.append(f"optimizer JSON payload must be an object: {relative}")
                return None
            return payload

        def require_format(payload: dict[str, Any], relative: str, expected: str) -> None:
            if payload.get("format") != expected:
                errors.append(
                    f"optimizer payload {relative} has format {payload.get('format')!r}; "
                    f"expected {expected!r}"
                )

        for pass_name in passes:
            if pass_name == "mtp_head_compilation":
                payload = read_object("speculation/mtp_config.json")
                if payload is not None:
                    require_format(payload, "speculation/mtp_config.json", "aether_mtp_v1")
                    heads = payload.get("heads")
                    if not isinstance(heads, list) or not heads:
                        errors.append("MTP claim has no compiled heads")
                    else:
                        for head in heads:
                            if not isinstance(head, dict):
                                errors.append("MTP head descriptor must be an object")
                                continue
                            blob = head.get("blob_file")
                            if isinstance(blob, str):
                                require_file(f"speculation/{blob}", nonempty=True)
                            else:
                                errors.append("MTP head descriptor has no blob_file")
            elif pass_name == "grammar_constraint_compilation":
                payload = read_object("grammar/fsm_config.json")
                blob = require_file("grammar/fsm.bin", nonempty=True)
                if payload is not None:
                    require_format(payload, "grammar/fsm_config.json", "aether_fsa_v1")
                    if payload.get("tokenizer_aware") is not True:
                        errors.append("grammar claim is not tokenizer-aware")
                    if not isinstance(payload.get("tokenizer_fingerprint"), str) or not payload["tokenizer_fingerprint"]:
                        errors.append("grammar claim has no tokenizer fingerprint")
                    for key in ("n_states", "n_transitions", "vocab_size"):
                        if not isinstance(payload.get(key), int) or payload[key] <= 0:
                            errors.append(f"grammar payload has invalid {key}")
                if blob is not None and blob.stat().st_size < 64:
                    errors.append("grammar FSM payload is shorter than its binary header")
            elif pass_name == "model_merging":
                payload = read_object("merging/merge_manifest.json")
                if payload is not None:
                    require_format(payload, "merging/merge_manifest.json", "aether_merge_manifest_v1")
                    if int(payload.get("n_sources", 0) or 0) <= 0:
                        errors.append("model merge claim has no source models")
                    if not isinstance(payload.get("coefficients"), list):
                        errors.append("model merge claim has no coefficient list")
            elif pass_name == "ttt_fast_weight_injection":
                payload = read_object("ttt/fast_weight_config.json")
                if payload is not None:
                    require_format(payload, "ttt/fast_weight_config.json", "aether_ttt_v1")
                    slots = payload.get("slots")
                    if not isinstance(slots, list) or not slots:
                        errors.append("TTT claim has no slot descriptors")
                    else:
                        for slot in slots:
                            if not isinstance(slot, dict) or not isinstance(slot.get("slot_file"), str):
                                errors.append("TTT slot descriptor has no slot_file")
                                continue
                            require_file(f"ttt/{slot['slot_file']}", nonempty=True)
            elif pass_name == "semantic_kv_compression":
                payload = read_object("graph/kv_compression_plan.json")
                if payload is not None:
                    require_format(payload, "graph/kv_compression_plan.json", "aether_kv_compression_v1")
                    if not isinstance(payload.get("layers"), list) or not payload["layers"]:
                        errors.append("semantic KV claim has no layer plan")
            elif pass_name == "cross_layer_kv_sharing":
                payload = read_object("graph/cross_layer_kv_plan.json")
                if payload is not None:
                    require_format(payload, "graph/cross_layer_kv_plan.json", "aether_cross_layer_kv_v1")
                    if not isinstance(payload.get("sharing_groups"), list):
                        errors.append("cross-layer KV claim has no sharing groups")
            elif pass_name == "green_energy_compilation":
                payload = read_object("metadata/green_profile.json")
                if payload is not None:
                    require_format(payload, "metadata/green_profile.json", "aether_green_profile_v1")
                    if not isinstance(payload.get("dvfs_hints"), list):
                        errors.append("green-energy claim has no DVFS hint list")
            elif pass_name == "tee_kernel_wrapping":
                payload = read_object("security/tee_config.json")
                hashes = read_object("security/weight_hash_manifest.json")
                if payload is not None:
                    require_format(payload, "security/tee_config.json", "aether_tee_v1")
                    if payload.get("backend") in {None, "none", ""}:
                        errors.append("TEE claim has no concrete backend")
                if hashes is not None:
                    require_format(hashes, "security/weight_hash_manifest.json", "aether_weight_hash_manifest_v1")
                    if not isinstance(hashes.get("weight_hashes"), dict):
                        errors.append("TEE claim has no weight hash map")
            elif pass_name == "mdlm_drafter_compilation":
                payload = read_object("diffusion/drafter_config.json")
                schedule = read_object("diffusion/schedule.json")
                head = read_object("graph/mdlm_draft_head_config.json")
                head_blob = require_file("graph/mdlm_draft_head.npz", nonempty=True)
                if payload is not None:
                    if payload.get("type") != "mdlm_drafter":
                        errors.append("MDLM drafter payload has the wrong type")
                    if int(payload.get("T_steps", 0) or 0) <= 0 or int(payload.get("K_block", 0) or 0) <= 0:
                        errors.append("MDLM drafter payload has invalid dimensions")
                if schedule is not None:
                    if not isinstance(schedule.get("alpha_t"), list) or not schedule["alpha_t"]:
                        errors.append("MDLM drafter schedule has no denoising coefficients")
                if head is not None:
                    require_format(head, "graph/mdlm_draft_head_config.json", "aether_mdlm_head_v1")
                    if head.get("backend") != "numpy_cpu":
                        errors.append("MDLM head does not declare the executable numpy_cpu backend")
                    if head.get("weight_file") != "mdlm_draft_head.npz":
                        errors.append("MDLM head has an invalid weight_file")
                    if not isinstance(head.get("weight_keys"), list) or not head["weight_keys"]:
                        errors.append("MDLM head has no declared weight keys")
            elif pass_name == "sub2bit_quantization":
                payload = read_object("quantization/sub2bit_manifest.json")
                if payload is not None:
                    require_format(payload, "quantization/sub2bit_manifest.json", "aether_sub2bit_v1")
                    if float(payload.get("bits_per_weight", 0) or 0) <= 0:
                        errors.append("sub-2-bit claim has invalid bits_per_weight")
            elif pass_name == "video_token_compression":
                payload = read_object("graph/video_compression_plan.json")
                if payload is not None:
                    require_format(payload, "graph/video_compression_plan.json", "aether_video_compression_v1")
                    retention = payload.get("retention_ratio")
                    if not isinstance(retention, (int, float)) or not 0 < retention <= 1:
                        errors.append("video compression claim has invalid retention_ratio")
            elif pass_name == "advanced_peft_compilation":
                payload = read_object("adapters/adapter_manifest.json")
                if payload is not None:
                    require_format(payload, "adapters/adapter_manifest.json", "aether_adapter_manifest_v1")
                    adapters = payload.get("adapters")
                    if not isinstance(adapters, list) or not adapters:
                        errors.append("PEFT claim has no adapter descriptors")
                    else:
                        for adapter in adapters:
                            if not isinstance(adapter, dict):
                                errors.append("PEFT adapter descriptor must be an object")
                                continue
                            for key in ("lora_A_ref", "lora_B_ref"):
                                reference = adapter.get(key)
                                if isinstance(reference, str):
                                    require_file(f"adapters/{reference}", nonempty=True)
                                else:
                                    errors.append(f"PEFT descriptor has no {key}")
            elif pass_name == "rlvr_verifier_head_injection":
                payload = read_object("training/rlvr_config.json")
                if payload is not None:
                    require_format(payload, "training/rlvr_config.json", "aether_rlvr_v1")
                    if int(payload.get("grpo_K", 0) or 0) < 2:
                        errors.append("RLVR claim has invalid GRPO group size")
                    if not isinstance(payload.get("opcodes"), list) or not payload["opcodes"]:
                        errors.append("RLVR claim has no training opcodes")

        return errors


    def _write_extended_artifacts(self) -> None:
        """Write optional AEG v3.x artifact manifests and content hashes."""
        if not self.manifest:
            return
        artifacts: dict[str, str] = {}
        graph_metadata = dict(self.metadata)
        defaults: dict[str, Any] = {
            "graph/reasoning_graph.aeg-ir": graph_metadata.get("reasoning_graph", {
                "version": "reasoning_graph/1.0",
                "entrypoint": "disabled",
                "nodes": [],
                "edges": [],
            }),
            "graph/rag_pipeline.aeg-ir": graph_metadata.get("rag_pipeline", {
                "version": "rag_pipeline/1.0",
                "stages": ["query_encode", "retrieve", "rerank", "context_pack", "generate"],
                "enabled": False,
            }),
            "graph/attention_head_patterns.json": graph_metadata.get("attention_head_patterns", {
                "version": "sparse_attention/1.0",
                "enabled": False,
                "patterns": [],
            }),
            "parallelism/prefill_decode_split.json": graph_metadata.get("prefill_decode_split", {
                "prefill_pool": {"role": "compute_bound", "chunk_size": 2048},
                "decode_pool": {"role": "memory_bound", "batching": "continuous"},
                "kv_transfer": "shared_memory_or_rdma",
            }),
            "inference/compute_profiles.json": graph_metadata.get("compute_profiles", {
                "greedy": {"relative_cost": 1.0, "quality_bias": 0.0},
                "beam": {"relative_cost": 2.5, "quality_bias": 0.04},
                "best_of_n": {"relative_cost": 4.0, "quality_bias": 0.07},
                "mcts": {"relative_cost": 8.0, "quality_bias": 0.12},
            }),
            "safety/prompt_guard.json": graph_metadata.get("prompt_guard", {
                "enabled": True,
                "threshold": 0.65,
                "signals": ["ignore_previous", "reveal_system", "tool_exfiltration"],
            }),
            "safety/output_filter.json": graph_metadata.get("output_filter", {
                "enabled": True,
                "policies": ["safety", "privacy", "copyright"],
                "action": "block_or_redact",
            }),
            "safety/audit_log.json": graph_metadata.get("audit_log", {"events": [], "hash_algorithm": "sha256"}),
            "provenance/manifest.json": graph_metadata.get("provenance", {
                "compiler_version": self.manifest.aether_version,
                "source_model": {"id": self.manifest.model_id, "license": "unknown"},
                "transformations": self.manifest.optimization.fusion_passes_applied,
                "eu_ai_act": {"risk_category": "unknown", "transparency_obligations_met": False},
            }),
            "provenance/fingerprint.json": graph_metadata.get("fingerprint", {
                "enabled": False,
                "trigger_count": 0,
                "ownership_threshold": 0.85,
            }),
            "watermark/config.json": graph_metadata.get("watermark", {
                "enabled": True,
                "algorithm": "greenlist_statistical",
                "context_width": 16,
                "delta": 1.0,
                "z_threshold": 4.0,
            }),
            "adapters/manifest.json": graph_metadata.get("adapters", {
                "mode": "none",
                "slots": 0,
                "adapters": [],
            }),
            "cuda_graphs/capture_manifest.json": graph_metadata.get("cuda_graphs", {
                "version": "cuda_graph_capture/1.0",
                "enabled": False,
                "piecewise": True,
                "decode_batch_sizes": [1, 2, 4, 8, 16, 32],
                "prefill_chunk_sizes": [512, 1024, 2048],
                "persistent_kernels": ["decode_attention", "rmsnorm", "moe_router"],
            }),
            "observability/eval_gates.json": graph_metadata.get("eval_gates", {
                "enabled": True,
                "max_relative_regression": 0.02,
                "required_benchmarks": ["hellaswag", "mmlu", "gsm8k", "math-500", "humaneval"],
                "action": "block_rollout_on_failure",
            }),
            "observability/drift_monitor.json": graph_metadata.get("drift_monitor", {
                "enabled": True,
                "baseline_win_rate": 0.5,
                "alert_drop": 0.05,
                "signals": ["win_rate", "ttft_ms", "tokens_per_second", "spec_accept_rate"],
            }),
            "observability/metrics_schema.json": graph_metadata.get("metrics_schema", {
                "version": "otel_metrics/1.0",
                "metrics": [
                    "tokens_per_second",
                    "ttft_ms",
                    "eagle3_accept_rate",
                    "kv_hit_rate",
                    "mla_compression_ratio",
                    "reasoning_budget_used",
                    "gpu_vram_utilization",
                ],
            }),
            "rollout/ab_config.json": graph_metadata.get("ab_rollout", {
                "enabled": True,
                "candidate_percent": 0.01,
                "assignment": "sha256_stable_bucket",
                "rollback_on": ["eval_gate_failure", "quality_drift_alert", "safety_alert"],
            }),
            "fleet/deployment_plan.json": graph_metadata.get("fleet_deployment", {
                "version": "fleet/1.0",
                "strategy": "heterogeneous_target_aware",
                "targets": self.manifest.kernels.targets,
                "hot_reload": True,
            }),
            "fleet/hot_reload.json": graph_metadata.get("hot_reload", {
                "enabled": True,
                "routing": "stable_hash",
                "rollback": "instant_switch_to_active",
            }),
            "distillation/plan.json": graph_metadata.get("distillation", {
                "version": "distillation/1.0",
                "modes": ["logit", "feature", "reasoning", "self"],
                "quality_gates": ["perplexity", "task_eval", "safety_regression"],
                "target_quality_retention": 0.95,
            }),
            "agentic/workflow_cache.json": graph_metadata.get("agentic_workflow", {
                "version": "agentic_workflow/1.0",
                "meta_tools": [],
                "routes": ["fast", "balanced", "deep"],
                "context_cache": {"enabled": True, "scope": "session_and_org_prefix"},
            }),
            "mla/plan.json": graph_metadata.get("mla_plan", {
                "version": "mla_compression/1.0",
                "enabled": self.manifest.architecture.attention_type.upper() == "MLA",
                "weight_absorption": self.manifest.architecture.attention_type.upper() == "MLA",
                "kernel": "aeg.mla_portable",
            }),
            "speculation/eagle3.json": graph_metadata.get("eagle3", {
                "version": "eagle3/1.0",
                "fusion_layers": list(range(min(self.manifest.architecture.layers, 8))),
                "tree_depth": 5 if self.manifest.architecture.context_length >= 65536 else 4,
                "branching_factor": 4,
                "attention_drift_correction": self.manifest.architecture.context_length >= 65536,
                "flattened_tree": True,
                "acceptance_floor": 0.75,
            }),
            "multimodal/graph.json": graph_metadata.get("multimodal_graph", {
                "version": "multimodal_graph/1.0",
                "enabled": False,
                "stages": ["modality_encode", "project", "llm_generate"],
                "optimizations": {"vit_data_parallel": True, "llm_tensor_parallel": True},
            }),
        }
        if "eval_report" in graph_metadata:
            defaults["observability/eval_report.json"] = graph_metadata["eval_report"]
        sparsity_plan = graph_metadata.get("sparsity_plan")
        if sparsity_plan is not None:
            defaults["weights/sparsity_masks.json"] = sparsity_plan
        if self.manifest.architecture.attention_type.upper() == "MLA":
            mla_dir = self.root / "weights" / "mla_compressed"
            mla_dir.mkdir(parents=True, exist_ok=True)
            latent_path = mla_dir / "latent_kv.manifest.json"
            latent_payload = {
                "enabled": True,
                "compression": "latent_kv",
                "estimated_kv_reduction": 0.90,
                "weight_absorption": True,
            }
            latent_path.write_text(json.dumps(latent_payload, indent=2), encoding="utf-8")
            artifacts["weights/mla_compressed/latent_kv.manifest.json"] = compute_file_hash(latent_path)
        for relative_path, payload in defaults.items():
            target = self.root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, str):
                target.write_text(payload, encoding="utf-8")
            else:
                target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
            artifacts[relative_path] = compute_file_hash(target)

        # Include all package payloads in the manifest, excluding the manifest
        # itself (which contains these hashes and would be self-referential).
        for target in self.root.rglob("*"):
            if not target.is_file():
                continue
            relative_path = target.relative_to(self.root).as_posix()
            if relative_path == AEG_MANIFEST_FILENAME:
                continue
            artifacts.setdefault(relative_path, compute_file_hash(target))
        self.manifest.artifacts = artifacts

    def get_backend_plan(self, target_id: str) -> str | None:
        """Return the backend plan descriptor for a hardware target."""
        if not self.manifest:
            return None
        return self.manifest.kernels.backend_plans.get(target_id)

    def supports_runtime_target(self, target_id: str) -> bool:
        """Return whether this artifact contains a variant for ``target_id``.

        AEG is portable at the IR/weight level, but executable kernel and
        backend variants remain target-specific.  CPU variants share the
        framework-free reference contract; accelerator execution requires an
        explicitly emitted matching target.  This prevents a GPU request from
        silently running a CPU artifact through an optional frontend backend.
        """
        if not self.manifest:
            return False
        kernels = self.manifest.kernels
        targets = set(kernels.targets)
        status = kernels.variant_status

        # New artifacts distinguish a real executable variant from a target
        # profile that was only used for planning.  Older AEGs did not carry
        # this field, so retain their historical exact-target behavior.
        if status:
            if status.get(target_id) == "executable":
                return True
            if status.get(target_id) == "portable" and _has_portable_target(target_id):
                return True
            if target_id.startswith("cpu_") and any(
                key.startswith("cpu_") and value in {"executable", "portable"}
                for key, value in status.items()
            ):
                return True
        elif target_id in targets:
            return True
        elif target_id.startswith("cpu_") and any(item.startswith("cpu_") for item in targets):
            return True

        # The graph/weight contract can be materialized by a real PyTorch
        # installation on CUDA, ROCm, or Apple MPS.  This is deliberately
        # limited to those device families; it must never make an NPU/FPGA/
        # TEE target appear executable without its vendor runtime.
        # A portable contract still has to name a target that Aether knows how
        # to map to a real device runtime.  Prefix matching alone would make
        # arbitrary values such as ``cuda_fake`` appear executable.
        if "pytorch" in kernels.portable_backends and _has_portable_target(target_id):
            return True
        return False

    def supports_portable_backend(self, backend_name: str) -> bool:
        """Return whether the artifact declares a portable backend contract."""
        return bool(self.manifest and backend_name in self.manifest.kernels.portable_backends)

    def get_sharding_plan(self, num_gpus: int) -> ShardingPlan | None:
        """Return the sharding plan for a specific GPU count."""
        return self.sharding_plans.get(num_gpus)

    def get_precision_map(self) -> dict[str, str]:
        """Return the per-layer precision assignments."""
        return dict(self.precision_map)

    def set_precision_map(self, precision_map: dict[str, str]) -> None:
        """Set the per-layer precision assignments."""
        self.precision_map = dict(precision_map)

    def set_sharding_plan(self, num_gpus: int, plan: ShardingPlan) -> None:
        """Set the sharding plan for a specific GPU count."""
        self.sharding_plans[num_gpus] = plan

    def set_backend_plan(self, target_id: str, plan_descriptor: str) -> None:
        """Set the backend plan descriptor for a hardware target."""
        if not self.manifest:
            msg = "Cannot set backend plan without manifest"
            raise AEGFormatError(msg)
        self.manifest.kernels.backend_plans[target_id] = plan_descriptor
        if target_id not in self.manifest.kernels.targets:
            self.manifest.kernels.targets.append(target_id)

    def compute_size(self) -> int:
        """Compute the total size of the package on disk in bytes."""
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def copy_to(self, destination: Path | str) -> AEGPackage:
        """Copy the package to a new location.

        Args:
            destination: Destination path.

        Returns:
            AEGPackage at the new location. The returned package is loaded if
            the destination contains a valid manifest.
        """
        dest = Path(destination).resolve()
        if self.root.exists():
            shutil.copytree(self.root, dest, dirs_exist_ok=True)
        new_package = AEGPackage(dest)
        if dest.exists() and (dest / AEG_MANIFEST_FILENAME).exists():
            new_package.load()
        return new_package

    def __repr__(self) -> str:
        return f"AEGPackage({self.root}, model_id={self.model_id}, loaded={self._is_loaded})"


def load_aeg_package(path: Path | str) -> AEGPackage:
    """Load an AEG package from disk (directory or archive).

    Args:
        path: Path to the AEG package (directory or .tar.gz archive).

    Returns:
        Loaded AEGPackage.
    """
    p = Path(path)
    if p.is_dir():
        package = AEGPackage(p)
        package.load()
        return package
    if p.suffix in (".gz", ".tgz"):
        return AEGPackage.load_from_archive(p)
    msg = f"Cannot load AEG package from: {p}. Expected directory or .tar.gz archive."
    raise AEGFormatError(msg)


def create_default_sharding_plans(architecture: ModelArchitecture) -> dict[int, ShardingPlan]:
    """Create lossless model-wide tensor plans for one through eight GPUs.

    Args:
        architecture: Model architecture metadata.

    Returns:
        Mapping of GPU count to ShardingPlan.
    """
    plans: dict[int, ShardingPlan] = {}
    params_gb = architecture.params_billion * 2.0  # Approximate GB at BF16
    for num_gpus in range(1, 9):
        # A compiled artifact records a model-wide tensor partition for every
        # visible-device count.  ``balanced_partition`` handles dimensions
        # that are not divisible by the count at execution time; no rank is a
        # full model replica. Pipeline/context layouts remain explicit opt-in
        # strategies rather than a hidden fallback.
        plans[num_gpus] = ShardingPlan(
            num_gpus=num_gpus,
            phase="prefill",
            tensor_parallel_degree=num_gpus,
            pipeline_stages=1,
            expert_parallel_degree=1,
            context_parallel_degree=1,
            memory_per_gpu_gb=params_gb / num_gpus,
        )
    return plans
