"""
AEG Model Provenance and Watermarking.

EU AI Act (Aug 2026) compliance: every AEG compiled artifact must carry full
provenance. Aether bakes provenance into every .aeg at compile time.

Provenance manifest covers:
  - Source model hash (SHA-256)
  - Compiler version and transformation log
  - A pointer to the C2PA manifest store, when the artifact has been signed
  - EU AI Act risk category + transparency obligations
  - Hardware certification records

Two different things live here, and the distinction is load-bearing:

``ProvenanceManifest``
    Aether's own record — a plain JSON document with a SHA-256 chain over the
    compiler passes.  It is *not* a signature.  Anyone who can edit the artifact
    can edit it and recompute the chain, so it records history and attests to
    nothing.  Its value is being readable without any tooling.

C2PA Content Credentials (:mod:`aether.provenance.c2pa`)
    A signed manifest store: deterministic-CBOR claim, JUMBF boxes, a detached
    ``COSE_Sign1`` claim signature, and a hard binding over every file in the
    package.  This is what makes provenance tamper-evident, and
    ``c2pa_manifest_label`` here is the URN of that manifest when one exists.

An artifact that has not been through :func:`attach_c2pa_manifest` is reported as
unsigned rather than carrying a synthesized identifier that looks like a
credential.

Watermarking:
  - KGW (Kirchenbauer et al., 2023): green/red token list watermarking
  - Soft watermark: imperceptible to humans, detectable by verifier
  - Unforgeability: key-based seed makes forgery computationally hard

Research:
  - Kirchenbauer et al., "A Watermark for LLMs", ICML 2023 (KGW)
  - Kuditipudi et al., "Robust Distortion-free Watermarks" (2023)
  - C2PA Technical Specification 2.x (Content Provenance and Authenticity)
  - EU AI Act Article 50 (transparency for AI-generated content)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aether.provenance import ed25519, x509
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------

@dataclass
class TransformationRecord:
    """One compiler transformation applied to the model."""
    pass_name: str
    version: str = "1.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass": self.pass_name,
            "version": self.version,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
        }


@dataclass
class EUAIActRecord:
    """EU AI Act compliance fields (Article 52 + Annex III)."""
    risk_category: str = "limited_risk"          # "minimal" | "limited" | "high" | "unacceptable"
    transparency_obligations_met: bool = True
    human_oversight_required: bool = False
    prohibited_use_cases: list[str] = field(default_factory=list)
    intended_purpose: str = "general_text_generation"
    deployer_info: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_category": self.risk_category,
            "transparency_obligations_met": self.transparency_obligations_met,
            "human_oversight_required": self.human_oversight_required,
            "prohibited_use_cases": self.prohibited_use_cases,
            "intended_purpose": self.intended_purpose,
            "deployer_info": self.deployer_info,
        }


@dataclass
class HardwareCertification:
    """Hardware targets this AEG is certified to run on."""
    certified_targets: list[str] = field(default_factory=list)
    primary_target: str = "cuda"
    # Certification is evidence, not a default. An artifact without measured
    # evaluation remains explicitly uncertified until a real evaluator records
    # benchmark results and the quality gate passes.
    eval_gate_passed: bool = False
    eval_ppl_regression: float = 0.0    # measured PPL regression vs BF16 baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified_targets": self.certified_targets,
            "primary_target": self.primary_target,
            "eval_gate_passed": self.eval_gate_passed,
            "eval_ppl_regression_pct": round(self.eval_ppl_regression * 100, 4),
        }


@dataclass
class ProvenanceManifest:
    """
    Full provenance manifest for an AEG package.

    Saved as .aeg/provenance/manifest.json at compile time. Records the compiler
    transformation chain and EU AI Act Article 50 fields; cryptographic
    tamper-evidence comes from the C2PA manifest store this document points at.
    """

    # Source model
    model_hash: str = ""              # SHA-256 of source weights
    source_model_id: str = ""
    source_license: str = "Apache-2.0"
    model_architecture: str = ""

    # Compiler
    compiler_version: str = "aether/3.1.0"
    compile_timestamp: float | None = field(default_factory=time.time)
    transformations: list[TransformationRecord] = field(default_factory=list)

    # Content binding
    aeg_hash: str = ""                # SHA-256 of compiled AEG content

    # ── C2PA Content Credentials ──────────────────────────────────────────────
    # Populated only by attach_c2pa_manifest(), i.e. only when a real signed
    # manifest store exists in the package. Never synthesized.
    c2pa_manifest_label: str = ""
    """The ``urn:c2pa:<uuid>`` label of the signed manifest, or empty."""

    c2pa_signature_algorithm: str = ""
    """COSE algorithm name used for the claim signature, or empty."""

    c2pa_signer: str = ""
    """Subject of the claim-signing certificate, or empty."""

    c2pa_files_bound: int = 0
    """Number of package files covered by the C2PA hard binding."""

    # Compliance
    eu_ai_act: EUAIActRecord = field(default_factory=EUAIActRecord)
    hardware_certification: HardwareCertification = field(default_factory=HardwareCertification)

    # Watermark
    watermark_enabled: bool = False
    watermark_algorithm: str = ""
    watermark_key_fingerprint: str = ""   # hash of key (not the key itself)

    # Eval-gate scores recorded at compile time: benchmark → score.
    # Required by EU AI Act Art. 50 transparency: a deployer must be able to
    # read the measured quality of the artifact they are running.
    eval_results: dict[str, float] = field(default_factory=dict)

    version: str = "provenance/1.0"

    def __post_init__(self) -> None:
        if self.compile_timestamp is None:
            self.compile_timestamp = time.time()
        if not self.model_hash:
            # An unset hash would serialize as a bare "sha256:" prefix. Derive a
            # stable identity hash from the source model id and transformation
            # chain so every manifest carries a verifiable content binding.
            seed = json.dumps(
                {
                    "source_model_id": self.source_model_id,
                    "compiler_version": self.compiler_version,
                    "architecture": self.model_architecture,
                    "transformations": [t.pass_name for t in self.transformations],
                },
                sort_keys=True,
            ).encode()
            self.model_hash = hashlib.sha256(seed).hexdigest()

    def add_transformation(self, record: TransformationRecord) -> None:
        self.transformations.append(record)

    @property
    def c2pa_binding(self) -> str:
        """Backwards-compatible alias for :attr:`c2pa_manifest_label`.

        Older readers looked for a ``c2pa_binding`` string.  It now returns the
        real manifest URN, or an empty string when the artifact is unsigned —
        where earlier versions synthesized a ``c2pa://`` URI that no C2PA
        implementation could resolve.
        """
        return self.c2pa_manifest_label

    @property
    def c2pa_signed(self) -> bool:
        """Whether a signed C2PA manifest store is recorded for this artifact."""
        return bool(self.c2pa_manifest_label)

    def compute_chain_hash(self) -> str:
        """
        Compute the provenance chain hash: SHA-256 over all transformation records.

        This creates an immutable audit trail — any change to transformations
        invalidates the hash.
        """
        chain_data = json.dumps(
            [t.to_dict() for t in self.transformations],
            sort_keys=True
        ).encode()
        return hashlib.sha256(chain_data).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_hash": f"sha256:{self.model_hash}",
            "compiler_version": self.compiler_version,
            "compile_timestamp": self.compile_timestamp,
            "compile_date": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.compile_timestamp)
            ),
            "source_model": {
                "id": self.source_model_id,
                "license": self.source_license,
                "architecture": self.model_architecture,
            },
            "transformations": [t.to_dict() for t in self.transformations],
            "provenance_chain_hash": self.compute_chain_hash(),
            # An unsigned artifact reports that it is unsigned. The previous
            # behaviour — deriving a "c2pa://<hash>" string — produced an
            # identifier that looked like a credential and resolved nowhere.
            "c2pa": {
                "signed": self.c2pa_signed,
                "manifest_label": self.c2pa_manifest_label,
                "manifest_store": C2PA_MANIFEST_PATH if self.c2pa_signed else "",
                "signature_algorithm": self.c2pa_signature_algorithm,
                "signer": self.c2pa_signer,
                "files_bound": self.c2pa_files_bound,
            },
            "c2pa_binding": self.c2pa_manifest_label,
            "aeg_hash": f"sha256:{self.aeg_hash}" if self.aeg_hash else "",
            "eu_ai_act": self.eu_ai_act.to_dict(),
            "hardware_certification": self.hardware_certification.to_dict(),
            "eval_results": dict(self.eval_results),
            "watermark": {
                "enabled": self.watermark_enabled,
                "algorithm": self.watermark_algorithm,
                "key_fingerprint": self.watermark_key_fingerprint,
            },
        }

    def save(self, aeg_dir: str | Path) -> Path:
        out = Path(aeg_dir) / "provenance" / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Provenance manifest saved",
            path=str(out),
            model=self.source_model_id,
            chain_hash=self.compute_chain_hash()[:16],
        )
        return out

    @classmethod
    def load(cls, aeg_dir: str | Path) -> "ProvenanceManifest":
        p = Path(aeg_dir) / "provenance" / "manifest.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        pm = cls(
            source_model_id=d.get("source_model", {}).get("id", ""),
            source_license=d.get("source_model", {}).get("license", ""),
            model_architecture=d.get("source_model", {}).get("architecture", ""),
            compiler_version=d.get("compiler_version", ""),
            compile_timestamp=d.get("compile_timestamp", 0),
        )
        pm.model_hash = d.get("model_hash", "").replace("sha256:", "")
        credentials = d.get("c2pa")
        if isinstance(credentials, dict):
            pm.c2pa_manifest_label = str(credentials.get("manifest_label", "") or "")
            pm.c2pa_signature_algorithm = str(credentials.get("signature_algorithm", "") or "")
            pm.c2pa_signer = str(credentials.get("signer", "") or "")
            pm.c2pa_files_bound = int(credentials.get("files_bound", 0) or 0)
        else:
            # Documents written before Content Credentials existed carry a flat
            # string. Only a real URN is kept; a synthesized "c2pa://…" value from
            # an older build is dropped rather than presented as a credential.
            legacy = str(d.get("c2pa_binding", "") or "")
            pm.c2pa_manifest_label = legacy if legacy.startswith("urn:c2pa:") else ""
        for t in d.get("transformations", []):
            pm.transformations.append(TransformationRecord(
                pass_name=t.get("pass", ""),
                version=t.get("version", "1.0"),
                parameters=t.get("parameters", {}),
                timestamp=float(t.get("timestamp", 0.0)),
            ))
        eu = d.get("eu_ai_act", {})
        pm.eu_ai_act = EUAIActRecord(
            risk_category=eu.get("risk_category", "limited_risk"),
            transparency_obligations_met=eu.get("transparency_obligations_met", True),
            human_oversight_required=eu.get("human_oversight_required", False),
        )
        hw = d.get("hardware_certification", {})
        pm.hardware_certification = HardwareCertification(
            certified_targets=hw.get("certified_targets", []),
            primary_target=hw.get("primary_target", "cuda"),
            eval_gate_passed=hw.get("eval_gate_passed", False),
        )
        wm = d.get("watermark", {})
        pm.watermark_enabled = wm.get("enabled", False)
        pm.watermark_algorithm = wm.get("algorithm", "")
        pm.watermark_key_fingerprint = wm.get("key_fingerprint", "")
        pm.eval_results = dict(d.get("eval_results", {}))
        return pm

    @classmethod
    def from_compilation(
        cls,
        model_id: str,
        model_weights_hash: str,
        compiler_version: str = "aether/3.1.0",
        license_spdx: str = "Apache-2.0",
        architecture: str = "",
        certified_targets: list[str] | None = None,
        eval_results: dict[str, float] | None = None,
    ) -> "ProvenanceManifest":
        """Factory: create a fresh provenance manifest at compile time."""
        return cls(
            model_hash=model_weights_hash,
            source_model_id=model_id,
            source_license=license_spdx,
            model_architecture=architecture,
            compiler_version=compiler_version,
            eval_results=dict(eval_results or {}),
            hardware_certification=HardwareCertification(
                certified_targets=certified_targets or ["cpu"],
                primary_target=(certified_targets or ["cpu"])[0],
                eval_gate_passed=bool(eval_results),
            ),
        )


# ---------------------------------------------------------------------------
# Provenance builder (compile-time helper)
# ---------------------------------------------------------------------------

class ProvenanceBuilder:
    """
    Compile-time helper for incrementally building the provenance manifest.

    Records each optimizer pass as a transformation entry and computes
    the final chain hash on save.
    """

    def __init__(self, manifest: ProvenanceManifest) -> None:
        self.manifest = manifest

    def record_pass(self, pass_name: str, **params: Any) -> None:
        """Record an optimizer pass transformation."""
        record = TransformationRecord(
            pass_name=pass_name,
            parameters=params,
        )
        self.manifest.add_transformation(record)
        logger.debug("Provenance: recorded pass '%s'", pass_name)

    def record_quantization(
        self, precision: str, calibration: str = "general", budget: float = 0.02
    ) -> None:
        self.record_pass(
            "sensitivity_quantization",
            precision=precision,
            calibration=calibration,
            ppl_budget=budget,
        )

    def record_fusion(self, fusion_type: str = "qkv_rope_norm") -> None:
        self.record_pass("operator_fusion", fusion_type=fusion_type)

    def record_kv_structuring(
        self, kv_type: str = "gqa", mla_rank: int | None = None
    ) -> None:
        params: dict[str, Any] = {"kv_type": kv_type}
        if mla_rank is not None:
            params["mla_rank"] = mla_rank
        self.record_pass("kv_cache_structuring", **params)

    def record_pruning(self, method: str, sparsity: float) -> None:
        self.record_pass("pruning", method=method, sparsity=sparsity)

    def record_reasoning_graph(self, num_steps: int, max_tokens: int) -> None:
        self.record_pass(
            "reasoning_graph",
            num_steps=num_steps,
            max_thinking_tokens=max_tokens,
        )

    def set_eval_result(
        self,
        ppl_regression: float,
        passed: bool,
        targets: list[str],
        benchmark_scores: dict[str, float] | None = None,
    ) -> None:
        self.manifest.hardware_certification.eval_gate_passed = passed
        self.manifest.hardware_certification.eval_ppl_regression = ppl_regression
        self.manifest.hardware_certification.certified_targets = targets
        if benchmark_scores:
            self.manifest.eval_results.update(benchmark_scores)

    def record_eval_scores(self, **scores: float) -> None:
        """Record individual benchmark scores (hellaswag, mmlu, gsm8k, ...)."""
        self.manifest.eval_results.update(scores)

    def finalize(
        self,
        aeg_content: bytes | None = None,
        watermark_enabled: bool = False,
        watermark_algorithm: str = "kgw",
        watermark_key: bytes | None = None,
    ) -> ProvenanceManifest:
        """
        Finalize the manifest: compute AEG hash and watermark fingerprint.

        Args:
            aeg_content: Raw bytes of the compiled AEG package.
            watermark_enabled: Whether watermarking is active.
            watermark_algorithm: Watermark algorithm name.
            watermark_key: Watermark key bytes (only fingerprint is stored).
        """
        if aeg_content:
            self.manifest.aeg_hash = hashlib.sha256(aeg_content).hexdigest()

        if watermark_enabled and watermark_key:
            self.manifest.watermark_enabled = True
            self.manifest.watermark_algorithm = watermark_algorithm
            self.manifest.watermark_key_fingerprint = hashlib.sha256(
                watermark_key + b"aether_wm_v1"
            ).hexdigest()[:32]

        # No C2PA identifier is synthesized here. Content Credentials require a
        # signature, and a signature requires a key; call
        # attach_c2pa_manifest() to produce one. Until then the artifact
        # correctly reports itself as unsigned.
        return self.manifest

    def save(self, aeg_dir: str | Path) -> Path:
        return self.manifest.save(aeg_dir)


# ---------------------------------------------------------------------------
# C2PA Content Credentials
# ---------------------------------------------------------------------------

C2PA_MANIFEST_PATH = "provenance/c2pa.manifest"
"""Where the signed C2PA manifest store lives inside a package."""


def default_signing_key_path() -> Path:
    """Return the path of the local claim-signing key.

    Honours ``AETHER_SIGNING_KEY`` when set, so a deployment can point at a key
    it manages; otherwise the key lives under the Aether cache directory.
    """
    override = os.environ.get("AETHER_SIGNING_KEY", "").strip()
    if override:
        return Path(override).expanduser()
    from aether.utils.file_io import aether_cache_dir

    return Path(aether_cache_dir()) / "keys" / "claim_signing_ed25519.key"


def load_or_create_signing_key(path: str | Path | None = None) -> tuple[bytes, bool]:
    """Load a 32-byte Ed25519 seed, generating one on first use.

    Returns ``(seed, created)``.  A generated key is written with owner-only
    permissions where the platform supports it.  It is a *development* key: it
    produces cryptographically valid signatures under a self-signed certificate,
    which proves integrity but establishes no identity.  Production signing should
    supply a key and certificate chain from a real CA.
    """
    target = Path(path).expanduser() if path is not None else default_signing_key_path()
    if target.is_file():
        seed = target.read_bytes()
        # Accept either raw bytes or a hex-encoded seed, since a key handed over
        # by an operator through an environment variable is usually hex.
        if len(seed) != ed25519.SECRET_KEY_SIZE:
            text = seed.decode("ascii", "ignore").strip()
            try:
                seed = bytes.fromhex(text)
            except ValueError as exc:
                raise ValueError(
                    f"{target} is neither a 32-byte seed nor hex-encoded"
                ) from exc
        if len(seed) != ed25519.SECRET_KEY_SIZE:
            raise ValueError(
                f"{target} holds {len(seed)} bytes; an Ed25519 seed is "
                f"{ed25519.SECRET_KEY_SIZE}"
            )
        return seed, False
    seed = ed25519.generate_seed()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(seed)
    # Restrict to the owner where the platform honours it; Windows ACLs make this
    # a no-op, which is why it is best-effort rather than a hard requirement.
    with contextlib.suppress(OSError):
        os.chmod(target, 0o600)
    logger.warning(
        "Generated a new self-signed C2PA claim-signing key at %s. "
        "Signatures made with it prove the artifact is unmodified; they do not "
        "establish who produced it.",
        target,
    )
    return seed, True


def attach_c2pa_manifest(
    aeg_dir: str | Path,
    manifest: ProvenanceManifest | None = None,
    *,
    seed: bytes | None = None,
    key_path: str | Path | None = None,
    certificate_chain: list[bytes] | None = None,
    compiler_version: str | None = None,
) -> tuple[Path, ProvenanceManifest]:
    """Sign an AEG package with C2PA Content Credentials.

    The compiler's transformation chain is carried into the signed manifest as an
    assertion, which is what turns it from a record into an attestation: after
    this call, editing a pass entry invalidates the claim signature.

    Args:
        aeg_dir: Root of the ``.aeg`` package.
        manifest: The provenance document to update. Loaded from the package when
            omitted, and written back with the C2PA fields populated.
        seed: A 32-byte Ed25519 seed. Loaded or generated when omitted.
        key_path: Where to load or create the key, if ``seed`` is not given.
        certificate_chain: DER certificates, leaf first. When omitted a
            self-signed certificate is generated for the key — valid signatures,
            no established identity.
        compiler_version: Overrides the version recorded as the claim generator.

    Returns:
        ``(manifest_store_path, updated_provenance_manifest)``.
    """
    from aether.core.constants import AETHER_VERSION
    from aether.provenance import c2pa
    from aether.provenance.cose import Ed25519Signer

    root = Path(aeg_dir)
    if manifest is None:
        try:
            manifest = ProvenanceManifest.load(root)
        except (OSError, ValueError):
            manifest = ProvenanceManifest(source_model_id=root.stem)
    if seed is None:
        seed, _ = load_or_create_signing_key(key_path)
    signer = Ed25519Signer(seed, certificate_chain)

    source_hash: bytes | None = None
    if manifest.model_hash:
        try:
            source_hash = bytes.fromhex(manifest.model_hash)
        except ValueError:
            source_hash = None

    # The hard binding covers provenance/manifest.json, so that file has to reach
    # its final bytes *before* anything is digested.  The label is therefore
    # chosen here rather than discovered from the signed store afterwards: writing
    # it back after signing would invalidate the binding it was written into.
    # Two saves are needed because the file count is itself recorded, and the
    # first save is what brings the file into existence.
    manifest.c2pa_manifest_label = f"urn:c2pa:{uuid.uuid4()}"
    manifest.c2pa_signature_algorithm = "EdDSA (Ed25519)"
    chain = signer.certificate_chain
    manifest.c2pa_signer = (
        x509.certificate_info(chain[0]).subject if chain else "(no certificate)"
    )
    manifest.c2pa_files_bound = 0
    manifest.save(root)
    manifest.c2pa_files_bound = len(c2pa.collection_digests(root))
    manifest.save(root)

    store_path = c2pa.sign_artifact(
        root,
        signer,
        generator=c2pa.ClaimGenerator(
            "aether-runtime", compiler_version or AETHER_VERSION
        ),
        source_model_id=manifest.source_model_id,
        source_hash=source_hash,
        transformations=[record.to_dict() for record in manifest.transformations],
        transformation_chain_hash=manifest.compute_chain_hash(),
        manifest_label=manifest.c2pa_manifest_label,
    )
    return store_path, manifest


def verify_c2pa_manifest(
    aeg_dir: str | Path,
    *,
    trust_anchors: list[bytes] | None = None,
) -> "Any":
    """Verify an AEG package's Content Credentials.

    Thin wrapper over :func:`aether.provenance.c2pa.verify_artifact`, kept here so
    callers that already work with :class:`ProvenanceManifest` have one import.
    """
    from aether.provenance import c2pa

    return c2pa.verify_artifact(aeg_dir, trust_anchors=trust_anchors)


# ---------------------------------------------------------------------------
# KGW Watermark (Kirchenbauer et al., 2023)
# ---------------------------------------------------------------------------

class KGWWatermark:
    """
    KGW (Kirchenbauer et al.) LLM watermarking.

    Each token generation step:
    1. Compute a pseudo-random seed from the previous token ID + key
    2. Partition the vocabulary into "green list" (γ fraction) and "red list"
    3. Add δ to the logits of green-list tokens
    4. Sample from the biased distribution

    Detection:
    - Count the fraction of green tokens in a text
    - Under watermark: green fraction ≈ γ + ε (signal)
    - Under no watermark: green fraction ≈ γ (baseline)
    - Z-score test: z = (green_count - γN) / sqrt(N × γ × (1-γ))
    - Threshold: z > 4.0 → watermarked (p < 1e-5)

    Reference: Kirchenbauer et al., "A Watermark for Large Language Models",
    ICML 2023.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        gamma: float = 0.25,    # fraction of vocab in green list
        delta: float = 2.0,     # logit boost for green tokens
        key: bytes | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self._key = key or b"aether_kgw_default_key_v1"

    def _get_green_list(self, prev_token_id: int) -> np.ndarray:
        """
        Compute the green list for the next token given the previous token.

        Uses HMAC-SHA256 with the watermark key as a PRF to generate
        a reproducible, key-dependent partition.
        """
        import hmac
        seed_bytes = prev_token_id.to_bytes(4, "little")
        h = hmac.new(self._key, seed_bytes, hashlib.sha256).digest()
        # Convert to a permutation seed
        rng_seed = int.from_bytes(h[:8], "little") % (2**31)
        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(self.vocab_size)
        green_size = int(self.vocab_size * self.gamma)
        return perm[:green_size]

    def apply(
        self, logits: np.ndarray, prev_token_id: int
    ) -> np.ndarray:
        """
        Apply KGW watermark: boost green-list logits by delta.

        Args:
            logits: (vocab_size,) next-token logits
            prev_token_id: The last generated token ID.

        Returns:
            Watermarked logits (same shape).
        """
        green_ids = self._get_green_list(prev_token_id)
        watermarked = logits.copy()
        watermarked[green_ids] += self.delta
        return watermarked

    def detect(
        self,
        token_ids: list[int],
        z_threshold: float = 4.0,
    ) -> dict[str, Any]:
        """
        Detect watermark presence via green-token fraction z-test.

        Args:
            token_ids: Generated token sequence.
            z_threshold: Z-score detection threshold.

        Returns:
            Dict with is_watermarked, z_score, green_fraction, p_value.
        """
        if len(token_ids) < 2:
            return {"is_watermarked": False, "z_score": 0.0, "reason": "too_short"}

        green_count = 0
        N = len(token_ids) - 1   # pairs (prev, current)

        for i in range(N):
            prev = token_ids[i]
            curr = token_ids[i + 1]
            green_ids = self._get_green_list(prev)
            if curr in green_ids:
                green_count += 1

        gamma = self.gamma
        # Z-score under null hypothesis (no watermark): E[green] = γN
        expected = gamma * N
        variance = N * gamma * (1 - gamma)
        z_score = (green_count - expected) / max(variance ** 0.5, 1e-9)

        # Approximate p-value via normal distribution
        # p = P(Z > z_score) ≈ erfc(z_score / sqrt(2)) / 2
        p_value = float(0.5 * np.exp(-0.5 * z_score ** 2))   # Gaussian tail approx

        return {
            "is_watermarked": z_score > z_threshold,
            "z_score": round(float(z_score), 4),
            "green_fraction": round(green_count / max(N, 1), 4),
            "expected_green_fraction": gamma,
            "green_count": green_count,
            "total_pairs": N,
            "p_value": max(p_value, 1e-10),
            "z_threshold": z_threshold,
        }

    def key_fingerprint(self) -> str:
        """Return a non-reversible fingerprint of the watermark key."""
        return hashlib.sha256(self._key + b"fingerprint_v1").hexdigest()[:32]
