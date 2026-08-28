"""C2PA Content Credentials for compiled AEG artifacts.

This module implements the parts of the C2PA specification an AEG package needs
to carry a verifiable, standards-shaped provenance record:

* a **claim** (``c2pa.claim.v2``) in deterministic CBOR, listing every assertion
  by hashed URI;
* an **assertion store** holding the hard binding, the action record, the source
  checkpoint as an ingredient, and Aether's compiler-pass chain;
* a **claim signature** — a detached ``COSE_Sign1`` over the claim bytes, with
  the signer's certificate chain in the protected header;
* the whole tree serialized as a **JUMBF manifest store**.

The hard binding is ``c2pa.hash.collection.data``, which is the assertion C2PA
defines for a manifest that covers a *set* of files rather than one stream — the
right fit for an ``.aeg``, which is a directory of weights, graph, tokenizer and
metadata.  Each file gets its own digest, so verification reports *which* file
changed rather than only that something did.

Scope, stated plainly
---------------------
What is implemented is real: RFC 8949 deterministic CBOR, RFC 9052 COSE_Sign1,
Ed25519 per RFC 8032, an X.509 chain in ``x5chain``, JUMBF boxes per ISO/IEC
19566-5, and hashed URIs computed over the byte ranges C2PA §11.1 specifies. A
tampered file, a tampered claim, a tampered assertion or a tampered signature all
fail verification.

What is *not* implemented, and is reported rather than assumed:

* **Trust.** Chain validation against the C2PA trust list, OCSP/CRL revocation
  and time-stamp (``sigTst2``) countersignatures are not performed. A verified
  Aether manifest proves integrity and key possession; it does not establish the
  signer's identity, and :class:`VerificationResult` says so in
  ``trust_reason`` instead of reporting a bare "verified".
* **Embedded manifests.** The store is written as a sidecar file. There is no
  JPEG/BMFF/RIFF embedding, because an AEG is none of those formats.
* Soft bindings (watermark/fingerprint assertions), redaction and update
  manifests.

References:
  * C2PA Technical Specification 2.x — §8 (manifests), §11.1 (JUMBF and hashed
    URIs), §13 (signing), §18 (hard bindings).
  * RFC 8949 (CBOR), RFC 9052/9053 (COSE), RFC 8032 (EdDSA), RFC 5280 (X.509).
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from aether.provenance import cose, jumbf
from aether.provenance.cbor import dumps, loads
from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "C2PAError",
    "MANIFEST_FILENAME",
    "CLAIM_LABEL",
    "SIGNATURE_LABEL",
    "ASSERTION_STORE_LABEL",
    "STORE_LABEL",
    "ClaimGenerator",
    "FileDigest",
    "VerificationResult",
    "collection_digests",
    "build_manifest_store",
    "sign_artifact",
    "verify_artifact",
    "read_manifest_store",
    "describe_manifest",
]

MANIFEST_FILENAME = "provenance/c2pa.manifest"
"""Sidecar path, relative to the ``.aeg`` root, holding the JUMBF store."""

STORE_LABEL = "c2pa"
CLAIM_LABEL = "c2pa.claim.v2"
SIGNATURE_LABEL = "c2pa.signature"
ASSERTION_STORE_LABEL = "c2pa.assertions"

HARD_BINDING_LABEL = "c2pa.hash.collection.data"
ACTIONS_LABEL = "c2pa.actions.v2"
INGREDIENT_LABEL = "c2pa.ingredient.v3"
TRANSFORMATIONS_LABEL = "dev.aether.transformations"

DEFAULT_HASH_ALG = "sha256"

_IPTC_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)

_CHUNK_BYTES = 1 << 20


class C2PAError(ValueError):
    """Raised when a manifest cannot be built or is structurally invalid."""


@dataclass(frozen=True)
class ClaimGenerator:
    """The tool that produced the claim (``claim_generator_info``)."""

    name: str = "aether-runtime"
    version: str = ""

    def to_cbor_map(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": self.name}
        if self.version:
            info["version"] = self.version
        return info


@dataclass(frozen=True)
class FileDigest:
    """One entry of the collection hard binding."""

    uri: str
    """Package-relative path, always with ``/`` separators."""

    hash: bytes
    size: int

    def to_cbor_map(self) -> dict[str, Any]:
        return {"uri": self.uri, "hash": self.hash, "size": self.size}


def _digest_file(path: Path, algorithm: str) -> tuple[bytes, int]:
    hasher = hashlib.new(algorithm)
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return hasher.digest(), size


def collection_digests(
    root: str | os.PathLike[str],
    *,
    algorithm: str = DEFAULT_HASH_ALG,
    exclude: Iterable[str] = (MANIFEST_FILENAME,),
) -> list[FileDigest]:
    """Digest every file in an AEG package, in a canonical order.

    The manifest sidecar is excluded by default for the obvious reason: it cannot
    contain a hash of itself.  Ordering is by POSIX-style relative path so two
    machines with different directory-iteration order produce the same
    assertion — the ordering *is* part of the signed data.

    Raises:
        C2PAError: If ``root`` is not a directory.
    """
    base = Path(root)
    if not base.is_dir():
        raise C2PAError(f"{base} is not an AEG package directory")
    excluded = {name.replace("\\", "/") for name in exclude}
    entries: list[FileDigest] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        digest, size = _digest_file(path, algorithm)
        entries.append(FileDigest(uri=relative, hash=digest, size=size))
    entries.sort(key=lambda entry: entry.uri)
    if not entries:
        raise C2PAError(f"AEG package {base} contains no files to bind")
    return entries


def _hashed_uri(box: jumbf.JUMBFBox, label: str, algorithm: str) -> dict[str, Any]:
    """Build a ``$hashed-uri-map`` pointing at an assertion in this manifest.

    The URI is the relative ``self#jumbf`` form C2PA permits inside a manifest,
    and the hash covers the assertion box exactly as §11.1 defines it.
    """
    return {
        "url": f"self#jumbf={ASSERTION_STORE_LABEL}/{label}",
        "hash": box.digest(algorithm),
    }


# ── Assertions ────────────────────────────────────────────────────────────────

def _hard_binding_assertion(
    digests: list[FileDigest], algorithm: str
) -> dict[str, Any]:
    """Build ``c2pa.hash.collection.data``: the binding to the package bytes."""
    return {
        "alg": algorithm,
        "uri_maps": [entry.to_cbor_map() for entry in digests],
    }


def _actions_assertion(
    generator: ClaimGenerator,
    source_model_id: str,
    *,
    when: float,
) -> dict[str, Any]:
    """Build ``c2pa.actions.v2`` describing what Aether did to produce the AEG.

    The action is ``c2pa.converted``: an AEG is a compilation of an existing
    checkpoint, not a newly created work.  ``digitalSourceType`` records that the
    subject is a trained model, which is the claim an EU AI Act Art. 50 reader
    needs from the artifact itself.
    """
    action: dict[str, Any] = {
        "action": "c2pa.converted",
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when)),
        "digitalSourceType": _IPTC_TRAINED_ALGORITHMIC_MEDIA,
        "softwareAgent": generator.to_cbor_map(),
    }
    if source_model_id:
        action["parameters"] = {"org.aether.source_model": source_model_id}
    return {"actions": [action]}


def _ingredient_assertion(
    source_model_id: str,
    source_hash: bytes | None,
    algorithm: str,
) -> dict[str, Any]:
    """Build ``c2pa.ingredient.v3`` for the source checkpoint.

    The checkpoint is the ingredient the AEG was derived from; recording its hash
    is what lets a deployer prove which weights an artifact came from after the
    original has been deleted.
    """
    ingredient: dict[str, Any] = {
        "dc:title": source_model_id or "unknown",
        "relationship": "inputTo",
    }
    if source_hash:
        ingredient["data"] = {"alg": algorithm, "hash": source_hash}
    return ingredient


def _transformations_assertion(
    transformations: list[dict[str, Any]], chain_hash: str
) -> dict[str, Any]:
    """Build Aether's compiler-pass assertion.

    This is the SHA-256 transformation chain the provenance manifest already
    tracked, promoted from a bare field into a signed C2PA assertion under a
    vendor-namespaced label.  Being signed is the difference: a hash chain nobody
    signs records history, it does not attest to it.
    """
    return {
        "chain_hash": chain_hash,
        "passes": [dict(entry) for entry in transformations],
    }


# ── Manifest store construction ───────────────────────────────────────────────

def build_manifest_store(
    digests: list[FileDigest],
    signer: cose.Signer,
    *,
    generator: ClaimGenerator | None = None,
    title: str = "",
    source_model_id: str = "",
    source_hash: bytes | None = None,
    transformations: list[dict[str, Any]] | None = None,
    transformation_chain_hash: str = "",
    algorithm: str = DEFAULT_HASH_ALG,
    manifest_label: str | None = None,
    instance_id: str | None = None,
    when: float | None = None,
) -> jumbf.JUMBFBox:
    """Assemble and sign a complete C2PA manifest store.

    The order of operations is forced by the format and worth stating: assertion
    boxes are built first, because the claim references them by hash; the claim is
    serialized next; and only then is the claim signed.  Any later edit to an
    assertion invalidates its hashed URI in the claim, and any edit to the claim
    invalidates the signature — which is precisely the tamper-evidence being
    bought.

    Returns:
        The ``c2pa`` store superbox. Serialize with ``.to_bytes()``.
    """
    generator = generator or ClaimGenerator()
    moment = time.time() if when is None else when
    label = manifest_label or f"urn:c2pa:{uuid.uuid4()}"

    assertions: list[tuple[str, dict[str, Any]]] = [
        (HARD_BINDING_LABEL, _hard_binding_assertion(digests, algorithm)),
        (ACTIONS_LABEL, _actions_assertion(generator, source_model_id, when=moment)),
        (
            INGREDIENT_LABEL,
            _ingredient_assertion(source_model_id, source_hash, algorithm),
        ),
    ]
    if transformations:
        assertions.append(
            (
                TRANSFORMATIONS_LABEL,
                _transformations_assertion(transformations, transformation_chain_hash),
            )
        )

    store = jumbf.JUMBFBox(
        content_type=jumbf.CONTENT_TYPE_C2PA_STORE, label=STORE_LABEL
    )
    manifest = store.add(
        jumbf.JUMBFBox(content_type=jumbf.CONTENT_TYPE_C2PA_MANIFEST, label=label)
    )
    assertion_store = manifest.add(
        jumbf.JUMBFBox(
            content_type=jumbf.CONTENT_TYPE_C2PA_ASSERTION_STORE,
            label=ASSERTION_STORE_LABEL,
        )
    )
    created: list[dict[str, Any]] = []
    for assertion_label, value in assertions:
        box = assertion_store.add(jumbf.cbor_box(assertion_label, dumps(value)))
        created.append(_hashed_uri(box, assertion_label, algorithm))

    claim: dict[str, Any] = {
        "instanceID": instance_id or f"xmp:iid:{uuid.uuid4()}",
        "claim_generator_info": generator.to_cbor_map(),
        "signature": f"self#jumbf={SIGNATURE_LABEL}",
        "created_assertions": created,
        "alg": algorithm,
    }
    if title:
        claim["dc:title"] = title
    claim_bytes = dumps(claim)
    manifest.add(
        jumbf.cbor_box(
            CLAIM_LABEL, claim_bytes, content_type=jumbf.CONTENT_TYPE_C2PA_CLAIM
        )
    )

    # Detached: the claim already exists in its own box, so the COSE envelope
    # carries nil rather than a second copy that could drift from the first.
    envelope = cose.sign_1(claim_bytes, signer, detached=True)
    manifest.add(
        jumbf.cbor_box(
            SIGNATURE_LABEL, envelope, content_type=jumbf.CONTENT_TYPE_C2PA_SIGNATURE
        )
    )
    return store


def sign_artifact(
    aeg_path: str | os.PathLike[str],
    signer: cose.Signer,
    *,
    generator: ClaimGenerator | None = None,
    source_model_id: str = "",
    source_hash: bytes | None = None,
    transformations: list[dict[str, Any]] | None = None,
    transformation_chain_hash: str = "",
    algorithm: str = DEFAULT_HASH_ALG,
    manifest_label: str | None = None,
) -> Path:
    """Sign an AEG package and write the manifest store beside its contents.

    The sidecar is written last, after every other file has been digested, and is
    itself excluded from the binding — otherwise the artifact could never verify.

    Note that every other file in the package must already be in its final state:
    the binding covers bytes, so a file rewritten after this call no longer
    matches what was signed.  Callers that update ``provenance/manifest.json``
    with the resulting label must therefore choose the label up front and pass it
    in, which is what :func:`aether.provenance.manifest.attach_c2pa_manifest`
    does.

    Returns:
        The path of the written manifest store.
    """
    root = Path(aeg_path)
    digests = collection_digests(root, algorithm=algorithm)
    store = build_manifest_store(
        digests,
        signer,
        generator=generator,
        title=root.name,
        source_model_id=source_model_id,
        source_hash=source_hash,
        transformations=transformations,
        transformation_chain_hash=transformation_chain_hash,
        algorithm=algorithm,
        manifest_label=manifest_label,
    )
    target = root / MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(store.to_bytes())
    logger.info(
        "C2PA manifest written",
        path=str(target),
        files_bound=len(digests),
        algorithm=algorithm,
        signature_alg=signer.alg,
    )
    return target


# ── Verification ──────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """The complete outcome of validating one AEG package's manifest.

    Every check is reported separately rather than reduced to one boolean, because
    the failures mean different things: a broken ``binding_valid`` means the
    weights changed, a broken ``claim_signature_valid`` means the claim was
    re-signed or corrupted, and ``trusted`` being false may mean nothing worse
    than "no trust list was configured".
    """

    manifest_present: bool = False
    structure_valid: bool = False
    claim_signature_valid: bool = False
    assertions_valid: bool = False
    binding_valid: bool = False
    trusted: bool = False
    trust_reason: str = ""
    manifest_label: str = ""
    instance_id: str = ""
    algorithm: str = ""
    signature: cose.SignatureVerification | None = None
    files_bound: int = 0
    changed_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)
    assertion_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def integrity_valid(self) -> bool:
        """True when the artifact is intact and its claim genuinely signed.

        This is integrity, deliberately not "trusted": it proves the bytes match
        a signed claim, not who signed it. See ``trusted`` and ``trust_reason``.
        """
        return (
            self.manifest_present
            and self.structure_valid
            and self.claim_signature_valid
            and self.assertions_valid
            and self.binding_valid
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_present": self.manifest_present,
            "structure_valid": self.structure_valid,
            "claim_signature_valid": self.claim_signature_valid,
            "assertions_valid": self.assertions_valid,
            "binding_valid": self.binding_valid,
            "integrity_valid": self.integrity_valid,
            "trusted": self.trusted,
            "trust_reason": self.trust_reason,
            "manifest_label": self.manifest_label,
            "instance_id": self.instance_id,
            "algorithm": self.algorithm,
            "files_bound": self.files_bound,
            "changed_files": list(self.changed_files),
            "missing_files": list(self.missing_files),
            "extra_files": list(self.extra_files),
            "assertion_labels": list(self.assertion_labels),
            "signature": self.signature.to_dict() if self.signature else None,
            "errors": list(self.errors),
        }


def read_manifest_store(aeg_path: str | os.PathLike[str]) -> jumbf.JUMBFBox:
    """Parse the manifest store sidecar of an AEG package."""
    path = Path(aeg_path) / MANIFEST_FILENAME
    if not path.is_file():
        raise C2PAError(f"no C2PA manifest at {path}")
    store, end = jumbf.parse_superbox(path.read_bytes())
    if store.label != STORE_LABEL:
        raise C2PAError(f"manifest store label is {store.label!r}, expected 'c2pa'")
    del end
    return store


def _active_manifest(store: jumbf.JUMBFBox) -> jumbf.JUMBFBox:
    """Return the store's single manifest.

    C2PA's active manifest is the last one in the store.  Aether writes exactly
    one; a store with none is malformed, and this refuses to guess.
    """
    manifests = [
        child
        for child in store.children
        if child.content_type
        in (jumbf.CONTENT_TYPE_C2PA_MANIFEST, jumbf.CONTENT_TYPE_C2PA_UPDATE_MANIFEST)
    ]
    if not manifests:
        raise C2PAError("manifest store contains no manifest box")
    return manifests[-1]


def verify_artifact(
    aeg_path: str | os.PathLike[str],
    *,
    trust_anchors: list[bytes] | tuple[bytes, ...] | None = None,
    public_key: bytes | None = None,
) -> VerificationResult:
    """Validate an AEG package against its C2PA manifest.

    The four checks are performed in dependency order — structure, signature,
    assertion hashes, then the file binding — and all of them run even when an
    earlier one fails, so a report names every problem rather than only the first.
    """
    result = VerificationResult()
    root = Path(aeg_path)
    try:
        store = read_manifest_store(root)
    except (C2PAError, jumbf.JUMBFError) as exc:
        result.errors.append(str(exc))
        return result
    result.manifest_present = True

    try:
        manifest = _active_manifest(store)
        claim_box = manifest.find(CLAIM_LABEL)
        signature_box = manifest.find(SIGNATURE_LABEL)
        assertion_store = manifest.find(ASSERTION_STORE_LABEL)
        if claim_box is None or signature_box is None or assertion_store is None:
            raise C2PAError(
                "manifest must contain a claim, a claim signature and an "
                "assertion store"
            )
        claim_bytes = jumbf.content_payload(claim_box)
        signature_bytes = jumbf.content_payload(signature_box)
        claim = loads(claim_bytes)
        if not isinstance(claim, dict):
            raise C2PAError("claim is not a CBOR map")
    except (C2PAError, jumbf.JUMBFError, ValueError) as exc:
        result.errors.append(f"malformed manifest: {exc}")
        return result

    result.structure_valid = True
    result.manifest_label = manifest.label
    result.instance_id = str(claim.get("instanceID", ""))
    result.algorithm = str(claim.get("alg", DEFAULT_HASH_ALG))

    # 1. The claim signature, over the claim bytes exactly as stored.
    verification = cose.verify_1(
        signature_bytes,
        claim_bytes,
        public_key=public_key,
        trust_anchors=trust_anchors,
    )
    result.signature = verification
    result.claim_signature_valid = verification.signature_valid
    result.trusted = verification.trusted
    result.trust_reason = verification.trust_reason
    result.errors.extend(verification.errors)

    # 2. Every assertion the claim references must still hash to its listed value.
    #    Without this a manifest could keep a valid signature while its assertions
    #    were swapped, since the signature only covers the claim.
    assertions = _verify_assertions(claim, assertion_store, result)

    # 3. The hard binding: do the files on disk match what was signed?
    binding = assertions.get(HARD_BINDING_LABEL)
    if binding is None:
        result.errors.append("claim declares no hard binding assertion")
    else:
        _verify_binding(root, binding, result)
    return result


def _verify_assertions(
    claim: dict[str, Any],
    assertion_store: jumbf.JUMBFBox,
    result: VerificationResult,
) -> dict[str, dict[str, Any]]:
    """Check each ``created_assertions`` hashed URI and decode the assertions."""
    declared = claim.get("created_assertions")
    if not isinstance(declared, list) or not declared:
        result.errors.append("claim lists no created_assertions")
        return {}
    default_alg = str(claim.get("alg", DEFAULT_HASH_ALG))
    decoded: dict[str, dict[str, Any]] = {}
    all_valid = True
    for entry in declared:
        if not isinstance(entry, dict):
            result.errors.append("created_assertions entry is not a map")
            all_valid = False
            continue
        url = str(entry.get("url", ""))
        expected = entry.get("hash")
        algorithm = str(entry.get("alg", default_alg))
        label = url.rsplit("/", 1)[-1] if url else ""
        box = assertion_store.find(label) if label else None
        if box is None:
            result.errors.append(f"assertion {url!r} does not resolve in the store")
            all_valid = False
            continue
        result.assertion_labels.append(label)
        if not isinstance(expected, bytes):
            result.errors.append(f"assertion {label} has no hash in the claim")
            all_valid = False
            continue
        try:
            actual = box.digest(algorithm)
        except jumbf.JUMBFError as exc:
            result.errors.append(f"assertion {label}: {exc}")
            all_valid = False
            continue
        if actual != expected:
            result.errors.append(
                f"assertion {label} does not match the hash in the signed claim"
            )
            all_valid = False
            continue
        try:
            value = loads(jumbf.content_payload(box))
        except (ValueError, jumbf.JUMBFError) as exc:
            result.errors.append(f"assertion {label} is not decodable CBOR: {exc}")
            all_valid = False
            continue
        if isinstance(value, dict):
            decoded[label] = value
    result.assertions_valid = all_valid
    return decoded


def _verify_binding(
    root: Path, binding: dict[str, Any], result: VerificationResult
) -> None:
    """Re-digest the package and compare it against the signed collection hash.

    Extra files are reported but do not fail the binding: an AEG package legally
    gains files after signing (a kernel cache, a local log), and the binding's job
    is to prove that nothing *it covers* changed.  A missing or altered covered
    file does fail, and is named.
    """
    algorithm = str(binding.get("alg", DEFAULT_HASH_ALG))
    entries = binding.get("uri_maps")
    if not isinstance(entries, list) or not entries:
        result.errors.append("hard binding lists no files")
        return
    result.files_bound = len(entries)
    covered: set[str] = set()
    valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            result.errors.append("hard binding entry is not a map")
            valid = False
            continue
        relative = str(entry.get("uri", ""))
        expected = entry.get("hash")
        covered.add(relative)
        path = root / relative
        if not path.is_file():
            result.missing_files.append(relative)
            valid = False
            continue
        try:
            actual, size = _digest_file(path, algorithm)
        except (OSError, ValueError) as exc:
            result.errors.append(f"{relative}: {exc}")
            valid = False
            continue
        declared_size = entry.get("size")
        size_mismatch = isinstance(declared_size, int) and declared_size != size
        if not isinstance(expected, bytes) or actual != expected or size_mismatch:
            result.changed_files.append(relative)
            valid = False
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    } - {MANIFEST_FILENAME}
    result.extra_files = sorted(present - covered)
    result.binding_valid = valid


def describe_manifest(aeg_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a human-readable summary of an AEG package's manifest.

    Structural only: it decodes and reports, and performs no verification. Use
    :func:`verify_artifact` to find out whether any of it is still valid.
    """
    store = read_manifest_store(aeg_path)
    manifest = _active_manifest(store)
    claim_box = manifest.find(CLAIM_LABEL)
    assertion_store = manifest.find(ASSERTION_STORE_LABEL)
    claim = loads(jumbf.content_payload(claim_box)) if claim_box else {}
    assertions: dict[str, Any] = {}
    if assertion_store is not None:
        for box in assertion_store.children:
            try:
                assertions[box.label] = loads(jumbf.content_payload(box))
            except (ValueError, jumbf.JUMBFError) as exc:
                assertions[box.label] = {"error": str(exc)}
    return {
        "store_label": store.label,
        "manifest_label": manifest.label,
        "claim": claim,
        "assertions": assertions,
    }
