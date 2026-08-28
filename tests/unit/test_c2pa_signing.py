"""C2PA Content Credentials: cryptography, format, and tamper-evidence.

Three separable things are checked, because a provenance implementation can be
wrong in three independent ways:

1. **The primitives** match their specifications — CBOR against RFC 8949's
   Appendix A vectors, Ed25519 against RFC 8032 §7.1, JUMBF against a byte-exact
   round trip.  Without this the manifest may be well-formed and still
   unverifiable by anyone else.
2. **The structure** is the one C2PA defines: a JUMBF store holding a claim, an
   assertion store and a detached COSE_Sign1 signature.
3. **Tampering is detected** at every layer — a changed weight file, a changed
   assertion, a re-encoded claim, a flipped signature bit, a substituted key.
   This is the whole point, and it is the part a hash chain without a signature
   cannot do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.provenance import c2pa, cose, ed25519, jumbf, x509
from aether.provenance.cbor import CBORDecodeError, CBORTag, Undefined, dumps, loads

SEED = bytes(range(32))


@pytest.fixture()
def package(tmp_path: Path) -> Path:
    """A minimal AEG-shaped package: nested directories and binary payloads."""
    root = tmp_path / "fixture.aeg"
    (root / "graph").mkdir(parents=True)
    (root / "weights").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"model": "fixture"}), encoding="utf-8")
    (root / "graph" / "computation_graph.aeg-ir").write_bytes(b"\x00graph\x01" * 64)
    (root / "weights" / "layer0.bin").write_bytes(bytes(range(256)) * 16)
    return root


@pytest.fixture()
def signer() -> cose.Ed25519Signer:
    return cose.Ed25519Signer(SEED)


# ── RFC 8949: deterministic CBOR ──────────────────────────────────────────────

RFC8949_VECTORS = [
    (0, "00"), (23, "17"), (24, "1818"), (1000, "1903e8"),
    (1000000000000, "1b000000e8d4a51000"),
    (-1, "20"), (-1000, "3903e7"),
    (0.0, "f90000"), (1.5, "f93e00"), (100000.0, "fa47c35000"),
    (1.1, "fb3ff199999999999a"),
    (b"\x01\x02\x03\x04", "4401020304"),
    ("IETF", "6449455446"), ("ü", "62c3bc"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
    (False, "f4"), (True, "f5"), (None, "f6"), (Undefined, "f7"),
]


@pytest.mark.parametrize(("value", "expected"), RFC8949_VECTORS)
def test_cbor_matches_rfc8949_appendix_a(value: object, expected: str) -> None:
    assert dumps(value).hex() == expected


def test_cbor_map_keys_use_bytewise_order() -> None:
    """§4.2.1 item 3: sort by the encoded key, not by the Python value."""
    encoded = dumps({"aa": 1, "b": 2, 10: 3, -1: 4}).hex()
    assert encoded == "a40a03200461620262616101"


def test_cbor_integer_valued_float_stays_a_float() -> None:
    """Narrowing 1.0 to 1 would change the data model, not just the bytes."""
    assert dumps(1.0) == bytes.fromhex("f93c00")
    assert isinstance(loads(dumps(1.0)), float)


def test_cbor_nan_has_one_encoding() -> None:
    """A second NaN encoding would make two signatures over one claim differ."""
    assert dumps(float("nan")).hex() == "f97e00"


def test_cbor_rejects_indefinite_length_items() -> None:
    with pytest.raises(CBORDecodeError, match="indefinite"):
        loads(bytes.fromhex("5f42010243030405ff"))


def test_cbor_rejects_trailing_bytes() -> None:
    """Trailing bytes after a signed payload are a smuggling vector."""
    with pytest.raises(CBORDecodeError, match="trailing"):
        loads(dumps(1) + b"\x00")


def test_cbor_round_trips_cose_shaped_structures() -> None:
    envelope = CBORTag(18, [b"\xa1\x01\x27", {}, None, b"\x00" * 64])
    assert loads(dumps(envelope)) == envelope


# ── RFC 8032 §7.1: Ed25519 ────────────────────────────────────────────────────

RFC8032_VECTORS = [
    (
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
        "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
    ),
    (
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da08"
        "5ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
    ),
    (
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18"
        "ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
    ),
]


@pytest.mark.parametrize(("seed", "public", "message", "signature"), RFC8032_VECTORS)
def test_ed25519_matches_rfc8032(seed: str, public: str, message: str, signature: str) -> None:
    seed_bytes = bytes.fromhex(seed)
    assert ed25519.public_key_from_seed(seed_bytes).hex() == public
    assert ed25519.sign(bytes.fromhex(message), seed_bytes).hex() == signature
    assert ed25519.verify(
        bytes.fromhex(message), bytes.fromhex(signature), bytes.fromhex(public)
    )


def test_ed25519_rejects_a_tampered_message() -> None:
    signature = ed25519.sign(b"claim", SEED)
    public = ed25519.public_key_from_seed(SEED)
    assert not ed25519.verify(b"claim!", signature, public)


def test_ed25519_rejects_a_scalar_at_or_above_the_group_order() -> None:
    """S >= L is a malleable re-encoding of a valid signature."""
    signature = bytearray(ed25519.sign(b"claim", SEED))
    signature[32:] = (2 ** 252 + 27742317777372353535851937790883648493).to_bytes(32, "little")
    public = ed25519.public_key_from_seed(SEED)
    assert not ed25519.verify(b"claim", bytes(signature), public)


def test_ed25519_rejects_a_non_curve_public_key() -> None:
    assert not ed25519.verify(b"claim", ed25519.sign(b"claim", SEED), b"\xff" * 32)


def test_ed25519_signing_is_deterministic() -> None:
    """No per-signature nonce means two signings produce identical bytes."""
    assert ed25519.sign(b"claim", SEED) == ed25519.sign(b"claim", SEED)


def test_pure_python_and_native_backends_agree() -> None:
    """If ``cryptography`` is installed, both paths must produce one signature."""
    pytest.importorskip("cryptography")
    signer = cose.Ed25519Signer(SEED)
    assert signer.backend == "cryptography"
    assert signer.sign(b"payload") == ed25519.sign(b"payload", SEED)


# ── ISO/IEC 19566-5: JUMBF ────────────────────────────────────────────────────

def test_jumbf_content_type_uuids_follow_the_4cc_convention() -> None:
    suffix = bytes.fromhex("00110010800000AA00389B71")
    for four_cc, uuid_value in (
        (b"c2pa", jumbf.CONTENT_TYPE_C2PA_STORE),
        (b"c2ma", jumbf.CONTENT_TYPE_C2PA_MANIFEST),
        (b"c2as", jumbf.CONTENT_TYPE_C2PA_ASSERTION_STORE),
        (b"c2cl", jumbf.CONTENT_TYPE_C2PA_CLAIM),
        (b"c2cs", jumbf.CONTENT_TYPE_C2PA_SIGNATURE),
        (b"cbor", jumbf.CONTENT_TYPE_CBOR),
    ):
        assert uuid_value == four_cc + suffix


def test_jumbf_round_trip_is_byte_exact() -> None:
    """A lossy round trip would break every hashed URI in a re-read manifest."""
    store = jumbf.JUMBFBox(content_type=jumbf.CONTENT_TYPE_C2PA_STORE, label="c2pa")
    manifest = store.add(
        jumbf.JUMBFBox(content_type=jumbf.CONTENT_TYPE_C2PA_MANIFEST, label="urn:c2pa:x")
    )
    assertions = manifest.add(
        jumbf.JUMBFBox(
            content_type=jumbf.CONTENT_TYPE_C2PA_ASSERTION_STORE,
            label="c2pa.assertions",
        )
    )
    assertions.add(jumbf.cbor_box("c2pa.actions.v2", dumps({"actions": []})))
    raw = store.to_bytes()
    parsed, end = jumbf.parse_superbox(raw)
    assert end == len(raw)
    assert parsed.to_bytes() == raw


def test_jumbf_box_hash_excludes_the_superbox_header() -> None:
    """C2PA §11.1: hashing covers the description + content boxes only."""
    box = jumbf.cbor_box("c2pa.actions.v2", b"\xa0")
    assert box.hashed_content() == box.to_bytes()[8:]


def test_jumbf_duplicate_labels_resolve_to_nothing() -> None:
    """An ambiguous reference must be unresolved, never one of the candidates."""
    parent = jumbf.JUMBFBox(content_type=jumbf.CONTENT_TYPE_C2PA_STORE, label="c2pa")
    parent.add(jumbf.cbor_box("same", b"\xa0"))
    parent.add(jumbf.cbor_box("same", b"\xa1\x00\x00"))
    assert parent.find("same") is None


# ── RFC 9052: COSE_Sign1 ──────────────────────────────────────────────────────

def test_cose_sign1_is_tagged_and_detached(signer: cose.Ed25519Signer) -> None:
    envelope = loads(cose.sign_1(b"claim", signer))
    assert isinstance(envelope, CBORTag) and envelope.tag == cose.COSE_SIGN1_TAG
    protected, unprotected, payload, signature = envelope.value
    assert payload is None, "C2PA requires the claim signature to be detached"
    assert unprotected == {}
    header = loads(protected)
    assert header[cose.HEADER_ALG] == cose.ALG_EDDSA
    assert cose.HEADER_X5CHAIN in header
    assert len(signature) == ed25519.SIGNATURE_SIZE


def test_cose_verifies_with_the_key_from_the_certificate(signer: cose.Ed25519Signer) -> None:
    result = cose.verify_1(cose.sign_1(b"claim", signer), b"claim")
    assert result.signature_valid
    assert result.certificate is not None and result.certificate.is_ed25519
    assert result.certificate.public_key == signer.public_key


def test_cose_signature_covers_the_payload(signer: cose.Ed25519Signer) -> None:
    envelope = cose.sign_1(b"claim", signer)
    assert not cose.verify_1(envelope, b"claim-modified").signature_valid


def test_cose_signature_covers_the_protected_header(signer: cose.Ed25519Signer) -> None:
    """Swapping the header must fail even though the payload is untouched."""
    envelope = loads(cose.sign_1(b"claim", signer))
    envelope.value[0] = dumps({cose.HEADER_ALG: cose.ALG_ES256})
    assert not cose.verify_1(dumps(envelope), b"claim").signature_valid


def test_cose_rejects_a_flipped_signature_bit(signer: cose.Ed25519Signer) -> None:
    envelope = loads(cose.sign_1(b"claim", signer))
    tampered = bytearray(envelope.value[3])
    tampered[0] ^= 0x01
    envelope.value[3] = bytes(tampered)
    assert not cose.verify_1(dumps(envelope), b"claim").signature_valid


def test_cose_rejects_a_substituted_signing_key(signer: cose.Ed25519Signer) -> None:
    """Re-signing with another key needs the matching certificate to verify."""
    other = cose.Ed25519Signer(bytes(range(1, 33)))
    envelope = loads(cose.sign_1(b"claim", other))
    envelope.value[0] = loads(cose.sign_1(b"claim", signer)).value[0]
    assert not cose.verify_1(dumps(envelope), b"claim").signature_valid


def test_cose_requires_a_detached_payload(signer: cose.Ed25519Signer) -> None:
    result = cose.verify_1(cose.sign_1(b"claim", signer))
    assert not result.signature_valid
    assert any("detached" in error for error in result.errors)


def test_cose_refuses_to_choose_between_two_payloads(signer: cose.Ed25519Signer) -> None:
    envelope = cose.sign_1(b"claim", signer, detached=False)
    result = cose.verify_1(envelope, b"different")
    assert not result.signature_valid
    assert any("differ" in error for error in result.errors)


def test_valid_signature_is_not_reported_as_trusted(signer: cose.Ed25519Signer) -> None:
    """The distinction the whole design turns on: integrity is not identity."""
    result = cose.verify_1(cose.sign_1(b"claim", signer), b"claim")
    assert result.signature_valid
    assert not result.trusted
    assert "self-signed" in result.trust_reason


def test_configured_trust_anchor_establishes_identity(signer: cose.Ed25519Signer) -> None:
    result = cose.verify_1(
        cose.sign_1(b"claim", signer),
        b"claim",
        trust_anchors=list(signer.certificate_chain),
    )
    assert result.signature_valid and result.trusted


# ── RFC 5280 / 8410: the claim-signing certificate ────────────────────────────

def test_generated_certificate_carries_the_c2pa_claim_signer_profile() -> None:
    certificate = x509.generate_self_signed_ed25519(SEED)
    info = x509.certificate_info(certificate)
    assert info.is_ed25519 and info.self_signed
    assert info.public_key == ed25519.public_key_from_seed(SEED)
    assert "CN=" in info.subject
    # keyUsage: digitalSignature and extendedKeyUsage: emailProtection are what
    # C2PA requires of a claim signer.
    assert x509._oid(x509.OID_KEY_USAGE) in certificate
    assert x509._oid(x509.OID_EKU_EMAIL_PROTECTION) in certificate


def test_certificate_self_signature_verifies() -> None:
    certificate = x509.generate_self_signed_ed25519(SEED)
    tag, body, _ = x509._read_tlv(certificate, 0)
    assert tag == 0x30
    children = x509._children(body)
    tbs = x509._tlv(0x30, children[0][1])
    signature = children[2][1][1:]  # strip the BIT STRING unused-bits octet
    assert ed25519.verify(tbs, signature, ed25519.public_key_from_seed(SEED))


def test_certificate_parser_rejects_garbage() -> None:
    with pytest.raises(x509.DERError):
        x509.certificate_info(b"\x30\x03not-der")


# ── The manifest store, end to end ────────────────────────────────────────────

def test_sign_then_verify_an_aeg_package(package: Path, signer: cose.Ed25519Signer) -> None:
    c2pa.sign_artifact(package, signer, source_model_id="Qwen/Qwen3-0.6B")
    result = c2pa.verify_artifact(package)
    assert result.integrity_valid
    assert result.claim_signature_valid
    assert result.assertions_valid
    assert result.binding_valid
    assert result.files_bound == 3
    assert not result.errors


def test_manifest_has_the_c2pa_box_structure(package: Path, signer: cose.Ed25519Signer) -> None:
    c2pa.sign_artifact(package, signer)
    store = c2pa.read_manifest_store(package)
    assert store.label == c2pa.STORE_LABEL
    assert store.content_type == jumbf.CONTENT_TYPE_C2PA_STORE
    manifest = store.children[0]
    assert manifest.label.startswith("urn:c2pa:")
    assert manifest.content_type == jumbf.CONTENT_TYPE_C2PA_MANIFEST
    assert manifest.find(c2pa.CLAIM_LABEL) is not None
    assert manifest.find(c2pa.SIGNATURE_LABEL) is not None
    assert manifest.find(c2pa.ASSERTION_STORE_LABEL) is not None


def test_claim_is_a_v2_claim_with_hashed_uri_assertions(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    described = c2pa.describe_manifest(package)
    claim = described["claim"]
    for required in ("instanceID", "claim_generator_info", "signature", "created_assertions"):
        assert required in claim, f"claim v2 requires {required}"
    assert claim["signature"] == f"self#jumbf={c2pa.SIGNATURE_LABEL}"
    assert claim["claim_generator_info"]["name"] == "aether-runtime"
    for entry in claim["created_assertions"]:
        assert entry["url"].startswith(f"self#jumbf={c2pa.ASSERTION_STORE_LABEL}/")
        assert isinstance(entry["hash"], bytes) and len(entry["hash"]) == 32
    labels = set(described["assertions"])
    assert c2pa.HARD_BINDING_LABEL in labels
    assert c2pa.ACTIONS_LABEL in labels
    assert c2pa.INGREDIENT_LABEL in labels


def test_hard_binding_covers_every_file_except_the_manifest(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    binding = c2pa.describe_manifest(package)["assertions"][c2pa.HARD_BINDING_LABEL]
    bound = {entry["uri"] for entry in binding["uri_maps"]}
    assert bound == {
        "manifest.json",
        "graph/computation_graph.aeg-ir",
        "weights/layer0.bin",
    }
    assert c2pa.MANIFEST_FILENAME not in bound, "a manifest cannot hash itself"


def test_collection_digests_are_order_independent(package: Path) -> None:
    """Directory iteration order must not change the signed bytes."""
    first = c2pa.collection_digests(package)
    assert [entry.uri for entry in first] == sorted(entry.uri for entry in first)
    assert first == c2pa.collection_digests(package)


# ── Tamper evidence ───────────────────────────────────────────────────────────

def test_changing_a_weight_file_fails_the_binding_and_names_it(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    (package / "weights" / "layer0.bin").write_bytes(b"tampered")
    result = c2pa.verify_artifact(package)
    assert not result.integrity_valid
    assert not result.binding_valid
    assert result.changed_files == ["weights/layer0.bin"]
    # The claim itself was not touched, so its signature still checks out — which
    # is exactly how a reader learns *what* went wrong.
    assert result.claim_signature_valid


def test_deleting_a_bound_file_is_reported_as_missing(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    (package / "graph" / "computation_graph.aeg-ir").unlink()
    result = c2pa.verify_artifact(package)
    assert result.missing_files == ["graph/computation_graph.aeg-ir"]
    assert not result.integrity_valid


def test_adding_an_uncovered_file_is_reported_but_not_a_failure(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    """A kernel cache appearing later is not evidence the artifact was altered."""
    c2pa.sign_artifact(package, signer)
    (package / "kernel_cache.bin").write_bytes(b"jit")
    result = c2pa.verify_artifact(package)
    assert result.integrity_valid
    assert result.extra_files == ["kernel_cache.bin"]


def test_rewriting_an_assertion_fails_its_hashed_uri(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    """The attack a claim-only signature would miss: swap the assertion bytes."""
    c2pa.sign_artifact(package, signer)
    store = c2pa.read_manifest_store(package)
    assertions = store.children[0].find(c2pa.ASSERTION_STORE_LABEL)
    target = assertions.find(c2pa.ACTIONS_LABEL)
    target.payload = jumbf.content_box(
        "cbor", dumps({"actions": [{"action": "c2pa.created"}]})
    )
    (package / c2pa.MANIFEST_FILENAME).write_bytes(store.to_bytes())
    result = c2pa.verify_artifact(package)
    assert result.claim_signature_valid, "the claim was not modified"
    assert not result.assertions_valid
    assert not result.integrity_valid


def test_rewriting_the_claim_fails_the_signature(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    store = c2pa.read_manifest_store(package)
    manifest = store.children[0]
    claim = loads(jumbf.content_payload(manifest.find(c2pa.CLAIM_LABEL)))
    claim["dc:title"] = "a different artifact"
    manifest.find(c2pa.CLAIM_LABEL).payload = jumbf.content_box("cbor", dumps(claim))
    (package / c2pa.MANIFEST_FILENAME).write_bytes(store.to_bytes())
    result = c2pa.verify_artifact(package)
    assert not result.claim_signature_valid
    assert not result.integrity_valid


def test_removing_the_manifest_is_reported_as_unsigned(
    package: Path, signer: cose.Ed25519Signer
) -> None:
    c2pa.sign_artifact(package, signer)
    (package / c2pa.MANIFEST_FILENAME).unlink()
    result = c2pa.verify_artifact(package)
    assert not result.manifest_present
    assert not result.integrity_valid


def test_verification_of_an_unsigned_package_does_not_raise(package: Path) -> None:
    result = c2pa.verify_artifact(package)
    assert not result.manifest_present
    assert result.errors


def test_signing_an_empty_package_is_refused(tmp_path: Path, signer: cose.Ed25519Signer) -> None:
    empty = tmp_path / "empty.aeg"
    empty.mkdir()
    with pytest.raises(c2pa.C2PAError, match="no files to bind"):
        c2pa.sign_artifact(empty, signer)


# ── Integration with the provenance manifest ──────────────────────────────────

def test_provenance_manifest_reports_unsigned_before_signing(package: Path) -> None:
    from aether.provenance.manifest import ProvenanceBuilder, ProvenanceManifest

    manifest = ProvenanceBuilder(
        ProvenanceManifest(source_model_id="demo", model_hash="ab" * 32)
    ).finalize(aeg_content=b"x")
    assert manifest.c2pa_signed is False
    assert manifest.c2pa_binding == ""


def test_attach_c2pa_manifest_binds_the_provenance_document_itself(package: Path) -> None:
    """The compiler-pass chain becomes signed data, not just a recorded hash."""
    from aether.provenance.manifest import (
        ProvenanceBuilder,
        ProvenanceManifest,
        attach_c2pa_manifest,
        verify_c2pa_manifest,
    )

    builder = ProvenanceBuilder(
        ProvenanceManifest(source_model_id="Qwen/Qwen3-0.6B", model_hash="ab" * 32)
    )
    builder.record_fusion()
    builder.record_quantization("Q4_K_M")
    _, manifest = attach_c2pa_manifest(package, builder.finalize(), seed=SEED)

    assert manifest.c2pa_manifest_label.startswith("urn:c2pa:")
    assert manifest.c2pa_signed
    assert manifest.c2pa_files_bound == 4  # three fixtures + provenance/manifest.json
    assert verify_c2pa_manifest(package).integrity_valid

    described = c2pa.describe_manifest(package)
    passes = described["assertions"][c2pa.TRANSFORMATIONS_LABEL]["passes"]
    assert [entry["pass"] for entry in passes] == [
        "operator_fusion",
        "sensitivity_quantization",
    ]

    # Editing the pass chain in the readable JSON now breaks the binding.
    document = json.loads((package / "provenance" / "manifest.json").read_text())
    document["transformations"] = []
    (package / "provenance" / "manifest.json").write_text(json.dumps(document))
    after = verify_c2pa_manifest(package)
    assert not after.integrity_valid
    assert after.changed_files == ["provenance/manifest.json"]
