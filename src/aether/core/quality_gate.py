"""Bind measured numerical admission to an exact AEG, without loading a framework.

This is an execution policy, not a parser-format requirement. Legacy artifacts
remain inspectable; executing an artifact without evidence requires an explicit
diagnostic decision. A recorded rejection cannot be waived that way.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import MISSING, fields
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from aether.core.hash_utils import verify_content_hash

if TYPE_CHECKING:
    from aether.core.aeg_format import AEGPackage

GATE_PATH = "quality/numerical_gate.json"
GATE_SCHEMA = "aether-numerical-quality/1"
EXECUTORS = ("cpu", "torch_cpu")
REQUIRED_CHECKS = ("logits", "tokens", "perplexity")
MAX_REPORT_BYTES = 1024 * 1024
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


class NumericalQualityError(ValueError):
    """Missing, rejected, inconsistent or unbound numerical quality evidence."""


def canonical_hash(value: object) -> str:
    """Hash strict deterministic JSON, rejecting NaN and unknown value objects."""
    content = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def artifact_binding(package: AEGPackage) -> dict[str, Any]:
    """Bind every prior payload and header while excluding the report's own hash."""
    if package.manifest is None:
        raise NumericalQualityError("AEG numerical quality gate requires a loaded manifest")
    header = package.manifest.to_dict()
    header.pop("manifest_hash", None)
    artifacts = dict(header.pop("artifacts"))
    for path in package.root.rglob("*"):
        _require(not path.is_symlink(), "linked candidate payload is not admissible")
        if path.is_file():
            name = path.relative_to(package.root).as_posix()
            _require(name == "manifest.json" or name in artifacts, f"undeclared payload {name!r}")
    artifacts.pop(GATE_PATH, None)
    return {"header_sha256": canonical_hash(header), "files": artifacts}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NumericalQualityError("AEG numerical quality gate: " + message)


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        _require(name not in result, f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _nonfinite(value: str) -> Any:
    raise NumericalQualityError(f"AEG numerical quality gate contains {value}")


def _source(source: Any) -> None:
    _require(isinstance(source, dict), "source binding is missing")
    files = source.get("files")
    _require(isinstance(files, dict) and bool(files), "source file evidence is empty")
    _require("config.json" in files, "source config is not bound")
    for name, evidence in files.items():
        _require(isinstance(name, str) and bool(name), "invalid source filename")
        path = PurePosixPath(name)
        windows = PureWindowsPath(name)
        _require(
            not path.is_absolute()
            and not windows.drive
            and ".." not in path.parts
            and "\\" not in name
            and "\x00" not in name,
            "source filename is not checkpoint-relative",
        )
        _require(isinstance(evidence, dict), "source file evidence must be an object")
        _require(_hash(evidence.get("sha256")), "source file hash is missing")
        size = evidence.get("bytes")
        _require(type(size) is int and size >= 0, "source file size is invalid")
    _require(
        any(name.endswith((".safetensors", ".bin", ".pt", ".pth")) for name in files),
        "source checkpoint weights are not bound",
    )
    _require(source.get("digest") == canonical_hash(files), "source digest disagrees with files")


def _reference(reference: Any) -> None:
    _require(isinstance(reference, dict), "HF reference evidence is missing")
    _require(reference.get("kind") == "huggingface", "reference must be actual Hugging Face")
    _require(reference.get("dtype") == "float32", "reference must preserve source FP32 values")
    versions = reference.get("versions")
    _require(isinstance(versions, dict), "reference versions are missing")
    for name in ("torch", "transformers"):
        version = versions.get(name)
        _require(isinstance(version, str) and 0 < len(version) <= 128, f"missing {name} version")
    _require(_hash(reference.get("evidence_sha256")), "measured HF evidence hash is missing")


def _samples(samples: Any) -> dict[str, dict[str, Any]]:
    _require(isinstance(samples, list) and 0 < len(samples) <= 128, "need 1-128 fixed samples")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        _require(isinstance(sample, dict), "sample evidence must be an object")
        name = sample.get("id")
        _require(isinstance(name, str) and bool(name) and name not in result, "invalid sample ID")
        for count in ("prompt_tokens", "label_tokens", "greedy_tokens", "vocab_size"):
            _require(_positive_int(sample.get(count)), f"invalid {count} count")
        inputs = sample.get("input")
        _require(isinstance(inputs, dict), "fixed token inputs are missing")
        for key, count in (("prompt_ids", "prompt_tokens"), ("continuation_ids", "label_tokens")):
            ids = inputs.get(key)
            _require(
                isinstance(ids, list)
                and len(ids) == sample[count]
                and all(type(token) is int and 0 <= token < sample["vocab_size"] for token in ids),
                "fixed token input count or vocabulary is invalid",
            )
        _require(inputs.get("greedy_tokens") == sample["greedy_tokens"], "greedy count disagrees")
        _require(
            sample.get("tokens_sha256") == canonical_hash(inputs),
            "sample token evidence is not bound",
        )
        result[name] = sample
    return result


def _component(raw: Any, kind: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise NumericalQualityError("AEG numerical quality gate metric evidence is missing")
    declared = {field.name: field for field in fields(kind)}
    required = {
        name
        for name, field in declared.items()
        if field.default is MISSING and field.default_factory is MISSING
    }
    _require(required <= raw.keys() <= declared.keys(), "metric fields do not match the schema")
    return raw


def validate_report(package: AEGPackage, report: Any) -> dict[str, Any]:
    """Recheck concrete metric DTOs, thresholds, cardinality and AEG bindings.

    ValidationResult owns the stable KL/cosine/greedy/fixed-label PPL consistency
    rules and their mathematical references. Its public evidence_passed property
    recomputes acceptance; a serialized success flag is never sufficient.
    """
    if not isinstance(report, dict):
        raise NumericalQualityError("AEG numerical quality gate report must be an object")
    _require(report.get("schema") == GATE_SCHEMA, "unsupported or missing evidence schema")
    _require(report.get("status") == "passed", "recorded numerical quality rejection")
    _source(report.get("source"))
    _reference(report.get("reference"))
    samples = _samples(report.get("samples"))
    if package.manifest is None:
        raise NumericalQualityError("AEG numerical quality gate: manifest is not loaded")
    budget = package.manifest.optimization.quality_budget_ppl_increase
    _require(_finite(budget) and 0 <= budget <= 1, "invalid requested perplexity budget")
    for sample in samples.values():
        _require(
            sample["vocab_size"] == package.manifest.architecture.vocab_size,
            "measured vocabulary disagrees with the artifact",
        )
    _require(
        report.get("artifact") == artifact_binding(package), "evidence belongs to other AEG bytes"
    )
    measured = report.get("measurements")
    if not isinstance(measured, dict) or set(measured) != set(EXECUTORS):
        raise NumericalQualityError("AEG numerical quality gate executor evidence is incomplete")

    # Lazy import keeps absent-evidence inspection/diagnostic paths independent
    # of metric machinery; the runtime never imports torch or transformers here.
    from aether.evaluation.quality_validator import (
        LogitValidation,
        PerplexityValidation,
        TokenValidation,
        ValidationResult,
    )

    for executor in EXECUTORS:
        rows = measured[executor]
        _require(
            isinstance(rows, dict) and rows.keys() == samples.keys(),
            "sample evidence is incomplete",
        )
        for name, raw in rows.items():
            sample = samples[name]
            _require(
                isinstance(raw, dict) and set(raw) == set(REQUIRED_CHECKS),
                "required checks are missing",
            )
            logits = LogitValidation(**_component(raw["logits"], LogitValidation))
            tokens = TokenValidation(**_component(raw["tokens"], TokenValidation))
            perplexity = PerplexityValidation(**_component(raw["perplexity"], PerplexityValidation))
            _require(
                _finite(logits.threshold_kl)
                and 0 <= logits.threshold_kl <= 0.01
                and _finite(logits.threshold_cosine)
                and 0.999 <= logits.threshold_cosine <= 1
                and _finite(tokens.threshold)
                and 0.95 <= tokens.threshold <= 1
                and _finite(perplexity.threshold_relative_delta)
                and 0 <= perplexity.threshold_relative_delta <= min(0.01, budget),
                "quality requirements were relaxed",
            )
            _require(
                logits.num_positions == sample["label_tokens"]
                and perplexity.num_tokens == sample["label_tokens"]
                and tokens.reference_token_count == sample["greedy_tokens"]
                and tokens.candidate_token_count == sample["greedy_tokens"],
                "measured positions do not match the fixed samples",
            )
            result = ValidationResult(
                logits=logits,
                tokens=tokens,
                perplexity=perplexity,
                passed=True,
                required_checks=REQUIRED_CHECKS,
            )
            _require(result.passed and result.evidence_passed, f"{executor}/{name} metrics failed")
    return report


def read_gate(package: AEGPackage, *, allow_uncertified: bool = False) -> dict[str, Any] | None:
    """Read authenticated numerical evidence; diagnostics waive only its absence."""
    _require(type(allow_uncertified) is bool, "diagnostic admission flag must be boolean")
    if package.manifest is None:
        raise NumericalQualityError("AEG numerical quality gate: manifest is not loaded")
    path = package.root / GATE_PATH
    expected = package.manifest.artifacts.get(GATE_PATH)
    try:
        _require(
            not path.is_symlink() and path.resolve().is_relative_to(package.root.resolve()),
            "report path escapes the artifact",
        )
        if not path.is_file():
            _require(expected is None, "declared numerical quality report is missing")
            if allow_uncertified:
                return None
            raise NumericalQualityError(
                "AEG has no measured numerical quality gate; use an explicitly diagnostic "
                "execution only for uncertified development artifacts"
            )
        if not isinstance(expected, str):
            raise NumericalQualityError(
                "AEG numerical quality report is not declared by the manifest"
            )
        _require(path.stat().st_size <= MAX_REPORT_BYTES, "numerical quality report is too large")
        with path.open("rb") as handle:
            content = handle.read(MAX_REPORT_BYTES + 1)
        _require(len(content) <= MAX_REPORT_BYTES, "numerical quality report is too large")
        _require(
            verify_content_hash(content, expected), "numerical quality report checksum mismatch"
        )
        report = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
        )
        return validate_report(package, report)
    except NumericalQualityError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise NumericalQualityError(f"AEG numerical quality gate is invalid: {exc}") from exc
