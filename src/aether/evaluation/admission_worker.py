"""Isolated original-source/HF and AEG numerical measurements.

The parent runs these workers serially. No reference weights survive into
compiler ingestion. Fixed-label alignment follows Transformers v4.57.3
``ForCausalLMLoss``: position t predicts token t+1. KL, greedy agreement and
PPL are evaluated by QualityValidator, which documents the mathematical oracle.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aether.core.hash_utils import compute_file_hash
from aether.core.quality_gate import canonical_hash

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_PROBES = (
    ("sentence", "The quick brown fox", " jumps over the lazy dog."),
    ("code", "def add(a, b):\n", "    return a + b\n"),
    ("reasoning", "There are three apples and two oranges.", " There are five fruits in total."),
)


def write_json(path: Path, value: object) -> None:
    """Write strict deterministic evidence, never NaN or fallback strings."""
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def tokenize_samples(
    records: list[dict[str, Any]], tokenizer: Any, vocab: int, context: int, greedy: int
) -> list[dict[str, Any]]:
    """Resolve a bounded, fixed probe set into explicit prefix and label IDs."""
    if not isinstance(records, list) or not 1 <= len(records) <= 128:
        raise ValueError("numerical quality requires 1-128 fixed probe records")
    samples = []
    names: set[str] = set()
    total_positions = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("quality probe record must be an object")
        name = record.get("id")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("quality probe IDs must be unique nonempty strings")
        names.add(name)
        text = {key: record.get(key) for key in ("prompt", "continuation")}
        if all(isinstance(value, str) and value for value in text.values()):
            if "prompt_ids" in record or "continuation_ids" in record:
                raise ValueError("a quality probe cannot mix text and explicit token IDs")
            ids = {
                key + "_ids": tokenizer.encode(value, add_special_tokens=False)
                for key, value in text.items()
            }
        elif any(value is not None for value in text.values()):
            raise ValueError("quality probes require both nonempty text fields")
        else:
            ids = {key: record.get(key) for key in ("prompt_ids", "continuation_ids")}
        for value in ids.values():
            if (
                not isinstance(value, list)
                or not value
                or any(type(token) is not int or not 0 <= token < vocab for token in value)
            ):
                raise ValueError("quality probes require nonempty in-vocabulary integer token IDs")
        prefix, labels = ids["prompt_ids"], ids["continuation_ids"]
        if max(len(prefix) + len(labels), len(prefix) + greedy) > context:
            raise ValueError("fixed quality probe exceeds the source context; refusing truncation")
        total_positions += len(labels)
        if total_positions > 4096:
            raise ValueError("quality probe set exceeds 4096 fixed-label positions")
        samples.append({"id": name, **ids, "greedy_tokens": greedy, **text})
    return samples


def _hf(job: dict[str, Any], work: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

    torch.set_num_threads(1)
    source = job["source"]
    # Inspect every stored dtype before HF can silently narrow an F64 source or
    # reinterpret a pre-quantized checkpoint as an original FP32 oracle.
    for shard in SafeTensorsLoader(source).discover_files():
        with safe_open(shard, framework="numpy") as handle:
            for name in sorted(handle.keys()):
                if handle.get_slice(name).get_dtype() not in {"F32", "F16", "BF16"}:
                    raise ValueError(
                        f"HF quality reference requires floating source weights: {name}"
                    )
    loaded: object = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("HF loader did not return model and loading evidence")
    model, loading = loaded
    if not isinstance(loading, dict) or any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"original HF checkpoint did not load completely: {loading}")
    model.eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("HF reference did not materialize all parameters as FP32")
    # AutoTokenizer's dynamic factory has no typed signature in Transformers
    # 4.57.3. Keep that third-party API boundary explicit without suppressions.
    tokenizer_factory: Callable[..., Any] = AutoTokenizer.from_pretrained
    tokenizer = tokenizer_factory(source, local_files_only=True, trust_remote_code=False)
    config = model.config
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("a pre-quantized source is not the original FP32 quality oracle")
    context = int(getattr(config, "max_position_embeddings", 0) or 0)
    if context <= 0:
        raise ValueError("HF quality reference requires an explicit source context bound")
    vocab = int(config.vocab_size)
    samples = tokenize_samples(job["probes"], tokenizer, vocab, context, job["greedy_tokens"])
    result: dict[str, Any] = {"samples": samples, "tokens": {}, "arrays": {}}
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            prefix, labels = sample["prompt_ids"], sample["continuation_ids"]
            # Do not include the unlabelled final position in logit/PPL checks.
            ids = torch.tensor([prefix + labels[:-1]], dtype=torch.long)
            rows = model(ids, use_cache=False).logits[0, len(prefix) - 1 :].float().numpy()
            if rows.shape != (len(labels), vocab) or not np.isfinite(rows).all():
                raise ValueError("HF reference produced invalid fixed-label logits")
            path = work / f"reference-{index}.npy"
            np.save(path, rows, allow_pickle=False)
            result["arrays"][sample["id"]] = compute_file_hash(path)
            generated = list(prefix)
            for _ in range(sample["greedy_tokens"]):
                logits = model(torch.tensor([generated]), use_cache=False).logits[0, -1]
                if logits.shape != (vocab,) or not torch.isfinite(logits).all():
                    raise ValueError("HF reference produced invalid greedy logits")
                generated.append(int(logits.argmax()))
            result["tokens"][sample["id"]] = generated[len(prefix) :]
    result["reference"] = {
        "kind": "huggingface",
        "dtype": "float32",
        "method": "original source; eager attention; full-prefix greedy; fixed-label NLL",
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "transformers")},
        "evidence_sha256": canonical_hash(result),
    }
    return result


def _candidate(job: dict[str, Any], work: Path) -> dict[str, Any]:
    from dataclasses import asdict

    import numpy as np

    from aether.core.aeg_format import AEGPackage
    from aether.core.quality_gate import NumericalQualityError, artifact_binding
    from aether.evaluation.quality_validator import QualityValidator, _log_probs
    from aether.runtime.aeg_loader import load_engine_from_package

    package = AEGPackage(job["artifact"]).load()
    binding = artifact_binding(package)
    if binding != job["artifact_binding"]:
        raise NumericalQualityError(
            "numerical quality worker candidate binding disagrees with parent"
        )
    # Candidate execution is deliberately diagnostic until its own evidence is
    # complete. This flag never authorizes a public deployment or a rejection.
    engine: Any = load_engine_from_package(package, allow_uncertified=True)
    if job["executor"] == "torch_cpu":
        import torch

        from aether.runtime.torch_engine import TorchAEGEngine

        torch.set_num_threads(1)
        if type(engine).__name__ != "CPUExecutionEngine":
            raise ValueError(
                "HF numerical admission of this specialized torch executor is not implemented"
            )
        engine = TorchAEGEngine(engine, "cpu")
    elif job["executor"] != "cpu":
        raise ValueError("unknown quality executor")
    reference = json.loads((work / "hf-result.json").read_text(encoding="utf-8"))
    validator = QualityValidator(max_perplexity_delta=job["max_perplexity_delta"])
    measurements = {}
    from aether.backends.native_cpu_backend import PackagedTokenizer

    tokenizer = PackagedTokenizer(package.root / "tokenizer" / "tokenizer.json")
    for index, sample in enumerate(reference["samples"]):
        prefix, labels = sample["prompt_ids"], sample["continuation_ids"]
        if sample["prompt"] is not None:
            for key in ("prompt", "continuation"):
                encoded = tokenizer(sample[key], add_special_tokens=False)["input_ids"][0].tolist()
                if encoded != sample[key + "_ids"]:
                    raise ValueError(
                        "AEG tokenizer disagrees with the original reference tokenizer"
                    )
        path = work / f"reference-{index}.npy"
        if compute_file_hash(path) != reference["arrays"][sample["id"]]:
            raise ValueError("reference logit evidence changed before comparison")
        original = np.load(path, allow_pickle=False)
        rows, _ = engine.forward_step(prefix + labels[:-1], None, return_all_logits=True)
        actual = np.asarray(rows)[len(prefix) - 1 :].copy()
        if actual.shape != original.shape:
            raise ValueError("candidate fixed-label shape disagrees with original HF")
        generated = []
        pending, cache = prefix, None
        for _ in range(sample["greedy_tokens"]):
            logits, cache = engine.forward_step(pending, cache)
            values = np.asarray(logits)
            if values.shape != (original.shape[-1],) or not np.isfinite(values).all():
                raise ValueError("candidate produced invalid greedy logits")
            pending = [int(values.argmax())]
            generated.extend(pending)
        positions = np.arange(len(labels))
        original_nll = -_log_probs(original.astype(np.float64))[positions, labels]
        actual_nll = -_log_probs(actual.astype(np.float64))[positions, labels]
        measurements[sample["id"]] = {
            "logits": asdict(validator.validate_logits(original, actual)),
            "tokens": asdict(
                validator.validate_greedy_agreement(reference["tokens"][sample["id"]], generated)
            ),
            "perplexity": asdict(validator.validate_perplexity(original_nll, actual_nll)),
        }
    package.verify_integrity()
    if artifact_binding(package) != binding:
        raise NumericalQualityError(
            "numerical quality worker candidate binding changed during execution"
        )
    return {"measurements": measurements, "artifact": binding}


def main() -> int:
    """Publish identity before framework imports, then one concrete worker result."""
    import psutil

    work = Path(sys.argv[1])
    name = sys.argv[2]
    process = psutil.Process()
    identity = {"pid": os.getpid(), "create_time": process.create_time()}
    write_json(work / "identity.json", identity)
    # Wait until the parent has proved this PID belongs to its child tree. This
    # also handles the Windows venv launcher, whose PID is not the Python worker.
    import time

    deadline = time.monotonic() + 30
    while not (work / "ack.json").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("quality parent did not acknowledge owned worker identity")
        time.sleep(0.01)
    job = json.loads((work / "job.json").read_text(encoding="utf-8"))
    try:
        result = _hf(job, work) if name == "hf" else _candidate(job, work)
        result.update(status="ok", identity=identity)
        write_json(work / f"{name}-result.json", result)
    except Exception as exc:
        traceback.print_exc()
        write_json(
            work / f"{name}-result.json",
            {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "identity": identity},
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
