"""Own serial quality workers and admit an exact unpublished AEG candidate.

Workers measure real HF/AEG execution. Hashes bind those measurements to the
checkpoint, fixed token probes, and every candidate payload. Admission is a
numerical regression check, not a corpus benchmark or hardware certification.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aether.core.exceptions import CompilationError
from aether.core.hash_utils import compute_file_hash
from aether.core.quality_gate import (
    EXECUTORS,
    GATE_PATH,
    GATE_SCHEMA,
    NumericalQualityError,
    artifact_binding,
    canonical_hash,
    read_gate,
    validate_report,
)
from aether.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import psutil

    from aether.compiler.config import CompilerConfig
    from aether.core.aeg_format import AEGPackage

logger = get_logger(__name__)
MIB = 1024 * 1024


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def materialize_source(model: str, config: CompilerConfig) -> Path:
    """Resolve config, HF reference and ingestion to a single revision snapshot."""
    local = Path(model)
    if local.is_dir():
        return local.absolute()
    if local.exists() or config.skip_download:
        raise NumericalQualityError(
            "HF numerical quality admission requires a local SafeTensors checkpoint directory; "
            "other formats require an original HF reference or explicit diagnostic compilation"
        )
    from huggingface_hub import HfApi, snapshot_download

    revision: str | None = config.model_revision
    if not isinstance(revision, str):
        raise NumericalQualityError("model_revision must be a string")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        info = HfApi().model_info(model, revision=revision)
        revision = info.sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise NumericalQualityError("Hub did not resolve the model to an immutable commit")
    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            cache_dir=str(Path(config.cache_dir) / "huggingface"),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
        )
    ).absolute()


def source_binding(source: Path) -> dict[str, Any]:
    """Stream-hash consumed source shards, indexes, config and tokenizer inputs."""
    from aether.compiler.stage1_ingestion.safetensors_loader import SafeTensorsLoader

    if (source / "adapter_config.json").exists():
        raise NumericalQualityError(
            "HF adapter-source admission needs an independent merged reference"
        )
    config_path = source / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("quantization_config") is not None:
        raise NumericalQualityError("HF numerical reference requires an unquantized source config")
    paths = {config_path, *SafeTensorsLoader(source).discover_files()}
    for pattern in ("*.json", "*.model", "*.txt", "*.tiktoken"):
        paths.update(source.glob(pattern))
    files = {}
    for path in sorted(paths):
        name = path.relative_to(source).as_posix()
        files[name] = {"sha256": compute_file_hash(path), "bytes": path.stat().st_size}
    return {"files": files, "digest": canonical_hash(files)}


def _owned(root: psutil.Process, created: float) -> list[psutil.Process]:
    """Inspect only a live child tree whose launcher identity still matches."""
    import psutil

    try:
        if root.create_time() != created or not root.is_running():
            return []
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def run_worker(
    work: Path,
    name: str,
    job: dict[str, Any],
    config: CompilerConfig,
    reserve: int,
) -> dict[str, Any]:
    """Admit and monitor one fresh worker; prove PID ownership before imports.

    Estimates and sampled RSS are kept in compiler logs, outside hashed quality
    evidence. The guard includes available host RAM after the parent's retained
    graph; it never assumes a Windows venv launcher is the Python worker.
    """
    import psutil

    headroom = config.quality_memory_headroom_mib * MIB
    available = int(psutil.virtual_memory().available)
    if available < reserve + headroom:
        raise NumericalQualityError(
            f"numerical quality {name} RAM admission refused: available={available}, "
            f"estimated_worker={reserve}, required_headroom={headroom} bytes"
        )
    for marker_name in ("identity.json", "ack.json"):
        (work / marker_name).unlink(missing_ok=True)
    _write(work / "job.json", job)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AETHER_TORCH_DTYPE": "float32",
            "AETHER_CACHE_DIR": str(work / "cache"),
        }
    )
    started = time.monotonic()
    identity: dict[str, Any] | None = None
    worker_peak, tree_peak = 0, 0
    minimum_available = available
    process: subprocess.Popen[bytes] | None = None
    root = None
    created = 0.0
    with (work / f"{name}.log").open("wb") as log:
        try:
            process = subprocess.Popen(
                [sys.executable, "-B", "-m", "aether.evaluation.admission_worker", str(work), name],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            root = psutil.Process(process.pid)
            created = root.create_time()
            while process.poll() is None:
                owned = _owned(root, created)
                if identity is None and (work / "identity.json").is_file():
                    # Read only after close: an incomplete write can be observed
                    # while the worker is starting; retry that specific state.
                    try:
                        marker = json.loads((work / "identity.json").read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        marker = None
                    if isinstance(marker, dict):
                        if not any(
                            marker == {"pid": item.pid, "create_time": item.create_time()}
                            for item in owned
                        ):
                            raise NumericalQualityError(
                                "quality worker PID is not an owned process"
                            )
                        identity = marker
                        _write(work / "ack.json", marker)
                tree_rss = 0
                for item in owned:
                    try:
                        rss = int(item.memory_info().rss)
                        tree_rss += rss
                        if identity == {"pid": item.pid, "create_time": item.create_time()}:
                            worker_peak = max(worker_peak, rss)
                    except psutil.NoSuchProcess:
                        continue  # This process exited between the tree and RSS snapshots.
                tree_peak = max(tree_peak, tree_rss)
                minimum_available = min(minimum_available, int(psutil.virtual_memory().available))
                if minimum_available < headroom:
                    raise NumericalQualityError(f"numerical quality {name} exhausted RAM headroom")
                if time.monotonic() - started > config.quality_worker_timeout_s:
                    raise NumericalQualityError(f"numerical quality {name} timed out")
                time.sleep(0.01)
            result_path = work / f"{name}-result.json"
            if not result_path.is_file():
                raise NumericalQualityError(f"numerical quality {name} exited without a result")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise NumericalQualityError("quality worker result must be an object")
            if identity is None or result.get("identity") != identity:
                raise NumericalQualityError("quality result has no verified owned worker identity")
            if process.returncode or result.get("status") != "ok":
                raise NumericalQualityError(
                    f"numerical quality {name} failed: {result.get('reason')}"
                )
            return result
        finally:
            if process is not None and process.poll() is None:
                if root is not None:
                    owned = _owned(root, created)
                    for child in reversed(owned):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            continue  # Owned process already exited; no unrelated PID is touched.
                    # Waiting for the Windows venv launcher alone does not wait
                    # for its Python child to release inherited log handles.
                    # Reap the entire proven tree before temporary files close.
                    _, alive = psutil.wait_procs(owned, timeout=10)
                    if alive:
                        raise NumericalQualityError(
                            "quality worker shutdown did not finish: "
                            + ", ".join(str(child.pid) for child in alive)
                        )
                process.wait(timeout=10)
            logger.info(
                "Numerical quality worker %s: seconds=%.6f worker_peak_rss=%d "
                "tree_peak_rss=%d available_min=%d estimated_reserve=%d identity=%s",
                name,
                time.monotonic() - started,
                worker_peak,
                tree_peak,
                minimum_available,
                reserve,
                identity,
            )


class NumericalAdmission:
    """A reference measured before ingestion, owned until candidate acceptance."""

    def __init__(self, source: Path, work: Path, config: CompilerConfig) -> None:
        from aether.evaluation.admission_worker import DEFAULT_PROBES

        self.source, self.work, self.config = source, work, config
        self.binding = source_binding(source)
        if config.quality_dataset is None:
            probes = [
                {"id": name, "prompt": prompt, "continuation": continuation}
                for name, prompt, continuation in DEFAULT_PROBES
            ]
        else:
            dataset = Path(config.quality_dataset)
            if dataset.stat().st_size > MIB:
                raise NumericalQualityError("quality probe JSONL exceeds 1 MiB")
            probes = [
                json.loads(line)
                for line in dataset.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.raw_bytes = sum(evidence["bytes"] for evidence in self.binding["files"].values())
        self.reference = run_worker(
            work,
            "hf",
            {
                "source": str(source),
                "probes": probes,
                "greedy_tokens": config.quality_greedy_tokens,
            },
            config,
            512 * MIB + self.raw_bytes * 4,
        )
        reference = self.reference["reference"]
        expected = canonical_hash(
            {key: self.reference[key] for key in ("samples", "tokens", "arrays")}
        )
        if reference["evidence_sha256"] != expected:
            raise NumericalQualityError("HF reference evidence digest disagrees with measurements")
        self.verify_source()

    def verify_source(self) -> None:
        """Reject changed input files instead of comparing different checkpoints."""
        if source_binding(self.source) != self.binding:
            raise NumericalQualityError(
                "source checkpoint changed during numerical quality admission"
            )

    def accept(self, package: AEGPackage) -> None:
        """Run required candidate engines and persist only passing, bound evidence."""
        if package.manifest is None:
            raise NumericalQualityError("quality candidate has no manifest")
        self.verify_source()
        binding = artifact_binding(package)
        measured = {}
        for executor in EXECUTORS:
            result = run_worker(
                self.work,
                executor,
                {
                    "artifact": str(package.root.absolute()),
                    "artifact_binding": binding,
                    "executor": executor,
                    "max_perplexity_delta": min(0.01, self.config.quality_budget),
                },
                self.config,
                (512 if executor == "torch_cpu" else 256) * MIB + self.raw_bytes * 4,
            )
            if result["artifact"] != binding:
                raise NumericalQualityError("quality worker returned another candidate binding")
            measured[executor] = result["measurements"]
        self.verify_source()
        samples = []
        for sample in self.reference["samples"]:
            inputs = {
                key: sample[key] for key in ("prompt_ids", "continuation_ids", "greedy_tokens")
            }
            samples.append(
                {
                    "id": sample["id"],
                    "prompt_tokens": len(sample["prompt_ids"]),
                    "label_tokens": len(sample["continuation_ids"]),
                    "greedy_tokens": sample["greedy_tokens"],
                    "vocab_size": package.manifest.architecture.vocab_size,
                    "input": inputs,
                    "tokens_sha256": canonical_hash(inputs),
                }
            )
        report = {
            "schema": GATE_SCHEMA,
            "status": "passed",
            "source": self.binding,
            "reference": self.reference["reference"],
            "samples": samples,
            "artifact": binding,
            "measurements": measured,
        }
        try:
            validate_report(package, report)
        except NumericalQualityError as exc:
            report["status"] = "rejected"
            raise CompilationError(
                f"Numerical quality gate rejected the candidate: {exc}",
                stage="numerical_quality",
                details=report,
            ) from exc
        path = package.root / GATE_PATH
        path.parent.mkdir(exist_ok=True)
        _write(path, report)
        package.save()
        package.verify_integrity()
        read_gate(package)


@contextmanager
def numerical_admission(model: str, config: CompilerConfig) -> Iterator[NumericalAdmission | None]:
    """Keep reference storage scoped across every compile exit, including cancellation."""
    if config.quality_mode == "diagnostic":
        yield None
        return
    try:
        unsupported = [
            name
            for enabled, name in (
                (config.enable_ttt, "TTT adaptation"),
                (config.enable_advanced_peft and bool(config.peft_adapter_paths), "PEFT adapters"),
                (
                    config.enable_model_merging and bool(config.model_merging_sources),
                    "model merging",
                ),
            )
            if enabled
        ]
        if unsupported:
            raise NumericalQualityError(
                "Original-base HF admission does not cover "
                + ", ".join(unsupported)
                + "; an independent expected transformed reference is required. "
                "Use explicit diagnostic compilation for these uncertified experimental transforms"
            )
        source = materialize_source(model, config)
        with tempfile.TemporaryDirectory(prefix="aether-quality-") as directory:
            session = NumericalAdmission(source, Path(directory), config)
            yield session
    except (NumericalQualityError, OSError, ValueError, TypeError, ImportError) as exc:
        raise CompilationError(
            f"Mandatory numerical quality admission could not complete: {exc}",
            model_id=model,
            stage="numerical_quality",
        ) from exc
