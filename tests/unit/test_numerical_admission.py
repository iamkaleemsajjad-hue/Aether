"""Ordinary deployment requires bound numerical evidence, not success labels."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from aether.backends.artifact_validation import ArtifactEvaluationError, validate_aeg_evaluation
from aether.compiler.config import CompilerConfig
from aether.core.quality_gate import GATE_PATH, NumericalQualityError, read_gate

if TYPE_CHECKING:
    from pathlib import Path

    from aether.backends.base import Backend
    from aether.core.aeg_format import AEGPackage


@pytest.mark.parametrize("budget", [True, float("nan"), float("inf"), -float("inf"), "0.01"])
def test_quality_budget_cannot_disable_numerical_requirements(budget: Any) -> None:
    """Invalid user policy must fail before reference execution or packaging."""
    from aether.core.exceptions import CompilerConfigError

    with pytest.raises(CompilerConfigError, match="quality_budget"):
        CompilerConfig(quality_budget=budget)


@pytest.mark.parametrize(
    "options",
    [
        {"enable_ttt": True},
        {"enable_advanced_peft": True, "peft_adapter_paths": ["adapter"]},
        {"enable_model_merging": True, "model_merging_sources": ["merged-source"]},
    ],
    ids=["ttt", "peft", "merge"],
)
def test_base_only_reference_cannot_certify_experimental_transforms(
    tmp_path: Path,
    options: dict[str, Any],
) -> None:
    """Reference absence cannot silently certify a different expected computation."""
    from aether import Compiler
    from aether.core.exceptions import CompilationError

    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_generic"],
            skip_download=True,
            cache_dir=str(tmp_path / "cache"),
            **options,
        )
    )
    output = tmp_path / "unmeasured.aeg"
    with pytest.raises(
        CompilationError, match="independent expected transformed reference"
    ) as caught:
        compiler.compile(str(tmp_path / "missing-source"), output_path=output)
    assert caught.value.stage == "numerical_quality"
    assert not output.exists()


@pytest.mark.parametrize(
    "claim",
    ["uncertified", "benchmark_passed", "status_only", "undeclared_report"],
)
def test_default_admission_refuses_missing_measured_hf_gate(
    minimal_aeg_package: AEGPackage,
    claim: str,
) -> None:
    """Deliberately unbacked test claims cannot authorize normal artifact execution."""
    package = minimal_aeg_package
    package.metadata["evaluation_status"] = "uncertified"
    if claim == "benchmark_passed":
        package.metadata["evaluation_status"] = "certified"
        package.metadata["eval_report"] = {"gate": {"passed": True}}
    elif claim == "status_only":
        package.metadata["numerical_quality_status"] = "passed"
    package.save()
    package.verify_integrity()
    assert package.manifest is not None
    if claim == "undeclared_report":
        path = package.root / "quality" / "numerical_gate.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"passed": True}), encoding="utf-8")
        assert "quality/numerical_gate.json" not in package.manifest.artifacts
    with pytest.raises(ArtifactEvaluationError, match=r"numerical.*quality|quality.*gate"):
        validate_aeg_evaluation(package)


def test_only_explicit_diagnostics_can_waive_absent_evidence(
    minimal_aeg_package: AEGPackage,
) -> None:
    """No success is asserted for a development artifact with no numerical run."""
    assert read_gate(minimal_aeg_package, allow_uncertified=True) is None
    validate_aeg_evaluation(minimal_aeg_package, allow_uncertified=True)
    with pytest.raises(ArtifactEvaluationError, match="numerical quality"):
        validate_aeg_evaluation(minimal_aeg_package)


@pytest.mark.parametrize(
    "payload",
    [
        '{"passed": true}',
        '{"schema": "aether-numerical-quality/1", "status": "rejected"}',
        '{"status": "passed", "status": "passed"}',
        '{"metric": NaN}',
        "[]",
        '"' + "x" * (1024 * 1024) + '"',
    ],
    ids=["success-label", "rejected", "duplicate-keys", "nan", "array", "oversized"],
)
def test_diagnostic_execution_never_waives_malformed_or_rejected_report(
    minimal_aeg_package: AEGPackage,
    payload: str,
) -> None:
    """Diagnostic permission waives absence alone, regardless of invalid payload kind."""
    package = minimal_aeg_package
    path = package.root / GATE_PATH
    path.parent.mkdir()
    path.write_text(payload, encoding="utf-8")
    package.save()
    package.verify_integrity()
    with pytest.raises(ArtifactEvaluationError, match="numerical quality"):
        validate_aeg_evaluation(package, allow_uncertified=True)


def test_declared_numerical_evidence_cannot_disappear_into_diagnostic_mode(
    minimal_aeg_package: AEGPackage,
) -> None:
    """Deleting declared evidence cannot turn a rejection into an absent optional file."""
    package = minimal_aeg_package
    path = package.root / GATE_PATH
    path.parent.mkdir()
    path.write_text('{"status": "rejected"}', encoding="utf-8")
    package.save()
    path.unlink()
    with pytest.raises(
        ArtifactEvaluationError, match="declared numerical quality report is missing"
    ):
        validate_aeg_evaluation(package, allow_uncertified=True)


def test_numerical_report_checksum_is_required_even_for_diagnostics(
    minimal_aeg_package: AEGPackage,
) -> None:
    """A declared report cannot be edited into acceptance without invalidating its hash."""
    package = minimal_aeg_package
    path = package.root / GATE_PATH
    path.parent.mkdir()
    path.write_text('{"status": "rejected"}', encoding="utf-8")
    package.save()
    path.write_text('{"status": "passed"}', encoding="utf-8")
    with pytest.raises(ArtifactEvaluationError, match="checksum"):
        validate_aeg_evaluation(package, allow_uncertified=True)


@pytest.mark.parametrize("value", [True, 1, "yes", None])
def test_diagnostic_admission_flag_requires_a_real_boolean(
    minimal_aeg_package: AEGPackage,
    value: Any,
) -> None:
    """Truthiness is insufficient to authorize diagnostic artifact execution."""
    if value is True:
        assert read_gate(minimal_aeg_package, allow_uncertified=True) is None
    else:
        with pytest.raises(NumericalQualityError, match="boolean"):
            # Exercise runtime validation without hiding the invalid input behind
            # a suppressed type diagnostic in this test.
            read_gate(minimal_aeg_package, **{"allow_uncertified": value})


@pytest.mark.parametrize("existing", [False, True])
def test_ram_refusal_precedes_ingestion_and_cannot_publish(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    """A resource-blocked original reference is a failed compile, never a certificate."""
    import psutil

    from aether import Compiler
    from aether.core.exceptions import CompilationError

    output = tmp_path / "candidate.aeg"
    if existing:
        output.mkdir()
        (output / "retained").write_bytes(b"previous publication is untouched")
    before = {path.name: path.read_bytes() for path in output.glob("*")}
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_generic"],
            cache_dir=str(tmp_path / "cache"),
            skip_download=True,
        )
    )

    def unexpected_ingestion(*args: object) -> None:
        raise AssertionError("ingestion ran before reference admission")

    monkeypatch.setattr(compiler, "_stage1_ingest", unexpected_ingestion)
    with pytest.raises(CompilationError, match="RAM admission refused") as caught:
        compiler.compile(str(tiny_local_safetensors_model), output_path=output)
    assert caught.value.stage == "numerical_quality"
    assert output.exists() is existing
    assert {path.name: path.read_bytes() for path in output.glob("*")} == before
    assert not list(tmp_path.glob(".aether-compile-*"))


def test_source_binding_detects_changes_to_checkpoint_and_tokenizer(
    tiny_local_safetensors_model: Path,
) -> None:
    """A tokenizer-only source change must invalidate the same measured input identity."""
    from aether.compiler.numerical_admission import source_binding

    source = tiny_local_safetensors_model
    before = source_binding(source)
    tokenizer = source / "tokenizer_config.json"
    tokenizer.write_text(tokenizer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = source_binding(source)
    assert before["digest"] != after["digest"]
    assert before["files"]["model.safetensors"] == after["files"]["model.safetensors"]
    assert before["files"]["tokenizer_config.json"] != after["files"]["tokenizer_config.json"]


def test_timeout_reaps_owned_worker_before_log_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled real child must release its inherited log before temp cleanup."""
    import subprocess
    import sys

    import psutil

    from aether.compiler.numerical_admission import run_worker

    helper = tmp_path / "waiting_worker.py"
    helper.write_text(
        "import json, os, pathlib, sys, time, psutil\n"
        "work = pathlib.Path(sys.argv[1])\n"
        "identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time()}\n"
        "(work / 'identity.json').write_text(json.dumps(identity), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    original_popen = subprocess.Popen
    started: list[subprocess.Popen[bytes]] = []

    def waiting_worker(arguments: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original_popen([sys.executable, str(helper), arguments[-2]], **kwargs)
        started.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", waiting_worker)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    config = CompilerConfig(quality_worker_timeout_s=0.5)
    with pytest.raises(NumericalQualityError, match="timed out"):
        run_worker(tmp_path, "hf", {}, config, reserve=0)
    assert len(started) == 1 and started[0].poll() is not None
    marker = tmp_path / "identity.json"
    if marker.exists():
        identity = json.loads(marker.read_text(encoding="utf-8"))
        try:
            worker = psutil.Process(identity["pid"])
        except psutil.NoSuchProcess:
            pass
        else:
            assert worker.create_time() != identity["create_time"] or not worker.is_running()
    # On Windows an unreaped launcher descendant can keep this inherited handle
    # open even after Popen.wait() has returned for the launcher itself.
    (tmp_path / "hf.log").unlink()


@pytest.mark.parametrize(
    "entry",
    [
        "loader",
        "native_backend",
        "torch_backend",
        "onnx_backend",
        "onnx_export",
        "runtime",
        "cli",
    ],
)
def test_diagnostic_compilation_requires_explicit_execution_permission(
    tmp_path: Path,
    tiny_local_safetensors_model: Path,
    entry: str,
) -> None:
    """Real development bytes can execute only through the explicit diagnostic API."""
    import numpy as np

    from aether import Compiler, Runtime, RuntimeConfig
    from aether.backends.native_cpu_backend import NativeCPUBackend
    from aether.runtime.aeg_loader import load_engine_from_path

    package = Compiler(
        CompilerConfig(
            quality_mode="diagnostic",
            targets=["cpu_generic"],
            skip_download=True,
            cache_dir=str(tmp_path / "cache"),
            precision="FP32",
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "diagnostic.aeg")
    assert package.metadata["numerical_quality_mode"] == "diagnostic"
    assert package.manifest is not None
    assert GATE_PATH not in package.manifest.artifacts
    package.verify_integrity()
    if entry == "loader":
        with pytest.raises(ArtifactEvaluationError, match="numerical quality"):
            load_engine_from_path(package.root, prepare_native_kernels=False)
        engine = load_engine_from_path(
            package.root,
            prepare_native_kernels=False,
            allow_uncertified=True,
        )
        logits, _ = engine.forward_step([1, 2])
        assert logits.shape == (32,) and np.isfinite(logits).all()
    elif entry in {"native_backend", "torch_backend", "onnx_backend"}:
        backend: Backend
        if entry == "torch_backend":
            pytest.importorskip("torch")
            from aether.backends.torch_backend import TorchBackend

            backend = TorchBackend()
        elif entry == "onnx_backend":
            pytest.importorskip("onnxruntime")
            pytest.importorskip("onnx")
            from aether.backends.onnx_backend import ONNXBackend

            backend = ONNXBackend()
        else:
            backend = NativeCPUBackend()
        with pytest.raises(ArtifactEvaluationError, match="numerical quality"):
            backend.load_model("development", str(package.root))
        handle = backend.load_model("development", str(package.root), allow_uncertified=True)
        assert not handle.numerical_quality_admitted
        with pytest.raises(ArtifactEvaluationError, match="Cached AEG"):
            backend.load_model("development")
        backend.unload_model("development")
    elif entry == "onnx_export":
        pytest.importorskip("onnx")
        from aether.runtime.onnx_export import export_aeg_to_onnx

        destination = tmp_path / "export"
        with pytest.raises(ArtifactEvaluationError, match="numerical quality"):
            export_aeg_to_onnx(package, destination)
        assert not destination.exists()
        bundle = export_aeg_to_onnx(package, destination, allow_uncertified=True)
        assert bundle.prefill_path.is_file() and bundle.decode_path.is_file()
    else:
        config = RuntimeConfig(
            backend_name="aether_cpu",
            hf_offline=True,
            model_cache_dir=str(tmp_path / "runtime-cache"),
            enable_semantic_cache=False,
            enable_diffusion_spec=False,
            speculative_decoding=False,
            dynamic_precision=False,
        )
        if entry == "cli":
            from click.testing import CliRunner

            from aether.cli import cli

            path = tmp_path / "runtime.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            arguments = [
                "--runtime-config",
                str(path),
                "run",
                str(package.root),
                "--prompt",
                "hello world",
                "--max-tokens",
                "2",
                "--temperature",
                "0",
                "--raw",
                "--non-interactive",
            ]
            refused = CliRunner().invoke(cli, arguments)
            assert refused.exit_code == 1 and "numerical quality" in refused.output
            executed = CliRunner().invoke(cli, ["--diagnostic", *arguments])
            assert executed.exit_code == 0, executed.output
            assert "Response:" in executed.output and "completion_tokens" in executed.output
            return
        with (
            Runtime(config) as runtime,
            pytest.raises(ArtifactEvaluationError, match="numerical quality"),
        ):
            runtime.generate(str(package.root), "hello world", max_tokens=2, temperature=0)
        diagnostic = RuntimeConfig.from_dict({**config.to_dict(), "allow_uncertified": True})
        with Runtime(diagnostic) as runtime:
            response = runtime.generate(
                str(package.root),
                "hello world",
                max_tokens=2,
                temperature=0,
            )
            assert response.usage["completion_tokens"] == 2
            assert response.metrics.backend_name == "aether_cpu"
