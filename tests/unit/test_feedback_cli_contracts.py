"""Public CLI and Python examples must reach real execution paths."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from aether import CompilerConfig
from aether.cli import cli
from aether.core.exceptions import BackendNotAvailableError


@pytest.mark.parametrize("precision", ["q4_k_m", "Q8_0", "bf16", "FP16", "fp32"])
def test_compile_forwards_explicit_precision(monkeypatch, precision):
    captured = {}

    class Compiler:
        def __init__(self, config):
            captured["config"] = config

        def compile(self, model, output_path=None):
            return SimpleNamespace(root="model.aeg", metadata={})

    monkeypatch.setattr("aether.cli.Compiler", Compiler)
    result = CliRunner().invoke(cli, ["compile", "local-model", "--precision", precision])
    assert result.exit_code == 0, result.output
    assert captured["config"].precision == precision.upper()
    assert captured["config"].precision_assignment_mode == "uniform"


def test_unknown_precision_fails_before_compilation():
    result = CliRunner().invoke(cli, ["compile", "local-model", "--precision", "made_up"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_explicit_precision_roundtrips_compiler_configuration():
    config = CompilerConfig(precision="q4_k_m")
    restored = CompilerConfig.from_dict(config.to_dict())
    assert restored.precision == "Q4_K_M"
    assert restored.precision_assignment_mode == "uniform"


def test_documented_backend_lookup_is_available():
    from aether.backends import NativeCPUBackend, get_backend

    assert isinstance(get_backend("aether_cpu"), NativeCPUBackend)
    with pytest.raises(BackendNotAvailableError, match="not installed or available"):
        get_backend("unknown-backend")


def _fake_runtime(monkeypatch):
    captured = {}

    class Runtime:
        def __init__(self, config):
            captured["config"] = config
            self._loaded_backends = {"model": SimpleNamespace(name="aether_cpu")}

        def close(self):
            captured["closed"] = captured.get("closed", 0) + 1

        def _load_model(self, model):
            captured["loaded"] = model

        def benchmark(self, model, max_tokens, prompt=None):
            return {
                "backend": "aether_cpu",
                "kernel_target": "cpu_avx2",
                "throughput_tps": 20.0,
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "duration_s": 0.1,
                "ttft_ms": 3.3,
            }

        def generate(self, **kwargs):
            raise AssertionError("streaming command called the blocking API")

        def generate_stream(self, **kwargs):
            captured["request"] = kwargs
            yield "first "
            yield "second"

    monkeypatch.setattr("aether.cli.Runtime", Runtime)
    return captured


def test_run_stream_reaches_incremental_generation(monkeypatch):
    captured = _fake_runtime(monkeypatch)
    result = CliRunner().invoke(cli, ["run", "model", "--stream", "--prompt", "hello"])
    assert result.exit_code == 0, result.output
    assert "first second" in result.output
    assert captured["request"]["prompt"] == "hello"
    assert captured["request"]["apply_chat_template"] is True


def test_cli_preserves_environment_backend_devices_and_offline_controls(monkeypatch):
    captured = _fake_runtime(monkeypatch)
    monkeypatch.setenv("AETHER_BACKEND", "aether_cpu")
    monkeypatch.setenv("AETHER_EXECUTION_DEVICES", "cpu")
    monkeypatch.setenv("AETHER_HF_OFFLINE", "1")
    result = CliRunner().invoke(cli, ["bench", "model", "--json"])
    assert result.exit_code == 0, result.output
    assert captured["config"].backend_name == "aether_cpu"
    assert captured["config"].execution_devices == ["cpu"]
    assert captured["config"].hf_offline
    assert json.loads(result.output)["kernel_target"] == "cpu_avx2"


def test_repeated_benchmark_uses_backend_token_counts_and_ttft(monkeypatch):
    _fake_runtime(monkeypatch)
    result = CliRunner().invoke(
        cli,
        [
            "benchmark",
            "model",
            "--runs",
            "1",
            "--warmup",
            "0",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    measured = report["runs"][0]
    assert report["provenance"]["hardware_target"] == "cpu_avx2"
    assert report["provenance"]["backend_name"] == "aether_cpu"
    assert measured["prompt_tokens"] == 7
    assert measured["output_tokens"] == 2
    assert measured["ttft_s"] == pytest.approx(0.0033)
    assert measured["token_counts_estimated"] is False
    assert measured["ttft_estimated"] is False
    assert measured["tbt_mean_s"] is None


def test_bench_saves_parseable_json_and_compares_a_saved_baseline(monkeypatch, tmp_path):
    _fake_runtime(monkeypatch)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"throughput_tps": 10.0}), encoding="utf-8")
    output = tmp_path / "measured.json"
    result = CliRunner().invoke(
        cli,
        [
            "bench",
            "model",
            "--compare",
            str(baseline),
            "--output",
            str(output),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.output)["comparison"]["throughput_speedup"] == 2.0


def test_bench_comparison_executes_the_requested_backend(monkeypatch):
    captured = _fake_runtime(monkeypatch)
    result = CliRunner().invoke(cli, ["bench", "model", "--compare", "pytorch", "--json"])
    assert result.exit_code == 0, result.output
    assert captured["config"].backend_name == "pytorch"
    assert json.loads(result.output)["comparison"]["throughput_speedup"] == 1.0


def test_missing_artifact_is_a_cli_error_without_a_traceback(tmp_path):
    result = CliRunner().invoke(cli, ["bench", str(tmp_path / "missing.aeg")])
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["info", "graph"])
def test_missing_artifact_inspection_exits_with_an_error(tmp_path, command):
    result = CliRunner().invoke(cli, [command, str(tmp_path / "missing.aeg")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


@pytest.mark.parametrize("precision", ["Q4_K_M", "Q8_0", "FP32"])
def test_cli_precision_reaches_serialized_weights(
    tiny_local_safetensors_model, tmp_path, precision
):
    from aether.core.aeg_format import load_aeg_package

    output = tmp_path / "model.aeg"
    result = CliRunner().invoke(
        cli,
        [
            "--cache-dir",
            str(tmp_path / "cache"),
            "compile",
            str(tiny_local_safetensors_model),
            "--target",
            "cpu_avx2",
            "--precision",
            precision,
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    package = load_aeg_package(output)
    assert package.get_precision_map()["layer_0"] == precision
    entries = package.weight_store().load_index()
    assert entries["layer_0_q_proj"].precision == precision
    package.verify_integrity()


def test_real_cpu_artifact_bench_run_stream_and_metrics_on_a_gpu_host(
    tiny_local_safetensors_model,
    tmp_path,
    monkeypatch,
):
    from dataclasses import replace

    from aether import Compiler, Runtime
    from aether.runtime.hardware import HardwareDetector

    artifact = Compiler(
        CompilerConfig(
            targets=["cpu_avx2"],
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "model.aeg")
    fingerprint = HardwareDetector().detect()
    cpu_target = next(
        target
        for target in fingerprint.attributes["available_targets"]
        if target.startswith("cpu_")
    )
    fingerprint = replace(
        fingerprint,
        target_id="cuda_sm80",
        attributes={
            "available_targets": ["cuda_sm80", cpu_target],
        },
    )
    monkeypatch.setattr(HardwareDetector, "detect", lambda _: fingerprint)
    runtime = Runtime()
    response = runtime.generate(str(artifact.root), "hello", max_tokens=2, temperature=0.0)
    assert response.usage["completion_tokens"] == 2
    assert response.metrics.kernel_target == cpu_target
    assert response.metrics.backend_name == "aether_cpu"
    runner = CliRunner()
    bench = runner.invoke(cli, ["bench", str(artifact.root), "--max-tokens", "2", "--json"])
    assert bench.exit_code == 0, bench.output
    assert json.loads(bench.output)["backend"] == "aether_cpu"
    assert json.loads(bench.output)["kernel_target"] == cpu_target
    stream = runner.invoke(
        cli,
        [
            "run",
            str(artifact.root),
            "--prompt",
            "hello",
            "--max-tokens",
            "2",
            "--stream",
        ],
    )
    assert stream.exit_code == 0, stream.output
    assert "Response:" in stream.output


def test_real_cpu_artifact_serves_openai_http(tiny_local_safetensors_model, tmp_path):
    """The installed test extra can serve a real artifact without Transformers."""
    import os
    import socket
    import subprocess
    import sys
    import time

    import httpx

    from aether import Compiler

    artifact = Compiler(
        CompilerConfig(targets=["cpu_avx2"], cache_dir=str(tmp_path / "cache"))
    ).compile(str(tiny_local_safetensors_model), output_path=tmp_path / "served.aeg")
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
    env = os.environ.copy()
    env["AETHER_HF_OFFLINE"] = "1"
    env["AETHER_BACKEND"] = "aether_cpu"
    env["AETHER_EXECUTION_DEVICES"] = "cpu"
    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aether",
                "--cache-dir",
                str(tmp_path / "cache"),
                "serve",
                str(artifact.root),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base_url, timeout=5.0, trust_env=False) as client:
                deadline = time.monotonic() + 60.0
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    pytest.fail(log_path.read_text(encoding="utf-8"))
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": str(artifact.root),
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2,
                        "temperature": 0.0,
                    },
                )
                assert response.status_code == 200, response.text
                assert response.json()["usage"]["completion_tokens"] == 2
                assert isinstance(response.json()["choices"][0]["message"]["content"], str)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
