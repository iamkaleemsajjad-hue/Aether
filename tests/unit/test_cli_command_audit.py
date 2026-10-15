"""Regression contracts from the full installed command-tree audit."""
from __future__ import annotations

import json
import shutil
import threading
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from aether.cli import cli


def _lifecycle_runtime(monkeypatch, *, failure=None):
    """Observe real command ownership without requiring an external checkpoint."""
    instances, events = [], []
    class OwnedRuntime:
        def __init__(self, config):
            self.name = config.backend_name or "primary"
            self.closed = False
            self.streaming = False
            self._loaded_backends = {"model": SimpleNamespace(name=self.name)}
            self._loaded_models = {}
            self.tracer = SimpleNamespace(export_otlp_json=lambda: {}, get_finished_spans=lambda: [])
            events.append(f"open:{self.name}")
            instances.append(self)
        def _load_model(self, model):
            events.append("load")
            if failure == "load":
                raise ValueError("model loading failed")
        def generate(self, *args, **kwargs):
            assert not self.closed
            if failure == "generate":
                raise ValueError("generation failed")
            return SimpleNamespace(text="answer", usage={"completion_tokens": 1},
                                   metrics=SimpleNamespace(to_dict=lambda: {"measured": True}))
        def generate_stream(self, **kwargs):
            self.streaming = True
            try:
                yield "first-token"
                if failure == "stream":
                    raise ValueError("stream failed")
                yield "last-token"
            finally:
                self.streaming = False
                events.append("stream_closed")
        def benchmark(self, model, max_tokens, prompt=None):
            assert not self.closed
            if failure == "benchmark" or (failure == "comparison" and self.name == "pytorch"):
                raise ValueError("benchmark execution failed")
            return {"backend": self.name, "kernel_target": "cpu_generic", "throughput_tps": 10.,
                    "prompt_tokens": 2, "completion_tokens": 1, "duration_s": .1, "ttft_ms": 1.}
        def _auxiliary(self, result):
            assert not self.closed
            if failure == "auxiliary":
                raise ValueError("auxiliary operation failed")
            return result
        def pull(self, model):
            return self._auxiliary(None)
        def info(self, model):
            return self._auxiliary({"model_id": model, "targets": []})
        def list(self):
            return self._auxiliary([])
        def remove(self, model):
            return self._auxiliary(None)
        def _resolve_aeg_path(self, model):
            return self._auxiliary(None)
        def eval_gate(self, model, **kwargs):
            return self._auxiliary({"passed": True})
        def set_model_task_weights(self, model_id, weights):
            return self._auxiliary(dict(weights))
        def grpo_train_step(self, *args, **kwargs):
            return self._auxiliary({"status": "complete"})
        def quantization_report(self, model):
            return self._auxiliary({})
        def semantic_cache_stats(self):
            return self._auxiliary({"enabled": False})
        def cxl_pool_status(self):
            return self._auxiliary({"mode": "memory"})
        def kv_transfer_stats(self):
            return self._auxiliary({"payload_transfers": 0})
        def close(self):
            assert not self.streaming, "stream must release its read gate before runtime shutdown"
            assert not self.closed, "command must close its owned runtime exactly once"
            self.closed = True
            events.append(f"close:{self.name}")
    monkeypatch.setattr("aether.cli.Runtime", OwnedRuntime)
    return instances, events


def test_reweight_passes_arbitrary_task_labels_as_mapping(monkeypatch):
    instances, events = _lifecycle_runtime(monkeypatch)
    result = CliRunner().invoke(cli, [
        "runtime", "reweight", "model", "--task", "weights=2", "--task", "self=1",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["weights"] == {"weights": 2., "self": 1.}
    assert instances[0].closed


@pytest.mark.parametrize("stream,failure", [(False, None), (True, None), (False, "load"),
                                             (False, "generate"), (True, "stream")])
def test_run_closes_owned_runtime_on_completion_and_failure(monkeypatch, stream, failure):
    instances, events = _lifecycle_runtime(monkeypatch, failure=failure)
    args = ["run", "model", "--non-interactive"] + (["--stream"] if stream else [])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == (1 if failure else 0), result.output
    assert len(instances) == 1 and instances[0].closed
    if stream:
        assert events.index("stream_closed") < events.index("close:primary")


def test_interrupted_stream_output_closes_iterator_before_runtime(monkeypatch):
    instances, events = _lifecycle_runtime(monkeypatch)
    original = click.echo
    def cancelled_output(message=None, *args, **kwargs):
        if message == "first-token":
            raise KeyboardInterrupt()
        return original(message, *args, **kwargs)
    monkeypatch.setattr(click, "echo", cancelled_output)
    result = CliRunner().invoke(cli, ["run", "model", "--stream", "--non-interactive"])
    assert result.exit_code != 0
    assert instances[0].closed and events[-2:] == ["stream_closed", "close:primary"]


@pytest.mark.parametrize("failure", [None, "comparison"])
def test_bench_releases_primary_before_comparison_and_closes_both(monkeypatch, failure):
    instances, events = _lifecycle_runtime(monkeypatch, failure=failure)
    result = CliRunner().invoke(cli, ["bench", "model", "--compare", "pytorch", "--json"])
    assert result.exit_code == (1 if failure else 0), result.output
    assert len(instances) == 2 and all(instance.closed for instance in instances)
    assert events.index("close:primary") < events.index("open:pytorch")


@pytest.mark.parametrize("command", ["bench", "benchmark"])
def test_benchmark_failure_closes_runtime_and_remains_a_failed_command(monkeypatch, command):
    instances, _ = _lifecycle_runtime(monkeypatch, failure="benchmark")
    args = [command, "model", "--json"]
    if command == "benchmark":
        args += ["--runs", "1", "--warmup", "0"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert len(instances) == 1 and instances[0].closed


def test_failed_benchmark_report_write_does_not_retain_backend(monkeypatch, tmp_path):
    instances, _ = _lifecycle_runtime(monkeypatch)
    output = tmp_path / "missing-directory" / "report.json"
    result = CliRunner().invoke(cli, ["bench", "model", "--output", str(output)])
    assert result.exit_code == 1 and instances[0].closed
    assert "Error:" in result.output


@pytest.mark.parametrize("args", [
    ["pull", "model"], ["info", "model"], ["list"], ["rm", "model"], ["trace"],
    ["trace", "model", "--prompt", "hello"], ["eval", "model"],
    ["runtime", "reweight", "model", "--task", "task=1"],
    ["train", "grpo", "model", "hello"], ["quantize-report", "model"],
    ["cache", "stats"], ["kv", "transfer-stats"], ["kv", "cxl-pool-status"],
])
@pytest.mark.parametrize("failure", [None, "auxiliary"])
def test_remaining_runtime_owners_close_on_success_error_and_early_return(monkeypatch, args, failure):
    instances, _ = _lifecycle_runtime(monkeypatch, failure=failure)
    result = CliRunner().invoke(cli, args, input="y\n" if args[0] == "rm" else None)
    # Trace's no-model path reads only its recorder, not an inference method.
    unaffected_trace = args[0] == "trace"
    assert result.exit_code == (1 if failure and not unaffected_trace else 0), result.output
    assert len(instances) == 1 and instances[0].closed


def test_coordinator_and_runtime_close_if_multi_agent_output_is_interrupted(monkeypatch):
    instances, events = _lifecycle_runtime(monkeypatch)
    class Session(dict):
        def close(self):
            events.append("coordinator_close")
    def coordinate(self, **kwargs):
        return Session({"agents": 2})
    # Construction occurs inside the command, so attach the actual public
    # method to the instrumented Runtime class rather than a prebuilt instance.
    from importlib import import_module
    monkeypatch.setattr(import_module("aether.cli").Runtime, "multi_agent_session", coordinate, raising=False)
    original = click.echo
    interrupted = False
    def interrupt(message=None, *args, **kwargs):
        nonlocal interrupted
        if isinstance(message, str) and '"agents"' in message and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(message, *args, **kwargs)
    monkeypatch.setattr(click, "echo", interrupt)
    result = CliRunner().invoke(cli, ["multi-agent", "--agents", "2"])
    assert result.exit_code == 1
    assert instances[0].closed and events[-2:] == ["coordinator_close", "close:primary"]


@pytest.mark.parametrize("failure", [None, "load", "grpc", "http", "stop"])
def test_serve_keeps_runtime_alive_then_closes_on_all_startup_and_shutdown_paths(monkeypatch, tmp_path, failure):
    pytest.importorskip("uvicorn")
    instances, events = _lifecycle_runtime(monkeypatch, failure="load" if failure == "load" else None)
    # create_app is the owner factory; the CLI must close the same instance
    # used by both transports, rather than allocating another runtime.
    def app(config):
        from importlib import import_module
        runtime = import_module("aether.cli").Runtime(config)
        return SimpleNamespace(state=SimpleNamespace(aether_runtime=runtime))
    monkeypatch.setattr("aether.server.app.create_app", app)
    class Stopped:
        def wait(self):
            events.append("grpc_wait")
            assert not instances[0].closed
            return True
    class Grpc:
        aether_port = 12345
        def stop(self, grace):
            assert grace == 0 and not instances[0].closed
            events.append("grpc_stop")
            if failure == "stop":
                raise OSError("gRPC stop failed")
            return Stopped()
    def start(runtime, **kwargs):
        assert runtime is instances[0] and not runtime.closed
        events.append("grpc_start")
        if failure == "grpc":
            raise OSError("gRPC startup failed")
        return Grpc()
    def http(application, **kwargs):
        assert application.state.aether_runtime is instances[0]
        assert not instances[0].closed
        events.append("http_run")
        if failure == "http":
            raise KeyboardInterrupt()
    monkeypatch.setattr("aether.server.grpc_service.start_grpc_server", start)
    monkeypatch.setattr("uvicorn.run", http)
    result = CliRunner().invoke(cli, ["--cache-dir", str(tmp_path), "serve", "model", "--grpc-port", "0"])
    assert result.exit_code == (0 if failure is None else 1), result.output
    assert len(instances) == 1 and instances[0].closed
    assert events[-1] == "close:primary"
    if failure not in {"load", "grpc", "stop"}:
        assert events.index("http_run") < events.index("grpc_stop") < events.index("grpc_wait") < events.index("close:primary")


@pytest.mark.parametrize("mode", ["completion", "stream", "cancel_stream"])
def test_real_cli_phase_workers_are_joined_before_command_returns(tmp_path, monkeypatch, mode):
    from dataclasses import fields
    from aether import Compiler, CompilerConfig
    from aether.backends.hardware_validation import _write_probe_checkpoint
    _write_probe_checkpoint(tmp_path / "checkpoint")
    config = CompilerConfig(targets=["cpu_generic"], precision="FP32", optimization_level=0,
                            cache_dir=str(tmp_path / "compiler-cache"))
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    artifact = tmp_path / "model.aeg"
    Compiler(config).compile(str(tmp_path / "checkpoint"), output_path=artifact)
    runtime_config = tmp_path / "runtime.json"
    runtime_config.write_text(json.dumps({"backend_name": "aether_cpu", "execution_devices": ["cpu"],
        "enable_semantic_cache": False, "speculative_decoding": False, "disaggregate_prefill_decode": True}), encoding="utf-8")
    prior = {thread.ident for thread in threading.enumerate()}
    seen = []
    cancelled = False
    original = click.echo
    def observe(message=None, *args, **kwargs):
        nonlocal cancelled
        seen.extend(thread for thread in threading.enumerate() if thread.ident not in prior
                    and thread.name.startswith(("aether-prefill-", "aether-decode-")))
        if mode == "cancel_stream" and not cancelled and message != "Response: ":
            cancelled = True
            raise KeyboardInterrupt()
        return original(message, *args, **kwargs)
    monkeypatch.setattr(click, "echo", observe)
    args = ["--runtime-config", str(runtime_config), "run", str(artifact), "--prompt", "hello world",
            "--max-tokens", "2", "--temperature", "0", "--raw", "--non-interactive"]
    if mode != "completion":
        args.append("--stream")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == (1 if mode == "cancel_stream" else 0), result.output
    if mode == "completion":
        assert "state_bytes_handed_off" in result.output
    assert not any(thread.is_alive() for thread in threading.enumerate() if thread.ident not in prior
                   and thread.name.startswith(("aether-prefill-", "aether-decode-")))
    if mode != "completion":
        assert seen, "stream output must have observed the actual live phase workers"


def _runner():
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def test_all_registered_commands_have_working_help():
    pending = [([], cli)]
    paths = []
    while pending:
        prefix, group = pending.pop(0)
        for name, command in group.commands.items():
            path = prefix + [name]
            paths.append(path)
            result = CliRunner().invoke(cli, path + ["--help"])
            assert result.exit_code == 0, (path, result.output)
            if isinstance(command, click.Group):
                pending.append((path, command))
    assert len(paths) >= 70
    assert cli.commands["hw"] is cli.commands["hardware"]
    assert {"as_json", "mtp"} <= {p.name for p in cli.commands["inspect"].params}


@pytest.mark.parametrize("args", [
    ["grammar", "missing.aeg"],
    ["merge", "missing.aeg"],
    ["merge", "a", "b", "--weights", "wrong"],
    ["merge", "a", "b", "--weights", "0.2"],
    ["merge", "a", "b", "--weights", "nan,1"],
    ["ttt-config", "a", "--layers", "wrong"],
    ["ttt-config", "a", "--adapter-rank", "0"],
    ["ttt-config", "a", "--ttl", "10"],
    ["ttt-config", "a", "--layers", "0,1"],
    ["kv-compress", "a", "--retention", "0"],
    ["kv-compress", "a", "--cross-layer-groups", "2"],
    ["mcp", "list", "missing.aeg"],
    ["run", "a", "--max-tokens", "0"],
    ["run", "a", "--temperature", "nan"],
    ["run", "a", "--top-p", "1.1"],
    ["serve", "--port", "65536"],
    ["logs", "--lines", "0"],
    ["green-profile", "a", "--renewable-threshold", "2"],
    ["green-route", "--regions", "us-east-1", "--latency-deadline", "nan"],
    ["slo-profile", "add", "x", "--ttft", "nan", "--tbt", "1"],
    ["trace", "--prompt", "hello"],
    ["trace", "missing.aeg"],
])
def test_invalid_arguments_or_unsupported_controls_fail_without_tracebacks(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0, result.output
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_inspect_accepts_path_cache_id_mtp_and_unwrapped_json(minimal_aeg_package, tmp_path, monkeypatch):
    monkeypatch.setattr("aether.cli.console.width", 24)
    source = minimal_aeg_package.root
    mtp_dir = source / "speculation"
    mtp_dir.mkdir(exist_ok=True)
    payload = {"description": "a long persisted MTP configuration " * 8, "heads": 2}
    (mtp_dir / "mtp_config.json").write_text(json.dumps(payload), encoding="utf-8")
    cache = tmp_path / "cache"
    destination = cache / "models" / "org--fixture.aeg"
    shutil.copytree(source, destination)
    for model in (str(source), "org/fixture", "org--fixture.aeg"):
        result = _runner().invoke(cli, ["--cache-dir", str(cache), "inspect", model, "--mtp", "--json"])
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        assert report["mtp"] == payload
        assert report["manifest"]["model_id"] == "test-model"


@pytest.mark.parametrize("args", [["trace"], ["models", "--json"], ["cache", "stats"], ["slo-status"], ["kv", "nika-policy", "--kv-size-gb", "1", "--bw-gbps", "100", "--decode-util", "0.5"]])
def test_json_output_survives_a_narrow_terminal(args, monkeypatch):
    monkeypatch.setattr("aether.cli.console.width", 24)
    result = _runner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert isinstance(json.loads(result.stdout), dict)


def test_grammar_prd_compile_consumes_schema_argument_and_checks_real_pass(tmp_path, monkeypatch):
    schema = tmp_path / "schema.gbnf"
    schema.write_text('root ::= "hello"', encoding="utf-8")
    result = CliRunner().invoke(cli, ["grammar", "compile", str(schema), "--target", "local.aeg", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Dry-run" in result.output


def test_merge_pass_gets_requested_models_method_and_coefficients(monkeypatch):
    captured = {}
    class Compiler:
        def __init__(self, config):
            captured["config"] = config
        def compile(self, *args, **kwargs):
            return SimpleNamespace(root="result.aeg", metadata={"optimizer_passes": ["model_merging"]})
    monkeypatch.setattr("aether.cli.Compiler", Compiler)
    result = CliRunner().invoke(cli, ["merge", "base", "task", "--method", "linear", "--weights", "0.3,0.7"])
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.model_merging_sources == ["base", "task"]
    assert config.model_merging_method == "task_arithmetic"
    assert config.model_merging_coefficients == [0.3, 0.7]


def test_force_local_pull_uses_compiler_not_cached_runtime(monkeypatch, tmp_path):
    captured = {}
    class Compiler:
        def __init__(self, config):
            captured["config"] = config
        def compile(self, model):
            captured["model"] = model
    monkeypatch.setattr("aether.cli.Compiler", Compiler)
    monkeypatch.setattr("aether.cli.Runtime", lambda *_: pytest.fail("forced compilation used Runtime.pull"))
    result = CliRunner().invoke(cli, ["--cache-dir", str(tmp_path), "pull", "local-source", "--compile-local"])
    assert result.exit_code == 0, result.output
    assert captured["model"] == "local-source"
    assert captured["config"].overwrite
    assert str(captured["config"].cache_dir) == str(tmp_path)


def test_grammar_mask_counts_each_bit(monkeypatch):
    class Runtime:
        def __init__(self, config):
            self.grammar_engine = SimpleNamespace(is_loaded=lambda: True, create_session=lambda: SimpleNamespace(get_token_mask=lambda: bytearray([255, 255])))
        def close(self):
            pass
        def _resolve_aeg_path(self, model):
            return "local.aeg"
        def _init_v4_layers(self, path):
            pass
    monkeypatch.setattr("aether.cli.Runtime", Runtime)
    result = _runner().invoke(cli, ["grammar", "test", "local.aeg"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["initial_allowed_token_count"] == 16


def test_mcp_inspects_real_list_tools_api_and_propagates_tool_errors(monkeypatch):
    class Runtime:
        def __init__(self, config):
            self.mcp_layer = SimpleNamespace(list_tools=lambda: [{"name": "echo", "description": "a" * 150}], call_tool=lambda *_: {"isError": True}, disconnect_all=lambda: None)
        def close(self):
            pass
        def _resolve_aeg_path(self, model):
            return "local.aeg"
        def _init_v4_layers(self, path):
            pass
    monkeypatch.setattr("aether.cli.Runtime", Runtime)
    result = _runner().invoke(cli, ["mcp", "local.aeg", "--tool", "echo"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["name"] == "echo"
    failed = _runner().invoke(cli, ["mcp", "local.aeg", "--tool", "echo", "--call", "{}"])
    assert failed.exit_code == 1
    assert json.loads(failed.stdout)["isError"]


def test_multi_agent_coordinates_requested_mode_and_explicit_scope(minimal_aeg_package):
    result = _runner().invoke(cli, ["multi-agent", "test", str(minimal_aeg_package.root), "--agents", "2", "--coordination", "swarm"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["coordination"] == "swarm"
    assert payload["models"] == [str(minimal_aeg_package.root)]
    assert payload["agent_count"] == 2
    assert payload["test"] == "local_kv_coordinator_initialization"
    assert not payload["model_execution_tested"]


def test_tee_fields_do_not_establish_attestation_trust(tmp_path):
    report = tmp_path / "attestation.json"
    report.write_text(json.dumps({"backend": "nvidia_cc", "token": "fabricated", "enclave_initialized": True, "hardware_backed": True, "generated_at": 1}), encoding="utf-8")
    result = _runner().invoke(cli, ["tee", "verify", "missing.aeg", "--report", str(report)])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["structural_valid"]
    assert not payload["verified"]
    assert payload["trust_level"] == "unverified"


def test_train_verify_requires_evidence_and_reports_actual_result():
    runner = _runner()
    no_truth = runner.invoke(cli, ["train", "verify", "--example", "2+2=4"])
    assert no_truth.exit_code == 2
    wrong = runner.invoke(cli, ["train", "verify", "--example", "2+2=5", "--ground-truth", "4"])
    assert wrong.exit_code == 1
    assert not json.loads(wrong.stdout)["verified"]
    correct = runner.invoke(cli, ["train", "verify", "--example", "2+2=4", "--ground-truth", "4"])
    assert correct.exit_code == 0
    assert json.loads(correct.stdout)["verified"]


def test_merge_info_reads_package_metadata(minimal_aeg_package):
    minimal_aeg_package.metadata["model_merging"] = {"method": "linear", "sources": ["base", "task"]}
    minimal_aeg_package.save()
    result = _runner().invoke(cli, ["merge-info", str(minimal_aeg_package.root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["merge"]["method"] == "linear"


def test_real_offline_cpu_command_workflow(tiny_local_safetensors_model, tmp_path, monkeypatch):
    """Exercise the public commands against real serialized weights and signing."""
    from aether.utils.file_io import aeg_cache_path
    monkeypatch.setenv("AETHER_BACKEND", "aether_cpu")
    monkeypatch.setenv("AETHER_EXECUTION_DEVICES", "cpu")
    monkeypatch.setenv("AETHER_HF_OFFLINE", "1")
    cache = tmp_path / "cache"
    prefix = ["--cache-dir", str(cache)]
    runner = _runner()
    compiled = runner.invoke(cli, prefix + ["compile", str(tiny_local_safetensors_model), "--target", "cpu_avx2", "--precision", "fp32"])
    assert compiled.exit_code == 0, compiled.output
    artifact = aeg_cache_path(str(tiny_local_safetensors_model), cache)
    assert (artifact / "manifest.json").is_file()
    from aether.core.aeg_format import load_aeg_package
    from aether.kernels.native_cpu import get_native_kernels
    package = load_aeg_package(artifact)
    if get_native_kernels().ensure_compiled():
        artifacts = package.metadata.get("kernel_artifacts", [])
        assert artifacts, "a working compiler must embed its native library even for a long source path"
        assert all((artifact / item["path"]).is_file() for item in artifacts)
    model = artifact.name
    for args in (
        ["info", model], ["inspect", model, "--json"], ["mla-stats", model],
        ["safety", model], ["status", model],
        ["bench", model, "--max-tokens", "2", "--json"],
        ["benchmark", model, "--max-tokens", "2", "--runs", "1", "--warmup", "0", "--json"],
        ["trace", model, "--prompt", "hello"],
    ):
        result = runner.invoke(cli, prefix + args)
        assert result.exit_code == 0, (args, result.output)
        payload = json.loads(result.stdout)
        assert isinstance(payload, dict)
        if args[0] == "safety":
            assert payload["integrity_verified"]
        if args[0] == "status":
            assert payload["artifact_available"] and not payload["execution_tested"]
        if args[0] == "bench":
            assert payload["backend"] == "aether_cpu"
            assert payload["completion_tokens"] == 2
    for args in (["graph", model], ["list"], ["run", model, "--prompt", "hello", "--max-tokens", "2", "--non-interactive"], ["run", model, "--prompt", "hello", "--max-tokens", "2", "--stream"]):
        result = runner.invoke(cli, prefix + args)
        assert result.exit_code == 0, (args, result.output)
    signed = runner.invoke(cli, ["sign", str(artifact), "--key", str(tmp_path / "signing.seed"), "--json"])
    assert signed.exit_code == 0, signed.output
    assert json.loads(signed.stdout)["files_bound"] > 0
    verified = runner.invoke(cli, ["verify", str(artifact), "--json"])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["integrity_valid"]
    removed = runner.invoke(cli, prefix + ["rm", model, "--yes"])
    assert removed.exit_code == 0, removed.output
    assert not artifact.exists()
