"""Offline end-to-end test for a real local SafeTensors model.

This deliberately uses a tiny Llama checkpoint written with the real
SafeTensors and Transformers tokenizer formats.  It verifies the contract
that matters to users: compile, close/reload the persisted AEG, and generate
through the public Runtime API without a network connection.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.core.aeg_format import AEGPackage, load_aeg_package
from aether.runtime.aeg_loader import load_engine_from_path


def _write_tiny_llama(path: Path, weights_format: str = "safetensors") -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    vocab_size = 32
    hidden = 16
    intermediate = 32
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 1,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": vocab_size,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "torch_dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(7)
    tensors = {
        "model.embed_tokens.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.norm.weight": np.ones(hidden, dtype="float32"),
        "lm_head.weight": rng.normal(size=(vocab_size, hidden)).astype("float32"),
        "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.post_attention_layernorm.weight": np.ones(hidden, dtype="float32"),
        "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(16, hidden)).astype("float32"),
        "model.layers.0.self_attn.k_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
        "model.layers.0.self_attn.v_proj.weight": rng.normal(size=(8, hidden)).astype("float32"),
        "model.layers.0.self_attn.o_proj.weight": rng.normal(size=(hidden, hidden)).astype("float32"),
        "model.layers.0.mlp.gate_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        "model.layers.0.mlp.up_proj.weight": rng.normal(size=(intermediate, hidden)).astype("float32"),
        "model.layers.0.mlp.down_proj.weight": rng.normal(size=(hidden, intermediate)).astype("float32"),
    }
    if weights_format == "safetensors":
        safetensors.save_file(tensors, str(path / "model.safetensors"))
    elif weights_format == "pytorch":
        torch = pytest.importorskip("torch")
        torch.save(
            {name: torch.from_numpy(value) for name, value in tensors.items()},
            str(path / "pytorch_model.bin"),
        )
    else:
        raise ValueError(f"unsupported test weight format: {weights_format}")

    vocab = {"<unk>": 0, "hello": 1, "world": 2}
    vocab.update({f"tok{i}": i + 3 for i in range(vocab_size - 3)})
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), unk_token="<unk>"
    ).save_pretrained(str(path))


@pytest.mark.integration
def test_local_safetensors_compile_reload_runtime(tmp_path: Path) -> None:
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "tiny.aeg"

    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=16,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    )
    compiler.compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.has_weights

    archive = tmp_path / "tiny.aeg.tar.gz"
    package.save_as_archive(archive)
    archived = AEGPackage.load_from_archive(archive)
    archived.verify_integrity()
    assert archived.has_weights
    archived_logits, _ = load_engine_from_path(archived.root).forward(
        np.asarray([1], dtype=np.int64)
    )
    assert archived_logits.shape == (1, 32)

    engine = load_engine_from_path(artifact)
    logits, _ = engine.forward(np.asarray([1, 2], dtype=np.int64))
    assert logits.shape == (2, 32)

    runtime = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=3))
    response = runtime.generate(str(artifact), prompt="hello world", max_tokens=3, temperature=0.0)
    assert response.usage["completion_tokens"] == 3
    assert response.text
    cached = runtime.generate(str(artifact), prompt="hello world", max_tokens=3, temperature=0.0)
    assert cached.text == response.text
    assert cached.metrics.extra.get("cache_hit") is True
    quant_report = runtime.quantization_report(str(artifact))
    assert quant_report["status"] == "measured"
    assert quant_report["weight_bytes"] > 0
    assert quant_report["memory_mb"] > 0
    assert quant_report["vs_bf16_reduction"].endswith("x")

    from fastapi.testclient import TestClient
    from aether.server.app import create_app

    app = create_app(RuntimeConfig(hf_offline=True, default_max_tokens=2))
    api_response = TestClient(app).post(
        "/v1/generate",
        json={"model": str(artifact), "prompt": "hello world", "max_tokens": 2, "temperature": 0.0},
    )
    assert api_response.status_code == 200, api_response.text
    body = api_response.json()
    assert body["usage"]["completion_tokens"] == 2
    assert body["text"]

    with TestClient(app).stream(
        "POST",
        "/v1/generate",
        json={
            "model": str(artifact),
            "prompt": "hello world",
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": True,
        },
    ) as streamed:
        assert streamed.status_code == 200
        events = [line for line in streamed.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    stream_text = "".join(
        json.loads(line[6:])["text"] for line in events[:-1] if line[6:] != "[DONE]"
    )
    assert stream_text


@pytest.mark.integration
def test_local_pytorch_checkpoint_compile_reload_runtime(tmp_path: Path) -> None:
    """A torch.save state dict must follow the same real AEG path as SafeTensors."""
    source = tmp_path / "tiny-llama-pytorch"
    source.mkdir()
    _write_tiny_llama(source, weights_format="pytorch")
    artifact = tmp_path / "tiny-pytorch.aeg"

    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "compiler-cache"),
        )
    ).compile(str(source), output_path=artifact)

    package = load_aeg_package(artifact)
    package.verify_integrity()
    assert package.has_weights
    response = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2)).generate(
        str(artifact), prompt="hello", max_tokens=2, temperature=0.0
    )
    assert response.text
    assert response.usage["completion_tokens"] == 2


@pytest.mark.integration
def test_cli_serve_exposes_real_tcp_api_for_local_aeg(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """The documented ``aether serve`` process must serve a real local AEG."""
    import httpx

    artifact = tmp_path / "served.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=artifact)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    environment = os.environ.copy()
    environment["AETHER_HF_OFFLINE"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aether.cli",
            "serve",
            str(artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20.0
        last_error = "server did not become ready"
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout is not None else ""
                    raise AssertionError(f"aether serve exited early: {output}")
                try:
                    health = client.get(f"{base_url}/health")
                    if health.status_code == 200:
                        break
                    last_error = f"health returned HTTP {health.status_code}"
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                time.sleep(0.1)
            else:
                output = process.stdout.read() if process.stdout is not None else ""
                raise AssertionError(f"{last_error}; server output: {output}")

            response = client.post(
                f"{base_url}/v1/generate",
                json={
                    "model": str(artifact),
                    "prompt": "hello",
                    "max_tokens": 2,
                    "temperature": 0.0,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["text"]
        assert body["usage"]["completion_tokens"] == 2
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.mark.integration
def test_agentic_session_reuses_real_cpu_aeg_kv_prefix(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """Agentic turns reuse only an exact token prefix from the CPU AEG cache."""
    artifact = tmp_path / "agentic.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(tiny_local_safetensors_model), output_path=artifact)

    runtime = Runtime(
        RuntimeConfig(
            hf_offline=True,
            default_max_tokens=2,
            enable_semantic_cache=False,
        )
    )

    async def exercise() -> tuple[dict[str, object], dict[str, object]]:
        async with runtime.agentic_session(str(artifact), system="Be concise") as session:
            first = await session.generate("hello", temperature=0.0)
            second = await session.generate("world", temperature=0.0)
            return first.metrics.to_dict(), second.metrics.to_dict()

    first_metrics, second_metrics = asyncio.run(exercise())
    assert first_metrics.get("kv_reuse") is False
    assert second_metrics.get("kv_reuse") is True
    assert int(second_metrics.get("kv_reused_tokens", 0)) > 0

    backend = runtime._loaded_backends[str(artifact)]  # noqa: SLF001 - cleanup contract
    handle = backend._models[str(artifact)]  # noqa: SLF001 - cleanup contract
    assert handle.session_caches == {}


@pytest.mark.integration
def test_enabled_optimizer_artifacts_are_persisted(tmp_path: Path) -> None:
    """Enabled passes must leave files in the saved AEG, not only reports."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "features.aeg"
    compiler = Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
            enable_grammar_constraint=True,
            grammar_schema='root ::= "hello"',
            enable_ttt=True,
            enable_green_energy=True,
            enable_tee=True,
            tee_backend="nvidia_cc",
            enable_mdlm_drafter=True,
        )
    )
    package = compiler.compile(str(source), output_path=artifact)
    assert package.manifest is not None
    assert package.manifest.format_version == "AEG/3.0"
    expected = [
        "grammar/fsm.bin",
        "ttt/fast_weight_config.json",
        "metadata/green_profile.json",
        "security/tee_config.json",
        "diffusion/drafter_config.json",
    ]
    for relative in expected:
        assert (package.root / relative).is_file(), relative
    package.verify_integrity()
    reloaded = AEGPackage(package.root).load()
    assert reloaded.format_version == "AEG/3.0"

    runtime = Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False))
    runtime._load_model(str(package.root))  # noqa: SLF001 - verify persisted layer reachability
    assert runtime.grammar_engine is not None
    assert runtime.ttt_engine is not None
    assert runtime.green_power_manager is not None
    assert runtime.tee_manager is not None


@pytest.mark.integration
def test_local_aeg_grpc_generate_and_stream(tmp_path: Path) -> None:
    """Exercise the gRPC transport against a real compiled CPU AEG."""
    pytest.importorskip("grpc")
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "grpc.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)

    from aether.server.grpc_service import AetherGrpcClient, start_grpc_server

    runtime = Runtime(RuntimeConfig(hf_offline=True, default_max_tokens=2, enable_semantic_cache=False))
    server = start_grpc_server(runtime, port=0, auth_token="test-token")
    client = AetherGrpcClient(f"127.0.0.1:{server.aether_port}", auth_token="test-token")
    try:
        assert client.health()["status"] == "ok"
        request = {"model_id": str(artifact), "prompt": "hello", "max_tokens": 2, "temperature": 0.0}
        response = client.generate(request)
        assert response["text"]
        chunks = list(client.generate_stream(request))
        assert chunks and chunks[-1]["final"] is True
        assert "".join(chunk["text"] for chunk in chunks) == response["text"]
    finally:
        client.close()
        server.stop(0)


@pytest.mark.integration
def test_local_aeg_evaluation_gate_measures_runtime_output(tmp_path: Path) -> None:
    """The configured evaluator must invoke the compiled model and gate it."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    artifact = tmp_path / "eval.aeg"
    Compiler(
        CompilerConfig(
            targets=["cpu_avx512"],
            overwrite=True,
            calibration_tokens=8,
            cache_dir=str(tmp_path / "cache"),
        )
    ).compile(str(source), output_path=artifact)

    dataset = tmp_path / "mmlu.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "prompt": "hello",
                "expected": "__this_answer_cannot_match__",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from aether.observability.ci_pipeline import JsonlBenchmarkEvaluator

    runtime = Runtime(
        RuntimeConfig(hf_offline=True, default_max_tokens=2, enable_semantic_cache=False)
    )
    evaluator = JsonlBenchmarkEvaluator(
        {"mmlu": dataset},
        lambda *, prompt, benchmark, max_tokens: runtime.generate(
            str(artifact), prompt=prompt, max_tokens=max_tokens, temperature=0.0
        ).text,
        max_tokens=2,
    )
    report = runtime.eval_gate(
        model=str(artifact),
        benchmarks=["mmlu"],
        max_regression=0.02,
        evaluator=evaluator,
        baselines={"mmlu": 1.0},
    )
    assert report.status == "failed"
    assert report.passed is False
    assert report["benchmarks"][0]["num_total"] == 1
    assert report["benchmarks"][0]["metadata"]["evaluator"] == "jsonl_exact_match"


@pytest.mark.integration
def test_compiler_rejects_and_runtime_blocks_failed_eval_artifact(
    tmp_path: Path, tiny_local_safetensors_model: Path
) -> None:
    """A failed measured gate cannot be returned or loaded for deployment."""
    from aether.core.exceptions import BackendError, CompilationError

    artifact = tmp_path / "rejected.aeg"

    def evaluator(benchmark, spec):
        return {
            "benchmark": benchmark,
            "score": 0.0,
            "num_correct": 0,
            "num_total": 1,
            "latency_ms": 1.0,
        }

    with pytest.raises(CompilationError, match="Evaluation gate failed"):
        Compiler(
            CompilerConfig(
                targets=["cpu_avx512"],
                overwrite=True,
                calibration_tokens=8,
                cache_dir=str(tmp_path / "cache"),
            )
        ).compile(
            str(tiny_local_safetensors_model),
            output_path=artifact,
            evaluation_evaluator=evaluator,
            eval_benchmarks=["mmlu"],
            eval_baselines={"mmlu": 1.0},
        )

    report_path = artifact / "observability" / "eval_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    with pytest.raises(BackendError, match="rejected by its persisted evaluation gate"):
        Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False)).generate(
            str(artifact), "hello", max_tokens=1, temperature=0.0
        )


@pytest.mark.integration
def test_runtime_merge_writes_and_reloads_real_aeg(tmp_path: Path) -> None:
    """Pass 12 must produce a loadable artifact, not a recorded config."""
    source = tmp_path / "tiny-llama"
    source.mkdir()
    _write_tiny_llama(source)
    base_path = tmp_path / "base.aeg"
    task_path = tmp_path / "task.aeg"
    compiler_config = CompilerConfig(
        targets=["cpu_avx512"],
        overwrite=True,
        calibration_tokens=8,
        cache_dir=str(tmp_path / "cache"),
    )
    Compiler(compiler_config).compile(str(source), output_path=base_path)
    Compiler(compiler_config).compile(str(source), output_path=task_path)

    # Make the source a genuine task vector while preserving its AEG format.
    task = AEGPackage(task_path).load()
    from aether.quantization.formats import quantize_tensor

    task.weights = {}
    for name, tensor in task.weight_store().dequantize_all().items():
        task.weights[name] = quantize_tensor((tensor + 0.01).astype("float32"), task.weight_store().entries[name].precision)
    task.save()

    runtime = Runtime(RuntimeConfig(hf_offline=True, enable_semantic_cache=False))
    result = runtime.merge(
        str(base_path),
        task_vectors=[{"name": "task", "path": str(task_path), "coefficient": 1.0}],
    )
    merged_path = Path(result["output_model"])
    assert result["status"] == "merged"
    assert merged_path.is_dir()
    merged = load_aeg_package(merged_path)
    merged.verify_integrity()
    assert merged.has_weights

    response = runtime.generate(str(merged_path), prompt="hello", max_tokens=2, temperature=0.0)
    assert response.text
