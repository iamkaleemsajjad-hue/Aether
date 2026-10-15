"""Offline checks for the fixed two-engine Mac workload and honest comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.macbook import cli, reporting
from benchmark.macbook.engine import Engine, GIB, estimate_working_bytes


def test_default_sweeps_and_smoke_keep_one_model() -> None:
    args = cli.parser().parse_args(["run"])
    cells = cli.cells_for(args)
    assert len(cells) == 9
    assert {c["batch_size"] for c in cells} == {1, 2, 4, 8, 16}
    assert {c["prompt_tokens"] for c in cells} == {32, 256, 1024}
    assert {c["output_tokens"] for c in cells} == {32, 128, 512}
    assert len({cli.case_id(c) for c in cells}) == 9
    args.matrix = "grid"
    assert len(cli.cells_for(args)) == 45
    args.smoke = True
    assert cli.cells_for(args) == [{"batch_size": 1, "prompt_tokens": 32, "output_tokens": 4}]
    assert cli.MODEL == "Qwen/Qwen3-8B"


@pytest.mark.parametrize("value", ["0", "1,-2", "", "1,x"])
def test_invalid_dimensions_rejected(value: str) -> None:
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["run", "--batch-sizes", value])


def test_exact_prompt_ids() -> None:
    tokenizer = SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=[10, 20, 30]))
    assert cli.build_prompt(tokenizer, 8) == [10, 20, 30, 10, 20, 30, 10, 20]


class FakeSession:
    def __init__(self):
        self.lengths = []

    def run(self, names, feed):
        ids = feed["input_ids"]
        batch, length = ids.shape
        past = feed["past_key_values.0.key"].shape[2]
        assert feed["attention_mask"].shape == (batch, past + length)
        self.lengths.append((batch, length, past))
        logits = np.zeros((batch, length, 10), dtype=np.float32)
        for row in range(batch):
            logits[row, -1, (ids[row, -1] + 1) % 10] = 1
        return [logits, np.zeros((batch, 8, past + length, 128), dtype=np.float32)]


def test_batched_decode_caches_reset_and_counts_are_real() -> None:
    engine = Engine.__new__(Engine)
    engine.session = FakeSession()
    engine.cache_names = ["past_key_values.0.key"]
    engine.outputs = ["logits", "present.0.key"]
    result = engine.generate([1, 2], 4, 3)
    assert result["token_ids"] == [[3, 4, 5]] * 4
    assert result["generated_tokens_total"] == 12
    assert engine.session.lengths == [(4, 2, 0), (4, 1, 2), (4, 1, 3)]
    assert result["aggregate_tokens_per_s"] == pytest.approx(12 / result["latency_s"])
    assert result["per_request_tokens_per_s"] == pytest.approx(3 / result["latency_s"])
    assert result["tpot_ms"] == pytest.approx(result["decode_s"] * 500)
    one = engine.generate([1], 1, 1)
    assert one["tpot_ms"] is None
    assert engine.session.lengths[-1] == (1, 1, 0)


def test_memory_estimate_accounts_for_batch_and_prompt() -> None:
    config = {
        "model": {
            "vocab_size": 151936,
            "decoder": {"num_hidden_layers": 36, "num_key_value_heads": 8, "head_size": 128},
        }
    }
    base = {"batch_size": 1, "prompt_tokens": 256, "output_tokens": 128}
    small = estimate_working_bytes(base, config, 6076825600)
    large = estimate_working_bytes({**base, "batch_size": 16}, config, 6076825600)
    assert 8 * GIB < small < 12 * GIB
    assert large > 12 * GIB


def measured(batch=1, prompt=256, output=128, latency=4.0, engine="onnxruntime") -> dict:
    return {
        "engine": engine,
        "prompt_ids_sha256": f"test-prompt-{prompt}",
        "batch_size": batch,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "harness_validation_only": True,
        "status": "MEASURED",
        "metrics": {
            "latency_s": {"median": latency, "n": 2},
            "aggregate_tokens_per_s": {"median": batch * output / latency, "n": 2},
            "per_request_tokens_per_s": {"median": output / latency, "n": 2},
            "ttft_s": {"median": 0.5, "n": 2},
            "tpot_ms": {"median": 20.0, "n": 2},
        },
        "worker_peak_rss_bytes": GIB,
        "sample_text": "<script>alert('x')</script>",
    }


def test_comparisons_hold_other_dimensions_fixed_and_exclude_failures() -> None:
    rows = [
        measured(),
        measured(batch=2, latency=5),
        measured(prompt=32, latency=3),
        {"batch_size": 4, "prompt_tokens": 256, "output_tokens": 128, "status": "OOM"},
    ]
    pairs = reporting.comparisons(rows)
    batch = next(p for p in pairs if p["sweep"] == "batch" and p["batch_size"] == 2)
    assert batch["throughput_ratio"] == pytest.approx(1.6)
    assert batch["batch_efficiency_percent"] == pytest.approx(80)
    assert batch["latency_ratio"] == pytest.approx(1.25)
    assert not any(p["batch_size"] == 4 for p in pairs)


def test_report_can_recover_partial_run_and_generate_real_plots(tmp_path: Path) -> None:
    cells = [
        {key: row[key] for key in ("batch_size", "prompt_tokens", "output_tokens")}
        for row in (measured(), measured(batch=2))
    ]
    cli.write_json(
        tmp_path / "run.json",
        {
            "schema": 2,
            "engines": ["aether", "onnxruntime"],
            "cells": cells,
            "smoke": False,
            "threads": 2,
            "harness_validation_only": True,
        },
    )
    cli.write_json(tmp_path / "raw" / "onnxruntime" / "b1_p256_o128.json", measured())
    reporting.render(tmp_path)
    result = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert len(result["records"]) == 4
    assert result["records"][0]["status"] == "NOT_COMPLETED"
    assert result["records"][1]["status"] == "MEASURED"
    assert len(result["graphs"]) == 8
    assert all((tmp_path / "graphs" / name).stat().st_size > 1000 for name in result["graphs"])
    assert "NOT_COMPLETED" in (tmp_path / "benchmark_results.csv").read_text()
    html = (tmp_path / "reports" / "REPORT.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "INCOMPLETE HEAD-TO-HEAD" in html


def test_actual_ort_cpu_session_with_tiny_graph(tmp_path: Path) -> None:
    """Exercise ORT itself with the supported signature, without downloading weights."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto as T, helper as h

    inputs = [
        h.make_tensor_value_info("input_ids", T.INT64, ["batch", "seq"]),
        h.make_tensor_value_info("attention_mask", T.INT64, ["batch", "total"]),
    ]
    outputs = [h.make_tensor_value_info("logits", T.FLOAT, ["batch", "seq", 10])]
    nodes = [
        h.make_node("OneHot", ["input_ids", "depth", "values"], ["logits"]),
        h.make_node("Cast", ["input_ids"], ["float_ids"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["float_ids", "axes"], ["kv_unit"]),
        h.make_node("Tile", ["kv_unit", "repeats"], ["new_kv"]),
    ]
    for layer in range(36):
        for kind in ("key", "value"):
            source, dest = f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}"
            inputs.append(h.make_tensor_value_info(source, T.FLOAT, ["batch", 8, "past", 128]))
            outputs.append(h.make_tensor_value_info(dest, T.FLOAT, ["batch", 8, "total", 128]))
            nodes.append(h.make_node("Concat", [source, "new_kv"], [dest], axis=2))
    graph = h.make_graph(
        nodes,
        "test-signature-only",
        inputs,
        outputs,
        [
            h.make_tensor("depth", T.INT64, [], [10]),
            h.make_tensor("values", T.FLOAT, [2], [0, 1]),
            h.make_tensor("axes", T.INT64, [2], [1, 3]),
            h.make_tensor("repeats", T.INT64, [4], [1, 8, 1, 128]),
        ],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, tmp_path / "model.onnx")
    engine = Engine(tmp_path, threads=1)
    result = engine.generate([1, 2, 3], batch=2, output_tokens=3)
    assert result["token_ids"] == [[3, 3, 3], [3, 3, 3]]
    assert result["generated_tokens_total"] == 6
    assert engine.session.get_providers() == ["CPUExecutionProvider"]


@pytest.mark.parametrize("mode", ["memory", "timeout", "crash"])
def test_worker_termination_preserves_failure_record(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    import psutil

    class Process:
        pid = 123
        returncode = 1 if mode == "crash" else None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=200 if mode == "memory" else 50)
        ),
    )
    times = iter([0, 10, 11, 12])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(times))
    job = {"cell": {"batch_size": 1, "prompt_tokens": 32, "output_tokens": 4}}
    result = cli.spawn(job, tmp_path, memory_limit=100, timeout=1)
    assert (
        result["status"]
        == {"memory": "MEMORY_LIMIT", "timeout": "TIMEOUT", "crash": "FAILED"}[mode]
    )
    assert "metrics" not in result
    assert json.loads((tmp_path / "b1_p32_o4.json").read_text())["status"] == result["status"]


def test_model_lock_checks_identity_sizes_and_hashes(tmp_path: Path) -> None:
    config = {"model": {"type": "qwen3", "decoder": {"num_hidden_layers": 36, "hidden_size": 4096}}}
    files = {}
    for name in cli.FILES:
        path = tmp_path / name
        path.write_text(
            json.dumps(config) if name == "genai_config.json" else "test", encoding="utf-8"
        )
        files[name] = {"path": name, "bytes": path.stat().st_size, "sha256": cli.sha256(path)}
    lock = {
        "model": cli.MODEL,
        "repository": cli.REPOSITORY,
        "subfolder": cli.SUBFOLDER,
        "files": files,
    }
    cli.write_json(tmp_path / "model.lock.json", lock)
    assert cli.load_artifact(tmp_path, verify_hashes=True)[2] == config
    (tmp_path / "model.onnx.data").write_text("evil", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        cli.load_artifact(tmp_path, verify_hashes=True)
    lock["model"] = "another/model"
    cli.write_json(tmp_path / "model.lock.json", lock)
    with pytest.raises(ValueError, match="Qwen3-8B"):
        cli.load_artifact(tmp_path)


def test_paired_ratios_require_two_engines_and_same_prompt() -> None:
    rows = [measured(), measured(engine="aether", latency=2)]
    paired = [p for p in reporting.comparisons(rows) if p["sweep"] == "engine"]
    assert len(paired) == 1
    assert paired[0]["throughput_ratio"] == 2
    assert paired[0]["latency_ratio"] == 0.5
    assert paired[0]["baseline_engine"] == "onnxruntime"
    assert "different representations" in paired[0]["comparison_scope"]
    rows[1]["prompt_ids_sha256"] = "different"
    assert not any(p["sweep"] == "engine" for p in reporting.comparisons(rows))
    rows[1]["status"] = "FAILED"
    assert not any(p["sweep"] == "engine" for p in reporting.comparisons(rows))


def test_engine_limits_do_not_fake_aether_batches_or_fallback() -> None:
    from benchmark.macbook.aether_engine import AETHER_LOAD_ESTIMATE, ENGINES

    assert ENGINES == ("aether", "onnxruntime")
    cell = {"batch_size": 1, "prompt_tokens": 32, "output_tokens": 4}
    assert (
        cli.case_unavailable("aether", cell, AETHER_LOAD_ESTIMATE, 12 * GIB, 14 * GIB, True)[0]
        == "SKIPPED_MEMORY"
    )
    assert (
        cli.case_unavailable(
            "aether", {**cell, "batch_size": 2}, AETHER_LOAD_ESTIMATE, 100 * GIB, 100 * GIB, True
        )[0]
        == "NOT_SUPPORTED"
    )
    assert (
        cli.case_unavailable("aether", cell, AETHER_LOAD_ESTIMATE, 100 * GIB, 100 * GIB, False)[0]
        == "NOT_PREPARED"
    )
    assert (
        cli.case_unavailable("aether", cell, AETHER_LOAD_ESTIMATE, 100 * GIB, 100 * GIB, True)
        is None
    )


def test_aether_preparation_refuses_small_host_before_download(tmp_path, monkeypatch) -> None:
    import psutil
    import huggingface_hub

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16 * GIB))

    def unexpected_download(*args, **kwargs):
        pytest.fail("Preparation must stop before downloading on a 16 GB host")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    with pytest.raises(ValueError, match="56 GiB"):
        cli.prepare_aether(tmp_path, "main")


def test_actual_required_ort_quantized_and_attention_kernels() -> None:
    pytest.importorskip("onnxruntime")
    from benchmark.macbook.probe import check_ort_kernels

    result = check_ort_kernels()
    assert result["MatMulNBits"] == "passed"
    assert result["GatherBlockQuantized"] == "passed"
    assert result["GroupQueryAttention"].startswith("passed")


def test_actual_aether_forward_adapter_resets_cache_and_refuses_batch() -> None:
    from aether.runtime.cpu_engine import CPUExecutionEngine
    from benchmark.macbook.aether_engine import AetherEngine
    from tests.unit.test_e2e_compile_run_cpu import _make_model_weights, HEADS, KV_HEADS

    engine = AetherEngine.__new__(AetherEngine)
    engine.executor = CPUExecutionEngine(
        _make_model_weights(), num_heads=HEADS, num_kv_heads=KV_HEADS
    )
    first = engine.generate([1, 2, 3], 1, 4)
    assert first["token_ids"] == engine.generate([1, 2, 3], 1, 4)["token_ids"]
    assert first["generated_tokens_total"] == 4
    assert first["aggregate_tokens_per_s"] == pytest.approx(4 / first["latency_s"])
    assert engine.generate([1], 1, 1)["tpot_ms"] is None
    with pytest.raises(ValueError, match="true batch"):
        engine.generate([1], 2, 2)


def test_old_single_engine_report_schema_is_rejected(tmp_path) -> None:
    cli.write_json(tmp_path / "run.json", {"schema": 1, "cells": []})
    with pytest.raises(ValueError, match="exactly Aether and ONNX"):
        reporting.render(tmp_path)


def test_run_schedules_exactly_two_engines_and_preserves_separate_records(tmp_path, monkeypatch):
    import psutil

    args = cli.parser().parse_args(
        [
            "run",
            "--matrix",
            "grid",
            "--batch-sizes",
            "1",
            "--prompt-tokens",
            "32,64",
            "--output-tokens",
            "4",
            "--max-memory-gib",
            "100",
            "--cooldown",
            "0",
            "--output-dir",
            str(tmp_path / "results"),
            "--aether-dir",
            str(tmp_path / "aether"),
        ]
    )
    cli.write_json(args.aether_dir / "aether.lock.json", {})
    config = {
        "model": {
            "context_length": 40960,
            "vocab_size": 151936,
            "decoder": {
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "head_size": 128,
            },
        }
    }
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        cli,
        "load_artifact",
        lambda *a, **kw: ({"files": {"model.onnx.data": {"bytes": 6076825600}}}, tmp_path, config),
    )
    monkeypatch.setattr(cli, "read_lock", lambda *a: ({"model": cli.MODEL}, tmp_path))
    monkeypatch.setattr(cli, "host_info", lambda: {"ram_bytes": 128 * GIB})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=120 * GIB))
    monkeypatch.setattr(reporting, "render", lambda *a: None)
    calls = []

    def fake_worker(job, directory, memory_limit, timeout):
        calls.append((job["engine"], job["cell"]["prompt_tokens"]))
        assert directory.name == job["engine"]
        return {**job["cell"], "status": "MEASURED"}

    monkeypatch.setattr(cli, "spawn", fake_worker)
    assert cli.run(args) == 0
    assert calls == [("aether", 32), ("onnxruntime", 32), ("onnxruntime", 64), ("aether", 64)]
    plan = json.loads((args.output_dir / "run.json").read_text())
    assert plan["engines"] == ["aether", "onnxruntime"]
    assert plan["model"] == "Qwen/Qwen3-8B"
    for engine, prompt in calls:
        row = json.loads((args.output_dir / "raw" / engine / f"b1_p{prompt}_o4.json").read_text())
        assert row["engine"] == engine
        assert row["prompt_tokens"] == prompt
