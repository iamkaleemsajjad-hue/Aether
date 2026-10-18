"""Focused contracts for the diagnostic harness, without benchmark timing."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tools.pass_audit import (
    COMBINATIONS,
    MIB,
    PASSES,
    admission,
    case_evidence,
    compare_numerics,
    compiler_options,
    diff_artifacts,
    inventory,
    main,
    parse_passes,
    reserve_bytes,
    run_worker,
    summarize,
)
from tools.pass_audit_fixtures import (
    FIXTURES,
    PROMPTS,
    create_fixture,
    geometry,
    parameter_count,
    prompt_text,
    write_json,
)


def _fixture_description() -> dict[str, Any]:
    return {"model": "source", "merge_source": "source_delta", "adapters": []}


def test_all_22_flags_explicitly_off_survive_real_config_clone(tmp_path: Path) -> None:
    """Cloning real config must preserve all 22 explicit disabled flags."""
    from aether.compiler.config import CompilerConfig
    from aether.compiler.stage2_optimizer.optimizer import OptimizerPipeline

    options = compiler_options([], _fixture_description(), tmp_path)
    config = CompilerConfig(**options).clone()
    pipeline = OptimizerPipeline(config)
    pipeline._sync_enabled_from_config()
    assert len(PASSES) == 22
    assert [p.name for p in pipeline._passes] == [p.name for p in PASSES]
    assert len(pipeline._pass_enabled) == 22
    assert all(not enabled for enabled in pipeline._pass_enabled.values())
    assert {p.flag for p in PASSES} <= options.keys()
    assert config.precision is None


@pytest.mark.parametrize("numbers", [[1], [3], [19], *map(list, COMBINATIONS)])
def test_selected_flags_are_the_only_enabled_passes(numbers: list[int], tmp_path: Path) -> None:
    """Each singleton and combination keeps unrelated passes disabled."""
    from aether.compiler.config import CompilerConfig

    config = CompilerConfig(**compiler_options(numbers, _fixture_description(), tmp_path)).clone()
    assert {p.number for p in PASSES if getattr(config, p.flag)} == set(numbers)
    assert config.precision == ("Q4_K_M" if 3 in numbers else None)


def test_mid_is_real_27m_geometry_without_allocating_it() -> None:
    """The mid fixture accounts for exactly 27,271,680 real parameters."""
    assert parameter_count("mid") == 27_271_680
    c = geometry("mid")
    assert (c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]) == (8, 512, 1536)
    assert c["torch_dtype"] == "float32"


@pytest.mark.parametrize("name", [name for name in FIXTURES if name != "mid"])
def test_real_safetensors_fixtures_are_deterministic_and_have_triggers(
    name: str, tmp_path: Path
) -> None:
    """SafeTensors, tokenizer and feature tensors must repeat byte for byte."""
    st = pytest.importorskip("safetensors.numpy")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer

    a = create_fixture(tmp_path / "a", name)
    b = create_fixture(tmp_path / "b", name)
    assert diff_artifacts(tmp_path / "a", tmp_path / "b")["byte_identical"]
    tensors = st.load_file(str(Path(a["model"]) / "model.safetensors"))
    assert all(tensor.dtype == np.float32 for tensor in tensors.values())
    assert sum(tensor.size for tensor in tensors.values()) == parameter_count(name)
    tokenizer = Tokenizer.from_file(str(Path(a["model"]) / "tokenizer.json"))
    for ids in PROMPTS.values():
        assert tokenizer.encode(prompt_text(ids)).ids == ids
    merged = st.load_file(str(Path(a["merge_source"]) / "model.safetensors"))
    key = "model.layers.0.self_attn.q_proj.weight"
    assert not np.array_equal(tensors[key], merged[key])
    if name == "moe":
        assert a["config"]["num_local_experts"] == 4
        assert len([key for key in tensors if ".experts." in key]) == 24
    if name == "mtp":
        assert tensors["mtp_head_0.weight"].shape == (128, 32)
        assert tensors["mtp_head_1.weight"].shape == (128, 32)
    if name == "adapter":
        adapter = st.load_file(str(Path(a["adapters"][0]) / "adapter_model.safetensors"))
        assert sorted(tensor.shape for tensor in adapter.values()) == [(2, 32), (32, 2)]
    assert a["parameters"] == b["parameters"]


def test_raw_diff_includes_manifest_binary_volatile_and_added_removed_files(tmp_path: Path) -> None:
    """Volatile fields remain visible in the raw every-file comparison."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_json(a / "manifest.json", {"compiled_at": "one", "duration_ms": 1, "real": 2})
    write_json(b / "manifest.json", {"compiled_at": "two", "duration_ms": 9, "real": 3})
    (a / "weight.bin").write_bytes(b"abc")
    (b / "weight.bin").write_bytes(b"abd")
    (a / "removed").write_bytes(b"gone")
    (b / "added").write_bytes(b"new")
    (a / "same").write_bytes(b"same")
    (b / "same").write_bytes(b"same")
    diff = diff_artifacts(a, b)
    rows = {row["file"]: row for row in diff["files"]}
    assert not diff["byte_identical"] and not diff["normalization_applied"]
    assert rows["weight.bin"]["before"]["sha256"] != rows["weight.bin"]["after"]["sha256"]
    assert rows["removed"]["status"] == "removed"
    assert rows["added"]["status"] == "added"
    assert rows["same"]["status"] == "identical"
    assert {row["path"] for row in diff["volatile_metadata_changes"]} == {
        "/compiled_at",
        "/duration_ms",
    }
    assert len(rows["manifest.json"]["json_changes"]) == 3


def _numerical_result(path: Path, logits: Any, tokens: list[int]) -> dict[str, Any]:
    np.savez(path, **dict.fromkeys(PROMPTS, logits))
    return {
        "status": "ok",
        "logits": str(path),
        "tokens": {label: list(tokens) for label in PROMPTS},
    }


def test_quality_metrics_are_stable_and_greedy_lengths_are_not_truncated(tmp_path: Path) -> None:
    """Extreme logits remain stable and missing generated tokens count as disagreement."""
    values = np.array([[1000.0, -1000.0, 0.0], [0, 0, 0]])
    ref = _numerical_result(tmp_path / "ref.npz", values, [1, 2])
    got = _numerical_result(tmp_path / "got.npz", values, [1])
    result = compare_numerics(ref, got)
    assert result["status"] == "ok"
    for prompt in result["prompts"].values():
        assert prompt["kl_reference_to_candidate_mean"] == pytest.approx(0)
        assert prompt["logit_cosine_mean"] == pytest.approx(1)
        assert prompt["greedy_token_agreement"] == 0.5
        assert not prompt["greedy_exact_match"]


def test_quality_metrics_catch_distribution_shift_and_invalid_logits(tmp_path: Path) -> None:
    """Changed distributions and nonfinite logits must be observable failures."""
    ref = _numerical_result(tmp_path / "ref.npz", np.array([[1.0, 2.0]]), [1])
    got = _numerical_result(tmp_path / "got.npz", np.array([[2.0, 1.0]]), [0])
    result = compare_numerics(ref, got)["prompts"]["A"]
    assert result["kl_reference_to_candidate_mean"] > 0
    assert result["teacher_forced_argmax_agreement"] == 0
    got = _numerical_result(tmp_path / "bad.npz", np.array([[float("nan"), 1]]), [1])
    assert compare_numerics(ref, got)["status"] == "failed"
    assert compare_numerics({"status": "blocked"}, got)["status"] == "unavailable"


def test_memory_guard_blocks_instead_of_starting_large_worker() -> None:
    """Admission must preserve headroom, including larger merge reserves."""
    assert admission(100 * MIB, 400 * MIB, 512 * MIB)["status"] == "blocked"
    assert admission(1000 * MIB, 400 * MIB, 512 * MIB)["status"] == "ok"
    assert reserve_bytes("mid", "compile", [12]) > reserve_bytes("mid", "compile", [])
    assert reserve_bytes("mid", "compile", []) + 512 * MIB > 1300 * MIB


def test_emitted_graph_or_successful_generation_never_becomes_working_claim() -> None:
    """Compilation and output alone cannot prove a specialized pass benefit."""
    compiled = {"status": "ok", "passes": {p.name: "applied" for p in PASSES}}
    runtime = {"load": {"status": "ok"}, "numerics": {"status": "ok"}, "public": {"status": "ok"}}
    rows = case_evidence([1, 10, 11, 17, 22], compiled, runtime, {"status": "ok"})
    assert all(not row["working_claim"] for row in rows)
    assert all(row["runtime_consumer"] == "not_exercised" for row in rows)
    assert all(row["benefit_status"] == "not_established" for row in rows)


def test_pruned_weights_have_dense_consumer_without_claiming_sparse_benefit() -> None:
    """Executing zeroed dense weights is consumption, not sparse acceleration."""
    compiled = {"status": "ok", "passes": {"pruning_sparsity": "applied"}}
    row = case_evidence([9], compiled, {"numerics": {"status": "ok"}}, {})[0]
    assert row["runtime_consumer"] == "decoded_weight_transform_executed"
    assert not row["working_claim"]
    assert row["benefit_status"] == "not_established"
    assert "sparse kernel" in row["consumer_scope"]
    absent = case_evidence([9], compiled, {"numerics": {"status": "failed"}}, {})[0]
    assert absent["runtime_consumer"] == "not_exercised"


def test_worker_compiles_and_runs_unicode_paths_with_ascii_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker logs must be UTF-8 even when the parent requests ASCII pipes."""
    pytest.importorskip("safetensors.numpy")
    pytest.importorskip("tokenizers")
    fixture = create_fixture(tmp_path / "source_λ_模型", "tiny")
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    args = argparse.Namespace(
        min_free_mib=128, native=False, epoch=1700000000, threads=1, timeout=120
    )
    job = {"fixture_name": "tiny", "fixture": fixture, "passes": [12], "tokens": 2}
    compiled = run_worker({**job, "action": "compile"}, tmp_path / "compile_λ", args)
    assert compiled["status"] == "ok", compiled
    assert compiled["passes"]["model_merging"] == "applied"
    log = Path(compiled["worker_log"]).read_text(encoding="utf-8")
    assert "source_λ_模型" in log
    numerics = run_worker(
        {**job, "action": "numerics", "artifact": compiled["artifact"]},
        tmp_path / "numerics_λ",
        args,
    )
    assert numerics["status"] == "ok", numerics
    assert all(len(tokens) == 2 for tokens in numerics["tokens"].values())
    assert set(numerics["tokens"]) == set(PROMPTS)


def test_small_plan_has_all_22_rows_and_honest_reference_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A narrow plan retains the complete pass catalog and explicit scope limits."""
    assert main(["--plan", "--fixtures", "tiny", "--passes", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["pass_matrix"]) == 22
    assert result["selected_cases"] == [[], [1]]
    assert result["reference_labels"]["hf"] == "not_requested"
    assert "BF16" in result["reference_labels"]["primary"]
    assert "not long-context" in result["sparse_scope"]
    assert "not audited" in result["precision_scope"]
    assert "idempotency" in result["determinism_scope"]
    assert result["timing"] == "skipped_not_requested"


@pytest.mark.parametrize("values", [["0"], ["23"], ["unknown"], ["all", "1"]])
def test_invalid_pass_selection_is_refused(values: list[str]) -> None:
    """Ambiguous or out-of-range selections must not launch workers."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_passes(values)


def test_spread_reports_actual_samples() -> None:
    """Median and spread describe the retained observations exactly."""
    assert summarize([1, 2, 6]) == {
        "samples": [1, 2, 6],
        "median": 2,
        "min": 1,
        "max": 6,
        "range": 5,
        "pstdev": pytest.approx(2.160246899),
    }


def test_continuation_ppl_uses_four_labels_and_excludes_unlabelled_last_row(tmp_path: Path) -> None:
    """An extra final logit cannot influence continuation NLL or relative PPL."""
    from tools.pass_audit_fixtures import CONTINUATION

    values = np.zeros((5, 16))
    changed = values.copy()
    changed[:4, CONTINUATION] = -0.01
    changed[4] = np.arange(16) * 1000
    reference = _numerical_result(tmp_path / "reference.npz", values, [1, 2])
    candidate = _numerical_result(tmp_path / "candidate.npz", changed, [1, 2])
    result = compare_numerics(reference, candidate)["prompts"]["A"]
    ppl = result["continuation_ppl"]
    expected_delta = np.log(12 + 4 * np.exp(-0.01)) + 0.01 - np.log(16)
    assert ppl["labels"] == CONTINUATION
    assert ppl["label_count"] == 4
    assert ppl["mean_nll_difference"] == pytest.approx(expected_delta)
    assert ppl["relative_ppl_change"] == pytest.approx(np.expm1(expected_delta))
    assert not result["numerical_gate"]["passed"]  # The changed last row still fails KL.


def test_continuation_ppl_reports_overflow_without_capping_or_json_infinity(tmp_path: Path) -> None:
    """Large loss remains visible in log space, with an explicit overflow marker."""
    from tools.pass_audit_fixtures import CONTINUATION

    reference = np.zeros((5, 16))
    candidate = reference.copy()
    candidate[np.arange(4), CONTINUATION] = -1000
    ref = _numerical_result(tmp_path / "reference.npz", reference, [1])
    got = _numerical_result(tmp_path / "candidate.npz", candidate, [1])
    result = compare_numerics(ref, got)
    ppl = result["prompts"]["A"]["continuation_ppl"]
    assert ppl["mean_nll_difference"] > 900
    assert ppl["relative_ppl_change"] is None
    assert ppl["relative_ppl_overflow"]
    assert not result["prompts"]["A"]["numerical_gate"]["passed"]
    json.dumps(result, allow_nan=False)


def test_unexpected_import_error_is_failed_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An internal dependency/import regression must not masquerade as a skip."""
    import tools.pass_audit as audit

    def broken_worker(job: dict[str, Any]) -> dict[str, Any]:
        raise ImportError("broken internal aether import")

    monkeypatch.setattr(audit, "_worker", broken_worker)
    job = tmp_path / "job.json"
    write_json(job, {"action": "numerics", "work": str(tmp_path)})
    audit.worker_main(job)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "ImportError"


def test_default_sensitivity_combo_uses_real_config_default(tmp_path: Path) -> None:
    """Sensitivity/precision coverage must not silently select explicit Q4."""
    from aether.compiler.config import CompilerConfig

    options = compiler_options(
        [2, 3], _fixture_description(), tmp_path, precision_mode="sensitivity"
    )
    config = CompilerConfig(**options)
    assert config.precision is None
    assert config.precision_assignment_mode == "sensitivity"
    assert config.enable_sensitivity and config.enable_precision_assignment


def test_idempotency_snapshot_hashes_full_tensor_not_truncated_repr() -> None:
    """A mutation in an array's undisplayed middle still changes the snapshot."""
    from tools.pass_audit import canonical_graph_value

    tensor = np.zeros(10000, dtype=np.float32)
    first = canonical_graph_value({"weight": tensor})
    tensor[5000] = 1
    second = canonical_graph_value({"weight": tensor})
    assert first != second


def test_memory_marker_identity_rejects_pid_reuse_and_unowned_process() -> None:
    """A phase marker cannot select a reused PID or a process outside its tree."""
    import psutil
    from tools.pass_audit import WorkerMemorySampler, memory_point

    current = psutil.Process()
    point = memory_point("accepted")
    assert point["schema"] == "aether-pass-audit-memory/2"
    assert point["pid"] == os.getpid()
    assert point["create_time"] == current.create_time()
    sampler = WorkerMemorySampler(current)
    sampler.sample({"phase": "legacy", "rss_bytes": 10**12})
    assert sampler.worker_identity is None
    sampler.sample({**point, "create_time": point["create_time"] + 1})
    assert sampler.worker_identity is None
    sampler.sample({**point, "pid": os.getppid()})
    assert sampler.worker_identity is None
    sampler.sample(point)
    assert sampler.worker_identity == (point["pid"], point["create_time"])
    sampler.finish({**point, "create_time": point["create_time"] + 1, "rss_bytes": 10**12})
    assert sampler.phase_peaks["accepted"] < 10**12


def test_worker_peak_samples_touched_allocation_behind_launcher(tmp_path: Path) -> None:
    """Measure the real child peak while both boundary markers have low RSS."""
    import psutil
    from tools.pass_audit import WorkerMemorySampler

    if psutil.virtual_memory().available < 256 * MIB:
        pytest.skip("64 MiB allocation probe requires at least 256 MiB available RAM")
    root = Path(__file__).resolve().parents[2]
    worker = tmp_path / "allocation_worker.py"
    worker.write_text(
        "import gc, json, sys, time\n"
        "from pathlib import Path\n"
        "from tools.pass_audit import mark_phase, memory_point\n"
        "work = Path(sys.argv[1])\n"
        "mark_phase(work, 'allocation')\n"
        "slab = bytearray(64 * 1024 * 1024)\n"
        "for index in range(0, len(slab), 4096): slab[index] = 17\n"
        "time.sleep(0.3)\n"
        "del slab\n"
        "gc.collect()\n"
        "mark_phase(work, 'released')\n"
        "(work / 'final.json').write_text(json.dumps(memory_point('released')))\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    # A Python wrapper guarantees a distinct worker even on POSIX. A Windows
    # venv adds its small launcher(s), reproducing the original Popen-PID bug.
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            "import subprocess, sys; subprocess.run(sys.argv[1:], check=True)",
            sys.executable,
            "-B",
            str(worker),
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    observed = psutil.Process(process.pid)
    sampler = WorkerMemorySampler(observed)
    started = time.monotonic()
    try:
        while process.poll() is None:
            marker = None
            with contextlib.suppress(OSError, ValueError):
                marker = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
            sampler.sample(marker)
            if time.monotonic() - started > 15:
                pytest.fail("Allocation subprocess did not complete within 15 seconds")
            time.sleep(0.005)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr.decode("utf-8", errors="replace")
        final = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
        sampler.finish(final)
        assert sampler.worker_identity == (final["pid"], final["create_time"])
        assert final["pid"] != process.pid
        assert sampler.phase_peaks["allocation"] >= final["rss_bytes"] + 48 * MIB
        assert sampler.tree_peaks["allocation"] >= sampler.phase_peaks["allocation"]
    finally:
        if process.poll() is None:
            for child in reversed(observed.children(recursive=True)):
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            process.kill()
        process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()


@pytest.mark.skipif(
    os.environ.get("AETHER_RUN_PASS_AUDIT_INTEGRATION") != "1",
    reason="Opt-in real subprocess compile/load/Runtime smoke, no timing",
)
def test_real_public_compile_audit_smoke(tmp_path: Path) -> None:
    """Opt-in smoke must exercise real compiler, loader and public text API."""
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "tools" / "pass_audit.py"),
            "--fixtures",
            "tiny",
            "--passes",
            "1",
            "--output",
            str(output),
            "--tokens",
            "2",
            "--timeout",
            "120",
            "--min-free-mib",
            "128",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    baseline, fusion = report["results"]
    assert set(baseline["builds"][0]["passes"].values()) == {"skipped"}
    assert baseline["runtime"]["public"]["status"] == "ok"
    assert fusion["runtime"]["public"]["status"] == "ok"
    assert all(row["repeatability_status"] == "identical" for row in (baseline, fusion))
    assert fusion["versus_unoptimized_aeg"]["status"] == "ok"
    assert {row["file"] for row in fusion["versus_baseline_artifacts"]["files"]} == inventory(
        Path(baseline["builds"][0]["artifact"])
    ).keys()
    assert baseline["runtime"]["timing"]["status"] == "skipped"
