"""Executable smoke coverage for CLI commands that do not need a model file."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from aether.cli import cli
from aether.core.aeg_format import load_aeg_package


@pytest.mark.parametrize(
    "argv",
    [
        ["version"],
        ["doctor", "--json"],
        ["hardware", "detect", "--json"],
        ["kernels"],
        ["multi-agent", "--agents", "2"],
        ["slo-status"],
        ["green-route", "--regions", "us-east-1", "--regions", "eu-north"],
        ["kv", "transfer-stats"],
        ["kv", "nika-policy", "--kv-size-gb", "2", "--bw-gbps", "400", "--decode-util", "0.7"],
        ["kv", "cxl-pool-status"],
        ["trace"],
        ["mcp"],
        ["train", "verify", "local.aeg", "--domain", "math", "--example", "2+2=4"],
        ["compile", "Qwen/Qwen3-0.6B", "--dry-run", "--target", "cpu_avx512"],
    ],
)
def test_cli_command_executes_without_placeholder_success(argv: list[str]) -> None:
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, f"{argv}:\n{result.output}"
    # Commands must emit either a real report or an explicit usage/status
    # message; an empty successful command is not a useful public API.
    assert result.output.strip(), argv


def test_reasoning_command_fails_closed_for_missing_graph(tmp_path) -> None:
    result = CliRunner().invoke(cli, ["reasoning", str(tmp_path)])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_kernel_generation_reports_capability_error_without_traceback() -> None:
    result = CliRunner().invoke(cli, ["kernel", "generate", "cuda_sm90", "rmsnorm"])
    assert result.exit_code != 0
    assert "nvcc" in result.output.lower()
    assert "traceback" not in result.output.lower()


def test_kernel_generation_accepts_prd_argument_order_without_traceback() -> None:
    result = CliRunner().invoke(
        cli,
        ["kernel", "generate", "rmsnorm", "--target", "cuda_sm90"],
    )
    assert result.exit_code != 0
    assert "nvcc" in result.output.lower()
    assert "traceback" not in result.output.lower()


def test_kernel_verify_loads_real_native_cpu_artifact(tmp_path) -> None:
    artifact = tmp_path / "rmsnorm.dll"
    generated = CliRunner().invoke(
        cli,
        [
            "kernel",
            "generate",
            "rmsnorm",
            "--target",
            "cpu_avx2",
            "--output",
            str(artifact),
        ],
    )
    assert generated.exit_code == 0, generated.output
    verified = CliRunner().invoke(
        cli,
        ["kernel", "verify", str(artifact), "--reference-op", "rmsnorm"],
    )
    assert verified.exit_code == 0, verified.output
    assert '"verified": true' in verified.output


def test_trace_without_request_reports_no_data_explicitly() -> None:
    result = CliRunner().invoke(cli, ["trace"])
    assert result.exit_code == 0, result.output
    assert "no_measured_requests" in result.output


def test_tee_prd_subcommands_fail_closed_for_missing_artifact(tmp_path) -> None:
    attest = CliRunner().invoke(cli, ["tee", "attest", "missing.aeg"])
    assert attest.exit_code != 0
    assert "not found" in attest.output.lower()

    report = tmp_path / "attestation.json"
    report.write_text("{}", encoding="utf-8")
    verify = CliRunner().invoke(
        cli,
        ["tee", "verify", "missing.aeg", "--report", str(report)],
    )
    assert verify.exit_code != 0
    assert "missing fields" in verify.output.lower()


def test_grammar_prd_subcommands_fail_closed_for_missing_artifact() -> None:
    listed = CliRunner().invoke(cli, ["grammar", "list", "missing.aeg"])
    assert listed.exit_code != 0
    assert "not found" in listed.output.lower()

    tested = CliRunner().invoke(cli, ["grammar", "test", "missing.aeg"])
    assert tested.exit_code != 0
    assert "not found" in tested.output.lower()


def test_multi_agent_prd_test_requires_a_real_aeg() -> None:
    result = CliRunner().invoke(
        cli,
        ["multi-agent", "test", "missing.aeg", "--coordination", "relay"],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_mcp_prd_subcommands_fail_closed_for_missing_artifact() -> None:
    listed = CliRunner().invoke(cli, ["mcp", "list", "missing.aeg"])
    assert listed.exit_code == 0
    assert "not found" in listed.output.lower()

    added = CliRunner().invoke(
        cli,
        ["mcp", "add", "missing.aeg", "--server", "filesystem", "--transport", "stdio"],
    )
    assert added.exit_code != 0
    assert "not found" in added.output.lower()


def test_mcp_add_persists_registry_and_updates_integrity(minimal_aeg_package) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "mcp",
            "add",
            str(minimal_aeg_package.root),
            "--server",
            "filesystem",
            "--transport",
            "stdio",
        ],
    )
    assert result.exit_code == 0, result.output
    config_path = minimal_aeg_package.root / "mcp" / "mcp_config.json"
    assert config_path.is_file()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["server_registry"][0]["id"] == "filesystem"
    reloaded = load_aeg_package(minimal_aeg_package.root)
    reloaded.verify_integrity()


def test_slo_profile_add_persists_profile(tmp_path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "--cache-dir",
            str(tmp_path),
            "slo-profile",
            "add",
            "interactive",
            "--ttft",
            "200",
            "--tbt",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"max_ttft_ms": 200.0' in result.output
    assert '"max_tbt_ms": 50.0' in result.output
    assert (tmp_path / "slo_profiles.json").is_file()


def _placement_manifest(root, **overrides) -> None:
    """Write the minimum AEG manifest ``aether plan`` needs: an architecture."""
    architecture = {
        "family": "qwen_family", "params_billion": 0.0, "layers": 28,
        "hidden_size": 1024, "num_attention_heads": 16, "num_kv_heads": 8,
        "head_dim": 128, "context_length": 32768, "vocab_size": 151936,
        "intermediate_size": 3072, "qk_norm": True, "ffn_type": "SwiGLU",
        "norm_type": "RMSNorm", "position_type": "RoPE",
    }
    architecture.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps({"model_id": "planner-fixture", "architecture": architecture}),
        encoding="utf-8",
    )


def test_plan_prints_the_decision_record(tmp_path) -> None:
    """``aether plan`` must answer from the manifest alone — no weights loaded."""
    package = tmp_path / "fixture.aeg"
    _placement_manifest(package)
    result = CliRunner().invoke(
        cli,
        ["plan", str(package), "--context", "256", "--generate", "32", "--no-probe"],
    )
    # Either a plan or a documented refusal; both are valid on an arbitrary host.
    assert result.exit_code in (0, 2), result.output
    if result.exit_code == 0:
        for section in ("AETHER EXECUTION PLAN", "FEASIBILITY", "DECISION", "LADDER"):
            assert section in result.output
    else:
        assert "no feasible placement" in result.output
        assert "What would make this feasible" in result.output


def test_plan_emits_machine_readable_json(tmp_path) -> None:
    package = tmp_path / "fixture.aeg"
    _placement_manifest(package)
    result = CliRunner().invoke(
        cli, ["plan", str(package), "--context", "256", "--json", "--no-probe"]
    )
    assert result.exit_code in (0, 2), result.output
    payload = json.loads(result.output)
    if result.exit_code == 0:
        assert payload["selected"]["plan"]["devices"]
        assert payload["model"]["kv_bytes_per_token"] > 0
    else:
        assert payload["feasible"] is False
        assert payload["remedies"]


def test_plan_rejects_a_path_without_a_manifest(tmp_path) -> None:
    empty = tmp_path / "empty.aeg"
    empty.mkdir()
    result = CliRunner().invoke(cli, ["plan", str(empty), "--no-probe"])
    assert result.exit_code != 0
    assert "manifest" in result.output.lower()


def test_plan_intent_is_accepted_and_validated(tmp_path) -> None:
    package = tmp_path / "fixture.aeg"
    _placement_manifest(package)
    good = CliRunner().invoke(
        cli, ["plan", str(package), "--intent", "capacity", "--context", "256", "--no-probe"]
    )
    assert good.exit_code in (0, 2), good.output
    bad = CliRunner().invoke(cli, ["plan", str(package), "--intent", "nonsense", "--no-probe"])
    assert bad.exit_code != 0
