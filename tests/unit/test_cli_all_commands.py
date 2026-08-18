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
