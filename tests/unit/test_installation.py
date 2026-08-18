"""Clean-install contract checks for the public CPU package surface."""

from __future__ import annotations

import importlib

from click.testing import CliRunner

from aether import AetherClient, AetherHub, Compiler, Runtime
from aether.cli import cli


def test_public_package_exports_import() -> None:
    assert all(value is not None for value in (AetherClient, AetherHub, Compiler, Runtime))


def test_core_modules_import_without_optional_accelerators() -> None:
    modules = (
        "aether.backends.hardware_detector",
        "aether.compiler.compiler",
        "aether.compiler.stage1_ingestion.ingestion",
        "aether.core.aeg_format",
        "aether.runtime.runtime",
        "aether.server.routes",
    )
    for module in modules:
        assert importlib.import_module(module) is not None


def test_doctor_reports_ten_local_checks() -> None:
    result = CliRunner().invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    assert '"total": 10' in result.output
    assert '"failed": 0' in result.output

