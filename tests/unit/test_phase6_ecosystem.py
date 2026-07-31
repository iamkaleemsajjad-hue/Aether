"""Tests for Phase 6 — Ecosystem (SDKs, VS Code Plugin, Hub).

Covers:
- TypeScriptSDKGenerator — output content, write to file
- GoSDKGenerator — Go syntax, write to file
- RustSDKGenerator — Rust syntax, write to file
- GitHubActionsGenerator — YAML structure, benchmark list
- VSCodePluginManifest — package.json structure, commands
- VSCodeCommandRegistry — command count, all required fields
- AEGInspectorProvider — TypeScript extension source
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# TypeScript SDK
# ---------------------------------------------------------------------------

class TestTypeScriptSDKGenerator:
    def test_generate_returns_string(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        gen = TypeScriptSDKGenerator()
        code = gen.generate()
        assert isinstance(code, str)
        assert len(code) > 100

    def test_contains_client_class(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate()
        assert "class AetherClient" in code

    def test_contains_generate_method(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate()
        assert "async generate" in code

    def test_contains_streaming(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate()
        assert "async *stream" in code or "AsyncGenerator" in code

    def test_contains_type_definitions(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate()
        assert "GenerateRequest" in code
        assert "GenerateResponse" in code
        assert "ChatMessage" in code

    def test_custom_base_url(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate(base_url="https://api.example.com")
        assert "https://api.example.com" in code

    def test_export_default(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        code = TypeScriptSDKGenerator().generate()
        assert "export default AetherClient" in code

    def test_write_creates_file(self):
        from aether.ecosystem.sdks import TypeScriptSDKGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            out = TypeScriptSDKGenerator().write(tmpdir)
            assert out.exists()
            assert out.name == "aether-sdk.ts"
            assert len(out.read_text()) > 500


# ---------------------------------------------------------------------------
# Go SDK
# ---------------------------------------------------------------------------

class TestGoSDKGenerator:
    def test_generate_returns_string(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        gen = GoSDKGenerator()
        code = gen.generate()
        assert isinstance(code, str)

    def test_contains_package_declaration(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate(package="aether")
        assert "package aether" in code

    def test_contains_client_struct(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate()
        assert "type Client struct" in code

    def test_contains_generate_method(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate()
        assert "func (c *Client) Generate" in code

    def test_contains_imports(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate()
        assert "net/http" in code
        assert "encoding/json" in code

    def test_contains_types(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate()
        assert "GenerateRequest" in code
        assert "GenerateResponse" in code
        assert "ChatMessage" in code

    def test_write_creates_file(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            out = GoSDKGenerator().write(tmpdir)
            assert out.exists()
            assert out.name == "aether_client.go"

    def test_custom_package_name(self):
        from aether.ecosystem.sdks import GoSDKGenerator
        code = GoSDKGenerator().generate(package="myaether")
        assert "package myaether" in code


# ---------------------------------------------------------------------------
# Rust SDK
# ---------------------------------------------------------------------------

class TestRustSDKGenerator:
    def test_generate_returns_string(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        gen = RustSDKGenerator()
        code = gen.generate()
        assert isinstance(code, str)

    def test_contains_struct_definitions(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        code = RustSDKGenerator().generate()
        assert "pub struct GenerateRequest" in code
        assert "pub struct GenerateResponse" in code
        assert "pub struct AetherClient" in code

    def test_contains_async_method(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        code = RustSDKGenerator().generate()
        assert "pub async fn generate" in code

    def test_contains_serde_derives(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        code = RustSDKGenerator().generate()
        assert "Serialize" in code
        assert "Deserialize" in code

    def test_write_creates_file(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            out = RustSDKGenerator().write(tmpdir)
            assert out.exists()
            assert out.name == "aether_client.rs"

    def test_health_method_present(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        code = RustSDKGenerator().generate()
        assert "pub async fn health" in code

    def test_contains_reqwest_import(self):
        from aether.ecosystem.sdks import RustSDKGenerator
        code = RustSDKGenerator().generate()
        assert "reqwest" in code


# ---------------------------------------------------------------------------
# GitHub Actions Generator
# ---------------------------------------------------------------------------

class TestGitHubActionsGenerator:
    def test_generate_returns_yaml_string(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        gen = GitHubActionsGenerator()
        yaml_str = gen.generate()
        assert isinstance(yaml_str, str)
        assert "name:" in yaml_str

    def test_contains_eval_gate_job(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        yaml_str = GitHubActionsGenerator().generate()
        assert "eval-gate" in yaml_str
        assert "CIEvalPipeline" in yaml_str

    def test_contains_pip_install(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        yaml_str = GitHubActionsGenerator().generate()
        assert "pip install" in yaml_str
        assert "aether-runtime" in yaml_str

    def test_contains_upload_artifact(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        yaml_str = GitHubActionsGenerator().generate()
        assert "upload-artifact" in yaml_str
        assert "eval_report.json" in yaml_str

    def test_benchmarks_in_workflow(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        yaml_str = GitHubActionsGenerator().generate(
            benchmarks=["hellaswag", "mmlu", "gsm8k"]
        )
        assert "hellaswag" in yaml_str
        assert "mmlu" in yaml_str

    def test_max_regression_in_workflow(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        yaml_str = GitHubActionsGenerator().generate(max_regression_pct=3.0)
        assert "0.030" in yaml_str or "3.0" in yaml_str or "0.03" in yaml_str

    def test_write_creates_workflow_file(self):
        from aether.ecosystem.sdks import GitHubActionsGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            out = GitHubActionsGenerator().write(tmpdir, model_id="test-model")
            assert out.exists()
            assert out.parent.name == "workflows"
            assert out.name == "aether-eval.yml"


# ---------------------------------------------------------------------------
# VS Code Plugin Manifest
# ---------------------------------------------------------------------------

class TestVSCodePluginManifest:
    def test_generate_returns_dict(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        manifest = VSCodePluginManifest().generate()
        assert isinstance(manifest, dict)

    def test_has_required_fields(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        assert "name" in m
        assert "version" in m
        assert "engines" in m
        assert "contributes" in m
        assert "main" in m

    def test_has_commands(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        commands = m["contributes"]["commands"]
        assert len(commands) >= 10  # All Aether CLI commands

    def test_aether_compile_command_present(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        cmd_ids = [c["command"] for c in m["contributes"]["commands"]]
        assert "aether.compile" in cmd_ids

    def test_custom_editor_for_aeg(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        editors = m["contributes"]["customEditors"]
        assert any("*.aeg" in e["selector"][0]["filenamePattern"] for e in editors)

    def test_language_contribution(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        langs = m["contributes"]["languages"]
        assert any(l["id"] == "aeg-ir" for l in langs)

    def test_write_package_json(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        with tempfile.TemporaryDirectory() as tmpdir:
            out = VSCodePluginManifest().write(tmpdir)
            assert out.exists()
            assert out.name == "package.json"
            data = json.loads(out.read_text())
            assert data["name"] == "aether-runtime"

    def test_configuration_properties(self):
        from aether.ecosystem.vscode_plugin import VSCodePluginManifest
        m = VSCodePluginManifest().generate()
        config = m["contributes"]["configuration"]["properties"]
        assert "aether.serverUrl" in config
        assert "aether.defaultTarget" in config
        assert "aether.defaultPrecision" in config


# ---------------------------------------------------------------------------
# VS Code Command Registry
# ---------------------------------------------------------------------------

class TestVSCodeCommandRegistry:
    def test_all_commands_have_required_fields(self):
        from aether.ecosystem.vscode_plugin import VSCodeCommandRegistry
        registry = VSCodeCommandRegistry()
        for cmd in registry.to_contribution_list():
            assert "command" in cmd
            assert "title" in cmd
            assert "category" in cmd

    def test_minimum_command_count(self):
        from aether.ecosystem.vscode_plugin import VSCodeCommandRegistry
        registry = VSCodeCommandRegistry()
        assert len(registry.to_contribution_list()) >= 10

    def test_all_aether_prefix(self):
        from aether.ecosystem.vscode_plugin import VSCodeCommandRegistry
        registry = VSCodeCommandRegistry()
        for cmd in registry.to_contribution_list():
            assert cmd["command"].startswith("aether.")


# ---------------------------------------------------------------------------
# AEG Inspector Provider
# ---------------------------------------------------------------------------

class TestAEGInspectorProvider:
    def test_generates_ts_source(self):
        from aether.ecosystem.vscode_plugin import AEGInspectorProvider
        provider = AEGInspectorProvider()
        ts = provider.generate_extension_ts()
        assert isinstance(ts, str)
        assert "export function activate" in ts
        assert "export function deactivate" in ts

    def test_registers_custom_editor(self):
        from aether.ecosystem.vscode_plugin import AEGInspectorProvider
        ts = AEGInspectorProvider().generate_extension_ts()
        assert "registerCustomEditorProvider" in ts
        assert "aether.aegInspector" in ts

    def test_registers_commands(self):
        from aether.ecosystem.vscode_plugin import AEGInspectorProvider
        ts = AEGInspectorProvider().generate_extension_ts()
        assert "aether.compile" in ts
        assert "aether.serve" in ts

    def test_write_creates_extension_ts(self):
        from aether.ecosystem.vscode_plugin import AEGInspectorProvider
        with tempfile.TemporaryDirectory() as tmpdir:
            out = AEGInspectorProvider().write(tmpdir)
            assert out.exists()
            assert out.name == "extension.ts"
            assert out.parent.name == "src"  # written to output_dir/src/
            content = out.read_text(encoding="utf-8")  # file is UTF-8 encoded
            assert "activate" in content
