"""CLI contract tests for documented v5 options and commands."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from aether.cli import cli


def test_run_replaces_terminal_incompatible_model_text(monkeypatch) -> None:
    """The CLI must not fail when a model emits Unicode on cp1252 stdout."""
    from aether.cli import _display_text

    class LegacyStream:
        encoding = "cp1252"

    monkeypatch.setattr("aether.cli.console.file", LegacyStream(), raising=False)
    assert "?" in _display_text("녕녕")


def test_compile_accepts_documented_v5_option_values(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeCompiler:
        def __init__(self, config) -> None:
            captured["config"] = config

        def compile(self, model: str, output_path: str | None = None):
            captured["model"] = model
            captured["output_path"] = output_path
            return SimpleNamespace(
                root="compiled.aeg",
                metadata={
                    "optimizer_passes": [
                        "sub2bit_quantization",
                        "mdlm_drafter_compilation",
                        "video_token_compression",
                    ]
                },
            )

    monkeypatch.setattr("aether.cli.Compiler", FakeCompiler)
    result = CliRunner().invoke(
        cli,
        [
            "compile",
            "local-model",
            "--sub2bit",
            "ternary",
            "--mdlm-drafter",
            "--mdlm-K",
            "4",
            "--mdlm-T",
            "3",
            "--video-compression",
            "storm",
            "--output",
            str(tmp_path / "model.aeg"),
        ],
    )
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.enable_sub2bit is True
    assert config.sub2bit_method == "bitnet"
    assert config.enable_mdlm_drafter is True
    assert config.mdlm_draft_block_size == 4
    assert config.mdlm_drafter_steps == 3
    assert config.enable_video_compression is True
    assert config.video_compression_backend == "storm"


def test_compile_bare_feature_flags_use_real_defaults(monkeypatch) -> None:
    captured = {}

    class FakeCompiler:
        def __init__(self, config) -> None:
            captured["config"] = config

        def compile(self, model: str, output_path: str | None = None):
            return SimpleNamespace(
                root="compiled.aeg",
                metadata={
                    "optimizer_passes": [
                        "sub2bit_quantization",
                        "video_token_compression",
                    ]
                },
            )

    monkeypatch.setattr("aether.cli.Compiler", FakeCompiler)
    result = CliRunner().invoke(
        cli,
        ["compile", "local-model", "--sub2bit", "--video-compression"],
    )
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.sub2bit_method == "bitnet"
    assert config.video_compression_backend == "stc"


def test_compile_rejects_false_success_when_requested_pass_is_skipped(monkeypatch) -> None:
    class FakeCompiler:
        def __init__(self, config) -> None:
            pass

        def compile(self, model: str, output_path: str | None = None):
            return SimpleNamespace(root="compiled.aeg", metadata={"optimizer_passes": []})

    monkeypatch.setattr("aether.cli.Compiler", FakeCompiler)
    result = CliRunner().invoke(cli, ["compile", "local-model", "--tee"])
    assert result.exit_code != 0
    assert "TEE compilation was not applied" in result.output


def test_cache_commands_and_quantize_report_are_registered() -> None:
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "quantize-report" in help_result.output
    assert "cache" in help_result.output

    stats = runner.invoke(cli, ["cache", "stats"])
    assert stats.exit_code == 0, stats.output
    assert '"enabled"' in stats.output

    flushed = runner.invoke(cli, ["cache", "flush"])
    assert flushed.exit_code == 0, flushed.output
    assert '"removed"' in flushed.output

    missing = runner.invoke(cli, ["quantize-report", "missing.aeg"])
    assert missing.exit_code != 0
    assert "No such command" not in missing.output


def test_eval_accepts_local_dataset_mapping(monkeypatch, tmp_path) -> None:
    dataset = tmp_path / "mmlu.csv"
    dataset.write_text(
        "question,A,B,C,D,answer\nWhat is 2+2?,3,4,5,6,B\n",
        encoding="utf-8",
    )

    class FakeRuntime:
        def __init__(self, config) -> None:
            self.config = config

        def generate(self, model, prompt, *, max_tokens, temperature):
            return SimpleNamespace(text="B")

        def eval_gate(self, model, **kwargs):
            result = kwargs["evaluator"](
                "mmlu", {"metric": "accuracy"}
            )
            assert result.score == 1.0
            assert kwargs["benchmarks"] == ["mmlu"]
            return {"passed": True, "status": "passed", "benchmarks": [result.to_dict()]}

    monkeypatch.setattr("aether.cli.Runtime", FakeRuntime)
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "local.aeg",
            "--dataset",
            f"mmlu={dataset}",
            "--examples",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"passed": true' in result.output
