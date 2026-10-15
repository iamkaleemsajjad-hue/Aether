"""
Aether CLI — command-line interface for compiling, running, and serving models.

Provides commands:
- compile
- pull
- run
- serve
- bench
- info
- graph
- list
- rm
- hw
- kernels
- logs
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import sys
import tempfile
import time
import builtins
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.compiler.config import SUPPORTED_COMPILATION_PRECISIONS
from aether.compiler.stage2_optimizer.pass16_green_energy import GreenEnergyCompilationPass
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import load_aeg_package
from aether.core.constants import AETHER_VERSION, DEFAULT_HUB_URL, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from aether.core.exceptions import AetherError
from aether.hub.client import HubClient
from aether.utils.file_io import aether_cache_dir, delete_model
from aether.utils.logging import configure_logging

console = Console()


@contextmanager
def _cli_runtime(config: RuntimeConfig) -> Iterator[Runtime]:
    """Own a command's runtime across returns, failures and Click exits."""
    runtime = Runtime(config)
    try:
        yield runtime
    finally:
        runtime.close()


def _display_text(value: object) -> str:
    """Return text that can be written by the active terminal encoding.

    Windows installations can expose a legacy cp1252 stdout even when the
    model emits Unicode.  Replacing only unencodable characters keeps the CLI
    command successful and preserves all text that the terminal can display.
    The runtime/API response remains unchanged; this is presentation-only.
    """
    text = str(value)
    stream = getattr(console, "file", None)
    encoding = getattr(stream, "encoding", None) or getattr(sys.stdout, "encoding", None)
    if not encoding:
        return text
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    return text


def _require_applied_pass(aeg: Any, pass_name: str, feature: str) -> None:
    """Refuse to report a feature that the compiler did not apply.

    Optimizer passes are allowed to skip when their real inputs or backend
    artifacts are unavailable.  A CLI command must not turn that explicit
    skip into a successful-looking artifact message.
    """
    metadata = getattr(aeg, "metadata", {}) or {}
    applied = metadata.get("optimizer_passes", [])
    if not isinstance(applied, builtins.list) or pass_name not in applied:
        raise click.ClickException(
            f"{feature} was not applied; the requested real backend/artifact is unavailable"
        )


def _runtime_config(ctx: click.Context, **overrides: Any) -> RuntimeConfig:
    """Preserve environment controls when a CLI command overrides settings."""
    data = RuntimeConfig.from_env().to_dict()
    config_path = ctx.obj.get("runtime_config")
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.stat().st_size > 1024**2:
            raise click.ClickException("runtime configuration exceeds the 1 MiB limit")
        supplied = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise click.ClickException("runtime configuration must be a JSON object")
        unknown = set(supplied) - set(data)
        if unknown:
            raise click.ClickException(f"unknown runtime configuration fields: {sorted(unknown)}")
        data.update(supplied)
    if ctx.obj.get("cache_dir") is not None:
        data["model_cache_dir"] = ctx.obj["cache_dir"]
    data.update(overrides)
    return RuntimeConfig.from_dict(data)


class _AetherGroup(click.Group):
    """Present expected configuration and execution errors as CLI errors."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (AetherError, OSError, ValueError) as exc:
            raise click.ClickException(_display_text(exc)) from exc
        except ImportError as exc:
            raise click.ClickException(f"missing optional dependency: {_display_text(exc)}") from exc


class _FiniteFloatRange(click.FloatRange):
    """Reject non-finite CLI values before they reach models or scheduling."""

    def convert(self, value: Any, param: Any, ctx: Any) -> float:
        converted = super().convert(value, param, ctx)
        if not math.isfinite(converted):
            self.fail("must be a finite number", param, ctx)
        return converted


@click.group(cls=_AetherGroup, context_settings={"max_content_width": 120})
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--cache-dir", type=click.Path(), help="Custom Aether cache directory.")
@click.option("--runtime-config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="RuntimeConfig JSON for serving, generation and execution features.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, cache_dir: str | None, runtime_config: Path | None) -> None:
    """Aether Runtime — compile any AI model, run it on any hardware."""
    configure_logging(level="DEBUG" if verbose else "INFO")
    # Rich captures the stream that existed when this module was imported.
    # Click (and test runners) replace stdout per invocation, so bind the
    # console to the active Click stream for every command invocation.
    console.file = sys.stdout
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["cache_dir"] = cache_dir
    ctx.obj["runtime_config"] = runtime_config


@cli.command("compile")
@click.argument("model")
@click.option("--target", "-t", multiple=True, help="Hardware target(s). Repeat for multiple targets.")
@click.option(
    "--precision", type=click.Choice(SUPPORTED_COMPILATION_PRECISIONS, case_sensitive=False),
    default=None, help="Explicit uniform layer precision; default preserves automatic assignment.",
)
@click.option("--quality-budget", type=_FiniteFloatRange(min=0.0), default=0.02, help="Maximum perplexity increase budget.")
@click.option("--calibration-dataset", default="wikitext-2", help="Calibration dataset for sensitivity analysis.")
@click.option("--upload", is_flag=True, help="Upload compiled AEG to Aether Hub.")
@click.option("--output", "-o", type=click.Path(), help="Output path for the AEG artifact.")
@click.option("--dry-run", is_flag=True, help="Plan compilation without producing an AEG.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing AEG package.")
@click.option("--mtp", "enable_mtp", is_flag=True, help="Compile native MTP heads.")
@click.option(
    "--sub2bit",
    type=click.Choice(["bitnet", "ternary", "btc_llm", "nanoq", "nanoquant"]),
    is_flag=False,
    flag_value="bitnet",
    default=None,
    help="Enable verified sub-2-bit quantization (optional mode: ternary, btc_llm, or nanoq).",
)
@click.option("--mdlm-drafter", is_flag=True, help="Compile an MDLM speculative drafter.")
@click.option(
    "--mdlm-weights",
    "mdlm_weights",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Trained MDLM head bundle (.npz or SafeTensors); required with --mdlm-drafter.",
)
@click.option(
    "--mdlm-K",
    "mdlm_k",
    type=click.IntRange(2, 64),
    default=None,
    help="MDLM draft block size K (requires --mdlm-drafter).",
)
@click.option(
    "--mdlm-T",
    "mdlm_t",
    type=click.IntRange(1, 64),
    default=None,
    help="MDLM denoising steps T (requires --mdlm-drafter).",
)
@click.option(
    "--video-compression",
    type=click.Choice(["stc", "storm", "streamingtom", "streaming_tom", "infotok", "mage_vl"]),
    is_flag=False,
    flag_value="stc",
    default=None,
    help="Enable VLM/video token compression (optional strategy).",
)
@click.option("--grammar-schema", type=click.Path(exists=True), help="Grammar schema for constrained decoding.")
@click.option("--ttt", "enable_ttt", is_flag=True, help="Inject TTT fast-weight slots.")
@click.option("--green", "enable_green", is_flag=True, help="Embed green energy profile.")
@click.option("--tee", "enable_tee", is_flag=True, help="Emit TEE wrapper metadata.")
@click.pass_context
def cmd_compile(
    ctx: click.Context,
    model: str,
    target: tuple[str, ...],
    precision: str | None,
    quality_budget: float,
    calibration_dataset: str,
    upload: bool,
    output: str | None,
    dry_run: bool,
    overwrite: bool,
    enable_mtp: bool,
    sub2bit: str | None,
    mdlm_drafter: bool,
    mdlm_weights: Path | None,
    mdlm_k: int | None,
    mdlm_t: int | None,
    video_compression: str | None,
    grammar_schema: str | None,
    enable_ttt: bool,
    enable_green: bool,
    enable_tee: bool,
) -> None:
    """Compile a model into an AEG artifact."""
    if not mdlm_drafter and any(value is not None for value in (mdlm_weights, mdlm_k, mdlm_t)):
        raise click.UsageError("--mdlm-weights/--mdlm-K/--mdlm-T require --mdlm-drafter")
    targets = list(target) if target else ["auto"]
    config = CompilerConfig(
        targets=targets,
        precision=precision,
        quality_budget=quality_budget,
        calibration_dataset=calibration_dataset,
        upload_kernels=upload,
        dry_run=dry_run,
        overwrite=overwrite,
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        verbose=ctx.obj.get("verbose", False),
        enable_mtp_compilation=enable_mtp,
        enable_sub2bit=sub2bit is not None,
        sub2bit_mode=sub2bit,
        enable_mdlm_drafter=mdlm_drafter,
        mdlm_drafter_weights_path=str(mdlm_weights) if mdlm_weights else None,
        mdlm_draft_block_size=mdlm_k or 8,
        mdlm_denoising_steps=mdlm_t or 6,
        enable_video_compression=video_compression is not None,
        video_compression_strategy=video_compression,
        enable_ttt=enable_ttt,
        enable_green_profile=enable_green,
        enable_tee=enable_tee,
    )
    if grammar_schema:
        config.enable_grammar_constraint = True
        config.grammar_schema = Path(grammar_schema).read_text(encoding="utf-8")
    compiler = Compiler(config)

    if dry_run:
        plan = compiler.plan(model)
        console.print(f"[bold]Compilation Plan for {model}[/bold]")
        console.print(f"Targets: {plan.targets}")
        console.print(f"Feasible: {plan.is_feasible}")
        console.print(f"Estimated memory: {plan.estimated_memory_gb:.1f} GB")
        console.print(f"Estimated compile time: {plan.estimated_compile_time_s:.1f} s")
        console.print(f"Estimated AEG size: {plan.estimated_aeg_size_gb:.1f} GB")
        console.print(f"Opportunities: {plan.total_opportunities}")
        for opp in plan.fusion_opportunities:
            console.print(f"  Fusion: {opp.description}")
        for opp in plan.precision_opportunities:
            console.print(f"  Precision: {opp.description}")
        for tgt, backend in plan.backend_recommendations.items():
            console.print(f"  Backend for {tgt}: {backend}")
        return

    with console.status(f"[bold green]Compiling {model}..."):
        aeg = compiler.compile(model, output_path=output)
    requested_passes = []
    if enable_mtp:
        requested_passes.append(("mtp_head_compilation", "MTP compilation"))
    if sub2bit is not None:
        requested_passes.append(("sub2bit_quantization", "sub-2-bit quantization"))
    if mdlm_drafter:
        requested_passes.append(("mdlm_drafter_compilation", "MDLM drafter compilation"))
    if video_compression is not None:
        requested_passes.append(("video_token_compression", "video compression"))
    if grammar_schema:
        requested_passes.append(("grammar_constraint_compilation", "grammar compilation"))
    if enable_ttt:
        requested_passes.append(("ttt_fast_weight_injection", "TTT compilation"))
    if enable_green:
        requested_passes.append(("green_energy_compilation", "green-energy compilation"))
    if enable_tee:
        requested_passes.append(("tee_kernel_wrapping", "TEE compilation"))
    for pass_name, feature in requested_passes:
        _require_applied_pass(aeg, pass_name, feature)
    console.print(f"[bold green]Compiled AEG saved to[/bold green] {aeg.root}")


@cli.command()
@click.argument("model")
@click.option("--compile-local", is_flag=True, help="Recompile the source locally even if a cached AEG exists.")
@click.pass_context
def pull(ctx: click.Context, model: str, compile_local: bool) -> None:
    """Download and compile a model to the local AEG cache."""
    with console.status(f"[bold green]Pulling {model}..."):
        if compile_local:
            Compiler(CompilerConfig(
                cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
                overwrite=True,
            )).compile(model)
        else:
            with _cli_runtime(_runtime_config(ctx)) as runtime:
                runtime.pull(model)
    console.print(f"[bold green]Model {model} ready.[/bold green]")


@cli.command()
@click.argument("model")
@click.option("--prompt", "-p", default="Hello, my name is", help="Prompt for generation.")
@click.option("--max-tokens", type=click.IntRange(1), default=128, help="Maximum tokens to generate.")
@click.option("--temperature", type=_FiniteFloatRange(min=0.0), default=0.7, help="Sampling temperature.")
@click.option("--top-p", type=_FiniteFloatRange(0.0, 1.0, min_open=True), default=0.9, help="Top-p sampling parameter.")
@click.option("--stream", is_flag=True, help="Stream output.")
@click.option(
    "--raw",
    is_flag=True,
    help=(
        "Send the prompt verbatim instead of as a chat turn. Use for base models, "
        "or to measure raw continuation."
    ),
)
@click.option(
    "--chat-template",
    "chat_template",
    default=None,
    help=(
        "Jinja chat template, or a path to one, for a checkpoint that packages none. "
        "Some instruction-tuned checkpoints record their turn format nowhere in the "
        "artifact; this is how you supply it."
    ),
)
@click.option("--non-interactive", is_flag=True, help="Run in non-interactive mode.")
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    raw: bool,
    chat_template: str | None,
    non_interactive: bool,
) -> None:
    """Run a model and generate text.

    A prompt typed here is a *question*, so it is delivered the way the checkpoint
    was trained to receive one: instruction-tuned artifacts package a chat template,
    and the prompt is rendered through it. Without that an instruction-tuned model
    does not answer — it continues the string, fluently and irrelevantly. Base models
    declare no template and are unaffected. ``--raw`` opts out.

    A checkpoint that packages no template cannot have its format recovered from the
    artifact, so a warning names ``--chat-template`` as the remedy.
    """
    rt = Runtime(_runtime_config(ctx))
    try:
        with console.status(f"[bold green]Loading {model}..."):
            rt._load_model(model)  # noqa: SLF001

        if not non_interactive:
            console.print(f"[bold]Aether Runtime[/bold] — model: {model} (backend: {rt._loaded_backends[model].name})")  # noqa: SLF001

        template = chat_template
        if template:
            candidate = Path(template)
            # A template is usually long and awkward to quote on a command line, so a path
            # is accepted wherever the string is.
            if candidate.is_file():
                template = candidate.read_text(encoding="utf-8")

        extra: dict[str, Any] = {"apply_chat_template": not raw}
        if template:
            extra["chat_template"] = template

        if stream:
            click.echo("Response: ", nl=False)
            iterator = rt.generate_stream(
                model_id=model, prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, **extra,
            )
            try:
                for delta in iterator:
                    click.echo(_display_text(delta), nl=False)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            click.echo()
            return

        with console.status("[bold green]Generating..."):
            response = rt.generate(
                model_id=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **extra,
            )
        console.print(f"[bold]Response:[/bold] {_display_text(response.text)}")
        console.print(f"[dim]Tokens: {response.usage}[/dim]")
        console.print(f"[dim]Metrics: {response.metrics.to_dict()}[/dim]")
    finally:
        rt.close()


@cli.command()
@click.argument("model", required=False)
@click.option("--port", type=click.IntRange(1, 65535), default=DEFAULT_SERVER_PORT, help="Server port.")
@click.option("--host", default=DEFAULT_SERVER_HOST, help="Server host.")
@click.option("--grpc-port", default=None, type=click.IntRange(0, 65535), help="Also expose authenticated gRPC on this port (0 selects a free port).")
@click.option(
    "--semantic-cache/--no-semantic-cache",
    default=None,
    help="Enable or disable the R11 semantic request cache.",
)
@click.option(
    "--threshold",
    "semantic_cache_threshold",
    type=_FiniteFloatRange(0.0, 1.0),
    default=0.92,
    show_default=True,
    help="Cosine similarity threshold for semantic cache hits.",
)
@click.pass_context
def serve(
    ctx: click.Context,
    model: str | None,
    port: int,
    host: str,
    grpc_port: int | None,
    semantic_cache: bool | None,
    semantic_cache_threshold: float,
) -> None:
    """Start the Aether REST server, optionally alongside gRPC."""
    try:
        from aether.server.app import create_app
        import uvicorn
    except ImportError:
        console.print("[red]fastapi and uvicorn are required for the server.[/red]")
        sys.exit(1)

    configured_slo_profiles: dict[str, dict[str, float | None]] = {}
    slo_profile_path = aether_cache_dir(ctx.obj.get("cache_dir")) / "slo_profiles.json"
    if slo_profile_path.is_file():
        try:
            loaded_profiles = json.loads(slo_profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"invalid SLO profile file {slo_profile_path}: {exc}") from exc
        if not isinstance(loaded_profiles, dict):
            raise click.ClickException(f"SLO profile file {slo_profile_path} must contain a JSON object")
        configured_slo_profiles = loaded_profiles

    config = _runtime_config(
        ctx,
        server_port=port,
        server_host=host,
        semantic_cache_threshold=semantic_cache_threshold,
        slo_profiles=configured_slo_profiles,
        **({"enable_semantic_cache": semantic_cache} if semantic_cache is not None else {}),
    )
    app = create_app(config)
    runtime = app.state.aether_runtime
    grpc_server = None
    try:
        if model:
            try:
                # A model argument is a startup contract, not decorative CLI
                # metadata: validate and load it before accepting requests.
                app.state.aether_runtime._load_model(model)  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(f"unable to load model {model!r}: {exc}") from exc
        if grpc_port is not None:
            try:
                from aether.server.grpc_service import start_grpc_server

                grpc_server = start_grpc_server(
                    app.state.aether_runtime,
                    host=host,
                    port=grpc_port,
                    auth_token=os.environ.get("AETHER_GRPC_API_KEY"),
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Unable to start gRPC: {exc}[/red]")
                sys.exit(1)
        console.print(f"[bold green]Starting Aether server at http://{host}:{port}[/bold green]")
        if grpc_server is not None:
            console.print(f"[bold green]gRPC listening at {host}:{grpc_server.aether_port}[/bold green]")
        uvicorn.run(app, host=host, port=port)
    finally:
        # Stop admitting RPCs and wait for their handlers before unloading
        # the shared REST/gRPC execution state.
        try:
            if grpc_server is not None:
                from aether.server.grpc_service import shutdown_grpc_server

                shutdown_grpc_server(grpc_server)
        finally:
            runtime.close()


@cli.command()
@click.argument("model")
@click.option("--compare", help="Compare against a saved bench JSON result or an installed backend.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=128, help="Maximum tokens per benchmark run.")
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Save measured results as JSON.")
@click.option("--json", "as_json", is_flag=True, help="Output measured results as JSON.")
@click.option("--video", "video_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option(
    "--speculative",
    type=click.Choice(["eagle3", "p-eagle", "mdlm"], case_sensitive=False),
    default=None,
    help="Speculative engine to benchmark when its real backend is available.",
)
@click.pass_context
def bench(
    ctx: click.Context,
    model: str,
    compare: str | None,
    max_tokens: int,
    output: str | None,
    as_json: bool,
    video_path: str | None,
    speculative: str | None,
) -> None:
    """Benchmark a model on the current hardware."""
    if video_path is not None:
        raise click.ClickException(
            "video benchmarking requires a real video/VLM runtime backend; "
            "the current installation has no executable video backend"
        )
    if speculative is not None:
        raise click.ClickException(
            f"speculative benchmark mode {speculative!r} is unavailable until its "
            "real draft/verification backend is loaded"
        )
    rt = Runtime(_runtime_config(ctx))
    try:
        with console.status(f"[bold green]Benchmarking {model}..."):
            result = rt.benchmark(model, max_tokens=max_tokens)
    finally:
        # Release the primary model before allocating a comparison backend.
        rt.close()
    if compare:
        baseline_path = Path(compare)
        if baseline_path.suffix.lower() == ".json":
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            comparison_runtime = Runtime(_runtime_config(ctx, backend_name=compare))
            try:
                baseline = comparison_runtime.benchmark(model, max_tokens=max_tokens)
            finally:
                comparison_runtime.close()
        import math

        rate = baseline.get("throughput_tps") if isinstance(baseline, dict) else None
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise click.ClickException("comparison result requires a positive finite throughput_tps")
        result["comparison"] = {
            "baseline": baseline,
            "throughput_speedup": result["throughput_tps"] / rate,
        }
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    table = Table(title=f"Benchmark: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)

@cli.command()
@click.argument("model")
@click.pass_context
def info(ctx: click.Context, model: str) -> None:
    """Show metadata and precision map for a compiled model."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        try:
            data = rt.info(model)
            click.echo(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument("model")
@click.pass_context
def graph(ctx: click.Context, model: str) -> None:
    """Print the AEG-IR graph of a compiled model."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if aeg_path is None:
            raise click.ClickException(f"model {model!r} was not found")
        aeg = load_aeg_package(aeg_path)
        if aeg.ir:
            console.print(aeg.ir.to_text())
        else:
            raise click.ClickException("AEG has no IR loaded")


@cli.command("list")
@click.pass_context
def cmd_list(ctx: click.Context) -> None:
    """List all compiled models in the local cache."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        models = rt.list()
        if not models:
            console.print("[dim]No compiled models found.[/dim]")
            return
        table = Table(title="Compiled Models")
        table.add_column("Model ID", style="cyan")
        for model in models:
            table.add_row(model)
        console.print(table)


@cli.command()
@click.argument("model")
@click.confirmation_option(prompt="Are you sure you want to remove this model?")
@click.pass_context
def rm(ctx: click.Context, model: str) -> None:
    """Remove a compiled model from the cache."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        rt.remove(model)
        console.print(f"[bold green]Removed {model}[/bold green]")


def _print_hardware_fingerprint() -> None:
    """Show the current device fingerprint without terminal JSON wrapping."""
    from aether.runtime.hardware import HardwareDetector
    fingerprint = HardwareDetector().detect()
    click.echo(json.dumps(fingerprint.to_dict(), indent=2, default=str))


@cli.command()
def kernels() -> None:
    """List active kernel targets and recommended backends."""
    target = HardwareProfile.auto()
    registry = TargetRegistry()
    table = Table(title="Kernel Targets")
    table.add_column("Target ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Recommended Backend", style="green")
    for tid in registry.supported_targets:
        profile = registry.get_profile(tid)
        table.add_row(tid, profile.name, profile.recommended_backend or "pytorch")
    console.print(table)
    console.print(f"[dim]Current target: {target.target_id}[/dim]")


@cli.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--lines", type=click.IntRange(1), default=50, help="Number of lines to show.")
@click.pass_context
def logs(ctx: click.Context, follow: bool, lines: int) -> None:
    """Show recent Aether runtime logs."""
    cache = aether_cache_dir(ctx.obj.get("cache_dir"))
    log_dir = cache / "logs"
    if not log_dir.exists():
        console.print("[dim]No logs found.[/dim]")
        return
    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        console.print("[dim]No log files found.[/dim]")
        return
    latest = log_files[-1]
    if follow:
        console.print(f"[dim]Following {latest}...[/dim]")
        # Simplified tail
        try:
            import tailer
            with latest.open(encoding="utf-8") as stream:
                for line in tailer.follow(stream):
                    click.echo(line, nl=False)
        except ImportError as exc:
            raise click.ClickException("Install 'tailer' for follow mode.") from exc
    else:
        content = latest.read_text(encoding="utf-8")
        for line in content.splitlines()[-lines:]:
            console.print(line)


@cli.command()
@click.argument("grammar_args", nargs=-1, required=True)
@click.option("--schema", type=click.Path(exists=True), help="Path to GBNF/LARK grammar schema file.")
@click.option("--mode", type=click.Choice(["gbnf", "lark", "regex"]), default="gbnf", help="Grammar format.")
@click.option("--dry-run", is_flag=True, help="Plan grammar constraint pass without recompiling.")
@click.option("--target", "target_model", type=click.Path(), help="Existing AEG/model to receive the grammar.")
@click.option("--name", "grammar_name", default="default", help="Persisted grammar name.")
@click.pass_context
def grammar(
    ctx: click.Context,
    grammar_args: tuple[str, ...],
    schema: str | None,
    mode: str,
    dry_run: bool,
    target_model: str | None,
    grammar_name: str,
) -> None:
    """Compile grammar constraints into the AEG artifact (Pass 11).

    Injects a GrammarConstraintCompilerPass FST/WFSA into the compiled model
    so token sampling is structurally constrained at inference time without
    any post-processing overhead.
    """
    args = list(grammar_args)
    action = "compile"
    if args and args[0].lower() in {"compile", "list", "test"}:
        action = args.pop(0).lower()

    if action == "list":
        if len(args) != 1:
            raise click.UsageError("grammar list requires <model.aeg>")
        with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
            path = rt._resolve_aeg_path(args[0])  # noqa: SLF001
            if path is None:
                raise click.ClickException(f"model {args[0]!r} was not found")
            root = Path(path)
            config_path = root / "grammar" / "fsm_config.json"
            if not config_path.is_file():
                raise click.ClickException(f"model {args[0]!r} has no persisted grammar FSM")
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise click.ClickException(f"invalid grammar metadata: {exc}") from exc
            click.echo(json.dumps({"model_id": args[0], "grammars": [payload]}, indent=2))
            return

    if action == "test":
        if len(args) != 1:
            raise click.UsageError("grammar test requires <model.aeg>")
        with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
            path = rt._resolve_aeg_path(args[0])  # noqa: SLF001
            if path is None:
                raise click.ClickException(f"model {args[0]!r} was not found")
            rt._init_v4_layers(path)  # noqa: SLF001
            engine = rt.grammar_engine
            if engine is None or not engine.is_loaded():
                raise click.ClickException(f"model {args[0]!r} has no loadable grammar FSM")
            session = engine.create_session()
            mask = session.get_token_mask()
            click.echo(json.dumps(
                        {
                            "model_id": args[0],
                            "grammar": grammar_name,
                            "loaded": True,
                            "initial_allowed_token_count": sum(byte.bit_count() for byte in mask),
                            "prompt_tested": False,
                            "message": "FSM loaded and initial token mask inspected; no model prompt was executed.",
                        },
                        indent=2,
                    ))
            return

    if args and target_model is None:
        # Legacy form: aether grammar <model> --schema schema.gbnf
        model = args.pop(0)
    else:
        # PRD form: aether grammar compile <schema.json> --target <model.aeg>
        if len(args) != 1:
            raise click.UsageError("grammar compile requires <schema> --target <model.aeg>")
        model = target_model
        schema = args.pop(0)
    if args:
        raise click.UsageError(f"unexpected grammar arguments: {' '.join(args)}")
    if model is None:
        raise click.UsageError("a grammar target model is required")
    if schema is None:
        raise click.UsageError("--schema is required to define the grammar")
    schema_path = Path(schema)
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_grammar_constraint=True,
    )
    config.grammar_schema = schema_path.read_text(encoding="utf-8")
    config.grammar_backend = mode
    if dry_run:
        console.print(f"[bold]Grammar constraint plan for {model}[/bold]")
        console.print(f"  Schema:  {schema_path}")
        console.print(f"  Format:  {mode}")
        console.print(f"  Pass:    GrammarConstraintCompilerPass (Pass 11)")
        console.print("[dim]Dry-run: no recompilation performed.[/dim]")
        return
    with console.status(f"[bold green]Injecting grammar constraints into {model}..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)
    _require_applied_pass(aeg, "grammar_constraint_compilation", "grammar compilation")
    console.print(
        f"[bold green]Grammar-constrained AEG saved to[/bold green] {aeg.root}"
    )


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--method", type=click.Choice(["ties", "dare", "linear", "slerp"]), default="ties",
              help="Model merging algorithm; slerp is reserved and currently unsupported.")
@click.option("--output", "-o", type=click.Path(), help="Output path for merged AEG.")
@click.option("--weights", default="", help="Comma-separated merge coefficients (e.g. '0.6,0.4').")
@click.pass_context
def merge(ctx: click.Context, models: tuple[str, ...], method: str, output: str | None, weights: str) -> None:
    """Merge two or more compiled AEG models (Pass 12).

    Supports TIES (trimming + elect sign + merge), DARE (dropout +
    rescale), and linear interpolation for compatible AEG inputs. SLERP
    is reserved and fails explicitly. Merged weights require a new quality
    evaluation; compatible execution contracts are preserved.

    Examples:
        aether merge model-a model-b --method ties
        aether merge model-a model-b model-c --method linear --weights 0.5,0.3,0.2
    """
    if len(models) < 2:
        raise click.UsageError("At least two models are required for merging")
    if method == "slerp":
        raise click.ClickException("SLERP is not implemented by the model-merging pass")
    merge_weights: list[float] = []
    if weights:
        try:
            merge_weights = [float(w.strip()) for w in weights.split(",")]
        except ValueError as exc:
            raise click.BadParameter("must be comma-separated floats", param_hint="--weights") from exc
        if len(merge_weights) != len(models) or not all(math.isfinite(weight) for weight in merge_weights):
            raise click.BadParameter("provide one finite coefficient per model", param_hint="--weights")
        if method == "linear" and not math.isclose(sum(merge_weights), 1.0, abs_tol=1e-9):
            raise click.BadParameter("linear interpolation coefficients must sum to 1", param_hint="--weights")
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_model_merging=True,
        model_merging_method="task_arithmetic" if method == "linear" else method,
        model_merging_sources=list(models),
        model_merging_coefficients=merge_weights or [1.0 / len(models)] * len(models),
    )

    with console.status(f"[bold green]Merging {len(models)} models via {method.upper()}..."):
        from aether.utils.file_io import aeg_cache_candidates, aeg_cache_path

        resolved = []
        for reference in models:
            candidates = [Path(reference), *aeg_cache_candidates(reference, config.cache_dir)]
            resolved.append(next((candidate for candidate in candidates if (candidate / "manifest.json").is_file()), None))
        if any(path is not None for path in resolved):
            if not all(path is not None for path in resolved):
                raise click.ClickException("Merging AEG inputs requires every input to resolve to an AEG package")
            from aether.compiler.aeg_merging import merge_aeg_packages

            destination = output or aeg_cache_path(f"merged/{models[0]}", config.cache_dir)
            try:
                aeg = merge_aeg_packages(resolved, destination, method=method, coefficients=config.model_merging_coefficients)
            except (ValueError, OSError) as exc:
                raise click.ClickException(str(exc)) from exc
        else:
            compiler = Compiler(config)
            aeg = compiler.compile(models[0], output_path=output)
    _require_applied_pass(aeg, "model_merging", "model merging")
    console.print(f"[bold green]Merged AEG saved to[/bold green] {aeg.root}")
    table = Table(title="Merge Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Weight", style="magenta")
    for m, w in zip(models, config.model_merging_coefficients):
        table.add_row(m, f"{w:.3f}")
    table.add_row("[bold]Method[/bold]", method.upper())
    console.print(table)


@cli.command("ttt-config")
@click.argument("model")
@click.option("--adapter-rank", type=click.IntRange(1), default=8, help="LoRA rank for TTT fast-weight adapters.")
@click.option("--ttl", type=click.IntRange(1), default=None, help="Reserved context TTL; unsupported until the TTT engine implements token expiry.")
@click.option("--layers", default="", help="Comma-separated layer indices to enable TTT (empty=all).")
@click.pass_context
def ttt_config(ctx: click.Context, model: str, adapter_rank: int, ttl: int | None, layers: str) -> None:
    """Configure Test-Time Training fast-weight injection (Pass 13).

    Compiles LoRA-style fast-weight adapters into the AEG that update
    continuously during inference, enabling the model to adapt to the
    current context window without gradient descent or full fine-tuning.
    """
    target_layers: list[int] | None = None
    if layers:
        try:
            target_layers = [int(x.strip()) for x in layers.split(",")]
        except ValueError as exc:
            raise click.BadParameter("must be comma-separated non-negative integers", param_hint="--layers") from exc
        if any(layer < 0 for layer in target_layers):
            raise click.BadParameter("must be non-negative", param_hint="--layers")
        raise click.ClickException("--layers selection is not implemented by the TTT compilation pass")
    if ttl is not None:
        raise click.ClickException("--ttl token expiry is not implemented by the TTT engine")

    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_ttt=True,
    )
    config.ttt_rank = adapter_rank
    config.ttt_layers = target_layers or []

    with console.status(f"[bold green]Injecting TTT fast-weight adapters into {model}..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)

    _require_applied_pass(aeg, "ttt_fast_weight_injection", "TTT compilation")
    console.print(f"[bold green]TTT-enabled AEG saved to[/bold green] {aeg.root}")
    table = Table(title="TTT Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Adapter rank (r)", str(adapter_rank))
    table.add_row("Context TTL", "not implemented")
    table.add_row("Target layers", str(target_layers or "all"))
    console.print(table)


@cli.command("kv-compress")
@click.argument("model")
@click.option("--method", type=click.Choice(["semantic", "cross-layer", "both"]), default="both",
              help="KV compression method: semantic (Pass 14), cross-layer (Pass 15), or both.")
@click.option("--retention", type=_FiniteFloatRange(0.0, 1.0, min_open=True), default=0.5,
              help="Fraction of KV tokens to retain (0.0–1.0).")
@click.option("--cross-layer-groups", type=click.IntRange(1), default=None,
              help="Reserved group count; current Pass 15 selects groups from its sharing threshold.")
@click.pass_context
def kv_compress(ctx: click.Context, model: str, method: str, retention: float, cross_layer_groups: int | None) -> None:
    """Compile KV cache compression into the AEG (Passes 14 and 15).

    Semantic KV (Pass 14) compresses the KV cache by identifying semantically
    redundant token representations and merging them.  Cross-layer KV sharing
    (Pass 15) groups layers to share a single KV cache, reducing memory footprint
    by up to 50% on deep models.
    """
    if cross_layer_groups is not None:
        raise click.ClickException("--cross-layer-groups is not implemented by the cross-layer KV compilation pass")

    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_semantic_kv=method in ("semantic", "both"),
        enable_cross_layer_kv=method in ("cross-layer", "both"),
    )
    config.semantic_kv_compression_ratio = retention
    config.cross_layer_kv_share_threshold = max(0.0, min(1.0, 1.0 - retention))

    with console.status(f"[bold green]Compiling KV compression into {model} (method={method})..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)

    if method in ("semantic", "both"):
        _require_applied_pass(aeg, "semantic_kv_compression", "semantic KV compression")
    if method in ("cross-layer", "both"):
        _require_applied_pass(aeg, "cross_layer_kv_sharing", "cross-layer KV sharing")
    console.print(f"[bold green]KV-compressed AEG saved to[/bold green] {aeg.root}")
    table = Table(title="KV Compression Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Method", method)
    table.add_row("Retention ratio", f"{retention:.1%}")
    if method in ("cross-layer", "both"):
        table.add_row("Cross-layer groups", "selected by compilation pass")
    console.print(table)


@cli.command("green-profile")
@click.argument("model")
@click.option("--carbon-region", default="", help="Electricity grid carbon region (e.g. 'us-west-2').")
@click.option("--renewable-threshold", type=_FiniteFloatRange(0.0, 1.0), default=0.80,
              help="Minimum renewable energy fraction to enable full compilation (0.0-1.0).")
@click.option("--defer/--no-defer", default=False,
              help="Defer compilation until renewable threshold is met.")
@click.pass_context
def green_profile(
    ctx: click.Context,
    model: str,
    carbon_region: str,
    renewable_threshold: float,
    defer: bool,
) -> None:
    """Compile a green-energy-aware AEG artifact (Pass 16).

    Green Energy Compilation (PRD ss16) embeds carbon-aware scheduling hints
    into the AEG so the runtime can defer batch execution to windows with
    higher renewable energy availability, using Electricity Maps or WattTime APIs.
    """
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_green_energy=True,
    )
    config.green_carbon_region = carbon_region or "auto"
    config.green_renewable_threshold = renewable_threshold
    config.green_defer_compilation = defer

    if defer:
        console.print(
            f"[bold yellow]Deferred mode:[/bold yellow] compilation will proceed when "
            f"renewable fraction >= {renewable_threshold:.0%} in region {carbon_region or 'auto'}."
        )

    with console.status(f"[bold green]Compiling green-energy profile for {model}..."):
        source = Path(model)
        if source.is_dir() and (source / "manifest.json").is_file():
            # A green profile is also a supported post-compilation operation.
            # Re-ingesting an AEG directory as if it were a Hugging Face source
            # loses its existing graph/weights and was the old failure mode.
            from aether.core.aeg_format import AEGPackage

            aeg = AEGPackage(source)
            aeg.load()
            if aeg.ir is None or aeg.manifest is None:
                raise click.ClickException("The AEG has no loadable graph or manifest")
            with tempfile.TemporaryDirectory(prefix="aether-green-profile-") as staging:
                setattr(aeg.ir, "output_dir", staging)
                _, report = GreenEnergyCompilationPass().run(
                    aeg.ir, aeg.manifest.architecture, config
                )
                if report.status != "applied":
                    raise click.ClickException(
                        f"Green-energy compilation failed: {report.details.get('error', 'unknown error')}"
                    )
                staged_profile = Path(staging) / "metadata" / "green_profile.json"
                if not staged_profile.is_file():
                    raise click.ClickException(
                        "Green-energy pass produced no persisted profile artifact"
                    )
                destination = aeg.root / "metadata" / "green_profile.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged_profile, destination)
                aeg.metadata["green_profile"] = json.loads(
                    destination.read_text(encoding="utf-8")
                )
                applied = list(aeg.metadata.get("optimizer_passes", []))
                if "green_energy_compilation" not in applied:
                    applied.append("green_energy_compilation")
                aeg.metadata["optimizer_passes"] = applied
                if not aeg.manifest.format_version.startswith(("AEG/2.", "AEG/3.")):
                    aeg.manifest.format_version = "AEG/2.0"
                aeg.save()
        else:
            compiler = Compiler(config)
            aeg = compiler.compile(model, output_path=None)

    _require_applied_pass(aeg, "green_energy_compilation", "green-energy compilation")
    console.print(f"[bold green]Green AEG saved to[/bold green] {aeg.root}")
    table = Table(title="Green Energy Profile")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Carbon region", carbon_region or "auto-detected")
    table.add_row("Renewable threshold", f"{renewable_threshold:.0%}")
    table.add_row("Defer mode", "enabled" if defer else "disabled")
    console.print(table)


@cli.command()
@click.argument("tee_args", nargs=-1, required=True)
@click.option("--backend", "--mode", type=click.Choice(["nvidia_cc", "intel_tdx", "amd_sev_snp", "openpcc"]),
              default="nvidia_cc", help="TEE backend.")
@click.option("--output", "-o", type=click.Path(), help="Output path for TEE-wrapped AEG.")
@click.option("--report", is_flag=True, help="Print TEE attestation report after compilation.")
@click.pass_context
def tee(ctx: click.Context, tee_args: tuple[str, ...], backend: str, output: str | None, report: bool) -> None:
    """Wrap a model in TEE-protected kernels (Pass 17).

    TEE Kernel Wrapping injects HMAC-guarded enter/exit guards around every
    compute kernel and generates a weight-hash manifest so the runtime can
    verify model integrity inside the trusted execution enclave.
    """
    action = "compile"
    args = list(tee_args)
    if args and args[0].lower() in {"compile", "attest", "verify"}:
        action = args.pop(0).lower()
    if not args:
        raise click.UsageError("provide an AEG/model path")
    model = args.pop(0)

    if action == "attest":
        if args:
            raise click.UsageError(f"unexpected TEE arguments: {' '.join(args)}")
        with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
            aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
            if aeg_path is None:
                raise click.ClickException(f"model {model!r} was not found")
            security_dir = Path(aeg_path) / "security"
            tee_config = security_dir / "tee_config.json"
            from aether.runtime.r8_tee_manager import TEERuntimeManager

            manager = TEERuntimeManager(
                backend=backend,
                tee_config_path=str(tee_config) if tee_config.is_file() else None,
            )
            try:
                if not manager.initialize():
                    raise click.ClickException("TEE initialization failed")
                attestation = manager.get_attestation_report()
                attestation["verification_scope"] = "local_platform_probe_and_report_generation"
                attestation["cryptographically_verified"] = False
                click.echo(json.dumps(attestation, indent=2, default=str))
            finally:
                manager.shutdown()
            return

    if action == "verify":
        report_path = args.pop(0) if args else None
        if not report:
            raise click.UsageError("tee verify requires --report <attestation.json>")
        if report_path is None:
            raise click.UsageError("tee verify requires a report path after --report")
        if args:
            raise click.UsageError(f"unexpected TEE arguments: {' '.join(args)}")
        path = Path(report_path)
        if not path.is_file():
            raise click.ClickException(f"attestation report {path} was not found")
        try:
            attestation = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"invalid attestation report: {exc}") from exc
        required_fields = {
            "backend", "token", "enclave_initialized", "hardware_backed", "generated_at"
        }
        missing = sorted(required_fields - set(attestation)) if isinstance(attestation, dict) else sorted(required_fields)
        if missing:
            raise click.ClickException(f"attestation report is missing fields: {missing}")
        result = {
            "model_id": model,
            "structural_valid": True,
            "verified": False,
            "hardware_backed_claim": attestation.get("hardware_backed"),
            "trust_level": "unverified",
            "verification_scope": "report_structure_only",
            "error": "vendor attestation signature, trust chain and artifact binding verification are not implemented",
            "report": attestation,
        }
        click.echo(json.dumps(result, indent=2, default=str))
        ctx.exit(1)

    if args:
        raise click.UsageError(f"unexpected TEE arguments: {' '.join(args)}")

    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_tee=True,
    )
    config.tee_backend = backend

    with console.status(f"[bold green]Wrapping {model} with TEE ({backend.upper()}) protection..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model, output_path=output)

    _require_applied_pass(aeg, "tee_kernel_wrapping", "TEE compilation")
    console.print(f"[bold green]TEE-wrapped AEG saved to[/bold green] {aeg.root}")

    if report:
        try:
            from aether.runtime.r8_tee_manager import TEERuntimeManager
            tee_mgr = TEERuntimeManager(backend=backend)
            if not tee_mgr.initialize():
                raise click.ClickException("TEE initialization failed")
            try:
                attestation = tee_mgr.get_attestation_report()
            finally:
                tee_mgr.shutdown()
            console.print(f"\n[bold]Attestation Report[/bold]")
            table = Table()
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="magenta")
            for k, v in attestation.items():
                table.add_row(k, str(v) if v is not None else "[dim]n/a[/dim]")
            console.print(table)
            hw_backed = attestation.get("hardware_backed", False)
            if hw_backed:
                console.print("[bold yellow]Platform reports TEE support; vendor attestation and model execution are unverified.[/bold yellow]")
            else:
                console.print("[bold yellow]WARN: Running in software simulation mode (no hardware TEE detected).[/bold yellow]")
        except Exception as exc:
            raise click.ClickException(f"Could not generate attestation report: {exc}") from exc


@cli.command()
@click.option("--list-tools", "-l", is_flag=True, help="List all MCP tools registered in the AEG.")
@click.option("--tool", default="", help="Name of a specific MCP tool to inspect.")
@click.option("--call", default="", help="JSON arguments to call a tool (requires --tool).")
@click.option("--server", default="", help="MCP server identifier for PRD add/test forms.")
@click.option("--transport", type=click.Choice(["stdio", "http", "ws", "websocket"]), default="stdio")
@click.option("--endpoint", default=None, help="MCP HTTP/WebSocket endpoint.")
@click.option("--command", "server_command", default=None, help="MCP stdio server command.")
@click.argument("mcp_args", nargs=-1)
@click.pass_context
def mcp(
    ctx: click.Context,
    list_tools: bool,
    tool: str,
    call: str,
    server: str,
    transport: str,
    endpoint: str | None,
    server_command: str | None,
    mcp_args: tuple[str, ...],
) -> None:
    """Inspect or invoke MCP tools registered in a compiled AEG (Runtime R6).

    The MCP (Model Context Protocol) integration layer (Runtime R6) allows
    models to call external tools such as web search, code execution, and
    database access through a structured JSON-RPC interface compiled into
    the AEG artifact.

    Examples:
        aether mcp mymodel --list-tools
        aether mcp mymodel --tool web_search --call '{\"query\": \"Aether Runtime\"}'
    """
    args = list(mcp_args)
    transport = "ws" if transport == "websocket" else transport
    action = "legacy"
    model: str | None = None
    if args and args[0].lower() in {"add", "list", "test"}:
        action = args.pop(0).lower()
        if args:
            model = args.pop(0)
        if args:
            raise click.UsageError(f"unexpected MCP arguments: {' '.join(args)}")
        if action == "list":
            list_tools = True
        elif action == "test":
            if not tool:
                raise click.UsageError("mcp test requires --tool <tool-name>")
            call = call or "{}"
        elif action == "add":
            if not model:
                raise click.UsageError("mcp add requires <model.aeg>")
            if not server:
                raise click.UsageError("mcp add requires --server <server-id>")
            if transport == "stdio" and not server_command:
                # PRD examples use well-known server IDs (for example
                # ``filesystem``).  Persist that ID as the executable; the
                # runtime will perform the real connection and fail closed if
                # it is not installed.
                server_command = server
            if transport in {"http", "ws"} and not endpoint:
                raise click.UsageError(f"{transport} MCP servers require --endpoint <url>")
            with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
                aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
                if aeg_path is None:
                    raise click.ClickException(f"model {model!r} not found in cache")
                package = load_aeg_package(aeg_path)
                root = Path(aeg_path)
                config_path = root / "mcp" / "mcp_config.json"
                existing: dict[str, Any] = {}
                if config_path.is_file():
                    try:
                        existing = json.loads(config_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as exc:
                        raise click.ClickException(f"invalid MCP config: {exc}") from exc
                if not isinstance(existing, dict):
                    raise click.ClickException("MCP config must be a JSON object")
                registry = existing.get("server_registry", [])
                if not isinstance(registry, list):
                    raise click.ClickException("MCP server_registry must be a JSON list")
                if any(isinstance(item, dict) and item.get("id") == server for item in registry):
                    raise click.ClickException(f"MCP server {server!r} is already registered")
                registry.append(
                    {
                        "id": server,
                        "transport": transport,
                        "endpoint": endpoint,
                        "command": server_command,
                        "tools": [],
                    }
                )
                payload = {
                    "format_version": "AEG/2.0" if package.manifest and package.manifest.format_version == "AEG/2.0" else "AEG/3.0",
                    "status": "enabled",
                    "enabled": True,
                    "server_registry": registry,
                    "default_timeout_ms": int(existing.get("default_timeout_ms", 5000)),
                    "max_parallel_tool_calls": int(existing.get("max_parallel_tool_calls", 4)),
                }
                registry_payload = {"servers": registry}
                config_path.parent.mkdir(parents=True, exist_ok=True)
                registry_path = root / "mcp" / "server_registry.json"
                config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                registry_path.write_text(json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8")
                if package.manifest is not None:
                    from aether.core.hash_utils import compute_file_hash

                    package.manifest.artifacts["mcp/mcp_config.json"] = compute_file_hash(config_path)
                    package.manifest.artifacts["mcp/server_registry.json"] = compute_file_hash(registry_path)
                    package.manifest.compute_and_set_manifest_hash()
                    (root / "manifest.json").write_text(
                        package.manifest.to_json(indent=2), encoding="utf-8"
                    )
                click.echo(json.dumps(
                            {"model_id": model, "server": registry[-1], "persisted": True},
                            indent=2,
                        ))
                return
    elif args:
        model = args.pop(0)
    if args:
        raise click.UsageError(f"unexpected MCP arguments: {' '.join(args)}")

    if not model:
        console.print("[dim]Usage: aether mcp <model> --list-tools[/dim]")
        return
    if call and not tool:
        raise click.UsageError("--call requires --tool")

    mcp_layer = None
    rt = None
    try:
        rt = Runtime(_runtime_config(ctx))
        aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if aeg_path is None:
            raise click.ClickException(f"model {model!r} not found in cache")

        # Load the artifact's persisted MCP registry.  Constructing a fresh
        # layer here silently discarded every server/tool compiled into the
        # AEG and made this command report an empty registry for valid files.
        rt._init_v4_layers(aeg_path)  # noqa: SLF001
        mcp_layer = rt.mcp_layer
        if mcp_layer is None:
            if tool:
                raise click.ClickException("MCP is not enabled by this AEG")
            console.print("[dim]MCP is not enabled by this AEG.[/dim]")
            return
        if list_tools:
            tools = mcp_layer.list_tools()
            if not tools:
                console.print("[dim]No MCP tools registered for this AEG.[/dim]")
                return
            table = Table(title=f"MCP Tools — {model}")
            table.add_column("Tool Name", style="cyan")
            table.add_column("Description", style="magenta")
            table.add_column("Schema", style="dim")
            for t in tools:
                table.add_row(
                    t.get("name", "?"),
                    t.get("description", ""),
                    json.dumps(t.get("input_schema", {}), indent=None)[:60],
                )
            console.print(table)

        elif tool and call:
            try:
                args = json.loads(call)
            except json.JSONDecodeError as exc:
                raise click.BadParameter(f"invalid JSON: {exc}", param_hint="--call") from exc
            if not isinstance(args, dict):
                raise click.BadParameter("arguments must be a JSON object", param_hint="--call")
            with console.status(f"[bold green]Calling MCP tool {tool!r}..."):
                result = mcp_layer.call_tool(tool, args)
            click.echo(json.dumps(result, indent=2, default=str))
            if isinstance(result, dict) and result.get("isError"):
                raise click.ClickException(f"MCP tool {tool!r} returned an error")

        elif tool:
            info = next((item for item in mcp_layer.list_tools() if item.get("name") == tool), None)
            if info:
                click.echo(json.dumps(info, indent=2, default=str))
            else:
                raise click.ClickException(f"Tool {tool!r} not found")
        else:
            console.print("[dim]Use --list-tools to see available tools, or --tool + --call to invoke.[/dim]")

    except Exception as exc:
        if isinstance(exc, click.ClickException):
            raise
        raise click.ClickException(f"MCP error: {exc}") from exc
    finally:
        try:
            if mcp_layer is not None:
                mcp_layer.disconnect_all()
        finally:
            if rt is not None:
                rt.close()


@cli.command()
def version() -> None:
    """Show Aether version."""
    console.print(f"Aether Runtime {AETHER_VERSION}")


_LEVEL_STYLE: dict[str, tuple[str, str]] = {
    "parity_verified": ("✅", "green"),
    "runs": ("🟡", "yellow"),
    "known_incorrect": ("🔬", "red"),
    "unsupported": ("❌", "dim"),
}


@cli.command("models")
@click.option("--counts", "counts_only", is_flag=True, help="Print only the family counts.")
@click.option("--level", "level_filter", default=None, help="Filter by support level.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def models(counts_only: bool, level_filter: str | None, as_json: bool) -> None:
    """Show the graded model-family support matrix.

    The numbers printed here are the same ones published in README.md and
    SUPPORTED_MODELS.md: all three read one registry, so a documented count can
    never drift away from what the compiler actually implements.
    """
    from aether.core.model_families import (
        MODEL_FAMILIES,
        SupportLevel,
        family_counts,
        support_summary,
    )

    counts = family_counts()
    if level_filter is not None:
        try:
            wanted = SupportLevel(level_filter)
        except ValueError:
            valid = ", ".join(level.value for level in SupportLevel)
            raise click.BadParameter(f"unknown level {level_filter!r}; expected one of: {valid}") from None
        rows = [family for family in MODEL_FAMILIES if family.level is wanted]
    else:
        rows = list(MODEL_FAMILIES)

    if as_json:
        payload: dict[str, Any] = {"counts": counts, "summary": support_summary()}
        if not counts_only:
            payload["families"] = [
                {
                    "key": family.key,
                    "name": family.name,
                    "kind": family.kind.value,
                    "level": family.level.value,
                    "representative_models": list(family.representative_models),
                    "numerics": family.numerics,
                    "detection_keys": list(family.detection_keys),
                    "aether_family": family.aether_family,
                    "note": family.note,
                }
                for family in rows
            ]
        click.echo(json.dumps(payload))
        return

    console.print(f"[bold]Aether model-family support[/bold] — {AETHER_VERSION}\n")
    summary = Table(show_header=True, header_style="bold")
    summary.add_column("Level")
    summary.add_column("Families", justify="right")
    summary.add_column("Guarantee")
    summary.add_row("✅ parity_verified", str(counts["parity_verified"]),
                    "logits match the reference (~1e-6), prefill and decode")
    summary.add_row("🟡 runs", str(counts["runs"]),
                    "executes with round-trip tests, no per-logit gate")
    summary.add_row("🔬 known_incorrect", str(counts["known_incorrect"]),
                    "executes with measured divergence — do not rely on output")
    summary.add_row("❌ unsupported", str(counts["unsupported"]),
                    "refused at compile time rather than emitting a wrong artifact")
    summary.add_row("[bold]executable[/bold]", f"[bold]{counts['executable']}[/bold]",
                    "parity_verified + runs + known_incorrect")
    console.print(summary)
    console.print(
        f"\n{counts['total']} architecture families · "
        f"{counts['detection_keys']} detected name / class spellings · "
        f"{counts['decoder']} decoder, {counts['encoder']} encoder, "
        f"{counts['encoder_decoder']} encoder-decoder, "
        f"{counts['ssm_hybrid']} SSM/hybrid, {counts['multimodal']} multimodal\n"
    )
    if counts_only:
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("Family")
    table.add_column("Kind")
    table.add_column("Representative models")
    table.add_column("Distinguishing contract")
    for family in rows:
        mark, style = _LEVEL_STYLE.get(family.level.value, ("?", "white"))
        detail = family.numerics if family.level.value == "parity_verified" else (
            family.note or family.numerics
        )
        table.add_row(
            mark,
            f"[{style}]{_display_text(family.name)}[/{style}]",
            family.kind.value,
            _display_text(", ".join(family.representative_models[:3])),
            _display_text(detail),
        )
    console.print(table)


@cli.command("eval")
@click.argument("model")
@click.option("--domain", default="general", help="Evaluation domain.")
@click.option("--examples", type=click.IntRange(1), default=100, help="Number of examples.")
@click.option("--threshold", type=_FiniteFloatRange(0.0, 1.0), default=0.98)
@click.option(
    "--dataset",
    "dataset_specs",
    multiple=True,
    help="Local benchmark dataset mapping BENCHMARK=PATH; repeat per benchmark.",
)
@click.option("--max-tokens", type=click.IntRange(1), default=256)
@click.option(
    "--allow-code-execution",
    is_flag=True,
    help="Allow explicit HumanEval subprocess execution for supplied datasets.",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    model: str,
    domain: str,
    examples: int,
    threshold: float,
    dataset_specs: tuple[str, ...],
    max_tokens: int,
    allow_code_execution: bool,
) -> None:
    """Run the runtime quality gate and fail if the configured threshold is missed."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        evaluator = None
        benchmarks = None
        if dataset_specs:
            datasets: dict[str, str] = {}
            for spec in dataset_specs:
                benchmark, separator, path = spec.partition("=")
                benchmark = benchmark.strip().lower()
                path = path.strip()
                if not separator or not benchmark or not path:
                    raise click.ClickException(
                        "--dataset must use BENCHMARK=PATH, for example --dataset mmlu=data.csv"
                    )
                if benchmark in datasets:
                    raise click.ClickException(f"duplicate dataset benchmark {benchmark!r}")
                datasets[benchmark] = path

            from aether.observability.ci_pipeline import DatasetBenchmarkEvaluator

            def generate_fn(*, prompt: str, benchmark: str, max_tokens: int) -> str:
                return rt.generate(
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                ).text

            evaluator = DatasetBenchmarkEvaluator(
                datasets,
                generate_fn,
                max_tokens=max_tokens,
                max_examples=examples,
                allow_code_execution=allow_code_execution,
            )
            # The module also exposes the ``aether list`` command, so its global
            # name shadows Python's built-in list constructor.
            benchmarks = builtins.list(datasets)

        report = rt.eval_gate(
            model,
            domain=domain,
            benchmarks=benchmarks,
            num_examples=examples,
            quality_threshold=threshold,
            evaluator=evaluator,
        )
        click.echo(json.dumps(report, indent=2, default=str))
        if not report.get("passed", False):
            raise click.ClickException("evaluation gate failed")


@cli.command("safety")
@click.argument("model")
@click.pass_context
def safety(ctx: click.Context, model: str) -> None:
    """Inspect persisted safety/provenance controls for an AEG artifact."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if path is None:
            raise click.ClickException(f"model {model!r} was not found")
        package = load_aeg_package(path)
        root = Path(path)
        safety_files = (
            sorted(str(p.relative_to(root)) for p in (root / "safety").rglob("*") if p.is_file())
            if (root / "safety").exists()
            else []
        )
        # Collect provenance data from the manifest's provenance field (always
        # populated — defaults to an empty ProvenanceInfo when not recorded).
        prov_dict: dict = {}
        if package.manifest:
            prov_obj = getattr(package.manifest, "provenance", None)
            if prov_obj is not None:
                prov_dict = prov_obj.to_dict() if callable(getattr(prov_obj, "to_dict", None)) else {}
        # Supplement with any raw provenance/manifest.json that may exist.
        prov_json_path = root / "provenance" / "manifest.json"
        if prov_json_path.is_file():
            try:
                raw_prov = json.loads(prov_json_path.read_text(encoding="utf-8"))
                prov_dict.update(raw_prov)
            except Exception:  # noqa: BLE001
                pass
        # Verify integrity flag: True when the manifest hash is present and valid.
        integrity_verified = False
        if package.manifest and package.manifest.manifest_hash:
            try:
                package.verify_integrity()
                integrity_verified = True
            except Exception:  # noqa: BLE001
                integrity_verified = False
        report = {
            "model_id": model,
            "integrity_verified": integrity_verified,
            "safety_files": safety_files,
            "provenance": prov_dict,
        }
        click.echo(json.dumps(report, indent=2, default=str))


@cli.command("trace")
@click.argument("model", required=False)
@click.option("--prompt", default="", help="Optional prompt to trace a real request.")
@click.pass_context
def trace(ctx: click.Context, model: str | None, prompt: str) -> None:
    """Export measured runtime spans as OTLP-compatible JSON."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        if prompt and not model:
            raise click.UsageError("--prompt requires an AEG model")
        if model and not prompt:
            raise click.UsageError("provide --prompt to trace a real model request")
        if model and prompt:
            rt.generate(model, prompt, max_tokens=1, temperature=0.0)
        payload = rt.tracer.export_otlp_json()
        if not rt.tracer.get_finished_spans():
            # Empty telemetry is valid when no request was executed, but it must
            # be distinguishable from a successful measured trace export.
            payload["status"] = "no_measured_requests"
            payload["message"] = (
                "No runtime spans were recorded. Provide an AEG model and --prompt "
                "to execute and trace a real request."
            )
        click.echo(json.dumps(payload, indent=2, default=str))


@cli.command("reasoning")
@click.argument("model")
@click.pass_context
def reasoning(ctx: click.Context, model: str) -> None:
    """Inspect the persisted reasoning graph for an AEG artifact."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if path is None:
            raise click.ClickException(f"model {model!r} was not found")
        root = Path(path)
        candidates = [
            root / "reasoning" / "reasoning_graph.json",
            root / "reasoning_graph.json",
            root / "graph" / "reasoning_graph.aeg-ir",
        ]
        graph_path = next((p for p in candidates if p.is_file()), None)
        if graph_path is None:
            raise click.ClickException(
                f"model {model!r} has no persisted reasoning graph; compile with "
                "reasoning-graph enabled"
            )
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise click.ClickException(f"reasoning graph for {model!r} is empty or malformed")
        if data.get("entrypoint") == "disabled" or data.get("enabled") is False:
            raise click.ClickException(f"reasoning graph for {model!r} is disabled")
        click.echo(json.dumps(data, indent=2, default=str))


@cli.command("mla-stats")
@click.argument("model")
@click.pass_context
def mla_stats(ctx: click.Context, model: str) -> None:
    """Show MLA/attention metadata persisted in the AEG manifest."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        info_data = rt.info(model)
        architecture = info_data.get("architecture", {})
        result = {"model_id": model, "mla_detected": bool(architecture.get("mla_enabled") or architecture.get("mla")), "architecture": architecture}
        click.echo(json.dumps(result, indent=2, default=str))


@cli.command("merge-info")
@click.argument("model")
@click.pass_context
def merge_info(ctx: click.Context, model: str) -> None:
    """Inspect the persisted task-vector merge metadata for an AEG artifact."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        try:
            path = rt._resolve_aeg_path(model)  # noqa: SLF001
            if path is None:
                raise click.ClickException(f"model {model!r} was not found")
            package = load_aeg_package(path)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc
        metadata = package.metadata
        merge = metadata.get("merge") or metadata.get("model_merging")
        if not isinstance(merge, dict):
            raise click.ClickException(
                f"model {model!r} has no persisted model-merging metadata; compile it with merging enabled"
            )
        click.echo(json.dumps({"model_id": model, "merge": merge}, indent=2, default=str))


@cli.group("runtime")
def runtime_commands() -> None:
    """Runtime-session controls."""


@runtime_commands.command("reweight")
@click.argument("model")
@click.option("--task", "task_specs", multiple=True, help="Task vector weight as NAME=VALUE; repeat per task.")
@click.option("--task1", type=float, default=None, help="Compatibility alias for task1.")
@click.option("--task2", type=float, default=None, help="Compatibility alias for task2.")
@click.option("--task3", type=float, default=None, help="Compatibility alias for task3.")
@click.option("--task4", type=float, default=None, help="Compatibility alias for task4.")
@click.pass_context
def runtime_reweight(
    ctx: click.Context,
    model: str,
    task_specs: tuple[str, ...],
    task1: float | None,
    task2: float | None,
    task3: float | None,
    task4: float | None,
) -> None:
    """Apply real persisted task-vector weights to a merged AEG session."""
    weights: dict[str, float] = {}
    for spec in task_specs:
        name, separator, raw_value = spec.partition("=")
        if not separator or not name.strip():
            raise click.ClickException("--task must use NAME=VALUE")
        try:
            weights[name.strip()] = float(raw_value)
        except ValueError as exc:
            raise click.ClickException(f"invalid task weight {spec!r}") from exc
        if not math.isfinite(weights[name.strip()]):
            raise click.ClickException(f"task weight {spec!r} must be finite")
    for name, value in (("task1", task1), ("task2", task2), ("task3", task3), ("task4", task4)):
        if value is not None:
            if not math.isfinite(value):
                raise click.ClickException(f"{name} weight must be finite")
            weights[name] = value
    if not weights:
        raise click.UsageError("provide at least one --task NAME=VALUE or --task1/--task2 weight")
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        try:
            normalized = rt.set_model_task_weights(model, weights)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps({"model_id": model, "weights": normalized}, indent=2))


@cli.command("kv-share")
@click.argument("model")
@click.option("--output", "output_path", type=click.Path(), help="Output AEG path.")
@click.pass_context
def kv_share(ctx: click.Context, model: str, output_path: str | None) -> None:
    """Compile with cross-layer KV sharing enabled (Pass 15)."""
    config = CompilerConfig(cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(), enable_cross_layer_kv=True)
    result = Compiler(config).compile(model, output_path=output_path)
    _require_applied_pass(result, "cross_layer_kv_sharing", "cross-layer KV sharing")
    console.print(f"[bold green]Cross-layer KV AEG saved to[/bold green] {result.root}")


@cli.command("multi-agent")
@click.argument("multi_agent_args", nargs=-1)
@click.option("--agents", type=click.IntRange(1), default=4)
@click.option("--shared-prefix", default="")
@click.option("--coordination", type=click.Choice(["relay", "kvcomm", "droidspeak", "swarm"]), default="relay")
@click.pass_context
def multi_agent(
    ctx: click.Context,
    multi_agent_args: tuple[str, ...],
    agents: int,
    shared_prefix: str,
    coordination: str,
) -> None:
    """Create a real multi-agent KV coordinator session."""
    args = list(multi_agent_args)
    action = "session"
    model = None
    if args and args[0].lower() == "test":
        action = "test"
        args.pop(0)
        if not args:
            raise click.UsageError("multi-agent test requires <model.aeg>")
        model = args.pop(0)
    if args:
        raise click.UsageError(f"unexpected multi-agent arguments: {' '.join(args)}")
    with _cli_runtime(_runtime_config(ctx)) as rt:
        if action == "test" and rt._resolve_aeg_path(model) is None:  # noqa: SLF001
            raise click.ClickException(f"model {model!r} was not found")
        result = rt.multi_agent_session(
            models=[model] if model else [],
            coordination=coordination,
            agent_count=agents,
            shared_prefix=shared_prefix,
        )
        try:
            if action == "test":
                result.update(
                    {
                        "model_id": model,
                        "coordination": coordination,
                        "test": "local_kv_coordinator_initialization",
                        "tested": True,
                        "model_execution_tested": False,
                    }
                )
            click.echo(json.dumps(result, indent=2, default=str))
        finally:
            result.close()


@cli.command("slo-status")
@click.pass_context
def slo_status(ctx: click.Context) -> None:
    """Show the live scheduler queue and configured SLO mode."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        report = {"scheduler": rt.config.scheduler, "queue": rt.scheduler.queue_snapshot(), "max_batch_size": rt.scheduler.max_batch_size}
        click.echo(json.dumps(report, indent=2, default=str))


@cli.command("green-route")
@click.option("--regions", multiple=True, required=True, help="Candidate deployment regions; repeat for multiple regions.")
@click.option("--latency-deadline", type=_FiniteFloatRange(min=0.0), default=1.0, show_default=True)
def green_route(regions: tuple[str, ...], latency_deadline: float) -> None:
    """Select the lowest-carbon candidate region within the latency deadline."""
    from aether.runtime.r7_green_power_manager import GreenPowerManager

    manager = GreenPowerManager()
    selected = manager.select_region(list(regions), latency_deadline_s=latency_deadline)
    click.echo(json.dumps(
                {
                    "selected_region": selected,
                    "candidate_regions": list(regions),
                    "latency_deadline_s": latency_deadline,
                    "policy": "lowest_carbon_within_latency_deadline",
                    "telemetry_source": "compiled/static_region_profile",
                },
                indent=2,
            ))


@cli.group("slo-profile")
def slo_profile() -> None:
    """Manage named scheduler SLO profiles."""


@slo_profile.command("add")
@click.argument("name")
@click.option("--ttft", type=_FiniteFloatRange(min=0.0), required=True, help="Maximum TTFT in milliseconds.")
@click.option("--tbt", type=_FiniteFloatRange(min=0.0), required=True, help="Maximum TBT in milliseconds.")
@click.pass_context
def slo_profile_add(ctx: click.Context, name: str, ttft: float, tbt: float) -> None:
    """Persist a named SLO profile in the Aether cache configuration."""
    cache_dir = aether_cache_dir(ctx.obj.get("cache_dir"))
    profile_path = cache_dir / "slo_profiles.json"
    profiles: dict[str, Any] = {}
    if profile_path.is_file():
        try:
            loaded = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"invalid SLO profile file {profile_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise click.ClickException(f"SLO profile file {profile_path} must contain a JSON object")
        profiles = loaded
    profiles[name] = {"max_ttft_ms": ttft, "max_tbt_ms": tbt}
    cache_dir.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")
    click.echo(json.dumps(
                {"name": name, "profile": profiles[name], "persisted_to": str(profile_path)},
                indent=2,
            ))


@cli.group("hub")
@click.option("--url", envvar="AETHER_HUB_URL", default=DEFAULT_HUB_URL, show_default=True, help="Aether Hub service URL.")
@click.pass_context
def hub(ctx: click.Context, url: str) -> None:
    """Authenticate and exchange AEG artifacts with Aether Hub."""
    ctx.obj["hub_url"] = url


def _hub_client(ctx: click.Context) -> HubClient:
    token_path = aether_cache_dir(ctx.obj.get("cache_dir")) / "hub_token"
    token = token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else None
    return HubClient(hub_url=ctx.obj.get("hub_url", DEFAULT_HUB_URL), auth_token=token, allow_local_cache=False)


@hub.command("login")
@click.option("--token", prompt=True, hide_input=True)
@click.pass_context
def hub_login(ctx: click.Context, token: str) -> None:
    if not token.strip():
        raise click.BadParameter("must not be empty", param_hint="--token")
    result = _hub_client(ctx).login(token)
    token_path = aether_cache_dir(ctx.obj.get("cache_dir")) / "hub_token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    click.echo(json.dumps(result, indent=2))


@hub.command("search")
@click.argument("query")
@click.pass_context
def hub_search(ctx: click.Context, query: str) -> None:
    click.echo(json.dumps([m.to_dict() for m in _hub_client(ctx).search(query)], indent=2))


@hub.command("push")
@click.argument("model_id")
@click.argument("package", type=click.Path(exists=True))
@click.pass_context
def hub_push(ctx: click.Context, model_id: str, package: str) -> None:
    manifest = _hub_client(ctx).upload(model_id, package)
    click.echo(json.dumps(manifest.to_dict(), indent=2))


@hub.command("pull")
@click.argument("model_id")
@click.argument("output", type=click.Path())
@click.pass_context
def hub_pull(ctx: click.Context, model_id: str, output: str) -> None:
    path = _hub_client(ctx).download(model_id, output)
    console.print(str(path))


@cli.group("train")
def train() -> None:
    """Training and RLVR commands."""


@train.command("grpo")
@click.argument("model")
@click.argument("prompts", nargs=-1, required=True)
@click.option("--group-size", type=click.IntRange(2), default=8)
@click.pass_context
def train_grpo(ctx: click.Context, model: str, prompts: tuple[str, ...], group_size: int) -> None:
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        report = rt.grpo_train_step(model, list(prompts), group_size=group_size)
        click.echo(json.dumps(report, indent=2, default=str))
        if report.get("status") == "failed":
            raise click.ClickException(report.get("error", "GRPO step failed"))


@train.command("verify")
@click.argument("model", required=False)
@click.option("--domain", default="math", show_default=True)
@click.option("--example", "example_text", required=True, help="Response text to verify.")
@click.option("--ground-truth", default=None)
@click.option("--test-code", default=None, help="Optional code used by the code verifier.")
@click.pass_context
def train_verify(
    ctx: click.Context,
    model: str | None,
    domain: str,
    example_text: str,
    ground_truth: str | None,
    test_code: str | None,
) -> None:
    """Verify one response with the deterministic RLVR verifier.

    ``model`` is accepted for PRD CLI compatibility and is reported in the
    result, but verification is deliberately performed by the rule-based
    verifier rather than pretending to run a training step.
    """
    from aether.compiler.stage2_optimizer.pass22_rlvr_verifier import GRPOTrainer

    if domain not in {"math", "code"}:
        raise click.ClickException(f"the {domain!r} domain has no deterministic verifier")
    if domain == "math" and ground_truth is None:
        raise click.UsageError("math verification requires --ground-truth")
    if domain == "code" and test_code is None:
        raise click.UsageError("code verification requires --test-code")
    try:
        reward = GRPOTrainer().verify_response(
            example_text,
            domain=domain,
            ground_truth=ground_truth,
            test_code=test_code,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(
                {
                    "status": "verified" if reward == 1.0 else "failed",
                    "verified": reward == 1.0,
                    "model": model,
                    "domain": domain,
                    "reward": reward,
                },
                indent=2,
                default=str,
            ))
    if reward != 1.0:
        ctx.exit(1)


@cli.group("kernel")
def kernel() -> None:
    """Target kernel planning commands."""


@kernel.command("generate")
@click.argument("kernel_args", nargs=-1)
@click.option("--target", "target_option", default=None, help="Hardware target (for example cpu_avx2 or cuda_sm90).")
@click.option("--output", type=click.Path())
def kernel_generate(kernel_args: tuple[str, ...], target_option: str | None, output: str | None) -> None:
    """Generate and verify an executable kernel artifact.

    CPU targets use Aether's native shared-library compiler. Vendor targets
    fail explicitly until their actual compiler/runtime integration is present.

    Both ``aether kernel generate <target> <op>`` (legacy) and the PRD form
    ``aether kernel generate <op> --target <target>`` are accepted.
    """
    from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
    from aether.core.exceptions import KernelError

    if target_option is not None:
        if len(kernel_args) != 1:
            raise click.UsageError("provide exactly one operation when --target is used")
        target = target_option
        op_name = kernel_args[0]
    elif len(kernel_args) == 2:
        target, op_name = kernel_args
    elif len(kernel_args) == 1:
        target = HardwareProfile.auto().target_id
        op_name = kernel_args[0]
    else:
        raise click.UsageError(
            "expected <op_name> --target <target> or legacy <target> <op_name>"
        )

    try:
        artifact = KernelEmitter(target).emit_executable(op_name, output)
    except KernelError as exc:
        # A missing vendor compiler/device is an expected capability failure,
        # not a Python traceback.  Keep the CLI contract machine-readable and
        # make the requested target failure explicit rather than claiming a
        # fallback kernel was produced.
        raise click.ClickException(str(exc)) from exc
    payload = json.dumps(artifact.to_dict(), indent=2)
    click.echo(payload)


@kernel.command("verify")
@click.argument("kernel_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--reference-op", required=True, help="Operation symbol that must be exported by the kernel.")
def kernel_verify(kernel_path: str, reference_op: str) -> None:
    """Verify an emitted native kernel file and its required exported symbol."""
    from aether.compiler.stage3_targeting.kernel_emitter import _cpu_symbol_for_op

    path = Path(kernel_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    suffix = path.suffix.lower()
    if suffix not in {".dll", ".so", ".dylib"}:
        raise click.ClickException(
            f"{path} is not a loadable native library (.dll, .so, or .dylib)"
        )
    try:
        symbol = _cpu_symbol_for_op(reference_op)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    try:
        import ctypes

        library = ctypes.CDLL(str(path))
        getattr(library, symbol)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(
            f"kernel failed native loading or does not export {symbol!r}: {exc}"
        ) from exc
    click.echo(json.dumps(
                {
                    "path": str(path),
                    "sha256": digest,
                    "reference_op": reference_op,
                    "exported_symbol": symbol,
                    "loadable": True,
                    "verified": True,
                },
                indent=2,
            ))


@cli.group("kv")
def kv() -> None:
    """KV infrastructure commands."""


@kv.command("transfer-stats")
@click.argument("model", required=False)
@click.pass_context
def kv_transfer_stats(ctx: click.Context, model: str | None) -> None:
    with _cli_runtime(_runtime_config(ctx)) as rt:
        if model and rt._resolve_aeg_path(model) is None:  # noqa: SLF001
            raise click.ClickException(f"model {model!r} was not found")
        report = rt.kv_transfer_stats()
        report["scope"] = "current_cli_process"
        report["model_id"] = model
        click.echo(json.dumps(report, indent=2, default=str))


@kv.command("nika-policy")
@click.option("--kv-size-gb", type=_FiniteFloatRange(min=0.0), required=True)
@click.option("--bw-gbps", type=_FiniteFloatRange(min=0.0, min_open=True), required=True)
@click.option("--decode-util", type=_FiniteFloatRange(0.0, 1.0), required=True)
def kv_nika_policy(kv_size_gb: float, bw_gbps: float, decode_util: float) -> None:
    """Evaluate the real NIKA transfer-versus-recompute policy."""
    from aether.runtime.r12_cxl_kv_pool import TraCTPolicy

    result = TraCTPolicy().nika_policy(kv_size_gb, bw_gbps, decode_util)
    result["execution_mode"] = "analytical_policy"
    result["physical_network_transfer"] = False
    click.echo(json.dumps(result, indent=2, default=str))


@kv.command("cxl-pool-status")
@click.pass_context
def kv_cxl_pool_status(ctx: click.Context) -> None:
    """Report the configured CXL KV pool and explicit local fallback state."""
    with _cli_runtime(_runtime_config(ctx)) as rt:
        click.echo(json.dumps(rt.cxl_pool_status(), indent=2, default=str))


@cli.command()
@click.argument("model")
@click.pass_context
def status(ctx: click.Context, model: str) -> None:
    """Inspect an artifact's presence and this CLI process's execution state."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        info_data = rt.info(model)
        click.echo(json.dumps({
            "model_id": info_data["model_id"],
            "artifact_available": True,
            "artifact_targets": info_data["targets"],
            "loaded": model in rt._loaded_models,  # noqa: SLF001
            "execution_tested": False,
            "scope": "current_cli_process",
            "message": "Artifact inspected; use run or serve to load and execute it.",
        }, indent=2, default=str))


@cli.command("quantize-report")
@click.argument("model")
@click.pass_context
def quantize_report(ctx: click.Context, model: str) -> None:
    """Show measured sub-2-bit quantization data for an AEG artifact."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        try:
            report = rt.quantization_report(model)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(json.dumps(report, indent=2, default=str))


@cli.group("cache")
def cache() -> None:
    """Inspect and control the semantic request cache."""


@cache.command("stats")
@click.pass_context
def cache_stats(ctx: click.Context) -> None:
    """Show measured semantic-cache statistics."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        click.echo(json.dumps(rt.semantic_cache_stats(), indent=2, default=str))


@cache.command("flush")
@click.pass_context
def cache_flush(ctx: click.Context) -> None:
    """Flush the semantic request cache and report the removed entry count."""
    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        rt._init_v5_layers()  # noqa: SLF001
        semantic_cache = getattr(rt, "_semantic_cache", None)  # noqa: SLF001
        if semantic_cache is None:
            raise click.ClickException("semantic cache is unavailable")
        console.print(json.dumps({"removed": semantic_cache.flush()}))


# ---------------------------------------------------------------------------
# aether doctor — full system diagnostics (PRD §42)
# ---------------------------------------------------------------------------

@cli.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of rich table.")
def doctor(as_json: bool) -> None:
    """Full system diagnostics: Python, dependencies, hardware, smoke test."""
    import importlib
    import subprocess
    import sys

    report: dict[str, Any] = {
        "aether_version": None,
        "python_version": sys.version,
        "platform": sys.platform,
        "checks": [],
    }

    def check(name: str, fn: Any) -> dict[str, Any]:
        try:
            result = fn()
            return {"name": name, "status": "pass", "detail": result}
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "status": "fail", "detail": str(exc)}

    # Aether version
    try:
        from aether.core.constants import AETHER_VERSION
        report["aether_version"] = AETHER_VERSION
    except ImportError:
        report["aether_version"] = "unknown"

    # Core dependencies.  Torch is an optional frontend/backend dependency in
    # pyproject.toml; the framework-free CPU/AEG path must remain healthy when
    # it is absent, so diagnose it separately instead of failing the install.
    for pkg in ["numpy", "click", "rich", "safetensors"]:
        def _check_import(p: str = pkg) -> str:
            mod = importlib.import_module(p)
            if p == "click":
                from importlib.metadata import version as package_version
                return package_version("click")
            return getattr(mod, "__version__", "installed")
        report["checks"].append(check(f"import:{pkg}", _check_import))

    try:
        torch_version = importlib.import_module("torch")
        report["checks"].append({
            "name": "optional:torch",
            "status": "pass",
            "detail": getattr(torch_version, "__version__", "installed"),
        })
    except ImportError:
        report["checks"].append({
            "name": "optional:torch",
            # Torch is an optional frontend/backend. Its absence is healthy
            # for the framework-free CPU/AEG product path and must not make a
            # clean CPU installation look broken.
            "status": "pass",
            "detail": "not installed; optional aether-runtime[pytorch] extra is available",
        })

    # Hardware detection
    try:
        from aether.backends.hardware_detector import detect_all_capabilities, validate_backend_environment
        caps = detect_all_capabilities()
        report["hardware"] = [c.to_dict() for c in caps]
        # CPU is always available — validate it
        cpu_val = validate_backend_environment("cpu")
        required_cpu_failures = [
            item for item in cpu_val.checks_failed
            if "pytorch" not in item.lower()
        ]
        report["checks"].append({
            "name": "hardware:cpu",
            "status": (
                "fail" if required_cpu_failures or not cpu_val.available
                else "warn" if cpu_val.warnings else "pass"
            ),
            "detail": {
                "passed": cpu_val.checks_passed,
                "failed": cpu_val.checks_failed,
                "optional_failures": [
                    item for item in cpu_val.checks_failed
                    if item not in required_cpu_failures
                ],
                "warnings": cpu_val.warnings,
                "execution_tested": cpu_val.execution_tested,
            },
        })
    except Exception as exc:  # noqa: BLE001
        report["checks"].append({"name": "hardware:detect", "status": "fail", "detail": str(exc)})

    # Smoke test: import and instantiate Runtime
    def _smoke() -> str:
        from aether import Runtime, RuntimeConfig
        with _cli_runtime(RuntimeConfig(hf_offline=True)) as rt:
            # Verify the runtime object was created properly without accessing private state
            assert rt is not None
            return f"Runtime instantiated OK (class={type(rt).__name__})"
    report["checks"].append(check("smoke:runtime_init", _smoke))

    # Native CPU kernels
    def _native_cpu() -> str | dict[str, Any]:
        from aether.kernels.native_cpu import get_native_kernels
        nk = get_native_kernels()
        if nk.ensure_compiled():
            return f"kernels={nk.available_kernels()}"
        return {
            "build_failed": nk.build_error,
            "fallback": "NumPy reference execution remains available; a C++ compiler is optional",
        }
    native_check = check("kernels:native_cpu", _native_cpu)
    if native_check["status"] == "fail":
        native_check["detail"] = {
            "build_failed": native_check["detail"],
            "fallback": "NumPy reference execution remains available; a C++ compiler is optional",
        }
    if isinstance(native_check.get("detail"), dict) and "build_failed" in native_check["detail"]:
        native_check["status"] = "warn"
    report["checks"].append(native_check)

    # Public package surface.  A clean installation is not healthy if the
    # documented top-level compiler/runtime entry points cannot be imported.
    def _public_api() -> str:
        from aether import AetherClient, AetherHub, Compiler, Runtime

        exports = (AetherClient, AetherHub, Compiler, Runtime)
        if any(value is None for value in exports):
            raise RuntimeError("one or more public API exports resolved to None")
        return "Runtime, Compiler, AetherClient, and AetherHub import successfully"

    report["checks"].append(check("api:public_exports", _public_api))

    # Hardware validation matrix file
    def _hw_matrix() -> str:
        # cli.py is at src/aether/cli.py — 3 parents = repo root
        candidates = [
            Path(__file__).parent.parent.parent / "hardware_validation_matrix.json",
            Path.cwd() / "hardware_validation_matrix.json",
        ]
        for p in candidates:
            if p.is_file():
                return f"found at {p}"
        return "not found (run: aether hardware detect --save hardware_validation_matrix.json)"
    report["checks"].append(check("file:hardware_validation_matrix", _hw_matrix))


    # Summarize
    passed = sum(1 for c in report["checks"] if c["status"] == "pass")
    failed = sum(1 for c in report["checks"] if c["status"] == "fail")
    warned = sum(1 for c in report["checks"] if c["status"] == "warn")
    report["summary"] = {
        "passed": passed, "failed": failed, "warnings": warned, "total": len(report["checks"])
    }

    if as_json:
        click.echo(json.dumps(report, indent=2, default=str))
        return

    # Rich table output
    table = Table(title="Aether Doctor Report")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail", style="dim")
    for c in report["checks"]:
        status_str = {"pass": "[green]PASS[/green]", "fail": "[red]FAIL[/red]",
                      "warn": "[yellow]WARN[/yellow]"}.get(c["status"], c["status"])
        detail = str(c.get("detail", ""))[:80]
        table.add_row(c["name"], status_str, detail)
    console.print(table)
    console.print(f"\n[bold]Hardware:[/bold]")
    hw_list = report.get("hardware", [])
    for h in hw_list[:6]:
        avail = "[green]YES[/green]" if h["available"] else "[red]NO[/red]"
        console.print(f"  {avail} {h['target_id']:20s} {h['vendor']} {h['device']}")
    if len(hw_list) > 6:
        console.print(f"  [dim]... and {len(hw_list) - 6} more (use aether hardware detect)[/dim]")
    console.print(f"\n[bold]Summary:[/bold] {passed}/{len(report['checks'])} checks passed", end="")
    if failed:
        console.print(f"  [red]{failed} failed[/red]")
    elif warned:
        console.print(f"  [yellow]{warned} warning(s); see fallback/environment details above[/yellow]")
    else:
        console.print("  [green]All OK[/green]")


# ---------------------------------------------------------------------------
# aether hardware — hardware detection and validation (PRD §41, §42)
# ---------------------------------------------------------------------------

def _hardware_status(value: bool | None, *, unchecked: str = "UNKNOWN") -> str:
    if value is None:
        return f"[dim]{unchecked}[/dim]"
    return "[green]YES[/green]" if value else "[red]NO[/red]"


def _hardware_version(value: str | None) -> str:
    if not value or value.lower().startswith("unknown"):
        return "unknown / not measured"
    return value


@cli.group("hardware", invoke_without_command=True)
@click.option("--json", "as_json", is_flag=True, help="Output the device fingerprint as JSON.")
@click.pass_context
def hardware(ctx: click.Context, as_json: bool) -> None:
    """Hardware detection, capability reporting, and backend validation."""
    if ctx.invoked_subcommand is None:
        _print_hardware_fingerprint()


@hardware.command("detect")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option(
    "--save", "save_path", type=click.Path(dir_okay=False, path_type=Path),
    help="Write the detection snapshot to this explicit file path.",
)
def hardware_detect(as_json: bool, save_path: Path | None) -> None:
    """Detect device/runtime capabilities without claiming successful execution."""
    from aether.backends.hardware_detector import detect_all_capabilities

    caps = detect_all_capabilities()
    data = {
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host_platform": sys.platform,
        "targets": [c.to_dict() for c in caps],
    }

    if save_path is not None:
        save_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        click.echo(f"Saved to {save_path}", err=True)

    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
        return

    table = Table(title="Hardware Capability Detection")
    table.add_column("Target ID", style="cyan")
    table.add_column("Device", style="white")
    table.add_column("Hardware")
    table.add_column("Runtime")
    table.add_column("Impl")
    table.add_column("Backend")
    table.add_column("Available", style="bold")
    table.add_column("Exec Tested", style="dim")
    for c in caps:
        table.add_row(
            c.target_id, c.device[:40], _hardware_status(c.hardware_present),
            _hardware_status(c.runtime_available), _hardware_status(c.implemented),
            _hardware_status(c.backend_available, unchecked="NOT CHECKED"),
            _hardware_status(c.available),
            "[green]YES[/green]" if c.execution_tested else "[dim]NOT TESTED[/dim]",
        )
    console.print(table)
    console.print(f"\n[dim]{len([c for c in caps if c.available])} available, "
                  f"{len([c for c in caps if not c.available])} unavailable[/dim]")
    console.print(
        "[dim]Available means detected device + vendor runtime. Backend readiness is separate. "
        "Detection does not execute inference; use hardware validate TARGET --execute.[/dim]"
    )


@hardware.command("capabilities")
@click.argument("target_id", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def hardware_capabilities(target_id: str | None, as_json: bool) -> None:
    """Show detailed capability matrix for a target (or all if no target given)."""
    from aether.backends.hardware_detector import detect_all_capabilities

    caps = detect_all_capabilities()
    if target_id:
        caps = [c for c in caps if c.target_id == target_id or c.vendor.lower() == target_id.lower()]
        if not caps:
            raise click.ClickException(f"No hardware target found for {target_id!r}")

    if as_json:
        click.echo(json.dumps([c.to_dict() for c in caps], indent=2, default=str))
        return

    for c in caps:
        table = Table(title=f"{c.target_id} — {c.vendor} {c.device}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("vendor", c.vendor)
        table.add_row("device", c.device)
        table.add_row("architecture", c.architecture)
        table.add_row("driver_version", _hardware_version(c.driver_version))
        table.add_row("runtime_version", _hardware_version(c.runtime_version))
        table.add_row("memory_bytes", f"{c.memory_bytes:,}" if c.memory_bytes > 0 else "unknown / not measured")
        runtime_formats = c.extra.get("precision_scope") == "runtime_formats"
        precision_label = "runtime_format" if runtime_formats else "native"
        table.add_row("precision_semantics", c.to_dict()["precision_semantics"])
        for precision in ("fp32", "fp16", "bf16", "fp8", "fp4", "int8", "int4"):
            confirmed = getattr(c, f"supports_{precision}")
            table.add_row(
                f"{precision_label}_{precision}", "confirmed" if confirmed else "not confirmed",
            )
        precision_execution = c.extra.get("precision_execution")
        if precision_execution:
            table.add_row("Aether precision paths", json.dumps(precision_execution, default=str))
        table.add_row("supports_tee", str(c.supports_tee))
        table.add_row("hardware_present", _hardware_status(c.hardware_present))
        table.add_row("runtime_available", _hardware_status(c.runtime_available))
        table.add_row("implemented", _hardware_status(c.implemented))
        table.add_row("backend_available", _hardware_status(c.backend_available, unchecked="NOT CHECKED"))
        if c.backend_dependencies:
            table.add_row("backend_dependencies", json.dumps(c.backend_dependencies))
        table.add_row("available", _hardware_status(c.available))
        table.add_row("compile_tested", "YES" if c.compile_tested else "NOT TESTED")
        table.add_row("execution_tested", "[green]YES[/green]" if c.execution_tested else "[dim]NOT TESTED[/dim]")
        table.add_row("production_validated", "[green]YES[/green]" if c.production_validated else "[dim]NOT VALIDATED[/dim]")
        if c.unavailable_reason:
            table.add_row("unavailable_reason", f"[dim]{c.unavailable_reason}[/dim]")
        console.print(table)


@hardware.command("validate")
@click.argument("target_id", default="cpu")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option("--execute", is_flag=True, help="Execute and check a real minimal workload on the requested backend.")
@click.option("--backend", "backend_name", help="Select an execution backend for the workload probe.")
@click.option("--device-index", type=click.IntRange(min=0), default=0, show_default=True)
def hardware_validate(
    target_id: str, as_json: bool, execute: bool, backend_name: str | None, device_index: int,
) -> None:
    """Validate environment contracts; --execute additionally tests real execution."""
    from aether.backends.hardware_detector import validate_backend_environment

    result = validate_backend_environment(
        target_id, execute=execute, backend_name=backend_name, device_index=device_index,
    )
    failed = not result.all_passed or (execute and not result.execution_tested)
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        if failed:
            raise click.exceptions.Exit(1)
        return

    table = Table(title=f"Backend Validation: {target_id}")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    for c in result.checks_passed:
        table.add_row(c, "[green]PASS[/green]")
    for c in result.checks_failed:
        table.add_row(c, "[red]FAIL[/red]")
    for w in result.warnings:
        table.add_row(w, "[yellow]WARN[/yellow]")
    console.print(table)
    status = "[red]VALIDATION FAILED[/red]" if failed else "[green]ENVIRONMENT READY[/green]"
    console.print(f"\nBackend {target_id!r}: {status}")
    if result.execution_tested:
        console.print(f"[green]Execution succeeded[/green] (scope: {result.execution_scope})")
    else:
        console.print("[dim]Execution: NOT TESTED. Use --execute for a real workload probe.[/dim]")
    console.print("[dim]Production validation: NOT VALIDATED by this command.[/dim]")
    if failed:
        raise click.ClickException(f"Backend {target_id!r} validation failed")


# ---------------------------------------------------------------------------
# aether backend — backend management (PRD §6)
# ---------------------------------------------------------------------------

@cli.group("backend")
def backend_group() -> None:
    """Backend management commands."""


@backend_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def backend_list(as_json: bool) -> None:
    """List hardware pipelines and separately report execution backend readiness."""
    from aether.backends.hardware_detector import detect_all_capabilities

    caps = detect_all_capabilities()
    if as_json:
        click.echo(json.dumps([c.to_dict() for c in caps], indent=2, default=str))
        return

    table = Table(title="Hardware Pipelines and Aether Backend Readiness")
    table.add_column("Target ID", style="cyan")
    table.add_column("Vendor", style="magenta")
    table.add_column("Device", style="white")
    table.add_column("Impl", style="dim")
    table.add_column("Runtime")
    table.add_column("Backend")
    table.add_column("Available", style="bold")
    table.add_column("Exec Tested", style="dim")
    for c in caps:
        table.add_row(
            c.target_id, c.vendor, c.device[:30], _hardware_status(c.implemented),
            _hardware_status(c.runtime_available),
            _hardware_status(c.backend_available, unchecked="NOT CHECKED"),
            _hardware_status(c.available),
            "[green]YES[/green]" if c.execution_tested else "[dim]NOT TESTED[/dim]",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# aether inspect — deep AEG inspection (PRD §44)
# ---------------------------------------------------------------------------

@cli.command("inspect")
@click.argument("aeg_path")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option("--mtp", is_flag=True, help="Inspect the persisted native MTP head artifact.")
@click.pass_context
def inspect_aeg(ctx: click.Context, aeg_path: str, as_json: bool, mtp: bool) -> None:
    """Deep inspection of a compiled AEG artifact."""
    from aether.core.aeg_format import load_aeg_package

    with _cli_runtime(_runtime_config(ctx, hf_offline=True)) as rt:
        resolved = rt._resolve_aeg_path(aeg_path)  # noqa: SLF001
        if resolved is None:
            raise click.ClickException(f"model {aeg_path!r} was not found")
        pkg = load_aeg_package(resolved)
        root = Path(resolved)
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

        report: dict[str, Any] = {
            "path": str(root),
            "model_id": pkg.manifest.model_id if pkg.manifest else aeg_path,
            "manifest": pkg.manifest.to_dict() if pkg.manifest else {},
            "files": files,
            "file_count": len(files),
        }
        if mtp:
            mtp_path = root / "speculation" / "mtp_config.json"
            if not mtp_path.is_file():
                raise click.ClickException(f"model {aeg_path!r} has no persisted MTP head; compile with --mtp")
            payload = json.loads(mtp_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise click.ClickException("MTP metadata must be a JSON object")
            report["mtp"] = payload

        if as_json or mtp:
            click.echo(json.dumps(report, indent=2, default=str))
            return

        console.print(f"[bold]AEG Artifact: {aeg_path}[/bold]")
        if pkg.manifest:
            m = pkg.manifest
            table = Table(title="Manifest")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            d = m.to_dict()
            for k, v in d.items():
                if isinstance(v, dict):
                    table.add_row(k, json.dumps(v, default=str)[:80])
                else:
                    table.add_row(k, str(v)[:80])
            console.print(table)
        console.print(f"\n[dim]{len(files)} files in artifact:[/dim]")
        for f in files[:20]:
            console.print(f"  {f}")
        if len(files) > 20:
            console.print(f"  [dim]... and {len(files) - 20} more[/dim]")


# ---------------------------------------------------------------------------
# aether sign / aether verify — C2PA Content Credentials
# ---------------------------------------------------------------------------

def _load_trust_anchors(paths: tuple[str, ...]) -> list[bytes]:
    """Read DER or PEM trust anchors from disk."""
    anchors: list[bytes] = []
    for entry in paths:
        raw = Path(entry).read_bytes()
        if b"-----BEGIN CERTIFICATE-----" in raw:
            import base64

            for block in raw.split(b"-----BEGIN CERTIFICATE-----")[1:]:
                body = block.split(b"-----END CERTIFICATE-----")[0]
                anchors.append(base64.b64decode(b"".join(body.split())))
        else:
            anchors.append(raw)
    return anchors


@cli.command("sign")
@click.argument("aeg_path", type=click.Path(exists=True, file_okay=False))
@click.option("--key", "key_path", type=click.Path(), default=None,
              help="Ed25519 signing seed (32 raw bytes or hex). Generated on first use.")
@click.option("--cert", "cert_paths", multiple=True, type=click.Path(exists=True),
              help="DER/PEM certificate chain, leaf first. Self-signed if omitted.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def sign_aeg(aeg_path: str, key_path: str | None, cert_paths: tuple[str, ...], as_json: bool) -> None:
    """Sign an AEG package with C2PA Content Credentials.

    Writes a JUMBF manifest store to provenance/c2pa.manifest containing a
    deterministic-CBOR claim, a collection hard binding over every file in the
    package, and a detached COSE_Sign1 claim signature.
    """
    from aether.provenance.manifest import attach_c2pa_manifest

    chain = _load_trust_anchors(cert_paths) if cert_paths else None
    try:
        store_path, manifest = attach_c2pa_manifest(
            aeg_path, key_path=key_path, certificate_chain=chain
        )
    except Exception as exc:  # noqa: BLE001 - surface the reason, not a traceback
        console.print(f"[red]Signing failed: {exc}[/red]")
        raise SystemExit(1) from exc

    payload = {
        "manifest_store": str(store_path),
        "manifest_label": manifest.c2pa_manifest_label,
        "signature_algorithm": manifest.c2pa_signature_algorithm,
        "signer": manifest.c2pa_signer,
        "files_bound": manifest.c2pa_files_bound,
        "self_signed": not cert_paths,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[green]Signed[/green] {aeg_path}")
    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(key, _display_text(str(value)))
    console.print(table)
    if not cert_paths:
        console.print(
            "[yellow]Self-signed:[/yellow] this signature proves the artifact is "
            "unmodified. It does not establish who produced it — pass --cert with a "
            "chain from a real CA for that."
        )



@cli.command("verify")
@click.argument("aeg_path", type=click.Path(exists=True, file_okay=False))
@click.option("--trust-anchor", "anchor_paths", multiple=True, type=click.Path(exists=True),
              help="DER/PEM trust anchor. Without one, identity is reported as unestablished.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def verify_aeg(aeg_path: str, anchor_paths: tuple[str, ...], as_json: bool) -> None:
    """Verify an AEG package's C2PA Content Credentials.

    Checks four things independently: the manifest structure, the COSE_Sign1
    claim signature, every assertion hash referenced by the claim, and the hard
    binding against the files on disk. Exits non-zero if integrity fails.
    """
    from aether.provenance.manifest import verify_c2pa_manifest

    anchors = _load_trust_anchors(anchor_paths) if anchor_paths else None
    result = verify_c2pa_manifest(aeg_path, trust_anchors=anchors)
    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        raise SystemExit(0 if result.integrity_valid else 1)

    checks = [
        ("manifest present", result.manifest_present),
        ("structure valid", result.structure_valid),
        ("claim signature valid", result.claim_signature_valid),
        ("assertion hashes valid", result.assertions_valid),
        ("file binding valid", result.binding_valid),
    ]
    table = Table(title=f"C2PA verification — {aeg_path}")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    for name, passed in checks:
        table.add_row(name, "[green]pass[/green]" if passed else "[red]FAIL[/red]")
    console.print(table)

    if result.signature is not None and result.signature.certificate is not None:
        certificate = result.signature.certificate
        console.print(f"Algorithm: {result.signature.algorithm_name}")
        console.print(f"Signer:    {_display_text(certificate.subject)}")
        console.print(f"Validity:  {certificate.not_before} -> {certificate.not_after}")
    console.print(f"Files bound: {result.files_bound}")
    for label, entries in (
        ("changed", result.changed_files),
        ("missing", result.missing_files),
    ):
        if entries:
            console.print(f"[red]{label}:[/red] {', '.join(entries[:10])}")
    if result.extra_files:
        console.print(
            f"[dim]not covered by the binding ({len(result.extra_files)}): "
            f"{', '.join(result.extra_files[:5])}[/dim]"
        )
    for error in result.errors:
        console.print(f"[red]*[/red] {_display_text(error)}")

    if result.integrity_valid:
        console.print("\n[green]Integrity verified.[/green]")
        if result.trusted:
            console.print(f"[green]Trusted:[/green] {result.trust_reason}")
        else:
            console.print(f"[yellow]Identity not established:[/yellow] {result.trust_reason}")
    else:
        console.print("\n[red]Integrity check FAILED.[/red]")
    raise SystemExit(0 if result.integrity_valid else 1)


# ---------------------------------------------------------------------------
# aether plan — hardware-aware placement decision
# ---------------------------------------------------------------------------

@cli.command("plan")
@click.argument("aeg_path", type=click.Path(exists=True))
@click.option("--batch", type=click.IntRange(1), default=1, help="Target batch size.")
@click.option("--context", type=click.IntRange(1), default=2048, help="Target prompt length.")
@click.option("--generate", type=click.IntRange(1), default=256, help="Target output length.")
@click.option("--batch-floor", type=click.IntRange(1), default=None,
              help="Batch that must work or the plan is infeasible. Defaults to 1.")
@click.option("--context-floor", type=click.IntRange(1), default=None,
              help="Prompt length that must work. Defaults to min(512, --context).")
@click.option("--intent", type=click.Choice(["balanced", "latency", "throughput", "capacity"]),
              default="balanced", help="What to optimise among feasible plans.")
@click.option("--devices", default=None,
              help="Comma-separated device list to consider, e.g. 'cuda:0,cuda:1'.")
@click.option("--kappa", type=_FiniteFloatRange(0.1, 1.0), default=None,
              help="Device-memory ceiling fraction. The only percentage in the model.")
@click.option("--z", "z_value", type=_FiniteFloatRange(0.0, 6.0), default=None,
              help="One-sided quantile for the transient-peak margin. 3.0 for serving.")
@click.option("--no-probe", is_flag=True,
              help="Skip the bandwidth and dispatch micro-benchmarks and use priors.")
@click.option("--json", "as_json", is_flag=True, help="Emit the full decision as JSON.")
def plan_cmd(
    aeg_path: str, batch: int, context: int, generate: int,
    batch_floor: int | None, context_floor: int | None, intent: str,
    devices: str | None, kappa: float | None, z_value: float | None,
    no_probe: bool, as_json: bool,
) -> None:
    """Show where Aether would run this model, and why.

    Reads the artifact's manifest — no weights are loaded — measures this machine,
    enumerates every structurally admissible placement, filters by a conservative
    memory residual, ranks the survivors by a three-roof cost model, and prints the
    derivation. Exits non-zero when no plan is feasible, with the arithmetic and the
    changes that would make it feasible.
    """
    from aether.placement import (
        CalibrationLedger, ExecutionPlanner, Intent, WorkloadEnvelope, take_census,
    )
    from aether.placement.memory import KAPPA_DEFAULT, Z_DEFAULT
    from aether.placement.model_profile import profile_from_manifest
    from aether.placement.planner import PlacementInfeasible

    manifest_path = Path(aeg_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.is_file():
        raise click.ClickException(f"no AEG manifest at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile = profile_from_manifest(manifest)
    except (ValueError, OSError) as exc:
        raise click.ClickException(f"could not read {manifest_path}: {exc}") from exc

    envelope = WorkloadEnvelope(
        batch_floor=batch_floor or 1, batch_target=batch,
        context_floor=context_floor or min(512, context), context_target=context,
        generate_floor=min(64, generate), generate_target=generate,
        intent=Intent(intent),
    )
    ledger = CalibrationLedger()
    census = take_census(
        device_ids=[item.strip() for item in devices.split(",") if item.strip()] if devices else None,
        ledger=ledger, probe_bandwidth=not no_probe, probe_dispatch=not no_probe,
    )
    planner = ExecutionPlanner(
        profile, census, ledger,
        kappa=KAPPA_DEFAULT if kappa is None else kappa,
        z=Z_DEFAULT if z_value is None else z_value,
    )
    try:
        decision = planner.plan(envelope)
    except PlacementInfeasible as exc:
        if as_json:
            # Raw, unwrapped JSON: a --json flag exists to be piped, and Rich's
            # pretty-printer inserts soft wraps that break json.loads downstream.
            click.echo(json.dumps(
                {"feasible": False, "error": str(exc), **exc.detail},
                indent=2, default=str,
            ))
        else:
            console.print(f"[red]{_display_text(str(exc))}[/red]\n")
            console.print("[bold]What would make this feasible[/bold]")
            for remedy in exc.detail.get("remedies", []):
                console.print(f"  - {_display_text(remedy)}")
        raise SystemExit(2) from None

    if as_json:
        click.echo(json.dumps(decision.to_dict(), indent=2, default=str))
        return
    # The record is fixed-width by construction; Rich would re-wrap the columns.
    click.echo(_display_text(decision.render()))



@cli.command("benchmark")
@click.argument("model")
@click.option("--prompts", "-p", multiple=True, default=["Hello, tell me about yourself."],
              help="Prompts to benchmark. Repeat for multiple prompts.")
@click.option("--max-tokens", type=click.IntRange(min=1), default=64, help="Max tokens per run.")
@click.option("--runs", type=click.IntRange(min=1), default=5, help="Number of measured runs.")
@click.option("--warmup", type=click.IntRange(min=0), default=1, help="Number of warmup runs (not measured).")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save benchmark report to JSON file.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.pass_context
def benchmark_cmd(
    ctx: click.Context,
    model: str,
    prompts: tuple[str, ...],
    max_tokens: int,
    runs: int,
    warmup: int,
    output: str | None,
    as_json: bool,
) -> None:
    """Run real measured inference benchmarks (TTFT, TBT, TPS, P95 latency).

    All measurements come from actual inference runs — no hardcoded values.
    Results include hardware provenance so they are reproducible.
    """
    from aether.observability.benchmark_runner import BenchmarkRunner

    rt = Runtime(_runtime_config(ctx, hf_offline=True))
    try:

        def _generate(prompt: str, max_tok: int) -> str:
            return rt.generate(model, prompt, max_tokens=max_tok, temperature=0.0).text

        runner = BenchmarkRunner(
            generate_fn=_generate,
            model_id=model,
            num_warmup_runs=warmup,
            num_measured_runs=runs,
            precision="artifact",
            measurement_fn=lambda prompt, max_tok: rt.benchmark(model, prompt=prompt, max_tokens=max_tok),
        )

        with console.status(f"[bold green]Benchmarking {model} ({warmup} warmup + {runs} runs)..."):
            report = runner.run(list(prompts), max_tokens=max_tokens, output_path=output)

        summary = report.to_dict()["summary"]
        failed = summary.get("failed_runs", 0)
        if as_json:
            click.echo(json.dumps(report.to_dict(), indent=2, default=str))
            if failed:
                ctx.exit(1)
            return

        table = Table(title=f"Benchmark Results — {model}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for k, v in summary.items():
            vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
            table.add_row(k, vstr)
        console.print(table)
        if output:
            console.print(f"[dim]Full report saved to {output}[/dim]")
        if failed:
            console.print(f"[yellow]Warning: {failed} run(s) failed[/yellow]")
            ctx.exit(1)
    finally:
        rt.close()


def main() -> None:
    """CLI entry point for `aether` command."""
    cli(obj={})


# Preserve the short v3 command while exposing the PRD spelling.
cli.add_command(hardware, name="hw")


if __name__ == "__main__":
    main()

