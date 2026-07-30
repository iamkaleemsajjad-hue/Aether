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
import sys
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.table import Table
from rich.text import Text

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import load_aeg_package
from aether.core.constants import AETHER_VERSION, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from aether.utils.file_io import aether_cache_dir, delete_model
from aether.utils.logging import configure_logging

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--cache-dir", type=click.Path(), help="Custom Aether cache directory.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, cache_dir: str | None) -> None:
    """Aether Runtime — compile any AI model, run it on any hardware."""
    configure_logging(level="DEBUG" if verbose else "INFO")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["cache_dir"] = cache_dir


@cli.command()
@click.argument("model")
@click.option("--target", "-t", multiple=True, help="Hardware target(s). Repeat for multiple targets.")
@click.option("--quality-budget", type=float, default=0.02, help="Maximum perplexity increase budget.")
@click.option("--calibration-dataset", default="wikitext-2", help="Calibration dataset for sensitivity analysis.")
@click.option("--upload", is_flag=True, help="Upload compiled AEG to Aether Hub.")
@click.option("--output", "-o", type=click.Path(), help="Output path for the AEG artifact.")
@click.option("--dry-run", is_flag=True, help="Plan compilation without producing an AEG.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing AEG package.")
@click.pass_context
def compile(
    ctx: click.Context,
    model: str,
    target: tuple[str, ...],
    quality_budget: float,
    calibration_dataset: str,
    upload: bool,
    output: str | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Compile a model into an AEG artifact."""
    targets = list(target) if target else ["auto"]
    config = CompilerConfig(
        targets=targets,
        quality_budget=quality_budget,
        calibration_dataset=calibration_dataset,
        upload_kernels=upload,
        dry_run=dry_run,
        overwrite=overwrite,
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        verbose=ctx.obj.get("verbose", False),
    )
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
        for target, backend in plan.backend_recommendations.items():
            console.print(f"  Backend for {target}: {backend}")
        return

    with console.status(f"[bold green]Compiling {model}..."):
        aeg = compiler.compile(model, output_path=output)
    console.print(f"[bold green]Compiled AEG saved to[/bold green] {aeg.root}")


@cli.command()
@click.argument("model")
@click.option("--compile-local", is_flag=True, help="Force local compilation even if Hub has the AEG.")
@click.pass_context
def pull(ctx: click.Context, model: str, compile_local: bool) -> None:
    """Download and compile a model to the local AEG cache."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    with console.status(f"[bold green]Pulling {model}..."):
        rt.pull(model)
    console.print(f"[bold green]Model {model} ready.[/bold green]")


@cli.command()
@click.argument("model")
@click.option("--prompt", "-p", default="Hello, my name is", help="Prompt for generation.")
@click.option("--max-tokens", default=128, help="Maximum tokens to generate.")
@click.option("--temperature", default=0.7, help="Sampling temperature.")
@click.option("--top-p", default=0.9, help="Top-p sampling parameter.")
@click.option("--stream", is_flag=True, help="Stream output.")
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
    non_interactive: bool,
) -> None:
    """Run a model and generate text."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    with console.status(f"[bold green]Loading {model}..."):
        rt._load_model(model)  # noqa: SLF001

    if not non_interactive:
        console.print(f"[bold]Aether Runtime[/bold] — model: {model} (backend: {rt._loaded_backends[model].name})")  # noqa: SLF001

    with console.status("[bold green]Generating..."):
        response = rt.generate(
            model_id=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    console.print(f"[bold]Response:[/bold] {response.text}")
    console.print(f"[dim]Tokens: {response.usage}[/dim]")
    console.print(f"[dim]Metrics: {response.metrics.to_dict()}[/dim]")


@cli.command()
@click.argument("model", required=False)
@click.option("--port", default=DEFAULT_SERVER_PORT, help="Server port.")
@click.option("--host", default=DEFAULT_SERVER_HOST, help="Server host.")
@click.pass_context
def serve(ctx: click.Context, model: str | None, port: int, host: str) -> None:
    """Start the Aether REST server."""
    try:
        from aether.server.app import create_app
        import uvicorn
    except ImportError:
        console.print("[red]fastapi and uvicorn are required for the server.[/red]")
        sys.exit(1)

    config = RuntimeConfig(
        model_cache_dir=ctx.obj.get("cache_dir"),
        server_port=port,
        server_host=host,
    )
    app = create_app(config)
    console.print(f"[bold green]Starting Aether server at http://{host}:{port}[/bold green]")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("model")
@click.option("--compare", help="Compare against another backend (e.g., 'vllm').")
@click.option("--max-tokens", default=128, help="Maximum tokens per benchmark run.")
@click.pass_context
def bench(ctx: click.Context, model: str, compare: str | None, max_tokens: int) -> None:
    """Benchmark a model on the current hardware."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    with console.status(f"[bold green]Benchmarking {model}..."):
        result = rt.benchmark(model, max_tokens=max_tokens)
    table = Table(title=f"Benchmark: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)

    if compare:
        console.print(f"[dim]Comparison against {compare} is not yet implemented in this version.[/dim]")


@cli.command()
@click.argument("model")
@click.pass_context
def info(ctx: click.Context, model: str) -> None:
    """Show metadata and precision map for a compiled model."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    try:
        data = rt.info(model)
        console.print(RichJSON(json.dumps(data, indent=2, default=str)))
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")


@cli.command()
@click.argument("model")
@click.pass_context
def graph(ctx: click.Context, model: str) -> None:
    """Print the AEG-IR graph of a compiled model."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
    if aeg_path is None:
        console.print(f"[red]Model {model} not found.[/red]")
        return
    aeg = load_aeg_package(aeg_path)
    if aeg.ir:
        console.print(aeg.ir.to_text())
    else:
        console.print("[red]AEG has no IR loaded.[/red]")


@cli.command()
@click.pass_context
def list(ctx: click.Context) -> None:
    """List all compiled models in the local cache."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
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
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    rt.remove(model)
    console.print(f"[bold green]Removed {model}[/bold green]")


@cli.command()
def hw() -> None:
    """Show hardware fingerprint."""
    from aether.runtime.hardware import HardwareDetector
    fingerprint = HardwareDetector().detect()
    console.print(RichJSON(json.dumps(fingerprint.to_dict(), indent=2, default=str)))


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
@click.option("--lines", default=50, help="Number of lines to show.")
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
            for line in tailer.follow(open(latest, encoding="utf-8")):  # noqa: SIM115
                console.print(line, end="")
        except ImportError:
            console.print("[red]Install 'tailer' for follow mode.[/red]")
    else:
        content = latest.read_text(encoding="utf-8")
        for line in content.splitlines()[-lines:]:
            console.print(line)


@cli.command()
def version() -> None:
    """Show Aether version."""
    console.print(f"Aether Runtime {AETHER_VERSION}")


@cli.command()
@click.argument("model")
@click.pass_context
def status(ctx: click.Context, model: str) -> None:
    """Show runtime status for a loaded model."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    if model not in rt._loaded_models:  # noqa: SLF001
        console.print(f"[dim]Model {model} is not loaded.[/dim]")
        return
    backend = rt._loaded_backends[model].name  # noqa: SLF001
    console.print(f"[bold]{model}[/bold] loaded on backend: {backend}")


def main() -> None:
    """CLI entry point for `aether` command."""
    cli(obj={})


if __name__ == "__main__":
    main()
