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
@click.argument("model")
@click.option("--schema", type=click.Path(exists=True), help="Path to GBNF/LARK grammar schema file.")
@click.option("--mode", type=click.Choice(["gbnf", "lark", "regex"]), default="gbnf", help="Grammar format.")
@click.option("--dry-run", is_flag=True, help="Plan grammar constraint pass without recompiling.")
@click.pass_context
def grammar(ctx: click.Context, model: str, schema: str | None, mode: str, dry_run: bool) -> None:
    """Compile grammar constraints into the AEG artifact (Pass 11).

    Injects a GrammarConstraintCompilerPass FST/WFSA into the compiled model
    so token sampling is structurally constrained at inference time without
    any post-processing overhead.
    """
    if schema is None:
        console.print("[red]--schema is required to define the grammar.[/red]")
        console.print("[dim]Example: aether grammar gpt2 --schema ./json.gbnf --mode gbnf[/dim]")
        return
    schema_path = Path(schema)
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_grammar_constraint=True,
    )
    config.grammar_schema_path = str(schema_path)
    config.grammar_format = mode
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
    console.print(
        f"[bold green]Grammar-constrained AEG saved to[/bold green] {aeg.root}"
    )


@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--method", type=click.Choice(["ties", "dare", "linear", "slerp"]), default="ties",
              help="Model merging algorithm (PRD §12).")
@click.option("--output", "-o", type=click.Path(), help="Output path for merged AEG.")
@click.option("--weights", default="", help="Comma-separated merge coefficients (e.g. '0.6,0.4').")
@click.pass_context
def merge(ctx: click.Context, models: tuple[str, ...], method: str, output: str | None, weights: str) -> None:
    """Merge two or more compiled AEG models (Pass 12).

    Supports TIES (trimming + elect sign + merge), DARE (dropout +
    rescale), linear interpolation, and SLERP. The merged AEG can be
    served directly like any single-model AEG.

    Examples:
        aether merge model-a model-b --method ties
        aether merge model-a model-b model-c --method linear --weights 0.5,0.3,0.2
    """
    if len(models) < 2:
        console.print("[red]At least two models are required for merging.[/red]")
        return
    merge_weights: list[float] = []
    if weights:
        try:
            merge_weights = [float(w.strip()) for w in weights.split(",")]
        except ValueError:
            console.print("[red]--weights must be comma-separated floats.[/red]")
            return
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_model_merging=True,
    )
    config.merge_method = method
    config.merge_model_ids = list(models)
    config.merge_weights = merge_weights or [1.0 / len(models)] * len(models)

    with console.status(f"[bold green]Merging {len(models)} models via {method.upper()}..."):
        compiler = Compiler(config)
        aeg = compiler.compile(models[0], output_path=output)
    console.print(f"[bold green]Merged AEG saved to[/bold green] {aeg.root}")
    table = Table(title="Merge Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Weight", style="magenta")
    for m, w in zip(models, config.merge_weights):
        table.add_row(m, f"{w:.3f}")
    table.add_row("[bold]Method[/bold]", method.upper())
    console.print(table)


@cli.command("ttt-config")
@click.argument("model")
@click.option("--adapter-rank", type=int, default=8, help="LoRA rank for TTT fast-weight adapters.")
@click.option("--ttl", type=int, default=512, help="Fast-weight context TTL (tokens).")
@click.option("--layers", default="", help="Comma-separated layer indices to enable TTT (empty=all).")
@click.pass_context
def ttt_config(ctx: click.Context, model: str, adapter_rank: int, ttl: int, layers: str) -> None:
    """Configure Test-Time Training fast-weight injection (Pass 13).

    Compiles LoRA-style fast-weight adapters into the AEG that update
    continuously during inference, enabling the model to adapt to the
    current context window without gradient descent or full fine-tuning.
    """
    target_layers: list[int] | None = None
    if layers:
        try:
            target_layers = [int(x.strip()) for x in layers.split(",")]
        except ValueError:
            console.print("[red]--layers must be comma-separated integers.[/red]")
            return

    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_ttt=True,
    )
    config.ttt_adapter_rank = adapter_rank
    config.ttt_context_ttl = ttl
    if target_layers is not None:
        config.ttt_target_layers = target_layers

    with console.status(f"[bold green]Injecting TTT fast-weight adapters into {model}..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)

    console.print(f"[bold green]TTT-enabled AEG saved to[/bold green] {aeg.root}")
    table = Table(title="TTT Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Adapter rank (r)", str(adapter_rank))
    table.add_row("Context TTL (tokens)", str(ttl))
    table.add_row("Target layers", str(target_layers or "all"))
    console.print(table)


@cli.command("kv-compress")
@click.argument("model")
@click.option("--method", type=click.Choice(["semantic", "cross-layer", "both"]), default="both",
              help="KV compression method: semantic (Pass 14), cross-layer (Pass 15), or both.")
@click.option("--retention", type=float, default=0.5,
              help="Fraction of KV tokens to retain (0.0–1.0).")
@click.option("--cross-layer-groups", type=int, default=4,
              help="Number of KV sharing groups for cross-layer (Pass 15).")
@click.pass_context
def kv_compress(ctx: click.Context, model: str, method: str, retention: float, cross_layer_groups: int) -> None:
    """Compile KV cache compression into the AEG (Passes 14 and 15).

    Semantic KV (Pass 14) compresses the KV cache by identifying semantically
    redundant token representations and merging them.  Cross-layer KV sharing
    (Pass 15) groups layers to share a single KV cache, reducing memory footprint
    by up to 50% on deep models.
    """
    if not (0.0 < retention <= 1.0):
        console.print("[red]--retention must be between 0.0 (exclusive) and 1.0.[/red]")
        return

    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_semantic_kv=method in ("semantic", "both"),
        enable_cross_layer_kv=method in ("cross-layer", "both"),
    )
    config.semantic_kv_retention = retention
    config.cross_layer_kv_groups = cross_layer_groups

    with console.status(f"[bold green]Compiling KV compression into {model} (method={method})..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)

    console.print(f"[bold green]KV-compressed AEG saved to[/bold green] {aeg.root}")
    table = Table(title="KV Compression Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Method", method)
    table.add_row("Retention ratio", f"{retention:.1%}")
    if method in ("cross-layer", "both"):
        table.add_row("Cross-layer groups", str(cross_layer_groups))
    console.print(table)


@cli.command("green-profile")
@click.argument("model")
@click.option("--carbon-region", default="", help="Electricity grid carbon region (e.g. 'us-west-2').")
@click.option("--renewable-threshold", type=float, default=0.80,
              help="Minimum renewable energy fraction to enable full compilation (0.0–1.0).")
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

    Green Energy Compilation (PRD §16) embeds carbon-aware scheduling hints
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
            f"renewable fraction ≥ {renewable_threshold:.0%} in region {carbon_region or 'auto'}."
        )

    with console.status(f"[bold green]Compiling green-energy profile for {model}..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model)

    console.print(f"[bold green]Green AEG saved to[/bold green] {aeg.root}")
    table = Table(title="Green Energy Profile")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Carbon region", carbon_region or "auto-detected")
    table.add_row("Renewable threshold", f"{renewable_threshold:.0%}")
    table.add_row("Defer mode", "enabled" if defer else "disabled")
    console.print(table)


@cli.command()
@click.argument("model")
@click.option("--backend", type=click.Choice(["nvidia_cc", "intel_tdx", "amd_sev_snp", "openpcc"]),
              default="nvidia_cc", help="TEE backend.")
@click.option("--output", "-o", type=click.Path(), help="Output path for TEE-wrapped AEG.")
@click.option("--report", is_flag=True, help="Print TEE attestation report after compilation.")
@click.pass_context
def tee(ctx: click.Context, model: str, backend: str, output: str | None, report: bool) -> None:
    """Wrap a model in TEE-protected kernels (Pass 17).

    TEE Kernel Wrapping injects HMAC-guarded enter/exit guards around every
    compute kernel and generates a weight-hash manifest so the runtime can
    verify model integrity inside the trusted execution enclave.
    """
    config = CompilerConfig(
        cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(),
        enable_tee=True,
    )
    config.tee_backend = backend

    with console.status(f"[bold green]Wrapping {model} with TEE ({backend.upper()}) protection..."):
        compiler = Compiler(config)
        aeg = compiler.compile(model, output_path=output)

    console.print(f"[bold green]TEE-wrapped AEG saved to[/bold green] {aeg.root}")

    if report:
        try:
            from aether.runtime.r8_tee_manager import TEERuntimeManager
            tee_mgr = TEERuntimeManager(backend=backend)
            tee_mgr.initialize()
            attestation = tee_mgr.get_attestation_report()
            console.print(f"\n[bold]Attestation Report[/bold]")
            table = Table()
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="magenta")
            for k, v in attestation.items():
                table.add_row(k, str(v) if v is not None else "[dim]n/a[/dim]")
            console.print(table)
            hw_backed = attestation.get("hardware_backed", False)
            if hw_backed:
                console.print("[bold green]✓ Hardware-backed TEE confirmed.[/bold green]")
            else:
                console.print("[bold yellow]⚠ Running in software simulation mode (no hardware TEE detected).[/bold yellow]")
        except Exception as exc:
            console.print(f"[red]Could not generate attestation report: {exc}[/red]")


@cli.command()
@click.option("--list-tools", "-l", is_flag=True, help="List all MCP tools registered in the AEG.")
@click.option("--tool", default="", help="Name of a specific MCP tool to inspect.")
@click.option("--call", default="", help="JSON arguments to call a tool (requires --tool).")
@click.argument("model", required=False)
@click.pass_context
def mcp(ctx: click.Context, list_tools: bool, tool: str, call: str, model: str | None) -> None:
    """Inspect or invoke MCP tools registered in a compiled AEG (Runtime R6).

    The MCP (Model Context Protocol) integration layer (Runtime R6) allows
    models to call external tools such as web search, code execution, and
    database access through a structured JSON-RPC interface compiled into
    the AEG artifact.

    Examples:
        aether mcp mymodel --list-tools
        aether mcp mymodel --tool web_search --call '{\"query\": \"Aether Runtime\"}'
    """
    if not model:
        console.print("[dim]Usage: aether mcp <model> --list-tools[/dim]")
        return

    try:
        from aether.runtime.r6_mcp_integration import MCPIntegrationLayer
        rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
        aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if aeg_path is None:
            console.print(f"[red]Model {model!r} not found in cache.[/red]")
            return

        mcp_layer = MCPIntegrationLayer()
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
                console.print(f"[red]Invalid JSON for --call: {exc}[/red]")
                return
            with console.status(f"[bold green]Calling MCP tool {tool!r}..."):
                result = mcp_layer.call_tool(tool, args)
            console.print(f"[bold]Result from {tool!r}:[/bold]")
            console.print(RichJSON(json.dumps(result, indent=2, default=str)))

        elif tool:
            info = mcp_layer.get_tool(tool)
            if info:
                console.print(RichJSON(json.dumps(info, indent=2, default=str)))
            else:
                console.print(f"[red]Tool {tool!r} not found.[/red]")
        else:
            console.print("[dim]Use --list-tools to see available tools, or --tool + --call to invoke.[/dim]")

    except Exception as exc:
        console.print(f"[red]MCP error: {exc}[/red]")


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
