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
import os
import shutil
import sys
import tempfile
import time
import builtins
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.table import Table
from rich.text import Text

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.compiler.stage2_optimizer.pass16_green_energy import GreenEnergyCompilationPass
from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
from aether.compiler.stage3_targeting.target_registry import TargetRegistry
from aether.core.aeg_format import load_aeg_package
from aether.core.constants import AETHER_VERSION, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT
from aether.hub.client import HubClient
from aether.utils.file_io import aether_cache_dir, delete_model
from aether.utils.logging import configure_logging

console = Console()


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


@click.group(context_settings={"max_content_width": 120})
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--cache-dir", type=click.Path(), help="Custom Aether cache directory.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, cache_dir: str | None) -> None:
    """Aether Runtime — compile any AI model, run it on any hardware."""
    configure_logging(level="DEBUG" if verbose else "INFO")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["cache_dir"] = cache_dir


@cli.command("compile")
@click.argument("model")
@click.option("--target", "-t", multiple=True, help="Hardware target(s). Repeat for multiple targets.")
@click.option("--quality-budget", type=float, default=0.02, help="Maximum perplexity increase budget.")
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
@click.option("--grpc-port", default=None, type=int, help="Also expose authenticated gRPC on this port.")
@click.option(
    "--semantic-cache/--no-semantic-cache",
    default=None,
    help="Enable or disable the R11 semantic request cache.",
)
@click.option(
    "--threshold",
    "semantic_cache_threshold",
    type=click.FloatRange(0.0, 1.0),
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

    config = RuntimeConfig(
        model_cache_dir=ctx.obj.get("cache_dir"),
        server_port=port,
        server_host=host,
        semantic_cache_threshold=semantic_cache_threshold,
        slo_profiles=configured_slo_profiles,
        **({"enable_semantic_cache": semantic_cache} if semantic_cache is not None else {}),
    )
    app = create_app(config)
    if model:
        try:
            # A model argument is a startup contract, not decorative CLI
            # metadata: validate and load it before accepting requests.
            app.state.aether_runtime._load_model(model)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(f"unable to load model {model!r}: {exc}") from exc
    grpc_server = None
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
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        if grpc_server is not None:
            grpc_server.stop(0)


@cli.command()
@click.argument("model")
@click.option("--compare", help="Compare against another backend (e.g., 'vllm').")
@click.option("--max-tokens", default=128, help="Maximum tokens per benchmark run.")
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
        raise click.ClickException(
            f"benchmark comparison backend {compare!r} is unavailable; "
            "only the active Aether backend was measured"
        )


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


@cli.command("inspect")
@click.argument("model")
@click.option("--mtp", is_flag=True, help="Inspect the persisted native MTP head artifact.")
@click.pass_context
def inspect_model(ctx: click.Context, model: str, mtp: bool) -> None:
    """Inspect an AEG manifest and optional persisted MTP configuration."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    try:
        info_data = rt.info(model)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    result: dict[str, Any] = {"model_id": model, "info": info_data}
    if mtp:
        path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if path is None:
            raise click.ClickException(f"model {model!r} was not found")
        mtp_path = Path(path) / "speculation" / "mtp_config.json"
        if not mtp_path.is_file():
            raise click.ClickException(
                f"model {model!r} has no persisted MTP head; compile with --mtp"
            )
        try:
            result["mtp"] = json.loads(mtp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"invalid MTP metadata: {exc}") from exc
    console.print(RichJSON(json.dumps(result, indent=2, default=str)))


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


@cli.command("list")
@click.pass_context
def cmd_list(ctx: click.Context) -> None:
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


@cli.command("hardware")
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
        rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
        console.print(RichJSON(json.dumps({"model_id": args[0], "grammars": [payload]}, indent=2)))
        return

    if action == "test":
        if len(args) != 1:
            raise click.UsageError("grammar test requires <model.aeg>")
        rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
        path = rt._resolve_aeg_path(args[0])  # noqa: SLF001
        if path is None:
            raise click.ClickException(f"model {args[0]!r} was not found")
        rt._init_v4_layers(path)  # noqa: SLF001
        engine = rt.grammar_engine
        if engine is None or not engine.is_loaded():
            raise click.ClickException(f"model {args[0]!r} has no loadable grammar FSM")
        session = engine.create_session()
        mask = session.get_token_mask()
        console.print(
            RichJSON(
                json.dumps(
                    {
                        "model_id": args[0],
                        "grammar": grammar_name,
                        "loaded": True,
                        "initial_allowed_token_count": sum(mask).bit_count(),
                        "prompt_tested": False,
                        "message": "FSM loaded and initial token mask inspected; no model prompt was executed.",
                    },
                    indent=2,
                )
            )
        )
        return

    if args and target_model is None:
        # Legacy form: aether grammar <model> --schema schema.gbnf
        model = args.pop(0)
    else:
        # PRD form: aether grammar compile <schema.json> --target <model.aeg>
        if len(args) != 1:
            raise click.UsageError("grammar compile requires <schema> --target <model.aeg>")
        model = target_model
        schema = args[0]
    if args:
        raise click.UsageError(f"unexpected grammar arguments: {' '.join(args)}")
    if model is None:
        raise click.UsageError("a grammar target model is required")
    if schema is None:
        console.print("[red]--schema is required to define the grammar.[/red]")
        console.print("[dim]Example: aether grammar gpt2 --schema ./json.gbnf --mode gbnf[/dim]")
        return
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
    config.merge_strategy = method
    config.merge_task_models = list(models)
    config.merge_coefficients = merge_weights or [1.0 / len(models)] * len(models)

    with console.status(f"[bold green]Merging {len(models)} models via {method.upper()}..."):
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
    config.kv_compression_ratio = retention
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
        table.add_row("Cross-layer groups", str(cross_layer_groups))
    console.print(table)


@cli.command("green-profile")
@click.argument("model")
@click.option("--carbon-region", default="", help="Electricity grid carbon region (e.g. 'us-west-2').")
@click.option("--renewable-threshold", type=float, default=0.80,
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
        rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
        if not manager.initialize():
            raise click.ClickException("TEE initialization failed")
        console.print(RichJSON(json.dumps(manager.get_attestation_report(), indent=2, default=str)))
        manager.shutdown()
        return

    if action == "verify":
        report_path = args.pop(0) if args else None
        if not report:
            raise click.UsageError("tee verify requires --report <attestation.json>")
        if report_path is None:
            raise click.UsageError("tee verify requires a report path after --report")
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
            "verified": True,
            "hardware_backed": bool(attestation.get("hardware_backed", False)),
            "trust_level": "hardware" if attestation.get("hardware_backed") else "software_simulation",
            "report": attestation,
        }
        console.print(RichJSON(json.dumps(result, indent=2, default=str)))
        return

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
                console.print("[bold green]PASS: Hardware-backed TEE confirmed.[/bold green]")
            else:
                console.print("[bold yellow]WARN: Running in software simulation mode (no hardware TEE detected).[/bold yellow]")
        except Exception as exc:
            console.print(f"[red]Could not generate attestation report: {exc}[/red]")


@cli.command()
@click.option("--list-tools", "-l", is_flag=True, help="List all MCP tools registered in the AEG.")
@click.option("--tool", default="", help="Name of a specific MCP tool to inspect.")
@click.option("--call", default="", help="JSON arguments to call a tool (requires --tool).")
@click.option("--server", default="", help="MCP server identifier for PRD add/test forms.")
@click.option("--transport", type=click.Choice(["stdio", "http", "websocket"]), default="stdio")
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
    action = "legacy"
    model: str | None = None
    if args and args[0].lower() in {"add", "list", "test"}:
        action = args.pop(0).lower()
        if args:
            model = args.pop(0)
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
            if transport in {"http", "websocket"} and not endpoint:
                raise click.UsageError(f"{transport} MCP servers require --endpoint <url>")
            rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
            console.print(
                RichJSON(
                    json.dumps(
                        {"model_id": model, "server": registry[-1], "persisted": True},
                        indent=2,
                    )
                )
            )
            return
    elif args:
        model = args.pop(0)
    if args:
        raise click.UsageError(f"unexpected MCP arguments: {' '.join(args)}")

    if not model:
        console.print("[dim]Usage: aether mcp <model> --list-tools[/dim]")
        return

    try:
        rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
        aeg_path = rt._resolve_aeg_path(model)  # noqa: SLF001
        if aeg_path is None:
            raise click.ClickException(f"model {model!r} not found in cache")

        # Load the artifact's persisted MCP registry.  Constructing a fresh
        # layer here silently discarded every server/tool compiled into the
        # AEG and made this command report an empty registry for valid files.
        rt._init_v4_layers(aeg_path)  # noqa: SLF001
        mcp_layer = rt.mcp_layer
        if mcp_layer is None:
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


@cli.command("eval")
@click.argument("model")
@click.option("--domain", default="general", help="Evaluation domain.")
@click.option("--examples", type=click.IntRange(1), default=100, help="Number of examples.")
@click.option("--threshold", type=click.FloatRange(0.0, 1.0), default=0.98)
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
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
    console.print(RichJSON(json.dumps(report, indent=2, default=str)))
    if not report.get("passed", False):
        raise click.ClickException("evaluation gate failed")


@cli.command("safety")
@click.argument("model")
@click.pass_context
def safety(ctx: click.Context, model: str) -> None:
    """Inspect persisted safety/provenance controls for an AEG artifact."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
            package.manifest.verify()
            integrity_verified = True
        except Exception:  # noqa: BLE001
            integrity_verified = False
    report = {
        "model_id": model,
        "integrity_verified": integrity_verified,
        "safety_files": safety_files,
        "provenance": prov_dict,
    }
    console.print(RichJSON(json.dumps(report, indent=2, default=str)))


@cli.command("trace")
@click.argument("model", required=False)
@click.option("--prompt", default="", help="Optional prompt to trace a real request.")
@click.pass_context
def trace(ctx: click.Context, model: str | None, prompt: str) -> None:
    """Export measured runtime spans as OTLP-compatible JSON."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
    console.print(RichJSON(json.dumps(payload, indent=2, default=str)))


@cli.command("reasoning")
@click.argument("model")
@click.pass_context
def reasoning(ctx: click.Context, model: str) -> None:
    """Inspect the persisted reasoning graph for an AEG artifact."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    path = rt._resolve_aeg_path(model)  # noqa: SLF001
    if path is None:
        raise click.ClickException(f"model {model!r} was not found")
    root = Path(path)
    candidates = [root / "reasoning" / "reasoning_graph.json", root / "reasoning_graph.json"]
    graph_path = next((p for p in candidates if p.is_file()), None)
    if graph_path is None:
        raise click.ClickException(
            f"model {model!r} has no persisted reasoning graph; compile with "
            "reasoning-graph enabled"
        )
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise click.ClickException(f"reasoning graph for {model!r} is empty or malformed")
    console.print(RichJSON(json.dumps(data, indent=2, default=str)))


@cli.command("mla-stats")
@click.argument("model")
@click.pass_context
def mla_stats(ctx: click.Context, model: str) -> None:
    """Show MLA/attention metadata persisted in the AEG manifest."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    info_data = rt.info(model)
    architecture = info_data.get("architecture", {})
    result = {"model_id": model, "mla_detected": bool(architecture.get("mla_enabled") or architecture.get("mla")), "architecture": architecture}
    console.print(RichJSON(json.dumps(result, indent=2, default=str)))


@cli.command("merge-info")
@click.argument("model")
@click.pass_context
def merge_info(ctx: click.Context, model: str) -> None:
    """Inspect the persisted task-vector merge metadata for an AEG artifact."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    try:
        data = rt.info(model)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    merge = metadata.get("merge") or metadata.get("model_merging")
    if not isinstance(merge, dict):
        raise click.ClickException(
            f"model {model!r} has no persisted model-merging metadata; compile it with merging enabled"
        )
    console.print(RichJSON(json.dumps({"model_id": model, "merge": merge}, indent=2, default=str)))


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
    for name, value in (("task1", task1), ("task2", task2), ("task3", task3), ("task4", task4)):
        if value is not None:
            weights[name] = value
    if not weights:
        raise click.UsageError("provide at least one --task NAME=VALUE or --task1/--task2 weight")
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    try:
        normalized = rt.set_task_weights(model, **weights)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    console.print(RichJSON(json.dumps({"model_id": model, "weights": normalized}, indent=2)))


@cli.command("kv-share")
@click.argument("model")
@click.option("--output", "output_path", type=click.Path(), help="Output AEG path.")
@click.pass_context
def kv_share(ctx: click.Context, model: str, output_path: str | None) -> None:
    """Compile with cross-layer KV sharing enabled (Pass 15)."""
    config = CompilerConfig(cache_dir=ctx.obj.get("cache_dir") or aether_cache_dir(), enable_cross_layer_kv=True)
    result = Compiler(config).compile(model, output_path=output_path)
    console.print(f"[bold green]Cross-layer KV AEG saved to[/bold green] {result.root}")


@cli.command("multi-agent")
@click.argument("multi_agent_args", nargs=-1)
@click.option("--agents", type=click.IntRange(1), default=4)
@click.option("--shared-prefix", default="")
@click.option("--coordination", type=click.Choice(["relay", "broadcast", "tree"]), default="relay")
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
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    if action == "test" and rt._resolve_aeg_path(model) is None:  # noqa: SLF001
        raise click.ClickException(f"model {model!r} was not found")
    result = rt.multi_agent_session(agents, shared_prefix)
    if action == "test":
        result.update(
            {
                "model_id": model,
                "coordination": coordination,
                "test": "local_kv_coordinator_initialization",
                "tested": True,
            }
        )
    console.print(RichJSON(json.dumps(result, indent=2, default=str)))


@cli.command("slo-status")
@click.pass_context
def slo_status(ctx: click.Context) -> None:
    """Show the live scheduler queue and configured SLO mode."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    report = {"scheduler": rt.config.scheduler, "queue": rt.scheduler.queue_snapshot(), "max_batch_size": rt.scheduler.max_batch_size}
    console.print(RichJSON(json.dumps(report, indent=2, default=str)))


@cli.command("green-route")
@click.option("--regions", multiple=True, required=True, help="Candidate deployment regions; repeat for multiple regions.")
@click.option("--latency-deadline", type=click.FloatRange(min=0.0), default=1.0, show_default=True)
def green_route(regions: tuple[str, ...], latency_deadline: float) -> None:
    """Select the lowest-carbon candidate region within the latency deadline."""
    from aether.runtime.r7_green_power_manager import GreenPowerManager

    manager = GreenPowerManager()
    selected = manager.select_region(list(regions), latency_deadline_s=latency_deadline)
    console.print(
        RichJSON(
            json.dumps(
                {
                    "selected_region": selected,
                    "candidate_regions": list(regions),
                    "latency_deadline_s": latency_deadline,
                    "policy": "lowest_carbon_within_latency_deadline",
                    "telemetry_source": "compiled/static_region_profile",
                },
                indent=2,
            )
        )
    )


@cli.group("slo-profile")
def slo_profile() -> None:
    """Manage named scheduler SLO profiles."""


@slo_profile.command("add")
@click.argument("name")
@click.option("--ttft", type=click.FloatRange(min=0.0), required=True, help="Maximum TTFT in milliseconds.")
@click.option("--tbt", type=click.FloatRange(min=0.0), required=True, help="Maximum TBT in milliseconds.")
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
    console.print(
        RichJSON(
            json.dumps(
                {"name": name, "profile": profiles[name], "persisted_to": str(profile_path)},
                indent=2,
            )
        )
    )


@cli.group("hub")
def hub() -> None:
    """Authenticate and exchange AEG artifacts with Aether Hub."""


def _hub_client() -> HubClient:
    token_path = aether_cache_dir() / "hub_token"
    token = token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else None
    return HubClient(auth_token=token)


@hub.command("login")
@click.option("--token", prompt=True, hide_input=True)
def hub_login(token: str) -> None:
    result = _hub_client().login(token)
    token_path = aether_cache_dir() / "hub_token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    console.print(RichJSON(json.dumps(result, indent=2)))


@hub.command("search")
@click.argument("query")
def hub_search(query: str) -> None:
    console.print(RichJSON(json.dumps([m.to_dict() for m in _hub_client().search(query)], indent=2)))


@hub.command("push")
@click.argument("model_id")
@click.argument("package", type=click.Path(exists=True))
def hub_push(model_id: str, package: str) -> None:
    manifest = _hub_client().upload(model_id, package)
    console.print(RichJSON(json.dumps(manifest.to_dict(), indent=2)))


@hub.command("pull")
@click.argument("model_id")
@click.argument("output", type=click.Path())
def hub_pull(model_id: str, output: str) -> None:
    path = _hub_client().download(model_id, output)
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
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    report = rt.grpo_train_step(model, list(prompts), group_size=group_size)
    console.print(RichJSON(json.dumps(report, indent=2, default=str)))
    if report.get("status") == "failed":
        raise click.ClickException(report.get("error", "GRPO step failed"))


@train.command("verify")
@click.argument("model", required=False)
@click.option("--domain", default="math", show_default=True)
@click.option("--example", "example_text", required=True, help="Response text to verify.")
@click.option("--ground-truth", default=None)
@click.option("--test-code", default=None, help="Optional code used by the code verifier.")
def train_verify(
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

    try:
        reward = GRPOTrainer().verify_response(
            example_text,
            domain=domain,
            ground_truth=ground_truth,
            test_code=test_code,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        RichJSON(
            json.dumps(
                {
                    "status": "verified",
                    "model": model,
                    "domain": domain,
                    "reward": reward,
                },
                indent=2,
                default=str,
            )
        )
    )


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
    console.print(RichJSON(payload))


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
    console.print(
        RichJSON(
            json.dumps(
                {
                    "path": str(path),
                    "sha256": digest,
                    "reference_op": reference_op,
                    "exported_symbol": symbol,
                    "loadable": True,
                    "verified": True,
                },
                indent=2,
            )
        )
    )


@cli.group("kv")
def kv() -> None:
    """KV infrastructure commands."""


@kv.command("transfer-stats")
@click.argument("model", required=False)
@click.pass_context
def kv_transfer_stats(ctx: click.Context, model: str | None) -> None:
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    console.print(RichJSON(json.dumps(rt.kv_transfer_stats(), indent=2, default=str)))


@kv.command("nika-policy")
@click.option("--kv-size-gb", type=click.FloatRange(min=0.0), required=True)
@click.option("--bw-gbps", type=click.FloatRange(min=0.0, min_open=True), required=True)
@click.option("--decode-util", type=click.FloatRange(0.0, 1.0), required=True)
def kv_nika_policy(kv_size_gb: float, bw_gbps: float, decode_util: float) -> None:
    """Evaluate the real NIKA transfer-versus-recompute policy."""
    from aether.runtime.r12_cxl_kv_pool import TraCTPolicy

    result = TraCTPolicy().nika_policy(kv_size_gb, bw_gbps, decode_util)
    result["execution_mode"] = "analytical_policy"
    result["physical_network_transfer"] = False
    console.print(RichJSON(json.dumps(result, indent=2, default=str)))


@kv.command("cxl-pool-status")
@click.pass_context
def kv_cxl_pool_status(ctx: click.Context) -> None:
    """Report the configured CXL KV pool and explicit local fallback state."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir")))
    console.print(RichJSON(json.dumps(rt.cxl_pool_status(), indent=2, default=str)))


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


@cli.command("quantize-report")
@click.argument("model")
@click.pass_context
def quantize_report(ctx: click.Context, model: str) -> None:
    """Show measured sub-2-bit quantization data for an AEG artifact."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    try:
        report = rt.quantization_report(model)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(RichJSON(json.dumps(report, indent=2, default=str)))


@cli.group("cache")
def cache() -> None:
    """Inspect and control the semantic request cache."""


@cache.command("stats")
@click.pass_context
def cache_stats(ctx: click.Context) -> None:
    """Show measured semantic-cache statistics."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
    console.print(RichJSON(json.dumps(rt.semantic_cache_stats(), indent=2, default=str)))


@cache.command("flush")
@click.pass_context
def cache_flush(ctx: click.Context) -> None:
    """Flush the semantic request cache and report the removed entry count."""
    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))
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
            "status": "warn",
            "detail": "not installed; install aether-runtime[pytorch] for PyTorch model execution",
        })

    # Hardware detection
    try:
        from aether.backends.hardware_detector import detect_all_capabilities, validate_backend_environment
        caps = detect_all_capabilities()
        report["hardware"] = [
            {
                "target_id": c.target_id,
                "vendor": c.vendor,
                "device": c.device,
                "available": c.available,
                "unavailable_reason": c.unavailable_reason,
            }
            for c in caps
        ]
        # CPU is always available — validate it
        cpu_val = validate_backend_environment("cpu")
        required_cpu_failures = [
            item for item in cpu_val.checks_failed
            if "pytorch" not in item.lower()
        ]
        report["checks"].append({
            "name": "hardware:cpu",
            "status": "pass" if not required_cpu_failures else "warn",
            "detail": {
                "passed": cpu_val.checks_passed,
                "failed": cpu_val.checks_failed,
                "optional_failures": [
                    item for item in cpu_val.checks_failed
                    if item not in required_cpu_failures
                ],
                "warnings": cpu_val.warnings,
            },
        })
    except Exception as exc:  # noqa: BLE001
        report["checks"].append({"name": "hardware:detect", "status": "fail", "detail": str(exc)})

    # Smoke test: import and instantiate Runtime
    def _smoke() -> str:
        from aether import Runtime, RuntimeConfig
        rt = Runtime(RuntimeConfig(hf_offline=True))
        # Verify the runtime object was created properly without accessing private state
        assert rt is not None
        return f"Runtime instantiated OK (class={type(rt).__name__})"
    report["checks"].append(check("smoke:runtime_init", _smoke))

    # Native CPU kernels
    def _native_cpu() -> str:
        from aether.kernels.native_cpu import get_native_kernels
        nk = get_native_kernels()
        if nk.ensure_compiled():
            return f"kernels={nk.available_kernels()}"
        return f"build_failed={nk.build_error}"
    report["checks"].append(check("kernels:native_cpu", _native_cpu))

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
        return "not found (run: aether hardware detect --save)"
    report["checks"].append(check("file:hardware_validation_matrix", _hw_matrix))


    # Summarize
    passed = sum(1 for c in report["checks"] if c["status"] == "pass")
    failed = sum(1 for c in report["checks"] if c["status"] == "fail")
    report["summary"] = {"passed": passed, "failed": failed, "total": len(report["checks"])}

    if as_json:
        console.print(RichJSON(json.dumps(report, indent=2, default=str)))
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
    else:
        console.print("  [green]All OK[/green]")


# ---------------------------------------------------------------------------
# aether hardware — hardware detection and validation (PRD §41, §42)
# ---------------------------------------------------------------------------

@cli.group("hardware")
def hardware() -> None:
    """Hardware detection, capability reporting, and backend validation."""


@hardware.command("detect")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.option("--save", is_flag=True, help="Update hardware_validation_matrix.json in the repo root.")
def hardware_detect(as_json: bool, save: bool) -> None:
    """Detect all hardware backends on this host and report real availability."""
    from aether.backends.hardware_detector import detect_all_capabilities

    caps = detect_all_capabilities()
    data = {
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host_platform": sys.platform if "sys" in dir() else __import__("sys").platform,
        "targets": [c.to_dict() for c in caps],
    }

    if save:
        import sys as _sys
        repo_root = Path(__file__).parent.parent.parent.parent
        matrix_path = repo_root / "hardware_validation_matrix.json"
        import json as _json
        matrix_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]Saved to {matrix_path}[/green]")

    if as_json:
        console.print(RichJSON(json.dumps(data, indent=2, default=str)))
        return

    table = Table(title="Hardware Capability Detection")
    table.add_column("Target ID", style="cyan")
    table.add_column("Vendor", style="magenta")
    table.add_column("Device", style="white")
    table.add_column("Available", style="bold")
    table.add_column("Exec Tested", style="dim")
    for c in caps:
        avail = "[green]YES[/green]" if c.available else "[red]NO[/red]"
        tested = "[green]YES[/green]" if c.execution_tested else "[dim]NO[/dim]"
        table.add_row(c.target_id, c.vendor, c.device[:40], avail, tested)
    console.print(table)
    console.print(f"\n[dim]{len([c for c in caps if c.available])} available, "
                  f"{len([c for c in caps if not c.available])} unavailable[/dim]")


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
        console.print(RichJSON(json.dumps([c.to_dict() for c in caps], indent=2, default=str)))
        return

    for c in caps:
        table = Table(title=f"{c.target_id} — {c.vendor} {c.device}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("vendor", c.vendor)
        table.add_row("device", c.device)
        table.add_row("architecture", c.architecture)
        table.add_row("driver_version", c.driver_version)
        table.add_row("memory_bytes", f"{c.memory_bytes:,}")
        table.add_row("supports_fp16", str(c.supports_fp16))
        table.add_row("supports_bf16", str(c.supports_bf16))
        table.add_row("supports_fp8", str(c.supports_fp8))
        table.add_row("supports_fp4", str(c.supports_fp4))
        table.add_row("supports_tee", str(c.supports_tee))
        table.add_row("implemented", "[green]YES[/green]" if c.implemented else "[red]NO[/red]")
        table.add_row("available", "[green]YES[/green]" if c.available else "[red]NO[/red]")
        table.add_row("execution_tested", "[green]YES[/green]" if c.execution_tested else "[dim]NO[/dim]")
        table.add_row("production_validated", "[green]YES[/green]" if c.production_validated else "[dim]NO[/dim]")
        if c.unavailable_reason:
            table.add_row("unavailable_reason", f"[dim]{c.unavailable_reason}[/dim]")
        console.print(table)


@hardware.command("validate")
@click.argument("target_id", default="cpu")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
def hardware_validate(target_id: str, as_json: bool) -> None:
    """Run environment contract checks for a backend target."""
    from aether.backends.hardware_detector import validate_backend_environment

    result = validate_backend_environment(target_id)
    if as_json:
        console.print(RichJSON(json.dumps(result.to_dict(), indent=2, default=str)))
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
    status = "[green]AVAILABLE[/green]" if result.all_passed else "[red]UNAVAILABLE[/red]"
    console.print(f"\nBackend {target_id!r}: {status}")
    if not result.all_passed and result.checks_failed:
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
    """List all registered backends with availability status."""
    from aether.backends.hardware_detector import detect_all_capabilities

    caps = detect_all_capabilities()
    if as_json:
        console.print(RichJSON(json.dumps(
            [{"target_id": c.target_id, "vendor": c.vendor, "device": c.device,
              "available": c.available, "implemented": c.implemented,
              "execution_tested": c.execution_tested,
              "unavailable_reason": c.unavailable_reason}
             for c in caps], indent=2)))
        return

    table = Table(title="Aether Backend Registry")
    table.add_column("Target ID", style="cyan")
    table.add_column("Vendor", style="magenta")
    table.add_column("Device", style="white")
    table.add_column("Impl", style="dim")
    table.add_column("Available", style="bold")
    table.add_column("Exec Tested", style="dim")
    for c in caps:
        impl = "[green]YES[/green]" if c.implemented else "[red]NO[/red]"
        avail = "[green]YES[/green]" if c.available else "[red]NO[/red]"
        tested = "[green]YES[/green]" if c.execution_tested else "[dim]NO[/dim]"
        table.add_row(c.target_id, c.vendor, c.device[:30], impl, avail, tested)
    console.print(table)


# ---------------------------------------------------------------------------
# aether inspect — deep AEG inspection (PRD §44)
# ---------------------------------------------------------------------------

@cli.command("inspect")
@click.argument("aeg_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@click.pass_context
def inspect_aeg(ctx: click.Context, aeg_path: str, as_json: bool) -> None:
    """Deep inspection of a compiled AEG artifact."""
    from aether.core.aeg_format import load_aeg_package

    pkg = load_aeg_package(aeg_path)
    root = Path(aeg_path)
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    report: dict[str, Any] = {
        "path": aeg_path,
        "manifest": pkg.manifest.to_dict() if pkg.manifest else {},
        "files": files,
        "file_count": len(files),
    }

    if as_json:
        console.print(RichJSON(json.dumps(report, indent=2, default=str)))
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
# aether benchmark — real measured performance (PRD §36)
# ---------------------------------------------------------------------------

@cli.command("benchmark")
@click.argument("model")
@click.option("--prompts", "-p", multiple=True, default=["Hello, tell me about yourself."],
              help="Prompts to benchmark. Repeat for multiple prompts.")
@click.option("--max-tokens", type=int, default=64, help="Max tokens per run.")
@click.option("--runs", type=int, default=5, help="Number of measured runs.")
@click.option("--warmup", type=int, default=1, help="Number of warmup runs (not measured).")
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

    rt = Runtime(RuntimeConfig(model_cache_dir=ctx.obj.get("cache_dir"), hf_offline=True))

    def _generate(prompt: str, max_tok: int) -> str:
        return rt.generate(model, prompt, max_tokens=max_tok, temperature=0.0).text

    runner = BenchmarkRunner(
        generate_fn=_generate,
        model_id=model,
        num_warmup_runs=warmup,
        num_measured_runs=runs,
    )

    with console.status(f"[bold green]Benchmarking {model} ({warmup} warmup + {runs} runs)..."):
        report = runner.run(list(prompts), max_tokens=max_tokens, output_path=output)

    if as_json:
        console.print(RichJSON(json.dumps(report.to_dict(), indent=2, default=str)))
        return

    table = Table(title=f"Benchmark Results — {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    summary = report.to_dict()["summary"]
    for k, v in summary.items():
        vstr = f"{v:.4f}" if isinstance(v, float) else str(v)
        table.add_row(k, vstr)
    console.print(table)
    if output:
        console.print(f"[dim]Full report saved to {output}[/dim]")
    failed = summary.get("failed_runs", 0)
    if failed:
        console.print(f"[yellow]Warning: {failed} run(s) failed[/yellow]")


def main() -> None:
    """CLI entry point for `aether` command."""
    cli(obj={})


# Preserve the short v3 command while exposing the PRD spelling.
cli.add_command(hardware, name="hw")


if __name__ == "__main__":
    main()

