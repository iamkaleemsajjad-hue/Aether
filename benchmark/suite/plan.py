"""The run plan: what will be measured, on what, with which settings.

One object describes an entire invocation, it is serialized into every result
file, and both the orchestrator and the worker processes read their instructions
from it. That is deliberate: if the plan is in the artifact, a reader can tell
exactly what was asked for, and a rerun cannot quietly differ from the run it
claims to reproduce.

The model list is not a knob. It is fixed in :mod:`benchmark.config` by the
benchmark's charter and by the hardware budget it was chosen for; ``--models``
can only narrow it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmark.config import MODELS
from benchmark.suite import engines as engine_registry

#: Batch widths, per the charter. Every one is attempted; one that does not fit is
#: recorded as OOM or UNSUPPORTED against that engine, never as a zero and never
#: as a reason to stop.
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16)

#: Prompt lengths in tokens for the prompt-scaling sweep.
PROMPT_TOKENS: tuple[int, ...] = (32, 256, 1024)

#: Generated-output lengths for the output-scaling sweep.
OUTPUT_TOKENS: tuple[int, ...] = (32, 128, 512)

#: The single configuration every headline number refers to. Batch 1 is
#: first-class here because it is the single-user, local-inference case, and a
#: large-batch aggregate must never be allowed to stand in for it.
PRIMARY_PROMPT_TOKENS = 256
PRIMARY_OUTPUT_TOKENS = 128

#: Precisions the suite can hold the whole field to, plus the resolver. Named once
#: so an engine's own table of what it can store is checkable against the set of
#: things it may be asked for, rather than against a list repeated in each place.
PRECISION_CHOICES: tuple[str, ...] = ("auto", "bf16", "fp16", "fp32")

#: Run counts at which the compile-once trade-off is evaluated. A build cost is
#: only justified by the number of inferences that follow it, so the report states
#: the total cost of ownership at several of them instead of picking one.
AMORTIZATION_RUNS: tuple[int, ...] = (1, 10, 100, 1_000, 10_000)


@dataclass
class SuiteConfig:
    """A complete, serializable description of one benchmark invocation."""

    models: list[str] = field(default_factory=lambda: list(MODELS))
    engines: list[str] = field(default_factory=lambda: list(engine_registry.KEYS))
    precision: str = "auto"
    batch_sizes: list[int] = field(default_factory=lambda: list(BATCH_SIZES))
    prompt_tokens: list[int] = field(default_factory=lambda: list(PROMPT_TOKENS))
    output_tokens: list[int] = field(default_factory=lambda: list(OUTPUT_TOKENS))
    primary_prompt_tokens: int = PRIMARY_PROMPT_TOKENS
    primary_output_tokens: int = PRIMARY_OUTPUT_TOKENS
    warmup_iters: int = 3
    measure_iters: int = 10
    seed: int = 1234
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    threads: int | None = None
    #: How many accelerators each engine may see. One by default, and that is a
    #: fairness decision rather than a limitation: a runtime that shards a model
    #: across every visible device is being measured on more hardware than one that
    #: does not, so the two numbers would describe different machines. Nothing in any
    #: engine's placement logic is modified - the worker restricts visibility, and
    #: each engine then does whatever it does with one device. Raise it to compare
    #: multi-device execution deliberately.
    devices: int | None = 1
    #: Whether the thread budget is pinned at all. Kept separate from ``threads``
    #: because "nobody chose" and "deliberately inherited" are different states, and
    #: the orchestrator fills in a default for the first but must not for the second.
    pin_threads: bool = True
    #: Engines the operator excluded. Recorded so the report can distinguish them
    #: from engines that were unavailable.
    excluded_engines: list[str] = field(default_factory=list)
    amortization_runs: list[int] = field(default_factory=lambda: list(AMORTIZATION_RUNS))
    output_dir: str = "benchmark_results"
    #: Second, independent process that reloads an engine's persisted artifact.
    #: This is the evidence for the compile-once claim; without it the claim would
    #: rest on a warm in-process reload, which is not the same thing.
    reuse_probe: bool = True
    #: Reuse raw result files from a previous invocation instead of re-measuring.
    resume: bool = False
    #: Seconds a single (engine, model) worker may take before it is killed and
    #: recorded as failed. Generous, because a first-time compile of a 3.8B model
    #: on a shared host is genuinely slow.
    worker_timeout_s: float = 7200.0
    cooldown_s: float = 0.0
    gpu_sample_interval_s: float = 0.1
    correctness: bool = True
    charts: bool = True
    #: Engine the report's detailed head-to-head drill-down describes. The standings,
    #: the rankings and the pairwise matrix are computed for every engine regardless;
    #: this only selects which engine gets the long-form section. None means the
    #: drill-down is printed for every engine that produced a measurement.
    focus: str | None = None
    #: Engine-specific knobs, passed through to the adapters that need them. Both
    #: engines with a build phase get one of each, and neither gets a knob that
    #: changes what is executed: a cache location, and the device to target.
    aeg_cache_dir: str | None = None
    openvino_cache_dir: str | None = None
    #: Which OpenVINO device to target. ``auto`` asks the adapter to pick the best
    #: device OpenVINO can actually see here, which is the same courtesy the CUDA
    #: engines get from torch; a named device is used as given and recorded.
    openvino_device: str = "auto"
    #: Recorded, not used for control flow: how the run was invoked.
    invocation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def workload_signature(self) -> dict[str, Any]:
        """The subset of the plan that defines the work, for cross-run comparison.

        Two runs whose signatures match were asked for the same work. Two whose
        signatures differ must not have their numbers compared, and the report says
        so by printing this next to the results.
        """
        return {
            "precision": self.precision,
            "batch_sizes": list(self.batch_sizes),
            "prompt_tokens": list(self.prompt_tokens),
            "output_tokens": list(self.output_tokens),
            "primary_prompt_tokens": self.primary_prompt_tokens,
            "primary_output_tokens": self.primary_output_tokens,
            "warmup_iters": self.warmup_iters,
            "measure_iters": self.measure_iters,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "threads": self.threads,
            "devices": self.devices,
        }


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def _str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python benchmark.py",
        description=(
            "Multi-engine inference benchmark: Aether Runtime against the field, on "
            "the same models, weights, hardware, prompts and generation settings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=_str_list, default=None,
                        help="Narrow the locked model list to these entries.")
    parser.add_argument("--engines", type=_str_list, default=None,
                        help=f"Engines to attempt. Known: {', '.join(engine_registry.KEYS)}")
    parser.add_argument("--exclude-engines", type=_str_list, default=None,
                        help="Engines to leave out; recorded as SKIPPED, not omitted.")
    parser.add_argument("--precision", default=None,
                        choices=list(PRECISION_CHOICES),
                        help="Precision every engine is held to. 'auto' resolves from the device.")
    parser.add_argument("--batch-sizes", type=_int_list, default=None)
    parser.add_argument("--prompt-tokens", type=_int_list, default=None)
    parser.add_argument("--output-tokens", type=_int_list, default=None)
    parser.add_argument("--primary-prompt-tokens", type=int, default=None)
    parser.add_argument("--primary-output-tokens", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--measure-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None,
                        help="Pin every engine to this many CPU threads (default: physical cores).")
    parser.add_argument("--devices", type=int, default=None,
                        help="Accelerators each engine may see. 1 keeps every engine on "
                             "the same single device, which is what makes the comparison "
                             "a comparison of engines rather than of machines. 0 means "
                             "no restriction.")
    parser.add_argument("--no-thread-pinning", action="store_true",
                        help="Leave thread counts inherited; the report marks them uncontrolled.")
    parser.add_argument("--amortization-runs", type=_int_list, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-reuse-probe", action="store_true",
                        help="Skip the second-process artifact reload measurement.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse raw result files already present in the output directory.")
    parser.add_argument("--worker-timeout", type=float, default=None)
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument("--no-correctness", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--focus", default=None,
                        help="Engine whose head-to-head section is printed in full. "
                             "Default: every measured engine gets one. Does not affect "
                             "any number, only which drill-downs the report includes.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Narrow preset for a pipeline check: one model, small matrix, few iterations.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smallest possible real run: verifies the pipeline end to end.")
    return parser


def add_engine_options(parser: argparse.ArgumentParser) -> None:
    """Per-engine knobs, grouped so ``--help`` stays readable."""
    group = parser.add_argument_group("engine-specific options")
    group.add_argument("--aeg-cache-dir", default=None,
                       help="Where Aether's compiled .aeg artifacts are kept.")
    group.add_argument("--openvino-cache-dir", default=None)
    group.add_argument("--openvino-device", default=None,
                       help="auto (default), CPU, GPU, NPU, or any OpenVINO device "
                            "name. auto picks the fastest device OpenVINO can see.")


def parse_args(argv: list[str] | None = None) -> SuiteConfig:
    """Build a :class:`SuiteConfig` from the command line."""
    import shlex
    import sys

    parser = build_parser()
    add_engine_options(parser)
    args = parser.parse_args(argv)

    config = SuiteConfig()
    config.invocation = " ".join(
        shlex.quote(part) for part in [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    )

    if args.smoke:
        # The smallest real run: one model, one batch, one length, two iterations.
        # It measures far too little to support a claim, and the report says so, but
        # it exercises every stage of the pipeline with genuine numbers.
        config.models = [MODELS[0]]
        config.batch_sizes = [1]
        config.prompt_tokens = [32]
        config.output_tokens = [16]
        config.primary_prompt_tokens = 32
        config.primary_output_tokens = 16
        config.warmup_iters = 1
        config.measure_iters = 2
    elif args.quick:
        config.models = [MODELS[0]]
        config.batch_sizes = [1, 2]
        config.prompt_tokens = [32, 256]
        config.output_tokens = [32]
        config.primary_prompt_tokens = 32
        config.primary_output_tokens = 32
        config.warmup_iters = 2
        config.measure_iters = 3

    for name, value in (
        ("models", args.models), ("engines", args.engines), ("precision", args.precision),
        ("batch_sizes", args.batch_sizes), ("prompt_tokens", args.prompt_tokens),
        ("output_tokens", args.output_tokens),
        ("primary_prompt_tokens", args.primary_prompt_tokens),
        ("primary_output_tokens", args.primary_output_tokens),
        ("warmup_iters", args.warmup_iters), ("measure_iters", args.measure_iters),
        ("seed", args.seed), ("temperature", args.temperature), ("top_p", args.top_p),
        ("top_k", args.top_k), ("threads", args.threads), ("devices", args.devices),
        ("amortization_runs", args.amortization_runs), ("output_dir", args.output_dir),
        ("worker_timeout_s", args.worker_timeout), ("cooldown_s", args.cooldown),
        ("aeg_cache_dir", args.aeg_cache_dir),
        ("openvino_cache_dir", args.openvino_cache_dir),
        ("openvino_device", args.openvino_device), ("focus", args.focus),
    ):
        if value is not None:
            setattr(config, name, value)

    config.excluded_engines = apply_engine_filter(config, args.exclude_engines)
    if args.no_reuse_probe:
        config.reuse_probe = False
    if args.resume:
        config.resume = True
    if args.no_correctness:
        config.correctness = False
    if args.no_charts:
        config.charts = False
    if args.no_thread_pinning:
        config.pin_threads = False
        config.threads = None
    validate(config, pinning_requested=config.pin_threads)
    return config


#: Sentinel recorded when the operator asked for inherited thread counts, so the
#: report can distinguish "not pinned by choice" from "nobody thought about it".
THREADS_INHERITED = "inherited"


def validate(config: SuiteConfig, pinning_requested: bool = True) -> None:
    """Reject a plan that could not produce a comparable result.

    Checked up front rather than discovered halfway through a two-hour run: a
    misspelled engine name or a model outside the charter is an operator error, and
    failing immediately is cheaper than failing at hour two.
    """
    from pathlib import Path

    unknown_engines = [key for key in config.engines if key not in engine_registry.KEYS]
    if unknown_engines:
        raise SystemExit(
            f"unknown engine(s): {unknown_engines}. Known: {', '.join(engine_registry.KEYS)}"
        )
    if config.focus and config.focus not in engine_registry.KEYS:
        raise SystemExit(
            f"--focus must name a known engine; got {config.focus!r}. "
            f"Known: {', '.join(engine_registry.KEYS)}"
        )
    unknown_models = [
        name for name in config.models
        if name not in MODELS and not Path(name).is_dir()
    ]
    if unknown_models:
        raise SystemExit(
            "the benchmark model list is fixed by the charter; --models can only "
            f"narrow it. Unknown: {unknown_models}. Known: {', '.join(MODELS)}"
        )
    if config.primary_prompt_tokens not in config.prompt_tokens:
        config.prompt_tokens = sorted({*config.prompt_tokens, config.primary_prompt_tokens})
    if config.primary_output_tokens not in config.output_tokens:
        config.output_tokens = sorted({*config.output_tokens, config.primary_output_tokens})
    if 1 not in config.batch_sizes:
        # Batch 1 is the single-user case and the denominator of every scaling
        # figure. A plan without it cannot produce either, so it is added back.
        config.batch_sizes = sorted({1, *config.batch_sizes})
    if config.measure_iters < 2:
        raise SystemExit("--measure-iters must be at least 2; one sample has no dispersion")
    if config.temperature > 0.0:
        # Not refused: a sampling run is a legitimate experiment. But correctness
        # cannot then be compared token-for-token, and the report must say why.
        config.correctness = config.correctness
    if not pinning_requested:
        config.pin_threads = False
        config.threads = None


def apply_engine_filter(config: SuiteConfig, exclude: list[str] | None) -> list[str]:
    """Remove excluded engines, returning the excluded keys for the record."""
    if not exclude:
        return []
    removed = [key for key in config.engines if key in exclude]
    config.engines = [key for key in config.engines if key not in exclude]
    return removed
