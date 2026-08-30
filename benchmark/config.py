"""Benchmark configuration — every knob lives here, nothing is hard-coded elsewhere.

The models under test are fixed by the benchmark's charter.  Everything
else (prompt lengths, batch sizes, precision, repetitions, seed) is configurable
from the command line so that no experiment requires editing source.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

#: The models this benchmark is defined over.  Every engine loads the same
#: repository at the same revision; see ``resolve_revision`` in system_info.
#: The list is fixed by the benchmark's charter and by the hardware budget it
#: was chosen for — a 135M, a 0.6B, a 350M and a 3.8B checkpoint, which is the
#: largest that fits one 15 GiB accelerator at 16-bit weights alongside a KV
#: cache.  Adding to it or substituting into it invalidates cross-run
#: comparison, so it is changed only by amending this tuple deliberately.
MODELS: tuple[str, ...] = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen3-0.6B",
    "SummerSigh/GPTNeo350M-Instruct-SFT",
    "microsoft/Phi-3.5-mini-instruct",
)

#: Prompt sizes in *tokens*, measured with each model's own tokenizer rather
#: than by character count.
PROMPT_TOKENS: tuple[int, ...] = (32, 256, 1024)

#: ``bf16`` is the primary comparison: all three checkpoints are published in
#: BF16, and the Aether compiler's default weight residency is also BF16, so
#: neither backend is rounding the other's weights.  See REPORT.md, "Precision".
PRECISIONS: tuple[str, ...] = ("bf16", "fp16", "fp32")


@dataclass
class BenchmarkConfig:
    """A complete, printable description of one benchmark invocation."""

    mode: str = "performance"
    models: list[str] = field(default_factory=lambda: list(MODELS))
    prompt_tokens: list[int] = field(default_factory=lambda: list(PROMPT_TOKENS))
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4])
    precisions: list[str] = field(default_factory=lambda: ["bf16"])
    max_new_tokens: int = 128
    warmup_iters: int = 2
    measure_iters: int = 5
    seed: int = 1234
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    devices: int | None = None
    multi_gpu: bool = True
    gpu_sample_interval_s: float = 0.1
    cooldown_s: float = 0.0
    output_dir: str = "benchmark/results"
    cache_dir: str | None = None
    keep_aeg: bool = True

    def describe(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split()]


def _str_list(value: str) -> list[str]:
    """Split on commas only.

    Model identifiers and local checkpoint paths may contain spaces, so
    whitespace cannot be a separator here without silently splitting a path in
    two and reporting it as an unknown model.
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    """Build a configuration from the command line.

    ``--quick`` is a preset, not a separate code path: it only narrows the
    matrix and repetition counts, so a quick run measures the same way a full
    run does.
    """
    parser = argparse.ArgumentParser(
        prog="python benchmark/run_benchmark.py",
        description="Neutral Aether Runtime vs Hugging Face Transformers benchmark.",
    )
    parser.add_argument(
        "--mode",
        default="performance",
        choices=["performance", "memory", "correctness", "profile", "multigpu", "batch",
                 "mixed", "all"],
        help="Which experiment to run. Each mode is independent; 'all' runs them in sequence.",
    )
    parser.add_argument("--models", type=_str_list, default=None,
                        help="Subset of the benchmark models (default: all).")
    parser.add_argument("--prompt-tokens", type=_int_list, default=None,
                        help="Prompt lengths in tokens, e.g. '32,256,1024'.")
    parser.add_argument("--batch-sizes", type=_int_list, default=None,
                        help="Batch sizes to attempt, e.g. '1,2,4'.")
    parser.add_argument("--precisions", type=_str_list, default=None,
                        help="Precisions to compare: bf16, fp16, fp32.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--measure-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None,
                        help="Limit visible GPUs to the first N (default: use all).")
    parser.add_argument("--no-multi-gpu", action="store_true",
                        help="Skip the multi-GPU experiment even if several GPUs exist.")
    parser.add_argument("--gpu-sample-interval", type=float, default=None,
                        help="GPU telemetry sampling period in seconds (default 0.1).")
    parser.add_argument("--cooldown", type=float, default=None,
                        help="Seconds to idle between major sections so the GPU can cool.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None,
                        help="Where compiled .aeg artifacts are written.")
    parser.add_argument("--quick", action="store_true",
                        help="Narrow preset for a sanity check: one model, one length, 2 iters.")
    args = parser.parse_args(argv)

    config = BenchmarkConfig(mode=args.mode)
    if args.quick:
        config.models = [MODELS[0]]
        config.prompt_tokens = [32]
        config.batch_sizes = [1]
        config.max_new_tokens = 32
        config.warmup_iters = 1
        config.measure_iters = 2
    for name, value in (
        ("models", args.models), ("prompt_tokens", args.prompt_tokens),
        ("batch_sizes", args.batch_sizes), ("precisions", args.precisions),
        ("max_new_tokens", args.max_new_tokens), ("warmup_iters", args.warmup_iters),
        ("measure_iters", args.measure_iters), ("seed", args.seed),
        ("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k),
        ("devices", args.devices), ("gpu_sample_interval_s", args.gpu_sample_interval),
        ("cooldown_s", args.cooldown), ("output_dir", args.output_dir),
        ("cache_dir", args.cache_dir),
    ):
        if value is not None:
            setattr(config, name, value)
    if args.no_multi_gpu:
        config.multi_gpu = False
    # The charter fixes the benchmark model list.  A filesystem path is also
    # accepted so the harness itself can be validated offline — that is a
    # pipeline check, not a benchmark result, and the runner labels it as such.
    from pathlib import Path

    unknown = [
        name for name in config.models
        if name not in MODELS and not Path(name).is_dir()
    ]
    if unknown:
        raise SystemExit(
            "--models accepts the benchmark models or an existing local "
            f"checkpoint directory; got {unknown}"
        )
    bad = set(config.precisions) - set(PRECISIONS)
    if bad:
        raise SystemExit(f"--precisions accepts {list(PRECISIONS)}; got {sorted(bad)}")
    return config


def is_charter_model(name: str) -> bool:
    """Whether a model entry is one of the models the benchmark is defined over."""
    return name in MODELS
