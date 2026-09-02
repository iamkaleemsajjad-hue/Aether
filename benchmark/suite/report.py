"""Assemble the human-readable report from the raw payload and the analysis.

The formatting helpers are the existing report's, imported rather than copied, so
a number is rendered the same way in both documents.

Two editorial rules, both enforced structurally rather than by care:

* every derived figure is printed next to its operands, so a percentage can be
  checked against the two measurements it came from without opening the JSON;
* the win table and the loss table are produced by the same function with the same
  formatting. There is no shorter, quieter rendering for the losses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.reporting import _bytes, _fmt, _table
from benchmark.suite import status as status_mod
from benchmark.suite.analysis import (
    DEVICE_DIFFERENCE,
    SAME_REPRESENTATION,
    WORK_DIFFERENCE,
)

#: How each comparability verdict is rendered in a table cell. Keyed on the verdict
#: itself so a label added to the analysis module shows up here as a KeyError-free
#: ``?`` rather than being silently printed as its opposite.
_COMPARABILITY_CELL: dict[str, str] = {
    SAME_REPRESENTATION: "like for like",
    "REPRESENTATION_DIFFERENCE": "format differs",
    DEVICE_DIFFERENCE: "hardware differs",
    WORK_DIFFERENCE: "work differs",
}


def _comparable_cell(label: str | None) -> str:
    """The short reading of one comparability verdict, for a table cell."""
    return _COMPARABILITY_CELL.get(str(label), str(label or "?").lower())


def _pct(value: Any, digits: int = 1) -> str:
    """Signed percentage, so a loss is never rendered as though it were a gain."""
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}%"


def _ratio(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}x"


def _short(model: str) -> str:
    return model.split("/")[-1]


def _seconds(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}s"


def _ms(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f} ms"


def executive_summary(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Answer the ten questions the charter puts first, or decline to.

    Each answer is derived from the rankings and the win/loss tables below it, so the
    summary cannot disagree with the body. A question the run cannot answer is
    answered with what is missing, not left blank.
    """
    rankings = analysis["rankings"]
    standings = analysis["standings"]
    lines = ["## Executive summary", ""]

    def leader(name: str) -> str:
        order = rankings.get(name, {}).get("order") or []
        if not order:
            return "not determined by this run"
        head = order[0]
        value = head["value"]
        # Rendered in the metric's own units. A byte count printed as a bare number
        # with two decimals is unreadable, and a latency printed as tok/s is wrong.
        if name in {"host_memory", "device_memory"}:
            rendered = _bytes(value)
        elif name in {"ttft", "latency", "cold_start"}:
            rendered = _seconds(value)
        elif isinstance(value, (int, float)):
            rendered = f"{float(value):,.2f} tok/s"
        else:
            rendered = str(value)
        model = head.get("model")
        return (f"**{head['engine']}** ({rendered}"
                + (f" on {_short(model)}" if model else "") + ")")

    overall = standings[0]["engine"] if standings else "not determined"
    lines += [
        f"1. **Best overall across the matrix**: **{overall}**"
        + (f" - median {standings[0]['median_percent_of_best']:.0f}% of the fastest "
           f"engine in each cell, winning {standings[0]['win_rate_percent']:.0f}% of its "
           f"{standings[0]['compared']} pairings" if standings else "")
        + ".",
        f"2. **Fastest single cell anywhere in the matrix**: {leader('peak_throughput')}.",
        f"3. **Fastest at batch 1** (the single-user case): {leader('batch_1')}.",
        f"4. **Best batch scaling**: {_scaling_leader(rankings)}.",
        f"5. **Lowest time to first token**: {leader('ttft')}"
        + (" - measured by more than one method across the field, so read it with the "
           "per-method note in the rankings section"
           if (rankings.get("ttft") or {}).get("mixed_methods") else "")
        + ".",
        f"6. **Lowest single-request latency**: {leader('latency')}.",
        f"7. **Lowest peak memory**: {leader('host_memory')}.",
        f"8. **Fastest cold start**: {leader('cold_start')}.",
        f"9. **Closest competitions**: {_closest_answer(analysis)}.",
        f"10. **Compilation trade-off**: {_compile_answer(analysis)}.",
        "",
        "Every one of those is a different workload, and the engine that wins one does "
        "not necessarily win another. The standings below score the whole matrix; the "
        "per-workload rankings are at the end of the report.",
        "",
        _standings_table(analysis),
        "",
    ]
    return "\n".join(lines)


def _standings_table(analysis: dict[str, Any]) -> str:
    """The league table: every engine's record against the whole field.

    ``% of best`` is the scale-free score: in each cell, an engine's throughput as a
    share of the fastest engine's throughput in that cell, then the median across
    cells. It does not depend on which engine is treated as the reference, which is
    what makes it the neutral summary.
    """
    standings = analysis["standings"]
    if not standings:
        return "_No engine produced a comparable measurement, so there are no standings._\n"
    rows = []
    for entry in standings:
        rows.append([
            str(entry["rank"]),
            f"`{entry['engine']}`",
            (f"{entry['median_percent_of_best']:.0f}%"
             if entry["median_percent_of_best"] is not None else "—"),
            f"{entry['wins']}/{entry['losses']}/{entry['ties']}",
            f"{entry['win_rate_percent']:.0f}%",
            _pct(entry["median_improvement_percent"]),
            str(entry["cells_measured"]),
            f"{entry['same_representation']}/{entry['compared']}",
            f"{entry.get('cross_device', 0)}/{entry['compared']}",
        ])
    return _table(
        ["Rank", "Engine", "% of best (median)", "W/L/T", "Win rate",
         "Median difference vs field", "Cells measured",
         "Same-representation pairings", "Pairings that crossed hardware"],
        rows,
    )


def _closest_answer(analysis: dict[str, Any]) -> str:
    """How many pairings were too close to call, at each stated threshold."""
    buckets = analysis["win_loss"].get("closest") or {}
    everything = analysis.get("pairwise") or []
    comparable = [item for item in everything if item["throughput"].get("comparable")]
    if not comparable:
        return "no comparable pairing was produced"
    counts = []
    for threshold in (1, 5, 10):
        within = [
            item for item in comparable
            if abs(item["throughput"]["subject_improvement_percent"]) <= threshold
        ]
        counts.append(f"{len(within)} within {threshold}%")
    del buckets
    return ", ".join(counts) + f" of {len(comparable)} ordered pairings"


def _rank_of(ranking: dict[str, Any], engine: str) -> str:
    for entry in ranking.get("order") or []:
        if entry["engine"] == engine:
            return f"{entry['rank']}"
    return "not ranked (no measurement)"


def _scaling_leader(rankings: dict[str, Any]) -> str:
    order = (rankings.get("batch_scaling") or {}).get("order") or []
    if not order:
        return "not determined; no engine completed more than one batch width"
    head = order[0]
    return (f"**{head['engine']}** at {head['value']:.0f}% of linear scaling up to "
            f"batch {head['batch_size']} on {_short(head['model'])}")


def _extreme(win_loss: dict[str, Any], key: str, engine: str) -> str:
    """The widest margin in one engine's favour, or against it."""
    entry = win_loss.get(key)
    if not entry:
        pool = (win_loss.get("all") or {}).get(
            "won" if key == "largest_advantage" else "lost"
        ) or []
        if not pool:
            return "no comparison of this kind was produced by this run"
        chooser = max if key == "largest_advantage" else min
        entry = chooser(
            pool, key=lambda item: item["throughput"]["subject_improvement_percent"]
        )
        return (f"{_extreme_text(entry)} - note: this comparison crosses a "
                "representation boundary")
    throughput = entry["throughput"]
    if key == "largest_disadvantage" and throughput["subject_improvement_percent"] >= 0:
        # Reporting a positive number under the word "disadvantage" would read as a
        # loss that did not happen. Say plainly that there was none, and give the
        # narrowest margin instead, which is the nearest thing the run measured.
        return (
            f"none - `{engine}` was not slower in any same-representation comparison. "
            f"Its narrowest margin was "
            f"{_pct(throughput['subject_improvement_percent'])} against "
            f"`{entry['competitor']}` on {_short(entry['model'])} at batch "
            f"{entry['batch_size']}"
        )
    return _extreme_text(entry)


def _extreme_text(entry: dict[str, Any]) -> str:
    """Render one comparison as a sentence, with both operands visible."""
    throughput = entry["throughput"]
    return (
        f"{_pct(throughput['subject_improvement_percent'])} against "
        f"`{entry['competitor']}` on {_short(entry['model'])} at batch "
        f"{entry['batch_size']}, prompt {entry['prompt_tokens']}, output "
        f"{entry['output_tokens']} "
        f"({throughput['subject']:,.2f} against {throughput['other']:,.2f} tok/s)"
    )


def _best_model(comparisons: list[dict[str, Any]], engine: str) -> str:
    """The model where one engine's median advantage over the field is largest."""
    import statistics

    by_model: dict[str, list[float]] = {}
    for item in comparisons:
        if (item["comparability"] == SAME_REPRESENTATION
                and item["throughput"].get("comparable")):
            by_model.setdefault(item["model"], []).append(
                item["throughput"]["subject_improvement_percent"]
            )
    suffix = ""
    if not by_model:
        for item in comparisons:
            if item["throughput"].get("comparable"):
                by_model.setdefault(item["model"], []).append(
                    item["throughput"]["subject_improvement_percent"]
                )
        suffix = " (no comparison in this set was like for like)"
    if not by_model:
        return "not determined; no comparison was produced by this run"
    scored = {model: statistics.median(values) for model, values in by_model.items()}
    best = max(scored, key=lambda key: scored[key])
    count = len(by_model[best])
    del engine
    return (f"`{_short(best)}`, median {_pct(scored[best])} across "
            f"{count} comparison{'' if count == 1 else 's'}{suffix}")


def _memory_answer(comparisons: list[dict[str, Any]], engine: str) -> str:
    import statistics

    host = [item for item in comparisons if item["host_memory"].get("comparable")]
    if not host:
        return "no comparable memory measurement"
    values = [item["host_memory"]["subject_improvement_percent"] for item in host]
    median = statistics.median(values)
    direction = "less" if median > 0 else "more"
    return (f"`{engine}` used a median {abs(median):.1f}% {direction} peak process "
            f"memory than the engines it was compared against, across {len(host)} "
            f"comparison{'' if len(host) == 1 else 's'} "
            f"(range {min(values):+.1f}% to {max(values):+.1f}%)")


def _compile_answer(analysis: dict[str, Any]) -> str:
    """The soonest *build* cost anywhere in the matrix to pay for itself.

    Restricted to engines that actually have a build phase. An engine with none has no
    compilation trade-off to report, and quoting its load time under that heading would
    answer a different question than the one asked.
    """
    breaks = [
        item for item in analysis["compile_economics"]["break_even"]
        if item.get("break_even_runs") is not None
        and item.get("subject_has_build_phase")
        and (item.get("subject_build_s") or 0.0) > 0
    ]
    if not breaks:
        return (
            "no build cost was repaid in this run; either no engine with a build phase "
            "was both slower to start and faster per request than another, or the "
            "start-up measurements needed for it are missing"
        )
    soonest = min(breaks, key=lambda item: item["break_even_runs"])
    return (
        f"on {_short(soonest['model'])}, {soonest['interpretation']} "
        f"(build {_seconds(soonest['subject_build_s'], 1)}, start-up "
        f"{_seconds(soonest['subject_startup_s'], 1)} against "
        f"{_seconds(soonest['competitor_startup_s'], 1)}, per request "
        f"{_seconds(soonest['subject_latency_s'])} against "
        f"{_seconds(soonest['competitor_latency_s'])})"
    )


def methodology_section(payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    plan = payload.get("plan") or {}
    threads = plan.get("threads")
    lines = [
        "## Methodology",
        "",
        "One variable is intended to change between rows: the inference stack. "
        "Everything else is held fixed and recorded.",
        "",
        "**Held fixed**",
        "",
        "- **Model and weights.** The same repository at the same revision for every "
        "engine. Revisions are listed per model below.",
        f"- **Precision.** {plan.get('resolved_precision')} for every engine. "
        f"Chosen because: {plan.get('precision_reason')}.",
        "- **Prompts.** Built once per model, before any engine starts, to an exact "
        "token count using that model's own tokenizer, then handed to every engine as "
        "the identical string. Each engine's tokenizer is then checked against the "
        "prompt-builder's, and any disagreement is reported in the compatibility table.",
        f"- **Generation settings.** max_new_tokens per cell as tabulated; "
        f"temperature {plan.get('temperature')}, top_p {plan.get('top_p')}, top_k "
        f"{plan.get('top_k')}, seed {plan.get('seed')}. Every engine is asked for a "
        "fixed number of tokens with early stopping suppressed, so no engine can appear "
        "faster by generating less.",
        "- **Threads.** "
        + (f"Pinned to {threads} for every engine, and set in OMP, MKL, OpenBLAS, "
           "NumExpr and torch before any library initializes."
           if threads else
           "Inherited from the environment and *not* controlled; the recorded value "
           "per engine is in the raw results, and CPU comparisons should be read with "
           "that in mind."),
        "- **Accelerators.** "
        + (f"Each engine saw {plan.get('devices')} device(s), enforced by restricting "
           "visibility in the worker before any CUDA context was created. No engine's "
           "placement logic was modified; each simply found that many devices. This is "
           "what stops a runtime that shards from being measured on more hardware than "
           "one that does not."
           if plan.get("devices") else
           "Device visibility was *not* restricted, so a runtime that shards may have "
           "used more hardware than one that does not. Each engine's device record is "
           "in the raw results."),
        f"- **Iterations.** {plan.get('warmup_iters')} warm-up iterations, executed and "
        f"discarded, then {plan.get('measure_iters')} measured iterations per cell. "
        "The first unwarmed call is reported separately as cold latency and is never "
        "averaged into steady state.",
        "",
        "**How each measurement is taken**",
        "",
        "- Every engine runs in its own process. One engine's failure, memory "
        "reservation or crash cannot affect another's numbers, and peak process memory "
        "is attributable to exactly one engine.",
        "- Only one worker runs at a time. Concurrent workers would contend for cores, "
        "memory bandwidth and the device, and every number would measure the contention.",
        "- Engine order is rotated per model, so a host that drifts thermally over a "
        "long run does not always penalize the same engine. The order used is recorded.",
        "- CUDA is synchronized on both edges of every timed region, so no asynchronous "
        "kernel is attributed to the wrong phase.",
        "- Device telemetry is sampled in dedicated extra iterations, never during the "
        "iterations whose latency is reported.",
        "- Download, build/compile, load, warm-up and steady state are timed as separate "
        "phases and never combined into one figure.",
        "",
        "**What each metric means**",
        "",
        _table(
            ["Metric", "Definition"],
            [
                [analysis["primary_metric_label"],
                 "generated tokens times batch width, divided by the wall time of the "
                 "whole generation call including prefill. Defined for every engine, "
                 "which is why headline comparisons use it."],
                ["decode tok/s",
                 "measured end-to-end latency minus measured prefill, over the tokens "
                 "the decode loop produced. Carries the uncertainty of both "
                 "measurements, and is only reported where the engine exposes a prefill "
                 "path."],
                ["prompt tok/s", "prompt tokens divided by measured prefill time."],
                ["TTFT", "time until the first token is available to a caller. The "
                 "method differs by engine (a real token stream, or a one-token "
                 "generation call) and each engine's method is stated."],
                ["TPOT / ITL", "decode seconds divided by the tokens the decode loop "
                 "produced; the average gap between consecutive tokens."],
                ["scaling efficiency",
                 "batch-N throughput divided by (batch-1 throughput times N), as a "
                 "percentage. 100% is perfectly linear."],
                ["per-request tok/s",
                 "aggregate throughput divided by batch width: what one caller inside "
                 "the batch experiences."],
                ["peak host / device memory",
                 "process resident set size sampled during inference, and the "
                 "allocator's peak reserved bytes."],
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def taxonomy_section(payload: dict[str, Any]) -> str:
    """Classify each system accurately, and say what its build phase leaves behind.

    The distinction matters for reading everything that follows: three of these are
    not compilers at all, and calling them one would make the compile-once section
    meaningless.
    """
    catalogue = payload.get("engine_catalogue") or {}
    rows = []
    for key, spec in catalogue.items():
        rows.append([
            f"`{key}`",
            spec.get("display", key),
            ", ".join(spec.get("taxonomy") or []),
            "yes" if spec.get("has_build_phase") else "no",
            str(spec.get("artifact_persistence")),
        ])
    return "\n".join([
        "## What each system is",
        "",
        "Not all of these are compilers, and the ones that compile do not all leave the "
        "same thing behind. The last column is the basis of the compile-once section: "
        "`portable-artifact` means a file or directory another process can load; "
        "`on-disk-cache` means machine-local generated code keyed to the library, the "
        "device and the graph; `process-local` means the work is redone on every start; "
        "`none` means there is no build phase at all.",
        "",
        _table(["Key", "System", "Classification", "Build phase", "What the build leaves"],
               rows),
        "",
    ])


def environment_section(payload: dict[str, Any]) -> str:
    from benchmark.reporting import environment_section as base_section

    hardware = payload.get("hardware") or {}
    plan = payload.get("plan") or {}
    extra = _table(
        ["Property", "Value"],
        [
            ["Accelerator class", str(hardware.get("accelerator"))],
            ["Accelerators present", _fmt(hardware.get("gpu_count"))],
            ["Compute capability",
             ", ".join(hardware.get("compute_capabilities") or []) or "-"],
            ["bf16 tensor cores (capability >= 8.0)",
             _fmt(hardware.get("bf16_native"))],
            ["Native fp16 on this device", _fmt(hardware.get("fp16_native"))],
            ["Benchmark precision", str(plan.get("resolved_precision"))],
            ["Precision chosen because", str(plan.get("precision_reason"))],
            ["Threads pinned to", _fmt(plan.get("threads"))],
            ["Accelerators visible per engine",
             _fmt(plan.get("devices")) if plan.get("devices") else "unrestricted"],
            ["Native bf16 per torch", _fmt(hardware.get("torch_reports_bf16"))],
            ["Suite version", str(payload.get("suite_version"))],
            ["Invocation", f"`{plan.get('invocation') or 'not recorded'}`"],
        ],
    )
    return "\n".join([
        base_section(payload.get("environment") or {}),
        "",
        "**Benchmark-controlled environment**",
        "",
        extra,
        "",
    ])


def compatibility_section(analysis: dict[str, Any]) -> str:
    """Which engines ran, which did not, and the reason for each that did not."""
    rows = []
    for entry in analysis["compatibility"]:
        rows.append([
            f"`{entry['engine']}`",
            _short(str(entry.get("model"))),
            entry["status"],
            _fmt(entry.get("version")),
            f"{entry['cells_measured']}/{entry['cells_attempted']}",
            str(entry.get("representation") or "—"),
            _fmt(entry.get("tokenizer_identical")),
            str(entry.get("reason") or "")[:110],
        ])
    return "\n".join([
        "## Engine compatibility",
        "",
        "A status other than MEASURED is a fact about this host or this model, not a "
        "performance result. `NOT_APPLICABLE` means the engine could never run here; "
        "`NOT_INSTALLED` means it is absent; `NOT_SUPPORTED` means it is present but "
        "cannot execute this configuration; `FAILED` means it tried and raised. None of "
        "them is a zero.",
        "",
        _table(
            ["Engine", "Model", "Status", "Version", "Cells", "Representation",
             "Tokenizer matches", "Reason"],
            rows,
        ),
        "",
        _execution_parity_table(analysis),
    ])


def _execution_parity_table(analysis: dict[str, Any]) -> str:
    """What each engine actually executed on, beside what it was asked to.

    The controlled variables are set once for the whole run, but an engine can decline
    one: a plugin with no driver for the accelerator falls back to the host CPU, and a
    plugin that cannot execute a precision widens it. Both are properties of the engine
    rather than of the configuration, and neither is visible in a throughput number - a
    row that ran on the CPU at fp32 just looks slow. Printing the requested and the
    reported value side by side is what turns that into a readable fact, and it is the
    evidence behind every DEVICE_DIFFERENCE label in the comparisons.
    """
    seen: dict[str, list[Any]] = {}
    for entry in analysis["compatibility"]:
        if entry.get("status") != status_mod.MEASURED:
            continue
        engine = str(entry.get("engine"))
        seen.setdefault(engine, [
            f"`{engine}`",
            str(entry.get("execution_device") or "not reported"),
            str(entry.get("execution_device_class") or "not reported"),
            str(entry.get("precision_requested") or "not reported"),
            str(entry.get("precision_executed") or "not reported"),
            _fmt(entry.get("threads")),
            str(entry.get("ttft_method") or "not reported").replace("_", " "),
        ])
    if not seen:
        return ""
    return "\n".join([
        "**What each engine executed on**",
        "",
        "Requested beside reported. A difference in either device class or precision is "
        "why a comparison involving that engine is labelled rather than quoted: the "
        "percentage would be describing two machines, or two numerical formats, instead "
        "of two stacks. The last column is the stopwatch behind that engine's "
        "time-to-first-token figure, because an engine with no stream to subscribe to "
        "is not measured by the same instrument as one that has.",
        "",
        _table(
            ["Engine", "Execution device", "Device class", "Precision asked for",
             "Precision reported", "CPU threads", "TTFT method"],
            [seen[engine] for engine in sorted(seen)],
        ),
        "",
    ])


def model_sections(payload: dict[str, Any], analysis: dict[str, Any],
                   charts: dict[str, Any]) -> str:
    """One section per model: what it is, then every measurement taken on it."""
    parts = ["## Model-by-model results", ""]
    written = set(charts.get("written") or [])
    off_charter = set(payload.get("harness_validation_only") or [])
    if off_charter:
        parts += [
            "> **Harness validation entries.** "
            + ", ".join(f"`{name}`" for name in sorted(off_charter))
            + " are not benchmark models. They were run to exercise the measurement "
            "pipeline, and their numbers describe this harness rather than the "
            "hardware or the engines. They must not be quoted as results.",
            "",
        ]
    for model_id, entry in (payload.get("models") or {}).items():
        facts = entry.get("facts") or {}
        if entry.get("status") == status_mod.FAILED:
            parts += [
                f"### {model_id}",
                "",
                f"**Not measured.** {entry.get('reason')}",
                "",
                "No engine was run on this model, so it contributes nothing to any "
                "comparison, ranking or figure below.",
                "",
            ]
            continue
        parts += [
            f"### {model_id}",
            "",
            _table(
                ["Property", "Value"],
                [
                    ["Model id", f"`{model_id}`"],
                    ["Revision", _fmt(facts.get("revision"))],
                    ["Architecture", _fmt(facts.get("architecture"))],
                    ["Parameters", _fmt(facts.get("parameters"))],
                    ["Layers / hidden / heads / KV heads",
                     f"{_fmt(facts.get('layers'))} / {_fmt(facts.get('hidden_size'))} / "
                     f"{_fmt(facts.get('attention_heads'))} / {_fmt(facts.get('kv_heads'))}"],
                    ["Vocabulary", _fmt(facts.get("vocab_size"))],
                    ["Declared context length", _fmt(facts.get("context_length"))],
                    ["Checkpoint dtype", _fmt(facts.get("checkpoint_dtype"))],
                    ["Tokenizer", _fmt(facts.get("tokenizer_class"))],
                    ["Tokenizer vocabulary", _fmt(facts.get("tokenizer_vocab_size"))],
                    ["Fits smallest visible device", _fmt(entry.get("fits_smallest_device"))],
                    ["Engine order used", ", ".join(entry.get("engine_order") or [])],
                ],
            ),
            "",
            "#### Batch 1 - the single-request case",
            "",
            _batch1_table(analysis, model_id),
            "",
            "#### Batch scaling",
            "",
            _batch_table(analysis, model_id),
            "",
            "#### Prompt-length and output-length sweeps",
            "",
            _sweep_table(analysis, model_id),
            "",
            "#### Aether against each engine on this model",
            "",
            _model_comparison_table(analysis, model_id),
            "",
        ]
        figures = [name for name in written if _short(model_id) in name]
        if figures:
            parts += [f"![{name}](../{charts.get('directory', 'graphs')}/{name})"
                      for name in sorted(figures)]
            parts += [""]
    return "\n".join(parts)


def _batch1_table(analysis: dict[str, Any], model_id: str) -> str:
    rows = []
    candidates = [
        row for row in analysis["rows"]
        if row.get("model") == model_id and row.get("is_primary")
    ]
    candidates.sort(
        key=lambda row: -(row.get(analysis["primary_metric"]) or 0.0)
    )
    for row in candidates:
        if not status_mod.is_measured(row):
            rows.append([f"`{row['engine']}`", row["status"], "—", "—", "—", "—", "—",
                         "—", "—", str(row.get("reason") or "")[:60]])
            continue
        stats = row.get("latency_stats") or {}
        rows.append([
            f"`{row['engine']}`",
            "MEASURED",
            _fmt(row.get(analysis["primary_metric"])),
            _fmt(row.get("decode_tokens_per_s")),
            _fmt(row.get("prompt_tokens_per_s")),
            _seconds(row.get("ttft_s")),
            _ms(row.get("tpot_ms")),
            _seconds(row.get("end_to_end_latency_s")),
            _bytes(row.get("peak_device_bytes") or row.get("peak_host_bytes")),
            f"n={_fmt(stats.get('n'))} cv={_fmt(stats.get('coefficient_of_variation'), 3)}",
        ])
    return _table(
        ["Engine", "Status", "tok/s", "decode tok/s", "prompt tok/s", "TTFT", "TPOT",
         "latency", "peak memory", "samples"],
        rows,
    )


def _batch_table(analysis: dict[str, Any], model_id: str) -> str:
    """Batch widths as measured, with both throughput figures side by side.

    Aggregate throughput and per-request throughput are printed together on purpose.
    Batching raises the first and does not raise the second; showing only the first
    would let a reader mistake a scheduling gain for a latency gain.
    """
    rows = []
    entries = [
        entry for entry in analysis["batch_scaling"] if entry["model"] == model_id
    ]
    entries.sort(key=lambda entry: (entry["engine"], entry["batch_size"]))
    for entry in entries:
        if not status_mod.is_measured(entry):
            rows.append([f"`{entry['engine']}`", str(entry["batch_size"]),
                         entry["status"], "—", "—", "—", "—",
                         str(entry.get("reason") or "")[:50]])
            continue
        rows.append([
            f"`{entry['engine']}`",
            str(entry["batch_size"]),
            "SUPPORTED",
            _fmt(entry.get("batch_tokens_per_s")),
            _fmt(entry.get("per_request_tokens_per_s")),
            _ratio(entry.get("scaling_vs_batch1")),
            (f"{entry['scaling_efficiency_percent']:.0f}%"
             if entry.get("scaling_efficiency_percent") is not None else "—"),
            _seconds(entry.get("end_to_end_latency_s")),
        ])
    return _table(
        ["Engine", "Batch", "Support", "batch tok/s", "per-request tok/s",
         "vs batch 1", "scaling efficiency", "latency"],
        rows,
    )


def _sweep_table(analysis: dict[str, Any], model_id: str) -> str:
    rows = []
    for row in analysis["rows"]:
        if row.get("model") != model_id or row.get("batch_size") != 1:
            continue
        sweeps = row.get("sweeps") or []
        if not ({"prompt", "output"} & set(sweeps)):
            continue
        rows.append([
            f"`{row['engine']}`",
            str(row.get("prompt_tokens")),
            str(row.get("output_tokens")),
            row["status"],
            _fmt(row.get(analysis["primary_metric"])) if status_mod.is_measured(row) else "—",
            _seconds(row.get("end_to_end_latency_s")) if status_mod.is_measured(row) else "—",
            _seconds(row.get("cold_latency_s")) if status_mod.is_measured(row) else "—",
        ])
    rows.sort(key=lambda item: (item[0], int(item[1]), int(item[2])))
    return _table(
        ["Engine", "Prompt tokens", "Output tokens", "Status", "tok/s", "latency",
         "cold first call"],
        rows,
    )


def _model_comparison_table(analysis: dict[str, Any], model_id: str) -> str:
    """Every engine on this model against the fastest engine in the same cell.

    The comparison is against whichever engine was quickest in that cell, so the
    reference changes per row and no engine is permanently the yardstick. The fastest
    engine's own row compares it against the runner-up, which is the only comparison
    that tells you anything about it.
    """
    rows = []
    grouped: dict[tuple[int, int, int], dict[str, float]] = {}
    for row in analysis["rows"]:
        if row.get("model") != model_id or row.get("batch_size") is None:
            continue
        if not status_mod.is_measured(row) or not row.get(analysis["primary_metric"]):
            continue
        key = (int(row["batch_size"]), int(row["prompt_tokens"]),
               int(row["output_tokens"]))
        grouped.setdefault(key, {})[row["engine"]] = float(
            row[analysis["primary_metric"]]
        )
    for key in sorted(grouped):
        ranked = sorted(grouped[key].items(), key=lambda pair: -pair[1])
        for position, (engine, value) in enumerate(ranked, start=1):
            # Compare against the leader, except for the leader itself, which is
            # compared against the engine immediately behind it.
            reference_engine, reference = ranked[1] if position == 1 else ranked[0]
            difference = (value - reference) / reference * 100.0 if reference else None
            rows.append([
                f"b{key[0]} p{key[1]} o{key[2]}",
                str(position),
                f"`{engine}`",
                _fmt(value),
                f"`{reference_engine}`",
                _pct(difference),
            ])
    if not rows:
        return "_No engine produced a comparable measurement on this model._\n"
    return _table(
        ["Cell", "Rank", "Engine", "tok/s", "Compared against", "Difference"],
        rows,
    )


def headline_tables(analysis: dict[str, Any]) -> str:
    """Who won each cell, and the full ordered pairwise matrix.

    The first table names the fastest engine in every cell and the margin over the
    runner-up: the neutral answer to "who is fastest here". The second is the complete
    matrix - every engine's median difference against every other - so no pairing is
    privileged and none is omitted. A dash means the pairing was never comparable,
    which is not the same as a zero.
    """
    metric = analysis["primary_metric"]
    grouped: dict[tuple[str, int, int, int], dict[str, float]] = {}
    for row in analysis["rows"]:
        if row.get("batch_size") is None or not status_mod.is_measured(row):
            continue
        if not row.get(metric):
            continue
        key = (row["model"], int(row["batch_size"]), int(row["prompt_tokens"]),
               int(row["output_tokens"]))
        grouped.setdefault(key, {})[row["engine"]] = float(row[metric])

    winner_rows = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key].items(), key=lambda pair: -pair[1])
        leader, best = ranked[0]
        runner_up, second = ranked[1] if len(ranked) > 1 else (None, None)
        margin = (best - second) / second * 100.0 if second else None
        winner_rows.append([
            _short(key[0]),
            f"b{key[1]}",
            f"p{key[2]}/o{key[3]}",
            f"**{leader}**",
            _fmt(best),
            f"`{runner_up}`" if runner_up else "—",
            _fmt(second),
            _pct(margin) if margin is not None else "—",
            str(len(ranked)),
        ])

    engines = analysis.get("engines_measured") or []
    matrix_rows = []
    for engine in engines:
        entry = (analysis["per_engine"].get(engine) or {}).get("per_competitor") or {}
        row = [f"`{engine}`"]
        for other in engines:
            if other == engine:
                row.append("—")
                continue
            record = entry.get(other)
            row.append(
                _pct(record["median_improvement_percent"]) if record else "—"
            )
        matrix_rows.append(row)

    return "\n".join([
        "## Who won each cell",
        "",
        "The fastest engine in every measured configuration, and how far ahead of the "
        "runner-up it was. `engines` is how many produced a comparable measurement "
        "there - a cell with fewer engines is an easier cell to lead.",
        "",
        _table(
            ["Model", "Batch", "Workload", "Fastest", "tok/s", "Runner-up", "tok/s",
             "Margin", "Engines"],
            winner_rows,
        ),
        "",
        "### Pairwise matrix",
        "",
        "Median difference of the row engine against the column engine, over every "
        "cell both measured. Positive means the row engine was faster. Every pairing "
        "appears in both directions, so the matrix is anti-symmetric by construction "
        "and no engine holds a privileged position in it.",
        "",
        _table(["Engine", *[f"vs {name}" for name in engines]], matrix_rows),
        "",
    ])


def win_loss_section(analysis: dict[str, Any]) -> str:
    """Head-to-head results, one subsection per engine, in identical detail.

    Every measured engine gets the same treatment: its record against the field, the
    cells it won, the cells it lost, the ties, and the pairings that could not be made.
    The sections are generated by one loop over the engine list, so there is no way for
    one engine to receive a fuller or a kinder accounting than another.
    """
    engines = analysis.get("engines_measured") or []
    focus = analysis.get("focus")
    shown = [focus] if focus and focus in engines else engines
    parts = [
        "## Head-to-head results",
        "",
        f"A difference of at most {analysis['tie_threshold_percent']:.0f}% is reported "
        "as a tie, because run-to-run variation at these iteration counts is of that "
        "order and a smaller gap is not evidence of a difference.",
        "",
        "Each engine below is scored against every other engine it shared a measured "
        "cell with. The same code produces every subsection.",
        "",
    ]
    if focus and focus in engines:
        parts += [
            f"_Only `{focus}` is shown, because the run was invoked with "
            f"`--focus {focus}`. Every other engine's record is in the standings table "
            "and in `benchmark_comparisons.csv`._",
            "",
        ]
    if not shown:
        parts += ["_No engine produced a comparable measurement._", ""]
        return "\n".join(parts)
    for engine in shown:
        parts += _engine_head_to_head(analysis, engine)
    return "\n".join(parts)


def _engine_head_to_head(analysis: dict[str, Any], engine: str) -> list[str]:
    """One engine's full record: wins, losses, ties, gaps, aggregated opponents."""
    view = analysis["per_engine"].get(engine) or {}
    win_loss = view.get("win_loss") or {}
    everything = win_loss.get("all") or {}
    same = win_loss.get("same_representation") or {}
    different = win_loss.get("representation_difference") or {}
    cross_device = win_loss.get("device_difference") or {}
    cross_work = win_loss.get("work_difference") or {}

    def render(entries: list[dict[str, Any]]) -> str:
        rows = [
            [
                _short(item["model"]),
                str(item["batch_size"]),
                f"p{item['prompt_tokens']}/o{item['output_tokens']}",
                f"`{item['competitor']}`",
                _fmt(item["throughput"]["subject"]),
                _fmt(item["throughput"]["other"]),
                _pct(item["throughput"]["subject_improvement_percent"]),
                _comparable_cell(item["comparability"]),
            ]
            for item in sorted(
                entries,
                key=lambda item: -item["throughput"]["subject_improvement_percent"],
            )
        ]
        return _table(
            ["Model", "Batch", "Workload", "Opponent", f"{engine} tok/s",
             "Opponent tok/s", "Difference", "Like for like?"],
            rows,
        )

    parts = [
        f"### `{engine}`",
        "",
        _table(
            ["Comparison set", "Compared", "Won", "Lost", "Tied"],
            [
                ["All valid comparisons", str(everything.get("compared", 0)),
                 str(everything.get("wins", 0)), str(everything.get("losses", 0)),
                 str(everything.get("ties", 0))],
                ["Same representation (supports execution claims)",
                 str(same.get("compared", 0)), str(same.get("wins", 0)),
                 str(same.get("losses", 0)), str(same.get("ties", 0))],
                ["Representation differs (does not)", str(different.get("compared", 0)),
                 str(different.get("wins", 0)), str(different.get("losses", 0)),
                 str(different.get("ties", 0))],
                ["Hardware differs (does not)", str(cross_device.get("compared", 0)),
                 str(cross_device.get("wins", 0)), str(cross_device.get("losses", 0)),
                 str(cross_device.get("ties", 0))],
                ["Work produced differs (does not)", str(cross_work.get("compared", 0)),
                 str(cross_work.get("wins", 0)), str(cross_work.get("losses", 0)),
                 str(cross_work.get("ties", 0))],
            ],
        ),
        "",
        f"- Largest advantage: {_extreme(win_loss, 'largest_advantage', engine)}.",
        f"- Largest disadvantage: {_extreme(win_loss, 'largest_disadvantage', engine)}.",
        f"- Strongest model: {_best_model(view.get('comparisons') or [], engine)}.",
        f"- Memory: {_memory_answer(view.get('comparisons') or [], engine)}.",
        "",
        f"#### Cells `{engine}` won",
        "",
        render(everything.get("won") or []) if everything.get("won") else "_No cell._\n",
        "",
        f"#### Cells `{engine}` lost",
        "",
        render(everything.get("lost") or []) if everything.get("lost") else "_No cell._\n",
        "",
        f"#### Statistical ties for `{engine}`",
        "",
        render(everything.get("tied") or []) if everything.get("tied") else "_No cell._\n",
        "",
        f"#### `{engine}` against each opponent, aggregated",
        "",
        _per_competitor_table(view.get("per_competitor") or {}, engine),
        "",
    ]
    incomparable = win_loss.get("incomparable") or []
    if incomparable:
        parts += [
            f"#### Pairings `{engine}` could not be compared on",
            "",
            _table(
                ["Model", "Batch", "Opponent", "Reason"],
                [[_short(item["model"]), str(item["batch_size"]),
                  f"`{item['competitor']}`", item["reason"]] for item in incomparable],
            ),
            "",
        ]
    return parts


def _per_competitor_table(per_competitor: dict[str, Any], engine: str) -> str:
    rows = []
    for opponent, entry in sorted(per_competitor.items()):
        rows.append([
            f"`{opponent}`",
            str(entry["cells"]),
            _pct(entry["median_improvement_percent"]),
            _pct(entry.get("median_improvement_percent_same_representation")),
            f"{_pct(entry['min_improvement_percent'])} to "
            f"{_pct(entry['max_improvement_percent'])}",
            f"{entry['wins']}/{entry['losses']}/{entry['ties']}",
            _comparable_cell(entry["comparability"]),
        ])
    return _table(
        ["Opponent", "Cells", f"Median {engine} difference",
         "Median, same representation only", "Range", "W/L/T", "Like for like?"],
        rows,
    )


def compile_section(analysis: dict[str, Any]) -> str:
    """Build costs, what they buy, and whether a second process gets them for free.

    This is the compile-once question answered with measurements: for every engine
    that claims a persistent build, a *separate process* was asked to load the artifact
    and run. An engine that has no build phase, or whose build is process-local, says
    so in the same table.
    """
    economics = analysis["compile_economics"]
    rows = []
    for entry in economics["entries"]:
        rows.append([
            f"`{entry['engine']}`",
            _short(str(entry["model"])),
            str(entry.get("spec_persistence")),
            _seconds(entry.get("build_s"), 1),
            _seconds(entry.get("load_s"), 1),
            _seconds(entry.get("total_s"), 1),
            _bytes(entry.get("artifact_bytes")),
            (_seconds(entry.get("second_process_load_s"), 2)
             if entry.get("second_process_load_s") is not None
             else str(entry.get("reuse_status") or "—")),
            _seconds(entry.get("first_inference_after_reload_s"), 2),
        ])
    parts = [
        "## Compilation economics",
        "",
        "Steady-state throughput excludes every cost in this section, and this section "
        "excludes steady-state throughput. Mixing them is how a compiled runtime gets "
        "credited with speed it only reaches after a cost the reader was not shown.",
        "",
        "**Second process** is the measurement that settles compile-once-use-everywhere: "
        "a brand-new OS process, holding nothing but what the first process wrote to "
        "disk, loading the artifact and running once.",
        "",
        _table(
            ["Engine", "Model", "Persistence", "Build", "Load", "Total start-up",
             "Artifact size", "Second process load", "First inference after reload"],
            rows,
        ),
        "",
        "### Break-even: how many requests justify the build",
        "",
        "Solved from the measured start-up costs and the measured per-request latency: "
        "with a fixed start-up S and per-request cost L on each side, the totals cross "
        "at N = (S_row - S_column) / (L_column - L_row). Computed for every ordered "
        "pair, so the question can be asked of any engine. A blank means the row engine "
        "was not faster per request there, so no number of requests would repay its "
        "build.",
        "",
        _table(
            ["Model", "Engine", "Opponent", "Engine start-up", "Opponent start-up",
             "Engine latency", "Opponent latency", "Break-even requests", "Reading"],
            [
                [
                    _short(item["model"]),
                    f"`{item['subject']}`",
                    f"`{item['competitor']}`",
                    _seconds(item["subject_startup_s"], 1),
                    _seconds(item["competitor_startup_s"], 1),
                    _seconds(item["subject_latency_s"]),
                    _seconds(item["competitor_latency_s"]),
                    (f"{item['break_even_runs']:.0f}"
                     if item.get("break_even_runs") is not None else "—"),
                    item["interpretation"],
                ]
                for item in economics["break_even"]
            ],
        ),
        "",
        "### Total cost of N requests",
        "",
        "Start-up plus N inferences, at each run count. The `warm` column uses the "
        "measured second-process artifact load instead of the build, and is blank for "
        "engines with nothing to reuse.",
        "",
        _total_cost_table(economics),
        "",
    ]
    return "\n".join(parts)


def _total_cost_table(economics: dict[str, Any]) -> str:
    counts = [str(count) for count in economics["run_counts"]]
    headers = ["Engine", "Model"]
    for count in counts:
        headers += [f"N={count} cold", f"N={count} warm"]
    rows = []
    for entry in economics["entries"]:
        totals = entry.get("total_cost_s")
        if not totals:
            continue
        row = [f"`{entry['engine']}`", _short(str(entry["model"]))]
        for count in counts:
            cell = totals.get(count) or {}
            row.append(_seconds(cell.get("cold_first_deployment"), 1))
            row.append(_seconds(cell.get("warm_reused_artifact"), 1))
        rows.append(row)
    return _table(headers, rows)


def correctness_section(analysis: dict[str, Any]) -> str:
    """Whether the engines computed the same thing, and how the difference is classed.

    Floating-point difference between two implementations of the same mathematics is
    expected, and an engine is not failed for it. What the classes will not do is call
    two genuinely different completions equivalent.
    """
    correctness = analysis["correctness"]
    rows = []
    for case in correctness["cases"]:
        tokens = case.get("tokens") or {}
        text = case.get("text") or {}
        rows.append([
            f"`{case['engine']}`",
            _short(case["model"]),
            case["classification"],
            str(case.get("basis") or "—"),
            _fmt(tokens.get("identical")),
            _fmt(text.get("identical")),
            _fmt(tokens.get("matching_prefix_tokens")),
            (f"{float(tokens['matching_prefix_fraction']) * 100:.0f}%"
             if tokens.get("matching_prefix_fraction") is not None else "—"),
            _fmt(tokens.get("first_divergence_index")),
            _fmt(case.get("candidate_completion_tokens")),
            str(case.get("token_ids_source") or "—"),
            str(case.get("reason") or "")[:60],
        ])
    return "\n".join([
        "## Correctness",
        "",
        "Every engine's greedy completion is compared against `"
        f"{correctness['reference_engine']}`, the reference implementation of these "
        "checkpoints, on the same prompt with the same settings.",
        "",
        _table(
            ["Class", "Meaning"],
            [
                ["EXACT_MATCH", "identical token ids, or identical decoded text"],
                ["NUMERICALLY_EQUIVALENT",
                 f"at least {correctness['prefix_agreement_fraction'] * 100:.0f}% of the "
                 "sequence agrees before diverging, which is what two implementations of "
                 "the same mathematics do when a near-tied argmax breaks differently"],
                ["EXPECTED_SAMPLING_DIFFERENCE",
                 "sampling was enabled, so two runs of the same engine would also "
                 "differ and the comparison carries no correctness information"],
                ["DIFFERENT_OUTPUT",
                 "the sequences diverge early and stay diverged: a different "
                 "computation, not rounding"],
                ["FAILURE", "one side produced no output to compare"],
            ],
        ),
        "",
        _table(
            ["Engine", "Model", "Class", "Basis", "Ids identical", "Text identical",
             "Matching prefix", "Prefix fraction", "First divergence",
             "Tokens produced", "Id source", "Note"],
            rows,
        ),
        "",
        "An engine whose ids were re-encoded from decoded text is marked as such: for "
        "those rows a difference in ids can come from the round trip rather than from "
        "the model, which is why the decoded-text comparison is reported beside it.",
        "",
    ])


def statistics_section(analysis: dict[str, Any]) -> str:
    statistics = analysis["statistics"]
    flagged = [entry for entry in statistics["entries"] if entry["outlier_suspected"]]
    noisy = [entry for entry in statistics["entries"] if entry["dispersion_flag"] == "high"]
    rows = []
    for entry in statistics["entries"]:
        rows.append([
            f"`{entry['engine']}`",
            _short(entry["model"]),
            f"b{entry['batch_size']} p{entry['prompt_tokens']} o{entry['output_tokens']}",
            _fmt(entry["iterations"]),
            _seconds(entry["mean_s"], 4),
            _seconds(entry["median_s"], 4),
            _seconds(entry["stdev_s"], 4),
            _seconds(entry["p95_s"], 4),
            _seconds(entry["p99_s"], 4),
            _fmt(entry["coefficient_of_variation"], 3),
            entry["dispersion_flag"],
        ])
    return "\n".join([
        "## Statistical quality",
        "",
        statistics["policy"],
        "",
        f"Cells with high dispersion (coefficient of variation above 10%): "
        f"**{len(noisy)}**. Cells whose range exceeds half their median, and are "
        f"therefore flagged for a possible outlier: **{len(flagged)}**. "
        f"Samples removed: **{statistics['outliers_removed']}**.",
        "",
        _table(
            ["Engine", "Model", "Cell", "n", "mean", "median", "stdev", "p95", "p99",
             "CoV", "Dispersion"],
            rows,
        ),
        "",
    ])


def rankings_section(analysis: dict[str, Any]) -> str:
    """Separate rankings, each labelled with the workload it describes."""
    rankings = analysis["rankings"]
    parts = [
        "## Final rankings",
        "",
        "There is no single ranking here, because there is no single workload. Each "
        "table below names the workload it was measured on, and an engine appears only "
        "if it produced that measurement.",
        "",
    ]
    titles = {
        "batch_1": "Best at batch 1 (single-user, local inference)",
        "peak_throughput": "Best peak throughput at any batch width",
        "ttft": "Lowest time to first token",
        "latency": "Lowest end-to-end latency",
        "host_memory": "Lowest peak host memory",
        "device_memory": "Lowest peak device memory",
        "cold_start": "Fastest cold start (first inference in a fresh process)",
        "batch_scaling": "Best batch scaling efficiency",
    }
    for key, title in titles.items():
        ranking = rankings.get(key) or {}
        order = ranking.get("order") or []
        parts += [f"### {title}", "", f"_{ranking.get('scope', '')}_", ""]
        if not order:
            parts += ["_No engine produced this measurement._", ""]
            continue
        rows = []
        for entry in order:
            value = entry["value"]
            if key in {"host_memory", "device_memory"}:
                rendered = _bytes(value)
            elif key == "batch_scaling":
                rendered = f"{float(value):.0f}% at batch {entry.get('batch_size')}"
            elif key in {"ttft", "latency", "cold_start"}:
                rendered = _seconds(value)
            else:
                rendered = f"{float(value):,.2f} tok/s"
            rows.append([
                str(entry["rank"]),
                f"`{entry['engine']}`",
                _short(str(entry.get("model") or "")),
                rendered,
            ])
        parts += [_table(["Rank", "Engine", "Model", "Value"], rows), ""]
        if ranking.get("mixed_methods"):
            # Ordering figures taken by different instruments is defensible only if the
            # table says which instrument produced each one.
            methods = ranking.get("methods") or {}
            parts += [
                "Measured by more than one method, so this ordering is not "
                "like-for-like: "
                + ", ".join(
                    f"`{engine}` by {str(method or 'an unreported method').replace('_', ' ')}"
                    for engine, method in sorted(methods.items())
                )
                + ".",
                "",
            ]
        missing = ranking.get("not_measured") or []
        if missing:
            parts += [
                f"Not ranked, because no measurement exists for this metric: "
                f"{', '.join(f'`{name}`' for name in missing)}.",
                "",
            ]
    return "\n".join(parts)


def cross_model_section(analysis: dict[str, Any]) -> str:
    """Whether the ordering of the field depends on the configuration.

    Asked of the whole field rather than of one engine: for each slice of the matrix,
    which engine led, how far ahead of the field's median it was, and how wide the
    spread was. An engine that leads everywhere is a general result; one that leads only
    at batch 16, or only on one model, is a conditional one, and this is the section
    that tells them apart.
    """
    import statistics as stats_mod

    metric = analysis["primary_metric"]
    measured = [
        row for row in analysis["rows"]
        if status_mod.is_measured(row) and row.get(metric)
        and row.get("batch_size") is not None
    ]

    def group(field: str) -> list[list[str]]:
        buckets: dict[Any, dict[str, list[float]]] = {}
        for row in measured:
            buckets.setdefault(row[field], {}).setdefault(row["engine"], []).append(
                float(row[metric])
            )
        rows = []
        for key, engines_here in sorted(buckets.items(), key=lambda item: str(item[0])):
            medians = {
                engine: stats_mod.median(values)
                for engine, values in engines_here.items()
            }
            ordered = sorted(medians.items(), key=lambda pair: -pair[1])
            leader, best = ordered[0]
            slowest = ordered[-1][1]
            field_median = stats_mod.median(list(medians.values()))
            rows.append([
                _short(str(key)) if field == "model" else str(key),
                str(len(medians)),
                f"**{leader}**",
                _fmt(best),
                _pct((best / field_median - 1.0) * 100.0 if field_median else None),
                (f"{best / slowest:.2f}x" if slowest else "—"),
            ])
        return rows

    headers = ["Engines", "Fastest", "tok/s (median)", "vs field median",
               "Fastest / slowest"]
    return "\n".join([
        "## Cross-configuration analysis",
        "",
        "Whether the ordering of the field is a property of the engines or of the "
        "configuration. Each row is a slice of the matrix: which engine led it, and how "
        "wide the field was.",
        "",
        "**By model**", "",
        _table(["Model", *headers], group("model")),
        "",
        "**By batch width**", "",
        _table(["Batch", *headers], group("batch_size")),
        "",
        "**By prompt length**", "",
        _table(["Prompt tokens", *headers], group("prompt_tokens")),
        "",
        "**By output length**", "",
        _table(["Output tokens", *headers], group("output_tokens")),
        "",
    ])


#: What this suite cannot settle. Stated so a reader does not extrapolate past what
#: was observed, and so a future run can tell which caveats it has removed.
LIMITATIONS: tuple[str, ...] = (
    "Every engine runs in its own process, which makes peak memory attributable and "
    "cold start real, but it means engine order cannot be alternated within a cell. "
    "Order is rotated per model instead, and the order used is recorded; a host that "
    "drifted sharply during a single model's sequence could still bias that model's "
    "rows.",
    "The headline metric is generated tokens per second over the whole generation "
    "call, prefill included. It is used because it is defined for every engine here. "
    "Decode-only throughput is reported wherever both sides expose a prefill path, and "
    "is never substituted for the headline figure.",
    "Time-to-first-token is obtained through a real token stream on some engines and "
    "by timing a one-token generation on others, because not every stack exposes a "
    "stream. Each engine's method is printed; the two are not the same machinery and "
    "TTFT is therefore a weaker comparison than throughput.",
    "OpenVINO executes a re-exported representation: the checkpoint is converted to "
    "OpenVINO IR and the weights are stored at the run's precision on the way out. It "
    "is the same numerical width as the framework engines hold, in a different "
    "container, and every percentage derived from a row whose storage width differs is "
    "labelled REPRESENTATION_DIFFERENCE rather than quoted as a difference between "
    "the two stacks.",
    "OpenVINO ships no CUDA plugin. On an NVIDIA host it can only execute on the CPU, "
    "while the three torch-based engines execute on the GPU, and no flag changes that: "
    "it is a property of the runtime, not of this configuration. Its rows are measured "
    "and it keeps its rank, but every pairing that crosses that boundary is labelled "
    "DEVICE_DIFFERENCE and counted separately in the standings, because the "
    "percentage describes two machines rather than two stacks. On an Intel host, where "
    "OpenVINO has a GPU plugin, --openvino-device GPU removes the caveat and the "
    "labelling follows the device the plugin reports back.",
    "The thread budget is applied to every engine by the same mechanism it responds "
    "to: the torch engines through the environment, OpenVINO through "
    "INFERENCE_NUM_THREADS, because its scheduler ignores the environment variable the "
    "others read. Without that, one engine would silently get every core while the "
    "rest were held to the pinned count.",
    "Comparability is judged on compute precision and on weight storage width. Two "
    "engines at the same compute precision holding the same 16-bit width in different "
    "containers - fp16 tensors against Aether's bf16 artifact, both derived from the "
    "same published bf16 checkpoint - are treated as comparable, with the storage "
    "difference printed next to every such comparison. That is a judgement, not a "
    "measurement: each side is one rounding step from the checkpoint, in a different "
    "direction, and a reader who wants the bit-exact configuration should run "
    "--precision bf16 on hardware with bf16 tensor cores.",
    "The benchmark precision is the widest 16-bit format the whole field can execute "
    "on the detected device. On pre-Ampere CUDA that is fp16, because engines in this "
    "field refuse bf16 below compute capability 8.0 - choosing bf16 there would exclude "
    "them rather than measure them. The consequence is that a pre-Ampere run is not the "
    "weight-exact configuration; --precision bf16 is, at the cost of those engines.",
    "Every engine sees the same number of accelerators, one by default, enforced by "
    "restricting device visibility in each worker rather than by altering any engine's "
    "placement logic. Equal visibility is not equal use: an engine with no plugin for "
    "the device present cannot execute on it, which is what the per-engine execution "
    "device column reports. A runtime that would shard across several devices therefore runs "
    "single-device here. That is what makes the comparison a comparison of engines; it "
    "is also not a measurement of what that runtime can do with more hardware, which "
    "this run does not test. Raise --devices to measure that deliberately.",
    "Aether's semantic response cache is disabled for measurement, through a public "
    "configuration flag, because the benchmark issues one prompt repeatedly and the "
    "cache would return a stored answer instead of running inference. The override is "
    "recorded per engine; no other default is changed, and no other engine in this "
    "field has a cache that would fire on a repeated prompt.",
    "Generation length is pinned rather than trusted: the three engines that expose a "
    "minimum-new-tokens control are given one, so an early stop cannot shorten a "
    "measured cell. Aether's public generate has no such control, so a cell where it "
    "stopped early is detected after the fact from the token count and labelled "
    "WORK_DIFFERENCE, which keeps it out of every percentage that claims to describe "
    "execution.",
    "Build costs are measured once per engine per model on this machine. A first-time "
    "compile on a shared host includes filesystem and network variance that a "
    "repeated compile would not, so a single build time is a weaker measurement than "
    "the steady-state figures, which are medians over many iterations.",
    "The second-process artifact reload demonstrates that a build can be reused on "
    "this machine. It does not demonstrate portability to a different machine, a "
    "different accelerator or a different library version, none of which this run "
    "tested.",
    "Iteration counts are bounded by the available compute budget. The dispersion "
    "statistics state how much confidence the sample size supports, and cells with "
    "high variation are flagged rather than smoothed.",
    "A shared or virtualized host (Kaggle, Colab, most CI) does not give the benchmark "
    "control of clocks or thermal state. Recorded temperatures show what happened "
    "rather than asserting parity.",
    "Correctness is compared on one prompt per model at greedy settings. It "
    "establishes that the engines compute the same thing on that input; it is not a "
    "quality evaluation and says nothing about downstream task accuracy.",
)


def failures_section(payload: dict[str, Any]) -> str:
    """Everything that did not work, with the reason, including worker crashes."""
    rows = []
    for run in payload.get("runs", []):
        if run.get("status") == status_mod.MEASURED:
            for cell in run.get("cells") or []:
                if cell.get("status") != status_mod.MEASURED:
                    rows.append([
                        f"`{run['engine']}`", _short(str(run["model"])),
                        f"b{cell.get('batch_size')} p{cell.get('prompt_tokens')}"
                        f" o{cell.get('output_tokens')}",
                        str(cell.get("status")), str(cell.get("reason") or "")[:160],
                    ])
            continue
        rows.append([
            f"`{run.get('engine')}`", _short(str(run.get("model"))), "whole engine",
            str(run.get("status")), str(run.get("reason") or "")[:160],
        ])
    processes = [
        process for process in payload.get("worker_processes", [])
        if process.get("timed_out") or process.get("returncode") not in (0, None)
    ]
    parts = ["## Failures and unavailable configurations", ""]
    parts.append(
        _table(["Engine", "Model", "Configuration", "Status", "Reason"], rows)
        if rows else "_Every attempted configuration produced a measurement._\n"
    )
    if processes:
        parts += [
            "",
            "**Worker processes that did not exit cleanly.** Recorded because a crash "
            "is a result about the engine, not a gap in the data.",
            "",
            _table(
                ["Engine", "Model", "Mode", "Exit", "Timed out", "Elapsed"],
                [
                    [f"`{item.get('engine')}`", _short(str(item.get("model"))),
                     str(item.get("mode")), str(item.get("returncode")),
                     _fmt(item.get("timed_out")), _seconds(item.get("elapsed_s"), 1)]
                    for item in processes
                ],
            ),
        ]
    parts.append("")
    return "\n".join(parts)


def build_report(payload: dict[str, Any], analysis: dict[str, Any],
                 charts: dict[str, Any]) -> str:
    """Assemble the whole document from whichever sections the run supports."""
    plan = payload.get("plan") or {}
    devices = plan.get("devices")
    parts = [
        "# Inference Engine Benchmark Report",
        "",
        f"Generated: {payload.get('generated_at', '—')}  ",
        f"Suite version: {payload.get('suite_version')}  ",
        f"Benchmark precision: {plan.get('resolved_precision')}  ",
        f"Accelerators visible to each engine: {devices if devices else 'unrestricted'} "
        "(visible; the device each engine actually executed on is reported per engine)  ",
        f"Models: {len(plan.get('models') or [])}  "
        f"Engines attempted: {len(plan.get('engines') or [])}",
        "",
        "This report compares inference **stacks** executing the same model "
        "architectures and the same weights. The stacks are not all the same kind of "
        "system - some are frameworks, some are runtimes, some are graph compilers, "
        "some are serving engines - and the section below classifies each one, because "
        "the comparison only means what it says if the reader knows what is being "
        "compared.",
        "",
        "Every engine is scored by the same code. The standings, the rankings and the "
        "pairwise matrix are computed for each engine identically, and no engine "
        "occupies a privileged position in any of them. The suite lives in the Aether "
        "Runtime repository, which is why Aether is in the field; it is not why any "
        "number here comes out the way it does, and where Aether loses the report says "
        "so in the same words it uses when Aether wins.",
        "",
        executive_summary(payload, analysis),
        "",
        taxonomy_section(payload),
        "",
        methodology_section(payload, analysis),
        "",
        environment_section(payload),
        "",
        compatibility_section(analysis),
        "",
        headline_tables(analysis),
        "",
        model_sections(payload, analysis, charts),
        "",
        cross_model_section(analysis),
        "",
        win_loss_section(analysis),
        "",
        compile_section(analysis),
        "",
        correctness_section(analysis),
        "",
        statistics_section(analysis),
        "",
        rankings_section(analysis),
        "",
        failures_section(payload),
        "",
        _charts_section(charts),
        "",
        "## Configuration",
        "",
        "Printed in full so a reader can confirm every engine received identical "
        "settings, and so the run can be repeated exactly.",
        "",
        "```json",
        json.dumps(plan, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Limitations",
        "",
    ]
    parts += [f"{index}. {text}" for index, text in enumerate(LIMITATIONS, start=1)]
    parts += ["", _reproducibility_section(payload), ""]
    return "\n".join(parts)


def _charts_section(charts: dict[str, Any]) -> str:
    written = charts.get("written") or []
    skipped = charts.get("skipped") or []
    directory = charts.get("directory", "graphs")
    parts = ["## Figures", ""]
    if written:
        parts += [
            "Axes start at zero and are linear. A missing point is a missing "
            "measurement: nothing is interpolated, smoothed or zero-filled, and panels "
            "name the engines that had no data for them.",
            "",
        ]
        parts += [f"- [`{name}`](../{directory}/{name})" for name in sorted(written)]
        parts += [""]
    if skipped:
        parts += [
            "",
            "Figures not produced, and why. A figure is skipped when the measurements "
            "it would need do not exist.",
            "",
            _table(["Figure", "Reason"],
                   [[item["chart"], item["reason"]] for item in skipped]),
        ]
    return "\n".join(parts)


def _reproducibility_section(payload: dict[str, Any]) -> str:
    software = ((payload.get("environment") or {}).get("software") or {})
    plan = payload.get("plan") or {}
    return "\n".join([
        "## Reproducibility",
        "",
        _table(
            ["Property", "Value"],
            [
                ["Aether commit", _fmt(software.get("aether_git_commit"))],
                ["Working tree dirty", _fmt(software.get("aether_git_dirty"))],
                ["Aether version", _fmt(software.get("aether_runtime")
                                        or software.get("aether_version_constant"))],
                ["Suite version", str(payload.get("suite_version"))],
                ["Run started", str(payload.get("generated_at"))],
                ["Run finished", str(payload.get("finished_at"))],
                ["Command", f"`{plan.get('invocation') or 'not recorded'}`"],
                ["Workload signature",
                 f"`{json.dumps(payload.get('workload_signature') or {}, sort_keys=True)}`"],
            ],
        ),
        "",
        "Two runs may only be compared when their workload signatures match. Raw "
        "per-engine records, the plan and the prompts are in `raw/`; every number in "
        "this report is derived from those files and nothing else.",
        "",
    ])


#: Columns of the machine-readable table. Fixed and flat, so the CSV can be loaded
#: by anything without a schema, and so a status is always present next to a value.
CSV_COLUMNS: tuple[str, ...] = (
    "model", "engine", "status", "reason", "precision", "representation", "quantized",
    "kind", "sweeps", "is_primary", "batch_size", "prompt_tokens", "output_tokens",
    "completion_tokens", "iterations", "total_tokens_per_s", "per_request_tokens_per_s",
    "decode_tokens_per_s", "prompt_tokens_per_s", "ttft_s", "tpot_ms",
    "end_to_end_latency_s", "cold_latency_s", "latency_median_s", "latency_stdev_s",
    "latency_p95_s", "latency_p99_s", "coefficient_of_variation", "peak_host_bytes",
    "peak_device_bytes", "taxonomy",
)


def write_csv(analysis: dict[str, Any], path: Path) -> str:
    """Write one row per (engine, model, cell), measured or not.

    The unmeasured rows are included with their status and empty metric columns, so a
    spreadsheet built from this file cannot accidentally average a missing engine in
    as a zero.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in analysis["rows"]:
            stats = row.get("latency_stats") or {}
            record = {column: row.get(column) for column in CSV_COLUMNS}
            record.update(
                sweeps="|".join(row.get("sweeps") or []),
                taxonomy="|".join(row.get("taxonomy") or []),
                latency_median_s=stats.get("median"),
                latency_stdev_s=stats.get("stdev"),
                latency_p95_s=stats.get("p95"),
                latency_p99_s=stats.get("p99"),
            )
            writer.writerow(record)
    return path.name


def write_comparison_csv(analysis: dict[str, Any], path: Path) -> str:
    """Write the full pairwise matrix, one row per ordered pair per cell.

    Ordered pairs, so every comparison appears from both sides and the file carries no
    assumption about which engine is the subject of interest. Column names name
    positions, not engines.
    """
    import csv

    columns = (
        "model", "batch_size", "prompt_tokens", "output_tokens", "subject",
        "competitor", "comparability", "comparability_note", "winner",
        "subject_tokens_per_s", "competitor_tokens_per_s",
        "subject_improvement_percent", "subject_latency_improvement_percent",
        "subject_host_memory_improvement_percent",
        "subject_device_memory_improvement_percent",
        "subject_ttft_improvement_percent", "ttft_same_method",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for item in analysis.get("pairwise") or analysis["comparisons"]:
            writer.writerow({
                "model": item["model"],
                "batch_size": item["batch_size"],
                "prompt_tokens": item["prompt_tokens"],
                "output_tokens": item["output_tokens"],
                "subject": item.get("subject"),
                "competitor": item["competitor"],
                "comparability": item["comparability"],
                "comparability_note": item.get("comparability_note"),
                "winner": item["winner"],
                "subject_tokens_per_s": item["throughput"].get("subject"),
                "competitor_tokens_per_s": item["throughput"].get("other"),
                "subject_improvement_percent":
                    item["throughput"].get("subject_improvement_percent"),
                "subject_latency_improvement_percent":
                    item["latency"].get("subject_improvement_percent"),
                "subject_host_memory_improvement_percent":
                    item["host_memory"].get("subject_improvement_percent"),
                "subject_device_memory_improvement_percent":
                    item["device_memory"].get("subject_improvement_percent"),
                "subject_ttft_improvement_percent":
                    item["ttft"].get("subject_improvement_percent"),
                # False where the two rows' first tokens were timed differently, which
                # a reader filtering this file on TTFT needs before the percentage.
                "ttft_same_method": item["ttft"].get("same_method"),
            })
    return path.name


def terminal_summary(payload: dict[str, Any], analysis: dict[str, Any],
                     paths: dict[str, Any]) -> str:
    """The concise summary printed when the run finishes.

    Deliberately the same numbers as the report's executive summary, read from the same
    analysis, so the terminal and the document can never disagree. Every measured engine
    appears; none is singled out.
    """
    rankings = analysis["rankings"]
    standings = analysis["standings"]
    lines = ["", "=" * 78, "BENCHMARK SUMMARY", "=" * 78, ""]

    for label, key, unit in (
        ("Best at batch 1", "batch_1", "tok/s"),
        ("Best peak throughput", "peak_throughput", "tok/s"),
        ("Lowest TTFT", "ttft", "s"),
        ("Lowest latency", "latency", "s"),
        ("Lowest host memory", "host_memory", "bytes"),
        ("Fastest cold start", "cold_start", "s"),
    ):
        order = (rankings.get(key) or {}).get("order") or []
        if not order:
            lines.append(f"  {label:22s} not measured")
            continue
        head = order[0]
        value = head["value"]
        rendered = (
            _bytes(value) if unit == "bytes"
            else f"{float(value):.3f} s" if unit == "s"
            else f"{float(value):,.2f} tok/s"
        )
        mixed = " [mixed methods]" if (rankings.get(key) or {}).get("mixed_methods") else ""
        lines.append(f"  {label:22s} {head['engine']}  ({rendered}){mixed}")

    lines += ["", "  Standings (median share of the fastest engine per cell):", ""]
    if standings:
        lines.append(f"    {'#':>2}  {'engine':18s} {'% of best':>9s}  "
                     f"{'W/L/T':>10s}  {'win rate':>8s}  cells")
        for entry in standings:
            share = (f"{entry['median_percent_of_best']:.0f}%"
                     if entry["median_percent_of_best"] is not None else "-")
            record = f"{entry['wins']}/{entry['losses']}/{entry['ties']}"
            lines.append(
                f"    {entry['rank']:>2}  {entry['engine']:18s} {share:>9s}  "
                f"{record:>10s}  {entry['win_rate_percent']:>7.0f}%  "
                f"{entry['cells_measured']}"
            )
    else:
        lines.append("    no engine produced a comparable measurement")

    lines += ["", "  Pairwise medians (row engine against column engine):", ""]
    engines = analysis.get("engines_measured") or []
    if len(engines) > 1:
        header = "    " + " " * 18 + "".join(f"{name[:9]:>11s}" for name in engines)
        lines.append(header)
        for engine in engines:
            entry = (analysis["per_engine"].get(engine) or {}).get("per_competitor") or {}
            cells = []
            for other in engines:
                if other == engine:
                    cells.append(f"{'-':>11s}")
                    continue
                record = entry.get(other)
                cells.append(
                    f"{record['median_improvement_percent']:>+10.1f}%" if record
                    else f"{'n/a':>11s}"
                )
            lines.append(f"    {engine:18s}" + "".join(cells))
    else:
        lines.append("    fewer than two engines produced a comparable measurement")

    unmeasured = [
        entry for entry in analysis["compatibility"]
        if entry["status"] != status_mod.MEASURED
    ]
    if unmeasured:
        lines += ["", "  Not measured (reason, not a zero):"]
        seen: set[tuple[str, str]] = set()
        for entry in unmeasured:
            key = (entry["engine"], str(entry["status"]))
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"    {entry['engine']:18s} {entry['status']:15s} "
                f"{str(entry.get('reason') or '')[:70]}"
            )

    lines += ["", "  Outputs:"]
    for label, value in paths.items():
        lines.append(f"    {label:22s} {value}")
    lines += ["", "=" * 78, ""]
    return "\n".join(lines)
