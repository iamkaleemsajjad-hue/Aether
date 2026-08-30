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
from benchmark.suite.analysis import SAME_REPRESENTATION, SUBJECT


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
    win_loss = analysis["win_loss"]
    lines = ["## Executive summary", ""]

    def leader(name: str) -> str:
        order = rankings.get(name, {}).get("order") or []
        if not order:
            return "not determined by this run"
        head = order[0]
        value = head["value"]
        rendered = (
            f"{float(value):,.2f}" if isinstance(value, (int, float)) else str(value)
        )
        model = head.get("model")
        return (f"**{head['engine']}** ({rendered}"
                + (f" on {_short(model)}" if model else "") + ")")

    subject_rank = _rank_of(rankings.get("batch_1", {}), SUBJECT)
    peak_rank = _rank_of(rankings.get("peak_throughput", {}), SUBJECT)
    same = win_loss.get("same_representation", {})
    # When no pairing held the same representation - which happens whenever the run's
    # precision differs from Aether's BF16 artifact residency - counting "0 of 0"
    # would read as a result. Fall back to every valid comparison and say so.
    counted = same if same.get("compared") else win_loss.get("all", {})
    scope = ("same-representation comparisons" if same.get("compared")
             else "valid comparisons, none of which held the same representation on "
                  "both sides (see the win/loss section for why)")
    lines += [
        f"1. **Fastest overall** (best cell anywhere in the matrix): {leader('peak_throughput')}.",
        f"2. **Fastest at batch 1** (the single-user case): {leader('batch_1')}.",
        f"3. **Best batch scaling**: {_scaling_leader(rankings)}.",
        f"4. **Where Aether wins**: {counted.get('aether_wins', 0)} of "
        f"{counted.get('compared', 0)} {scope}. "
        "The full list is in the win/loss section; nothing is summarized away.",
        f"5. **Where Aether loses**: {counted.get('aether_losses', 0)} of "
        f"{counted.get('compared', 0)}, "
        f"plus {counted.get('ties', 0)} statistical ties.",
        f"6. **Largest measured Aether advantage**: {_extreme(win_loss, 'largest_advantage')}.",
        f"7. **Largest measured Aether disadvantage**: "
        f"{_extreme(win_loss, 'largest_disadvantage')}.",
        f"8. **Strongest Aether model**: {_best_model(analysis)}.",
        f"9. **Memory**: {_memory_answer(analysis)}.",
        f"10. **Compilation trade-off**: {_compile_answer(analysis)}.",
        "",
        f"Aether's own position: rank {subject_rank} at batch 1, "
        f"rank {peak_rank} on peak throughput, out of "
        f"{len(rankings.get('batch_1', {}).get('order') or [])} engines that produced a "
        "batch-1 measurement.",
        "",
    ]
    return "\n".join(lines)


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


def _extreme(win_loss: dict[str, Any], key: str) -> str:
    entry = win_loss.get(key)
    if not entry:
        # No same-representation pairing exists, so fall back to the widest
        # comparison the run did make and label what it is.
        pool = (win_loss.get("all") or {}).get(
            "wins" if key == "largest_advantage" else "losses"
        ) or []
        if not pool:
            return "no comparison of this kind was produced by this run"
        chooser = max if key == "largest_advantage" else min
        entry = chooser(
            pool, key=lambda item: item["throughput"]["subject_improvement_percent"]
        )
        rendered = _extreme_text(entry)
        return f"{rendered} - note: this comparison crosses a representation boundary"
    throughput = entry["throughput"]
    if key == "largest_disadvantage" and throughput["subject_improvement_percent"] >= 0:
        # Reporting a positive number under the word "disadvantage" would read as a
        # loss that did not happen. Say plainly that there was none, and give the
        # narrowest margin instead, which is the nearest thing the run measured.
        return (
            "none - Aether was not slower in any same-representation comparison. Its "
            f"narrowest margin was {_pct(throughput['subject_improvement_percent'])} "
            f"against `{entry['competitor']}` on {_short(entry['model'])} at batch "
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


def _best_model(analysis: dict[str, Any]) -> str:
    """The model where Aether's median same-representation advantage is largest."""
    by_model: dict[str, list[float]] = {}
    for item in analysis["comparisons"]:
        if (item["comparability"] == SAME_REPRESENTATION
                and item["throughput"].get("comparable")):
            by_model.setdefault(item["model"], []).append(
                item["throughput"]["subject_improvement_percent"]
            )
    suffix = ""
    if not by_model:
        for item in analysis["comparisons"]:
            if item["throughput"].get("comparable"):
                by_model.setdefault(item["model"], []).append(
                    item["throughput"]["subject_improvement_percent"]
                )
        suffix = " (comparisons cross a representation boundary)"
    if not by_model:
        return "not determined; no comparison was produced by this run"
    import statistics

    scored = {
        model: statistics.median(values) for model, values in by_model.items()
    }
    best = max(scored, key=lambda key: scored[key])
    count = len(by_model[best])
    return (f"`{_short(best)}`, median {_pct(scored[best])} across "
            f"{count} comparison{'' if count == 1 else 's'}{suffix}")


def _memory_answer(analysis: dict[str, Any]) -> str:
    host = [
        item for item in analysis["comparisons"]
        if item["host_memory"].get("comparable")
    ]
    if not host:
        return "no comparable memory measurement"
    import statistics

    values = [item["host_memory"]["subject_improvement_percent"] for item in host]
    median = statistics.median(values)
    direction = "less" if median > 0 else "more"
    return (f"Aether's peak process memory was a median {abs(median):.1f}% {direction} "
            f"than its competitors' across {len(host)} "
            f"comparison{'' if len(host) == 1 else 's'} "
            f"(range {min(values):+.1f}% to {max(values):+.1f}%)")


def _compile_answer(analysis: dict[str, Any]) -> str:
    breaks = [
        item for item in analysis["compile_economics"]["break_even"]
        if item.get("break_even_runs") is not None
    ]
    if not breaks:
        return (
            "no break-even point was computable; either Aether was not faster per "
            "request in the measured configuration, or no competitor produced a "
            "comparable start-up measurement"
        )
    soonest = min(breaks, key=lambda item: item["break_even_runs"])
    return (
        f"against `{soonest['competitor']}` on {_short(soonest['model'])}: "
        f"{soonest['interpretation']} "
        f"(start-up {_seconds(soonest['aether_startup_s'], 1)} against "
        f"{_seconds(soonest['competitor_startup_s'], 1)}, per request "
        f"{_seconds(soonest['aether_latency_s'])} against "
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
            ["Native bf16 on this device", _fmt(hardware.get("bf16_native"))],
            ["Native fp16 on this device", _fmt(hardware.get("fp16_native"))],
            ["Benchmark precision", str(plan.get("resolved_precision"))],
            ["Precision chosen because", str(plan.get("precision_reason"))],
            ["Threads pinned to", _fmt(plan.get("threads"))],
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
    rows = []
    for item in analysis["comparisons"]:
        if item["model"] != model_id:
            continue
        throughput = item["throughput"]
        rows.append([
            str(item["batch_size"]),
            f"`{item['competitor']}`",
            _fmt(throughput.get("subject")),
            _fmt(throughput.get("other")),
            _pct(throughput.get("subject_improvement_percent")),
            _pct(item["latency"].get("subject_improvement_percent")),
            _pct(item["host_memory"].get("subject_improvement_percent")),
            item["winner"],
            "same" if item["comparability"] == SAME_REPRESENTATION else "differs",
        ])
    rows.sort(key=lambda item: (int(item[0]), item[1]))
    return _table(
        ["Batch", "Competitor", "Aether tok/s", "Competitor tok/s",
         "Aether throughput", "Aether latency", "Aether memory", "Winner",
         "Representation"],
        rows,
    )


def headline_tables(analysis: dict[str, Any]) -> str:
    """The two tables the charter asks for by name.

    The first is Aether against the *best* competitor in every cell, which is the
    hardest comparison available and therefore the only honest headline. The second is
    Aether's position against each named competitor, with empty cells where the
    comparison was not valid rather than a fabricated number.
    """
    best_rows = []
    grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for item in analysis["comparisons"]:
        if item["throughput"].get("comparable"):
            key = (item["model"], item["batch_size"], item["prompt_tokens"],
                   item["output_tokens"])
            grouped.setdefault(key, []).append(item)
    for key in sorted(grouped):
        entries = grouped[key]
        best = max(entries, key=lambda item: item["throughput"]["other"])
        best_rows.append([
            _short(key[0]),
            str(key[1]),
            _fmt(best["throughput"]["subject"]),
            f"`{best['competitor']}`",
            _fmt(best["throughput"]["other"]),
            _pct(best["throughput"]["subject_improvement_percent"]),
            "Aether" if best["winner"] == "aether"
            else ("tie" if best["winner"] == "tie" else best["competitor"]),
        ])

    competitors = sorted(analysis["per_competitor"])
    matrix_rows = []
    for key in sorted(grouped):
        entries = {item["competitor"]: item for item in grouped[key]}
        subject = next(iter(grouped[key]))["throughput"]["subject"]
        ranked = sorted(
            [(item["competitor"], item["throughput"]["other"]) for item in grouped[key]]
            + [(SUBJECT, subject)],
            key=lambda pair: -pair[1],
        )
        position = next(
            index for index, pair in enumerate(ranked, start=1) if pair[0] == SUBJECT
        )
        row = [_short(key[0]), str(key[1]), f"{position}", f"{len(ranked)}"]
        for competitor in competitors:
            item = entries.get(competitor)
            row.append(
                _pct(item["throughput"]["subject_improvement_percent"]) if item else "—"
            )
        matrix_rows.append(row)

    return "\n".join([
        "## Headline comparison",
        "",
        "Aether against the fastest engine that produced a measurement in each cell. "
        "A cell where Aether is not the winner says so.",
        "",
        _table(
            ["Model", "Batch", "Aether tok/s", "Best competitor",
             "Best competitor tok/s", "Aether vs best", "Winner"],
            best_rows,
        ),
        "",
        "Aether's rank in each cell, and its position against every engine "
        "individually. A dash means the comparison was not valid, which is not the "
        "same as a zero.",
        "",
        _table(
            ["Model", "Batch", "Aether rank", "Engines compared",
             *[f"vs {name}" for name in competitors]],
            matrix_rows,
        ),
        "",
    ])


def win_loss_section(analysis: dict[str, Any]) -> str:
    """Wins, losses, ties and invalid comparisons, in that order and equal detail."""
    win_loss = analysis["win_loss"]

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
                "same" if item["comparability"] == SAME_REPRESENTATION else "differs",
            ]
            for item in sorted(
                entries,
                key=lambda item: -item["throughput"]["subject_improvement_percent"],
            )
        ]
        return _table(
            ["Model", "Batch", "Workload", "Competitor", "Aether tok/s",
             "Competitor tok/s", "Difference", "Representation"],
            rows,
        )

    everything = win_loss["all"]
    same = win_loss["same_representation"]
    different = win_loss["representation_difference"]
    parts = [
        "## Aether win/loss analysis",
        "",
        f"A difference of at most {analysis['tie_threshold_percent']:.0f}% is reported "
        "as a tie, because run-to-run variation at these iteration counts is of that "
        "order and a smaller gap is not evidence of a difference.",
        "",
        _table(
            ["Comparison set", "Compared", "Aether wins", "Aether loses", "Ties"],
            [
                ["All valid comparisons", str(everything["compared"]),
                 str(everything["aether_wins"]), str(everything["aether_losses"]),
                 str(everything["ties"])],
                ["Same representation (supports execution claims)",
                 str(same["compared"]), str(same["aether_wins"]),
                 str(same["aether_losses"]), str(same["ties"])],
                ["Representation differs (does not)", str(different["compared"]),
                 str(different["aether_wins"]), str(different["aether_losses"]),
                 str(different["ties"])],
            ],
        ),
        "",
        "### Where Aether wins",
        "",
        render(everything["wins"]) if everything["wins"] else "_No cell._\n",
        "",
        "### Where Aether loses",
        "",
        render(everything["losses"]) if everything["losses"] else "_No cell._\n",
        "",
        "### Statistical ties",
        "",
        render(everything["tied"]) if everything["tied"] else "_No cell._\n",
        "",
        "### Closest competitions",
        "",
    ]
    for label, entries in win_loss["closest"].items():
        threshold = label.replace("within_", "").replace("_percent", "")
        parts.append(f"- within ±{threshold}%: **{len(entries)}** comparisons")
    parts += ["", "### Comparisons that could not be made", ""]
    incomparable = win_loss["incomparable"]
    if incomparable:
        parts.append(_table(
            ["Model", "Batch", "Competitor", "Reason"],
            [[_short(item["model"]), str(item["batch_size"]),
              f"`{item['competitor']}`", item["reason"]] for item in incomparable],
        ))
    else:
        parts.append("_Every attempted pairing produced a comparison._\n")
    parts += ["", "### Aether against each competitor, aggregated", "",
              _per_competitor_table(analysis), ""]
    return "\n".join(parts)


def _per_competitor_table(analysis: dict[str, Any]) -> str:
    rows = []
    for engine, entry in sorted(analysis["per_competitor"].items()):
        rows.append([
            f"`{engine}`",
            str(entry["cells"]),
            _pct(entry["median_improvement_percent"]),
            _pct(entry.get("median_improvement_percent_same_representation")),
            f"{_pct(entry['min_improvement_percent'])} to "
            f"{_pct(entry['max_improvement_percent'])}",
            f"{entry['aether_wins']}/{entry['aether_losses']}/{entry['ties']}",
            "same" if entry["comparability"] == SAME_REPRESENTATION else "differs",
        ])
    return _table(
        ["Competitor", "Cells", "Median Aether difference",
         "Median, same representation only", "Range", "W/L/T", "Representation"],
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
        "at N = (S_aether - S_other) / (L_other - L_aether). A blank means Aether was "
        "not faster per request there, so no number of requests would repay the build.",
        "",
        _table(
            ["Model", "Competitor", "Aether start-up", "Competitor start-up",
             "Aether latency", "Competitor latency", "Break-even requests",
             "Reading"],
            [
                [
                    _short(item["model"]),
                    f"`{item['competitor']}`",
                    _seconds(item["aether_startup_s"], 1),
                    _seconds(item["competitor_startup_s"], 1),
                    _seconds(item["aether_latency_s"]),
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
        missing = ranking.get("not_measured") or []
        if missing:
            parts += [
                f"Not ranked, because no measurement exists for this metric: "
                f"{', '.join(f'`{name}`' for name in missing)}.",
                "",
            ]
    return "\n".join(parts)


def cross_model_section(analysis: dict[str, Any]) -> str:
    """Whether Aether's position depends on the model, the batch, or the length."""
    import statistics as stats_mod

    def group(field: str) -> list[list[str]]:
        buckets: dict[Any, list[float]] = {}
        for item in analysis["comparisons"]:
            if (item["comparability"] == SAME_REPRESENTATION
                    and item["throughput"].get("comparable")):
                buckets.setdefault(item[field], []).append(
                    item["throughput"]["subject_improvement_percent"]
                )
        return [
            [
                _short(str(key)) if field == "model" else str(key),
                str(len(values)),
                _pct(stats_mod.median(values)),
                f"{_pct(min(values))} to {_pct(max(values))}",
            ]
            for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
        ]

    return "\n".join([
        "## Cross-model analysis",
        "",
        "Whether Aether's measured advantage is a property of the runtime or of the "
        "configuration. Same-representation comparisons only, since those are the ones "
        "that describe execution.",
        "",
        "**By model**", "",
        _table(["Model", "Comparisons", "Median Aether difference", "Range"],
               group("model")),
        "",
        "**By batch width**", "",
        _table(["Batch", "Comparisons", "Median Aether difference", "Range"],
               group("batch_size")),
        "",
        "**By prompt length**", "",
        _table(["Prompt tokens", "Comparisons", "Median Aether difference", "Range"],
               group("prompt_tokens")),
        "",
        "**By output length**", "",
        _table(["Output tokens", "Comparisons", "Median Aether difference", "Range"],
               group("output_tokens")),
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
    "Engines that execute a re-exported or quantized representation (ONNX Runtime, "
    "OpenVINO, llama.cpp, ExLlamaV2, MLC) are not holding the same weights as the "
    "framework engines. Their rows are measured and reported because they are how "
    "people actually deploy, and every percentage derived from them is labelled "
    "REPRESENTATION_DIFFERENCE.",
    "Serving engines reserve device memory by policy rather than by need, so their "
    "peak-memory rows describe a reservation, not a working set. The reservation "
    "fraction is recorded with each result.",
    "Aether's semantic response cache and SGLang's prefix cache are both disabled for "
    "measurement, through public configuration flags, because the benchmark issues one "
    "prompt repeatedly and both would return a cached answer instead of running "
    "inference. Both overrides are recorded per engine; no other default is changed.",
    "Aether's compiled artifact stores weights at BF16 by default, so at fp16 or fp32 "
    "it is not holding bit-identical values to the published checkpoint the framework "
    "engines load. Where the hardware supports bf16 natively, bf16 is the primary "
    "configuration for exactly this reason.",
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
    parts = [
        "# Aether Runtime Inference Benchmark Report",
        "",
        f"Generated: {payload.get('generated_at', '—')}  ",
        f"Suite version: {payload.get('suite_version')}  ",
        f"Benchmark precision: {plan.get('resolved_precision')}  ",
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
    """Write the win/loss matrix as its own table, one row per comparison."""
    import csv

    columns = (
        "model", "batch_size", "prompt_tokens", "output_tokens", "competitor",
        "comparability", "winner", "aether_tokens_per_s", "competitor_tokens_per_s",
        "aether_improvement_percent", "aether_latency_improvement_percent",
        "aether_host_memory_improvement_percent",
        "aether_device_memory_improvement_percent", "aether_ttft_improvement_percent",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for item in analysis["comparisons"]:
            writer.writerow({
                "model": item["model"],
                "batch_size": item["batch_size"],
                "prompt_tokens": item["prompt_tokens"],
                "output_tokens": item["output_tokens"],
                "competitor": item["competitor"],
                "comparability": item["comparability"],
                "winner": item["winner"],
                "aether_tokens_per_s": item["throughput"].get("subject"),
                "competitor_tokens_per_s": item["throughput"].get("other"),
                "aether_improvement_percent":
                    item["throughput"].get("subject_improvement_percent"),
                "aether_latency_improvement_percent":
                    item["latency"].get("subject_improvement_percent"),
                "aether_host_memory_improvement_percent":
                    item["host_memory"].get("subject_improvement_percent"),
                "aether_device_memory_improvement_percent":
                    item["device_memory"].get("subject_improvement_percent"),
                "aether_ttft_improvement_percent":
                    item["ttft"].get("subject_improvement_percent"),
            })
    return path.name


def terminal_summary(payload: dict[str, Any], analysis: dict[str, Any],
                     paths: dict[str, Any]) -> str:
    """The concise summary printed when the run finishes.

    Deliberately the same numbers as the report's executive summary, read from the same
    analysis, so the terminal and the document can never disagree.
    """
    rankings = analysis["rankings"]
    win_loss = analysis["win_loss"]["all"]
    same = analysis["win_loss"]["same_representation"]
    lines = ["", "=" * 72, "AETHER BENCHMARK SUMMARY", "=" * 72, ""]

    for label, key, unit in (
        ("Best batch-1 engine", "batch_1", "tok/s"),
        ("Best peak throughput", "peak_throughput", "tok/s"),
        ("Best TTFT", "ttft", "s"),
        ("Best latency", "latency", "s"),
        ("Best host memory", "host_memory", "bytes"),
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
        lines.append(f"  {label:22s} {head['engine']}  ({rendered})")

    lines += ["", "  Aether:"]
    lines.append(f"    batch-1 rank        {_rank_of(rankings.get('batch_1', {}), SUBJECT)}"
                 f" / {len(rankings.get('batch_1', {}).get('order') or [])}")
    subject_batch1 = next(
        (entry["value"] for entry in (rankings.get("batch_1") or {}).get("order") or []
         if entry["engine"] == SUBJECT), None
    )
    subject_peak = next(
        (entry["value"] for entry in (rankings.get("peak_throughput") or {}).get("order") or []
         if entry["engine"] == SUBJECT), None
    )
    lines.append("    batch-1 throughput  "
                 + (f"{float(subject_batch1):,.2f} tok/s" if subject_batch1 else "not measured"))
    lines.append("    best throughput     "
                 + (f"{float(subject_peak):,.2f} tok/s" if subject_peak else "not measured"))

    lines += ["", "  Aether against each engine (median across cells):"]
    if analysis["per_competitor"]:
        for engine, entry in sorted(analysis["per_competitor"].items()):
            marker = "" if entry["comparability"] == SAME_REPRESENTATION \
                else "   [representation differs]"
            lines.append(
                f"    vs {engine:16s} {entry['median_improvement_percent']:+7.1f}%"
                f"   W/L/T {entry['aether_wins']}/{entry['aether_losses']}/"
                f"{entry['ties']}{marker}"
            )
    else:
        lines.append("    no valid comparison was produced by this run")

    lines += [
        "",
        f"  Wins {win_loss['aether_wins']}   Losses {win_loss['aether_losses']}   "
        f"Ties {win_loss['ties']}   (of {win_loss['compared']} valid comparisons)",
        f"  Same-representation only: wins {same['aether_wins']}, "
        f"losses {same['aether_losses']}, ties {same['ties']} of {same['compared']}",
        "",
        "  Outputs:",
    ]
    for label, value in paths.items():
        lines.append(f"    {label:22s} {value}")
    lines += ["", "=" * 72, ""]
    return "\n".join(lines)
