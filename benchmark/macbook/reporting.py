"""Rebuild two-engine comparisons, tables and figures from saved observations."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from benchmark.macbook.engine import GIB
from benchmark.macbook.aether_engine import ENGINES


def metric(row: dict, name: str, statistic: str = "median") -> float | None:
    return row.get("metrics", {}).get(name, {}).get(statistic)


def comparisons(records: list[dict]) -> list[dict]:
    """Compare only measured cells, holding every other workload dimension fixed."""
    measured = {
        tuple(row[key] for key in ("engine", "batch_size", "prompt_tokens", "output_tokens")): row
        for row in records
        if row["status"] == "MEASURED"
    }
    result = []
    for (engine, batch, prompt, output), row in measured.items():
        baselines = [
            ("batch", (engine, 1, prompt, output)),
            ("prompt", (engine, batch, 256, output)),
            ("output", (engine, batch, prompt, 128)),
        ]
        if engine == "aether":
            baselines.append(("engine", ("onnxruntime", batch, prompt, output)))
        for sweep, baseline in baselines:
            other = measured.get(baseline)
            if other is None:
                continue
            if sweep == "engine" and (
                not row.get("prompt_ids_sha256")
                or row["prompt_ids_sha256"] != other.get("prompt_ids_sha256")
            ):
                continue
            rate, base_rate = (metric(item, "aggregate_tokens_per_s") for item in (row, other))
            latency, base_latency = (metric(item, "latency_s") for item in (row, other))
            if not rate or not base_rate or not latency or not base_latency:
                continue
            result.append(
                {
                    "sweep": sweep,
                    "engine": engine,
                    "baseline_engine": baseline[0],
                    "comparison_scope": "different representations: Aether FP32 / ORT INT4"
                    if sweep == "engine"
                    else "same engine and representation",
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "output_tokens": output,
                    "baseline_batch": baseline[1],
                    "baseline_prompt": baseline[2],
                    "baseline_output": baseline[3],
                    "throughput_ratio": rate / base_rate,
                    "throughput_change_percent": (rate / base_rate - 1) * 100,
                    "latency_ratio": latency / base_latency,
                    "batch_efficiency_percent": rate / base_rate / batch * 100
                    if sweep == "batch"
                    else None,
                }
            )
    return result


def csv_file(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plots(records: list[dict], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = [row for row in records if row["status"] == "MEASURED"]
    if not measured:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("batch_throughput", "batch_size", "aggregate_tokens_per_s", "Aggregate output tokens/s"),
        (
            "batch_per_request",
            "batch_size",
            "per_request_tokens_per_s",
            "Output tokens/s per request",
        ),
        ("batch_latency", "batch_size", "latency_s", "Generation latency (s)"),
        ("batch_memory", "batch_size", "memory", "Worker peak RSS (GiB; sampled)"),
        (
            "prompt_throughput",
            "prompt_tokens",
            "aggregate_tokens_per_s",
            "Aggregate output tokens/s",
        ),
        ("prompt_ttft", "prompt_tokens", "ttft_s", "Time to first token (s)"),
        (
            "output_throughput",
            "output_tokens",
            "aggregate_tokens_per_s",
            "Aggregate output tokens/s",
        ),
        ("output_latency", "output_tokens", "latency_s", "Generation latency (s)"),
    ]
    written = []
    dimensions = ("batch_size", "prompt_tokens", "output_tokens")
    for name, axis, value, label in specifications:
        others = [key for key in dimensions if key != axis]
        groups: dict[tuple, list] = {}
        for row in measured:
            y = (
                row.get("worker_peak_rss_bytes", 0) / GIB
                if value == "memory"
                else metric(row, value)
            )
            if y is not None:
                group = (row["engine"], *(row[key] for key in others))
                groups.setdefault(group, []).append((row[axis], y))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.20, top=0.84)
        for group, points in sorted(groups.items()):
            points.sort()
            # Keep an Aether singleton whenever ORT has the same fixed dimensions.
            if (
                len(points) == 1
                and any(len(items) > 1 for items in groups.values())
                and not any(
                    len(items) > 1 and other[1:] == group[1:] for other, items in groups.items()
                )
            ):
                continue
            ax.plot(
                [x for x, _ in points],
                [y for _, y in points],
                marker="o",
                label=group[0]
                + ": "
                + ", ".join(f"{key}={val}" for key, val in zip(others, group[1:])),
            )
        title = "Qwen3-8B · Aether vs ONNX Runtime · CPU"
        if any(row.get("harness_validation_only") for row in measured):
            title = "SYNTHETIC TEST DATA — not performance evidence\n" + title
        ax.set(title=title, xlabel=axis.replace("_", " "), ylabel=label, ylim=(0, None))
        ax.set_xticks(sorted({row[axis] for row in measured}))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.text(
            0.5,
            0.025,
            "Measured cells only. Aether FP32 / ORT INT4; missing cases are listed in the report.",
            ha="center",
            fontsize=8,
        )
        filename = name + ".png"
        fig.savefig(directory / filename, dpi=160)
        plt.close(fig)
        written.append(filename)
    return written


def render(directory: Path) -> None:
    """Generate JSON, CSV, Markdown and a self-contained-navigation HTML report."""
    plan = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if plan.get("schema") != 2 or plan.get("engines") != list(ENGINES):
        raise ValueError("Expected schema 2 with exactly Aether and ONNX Runtime")
    records = []
    for cell in plan["cells"]:
        key = f"b{cell['batch_size']}_p{cell['prompt_tokens']}_o{cell['output_tokens']}"
        for engine in ENGINES:
            path = directory / "raw" / engine / f"{key}.json"
            row = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {
                    **cell,
                    "engine": engine,
                    "status": "NOT_COMPLETED",
                    "reason": "No completed worker record",
                }
            )
            if row.get("engine") != engine or any(row.get(k) != v for k, v in cell.items()):
                raise ValueError(f"Mismatched engine/workload record: {path}")
            records.append(row)
    pairs = comparisons(records)
    chart_names = plots(records, directory / "graphs")
    report_dir = directory / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan, "records": records, "comparisons": pairs, "graphs": chart_names}
    (directory / "benchmark_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    fields = [
        "engine",
        "representation",
        "batch_size",
        "prompt_tokens",
        "output_tokens",
        "status",
        "reason",
        "load_s",
        "cold_latency_s",
        "worker_peak_rss_bytes",
        "estimated_working_bytes",
        "swap_used_bytes_before",
        "swap_used_bytes_after",
    ]
    metrics = (
        "latency_s",
        "ttft_s",
        "decode_s",
        "tpot_ms",
        "aggregate_tokens_per_s",
        "per_request_tokens_per_s",
    )
    fields += [
        f"{name}_{stat}"
        for name in metrics
        for stat in ("n", "median", "mean", "stdev", "p90", "p95", "coefficient_of_variation")
    ]
    flat = []
    for row in records:
        entry = {key: row.get(key) for key in fields}
        entry.update(
            {
                f"{name}_{stat}": value
                for name in metrics
                for stat, value in row.get("metrics", {}).get(name, {}).items()
            }
        )
        flat.append(entry)
    csv_file(directory / "benchmark_results.csv", flat, fields)
    csv_file(
        directory / "benchmark_comparisons.csv",
        pairs,
        [
            "sweep",
            "engine",
            "baseline_engine",
            "comparison_scope",
            "batch_size",
            "prompt_tokens",
            "output_tokens",
            "baseline_batch",
            "baseline_prompt",
            "baseline_output",
            "throughput_ratio",
            "throughput_change_percent",
            "latency_ratio",
            "batch_efficiency_percent",
        ],
    )

    def number(value: Any) -> str:
        return "—" if value is None else f"{value:,.3f}"

    table = [
        "| Engine | Batch | Prompt | Output | Status | Output tok/s | Per request tok/s | "
        "Latency s | "
        "TTFT s | TPOT ms | Worker RSS GiB |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        values = [
            row["engine"],
            row["batch_size"],
            row["prompt_tokens"],
            row["output_tokens"],
            row["status"],
        ]
        values += [
            number(metric(row, name))
            for name in (
                "aggregate_tokens_per_s",
                "per_request_tokens_per_s",
                "latency_s",
                "ttft_s",
                "tpot_ms",
            )
        ]
        values += [
            number(row["worker_peak_rss_bytes"] / GIB) if row.get("worker_peak_rss_bytes") else "—"
        ]
        table.append("| " + " | ".join(str(value) for value in values) + " |")
    measured = sum(row["status"] == "MEASURED" for row in records)
    notes = [
        f"Measured {measured}/{len(records)} requested cases. "
        + (
            "Smoke run: pipeline validation only."
            if plan["smoke"]
            else "All failures and skipped cases remain visible."
        ),
        "Scope: exactly Aether and ONNX Runtime; Qwen3-8B only. Aether uses its actual "
        "CPUExecutionEngine; ONNX uses CPUExecutionProvider. No GPU provider or engine fallback.",
        "Representations differ: Aether BF16 storage expanded to FP32 computation; ONNX published "
        "INT4 KLD block-128 weights and FP32 KV cache. This compares available execution paths, "
        "not identical precision or established accuracy equivalence.",
        f"Paired engine comparisons: {sum(p['sweep'] == 'engine' for p in pairs)}. "
        "Engine ratios are Aether / ONNX for identical workload and prompt hashes; throughput "
        "above 1 favors Aether, latency below 1 favors Aether. "
        "No missing result establishes a winner.",
        "Current Aether limitations: weights expand to FP32 (cannot fit this 8B model in 16 GB); "
        "true batch above 1 is unsupported. Such cells remain explicit and have no ratios.",
        "Workload: exact token IDs from the existing benchmark corpus, repeated identically "
        "across rows. "
        "Greedy decoding, no chat template, EOS ignored, fixed output length. This measures a "
        "controlled "
        "generation workload, not Qwen thinking-mode answer quality.",
        "Timing includes input/cache allocation, synchronous engine calls, greedy selection, "
        "finite-logit checks "
        "and the Python decode loop. Tokenization, download, session loading and text "
        "rendering are excluded. "
        "TTFT is observed in the same generation; TPOT=(end-first token)/(output tokens-1). "
        "Aggregate tok/s=batch×output/latency; per-request tok/s=output/latency.",
        "Every case loads a fresh process/session, takes one cold call, discards warmups, then "
        "measures repetitions. "
        "Load times include ORT graph optimization or Aether weight loading; filesystem caches "
        "may already be warm. Artifact preparation is separate from generation timings.",
        "Median and dispersion are computed from per-iteration observations. Full "
        "n/mean/stdev/p90/p95/CV "
        "are in CSV/JSON. Percentiles from small samples are descriptive, not production "
        "tail-latency estimates.",
        "Memory estimates are heuristic. The parent samples worker RSS every 200 ms (including "
        "load); "
        "a separate telemetry generation samples RSS/CPU every 50 ms. Monitoring overhead "
        "still exists. "
        "Transient peaks can be missed; macOS compression, swap and other applications can "
        "affect results. "
        "A memory cap is best effort and does not guarantee freedom from OS termination.",
        "INT4 correctness against the source model is not tested. Only finite logits, exact "
        "row/token counts, "
        "greedy repeatability and agreement across duplicated batch rows are checked.",
        f"Requested CPU threads: {plan['threads']}; "
        "ORT intra-op / 1 inter-op; sequential execution; "
        "graph optimization ALL; "
        "CPU arena and memory pattern disabled. Cases run sequentially; thermal drift is not "
        "corrected.",
    ]
    if plan.get("harness_validation_only"):
        notes.insert(
            0, "SYNTHETIC TEST DATA: validates reporting only; not model performance evidence."
        )
    if not any(p["sweep"] == "engine" for p in pairs):
        notes.insert(
            0, "INCOMPLETE HEAD-TO-HEAD: no matched engine measurements; no winner reported."
        )
    lines = ["# MacBook benchmark — Qwen3-8B / Aether vs ONNX Runtime", ""]
    lines += [item + "\n" for item in notes]
    lines += [
        "## Results",
        "",
        *table,
        "",
        "## Loading and sample dispersion",
        "",
        "Load and cold-call times are separate from steady-state medians. "
        "CV is latency standard deviation divided by mean.",
        "",
        "| Cell (b/p/o) | Session load s | Cold call s | Samples | Latency p95 s | Latency CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        if row["status"] == "MEASURED":
            lines.append(
                f"| {row['engine']} "
                f"{row['batch_size']}/{row['prompt_tokens']}/{row['output_tokens']} "
                f"| {number(row.get('load_s'))} | {number(row.get('cold_latency_s'))} "
                f"| {metric(row, 'latency_s', 'n')} "
                f"| {number(metric(row, 'latency_s', 'p95'))} "
                f"| {number(metric(row, 'latency_s', 'coefficient_of_variation'))} |"
            )
    lines += [
        "",
        "## Comparisons",
        "",
        "Ratios compare measured medians with the listed baseline; no missing value becomes zero.",
        "",
        "| Sweep | Cell (b/p/o) | Baseline (b/p/o) | Throughput ratio | Latency ratio | Batch "
        "efficiency % |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in pairs:
        lines.append(
            f"| {item['sweep']} ({item['engine']} / {item['baseline_engine']}) | "
            f"{item['batch_size']}/{item['prompt_tokens']}/{item['output_tokens']} "
            f"| {item['baseline_batch']}/{item['baseline_prompt']}/{item['baseline_output']} "
            f"| {number(item['throughput_ratio'])} | {number(item['latency_ratio'])} "
            f"| {number(item['batch_efficiency_percent'])} |"
        )
    lines += ["", "## Missing measurements", ""]
    lines += [
        f"- {r['engine']} b{r['batch_size']}/p{r['prompt_tokens']}/o{r['output_tokens']}: "
        f"{r['status']} — {r.get('reason', '')}"
        for r in records
        if r["status"] != "MEASURED"
    ]
    lines += [
        "",
        "## Reproducibility",
        "",
        "```json",
        json.dumps(plan, indent=2),
        "```",
        "",
        "## Graphs",
        "",
    ]
    for name in chart_names:
        lines += [f"![{name}](../graphs/{name})", ""]
    lines += [
        "## Generated text samples (first batch row)",
        "",
        "These are raw fixed-length continuations; special tokens are retained.",
        "",
    ]
    for row in records:
        if row["status"] == "MEASURED":
            lines += [
                f"### {row['engine']} "
                f"b{row['batch_size']}/p{row['prompt_tokens']}/o{row['output_tokens']}",
                "",
                "```text",
                row.get("sample_text", "").replace("```", "'''"),
                "```",
                "",
            ]
    markdown = "\n".join(lines)
    (report_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    # No JS/CDN dependencies: opening the file works offline in a Mac browser.
    body = "".join(f"<p>{html.escape(item)}</p>" for item in notes)
    for title, rows in (("Results", table),):
        body += f"<h2>{title}</h2><div class='scroll'><table>"
        for index, line in enumerate(rows):
            if index == 1:
                continue
            tag = "th" if index == 0 else "td"
            body += (
                "<tr>"
                + "".join(
                    f"<{tag}>{html.escape(v.strip())}</{tag}>" for v in line.strip("|").split("|")
                )
                + "</tr>"
            )
        body += "</table></div>"
    body += "<h2>Graphs</h2>" + "".join(
        f'<img src="../graphs/{name}" alt="{name}">' for name in chart_names
    )
    body += (
        "<h2>Detailed report, comparisons and provenance</h2><pre>"
        + html.escape(markdown)
        + "</pre>"
    )
    (report_dir / "REPORT.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>MacBook ONNX benchmark</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;"
        "color:#172b3a}p{line-height:1.6}table{border-collapse:collapse;width:100%}"
        "td,th{padding:8px;border-bottom:1px solid #ccd;text-align:right}.scroll{overflow:auto}"
        "img{width:100%;margin:20px "
        "0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}</style>"
        "<h1>Qwen3-8B · Aether vs ONNX Runtime CPU</h1>" + body + "</html>",
        encoding="utf-8",
    )
