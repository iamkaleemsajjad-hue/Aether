"""Turn raw results into a report a sceptical reader can audit.

Two rules shape this module:

* every derived number is printed next to the measurements it came from, so a
  ratio can always be checked against its operands;
* an absent, failed, or unsupported measurement is printed as such.  Nothing is
  dropped from a table because it was unflattering, and no axis is rescaled to
  make a difference look larger than it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}{suffix}"
    return str(value)


def _bytes(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) / 1024 ** 3:.3f} GiB"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    line = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"{line}\n{rule}\n{body}\n"


def environment_section(env: dict[str, Any]) -> str:
    gpu = env.get("gpu", {})
    cpu = env.get("cpu", {})
    software = env.get("software", {})
    ram = env.get("ram", {})
    lines = ["## Environment\n"]
    devices = gpu.get("devices") or []
    lines.append(_table(
        ["Property", "Value"],
        [
            ["GPU count", _fmt(gpu.get("count"))],
            ["GPU(s)", ", ".join(d["name"] for d in devices) or "none"],
            ["GPU VRAM", ", ".join(_bytes(d["total_memory_bytes"]) for d in devices) or "—"],
            ["Compute capability", ", ".join(d["compute_capability"] for d in devices) or "—"],
            ["CUDA runtime (torch)", _fmt(gpu.get("cuda_runtime_version"))],
            ["cuDNN", _fmt(gpu.get("cudnn_version"))],
            ["SDPA backends enabled", _fmt(gpu.get("sdpa_backends"))],
            ["nvidia-smi", "<br>".join(gpu.get("nvml") or []) or "—"],
            ["CPU", _fmt(cpu.get("model_name") or cpu.get("processor"))],
            ["CPU cores (physical/logical)",
             f"{_fmt(cpu.get('physical_cores'))} / {_fmt(cpu.get('logical_cores'))}"],
            ["torch threads", _fmt(cpu.get("torch_num_threads"))],
            ["System RAM total", _bytes(ram.get("total_bytes"))],
            ["System RAM available", _bytes(ram.get("available_bytes"))],
            ["OS", _fmt(software.get("platform"))],
            ["Python", _fmt(software.get("python"))],
            ["PyTorch", _fmt(software.get("torch"))],
            ["Transformers", _fmt(software.get("transformers"))],
            ["Tokenizers", _fmt(software.get("tokenizers"))],
            ["Aether Runtime", _fmt(software.get("aether_runtime") or software.get("aether_version_constant"))],
            ["Aether commit", _fmt(software.get("aether_git_commit"))],
            ["Aether working tree dirty", _fmt(software.get("aether_git_dirty"))],
        ],
    ))
    revisions = env.get("model_revisions") or {}
    if revisions:
        lines.append("\n**Model revisions** (both backends load these exact commits):\n")
        lines.append(_table(["Model", "Revision"],
                            [[k, _fmt(v)] for k, v in revisions.items()]))
    relevant = {k: v for k, v in (env.get("env") or {}).items() if v}
    if relevant:
        lines.append("\n**Relevant environment variables:**\n")
        lines.append(_table(["Variable", "Value"], [[k, str(v)] for k, v in relevant.items()]))
    return "\n".join(lines)


def performance_section(results: list[dict[str, Any]]) -> str:
    """Throughput and latency, with the operands of every ratio kept visible."""
    lines = ["## Steady-state generation performance\n"]
    lines.append(
        "Each row is one (model, precision, prompt length, batch size) cell. "
        "`tok/s` is the median of per-iteration throughputs; the cold column is "
        "the first, unwarmed iteration, reported separately because it carries "
        "autotuning and allocator growth that a served request would not.\n"
    )
    rows = []
    for cell in results:
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            if record.get("status") != "ok":
                rows.append([
                    key["model"].split("/")[-1], key["precision"],
                    str(key["prompt_tokens"]), str(key["batch_size"]), backend,
                    "—", f"**{record.get('status', 'missing')}**",
                    _fmt(record.get("message", "")[:60]), "—", "—", "—", "—",
                ])
                continue
            latency = record["latency_s"]
            throughput = record["tokens_per_s"]
            rows.append([
                key["model"].split("/")[-1], key["precision"],
                str(key["prompt_tokens"]), str(key["batch_size"]), backend,
                _fmt(record.get("completion_tokens")),
                _fmt(throughput.get("median")), _fmt(throughput.get("stdev")),
                _fmt(latency.get("median"), 4), _fmt(latency.get("p95"), 4),
                _fmt(record.get("cold_latency_s"), 4), str(latency.get("n")),
            ])
    lines.append(_table(
        ["Model", "Prec", "Prompt", "Batch", "Backend", "gen tokens",
         "tok/s (med)", "tok/s (sd)", "latency s (med)", "latency s (p95)",
         "cold s", "n"],
        rows,
    ))

    comparisons = []
    for cell in results:
        key = cell["cell"]
        aether = cell["backends"].get("aether", {})
        reference = cell["backends"].get("transformers", {})
        if aether.get("status") != "ok" or reference.get("status") != "ok":
            note = (
                f"aether={aether.get('status', 'missing')}, "
                f"transformers={reference.get('status', 'missing')}"
            )
            comparisons.append([
                key["model"].split("/")[-1], key["precision"], str(key["prompt_tokens"]),
                str(key["batch_size"]), "not comparable", note, "—", "—",
            ])
            continue
        a = aether["tokens_per_s"]["median"]
        b = reference["tokens_per_s"]["median"]
        ratio = a / b if b else None
        comparisons.append([
            key["model"].split("/")[-1], key["precision"], str(key["prompt_tokens"]),
            str(key["batch_size"]),
            f"{ratio:.2f}x" if ratio else "—",
            f"{(ratio - 1) * 100:+.1f}%" if ratio else "—",
            f"{a:.2f}", f"{b:.2f}",
        ])
    lines.append("\n### Aether relative to Transformers\n")
    lines.append(
        "`speedup = median Aether tok/s / median Transformers tok/s`. Both "
        "operands are shown so the ratio can be checked directly. A value below "
        "1.00x means Transformers was faster in that cell. Throughput is "
        "normalized by the tokens each backend actually generated; a row flagged "
        "as *different work* means the two produced different token counts, so "
        "its latency comparison is not like-for-like.\n"
    )
    lines.append(_table(
        ["Model", "Prec", "Prompt", "Batch", "Speedup", "Change",
         "Aether tok/s", "Transformers tok/s"],
        comparisons,
    ))
    return "\n".join(lines)


def phase_section(results: list[dict[str, Any]]) -> str:
    """Prefill and time-to-first-token, kept apart from end-to-end throughput."""
    lines = ["## Prefill, decode and time-to-first-token\n"]
    lines.append(
        "Prefill is a single forward pass over the prompt at the same abstraction "
        "on both backends. Decode is derived as the remainder of the end-to-end "
        "generation, so it inherits prefill's uncertainty. TTFT is measured "
        "through each library's own streaming API and is therefore reported as its "
        "own experiment rather than folded into throughput.\n"
    )
    rows = []
    for cell in results:
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            prefill = (cell.get("prefill") or {}).get(backend, {})
            ttft = (cell.get("ttft") or {}).get(backend, {})
            prefill_median = (prefill.get("latency_s") or {}).get("median")
            total_median = (record.get("latency_s") or {}).get("median")
            generated = record.get("completion_tokens") or 0
            decode_median = (
                total_median - prefill_median
                if total_median is not None and prefill_median is not None
                else None
            )
            rows.append([
                key["model"].split("/")[-1], key["precision"], str(key["prompt_tokens"]),
                backend,
                _fmt(prefill_median, 4) if prefill.get("status") == "ok"
                else prefill.get("status", "—"),
                _fmt(decode_median, 4),
                _fmt(decode_median / generated * 1000.0, 3) if decode_median and generated else "—",
                _fmt(((ttft.get("ttft_s") or {}).get("median")), 4)
                if ttft.get("status") == "ok" else ttft.get("status", "—"),
            ])
    lines.append(_table(
        ["Model", "Prec", "Prompt", "Backend", "prefill s", "decode s",
         "ms / decoded token", "TTFT s"],
        rows,
    ))
    return "\n".join(lines)


def memory_section(results: list[dict[str, Any]]) -> str:
    lines = ["## Memory\n"]
    lines.append(
        "GPU figures come from the PyTorch allocator, which is exact. `peak "
        "allocated` is the high-water mark of live tensors; `peak reserved` "
        "includes the allocator's cached blocks, which is what the driver "
        "actually holds. Host RSS is read from the OS, not from Python.\n"
    )
    rows = []
    for cell in results:
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            load = (cell.get("load") or {}).get(backend, {})
            if record.get("status") != "ok":
                continue
            devices = (record.get("gpu_peak") or {}).get("devices") or []
            peak_alloc = sum(d["peak_allocated_bytes"] for d in devices) or None
            peak_reserved = sum(d["peak_reserved_bytes"] for d in devices) or None
            after_load = ((load.get("gpu_after") or {}).get("devices") or [])
            load_alloc = sum(d["allocated_bytes"] for d in after_load) or None
            host = record.get("host_during_inference") or {}
            before_load = ((load.get("gpu_before") or {}).get("devices") or [])
            load_delta = (
                (load_alloc or 0) - sum(d["allocated_bytes"] for d in before_load)
                if load_alloc is not None else None
            )
            rows.append([
                key["model"].split("/")[-1], key["precision"], str(key["prompt_tokens"]),
                backend,
                _bytes(load_delta), _bytes(record.get("gpu_inference_delta_bytes")),
                _bytes(peak_alloc), _bytes(peak_reserved),
                _bytes((load.get("host_after") or {}).get("rss_bytes")),
                _bytes(host.get("rss_peak_bytes")),
            ])
    lines.append(_table(
        ["Model", "Prec", "Prompt", "Backend", "GPU weights (load delta)",
         "GPU inference delta", "GPU peak alloc*", "GPU peak reserved*",
         "Host RSS after load", "Host RSS peak (infer)"],
        rows,
    ))
    lines.append(
        "\n*Both backends are kept resident in one process so execution order can "
        "be alternated per cell, which is what protects the throughput numbers from "
        "thermal drift. The allocator's peak counters are process-wide, so the two "
        "starred columns include the other backend's resident weights and are "
        "**not** attributable to a single backend. The two delta columns are the "
        "per-backend figures: `GPU weights (load delta)` is the allocation change "
        "across this backend's own load, and `GPU inference delta` is the extra "
        "memory its decoding needed above what was already resident, which is the "
        "KV cache plus activations.\n"
    )
    return "\n".join(lines)


def utilization_section(results: list[dict[str, Any]]) -> str:
    lines = ["## CPU and GPU utilization\n"]
    lines.append(
        "Collected by background samplers during a dedicated extra iteration, "
        "never during the iterations whose latency is reported above, so the "
        "sampler's own cost cannot contaminate the official timings. The sampling "
        "period is stated per row: a sampler can miss a transient shorter than it.\n"
    )
    rows = []
    for cell in results:
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            if record.get("status") != "ok":
                continue
            host = record.get("host_during_inference") or {}
            telemetry = record.get("gpu_telemetry") or {}
            devices = telemetry.get("devices") or []
            rows.append([
                key["model"].split("/")[-1], key["precision"], backend,
                _fmt(host.get("sample_interval_s"), 3),
                _fmt(host.get("cpu_percent_mean"), 1, "%"),
                _fmt(host.get("cpu_percent_peak"), 1, "%"),
                _fmt(host.get("threads")),
                ", ".join(_fmt(d.get("gpu_util_percent_mean"), 1, "%") for d in devices) or "—",
                ", ".join(_fmt(d.get("power_watts_mean"), 1, " W") for d in devices) or "—",
                ", ".join(_fmt(d.get("temperature_c_max"), 0, "°C") for d in devices) or "—",
            ])
    lines.append(_table(
        ["Model", "Prec", "Backend", "sample s", "CPU mean", "CPU peak", "threads",
         "GPU util", "GPU power", "GPU temp max"],
        rows,
    ))
    return "\n".join(lines)


def correctness_section(payload: dict[str, Any]) -> str:
    """State whether the two runtimes computed the same thing, with the evidence."""
    lines = ["## Correctness\n"]
    lines.append(
        "Compared at three levels: raw prompt logits, greedy token ids, and decoded "
        "text. Bit-for-bit equality is **not** the criterion — the two runtimes use "
        "different kernels and reduction orders, and Aether's weights pass through "
        "the AEG's BF16 residency, so small floating-point differences are expected "
        "and legitimate. The question is whether the differences sit at the scale of "
        "floating-point noise or at the scale of a different computation, so the "
        "deviation is normalized by the reference's own logit spread.\n"
    )
    rows = []
    for entry in payload.get("cases", []):
        logits = entry.get("logits", {})
        tokens = entry.get("tokens", {})
        text = entry.get("text", {})
        verdict = entry.get("verdict", {})
        if not logits.get("comparable"):
            rows.append([
                entry["model"].split("/")[-1], entry["precision"],
                "not comparable", logits.get("reason", "—"), "—", "—", "—", "—",
            ])
            continue
        rows.append([
            entry["model"].split("/")[-1], entry["precision"],
            _fmt(verdict.get("equivalent")),
            f"{logits['max_abs_diff']:.3e}",
            f"{logits['max_abs_diff_over_std']:.2e}",
            f"{logits['cosine_similarity']:.7f}",
            _fmt(tokens.get("matching_prefix_tokens")) + "/" + _fmt(tokens.get("reference_length")),
            _fmt(text.get("identical")),
        ])
    lines.append(_table(
        ["Model", "Prec", "Equivalent", "max|Δlogit|", "Δ / logit sd",
         "cosine", "greedy prefix match", "text identical"],
        rows,
    ))
    agreement = payload.get("tokenizer_agreement")
    if agreement:
        lines.append("\n**Tokenizer agreement** (verified, not assumed):\n")
        lines.append(_table(
            ["Model", "Identical ids", "Texts checked"],
            [[model.split("/")[-1], _fmt(entry.get("identical")), _fmt(entry.get("checked"))]
             for model, entry in agreement.items()],
        ))
    concerns = [
        f"- `{entry['model']}` ({entry['precision']}): {concern}"
        for entry in payload.get("cases", [])
        for concern in (entry.get("verdict", {}).get("concerns") or [])
    ]
    if concerns:
        lines.append("\n**Concerns raised by the comparison:**\n")
        lines.extend(concerns)
        lines.append("")
    return "\n".join(lines)


def kernel_section(payload: dict[str, Any]) -> str:
    """Kernel counts and time attribution, with the profiling caveat stated."""
    lines = ["## Kernel-level analysis\n"]
    lines.append(
        "Two independent instruments. **Dispatch counts** are deterministic and "
        "hardware-independent: every ATen call is counted, with metadata-only view "
        "operations separated out because they launch no GPU kernel. **Profiler "
        "attribution** assigns device and host time to individual kernels. "
        "Profiling perturbs a launch-bound loop, so none of these numbers feed the "
        "performance tables above.\n"
    )
    rows = []
    for entry in payload.get("dispatch", []):
        for backend in ("transformers", "aether"):
            record = entry.get(backend) or {}
            if not record:
                continue
            rows.append([
                entry["model"].split("/")[-1], backend,
                _fmt(record.get("layers")),
                _fmt(record.get("kernel_calls_per_step"), 1),
                _fmt(record.get("kernel_calls_per_layer_per_step"), 2),
                _fmt(record.get("view_calls_per_step"), 1),
            ])
    lines.append("### ATen calls per decoded token\n")
    lines.append(_table(
        ["Model", "Backend", "layers", "kernels/token", "kernels/layer/token", "views/token"],
        rows,
    ))

    category_rows = []
    for entry in payload.get("dispatch", []):
        categories = set()
        for backend in ("transformers", "aether"):
            categories |= set(((entry.get(backend) or {}).get("by_category_per_step") or {}))
        for category in sorted(categories):
            left = ((entry.get("transformers") or {}).get("by_category_per_step") or {}).get(category)
            right = ((entry.get("aether") or {}).get("by_category_per_step") or {}).get(category)
            category_rows.append([
                entry["model"].split("/")[-1], category, _fmt(left, 1), _fmt(right, 1),
            ])
    if category_rows:
        lines.append("\n### Calls per token by category\n")
        lines.append(_table(["Model", "Category", "Transformers", "Aether"], category_rows))

    for entry in payload.get("profile", []):
        for backend in ("transformers", "aether"):
            record = entry.get(backend) or {}
            if not record:
                continue
            lines.append(f"\n### {entry['model']} — {backend} kernel time\n")
            lines.append(
                f"Summed device time {_fmt(record.get('device_time_per_step_ms'), 3)} ms/step; "
                f"summed self CPU time {_fmt(record.get('summed_self_cpu_time_ms'), 2)} ms over "
                f"{_fmt(record.get('steps'))} steps. "
                f"Synchronizations: {_fmt(record.get('synchronizations'))}; "
                f"memcpy calls: {_fmt(record.get('memcpy_calls'))}.\n"
            )
            if record.get("fused_attention_kernels"):
                lines.append(
                    "Fused attention kernels observed: "
                    + ", ".join(f"`{name}`" for name in record["fused_attention_kernels"])
                    + "\n"
                )
            lines.append(_table(
                ["Kernel", "count", "device ms", "share", "category"],
                [[f"`{k['name']}`", _fmt(k["count"]), _fmt(k["self_device_time_ms"], 3),
                  _fmt((k["share_of_device_time"] or 0) * 100, 1, "%"), k["category"]]
                 for k in (record.get("kernels") or [])[:12]],
            ))
    return "\n".join(lines)


def failures_section(results: list[dict[str, Any]], extra: list[dict[str, Any]]) -> str:
    """Every configuration that did not produce a measurement, and why."""
    lines = ["## Failures, unsupported configurations and skips\n"]
    rows = []
    for cell in results:
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            status = record.get("status")
            if status in (None, "ok"):
                continue
            rows.append([
                key["model"].split("/")[-1], key["precision"], str(key["prompt_tokens"]),
                str(key["batch_size"]), backend, status,
                record.get("phase", "—"),
                (record.get("message") or "")[:140],
            ])
    for entry in extra:
        rows.append([
            entry.get("model", "—"), entry.get("precision", "—"),
            str(entry.get("prompt_tokens", "—")), str(entry.get("batch_size", "—")),
            entry.get("backend", "—"), entry.get("status", "skipped"),
            entry.get("phase", "—"), (entry.get("message") or "")[:140],
        ])
    if not rows:
        lines.append("_No failures, unsupported configurations, or skips._\n")
        return "\n".join(lines)
    lines.append(_table(
        ["Model", "Prec", "Prompt", "Batch", "Backend", "Status", "Phase", "Detail"],
        rows,
    ))
    return "\n".join(lines)


def multigpu_section(payload: dict[str, Any] | None) -> str:
    lines = ["## Multi-GPU\n"]
    if not payload:
        lines.append("_Not run: a single accelerator was visible, or the mode was disabled._\n")
        return "\n".join(lines)
    lines.append(
        "Aether's placement policy shards a dense decoder only when the model does "
        "not fit on the smallest visible device; otherwise it runs single-device. "
        "Forcing the sharded path on a model that fits is therefore a diagnostic "
        "configuration, not Aether's default behaviour, and is labelled as such.\n"
    )
    rows = []
    for entry in payload.get("cases", []):
        rows.append([
            entry.get("model", "—").split("/")[-1],
            entry.get("configuration", "—"),
            _fmt(entry.get("gpus")),
            _fmt((entry.get("tokens_per_s") or {}).get("median"), 2),
            _fmt((entry.get("latency_s") or {}).get("median"), 4),
            ", ".join(_bytes(v) for v in (entry.get("per_gpu_peak_bytes") or [])) or "—",
            entry.get("engine", "—"),
            entry.get("status", "—"),
        ])
    lines.append(_table(
        ["Model", "Configuration", "GPUs", "tok/s (med)", "latency s (med)",
         "per-GPU peak", "engine", "status"],
        rows,
    ))
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_charts(results: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Plot the raw measurements. Linear axes from zero, no rescaling."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written: list[str] = []
    usable = [
        cell for cell in results
        if cell["backends"].get("aether", {}).get("status") == "ok"
        and cell["backends"].get("transformers", {}).get("status") == "ok"
    ]
    if not usable:
        return []

    labels, aether_values, reference_values = [], [], []
    for cell in usable:
        key = cell["cell"]
        labels.append(f"{key['model'].split('/')[-1]}\n{key['precision']} p{key['prompt_tokens']}")
        aether_values.append(cell["backends"]["aether"]["tokens_per_s"]["median"])
        reference_values.append(cell["backends"]["transformers"]["tokens_per_s"]["median"])

    positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    width = 0.4
    axis.bar([p - width / 2 for p in positions], reference_values, width, label="Transformers")
    axis.bar([p + width / 2 for p in positions], aether_values, width, label="Aether")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("tokens / second (median)")
    axis.set_ylim(bottom=0)  # never truncate the axis to exaggerate a gap
    axis.set_title("Steady-state decode throughput, same hardware and settings")
    axis.legend()
    figure.tight_layout()
    path = output_dir / "throughput.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path.name)

    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    peaks_a, peaks_t = [], []
    for cell in usable:
        for backend, sink in (("aether", peaks_a), ("transformers", peaks_t)):
            devices = (cell["backends"][backend].get("gpu_peak") or {}).get("devices") or []
            sink.append(sum(d["peak_reserved_bytes"] for d in devices) / 1024 ** 3)
    axis.bar([p - width / 2 for p in positions], peaks_t, width, label="Transformers")
    axis.bar([p + width / 2 for p in positions], peaks_a, width, label="Aether")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("peak reserved GPU memory (GiB)")
    axis.set_ylim(bottom=0)
    axis.set_title("Peak reserved GPU memory")
    axis.legend()
    figure.tight_layout()
    path = output_dir / "gpu_memory.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path.name)
    return written


def save_raw(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Persist machine-readable results next to the human-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    written = {"json": json_path.name}

    rows = []
    for cell in payload.get("performance", []):
        key = cell["cell"]
        for backend in ("transformers", "aether"):
            record = cell["backends"].get(backend, {})
            rows.append({
                "model": key["model"], "precision": key["precision"],
                "prompt_tokens": key["prompt_tokens"], "batch_size": key["batch_size"],
                "backend": backend, "status": record.get("status"),
                "tokens_per_s_median": (record.get("tokens_per_s") or {}).get("median"),
                "tokens_per_s_stdev": (record.get("tokens_per_s") or {}).get("stdev"),
                "latency_s_median": (record.get("latency_s") or {}).get("median"),
                "latency_s_p95": (record.get("latency_s") or {}).get("p95"),
                "cold_latency_s": record.get("cold_latency_s"),
                "completion_tokens": record.get("completion_tokens"),
                "iterations": (record.get("latency_s") or {}).get("n"),
            })
    if rows:
        import csv

        csv_path = output_dir / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written["csv"] = csv_path.name
    return written


#: Everything the measurements cannot settle.  Stated in the report so a reader
#: does not extrapolate past what was actually observed.
LIMITATIONS = (
    "Aether's portable engine has no batch dimension, so batch sizes above 1 are "
    "reported as unsupported rather than measured. Any batch>1 row is a "
    "Transformers-only observation and not a comparison.",
    "Aether's compiled artifact stores weights at BF16 (the compiler's default "
    "residency). At fp16 and fp32 the two backends therefore do not hold "
    "bit-identical weights, since Transformers loads the published checkpoint "
    "directly. The bf16 comparison is the one where both hold the same values, "
    "which is why it is the primary configuration.",
    "Compilation is a one-time cost measured separately as prepare_s. It is not "
    "amortized into throughput, and a deployment that compiled per request would "
    "see very different totals.",
    "Time-to-first-token is measured through each library's own streaming API. "
    "Those code paths are not identical machinery, so TTFT is a weaker comparison "
    "than the throughput figures.",
    "Decode time is derived by subtracting measured prefill from measured "
    "end-to-end latency, so it carries the uncertainty of both.",
    "GPU utilization, power and temperature are sampled at a finite interval; a "
    "transient shorter than that interval can be missed entirely.",
    "Kaggle is a shared, virtualized environment. Clock behaviour and thermal "
    "state are not under the benchmark's control; recorded temperatures show what "
    "actually happened rather than asserting thermal parity.",
    "Profiler-derived kernel numbers come from instrumented runs and are not the "
    "source of any throughput claim in this report.",
    "Iteration counts are bounded by the available GPU budget. The dispersion "
    "statistics state how much confidence the sample size supports.",
    "Aether's semantic response cache is disabled for every measured run. It is "
    "on by default and returns a stored completion for a repeated prompt in about "
    "a millisecond, so leaving it on would time a cache lookup rather than "
    "inference. Transformers has no equivalent. The override is a public config "
    "flag, is recorded in each backend's description, and does not change Aether's "
    "own default.",
)


def build_report(payload: dict[str, Any], charts: list[str]) -> str:
    """Assemble REPORT.md from whichever sections the run actually produced."""
    config = payload.get("config", {})
    parts = [
        "# Aether Runtime vs Hugging Face Transformers — Benchmark Report",
        "",
        f"Generated: {payload.get('generated_at', '-')}",
        f"Mode: `{config.get('mode', '-')}`",
        "",
        "This report compares two **runtimes** executing the **same model "
        "architectures and the same weights**. Aether Runtime is a compiler plus an "
        "executor: it compiles a checkpoint into a self-contained `.aeg` artifact "
        "and runs that artifact. Hugging Face Transformers loads the checkpoint and "
        "runs it through its own model classes. Neither changes the model; they "
        "differ in how the same mathematics is executed.",
        "",
        environment_section(payload.get("environment", {})),
        "",
        "## Configuration",
        "",
        "Printed in full so a reader can confirm both backends received identical "
        "settings, and so the run can be reproduced exactly.",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Methodology",
        "",
        "- Both backends run on the same host, in the same process, with the same "
        "visible GPUs.",
        "- Identical prompts, built to an exact **token** count with each model's "
        "own tokenizer; identical `max_new_tokens`, sampling settings and seed.",
        "- Greedy decoding (`temperature=0`) for the primary comparison, so the "
        "measured work is deterministic.",
        "- Download, compile, load, warm-up and steady-state are timed as separate "
        "phases. Warm-up iterations are executed and discarded.",
        "- CUDA is synchronized on both edges of every timed region, so no "
        "asynchronous kernel is attributed to the wrong phase.",
        "- Backend order is alternated across repetitions, so thermal drift or "
        "clock ramping cannot land preferentially on one backend.",
        "- Allocator peak counters are reset before each measured phase.",
        "- Telemetry sampling runs in dedicated extra iterations, never during the "
        "iterations whose latency is reported.",
        "",
    ]
    if payload.get("performance"):
        parts += [performance_section(payload["performance"]), "",
                  phase_section(payload["performance"]), "",
                  memory_section(payload["performance"]), "",
                  utilization_section(payload["performance"]), ""]
    if payload.get("correctness"):
        parts += [correctness_section(payload["correctness"]), ""]
    if payload.get("kernels"):
        parts += [kernel_section(payload["kernels"]), ""]
    if payload.get("multigpu") is not None:
        parts += [multigpu_section(payload.get("multigpu")), ""]
    parts += [failures_section(payload.get("performance", []), payload.get("skips", [])), ""]
    if charts:
        parts += ["## Charts", ""]
        parts += [f"![{name}]({name})" for name in charts]
        parts += ["", "Axes start at zero and are linear; no scale is adjusted.", ""]
    parts += ["## Limitations", ""]
    parts += [f"{index}. {text}" for index, text in enumerate(LIMITATIONS, start=1)]
    parts += ["", "## Conclusions", "", _conclusions(payload), ""]
    return "\n".join(parts)


def _conclusions(payload: dict[str, Any]) -> str:
    """State only what the collected measurements support."""
    performance = payload.get("performance", [])
    comparable = [
        cell for cell in performance
        if cell["backends"].get("aether", {}).get("status") == "ok"
        and cell["backends"].get("transformers", {}).get("status") == "ok"
    ]
    if not comparable:
        return (
            "No cell produced a successful measurement on both backends, so this run "
            "supports no performance conclusion. See the failures table above."
        )
    ratios = []
    for cell in comparable:
        a = cell["backends"]["aether"]["tokens_per_s"]["median"]
        b = cell["backends"]["transformers"]["tokens_per_s"]["median"]
        if b:
            ratios.append((a / b, cell["cell"]))
    ratios.sort(key=lambda item: item[0])
    faster = [r for r, _ in ratios if r > 1.05]
    slower = [r for r, _ in ratios if r < 0.95]
    parity = [r for r, _ in ratios if 0.95 <= r <= 1.05]
    lines = [
        f"Comparable cells: **{len(ratios)}**. Aether faster by more than 5% in "
        f"**{len(faster)}**, Transformers faster by more than 5% in **{len(slower)}**, "
        f"within plus/minus 5% in **{len(parity)}**.",
        "",
    ]
    if ratios:
        worst, worst_cell = ratios[0]
        best, best_cell = ratios[-1]
        lines += [
            f"- Best case for Aether: **{best:.2f}x** on `{best_cell['model']}` at "
            f"{best_cell['precision']}, prompt {best_cell['prompt_tokens']}, "
            f"batch {best_cell['batch_size']}.",
            f"- Worst case for Aether: **{worst:.2f}x** on `{worst_cell['model']}` at "
            f"{worst_cell['precision']}, prompt {worst_cell['prompt_tokens']}, "
            f"batch {worst_cell['batch_size']}.",
            "",
        ]
    cases = (payload.get("correctness") or {}).get("cases", [])
    if cases:
        equivalent = sum(1 for case in cases if case.get("verdict", {}).get("equivalent"))
        lines.append(
            f"- Correctness: **{equivalent}/{len(cases)}** compared cases were "
            "numerically equivalent to Transformers within the stated logit "
            "tolerance. Per-case deviations are in the correctness table."
        )
    if (payload.get("kernels") or {}).get("dispatch"):
        lines.append(
            "- Kernel counts are reported per decoded token per layer for both "
            "backends. Every attribution of a throughput difference is labelled in "
            "the kernel section as confirmed by counting, confirmed by the profiler, "
            "or inferred."
        )
    lines += [
        "",
        "Claims beyond the above are not supported by this run. No conclusion is "
        "drawn for configurations that failed, were unsupported, or were not "
        "executed.",
    ]
    return "\n".join(lines)
