"""Plot the measurements, and nothing else.

Rules this module exists to enforce, all of them about what a chart is not allowed
to do:

* an axis starts at zero and stays linear, so a small difference cannot be drawn as
  a large one;
* a missing measurement is a missing point. It is never zero, never interpolated,
  and never smoothed over. Engines with no data for a panel are listed on the panel
  as unavailable, with the status that explains them;
* no series is reordered or omitted to improve the picture.

Every figure degrades to "not produced" if matplotlib is absent, and one figure
failing never prevents the others.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.suite import status as status_mod

#: Fixed engine colours, so the same engine is the same colour in every figure and a
#: reader can follow one stack across twelve panels. Chosen to stay distinguishable
#: in greyscale and under the common forms of colour blindness.
COLORS: dict[str, str] = {
    "aether": "#0F62FE",
    "transformers": "#6F6F6F",
    "pytorch_native": "#8A3FFC",
    "torch_compile": "#1192E8",
    "onnxruntime": "#005D5D",
    "openvino": "#009D9A",
    "llama_cpp": "#A56EFF",
    "vllm": "#EE538B",
    "sglang": "#FA4D56",
    "tensorrt_llm": "#198038",
    "deepspeed": "#B28600",
    "exllamav2": "#9F1853",
    "mlc": "#570408",
}
FALLBACK_COLOR = "#525252"

#: Aether is drawn last so it sits on top where series overlap, and is drawn thicker.
#: That is presentational emphasis on the subject of the report, not a change to any
#: value.
SUBJECT = "aether"


def _color(engine: str) -> str:
    return COLORS.get(engine, FALLBACK_COLOR)


def _short(model: str) -> str:
    return model.split("/")[-1]


def _pyplot() -> Any:
    """Import matplotlib in headless mode, or return None if it is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        return plt
    except ImportError:
        return None


def _note_unavailable(axis: Any, missing: dict[str, str]) -> None:
    """Print, on the figure, which engines have no data here and why.

    A chart that simply lacks a bar invites the reader to assume the engine was slow.
    Naming the absent engines and their status on the panel itself is what prevents
    that reading.
    """
    if not missing:
        return
    lines = [f"not measured: {engine} ({reason})" for engine, reason in
             sorted(missing.items())]
    axis.text(
        0.99, 0.02, "\n".join(lines[:6]), transform=axis.transAxes,
        ha="right", va="bottom", fontsize=6.5, color="#525252",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#E0E0E0", "pad": 3},
    )


def _save(figure: Any, plt: Any, directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / name)
    plt.close(figure)
    return name


def _measured(rows: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    """Rows that carry a measurement and match every filter given."""
    selected = []
    for row in rows:
        if not status_mod.is_measured(row):
            continue
        if all(row.get(key) == value for key, value in filters.items()):
            selected.append(row)
    return selected


def _unavailable(rows: list[dict[str, Any]], model: str) -> dict[str, str]:
    """Engines with no measurement for a model, mapped to their status."""
    statuses: dict[str, str] = {}
    for row in rows:
        if row.get("model") != model:
            continue
        engine = row.get("engine")
        if status_mod.is_measured(row):
            statuses.pop(engine, None)
            statuses[engine] = ""
        elif engine not in statuses:
            statuses[engine] = str(row.get("status"))
    return {engine: reason for engine, reason in statuses.items() if reason}


def throughput_ranking(plt: Any, analysis: dict[str, Any], directory: Path,
                       model: str) -> str | None:
    """Graph 1: every engine's batch-1 generation throughput, ranked."""
    rows = _measured(analysis["rows"], model=model, is_primary=True)
    rows = [row for row in rows if row.get(analysis["primary_metric"])]
    if not rows:
        return None
    rows.sort(key=lambda row: float(row[analysis["primary_metric"]]))
    figure, axis = plt.subplots(figsize=(7.5, max(2.6, 0.42 * len(rows) + 1.4)))
    positions = range(len(rows))
    values = [float(row[analysis["primary_metric"]]) for row in rows]
    axis.barh(
        list(positions), values,
        color=[_color(row["engine"]) for row in rows],
        edgecolor=["#161616" if row["engine"] == SUBJECT else "none" for row in rows],
        linewidth=[1.2 if row["engine"] == SUBJECT else 0 for row in rows],
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels([row["engine"] for row in rows])
    axis.set_xlabel(analysis["primary_metric_label"])
    axis.set_xlim(left=0)
    axis.set_title(f"Batch-1 generation throughput - {_short(model)}")
    for position, value in zip(positions, values, strict=False):
        axis.text(value, position, f" {value:,.1f}", va="center", fontsize=7.5)
    _note_unavailable(axis, _unavailable(analysis["rows"], model))
    return _save(figure, plt, directory, f"01_throughput_ranking__{_short(model)}.png")


def batch_scaling(plt: Any, analysis: dict[str, Any], directory: Path,
                  model: str) -> str | None:
    """Graph 2: aggregate throughput against batch width, one line per engine."""
    entries = [
        entry for entry in analysis["batch_scaling"]
        if entry["model"] == model and entry.get("batch_tokens_per_s")
        and status_mod.is_measured(entry)
    ]
    if not entries:
        return None
    by_engine: dict[str, list[tuple[int, float]]] = {}
    for entry in entries:
        by_engine.setdefault(entry["engine"], []).append(
            (int(entry["batch_size"]), float(entry["batch_tokens_per_s"]))
        )
    if not any(len(points) > 1 for points in by_engine.values()):
        # Only one batch width was measured. A single-point line would suggest a
        # trend the run never observed.
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for engine in sorted(by_engine, key=lambda name: name == SUBJECT):
        points = sorted(by_engine[engine])
        axis.plot(
            [point[0] for point in points], [point[1] for point in points],
            marker="o", markersize=4, color=_color(engine), label=engine,
            linewidth=2.4 if engine == SUBJECT else 1.5,
        )
    axis.set_xscale("log", base=2)
    widths = sorted({point[0] for points in by_engine.values() for point in points})
    axis.set_xticks(widths)
    axis.set_xticklabels([str(width) for width in widths])
    axis.set_xlabel("batch size (log2 scale; measured widths only)")
    axis.set_ylabel(analysis["primary_metric_label"])
    axis.set_ylim(bottom=0)
    axis.set_title(f"Throughput against batch width - {_short(model)}")
    axis.legend(fontsize=7, ncol=2)
    _note_unavailable(axis, {
        entry["engine"]: str(entry["status"]) for entry in analysis["batch_scaling"]
        if entry["model"] == model and not status_mod.is_measured(entry)
    })
    return _save(figure, plt, directory, f"02_batch_scaling__{_short(model)}.png")


def scaling_efficiency(plt: Any, analysis: dict[str, Any], directory: Path,
                       model: str) -> str | None:
    """Graph 3: how much of ideal linear batch scaling each engine achieved."""
    entries = [
        entry for entry in analysis["batch_scaling"]
        if entry["model"] == model and entry.get("scaling_efficiency_percent") is not None
    ]
    if not entries:
        return None
    by_engine: dict[str, list[tuple[int, float]]] = {}
    for entry in entries:
        if int(entry["batch_size"]) == 1:
            # Batch 1 is the denominator; plotting it as 100% adds a point that is
            # true by construction rather than measured.
            continue
        by_engine.setdefault(entry["engine"], []).append(
            (int(entry["batch_size"]), float(entry["scaling_efficiency_percent"]))
        )
    if not by_engine:
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for engine in sorted(by_engine, key=lambda name: name == SUBJECT):
        points = sorted(by_engine[engine])
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                  markersize=4, color=_color(engine), label=engine,
                  linewidth=2.4 if engine == SUBJECT else 1.5)
    axis.axhline(100.0, color="#161616", linestyle="--", linewidth=1.0)
    axis.text(0.01, 100.0, " 100% = perfectly linear scaling", fontsize=7,
              va="bottom", transform=axis.get_yaxis_transform())
    axis.set_xscale("log", base=2)
    widths = sorted({p[0] for points in by_engine.values() for p in points})
    axis.set_xticks(widths)
    axis.set_xticklabels([str(width) for width in widths])
    axis.set_xlabel("batch size")
    axis.set_ylabel("scaling efficiency (% of linear)")
    axis.set_ylim(bottom=0)
    axis.set_title(f"Batch scaling efficiency - {_short(model)}")
    axis.legend(fontsize=7, ncol=2)
    return _save(figure, plt, directory, f"03_scaling_efficiency__{_short(model)}.png")


def _simple_bar(plt: Any, analysis: dict[str, Any], directory: Path, model: str,
                metric: str, ylabel: str, title: str, filename: str,
                lower_is_better: bool, scale: float = 1.0) -> str | None:
    rows = [
        row for row in _measured(analysis["rows"], model=model, is_primary=True)
        if row.get(metric)
    ]
    if not rows:
        return None
    rows.sort(key=lambda row: float(row[metric]), reverse=not lower_is_better)
    figure, axis = plt.subplots(figsize=(max(5.5, 0.85 * len(rows) + 2.2), 4.0))
    values = [float(row[metric]) * scale for row in rows]
    axis.bar(
        range(len(rows)), values,
        color=[_color(row["engine"]) for row in rows],
        edgecolor=["#161616" if row["engine"] == SUBJECT else "none" for row in rows],
        linewidth=[1.2 if row["engine"] == SUBJECT else 0 for row in rows],
    )
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([row["engine"] for row in rows], rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0)
    suffix = " (lower is better)" if lower_is_better else " (higher is better)"
    axis.set_title(f"{title} - {_short(model)}{suffix}")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:,.2f}", ha="center", va="bottom", fontsize=7)
    _note_unavailable(axis, _unavailable(analysis["rows"], model))
    return _save(figure, plt, directory, filename)


def compile_tradeoff(plt: Any, analysis: dict[str, Any], directory: Path) -> str | None:
    """Graph 7: build cost against the steady-state throughput it buys.

    A scatter rather than a bar chart, because the question is a trade-off and not a
    ranking: up and to the left is a cheap build that bought speed, down and to the
    right is an expensive one that did not. Engines with no build phase are drawn at
    zero on the x axis, which is their true build cost, not a missing value.
    """
    entries = [
        entry for entry in analysis["compile_economics"]["entries"]
        if entry.get("steady_state_latency_s") and entry.get("total_s") is not None
    ]
    if not entries:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for entry in entries:
        throughput = 1.0 / float(entry["steady_state_latency_s"])
        build = float(entry.get("build_s") or 0.0)
        axis.scatter(build, throughput, s=70 if entry["engine"] == SUBJECT else 45,
                     color=_color(entry["engine"]),
                     edgecolor="#161616" if entry["engine"] == SUBJECT else "none",
                     zorder=3)
        axis.annotate(
            f"{entry['engine']}\n{_short(entry['model'])}",
            (build, throughput), textcoords="offset points", xytext=(6, 4), fontsize=6.5,
        )
    axis.set_xlabel("build / compile time (s); 0 means the engine has no build phase")
    axis.set_ylabel("requests per second at batch 1 (1 / end-to-end latency)")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_title("Build cost against the steady-state speed it buys")
    return _save(figure, plt, directory, "07_compile_tradeoff.png")


def improvement_bars(plt: Any, analysis: dict[str, Any], directory: Path) -> str | None:
    """Graph 8: Aether's median improvement against each competitor.

    Signed, with zero drawn: a bar below the line is Aether losing, and it is
    rendered exactly as prominently as a bar above it. Bars whose comparison crosses
    a representation boundary are hatched, so a quantized competitor's result cannot
    be read as a same-weights claim.
    """
    per_competitor = analysis["per_competitor"]
    if not per_competitor:
        return None
    engines = sorted(per_competitor, key=lambda key: per_competitor[key][
        "median_improvement_percent"])
    values = [per_competitor[key]["median_improvement_percent"] for key in engines]
    figure, axis = plt.subplots(figsize=(max(5.5, 0.9 * len(engines) + 2.4), 4.2))
    bars = axis.bar(range(len(engines)), values,
                    color=["#24A148" if value >= 0 else "#DA1E28" for value in values])
    for index, key in enumerate(engines):
        if per_competitor[key]["comparability"] != "SAME_REPRESENTATION":
            bars[index].set_hatch("//")
            bars[index].set_edgecolor("#161616")
    axis.axhline(0.0, color="#161616", linewidth=1.0)
    axis.set_xticks(range(len(engines)))
    axis.set_xticklabels(engines, rotation=30, ha="right")
    axis.set_ylabel("Aether median improvement (%)")
    axis.set_title(
        "Aether against each competitor - positive is Aether faster, "
        "hatched crosses a representation boundary"
    )
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:+.1f}%", ha="center",
                  va="bottom" if value >= 0 else "top", fontsize=7)
    return _save(figure, plt, directory, "08_aether_improvement.png")


def _sweep_line(plt: Any, analysis: dict[str, Any], directory: Path, model: str,
                sweep: str, axis_key: str, xlabel: str, title: str,
                filename: str) -> str | None:
    """Graphs 9 and 10: throughput against prompt length, and against output length."""
    rows = [
        row for row in _measured(analysis["rows"], model=model)
        if sweep in (row.get("sweeps") or []) and row.get(analysis["primary_metric"])
        and row.get("batch_size") == 1
    ]
    if not rows:
        return None
    by_engine: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_engine.setdefault(row["engine"], []).append(
            (int(row[axis_key]), float(row[analysis["primary_metric"]]))
        )
    if not any(len(points) > 1 for points in by_engine.values()):
        # A single point per engine is not a curve; drawing one would suggest a
        # trend that was never measured.
        return None
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for engine in sorted(by_engine, key=lambda name: name == SUBJECT):
        points = sorted(by_engine[engine])
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                  markersize=4, color=_color(engine), label=engine,
                  linewidth=2.4 if engine == SUBJECT else 1.5)
    lengths = sorted({p[0] for points in by_engine.values() for p in points})
    axis.set_xticks(lengths)
    axis.set_xticklabels([str(length) for length in lengths])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(analysis["primary_metric_label"])
    axis.set_ylim(bottom=0)
    axis.set_title(f"{title} - {_short(model)}")
    axis.legend(fontsize=7, ncol=2)
    return _save(figure, plt, directory, filename)


def model_scaling(plt: Any, analysis: dict[str, Any], directory: Path) -> str | None:
    """Graph 11: every engine across every model, at batch 1."""
    rows = [
        row for row in _measured(analysis["rows"], is_primary=True)
        if row.get(analysis["primary_metric"])
    ]
    if not rows:
        return None
    models = sorted({row["model"] for row in rows})
    engines = sorted({row["engine"] for row in rows}, key=lambda name: name == SUBJECT)
    if len(models) < 2:
        return None
    figure, axis = plt.subplots(figsize=(max(7.0, 1.9 * len(models) + 2.0), 4.4))
    width = 0.8 / max(len(engines), 1)
    lookup = {(row["model"], row["engine"]): float(row[analysis["primary_metric"]])
              for row in rows}
    for index, engine in enumerate(engines):
        offsets, values = [], []
        for position, model in enumerate(models):
            value = lookup.get((model, engine))
            if value is None:
                # Omitted, not zeroed: a bar of height zero would claim the engine
                # ran this model and produced nothing.
                continue
            offsets.append(position + index * width - 0.4 + width / 2)
            values.append(value)
        if values:
            axis.bar(offsets, values, width, color=_color(engine), label=engine)
    axis.set_xticks(range(len(models)))
    axis.set_xticklabels([_short(model) for model in models], rotation=15, ha="right")
    axis.set_ylabel(analysis["primary_metric_label"])
    axis.set_ylim(bottom=0)
    axis.set_title("Batch-1 throughput across models (absent bars were not measured)")
    axis.legend(fontsize=7, ncol=3)
    return _save(figure, plt, directory, "11_model_scaling.png")


def heatmap(plt: Any, analysis: dict[str, Any], directory: Path) -> str | None:
    """Graph 12: engine by (model, batch), coloured by throughput.

    Cells with no measurement are left blank and labelled with their status, so the
    grid shows the shape of the field's coverage as well as its speeds. A colour scale
    cannot represent "not measured", so it is not asked to.
    """
    rows = [
        row for row in analysis["rows"]
        if row.get("batch_size") is not None and "batch" in (row.get("sweeps") or [])
    ]
    if not rows:
        return None
    columns = sorted({(row["model"], int(row["batch_size"])) for row in rows},
                     key=lambda item: (item[0], item[1]))
    # Every engine the run attempted gets a row, including the ones that never
    # produced a cell. A grid that silently drops them would show the field as
    # smaller than it was and hide which stacks this host could not run.
    engines = sorted({row["engine"] for row in analysis["rows"] if row.get("engine")})
    engine_status = {
        row["engine"]: str(row.get("status"))
        for row in analysis["rows"] if row.get("batch_size") is None
    }
    lookup: dict[tuple[str, tuple[str, int]], Any] = {}
    for engine in engines:
        for column in columns:
            lookup[(engine, column)] = engine_status.get(engine, status_mod.SKIPPED)
    for row in rows:
        key = (row["engine"], (row["model"], int(row["batch_size"])))
        if status_mod.is_measured(row) and row.get(analysis["primary_metric"]):
            lookup[key] = float(row[analysis["primary_metric"]])
        else:
            lookup[key] = str(row.get("status"))

    import numpy as np

    grid = np.full((len(engines), len(columns)), np.nan)
    for row_index, engine in enumerate(engines):
        for column_index, column in enumerate(columns):
            value = lookup.get((engine, column))
            if isinstance(value, float):
                grid[row_index, column_index] = value

    figure, axis = plt.subplots(
        figsize=(max(7.0, 0.95 * len(columns) + 3.0), max(3.4, 0.42 * len(engines) + 1.8))
    )
    image = axis.imshow(grid, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([f"{_short(model)}\nb{batch}" for model, batch in columns],
                         fontsize=7)
    axis.set_yticks(range(len(engines)))
    axis.set_yticklabels(engines, fontsize=8)
    axis.set_title(f"{analysis['primary_metric_label']} - blank cells were not measured")
    axis.grid(False)
    finite = grid[~np.isnan(grid)]
    midpoint = (float(finite.min()) + float(finite.max())) / 2.0 if finite.size else 0.0
    for row_index, engine in enumerate(engines):
        for column_index, column in enumerate(columns):
            value = lookup.get((engine, column))
            if isinstance(value, float):
                # viridis runs dark to light, so the label has to flip with the
                # cell or half the numbers become unreadable.
                axis.text(column_index, row_index, f"{value:,.0f}", ha="center",
                          va="center", fontsize=6.5,
                          color="#161616" if value > midpoint else "white")
            else:
                axis.text(column_index, row_index, _abbreviate(value), ha="center",
                          va="center", fontsize=5.5, color="#525252")
    figure.colorbar(image, ax=axis, label=analysis["primary_metric_label"])
    return _save(figure, plt, directory, "12_heatmap.png")


#: Short forms for the status vocabulary, so a heatmap cell can carry the reason it
#: is empty without the label overflowing the cell.
_ABBREVIATIONS = {
    status_mod.NOT_INSTALLED: "n/inst",
    status_mod.NOT_SUPPORTED: "n/supp",
    status_mod.NOT_APPLICABLE: "n/appl",
    status_mod.FAILED: "failed",
    status_mod.OOM: "OOM",
    status_mod.SKIPPED: "skip",
}


def _abbreviate(value: Any) -> str:
    return _ABBREVIATIONS.get(str(value), "n/a")


def write_all(analysis: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Produce every figure the data supports, and record every one it does not.

    A figure is skipped when the measurements for it do not exist - never filled in.
    The returned manifest lists what was drawn and what was not, and the report prints
    the second list so an absent figure reads as absent data rather than an omission.
    """
    plt = _pyplot()
    if plt is None:
        return {
            "written": [],
            "skipped": [{"chart": "all", "reason": "matplotlib is not installed"}],
        }
    directory.mkdir(parents=True, exist_ok=True)
    models = sorted({
        row["model"] for row in analysis["rows"] if row.get("model")
    })
    written: list[str] = []
    skipped: list[dict[str, str]] = []

    def attempt(name: str, function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            result = function(plt, analysis, directory, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - one bad figure is not fatal
            skipped.append({"chart": name, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            return
        if result:
            written.append(result)
        else:
            skipped.append({"chart": name, "reason": "no measurement supports this figure"})

    for model in models:
        short = _short(model)
        attempt(f"01 throughput ranking [{short}]", throughput_ranking, model)
        attempt(f"02 batch scaling [{short}]", batch_scaling, model)
        attempt(f"03 scaling efficiency [{short}]", scaling_efficiency, model)
        attempt(f"04 batch-1 competition [{short}]", _simple_bar, model,
                analysis["primary_metric"], analysis["primary_metric_label"],
                "Batch-1 competition", f"04_batch1__{short}.png", False)
        attempt(f"05 time to first token [{short}]", _simple_bar, model, "ttft_s",
                "time to first token (s)", "Time to first token",
                f"05_ttft__{short}.png", True)
        attempt(f"06 peak host memory [{short}]", _simple_bar, model,
                "peak_host_bytes", "peak process RSS (GiB)", "Peak host memory",
                f"06_memory_host__{short}.png", True, 1.0 / 1024 ** 3)
        attempt(f"06b peak device memory [{short}]", _simple_bar, model,
                "peak_device_bytes", "peak reserved VRAM (GiB)", "Peak device memory",
                f"06b_memory_device__{short}.png", True, 1.0 / 1024 ** 3)
        attempt(f"09 throughput vs prompt length [{short}]", _sweep_line, model,
                "prompt", "prompt_tokens", "prompt length (tokens)",
                "Throughput against prompt length", f"09_prompt_length__{short}.png")
        attempt(f"10 throughput vs output length [{short}]", _sweep_line, model,
                "output", "output_tokens", "generated tokens requested",
                "Throughput against output length", f"10_output_length__{short}.png")
    attempt("07 compile trade-off", compile_tradeoff)
    attempt("08 Aether improvement", improvement_bars)
    attempt("11 model scaling", model_scaling)
    attempt("12 heatmap", heatmap)
    return {"written": written, "skipped": skipped}
