"""The one command: run the whole multi-engine benchmark and write every artifact.

    python benchmark.py                 # the full suite, every locked model, every engine
    python benchmark.py --smoke         # smallest real run: proves the pipeline works
    python benchmark.py --engines aether,transformers --models Qwen/Qwen3-0.6B
    python benchmark.py --resume        # reuse raw results already on disk

What it does, in order: detect the hardware, resolve the precision the hardware can
actually execute, survey which engines are installed and applicable, build the shared
prompts, run each engine in its own process, collect every measurement and every
reason a measurement is missing, derive the comparisons, draw the figures, and write
the JSON, the CSVs and the report.

Nothing here fabricates a number. An engine that could not run appears in the output
with the reason it could not, never as a zero, and never omitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
for _candidate in (_ROOT, _ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from benchmark.suite import analysis as analysis_mod  # noqa: E402
from benchmark.suite import charts as charts_mod  # noqa: E402
from benchmark.suite import orchestrate  # noqa: E402
from benchmark.suite import plan as plan_mod  # noqa: E402
from benchmark.suite import report as report_mod  # noqa: E402

#: Printed as the run progresses, so a long run shows what has completed. The list is
#: derived from what actually happened, not asserted up front.
STAGES = (
    "Environment detected",
    "Engines surveyed",
    "Prompts built",
    "Engines measured",
    "Correctness validated",
    "Batch scaling completed",
    "Statistics calculated",
    "Figures generated",
    "JSON written",
    "CSV written",
    "Report generated",
)


def _log(message: str) -> None:
    print(message, flush=True)


def _tick(label: str, detail: str = "") -> None:
    _log(f"  [ok] {label}" + (f" - {detail}" if detail else ""))


def _cross(label: str, detail: str) -> None:
    _log(f"  [--] {label} - {detail}")


def main(argv: list[str] | None = None) -> int:
    config = plan_mod.parse_args(argv)
    root = Path(config.output_dir)

    payload = orchestrate.run(config)

    raw_dir = root / orchestrate.RAW_DIR
    graph_dir = root / orchestrate.GRAPH_DIR
    report_dir = root / orchestrate.REPORT_DIR
    for directory in (raw_dir, graph_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _log(f"\n{'=' * 78}\nANALYSIS\n{'=' * 78}\n")
    analysis = analysis_mod.analyze(payload)
    measured = sum(
        1 for row in analysis["rows"] if row.get("status") == "MEASURED"
    )
    engines_measured = sorted({
        row["engine"] for row in analysis["rows"] if row.get("status") == "MEASURED"
    })
    _tick("Environment detected",
          f"{payload['hardware']['accelerator']}, precision "
          f"{payload['plan']['resolved_precision']}")
    _tick("Engines surveyed", f"{len(config.engines)} attempted")
    _tick("Prompts built", f"{len(config.prompt_tokens)} lengths per model")
    if engines_measured:
        _tick("Engines measured", ", ".join(engines_measured))
    else:
        _cross("Engines measured", "no engine produced a measurement")
    correctness_cases = analysis["correctness"]["cases"]
    if correctness_cases:
        _tick("Correctness validated", f"{len(correctness_cases)} engine comparisons")
    else:
        _cross("Correctness validated", "no engine pair produced comparable output")
    scaled = [
        entry for entry in analysis["batch_scaling"]
        if entry.get("scaling_efficiency_percent") is not None
        and int(entry.get("batch_size") or 1) > 1
    ]
    if scaled:
        _tick("Batch scaling completed", f"{len(scaled)} cells beyond batch 1")
    else:
        _cross("Batch scaling completed", "no engine completed a batch width above 1")
    _tick("Statistics calculated", f"{measured} measured cells")

    charts: dict[str, Any] = {"written": [], "skipped": [], "directory": orchestrate.GRAPH_DIR}
    if config.charts:
        charts.update(charts_mod.write_all(analysis, graph_dir))
        charts["directory"] = orchestrate.GRAPH_DIR
        if charts["written"]:
            _tick("Figures generated", f"{len(charts['written'])} written to {graph_dir}")
        else:
            _cross("Figures generated",
                   "; ".join(item["reason"] for item in charts["skipped"][:2]))
    else:
        _cross("Figures generated", "disabled with --no-charts")

    results_path = root / "benchmark_results.json"
    results_path.write_text(
        json.dumps({"raw": payload, "analysis": analysis}, indent=2, default=str),
        encoding="utf-8",
    )
    _tick("JSON written", str(results_path))

    csv_path = root / "benchmark_results.csv"
    report_mod.write_csv(analysis, csv_path)
    comparison_csv = root / "benchmark_comparisons.csv"
    report_mod.write_comparison_csv(analysis, comparison_csv)
    _tick("CSV written", f"{csv_path}, {comparison_csv}")

    report_path = report_dir / "BENCHMARK_REPORT.md"
    report_path.write_text(
        report_mod.build_report(payload, analysis, charts), encoding="utf-8"
    )
    _tick("Report generated", str(report_path))

    _log(report_mod.terminal_summary(payload, analysis, {
        "report": str(report_path),
        "raw records": str(raw_dir),
        "figures": str(graph_dir),
        "results JSON": str(results_path),
        "results CSV": str(csv_path),
        "comparisons CSV": str(comparison_csv),
    }))
    # Zero even when engines failed: a run that correctly recorded a field of
    # unavailable engines did its job. A non-zero exit is reserved for a run that
    # produced nothing at all, which is a harness problem rather than a result.
    return 0 if measured else 1


if __name__ == "__main__":
    raise SystemExit(main())
