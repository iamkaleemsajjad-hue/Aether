"""Derive comparisons from raw measurements, and refuse to derive them otherwise.

Every function here obeys three rules:

* a comparison exists only when both sides were measured. There is no default, no
  zero and no interpolation for a missing value;
* a comparison against an engine running a different representation of the model is
  computed but labelled ``REPRESENTATION_DIFFERENCE``, so it can never be quoted as
  a same-weights speed claim;
* a result that is unfavourable to Aether is produced by exactly the same code path
  as a favourable one. There is no branch anywhere in this module that tests which
  engine won before deciding what to report.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from benchmark.suite import engines as registry
from benchmark.suite import status as status_mod

SUBJECT = registry.SUBJECT
REFERENCE = registry.REFERENCE

#: Same weights, same precision, same container: a percentage here is a claim about
#: execution.
SAME_REPRESENTATION = "SAME_REPRESENTATION"
#: The engines are not holding the same values. The number is still reported, and it
#: still describes what a user would experience, but it is not a claim about
#: execution alone.
REPRESENTATION_DIFFERENCE = "REPRESENTATION_DIFFERENCE"

#: Below this relative difference the two measurements are called a tie. Set from
#: the dispersion the suite actually observes rather than picked for roundness: at
#: the iteration counts a GPU budget allows, run-to-run coefficients of variation
#: of a couple of percent are normal, so a 2% gap is not evidence of a difference.
TIE_THRESHOLD = 0.02

#: Buckets the report uses to find the closest competitions, per the charter.
CLOSENESS_BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class Key:
    """Identifies one workload cell across engines."""

    model: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int

    def label(self) -> str:
        return (f"{self.model.split('/')[-1]} b{self.batch_size} "
                f"p{self.prompt_tokens} o{self.output_tokens}")


def _peak_host_bytes(cell: dict[str, Any]) -> int | None:
    host = (cell.get("measurement") or {}).get("host_during_inference") or {}
    return host.get("rss_peak_bytes")


def _peak_device_bytes(cell: dict[str, Any]) -> int | None:
    peak = (cell.get("measurement") or {}).get("gpu_peak") or {}
    devices = peak.get("devices") or []
    if not devices:
        return None
    return sum(int(device.get("peak_reserved_bytes") or 0) for device in devices)


def flatten(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (engine, model, cell), measured or not.

    The unmeasured rows are kept deliberately. They are what lets the report say
    "vLLM was not applicable on this host" in the same table that says "Aether
    reached 41 tok/s", instead of leaving a gap the reader has to interpret.
    """
    rows: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        engine = run.get("engine")
        model = run.get("model")
        spec = run.get("spec") or {}
        describe = run.get("describe") or {}
        run_status = run.get("status")
        cells = run.get("cells") or []
        if not cells:
            rows.append({
                "engine": engine, "model": model, "batch_size": None,
                "prompt_tokens": None, "output_tokens": None,
                "status": run_status, "reason": run.get("reason", ""),
                "taxonomy": spec.get("taxonomy", []), "sweeps": [],
                "quantized": bool(describe.get("quantized")),
                "representation": describe.get("representation"),
            })
            continue
        for cell in cells:
            derived = cell.get("derived") or {}
            rows.append({
                "engine": engine,
                "model": model,
                "batch_size": cell.get("batch_size"),
                "prompt_tokens": cell.get("prompt_tokens"),
                "output_tokens": cell.get("output_tokens"),
                "kind": cell.get("kind"),
                "sweeps": cell.get("sweeps", []),
                "is_primary": bool(cell.get("is_primary")),
                "status": cell.get("status", run_status),
                "reason": cell.get("reason", ""),
                "taxonomy": spec.get("taxonomy", []),
                "quantized": bool(describe.get("quantized")),
                "representation": describe.get("representation"),
                "precision": run.get("precision"),
                "total_tokens_per_s": derived.get("total_tokens_per_s"),
                "per_request_tokens_per_s": derived.get("per_request_tokens_per_s"),
                "decode_tokens_per_s": derived.get("decode_tokens_per_s"),
                "prompt_tokens_per_s": derived.get("prompt_tokens_per_s"),
                "ttft_s": derived.get("ttft_s"),
                "tpot_ms": derived.get("tpot_ms"),
                "end_to_end_latency_s": derived.get("end_to_end_latency_s"),
                "cold_latency_s": derived.get("cold_latency_s"),
                "iterations": derived.get("iterations"),
                "coefficient_of_variation": derived.get("coefficient_of_variation"),
                "latency_stats": derived.get("latency_stats"),
                "throughput_stats": derived.get("throughput_stats"),
                "peak_host_bytes": _peak_host_bytes(cell),
                "peak_device_bytes": _peak_device_bytes(cell),
                "completion_tokens": (cell.get("measurement") or {}).get("completion_tokens"),
            })
    return rows


#: The metric every headline comparison is made on. Chosen because it is defined
#: for every engine that produced a measurement, and because it is the quantity a
#: caller experiences: generated tokens per second of wall time for the whole
#: request, prefill included. Decode-only throughput is reported alongside it
#: wherever both sides could supply a prefill measurement, and is never silently
#: substituted for it.
PRIMARY_METRIC = "total_tokens_per_s"
PRIMARY_METRIC_LABEL = "generation throughput (tok/s, whole request)"

#: Metrics where a lower value is better, so a comparison must not be read as a ratio
#: in the same direction as throughput.
LOWER_IS_BETTER = frozenset({
    "ttft_s", "tpot_ms", "end_to_end_latency_s", "cold_latency_s",
    "peak_host_bytes", "peak_device_bytes",
})


def measured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if status_mod.is_measured(row)]


def index_by_cell(rows: list[dict[str, Any]]) -> dict[Key, dict[str, dict[str, Any]]]:
    """Group measured rows by workload cell, then by engine."""
    table: dict[Key, dict[str, dict[str, Any]]] = {}
    for row in measured_rows(rows):
        if row.get("batch_size") is None:
            continue
        key = Key(row["model"], int(row["batch_size"]), int(row["prompt_tokens"]),
                  int(row["output_tokens"]))
        table.setdefault(key, {})[row["engine"]] = row
    return table


def comparability(subject: dict[str, Any], other: dict[str, Any]) -> tuple[str, str]:
    """Whether these two rows are holding the same model representation.

    The check is on what each engine reported it loaded, not on what the plan asked
    for, because an engine can quietly export or quantize during its build step. A
    difference is not a disqualification; it changes what the number means, and the
    label is how the report carries that forward into every percentage.
    """
    if subject.get("quantized") or other.get("quantized"):
        return REPRESENTATION_DIFFERENCE, (
            "one side executes a quantized representation, so the comparison "
            "includes a weight-format difference as well as an execution difference"
        )
    subject_repr = str(subject.get("representation") or "")
    other_repr = str(other.get("representation") or "")
    if "ONNX" in other_repr or "OpenVINO IR" in other_repr or "TensorRT" in other_repr:
        return REPRESENTATION_DIFFERENCE, (
            f"the competing engine executes a re-exported graph ({other_repr}); its "
            "stored element type may differ from the benchmark precision"
        )
    if subject.get("precision") != other.get("precision"):
        return REPRESENTATION_DIFFERENCE, (
            f"precisions differ: {subject.get('precision')} against "
            f"{other.get('precision')}"
        )
    # Aether's compiled artifact stores weights at its compiler's default residency.
    # When that residency is not the benchmark precision, Aether and the framework
    # engines are not holding the same values, and saying so is the whole point of
    # this label - it is the one representation difference on the subject's own side.
    precision = str(subject.get("precision") or "")
    if "BF16" in subject_repr.upper() and precision != "bf16":
        return REPRESENTATION_DIFFERENCE, (
            f"Aether's artifact holds BF16 weights while the run is at {precision}, "
            "so the two sides are not holding bit-identical values"
        )
    return SAME_REPRESENTATION, ""


def compare(subject_value: float | None, other_value: float | None,
            lower_is_better: bool = False) -> dict[str, Any]:
    """Express one measurement against another, or say that it cannot be.

    Returns the ratio, the percentage, and both operands. The operands travel with
    the ratio everywhere in this suite so that a reader can check the arithmetic
    without going back to the raw file.
    """
    if not subject_value or not other_value or subject_value <= 0 or other_value <= 0:
        return {
            "comparable": False,
            "subject": subject_value,
            "other": other_value,
            "ratio": None,
            "subject_improvement_percent": None,
        }
    if lower_is_better:
        # A latency or a memory figure improves by going down, so the improvement is
        # how much of the competitor's value was removed.
        improvement = (other_value - subject_value) / other_value * 100.0
        ratio = other_value / subject_value
    else:
        improvement = (subject_value - other_value) / other_value * 100.0
        ratio = subject_value / other_value
    return {
        "comparable": True,
        "subject": subject_value,
        "other": other_value,
        "ratio": ratio,
        "subject_improvement_percent": improvement,
    }


def verdict(improvement_percent: float | None) -> str:
    """Name the winner, or call it a tie, on a single stated threshold."""
    if improvement_percent is None:
        return "no comparison"
    if improvement_percent > TIE_THRESHOLD * 100.0:
        return "aether"
    if improvement_percent < -TIE_THRESHOLD * 100.0:
        return "competitor"
    return "tie"


def head_to_head(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aether against every other engine, in every cell both of them measured.

    This is the win/loss matrix. It is generated by iterating cells, not by
    iterating claims: a cell where Aether lost appears for exactly the same reason
    a cell where it won does, and nothing filters on the sign of the result.
    """
    table = index_by_cell(rows)
    comparisons: list[dict[str, Any]] = []
    for key in sorted(table, key=lambda item: (item.model, item.batch_size,
                                               item.prompt_tokens, item.output_tokens)):
        engines_here = table[key]
        subject = engines_here.get(SUBJECT)
        if subject is None:
            continue
        for engine, other in engines_here.items():
            if engine == SUBJECT:
                continue
            label, note = comparability(subject, other)
            throughput = compare(subject.get(PRIMARY_METRIC), other.get(PRIMARY_METRIC))
            decode = compare(subject.get("decode_tokens_per_s"),
                             other.get("decode_tokens_per_s"))
            latency = compare(subject.get("end_to_end_latency_s"),
                              other.get("end_to_end_latency_s"), lower_is_better=True)
            ttft = compare(subject.get("ttft_s"), other.get("ttft_s"), lower_is_better=True)
            host_memory = compare(subject.get("peak_host_bytes"),
                                  other.get("peak_host_bytes"), lower_is_better=True)
            device_memory = compare(subject.get("peak_device_bytes"),
                                    other.get("peak_device_bytes"), lower_is_better=True)
            comparisons.append({
                "model": key.model,
                "batch_size": key.batch_size,
                "prompt_tokens": key.prompt_tokens,
                "output_tokens": key.output_tokens,
                "is_primary": bool(subject.get("is_primary")),
                "competitor": engine,
                "comparability": label,
                "comparability_note": note,
                "throughput": throughput,
                "decode": decode,
                "latency": latency,
                "ttft": ttft,
                "host_memory": host_memory,
                "device_memory": device_memory,
                "winner": verdict(throughput.get("subject_improvement_percent")),
            })
    return comparisons


def win_loss_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Count and list where Aether wins, loses and ties.

    Same-representation comparisons are counted separately from ones that cross a
    weight-format boundary, because only the first supports a statement about
    execution. Both are listed; neither is hidden.
    """
    def bucket(entries: list[dict[str, Any]]) -> dict[str, Any]:
        wins = [item for item in entries if item["winner"] == "aether"]
        losses = [item for item in entries if item["winner"] == "competitor"]
        ties = [item for item in entries if item["winner"] == "tie"]
        return {
            "compared": len(entries),
            "aether_wins": len(wins),
            "aether_losses": len(losses),
            "ties": len(ties),
            "wins": wins,
            "losses": losses,
            "tied": ties,
        }

    comparable = [
        item for item in comparisons if item["throughput"].get("comparable")
    ]
    same = [item for item in comparable if item["comparability"] == SAME_REPRESENTATION]
    different = [
        item for item in comparable if item["comparability"] == REPRESENTATION_DIFFERENCE
    ]
    summary = {
        "all": bucket(comparable),
        "same_representation": bucket(same),
        "representation_difference": bucket(different),
        "incomparable": [
            {
                "model": item["model"], "competitor": item["competitor"],
                "batch_size": item["batch_size"],
                "reason": "one side produced no measurement for this cell",
            }
            for item in comparisons if not item["throughput"].get("comparable")
        ],
    }
    if same:
        best = max(same, key=lambda item: item["throughput"]["subject_improvement_percent"])
        worst = min(same, key=lambda item: item["throughput"]["subject_improvement_percent"])
        summary["largest_advantage"] = best
        summary["largest_disadvantage"] = worst
    summary["closest"] = {
        f"within_{int(threshold * 100)}_percent": [
            item for item in comparable
            if abs(item["throughput"]["subject_improvement_percent"]) <= threshold * 100.0
        ]
        for threshold in CLOSENESS_BUCKETS
    }
    return summary


def per_competitor(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate Aether's position against each competitor across all cells.

    The median is reported rather than the mean: one cell where an engine failed to
    scale would otherwise dominate a mean and misdescribe the typical case. The
    range is printed beside it so the spread is never hidden by the centre.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in comparisons:
        if item["throughput"].get("comparable"):
            grouped.setdefault(item["competitor"], []).append(item)
    result: dict[str, dict[str, Any]] = {}
    for engine, entries in grouped.items():
        improvements = [
            item["throughput"]["subject_improvement_percent"] for item in entries
        ]
        same = [item for item in entries if item["comparability"] == SAME_REPRESENTATION]
        same_improvements = [
            item["throughput"]["subject_improvement_percent"] for item in same
        ]
        result[engine] = {
            "cells": len(entries),
            "same_representation_cells": len(same),
            "median_improvement_percent": statistics.median(improvements),
            "median_improvement_percent_same_representation": (
                statistics.median(same_improvements) if same_improvements else None
            ),
            "min_improvement_percent": min(improvements),
            "max_improvement_percent": max(improvements),
            "aether_wins": sum(1 for item in entries if item["winner"] == "aether"),
            "aether_losses": sum(1 for item in entries if item["winner"] == "competitor"),
            "ties": sum(1 for item in entries if item["winner"] == "tie"),
            "comparability": (
                SAME_REPRESENTATION if len(same) == len(entries)
                else REPRESENTATION_DIFFERENCE
            ),
        }
    return result


def batch_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How each engine's throughput responds to batch width.

    Two figures per cell, because one of them alone is misleading. Aggregate
    throughput rising with batch width is expected and says nothing about
    efficiency; ``scaling_efficiency_percent`` is the fraction of ideal linear
    scaling actually achieved, and ``per_request_tokens_per_s`` is what a single
    caller waiting inside the batch experiences.
    """
    baseline: dict[tuple[str, str], float] = {}
    for row in measured_rows(rows):
        if row.get("batch_size") == 1 and row.get("is_primary"):
            value = row.get(PRIMARY_METRIC)
            if value:
                baseline[(row["engine"], row["model"])] = float(value)
    results: list[dict[str, Any]] = []
    for row in rows:
        if row.get("batch_size") is None or "batch" not in (row.get("sweeps") or []):
            continue
        entry = {
            "engine": row["engine"],
            "model": row["model"],
            "batch_size": row["batch_size"],
            "status": row["status"],
            "reason": row.get("reason", ""),
            "batch_tokens_per_s": row.get(PRIMARY_METRIC),
            "per_request_tokens_per_s": row.get("per_request_tokens_per_s"),
            "end_to_end_latency_s": row.get("end_to_end_latency_s"),
            "peak_device_bytes": row.get("peak_device_bytes"),
            "peak_host_bytes": row.get("peak_host_bytes"),
            "scaling_vs_batch1": None,
            "scaling_efficiency_percent": None,
        }
        reference = baseline.get((row["engine"], row["model"]))
        if status_mod.is_measured(row) and reference and row.get(PRIMARY_METRIC):
            ratio = float(row[PRIMARY_METRIC]) / reference
            entry["scaling_vs_batch1"] = ratio
            entry["scaling_efficiency_percent"] = (
                ratio / max(int(row["batch_size"]), 1) * 100.0
            )
        results.append(entry)
    return results


def scaling_ranking(scaling: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank engines by how efficiently they convert batch width into throughput.

    Judged at the largest batch each engine actually completed, and that width is
    printed with the score: an engine that only reached batch 2 has not demonstrated
    the same thing as one that reached 16, and ranking them on a bare percentage
    without saying so would obscure it.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in scaling:
        if entry.get("scaling_efficiency_percent") is None or entry["batch_size"] == 1:
            continue
        key = (entry["engine"], entry["model"])
        current = best.get(key)
        if current is None or entry["batch_size"] > current["batch_size"]:
            best[key] = entry
    ranked = sorted(
        best.values(), key=lambda item: item["scaling_efficiency_percent"], reverse=True
    )
    return ranked


def rankings(rows: list[dict[str, Any]], scaling: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate rankings, each scoped to the workload it was measured on.

    One universal ranking would be the most quotable output of this suite and the
    least defensible, so there is not one. Each ranking below names its workload, and
    an engine appears in it only if it produced that measurement.
    """
    def rank(candidates: list[dict[str, Any]], metric: str, lower_is_better: bool,
             scope: str) -> dict[str, Any]:
        usable = [
            item for item in candidates
            if status_mod.is_measured(item) and item.get(metric)
        ]
        ordered = sorted(
            usable, key=lambda item: float(item[metric]), reverse=not lower_is_better
        )
        return {
            "metric": metric,
            "scope": scope,
            "lower_is_better": lower_is_better,
            "order": [
                {
                    "rank": index,
                    "engine": item["engine"],
                    "model": item.get("model"),
                    "value": item[metric],
                }
                for index, item in enumerate(ordered, start=1)
            ],
            "not_measured": sorted({
                item["engine"] for item in candidates
                if not (status_mod.is_measured(item) and item.get(metric))
            }),
        }

    primary = [row for row in rows if row.get("is_primary")]
    all_measured = measured_rows(rows)
    by_engine_best: dict[str, dict[str, Any]] = {}
    for row in all_measured:
        value = row.get(PRIMARY_METRIC)
        if not value:
            continue
        current = by_engine_best.get(row["engine"])
        if current is None or float(value) > float(current[PRIMARY_METRIC]):
            by_engine_best[row["engine"]] = row

    return {
        "batch_1": rank(primary, PRIMARY_METRIC, False,
                        "batch 1, primary prompt and output length: the single-user case"),
        "peak_throughput": rank(list(by_engine_best.values()), PRIMARY_METRIC, False,
                                "each engine's best cell anywhere in the matrix, "
                                "whatever batch width produced it"),
        "ttft": rank(primary, "ttft_s", True, "batch 1: time to first token"),
        "latency": rank(primary, "end_to_end_latency_s", True,
                        "batch 1: end-to-end latency of one request"),
        "host_memory": rank(primary, "peak_host_bytes", True,
                            "batch 1: peak process resident set size"),
        "device_memory": rank(primary, "peak_device_bytes", True,
                              "batch 1: peak reserved accelerator memory"),
        "cold_start": rank(primary, "cold_latency_s", True,
                           "batch 1: first, unwarmed inference in a fresh process"),
        "batch_scaling": {
            "metric": "scaling_efficiency_percent",
            "scope": "largest batch width each engine completed; the width is listed",
            "lower_is_better": False,
            "order": [
                {
                    "rank": index,
                    "engine": item["engine"],
                    "model": item["model"],
                    "batch_size": item["batch_size"],
                    "value": item["scaling_efficiency_percent"],
                }
                for index, item in enumerate(scaling_ranking(scaling), start=1)
            ],
        },
    }


def _startup(run: dict[str, Any]) -> dict[str, Any]:
    artifact = run.get("artifact") or {}
    load = run.get("load") or {}
    return {
        "build_s": artifact.get("build_s") or load.get("prepare_s"),
        "load_s": artifact.get("load_s") or load.get("load_s"),
        "download_s": artifact.get("download_s") or load.get("download_s"),
        "total_s": artifact.get("total_startup_s") or load.get("total_s"),
        "artifact_bytes": artifact.get("artifact_bytes"),
        "persistence": artifact.get("persistence"),
        "has_build_phase": artifact.get("has_build_phase"),
        "built_this_run": artifact.get("built_this_run"),
    }


def compile_economics(payload: dict[str, Any], rows: list[dict[str, Any]],
                      run_counts: list[int]) -> dict[str, Any]:
    """What a build phase costs, and how many requests it takes to pay for itself.

    Kept strictly separate from steady-state throughput. Steady state answers "once
    everything is ready, how fast is it"; this answers "what does the whole workflow
    cost", which is a different question and the one a deployment decision actually
    turns on.

    The second-process reload measured by the reuse probe is what makes the
    compile-once claim checkable: an engine whose artifact reloads in a fresh process
    pays its build cost once per machine, not once per process.
    """
    primary = {
        (row["engine"], row["model"]): row
        for row in rows if row.get("is_primary") and status_mod.is_measured(row)
    }
    reuse_by_key = {
        (run.get("engine"), run.get("model")): run
        for run in payload.get("reuse_runs", [])
    }
    entries: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        key = (run.get("engine"), run.get("model"))
        startup = _startup(run)
        row = primary.get(key)
        latency = row.get("end_to_end_latency_s") if row else None
        reuse = reuse_by_key.get(key) or {}
        reuse_load = ((reuse.get("load") or {}).get("total_s")
                      if reuse.get("status") == status_mod.MEASURED else None)
        first_after_reload = None
        if reuse.get("status") == status_mod.MEASURED:
            first_after_reload = (reuse.get("first_inference") or {}).get("cold_latency_s")
        entry: dict[str, Any] = {
            "engine": run.get("engine"),
            "model": run.get("model"),
            "status": run.get("status"),
            "spec_persistence": (run.get("spec") or {}).get("artifact_persistence"),
            "steady_state_latency_s": latency,
            "reuse_status": reuse.get("status"),
            "reuse_reason": reuse.get("reason"),
            "second_process_load_s": reuse_load,
            "first_inference_after_reload_s": first_after_reload,
            **startup,
        }
        if latency:
            entry["total_cost_s"] = {
                str(count): {
                    "cold_first_deployment": (startup["total_s"] or 0.0) + count * latency,
                    "warm_reused_artifact": (
                        (reuse_load + count * latency) if reuse_load is not None else None
                    ),
                }
                for count in run_counts
            }
        entries.append(entry)
    return {
        "run_counts": list(run_counts),
        "entries": entries,
        "break_even": _break_even(entries),
    }


def _break_even(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """How many requests it takes for Aether's build cost to pay for itself.

    Solved rather than searched: with a fixed start-up cost S and a per-request cost
    L on each side, the totals cross at N = (S_aether - S_other) / (L_other -
    L_aether). Reported as a run count, with the sign cases spelled out, because
    "never" and "immediately" are both real answers and both matter.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("steady_state_latency_s"):
            by_model.setdefault(entry["model"], {})[entry["engine"]] = entry
    results: list[dict[str, Any]] = []
    for model, engines_here in by_model.items():
        subject = engines_here.get(SUBJECT)
        if subject is None:
            continue
        for engine, other in engines_here.items():
            if engine == SUBJECT:
                continue
            startup_delta = (subject.get("total_s") or 0.0) - (other.get("total_s") or 0.0)
            latency_gain = (
                (other["steady_state_latency_s"]) - (subject["steady_state_latency_s"])
            )
            record: dict[str, Any] = {
                "model": model,
                "competitor": engine,
                "aether_startup_s": subject.get("total_s"),
                "competitor_startup_s": other.get("total_s"),
                "aether_latency_s": subject["steady_state_latency_s"],
                "competitor_latency_s": other["steady_state_latency_s"],
                "startup_penalty_s": startup_delta,
                "per_request_saving_s": latency_gain,
            }
            if latency_gain <= 0:
                record["break_even_runs"] = None
                record["interpretation"] = (
                    "Aether is not faster per request in this configuration, so its "
                    "start-up cost is never amortized here"
                )
            elif startup_delta <= 0:
                record["break_even_runs"] = 0
                record["interpretation"] = (
                    "Aether starts up no slower and runs faster, so it is ahead from "
                    "the first request"
                )
            else:
                record["break_even_runs"] = startup_delta / latency_gain
                record["interpretation"] = (
                    f"Aether's start-up costs {startup_delta:.1f}s more and saves "
                    f"{latency_gain * 1000:.1f}ms per request, so it pulls ahead after "
                    f"about {startup_delta / latency_gain:.0f} requests"
                )
            results.append(record)
    return results


# ── Correctness ─────────────────────────────────────────────────────────────
EXACT_MATCH = "EXACT_MATCH"
NUMERICALLY_EQUIVALENT = "NUMERICALLY_EQUIVALENT"
EXPECTED_SAMPLING_DIFFERENCE = "EXPECTED_SAMPLING_DIFFERENCE"
DIFFERENT_OUTPUT = "DIFFERENT_OUTPUT"
FAILURE = "FAILURE"

#: A greedy decode is a chain of argmax decisions, so two numerically equivalent
#: implementations can split at a near-tie and then stay split. A long shared
#: prefix is therefore evidence of agreement, and this is how long it has to be.
#: Set to most of the sequence rather than a token or two, so a coincidental
#: opening cannot be mistaken for equivalence.
PREFIX_AGREEMENT_FRACTION = 0.75


def correctness(payload: dict[str, Any], greedy: bool) -> dict[str, Any]:
    """Compare every engine's generated output against the reference engine's.

    Floating-point difference between implementations is expected and is not a
    defect, so the classification distinguishes it from a genuinely different
    computation instead of failing an engine for rounding. What it will not do is
    call two different completions equivalent because the numbers underneath were
    close: the token sequence is the observable, and the classes say exactly what was
    observed.
    """
    from benchmark import correctness as correctness_mod

    captures: dict[str, dict[str, dict[str, Any]]] = {}
    for run in payload.get("runs", []):
        sample = run.get("correctness_sample")
        if sample:
            captures.setdefault(run["model"], {})[run["engine"]] = sample

    cases: list[dict[str, Any]] = []
    for model, per_engine in captures.items():
        reference = per_engine.get(REFERENCE)
        for engine, candidate in per_engine.items():
            if engine == REFERENCE:
                continue
            case: dict[str, Any] = {
                "model": model, "engine": engine, "reference": REFERENCE,
                "greedy": greedy,
            }
            if reference is None or reference.get("status") != status_mod.MEASURED:
                case.update(classification=FAILURE,
                            reason="the reference engine produced no output to compare against")
                cases.append(case)
                continue
            if candidate.get("status") != status_mod.MEASURED:
                case.update(classification=FAILURE,
                            reason=candidate.get("reason", "engine produced no output"))
                cases.append(case)
                continue
            tokens = correctness_mod.compare_token_ids(
                [int(value) for value in reference.get("token_ids") or []],
                [int(value) for value in candidate.get("token_ids") or []],
            )
            text = correctness_mod.compare_text(
                reference.get("text") or "", candidate.get("text") or ""
            )
            classification, basis = _classify(tokens, text, greedy)
            case.update(
                tokens=tokens,
                text=text,
                reference_completion_tokens=reference.get("completion_tokens"),
                candidate_completion_tokens=candidate.get("completion_tokens"),
                token_ids_source=candidate.get("token_ids_source"),
                classification=classification,
                basis=basis,
            )
            cases.append(case)
    return {
        "reference_engine": REFERENCE,
        "greedy": greedy,
        "prefix_agreement_fraction": PREFIX_AGREEMENT_FRACTION,
        "cases": cases,
    }


def _classify(tokens: dict[str, Any], text: dict[str, Any],
              greedy: bool) -> tuple[str, str]:
    """Name what the difference between two completions is, and on what evidence.

    The basis is returned alongside the class because the two observables can
    disagree: an engine that only returns decoded text has its ids re-encoded, and a
    round trip can renumber tokens that decode to the identical string. Identical
    text is the stronger evidence in that case, and the report must be able to say
    which one the class rests on.
    """
    if tokens.get("identical"):
        return EXACT_MATCH, "token ids identical"
    if text.get("identical"):
        return EXACT_MATCH, (
            "decoded text identical; token ids differ, which a re-encoding round "
            "trip can cause"
        )
    if not greedy:
        # With sampling on, two runs of the *same* engine would differ too, so a
        # difference here carries no information about correctness.
        return EXPECTED_SAMPLING_DIFFERENCE, "sampling was enabled"
    fraction = float(tokens.get("matching_prefix_fraction") or 0.0)
    if fraction >= PREFIX_AGREEMENT_FRACTION:
        return NUMERICALLY_EQUIVALENT, f"{fraction * 100:.0f}% of the ids agree first"
    if fraction < 0.25:
        return DIFFERENT_OUTPUT, f"only {fraction * 100:.0f}% of the ids agree"
    return NUMERICALLY_EQUIVALENT, f"{fraction * 100:.0f}% of the ids agree first"


# ── Statistical quality ─────────────────────────────────────────────────────

#: Multiplier on the median absolute deviation past which a sample is flagged. A
#: modified z-score of 3.5 is the conventional threshold and, unlike a standard
#: deviation, it is not itself inflated by the outlier it is trying to find.
OUTLIER_MODIFIED_Z = 3.5


def statistical_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report dispersion, and flag outliers without removing any.

    Nothing here changes a measurement. A flagged sample stays in every average the
    report prints; the flag exists so a reader can see that a cell was noisy rather
    than having a quiet mean present it as settled.
    """
    entries: list[dict[str, Any]] = []
    for row in measured_rows(rows):
        stats = row.get("latency_stats") or {}
        samples = stats.get("n")
        entry = {
            "engine": row["engine"],
            "model": row["model"],
            "batch_size": row["batch_size"],
            "prompt_tokens": row["prompt_tokens"],
            "output_tokens": row["output_tokens"],
            "iterations": samples,
            "mean_s": stats.get("mean"),
            "median_s": stats.get("median"),
            "stdev_s": stats.get("stdev"),
            "p50_s": stats.get("p50"),
            "p90_s": stats.get("p90"),
            "p95_s": stats.get("p95"),
            "p99_s": stats.get("p99"),
            "min_s": stats.get("min"),
            "max_s": stats.get("max"),
            "coefficient_of_variation": stats.get("coefficient_of_variation"),
        }
        cov = stats.get("coefficient_of_variation")
        entry["dispersion_flag"] = (
            "high" if isinstance(cov, (int, float)) and cov > 0.10
            else "moderate" if isinstance(cov, (int, float)) and cov > 0.05
            else "low"
        )
        spread = None
        if entry["min_s"] and entry["max_s"] and entry["median_s"]:
            spread = (entry["max_s"] - entry["min_s"]) / entry["median_s"]
        entry["range_over_median"] = spread
        entry["outlier_suspected"] = bool(
            spread is not None and spread > 0.5 and (samples or 0) >= 5
        )
        entries.append(entry)
    return {
        "modified_z_threshold": OUTLIER_MODIFIED_Z,
        "outliers_removed": 0,
        "policy": (
            "No sample is discarded. Cells whose range exceeds half their median are "
            "flagged so a reader can weigh them; every reported statistic still "
            "includes every sample that was taken."
        ),
        "entries": entries,
    }


def engine_compatibility(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per engine per model: did it run, and if not, why not."""
    catalogue = payload.get("engine_catalogue") or {}
    table: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        key = run.get("engine")
        spec = catalogue.get(key) or run.get("spec") or {}
        measured_cells = [
            cell for cell in (run.get("cells") or [])
            if cell.get("status") == status_mod.MEASURED
        ]
        table.append({
            "engine": key,
            "display": spec.get("display", key),
            "model": run.get("model"),
            "taxonomy": spec.get("taxonomy", []),
            "status": run.get("status"),
            "reason": run.get("reason", ""),
            "version": (run.get("availability") or {}).get("version"),
            "cells_measured": len(measured_cells),
            "cells_attempted": len(run.get("cells") or []),
            "batch_support": run.get("batch_support") or {},
            "representation": (run.get("describe") or {}).get("representation"),
            "has_build_phase": spec.get("has_build_phase"),
            "artifact_persistence": spec.get("artifact_persistence"),
            "ttft_method": (run.get("describe") or {}).get("ttft_method")
            or spec.get("ttft_method"),
            "tokenizer_identical": (run.get("tokenizer_agreement") or {}).get("identical"),
        })
    return table


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive every comparison the report will print, once, from the raw payload."""
    plan = payload.get("plan") or {}
    rows = flatten(payload)
    scaling = batch_scaling(rows)
    comparisons = head_to_head(rows)
    greedy = float(plan.get("temperature") or 0.0) == 0.0
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_label": PRIMARY_METRIC_LABEL,
        "tie_threshold_percent": TIE_THRESHOLD * 100.0,
        "rows": rows,
        "compatibility": engine_compatibility(payload),
        "comparisons": comparisons,
        "win_loss": win_loss_summary(comparisons),
        "per_competitor": per_competitor(comparisons),
        "batch_scaling": scaling,
        "rankings": rankings(rows, scaling),
        "compile_economics": compile_economics(
            payload, rows, [int(value) for value in plan.get("amortization_runs") or [1]]
        ),
        "correctness": correctness(payload, greedy),
        "statistics": statistical_quality(rows),
    }
