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
                "weight_storage_bits": describe.get("weight_storage_bits"),
                "weight_storage_format": describe.get("weight_storage_format"),
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
                "weight_storage_bits": describe.get("weight_storage_bits"),
                "weight_storage_format": describe.get("weight_storage_format"),
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
    """Whether a speed comparison between these two rows is a speed comparison.

    Judged on what each engine reported it actually loaded, not on what the plan
    asked for, because an engine can export or quantize during its build step.

    Two things are separated here, and conflating them is the mistake this function
    exists to avoid:

    * **compute precision** - the dtype the arithmetic runs in. If it differs, the two
      engines are not doing the same work, and the comparison is labelled.
    * **weight storage** - the container the values sit in. Every engine here derives
      from the same published bf16 checkpoint, and each renders it into its own 16-bit
      form: the framework engines cast to fp16 tensors, Aether's artifact keeps the
      bf16 values, an ONNX export widens to fp32. At equal compute precision and equal
      storage width that is one rounding step in different directions, which is a
      *disclosed* detail rather than a different experiment. Below 16 bits it is a
      different experiment, because fewer bits per weight is less memory traffic and
      decode is memory bound.

    The returned note carries the storage detail even when the verdict is
    SAME_REPRESENTATION, so it reaches the report either way.
    """
    subject_repr = str(subject.get("representation") or "")
    other_repr = str(other.get("representation") or "")

    if subject.get("quantized") or other.get("quantized"):
        quantized = "both engines" if (
            subject.get("quantized") and other.get("quantized")
        ) else ("this engine" if subject.get("quantized") else "the competing engine")
        return REPRESENTATION_DIFFERENCE, (
            f"{quantized} executes a quantized representation ({subject_repr} against "
            f"{other_repr}), so the comparison includes a weight-format difference as "
            "well as an execution difference"
        )
    if subject.get("precision") != other.get("precision"):
        return REPRESENTATION_DIFFERENCE, (
            f"compute precisions differ: {subject.get('precision')} against "
            f"{other.get('precision')}, so the two engines are not performing the "
            "same arithmetic"
        )

    subject_bits = subject.get("weight_storage_bits")
    other_bits = other.get("weight_storage_bits")
    if subject_bits and other_bits and int(subject_bits) != int(other_bits):
        return REPRESENTATION_DIFFERENCE, (
            f"weights are stored at different widths: {subject_bits}-bit "
            f"({subject.get('weight_storage_format')}) against {other_bits}-bit "
            f"({other.get('weight_storage_format')})"
        )
    if any(bits and int(bits) < 16 for bits in (subject_bits, other_bits)):
        return REPRESENTATION_DIFFERENCE, (
            "at least one engine stores weights below 16 bits, which changes memory "
            "traffic and therefore decode speed independently of execution"
        )

    subject_format = str(subject.get("weight_storage_format") or "")
    other_format = str(other.get("weight_storage_format") or "")
    if subject_format and other_format and subject_format != other_format:
        return SAME_REPRESENTATION, (
            f"same compute precision and the same storage width, but different 16-bit "
            f"containers ({subject_format} against {other_format}). Both are one "
            "rounding step from the published checkpoint, in different directions."
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
    """Name which side won, or call it a tie, on a single stated threshold.

    ``subject`` and ``competitor`` are positions in a comparison, not engines. The
    same function decides every pairing in the matrix, in both directions, so no
    engine can be favoured by the arithmetic that scores it.
    """
    if improvement_percent is None:
        return "no comparison"
    if improvement_percent > TIE_THRESHOLD * 100.0:
        return "subject"
    if improvement_percent < -TIE_THRESHOLD * 100.0:
        return "competitor"
    return "tie"


def _compare_pair(subject: dict[str, Any], other: dict[str, Any], key: Key,
                  subject_engine: str, other_engine: str) -> dict[str, Any]:
    """Every metric of one engine against another, in one cell."""
    label, note = comparability(subject, other)
    throughput = compare(subject.get(PRIMARY_METRIC), other.get(PRIMARY_METRIC))
    return {
        "model": key.model,
        "batch_size": key.batch_size,
        "prompt_tokens": key.prompt_tokens,
        "output_tokens": key.output_tokens,
        "is_primary": bool(subject.get("is_primary")),
        "subject": subject_engine,
        "competitor": other_engine,
        "comparability": label,
        "comparability_note": note,
        "throughput": throughput,
        "decode": compare(subject.get("decode_tokens_per_s"),
                          other.get("decode_tokens_per_s")),
        "latency": compare(subject.get("end_to_end_latency_s"),
                           other.get("end_to_end_latency_s"), lower_is_better=True),
        "ttft": compare(subject.get("ttft_s"), other.get("ttft_s"),
                        lower_is_better=True),
        "host_memory": compare(subject.get("peak_host_bytes"),
                               other.get("peak_host_bytes"), lower_is_better=True),
        "device_memory": compare(subject.get("peak_device_bytes"),
                                 other.get("peak_device_bytes"), lower_is_better=True),
        "winner": verdict(throughput.get("subject_improvement_percent")),
    }


def pairwise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ordered pair of engines, in every cell both of them measured.

    Ordered, so each pairing appears from both sides and neither engine occupies a
    privileged position. Generated by iterating cells rather than claims: a cell where
    an engine lost is produced by the same loop that produces one where it won, and
    nothing anywhere filters on the sign of the result.
    """
    table = index_by_cell(rows)
    comparisons: list[dict[str, Any]] = []
    for key in sorted(table, key=lambda item: (item.model, item.batch_size,
                                               item.prompt_tokens, item.output_tokens)):
        engines_here = table[key]
        for subject_engine, subject in sorted(engines_here.items()):
            for other_engine, other in sorted(engines_here.items()):
                if subject_engine == other_engine:
                    continue
                comparisons.append(
                    _compare_pair(subject, other, key, subject_engine, other_engine)
                )
    return comparisons


def head_to_head(rows: list[dict[str, Any]],
                 engine: str = SUBJECT) -> list[dict[str, Any]]:
    """One engine against every other, in every cell both of them measured.

    A view of :func:`pairwise` filtered to one left-hand engine. Which engine that is
    is an argument, not a property of the code, so the same tables can be produced for
    any of them.
    """
    return [item for item in pairwise(rows) if item["subject"] == engine]


def win_loss_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Count and list where the left-hand engine of each comparison wins and loses.

    Position-neutral: ``wins`` are the cells where the comparison's subject was
    faster, whichever engine that happens to be. Same-representation comparisons are
    counted separately from ones that cross a weight-format boundary, because only the
    first supports a statement about execution. Both are listed; neither is hidden.
    """
    def bucket(entries: list[dict[str, Any]]) -> dict[str, Any]:
        wins = [item for item in entries if item["winner"] == "subject"]
        losses = [item for item in entries if item["winner"] == "competitor"]
        ties = [item for item in entries if item["winner"] == "tie"]
        return {
            "compared": len(entries),
            "wins": len(wins),
            "losses": len(losses),
            "ties": len(ties),
            "won": wins,
            "lost": losses,
            "tied": ties,
            # Retained under the older names so a consumer of the JSON written by an
            # earlier run does not silently read a missing key as zero.
            "aether_wins": len(wins),
            "aether_losses": len(losses),
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
                "model": item["model"], "subject": item["subject"],
                "competitor": item["competitor"],
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
    """Aggregate one engine's position against each of its opponents.

    Expects comparisons that already share a single subject (what
    :func:`head_to_head` returns). The median is reported rather than the mean: one
    cell where an engine failed to scale would otherwise dominate a mean and
    misdescribe the typical case. The range is printed beside it so the spread is
    never hidden by the centre.
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
            "wins": sum(1 for item in entries if item["winner"] == "subject"),
            "losses": sum(1 for item in entries if item["winner"] == "competitor"),
            "ties": sum(1 for item in entries if item["winner"] == "tie"),
            "comparability": (
                SAME_REPRESENTATION if len(same) == len(entries)
                else REPRESENTATION_DIFFERENCE
            ),
        }
    return result


def standings(comparisons: list[dict[str, Any]],
              rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A league table: every engine's record against the whole field.

    Ordered by win rate over its own comparisons, so an engine that only ran the easy
    cells cannot climb by having fewer hard ones - the count of comparisons is printed
    beside the rate, and an engine that ran fewer of them is visibly not measured on
    the same amount of work.

    This is the neutral summary. There is no privileged engine in it, and it is the
    table the report leads with.
    """
    by_engine: dict[str, dict[str, Any]] = {}
    for item in comparisons:
        if not item["throughput"].get("comparable"):
            continue
        entry = by_engine.setdefault(item["subject"], {
            "engine": item["subject"], "compared": 0, "wins": 0, "losses": 0,
            "ties": 0, "improvements": [], "same_representation": 0,
        })
        entry["compared"] += 1
        entry["improvements"].append(item["throughput"]["subject_improvement_percent"])
        if item["comparability"] == SAME_REPRESENTATION:
            entry["same_representation"] += 1
        if item["winner"] == "subject":
            entry["wins"] += 1
        elif item["winner"] == "competitor":
            entry["losses"] += 1
        else:
            entry["ties"] += 1

    best_by_cell: dict[tuple[str, int, int, int], float] = {}
    for row in measured_rows(rows):
        value = row.get(PRIMARY_METRIC)
        if row.get("batch_size") is None or not value:
            continue
        key = (row["model"], int(row["batch_size"]), int(row["prompt_tokens"]),
               int(row["output_tokens"]))
        best_by_cell[key] = max(best_by_cell.get(key, 0.0), float(value))
    shares: dict[str, list[float]] = {}
    for row in measured_rows(rows):
        value = row.get(PRIMARY_METRIC)
        if row.get("batch_size") is None or not value:
            continue
        key = (row["model"], int(row["batch_size"]), int(row["prompt_tokens"]),
               int(row["output_tokens"]))
        best = best_by_cell.get(key)
        if best:
            shares.setdefault(row["engine"], []).append(float(value) / best * 100.0)

    table: list[dict[str, Any]] = []
    for engine, entry in by_engine.items():
        improvements = entry.pop("improvements")
        table.append({
            **entry,
            "win_rate_percent": entry["wins"] / entry["compared"] * 100.0,
            "median_improvement_percent": statistics.median(improvements),
            # Share of the fastest engine's throughput in each cell, median across
            # cells: a scale-free score that does not depend on which engine happens
            # to be the reference.
            "median_percent_of_best": (
                statistics.median(shares[engine]) if shares.get(engine) else None
            ),
            "cells_measured": len(shares.get(engine, [])),
        })
    table.sort(
        key=lambda item: (
            item["median_percent_of_best"] if item["median_percent_of_best"] is not None
            else -1.0,
            item["win_rate_percent"],
        ),
        reverse=True,
    )
    for position, entry in enumerate(table, start=1):
        entry["rank"] = position
    return table


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
    """How many requests it takes a build cost to pay for itself, for every pairing.

    Solved rather than searched: with a fixed start-up cost S and a per-request cost L
    on each side, the totals cross at N = (S_subject - S_other) / (L_other -
    L_subject). Reported as a run count, with the sign cases spelled out, because
    "never" and "immediately" are both real answers and both matter.

    Computed for every ordered pair, so the question can be asked of any engine that
    has a build phase rather than only of one.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("steady_state_latency_s"):
            by_model.setdefault(entry["model"], {})[entry["engine"]] = entry
    results: list[dict[str, Any]] = []
    for model, engines_here in by_model.items():
        for subject_engine, subject in sorted(engines_here.items()):
            for engine, other in sorted(engines_here.items()):
                if engine == subject_engine:
                    continue
                startup_delta = (
                    (subject.get("total_s") or 0.0) - (other.get("total_s") or 0.0)
                )
                latency_gain = (
                    other["steady_state_latency_s"] - subject["steady_state_latency_s"]
                )
                record: dict[str, Any] = {
                    "model": model,
                    "subject": subject_engine,
                    "competitor": engine,
                    # Whether the subject actually has a build phase decides whether
                    # this pairing answers the *compilation* question or merely the
                    # start-up one. Recorded so the report can ask the narrower one.
                    "subject_has_build_phase": bool(subject.get("has_build_phase")),
                    "competitor_has_build_phase": bool(other.get("has_build_phase")),
                    "subject_build_s": subject.get("build_s"),
                    "subject_startup_s": subject.get("total_s"),
                    "competitor_startup_s": other.get("total_s"),
                    "subject_latency_s": subject["steady_state_latency_s"],
                    "competitor_latency_s": other["steady_state_latency_s"],
                    "startup_penalty_s": startup_delta,
                    "per_request_saving_s": latency_gain,
                }
                if latency_gain <= 0:
                    record["break_even_runs"] = None
                    record["interpretation"] = (
                        f"{subject_engine} is not faster per request than {engine} "
                        "here, so no number of requests repays its start-up cost"
                    )
                elif startup_delta <= 0:
                    record["break_even_runs"] = 0
                    record["interpretation"] = (
                        f"{subject_engine} starts up no slower than {engine} and runs "
                        "faster, so it is ahead from the first request"
                    )
                else:
                    record["break_even_runs"] = startup_delta / latency_gain
                    record["interpretation"] = (
                        f"{subject_engine} costs {startup_delta:.1f}s more to start "
                        f"than {engine} and saves {latency_gain * 1000:.1f}ms per "
                        f"request, so it pulls ahead after about "
                        f"{startup_delta / latency_gain:.0f} requests"
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
    """Derive every comparison the report will print, once, from the raw payload.

    The pairwise matrix is the primary product and is computed for every ordered pair
    of engines. Per-engine views are slices of it, so no engine's numbers come from a
    different code path than any other's.
    """
    plan = payload.get("plan") or {}
    rows = flatten(payload)
    scaling = batch_scaling(rows)
    all_pairs = pairwise(rows)
    engines_measured = sorted({
        row["engine"] for row in measured_rows(rows) if row.get("engine")
    })
    greedy = float(plan.get("temperature") or 0.0) == 0.0
    per_engine = {
        engine: {
            "comparisons": [item for item in all_pairs if item["subject"] == engine],
            "win_loss": win_loss_summary(
                [item for item in all_pairs if item["subject"] == engine]
            ),
            "per_competitor": per_competitor(
                [item for item in all_pairs if item["subject"] == engine]
            ),
        }
        for engine in engines_measured
    }
    focus = plan.get("focus") or None
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_label": PRIMARY_METRIC_LABEL,
        "tie_threshold_percent": TIE_THRESHOLD * 100.0,
        "rows": rows,
        "engines_measured": engines_measured,
        "compatibility": engine_compatibility(payload),
        "pairwise": all_pairs,
        "standings": standings(all_pairs, rows),
        "per_engine": per_engine,
        "focus": focus,
        # The single-engine views below are slices of `pairwise`, kept as top-level
        # keys because the report and the CSV writers read them directly. `focus`
        # selects which engine they describe; with no focus they describe the engine
        # this repository is developing, which is a choice of subject, not of scoring.
        "comparisons": (
            per_engine.get(focus or SUBJECT, {}).get("comparisons", [])
        ),
        "win_loss": per_engine.get(focus or SUBJECT, {}).get(
            "win_loss", win_loss_summary([])
        ),
        "per_competitor": per_engine.get(focus or SUBJECT, {}).get("per_competitor", {}),
        "subject": focus or SUBJECT,
        "batch_scaling": scaling,
        "rankings": rankings(rows, scaling),
        "compile_economics": compile_economics(
            payload, rows, [int(value) for value in plan.get("amortization_runs") or [1]]
        ),
        "correctness": correctness(payload, greedy),
        "statistics": statistical_quality(rows),
    }
