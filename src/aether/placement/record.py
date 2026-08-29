"""The decision record — the planner's actual deliverable.

A planner that cannot be audited will be overridden by an environment variable
within a month.  Every number rendered here is one the planner already computed, so
printing it costs nothing and makes the whole design falsifiable in the field: an
operator who disagrees can point at the line that is wrong.

The layout deliberately puts the *derivation* before the verdict.  A reader who only
wants the answer reads the last four lines; a reader debugging an OOM reads the
feasibility block and sees exactly which term consumed the device.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from aether.placement.planner import PlanEvaluation, PlannerDecision

__all__ = ["render_record", "render_compact"]

_GIB = 1024 ** 3
_MAX_COLUMNS = 3
"""Plans shown side by side. Three is what fits an 80-column terminal legibly."""


def _fmt_gib(value: float) -> str:
    return f"{value / _GIB:8.2f}"


def _fmt_ms(value: float) -> str:
    return f"{value * 1e3:8.2f}"


def _binding_budget(evaluation: "PlanEvaluation") -> Any:
    """The device that limits this plan - the only one whose numbers decide anything."""
    return min(evaluation.budgets, key=lambda budget: budget.tokens_max)


def _columns(decision: "PlannerDecision") -> "list[PlanEvaluation]":
    """The selected plan first, then the strongest alternatives it beat."""
    selected = decision.selected
    others = [
        candidate for candidate in decision.candidates
        if candidate is not selected
    ]
    # Feasible alternatives are more informative than infeasible ones, except when
    # the infeasible ones explain why hardware was added — so keep one of each.
    feasible = sorted(
        (candidate for candidate in others if candidate.feasible),
        key=lambda candidate: candidate.objective(decision.workload),
    )
    infeasible = sorted(
        (candidate for candidate in others if not candidate.feasible),
        key=lambda candidate: -candidate.tokens_max,
    )
    chosen = [selected] + feasible[: _MAX_COLUMNS - 1]
    if infeasible and len(chosen) <= _MAX_COLUMNS:
        chosen.append(infeasible[0])
    return chosen[: _MAX_COLUMNS + 1]


def _row(label: str, values: "list[str]", width: int = 24) -> str:
    return label.ljust(width) + "".join(value.rjust(14) for value in values)


def render_record(decision: "PlannerDecision") -> str:
    """Render the full decision record as plain text."""
    profile = decision.profile
    census = decision.census
    workload = decision.workload
    selected = decision.selected
    columns = _columns(decision)
    lines: list[str] = []

    stamp = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
    header = "AETHER EXECUTION PLAN"
    trailer = f"{profile.model_id} | {stamp}"
    lines.append(header + trailer.rjust(max(1, 78 - len(header))))
    lines.append("")

    # ── inputs ────────────────────────────────────────────────────────────────
    accelerators = census.accelerators or census.devices
    kinds: dict[str, int] = {}
    sizes: dict[str, float] = {}
    for gpu in accelerators:
        kinds[gpu.name] = kinds.get(gpu.name, 0) + 1
        sizes[gpu.name] = gpu.total_bytes / _GIB
    fleet = ", ".join(
        f"{count} x {name} {sizes[name]:.1f} GiB" for name, count in kinds.items()
    )
    lead = accelerators[0]
    fabrics = len(census.fabric_groups())
    link_kinds = sorted({link.kind for link in census.links}) or ["none"]
    lines.append(f"hardware   {fleet}")
    lines.append(
        f"           {'/'.join(link_kinds)} | {fabrics} fabric class"
        f"{'es' if fabrics != 1 else ''} | "
        f"bw {lead.effective_bandwidth_bps / 1e9:.0f} GB/s | "
        f"flops {lead.effective_flops / 1e12:.1f} TF"
    )
    measured = ", ".join(lead.measured) or "priors only"
    binding = _binding_budget(selected)
    lines.append(
        f"           measured: {measured} | "
        f"calibrated: {'yes' if decision.calibrated else 'no'} | "
        f"fragmentation x{binding.fragmentation:.2f}"
    )
    lines.append(
        f"model      {profile.params / 1e9:.2f} B params | "
        f"{profile.weight_bytes / _GIB:.2f} GiB at {profile.weight_dtype_bytes:.0f} B/param | "
        f"{profile.layers} layers | h {profile.hidden_size}"
    )
    lines.append(
        f"           {profile.num_kv_heads} KV heads | "
        f"{profile.kv_bytes_per_token / 1024:.0f} KiB KV/token | "
        f"{profile.ops_per_token} host ops/token | "
        f"{'exact' if profile.exact_weights else 'declared'} weights"
    )
    lines.append(
        f"workload   floor  b={workload.batch_floor} ctx={workload.context_floor} "
        f"gen={workload.generate_floor}   "
        f"target  b={workload.batch_target} ctx={workload.context_target} "
        f"gen={workload.generate_target}"
    )
    lines.append(f"intent     {workload.intent.value}")
    lines.append("")

    # ── feasibility ───────────────────────────────────────────────────────────
    labels = [evaluation.label[:13] for evaluation in columns]
    budgets = [_binding_budget(evaluation) for evaluation in columns]
    lines.append(_row("FEASIBILITY  binding device", labels))
    lines.append(_row("  C_safe           GiB", [_fmt_gib(b.safe_capacity_bytes) for b in budgets]))
    lines.append(_row("  static S         GiB", [_fmt_gib(b.static_bytes) for b in budgets]))
    lines.append(_row("  transient T      GiB", [_fmt_gib(b.transient_bytes) for b in budgets]))
    lines.append(_row("  margin z*sigma    GiB", [_fmt_gib(b.transient_margin_bytes) for b in budgets]))
    lines.append(_row("  KV budget K      GiB", [_fmt_gib(b.kv_budget_bytes) for b in budgets]))
    lines.append(_row("  KV per token     KiB", [f"{b.kv_bytes_per_token / 1024:8.1f}" for b in budgets]))
    lines.append(_row("  tokens_max", [f"{e.tokens_max:>8,}" for e in columns]))
    lines.append(_row("  verdict", [
        "feasible" if evaluation.feasible else "INFEASIBLE" for evaluation in columns
    ]))
    for evaluation in columns:
        if not evaluation.feasible:
            lines.append(f"           {evaluation.label}: {evaluation.reason}")
    lines.append("")

    # ── performance ───────────────────────────────────────────────────────────
    lines.append(_row("PERFORMANCE  decode/token", labels))
    lines.append(_row("  compute roof      ms", [
        _fmt_ms(max(s.roofs.compute_s for s in e.stage_costs)) for e in columns
    ]))
    lines.append(_row("  bandwidth roof    ms", [
        _fmt_ms(sum(s.roofs.bandwidth_s for s in e.stage_costs)) for e in columns
    ]))
    lines.append(_row("  dispatch roof     ms", [
        _fmt_ms(sum(s.roofs.dispatch_s for s in e.stage_costs)) for e in columns
    ]))
    lines.append(_row("  communication     ms", [
        _fmt_ms(sum(s.comm_s for s in e.stage_costs)) for e in columns
    ]))
    lines.append(_row("  predicted TPOT    ms", [_fmt_ms(e.decode_seconds) for e in columns]))
    lines.append(_row("  prefill total     ms", [_fmt_ms(e.prefill_seconds) for e in columns]))
    lines.append(_row("  blended/token     ms", [_fmt_ms(e.blended_token_seconds) for e in columns]))
    lines.append(_row("  +/- sigma         ms", [
        _fmt_ms(e.sigma_objective(workload)) for e in columns
    ]))
    lines.append(_row("  binding roof", [e.binding_roof for e in columns]))
    lines.append("")

    # ── decision ──────────────────────────────────────────────────────────────
    placement = " | ".join(
        f"[{','.join(stage.devices)}] layers {stage.layer_start}-{stage.layer_end - 1}"
        + (f" TP={stage.tp_degree}" if stage.tp_degree > 1 else "")
        + (
            " split " + "/".join(f"{value * 100:.0f}%" for value in stage.shard_fractions)
            if stage.tp_degree > 1 else ""
        )
        for stage in selected.plan.stages
    )
    lines.append(f"DECISION   {selected.label} - {selected.plan.kind.value}")
    lines.append(f"           {placement}")
    lines.append(_wrap("REASON     ", decision.reason))

    headroom = selected.headroom(workload)
    ceiling = selected.batch_ceiling(workload)
    lines.append(
        f"HEADROOM   tokens_max {selected.tokens_max:,} vs target "
        f"{workload.target_kv_tokens:,} -> {headroom:.2f}x headroom"
    )
    lines.append(
        f"           batch may reach {ceiling} at ctx {workload.context_target} "
        f"before a replan is required"
    )
    if selected.plan.max_tp_degree > 1:
        lines.append(
            f"           TP prefill comm/compute ratio {selected.comm_ratio:.3f} "
            f"(<<1 means the fabric can carry it)"
        )

    lines.append("LADDER")
    for rung in decision.ladder:
        lines.append(f"           {rung}")

    if decision.flags:
        lines.append("FLAGGED")
        for flag in decision.flags:
            lines.append(_wrap("           ", flag, indent=11))
    lines.append("")
    lines.append(
        f"           {len(decision.candidates)} plans evaluated, "
        f"{len(decision.feasible_candidates)} feasible, in "
        f"{decision.planning_seconds * 1e3:.2f} ms"
    )
    return "\n".join(lines)


def _wrap(prefix: str, text: str, *, indent: int | None = None, width: int = 78) -> str:
    """Wrap prose to the record's width, keeping the label column intact."""
    import textwrap

    pad = " " * (indent if indent is not None else len(prefix))
    wrapped = textwrap.wrap(text, width=width - len(prefix)) or [""]
    return "\n".join(
        (prefix if index == 0 else pad) + line for index, line in enumerate(wrapped)
    )


def render_compact(decision: "PlannerDecision") -> str:
    """One line, for a log: the decision plus the number that justifies it."""
    selected = decision.selected
    return (
        f"{selected.label}: {selected.tokens_max:,} KV tokens, "
        f"{selected.blended_token_seconds * 1e3:.2f} ms/token "
        f"({selected.binding_roof}-bound on {selected.binding_device}), "
        f"{selected.headroom(decision.workload):.2f}x headroom"
    )