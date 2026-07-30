"""Agentic workflow optimizer.

The PRD describes agentic sessions as a first-class runtime workload: tool calls,
long-lived context, cascade routing, and cross-session KV reuse. This module
implements deterministic planning primitives for those workloads. It does not
invoke external tools; it compiles observed call traces into cacheable workflow
plans that runtimes can execute and monitor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool invocation in an agent trace."""

    name: str
    arguments_schema: dict[str, str] = field(default_factory=dict)
    reads_context: bool = True
    writes_context: bool = False
    average_latency_ms: float = 0.0
    success_rate: float = 1.0

    def signature(self) -> str:
        """Return a stable signature used for meta-tool mining."""
        schema = ",".join(f"{key}:{value}" for key, value in sorted(self.arguments_schema.items()))
        return f"{self.name}({schema})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments_schema": dict(self.arguments_schema),
            "reads_context": self.reads_context,
            "writes_context": self.writes_context,
            "average_latency_ms": self.average_latency_ms,
            "success_rate": self.success_rate,
            "signature": self.signature(),
        }


@dataclass(frozen=True)
class MetaTool:
    """A frequent tool sequence compiled into one runtime workflow node."""

    meta_tool_id: str
    sequence: tuple[str, ...]
    frequency: int
    estimated_latency_ms: float
    kv_reuse_score: float
    failure_policy: str = "split_and_retry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_tool_id": self.meta_tool_id,
            "sequence": list(self.sequence),
            "frequency": self.frequency,
            "estimated_latency_ms": round(self.estimated_latency_ms, 4),
            "kv_reuse_score": round(self.kv_reuse_score, 4),
            "failure_policy": self.failure_policy,
        }


@dataclass(frozen=True)
class CascadeRoute:
    """Complexity-based model route for agentic requests."""

    route_id: str
    model_tier: str
    min_complexity: float
    max_complexity: float
    reasoning_budget_tokens: int
    speculative_decoding: bool

    def matches(self, complexity: float) -> bool:
        return self.min_complexity <= complexity <= self.max_complexity

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "model_tier": self.model_tier,
            "min_complexity": self.min_complexity,
            "max_complexity": self.max_complexity,
            "reasoning_budget_tokens": self.reasoning_budget_tokens,
            "speculative_decoding": self.speculative_decoding,
        }


class AgentWorkflowOptimizer:
    """Compile observed agent traces into runtime-ready workflow metadata."""

    def __init__(self, min_sequence_frequency: int = 2, max_sequence_length: int = 4) -> None:
        if min_sequence_frequency < 1:
            raise ValueError("min_sequence_frequency must be >= 1")
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be >= 2")
        self.min_sequence_frequency = min_sequence_frequency
        self.max_sequence_length = max_sequence_length
        self._routes = (
            CascadeRoute("fast", "small", 0.0, 0.35, 256, True),
            CascadeRoute("balanced", "medium", 0.35, 0.75, 1024, True),
            CascadeRoute("deep", "large", 0.75, 1.0, 4096, False),
        )

    def compile(self, traces: Iterable[Iterable[ToolCall]]) -> dict[str, Any]:
        """Compile traces into meta-tools, cache policy, and cascade routes."""
        normalized_traces = [list(trace) for trace in traces]
        sequence_counts: dict[tuple[str, ...], int] = {}
        latency_by_sequence: dict[tuple[str, ...], float] = {}
        kv_scores: dict[tuple[str, ...], float] = {}
        for trace in normalized_traces:
            signatures = [call.signature() for call in trace]
            for length in range(2, min(self.max_sequence_length, len(trace)) + 1):
                for start in range(0, len(trace) - length + 1):
                    window = tuple(signatures[start : start + length])
                    calls = trace[start : start + length]
                    sequence_counts[window] = sequence_counts.get(window, 0) + 1
                    latency_by_sequence[window] = latency_by_sequence.get(window, 0.0) + sum(
                        max(0.0, call.average_latency_ms) for call in calls
                    )
                    kv_scores[window] = kv_scores.get(window, 0.0) + self._kv_reuse_score(calls)
        meta_tools = []
        for sequence, frequency in sorted(sequence_counts.items(), key=lambda item: (-item[1], item[0])):
            if frequency < self.min_sequence_frequency:
                continue
            digest = hashlib.sha256("|".join(sequence).encode("utf-8")).hexdigest()[:16]
            meta_tools.append(
                MetaTool(
                    meta_tool_id=f"meta_{digest}",
                    sequence=sequence,
                    frequency=frequency,
                    estimated_latency_ms=latency_by_sequence[sequence] / frequency,
                    kv_reuse_score=kv_scores[sequence] / frequency,
                ).to_dict()
            )
        return {
            "version": "agentic_workflow/1.0",
            "trace_count": len(normalized_traces),
            "meta_tools": meta_tools,
            "routes": [route.to_dict() for route in self._routes],
            "context_cache": self.context_cache_policy(normalized_traces),
        }

    def route_for_prompt(self, prompt: str, tool_count: int = 0) -> CascadeRoute:
        """Return a deterministic cascade route from prompt complexity."""
        complexity = self.estimate_complexity(prompt, tool_count=tool_count)
        for route in self._routes:
            if route.matches(complexity):
                return route
        return self._routes[-1]

    def estimate_complexity(self, prompt: str, tool_count: int = 0) -> float:
        """Estimate request complexity using length, reasoning cues, and tools."""
        token_estimate = max(1, len(prompt.split()))
        length_score = min(0.45, token_estimate / 4000)
        reasoning_cues = ("prove", "derive", "plan", "debug", "analyze", "compare", "research")
        cue_score = min(0.35, sum(1 for cue in reasoning_cues if cue in prompt.lower()) * 0.08)
        tool_score = min(0.25, tool_count * 0.06)
        return min(1.0, length_score + cue_score + tool_score)

    def context_cache_policy(self, traces: Iterable[Iterable[ToolCall]]) -> dict[str, Any]:
        """Build a KV reuse policy for agent sessions."""
        read_count = 0
        write_count = 0
        total = 0
        for trace in traces:
            for call in trace:
                total += 1
                read_count += int(call.reads_context)
                write_count += int(call.writes_context)
        reuse_ratio = read_count / max(1, total)
        invalidation = "on_context_write" if write_count else "ttl_only"
        return {
            "enabled": True,
            "scope": "session_and_org_prefix",
            "reuse_ratio_estimate": round(reuse_ratio, 4),
            "invalidation": invalidation,
            "pinned_prefixes": ["system_prompt", "tool_schema", "rag_system_prompt"],
            "ttl_seconds": 3600,
        }

    def _kv_reuse_score(self, calls: list[ToolCall]) -> float:
        if not calls:
            return 0.0
        reads = sum(1 for call in calls if call.reads_context)
        writes = sum(1 for call in calls if call.writes_context)
        success = sum(call.success_rate for call in calls) / len(calls)
        return max(0.0, min(1.0, (reads / len(calls)) * success - writes * 0.05))
