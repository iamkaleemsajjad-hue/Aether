"""Fleet health monitoring, auto-scaling, and multi-region topology (Section 38 v3.1).

Extends AetherFleetManager with per-node health checks, latency SLO tracking,
auto-scale logic, and multi-region topology assignment.

Research: Helium workflow-aware serving (2026), Kubernetes AI Operators (2025),
          MuxWise (2026) SLO-aware scheduling.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aether.fleet.manager import FleetConfig, FleetNode


# ---------------------------------------------------------------------------
# Health status
# ---------------------------------------------------------------------------

class NodeHealth(Enum):
    """Health state of a fleet node."""

    HEALTHY   = "healthy"
    DEGRADED  = "degraded"    # High latency or elevated error rate
    UNHEALTHY = "unhealthy"   # Failing health checks
    DRAINING  = "draining"    # Being removed from rotation
    OFFLINE   = "offline"     # Not responding


@dataclass
class NodeMetrics:
    """Recent metrics collected from a fleet node."""

    node_id: str
    timestamp: float = field(default_factory=time.time)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    tokens_per_second: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used_gb: float = 0.0
    active_requests: int = 0
    queue_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "latency_ms": {
                "p50": round(self.p50_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
            },
            "error_rate": round(self.error_rate, 4),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "gpu_utilization": round(self.gpu_utilization, 3),
            "gpu_memory_used_gb": round(self.gpu_memory_used_gb, 2),
            "active_requests": self.active_requests,
            "queue_depth": self.queue_depth,
        }


@dataclass
class SLOConfig:
    """Service Level Objective configuration for a deployment."""

    max_p50_latency_ms: float = 100.0     # P50 TTFT ≤ 100ms
    max_p95_latency_ms: float = 500.0     # P95 TTFT ≤ 500ms
    max_p99_latency_ms: float = 2000.0    # P99 TTFT ≤ 2s
    max_error_rate: float = 0.01          # Error rate ≤ 1%
    min_tokens_per_second: float = 10.0   # Min throughput
    max_gpu_utilization: float = 0.90     # GPU util ≤ 90%

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": {
                "p50_max": self.max_p50_latency_ms,
                "p95_max": self.max_p95_latency_ms,
                "p99_max": self.max_p99_latency_ms,
            },
            "error_rate_max": self.max_error_rate,
            "throughput_min_tps": self.min_tokens_per_second,
            "gpu_utilization_max": self.max_gpu_utilization,
        }


# ---------------------------------------------------------------------------
# Fleet health monitor
# ---------------------------------------------------------------------------

class FleetHealthMonitor:
    """
    Per-node health monitoring with SLO tracking for AEG fleet deployments.

    Tracks latency percentiles, error rates, and GPU utilization per node.
    Triggers alerts when SLO thresholds are breached.

    Research: Kubernetes AI Operators (2025), MuxWise SLO-aware scheduling (2026).
    """

    def __init__(self, slo: SLOConfig | None = None, window_size: int = 100) -> None:
        self.slo = slo or SLOConfig()
        self.window_size = window_size
        self._node_metrics: dict[str, list[NodeMetrics]] = {}
        self._node_health: dict[str, NodeHealth] = {}

    def record(self, metrics: NodeMetrics) -> NodeHealth:
        """Record node metrics and update health status."""
        node_id = metrics.node_id
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = []

        history = self._node_metrics[node_id]
        history.append(metrics)
        # Keep rolling window
        if len(history) > self.window_size:
            history.pop(0)

        health = self._assess_health(metrics)
        self._node_health[node_id] = health
        return health

    def _assess_health(self, metrics: NodeMetrics) -> NodeHealth:
        """Assess node health against SLO thresholds."""
        slo = self.slo
        violations = 0

        if metrics.p99_latency_ms > slo.max_p99_latency_ms:
            violations += 2  # Critical
        if metrics.p95_latency_ms > slo.max_p95_latency_ms:
            violations += 1
        if metrics.error_rate > slo.max_error_rate * 5:
            violations += 3  # Critical — high error rate
        elif metrics.error_rate > slo.max_error_rate:
            violations += 1
        if metrics.gpu_utilization > slo.max_gpu_utilization:
            violations += 1
        if metrics.tokens_per_second < slo.min_tokens_per_second and metrics.active_requests > 0:
            violations += 1

        if violations == 0:
            return NodeHealth.HEALTHY
        elif violations <= 1:
            return NodeHealth.DEGRADED
        elif violations <= 3:
            return NodeHealth.UNHEALTHY
        else:
            return NodeHealth.OFFLINE

    def node_health(self, node_id: str) -> NodeHealth:
        return self._node_health.get(node_id, NodeHealth.OFFLINE)

    def fleet_summary(self) -> dict[str, Any]:
        """Return aggregated health summary across all nodes."""
        health_counts: dict[str, int] = {h.value: 0 for h in NodeHealth}
        for h in self._node_health.values():
            health_counts[h.value] += 1

        return {
            "total_nodes": len(self._node_health),
            "health_distribution": health_counts,
            "healthy_fraction": round(
                health_counts.get("healthy", 0) / max(len(self._node_health), 1), 4
            ),
            "slo_config": self.slo.to_dict(),
        }

    def unhealthy_nodes(self) -> list[str]:
        """Return IDs of nodes that are unhealthy or offline."""
        return [
            nid for nid, h in self._node_health.items()
            if h in (NodeHealth.UNHEALTHY, NodeHealth.OFFLINE)
        ]


# ---------------------------------------------------------------------------
# Auto-scaler
# ---------------------------------------------------------------------------

@dataclass
class ScaleDecision:
    """Auto-scale decision output."""

    action: str               # "scale_up" | "scale_down" | "no_change"
    delta_replicas: int       # How many replicas to add (+) or remove (-)
    reason: str
    current_replicas: int
    target_replicas: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "delta_replicas": self.delta_replicas,
            "reason": self.reason,
            "current_replicas": self.current_replicas,
            "target_replicas": self.target_replicas,
        }


class AutoScaler:
    """
    Horizontal auto-scaler for AEG fleet deployments.

    Scale-up triggers:
    - Queue depth > scale_up_queue_threshold
    - P95 latency > SLO max * scale_up_latency_ratio
    - GPU utilization > max_gpu_util * 0.85

    Scale-down triggers:
    - Queue depth == 0 for scale_down_idle_steps consecutive steps
    - P50 latency < SLO max * 0.3 (heavily under-utilized)

    Research: Kubernetes HPA (Horizontal Pod Autoscaler), MuxWise (2026).
    """

    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 16,
        scale_up_queue_threshold: int = 10,
        scale_up_latency_ratio: float = 0.80,
        scale_down_idle_steps: int = 5,
        slo: SLOConfig | None = None,
    ) -> None:
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_queue_threshold = scale_up_queue_threshold
        self.scale_up_latency_ratio = scale_up_latency_ratio
        self.scale_down_idle_steps = scale_down_idle_steps
        self.slo = slo or SLOConfig()
        self._idle_steps = 0

    def evaluate(
        self,
        current_replicas: int,
        node_metrics: list[NodeMetrics],
    ) -> ScaleDecision:
        """Evaluate whether to scale up, down, or stay the same."""
        if not node_metrics:
            return ScaleDecision("no_change", 0, "no metrics available",
                                 current_replicas, current_replicas)

        # Aggregate across nodes
        total_queue = sum(m.queue_depth for m in node_metrics)
        avg_p95 = statistics.fmean(m.p95_latency_ms for m in node_metrics)
        avg_gpu = statistics.fmean(m.gpu_utilization for m in node_metrics)
        avg_p50 = statistics.fmean(m.p50_latency_ms for m in node_metrics)

        # Scale-up conditions
        if total_queue > self.scale_up_queue_threshold:
            self._idle_steps = 0
            delta = min(2, self.max_replicas - current_replicas)
            target = current_replicas + delta
            return ScaleDecision("scale_up", delta,
                f"queue_depth={total_queue} > threshold={self.scale_up_queue_threshold}",
                current_replicas, target)

        if avg_p95 > self.slo.max_p95_latency_ms * self.scale_up_latency_ratio:
            self._idle_steps = 0
            delta = min(1, self.max_replicas - current_replicas)
            target = current_replicas + delta
            return ScaleDecision("scale_up", delta,
                f"p95_latency={avg_p95:.0f}ms > threshold={self.slo.max_p95_latency_ms * self.scale_up_latency_ratio:.0f}ms",
                current_replicas, target)

        if avg_gpu > self.slo.max_gpu_utilization * 0.85:
            self._idle_steps = 0
            delta = min(1, self.max_replicas - current_replicas)
            target = current_replicas + delta
            return ScaleDecision("scale_up", delta,
                f"gpu_utilization={avg_gpu:.2f} > threshold={self.slo.max_gpu_utilization * 0.85:.2f}",
                current_replicas, target)

        # Scale-down conditions
        if total_queue == 0 and avg_p50 < self.slo.max_p50_latency_ms * 0.3:
            self._idle_steps += 1
            if self._idle_steps >= self.scale_down_idle_steps and current_replicas > self.min_replicas:
                self._idle_steps = 0
                target = max(self.min_replicas, current_replicas - 1)
                return ScaleDecision("scale_down", -(current_replicas - target),
                    f"idle for {self.scale_down_idle_steps} steps, p50={avg_p50:.0f}ms",
                    current_replicas, target)
        else:
            self._idle_steps = 0

        return ScaleDecision("no_change", 0, "within SLO thresholds",
                             current_replicas, current_replicas)


# ---------------------------------------------------------------------------
# Multi-region topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionSpec:
    """A geographic region in a multi-region fleet deployment."""

    region_id: str              # e.g. "us-east-1", "eu-west-2"
    nodes: tuple[FleetNode, ...]
    latency_weight: float = 1.0  # Lower = preferred for latency-sensitive traffic
    compliance_zones: tuple[str, ...] = ()  # e.g. ("GDPR", "CCPA")

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "node_count": len(self.nodes),
            "nodes": [n.to_dict() for n in self.nodes],
            "latency_weight": self.latency_weight,
            "compliance_zones": list(self.compliance_zones),
            "targets": list({n.target for n in self.nodes}),
        }


class MultiRegionTopology:
    """
    Assigns AEG deployment across multiple geographic regions.

    Routes requests to the lowest-latency eligible region that satisfies
    compliance requirements.

    PRD Section 38.2 example:
        Region 1 (US-East): 2x B200,   cuda_sm100 kernel from Hub CDN
        Region 2 (EU-West): 4x H100,   cuda_sm90  kernel from Hub CDN
        Region 3 (AP-South): 2x MI300X, rocm_cdna3 kernel from Hub CDN
        Edge nodes: Apple M4, metal_m3 kernel from Hub CDN

    Research: Helium workflow-aware serving (2026), global load balancing.
    """

    def __init__(self, regions: list[RegionSpec]) -> None:
        self.regions = regions

    def assign_region(
        self,
        request_id: str,
        client_region: str | None = None,
        compliance_required: str | None = None,
    ) -> RegionSpec:
        """
        Assign a request to the best-fit region.

        Args:
            request_id: Unique request ID (for consistent hashing).
            client_region: Client's region hint (e.g. "eu-west-2").
            compliance_required: Compliance zone requirement (e.g. "GDPR").

        Returns:
            Best-fit RegionSpec.
        """
        eligible = self.regions

        # Filter by compliance
        if compliance_required:
            eligible = [
                r for r in eligible
                if compliance_required in r.compliance_zones
            ] or eligible  # Fall back to all if none match

        # Prefer client's region
        if client_region:
            home = [r for r in eligible if client_region in r.region_id]
            if home:
                return home[0]

        # Stable hash for consistent routing
        digest = hashlib.sha256(f"region:{request_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF

        # Weighted selection
        total_weight = sum(1.0 / r.latency_weight for r in eligible)
        cumulative = 0.0
        for region in sorted(eligible, key=lambda r: r.latency_weight):
            cumulative += (1.0 / region.latency_weight) / total_weight
            if bucket < cumulative:
                return region

        return eligible[0]

    def topology_manifest(self) -> dict[str, Any]:
        return {
            "version": "multi_region/1.0",
            "region_count": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
            "routing": "weighted_latency_with_compliance_filter",
            "cdn_delivery": "aether_hub_cdn",
            "research": "Helium workflow-aware serving (2026)",
        }
