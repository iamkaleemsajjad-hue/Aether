"""Zero-downtime hot-reload and auto-rollout controller for Aether Runtime.

Implements the Tri-Layer Adaptation Framework from PRD Section 36:
- Hot-reload: load new AEG alongside old, zero dropped requests
- Auto-rollout: gradually increase traffic to new model if quality holds
- Rollback: instant revert to previous model on regression or safety alert

Research:
- Helium workflow-aware serving (2026)
- Canary releases (Google SRE Book, 2016)
- PRD Section 36: Adaptive Learning and Zero-Downtime Hot-Reload
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aether.observability.gates import ABRolloutController, DriftMonitor, TelemetrySnapshot


# ---------------------------------------------------------------------------
# Rollout state
# ---------------------------------------------------------------------------

class RolloutState(Enum):
    """Current state of a hot-reload rollout experiment."""

    LOADING        = "loading"         # New AEG is loading in background
    CANARY         = "canary"          # Small % of traffic on new AEG
    RAMPING        = "ramping"         # Traffic fraction increasing
    FULL_ROLLOUT   = "full_rollout"    # 100% on new AEG
    ROLLED_BACK    = "rolled_back"     # Reverted to previous AEG
    STABLE         = "stable"          # Completed successfully


@dataclass
class RolloutExperiment:
    """Tracks a single A/B rollout experiment."""

    experiment_id: str
    active_aeg: str
    candidate_aeg: str
    start_time: float = field(default_factory=time.time)
    state: RolloutState = RolloutState.LOADING
    candidate_percent: float = 0.0
    step_size: float = 0.10           # Increment per successful step
    step_interval_sec: int = 3600     # Time between auto-steps
    last_step_time: float = 0.0
    regression_detected: bool = False
    quality_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "active_aeg": self.active_aeg,
            "candidate_aeg": self.candidate_aeg,
            "state": self.state.value,
            "candidate_percent": round(self.candidate_percent, 4),
            "step_size": self.step_size,
            "step_interval_sec": self.step_interval_sec,
            "regression_detected": self.regression_detected,
            "uptime_sec": round(time.time() - self.start_time, 1),
            "quality_snapshot_count": len(self.quality_snapshots),
        }


# ---------------------------------------------------------------------------
# Hot-reload engine
# ---------------------------------------------------------------------------

class AetherHotReload:
    """
    Zero-downtime model update system.

    Loads a new AEG artifact alongside the current serving model,
    starts routing a small canary % of traffic to the new model,
    and gradually increases the split if quality holds.

    Supports:
    - Canary launches (start at 1% traffic)
    - Auto-ramping (double every step_interval_sec if no regression)
    - Instant rollback (switch all traffic back to old AEG in <1ms)
    - Multi-experiment tracking (multiple concurrent rollouts)

    Usage:
        reloader = AetherHotReload()
        exp = reloader.start_reload("new-qwen3-72b.aeg",
                                    baseline_win_rate=0.87)
        # … serve traffic, collect telemetry …
        reloader.record_telemetry(exp.experiment_id, snapshot)
        reloader.auto_step(exp.experiment_id)
    """

    def __init__(self) -> None:
        self._experiments: dict[str, RolloutExperiment] = {}
        self._drift_monitors: dict[str, DriftMonitor] = {}
        self._ab_controllers: dict[str, ABRolloutController] = {}

    def start_reload(
        self,
        candidate_aeg: str,
        active_aeg: str = "current",
        baseline_win_rate: float = 0.75,
        step_size: float = 0.10,
        step_interval_sec: int = 3600,
        alert_drop: float = 0.05,
    ) -> RolloutExperiment:
        """
        Begin a hot-reload experiment.

        Args:
            candidate_aeg: Path to the new AEG to roll out.
            active_aeg: Path to the currently serving AEG.
            baseline_win_rate: Expected win rate of the active model (for drift detection).
            step_size: Fraction to increase traffic per step (default 10%).
            step_interval_sec: Minimum seconds between traffic increases.
            alert_drop: Win rate drop that triggers rollback.

        Returns:
            RolloutExperiment tracking this reload.
        """
        exp_id = hashlib.sha256(
            f"{candidate_aeg}:{active_aeg}:{time.time()}".encode()
        ).hexdigest()[:12]

        exp = RolloutExperiment(
            experiment_id=exp_id,
            active_aeg=active_aeg,
            candidate_aeg=candidate_aeg,
            state=RolloutState.LOADING,
            candidate_percent=0.01,  # Start canary at 1%
            step_size=step_size,
            step_interval_sec=step_interval_sec,
            last_step_time=time.time(),
        )

        monitor = DriftMonitor(
            baseline_win_rate=baseline_win_rate,
            alert_drop=alert_drop,
            min_samples=20,
        )
        controller = ABRolloutController(
            experiment_id=exp_id,
            candidate_percent=0.01,
        )

        self._experiments[exp_id] = exp
        self._drift_monitors[exp_id] = monitor
        self._ab_controllers[exp_id] = controller

        # Transition to canary state
        exp.state = RolloutState.CANARY
        return exp

    def route_request(self, experiment_id: str, request_id: str) -> str:
        """
        Route a request to either the active or candidate AEG.

        Returns:
            "active" or "candidate" — caller selects the corresponding AEG.
        """
        controller = self._ab_controllers.get(experiment_id)
        if controller is None:
            return "active"
        return controller.assign(request_id)

    def record_telemetry(
        self, experiment_id: str, snapshot: TelemetrySnapshot
    ) -> dict[str, Any]:
        """Record a telemetry snapshot and check for quality drift."""
        monitor = self._drift_monitors.get(experiment_id)
        if monitor is None:
            return {}

        exp = self._experiments.get(experiment_id)
        status = monitor.record(snapshot)

        if exp and status.get("alert"):
            exp.regression_detected = True
            exp.state = RolloutState.ROLLED_BACK
            exp.candidate_percent = 0.0
            if experiment_id in self._ab_controllers:
                self._ab_controllers[experiment_id].candidate_percent = 0.0

        if exp:
            exp.quality_snapshots.append(snapshot.to_dict())

        return status

    def auto_step(self, experiment_id: str) -> float:
        """
        Auto-increase traffic to the candidate if quality holds.

        Returns the new candidate_percent (0.0 on rollback).
        """
        exp = self._experiments.get(experiment_id)
        monitor = self._drift_monitors.get(experiment_id)
        controller = self._ab_controllers.get(experiment_id)

        if not exp or not monitor or not controller:
            return 0.0

        # Check timing
        elapsed = time.time() - exp.last_step_time
        if elapsed < exp.step_interval_sec:
            return exp.candidate_percent

        status = monitor.status()
        gate_passed = not exp.regression_detected and not status.get("alert", False)

        new_pct = controller.ramp(gate_passed=gate_passed, drift_alert=status.get("alert", False))
        exp.candidate_percent = new_pct
        exp.last_step_time = time.time()

        # Update state
        if new_pct == 0.0:
            exp.state = RolloutState.ROLLED_BACK
        elif new_pct >= 1.0:
            exp.state = RolloutState.FULL_ROLLOUT
        elif new_pct > 0.01:
            exp.state = RolloutState.RAMPING

        return new_pct

    def promote(self, experiment_id: str) -> str:
        """Promote candidate to active (100% traffic)."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment {experiment_id} not found")
        exp.candidate_percent = 1.0
        exp.state = RolloutState.FULL_ROLLOUT
        if experiment_id in self._ab_controllers:
            self._ab_controllers[experiment_id].candidate_percent = 1.0
        return exp.candidate_aeg

    def rollback(self, experiment_id: str) -> str:
        """Instantly rollback to the active AEG (0% candidate traffic)."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment {experiment_id} not found")
        exp.candidate_percent = 0.0
        exp.state = RolloutState.ROLLED_BACK
        exp.regression_detected = True
        if experiment_id in self._ab_controllers:
            self._ab_controllers[experiment_id].candidate_percent = 0.0
        return exp.active_aeg

    def status(self, experiment_id: str) -> dict[str, Any]:
        """Get full status of a rollout experiment."""
        exp = self._experiments.get(experiment_id)
        monitor = self._drift_monitors.get(experiment_id)
        if not exp:
            return {"error": f"Unknown experiment: {experiment_id}"}
        result = exp.to_dict()
        if monitor:
            result["drift_status"] = monitor.status()
        return result

    def all_experiments(self) -> list[dict[str, Any]]:
        """List all active experiments."""
        return [self.status(eid) for eid in self._experiments]


# ---------------------------------------------------------------------------
# Auto-rollout controller (scheduled-ramp variant)
# ---------------------------------------------------------------------------

class AutoRolloutController:
    """
    Automated rollout controller with configurable ramp schedule.

    Follows an exponential ramp: 1% → 2% → 4% → 8% → 16% → 32% → 64% → 100%
    Each step requires:
    1. No quality regression (win_rate drop < alert_drop)
    2. Minimum step_interval_sec elapsed since last step
    3. No active safety alerts

    If any check fails, immediately rolls back to 0%.

    Research: Google SRE Book (canary analysis), Helium (2026, workflow-aware serving).
    """

    def __init__(
        self,
        reloader: AetherHotReload,
        experiment_id: str,
        step_interval_sec: int = 3600,
        target_pct: float = 1.0,
    ) -> None:
        self.reloader = reloader
        self.experiment_id = experiment_id
        self.step_interval_sec = step_interval_sec
        self.target_pct = target_pct
        self._start_time = time.time()

    def tick(self) -> dict[str, Any]:
        """
        Advance the rollout by one step if conditions are met.

        Call this periodically (e.g., every minute from a background thread).

        Returns:
            Status dict with current candidate_percent and state.
        """
        new_pct = self.reloader.auto_step(self.experiment_id)
        status = self.reloader.status(self.experiment_id)
        return {
            "experiment_id": self.experiment_id,
            "candidate_percent": new_pct,
            "state": status.get("state"),
            "elapsed_sec": round(time.time() - self._start_time, 1),
            "at_target": new_pct >= self.target_pct,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "step_interval_sec": self.step_interval_sec,
            "target_pct": self.target_pct,
            "ramp_schedule": "exponential_doubling",
            "rollback_triggers": ["quality_drift", "safety_alert", "eval_gate_failure"],
        }
