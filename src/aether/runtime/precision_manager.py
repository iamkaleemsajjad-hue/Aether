"""
Dynamic precision manager.

Adjusts the active precision of a running model based on memory pressure,
quality budget, and observed perplexity. This provides a runtime feedback
loop that can downgrade precision to avoid OOMs or upgrade precision when
quality is degraded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aether.core.constants import PrecisionConstants
from aether.core.exceptions import PrecisionAdjustmentError
from aether.core.types import Precision
from aether.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PrecisionState:
    """Current precision state for a loaded model."""

    model_id: str
    active_map: dict[str, str] = field(default_factory=dict)
    memory_pressure: float = 0.0
    quality_budget: float = 0.02
    last_adjustment: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: dict[str, Any]) -> None:
        """Record a precision adjustment event."""
        event["timestamp"] = time.time()
        self.history.append(event)
        self.last_adjustment = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "memory_pressure": self.memory_pressure,
            "quality_budget": self.quality_budget,
            "last_adjustment": self.last_adjustment,
            "active_map": self.active_map,
        }


class PrecisionManager:
    """Manages dynamic precision adjustment at runtime."""

    def __init__(self, quality_budget: float = 0.02, cooldown_seconds: float = 5.0) -> None:
        self.quality_budget = quality_budget
        self.cooldown_seconds = cooldown_seconds
        self._states: dict[str, PrecisionState] = {}

    def register(self, model_id: str, precision_map: dict[str, str]) -> PrecisionState:
        """Register a model with an initial precision map."""
        state = PrecisionState(model_id=model_id, active_map=dict(precision_map))
        self._states[model_id] = state
        logger.info("Registered precision state", model_id=model_id, layers=len(precision_map))
        return state

    def get_state(self, model_id: str) -> PrecisionState | None:
        """Return the precision state for a model."""
        return self._states.get(model_id)

    def update_memory_pressure(self, model_id: str, pressure: float) -> None:
        """Update the observed memory pressure for a model."""
        state = self._states.get(model_id)
        if state is None:
            return
        state.memory_pressure = max(0.0, min(1.0, pressure))

    def _can_adjust(self, state: PrecisionState) -> bool:
        """Check whether enough time has passed since the last adjustment."""
        if self.cooldown_seconds <= 0:
            return True
        return time.time() - state.last_adjustment > self.cooldown_seconds

    def _downgrade_layer(self, precision: str) -> str:
        """Return the next lower precision tier for a layer."""
        tiers: list[str] = ["BF16", "FP8", "Q4_K_M", "Q3_K", "Q2_K"]
        try:
            idx = tiers.index(precision.upper())
        except ValueError:
            idx = 0
        if idx < len(tiers) - 1:
            return tiers[idx + 1]
        return precision

    def _upgrade_layer(self, precision: str) -> str:
        """Return the next higher precision tier for a layer."""
        tiers: list[str] = ["BF16", "FP8", "Q4_K_M", "Q3_K", "Q2_K"]
        try:
            idx = tiers.index(precision.upper())
        except ValueError:
            idx = 0
        if idx > 0:
            return tiers[idx - 1]
        return precision

    def adjust(self, model_id: str, observed_perplexity_delta: float | None = None) -> dict[str, Any]:
        """Adjust precision based on memory pressure and quality feedback.

        Args:
            model_id: Model to adjust.
            observed_perplexity_delta: Optional observed perplexity increase.

        Returns:
            A dictionary describing the adjustment.
        """
        state = self._states.get(model_id)
        if state is None:
            msg = f"Model {model_id} is not registered with the precision manager"
            raise PrecisionAdjustmentError(msg, model_id=model_id)
        if not self._can_adjust(state):
            return {"model_id": model_id, "action": "cooldown", "reason": "adjustment cooldown"}

        action = "none"
        changes: dict[str, tuple[str, str]] = {}

        if state.memory_pressure > 0.9:
            action = "downgrade"
            for layer, precision in state.active_map.items():
                if layer in ("embedding", "lm_head"):
                    continue
                new_precision = self._downgrade_layer(precision)
                if new_precision != precision:
                    changes[layer] = (precision, new_precision)
                    state.active_map[layer] = new_precision
        elif state.memory_pressure > 0.75:
            action = "downgrade_selective"
            for layer, precision in state.active_map.items():
                if layer in ("embedding", "lm_head"):
                    continue
                if precision.upper() in ("BF16", "FP8"):
                    new_precision = self._downgrade_layer(precision)
                    changes[layer] = (precision, new_precision)
                    state.active_map[layer] = new_precision

        if observed_perplexity_delta is not None and observed_perplexity_delta > self.quality_budget:
            action = "upgrade" if action == "none" else f"{action}_then_upgrade"
            for layer, (old, new) in list(changes.items()):
                if PrecisionConstants.bit_width(old) > PrecisionConstants.bit_width(new):
                    state.active_map[layer] = old
                    del changes[layer]
            for layer, precision in state.active_map.items():
                if layer in ("embedding", "lm_head"):
                    continue
                if precision.upper() in ("Q2_K", "Q3_K"):
                    new_precision = self._upgrade_layer(precision)
                    if new_precision != precision:
                        changes[layer] = (precision, new_precision)
                        state.active_map[layer] = new_precision

        state.record(
            {
                "action": action,
                "memory_pressure": state.memory_pressure,
                "observed_perplexity_delta": observed_perplexity_delta,
                "changes": changes,
            }
        )
        logger.info(
            "Precision adjustment",
            model_id=model_id,
            action=action,
            changes=len(changes),
            memory_pressure=state.memory_pressure,
        )
        return {
            "model_id": model_id,
            "action": action,
            "changes": {k: {"from": v[0], "to": v[1]} for k, v in changes.items()},
            "memory_pressure": state.memory_pressure,
        }

    def reset(self, model_id: str, precision_map: dict[str, str]) -> None:
        """Reset the precision map to the original compiled values."""
        state = self._states.get(model_id)
        if state is None:
            return
        state.active_map = dict(precision_map)
        state.memory_pressure = 0.0
        state.record({"action": "reset"})
        logger.info("Precision reset", model_id=model_id)

    def summary(self, model_id: str) -> dict[str, Any]:
        """Return a summary of the precision state."""
        state = self._states.get(model_id)
        if state is None:
            return {"model_id": model_id, "registered": False}
        return {
            "model_id": model_id,
            "registered": True,
            "memory_pressure": state.memory_pressure,
            "quality_budget": state.quality_budget,
            "active_map": state.active_map,
            "history_count": len(state.history),
        }

    def __repr__(self) -> str:
        return f"PrecisionManager(models={len(self._states)}, budget={self.quality_budget})"
