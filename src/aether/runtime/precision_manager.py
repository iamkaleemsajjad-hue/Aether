"""
Dynamic precision manager.

At runtime, Aether monitors memory pressure and quality signals and adjusts
the active precision tier per-layer. The precision ladder is:

  BF16 → FP8_E4M3 → Q4_K_M → Q3_K → Q2_K

Each downgrade step reduces VRAM by ~2× with a bounded quality penalty.
The manager also handles the reverse path (upgrade) when pressure drops.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

# Ordered from highest quality/cost to lowest
_PRECISION_LADDER: list[str] = [
    "BF16",
    "FP16",
    "FP8_E4M3",
    "FP8",
    "Q8_0",
    "Q6_K",
    "Q4_K_M",
    "Q4_0",
    "Q3_K",
    "Q2_K",
]

# Approximate memory footprint relative to BF16=1.0
_PRECISION_COST: dict[str, float] = {
    "BF16":     1.0,
    "FP16":     1.0,
    "FP8_E4M3": 0.5,
    "FP8":      0.5,
    "Q8_0":     0.5,
    "Q6_K":     0.375,
    "Q4_K_M":   0.25,
    "Q4_0":     0.25,
    "Q3_K":     0.1875,
    "Q2_K":     0.125,
}

# Approximate perplexity penalty (relative increase) vs BF16
_PRECISION_PPL_PENALTY: dict[str, float] = {
    "BF16":     0.0,
    "FP16":     0.0,
    "FP8_E4M3": 0.001,
    "FP8":      0.002,
    "Q8_0":     0.005,
    "Q6_K":     0.01,
    "Q4_K_M":   0.03,
    "Q4_0":     0.05,
    "Q3_K":     0.09,
    "Q2_K":     0.18,
}

# Threshold for downgrade / upgrade
_DOWNGRADE_PRESSURE_THRESHOLD = 0.85  # fraction of VRAM used
_UPGRADE_PRESSURE_THRESHOLD   = 0.60


@dataclass
class PrecisionState:
    """Current precision state for a single layer."""

    layer_id: str
    target_precision: str           # precision specified at compile time
    active_precision: str           # current runtime precision
    sensitivity: float = 0.5        # 0=insensitive, 1=critical
    locked: bool = False            # True for embedding/lm_head

    def can_downgrade(self) -> bool:
        idx = _PRECISION_LADDER.index(self.active_precision) if self.active_precision in _PRECISION_LADDER else 0
        return not self.locked and idx < len(_PRECISION_LADDER) - 1

    def can_upgrade(self) -> bool:
        target_idx = _PRECISION_LADDER.index(self.target_precision) if self.target_precision in _PRECISION_LADDER else 0
        active_idx = _PRECISION_LADDER.index(self.active_precision) if self.active_precision in _PRECISION_LADDER else 0
        return not self.locked and active_idx > target_idx

    def downgrade(self) -> str | None:
        if not self.can_downgrade():
            return None
        idx = _PRECISION_LADDER.index(self.active_precision)
        self.active_precision = _PRECISION_LADDER[idx + 1]
        return self.active_precision

    def upgrade(self) -> str | None:
        if not self.can_upgrade():
            return None
        idx = _PRECISION_LADDER.index(self.active_precision)
        self.active_precision = _PRECISION_LADDER[idx - 1]
        return self.active_precision

    def memory_ratio(self) -> float:
        return _PRECISION_COST.get(self.active_precision, 1.0)

    def ppl_penalty(self) -> float:
        return _PRECISION_PPL_PENALTY.get(self.active_precision, 0.0)


@dataclass
class PrecisionSnapshot:
    """Snapshot of precision state and memory statistics."""

    timestamp: float
    memory_pressure: float
    layers: dict[str, str]          # layer_id → active_precision
    downgrades: int = 0
    upgrades: int = 0
    estimated_ppl_delta: float = 0.0


class DynamicPrecisionManager:
    """
    Monitors VRAM pressure and adjusts active precision per layer.

    Usage:
        mgr = DynamicPrecisionManager(precision_map={...})
        # On each generation step:
        mgr.update(memory_pressure=0.9)
        active = mgr.get_active_precision("layer_3")
    """

    def __init__(
        self,
        precision_map: dict[str, str] | None = None,
        sensitivity_map: dict[str, float] | None = None,
        max_ppl_delta: float = 0.05,
    ) -> None:
        self._lock = threading.RLock()
        self._states: dict[str, PrecisionState] = {}
        self._max_ppl_delta = max_ppl_delta
        self._history: list[PrecisionSnapshot] = []
        self._memory_pressure: float = 0.0
        self._downgrade_count: int = 0
        self._upgrade_count: int = 0

        if precision_map:
            self._initialize(precision_map, sensitivity_map or {})

    def _initialize(
        self,
        precision_map: dict[str, str],
        sensitivity_map: dict[str, float],
    ) -> None:
        """Build per-layer precision states from maps."""
        with self._lock:
            for layer_id, precision in precision_map.items():
                sensitivity = sensitivity_map.get(layer_id, 0.5)
                locked = layer_id in {"embedding", "lm_head"}
                self._states[layer_id] = PrecisionState(
                    layer_id=layer_id,
                    target_precision=precision,
                    active_precision=precision,
                    sensitivity=sensitivity,
                    locked=locked,
                )
        logger.info(
            "Precision manager initialized",
            layers=len(self._states),
            max_ppl_delta=self._max_ppl_delta,
        )

    def load_from_precision_map(
        self,
        precision_map: dict[str, str],
        sensitivity_map: dict[str, float] | None = None,
    ) -> None:
        """Load or reload from an AEG precision map."""
        self._initialize(precision_map, sensitivity_map or {})

    # ------------------------------------------------------------------
    # Pressure response
    # ------------------------------------------------------------------

    def update(self, memory_pressure: float) -> PrecisionSnapshot:
        """
        Respond to current memory pressure by adjusting precision.

        Args:
            memory_pressure: Fraction of VRAM consumed (0.0 – 1.0).

        Returns:
            A PrecisionSnapshot with the changes made this step.
        """
        with self._lock:
            self._memory_pressure = max(0.0, min(1.0, memory_pressure))
            downgrades = 0
            upgrades = 0

            if self._memory_pressure >= _DOWNGRADE_PRESSURE_THRESHOLD:
                downgrades = self._apply_downgrades()
            elif self._memory_pressure <= _UPGRADE_PRESSURE_THRESHOLD:
                upgrades = self._apply_upgrades()

            snapshot = PrecisionSnapshot(
                timestamp=time.monotonic(),
                memory_pressure=self._memory_pressure,
                layers=self.get_active_map(),
                downgrades=downgrades,
                upgrades=upgrades,
                estimated_ppl_delta=self.estimated_ppl_delta(),
            )
            self._history.append(snapshot)
            if len(self._history) > 100:
                self._history = self._history[-50:]
            return snapshot

    def _apply_downgrades(self) -> int:
        """
        Downgrade the least-sensitive unlocked layers until pressure is acceptable
        or we hit the max_ppl_delta budget.

        Returns the number of layers downgraded.
        """
        count = 0
        # Sort by sensitivity ascending: least sensitive first
        candidates = sorted(
            [s for s in self._states.values() if s.can_downgrade()],
            key=lambda s: (s.sensitivity, s.layer_id),
        )
        for state in candidates:
            current_ppl = self.estimated_ppl_delta()
            # Estimate penalty increase for this downgrade
            idx = _PRECISION_LADDER.index(state.active_precision)
            next_prec = _PRECISION_LADDER[idx + 1]
            delta_increase = _PRECISION_PPL_PENALTY[next_prec] - _PRECISION_PPL_PENALTY[state.active_precision]
            if current_ppl + delta_increase > self._max_ppl_delta:
                break  # Quality budget exhausted
            old = state.active_precision
            new = state.downgrade()
            if new:
                self._downgrade_count += 1
                count += 1
                logger.debug("Precision: %s %s→%s (pressure=%.2f)", state.layer_id, old, new, self._memory_pressure)
            if self._memory_pressure < _DOWNGRADE_PRESSURE_THRESHOLD:
                break
        return count

    def _apply_upgrades(self) -> int:
        """
        Upgrade layers that can return to their target precision.

        Returns the number of layers upgraded.
        """
        count = 0
        candidates = sorted(
            [s for s in self._states.values() if s.can_upgrade()],
            key=lambda s: (-s.sensitivity, s.layer_id),
        )
        for state in candidates:
            old = state.active_precision
            new = state.upgrade()
            if new:
                self._upgrade_count += 1
                count += 1
                logger.debug("Precision: %s %s→%s (upgrade)", state.layer_id, old, new)
        return count

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_precision(self, layer_id: str) -> str:
        """Return the current active precision for a layer."""
        with self._lock:
            state = self._states.get(layer_id)
            return state.active_precision if state else "BF16"

    def get_active_map(self) -> dict[str, str]:
        """Return {layer_id: active_precision} for all layers."""
        with self._lock:
            return {lid: s.active_precision for lid, s in self._states.items()}

    def estimated_ppl_delta(self) -> float:
        """Estimate the total perplexity increase from current precision assignments."""
        with self._lock:
            if not self._states:
                return 0.0
            penalties = [
                s.ppl_penalty() * (1.0 + s.sensitivity)
                for s in self._states.values()
            ]
            return sum(penalties) / max(len(penalties), 1)

    def memory_pressure(self) -> float:
        """Return the last recorded memory pressure."""
        with self._lock:
            return self._memory_pressure

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "layer_count": len(self._states),
                "memory_pressure": self._memory_pressure,
                "estimated_ppl_delta": self.estimated_ppl_delta(),
                "downgrade_count": self._downgrade_count,
                "upgrade_count": self._upgrade_count,
                "active_map": self.get_active_map(),
                "max_ppl_delta": self._max_ppl_delta,
            }

    def reset(self) -> None:
        """Reset all layers to their target precision."""
        with self._lock:
            for state in self._states.values():
                state.active_precision = state.target_precision
            logger.info("Precision manager reset to target precisions")

    def __repr__(self) -> str:
        return (
            f"DynamicPrecisionManager(layers={len(self._states)}, "
            f"pressure={self._memory_pressure:.2f}, "
            f"ppl_delta={self.estimated_ppl_delta():.4f})"
        )


# ---------------------------------------------------------------------------
# Multi-model precision manager
# ---------------------------------------------------------------------------

@dataclass
class ModelPrecisionState:
    """Runtime precision state for one loaded model."""

    model_id: str
    #: Precision map as compiled into the AEG package.
    target_map: dict[str, str]
    #: Precision map currently in force; diverges from target under pressure.
    active_map: dict[str, str]
    memory_pressure: float = 0.0
    last_adjust_at: float = 0.0
    downgrades: int = 0
    upgrades: int = 0

    @property
    def is_degraded(self) -> bool:
        """True when any layer is running below its compiled precision."""
        return self.active_map != self.target_map

    def estimated_memory_ratio(self) -> float:
        """Mean memory footprint of the active map relative to BF16."""
        if not self.active_map:
            return 1.0
        costs = [_PRECISION_COST.get(p, 1.0) for p in self.active_map.values()]
        return sum(costs) / len(costs)

    def estimated_ppl_delta(self) -> float:
        """Mean perplexity penalty implied by the active map."""
        if not self.active_map:
            return 0.0
        penalties = [
            _PRECISION_PPL_PENALTY.get(p, 0.0) for p in self.active_map.values()
        ]
        return sum(penalties) / len(penalties)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "memory_pressure": round(self.memory_pressure, 4),
            "is_degraded": self.is_degraded,
            "downgrades": self.downgrades,
            "upgrades": self.upgrades,
            "active_map": dict(self.active_map),
            "target_map": dict(self.target_map),
            "estimated_memory_ratio": round(self.estimated_memory_ratio(), 4),
            "estimated_ppl_delta": round(self.estimated_ppl_delta(), 6),
        }


class PrecisionManager:
    """
    Precision manager for the multi-model runtime.

    Where :class:`DynamicPrecisionManager` tracks layers of a single model,
    this manages every model resident in the runtime and gates changes behind
    a cooldown. The cooldown matters: precision changes invalidate compiled
    kernels and CUDA graphs, so reacting to every pressure spike would thrash
    the kernel cache and cost more than the memory saved.

    Usage:
        mgr = PrecisionManager(cooldown_seconds=30.0)
        mgr.register("qwen3-8b", {"layer_0": "BF16", ...})
        mgr.update_memory_pressure("qwen3-8b", 0.94)
        result = mgr.adjust("qwen3-8b")
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        downgrade_threshold: float = _DOWNGRADE_PRESSURE_THRESHOLD,
        upgrade_threshold: float = _UPGRADE_PRESSURE_THRESHOLD,
        max_ppl_delta: float = 0.05,
    ) -> None:
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.downgrade_threshold = downgrade_threshold
        self.upgrade_threshold = upgrade_threshold
        self.max_ppl_delta = max_ppl_delta
        self._lock = threading.RLock()
        self._states: dict[str, ModelPrecisionState] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self, model_id: str, precision_map: dict[str, str]
    ) -> ModelPrecisionState:
        """
        Register a model's compiled precision map.

        Returns the tracked state. Re-registering resets the model to its
        compiled precision.
        """
        with self._lock:
            state = ModelPrecisionState(
                model_id=model_id,
                target_map=dict(precision_map),
                active_map=dict(precision_map),
                # Start the cooldown clock at registration: a model's memory
                # footprint is still settling right after load, so the first
                # pressure reading is not yet a reason to change precision.
                last_adjust_at=time.monotonic(),
            )
            self._states[model_id] = state
            logger.info(
                "Precision manager registered model",
                model_id=model_id,
                layers=len(precision_map),
            )
            return state

    def unregister(self, model_id: str) -> bool:
        """Drop a model's precision state. Returns True if it was present."""
        with self._lock:
            return self._states.pop(model_id, None) is not None

    def get_state(self, model_id: str) -> ModelPrecisionState | None:
        """Return the tracked state for a model, or None if unregistered."""
        with self._lock:
            return self._states.get(model_id)

    def get_active_precision(self, model_id: str, layer_id: str) -> str:
        """Active precision for one layer, falling back to BF16."""
        with self._lock:
            state = self._states.get(model_id)
            if state is None:
                return "BF16"
            return state.active_map.get(layer_id, "BF16")

    # ------------------------------------------------------------------
    # Pressure tracking and adjustment
    # ------------------------------------------------------------------

    def update_memory_pressure(self, model_id: str, pressure: float) -> float:
        """
        Record current memory pressure for a model.

        Args:
            pressure: Fraction of device memory consumed, clamped to [0, 1].

        Returns:
            The clamped pressure that was recorded.
        """
        with self._lock:
            state = self._states.get(model_id)
            if state is None:
                raise KeyError(f"Model {model_id!r} is not registered")
            state.memory_pressure = max(0.0, min(1.0, pressure))
            return state.memory_pressure

    def adjust(self, model_id: str) -> dict[str, Any]:
        """
        Apply a precision adjustment if pressure warrants and cooldown allows.

        Returns a dict whose ``action`` is one of:

        * ``cooldown``  — a change happened too recently; nothing done.
        * ``downgrade`` — every eligible layer stepped down one rung.
        * ``downgrade_selective`` — some layers stepped down; others were held
          back by the quality budget or the bottom of the ladder.
        * ``upgrade``   — layers moved back toward compiled precision.
        * ``none``      — pressure is in the neutral band, or nothing to do.
        """
        with self._lock:
            state = self._states.get(model_id)
            if state is None:
                raise KeyError(f"Model {model_id!r} is not registered")

            now = time.monotonic()
            if self.cooldown_seconds > 0.0 and state.last_adjust_at > 0.0:
                elapsed = now - state.last_adjust_at
                if elapsed < self.cooldown_seconds:
                    return {
                        "action": "cooldown",
                        "model_id": model_id,
                        "seconds_remaining": round(self.cooldown_seconds - elapsed, 3),
                    }

            if state.memory_pressure >= self.downgrade_threshold:
                result = self._downgrade(state)
            elif state.memory_pressure <= self.upgrade_threshold:
                result = self._upgrade(state)
            else:
                return {"action": "none", "model_id": model_id, "reason": "neutral_band"}

            if result["action"] != "none":
                state.last_adjust_at = now
            return result

    def _downgrade(self, state: ModelPrecisionState) -> dict[str, Any]:
        """Step eligible layers one rung down the precision ladder."""
        changed: dict[str, tuple[str, str]] = {}
        blocked = 0

        # Least-sensitive first is not knowable here, so walk deterministically
        # and stop individual layers that would breach the quality budget.
        for layer_id in sorted(state.active_map):
            current = state.active_map[layer_id]
            if layer_id in {"embedding", "lm_head"}:
                blocked += 1
                continue
            if current not in _PRECISION_LADDER:
                blocked += 1
                continue
            idx = _PRECISION_LADDER.index(current)
            if idx >= len(_PRECISION_LADDER) - 1:
                blocked += 1
                continue
            nxt = _PRECISION_LADDER[idx + 1]
            if _PRECISION_PPL_PENALTY.get(nxt, 0.0) > self.max_ppl_delta:
                blocked += 1
                continue
            state.active_map[layer_id] = nxt
            changed[layer_id] = (current, nxt)

        if not changed:
            return {
                "action": "none",
                "model_id": state.model_id,
                "reason": "no_downgrade_candidates",
            }

        state.downgrades += len(changed)
        action = "downgrade" if blocked == 0 else "downgrade_selective"
        logger.info(
            "Precision downgrade",
            model_id=state.model_id,
            layers_changed=len(changed),
            layers_held=blocked,
            pressure=round(state.memory_pressure, 3),
        )
        return {
            "action": action,
            "model_id": state.model_id,
            "changed": {k: {"from": v[0], "to": v[1]} for k, v in changed.items()},
            "layers_changed": len(changed),
            "layers_held": blocked,
            "estimated_memory_ratio": round(state.estimated_memory_ratio(), 4),
        }

    def _upgrade(self, state: ModelPrecisionState) -> dict[str, Any]:
        """Step degraded layers one rung back toward compiled precision."""
        changed: dict[str, tuple[str, str]] = {}

        for layer_id in sorted(state.active_map):
            current = state.active_map[layer_id]
            target = state.target_map.get(layer_id, current)
            if current == target:
                continue
            if current not in _PRECISION_LADDER or target not in _PRECISION_LADDER:
                continue
            cur_idx = _PRECISION_LADDER.index(current)
            tgt_idx = _PRECISION_LADDER.index(target)
            if cur_idx <= tgt_idx:
                continue
            nxt = _PRECISION_LADDER[cur_idx - 1]
            state.active_map[layer_id] = nxt
            changed[layer_id] = (current, nxt)

        if not changed:
            return {
                "action": "none",
                "model_id": state.model_id,
                "reason": "already_at_target",
            }

        state.upgrades += len(changed)
        logger.info(
            "Precision upgrade",
            model_id=state.model_id,
            layers_changed=len(changed),
            pressure=round(state.memory_pressure, 3),
        )
        return {
            "action": "upgrade",
            "model_id": state.model_id,
            "changed": {k: {"from": v[0], "to": v[1]} for k, v in changed.items()},
            "layers_changed": len(changed),
            "estimated_memory_ratio": round(state.estimated_memory_ratio(), 4),
        }

    def reset(
        self, model_id: str, precision_map: dict[str, str] | None = None
    ) -> ModelPrecisionState:
        """
        Reset a model to its compiled precision, or to a new map.

        Passing ``precision_map`` replaces both the target and active maps —
        use this after a hot-reload changes the compiled precision.
        """
        with self._lock:
            state = self._states.get(model_id)
            if state is None:
                if precision_map is None:
                    raise KeyError(f"Model {model_id!r} is not registered")
                return self.register(model_id, precision_map)

            if precision_map is not None:
                state.target_map = dict(precision_map)
            state.active_map = dict(state.target_map)
            state.last_adjust_at = 0.0
            logger.info("Precision manager reset model", model_id=model_id)
            return state

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "models": len(self._states),
                "cooldown_seconds": self.cooldown_seconds,
                "downgrade_threshold": self.downgrade_threshold,
                "upgrade_threshold": self.upgrade_threshold,
                "degraded_models": sum(
                    1 for s in self._states.values() if s.is_degraded
                ),
                "states": {mid: s.to_dict() for mid, s in self._states.items()},
            }

    def __repr__(self) -> str:
        return (
            f"PrecisionManager(models={len(self._states)}, "
            f"cooldown={self.cooldown_seconds}s)"
        )
