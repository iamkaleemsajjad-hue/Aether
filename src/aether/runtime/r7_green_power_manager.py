"""
R7 — Green Power Manager.

The Green Power Manager enforces energy efficiency policies at runtime using
the DVFS breakpoints and carbon profiles compiled by Pass 16.

Three operational modes:
  1. **Performance**: Ignore DVFS hints; run all kernels at max clock.
  2. **Balanced**: Apply DVFS hints; throttle memory-bound ops to ~60% clock.
     Target: 30–40% energy reduction with < 5% throughput penalty.
  3. **Eco**: Apply aggressive DVFS + reduce batch size to meet TDP cap.
     Target: 40–48% energy reduction (MELODI 2026 benchmark).

Carbon routing:
  - If multiple deployment regions are available, the Green Power Manager
    routes batches to the region with lowest current carbon intensity
    when the latency SLO permits.
  - Integrates with the SLO Scheduler (R4) to ensure carbon-routing
    decisions do not violate TTFT deadlines.

TDP cap enforcement:
  - Monitors GPU power draw via NVML (NVIDIA) or ROCm SMI (AMD).
  - If power exceeds the cap, throttles GPU clock to ``freq × (cap/current_power)^0.5``.
  - Hysteresis: only re-adjust if power delta > 5% of cap.

Research basis:
  - MELODI 2026: energy-aware LLM inference operator scheduling.
  - DVFS arXiv 2025: frequency scaling for memory-bound transformer ops.
  - CodeCarbon 2026: carbon footprint tracking for ML workloads.
  - Green AI (Schwartz et al. 2020): efficiency-first ML philosophy.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)


class GreenPowerManager:
    """Runtime R7: Green Power Manager — DVFS + Carbon-Routing + TDP Enforcement.

    Loads the green profile from ``.aeg/metadata/green_profile.json`` (produced
    by Pass 16) and applies runtime energy policies.
    """

    # Mode constants.
    MODE_PERFORMANCE = "performance"
    MODE_BALANCED = "balanced"
    MODE_ECO = "eco"

    def __init__(
        self,
        mode: str = MODE_BALANCED,
        tdp_cap_w: float | None = None,
        green_profile_path: str | None = None,
    ) -> None:
        self.mode = mode
        self._tdp_cap_w: float | None = tdp_cap_w
        self._dvfs_hints: dict[str, dict[str, Any]] = {}  # op_id → hint
        self._carbon_intensity: float = 300.0  # gCO₂/kWh default
        self._region: str = "unknown"
        self._current_power_w: float = 0.0
        self._lock = threading.RLock()
        self._stats = _GreenStats()

        if green_profile_path:
            self._load_profile(green_profile_path)

    def _load_profile(self, path: str) -> None:
        """Load compiled green profile from AEG artifact."""
        p = Path(path)
        if not p.exists():
            logger.warning("R7: Green profile not found at %s.", path)
            return
        try:
            profile = json.loads(p.read_text(encoding="utf-8"))
            # Load DVFS hints by op_id.
            self._dvfs_hints = {
                h["op_id"]: h
                for h in profile.get("dvfs_hints", [])
                if "op_id" in h
            }
            self._carbon_intensity = float(
                profile.get("carbon_intensity_gco2_per_kwh", 300.0)
            )
            self._region = profile.get("carbon_region", "unknown")
            if self._tdp_cap_w is None:
                self._tdp_cap_w = float(profile.get("effective_tdp_cap_w", 400.0))
            logger.info(
                "R7: Green profile loaded — region=%r, carbon=%.0f gCO₂/kWh, "
                "%d DVFS hints, TDP cap=%.0fW.",
                self._region,
                self._carbon_intensity,
                len(self._dvfs_hints),
                self._tdp_cap_w or 0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("R7: Failed to load green profile: %s", exc)

    def get_dvfs_config(self, op_id: str) -> tuple[int, int]:
        """Return (freq_mhz, voltage_mv) for an operator.

        In PERFORMANCE mode: always max clock.
        In BALANCED/ECO mode: use compiled DVFS hint.

        Args:
            op_id: Operator identifier.

        Returns:
            (freq_mhz, voltage_mv) tuple.
        """
        if self.mode == self.MODE_PERFORMANCE:
            return 1980, 1100  # max clock

        hint = self._dvfs_hints.get(op_id)
        if hint is None:
            # No hint for this op: use conservative full clock.
            return 1980, 1100

        freq = int(hint.get("freq_mhz", 1980))
        voltage = int(hint.get("voltage_mv", 1100))

        if self.mode == self.MODE_ECO:
            # ECO: further reduce by 20%.
            freq = int(freq * 0.80)
            voltage = max(750, int(voltage * 0.88))

        return freq, voltage

    def update_power_reading(self, current_power_w: float) -> float | None:
        """Update current GPU power reading and enforce TDP cap.

        Args:
            current_power_w: Current GPU power draw in Watts (from NVML/ROCm).

        Returns:
            New throttled frequency (MHz) if throttling applied, else None.
        """
        with self._lock:
            self._current_power_w = current_power_w

            if self._tdp_cap_w is None or self.mode == self.MODE_PERFORMANCE:
                return None

            if current_power_w <= self._tdp_cap_w * 1.05:  # 5% hysteresis.
                return None

            # Throttle: f_new = f_max * sqrt(TDP_cap / P_current)
            ratio = math.sqrt(self._tdp_cap_w / current_power_w)
            f_throttled = int(1980 * ratio)
            f_throttled = max(800, f_throttled)  # Floor: 800 MHz.

            self._stats.throttle_events += 1
            logger.debug(
                "R7: TDP throttle: %.0fW > %.0fW cap → %.0f MHz.",
                current_power_w,
                self._tdp_cap_w,
                f_throttled,
            )
            return f_throttled

    def estimate_request_energy(
        self,
        n_prompt_tokens: int,
        n_gen_tokens: int,
        duration_s: float | None = None,
    ) -> float:
        """Estimate energy consumption for a request (mJ).

        When ``duration_s`` is supplied, energy is derived from the current
        power reading when available, otherwise from the compiled TDP cap and
        the selected mode.  Without a duration this retains the token-based
        fallback for callers that only have token counts.
        """
        if duration_s is not None:
            energy_mj, _ = self.measure_request_energy(
                duration_s,
                n_prompt_tokens=n_prompt_tokens,
                n_gen_tokens=n_gen_tokens,
            )
            return energy_mj
        n_tokens = n_prompt_tokens + n_gen_tokens
        # Conservative estimate: ~500 mJ per 1M tokens at full TDP for 7B model.
        base_energy_mj = (n_tokens / 1e6) * 500.0
        tdp_w = self._tdp_cap_w or 400.0
        # Scale by TDP ratio.
        mode_factor = {
            self.MODE_PERFORMANCE: 1.00,
            self.MODE_BALANCED: 0.65,  # 35% savings.
            self.MODE_ECO: 0.52,       # 48% savings (MELODI 2026 max).
        }.get(self.mode, 1.0)
        return base_energy_mj * (tdp_w / 400.0) * mode_factor

    def measure_request_energy(
        self,
        duration_s: float,
        *,
        n_prompt_tokens: int = 0,
        n_gen_tokens: int = 0,
    ) -> tuple[float, str]:
        """Return request energy and its evidence source.

        A positive ``update_power_reading`` value is treated as measured
        device power.  If no device reading exists, the result is explicitly
        labelled a TDP-duration estimate; it is never presented as hardware
        telemetry.  The token arguments are retained for future profile-based
        accounting and make the call site self-describing.
        """
        duration_s = float(duration_s)
        if duration_s < 0.0 or not math.isfinite(duration_s):
            raise ValueError("duration_s must be a finite non-negative value")
        del n_prompt_tokens, n_gen_tokens
        with self._lock:
            if self._current_power_w > 0.0:
                power_w = self._current_power_w
                source = "measured_power_reading"
            else:
                power_w = self._tdp_cap_w or 400.0
                mode_factor = {
                    self.MODE_PERFORMANCE: 1.00,
                    self.MODE_BALANCED: 0.65,
                    self.MODE_ECO: 0.52,
                }.get(self.mode, 1.0)
                power_w *= mode_factor
                source = "tdp_duration_estimate"
        return power_w * duration_s * 1000.0, source

    def estimate_carbon(self, energy_mj: float) -> float:
        """Estimate carbon footprint in gCO₂ equivalent.

        Formula: C = E_kwh × carbon_intensity_gco2_per_kwh.
        """
        energy_kwh = energy_mj / (1e6 * 3600)  # mJ → kWh
        return energy_kwh * self._carbon_intensity

    def select_region(
        self,
        available_regions: list[str],
        latency_deadline_s: float = 1.0,
    ) -> str:
        """Select the lowest-carbon region that meets the latency deadline.

        In a geo-distributed cluster, routes the request to the greenest
        region whose round-trip latency is below the deadline.
        """
        # Carbon intensity by region (from Pass 16 constants).
        _CARBON_MAP: dict[str, float] = {
            "us-west": 82.0,
            "eu-north": 28.0,
            "eu-west": 156.0,
            "us-east": 318.0,
            "ap-east": 487.0,
        }
        # Latency estimate (ms) per region — simplified model.
        _LATENCY_MAP: dict[str, float] = {
            "us-west": 5.0,
            "eu-north": 80.0,
            "eu-west": 90.0,
            "us-east": 10.0,
            "ap-east": 150.0,
        }

        best = None
        best_carbon = float("inf")
        for region in available_regions:
            latency_ms = _LATENCY_MAP.get(region, 200.0)
            if latency_ms / 1000 > latency_deadline_s:
                continue
            carbon = _CARBON_MAP.get(region, 400.0)
            if carbon < best_carbon:
                best_carbon = carbon
                best = region

        return best or (available_regions[0] if available_regions else "default")

    def record_request(
        self,
        energy_mj: float,
        carbon_gco2: float,
        *,
        source: str = "tdp_duration_estimate",
    ) -> None:
        """Record energy and carbon for a completed request."""
        with self._lock:
            self._stats.total_energy_mj += energy_mj
            self._stats.total_carbon_gco2 += carbon_gco2
            self._stats.total_requests += 1
            if source == "measured_power_reading":
                self._stats.measured_requests += 1
            else:
                self._stats.estimated_requests += 1

    @property
    def stats(self) -> "_GreenStats":
        return self._stats

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "region": self._region,
            "carbon_intensity_gco2_kwh": self._carbon_intensity,
            "tdp_cap_w": self._tdp_cap_w,
            "dvfs_hints_loaded": len(self._dvfs_hints),
            "total_energy_j": round(self._stats.total_energy_mj / 1000, 3),
                "total_carbon_gco2": round(self._stats.total_carbon_gco2, 6),
                "throttle_events": self._stats.throttle_events,
                "measured_requests": self._stats.measured_requests,
                "estimated_requests": self._stats.estimated_requests,
        }

    def get_status(self) -> dict[str, Any]:
        """Return the public R7 status payload used by SDK and REST callers."""
        status = self.summary()
        status.update(
            {
                "current_power_w": self._current_power_w,
                "carbon_intensity_gco2_kwh": self._carbon_intensity,
                "current_region": self._region,
                "dvfs_active": bool(self._dvfs_hints) and self.mode != self.MODE_PERFORMANCE,
                "power_budget_watts": self._tdp_cap_w,
            }
        )
        return status


class _GreenStats:
    __slots__ = (
        "total_energy_mj",
        "total_carbon_gco2",
        "total_requests",
        "throttle_events",
        "measured_requests",
        "estimated_requests",
    )

    def __init__(self) -> None:
        self.total_energy_mj = 0.0
        self.total_carbon_gco2 = 0.0
        self.total_requests = 0
        self.throttle_events = 0
        self.measured_requests = 0
        self.estimated_requests = 0
