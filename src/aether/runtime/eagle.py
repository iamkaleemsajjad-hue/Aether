"""EAGLE-3 speculative decoding planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EAGLE3Plan:
    """Runtime plan for EAGLE-3 multi-layer feature fusion."""

    draft_model_id: str | None
    fusion_layers: tuple[int, ...]
    tree_depth: int
    branching_factor: int
    attention_drift_correction: bool = True
    flattened_tree: bool = True
    acceptance_floor: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "eagle3/1.0",
            "draft_model_id": self.draft_model_id,
            "fusion_layers": list(self.fusion_layers),
            "tree_depth": self.tree_depth,
            "branching_factor": self.branching_factor,
            "attention_drift_correction": self.attention_drift_correction,
            "flattened_tree": self.flattened_tree,
            "acceptance_floor": self.acceptance_floor,
            "fallback": "standard_decode_on_low_acceptance",
        }


class EAGLE3Planner:
    """Choose EAGLE-3 fusion layers and draft tree shape."""

    def plan(self, architecture: Any, draft_model_id: str | None = None, target_acceptance: float = 0.8) -> EAGLE3Plan:
        layers = max(1, int(getattr(architecture, "layers", 1)))
        if layers <= 8:
            fusion_layers = tuple(range(layers))
        else:
            step = max(1, layers // 8)
            fusion_layers = tuple(sorted(set([0, layers - 1, *range(step - 1, layers, step)])))[:10]
        context = int(getattr(architecture, "context_length", 32768))
        tree_depth = 5 if context >= 65536 else 4
        branching_factor = 4 if target_acceptance >= 0.8 else 3
        return EAGLE3Plan(
            draft_model_id=draft_model_id,
            fusion_layers=fusion_layers,
            tree_depth=tree_depth,
            branching_factor=branching_factor,
            attention_drift_correction=context >= 65536,
            flattened_tree=True,
            acceptance_floor=max(0.5, min(0.95, target_acceptance - 0.05)),
        )

    def verify_acceptance(self, accepted_tokens: int, proposed_tokens: int, plan: EAGLE3Plan) -> dict[str, Any]:
        rate = accepted_tokens / max(1, proposed_tokens)
        return {
            "accepted_tokens": accepted_tokens,
            "proposed_tokens": proposed_tokens,
            "acceptance_rate": rate,
            "use_speculation": rate >= plan.acceptance_floor,
            "fallback": None if rate >= plan.acceptance_floor else "standard_decode",
        }
