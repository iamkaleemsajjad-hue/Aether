"""
R5 — TTT Fast-Weight Engine.

Test-Time Training (TTT) enables models to adapt domain-specifically during
inference.  The TTT Engine executes online gradient steps on fast-weight
parameters (injected by Pass 13) without storing optimizer state across requests.

Execution model:
  1. Before generating the first token, run one forward pass on the input
     to compute hidden states h.
  2. Compute self-supervised loss L = ||LayerNorm(h; µ, σ) - h_target||².
  3. Compute gradient ∇L w.r.t. fast-weight parameters (µ, σ, A, B).
  4. Apply gradient step: µ ← µ - η∇µL, σ ← σ - η∇σL.
  5. Continue with generation using the updated fast weights.

This process adapts the model to the specific domain/style of the current
input without modifying the frozen base weights.

Memory model:
  - Fast weights are stored in a separate TTT parameter buffer (loaded from
    the ``.aeg/ttt/`` artifacts written by Pass 13).
  - Fast weights are request-scoped: reset to base values after each request
    (stateless) unless session-scoped TTT is enabled.

Research basis:
  - In-Place TTT (arXiv 2026): LoRA-style fast weights, single gradient step.
  - VDS-TTT (NeurIPS 2026): video-domain TTT with streaming updates.
  - SDFT 2026: sparse dynamic fine-tuning.
  - Sun et al. 2024: original TTT framework.
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


class TTTFastWeightEngine:
    """Runtime R5: TTT Fast-Weight Online Adaptation Engine.

    Manages fast-weight parameter buffers per layer and executes in-place
    gradient steps during inference.  Thread-safe via per-request slot locking.
    """

    def __init__(
        self,
        n_layers: int = 32,
        hidden_size: int = 4096,
        rank: int = 16,
        learning_rate: float = 1e-4,
        session_scoped: bool = False,
        ttt_config_path: str | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rank = rank
        self.learning_rate = learning_rate
        self.session_scoped = session_scoped

        # Fast weight buffers: per-layer (A, B, mu, sigma) initialized to zeros/identity.
        self._base_weights: list[dict[str, list[float]]] = [
            self._init_slot(hidden_size, rank) for _ in range(n_layers)
        ]
        # Per-request active weights (copy of base, modified per request).
        self._active_weights: dict[str, list[dict]] = {}
        self._lock = threading.RLock()
        self._stats = _TTTStats()

        if ttt_config_path:
            self._load_config(ttt_config_path)

    def _init_slot(self, hidden_size: int, rank: int) -> dict[str, list[float]]:
        """Initialize a fast-weight slot with zero A, zero B, unit LayerNorm."""
        return {
            "A": [0.0] * (hidden_size * rank),       # LoRA A matrix (flattened)
            "B": [0.0] * (rank * hidden_size),        # LoRA B matrix (flattened)
            "mu": [0.0] * hidden_size,                 # LayerNorm shift
            "sigma": [1.0] * hidden_size,              # LayerNorm scale (init to 1)
            "momentum_A": [0.0] * (hidden_size * rank),
            "momentum_B": [0.0] * (rank * hidden_size),
        }

    def _load_config(self, config_path: str) -> None:
        """Load TTT configuration from AEG artifact."""
        p = Path(config_path)
        if not p.exists():
            return
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            self.n_layers = int(cfg.get("n_layers", self.n_layers))
            self.hidden_size = int(cfg.get("hidden_size", self.hidden_size))
            self.rank = int(cfg.get("rank", self.rank))
            self.learning_rate = float(cfg.get("learning_rate", self.learning_rate))
            # Reinitialize slots if config changed dimensions.
            self._base_weights = [
                self._init_slot(self.hidden_size, self.rank) for _ in range(self.n_layers)
            ]
            logger.info(
                "R5: TTT config loaded: %d layers, hidden=%d, rank=%d, lr=%.2e.",
                self.n_layers,
                self.hidden_size,
                self.rank,
                self.learning_rate,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("R5: Failed to load TTT config: %s", exc)

    def begin_request(self, request_id: str) -> None:
        """Initialize fast weights for a new request.

        If session_scoped=False: copy base weights (stateless per-request adaptation).
        If session_scoped=True: maintain weights across requests in the same session.
        """
        with self._lock:
            if not self.session_scoped or request_id not in self._active_weights:
                self._active_weights[request_id] = [
                    {k: list(v) for k, v in slot.items()}
                    for slot in self._base_weights
                ]

    def adapt(
        self,
        request_id: str,
        hidden_states: list[list[float]],
        layer_idx: int = -1,
    ) -> float:
        """Run one online gradient step on the fast weights for this request.

        Algorithm (In-Place TTT, arXiv 2026):
          1. Compute reconstruction loss: L = ||A @ B @ h - h||² / hidden_size
          2. Gradient: ∇_A L = (2/N) (AB h - h) h^T B^T
                       ∇_B L = (2/N) A^T (AB h - h) h^T
          3. Update: A ← A - lr * ∇_A L
                     B ← B - lr * ∇_B L
          4. Update LayerNorm: mu ← mean(h), sigma ← std(h)

        Args:
            request_id: The request to adapt.
            hidden_states: List of hidden state vectors (one per token).
            layer_idx: Which layer to adapt (-1 = all layers).

        Returns:
            Reconstruction loss (float) for monitoring.
        """
        start = time.perf_counter()
        with self._lock:
            if request_id not in self._active_weights:
                self.begin_request(request_id)
            weights = self._active_weights[request_id]

        layers_to_adapt = range(self.n_layers) if layer_idx < 0 else [layer_idx]
        total_loss = 0.0

        for l_idx in layers_to_adapt:
            if l_idx >= len(weights):
                continue
            slot = weights[l_idx]
            loss = self._gradient_step(slot, hidden_states)
            total_loss += loss

        avg_loss = total_loss / max(1, len(list(layers_to_adapt)))
        self._stats.total_adapt_steps += 1
        self._stats.total_adapt_time_ms += (time.perf_counter() - start) * 1000

        logger.debug(
            "R5: adapt request=%r layers=%s loss=%.4f",
            request_id[:8],
            "all" if layer_idx < 0 else str(layer_idx),
            avg_loss,
        )
        return avg_loss

    def _gradient_step(
        self,
        slot: dict[str, list[float]],
        hidden_states: list[list[float]],
    ) -> float:
        """Compute and apply one gradient step on a single layer's fast weights.

        Simplified In-Place TTT:
          - h_mean = mean over tokens
          - loss = ||A @ B @ h_mean - h_mean||² / hidden_size
          - Update mu, sigma (LayerNorm parameters)
        """
        if not hidden_states:
            return 0.0

        H = self.hidden_size
        R = self.rank
        lr = self.learning_rate

        # Mean hidden state over all tokens.
        h_mean = [
            sum(hidden_states[t][j] if j < len(hidden_states[t]) else 0.0
                for t in range(len(hidden_states))) / len(hidden_states)
            for j in range(H)
        ]

        # Update LayerNorm mu and sigma.
        mean_h = sum(h_mean) / H
        var_h = sum((x - mean_h) ** 2 for x in h_mean) / H
        std_h = math.sqrt(var_h + 1e-8)

        slot["mu"] = [slot["mu"][j] - lr * (slot["mu"][j] - mean_h) for j in range(H)]
        slot["sigma"] = [slot["sigma"][j] - lr * (slot["sigma"][j] - std_h) for j in range(H)]

        # LoRA reconstruction loss: L = ||B @ A @ h - h||² / H
        A = slot["A"]  # flattened H × R
        B = slot["B"]  # flattened R × H

        # Compute AB @ h_mean (shape H).
        Ah = [
            sum(A[j * R + k] * h_mean[j] for j in range(H))
            for k in range(R)
        ]  # R-dim
        BAh = [
            sum(B[k * H + j] * Ah[k] for k in range(R))
            for j in range(H)
        ]  # H-dim

        # Residual: e = BAh - h_mean.
        residual = [BAh[j] - h_mean[j] for j in range(H)]
        loss = sum(r * r for r in residual) / H

        # Gradient step on A and B (simplified outer product).
        for j in range(H):
            for k in range(R):
                grad_A = 2.0 / H * residual[j] * Ah[k]
                slot["A"][j * R + k] -= lr * grad_A

        for k in range(R):
            for j in range(H):
                grad_B = 2.0 / H * Ah[k] * residual[j]
                slot["B"][k * H + j] -= lr * grad_B

        return loss

    def get_fast_weights(self, request_id: str, layer_idx: int) -> dict[str, list[float]] | None:
        """Return the current fast weights for a layer in a request."""
        with self._lock:
            weights = self._active_weights.get(request_id)
            if weights is None or layer_idx >= len(weights):
                return None
            return weights[layer_idx]

    def end_request(self, request_id: str) -> None:
        """Release fast weights for a completed request."""
        with self._lock:
            if not self.session_scoped:
                self._active_weights.pop(request_id, None)
            self._stats.total_requests += 1

    @property
    def stats(self) -> "_TTTStats":
        return self._stats


class _TTTStats:
    __slots__ = ("total_adapt_steps", "total_adapt_time_ms", "total_requests")

    def __init__(self) -> None:
        self.total_adapt_steps = 0
        self.total_adapt_time_ms = 0.0
        self.total_requests = 0

    @property
    def avg_adapt_time_ms(self) -> float:
        return self.total_adapt_time_ms / max(1, self.total_adapt_steps)
