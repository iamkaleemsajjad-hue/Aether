"""
Native CUDA execution backend for compiled AEG artifacts.

This backend uses the JIT-compiled CUDA kernels from
:mod:`aether.kernels.native_cuda` to run a transformer forward pass
entirely via the CUDA Driver API, without requiring PyTorch.

Design
------
* **No PyTorch required.** All tensor math is performed by Aether's own
  CUDA kernels, loaded at runtime via ctypes.  PyTorch is never imported.
* **GPU memory managed by Aether.** Weights are uploaded to device memory
  using ``cudaMalloc``/``cudaMemcpy`` called through ctypes.  The runtime
  owns these allocations for the lifetime of the engine.
* **CPU fallback.** If no CUDA toolkit is detected (no nvcc, or the GPU is
  absent), the backend refuses to load, and the registry falls back to the
  next available backend (NativeCPU, TorchBackend, etc.).

Call graph
----------
    NativeCUDABackend.generate()
        → AetherCUDAEngine.generate_with_cache()
            → AetherCUDAEngine.forward()
                → NativeCUDAKernels.rmsnorm()
                → NativeCUDAKernels.sgemm()
                → NativeCUDAKernels.rope()
                → NativeCUDAKernels.flash_attn()
                → NativeCUDAKernels.swiglu()
                → NativeCUDAKernels.sgemm() (LM head)

GPU validation
--------------
This module is written for future GPU execution.  Tests marked
``@pytest.mark.gpu`` in ``tests/gpu/`` verify correct output on CUDA
hardware once available.  On CPU-only machines the module imports cleanly
but ``is_available()`` returns False and ``load_model()`` raises
``BackendError``.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.exceptions import BackendError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AetherCUDAEngine
# ---------------------------------------------------------------------------

class AetherCUDAEngine:
    """
    Native CUDA transformer engine.

    Loads weight tensors to GPU via the CUDA Driver API (ctypes), then runs
    a decoder-only transformer forward pass using Aether's own CUDA kernels.
    No PyTorch is imported.

    This class intentionally mirrors the interface of ``CPUExecutionEngine``
    so that ``DraftTargetSpecDecoder``, ``ContinuousBatcher``, and any other
    runtime subsystem that uses the ``ForwardStepEngine`` protocol can drive
    it without modification.
    """

    def __init__(
        self,
        weights: Any,             # ModelWeights-like object (same as cpu_engine)
        cuda_kernels: Any,        # NativeCUDAKernels instance
        device_id: int = 0,
    ) -> None:
        self.weights = weights
        self._kernels = cuda_kernels
        self.device_id = device_id
        self._gpu_weights: dict[str, Any] = {}  # name → GPU buffer handle
        self._weights_on_device = False

        # Upload weights to GPU
        self._upload_weights()

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _upload_weights(self) -> None:
        """Copy every weight tensor to the GPU memory pool."""
        if not hasattr(self._kernels, "upload_tensor"):
            # NativeCUDAKernels not yet compiled — skip upload, will fail on forward
            logger.warning(
                "NativeCUDAKernels.upload_tensor not available; "
                "GPU weight upload deferred until first forward pass"
            )
            return
        try:
            # Upload embedding
            self._gpu_weights["embedding"] = self._kernels.upload_tensor(
                self.weights.embedding.astype(np.float16)
            )
            # Upload per-layer weights
            for layer_idx, layer in enumerate(self.weights.layers):
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj",
                             "input_layernorm", "post_attention_layernorm"):
                    w = getattr(layer, attr, None)
                    if w is not None:
                        key = f"layer_{layer_idx}_{attr}"
                        self._gpu_weights[key] = self._kernels.upload_tensor(
                            np.asarray(w, dtype=np.float16)
                        )
            # Upload LM head and final norm
            if self.weights.lm_head is not None:
                self._gpu_weights["lm_head"] = self._kernels.upload_tensor(
                    self.weights.lm_head.astype(np.float16)
                )
            if self.weights.norm_weight is not None:
                self._gpu_weights["final_norm"] = self._kernels.upload_tensor(
                    self.weights.norm_weight.astype(np.float32)
                )
            self._weights_on_device = True
            logger.info(
                "AetherCUDAEngine: uploaded %d weight tensors to GPU %d",
                len(self._gpu_weights), self.device_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU weight upload failed: %s; forward() will fail", exc)

    # ------------------------------------------------------------------
    # ForwardStepEngine protocol
    # ------------------------------------------------------------------

    def forward_step(
        self,
        token_ids: np.ndarray,
        kv_cache: Any = None,
        return_all_logits: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """
        Single-step forward pass.

        Satisfies the ``ForwardStepEngine`` protocol used by
        ``DraftTargetSpecDecoder`` and ``ContinuousBatcher``.

        Args:
            token_ids: 1-D or 2-D int64 array of token IDs.
            kv_cache: Existing KV cache object, or None to start fresh.
            return_all_logits: If True return (seq, vocab); else (vocab,).

        Returns:
            ``(logits_numpy, updated_kv_cache)``
        """
        logits, new_cache = self.forward(token_ids, cache=kv_cache)
        if not return_all_logits:
            logits = logits[-1]  # last position for decode
        return logits, new_cache

    def forward(
        self,
        token_ids: np.ndarray,
        cache: Any = None,
    ) -> tuple[np.ndarray, Any]:
        """
        Full transformer forward pass on GPU.

        Raises ``BackendError`` when the CUDA toolkit is absent or GPU weights
        were not uploaded successfully.  The registry will not select this
        backend unless ``is_available()`` returns True.
        """
        if not self._weights_on_device:
            raise BackendError(
                "AetherCUDAEngine: GPU weights not uploaded. "
                "Ensure CUDA toolkit is present and NativeCUDAKernels compiled.",
                backend_name="aether_cuda",
            )

        ids = np.ascontiguousarray(token_ids, dtype=np.int64).reshape(-1)
        seq_len = int(ids.size)

        # ── Embedding lookup ──────────────────────────────────────────────────
        # Performed on CPU (embedding table is small; GPU lookup overhead is
        # not worthwhile for seq_len=1 decode).  The embedding result is then
        # uploaded to the GPU as the initial hidden state.
        hidden_cpu = self.weights.embedding[ids].astype(np.float16)
        hidden_gpu = self._kernels.upload_tensor(hidden_cpu)

        # ── KV cache ─────────────────────────────────────────────────────────
        from aether.runtime.cpu_engine import KVCache  # reuse CPU cache structure
        cache = cache or KVCache(num_layers=self.weights.num_layers)

        # ── Transformer layers ────────────────────────────────────────────────
        for layer_idx, layer in enumerate(self.weights.layers):
            past = cache.length // max(len(self.weights.layers), 1)

            # Pre-attention RMSNorm (GPU)
            norm_w_gpu = self._gpu_weights.get(f"layer_{layer_idx}_input_layernorm")
            normed_gpu = self._kernels.rmsnorm(
                hidden_gpu, norm_w_gpu, seq_len,
                int(self.weights.hidden_size), float(self.weights.norm_eps),
            )

            # Q, K, V projections (GPU SGEMM)
            q_w = self._gpu_weights.get(f"layer_{layer_idx}_q_proj")
            k_w = self._gpu_weights.get(f"layer_{layer_idx}_k_proj")
            v_w = self._gpu_weights.get(f"layer_{layer_idx}_v_proj")
            q_gpu = self._kernels.sgemm(normed_gpu, q_w, seq_len,
                                         self.weights.hidden_size,
                                         self.weights.num_heads * self.weights.head_dim)
            k_gpu = self._kernels.sgemm(normed_gpu, k_w, seq_len,
                                         self.weights.hidden_size,
                                         self.weights.num_kv_heads * self.weights.head_dim)
            v_gpu = self._kernels.sgemm(normed_gpu, v_w, seq_len,
                                         self.weights.hidden_size,
                                         self.weights.num_kv_heads * self.weights.head_dim)

            # RoPE (GPU)
            if hasattr(self._kernels, "rope"):
                q_gpu, k_gpu = self._kernels.rope(q_gpu, k_gpu, past, seq_len,
                                                    self.weights.head_dim)

            # Download K, V to CPU for KV cache append (GPU paged attention pending)
            # TODO: replace with GPU-resident paged attention once NativeCUDAKernels
            # exposes paged_attn() — see docs/audit/KV_CACHE_AUDIT.md
            k_cpu = self._kernels.download_tensor(k_gpu, (seq_len, self.weights.num_kv_heads, self.weights.head_dim))
            v_cpu = self._kernels.download_tensor(v_gpu, (seq_len, self.weights.num_kv_heads, self.weights.head_dim))
            cache.append_kv(layer_idx, k_cpu, v_cpu)

            # Full K, V (past + present) — upload accumulated cache to GPU
            k_full = np.concatenate(
                [cache.keys[layer_idx][:-seq_len], k_cpu], axis=0
            ) if cache.length > seq_len else k_cpu
            v_full = np.concatenate(
                [cache.values[layer_idx][:-seq_len], v_cpu], axis=0
            ) if cache.length > seq_len else v_cpu
            k_full_gpu = self._kernels.upload_tensor(k_full.astype(np.float16))
            v_full_gpu = self._kernels.upload_tensor(v_full.astype(np.float16))

            # FlashAttention (GPU)
            attn_gpu = self._kernels.flash_attn(
                q_gpu, k_full_gpu, v_full_gpu,
                seq_len, cache.length, self.weights.num_heads,
                self.weights.num_kv_heads, self.weights.head_dim,
            )

            # Output projection (GPU)
            o_w = self._gpu_weights.get(f"layer_{layer_idx}_o_proj")
            attn_out_gpu = self._kernels.sgemm(attn_gpu, o_w, seq_len,
                                                self.weights.num_heads * self.weights.head_dim,
                                                self.weights.hidden_size)

            # Residual (GPU)
            hidden_gpu = self._kernels.add(hidden_gpu, attn_out_gpu, seq_len * self.weights.hidden_size)

            # Post-attention RMSNorm (GPU)
            post_norm_w_gpu = self._gpu_weights.get(f"layer_{layer_idx}_post_attention_layernorm")
            ffn_in_gpu = self._kernels.rmsnorm(
                hidden_gpu, post_norm_w_gpu, seq_len,
                int(self.weights.hidden_size), float(self.weights.norm_eps),
            )

            # SwiGLU FFN (GPU)
            gate_w = self._gpu_weights.get(f"layer_{layer_idx}_gate_proj")
            up_w   = self._gpu_weights.get(f"layer_{layer_idx}_up_proj")
            down_w = self._gpu_weights.get(f"layer_{layer_idx}_down_proj")
            intermediate = getattr(layer, "intermediate_size", None) or \
                           int(self.weights.hidden_size * 8 / 3)
            gate_gpu = self._kernels.sgemm(ffn_in_gpu, gate_w, seq_len,
                                            self.weights.hidden_size, intermediate)
            up_gpu   = self._kernels.sgemm(ffn_in_gpu, up_w, seq_len,
                                            self.weights.hidden_size, intermediate)
            swiglu_gpu = self._kernels.swiglu(gate_gpu, up_gpu, seq_len * intermediate)
            ffn_out_gpu = self._kernels.sgemm(swiglu_gpu, down_w, seq_len,
                                               intermediate, self.weights.hidden_size)

            # Residual (GPU)
            hidden_gpu = self._kernels.add(hidden_gpu, ffn_out_gpu, seq_len * self.weights.hidden_size)

        # ── Final RMSNorm (GPU) ───────────────────────────────────────────────
        final_norm_w_gpu = self._gpu_weights.get("final_norm")
        hidden_gpu = self._kernels.rmsnorm(
            hidden_gpu, final_norm_w_gpu, seq_len,
            int(self.weights.hidden_size), float(self.weights.norm_eps),
        )

        # ── LM head (GPU SGEMM) ───────────────────────────────────────────────
        lm_head_gpu = self._gpu_weights.get("lm_head")
        logits_gpu = self._kernels.sgemm(
            hidden_gpu, lm_head_gpu,
            seq_len, self.weights.hidden_size, self.weights.vocab_size,
        )

        # ── Download logits to CPU for sampling ───────────────────────────────
        logits_cpu = self._kernels.download_tensor(
            logits_gpu, (seq_len, self.weights.vocab_size)
        ).astype(np.float32)

        return logits_cpu, cache

    # ------------------------------------------------------------------
    # Generation loop (mirrors CPUExecutionEngine)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        generated, _ = self.generate_with_cache(
            prompt_ids, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
        )
        return generated

    def generate_with_cache(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 16,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        cache: Any = None,
        **kwargs: Any,
    ) -> tuple[list[int], Any]:
        """Autoregressive generation using GPU forward pass."""
        ids = np.ascontiguousarray(prompt_ids, dtype=np.int64).reshape(-1)
        eos_set: set[int] = set()
        if isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        elif eos_token_id is not None:
            eos_set = set(eos_token_id)

        generated: list[int] = []
        # Prefill
        logits, cache = self.forward(ids, cache=cache)
        next_token = int(self._sample(logits[-1], temperature, top_k, top_p))
        generated.append(next_token)

        # Decode
        while len(generated) < max_tokens:
            if eos_set and next_token in eos_set:
                break
            logits, cache = self.forward(np.array([next_token], dtype=np.int64), cache=cache)
            next_token = int(self._sample(logits[-1], temperature, top_k, top_p))
            generated.append(next_token)

        return generated, cache

    @staticmethod
    def _sample(
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        if temperature <= 1e-8:
            return int(np.argmax(logits))
        logits = logits.astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits / temperature)
        if top_k > 0:
            kth = np.partition(probs, -min(top_k, len(probs)))[-min(top_k, len(probs))]
            probs = np.where(probs >= kth, probs, 0.0)
        if top_p < 1.0:
            sorted_p = np.sort(probs)[::-1]
            cumsum = np.cumsum(sorted_p)
            cutoff = sorted_p[np.searchsorted(cumsum, top_p * probs.sum())]
            probs = np.where(probs >= cutoff, probs, 0.0)
        total = probs.sum()
        if total < 1e-30:
            return int(np.argmax(logits))
        probs /= total
        return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# NativeCUDABackend
# ---------------------------------------------------------------------------

class NativeCUDABackend(Backend):
    """
    Backend adapter for AetherCUDAEngine.

    Registered at position 2 in the backend registry (after NativeCPU,
    before ONNX, vLLM, TorchBackend).  Returns is_available() = False on
    machines without a CUDA toolkit, so the registry transparently falls
    through to the next backend.
    """

    name: str = "aether_cuda"

    def __init__(self) -> None:
        self._kernels: Any = None
        self._models: dict[str, AetherCUDAEngine] = {}
        self._available: bool = False
        self._init_kernels()

    def get_capabilities(self) -> list[str]:
        """Return backend capabilities."""
        caps = ["native_cuda", "jit_compiled", "pytorch_free"]
        if self._available:
            caps.append("cuda_available")
        return caps

    def _init_kernels(self) -> None:
        try:
            from aether.kernels.native_cuda import NativeCUDAKernels, detect_cuda_toolchain
            toolchain = detect_cuda_toolchain()
            if toolchain is None:
                logger.debug("NativeCUDABackend: no nvcc found; backend unavailable")
                return
            self._kernels = NativeCUDAKernels()
            self._kernels.compile()   # JIT-compile CUDA_KERNEL_SOURCE via nvcc
            self._available = self._kernels.is_available()
            if self._available:
                logger.info("NativeCUDABackend: CUDA kernels compiled and loaded")
            else:
                logger.debug("NativeCUDABackend: compiled but GPU not detected at runtime")
        except Exception as exc:  # noqa: BLE001
            logger.debug("NativeCUDABackend init failed: %s", exc)

    def is_available(self) -> bool:
        return self._available

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            version="1.3.0",
            description="Aether native CUDA backend (JIT-compiled, no PyTorch)",
            supports_streaming=False,
            supports_batching=False,
            requires_gpu=True,
        )

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        if not self._available:
            raise BackendError(
                "NativeCUDABackend: CUDA toolkit or GPU not available",
                backend_name=self.name,
            )
        aeg_path = kwargs.get("aeg_path")
        if aeg_path is None:
            raise BackendError("aeg_path required for NativeCUDABackend", backend_name=self.name)

        from aether.runtime.aeg_loader import load_engine_from_path
        # load_engine_from_path returns a CPUExecutionEngine; we reuse its weights
        cpu_engine = load_engine_from_path(aeg_path)
        self._models[model_id] = AetherCUDAEngine(
            weights=cpu_engine.weights,
            cuda_kernels=self._kernels,
            device_id=int(kwargs.get("device_id", 0)),
        )
        logger.info("NativeCUDABackend: loaded model %s on GPU", model_id)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.model_id not in self._models:
            self.load_model(request.model_id, aeg_path=request.extra.get("aeg_path"))
        engine = self._models[request.model_id]

        # Tokenise (re-use AEG packaged tokenizer via CPU backend handle)
        from aether.runtime.aeg_loader import load_engine_from_path
        aeg_path = request.extra.get("aeg_path", "")
        from aether.backends.native_cpu_backend import PackagedTokenizer
        tokenizer_path = Path(aeg_path) / "tokenizer" / "tokenizer.json"
        tokenizer = PackagedTokenizer(tokenizer_path)

        text = request.prompt or ""
        prompt_ids = np.asarray(
            tokenizer.encode(text).ids, dtype=np.int64
        )

        start = time.perf_counter()
        generated, _ = engine.generate_with_cache(
            prompt_ids,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - start
        generated_text = tokenizer.decode(generated)

        return GenerationResult(
            text=generated_text,
            prompt_tokens=int(prompt_ids.size),
            completion_tokens=len(generated),
            finish_reason="stop" if len(generated) < request.max_tokens else "length",
            backend_name=self.name,
            metrics={
                "ttft_ms": elapsed * 1000.0,
                "throughput_tps": len(generated) / max(elapsed, 1e-9),
                "device": f"cuda:{engine.device_id}",
                "framework_free": True,
                "native_cuda": True,
            },
        )
