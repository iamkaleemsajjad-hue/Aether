"""
llama.cpp backend — GGUF inference via llama-cpp-python or llama-server.

This backend provides two execution paths:
  1. **In-process** (preferred): via the ``llama-cpp-python`` pip package,
     which bundles prebuilt libllama shared libraries.
  2. **Out-of-process**: spawns a ``llama-server`` or ``llama-cli`` subprocess
     when llama-cpp-python is unavailable, communicating via stdin/stdout or
     HTTP REST (llama-server's OpenAI-compatible endpoint).

Both paths support GGUF models with K-quant precision (Q2_K … Q8_0, F16, BF16).
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaCppBackend(Backend):
    """
    llama.cpp backend supporting in-process (llama-cpp-python) and
    out-of-process (llama-server HTTP) execution modes.
    """

    def __init__(self) -> None:
        info = BackendInfo(
            name="llamacpp",
            version="2.0.0",
            supported_targets=[
                "cpu_avx512", "cpu_neon", "cpu_avx2",
                "cuda_sm70", "cuda_sm80", "cuda_sm89", "cuda_sm90",
                "metal_m1", "metal_m3", "rocm_rdna3",
            ],
            capabilities=["generate", "chat", "embed", "gguf", "cpu_offload", "k_quant"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}
        self._server_urls: dict[str, str] = {}   # model_id → server base URL
        self._server_procs: dict[str, subprocess.Popen] = {}

    # ------------------------------------------------------------------
    # Availability detection
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if llama-cpp-python is installed or llama-server is on PATH."""
        if self._has_llama_cpp_python():
            return True
        return self._find_llama_binary() is not None

    def _has_llama_cpp_python(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _find_llama_binary(self) -> str | None:
        """Locate ``llama-server`` or ``llama-cli`` on PATH."""
        import shutil

        for name in ("llama-server", "llama-cli", "llama"):
            path = shutil.which(name)
            if path:
                return path
        return None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """
        Load a GGUF model.

        Preference order:
        1. llama-cpp-python in-process (fastest, no latency)
        2. llama-server subprocess (REST API mode)

        Args:
            model_id: Path to a .gguf file, or HuggingFace model ID.
            aeg_path: Optional AEG package path (not used for GGUF; ignored).
            kwargs: n_ctx, n_threads, n_gpu_layers, server_port, etc.
        """
        if model_id in self._models or model_id in self._server_urls:
            return self._models.get(model_id, model_id)

        if self._has_llama_cpp_python():
            return self._load_inprocess(model_id, **kwargs)
        return self._load_subprocess(model_id, **kwargs)

    def _load_inprocess(self, model_id: str, **kwargs: Any) -> Any:
        """Load via llama-cpp-python (in-process)."""
        from llama_cpp import Llama

        n_ctx = kwargs.get("n_ctx", 4096)
        n_threads = kwargs.get("n_threads", None)
        n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        flash_attn = kwargs.get("flash_attn", True)
        model = Llama(
            model_path=model_id,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            flash_attn=flash_attn,
            verbose=False,
        )
        self._models[model_id] = model
        logger.info("llama.cpp: loaded in-process model %s", model_id)
        return model

    def _load_subprocess(self, model_id: str, **kwargs: Any) -> str:
        """
        Launch llama-server as a subprocess and wait for it to be ready.

        Returns the server base URL (e.g., "http://127.0.0.1:8380").
        """
        binary = self._find_llama_binary()
        if binary is None:
            msg = "Neither llama-cpp-python nor llama-server is installed"
            raise ImportError(msg)

        port = kwargs.get("server_port", 8380)
        host = "127.0.0.1"
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        n_threads = kwargs.get("n_threads", 4)

        cmd = [
            binary,
            "--model", model_id,
            "--port", str(port),
            "--host", host,
            "--ctx-size", str(n_ctx),
            "--n-gpu-layers", str(n_gpu_layers),
            "--threads", str(n_threads),
            "--no-mmap",
        ]

        logger.info("llama.cpp: starting server process: %s", " ".join(cmd))
        proc = subprocess.Popen(  # nosec B603 B607
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._server_procs[model_id] = proc

        # Poll until server is ready (max 30s)
        base_url = f"http://{host}:{port}"
        for _ in range(60):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(f"{base_url}/health", timeout=1)
                logger.info("llama.cpp server ready at %s", base_url)
                self._server_urls[model_id] = base_url
                return model_id
            except Exception:
                pass

        logger.warning("llama.cpp server did not become ready in time; may fail")
        self._server_urls[model_id] = base_url
        return model_id

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using llama.cpp."""
        # In-process path
        model = self._models.get(request.model_id)
        if model is not None:
            return self._generate_inprocess(model, request)

        # Server path
        server_url = self._server_urls.get(request.model_id)
        if server_url is not None:
            return self._generate_server(server_url, request)

        # Try to auto-load
        self.load_model(request.model_id)
        model = self._models.get(request.model_id)
        if model is not None:
            return self._generate_inprocess(model, request)
        server_url = self._server_urls.get(request.model_id)
        if server_url is not None:
            return self._generate_server(server_url, request)

        msg = f"Model {request.model_id} could not be loaded by llamacpp backend"
        raise RuntimeError(msg)

    def _generate_inprocess(self, model: Any, request: GenerationRequest) -> GenerationResult:
        """Generate via llama-cpp-python in-process API."""
        text = self._build_prompt(request)
        start = time.perf_counter()
        output = model(
            text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else 40,
            stop=request.stop or [],
            echo=False,
        )
        elapsed = time.perf_counter() - start
        choice = output["choices"][0]
        usage = output.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(choice["text"].split()))
        return GenerationResult(
            text=choice["text"],
            prompt_tokens=usage.get("prompt_tokens", len(text.split())),
            completion_tokens=completion_tokens,
            finish_reason=choice.get("finish_reason", "stop"),
            backend_name=self.name,
            metrics={
                "ttft_ms": elapsed * 1000,
                "throughput_tps": completion_tokens / max(elapsed, 1e-6),
                "device": "llama_cpp",
            },
        )

    def _generate_server(self, base_url: str, request: GenerationRequest) -> GenerationResult:
        """Generate via llama-server OpenAI-compatible REST API."""
        text = self._build_prompt(request)
        payload = {
            "prompt": text,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k if request.top_k > 0 else 40,
            "stop": request.stop or [],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            msg = f"llama-server request failed: {exc}"
            raise RuntimeError(msg) from exc
        elapsed = time.perf_counter() - start
        gen_text = result.get("content", "")
        completion_tokens = result.get("tokens_predicted", len(gen_text.split()))
        prompt_tokens = result.get("tokens_evaluated", len(text.split()))
        return GenerationResult(
            text=gen_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=result.get("stop_type", "stop"),
            backend_name=self.name,
            metrics={
                "ttft_ms": elapsed * 1000,
                "throughput_tps": completion_tokens / max(elapsed, 1e-6),
                "device": "llama_server",
            },
        )

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Format prompt text from a GenerationRequest."""
        if request.messages:
            parts: list[str] = []
            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
            parts.append("<|assistant|>")
            return "\n".join(parts)
        return request.prompt or ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate any spawned llama-server subprocesses."""
        for model_id, proc in list(self._server_procs.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            logger.info("llama.cpp: terminated server for %s", model_id)
        self._server_procs.clear()
        self._server_urls.clear()

    def get_capabilities(self) -> list[str]:
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"LlamaCppBackend(models={len(self._models)}, servers={len(self._server_urls)})"
