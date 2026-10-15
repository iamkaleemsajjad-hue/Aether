"""llama.cpp backend — GGUF inference via llama-cpp-python or llama-server.

This backend provides two execution paths:
  1. **In-process** (preferred): via the ``llama-cpp-python`` pip package,
     which bundles prebuilt libllama shared libraries.
  2. **Out-of-process**: spawns a dedicated ``llama-server`` when
     llama-cpp-python is unavailable, using its HTTP API and a unique alias
     to verify the launched model before accepting requests.

Both paths support GGUF models with K-quant precision (Q2_K … Q8_0, F16, BF16).
"""

from __future__ import annotations

import ctypes
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from aether.backends.base import Backend, BackendInfo, GenerationRequest, GenerationResult
from aether.core.constants import SUPPORTED_TARGET_IDS
from aether.core.exceptions import BackendError
from aether.utils.logging import get_logger

logger = get_logger(__name__)


class LlamaCppBackend(Backend):
    """Execute GGUF with llama-cpp-python or a dedicated llama-server.

    The in-process adapter is preferred when it is installed.
    """

    model_formats = frozenset({"gguf"})

    def __init__(self) -> None:
        info = BackendInfo(
            name="llamacpp",
            version="2.0.0",
            supported_targets=sorted(
                target
                for target in SUPPORTED_TARGET_IDS
                if target.startswith(("cpu_", "cuda_", "metal", "rocm_", "amd_mi"))
                and not target.endswith("_tee")
            ),
            capabilities=["generate", "chat", "stream", "gguf", "cpu_offload", "k_quant"],
        )
        super().__init__(info)
        self._models: dict[str, Any] = {}
        self._server_urls: dict[str, str] = {}  # model_id → server base URL
        self._server_procs: dict[str, subprocess.Popen] = {}
        self._model_devices: dict[str, tuple[str, ...]] = {}

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
        except (ImportError, OSError):
            return False

    def _find_llama_binary(self) -> str | None:
        """Locate the HTTP-capable ``llama-server`` on PATH."""
        import shutil

        for name in ("llama-server",):
            path = shutil.which(name)
            if path:
                return path
        return None

    def _gpu_devices(self) -> list[dict[str, Any]]:
        """Enumerate local GGML devices, never RPC/offload support alone.

        Older builds without device enumeration remain usable on CPU. C ABI
        signatures are explicit so pointer return values are not truncated.
        """
        if self._has_llama_cpp_python():
            try:
                from llama_cpp import llama_cpp

                library = llama_cpp._lib
                signatures = {
                    "ggml_backend_dev_count": ([], ctypes.c_size_t),
                    "ggml_backend_dev_get": ([ctypes.c_size_t], ctypes.c_void_p),
                    "ggml_backend_dev_type": ([ctypes.c_void_p], ctypes.c_int),
                    "ggml_backend_dev_name": ([ctypes.c_void_p], ctypes.c_char_p),
                    "ggml_backend_dev_backend_reg": ([ctypes.c_void_p], ctypes.c_void_p),
                    "ggml_backend_reg_name": ([ctypes.c_void_p], ctypes.c_char_p),
                }
                for name, (arguments, returns) in signatures.items():
                    function = getattr(library, name)
                    function.argtypes = arguments
                    function.restype = returns
                initialize = getattr(library, "ggml_backend_load_all", None)
                if initialize is not None:
                    initialize.argtypes = []
                    initialize.restype = None
                    initialize()
                devices = []
                for index in range(int(library.ggml_backend_dev_count())):
                    device = library.ggml_backend_dev_get(index)
                    if not device or library.ggml_backend_dev_type(device) not in (1, 2):
                        continue
                    name = library.ggml_backend_dev_name(device).decode("utf-8", "replace")
                    registry = library.ggml_backend_dev_backend_reg(device)
                    backend = library.ggml_backend_reg_name(registry).decode("utf-8", "replace")
                    # main_gpu is an index in llama's GPU device list, not the
                    # absolute GGML registry index (which also includes CPU).
                    devices.append(
                        {
                            "name": name,
                            "backend": backend,
                            "type": library.ggml_backend_dev_type(device),
                        }
                    )
                # llama's default list deduplicates devices across registries,
                # puts RPC first and omits iGPUs when discrete GPUs exist. Do
                # not guess its main_gpu ordinal for ambiguous mixed builds.
                if any(item["backend"].lower() == "rpc" for item in devices):
                    return []
                devices = [item for item in devices if item["type"] == 1] or devices
                if len({item["backend"].lower() for item in devices}) > 1:
                    return []
                return [dict(item, main_gpu=index) for index, item in enumerate(devices)]
            except (ImportError, OSError, AttributeError, TypeError, ValueError):
                return []
        binary = self._find_llama_binary()
        if not binary:
            return []
        try:
            result = subprocess.run(
                [binary, "--list-devices"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode != 0:
                return []
            devices = []
            for line in (result.stdout + "\n" + result.stderr).splitlines():
                match = re.match(r"\s*((CUDA|HIP|ROCm|Metal)\d*)\s*:", line, re.IGNORECASE)
                if match:
                    devices.append(
                        {"name": match[1], "backend": match[2], "main_gpu": len(devices)}
                    )
            return devices
        except (OSError, subprocess.SubprocessError):
            return []

    def available_for_target(self, target_id: str) -> bool:
        """Require matching local device evidence for accelerator targets."""
        if target_id not in self.info.supported_targets or not self.is_available():
            return False
        if target_id.startswith("cpu_"):
            return True
        family = (
            "cuda"
            if target_id.startswith("cuda_")
            else "metal"
            if target_id.startswith("metal")
            else "rocm"
        )
        return any(self._device_family(item) == family for item in self._gpu_devices())

    def automatic_execution_devices(self, target_id: str) -> list[str] | None:
        """Translate a selected target into a verified provider ordinal."""
        if target_id.startswith("cpu_"):
            return ["cpu"]
        if target_id.startswith("metal"):
            return ["mps"]
        family = "cuda" if target_id.startswith("cuda_") else "rocm"
        for device in self._gpu_devices():
            if self._device_family(device) == family:
                ordinal = re.search(r"(\d+)$", device["name"])
                if ordinal:
                    return [f"cuda:{int(ordinal[1])}"]
        raise BackendError(
            f"No verified llama.cpp device ordinal for {target_id}", backend_name=self.name
        )

    @staticmethod
    def _device_family(device: dict[str, Any]) -> str:
        backend = str(device["backend"]).lower()
        if backend in {"hip", "rocm"}:
            return "rocm"
        return backend

    def _execution_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        devices = kwargs.get("execution_devices")
        if devices is None:
            return {"n_gpu_layers": kwargs.get("n_gpu_layers", 0)}
        if devices == ["cpu"]:
            if kwargs.get("n_gpu_layers", 0) != 0:
                raise BackendError(
                    "CPU execution conflicts with n_gpu_layers", backend_name=self.name
                )
            return {"n_gpu_layers": 0}
        if len(devices) != 1:
            raise BackendError(
                "llama.cpp adapter supports one explicit GPU or CPU; "
                "multi-device meshes require another backend",
                backend_name=self.name,
            )
        requested = devices[0]
        inventory = self._gpu_devices()
        if requested == "mps":
            candidates = [item for item in inventory if self._device_family(item) == "metal"]
        elif re.fullmatch(r"cuda(?::\d+)?", requested):
            ordinal = int(requested.partition(":")[2] or 0)
            candidates = [
                item
                for item in inventory
                if self._device_family(item) in {"cuda", "rocm"}
                and re.fullmatch(rf"(?:CUDA|HIP|ROCm){ordinal}", item["name"], re.IGNORECASE)
            ]
        else:
            candidates = []
        if len(candidates) != 1:
            raise BackendError(
                f"llama.cpp cannot verify a local GGML device for {requested}",
                backend_name=self.name,
            )
        layers = kwargs.get("n_gpu_layers", -1)
        if layers == 0:
            raise BackendError(
                "GPU execution conflicts with n_gpu_layers=0", backend_name=self.name
            )
        return {"n_gpu_layers": layers, "main_gpu": candidates[0]["main_gpu"], "split_mode": 0}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, aeg_path: str | None = None, **kwargs: Any) -> Any:
        """Load a GGUF model.

        Preference order:
        1. llama-cpp-python in-process (fastest, no latency)
        2. llama-server subprocess (REST API mode)

        Args:
            model_id: Path to a .gguf file, or HuggingFace model ID.
            aeg_path: AEG is unsupported by this source frontend.
            kwargs: n_ctx, n_threads, n_gpu_layers, server_port, etc.
        """
        if not self.available_for_model(model_id, aeg_path):
            raise BackendError(
                "llama.cpp loads GGUF files; it does not implicitly convert AEG/source/ONNX models",
                backend_name=self.name,
            )
        options = self._execution_options(kwargs)
        mesh = tuple(kwargs.get("execution_devices") or ())
        if model_id in self._model_devices and self._model_devices[model_id] != mesh:
            raise BackendError(
                "Model is already loaded on another execution mesh; unload it first",
                backend_name=self.name,
            )
        if model_id in self._models or model_id in self._server_urls:
            return self._models.get(model_id, model_id)
        kwargs.update(options)
        if self._has_llama_cpp_python():
            result = self._load_inprocess(model_id, **kwargs)
        else:
            result = self._load_subprocess(model_id, **kwargs)
        self._model_devices[model_id] = mesh
        return result

    def _load_inprocess(self, model_id: str, **kwargs: Any) -> Any:
        """Load via llama-cpp-python (in-process)."""
        from llama_cpp import Llama

        n_ctx = kwargs.get("n_ctx", 4096)
        n_threads = kwargs.get("n_threads")
        n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        flash_attn = kwargs.get("flash_attn", True)
        model = Llama(
            model_path=model_id,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            main_gpu=kwargs.get("main_gpu", 0),
            split_mode=kwargs.get("split_mode", 1),
            flash_attn=flash_attn,
            verbose=False,
        )
        self._models[model_id] = model
        logger.info("llama.cpp: loaded in-process model %s", model_id)
        return model

    def _load_subprocess(self, model_id: str, **kwargs: Any) -> str:
        """Launch llama-server as a subprocess and wait for it to be ready.

        Returns the server base URL (e.g., "http://127.0.0.1:8380").
        """
        binary = self._find_llama_binary()
        if binary is None:
            msg = "Neither llama-cpp-python nor llama-server is installed"
            raise ImportError(msg)

        host = "127.0.0.1"
        port = kwargs.get("server_port", 0)
        if type(port) is not int or not 0 <= port <= 65535:
            raise BackendError(
                "server_port must be an integer from 0 to 65535", backend_name=self.name
            )
        # Never reuse a running server's port. There is a release/bind race
        # before the child starts, so also verify its unique model alias below.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                reservation.bind((host, port))
                port = reservation.getsockname()[1]
        except OSError as exc:
            raise BackendError(
                f"llama-server port is unavailable: {port}", backend_name=self.name
            ) from exc
        instance_alias = f"aether-{uuid.uuid4().hex}"
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", 0)
        n_threads = kwargs.get("n_threads", 4)

        cmd = [
            binary,
            "--model",
            model_id,
            "--alias",
            instance_alias,
            "--port",
            str(port),
            "--host",
            host,
            "--ctx-size",
            str(n_ctx),
            "--n-gpu-layers",
            str(n_gpu_layers),
            "--threads",
            str(n_threads),
            "--no-mmap",
        ]
        if "main_gpu" in kwargs:
            cmd.extend(["--main-gpu", str(kwargs["main_gpu"]), "--split-mode", "none"])

        logger.info("llama.cpp: starting server process: %s", " ".join(cmd))
        proc = subprocess.Popen(  # nosec B603 B607
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._server_procs[model_id] = proc

        # Poll until server is ready (max 30s)
        base_url = f"http://{host}:{port}"
        for _ in range(60):
            if proc.poll() is not None:
                self.unload_model(model_id)
                raise BackendError("llama-server exited before readiness", backend_name=self.name)
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                    pass
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1) as response:
                    models = json.load(response)
                if (
                    not isinstance(models, dict)
                    or not isinstance(models.get("data"), list)
                    or not any(
                        isinstance(item, dict) and item.get("id") == instance_alias
                        for item in models.get("data", [])
                    )
                ):
                    self.unload_model(model_id)
                    raise BackendError(
                        "llama-server identity does not match the launched model",
                        backend_name=self.name,
                    )
                if proc.poll() is not None:
                    self.unload_model(model_id)
                    raise BackendError(
                        "llama-server exited during readiness", backend_name=self.name
                    )
                logger.info("llama.cpp server ready at %s", base_url)
                self._server_urls[model_id] = base_url
                return model_id
            except (OSError, ValueError):
                pass

        self.unload_model(model_id)
        raise BackendError(
            "llama-server did not become ready before timeout", backend_name=self.name
        )

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
        parameters = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop or [],
        }
        if request.messages:
            output = model.create_chat_completion(messages=request.messages, **parameters)
        else:
            output = model(text, echo=False, **parameters)
        elapsed = time.perf_counter() - start
        choice = output["choices"][0]
        usage = output.get("usage", {})
        generated = choice.get("text", choice.get("message", {}).get("content", ""))
        completion_tokens = usage.get("completion_tokens", 0)
        return GenerationResult(
            text=generated,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            finish_reason=choice.get("finish_reason", "stop"),
            backend_name=self.name,
            metrics={
                "throughput_tps": completion_tokens / max(elapsed, 1e-6),
                "device": "llama_cpp",
            },
        )

    def generate_stream(self, request: GenerationRequest) -> Any:
        """Yield real provider completion or chat events."""
        if request.model_id not in self._models and request.model_id not in self._server_urls:
            self.load_model(request.model_id)
        parameters = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop or [],
            "stream": True,
        }
        model = self._models.get(request.model_id)
        if model is not None:
            if request.messages:
                events = model.create_chat_completion(messages=request.messages, **parameters)
            else:
                events = model(request.prompt or "", echo=False, **parameters)
            for event in events:
                for choice in event.get("choices", []):
                    text = choice.get("text", choice.get("delta", {}).get("content", ""))
                    if text:
                        yield text
            return
        base_url = self._server_urls[request.model_id]
        if request.messages:
            parameters["messages"] = request.messages
            endpoint = "/v1/chat/completions"
        else:
            parameters["prompt"] = request.prompt or ""
            parameters["n_predict"] = parameters.pop("max_tokens")
            endpoint = "/completion"
        query = urllib.request.Request(
            base_url + endpoint,
            data=json.dumps(parameters).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(query, timeout=120) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = json.loads(data)
                if request.messages:
                    for choice in event.get("choices", []):
                        text = choice.get("delta", {}).get("content", "")
                        if text:
                            yield text
                elif event.get("content"):
                    yield event["content"]

    def _generate_server(self, base_url: str, request: GenerationRequest) -> GenerationResult:
        """Generate via llama-server OpenAI-compatible REST API."""
        text = self._build_prompt(request)
        payload = {
            "prompt": text,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop": request.stop or [],
            "stream": False,
        }
        endpoint = "/completion"
        if request.messages:
            endpoint = "/v1/chat/completions"
            payload.pop("prompt")
            payload.pop("n_predict")
            payload.update({"messages": request.messages, "max_tokens": request.max_tokens})
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}{endpoint}",
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
        if request.messages:
            choice = result["choices"][0]
            gen_text = choice["message"]["content"]
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            finish_reason = choice.get("finish_reason", "stop")
        else:
            gen_text = result.get("content", "")
            completion_tokens = result.get("tokens_predicted", 0)
            prompt_tokens = result.get("tokens_evaluated", 0)
            finish_reason = result.get("stop_type", "stop")
        return GenerationResult(
            text=gen_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            backend_name=self.name,
            metrics={
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
        for model_id in set(self._server_procs) | set(self._models):
            self.unload_model(model_id)

    def unload_model(self, model_id: str) -> None:
        """Release the selected model and its owned server process."""
        model = self._models.pop(model_id, None)
        if model is not None and callable(getattr(model, "close", None)):
            model.close()
        proc = self._server_procs.pop(model_id, None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            logger.info("llama.cpp: terminated server for %s", model_id)
        self._server_urls.pop(model_id, None)
        self._model_devices.pop(model_id, None)

    def get_capabilities(self) -> list[str]:
        """Return implemented frontend operations."""
        return self.info.capabilities

    def __repr__(self) -> str:
        return f"LlamaCppBackend(models={len(self._models)}, servers={len(self._server_urls)})"
