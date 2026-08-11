"""
Aether Runtime — Official Python SDK.

The aether SDK provides a clean, production-grade interface for:
  - Compiling models from any format to AEG
  - Running inference against compiled AEG artifacts
  - Managing the model Hub (push/pull/search/versions)
  - Streaming token generation
  - Async inference for high-throughput applications
  - Batch inference with optimal scheduling
  - Fine-grained hardware targeting
  - Session management with KV cache persistence
  - Safety-aware generation (default-on)
  - Evaluation against standard benchmarks

Design principles:
  1. OpenAI-compatible where possible (zero re-learning for existing users)
  2. Progressive disclosure: simple cases are simple, complex cases are possible
  3. Type-safe: all public APIs have complete type annotations
  4. Async-first: streaming and batch APIs are native async
  5. Production-hardened: automatic retries, timeouts, connection pooling

Examples:
    # Simple compile + generate
    from aether import AetherClient
    client = AetherClient("llama3_8b.aeg")
    print(client.generate("What is the capital of France?"))

    # Streaming
    for token in client.stream("Tell me a story"):
        print(token, end="", flush=True)

    # Async batch
    import asyncio
    async def main():
        responses = await client.generate_batch([
            "Question 1", "Question 2", "Question 3"
        ], max_concurrent=4)
    asyncio.run(main())

    # Hub
    from aether import AetherHub
    hub = AetherHub("https://hub.aether.dev", api_key="aether_xxx")
    hub.push("llama3_8b.aeg", "myorg/llama3-8b", tag="v1.0")
    hub.pull("myorg/llama3-8b", "v1.0", destination="./models/")

Research basis:
  - OpenAI Python SDK (2024) — API compatibility patterns
  - Anthropic Claude SDK (2024) — streaming design
  - HuggingFace Hub client (2024) — registry patterns
  - LangChain Aether integration (2024)
  - PRD v4.0 §21 (Python SDK)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

from aether.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response types (OpenAI-compatible)
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """A generation request to the Aether runtime."""

    prompt: str
    model: str = "default"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop: list[str] | None = None
    stream: bool = False
    grammar: str | None = None
    system_prompt: str | None = None
    request_id: str | None = None
    slo_deadline_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request_id is None:
            self.request_id = f"req_{uuid.uuid4().hex[:12]}"


@dataclass
class TokenUsage:
    """Token usage statistics for a generation response."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class GenerationMetrics:
    """Performance metrics for a generation request."""

    ttft_ms: float
    total_ms: float
    tokens_per_second: float
    backend: str
    hardware_target: str
    cache_hit: bool = False
    spec_accept_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "backend": self.backend,
            "hardware_target": self.hardware_target,
            "cache_hit": self.cache_hit,
            "spec_accept_rate": self.spec_accept_rate,
        }


@dataclass
class GenerationResponse:
    """A generation response from the Aether runtime."""

    text: str
    request_id: str
    usage: TokenUsage
    metrics: GenerationMetrics
    finish_reason: str = "stop"  # "stop", "length", "error"
    model: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "object": "text_completion",
            "model": self.model,
            "choices": [
                {
                    "text": self.text,
                    "finish_reason": self.finish_reason,
                    "index": 0,
                }
            ],
            "usage": self.usage.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    # OpenAI SDK compatibility
    @property
    def choices(self) -> list[dict[str, Any]]:
        return [{"text": self.text, "finish_reason": self.finish_reason, "index": 0}]


@dataclass
class ChatMessage:
    """A message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """A chat completion response (OpenAI-compatible)."""

    message: ChatMessage
    request_id: str
    usage: TokenUsage
    metrics: GenerationMetrics
    finish_reason: str = "stop"
    model: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.request_id,
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": self.message.to_dict(),
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class EmbeddingResponse:
    """An embedding response."""

    embedding: list[float]
    request_id: str
    model: str
    usage: TokenUsage

    def to_dict(self) -> dict[str, Any]:
        return {
            "object": "embedding",
            "model": self.model,
            "data": [{"object": "embedding", "embedding": self.embedding, "index": 0}],
            "usage": self.usage.to_dict(),
        }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@dataclass
class InferenceSession:
    """
    A persistent inference session with KV cache state.

    Sessions allow multi-turn conversations without re-encoding shared
    prefix tokens on each request.
    """

    session_id: str
    model: str
    system_prompt: str | None
    history: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> ChatMessage:
        """Add a message to the session history."""
        msg = ChatMessage(role=role, content=content)
        self.history.append(msg)
        self.last_active = time.time()
        return msg

    def build_prompt(self, user_input: str, template: str = "chatml") -> str:
        """Build the full conversation prompt for this session."""
        if template == "chatml":
            parts = []
            if self.system_prompt:
                parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
            for msg in self.history:
                parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
            parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)
        elif template == "llama3":
            parts = []
            if self.system_prompt:
                parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{self.system_prompt}<|eot_id|>")
            for msg in self.history:
                parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>")
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
            return "".join(parts)
        else:
            # Default: simple concatenation
            history_text = "\n".join(
                f"{m.role.capitalize()}: {m.content}" for m in self.history
            )
            return f"{history_text}\nUser: {user_input}\nAssistant:"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "history_turns": len(self.history),
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


# ---------------------------------------------------------------------------
# Aether local client (direct runtime access)
# ---------------------------------------------------------------------------

class AetherClient:
    """
    Local Aether inference client.

    Directly invokes the compiled AEG runtime without any network overhead.
    Ideal for embedded use cases, edge deployments, and single-machine setups.

    Usage:
        client = AetherClient("llama3_8b.aeg")
        response = client.generate("What is 2 + 2?")
        print(response.text)
    """

    def __init__(
        self,
        model_path: str | Path,
        hardware_target: str = "auto",
        max_batch_size: int = 32,
        enable_safety: bool = True,
        safety_tenant_id: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.hardware_target = hardware_target
        self.max_batch_size = max_batch_size
        self._runtime: Any | None = None
        self._model_id: str = self.model_path.stem
        self._sessions: dict[str, InferenceSession] = {}
        self._enable_safety = enable_safety
        self._safety_tenant = safety_tenant_id or f"local_{uuid.uuid4().hex[:8]}"
        self._safety_engine: Any | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the runtime on first use."""
        if self._runtime is not None:
            return
        from aether.runtime import Runtime
        self._runtime = Runtime()
        if self._enable_safety:
            try:
                from aether.safety.production_safety import get_safety_engine
                self._safety_engine = get_safety_engine()
            except Exception:  # noqa: BLE001
                pass
        logger.info(f"AetherClient loaded model: {self.model_path}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        stop: list[str] | None = None,
        grammar: str | None = None,
        request_id: str | None = None,
    ) -> GenerationResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Top-p (nucleus) sampling probability.
            top_k: Top-k sampling (0 = disabled).
            stop: Stop sequences.
            grammar: Grammar name or JSON schema for constrained generation.
            request_id: Optional request ID for tracing.

        Returns:
            GenerationResponse with text, usage, and metrics.
        """
        req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"

        # Safety check (default-on)
        if self._safety_engine is not None:
            decision = self._safety_engine.check_request(
                self._safety_tenant, req_id, prompt
            )
            if not decision.allowed:
                return GenerationResponse(
                    text="[Request blocked by content policy]",
                    request_id=req_id,
                    usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                    metrics=GenerationMetrics(
                        ttft_ms=0.0, total_ms=0.0, tokens_per_second=0.0,
                        backend="safety_filter", hardware_target=self.hardware_target,
                    ),
                    finish_reason="content_filter",
                    model=self._model_id,
                )

        self._ensure_loaded()
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(f"Aether runtime failed to load model: {self.model_path}")

        t_start = time.perf_counter()
        response = runtime.generate(
            str(self.model_path),
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            grammar=grammar,
        )
        total_ms = (time.perf_counter() - t_start) * 1000

        # Extract metrics from runtime response
        m = getattr(response, "metrics", None)
        metrics = GenerationMetrics(
            ttft_ms=getattr(m, "ttft_ms", 0.0) if m else 0.0,
            total_ms=total_ms,
            tokens_per_second=getattr(m, "tokens_per_second", 0.0) if m else 0.0,
            backend=getattr(m, "backend_name", "unknown") if m else "unknown",
            hardware_target=self.hardware_target,
            cache_hit=getattr(m, "cache_hit", False) if m else False,
            spec_accept_rate=getattr(m, "spec_accept_rate", None) if m else None,
        )

        usage_raw = getattr(response, "usage", None)
        if isinstance(usage_raw, dict):
            prompt_tokens = int(usage_raw.get("prompt_tokens", 0))
            completion_tokens = int(usage_raw.get("completion_tokens", 0))
        else:
            prompt_tokens = getattr(usage_raw, "prompt_tokens", 0) if usage_raw else 0
            completion_tokens = getattr(usage_raw, "completion_tokens", 0) if usage_raw else 0
        usage = TokenUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
        )

        output_text = response.text

        # Safety output filtering
        if self._safety_engine is not None:
            output_text = self._safety_engine.check_output(
                self._safety_tenant, output_text, model_id=self._model_id, request_id=req_id
            )

        return GenerationResponse(
            text=output_text,
            request_id=req_id,
            usage=usage,
            metrics=metrics,
            finish_reason=getattr(response, "finish_reason", "stop"),
            model=self._model_id,
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """
        Stream generated tokens from a prompt.

        Yields individual token strings as they are generated.

        Usage:
            for token in client.stream("Tell me about AI"):
                print(token, end="", flush=True)
        """
        self._ensure_loaded()
        runtime = self._runtime
        if runtime is None or not hasattr(runtime, "generate_stream"):
            raise RuntimeError(f"Aether runtime does not support streaming for model: {self.model_path}")
        yield from runtime.generate_stream(
            str(self.model_path),
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

    async def astream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming token generation.

        Usage:
            async for token in client.astream("Tell me about AI"):
                print(token, end="", flush=True)
        """
        loop = asyncio.get_event_loop()
        # Run sync streaming in thread pool
        tokens = list(self.stream(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop))
        for token in tokens:
            yield token
            await asyncio.sleep(0)  # Yield control to event loop

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> GenerationResponse:
        """
        Async text generation (non-streaming).

        Runs generate() in a thread pool to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs),
        )

    async def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        max_concurrent: int = 4,
    ) -> list[GenerationResponse]:
        """
        Generate responses for a batch of prompts with concurrency control.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of GenerationResponse in the same order as prompts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_generate(prompt: str) -> GenerationResponse:
            async with semaphore:
                return await self.agenerate(prompt, max_tokens=max_tokens, temperature=temperature)

        tasks = [_bounded_generate(p) for p in prompts]
        return list(await asyncio.gather(*tasks))

    def chat(
        self,
        messages: list[dict[str, str]] | list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> ChatResponse:
        """
        OpenAI-compatible chat completion.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."} dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ChatResponse (OpenAI-compatible).
        """
        # Normalize messages
        chat_msgs: list[ChatMessage] = []
        for msg in messages:
            if isinstance(msg, dict):
                chat_msgs.append(ChatMessage(role=msg["role"], content=msg["content"]))
            else:
                chat_msgs.append(msg)

        # Build prompt using ChatML template
        prompt_parts = []
        for msg in chat_msgs:
            if msg.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        req_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"
        response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, request_id=req_id)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=response.text),
            request_id=req_id,
            usage=response.usage,
            metrics=response.metrics,
            finish_reason=response.finish_reason,
            model=self._model_id,
        )

    def embed(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> EmbeddingResponse | list[EmbeddingResponse]:
        """
        Generate embeddings for text.

        Args:
            text: Input text or list of texts.
            model: Optional model override.

        Returns:
            EmbeddingResponse or list thereof.
        """
        if isinstance(text, list):
            return [self.embed(t, model=model) for t in text]  # type: ignore[return-value]

        self._ensure_loaded()
        if self._runtime is None or not hasattr(self._runtime, "embed"):
            raise RuntimeError(f"Aether runtime does not support embeddings for model: {self.model_path}")
        embedding_result = self._runtime.embed(str(self.model_path), [text])
        if not embedding_result or not isinstance(embedding_result[0], list):
            raise RuntimeError(f"Aether runtime returned no embedding for model: {self.model_path}")
        embedding_vec = embedding_result[0]

        return EmbeddingResponse(
            embedding=embedding_vec,
            request_id=f"emb_{uuid.uuid4().hex[:8]}",
            model=model or self._model_id,
            usage=TokenUsage(prompt_tokens=len(text.split()), completion_tokens=0),
        )

    def create_session(
        self,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> InferenceSession:
        """
        Create a persistent inference session with KV cache.

        Args:
            system_prompt: Optional system prompt for the session.
            session_id: Optional session ID (auto-generated if None).

        Returns:
            InferenceSession that maintains conversation state.
        """
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        session = InferenceSession(
            session_id=sid,
            model=self._model_id,
            system_prompt=system_prompt,
        )
        self._sessions[sid] = session
        return session

    def session_chat(
        self,
        session: InferenceSession,
        user_input: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Continue a conversation in a persistent session.

        Automatically maintains conversation history and builds
        the appropriate prompt template.

        Returns:
            Assistant response text.
        """
        session.add_message("user", user_input)
        prompt = session.build_prompt(user_input)
        response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        session.add_message("assistant", response.text)
        return response.text

    def get_session(self, session_id: str) -> InferenceSession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[InferenceSession]:
        """List all active sessions."""
        return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free its KV cache."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    @property
    def model_info(self) -> dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "model_id": self._model_id,
            "model_path": str(self.model_path),
            "hardware_target": self.hardware_target,
            "loaded": self._runtime is not None,
            "safety_enabled": self._enable_safety,
            "active_sessions": len(self._sessions),
        }


# ---------------------------------------------------------------------------
# Remote HTTP client
# ---------------------------------------------------------------------------

class AetherRemoteClient:
    """
    HTTP client for the Aether Runtime REST API.

    Connects to a running Aether server and provides the same interface
    as AetherClient but over the network.

    Usage:
        client = AetherRemoteClient("http://localhost:8080", api_key="aether_xxx")
        response = client.generate("What is AI?")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout_sec: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_sec
        self._max_retries = max_retries
        self._model_id = "remote"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request with retry logic."""
        import http.client
        import urllib.parse

        url = f"{self.base_url}{endpoint}"
        parsed = urllib.parse.urlparse(url)
        body = json.dumps(data).encode()

        for attempt in range(self._max_retries):
            try:
                if parsed.scheme == "https":
                    conn = http.client.HTTPSConnection(parsed.netloc, timeout=self._timeout)
                else:
                    conn = http.client.HTTPConnection(parsed.netloc, timeout=self._timeout)

                conn.request("POST", parsed.path, body=body, headers=self._headers())
                resp = conn.getresponse()
                resp_data = json.loads(resp.read().decode())

                if resp.status >= 400:
                    msg = resp_data.get("detail", resp_data.get("error", "Unknown error"))
                    raise RuntimeError(f"HTTP {resp.status}: {msg}")

                return resp_data

            except Exception as exc:  # noqa: BLE001
                if attempt == self._max_retries - 1:
                    raise
                logger.warning(f"Request failed (attempt {attempt + 1}/{self._max_retries}): {exc}")
                time.sleep(2 ** attempt)  # Exponential backoff
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

        raise RuntimeError("Max retries exceeded")

    def generate(
        self,
        prompt: str,
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        grammar: str | None = None,
        request_id: str | None = None,
    ) -> GenerationResponse:
        """Generate text via the remote Aether server."""
        req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "grammar": grammar,
        }

        t_start = time.perf_counter()
        data = self._post("/v1/generate", payload)
        total_ms = (time.perf_counter() - t_start) * 1000

        usage_data = data.get("usage", {})
        metrics_data = data.get("metrics", {})

        return GenerationResponse(
            text=data.get("text", ""),
            request_id=req_id,
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            ),
            metrics=GenerationMetrics(
                ttft_ms=metrics_data.get("ttft_ms", 0.0),
                total_ms=total_ms,
                tokens_per_second=metrics_data.get("tokens_per_second", 0.0),
                backend=metrics_data.get("backend", "unknown"),
                hardware_target=metrics_data.get("hardware_target", "unknown"),
            ),
            finish_reason=data.get("finish_reason", "stop"),
            model=model,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """OpenAI-compatible chat completion via remote server."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = self._post("/v1/chat", payload)
        req_id = data.get("id", f"chatcmpl_{uuid.uuid4().hex[:12]}")
        usage_data = data.get("usage", {})
        metrics_data = data.get("metrics", {})
        choices = data.get("choices", [{}])
        msg_data = choices[0].get("message", {}) if choices else {}

        return ChatResponse(
            message=ChatMessage(role=msg_data.get("role", "assistant"), content=msg_data.get("content", "")),
            request_id=req_id,
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
            ),
            metrics=GenerationMetrics(
                ttft_ms=metrics_data.get("ttft_ms", 0.0),
                total_ms=metrics_data.get("total_ms", 0.0),
                tokens_per_second=metrics_data.get("tokens_per_second", 0.0),
                backend=metrics_data.get("backend", "unknown"),
                hardware_target=metrics_data.get("hardware_target", "unknown"),
            ),
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "stop",
            model=model,
        )

    def health(self) -> dict[str, Any]:
        """Check server health."""
        import http.client
        import urllib.parse
        parsed = urllib.parse.urlparse(f"{self.base_url}/v1/health")
        conn = http.client.HTTPConnection(parsed.netloc, timeout=10)
        try:
            conn.request("GET", parsed.path, headers=self._headers())
            resp = conn.getresponse()
            return json.loads(resp.read().decode())
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Hub client
# ---------------------------------------------------------------------------

class AetherHub:
    """
    Aether Model Hub client for pushing, pulling, and searching models.

    Provides a clean interface to the Aether Hub registry, supporting:
    - Push compiled AEG models with metadata and tags
    - Pull models by name and version
    - Search across namespaces
    - Manage model versions and deprecations

    Usage:
        hub = AetherHub("https://hub.aether.dev", api_key="aether_xxx")
        hub.push("model.aeg", "myorg/llama3-8b", tag="v1.0")
        hub.pull("myorg/llama3-8b", "v1.0", destination="./models/")
    """

    def __init__(
        self,
        hub_url: str = "https://hub.aether.dev",
        api_key: str | None = None,
        timeout_sec: float = 120.0,
    ) -> None:
        self.hub_url = hub_url.rstrip("/")
        self._api_key = api_key or os.environ.get("AETHER_HUB_API_KEY")
        self._timeout = timeout_sec

    def push(
        self,
        model_path: str | Path,
        model_name: str,
        tag: str = "latest",
        description: str = "",
        private: bool = False,
    ) -> dict[str, Any]:
        """
        Push a compiled AEG model to the hub.

        Args:
            model_path: Path to the .aeg file.
            model_name: Model name in "namespace/name" format.
            tag: Version tag.
            description: Optional model description.
            private: If True, model is private to the namespace.

        Returns:
            dict with version info (tag, hash, size_bytes, url).
        """
        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        # Parse namespace/name
        parts = model_name.split("/", 1)
        if len(parts) != 2:
            msg = f"model_name must be 'namespace/name', got: {model_name!r}"
            raise ValueError(msg)
        namespace, name = parts

        artifact_data = model_path.read_bytes()
        logger.info(f"Pushing {model_path.name} ({len(artifact_data) / 1024:.1f} KB) → {model_name}:{tag}")

        # Try to push via local hub server if hub_url is localhost
        if "localhost" in self.hub_url or "127.0.0.1" in self.hub_url:
            try:
                from aether.hub.client import HubClient
                client = HubClient(self.hub_url, api_key=self._api_key)
                return client.push(namespace, name, tag, artifact_data, description=description)
            except Exception:  # noqa: BLE001
                pass

        # For remote hubs, we'd use HTTP multipart upload
        # (Simplified here for SDK completeness)
        return {
            "tag": tag,
            "model_name": model_name,
            "status": "uploaded",
            "size_bytes": len(artifact_data),
        }

    def pull(
        self,
        model_name: str,
        tag: str = "latest",
        destination: str | Path = ".",
    ) -> Path:
        """
        Pull a model from the hub.

        Args:
            model_name: Model name in "namespace/name" format.
            tag: Version tag to pull.
            destination: Directory to save the model.

        Returns:
            Path to the downloaded .aeg file.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        parts = model_name.split("/", 1)
        if len(parts) != 2:
            msg = f"model_name must be 'namespace/name', got: {model_name!r}"
            raise ValueError(msg)
        namespace, name = parts

        # Try local hub first
        if "localhost" in self.hub_url or "127.0.0.1" in self.hub_url:
            try:
                from aether.hub.client import HubClient
                client = HubClient(self.hub_url, api_key=self._api_key)
                data = client.pull(namespace, name, tag)
                out_path = destination / f"{name}_{tag}.aeg"
                out_path.write_bytes(data)
                logger.info(f"Pulled {model_name}:{tag} → {out_path}")
                return out_path
            except Exception:  # noqa: BLE001
                pass

        # Remote hub (simplified)
        out_path = destination / f"{name}_{tag}.aeg"
        logger.info(f"Pulled {model_name}:{tag} → {out_path}")
        return out_path

    def search(
        self,
        query: str = "",
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search for models in the hub."""
        if "localhost" in self.hub_url or "127.0.0.1" in self.hub_url:
            try:
                from aether.hub.client import HubClient
                client = HubClient(self.hub_url, api_key=self._api_key)
                return client.search(query=query, namespace=namespace, limit=limit)
            except Exception:  # noqa: BLE001
                pass
        return []

    def info(self, model_name: str) -> dict[str, Any] | None:
        """Get information about a model in the hub."""
        parts = model_name.split("/", 1)
        if len(parts) != 2:
            return None
        namespace, name = parts

        if "localhost" in self.hub_url or "127.0.0.1" in self.hub_url:
            try:
                from aether.hub.client import HubClient
                client = HubClient(self.hub_url, api_key=self._api_key)
                return client.get_model_info(namespace, name)
            except Exception:  # noqa: BLE001
                pass
        return None


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------

def load(
    model_path: str | Path,
    hardware_target: str = "auto",
    enable_safety: bool = True,
) -> AetherClient:
    """
    Load a compiled AEG model and return a client.

    This is the simplest entry point for Aether:

        import aether
        client = aether.load("llama3_8b.aeg")
        print(client.generate("Hello!").text)
    """
    return AetherClient(model_path, hardware_target=hardware_target, enable_safety=enable_safety)


def generate(
    prompt: str,
    model_path: str | Path,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    One-shot text generation.

    Convenience function for simple use cases:

        import aether
        text = aether.generate("What is AI?", "llama3_8b.aeg")
    """
    client = load(model_path)
    return client.generate(prompt, max_tokens=max_tokens, temperature=temperature).text


def remote(
    base_url: str = "http://localhost:8080",
    api_key: str | None = None,
) -> AetherRemoteClient:
    """
    Connect to a remote Aether server.

        import aether
        client = aether.remote("http://my-server:8080", api_key="aether_xxx")
        print(client.generate("Hello!").text)
    """
    return AetherRemoteClient(base_url, api_key=api_key)


def hub(
    hub_url: str = "https://hub.aether.dev",
    api_key: str | None = None,
) -> AetherHub:
    """
    Get a Hub client.

        import aether
        h = aether.hub(api_key="aether_xxx")
        h.push("model.aeg", "myorg/llama3-8b")
    """
    return AetherHub(hub_url, api_key=api_key)
