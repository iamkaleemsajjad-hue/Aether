"""
REST API routes for Aether — v4.0 extended.

Implements all PRD v3.1 endpoints plus new PRD v4.0 endpoints:

v3.1 endpoints (existing, preserved):
  POST /v1/generate           — text completion
  POST /v1/chat               — chat completion (OpenAI-compatible)
  POST /v1/embeddings         — embedding generation
  POST /v1/rerank             — document reranking
  POST /v1/transcribe         — audio transcription
  POST /v1/compile            — compile a model (async job)
  GET  /v1/compile/{job_id}   — compilation job status
  GET  /v1/models             — list compiled models
  POST /v1/models/pull        — download and compile
  GET  /v1/models/{name}      — model info
  DELETE /v1/models/{name}    — remove compiled model
  GET  /v1/hardware           — hardware fingerprint
  GET  /v1/kernels            — active kernel targets
  GET  /v1/metrics            — Prometheus metrics
  GET  /v1/health             — health check

v4.0 NEW endpoints (PRD §22 Extended Developer API v4.0):
  POST /v1/tools/call         — MCP tool call (R6 MCP Native Integration)
  POST /v1/grammar/compile    — pre-compile a grammar FSM (Pass 11)
  GET  /v1/grammar/list       — list compiled grammars
  POST /v1/models/{name}/merge— apply model merging task vectors (Pass 12)
  POST /v1/models/{name}/ttt  — TTT fast-weight update for a session (Pass 13)
  GET  /v1/targets            — list all supported hardware targets
  GET  /v1/targets/{target_id}— single target hardware profile
  GET  /v1/green/status       — carbon / energy status (R7 Green Power Manager)
  POST /v1/tee/session        — start a TEE confidential inference session (R8)
  DELETE /v1/tee/session/{id} — close a TEE session
"""

import json
import asyncio
import builtins
import time
import uuid
from pathlib import Path
from typing import Any

from aether.runtime import Runtime
from aether.core.constants import AETHER_VERSION


def create_router(runtime: Runtime) -> Any:
    """Create a FastAPI router with all v3.1 + v4.0 endpoints."""
    try:
        from fastapi import APIRouter, HTTPException
        from starlette.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError:
        msg = "fastapi and pydantic are required for the server"
        raise ImportError(msg)

    router = APIRouter()
    eval_jobs: dict[str, dict[str, Any]] = {}
    merge_jobs: dict[str, dict[str, Any]] = {}
    grpo_jobs: dict[str, dict[str, Any]] = {}
    video_jobs: dict[str, dict[str, Any]] = {}
    ab_experiments: dict[str, dict[str, Any]] = {}
    multi_agent_sessions: dict[str, Any] = {}
    kernel_artifacts: dict[str, dict[str, Any]] = {}

    # ── Request / Response models ─────────────────────────────────────────────

    class GenerateRequest(BaseModel):
        model: str
        prompt: str
        max_tokens: int = 1024
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 0
        stream: bool = False
        stop: list[str] | None = None
        grammar: str | None = Field(
            default=None,
            description="Grammar name or inline JSON schema for structured output (R3/Pass 11)",
        )
        slo_deadline_ms: float | None = Field(
            default=None,
            description="Request deadline in milliseconds for R4 SLO-Aware Scheduler",
        )

    class GenerateResponse(BaseModel):
        text: str
        usage: dict[str, int]
        metrics: dict[str, Any]

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        max_tokens: int = 1024
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        grammar: str | None = Field(
            default=None,
            description="Structured output grammar name or JSON schema",
        )
        response_format: dict[str, Any] | None = Field(
            default=None,
            description="OpenAI-compatible response_format (type: json_object | json_schema)",
        )
        slo_deadline_ms: float | None = None

    class EmbedRequest(BaseModel):
        model: str
        input: list[str]

    class RerankRequest(BaseModel):
        model: str
        query: str
        documents: list[str]

    class TranscribeRequest(BaseModel):
        model: str
        audio: str
        language: str | None = None

    class CompileRequest(BaseModel):
        model: str
        target: str = "auto"
        quantization: str | None = None
        quality_budget: float = 0.98
        enable_mtp: bool = False
        enable_grammar: bool = False
        enable_tee: bool = False
        enable_green: bool = False
        merge_tasks: list[str] | None = None

    class PullRequest(BaseModel):
        model: str

    class CascadeRequest(BaseModel):
        query: str
        model_routing: dict[str, str] | None = None
        max_tokens: int = 1024
        temperature: float = 0.7

    class StructuredRequest(BaseModel):
        model: str
        prompt: str
        schema: dict[str, Any] | None = None
        grammar: str | None = None
        regex: str | None = None
        max_tokens: int = 1024
        temperature: float = 0.0

    class EvalRequest(BaseModel):
        model: str
        domain: str = "general"
        num_examples: int = Field(default=100, gt=0)
        quality_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
        datasets: dict[str, str] | None = Field(
            default=None,
            description=(
                "Benchmark-to-file mapping. Paths are relative to the server's "
                "configured eval_data_dir."
            ),
        )
        max_tokens: int = Field(default=256, gt=0)
        allow_code_execution: bool = False

    class ABStartRequest(BaseModel):
        model_a: str
        model_b: str
        prompt: str
        traffic_split_pct: int = 50
        max_tokens: int = 128

    class ABRollbackRequest(BaseModel):
        experiment_id: str

    class MergeRequest(BaseModel):
        model: str
        task_vectors: list[dict[str, Any]]
        merge_method: str = "task_arithmetic"
        density: float = 1.0

    class ReweightRequest(BaseModel):
        model: str
        weights: dict[str, float]

    class MultiAgentRequest(BaseModel):
        agent_count: int = 4
        shared_prefix: str = ""
        model: str = ""

    class MultiAgentSpawnRequest(BaseModel):
        session_id: str
        model: str
        context: str = ""
        inherit_agent_id: str | None = None

    class TTTAdaptRequest(BaseModel):
        model: str
        session_id: str
        hidden_states: list[list[float]]
        layer_idx: int = -1

    class MCPRegisterRequest(BaseModel):
        server_id: str
        transport: str = "stdio"
        endpoint: str | None = None
        command: str | None = None

    class GreenRouteRequest(BaseModel):
        regions: list[str]
        latency_deadline_s: float = 1.0

    class VideoRequest(BaseModel):
        model: str
        video_path: str
        prompt: str
        compression: str = "stc"
        max_visual_tokens: int = 4096

    class GRPORequest(BaseModel):
        model: str
        prompts: list[str]
        group_size: int = 8
        domain: str = "math"
        learning_rate: float = 1e-6
        max_tokens: int = 2048

    class KernelGenerateRequest(BaseModel):
        target: str
        op_name: str

    # ── v4.0 NEW request models ───────────────────────────────────────────────

    class MCPToolCallRequest(BaseModel):
        """R6 MCP Native Integration — call a registered MCP tool."""
        tool_id: str = Field(..., description="MCP tool identifier (server/tool_name)")
        arguments: dict[str, Any] = Field(default_factory=dict)
        model: str | None = Field(
            default=None,
            description="Model to use for tool-augmented generation (optional)",
        )

    class GrammarCompileRequest(BaseModel):
        """Pass 11 Grammar Constraint — pre-compile a grammar FSM."""
        grammar_name: str = Field(..., description="Name to register this grammar under")
        grammar_type: str = Field(
            default="json_schema",
            description="One of: json_schema, regex, ebnf, openai_tool_call",
        )
        grammar_spec: str | dict[str, Any] = Field(
            ..., description="Grammar specification: JSON schema dict, regex string, or EBNF string"
        )
        model: str | None = Field(
            default=None, description="Model to use for token-specific FSM compilation"
        )

    class ModelMergeRequest(BaseModel):
        """Pass 12 Model Merging — apply task vector merging."""
        task_vectors: list[dict[str, Any]] = Field(
            ...,
            description="List of task vector configs: [{name: str, coefficient: float, path: str}]",
        )
        merge_method: str = Field(
            default="task_arithmetic",
            description="Merge method: task_arithmetic | ties | dare | evolutionary",
        )
        density: float = Field(default=1.0, description="Pruning density for DARE/TIES (0.0-1.0)")

    class TTTUpdateRequest(BaseModel):
        """Pass 13 / R5 TTT — update fast weights for a session."""
        session_id: str = Field(..., description="Active runtime session ID")
        context: str = Field(..., description="Context text for TTT domain adaptation")
        learning_rate: float = Field(default=1e-4, description="Fast-weight update learning rate")
        max_steps: int = Field(default=10, description="Number of gradient steps")

    class TEESessionRequest(BaseModel):
        """R8 TEE — start a confidential inference session."""
        model: str
        tee_backend: str = Field(
            default="auto",
            description="TEE backend: auto | nvidia_cc | intel_tdx | amd_sev_snp",
        )
        seal_weights: bool = Field(
            default=False,
            description="Whether to encrypt model weights into the enclave memory",
        )
        attestation_required: bool = Field(
            default=True, description="Request attestation report from hardware"
        )

    # ── v3.1 Routes (preserved) ────────────────────────────────────────────────

    @router.get("/health", tags=["System"])
    async def api_health():
        """Return the PRD-defined versioned health endpoint."""
        return {
            "status": "healthy",
            "version": AETHER_VERSION,
            "target": runtime.fingerprint.target_id,
            "loaded_models": len(runtime._loaded_models),
        }

    @router.post("/generate", tags=["Generation"])
    async def generate(req: GenerateRequest):
        """Text completion with optional structured output and SLO deadline."""
        try:
            if req.stream:
                async def event_stream() -> Any:
                    try:
                        if req.grammar:
                            stream = runtime.generate_constrained_stream(
                                model_id=req.model,
                                prompt=req.prompt,
                                grammar=req.grammar,
                                max_tokens=req.max_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                top_k=req.top_k,
                                stop=req.stop,
                                slo_deadline_ms=req.slo_deadline_ms,
                            )
                        else:
                            stream = runtime.generate_stream(
                                model_id=req.model,
                                prompt=req.prompt,
                                max_tokens=req.max_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                top_k=req.top_k,
                                stop=req.stop,
                                slo_deadline_ms=req.slo_deadline_ms,
                            )
                        for index, chunk in enumerate(
                            stream
                        ):
                            yield f"data: {json.dumps({'text': chunk, 'index': index})}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as exc:  # noqa: BLE001
                        # Once an SSE response starts, HTTP status cannot be
                        # changed. Emit an explicit terminal error event and do
                        # not emit [DONE], so clients cannot mistake failure for
                        # a successful completion.
                        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")
            if req.grammar:
                response = runtime.generate_constrained(
                    model_id=req.model,
                    prompt=req.prompt,
                    grammar=req.grammar,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    stop=req.stop,
                    slo_deadline_ms=req.slo_deadline_ms,
                )
            else:
                response = runtime.generate(
                    model_id=req.model,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    stop=req.stop,
                    slo_deadline_ms=req.slo_deadline_ms,
                )
            return GenerateResponse(
                text=response.text,
                usage=response.usage,
                metrics=response.metrics.to_dict(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/chat", tags=["Chat"])
    async def chat(req: ChatRequest):
        """Chat completion (OpenAI-compatible) with structured output and SLO support."""
        try:
            messages = [m.model_dump() for m in req.messages]
            response_schema: dict[str, Any] | None = None
            if req.response_format is not None:
                if req.grammar:
                    raise HTTPException(status_code=422, detail="grammar and response_format are mutually exclusive")
                format_type = req.response_format.get("type")
                if format_type != "json_schema":
                    raise HTTPException(
                        status_code=422,
                        detail="response_format requires a tokenizer-aware json_schema constraint",
                    )
                schema_payload = req.response_format.get("json_schema")
                if not isinstance(schema_payload, dict):
                    raise HTTPException(status_code=422, detail="response_format.json_schema must be an object")
                response_schema = schema_payload.get("schema")
                if not isinstance(response_schema, dict):
                    raise HTTPException(status_code=422, detail="response_format.json_schema.schema must be an object")
            if req.stream:
                if req.grammar or response_schema is not None:
                    stream = runtime.generate_constrained_stream(
                        model_id=req.model,
                        messages=messages,
                        grammar=req.grammar,
                        schema=response_schema,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        slo_deadline_ms=req.slo_deadline_ms,
                    )
                else:
                    stream = runtime.generate_stream(
                        model_id=req.model,
                        messages=messages,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        slo_deadline_ms=req.slo_deadline_ms,
                    )

                async def event_stream() -> Any:
                    try:
                        for index, chunk in enumerate(stream):
                            yield f"data: {json.dumps({'text': chunk, 'index': index})}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as exc:  # noqa: BLE001
                        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            if req.grammar or response_schema is not None:
                response = runtime.generate_constrained(
                    model_id=req.model,
                    messages=messages,
                    grammar=req.grammar,
                    schema=response_schema,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    slo_deadline_ms=req.slo_deadline_ms,
                )
            else:
                response = runtime.chat(
                    model_id=req.model,
                    messages=messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    slo_deadline_ms=req.slo_deadline_ms,
                )
            return {
                "model": req.model,
                "text": response.text,
                "usage": response.usage,
                "metrics": response.metrics.to_dict(),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/embeddings", tags=["Embeddings"])
    async def embeddings(req: EmbedRequest):
        """Embedding generation."""
        try:
            vectors = runtime.embed(req.model, req.input)
            return {
                "model": req.model,
                "vectors": vectors,
                "usage": {"prompt_tokens": 0},
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/rerank", tags=["Rerank"])
    async def rerank(req: RerankRequest):
        """Document reranking."""
        try:
            results = runtime.rerank(req.model, req.query, req.documents)
            return {"results": results}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/transcribe", tags=["Transcription"])
    async def transcribe(req: TranscribeRequest):
        """Audio transcription."""
        try:
            text = runtime.transcribe(req.model, req.audio, language=req.language)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/compile", tags=["Compilation"])
    async def compile_model(req: CompileRequest):
        """Compile a model for a target asynchronously. Returns a job_id."""
        job_id = str(uuid.uuid4())
        try:
            # Validate target before enqueuing
            from aether.core.constants import SUPPORTED_TARGET_IDS
            if req.target != "auto" and req.target not in SUPPORTED_TARGET_IDS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown target '{req.target}'. Valid targets: {sorted(SUPPORTED_TARGET_IDS)}",
                )
            # Enqueue compilation job
            if hasattr(runtime, "compile_async"):
                runtime.compile_async(
                    model_id=req.model,
                    job_id=job_id,
                    target=req.target,
                    quantization=req.quantization,
                    quality_budget=req.quality_budget,
                    enable_mtp=req.enable_mtp,
                    enable_grammar=req.enable_grammar,
                    enable_tee=req.enable_tee,
                    enable_green=req.enable_green,
                )
            return {
                "job_id": job_id,
                "status": "queued",
                "model": req.model,
                "target": req.target,
                "submitted_at": time.time(),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/compile/{job_id}", tags=["Compilation"])
    async def get_compile_status(job_id: str):
        """Get compilation job status."""
        try:
            if hasattr(runtime, "get_compile_status"):
                status = runtime.get_compile_status(job_id)
                return status
            return {"job_id": job_id, "status": "unknown"}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/models", tags=["Model Management"])
    async def list_models():
        """List compiled models."""
        models = runtime.list()
        return {"models": models}

    @router.post("/models/pull", tags=["Model Management"])
    async def pull_model(req: PullRequest):
        """Download and compile a model from the hub."""
        try:
            runtime.pull(req.model)
            return {"status": "success", "model": req.model}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/generate/cascade", tags=["Generation"])
    async def generate_cascade(req: CascadeRequest):
        try:
            response = runtime.generate_cascade(
                req.query,
                model_routing=req.model_routing,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            return response.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/generate/structured", tags=["Structured Output"])
    async def generate_structured(req: StructuredRequest):
        if sum(value is not None for value in (req.schema, req.grammar, req.regex)) != 1:
            raise HTTPException(status_code=422, detail="exactly one of schema, grammar, or regex is required")
        try:
            response = runtime.generate_constrained(
                req.model,
                req.prompt,
                schema=req.schema,
                grammar=req.grammar,
                regex=req.regex,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            return response.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/models/{name:path}/graph", tags=["Model Management"])
    async def model_graph(name: str):
        try:
            package = __import__("aether.core.aeg_format", fromlist=["load_aeg_package"]).load_aeg_package(
                runtime._resolve_aeg_path(name) or name
            )
            if package.ir is None:
                raise ValueError("AEG contains no AEG-IR module")
            return {"model": name, "ir": package.ir.to_dict(), "text": package.ir.to_text()}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/models/{name:path}/mla", tags=["Model Management"])
    async def model_mla(name: str):
        try:
            info = runtime.info(name)
            return {"model": name, "mla": info.get("architecture", {}).get("attention_type", "MHA").upper() == "MLA", "plan": info}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.get("/models/{name:path}/reasoning", tags=["Model Management"])
    async def model_reasoning(name: str):
        try:
            root = runtime._resolve_aeg_path(name) or name
            path = Path(root) / "graph" / "reasoning_graph.aeg-ir"
            if not path.exists():
                raise FileNotFoundError(path)
            return {"model": name, "reasoning_graph": json.loads(path.read_text(encoding="utf-8"))}
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.post("/eval", tags=["Evaluation"])
    async def start_eval(req: EvalRequest):
        job_id = str(uuid.uuid4())
        try:
            evaluator = None
            benchmarks = None
            if req.datasets:
                root_value = (runtime.config.extra or {}).get("eval_data_dir")
                if not isinstance(root_value, str) or not root_value.strip():
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "dataset evaluation is unavailable: configure "
                            "RuntimeConfig.extra['eval_data_dir']"
                        ),
                    )
                root = Path(root_value).expanduser().resolve()
                if not root.is_dir():
                    raise HTTPException(status_code=503, detail=f"evaluation data root not found: {root}")
                resolved: dict[str, str] = {}
                for benchmark, requested_path in req.datasets.items():
                    if not benchmark.strip() or not isinstance(requested_path, str) or not requested_path.strip():
                        raise HTTPException(status_code=422, detail="datasets must map non-empty names to paths")
                    candidate = (root / requested_path).resolve()
                    try:
                        candidate.relative_to(root)
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=422,
                            detail=f"dataset path escapes configured eval_data_dir: {requested_path!r}",
                        ) from exc
                    if not candidate.is_file():
                        raise HTTPException(status_code=422, detail=f"dataset file not found: {requested_path!r}")
                    resolved[benchmark.strip().lower()] = str(candidate)

                from aether.observability.ci_pipeline import DatasetBenchmarkEvaluator

                def generate_fn(*, prompt: str, benchmark: str, max_tokens: int) -> str:
                    return runtime.generate(
                        req.model,
                        prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    ).text

                evaluator = DatasetBenchmarkEvaluator(
                    resolved,
                    generate_fn,
                    max_tokens=req.max_tokens,
                    max_examples=req.num_examples,
                    allow_code_execution=req.allow_code_execution,
                )
                benchmarks = builtins.list(resolved)

            # Generation-backed evaluation is CPU/GPU-bound synchronous work;
            # keep the FastAPI event loop responsive while preserving the
            # existing job record and explicit failed status semantics.
            result = await asyncio.to_thread(
                runtime.eval_gate,
                req.model,
                req.domain,
                req.num_examples,
                req.quality_threshold,
                benchmarks=benchmarks,
                evaluator=evaluator,
            )
            # An unavailable or failed quality gate is not a successful job.
            # Returning ``succeeded`` here allowed callers to deploy an
            # artifact whose evaluation never ran.
            passed = bool(result.get("passed", False)) if isinstance(result, dict) else False
            eval_status = "succeeded" if passed else "failed"
            eval_jobs[job_id] = {"job_id": job_id, "status": eval_status, "result": result}
            return eval_jobs[job_id]
        except HTTPException:
            raise
        except Exception as exc:
            eval_jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/eval/{job_id}", tags=["Evaluation"])
    async def get_eval(job_id: str):
        result = eval_jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"evaluation job {job_id!r} not found")
        return result

    @router.post("/ab/start", tags=["Rollout"])
    async def start_ab(req: ABStartRequest):
        if not 0 <= req.traffic_split_pct <= 100:
            raise HTTPException(status_code=422, detail="traffic_split_pct must be between 0 and 100")
        experiment_id = str(uuid.uuid4())
        try:
            result = runtime.ab_rollout(req.model_a, req.model_b, req.prompt, req.traffic_split_pct, max_tokens=req.max_tokens)
            ab_experiments[experiment_id] = {"experiment_id": experiment_id, "status": "active", "config": req.model_dump(), "last_result": result}
            return ab_experiments[experiment_id]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/ab/{experiment_id}", tags=["Rollout"])
    async def get_ab(experiment_id: str):
        result = ab_experiments.get(experiment_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"A/B experiment {experiment_id!r} not found")
        return result

    @router.post("/ab/rollback", tags=["Rollout"])
    async def rollback_ab(req: ABRollbackRequest):
        result = ab_experiments.get(req.experiment_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"A/B experiment {req.experiment_id!r} not found")
        result["status"] = "rolled_back"
        return {"experiment_id": req.experiment_id, "status": result["status"]}

    @router.get("/traces", tags=["Observability"])
    async def traces():
        return runtime.tracer.export_otlp_json()

    @router.post("/merge", tags=["Model Management"])
    async def merge(req: MergeRequest):
        job_id = str(uuid.uuid4())
        try:
            result = runtime.merge(req.model, req.task_vectors, req.merge_method, req.density)
            merge_jobs[job_id] = {"job_id": job_id, "status": "succeeded", "result": result}
            return merge_jobs[job_id]
        except Exception as exc:
            merge_jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/merge/{job_id}", tags=["Model Management"])
    async def get_merge(job_id: str):
        result = merge_jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"merge job {job_id!r} not found")
        return result

    @router.post("/merge/reweight", tags=["Model Management"])
    async def reweight(req: ReweightRequest):
        try:
            return {"model": req.model, "weights": runtime.set_task_weights(req.model, **req.weights)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/multi_agent/session", tags=["Multi-Agent"])
    async def create_multi_agent(req: MultiAgentRequest):
        if req.agent_count < 1:
            raise HTTPException(status_code=422, detail="agent_count must be positive")
        result = runtime.multi_agent_session(
            models=[req.model] if req.model else [],
            coordination="relay",
            agent_count=req.agent_count,
            shared_prefix=req.shared_prefix,
        )
        multi_agent_sessions[result["session_id"]] = result
        return result

    @router.post("/multi_agent/spawn", tags=["Multi-Agent"])
    async def spawn_multi_agent(req: MultiAgentSpawnRequest):
        """Spawn an agent while explicitly preserving KV lineage when requested."""
        session = multi_agent_sessions.get(req.session_id)
        if session is None or not hasattr(session, "spawn_agent"):
            raise HTTPException(status_code=404, detail="multi-agent session not found")
        parent = None
        if req.inherit_agent_id:
            parent = getattr(session, "_agents", {}).get(req.inherit_agent_id)
            if parent is None:
                raise HTTPException(status_code=404, detail="inherited agent not found")
        try:
            agent = await session.spawn_agent(
                req.model,
                context=req.context,
                inherit_kv_from=parent,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "session_id": req.session_id,
            "agent_session_id": agent.session_id,
            "model": agent.model_id,
            "inherited_kv": parent is not None and agent.prefix_hash == parent.prefix_hash,
            "prefix_hash": agent.prefix_hash,
            "agent_count": session.get("agent_count", 0),
        }

    @router.delete("/multi_agent/session/{session_id}", tags=["Multi-Agent"])
    async def delete_multi_agent(session_id: str):
        session = multi_agent_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="multi-agent session not found")
        session.close()
        multi_agent_sessions.pop(session_id, None)
        return {"session_id": session_id, "status": "closed"}

    @router.get("/slo/status", tags=["Scheduling"])
    async def slo_status():
        scheduler = runtime.slo_scheduler or runtime.scheduler
        return scheduler.summary()

    @router.post("/slo/profile", tags=["Scheduling"])
    async def slo_profile(profile: dict[str, Any]):
        name = profile.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(status_code=422, detail="profile.name is required")
        values = {key: profile.get(key) for key in ("max_ttft_ms", "max_tbt_ms")}
        runtime.config.slo_profiles[name] = values
        return {"name": name, "profile": values}

    @router.post("/ttt/adapt", tags=["TTT"])
    async def ttt_adapt(req: TTTAdaptRequest):
        runtime._init_v4_layers(runtime._resolve_aeg_path(req.model))
        if runtime.ttt_engine is None:
            raise HTTPException(status_code=503, detail="TTT is not enabled by the loaded AEG")
        runtime.ttt_engine.begin_request(req.session_id)
        loss = runtime.ttt_engine.adapt(req.session_id, req.hidden_states, req.layer_idx)
        return {"session_id": req.session_id, "loss": loss, "status": "updated"}

    @router.post("/ttt/reset", tags=["TTT"])
    async def ttt_reset(req: dict[str, Any]):
        session_id = req.get("session_id")
        if not isinstance(session_id, str):
            raise HTTPException(status_code=422, detail="session_id is required")
        if runtime.ttt_engine is None:
            raise HTTPException(status_code=503, detail="TTT is not enabled")
        runtime.ttt_engine.end_request(session_id)
        return {"session_id": session_id, "status": "reset"}

    @router.get("/mcp/tools", tags=["MCP Tools"])
    async def mcp_tools():
        if runtime.mcp_layer is None:
            return {"tools": [], "connected_servers": [], "enabled": False}
        return {"tools": runtime.mcp_layer.list_tools(), "connected_servers": runtime.mcp_layer.connected_servers, "enabled": True}

    @router.post("/mcp/server/register", tags=["MCP Tools"])
    async def mcp_register(req: MCPRegisterRequest):
        if runtime.mcp_layer is None:
            raise HTTPException(status_code=503, detail="MCP is not enabled by the loaded AEG")
        command = req.command or req.server_id
        connect = getattr(runtime.mcp_layer, "add_server", None)
        if not callable(connect):
            raise HTTPException(status_code=503, detail="MCP server registration is unavailable")
        connected = connect(
            req.server_id,
            transport=req.transport,
            endpoint=req.endpoint,
            command=command,
        )
        if not connected:
            raise HTTPException(status_code=502, detail=f"MCP server {req.server_id!r} failed to connect")
        return {"server_id": req.server_id, "connected": True, "tools": runtime.mcp_layer.list_tools()}

    @router.get("/green/metrics", tags=["Green Inference"])
    async def green_metrics():
        return runtime.green_power_manager.get_status() if runtime.green_power_manager is not None else {"enabled": False}

    @router.get("/green/carbon_intensity", tags=["Green Inference"])
    async def green_carbon_intensity():
        if runtime.green_power_manager is None:
            return {"enabled": False, "regions": {}}
        return {"enabled": True, "status": runtime.green_power_manager.get_status()}

    @router.post("/green/route", tags=["Green Inference"])
    async def green_route(req: GreenRouteRequest):
        if runtime.green_power_manager is None:
            raise HTTPException(status_code=503, detail="green power manager is not enabled")
        return {"region": runtime.green_power_manager.select_region(req.regions, req.latency_deadline_s)}

    @router.get("/tee/attestation", tags=["Confidential Inference"])
    async def tee_attestation():
        report = runtime.get_attestation_report()
        if not report.get("enabled", False):
            raise HTTPException(status_code=503, detail=report.get("reason", "TEE unavailable"))
        return report

    @router.post("/tee/verify", tags=["Confidential Inference"])
    async def tee_verify(report: dict[str, Any]):
        if runtime.tee_manager is None:
            raise HTTPException(status_code=503, detail="TEE unavailable")
        expected = runtime.tee_manager.get_attestation_report()
        if not expected.get("enclave_initialized", False) or not expected.get("hardware_backed", False):
            raise HTTPException(
                status_code=503,
                detail="hardware-backed TEE attestation is unavailable",
            )
        if report.get("model_hash") != expected.get("model_hash"):
            raise HTTPException(status_code=400, detail="attestation model hash mismatch")
        if report.get("token") != expected.get("token"):
            raise HTTPException(status_code=400, detail="attestation token mismatch")
        return {
            "verified": True,
            "model_hash": expected.get("model_hash"),
            "backend": expected.get("backend"),
        }

    @router.get("/tee/status", tags=["Confidential Inference"])
    async def tee_status():
        if runtime.tee_manager is None:
            return {"enabled": False, "hardware_backed": False}
        report = runtime.tee_manager.get_attestation_report()
        return {
            "enabled": bool(runtime.tee_manager.is_initialized() and report.get("hardware_backed", False)),
            **report,
        }

    @router.get("/hardware/rubin", tags=["System"])
    async def hardware_rubin():
        from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
        profile = HardwareProfile.from_target_id("cuda_sm120")
        if profile is None:
            raise HTTPException(status_code=404, detail="Rubin sm120 profile unavailable")
        return profile.to_dict()

    @router.post("/kernels/rubin/profile", tags=["System"])
    async def rubin_profile(payload: dict[str, Any]):
        if runtime.fingerprint.target_id != "cuda_sm120":
            raise HTTPException(status_code=503, detail="Rubin kernel profiling requires cuda_sm120 hardware")
        # A profile request must not be reported as accepted until a real
        # Rubin profiler and kernel execution backend are connected.
        raise HTTPException(
            status_code=501,
            detail="Rubin kernel profiling backend is not implemented in this runtime",
        )

    @router.post("/video/generate", tags=["Video"])
    async def video_generate(req: VideoRequest):
        job_id = str(uuid.uuid4())
        try:
            response = runtime.generate_video(
                req.model, req.video_path, req.prompt, req.compression, req.max_visual_tokens
            )
            result = {"job_id": job_id, "status": "succeeded", "response": response.to_dict()}
            video_jobs[job_id] = result
            return result
        except Exception as exc:
            video_jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}
            raise HTTPException(status_code=501, detail={"job_id": job_id, "error": str(exc)})

    @router.get("/video/{job_id}/stats", tags=["Video"])
    async def video_stats(job_id: str):
        result = video_jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"video job {job_id!r} not found")
        response = result.get("response")
        if isinstance(response, dict):
            metrics = response.get("metrics", {})
            return {
                "job_id": job_id,
                "status": result["status"],
                "visual_tokens_used": metrics.get("visual_tokens_used"),
                "compression_ratio": metrics.get("video_compression_ratio"),
                "error": result.get("error"),
            }
        return {"job_id": job_id, "status": result["status"], "error": result.get("error")}

    @router.get("/cache/semantic/stats", tags=["Caching"])
    async def semantic_cache_stats():
        return runtime.semantic_cache_stats()

    @router.post("/cache/semantic/flush", tags=["Caching"])
    async def semantic_cache_flush():
        runtime._init_v5_layers()
        cache = getattr(runtime, "_semantic_cache", None)
        if cache is None:
            raise HTTPException(status_code=503, detail="semantic cache unavailable")
        return {"removed": cache.flush()}

    @router.post("/cache/semantic/bypass", tags=["Caching"])
    async def semantic_cache_bypass(payload: dict[str, Any]):
        runtime._init_v5_layers()
        cache = getattr(runtime, "_semantic_cache", None)
        if cache is None:
            raise HTTPException(status_code=503, detail="semantic cache unavailable")
        model = payload.get("model")
        prompt = payload.get("prompt")
        if not isinstance(model, str) or not model:
            raise HTTPException(status_code=422, detail="model is required")
        if not isinstance(prompt, str) or not prompt:
            raise HTTPException(status_code=422, detail="prompt is required")
        try:
            response = runtime.generate(
                model,
                prompt,
                max_tokens=payload.get("max_tokens"),
                temperature=payload.get("temperature"),
                top_p=payload.get("top_p"),
                cache_bypass=True,
            )
            return {
                "bypass": True,
                "prompt_hash": cache._hash(cache._normalize(prompt), model, {}),
                "response": response.to_dict(),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/train/grpo/start", tags=["Training"])
    async def grpo_start(req: GRPORequest):
        job_id = str(uuid.uuid4())
        try:
            result = runtime.grpo_train_step(
                req.model,
                req.prompts,
                req.group_size,
                req.domain,
                req.learning_rate,
                max_tokens=req.max_tokens,
            )
            if result.get("status") == "failed":
                grpo_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "failed",
                    "result": result,
                }
                raise HTTPException(
                    status_code=501,
                    detail={"job_id": job_id, "error": result.get("error", "GRPO unavailable")},
                )
            grpo_jobs[job_id] = {"job_id": job_id, "status": result.get("status", "unknown"), "result": result}
            return grpo_jobs[job_id]
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            grpo_jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(exc)}
            raise HTTPException(status_code=400, detail={"job_id": job_id, "error": str(exc)})

    @router.get("/train/grpo/{job_id}", tags=["Training"])
    async def grpo_status(job_id: str):
        result = grpo_jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"GRPO job {job_id!r} not found")
        return result

    @router.post("/train/grpo/verify", tags=["Training"])
    async def grpo_verify(payload: dict[str, Any]):
        response = payload.get("response")
        if not isinstance(response, str) or not response:
            raise HTTPException(status_code=422, detail="response is required")
        domain = payload.get("domain", payload.get("verifier_domain", "math"))
        if not isinstance(domain, str):
            raise HTTPException(status_code=422, detail="domain must be a string")
        try:
            from aether.compiler.stage2_optimizer.pass22_rlvr_verifier import GRPOTrainer

            reward = GRPOTrainer().verify_response(
                response,
                domain=domain,
                ground_truth=payload.get("ground_truth"),
                test_code=payload.get("test_code"),
            )
            return {"status": "verified", "domain": domain, "reward": reward}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/kv/transfer/stats", tags=["KV"])
    async def kv_transfer_stats():
        return runtime.kv_transfer_stats()

    @router.get("/kv/cxl/pool", tags=["KV"])
    async def cxl_pool():
        return runtime.cxl_pool_status()

    @router.post("/kv/cxl/defrag", tags=["KV"])
    async def cxl_defrag():
        runtime._init_v5_layers()
        pool = getattr(runtime, "_cxl_pool", None)
        if pool is None:
            raise HTTPException(status_code=503, detail="CXL pool is not configured")
        return pool.defragment()


    @router.get("/hardware", tags=["System"])
    async def hardware():
        """Return hardware fingerprint with v4.0 fields (TEE, ternary, MXFP6, RISC-V NPU)."""
        hw = runtime.hardware()
        # Attach additional v4.0 profile fields if available
        try:
            from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
            profile = HardwareProfile.from_target_id(hw.get("target_id", "cpu_avx512"))
            if profile:
                hw.update({
                    "supports_fp4": profile.supports_fp4,
                    "supports_fp8": profile.supports_fp8,
                    "supports_ternary": profile.supports_ternary,
                    "supports_mxfp6": profile.supports_mxfp6,
                    "supports_tee": profile.supports_tee,
                    "is_riscv_npu": profile.is_riscv_npu,
                    "flops_fp4": profile.flops_fp4,
                    "nvlink_bandwidth_gb_s": profile.nvlink_bandwidth_gb_s,
                    "tdp_watts": profile.tdp_watts,
                })
        except Exception:
            pass
        return hw

    @router.get("/kernels", tags=["System"])
    async def kernels():
        """Return active kernel targets."""
        from aether.kernels.native_cpu import get_native_kernels

        native = get_native_kernels()
        return {
            "target": runtime.fingerprint.target_id,
            "native_cpu": {
                "loaded": native.is_native,
                "toolchain_detected": native.toolchain is not None,
                "library": str(native.library_path) if native.library_path else None,
                "symbols": native.available_kernels() if native.is_native else [],
            },
        }

    @router.post("/kernels/generate", tags=["System"])
    async def generate_kernel(req: KernelGenerateRequest):
        """Generate a real executable kernel or return a controlled capability error."""
        from aether.compiler.stage3_targeting.kernel_emitter import KernelEmitter
        from aether.core.exceptions import KernelError

        try:
            artifact = KernelEmitter(req.target).emit_executable(req.op_name)
        except KernelError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        record = artifact.to_dict()
        record["verified"] = bool(artifact.artifact_path.is_file() and artifact.sha256)
        kernel_artifacts[f"{req.target}/{req.op_name}"] = record
        return record

    @router.get("/kernels/{name:path}/verified", tags=["System"])
    async def verify_kernel(name: str):
        """Verify that a previously generated kernel artifact still matches its hash."""
        import hashlib

        record = kernel_artifacts.get(name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"kernel {name!r} has not been generated")
        artifact_path = Path(record["artifact_path"])
        if not artifact_path.is_file():
            raise HTTPException(status_code=410, detail="kernel artifact no longer exists")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        result = {**record, "verified": digest == record.get("sha256"), "current_sha256": digest}
        if not result["verified"]:
            raise HTTPException(status_code=409, detail=result)
        return result

    @router.get("/metrics", tags=["System"])
    async def metrics():
        """Return Prometheus-compatible metrics."""
        return {
            "runtime_up": 1,
            "loaded_models": len(runtime._loaded_models),  # type: ignore[attr-defined]
            "kv_cache_blocks": runtime.kv_cache.block_count,
            "kv_cache_hit_rate": runtime.kv_cache.hit_rate(),
        }

    # ── v4.0 NEW Routes ────────────────────────────────────────────────────────

    @router.post("/tools/call", tags=["MCP Tools"])
    async def mcp_tool_call(req: MCPToolCallRequest):
        """Call an MCP tool via the R6 MCP Native Integration layer.

        The tool_id is in the format 'server_name/tool_name'.
        Returns the raw tool result as returned by the MCP server.

        Research basis: Model Context Protocol v1.0 spec (Anthropic 2024-2026);
        PRD §19 Runtime R6 MCP Native Integration Layer.
        """
        try:
            if hasattr(runtime, "mcp_layer") and runtime.mcp_layer is not None:
                result = runtime.mcp_layer.call_tool(req.tool_id, req.arguments)
                if not isinstance(result, dict):
                    raise ValueError("MCP layer returned a non-object result")
                tool_error = bool(result.get("isError", False))
                return {
                    "tool_id": req.tool_id,
                    "result": result,
                    "success": not tool_error,
                }
            # Fallback: MCP layer not initialized
            return {
                "tool_id": req.tool_id,
                "result": None,
                "success": False,
                "error": "MCP integration layer not initialized. Enable with --enable-mcp.",
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"MCP tool call failed: {exc}")

    @router.post("/grammar/compile", tags=["Structured Output"])
    async def compile_grammar(req: GrammarCompileRequest):
        """Pre-compile a grammar FSM for zero-overhead structured output.

        Compiles a JSON schema, regex, EBNF, or OpenAI tool call grammar into a
        finite-state machine (FSM) binary stored in the model's AEG structured_output/
        directory. The FSM is then used by R3 Grammar FSM Engine at decode time.

        Research basis: XGrammar (MLC 2026), LLGuidance (MSR 2026), Outlines 2026;
        PRD §16 Pass 11: Grammar-Guided Constraint Compiler.
        """
        try:
            result: dict[str, Any] = {
                "grammar_name": req.grammar_name,
                "grammar_type": req.grammar_type,
                "status": "compiled",
                "states": 0,
                "transitions": 0,
            }
            if hasattr(runtime, "grammar_engine") and runtime.grammar_engine is not None:
                fsm_info = runtime.grammar_engine.compile(
                    grammar_name=req.grammar_name,
                    grammar_type=req.grammar_type,
                    grammar_spec=req.grammar_spec,
                    model=req.model,
                )
                result.update(fsm_info)
            else:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Grammar compilation requires a loaded tokenizer-aware grammar backend; "
                        "the request was not queued or accepted"
                    ),
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Grammar compilation failed: {exc}")

    @router.get("/grammar/list", tags=["Structured Output"])
    async def list_grammars():
        """List all pre-compiled grammar FSMs.

        Returns the grammar manifest from the model's AEG structured_output/ directory.
        """
        try:
            if hasattr(runtime, "grammar_engine") and runtime.grammar_engine is not None:
                grammars = runtime.grammar_engine.list_grammars()
            else:
                grammars = []
            return {"grammars": grammars, "count": len(grammars)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/models/{name:path}/merge", tags=["Model Management"])
    async def merge_model(name: str, req: ModelMergeRequest):
        """Apply model merging task vectors to a compiled model.

        Uses task arithmetic (or TIES/DARE/Evolutionary) to combine a base model
        with multiple task-specific delta-weight vectors. The merged result is a
        single AEG artifact that performs multiple tasks at single-model inference cost.

        Research basis: Task Arithmetic (Ilharco ICLR 2023), FREE-Merging 2026,
        Evolutionary Model Merge 2026; PRD §17 Pass 12: Model Merging.
        """
        try:
            if hasattr(runtime, "merge"):
                result = runtime.merge(
                    model_id=name,
                    task_vectors=req.task_vectors,
                    method=req.merge_method,
                    density=req.density,
                )
                return {
                    "model": name,
                    "status": "merged",
                    "method": req.merge_method,
                    "task_count": len(req.task_vectors),
                    "result": result,
                }
            raise HTTPException(
                status_code=501,
                detail="Model merging is not available in this runtime build",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/models/{name:path}/ttt", tags=["Model Management"])
    async def ttt_update(name: str, req: TTTUpdateRequest):
        """Apply Test-Time Training (TTT) fast-weight update for a session.

        Runs micro-gradient-descent on the pre-allocated fast-weight parameter slots
        (Pass 13) using the provided context text. Adapts model behavior to the
        domain without full recompilation.

        Research basis: In-Place TTT (arXiv 2026), VDS-TTT (NeurIPS 2026),
        SDFT 2026; PRD §18 Pass 13: TTT Fast-Weight Injection.
        """
        try:
            if hasattr(runtime, "ttt_engine") and runtime.ttt_engine is not None:
                result = runtime.ttt_engine.update(
                    session_id=req.session_id,
                    context=req.context,
                    learning_rate=req.learning_rate,
                    max_steps=req.max_steps,
                )
                return {
                    "session_id": req.session_id,
                    "model": name,
                    "status": "updated",
                    "steps_completed": result.get("steps", req.max_steps),
                    "loss": result.get("loss", 0.0),
                }
            return {
                "session_id": req.session_id,
                "model": name,
                "status": "unsupported",
                "error": "TTT engine not initialized. Compile model with enable_ttt=True.",
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/targets", tags=["System"])
    async def list_targets():
        """List all supported hardware targets with v4.0 profile details.

        Returns all target IDs from SUPPORTED_TARGET_IDS with their hardware
        capabilities: FP4, FP8, TEE, ternary, MXFP6, RISC-V NPU flags,
        NVLink bandwidth, and TDP.
        """
        from aether.core.constants import SUPPORTED_TARGETS
        from aether.compiler.stage3_targeting.hardware_profile import (
            HardwareProfile, _TARGET_PROFILES,
        )
        targets = []
        for tid, description in SUPPORTED_TARGETS.items():
            profile = HardwareProfile.from_target_id(tid)
            entry: dict[str, Any] = {
                "target_id": tid,
                "description": description,
            }
            if profile:
                entry.update(profile.to_dict())
            targets.append(entry)
        return {
            "targets": targets,
            "count": len(targets),
            "aeg_format_version": "AEG/2.0",
        }

    @router.get("/targets/{target_id}", tags=["System"])
    async def get_target(target_id: str):
        """Get full hardware profile for a specific target ID."""
        from aether.core.constants import SUPPORTED_TARGET_IDS
        if target_id not in SUPPORTED_TARGET_IDS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown target '{target_id}'. Use GET /v1/targets to list valid targets.",
            )
        from aether.compiler.stage3_targeting.hardware_profile import HardwareProfile
        profile = HardwareProfile.from_target_id(target_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"No profile data for target '{target_id}'")
        return profile.to_dict()

    @router.get("/green/status", tags=["Green Inference"])
    async def green_status():
        """Return current carbon intensity and energy status.

        Used by R7 Green Power Manager to provide real-time visibility into
        the carbon routing and DVFS state.

        Research basis: MELODI 2026, CodeCarbon 2026, DVFS arXiv 2025;
        PRD §20 Runtime R7: Green Inference Power Manager.
        """
        try:
            if hasattr(runtime, "green_power_manager") and runtime.green_power_manager is not None:
                status = runtime.green_power_manager.get_status()
                return {"status": "active", **status}
            # Return a sensible default when not initialized
            return {
                "status": "inactive",
                "carbon_intensity_gco2_kwh": None,
                "current_region": None,
                "dvfs_active": False,
                "power_budget_watts": None,
                "estimated_co2_per_token_mg": None,
                "note": "Green power manager not initialized. Compile with enable_green=True.",
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/tee/session", tags=["Confidential Inference"])
    async def create_tee_session(req: TEESessionRequest):
        """Start a Trusted Execution Environment (TEE) confidential inference session.

        Creates an encrypted enclave session for the specified model.
        All inference in this session runs inside a hardware-attested TEE
        (NVIDIA CC mode, Intel TDX, or AMD SEV-SNP).

        Research basis: Intel TDX + NVIDIA CC Joint Paper 2026, Tinfoil Red Hat 2026;
        PRD §21 Runtime R8: Confidential Inference TEE Runtime.
        """
        session_id = str(uuid.uuid4())
        try:
            if hasattr(runtime, "tee_manager") and runtime.tee_manager is not None:
                session_info = runtime.tee_manager.create_session(
                    model=req.model,
                    tee_backend=req.tee_backend,
                    seal_weights=req.seal_weights,
                    attestation_required=req.attestation_required,
                    session_id=session_id,
                )
                return {
                    "session_id": session_id,
                    "status": "active",
                    "tee_backend": session_info.get("tee_backend", req.tee_backend),
                    "attestation_report": session_info.get("attestation_report"),
                    "enclave_created_at": time.time(),
                }
            raise HTTPException(
                status_code=503,
                detail=(
                    "TEE manager not initialized. Target must be cuda_sm100_tee or "
                    "a host with Intel TDX/AMD SEV-SNP support."
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TEE session creation failed: {exc}")

    @router.delete("/tee/session/{session_id}", tags=["Confidential Inference"])
    async def close_tee_session(session_id: str):
        """Close a TEE confidential inference session and destroy the enclave."""
        try:
            if hasattr(runtime, "tee_manager") and runtime.tee_manager is not None:
                runtime.tee_manager.close_session(session_id)
                return {"session_id": session_id, "status": "closed"}
            return {"session_id": session_id, "status": "not_found"}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/models/{name:path}/sub2bit", tags=["Quantization"])
    async def sub2bit_report(name: str):
        """Return measured quantization storage data for an AEG artifact."""
        try:
            return runtime.quantization_report(name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # These catch-all model routes must be registered last.  FastAPI matches
    # path parameters in declaration order, and placing them earlier would
    # swallow /models/{name}/graph, /merge, and /ttt requests.
    @router.get("/models/{name:path}", tags=["Model Management"])
    async def get_model(name: str):
        try:
            return runtime.info(name)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.delete("/models/{name:path}", tags=["Model Management"])
    async def delete_model(name: str):
        try:
            runtime.remove(name)
            return {"status": "success", "model": name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
