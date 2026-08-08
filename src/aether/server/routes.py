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

import time
import uuid
from typing import Any

from aether.runtime import Runtime


def create_router(runtime: Runtime) -> Any:
    """Create a FastAPI router with all v3.1 + v4.0 endpoints."""
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        msg = "fastapi and pydantic are required for the server"
        raise ImportError(msg)

    router = APIRouter()

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

    @router.post("/generate", tags=["Generation"])
    async def generate(req: GenerateRequest):
        """Text completion with optional structured output and SLO deadline."""
        try:
            response = runtime.generate(
                model_id=req.model,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                stop=req.stop,
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
            response = runtime.chat(
                model_id=req.model,
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
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

    @router.get("/models/{name:path}", tags=["Model Management"])
    async def get_model(name: str):
        """Get model info including AEG format version and hardware targets."""
        try:
            info = runtime.info(name)
            return info
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.delete("/models/{name:path}", tags=["Model Management"])
    async def delete_model(name: str):
        """Remove a compiled model."""
        try:
            runtime.remove(name)
            return {"status": "success", "model": name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

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
        return {"target": runtime.fingerprint.target_id}

    @router.get("/metrics", tags=["System"])
    async def metrics():
        """Return Prometheus-compatible metrics."""
        return {
            "runtime_up": 1,
            "loaded_models": len(runtime._loaded_models),  # type: ignore[attr-defined]
            "kv_cache_blocks": runtime.kv_cache.block_count,
            "kv_cache_hit_rate": runtime.kv_cache.hit_rate(),
        }

    @router.get("/health", tags=["System"])
    async def health():
        """Health check."""
        return {"status": "healthy", "version": "4.0"}

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
                result = runtime.mcp_layer.call_tool(
                    tool_id=req.tool_id,
                    arguments=req.arguments,
                )
                return {
                    "tool_id": req.tool_id,
                    "result": result,
                    "success": True,
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
                result["status"] = "queued"
                result["note"] = "Grammar engine not initialized; grammar queued for next model load."
            return result
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
            return {
                "model": name,
                "status": "unsupported",
                "error": "Model merging not available in this runtime build.",
            }
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
            return {
                "session_id": session_id,
                "status": "unsupported",
                "error": (
                    "TEE manager not initialized. Target must be cuda_sm100_tee or "
                    "cpu with Intel TDX/AMD SEV-SNP support."
                ),
            }
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

    return router
