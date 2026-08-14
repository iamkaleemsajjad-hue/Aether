"""Type stubs for the Aether Runtime public SDK.

This file is auto-generated from the implementation; do not edit by hand.
Regenerate with: python scripts/generate_stubs.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Iterator

# ---------------------------------------------------------------------------
# GenerationMetrics
# ---------------------------------------------------------------------------

class GenerationMetrics:
    ttft_ms: float
    throughput_tps: float
    total_ms: float
    prompt_tokens: int
    completion_tokens: int

# ---------------------------------------------------------------------------
# GenerationResponse / StreamChunk
# ---------------------------------------------------------------------------

class StreamChunk:
    text: str
    finish_reason: str | None
    index: int

class GenerationResponse:
    text: str
    usage: dict[str, int]
    metrics: GenerationMetrics
    finish_reason: str

# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

class QualityReport:
    model_id: str
    perplexity_delta: float
    quality_budget: float
    budget_met: bool
    layer_scores: dict[str, float]

# ---------------------------------------------------------------------------
# CompilationPlan
# ---------------------------------------------------------------------------

class CompilationPlan:
    model_id: str
    target_hardware: list[str]
    estimated_size_mb: float
    passes: list[str]
    precision_map: dict[str, str]

# ---------------------------------------------------------------------------
# AEGPackage
# ---------------------------------------------------------------------------

class AEGPackage:
    aeg_path: Path
    format_version: str
    architecture: Any
    precision_map: dict[str, str]
    manifest: dict[str, Any]
    size_mb: float

    @classmethod
    def load(cls, path: str | Path) -> AEGPackage: ...
    def verify(self) -> None: ...

# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

class RuntimeConfig:
    optimize_for: str
    speculative_decoding: bool
    prefill_chunk_size: int
    max_batch_size: int
    kv_cache_dtype: str
    dynamic_precision: bool
    slo_ttft_ms: float | None
    slo_throughput_tps: float | None
    enable_grammar: bool
    enable_mcp: bool
    mcp_servers: list[Any]
    enable_green: bool
    target_carbon_intensity: float | None
    otlp_endpoint: str | None
    metrics_port: int

    def __init__(
        self,
        optimize_for: str = "latency",
        speculative_decoding: bool = True,
        prefill_chunk_size: int = 2048,
        max_batch_size: int = 256,
        kv_cache_dtype: str = "fp8",
        dynamic_precision: bool = True,
        slo_ttft_ms: float | None = None,
        slo_throughput_tps: float | None = None,
        enable_grammar: bool = True,
        enable_mcp: bool = True,
        mcp_servers: list[Any] | None = None,
        enable_green: bool = False,
        target_carbon_intensity: float | None = None,
        otlp_endpoint: str | None = None,
        metrics_port: int = 9090,
    ) -> None: ...

# ---------------------------------------------------------------------------
# CompilerConfig
# ---------------------------------------------------------------------------

class CompilerConfig:
    quality_budget: float
    calibration_dataset: str
    calibration_samples: int
    targets: list[str]
    optimization_level: int
    enable_fusion: bool
    enable_sensitivity: bool
    enable_precision_assignment: bool
    enable_kv_cache_structuring: bool
    enable_moe_routing: bool
    enable_parallelism_discovery: bool
    enable_reasoning_graph: bool
    enable_sparse_attention: bool
    enable_pruning: bool
    upload_kernels: bool
    hub_url: str

    def __init__(
        self,
        quality_budget: float = 0.02,
        calibration_dataset: str = "wikitext-2",
        calibration_samples: int = 512,
        targets: list[str] | None = None,
        optimization_level: int = 2,
        enable_fusion: bool = True,
        enable_sensitivity: bool = True,
        enable_precision_assignment: bool = True,
        enable_kv_cache_structuring: bool = True,
        enable_moe_routing: bool = True,
        enable_parallelism_discovery: bool = True,
        enable_reasoning_graph: bool = False,
        enable_sparse_attention: bool = False,
        enable_pruning: bool = False,
        upload_kernels: bool = False,
        hub_url: str = "https://hub.aether.dev",
    ) -> None: ...

# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class Compiler:
    def __init__(self, config: CompilerConfig | None = None) -> None: ...
    def plan(
        self,
        model_id: str,
        hardware: list[str] | None = None,
    ) -> CompilationPlan: ...
    def compile(
        self,
        model_id: str,
        targets: list[str] | None = None,
        quality_budget: float | None = None,
        calibration_dataset: str | None = None,
        output_path: str | Path | None = None,
    ) -> AEGPackage: ...
    def quality_report(self, aeg: AEGPackage) -> QualityReport: ...

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class Runtime:
    def __init__(self, config: RuntimeConfig | None = None) -> None: ...

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stream: bool = False,
        stop: list[str] | None = None,
        adapter_id: str | None = None,
        grammar: str | None = None,
        seed: int | None = None,
    ) -> GenerationResponse | Iterator[StreamChunk]: ...

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stream: bool = False,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResponse | Iterator[StreamChunk]: ...

    def embed(
        self,
        model_id: str,
        input: str | list[str],
    ) -> list[list[float]]: ...

    def rerank(
        self,
        model_id: str,
        query: str,
        documents: list[str],
    ) -> list[dict[str, Any]]: ...

    def transcribe(
        self,
        model_id: str,
        audio: str | Path,
        language: str | None = None,
    ) -> str: ...

    def benchmark(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 128,
    ) -> dict[str, float]: ...

    def merge(
        self,
        model_a: str | AEGPackage,
        model_b: str | AEGPackage,
        alpha: float = 0.5,
        method: str = "slerp",
    ) -> AEGPackage: ...

    def pull(self, model_id: str) -> AEGPackage: ...
    def list(self) -> list[dict[str, Any]]: ...
    def info(self, model_id: str) -> dict[str, Any]: ...
    def remove(self, model_id: str) -> None: ...

# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def compile(
    model_id: str,
    *,
    output: str | Path | None = None,
    targets: list[str] | None = None,
    quality_budget: float = 0.02,
    calibration_dataset: str = "wikitext-2",
) -> AEGPackage: ...

def run(
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 1.0,
    stream: bool = False,
) -> GenerationResponse: ...
