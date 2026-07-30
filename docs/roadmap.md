# Roadmap

This roadmap translates the PRD into implementation gates. A feature is complete only when it has code, tests, docs, and a repeatable validation path.

## Phase 1: Compiler Foundation

- AEG package layout with manifest hashes, graph IR, precision map, kernel metadata, and sharding plans.
- Model ingestion for architecture metadata and graph construction from local or named models.
- Operator fusion pass that annotates or fuses high-value transformer patterns.
- CPU/PyTorch fallback backend so the API works without specialized hardware.
- CLI, Python SDK, OpenAI-compatible API surface, and documentation.

## Phase 2: Optimizer Depth

- Calibration-backed sensitivity analysis with backend logits when available and deterministic proxy fallback.
- Mixed-precision assignment constrained by quality budget and layer sensitivity.
- Paged KV cache metadata and runtime hit-rate metrics.
- Chunked prefill scheduling and decode batching.
- Tree-speculative decoding acceptance tracking and branch pruning.

## Phase 3: Parallelism and Scale

- Cost model for tensor, pipeline, context, and expert parallelism.
- Separate prefill and decode sharding plans stored in AEG.
- Multi-process worker prototype with explicit KV handoff.
- MoE expert tiering with hot, warm, and cold placement policies.
- Benchmark harness for TTFT, TPOT, goodput, MFU, and memory pressure.

## Phase 4: Ecosystem

- Hub client with signed content-addressed kernel artifacts.
- HuggingFace discovery for precompiled AEG variants.
- Backend plugin SDK with conformance tests.
- Release artifacts for wheels, docs, SBOM, and provenance attestations.
- Public benchmark reports with reproducible configs.

## Phase 5: Compiler Platform

- Stable AEG compatibility policy and migration tooling.
- Third-party hardware target onboarding guide.
- External compiler pass API.
- Multimodal graph support for vision-language and audio-language models.
- Enterprise controls for private registries, audit logs, and role-based access.
