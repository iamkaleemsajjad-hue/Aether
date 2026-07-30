# Changelog

All notable changes to Aether Runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and foundational packages.
- `aether.core` package: AEG format, AEG-IR, computation graph, tensor types, and content-addressed hashing.
- `aether.compiler` skeleton: `Compiler` and `CompilerConfig`, dry-run planning, quality reports, and optimizer pass manager.
- Stage 1 ingestion loaders: SafeTensors, GGUF, ONNX, MLX, PyTorch.
- Architecture detector for Llama, Qwen, Gemma, Mistral, DeepSeek, and MoE families.
- Stage 2 optimizer passes 1–6: operator fusion, sensitivity analysis, precision assignment, KV cache structuring, MoE routing, and automatic parallelism discovery.
- Stage 3 targeting: hardware profiles, target registry, backend selector, and kernel emitter.
- Backend plugin system with stable `Backend` interface.
- Reference backend plugins: vLLM, llama.cpp, TensorRT-LLM, MLX, ONNX Runtime, and PyTorch fallback.
- Stage 4 runtime: hardware fingerprinting, model registry, executor, disaggregated prefill/decode scheduler, global KV cache manager, tree-speculative decoding engine, and dynamic precision manager.
- REST server with OpenAI-compatible endpoints and Prometheus metrics.
- `aether` CLI with `compile`, `pull`, `run`, `serve`, `bench`, `info`, `graph`, `list`, `rm`, `hw`, `kernels`, and `logs` commands.
- Python SDK: `Runtime`, `Compiler`, `CompilerConfig`, `RuntimeConfig`, and response objects.
- Aether Hub client with local content-addressed kernel cache.
- Quantization package: formats, sensitivity scoring, precision assignment, and packing utilities.
- MoE package: router, expert manager, sparsity analysis, and planner.
- Parallelism package: planner, mesh, sharding, and distributed primitives.
- Targets package: hardware target profiles and Triton-style kernel templates.
- Utilities: logging, profiling, telemetry, memory management, file I/O, and threading.
- Comprehensive test suite: unit tests, integration tests with real HuggingFace models, and backend discovery tests.
- CI/CD pipelines: Linux, macOS, and Windows test matrix, integration tests, documentation build, and release pipeline.
- Documentation: README, architecture docs, AEG format specification, optimizer passes, runtime, API reference, and research foundation.
- Examples: compile-and-run, serve OpenAI, benchmark, and custom kernel target.
- Benchmark suite with comparison tooling and public leaderboard hooks.

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2026-07-27

### Added
- Initial alpha release of Aether Runtime.
- Foundation compiler and runtime as described in the PRD v2.0.
- Public Python SDK and CLI.
- OpenAI-compatible REST API.
- Research-backed optimizer passes and runtime intelligence.

[Unreleased]: https://github.com/aether-dev/aether-runtime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aether-dev/aether-runtime/releases/tag/v0.1.0
