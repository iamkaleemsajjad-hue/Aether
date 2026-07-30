# Aether Research Foundation

This document maps the research papers and open-source projects that inform Aether's design decisions.

## Compiler and Intermediate Representations

| Paper / Project | Year | Aether Feature |
|-------------------|------|----------------|
| MLIR: A Compiler Infrastructure for the End of Moore's Law | 2021 | AEG-IR dialect design; multi-level lowering; pass infrastructure |
| IREE: Intermediate Representation Execution Environment | 2022+ | Hardware-universal runtime; HAL abstraction; MLIR lowering pipeline |
| StableHLO: Portability and Stability for ML Compilers | 2023+ | AEG-IR versioning and stability guarantee |
| Meta LLM Compiler: Foundation Models of Compiler Optimization | 2024 | AI-driven pass ordering and optimization selection |
| Modular MAX Graph Compiler | 2024+ | Graph compilation performance reference; operator fusion benchmarks |
| ClusterFusion: Intra-Kernel Communication for Transformer Fusion | 2025 | Megakernel fusion; ClusterReduce/ClusterGather primitives |

## Speculative Decoding and Tree Attention

| Paper | Year | Aether Feature |
|-------|------|----------------|
| SpecInfer: Tree-based Speculative Inference | 2023 | Tree speculation foundation |
| OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure | 2024 | Adaptive draft tree construction |
| DeFT: Decoding with Flash Tree-Attention | 2025 | KV-Guided Grouping; tree-masked attention kernel |
| JetSpec: Scaling Speculative Decoding | 2026 | Causal parallel draft heads |
| EDD: Effective Draft Decoder via Soft Prompts | 2025 | Higher-quality draft generation |
| PCT: Pruned Candidate Trees | 2025 | Dynamic tree pruning |
| EAGLE-3: Scalable Speculative Decoding | 2025 | Draft model architecture |

## KV Cache and Memory Management

| Paper | Year | Aether Feature |
|-------|------|----------------|
| PagedAttention: Efficient Memory Management for LLM Serving | 2023 | Paged KV block management |
| SGLang: Efficient Execution of Structured LM Programs | 2024 | RadixAttention prefix cache |
| DistServe: Disaggregating Prefill and Decoding | 2024 | Disaggregated scheduler |
| Mooncake: A KVCache-centric Disaggregated Architecture | 2024 | KV-centric disaggregation; production results |
| EvolKV: Evolutionary KV Cache Optimization | 2025 | Adaptive KV allocation |
| FlexGen: High-Throughput Inference with a Single GPU | 2023 | NVMe KV cache offloading |
| LoopServe: Multi-Turn KV Cache Reuse | 2025 | Cross-session KV sharing |

## Quantization and Mixed Precision

| Paper | Year | Aether Feature |
|-------|------|----------------|
| GPTQ: Accurate Post-Training Quantization | 2022 | Quantization sensitivity reference |
| AWQ: Activation-aware Weight Quantization | 2023 | Activation-weighted quantization |
| AutoMixQ: Automated Mixed-Precision Quantization | 2025 | Sensitivity analysis pass design |
| AMQ: Accurate Mixed-Precision Quantization | 2025 | Quality benchmarks |
| MoQAE: Mixture of Quantization-Aware Experts | 2025 | Per-expert precision |
| ExLlamaV2 | 2024 | INT4 GEMM kernel integration |

## MoE Inference Optimization

| Paper | Year | Aether Feature |
|-------|------|----------------|
| MoE-Infinity: Offloading-Efficient MoE Serving | 2025 | Expert offload tiering |
| CommitMoE: Expert Prefetching for Memory-Constrained Serving | 2025 | Expert prefetch scheduling |
| FinDEP: Fine-Grained Disaggregated Expert Parallelism | 2025 | Expert compute/communication overlap |
| DynaMoE: Dynamic Expert Allocation | 2025 | Threshold-based routing |
| DA-MoE: Attention-Guided Dynamic Expert Allocation | 2025 | Token importance-based routing |
| Intra-Expert Sparsity Analysis | 2025 | Dead-channel skipping |

## Parallelism and Distributed Inference

| Paper | Year | Aether Feature |
|-------|------|----------------|
| Alpa: Automating Inter/Intra-Operator Parallelism | 2022 | Parallelism solver design |
| Megatron-LM: Training Multi-Billion Parameter Models | 2019–2025 | TP/PP/DP/EP/CP primitives |
| Seesaw: Dynamic Model Re-sharding for LLM Inference | 2025 | Stage-aware re-sharding |
| Ring Attention / Ulysses Context Parallelism | 2023–2024 | Long-context parallelism |
| Splitwise: Efficient Generative LLM Inference via Phase Splitting | 2023 | Prefill/decode disaggregation analysis |

## Open-Source Projects Studied

| Project | What Aether Learns | What Aether Does Differently |
|---------|--------------------|------------------------------|
| vLLM | PagedAttention; continuous batching | Owns the computation graph; compiles to AEG |
| SGLang | RadixAttention; prefix cache | AEG-IR carries radix hints at compile time |
| TensorRT-LLM | Kernel fusion; FP8 GEMM | Open-source; hardware-universal; open format |
| llama.cpp | GGUF format; cross-platform | AEG supersedes GGUF as distribution format |
| MLX | Apple Silicon native; unified memory | AEG ingests MLX; Metal backends emitted |
| ONNX Runtime | Execution Provider model | AEG-IR is LLM-specialized |
| IREE | MLIR lowering; hardware universality | LLM-specialized ops; developer UX; Aether Hub |
| Modular MAX | MLIR-based graph compilation | Open-source; open format; community |
| Ollama | Docker-like UX | Compilation, not wrapping |
| NVIDIA Dynamo | Disaggregated serving | Not NVIDIA-exclusive; open runtime |
