# Aether Optimizer Passes

Aether's Stage 2 optimizer runs **22 graph-level compiler passes** in sequence.
Each pass is a self-contained transformation that can be enabled or disabled
independently. All 22 passes fire on every `aether compile` invocation at the
default optimization level (O2).

---

## Pass 1: Operator Fusion

Merges sequences of operations into single fused kernels to reduce kernel
launches and eliminate intermediate DRAM round-trips.

Typical fusion patterns:

- `RMSNorm → QKV projection → RoPE` → `fused_qkv_rope_norm`
- `Attention → output projection → residual add` → `fused_attn_out_residual`
- `FFN gate/up/down projections + residual` → `fused_ffn_residual`
- `Softmax → Dropout → MatMul` → `fused_softmax_attn`

Research: ClusterFusion (NeurIPS 2025), FlashAttention-3, TensorRT-LLM kernel fusion.

---

## Pass 2: Sensitivity Analysis

Computes a per-layer sensitivity score using calibration data:

```
sensitivity[L] = Δperplexity / bits_saved(L)
```

Higher sensitivity = layer is more important to preserve precision.
Calibration runs on WikiText-103 (50-sequence sample) by default.

Research: AutoMixQ (2024), AMQ, GPTQ, AWQ sensitivity analysis.

---

## Pass 3: Precision Assignment

Using the sensitivity map, each layer receives an optimal precision:

| Sensitivity | Precision |
|-------------|-----------|
| > 0.9 | BF16 |
| 0.7–0.9 | FP8 or Q6_K |
| 0.4–0.7 | Q4_K_M |
| < 0.4 | Q3_K or IQ3_XS |

Modes: `sensitivity` (default), `uniform`, `manual`.

---

## Pass 4: KV Cache Structuring

Annotates the AEG-IR with explicit KV cache nodes and policies:

- Paged blocks aligned to hardware page size (64 KB default)
- RadixTree prefix hints for shared system prompts
- Memory tier thresholds: GPU HBM → CPU DRAM → NVMe SSD → Aether Hub
- Cross-session sharing policies
- Sub-2-bit KV quantization metadata (Q2K_KV)

Research: PagedAttention (vLLM 2023), SGLang RadixAttention, DistServe, Mooncake.

---

## Pass 5: MoE Expert Routing

For MoE models (Mixtral, DeepSeek-MoE, etc.):

1. Profiles expert activation on calibration set.
2. Classifies experts as hot (>5%), warm (0.1–5%), or cold (<0.1%).
3. Pins hot experts to GPU HBM; stages warm to CPU DRAM; lazy-loads cold from NVMe.
4. Replaces top-K routing with adaptive threshold-based routing (DynaMoE).
5. Emits intra-expert sparsity kernels for dead activation channels.

Research: MoE-Infinity, CommitMoE, FinDEP, DynaMoE.

---

## Pass 6: Automatic Parallelism Discovery

Searches over the strategy space of TP/PP/EP/CP parallelism.
Produces separate prefill and decode sharding plans for 1/2/4/8/16 GPU configurations.

Search space:
- Tensor parallelism: 1, 2, 4, 8
- Pipeline stages: 1, 2, 4
- Expert parallelism (MoE only): 1, 2, 4
- Context parallelism (long context): 1, 2, 4

Research: Alpa, Megatron-LM, Seesaw, Ring Attention.

---

## Pass 7: Reasoning Graph Compiler

Emits `graph/reasoning_graph.aeg-ir` with explicit reasoning metadata:
- Budget/token allocation per reasoning step
- Early-exit conditions and confidence thresholds
- Verification step injection points
- Speculative chain-of-thought (CoT) metadata

Research: DeepSeek-R1, Sky-T1, ARC reasoning (2025).

---

## Pass 8: Sparse Attention

For prompts exceeding the long-context threshold, emits sparse attention patterns:
- MInference vertical-slash pattern
- Block-sparse (sliding window)
- A-shape (dense for first tokens)

Output: `graph/attention_head_patterns.json`
Falls back to dense attention below the threshold.

Research: MInference (Microsoft 2024), Longformer, BigBird.

---

## Pass 9: Pruning and Sparsity

Emits weight sparsity masks using Wanda/SparseGPT-inspired importance scoring:

- **2:4 structured sparsity** for NVIDIA tensor cores (sm80+)
- **Unstructured sparsity** for CPU targets (up to 70% sparsity)
- Sensitive layers receive lower sparsity ratios

Output: `weights/sparsity_masks.json`

Research: SparseGPT (2023), Wanda (2024), 2:4 Structured Pruning.

---

## Pass 10: Speculative Decoding Configuration

Embeds speculative decoding configuration for draft/verify inference:

- Selects optimal draft model (Eagle-2, Medusa-style, or n-gram)
- Configures speculation depth (default: 5 tokens per draft)
- Tunes acceptance threshold per model quality

Research: Eagle (2024), Medusa, SpecInfer, BiLD.

---

## Pass 11: LoRA/PEFT Adapter Merging

Merges LoRA/QLoRA/DoRA adapter weights into the base model graph:

- Fuses `W + A·B` into a single weight matrix when rank ≤ 64
- Validates adapter compatibility (matching hidden dim, attention heads)
- Preserves adapter metadata for multi-LoRA serving

Research: LoRA (Hu et al. 2021), QLoRA (Dettmers 2023), DoRA (2024).

---

## Pass 12: Embedding Optimization

Optimizes the embedding and unembedding (LM head) layers:

- Quantizes vocabulary embeddings to INT8 or lower
- Ties input/output embeddings when weights are shared
- Applies vocab pruning for domain-specific models

---

## Pass 13: Long-Context Adaptation

For models operating above 32K context:

- Applies dynamic NTK RoPE scaling for the configured context length
- Inserts ring attention boundaries for multi-device long-context
- Configures chunked prefill block sizes

Research: YaRN (2023), RoPE NTK scaling, Ring Attention.

---

## Pass 14: Video KV Management

For vision-language models processing video:

- Extracts per-frame KV cache plans
- Inserts temporal compression nodes (deduplicate similar frames)
- Configures cross-frame KV sharing policies

Research: Video LLaVA (2024), Aether AEG/3.0 video payload spec.

---

## Pass 15: Multi-Agent KV Coordination

For multi-agent inference workloads:

- Emits shared prefix KV cache policies across agents
- Configures SwarmKV-style distributed KV sharing
- Inserts inter-agent token visibility boundaries

Research: SwarmKV (2026), vLLM prefix caching.

---

## Pass 16: Grammar-Constrained Decoding

Embeds structured output configuration:

- Compiles JSON schema / regex grammar into a finite-state machine (FSM)
- Emits grammar FSM as `graph/grammar_fsm.bin`
- Configures token masking at each decode step for constrained generation

Research: Outlines (2024), LM-Format-Enforcer, Guidance.

---

## Pass 17: TEE Wrapping

When the target platform supports Intel TDX, AMD SEV-SNP, or NVIDIA Hopper CC:

- Wraps weight loading in a Trusted Execution Environment enclave
- Emits remote attestation metadata into `provenance/tee_attestation.json`
- Ensures weight decryption occurs only inside the enclave

Falls back to plaintext execution with explicit warning when TEE is unavailable.

Research: Intel TDX (2024), AMD SEV-SNP, NVIDIA Hopper Confidential Computing.

---

## Pass 18: MDLM Drafter Injection

For models using MDLM-style diffusion speculative decoding:

- Loads drafter weights from a supplied MDLM checkpoint
- Fuses drafter tokenizer alignment layer
- Configures diffusion timestep schedule for 5-token draft lookahead

Requires: `--drafter-path /path/to/mdlm_drafter.safetensors`

Research: MDLM (2024), DiffuSeq, PLM-based diffusion language models.

---

## Pass 19: SLO-Aware Scheduling Configuration

Embeds per-model Service Level Objective metadata:

- Configures TTFT deadline (default: 2000 ms for streaming, 5000 ms for batch)
- Binds per-tenant priority weights for EDF (Earliest Deadline First) scheduling
- Sets preemption thresholds for long-running requests

Research: Orca (2022), S3 (2023), DejaVu SLO-aware scheduling.

---

## Pass 20: Green Power Optimization

Embeds power-aware scheduling hints:

- Profiles model thermal design power (TDP) per hardware target
- Annotates idle periods for CPU/GPU power gating
- Configures carbon-intensity-aware batch scheduling

Research: Green Scheduling (2024), NVIDIA GPU power state management.

---

## Pass 21: MCP Tool Integration

For agentic models using the Model Context Protocol:

- Validates tool schema compatibility (JSON Schema draft-07)
- Embeds tool call routing tables into `graph/mcp_tool_manifest.json`
- Configures tool-call output parsing and result injection

Research: Model Context Protocol spec (Anthropic 2024), OpenAI function calling.

---

## Pass 22: RLVR Verifier Injection

For GRPO/RLVR-style reinforcement learning from verifiable rewards:

- Injects a verifier head into the model graph for process reward model (PRM) scoring
- Configures GRPO training loop hyperparameters: group size, KL penalty, clip ratio
- Emits `graph/rlvr_config.json` for online RL training integration

Research: DeepSeek-R1 GRPO (2025), RLVR (2025), OpenR reasoning (2025).

---

## Configuration

All 22 passes are controlled via `CompilerConfig`:

```python
from aether import CompilerConfig

config = CompilerConfig(
    # Core passes (O1+)
    enable_fusion=True,
    enable_sensitivity=True,
    enable_precision_assignment=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,            # Auto-detected for MoE models
    enable_parallelism_discovery=True,

    # Reasoning passes (O2+)
    enable_reasoning_graph=True,
    enable_sparse_attention=True,
    enable_pruning=True,

    # Advanced passes (O2+)
    enable_speculative_decoding=True,
    enable_lora_merging=True,           # Auto-detected if adapters present
    enable_long_context=True,           # Auto-detected above 32K ctx

    # Specialized passes (explicit opt-in)
    enable_tee_wrapping=False,          # Requires TEE hardware
    enable_mdlm_drafter=False,          # Requires drafter weights
    enable_rlvr=False,                  # Requires training mode

    # Quality
    calibration_samples=50,
    quality_threshold=0.99,             # Perplexity ratio vs BF16
)
```

Or via CLI:

```bash
# Default (O2 — all 22 passes)
aether compile model.safetensors --target cpu_avx512

# Minimal (O1 — core passes only)
aether compile model.safetensors --target cpu_avx512 --opt-level 1

# With TEE wrapping (requires compatible hardware)
aether compile model.safetensors --target cuda_sm100_tee --enable-tee

# With MDLM drafter
aether compile model.safetensors --target cuda_sm90 \
    --drafter path/to/drafter.safetensors
```

---

## Pass Ordering and Dependencies

```
Pass 1  (Fusion)
  └── Pass 2  (Sensitivity)
        └── Pass 3  (Precision)
              ├── Pass 4  (KV Cache)
              ├── Pass 9  (Pruning)
              └── Pass 11 (LoRA merge)
Pass 5  (MoE) — depends on Pass 3
Pass 6  (Parallelism) — depends on Pass 3, 4, 5
Pass 7  (Reasoning) — independent
Pass 8  (Sparse Attention) — depends on Pass 4
Pass 10 (Speculative) — independent
Pass 12–16 — independent (model family specific)
Pass 17 (TEE) — depends on all weight passes
Pass 18 (MDLM) — depends on Pass 10
Pass 19–21 — independent
Pass 22 (RLVR) — independent
```

---

## Research Foundation

| Pass | Key Research |
|------|-------------|
| 1 | ClusterFusion NeurIPS 2025, FlashAttention-3 |
| 2 | GPTQ, AWQ, AutoMixQ 2024 |
| 3 | Mixed-precision inference, FP8 LLM survey 2024 |
| 4 | PagedAttention, RadixAttention, Mooncake |
| 5 | MoE-Infinity, DynaMoE, CommitMoE |
| 6 | Alpa, Megatron-LM, Ring Attention |
| 7 | DeepSeek-R1, Sky-T1 2025 |
| 8 | MInference Microsoft 2024 |
| 9 | SparseGPT, Wanda, 2:4 Pruning |
| 10 | Eagle 2024, Medusa, SpecInfer |
| 11 | LoRA, QLoRA, DoRA 2024 |
| 13 | YaRN, RoPE NTK, Ring Attention |
| 16 | Outlines, LM-Format-Enforcer |
| 17 | Intel TDX, AMD SEV-SNP 2024 |
| 18 | MDLM 2024, DiffuSeq |
| 22 | DeepSeek-R1 GRPO, RLVR 2025 |

See [research.md](research.md) for the full paper mapping.
