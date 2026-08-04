# PRD v4.0 + v5.0 Runtime Layers — API Reference

> **Aether Runtime R1–R12**
> Companion implementation to the PRD v4.0 + v5.0 specification.

---

## Quick Reference

| Layer | Class | Import path | Key method |
|-------|-------|-------------|------------|
| R1 | `PEAGLEEngine` | `aether.runtime.r1_peagle_engine` | `propose(hidden_state, target_fn, ctx_tokens)` |
| R2 | `MultiAgentKVCoordinator` | `aether.runtime.r2_multi_agent_kv` | `register_session(sid, tokens, kv)` |
| R3 | `GrammarFSMEngine` | `aether.runtime.r3_grammar_fsm` | `create_session(grammar_id)` |
| R4 | `SLOScheduler` | `aether.runtime.r4_slo_scheduler` | `submit(rid, tokens, slo_tier)` |
| R5 | `TTTFastWeightEngine` | `aether.runtime.r5_ttt_engine` | `adapt(rid, hidden_states, layer_idx)` |
| R6 | `MCPIntegrationLayer` | `aether.runtime.r6_mcp_integration` | `call_tool(name, arguments)` |
| R7 | `GreenPowerManager` | `aether.runtime.r7_green_power_manager` | `get_dvfs_config(op_id)` |
| R8 | `TEERuntimeManager` | `aether.runtime.r8_tee_manager` | `initialize()` / `enter_kernel(kid)` |
| R9 | `Sub2BitKVWeightCache` | `aether.runtime.r9_sub2bit_kv_cache` | `store_kv_ternary(bid, keys, vals)` |
| R10 | `VideoFrameKVManager` | `aether.runtime.r10_video_kv_manager` | `ingest_frame(idx, tokens, prev)` |
| R11 | `SemanticKVCache` | `aether.runtime.r11_semantic_kv_cache` | `lookup(embedding)` |
| R12 | `RLVRTrainingHarness` | `aether.runtime.r12_rlvr_harness` | `train_step(prompt, ground_truth)` |

---

## R1 — PEAGLEEngine

**File**: `src/aether/runtime/r1_peagle_engine.py`

P-EAGLE (Parallel EAGLE) extends EAGLE-3 speculative decoding with hardware-level
SM partitioning. The draft model runs concurrently with the target model on separate
CUDA SM partitions.

### Modes

| Mode | Description | Typical Speedup |
|------|-------------|-----------------|
| `mtp` | MTP heads from Pass 10 | 1.8–2.5× AR |
| `eagle3` | EAGLE drafter model | 3.0–3.5× AR |
| `mdlm` | MDLM diffusion drafter | 2.5–3.2× AR |
| `hybrid` | MTP + EAGLE interleaved | up to 4.0× AR |

### Usage

```python
from aether.runtime import PEAGLEEngine

engine = PEAGLEEngine(
    draft_K=5,
    mode="mtp",                          # or "eagle3", "mdlm", "hybrid"
    target_acceptance_rate=0.70,
    draft_sm_fraction=0.30,
    mtp_config_path=".aeg/speculation/mtp_config.json",
)

# One decode step.
proposal = engine.propose(
    hidden_state=model_last_hidden,      # Any tensor or list
    target_forward_fn=model.forward,     # Callable[hidden] → logits
    context_tokens=token_ids,
)

print(f"Accepted {proposal.n_accepted}/{len(proposal.draft_tokens)} tokens")
print(f"Acceptance rate: {proposal.acceptance_rate:.2%}")

# Adaptive K adjustment (call after each step).
new_K = engine.adaptive_adjust_K()

# SM partition config for CUDA backend.
sm_cfg = engine.get_sm_partition_config()
```

### Acceptance Criterion

From Leviathan et al. 2023 (speculative decoding):

```
p_accept(x) = min(1, p_target(x) / p_draft(x))
```

When `p_accept ≥ 0.5`, the token is accepted. The engine tracks rolling
acceptance rate over the last 100 steps and adjusts K accordingly.

---

## R2 — MultiAgentKVCoordinator

**File**: `src/aether/runtime/r2_multi_agent_kv.py`

Enables zero-copy KV sharing across agent sessions with identical prefixes.
Implements RadixAttention-style prefix deduplication and copy-on-write for
diverged private tails.

### Usage

```python
from aether.runtime import MultiAgentKVCoordinator

coord = MultiAgentKVCoordinator(
    max_shared_blocks=10_000,
    eviction_policy="lru",
)

# Register agents sharing a system prompt.
sess1 = coord.register_session("agent_1", prefix_tokens=[1, 2, 3, 4], prefix_kv=kv_data)
sess2 = coord.register_session("agent_2", prefix_tokens=[1, 2, 3, 4])  # Reuses sess1's KV.

assert sess1.shared_block_id == sess2.shared_block_id  # Same block.

# After prefix, agents diverge.
coord.append_private_kv("agent_1", new_tokens=[5, 6], new_kv=agent1_kv)
coord.append_private_kv("agent_2", new_tokens=[7, 8], new_kv=agent2_kv)

# Retrieve for attention computation.
shared_kv, private_kv = coord.get_full_kv("agent_1")

# Release when done.
coord.release_session("agent_1")
coord.release_session("agent_2")
```

### Memory Savings

With N agents sharing a K-token system prompt:
- **Without R2**: N × K KV pairs computed and stored.
- **With R2**: 1 × K KV pairs (shared) + N × divergence KV pairs.

---

## R3 — GrammarFSMEngine

**File**: `src/aether/runtime/r3_grammar_fsm.py`

Enforces grammar constraints at decode time by masking invalid tokens via a
pre-compiled FSA. Loads the binary blob written by Pass 11.

### Usage

```python
from aether.runtime import GrammarFSMEngine

fsm = GrammarFSMEngine()

# Load grammar (FSA blob from Pass 11).
fsm.load(".aeg/grammar/fsm.bin", grammar_id="json")

# Per-request session (each request is independent).
session = fsm.create_session("json")

# Decode loop.
for _ in range(max_tokens):
    mask = session.get_token_mask()           # bytearray[ceil(vocab/8)]
    token_id = sample_with_mask(logits, mask) # Your sampler
    next_state = session.advance(token_id)
    if session.is_accepting():
        break                                 # Valid complete output
```

### Token Mask Format

```
mask[i // 8] & (1 << (i % 8)) == 1  →  token i is valid
```

Bit-packed bytearray of length `ceil(vocab_size / 8)`. Can be consumed
directly by CUDA sampling kernels without conversion.

### Performance

- O(1) mask lookup via pre-built state→mask dict.
- < 50 µs overhead per decode step (XGrammar 2026 benchmark).

---

## R4 — SLOScheduler

**File**: `src/aether/runtime/r4_slo_scheduler.py`

Multi-objective scheduler maintaining per-SLO-tier priority queues with
chunked prefill (Sarathi-Serve) for TTFT control.

### SLO Tiers

| Tier | TTFT Deadline | Priority | Use case |
|------|--------------|----------|----------|
| `AGENTIC` | 100 ms | Highest | Tool call responses |
| `LATENCY` | 200 ms | High | Interactive chat |
| `BALANCED` | 1 s | Medium | Mixed workloads |
| `THROUGHPUT` | 30 s | Low | Batch processing |

### Usage

```python
from aether.runtime import SLOScheduler, SLOTier

sched = SLOScheduler(
    max_batch_tokens=8192,
    max_prefill_chunk_tokens=4096,
    scheduling_algo="mlfq",            # or "sjf", "fcfs"
)

# Submit requests.
req = sched.submit(
    "req_001",
    prompt_tokens=512,
    max_new_tokens=1024,
    slo_tier=SLOTier.LATENCY,
)

# Dispatch loop.
while True:
    batch = sched.next_batch(token_budget=8192)
    if not batch:
        break
    results = process_batch(batch)
    sched.record_batch_latency(elapsed_ms=results.latency_ms)
```

### Deadline Boosting

When a request's TTFT deadline is within 50 ms, its priority is automatically
boosted to `-1000` (emergency priority) by `_rebalance_priorities()`.

---

## R5 — TTTFastWeightEngine

**File**: `src/aether/runtime/r5_ttt_engine.py`

Implements In-Place TTT (arXiv 2026) — online adaptation of per-layer fast
weights during inference without storing optimizer state across requests.

### Usage

```python
from aether.runtime import TTTFastWeightEngine

ttt = TTTFastWeightEngine(
    n_layers=32,
    hidden_size=4096,
    rank=16,
    learning_rate=1e-4,
    session_scoped=False,              # True = maintain weights across turns
    ttt_config_path=".aeg/ttt/ttt_config.json",
)

# Before generation.
ttt.begin_request("req_001")

# Adapt to input domain using hidden states.
loss = ttt.adapt("req_001", hidden_states=h_list, layer_idx=-1)

# During generation: get fast weights for layer-level injection.
weights = ttt.get_fast_weights("req_001", layer_idx=5)

# After generation.
ttt.end_request("req_001")
```

### Algorithm (In-Place TTT)

```
h_mean = mean(h_1, ..., h_N)                  # Mean pooling over tokens
mu ← mu - lr × (mu - mean(h_mean))            # LayerNorm shift update
sigma ← sigma - lr × (sigma - std(h_mean))    # LayerNorm scale update
A ← A - lr × ∇_A ||BAh - h||² / hidden_size  # LoRA A gradient
B ← B - lr × ∇_B ||BAh - h||² / hidden_size  # LoRA B gradient
```

---

## R6 — MCPIntegrationLayer

**File**: `src/aether/runtime/r6_mcp_integration.py`

Native MCP (Model Context Protocol) integration with multi-server connection
management and three-strategy tool call detection.

### Usage

```python
from aether.runtime import MCPIntegrationLayer

mcp = MCPIntegrationLayer(timeout_s=30.0, max_concurrent_calls=16)

# Connect to MCP servers.
mcp.add_server("file_system", transport="stdio")
mcp.add_server("web_search", transport="http", endpoint="http://localhost:3001")

# Detect tool call from model output.
tool_call = mcp.detect_tool_call(generated_text)
if tool_call:
    result = mcp.call_tool(tool_call["tool"], tool_call["arguments"])
    # Inject result back into context.

mcp.disconnect_all()
```

### Detection Strategies

1. **JSON pattern**: `{"tool": "name", "arguments": {...}}`
2. **XML tag**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
3. **Function call XML**: `<function_calls><invoke name="...">...</invoke></function_calls>`

---

## R7 — GreenPowerManager

**File**: `src/aether/runtime/r7_green_power_manager.py`

DVFS enforcement, TDP cap throttling, and carbon-aware routing. Loads
the green profile written by Pass 16.

### Usage

```python
from aether.runtime import GreenPowerManager

gpm = GreenPowerManager(
    mode="balanced",                   # "performance" | "balanced" | "eco"
    tdp_cap_w=400.0,
    green_profile_path=".aeg/metadata/green_profile.json",
)

# Get DVFS frequency for an operator.
freq_mhz, voltage_mv = gpm.get_dvfs_config("gemm_layer_5")

# Update with real GPU power reading (e.g., from NVML).
throttled_freq = gpm.update_power_reading(current_power_w=620.0)
if throttled_freq:
    set_gpu_clock(throttled_freq)

# Select greenest region within latency deadline.
region = gpm.select_region(
    available_regions=["us-west", "eu-north", "ap-east"],
    latency_deadline_s=1.0,
)

# Record energy for telemetry.
energy_mj = gpm.estimate_request_energy(n_prompt_tokens=512, n_gen_tokens=1024)
carbon_gco2 = gpm.estimate_carbon(energy_mj)
gpm.record_request(energy_mj, carbon_gco2)
```

### Energy Reduction by Mode

| Mode | Energy Savings | Throughput Impact |
|------|---------------|-------------------|
| `performance` | 0% | 0% |
| `balanced` | ~35% | < 5% |
| `eco` | ~48% | 5–15% |

---

## R8 — TEERuntimeManager

**File**: `src/aether/runtime/r8_tee_manager.py`

TEE enclave lifecycle manager for confidential LLM inference. Wraps each
compute kernel with enter/exit guards and verifies weight integrity.

### Backends

| Backend | Platform | Attestation |
|---------|----------|-------------|
| `nvidia_cc` | NVIDIA H100/H200 CC mode | NVIDIA RATS SDK |
| `intel_tdx` | Intel TDX Trust Domain | TDCALL(TDATTEST) |
| `amd_sev_snp` | AMD SEV-SNP | /dev/sev-guest ioctl |

### Usage

```python
from aether.runtime import TEERuntimeManager

tee = TEERuntimeManager(
    backend="nvidia_cc",
    tee_config_path=".aeg/security/",
    enable_heartbeat=True,
    heartbeat_interval_s=30.0,
)

# Initialize enclave (idempotent).
assert tee.initialize()

# Verify weights before loading.
valid, failed = tee.verify_weights(model.state_dict())
if not valid:
    raise RuntimeError(f"Weight integrity failure: {failed}")

# Kernel dispatch.
for kernel_id, kernel_fn in kernels:
    tee.enter_kernel(kernel_id)
    kernel_fn()
    tee.exit_kernel(kernel_id)

# Remote attestation.
report = tee.get_attestation_report()
```

---

## R9 — Sub2BitKVWeightCache

**File**: `src/aether/runtime/r9_sub2bit_kv_cache.py`

Sub-2-bit KV cache storage (ternary {-1, 0, +1}) and LRU decompressed
weight cache. Loads quantization plan from Pass 19's manifest.

### Ternary Encoding

- **Format**: 4 ternary values packed per byte (2 bits each).
- **Encoding**: `00 → 0`, `01 → +1`, `10 → -1`.
- **Scale**: float32 absmean per vector.
- **Effective bits**: 1.58 bits/weight (log₂(3)).

### Usage

```python
from aether.runtime import Sub2BitKVWeightCache

cache = Sub2BitKVWeightCache(
    weight_cache_budget_gb=4.0,
    sub2bit_manifest_path=".aeg/quantization/sub2bit_manifest.json",
)

# Store KV in ternary format.
cache.store_kv_ternary("block_001", key_vectors, value_vectors)

# Load (decompress) for attention.
keys, values = cache.load_kv("block_001")

# Ternary GEMM.
output = cache.ternary_gemm(W_ternary_packed, scale=2.0, x=activation, out_features=4096, in_features=4096)

# Weight decompression cache.
weights = cache.get_weights("transformer.layer.5.self_attn.q_proj")
if weights is None:
    weights = decompress_from_blob("...")
    cache.store_weights("transformer.layer.5.self_attn.q_proj", weights, size_bytes=8192)
```

---

## R10 — VideoFrameKVManager

**File**: `src/aether/runtime/r10_video_kv_manager.py`

Scene-adaptive KV management for video VLM inference. Implements
StreamingTOM-style bounded KV window with time-decaying importance scoring.

### Usage

```python
from aether.runtime import VideoFrameKVManager

vmgr = VideoFrameKVManager(
    max_kv_slots=512,
    tokens_per_frame_raw=256,
    compression_ratio=0.25,
    scene_change_threshold=0.3,
    decay_rate=0.01,
    compression_plan_path=".aeg/graph/video_compression_plan.json",
)

# Process video frame by frame.
prev_tokens = None
for frame_idx, frame in enumerate(video_frames):
    visual_tokens = vit_encoder(frame)
    slot = vmgr.ingest_frame(
        frame_idx,
        visual_tokens,
        prev_frame_tokens=prev_tokens,
        task_relevance=1.0,
    )
    prev_tokens = visual_tokens

    # Periodic importance decay.
    if frame_idx % 10 == 0:
        vmgr.decay_importance()

# Get tiered attention context for a query frame.
ctx = vmgr.get_attention_context(query_frame_idx=100, recent_window=8)
# ctx["recent"]   → Full KV for last 8 frames.
# ctx["mid_term"] → STC-compressed KV for frames 69–92.
# ctx["summary"]  → Mean-pooled summary tokens for frames 0–68.
```

---

## R11 — SemanticKVCache

**File**: `src/aether/runtime/r11_semantic_kv_cache.py`

HNSW-based semantic cross-request KV deduplication. When a new request is
semantically similar to a previous one (cosine similarity > threshold),
it reuses the cached KV without recomputation.

### Usage

```python
from aether.runtime import SemanticKVCache

skv = SemanticKVCache(
    dim=128,
    similarity_threshold=0.92,
    max_kv_blocks=10_000,
    max_kv_memory_gb=16.0,
)

# Embed the prompt.
embedding = skv.embed_prompt(prompt_token_ids)

# Look up cached KV.
kv_data, block_id, similarity = skv.lookup(embedding)
if kv_data is not None:
    # Cache HIT — reuse KV directly.
    use_kv_from_cache(kv_data)
else:
    # Cache MISS — compute normally, then store.
    kv_data = compute_kv(prompt)
    skv.store("new_block", embedding, kv_data, kv_size_bytes=estimate_size(kv_data))

print(f"Hit rate: {skv.hit_rate():.2%}")
```

### HNSW Acceleration

- Uses `hnswlib` when installed (`pip install hnswlib`): sub-millisecond ANN.
- Falls back to pure-Python linear scan (correct but O(n) per lookup).

---

## R12 — RLVRTrainingHarness

**File**: `src/aether/runtime/r12_rlvr_harness.py`

Full GRPO (Group Relative Policy Optimization) training loop with
verifiable reward backends. Loads config from Pass 22's `rlvr_config.json`.

### Verifier Backends

| Backend | Correctness | Use case |
|---------|------------|----------|
| `sympy` | Symbolic equality | Math, algebra |
| `pytest` | Test pass rate | Code generation |
| `llm_judge` | Semantic overlap | Open-ended QA |
| `human` | Deferred (0.5 neutral) | RLHF annotation |

### GRPO Algorithm

```
For each prompt:
  1. Sample K solutions: {s_1, ..., s_K}
  2. Compute rewards: r_i = verifier(s_i, ground_truth)
  3. K2V sub-task rewards: r_sub_i += 0.1 × subtask_completion(s_i)
  4. Advantages: A_i = (r_i - mean(r)) / (std(r) + ε)
  5. Loss: L = -Σ clip(π/π_old, 1-ε, 1+ε) × A_i
  6. Gradient step via optimizer_step_fn(loss)
```

### Usage

```python
from aether.runtime import RLVRTrainingHarness

harness = RLVRTrainingHarness(
    rlvr_config_path=".aeg/training/rlvr_config.json",
    model_forward_fn=lambda prompt, **kw: model.generate(prompt, **kw),
    optimizer_step_fn=lambda loss: optimizer.step(loss),
)

# One training step.
result = harness.train_step(
    prompt="Compute the derivative of x³ + 2x.",
    ground_truth="3*x**2 + 2",
)

print(f"GRPO loss: {result.loss:.4f}")
print(f"pass@{harness._K}: {result.pass_at_k:.2%}")
print(f"Avg reward: {sum(result.rewards)/len(result.rewards):.3f}")
```

### pass@k Metric

From Chen et al. 2021 (HumanEval):
```
pass@k = 1 - C(n-c, k) / C(n, k)
```
where `n` = total samples, `c` = correct samples, `k` = evaluation budget.

---

## Integration: Composing Layers

All layers are independent and can be composed. Example: full request
lifecycle using R1 + R4 + R7 + R11:

```python
from aether.runtime import (
    PEAGLEEngine, SLOScheduler, SLOTier,
    GreenPowerManager, SemanticKVCache,
)

# Initialize.
spec = PEAGLEEngine(draft_K=5, mode="mtp")
sched = SLOScheduler(max_batch_tokens=8192)
gpm = GreenPowerManager(mode="balanced", tdp_cap_w=400.0)
skv = SemanticKVCache(dim=128, similarity_threshold=0.90)

# Request arrives.
req = sched.submit("r001", prompt_tokens=512, max_new_tokens=1024, slo_tier=SLOTier.LATENCY)

# Check semantic KV cache.
emb = skv.embed_prompt(token_ids)
cached_kv, _, sim = skv.lookup(emb)

# Route to greenest region.
region = gpm.select_region(["us-west", "eu-north"], latency_deadline_s=0.2)

# Generate with speculative decoding.
for step in range(max_steps):
    proposal = spec.propose(hidden, model.forward, token_ids)
    token_ids.extend(proposal.accepted_tokens)
    if eos_token in proposal.accepted_tokens:
        break

# Record energy.
e = gpm.estimate_request_energy(512, len(token_ids) - 512)
gpm.record_request(e, gpm.estimate_carbon(e))

# Store KV for future semantic reuse.
skv.store("r001", emb, computed_kv, kv_size_bytes=4096)
```

---

## AEG Artifact Dependencies

Each runtime layer loads from the corresponding AEG artifact:

```
R1  ← .aeg/speculation/mtp_config.json      (Pass 10)
R3  ← .aeg/grammar/fsm.bin                  (Pass 11)
R5  ← .aeg/ttt/ttt_config.json             (Pass 13)
R7  ← .aeg/metadata/green_profile.json      (Pass 16)
R8  ← .aeg/security/tee_config.json         (Pass 17)
R9  ← .aeg/quantization/sub2bit_manifest.json (Pass 19)
R10 ← .aeg/graph/video_compression_plan.json  (Pass 20)
R12 ← .aeg/training/rlvr_config.json         (Pass 22)
```

All paths can be overridden via the constructor `*_config_path` argument.
