# Aether Runtime — Supported Model Matrix

Aether detects model architecture from `config.json`, GGUF headers, SafeTensors
metadata, or structural graph analysis. Compilation and execution require
supported operators, a compatible backend and passing numerical admission.
The table below records verified contracts and historical family gaps.

## Support at a glance

Aether classifies **40 architecture families** — distinct computation contracts,
each one code path through the compiler and the runtime — reached through **164**
model-name and Hugging Face architecture-class spellings.

| Level | Families | Meaning |
|-------|---------:|---------|
| ✅ Parity-verified | **26** | every logit matches the reference (~1e-6) on the CPU, PyTorch and tensor-parallel engines, prefill and decode |
| 🟡 Runs, not gated | **6** | compiles and executes with round-trip tests; no per-logit comparison yet |
| 🔬 Known-incorrect | **4** | historical divergence or incomplete specialized contracts; scoped corrections below |
| ❌ Refused | **4** | detected, then rejected at compile time rather than emitting a wrong artifact |
| **Engine paths** | **36** | registered engine families, including degraded contracts |

Three quantities are easy to conflate, so they are kept apart here:
**families** are computation contracts (40); **detection keys** are the names and
class spellings that resolve to one (164); **checkpoints** are individual
published weights, which Aether does not count and makes no claim about — a
family covers every checkpoint that shares its contract.

These numbers are not prose. They are computed from
[`src/aether/core/model_families.py`](src/aether/core/model_families.py), the
single source of truth, and `tests/unit/test_model_family_registry.py` fails the
build if this document and the registry disagree. Reproduce them with:

```bash
aether models --counts
python -c "from aether.core.model_families import support_summary; print(support_summary())"
```

## How this matrix is verified

Registry grades retain the historical family checks described here. Ordinary
compilation additionally requires independent original-FP32 HF/CPU/Torch CPU
numerical admission; the scoped Mamba-1 results are linked below.

Every family marked **✅ Parity-verified** is checked by
[`scripts/validate_family_parity.py`](../scripts/validate_family_parity.py),
which for each family:

1. builds a small model of that architecture with 🤗 Transformers,
2. compiles it through the full Aether pipeline to an `.aeg`,
3. runs the artifact on the NumPy CPU engine, the PyTorch engine, and the
   tensor-parallel engine — for both prefill and incremental decode, and
4. compares every logit against the Transformers reference.

The gate is the maximum logit deviation relative to the reference logit spread,
with the reference weights rounded to BF16 first (the compiler's default
residency) so the test measures the forward pass rather than quantization. A
correct implementation lands near **1e-6**; a wrong attention scale, rotary
convention, or permuted head is orders of magnitude larger. Reproduce with:

```bash
python scripts/validate_family_parity.py llama qwen3 gemma2 gpt_neo   # etc.
```

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | **Parity-verified**: compiled output matches the 🤗 Transformers reference to ~1e-6 on prefill and decode, on the CPU, PyTorch, and tensor-parallel engines. |
| 🟡 | **Runs, not parity-gated**: compiles and executes through the shared decoder path, but has no automatic per-logit comparison against a reference implementation yet. |
| 🔬 | **Specialized contract with a degraded grade**: historical divergence or incomplete execution remains recorded; read the scoped corrections and remaining validation below. |
| ❌ | **Not yet supported**: fails to compile or has no execution path. |

---

## Decoder families — parity-verified ✅ (26)

These share the standard decoder graph and are verified against Transformers on
every execution path. Each row's *distinguishing numerics* are the constants
Aether derives from the checkpoint and applies at runtime; getting any of them
wrong changes every logit, which is why they are individually pinned by
[`tests/unit/test_execution_numerics.py`](../tests/unit/test_execution_numerics.py).

| Family | Representative models | Distinguishing numerics Aether handles |
|--------|-----------------------|----------------------------------------|
| **Llama 3.x** | Llama-3.1-8B, 3.2-1B/3B, 3.3-70B | GQA · SwiGLU · RMSNorm · RoPE (the baseline block) |
| **Qwen 2 / 2.5** | Qwen2-7B/72B, Qwen2.5 | GQA · SwiGLU; sliding-window honored only when the schedule enables it |
| **Qwen 3** | Qwen3-0.6B/1.5B/8B/32B/72B | per-head Q/K-norm |
| **Qwen 3 MoE** | Qwen3-MoE | routed experts **without** top-k renormalization (`norm_topk_prob: false`) |
| **Mistral** | Mistral-7B v0.1–v0.3 | GQA · SwiGLU |
| **Mixtral** | Mixtral-8x7B, 8x22B | top-2 of 8 experts **with** top-k renormalization |
| **Gemma 2** | Gemma-2-2B/9B/27B | ×√H embedding scale · `(1+w)` norms · sandwich norm · `query_pre_attn_scalar` scale · attention & final logit soft-caps · GeGLU |
| **Gemma 3** | Gemma-3-1B/4B/12B/27B (text) | all of Gemma-2 plus a separate rotary base for sliding-window layers |
| **GPT-2** | GPT-2 117M–1.5B, DialoGPT | MHA · Conv1D weight layout · GELU-tanh · learned absolute positions · LayerNorm |
| **GPT-Neo** | GPT-Neo 125M/1.3B/2.7B | **unscaled attention** (no 1/√d) · local/global layer schedule |
| **GPT-NeoX** | GPT-NeoX-20B, Pythia 70M–12B | 25% partial rotary · per-head-interleaved fused QKV · parallel residual · exact-erf GELU |
| **GPT-J** | GPT-J-6B | interleaved (rotate-every-two) rotary · partial rotary · parallel residual |
| **Phi-3** | Phi-3-mini/small/medium, Phi-4 | fused QKV · SwiGLU · LongRoPE factor tables keyed on the pretrained context length |
| **Falcon** | Falcon-7B/40B (new decoder arch) | per-KV-group interleaved fused QKV · parallel residual · single/dual block norms |
| **BLOOM** | BLOOM 560M–176B | **ALiBi** positions · per-head-interleaved fused QKV · embedding LayerNorm |
| **MPT** | MPT-7B/30B | ALiBi · `d_model`/`n_heads`/`n_layers` config spellings · low-precision LayerNorm |
| **StarCoder2** | StarCoder2-3B/7B/15B | GQA · GELU-tanh · sliding-window schedule |
| **Cohere / Command-R** | Command-R, Command-R+ | interleaved rotary · `logit_scale` · parallel residual · one block norm |
| **OLMo 2** | OLMo-2-7B/13B | **post-norm** block · full-projection Q/K-norm |
| **OLMoE** | OLMoE-1B-7B | pre-norm · full-projection Q/K-norm · experts without top-k renormalization |
| **StableLM** | StableLM-2, StableLM-3B | 25% partial rotary |
| **Granite** | Granite-3.x dense | embedding / residual / attention / logit multipliers |
| **EXAONE 4** | EXAONE-4 | post-norm · per-head Q/K-norm · **NoPE global layers** (rotary only on sliding-window layers) |
| **SmolLM 3** | SmolLM3-3B | interleaved NoPE layers among RoPE layers |
| **GLM-4** | GLM-4-9B, GLM-4 (text) | interleaved rotary · 50% partial rotary · GLM-spelled sandwich norm |
| **Nemotron** | Nemotron dense | `LayerNorm1P` (`(1+w)` LayerNorm) · squared-ReLU FFN · 50% partial rotary |

**These families cover a large share of the open-weight ecosystem by shared
graph.** A checkpoint that is architecturally a member of one of the rows above —
for example Yi, InternLM, TinyLlama, Zephyr, Dolphin, Nous-Hermes, OpenChat,
Tulu, SOLAR, MiniCPM, and the many Llama/Qwen/Mistral fine-tunes — executes
through the same verified path, because detection keys on structure, not name.
They are covered transitively; only the base architectures above are directly
parity-gated.

---

## Encoder & encoder-decoder — runs, not parity-gated 🟡 (6)

These have dedicated engines with round-trip tests (compile → load → execute,
with the portable and CPU engines cross-checked), but no automatic per-logit
comparison against Transformers. They execute; they are not yet certified
bit-for-bit.

| Family | Representative models | Engine |
|--------|-----------------------|--------|
| **BERT** | BERT-base/large (uncased/cased) | bidirectional encoder |
| **RoBERTa** | RoBERTa-base/large | bidirectional encoder |
| **DeBERTa** | DeBERTa-v3-base/large | bidirectional encoder |
| **ELECTRA** | ELECTRA-base/large | bidirectional encoder |
| **ALBERT** | ALBERT-v2 | shared-layer encoder |
| **T5 / mT5 / FLAN-T5** | T5-small → T5-11B, BART, Pegasus | encoder-decoder + relative attention |

---

## State-space & hybrid — historical divergence / incomplete validation 🔬 (4)

These families retain the registry's degraded grades while trained checkpoints
and complete execution contracts are revalidated. Historical reference checks,
scoped corrections and incomplete bindings are recorded individually below:

| Family | Status detail |
|--------|---------------|
| **Mamba** | Historical trained-family check diverged (cosine ≈ 0.97). Checkpoint normalization, expansion, biases and finite-value handling are now corrected; nine untrained FP32 contracts pass original-HF/CPU/Torch CPU, public API and CLI checks. Trained checkpoints remain unverified and the family grade is unchanged; see [measurements and limits](docs/mission-mamba1-fidelity.md). |
| **Mamba-2 / SSD** | Historical parity check diverged (cosine ≈ 0.21). Learned normalization, HF expansion and output bias are now corrected; two untrained FP32 HF fixtures pass CPU/Torch CPU, API and CLI checks. Trained checkpoints and other config/normalization variants remain unverified; see [executed evidence](docs/mission-mamba2-normalization.md). |
| **RWKV-7** | Does not currently bind its time-mix/channel-mix weights during compile. |
| **Jamba** | Hybrid attention+SSM+MoE (also Zamba2, Hymba, Falcon-H1, Bamba, PLaMo); the reference path requires CUDA Mamba kernels, so it is unverified on CPU. |

---

## Not yet supported ❌ (4)

| Family | Reason |
|--------|--------|
| **DeepSeek V3 / R1 (MLA)** | Multi-head latent attention fuses `k_rope` into `kv_a_proj`, adds shared experts alongside routed ones, and uses sigmoid group-limited routing. The MLA engine does not yet reconstruct this contract; compile fails on the fused projection. |
| **MiniMax** | Lightning attention (alternating linear/softmax layers) is a distinct architecture class without an engine. |
| **Vision-language** | LLaVA, InternVL, PaliGemma, Qwen2-VL, Pixtral are detected and ingested, but the vision tower has no verified execution contract. |
| **Whisper / audio** | The Conv1D log-Mel front-end is detected and ingested, but has no verified execution contract. |

---

## Compile-once, run-everywhere

Portable AEG execution uses a compatible backend that supports the artifact's
operators. The [hardware matrix](hardware_validation_matrix.json) records
per-target execution and validation evidence:

* Supported CPU/Torch CPU contracts have compile/save/load and numerical
  comparisons. The nine untrained Mamba-1 contracts above execute through both
  public CPU backends from the same admitted artifact. Accelerator and ONNX
  parity require their own supported operators and execution evidence.
* The `.aeg` is **fully self-contained**: it embeds the quantized weights, the
  complete supported tokenizer assets, the computation graph, and integrity hashes. After
  compilation the original checkpoint can be **deleted**, and
  `aether run model.aeg --prompt "…"` still loads and generates correctly with
  no network access and no reference to the source model.

Verify both properties yourself:

```bash
aether compile ./some-model -t cpu_generic --precision FP32 -o model.aeg
mv ./some-model ./some-model-source-backup
aether run model.aeg --prompt "The capital of France is" --temperature 0
# Inspect the actual generated text and admission report for this checkpoint.
```

---

## GGUF format

Any GGUF file whose `general.architecture` maps to a parity-verified decoder
family above is supported; the architecture is read from the header, not the
filename. Q4_0/Q4_1, Q4_K_M/Q4_K_S, Q5_K_M, Q6_K, Q8_0, and F16/F32 tensors are
dequantized on load.

---

## A note on decode throughput

Small-model decode can be limited by per-layer operations and dispatch overhead.
Parameter count alone does not predict throughput across models and hardware.
Measure the per-layer operation count for any family with:

```bash
python scripts/benchmark_dispatch.py llama     # Aether vs HuggingFace, per layer
python scripts/profile_decode.py model.aeg cuda 100
```

Learned-position, ALiBi and rotary architectures have different operator
schedules. Profile their actual attention and projection paths when comparing
latency; operation counts alone do not establish a hardware speedup.

## A note on accelerator memory

Position tables are part of runtime memory. Their allocation depends on the
artifact's context/rotary contract and the selected executor. Measure the actual
table footprint alongside weights, KV, activations and overhead at the intended
workload ceiling.

## A note on multi-GPU

Sharding is decided by a **planner**, not a threshold. Aether measures the machine,
reads the model's exact tensor geometry from the AEG, enumerates every structurally
admissible placement, filters them by a conservative memory residual, and ranks the
survivors on a three-roof cost model — compute, memory bandwidth, and host dispatch.

Three consequences worth knowing:

* **Tensor parallelism is not automatically a speed-up.** For small models decode is
  *dispatch*-bound, not bandwidth-bound, and TP roughly doubles the host op count. The
  planner predicts that and keeps such models on one device.
* **The planner estimates sharding benefit** from compute, weight bandwidth,
  dispatch and fabric costs. Predicted gains require execution measurements on
  the proposed hardware and workload.
* **The planner reports estimated capacity.** Its `tokens_max` follows a memory
  model; actual allocations and concurrent memory use still require admission
  control and measurement.

Inspect the decision for any artifact and any workload:

```bash
aether plan model.aeg --batch 4 --context 8192 --intent balanced
```

`AETHER_FORCE_TENSOR_PARALLEL=1` still overrides the planner, and the override is
recorded in the decision rather than hidden. Design:
[`docs/architecture-execution-planner.html`](docs/architecture-execution-planner.html).
