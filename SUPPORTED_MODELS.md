# Aether Runtime — Supported Model Matrix

Aether detects model architecture from `config.json`, GGUF headers, SafeTensors
metadata, or structural graph analysis — **not** from the model name. Any model
whose architecture is recognized by the detector can be compiled to a
self-contained `.aeg` artifact and executed on any supported target.

## Support at a glance

Aether classifies **40 architecture families** — distinct computation contracts,
each one code path through the compiler and the runtime — reached through **164**
model-name and Hugging Face architecture-class spellings.

| Level | Families | Meaning |
|-------|---------:|---------|
| ✅ Parity-verified | **26** | every logit matches the reference (~1e-6) on the CPU, PyTorch and tensor-parallel engines, prefill and decode |
| 🟡 Runs, not gated | **6** | compiles and executes with round-trip tests; no per-logit comparison yet |
| 🔬 Known-incorrect | **4** | executes with *measured* divergence from the reference |
| ❌ Refused | **4** | detected, then rejected at compile time rather than emitting a wrong artifact |
| **Executable** | **36** | parity-verified + runs + known-incorrect |

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
| 🔬 | **Specialized engine, known-incorrect**: ingests and runs, but its dedicated engine does **not** currently match the reference. Do not rely on it. |
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

## State-space & hybrid — specialized engine, currently incorrect 🔬 (4)

These ingest and run through their own engines, but **do not currently match the
reference implementation** when checked with `validate_family_parity.py`, so they
must not be relied on for correct output yet. This is a deliberate downgrade from
earlier documentation, which asserted support without a parity check:

| Family | Status detail |
|--------|---------------|
| **Mamba** | Runs; selective-scan output diverges from the reference (cosine ≈ 0.97). Not correct. |
| **Mamba-2 / SSD** | Runs; output badly diverges (cosine ≈ 0.21). Not correct. |
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

The portability promise holds and is verified:

* An `.aeg` compiled for **any** target (`cuda_sm70`, `metal_m1`, `cpu_avx512`, …)
  executes on **any** host with a supported backend. Artifacts built for the
  three targets above produce **identical** logits when run on the same machine.
* The `.aeg` is **fully self-contained**: it embeds the quantized weights, the
  complete tokenizer, the computation graph, and integrity hashes. After
  compilation the original checkpoint can be **deleted**, and
  `aether run model.aeg --prompt "…"` still loads and generates correctly with
  no network access and no reference to the source model.

Verify both properties yourself:

```bash
aether compile ./some-model -t cpu_avx512 -o model.aeg
rm -rf ./some-model                 # delete the original entirely
aether run model.aeg --prompt "The capital of France is" --temperature 0
# → "Paris. …"  (runs from the artifact alone)
```

---

## GGUF format

Any GGUF file whose `general.architecture` maps to a parity-verified decoder
family above is supported; the architecture is read from the header, not the
filename. Q4_0/Q4_1, Q4_K_M/Q4_K_S, Q5_K_M, Q6_K, Q8_0, and F16/F32 tensors are
dequantized on load.

---

## A note on decode throughput

Small-model decode is bound by the *number of tensor operations per layer*, not
by arithmetic. A 135M model with 30 layers issues more work per token than a
350M model with 24, which is why parameter count alone does not predict speed.
Measure the per-layer operation count for any family with:

```bash
python scripts/benchmark_dispatch.py llama     # Aether vs HuggingFace, per layer
python scripts/profile_decode.py model.aeg cuda 100
```

Architectures without rotary embeddings (GPT-2, GPT-Neo, BLOOM, MPT — all of
which use learned or ALiBi positions) issue roughly a third fewer operations per
layer than rotary families, so they decode faster at equal depth. The rotary
path is evaluated in its minimal five-operation form, with the sin/cos tables
pre-expanded at build time and Q/K rotated in a single pass.

## A note on accelerator memory

Position-indexed tables (rotary sin/cos) are sized by sequence length and capped
by the artifact's declared context length, so their footprint stays proportional
to the tokens actually generated. They are never grown multiplicatively: a
40960-position context would otherwise reserve gigabytes of accelerator memory
for positions the request can never reach.

## A note on multi-GPU

Tensor-parallel sharding is a **memory-capacity** mechanism, not a throughput
boost for small models. Splitting a model that already fits on one accelerator
adds a cross-device copy to every layer and makes decode *slower*. Aether
therefore auto-shards only when the model does not fit on the smallest device;
otherwise it runs single-device. Force sharding for benchmarking with
`AETHER_FORCE_TENSOR_PARALLEL=1`.
