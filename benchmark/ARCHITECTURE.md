# Aether Runtime and Hugging Face Transformers — Architectural Comparison

Written from inspection of the Aether Runtime source tree in this repository, not
from documentation or assumption. Every claim below is labelled:

- **[source]** — read directly from the implementation, with the file named.
- **[measured]** — observed by the benchmark harness or the parity harness.
- **[inferred]** — a reasonable explanation that has *not* been directly
  confirmed. Treat these as hypotheses.

This document describes the systems. It deliberately contains no performance
numbers: those belong in `results/REPORT.md`, generated from an actual run on
actual hardware.

---

## 1. What each system is

**Hugging Face Transformers** is a model library: for each architecture it
provides a Python `nn.Module` implementation, and generation is a Python loop
(`GenerationMixin.generate`) that calls that module once per step.

**Aether Runtime** is a compiler plus an executor **[source]**. It has two
distinct phases:

1. `aether.compiler.compiler.Compiler.compile` ingests a checkpoint, detects the
   architecture from `config.json` (never from the model name), runs optimizer
   passes, quantizes the weights, and writes a self-contained `.aeg` package.
2. `aether.runtime.runtime.Runtime.generate` loads that package and executes it
   through one of several engines.

The important consequence: **Aether does not import or execute the model's
Transformers class at inference time.** It reconstructs the forward pass from
tensors and an architecture description carried in the artifact.

**Both execute the same model architecture and the same weights.** Aether is not
a different model — it is a different way of running the same one.

---

## 2. Model loading

| | Transformers | Aether |
|---|---|---|
| Source | checkpoint on disk / Hub | compiled `.aeg` artifact |
| Weight format | as published (BF16 for all three benchmark models) | quantized blob, default residency **BF16** **[source: `compiler.py`, `default_precision="BF16"`]** |
| Tokenizer | loaded from the repo | **embedded in the artifact** **[source: `aeg_format.py`, `tokenizer/` subtree]** |
| Integrity | none by default | every payload hashed and verified on load **[source: `AEGPackage.verify_integrity`]** |
| Self-contained | no | yes — the original checkpoint can be deleted **[measured]** |

Aether's load path is `aeg_loader.load_engine_from_package`: it dequantizes each
tensor to float32 NumPy, groups them into `LayerWeights`, validates shape
invariants against the manifest, and builds a `CPUExecutionEngine`. The
accelerator engine then uploads those arrays to the device at the compute dtype.

**Cost implication [source]:** loading materializes the model twice — once as
float32 NumPy on the host, then again on the device. This is why Aether's host RSS
at load is expected to exceed Transformers'; the benchmark measures it rather than
assuming.

---

## 3. Engine selection

`TorchBackend._try_load_compiled_aeg` dispatches on architecture **[source]**:

| Artifact contains | Engine |
|---|---|
| dense decoder | `TorchAEGEngine` |
| dense decoder, model does not fit one device | `TorchTensorParallelAEGEngine` |
| encoder (BERT-family) | `TorchEncoderAEGEngine` |
| encoder-decoder (T5) | `TorchSeq2SeqAEGEngine` |
| MLA (DeepSeek) | `TorchMLAAEGEngine` |
| Mamba / Mamba-2 / RWKV | `TorchMambaAEGEngine`, `TorchMamba2AEGEngine`, `TorchRWKVAEGEngine` |

All three benchmark models take `TorchAEGEngine`.

---

## 4. Precision

**Transformers:** `torch_dtype` selects both storage and compute.

**Aether [source]:** two independent stages.
- *Storage* is fixed by the compiler at BF16 by default, inside the artifact.
- *Compute* is chosen at engine construction from `AETHER_TORCH_DTYPE`
  (`fp16`/`bf16`/`fp32`, or `auto` → FP16 on CUDA, FP32 on CPU).

**This is the benchmark's most important fairness caveat.** At `bf16` both
backends hold the same values, because the checkpoints are natively BF16. At
`fp16` or `fp32`, Aether's weights have been through BF16 storage and
Transformers' have not. The report states this; it is why `bf16` is the primary
configuration.

Note also that RMSNorm accumulates its reduction in FP32 regardless of compute
dtype **[source: `torch_engine._norm`]**, which is the standard stable evaluation.

---

## 5. Attention

Both call `torch.nn.functional.scaled_dot_product_attention` **[source]**, so both
get whatever fused kernel PyTorch selects for the device — FlashAttention on
compute capability 8.0+, the memory-efficient CUTLASS path on older cards. Neither
ships a hand-written attention kernel for the GPU path.

Aether's `_attention` **[source]** adds:
- an explicit `scale` argument, because not every family uses `1/√d` (GPT-Neo's
  published attention is unscaled);
- a fallback to explicit math when the architecture needs something SDPA cannot
  express — ALiBi bias, or Gemma-2's attention logit soft-cap;
- omission of the sliding-window mask when the whole cache already fits inside the
  window, which keeps SDPA on its fused path.

**There are no custom CUDA kernels and no Triton kernels in the GPU path**
**[source]**. `aether/kernels/native_cpu.py` is a C++/OpenMP library, and it is
used only by the NumPy CPU engine.

---

## 6. KV cache

**Transformers** uses its `Cache` classes; `DynamicCache` grows by concatenation.

**Aether [source: `torch_engine._append_kv`]** preallocates. The generation loop
knows the final length (`prompt + max_new_tokens`) and passes it as `reserve`, so
the cache is allocated once and each step writes into a slice. No reallocation and
no prefix copy during decode.

---

## 7. The decode loop, and where the two differ most

This is the substantive architectural difference for small models.

**Transformers** calls the module per step; each call re-enters the Python
`nn.Module` hierarchy, and `generate` runs its `LogitsProcessor` and
`StoppingCriteria` machinery around it.

**Aether [source: `torch_engine._forward_device`]** is a single flat function over
a list of pre-converted weight dictionaries. Several things are resolved once at
load time instead of per step:

- projection orientation (a GPT-2 Conv1D weight is transposed once, not per call);
- Q/K/V packed into one GEMM, gate/up packed into one GEMM;
- rotary sin/cos tables pre-expanded to the rotated width in the layout the
  model's convention needs;
- a per-layer plan of `(is_local, window, attention_scale, uses_rope)`, so the
  loop performs no string comparison or attribute probing per layer.

And one structural change in the loop itself **[source:
`_generate_pipelined`]**: the next step's forward pass and sampling are *queued
before* the current token is read back to the host. A decode step must move its
sampled token to the host, and doing that immediately after sampling stalls the
host until every kernel of that step has retired. Queuing first lets the host's
launch work overlap device execution.

**[measured]** These reduce the ATen call count per layer per token relative to
Transformers. The benchmark's `profile` mode counts both, so the report states the
actual numbers rather than this document asserting them.

**[source] Aether does *not* use CUDA graphs.** There is no graph capture in the
decode path. If a report claims a launch-overhead advantage, it comes from issuing
fewer operations, not from replaying a captured graph.

---

## 8. Batching

**[source]** `TorchAEGEngine._forward_device` treats the token tensor as 1-D:
`hidden` has shape `(seq, hidden)` with no batch dimension. A 2-D input is
flattened into a single sequence — **[measured]** a `(2,3)` input produces 6 logit
rows, not 2 sequences of 3.

**Aether cannot serve batch > 1 on this path.** The benchmark reports that as
`unsupported` rather than measuring something incorrect. Transformers batches
normally, so batch>1 rows are Transformers-only observations.

---

## 9. Multi-GPU

**Transformers** offers `device_map` for pipeline-style layer sharding.

**Aether [source: `torch_tensor_parallel.py`]** implements single-process tensor
parallelism: Q/K/V and gate/up are row-sharded, attention-output and down are
column-sharded, embedding and LM head are vocabulary-sharded, with explicit
device-to-device copies for the collectives.

**[source: `torch_backend._should_shard`]** It activates only when the model does
not fit on the smallest visible device, because sharding a model that *does* fit
adds a cross-device copy per layer and makes decode slower. For all three
benchmark models on a 15 GiB card, the default is single-device. The multi-GPU
mode measures both the default and the forced sharded path, labelled separately.

---

## 10. Sampling

**Transformers** runs a `LogitsProcessorList`.

**Aether [source: `_sample_device`]** implements greedy, temperature, top-k and
nucleus sampling directly, and keeps the result on the device so the next step can
be queued before the token is read back. Nucleus sampling sorts the full
vocabulary; a cheaper top-k pre-truncation was considered and rejected because
testing its exactness requires reading a value to the host, which would reintroduce
the stall the pipelining exists to avoid **[source: comment in `_sample_top_p`]**.

---

## 10a. The semantic response cache

**[source: `RuntimeConfig.enable_semantic_cache`, default `True`]** The runtime
keeps a semantic cache of prompt to completion. A repeated (or sufficiently
similar) prompt is answered from it without executing the model.

**[measured]** With it enabled, a repeat of the same prompt returns in ~0.001 s
against ~15 s for real generation.

This is a genuine product feature and a real latency win for repeated traffic. It
is **not** an inference optimization, and Transformers has no counterpart, so the
benchmark disables it: otherwise every iteration after the first would time a
lookup. See the README for the measurement and the reasoning.

## 11. Where a difference could come from, and how to tell

If the benchmark shows a throughput difference, these are the candidate causes and
the evidence that would distinguish them:

| Candidate | How to confirm | Status |
|---|---|---|
| Fewer ATen calls per token | `profile` mode dispatch counts | **[measured]** — counted for both |
| Fused QKV / gate-up GEMMs | source, plus GEMM count per layer | **[source]** + countable |
| Preallocated KV cache | source; absence of realloc in profile | **[source]** |
| Lower host stall from decode pipelining | profiler: wall clock minus summed device time | **[source]**, testable |
| Different attention kernel | profiler kernel names | both use SDPA **[source]**; profile confirms |
| Custom CUDA / Triton kernels | source | **none exist** **[source]** |
| CUDA graphs | profiler | **not used** **[source]** |
| Reduced precision | correctness mode | BF16 storage; `bf16` run is like-for-like **[source]** |
| Approximation / fewer math ops | correctness mode logit comparison | **[measured]** by parity harness |

The last two matter most for the question "does Aether trade quality for speed?".
The `correctness` mode answers it with logit deviations normalized by the
reference's own logit spread, greedy-token agreement, and text comparison.

---

## 12. Honest asymmetries

1. **Compile cost.** Aether pays it once; the benchmark times it separately as
   `prepare_s` and never amortizes it into throughput.
2. **Load memory.** Aether dequantizes to float32 NumPy on the host before
   uploading, so peak host RSS at load is expected to be higher.
3. **Coverage.** Transformers supports far more architectures. Aether's verified
   set is enumerated in `SUPPORTED_MODELS.md`, and its state-space engines are
   currently marked as not matching their reference.
4. **Batching.** Section 8.
5. **Maturity.** Transformers is heavily exercised in production across many
   hardware configurations; Aether is not.
