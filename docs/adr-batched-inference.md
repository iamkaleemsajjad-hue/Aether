# Batched inference in Aether Runtime — architecture decision record

Status: accepted
Scope: `aether.runtime` portable executors, runtime/backend APIs, benchmark harness
Non-scope: AEG on-disk format, compiler IR, native CPU (NumPy) kernels

---

## 1. Problem

Aether's decoder executors were sequence-major. One forward pass carried one
sequence: a single monotonic run of positions, one KV cache per layer with no
batch axis, and a sampler that reduced a rank-1 logit vector to one scalar
token. Serving *N* independent sequences per pass was therefore not expressible,
and the benchmark reported `batch_size > 1` as unsupported rather than
approximating it.

The specific code-level constraints found in the audit:

| Location | Assumption |
|---|---|
| `TorchAEGEngine._forward_device` | `ids = token_ids.reshape(-1)` — collapses `[B,S]` into `[B*S]`, destroying sequence identity |
| `TorchKVCache.keys[layer]` | `(capacity, kv_heads, head_dim)` — no batch axis |
| `TorchAEGEngine._attention` | `q.transpose(0,1).unsqueeze(0)` — a literal batch of one |
| `TorchAEGEngine._append_kv` | `result[past:total]` — one frontier for the whole cache |
| `cache.last_logits` | `logits[-1]` — one row of logits |
| `_sample_device` | `torch.argmax(values)` over a rank-1 tensor — one token |

## 2. What the audit settled about AEG and the compiler

**The AEG format and compiler IR do not assume `batch = 1`, and must not be
changed.**

- `aether/compiler/stage1_ingestion/ingestion.py:1983` sets `batch_dim = None`
  with the comment `# dynamic`, and builds the `input_ids` graph node as
  `TensorShape.from_list([batch_dim])`.
- `aether/core/types.py:724` defines `TensorShape.dims: tuple[int | None, ...]`,
  where `None` *is* the representation of a dynamic dimension
  (`is_static()` returns `all(d is not None for d in self.dims)`).
- The compiler declares `kernels.portable_backends = ["aether_cpu", "pytorch"]`
  (`compiler.py:1347`). Both are **eager tensor interpreters**: they materialize
  authenticated weights and evaluate the transformer equations op by op. Neither
  executes a shape-specialized compiled kernel emitted at compile time.

Consequences, and they are the load-bearing decisions of this ADR:

1. **No AEG format change.** A batch axis is not a property of the artifact. The
   artifact stores weights plus an execution-numerics contract; the batch axis is
   a property of a single call into the executor.
2. **No batch-specialized graph variants (`B=1` / `B=2` / `B=4`).** The prompt
   asked whether a highly optimized solution should compile specialized graphs
   per batch size. For a compiler emitting static kernels that trade-off is real.
   Here it is not: there is no kernel to specialize, so variants would add
   artifact size, compile time, and a combinatorial validation surface while
   changing not one dispatched operation. The dynamic dimension the IR already
   declares is the correct representation.
3. **Batching is a runtime-executor feature.** All work belongs in
   `aether.runtime`, behind an explicit, queryable capability.

Rather than encoding a capability into the artifact, the executor advertises it
(`supports_batch()`, `max_batch_size`). That keeps existing artifacts valid: an
AEG compiled before this change gains batching by being run on a newer runtime,
which is the correct outcome for a weights-plus-metadata artifact.

## 3. Techniques considered

| Technique | Solves | Decision |
|---|---|---|
| **Left-padded static batch, contiguous KV** | independent rows, uniform decode frontier | **Adopted** |
| Right-padded static batch | same | Rejected — each row's frontier lands at a different index, forcing a per-row scatter every decode step and a gather to find each row's last logits |
| Variable-length packing (`cu_seqlens`, FlashAttention varlen) | zero pad waste | Rejected — needs a varlen attention kernel; `torch.nn.functional.scaled_dot_product_attention` exposes none, and the portable executor must run on CUDA/ROCm/MPS/CPU with no extra dependency |
| **Paged KV cache / PagedAttention** (Kwon et al., SOSP 2023) | fragmentation across many long-lived concurrent sequences | Rejected for this change — the win is a memory-allocator win at high concurrency; at `B ≤ 4` static batching it adds a block-table indirection on every attention call and needs a paged kernel to not lose throughput. The layout chosen below leaves a clean path to it (§6) |
| **Continuous batching** (Orca, Yu et al., OSDI 2022) | throughput under a stream of arrivals | Rejected for this change — it is a *scheduler*, and presupposes exactly the batched executor this ADR builds. Correct next layer, wrong first step |
| Per-request sequential loop | nothing | Rejected — this is the thing the change exists to avoid. It is not batching |
| Duplicating the model per row | nothing | Rejected — multiplies resident weights, gains nothing |

Reference implementations consulted for the batched-decode contract: Hugging
Face Transformers (`GenerationMixin`, left padding + `position_ids` from the
attention mask + `unfinished_sequences`), vLLM (block-table KV indirection),
llama.cpp (`n_seq_max`, sequence-id-tagged unified cache), TensorRT-LLM
(`max_batch_size` engine specialization — the case that does *not* apply here,
because Aether's portable path emits no static kernel).

## 4. Adopted architecture

### 4.1 Left padding, right alignment

Sequences are right-aligned inside the padded window:

```
prompt A (5 tokens)   .  .  .  a  a  a  a  a
prompt B (8 tokens)   b  b  b  b  b  b  b  b
prompt C (2 tokens)   .  .  .  .  .  .  c  c
                      ^ pad                ^ every row's last real token
```

Right alignment is the property that makes decode uniform: after prefill, every
row's next token lands at the *same* padded index. One write index serves the
whole batch, so a decode step is one `cache[:, w].copy_(...)` and not a per-row
scatter.

The cost is that pad slots occupy KV rows and pass through the prefill GEMMs.
That cost is bounded by the spread of prompt lengths in a batch, and is zero for
the equal-length batches the benchmark issues.

### 4.2 Positions are per row, not per padded index

A pad slot contributes no position. Row `b`'s first *real* token is at position
0 regardless of how much padding precedes it:

```
position(b, i) = max(0, i - pad_count(b))
```

This is the invariant that makes a batched result *equal* the same sequence
decoded alone. Using the padded index as the position would shift every rotary
angle (and every learned absolute position embedding, which is how GPT-Neo is
positioned) by that row's pad count — a silent, model-wide numerical error that
no shape check would catch.

### 4.3 KV cache layout: `(B, capacity, kv_heads, head_dim)`

Batch-major, contiguous, one tensor per layer.

Chosen over `(layers, B, heads, seq, head_dim)` and over paged blocks because:

- **The append is one slice.** `keys[:, past:total]` writes every row at once,
  which is exactly what right alignment buys.
- **It is already the shape SDPA wants.** `(B, S, H, D).transpose(1, 2)` is
  `(B, H, S, D)` — a view, not a copy. The unbatched path needed
  `transpose(0,1).unsqueeze(0)`; the batched path needs one transpose.
- **Row-major contiguity puts a row's own timeline adjacent in memory**, which is
  the access pattern attention actually has (one row reads its whole history),
  and gives coalesced loads on GPU for the head/dim inner dims.
- **Isolation is structural, not enforced.** Because the batch axis is the
  outermost axis of every activation and cache tensor, and every operation in the
  layer loop is either elementwise, last-dim (`linear`, norms), or explicitly
  batch-blocked (SDPA), row `i` cannot reach row `j`'s state. There is no code
  path that would have to be *checked* for isolation. §4.6 lists the two
  operations where that is not automatic and how each is handled.

A `live` mask of shape `(B, capacity)` travels with the cache and records, per
row, which KV slots hold a real token. It is the single source of truth for
masking: pad slots are never live, decode slots become live as they are written.

### 4.4 One rank-generic forward, not two implementations

`_forward_device` is generalized to accept either a rank-1 id tensor `(S,)` or a
rank-2 id tensor `(B, S)`, and carries the leading shape through the layer loop.
The ~200 lines of architecture-variant handling (MoE, ALiBi, parallel residual,
post/sandwich norm, Q/K norm scope, partial and interleaved rotary, sliding
window, logit softcap) are **not duplicated**. Duplicating them is the largest
correctness hazard available here, because the two copies would drift.

Only five sites are rank-aware, and each branches on `ids.dim()`:

| Site | Unbatched | Batched |
|---|---|---|
| embedding gather | `index_select` → `(S, h)` | gather + reshape → `(B, S, h)` |
| head reshape | `(S, H, D)` | `(B, S, H, D)` |
| rotary factors | `(S, 1, D)` | `(B, S, 1, D)` |
| KV append | `[past:total]` | `[:, past:total]` |
| last logits | `logits[-1]` | `logits[:, -1]` |

`B = 1` batched is a strict special case of the batched path, and the existing
unbatched path is preserved byte-for-byte for every current caller.

### 4.5 Fast paths preserved

The unbatched path's tuned behaviour is not regressed:

- Unbatched prefill still takes `is_causal=True` with `attn_mask=None`.
- Unbatched decode still takes `attn_mask=None`.
- **A batch whose rows are equal-length has no padding**, therefore `live` is
  all-true, therefore it *also* takes `is_causal=True` / `attn_mask=None` and
  keeps the fused SDPA kernel. This is the benchmark's case.
- A materialized boolean mask is built only when padding actually exists.

### 4.6 The two operations that are not automatically batch-safe

1. **MoE expert routing** (`_moe_ffn`) dispatches with
   `torch.where(selected == e)` over a rank-2 `(tokens, experts)` tensor. The
   batched path flattens `(B, S, ·)` to `(B*S, ·)` for routing and restores the
   shape afterwards. This flatten is *safe* — expert routing is strictly
   per-token with no cross-token interaction — and is called out here precisely
   because the same flatten applied to attention is the bug this whole design
   exists to prevent.
2. **Sampling** must reduce `(B, vocab)` to `(B,)`, not to a scalar. Greedy,
   top-k, and nucleus all become last-dim reductions; `torch.multinomial`
   already samples rank-2 input per row.

### 4.7 Per-row stopping

A `finished` mask of shape `(B,)` tracks stop-token arrival. A finished row stops
*recording* tokens; the batch keeps its width until every row is finished. Rows
are independent, so continuing to compute a finished row produces output that is
simply discarded — it cannot perturb another row.

This trades some wasted compute on skewed batches for a decode loop with no
mid-flight reshaping. Shrinking the batch as rows retire is the natural
follow-up, and belongs with the scheduler in §6.

### 4.8 Determinism note

Batched sampling draws all rows from one `torch.multinomial` call, so a sampled
batched run is not token-identical to *N* separately-seeded single-sequence runs.
This is true of every batched runtime, Transformers included. Greedy decoding
(`temperature = 0`) *is* comparable, and is what the equivalence tests assert.

## 5. Consequences

- Existing single-sequence callers (`forward`, `generate`, `generate_iter`,
  `generate_with_cache`, session KV reuse, the R2 shared-prefix coordinator,
  TTT, speculative decoding) are unchanged and keep using `TorchKVCache`.
- Batched callers use the new explicitly-batched entry points and
  `BatchedKVCache`.
- The tensor-parallel executor (`torch_tensor_parallel.py`) carries its own copy
  of the forward loop and is **not** batched by this change. It remains
  single-sequence and refuses a batched call explicitly rather than flattening it.
  Batching it is mechanical but doubles the validation surface, and it is selected
  only for models that do not fit on one device — a different problem.
- The NumPy CPU executor keeps its sequence-major kernels; they remain the faster
  path for one sequence, and single-sequence CPU requests are unchanged. A batched
  request on a CPU host is served by promoting the same authenticated weights onto
  the portable tensor executor at `device="cpu"` — same artifact, same equations,
  different kernel provider. `aether.backends.batched_generation` does this lazily,
  at most once per loaded handle, and only when a batch is actually asked for,
  because it materializes the weights a second time as tensors. Where PyTorch is
  absent, batch > 1 is refused explicitly rather than emulated by a sequential
  loop.
- Packing, promotion, row-splitting and metric labelling live once in
  `aether.backends.batched_generation`, shared by the native CPU backend and the
  PyTorch backend. Both load an AEG into the same `CompiledAEGHandle` and would
  otherwise have carried two copies of this logic, which is how the two would
  drift.

## 6. Path forward

The layout was chosen so that each next step is additive, not a rewrite:

1. **Retire finished rows mid-decode** — compact the batch when a row stops.
2. **Continuous batching** — an arrival scheduler on top of the batched
   executor; the `live` mask and per-row positions are already the state a
   scheduler needs to admit a new sequence into a running batch.
3. **Paged KV** — replace the contiguous per-layer tensor with a block table.
   The `live` mask generalizes to a block-occupancy map; nothing above the cache
   boundary changes.
4. **Varlen packing** — becomes attractive if/when a varlen attention kernel is
   available on the portable path, removing pad compute entirely.
