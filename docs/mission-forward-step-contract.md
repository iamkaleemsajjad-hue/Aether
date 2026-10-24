# Torch last-position forward-step output

`TorchAEGEngine.forward_step(..., return_all_logits=False)` now returns a NumPy
FP32 vocabulary vector, matching its documented contract and the CPU engine.
The internal tensor executor still keeps its sequence dimension; the public
adapter selects its last row. This is a view operation, without a new tensor
projection or a numerical transformation.

The real AEG CPU/Torch comparison exposed the issue after the independent
calibration-ledger stall was fixed. The quality validator correctly rejected
Torch's `(1, vocab)` output where `(vocab,)` was required. Read scope covered the
Torch engine, CPU forward-step implementation, the shared protocol,
continuous-batching and speculative callers, portable-executor/speculation
tests and public runtime contracts. Existing scheduler/fallback callers flatten
last-position logits; all-position speculative verification consumes the
two-dimensional output, which is retained.

| Check | Before | After |
|---|---|---|
| Eight real CPU/Torch regression variants | 4 failed, 4 passed | All pass in the 234-test focused selection |
| Last-position prefill and decode | `(1, 17)`, contrary to contract | `(17,)`, FP32 NumPy |
| All-position prefill and decode | `(input_length, 17)` | Preserved |
| Fresh `None` and explicitly initialized caches | Four last-position shape failures | Both preserve the actual consumed token count |
| Reference logits and greedy IDs | All-position variants already agreed | Every variant agrees within `rtol=2e-4`, `atol=2e-4`; greedy IDs match |
| Existing touched-file Ruff / scoped source mypy diagnostics | 116 / 46 | 116 / 46; no new diagnostic messages |

Evidence: `build/mission/forward-step-before.*`,
`quality-combined-focused.*` and `forward-step-static-comparison.json`.
The full collection result is recorded in [mission validation](mission-validation.md).
The 234-test selection includes ledger failures, strict quality math, real AEG
CPU/Torch parity, existing evaluation contracts, portable execution and real
Torch speculative execution. It passed in 15.23 seconds.

This fixes a public execution contract; it makes no generation-throughput, TTFT,
RSS or model-quality improvement claim. The same artifact is used on both CPU
engines. It does not establish ONNX parity or accelerator execution. Physical
CPU evidence is current; other hardware tiers and the
[22-pass audit](mission-pass-audit.md) are unchanged.
