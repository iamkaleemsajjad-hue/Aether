# Preserve explicitly requested FP32 weights

Executed 2026-10-20 on the physical Intel Core i5-1334U, Windows AMD64,
Python 3.10.11, NumPy 1.26.4, Torch 2.9.1+cpu and Transformers 4.57.3.

Reading the complete optimizer and weight packer, their callers and regression
tests confirmed that uniform FP32 still forced embedding/LM-head globals and
projection biases through BF16. Manual global assignments were also overwritten.
Three regression cases failed before the correction. P3 now preserves explicit
global assignments; uniform FP32 reaches globals and biases in the real packer.
Compact-format defaults retain their existing BF16 global/bias policy.

The tests round-trip eleven actual encoded tensors, including global tensors,
fused-QKV biases and ordinary projection biases, and compare exact FP32 values.
The optimizer/packer/end-to-end selection passed 165 cases with two existing
weightless synthetic-fixture skips. The wider mission's verification history is
recorded in the [validation journal](mission-validation.md).

The original-HF comparison used the deterministic untrained 26,784-parameter
Llama fixture from `tools/pass_audit_fixtures.py`, seed 1729. Two fixed prefixes
contain 4 and 64 tokens, with four teacher-forced labels and eight greedy output
IDs each. Both real CPU and Torch CPU executors ran against original FP32 HF.
These observations do not establish trained-model or serving performance.
The measurement used the separate admission implementation in the active
worktree; the FP32 correction itself does not introduce that gate.

| Observation | Before | After |
|---|---:|---:|
| Worst per-position KL, both executors/probes | 2.10033e-8 | 4.09313e-16 |
| Minimum cosine | 0.999998402164 | 0.9999999999999749 |
| Greedy agreement, probe A per executor | 6/8 | 8/8 |
| Greedy agreement, probe B per executor | 8/8 | 8/8 |
| Largest positive relative PPL change | 1.49878e-5 | 1.14994e-8 |
| Actual weight blob bytes | 98,880 | 107,136 |
| Mandatory admission | Rejected; no output published | Passed; strict public generation executed |

The additional 8,256 bytes are the cost of honoring the requested FP32 storage.
This change claims numerical preservation, not a memory or speed improvement.
Admission worker wall times and sampled RSS are reported separately; no clean
serving tokens/s or TTFT comparison was run for this correction.

Raw evidence remains under `build/mission/fp32-packaging-before-2.*`,
`fp32-packaging-after.*`, `numerical-live-5.log`, `numerical-live-6/`,
`numerical-live-6.log` and `numerical-live-comparison.json`. The before run really
completed all workers and failed its gate; it was not assigned an artificial
passing result. A later replay was RAM-refused before loading HF, as recorded
in the admission report.
