# Dense load measurement

The architecture remains the one recorded in
[the pre-change architecture note](mission-architecture-note.md): engines execute
their weight containers; graph rewrites do not select runtime kernels.
`WeightStore.load_tensor` reads each payload into a new writable NumPy allocation.
The dense loader then calls `dequantize_tensor`, whose FP32 passthrough creates
another array. This is a confirmed redundant copy, distinct from mmap support
and from decoding genuinely quantized weights into resident FP32 projections.

Before changing ownership, the public decoder's default independent-copy
behavior must remain intact for existing compiler, merge, kernel and test
callers. Only a consumer that owns its input may safely opt into reuse.

## Executed local baseline

On 2026-10-24, Windows build 26200, Intel Core i5-1334U, 12 logical CPUs,
8,259,321,856 bytes RAM, Python 3.10.11 and NumPy 1.26.4, a real
2,048 by 1,024 FP32 payload at revision
`66f0e101d31f12121d07445c5ecb798ae155defd` produced an independent output:

| Observation | Measured baseline |
| --- | ---: |
| Input payload | 8,388,608 bytes |
| Additional traced current allocation | 8,388,800 bytes |
| Additional traced peak allocation | 8,388,941 bytes |
| Decode median, 31 samples after 7 warmups | 3.9168 ms |
| Decode minimum / maximum | 3.5555 / 8.8027 ms |
| `astype` cumulative time, 100 profiled decodes | 0.314 / 0.317 s total |

These are a local isolated decode baseline, not a full-model load improvement,
RSS attribution or generation speed claim. No production source was changed for
this baseline. The probe confirmed that writing the decoded array leaves the
encoded input unchanged.

Raw records: [decode samples](measurements/dense-fp32-copy-baseline.json),
[cProfile](measurements/dense-fp32-copy-baseline-cprofile.txt), and
[full-model admission refusal](measurements/dense-local-admission.json).

The new full-model command refused the 27,271,680-parameter fixture before
allocation: 681,508,864 bytes were available, below its unchanged 948,086,784-byte
fixture reserve plus 536,870,912-byte headroom. This refusal is not a model run.

## Reproduction and measurement contract

```powershell
python tools/validate_dense_hf.py --output dense-results --baseline-src historical-dense/src
```

The historical checkout must contain the intended baseline `src` tree. The tool
records both Git revisions and worker source-file hashes. It uses deterministic,
complete untrained tiny and 27M Llama checkpoints from `pass_audit_fixtures.py`,
with all 22 optimizer passes explicitly disabled and FP32 packaging. Current
compilation requires the normal original-Transformers quality admission;
historical compilation is explicitly diagnostic.

Actual CPU, Torch CPU and ONNX CPU outputs are checked against original HF
fixed-label logits and eight greedy tokens on short and 64-token prompts.
Checks include cached continuation, KL, cosine, fixed-label PPL, ordinary SDK
generation and streaming, and the ordinary streaming CLI. A failure stops the
command and retains its available evidence.

Load workers are separate owned processes with existing RAM guards. Native
library preparation occurs before the load timer. There are two warmup loads
per role, then seven retained samples with alternating current/baseline order.
Reports retain every load's time, sampled RSS, traced allocation, and the union
of contiguous weight-array spans. Inference reports separate allocated cache
array spans and logical KV bytes. This is warm filesystem/code measurement;
brief RSS peaks and native allocations are not fully observable by these tools.
The generation timer reports engine first output rather than server TTFT.

The pinned Ubuntu CI job runs the same command and uploads source/reference
arrays, worker identities, failures, raw samples and profiles. Complete compiled
payload hashes are retained; repeated 109 MB fixture/AEG/ONNX payloads are
regenerated instead of uploaded. The job is not evidence of success until it
has actually completed. Hosted measurements and before/after results remain
pending at this checkpoint.

No pass or hardware tier is promoted by adding this measurement. All 22 pass
dispositions and backend limits remain those in the
[latest pass and hardware tables](mission-mamba2-source-parameters.md).
