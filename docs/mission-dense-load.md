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
with only P3 enabled for uniform FP32 packaging. Current
compilation and baseline compilation require normal original-Transformers
quality admission. Both timed readers load the identical current certified
artifact and record its manifest digest. Separate baseline compilation remains
available for artifact-file comparison, without changing the timed input.

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

The first hosted attempt on revision `7d5cab5` stopped at configuration
validation: all passes off plus an explicit FP32 request is deliberately
invalid. Its original-HF reference ran, but compilation, load and generation
did not run. The corrected harness selects only P3, preserving the compiler's
precision contract, and tests this configuration against real validation.

A real-artifact regression also found two compiler discoveries per load worker:
warming a standalone provider left the loader's process-wide provider cold.
The harness now warms `get_native_kernels()` itself. The regression observes
one discovery; it failed with two before this correction. Any earlier timings
include that additional discovery and must not be labeled as pure warmed load.
The first corrected-warmup report still compared certified versus diagnostic
artifacts. It is retained as historical evidence; the final command requires
ordinary admission for both compilers and times one shared artifact, so their
reader comparison cannot be explained by different certification payloads.

No pass or hardware tier is promoted by adding this measurement. All 22 pass
dispositions and backend limits remain those in the
[latest pass and hardware tables](mission-mamba2-source-parameters.md).

## Owned-buffer change

`dequantize_tensor` now accepts keyword-only `copy=False`. Its default still
produces independent storage. The vocabulary validator and tensor-loading loop
opt in because `WeightStore.load_tensor` returns fresh writable allocations.
The engine retains those allocations through NumPy views. Separate model loads
remain independent, and closing or removing the weight file cannot invalidate
the loaded arrays. This does not introduce mapped inference weights or remove
resident decoded FP32 arrays for compressed codecs.

This uses the existing [NumPy `astype` ownership contract](https://numpy.org/doc/1.26/reference/generated/numpy.ndarray.astype.html).
FP16 conversion, uint16 BF16 expansion, block decoding and padding retain their
previous arithmetic. Read-only input stays read-only only when reuse is
explicitly requested; byte-order or dtype conversion still allocates. The
shared decoder, direct source/kernel/merge callers, their tests and smoke
scripts were read before the change. The similarly named GGUF reader method
does not import this shared function.

Before the change, six default-ownership checks passed and eleven ownership/
integration checks failed. Afterward all 17 pass, including a real compiled AEG
whose loaded weights survive closing and deleting the blob. The related codec,
loader, source ingestion, pruning and merge selection passes: 552 passed,
2 skipped. The new test module and benchmark pass Ruff and mypy. For the two
edited core modules, diagnostic comparison retains the existing 82 Ruff and
48 mypy findings with no added messages; this is not a clean core typecheck.

The first hosted baseline (`bdf20e3`, CI run `40089097194`) passes all 24 jobs:
the complete pinned-dependency suite has 5,541 passed, 51 skipped, zero failures,
and 45,277/57,445 covered lines. Actual tiny/27M CPU, Torch CPU, ONNX CPU, cached
continuation, SDK generation/streaming and CLI outputs pass the source checks.
Independent download verification reopens 16 paired arrays and checks 528
worker source hashes, 44 owned worker identities and 50 retained artifact files.
Twenty omitted payload files were hashed and verified inside the worker, and
were not independently reopened after download. The earlier compiler-discovery
timings remain historical measurements rather than warmed-load claims.

The local paired probe on the same Windows host uses one actual 8,388,608-byte
FP32 payload, seven warmups and 31 alternating samples per mode. Separate traced
measurements have 11 samples. Outputs match the input's uint32 bits in both modes.

| Isolated decoder observation | Default copy | Explicit reuse |
| --- | ---: | ---: |
| Median time | 6.0033 ms | 0.0489 ms |
| Minimum / maximum time | 5.0215 / 9.5545 ms | 0.0294 / 0.1320 ms |
| Traced current allocation range | 8,389,144 B | 440–5,072 B |
| Traced peak allocation range | 8,389,285 B | 581–5,072 B |

This establishes removal of the redundant payload allocation. It is not a
generation speedup, process peak-RSS reduction, or result for quantized matrix
execution. Full-suite validation of this change and the corrected fresh-process
27M A/B measurements remain pending until their runs finish.
