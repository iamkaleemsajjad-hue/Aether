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
regenerated instead of uploaded. The job is evidence of success only after it
has actually completed. Completed measurements and validation limits are
recorded in the final shared-artifact section below.

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

A further import-cache control reproduces Python `-B` reading the parent's
existing bytecode while compiling a cache-cold baseline. A real subprocess
audit observes compile counts `[0, 1]` before cache preparation and `[0, 0]`
after it. The tool precompiles both source trees with checked hashes before
launching workers, and records actual cache presence for each reader.
Earlier load timings can include unequal source-compilation work. Their
loading speed and allocation comparisons remain provisional; the completed
balanced run is reported below. The ownership regressions are separate.

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
execution. Full-suite validation and the controlled measurements follow.

## Final shared-artifact measurements

The final [CI run at
`d277e3d`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/40121721368) passes all 24
jobs; security passes separately. Its production source tree equals the loader implementation at
`54ca509`. Both compilers require ordinary original-HF admission. Every timed reader loads the
same current certified artifact; manifest digests, native-library build identities and dispatch
match across all readers. Both source trees have 302 checked-hash bytecode caches prepared
before workers, with cache presence verified in every measured load. The earlier 14.79%
load-time difference is not treated as a runtime speedup.

The physical dense host is an AMD EPYC 7763, four logical CPUs, 16,766,414,848 bytes RAM, Linux
6.17.0-1022-azure x86_64, Python 3.11.14, NumPy 1.26.4, Torch 2.9.1+cpu, Transformers 4.57.3,
SafeTensors 0.8.0, Tokenizers 0.22.2, psutil 7.2.2 and ONNX Runtime 1.23.2. Numerical threads
are fixed at one. Two warmup workers precede seven fresh measured workers per reader, with
alternating role order. Values are median [minimum,maximum]. MB below means 1,000,000 bytes.

| Fixture / observation | Old reader | Current reader |
|---|---:|---:|
| tiny / load ms | 562.663 [557.097,566.884] | 559.600 [556.689,568.042] |
| tiny / RSS load delta MB | 8.794 [8.716,8.860] | 8.782 [8.741,8.823] |
| tiny / sampled load-phase peak RSS MB | 68.653 [68.448,71.033] | 68.731 [68.583,71.213] |
| tiny / traced peak MB | 4.088 [4.088,4.088] | 4.089 [4.088,4.089] |
| mid / load ms | 726.886 [716.155,747.468] | 725.583 [717.066,736.729] |
| mid / RSS load delta MB | 132.149 [132.092,132.186] | 121.905 [121.897,122.016] |
| mid / sampled load-phase peak RSS MB | 192.057 [191.816,192.090] | 185.663 [185.532,185.676] |
| mid / traced peak MB | 118.940 [118.940,118.940] | 118.944 [118.944,118.944] |

For the 27M fixture, median load time is effectively flat (726.886 to 725.583 ms) with
overlapping ranges; no load speedup is assigned. The RSS load delta decreases by 10,244,096
bytes and sampled load-phase peak RSS decreases by 6,393,856 bytes. Traced peak is almost
unchanged: this experiment does not establish a large peak-tracemalloc reduction. RSS includes
allocator retention, code and metadata, so its component attribution is not exact. Sampling can
miss brief peaks, and cold-filesystem load was not measured.

| Fixture | Logical weight array bytes, old = current | Allocated cache array spans / logical KV bytes, old = current | Same timed artifact bytes |
|---|---:|---:|---:|
| tiny | 107136 | 18048 / 16384 | 213189 |
| mid | 109086720 | 1062912 / 1048576 | 109323657 |

The weight blobs are identical between compilers; independent package metadata can differ
slightly. There is no artifact compression or steady-state logical-weight/KV reduction.
Activation and general runtime overhead are not separately attributed.

| Fixture / executor | Worst-position KL | Minimum cosine | Greedy agreement | Worst relative fixed-label PPL change |
|---|---:|---:|---:|---:|
| tiny / CPU | 9.70878e-16 | 0.999999999999976 | 16/16 | +8.0507388e-07% |
| tiny / torch_cpu | 9.96082e-16 | 0.999999999999973 | 16/16 | +5.6359673e-07% |
| tiny / onnx_cpu | 9.19353e-16 | 0.999999999999982 | 16/16 | +3.5377408e-07% |
| mid / CPU | 1.19133e-13 | 0.999999999999430 | 16/16 | +2.4361592e-05% |
| mid / torch_cpu | 3.67185e-14 | 0.999999999999827 | 16/16 | +3.2068374e-06% |
| mid / onnx_cpu | 1.15703e-13 | 0.999999999999443 | 16/16 | +2.4642257e-05% |

CPU saved logits are byte-identical before/after on both probes for each fixture. Actual
CPU/Torch/ONNX, cached continuation, public CPU/Torch SDK generation and streaming, and the
ordinary CLI pass. These are complete untrained FP32 models, two probes and four fixed
continuation labels per probe; corpus PPL, trained-model utility, larger contexts and 8B
performance remain unverified.

| Fixture / prompt IDs | Old decode IDs/s | Current decode IDs/s | Old first-output ms | Current first-output ms |
|---|---:|---:|---:|---:|
| tiny / 4 | 2171.85 [2099.09,2172.21] | 2138.74 [1942.54,2161.26] | 0.572 [0.529,1.224] | 0.575 [0.546,1.116] |
| tiny / 64 | 2133.06 [2105.74,2155.71] | 2070.72 [1972.87,2100.76] | 0.960 [0.947,1.360] | 0.954 [0.944,1.050] |
| mid / 4 | 97.94 [96.31,99.59] | 94.90 [94.47,96.62] | 25.206 [24.783,25.675] | 25.614 [25.066,26.087] |
| mid / 64 | 93.12 [91.97,96.34] | 92.37 [89.39,92.85] | 65.794 [65.685,67.001] | 65.970 [65.555,66.267] |

Generation timings are seven repetitions within one worker per reader, after two warmups. They
are not interleaved independent-process inference trials or server TTFT. The 27M decode medians
decrease by 3.11%/0.80% for short/long prompts; all four decode rows are slower in this run.
First-output medians increase in three rows. These losses remain visible. A generation speedup
and broad timing non-regression are not established by this loader change.

The complete hosted suite has 5,560 passed, 51 skipped, zero failures/errors in 169.600 s. Final
line coverage is 45,274/57,445 = 78.812777439%. This is three covered lines below the first
dense baseline and fifteen below the intermediate `54ca509` run despite an identical production
source tree. The lost/gained line list is retained; the cause of this execution-path variation
is unresolved. All 14 current Mamba-2 and nine current Mamba-1 source contracts also pass
admission/API/CLI controls. Historical failing Mamba-1 candidates remain in the raw controls.

The complete local Windows run has 5,564 passed, 43 skipped and 3 failures in 439.624 s, with
45,285/57,445 = 78.831926190% covered lines. The failures are the two existing MCP HTTP framing
cases and audit-worker RAM admission (160,980,992 bytes available versus 406,938,624 reserve
plus 134,217,728 headroom). The isolated audit repeat also refuses before allocation at
354,521,088 available bytes. The 64 MiB allocation control skips for low RAM. No test, admission
reserve or guard was relaxed; the local suite remains red.

Independent verification reopens 16 paired array comparisons and 52 retained artifact files,
checks 528 source-file hashes and 44 actual worker identities, and verifies the shared
manifest/native dispatch. Twenty large payloads were worker-hashed/integrity-checked and are not
independently reopened after download. GitHub's temporary merge `4438390` has the same complete
Git tree as `d277e3d`, confirmed through the API and local Git objects.

Raw records: [final report](measurements/dense-owned-hf.json), [independent
metrics](measurements/dense-owned-hf-summary.json), [provenance and all validation
limits](measurements/dense-owned-provenance.json), [dependency
freeze](measurements/dense-owned-freeze.txt), [local paired allocation
samples](measurements/dense-owned-local.json), [initial
baseline](measurements/dense-baseline-initial-hf.json), [provisional different-certification
comparison](measurements/dense-owned-provisional-hf.json), [provisional unequal-cache
comparison](measurements/dense-owned-source-cache-provisional-hf.json), [Mamba-1
controls](measurements/dense-owned-mamba1-controls.json) and [Mamba-2
controls](measurements/dense-owned-mamba2-controls.json). The four dense cProfile files in
`docs/measurements/dense-owned-*-cprofile.txt` preserve measured inference hot paths.

## Pass and hardware disposition

This loader change does not promote an optimizer pass. The prior dispositions
remain in force; this area measures ordinary FP32 loading with P3 selected.

| Pass | Default/disposition | Runtime benefit status |
|---|---|---|
| 1 fusion | Off, experimental IR | No schedule consumer |
| 2 sensitivity | Off, diagnostic | Reconstruction proxy |
| 3 precision | On | Stored formats consumed; these checks use FP32 |
| 4 KV structure | Off, experimental IR | Engine caches independent |
| 5 MoE tiers | Off, experimental hints | Placement benefit unproved |
| 6 parallelism | Off, inspection | No device-placement consumer |
| 7 reasoning | Off, experimental IR | Serving/quality benefit unproved |
| 8 sparse attention | Off, experimental lossy | Dense masking; sparse benefit unproved |
| 9 pruning | Off, experimental lossy | No sparse-kernel benefit |
| 10 MTP | Off, supplied heads | End-to-end benefit unproved |
| 11 grammar | Off, explicit functional | FSM consumer; full constrained A/B pending |
| 12 merging | Off, explicit transform | Weights consumed; task quality requires evaluation |
| 13 TTT | Off, experimental | Useful slot/model effect unproved |
| 14 semantic KV | Off, experimental lossy | CPU consumer; quality losses recorded |
| 15 shared KV | Off, experimental lossy | CPU consumer; quality losses recorded |
| 16 green | Off, accounting | No measured energy reduction |
| 17 TEE | Off, unsupported execution | No attested backend kernel |
| 18 MDLM | Off, supplied trained weights | Full draft/verify quality and benefit unproved |
| 19 sub-2-bit | Off, experimental lossy | No packed-kernel benefit |
| 20 video | Off, unsupported execution | No validated end-to-end VLM path |
| 21 PEFT | Off, explicit adapter | Broader task parity unresolved |
| 22 RLVR | Off, experimental training | Gradient backend required |

| Backend scope | Verification tier and technique in this area |
|---|---|
| NumPy/native CPU | Physical Windows i5-1334U ownership/codec/AEG checks and complete local run; physical hosted EPYC tiny/27M HF, cached decode and same-artifact reader A/B |
| Torch CPU | Physical hosted CPU tensors, original-HF logits/tokens, ordinary SDK/streaming and CLI; 23 current Mamba source controls in a separate hosted CPU job |
| ONNX CPU | Actual ONNX export, CPUExecutionProvider sessions, cached continuation and original-HF parity on both dense fixtures |
| CUDA, ROCm, MPS, DirectML and other targets | No new physical-device execution; existing matrix tiers remain unchanged |

Generation non-regression, trained/corpus quality, larger models, mapped
inference weights, packed quantized execution, long-context serving memory,
full pass composition, additional hardware execution, archive hardening and
blocking core/runtime types remain open. The measured inference profiles and
the engine's repeated normalization before native Q projection provide a
concrete next profiling target; any arithmetic change still needs independent
reference equations, numerical gates and before/after measurements. The
[hardware matrix](../hardware_validation_matrix.json), generated README table
and [pass audit](mission-pass-audit.md) retain their existing scopes.
