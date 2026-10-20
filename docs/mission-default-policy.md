# Measured default-pass policy correction

Executed on 2026-10-18, Intel Core i5-1334U, Windows AMD64, Python 3.10.11,
NumPy 1.26.4, Torch 2.9.1+cpu and Transformers 4.57.3. This is an incomplete
mission checkpoint. The architecture decision remains engine-owned execution.

## Confirmed issues and correction

Read scope covered compiler/config/constants, the optimizer and all registered
pass implementations, their engine/loader consumers, public callers, affected
unit/integration tests, validation script and optimizer/audit documentation.
CPU/Torch execution does not interpret fused IR groups. P2's reconstruction
scores do not change default P3's BF16 selection; P4/P5/P6/P7 annotations do not
establish their advertised serving benefits. Source MoE routing and runtime KV
caching operate independently of those passes.

P8 is a real consumer, not a graph-only no-op: merely attaching its plan prevents
native CPU decode attention even when the prompt is below the masking threshold.
Its masking path still computes dense scores. Default-on therefore imposed an
unnecessary dispatch change without long-context quality or sparse-kernel evidence.

P1/P2/P4/P5/P6/P7/P8 now require explicit opt-in. Only P3 remains enabled by
default; P9–P22 were already off. Existing explicit flags remain available.
The pipeline constructor now reflects the actual config, and public pipeline
toggles survive config synchronization. Selected passes receive an effective
config with the same canonical flags, including internally gated grammar
compilation; the caller's config and aliases are not mutated.

Packaging now persists sharding plans only from an applied P6 producer, rather
than manufacturing generic GPU plans when P6 is disabled. Fusion reports count
actual groups/patterns, leave skipped fusion empty, and claim zero runtime
launch/memory savings. Planning descriptions and SDK flag docs no longer promise
unmeasured throughput, perplexity, energy or hardware gains.

Regression evidence before correction: initial policy/toggle tests returned
5 failed / 1 passed; added packaging/preview checks returned 2 failed / 1 passed;
the internal grammar-gate pair returned 1 failed / 1 passed. The focused
compiler/optimizer/harness selection finally passed 200 tests with 2 skips.
The final complete-suite and static results are recorded in the
[validation journal](mission-validation.md).

## Serial measurements and limits

`tools/audit_default_passes.py` uses the existing real-checkpoint audit workers,
sampled worker identities, admission guards and every-file artifact diffs. It
records actual config flags and rejects a declared pass list that differs from
them. Both controller and workers use this checkout's source. The optional
`--historical-baseline` reconstructs the former seven-on policy explicitly, so
the comparison remains repeatable after defaults change.

Fixtures: untrained deterministic FP32 tiny Llama (26,784 parameters) and MoE
(63,904 parameters, four experts, top-2 routing), created twice with all source
file hashes matching. Both declare context 32768; actual prompts contain 4/64
IDs. This is **not a 32k-token benchmark**. Execution uses native CPU kernels,
one BLAS/OpenMP thread, eight output IDs, fresh request caches, two warmups and
seven timing repeats per prompt. Three fresh cold-cache builds alternate profile
order. Timings exclude loading; TTFT includes prefill through first yielded ID.

The table compares the historical flags with current defaults using the same
post-correction source and fixture. Values are observations on a busy,
memory-constrained host, with an explicit 128 MiB free-memory reserve. They are
not clean-host, pretrained-model or production-performance certification.

| Fixture/metric | Former seven-on flags | Current defaults |
|---|---|---|
| Tiny decode tok/s, median [min,max] | 898.1 [590.2,1276.7] | 1244.3 [1080.9,1947.8] |
| Tiny TTFT ms, median [min,max] | 1.553 [1.151,2.017] | 1.395 [1.105,2.067] |
| Tiny compile + integrity seconds, median [min,max] | 3.445 [3.436,3.761] | 2.821 [2.759,2.905] |
| Tiny artifact bytes / files | 197446 / 26 | 190982 / 16 |
| Tiny sampled generation worker RSS MiB | 58.965 | 58.762 |
| MoE decode tok/s, median [min,max] | 653.5 [417.2,815.6] | 839.8 [554.5,889.8] |
| MoE TTFT ms, median [min,max] | 2.137 [1.759,2.636] | 3.365 [2.410,3.658] |
| MoE compile + integrity seconds, median [min,max] | 2.651 [2.568,2.812] | 2.646 [2.380,2.777] |
| MoE artifact bytes / files | 313634 / 26 | 303915 / 16 |
| MoE sampled generation worker RSS MiB | 59.332 | 59.117 |

The initial MoE TTFT median worsened. A targeted serial recheck used three fresh
worker blocks per profile, alternated profile order, five warmups and 31 repeats
per block. All samples, including outliers and the original run, are retained.

| Fixture/metric, 93 recheck samples per profile | Former seven-on | Current defaults |
|---|---|---|
| Tiny decode tok/s, median [min,max] | 891.2 [205.6,1257.9] | 1183.3 [321.9,1973.8] |
| Tiny TTFT ms, median [min,max] | 1.624 [1.028,5.856] | 1.862 [1.100,4.573] |
| MoE decode tok/s, median [min,max] | 528.5 [69.7,874.0] | 895.2 [300.5,1258.1] |
| MoE TTFT ms, median [min,max] | 3.094 [1.690,20.648] | 2.560 [1.671,6.542] |

Decode medians improved in both cohorts. TTFT direction changed between cohorts;
there is no universal TTFT improvement or proven absence of latency regression.
The small sampled RSS differences do not establish a resident-memory gain.
Weights/KV/activations/overhead are not separately attributed. This policy removes
unverified defaults and restores an existing decode path; it does not constitute
the requested general engine optimization or certify every speed/memory tradeoff.

Separate `cProfile` requests after two warmups confirmed the mechanism on both
fixtures: former flags made 18 `_attention` calls and zero native `flash_attn`
calls; current defaults made two `_attention` prefill calls and 16 native decode
calls. All eight generated IDs matched each profile's numerical worker. Profiler
times are not the benchmark samples, and the function name does not certify a
specific FlashAttention paper implementation.

Both profiles preserved exact teacher-forced logits (maximum absolute error 0,
KL 0), all 16 generated IDs, and fixed-label continuation NLL/PPL across A/B.
They also matched the corresponding pre-change-code artifacts numerically.
Tiny's actual original-FP32 HF oracle ran: maximum KL `5.449e-8`, minimum cosine
`0.99999538`, greedy agreement 100%, maximum fixed-label PPL increase 0.01509%.
MoE has no executed HF oracle. Synthetic agreement is not corpus/task quality.
Every file in each profile's three repeated builds matched byte-for-byte;
runtime execution did not mutate an artifact. Source/runtime hashes and raw
inventories remain in the reports. Existing artifacts still load their stored
plans; newly compiled graph hashes can differ from older defaults.

An earlier run against unchanged production code also observed exact A/B logits
and faster decode with the seven flags explicitly off. Its mid fixture was
created twice (27,271,680 parameters), but compilation and HF were blocked by
memory admission. The later small-only audit did not silently count mid as run.
The first attempt blocked even tiny workers and remains recorded as such.

Small-fixture HF admission now estimates 512 MiB plus independent live headroom,
based on retained actual small-oracle tree RSS below 300 MiB. Mid retains its
1 GiB estimate; non-HF overhead and live headroom checks remain intact. These are
admission estimates, not measured model footprints or allocation guarantees.

Evidence under `build/mission/`: `default-policy-before/` (blocked),
`default-policy-before-retry/report.json` (mid blocked, exit 1),
`default-policy-after/report.json` (tiny/MoE execution, exit 0),
`default-policy-after/timing-recheck/report.json` (exit 0), each profile's
`dispatch-profile.json` / `.pstats`, and `default-policy-measurements.json`.
Exact dependency versions remain in `clean-freeze.txt`.

## All 22 dispositions and remaining work

| Pass | Default / disposition | Evidence and limitation |
|---|---|---|
| 1 fusion | Off; experimental IR | Actual group reporting; no engine schedule consumer |
| 2 sensitivity | Off; experimental diagnostic | Reconstruction proxy, no default P3 decision change |
| 3 precision | On | Encoded weights consumed; default BF16; explicit FP32 preservation corrected; publication's independent numerical gate is described below |
| 4 KV structure | Off; experimental IR | Engine caches independent; duplicate-application defect remains |
| 5 MoE tiers | Off; experimental hints | Source router/top-k remain executable; placement benefit unproved |
| 6 parallelism | Off; experimental inspection | Applied producer required for files; no device-placement consumer |
| 7 reasoning | Off; experimental IR | No demonstrated serving execution/quality gain |
| 8 sparse attention | Off; experimental approximation | Native dispatch restored by default; long-context quality unverified |
| 9 pruning | Off; experimental lossy | Zeroed dense weights consumed; no sparse-kernel benefit |
| 10 MTP | Off; experimental supplied heads | Full requested-head verification and benefit remain open |
| 11 grammar | Off; explicit functional feature | Actual tokenizer-aware FSM emitted in toggle regression; full constrained A/B pending |
| 12 merging | Off; explicit model transform | Changed weights consumed; task quality requires evaluation |
| 13 TTT | Off; experimental | Useful adaptation and slot-to-model effect remain unproved |
| 14 semantic KV | Off; experimental lossy | CPU retention consumed; synthetic generation regressions recorded |
| 15 shared KV | Off; experimental lossy | CPU pointers consumed; synthetic quality regressions recorded |
| 16 green | Off; experimental accounting | No measured device energy reduction |
| 17 TEE | Off; unsupported execution | No backend-emitted attested kernels; explicit CLI request refused |
| 18 MDLM | Off; experimental supplied trained heads | Missing weights refused/skipped; real draft benefit unproved |
| 19 sub-2-bit | Off; experimental lossy storage | Dense dequantization; independent compiler numerical gate added; optimized runtime kernel and QAT quality remain unverified |
| 20 video | Off; unsupported CPU execution | No end-to-end VLM benefit verified |
| 21 adapter | Off; experimental supplied tensors | Dense adapter consumer; source scaling/parity unresolved |
| 22 RLVR | Off; experimental training integration | Gradient callbacks required; generation cannot validate training |

The [pass reference](optimizer-passes.md) records flags and consumers; the
[historical 22-row audit](mission-pass-audit.md) preserves the individual measured
losses. Full ordering/idempotency, model-family
math, trained-model and larger-model performance, memory attribution/soak,
hardware execution, archive/download robustness and blocking CI remain open.
Current CPU evidence is physical execution; historical DirectML evidence is not
rerun, and no accelerator tier is promoted by this correction.

On 2026-10-20, [independent numerical admission](mission-numerical-admission.md)
was added before ordinary publication/loading, comparing actual CPU and Torch
CPU against original FP32 HF. It is separate from P3 and checks every compiled
base candidate. Diagnostic audits explicitly retain uncertified execution so
they can measure rejected experimental cases. This does not establish runtime
benefits for the remaining passes or certify request-time transformations.
