# Mamba-2 finite-input numerical fidelity

The CPU SSD engine now evaluates SiLU and softplus without clipping their
finite inputs to +/-60. The [execution architecture](mission-architecture-note.md)
remains engine-owned: this changes the handwritten forward path, with no graph
rewrite, weight transformation, new artifact field or cache-layout change.

## Reference and reproduction

Read the complete Mamba-2 CPU engine, fixture, normalization/convolution tests,
strict HF tool, Torch state engines and the cache/session/worker call paths.
The primary [Mamba-2 paper](https://arxiv.org/html/2405.21060v1), including
appendices, was read in the earlier normalization area. Rechecked its reference
[`Mamba2Simple`](https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/modules/mamba2_simple.py)
and [`SSD minimal`](https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/modules/ssd_minimal.py).
The official v2.2.4 tag was resolved to exact commit
`95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` through the GitHub API. The installed
Transformers 4.57.3 reference uses SiLU for the post-gate norm and its configured
convolution activation, and softplus for projected time steps.

The source equations are `SiLU(x) = x * sigmoid(x)` and
`delta = log(1 + exp(raw_dt + bias))`. The SSD recurrence remains
`h = exp(delta*A)*h_previous + delta*x*B`, with the learned D skip, SiLU gate,
learned output RMSNorm and output projection. It does not substitute the exact
continuous-system ZOH input discretization. See paper equations 2/3 and section
7.1, plus the [SiLU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.silu.html)
and [softplus](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.softplus.html)
reference definitions.

Before this fix, large positive time steps saturated at roughly 60. Large
negative convolution/gate values acquired false nonzero activations: finite
float32 input `-1e20` returned about `-8.7565e-7`, while the independent FP64
reference rounds to zero. At the largest negative finite float32 value, the
old result was about `-2.98e12`. These are changed functions, not an overflow
avoidance identity.

Complete deterministic, untrained HF checkpoints isolate large delta, negative
gate and negative X-convolution inputs alongside a default control. B/C weights
and convolution channels remain nonzero; no fabricated target logits or zero
recurrence replaces execution. The independent FP64 oracle constructs the full
semiseparable matrix and full causal convolution, without calling a runtime
recurrence or activation. Comparisons against the original HF model separately
validate this FP64 oracle.

The initial pre-fix regression returned **five failures, nine passes** in 75.72
seconds. Replaying the final tests against the exact retained Git archive
reproduced the same five failures and nine passes in 32.83 seconds. These are
test-run durations under changing host load, not inference benchmarks.
Both scalar dtype checks and the three CPU stress forwards failed. All four
oracle comparisons against original HF, all four Torch CPU controls and the
default CPU control passed. Maximum absolute CPU/model logit errors against the
matrix oracle were **2.10043** (large delta), **1.89717** (negative gate), and
**1.56770** (negative convolution). This distinguishes the CPU defect from an
incorrect source fixture or oracle.

The initial correction uses `logaddexp(0, x)` for softplus. For SiLU, set
`t = exp(-abs(x))`; the sigmoid is `t/(1+t)` for negative x and its complement
for nonnegative x. The exponential never receives a positive argument.
Temporary results are owned arrays, and the input is not modified. Scalar
float32/64 tests include the finite dtype endpoints and use the dtype's normal
precision floor for underflow-sized results. No input threshold or empirical
model-dependent approximation replaces the equations.

Focused validation returned **112 passed** in 79.39 seconds, including all 14
new cases. It covers full prefill, cloned cache continuation, the final-vector
and all-row protocols, convolution-history bounds, original-HF oracle checks,
existing convolution/epsilon contracts and actual artifact round trips.

## Validation and measurement scope

Local checks ran on Windows x86 CPU with Python 3.10.11, NumPy 1.26.4, Torch
2.9.1 and Transformers 4.57.3. The complete run returned **5,473 passed, 42
skipped, two failures** (5,517 cases); all 14 new cases passed. The two failures
are the existing MCP HTTP handshake cases, also present in the previous complete
local run. The local suite remains red. The earlier
[wire observation](measurements/mcp-local-framing-observation.json) records
the framing mismatch without weakening the HTTP tests or claiming an environment
fix. Existing Ruff findings remain **19**, and scoped existing
mypy errors **40**, with no new file/code/message diagnostics. The new test
file passes scoped mypy. Critical Ruff and workflow YAML parsing pass. Existing
static backlog is not cleared. The
[local validation record](measurements/mamba2-stability-local-validation.json)
binds JUnit, coverage, exact baseline archive and source hashes.

| Complete local suite | Previous epsilon area | Stable math |
|---|---:|---:|
| Passed / skipped / failed | 5,459 / 42 / 2 | 5,473 / 42 / 2 |
| Covered / executable lines | 45,186 / 57,356 | 45,202 / 57,360 |
| Line coverage | 78.7816445% | 78.8040446% |
| Covered / executable branches | 10,837 / 17,290 | 10,838 / 17,290 |

The strict [HF harness](../tools/validate_mamba2_hf.py) includes the three stress
cases in addition to its default, expanded/biased and zero-epsilon controls.
Its blocking CI baseline is immediate pre-fix
`c7af1b6331681b9e8e6b337cb5c06368825143b2`. Earlier epsilon measurements remain
in [their original report](mission-normalization-epsilon.md).

The first local strict attempt executed an original-HF reference, then the
compile's own reference worker was refused by RAM admission: **785,047,552**
available bytes versus **536,923,100** reserved worker bytes plus
**268,435,456** bytes of live headroom. The raw partial result is retained under
`build/mission/mamba2-stability-hf-local/`, with an LF-normalized, JSON-identical
[retained copy](measurements/mamba2-stability-local-hf-attempt.json); no current admitted artifact or
benchmark success is assigned to that refusal. In-process pytest comparisons
are separate evidence and do not establish subprocess admission.

Raw local source snapshots, static checks and JUnit files use
`build/mission/mamba2-stability-*`. The exact 304-file pre-fix runtime Git
archive is retained and verified there. Peak inference RSS, activation/overhead
attribution, trained/corpus quality and controlled larger-model performance
require their own measurements. Finite weights alone do not guarantee finite
intermediate model arithmetic for every possible input.

## Hosted comparison at `5dbe26d`

[All 23 CI jobs passed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39903522524);
[security passed separately](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39903522529).
The complete single hosted run returned **5,466 passed, 51 skipped, zero
failures** (5,517 cases). All six Mamba-2 contracts and nine Mamba-1 controls
passed mandatory original-HF/CPU/Torch CPU admission, public API/streaming and
CLI checks. These are complete untrained FP32 checkpoints, not trained/corpus
or accelerator validation.

The [raw Mamba-2 report](measurements/mamba2-stability-hf.json),
[Mamba-1 controls](measurements/mamba1-control-during-mamba2-stability.json),
[provenance](measurements/mamba2-stability-provenance.json) and
[dependency freeze](measurements/mamba2-stability-freeze.txt) retain all cases.
Verification matched **270** reported implementation-file hashes to the exact
Git revisions, both fixture hashes and **105** source/reference files. The
baseline is `c7af1b6`; Mamba-1 retains its separate historical comparison.

Hardware was AMD EPYC 7763, four logical CPUs and 16,766,414,848 host RAM bytes;
Linux 6.17.0-1022 Azure x86_64, Python 3.11.14, NumPy 1.26.4, Torch 2.9.1+cpu,
Transformers 4.57.3, SafeTensors 0.8.0, Tokenizers 0.22.2 and psutil 7.2.2.
The probes use one/64 prompt IDs and four fixed-label continuation positions
each. Greedy comparisons use eight generated IDs per probe.

| Stress contract | Worst KL, before → after | Minimum cosine before | Max absolute error, before → after | Greedy IDs, CPU before → each executor after |
|---|---:|---:|---:|---:|
| Large delta | 0.328273 → 1.85889e-14 | -0.928404 | 1.87893 → 4.76837e-7 | 4/16 → 16/16 |
| Negative gate | 0.155047 → 1.34107e-15 | -0.0150425 | 1.43668 → 1.19209e-7 | 2/16 → 16/16 |
| Negative convolution | 0.219274 → 3.97123e-15 | -0.668373 | 1.52825 → 2.38419e-7 | 2/16 → 16/16 |

Across all six corrected contracts and both executors, minimum cosine was
**0.9999999999998017**, worst KL **1.85889e-14**, maximum absolute error
**4.76837e-7** and greedy agreement **16/16 per executor per contract**.
Default, expanded/biased and zero-epsilon controls remain within the declared
gate, though their saved CPU logits are no longer bit-identical after the
arithmetic change. Worst relative fixed-label PPL changes for large delta,
negative gate and negative convolution were respectively **-5.74002%, +35.1230%,
+37.8811%** before, versus **+0.0000111583%, +0.00000314421%, +0.00000389312%**
after. The lower historical PPL on the large-delta labels does not establish
source fidelity; its KL, cosine and greedy results fail.

| Source contract | Logical weight bytes, before = after | SSM state / history bytes, before = after | Artifact bytes, before → after | Compile seconds, diagnostic before → admitted after |
|---|---:|---:|---:|---:|
| Default | 10672 | 512 / 384 | 93902 → 101923 | 0.309359 → 8.95374 |
| Expanded/biased | 18112 | 1024 / 640 | 103568 → 111627 | 0.309359 → 6.14129 |
| Zero epsilon | 10672 | 512 / 384 | 93901 → 101965 | 0.306715 → 6.12517 |
| Large delta | 10672 | 512 / 384 | 93906 → 101968 | 0.308179 → 6.17393 |
| Negative gate | 11072 | 512 / 384 | 96504 → 104573 | 0.306492 → 6.14279 |
| Negative convolution | 10672 | 512 / 384 | 93908 → 101966 | 0.308190 → 6.16345 |

These are logical arrays, not peak RSS or activation/overhead measurements.
Artifact growth includes mandatory numerical-admission evidence; the diagnostic
baseline omits that verification workload. Compile times are consequently not
a like-for-like performance comparison.

Engine timing uses one BLAS/OpenMP thread, two warmups, seven measured repeats
and eight generated IDs without EOS stopping. Values are median [minimum,
maximum]. First output excludes artifact load and tokenization; it is not API
TTFT. The three stress cases also generate different baseline/corrected IDs.

| Source contract / prompt IDs | Decode IDs/s before | Decode IDs/s after | First output ms before | First output ms after |
|---|---:|---:|---:|---:|
| Default / 1 | 5271 [5056,5457] | 5132 [4953,5227] | 0.173284 [0.169737,0.201908] | 0.178483 [0.173325,0.217627] |
| Default / 64 | 5177 [4956,5325] | 5162 [5048,5236] | 8.58372 [8.52033,8.65178] | 8.82510 [8.75434,9.16187] |
| Expanded/biased / 1 | 5153 [4831,5233] | 5022 [4562,5161] | 0.182101 [0.174987,0.203010] | 0.185588 [0.183192,0.207348] |
| Expanded/biased / 64 | 5200 [5113,5263] | 4983 [4802,5062] | 8.80880 [8.77060,8.92203] | 9.17573 [9.11249,9.39586] |
| Zero epsilon / 1 | 5336 [5238,5517] | 5133 [4340,5259] | 0.174717 [0.167744,0.205535] | 0.178584 [0.173534,0.245189] |
| Zero epsilon / 64 | 5388 [5252,5452] | 5107 [4861,5222] | 8.42490 [8.38506,8.51809] | 8.81210 [8.72125,9.12119] |
| Large delta / 1 | 5185 [4146,5377] | 5077 [5034,5233] | 0.173886 [0.170760,0.196659] | 0.180077 [0.175528,0.192871] |
| Large delta / 64 | 5277 [5130,5384] | 5077 [4930,5162] | 8.50489 [8.48341,8.55019] | 8.93379 [8.77711,9.16178] |
| Negative gate / 1 | 5279 [4980,5395] | 4983 [4877,5213] | 0.178694 [0.172813,0.214672] | 0.181489 [0.180006,0.226133] |
| Negative gate / 64 | 5233 [5095,5300] | 4942 [4608,5161] | 8.55244 [8.52804,8.63252] | 9.03840 [8.87579,9.33815] |
| Negative convolution / 1 | 5390 [5314,5436] | 5184 [5005,5277] | 0.196076 [0.171049,0.208340] | 0.180127 [0.174667,0.214371] |
| Negative convolution / 64 | 5216 [5078,5418] | 5190 [4906,5229] | 8.54492 [8.50506,8.81927] | 8.83402 [8.77215,8.96892] |

All twelve decode medians decreased, by **0.29–5.61%**. Eleven first-output
medians increased, by **1.56–5.68%**; negative-convolution short-prompt latency
decreased **8.13%**. Shared-worker spread limits attribution. This is a measured
correctness improvement with unfavorable timing observations, not a speedup
or a no-regression certificate. Activation profiling and a lower-dispatch-cost
stable identity are being evaluated separately.

Hosted line coverage was **45,191 / 57,360 = 78.7848675%**, versus the previous
epsilon run's **45,192 / 57,356 = 78.7921054%**. The 78.6% blocking floor passed,
but the overall non-reduction criterion remains open. Mamba-2 covered lines
increased from 134 to 139; six previously covered lines in the unchanged R9
speculative rejection path were not hit, with one-line differences in two other
unchanged runtime modules. These runs are retained separately, not combined
into a more favorable coverage number.

## Profiling follow-up: stable log-sigmoid

The first implementation's timing losses motivated profiling before another
runtime edit. A local `cProfile` run of 300 default tiny-model generations
recorded 376,501 calls in 0.696 seconds; 5,400 SiLU calls accounted for about
0.068 seconds. The
[profile](measurements/mamba2-logsigmoid-cprofile.txt) is retained, with its
[measurement and validation record](measurements/mamba2-logsigmoid-local-validation.json).

The follow-up evaluates
`SiLU(x) = x * exp(-logaddexp(0, -x))`. The exponential argument remains
nonpositive for every finite x. This reduces NumPy dispatches and removes the
per-call sign mask while preserving the original function, dtype and input
ownership. Softplus, cache layouts and weights are unchanged.

Before adopting it, a single-process, randomized interleaved comparison used
the exact pre-fix engine module, the initial piecewise engine and a subclass
containing only the proposed activation. They shared identical loaded weights
and owned independent caches. All three produced identical eight-ID greedy
outputs on both default control prompts. Each had five warmups, 45 samples and
five complete generations per sample, with one BLAS/OpenMP thread.

| Default prompt | Pre-fix clipped generation ms | Initial piecewise generation ms | Log-sigmoid candidate generation ms |
|---|---:|---:|---:|
| One ID | 1.22818 [1.01470,2.18338] | 1.23554 [1.00422,2.11100] | 1.09740 [0.92976,2.32874] |
| 64 IDs | 9.03444 [7.53362,14.06198] | 9.36178 [7.92412,12.41560] | 8.51244 [7.32740,10.65932] |

Values are median [minimum, maximum] for complete eight-ID engine generation,
excluding artifact loading and tokenization. These local results support
testing the candidate; they do not replace hosted decode/first-output A/Bs,
larger-model profiling, peak RSS or activation/overhead attribution. The raw
45-sample series and exact module hashes are retained in the record.

The adopted identity passes all **112 focused checks**, including all 14
endpoint/original-HF/matrix/compiled-cache regressions, in 28.70 seconds.
Critical Ruff passes; the touched module retains **19** existing Ruff findings
and **30** existing mypy errors, with no new diagnostic messages. The complete
local run returned **5,470 passed, 42 skipped, five failures** in 543.62 seconds:
two existing MCP HTTP cases, a Unicode-worker RAM refusal (458,153,984 available
bytes versus 406,938,624 estimated worker bytes plus 134,217,728 headroom), and
two exact-equality Torch assertions outside the Mamba-2 execution path. Their
maximum differences were 2.9802322e-8 and 5.9604645e-8. All 14 Mamba-2 stability
cases passed.

A separate run of the three additional failures passed all three in 16.65
seconds. The two Torch cases also passed against the unchanged pre-stability
runtime in 6.89 seconds. This does not explain or erase their complete-run
failures. No numerical tolerance, failing test or resource guard was relaxed.
Coverage was **45,190 / 57,356 = 78.7886185%**, below the initial correction's
local **78.8040446%**; local non-reduction remains open. The hosted follow-up
below is a separate complete run.

## Hosted optimized result at `bd3db30`

[All 23 CI jobs passed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/40071277073)
and [security passed separately](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/40071277054).
The complete hosted run returned **5,466 passed, 51 skipped, zero failures**
in 145.439 JUnit seconds. All six Mamba-2 contracts and nine Mamba-1 controls
passed original-HF/CPU/Torch CPU admission, public API/streaming and CLI checks.
Coverage was **45,192 / 57,356 = 78.7921054%**, equal to the recorded epsilon
hosted run and above the initial piecewise run's **78.7848675%**. The local
failures and local coverage decrease are retained above.

The [raw optimized Mamba-2 measurements](measurements/mamba2-logsigmoid-hf.json),
[Mamba-1 controls](measurements/mamba1-control-during-mamba2-logsigmoid.json),
[provenance](measurements/mamba2-logsigmoid-provenance.json) and
[dependency freeze](measurements/mamba2-logsigmoid-freeze.txt) retain all cases.
Again, verification matched **270** implementation-file hashes, both fixture
hashes and **105** source/reference files against actual bytes and Git revisions.

This worker was **Intel Xeon Platinum 8573C**, four logical CPUs and
16,765,378,560 host RAM bytes, with Linux 6.17.0-1022 Azure x86_64. Python
3.11.14, NumPy 1.26.4, Torch 2.9.1+cpu, Transformers 4.57.3, SafeTensors 0.8.0,
Tokenizers 0.22.2 and psutil 7.2.2 were pinned. Compare current and pre-fix
`c7af1b6` within this worker; absolute rates from the earlier EPYC run are
not a controlled comparison against this Xeon run.

Across all six corrected contracts and both executors, worst KL was
**1.58106e-14**, minimum cosine **0.999999999999793**, maximum absolute logit
error **3.87430e-7**, greedy agreement **16/16 per executor per contract**,
and worst relative fixed-label PPL increase **0.00000413237%**.

| Stress contract | Worst KL, pre-fix → optimized | Max absolute logit error, pre-fix → optimized | Greedy IDs, CPU pre-fix → each optimized executor |
|---|---:|---:|---:|
| Large delta | 0.328273 → 1.04175e-14 | 1.87893 → 3.57628e-7 | 4/16 → 16/16 |
| Negative gate | 0.155047 → 1.55532e-15 | 1.43668 → 1.19209e-7 | 2/16 → 16/16 |
| Negative convolution | 0.219274 → 4.34282e-15 | 1.52825 → 2.38419e-7 | 2/16 → 16/16 |

Timing follows the same two-warmup, seven-repeat, one-thread, eight-ID protocol
as the initial hosted run. These are tiny, untrained 2,668–4,528-parameter
FP32 fixtures; first output is an engine measurement, not API TTFT.

| Contract / prompt IDs | Decode IDs/s pre-fix | Decode IDs/s optimized | First output ms pre-fix | First output ms optimized |
|---|---:|---:|---:|---:|
| Default / 1 | 8591 [8384,9051] | 9665 [8383,10029] | 0.109026 [0.105117,0.112548] | 0.107827 [0.098962,0.153011] |
| Default / 64 | 8421 [7702,8729] | 9032 [8553,9814] | 5.41704 [5.37742,5.63994] | 4.99575 [4.87679,5.03639] |
| Expanded/biased / 1 | 8742 [8574,8924] | 9225 [8581,9781] | 0.111479 [0.106463,0.141661] | 0.104953 [0.102857,0.115971] |
| Expanded/biased / 64 | 8134 [7733,8549] | 9161 [8585,9269] | 5.70316 [5.61278,5.92421] | 5.22402 [5.19318,5.40503] |
| Zero epsilon / 1 | 8935 [8798,9043] | 9602 [9197,9974] | 0.106950 [0.102795,0.109735] | 0.103945 [0.099568,0.142300] |
| Zero epsilon / 64 | 8651 [7637,8944] | 9579 [8510,9762] | 5.32293 [5.24010,6.56933] | 4.87388 [4.79839,5.06074] |
| Large delta / 1 | 8902 [8612,9010] | 9423 [8856,9850] | 0.108799 [0.105781,0.146753] | 0.106629 [0.097531,0.149792] |
| Large delta / 64 | 8683 [8475,8780] | 9108 [8795,9545] | 5.39401 [5.28945,5.61247] | 5.01133 [4.90482,5.23032] |
| Negative gate / 1 | 8739 [8207,8975] | 9120 [8938,9690] | 0.112295 [0.105960,0.128624] | 0.101515 [0.097873,0.110951] |
| Negative gate / 64 | 8496 [8141,8940] | 9294 [8819,9541] | 5.42468 [5.34398,5.58136] | 5.02178 [4.98421,5.24124] |
| Negative convolution / 1 | 8904 [8745,9030] | 9743 [8941,10124] | 0.109426 [0.103821,0.112115] | 0.099694 [0.097807,0.125282] |
| Negative convolution / 64 | 8505 [8348,8941] | 9353 [8568,9692] | 5.39322 [5.27238,5.56860] | 5.00476 [4.90373,5.14194] |

All twelve observed decode medians increased **4.36–12.63%** and first-output
medians decreased **1.10–9.60%**. Shared-host spreads still matter. The stress
cases correct different generated sequences, while the three ordinary controls
retain greedy agreement. These results support this tiny CPU workload; they do
not certify larger/trained models, accelerators or every serving workload.

| Contract | Logical weights, pre-fix = optimized | SSM state / history bytes, pre-fix = optimized | Artifact bytes, pre-fix → optimized | Compile seconds, diagnostic pre-fix → admitted optimized |
|---|---:|---:|---:|---:|
| Default | 10672 | 512 / 384 | 94140 → 102204 | 0.262625 → 6.67577 |
| Expanded/biased | 18112 | 1024 / 640 | 103806 → 111874 | 0.264225 → 5.24044 |
| Zero epsilon | 10672 | 512 / 384 | 94139 → 102204 | 0.262455 → 5.13885 |
| Large delta | 10672 | 512 / 384 | 94144 → 102204 | 0.264175 → 5.15229 |
| Negative gate | 11072 | 512 / 384 | 96742 → 104804 | 0.263130 → 5.14452 |
| Negative convolution | 10672 | 512 / 384 | 94146 → 102211 | 0.262103 → 5.11995 |

The admitted build includes numerical-evidence files and reference execution;
artifact growth and compile time are not a like-for-like performance loss
attributable to activation arithmetic. Logical weight/cache bytes are unchanged.

A separate physical Windows CPU
[allocation probe](measurements/mamba2-logsigmoid-traced-allocations.json)
measured warmed generation after weights and prompt allocation. Five warmups
and 19 samples produced the following peak traced Python/NumPy bytes:

| Default prompt | Pre-fix median [min,max] | Optimized median [min,max] |
|---|---:|---:|
| One ID | 19120 [19120,19300] | 14128 [14128,16615] |
| 64 IDs | 52496 [52496,55828] | 52040 [52040,55301] |

All sampled generations retained identical eight-ID greedy outputs. An explicit
4,194,304-byte NumPy allocation produced 4,194,400 traced bytes, verifying that
the installed NumPy path exposes array-data allocations to this tracer. The
source weights/tokenizer bytes match the hosted default fixture; source config
values and LF-normalized bytes match, while the local raw config uses Windows
CRLF. This is diagnostic local execution, not a new local numerical admission.
These traced peaks combine transient arrays, caches, outputs and Python objects;
they exclude loaded weights, prompt and untraced native allocations. Process
RSS, BLAS-native memory, independent activation/overhead attribution and soak
behavior remain unmeasured here.

## Unchanged passes and hardware scope

This is a runtime numerical correction. It establishes no new optimizer-pass
benefit or sparse/quantized/accelerator kernel claim.

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
| 17 TEE | Off, unsupported execution | No attested backend kernels |
| 18 MDLM | Off, supplied trained heads | Real-draft benefit unproved |
| 19 sub-2-bit | Off, experimental lossy | Dense dequantization; optimized kernel/QAT unverified |
| 20 video | Off, unsupported CPU execution | No verified VLM benefit |
| 21 adapter | Off, supplied tensors | Scaling/parity unresolved |
| 22 RLVR | Off, experimental training | Gradient backend required |

| Backend scope | Executed technique in this area |
|---|---|
| NumPy CPU | Physical Windows and hosted Linux CPU; independent FP64 scalar/matrix checks, original-HF admission, compiled prefill/cached continuation, public API/CLI |
| Torch CPU | Actual CPU tensors and original-HF/matrix controls on complete source checkpoints; hosted admission and public API/CLI |
| ONNX | Existing general controls; no Mamba-2 ONNX claim |
| CUDA, ROCm, MPS, DirectML and other targets | No new physical-device execution; existing matrix tiers unchanged |

Source-defined nondefault time-step limits and convolution activations are
covered by a subsequent [source-parameter correction](mission-mamba2-source-parameters.md).
Trained Mamba-2, other model families, larger-model
speed, quantized weight execution, bounded serving memory, pass composition,
hardware validation, archive hardening and blocking core/runtime type checks
remain open. The [hardware matrix](../hardware_validation_matrix.json),
generated README table and [pass audit](mission-pass-audit.md) retain their scopes.
