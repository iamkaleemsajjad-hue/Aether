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
semiseparable matrix and full causal convolution. Its log-sigmoid formulation
is independent of the runtime's new piecewise sigmoid evaluation.

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

The correction uses `logaddexp(0, x)` for softplus. For SiLU, set
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
| NumPy CPU | Physical Windows CPU; independent FP64 scalar/matrix checks and actual compiled prefill/cached continuation |
| Torch CPU | Actual CPU tensors and original-HF/matrix controls on complete source checkpoints |
| ONNX | Existing general controls; no Mamba-2 ONNX claim |
| CUDA, ROCm, MPS, DirectML and other targets | No new physical-device execution; existing matrix tiers unchanged |

Source-defined nondefault time-step limits and convolution activations remain
separate correctness work. Trained Mamba-2, other model families, larger-model
speed, quantized weight execution, bounded serving memory, pass composition,
hardware validation, archive hardening and blocking core/runtime type checks
remain open. The [hardware matrix](../hardware_validation_matrix.json),
generated README table and [pass audit](mission-pass-audit.md) retain their scopes.
