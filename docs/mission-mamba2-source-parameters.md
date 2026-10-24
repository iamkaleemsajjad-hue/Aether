# Mamba-2 source execution parameters

The compiler now preserves source-defined convolution activation and time-step
bounds through AEG packaging and both actual Mamba-2 engines. This follows the
[engine-owned execution decision](mission-architecture-note.md). It changes
executed numerical behavior; no optimizer-pass benefit is inferred.

## Source contract and reproduction

Reviewed the Mamba-2 engines, canonical loader, ingestion paths and callers,
architecture serialization, manifest integrity, cache/session/worker contracts,
existing convolution/normalization/stability tests and the strict HF tool.
The [Mamba-2 paper](https://arxiv.org/html/2405.21060v1), equations 2/3 and
section 7.1, supplies the recurrence and gate order. The exact source execution
contract is [Transformers 4.57.3 Mamba2Mixer.torch_forward](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mamba2/modeling_mamba2.py).
It applies `ACT2FN[hidden_act]` to causal-convolution output and clamps
`softplus(dt+bias)` before **both** decay and input injection. The output gate
remains SiLU regardless of convolution activation.

Also reviewed the official
[`Mamba2` implementation at v2.2.4](https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/modules/mamba2.py),
alongside the previously reviewed simple block and SSD minimal implementation.
That official version fixes convolution activation to SiLU and passes
`dt_limit` to its full scan, but its token `step` omits the limit. Its token
step therefore is not the oracle for HF's declared bounded-time-step contract.
[PyTorch clamp semantics](https://docs.pytorch.org/docs/2.9/generated/torch.clamp.html)
define open endpoints and inverted bounds: `min > max` yields `max`.
Actual HF forwards and the independent FP64 full-matrix oracle cover these
cases; the implementation does not invent an ordering rejection.

The final 33-case regression against a verified archive of
`19dad4ef58137ea6f2db0f277bfd4ee261b05ca3` returned **18 failed, 15 passed**
in 18.59 seconds. All 11 original-HF/matrix controls passed; default/unbounded
CPU/Torch cases passed. Nine nondefault cases failed in each actual AEG engine.
The corrected implementation passes all **33** cases. An expanded run adds
malformed-source refusal, publication preservation, serialization, authenticated
parameter tampering and a real serving-worker clone: **70 passed** in 22.75
seconds. Test durations are not inference benchmarks.

Earlier diagnostic reproduction on seven teacher-forced positions showed why
an explicit source-contract regression matters: ReLU convolution produced only
**2/7 matching argmax rows**, worst KL about **0.0451** and minimum cosine about
**0.3843** in both engines. Smaller bound omissions could pass the coarse quality
thresholds while still computing the wrong declared function.

## Executed correction and compatibility

Nondefault settings are encoded in the existing authenticated architecture
header as `ssm_runtime_parameters`. Immutable validated parameters are consumed
by the canonical loader, NumPy engine and Torch engine. Convolution activation
is selected once. Time-step bounds clamp in place after softplus; no new
weight arrays or cache layouts are introduced. SiLU, its `swish` alias and ReLU
are supported; unsupported activations and malformed/nonfinite stored bounds
fail clearly. Source infinity endpoints normalize to JSON `null`.

Validation runs after ingestion loader/fallback paths, including a materialized
snapshot. The existing architecture refresh updates the caller's object in
place; the apparent stale-reference concern on that path was rejected after
tracing `_refresh_architecture_from_snapshot`. Unsupported parameters fail
before publication, including diagnostic compilation and requested overwrite.

The header field is omitted for default SiLU/unbounded execution. Six isolated
old/new compiler/reader jobs used the actual retained old source, rather than
an emulated old parser. All four default compiler/reader combinations generated
byte-identical saved logits. Both diagnostic default artifacts were **133,213
bytes** in this local run. The custom ReLU/bounded artifact was **133,486
bytes**; this is a different source contract, not a same-model storage comparison.
The old reader refused the new custom artifact through manifest verification.
New readers load old default artifacts; readers predating the required settings
must be upgraded for nondefault artifacts. No opaque inspection payload can be
silently skipped to authorize different arithmetic.

The numerical admission report binds the architecture header and payloads.
The benchmark separately records implementation hashes, including the new
parameter module and architecture serializer; it is audit provenance, not a
code-version restriction on portable artifact execution.

## Validation and measurement scope

The complete local run returned **5,543 passed, 42 skipped, two failures** in
602.19 seconds. Both failures are the previously recorded MCP HTTP handshake
cases; all new parameter cases passed. Line coverage was **45,244/57,440 =
78.7674095%**, versus the prior local **45,190/57,356 = 78.7886185%**. Covered
lines increased, but the percentage decreased by **0.021209 percentage points**;
the local coverage non-reduction criterion remains open. A separate final
control run passed **151** cases, including normalization, stable activation,
convolution and SSD integration. It does not replace the complete red run or
its coverage. [Raw failures, counts and scoped diagnostics](measurements/mamba2-parameters-local-validation.json)
retain the evidence; [actual compatibility jobs](measurements/mamba2-parameters-compatibility.json)
retain reader identities and outputs.

An interleaved local default-path comparison used 45 samples, five eight-ID
generations per sample and seven warmups; allocations used 19 separate samples.
The actual retained old engine and current engine shared the same immutable
default weights and generated identical IDs and prefill logits. These are
whole-engine generation times, including prefill, rather than decode-only
throughput or public API TTFT. Local thread/affinity settings were not pinned.

| Prompt | Old generation ms median [min,max] | Current generation ms median [min,max] | Old/current traced peak bytes median [min,max] |
|---|---:|---:|---:|
| One ID | 0.64132 [0.61046,1.11644] | 0.64242 [0.60872,0.83706] | 8544 [8544,8544] / 8544 [8544,8544] |
| 64 IDs | 4.66040 [4.27222,5.33980] | 4.75960 [4.26578,6.27412] | 37712 [37712,37712] / 37712 [37712,37712] |

Observed medians were **0.17% and 2.13% longer** with overlapping spreads;
no speed improvement or proof of non-regression is assigned to this local
sample. Traced peaks were unchanged. The NumPy tracing control observed at
least the 4,194,304-byte array allocation. Tracing excludes loaded weights,
prompt and untraced native allocations; peak RSS and separate activation/runtime
overhead remain unmeasured. The [complete local series](measurements/mamba2-parameters-default-observations.json)
preserves losses and limits. Hosted strict measurements are pending; no new
peak-RSS, corpus-quality or accelerator improvement is claimed here.

New modules/tests and the HF tool pass Ruff; scoped new/contract files pass
mypy. Existing touched-file Ruff diagnostics decreased **250 to 249**, existing
scoped mypy diagnostics remain **110 to 110**, and serializer/tool Ruff
diagnostics remain **13 to 13**, with no added file/code/message diagnostics.
The full repository style/type backlog remains open.

Repeat the functional checks with:

```powershell
python -m pytest tests/unit/test_mamba2_source_parameters.py tests/unit/test_mamba2_parameter_contract.py -q --no-cov
```

The committed `tools/validate_mamba2_hf.py --output NEW_DIRECTORY --baseline-src
PATH_TO_19DAD4E_SRC` runs 14 complete, untrained Mamba-2 source contracts against
original HF, CPU/Torch CPU numerical admission, public generation/streaming and
CLI. It preserves the compiler's RAM limits. `--family mamba` runs nine separate
Mamba-1 controls. Historical compilation is explicitly diagnostic; current
compilation requires the real numerical gate. Hardware/dependencies, source and
implementation hashes, two warmups, seven timing samples, logical weight/state
bytes, artifact size and compile time are retained by the tool. Engine first
output is not public API TTFT. Trained models and large-model comparisons remain
required mission work.

## Pass and hardware disposition

No pass is promoted by this runtime correction.

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

| Backend scope | Verification in this area |
|---|---|
| NumPy CPU | Physical Windows CPU, actual source compile/load/forward/cached continuation, FP64 matrix oracle and original-HF controls |
| Torch CPU | Actual CPU tensors, original-HF/matrix comparisons and cached continuation |
| ONNX | No Mamba-2 ONNX execution claim |
| CUDA, ROCm, MPS, DirectML and other targets | No new device execution; existing matrix tiers unchanged |

Trained Mamba-2 and other model families, larger-model speed, packed quantized
execution, bounded serving memory, full pass composition, hardware validation,
archive hardening and blocking core/runtime types remain open. The
[hardware matrix](../hardware_validation_matrix.json), generated README table
and [pass audit](mission-pass-audit.md) retain their verification scopes.
