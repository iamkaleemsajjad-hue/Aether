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
preserves losses and limits. The separate strict local tool passed its default
contract through original-HF admission, CPU/Torch CPU, API/streaming and CLI.
Its expanded/biased contract then refused the Torch worker for live RAM:
738,463,744 bytes available versus a 536,953,696-byte worker estimate plus
268,435,456-byte headroom. The [partial report](measurements/mamba2-parameters-local-hf-attempt.json)
retains both cases. No memory guard was relaxed. No new peak-RSS,
corpus-quality or accelerator improvement is claimed here.

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

## Hosted original-HF measurements

[CI at `5c14973`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/40076617018)
passed all **23 jobs**; [security](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/40076617020)
passed separately. The complete pinned CPU suite returned **5,536 passed,
51 skipped, zero failures**. Line coverage was **45,271/57,440 = 78.8144150%**,
above the prior hosted **45,192/57,356 = 78.7921054%** and the blocking 78.6%
floor. The complete local failures and lower local percentage remain recorded.

All 14 Mamba-2 source contracts and nine Mamba-1 controls passed ordinary
original-HF admission, CPU/Torch CPU, public generation/streaming and CLI checks.
Verified **496 implementation hashes**, **161 source/reference files** and
reopened/reverified **43 downloaded artifact packages**. The other three
historical Mamba-1 candidates failed and remain recorded; no current candidate
failed. The dependency freeze identifies GitHub's temporary PR merge checkout
`bba2823`; its entire Git tree equals the pushed `5c14973` tree, independently
checked through the GitHub API and local Git objects.

The [Mamba-2 raw report](measurements/mamba2-parameters-hf.json),
[Mamba-1 controls](measurements/mamba1-control-during-mamba2-parameters.json),
[verification record](measurements/mamba2-parameters-provenance.json) and
[dependency freeze](measurements/mamba2-parameters-freeze.txt) retain source
identities, all repeats, losses and limits. The physical host was an Intel Xeon
Platinum 8370C, four logical CPUs, 16,765,378,560 bytes RAM, Linux
6.17.0-1022-azure x86_64, Python 3.11.14, NumPy 1.26.4, Torch 2.9.1+cpu,
Transformers 4.57.3, SafeTensors 0.8.0, Tokenizers 0.22.2 and psutil 7.2.2.
Timing used one numerical thread, two warmups and seven repeats per probe.

Every current Mamba-2 contract matched **16/16 greedy IDs per executor**.
Across CPU and Torch CPU, worst KL was **2.07007e-14**, minimum cosine
**0.9999999999997122**, maximum absolute logit error **4.17233e-7**, and worst
relative fixed-label PPL increase **8.76456e-8** (**0.00000876456%**).
The two ReLU source contracts recovered old CPU agreement from **10/16 to
16/16** each; their old PPL increases of **7.79%/7.81%** fell to numerical
roundoff. Bounded-only contracts already passed coarse gates but now preserve
their declared function to FP32/matrix-reference tolerance. Six unaffected
control contracts retained byte-identical saved fixed-label logits.

Default decode medians changed **-1.15% / +0.37%** for one/64-ID prompts;
first-output medians changed **-0.68% / +0.52%**. The six bounded-SiLU
contracts decoded **2.30–6.20% slower**, with first output **0.78–7.74% longer**.
These losses include previously omitted required clamp work. ReLU without
bounds measured faster but executes a different source activation, so no
general optimization benefit is assigned. Timing non-regression remains open;
profiling the correctly bounded path is follow-up work. All spreads follow.

Quality uses two fixed probes; baseline agreement is CPU-only (16 IDs), while
the current total covers CPU plus Torch CPU (32 IDs). PPL is fixed-label
synthetic loss, not trained-model corpus quality. Timing is engine decode and
first output, not public API TTFT. Logical weights/state/history remain
unchanged; peak RSS and separate activation/runtime-overhead attribution are
unmeasured. Artifact/compile columns compare diagnostic historical compilation
with current numerical admission and therefore perform different work.

| Source contract | Old CPU KL / min cosine | Old CPU greedy IDs | Current CPU + Torch KL / min cosine | Current greedy IDs |
|---|---:|---:|---:|---:|
| default | 5.2046e-15 / 0.999999999999946 | 16/16 | 9.18477e-15 / 0.999999999999911 | 32/32 |
| expanded-biased | 9.89457e-15 / 0.999999999999897 | 16/16 | 9.89457e-15 / 0.999999999999897 | 32/32 |
| zero-epsilon | 2.07007e-14 / 0.999999999999712 | 16/16 | 2.07007e-14 / 0.999999999999712 | 32/32 |
| large-delta | 1.13713e-14 / 0.999999999999935 | 16/16 | 1.13713e-14 / 0.999999999999905 | 32/32 |
| negative-gate | 1.30465e-15 / 0.999999999999994 | 16/16 | 1.30465e-15 / 0.999999999999994 | 32/32 |
| negative-conv | 3.97123e-15 / 0.999999999999988 | 16/16 | 3.97123e-15 / 0.999999999999986 | 32/32 |
| upper-limit | 1.69879e-07 / 0.999997691461183 | 16/16 | 5.8319e-15 / 0.999999999999932 | 32/32 |
| lower-limit | 4.11152e-06 / 0.999936905251601 | 16/16 | 1.01358e-14 / 0.99999999999991 | 32/32 |
| zero-limit | 2.40942e-07 / 0.999996720816202 | 16/16 | 8.24053e-15 / 0.999999999999875 | 32/32 |
| open-upper-limit | 4.11152e-06 / 0.999936905251601 | 16/16 | 1.01358e-14 / 0.99999999999991 | 32/32 |
| open-lower-limit | 1.69879e-07 / 0.999997691461183 | 16/16 | 5.8319e-15 / 0.999999999999932 | 32/32 |
| inverted-limits | 1.92961e-08 / 0.999999686870055 | 16/16 | 7.76608e-15 / 0.9999999999999 | 32/32 |
| relu-convolution | 0.0452188 / 0.383751683350911 | 10/16 | 6.13044e-15 / 0.999999999999925 | 32/32 |
| relu-bounded | 0.045148 / 0.384204171397972 | 10/16 | 1.25203e-14 / 0.999999999999879 | 32/32 |

| Contract / prompt IDs | Old decode IDs/s median [min,max] | Current decode IDs/s median [min,max] | Old first-output ms median [min,max] | Current first-output ms median [min,max] |
|---|---:|---:|---:|---:|
| default / 1 | 7376.33 [7278.72,7486.78] | 7291.23 [6628.8,7337.88] | 0.129747 [0.125531,0.138841] | 0.128866 [0.124884,0.158471] |
| default / 64 | 7312.32 [7230.57,7364.51] | 7339.52 [6818.42,7427.62] | 6.29509 [6.27281,6.34638] | 6.32778 [6.27355,6.64767] |
| expanded-biased / 1 | 7002.04 [6847.38,7091.97] | 6974.42 [6852.61,7112.06] | 0.135443 [0.131997,0.145251] | 0.13325 [0.130976,0.162215] |
| expanded-biased / 64 | 6988.98 [6899.81,6996.68] | 7003.76 [6580.95,7070.32] | 6.70251 [6.68538,6.83067] | 6.69326 [6.64657,6.7998] |
| zero-epsilon / 1 | 7221.5 [7192.77,7280.96] | 7387.54 [7163.6,7501.51] | 0.130288 [0.127042,0.168305] | 0.128507 [0.12581,0.180546] |
| zero-epsilon / 64 | 7277.7 [7214.02,7370.38] | 7344.02 [7302.66,7379.68] | 6.32369 [6.31095,6.3764] | 6.33162 [6.29137,6.42859] |
| large-delta / 1 | 7188.05 [6819.26,7332.25] | 7273.86 [6974.24,7403] | 0.130854 [0.127015,0.157577] | 0.129617 [0.125698,0.144876] |
| large-delta / 64 | 7268.45 [7218.87,7296.34] | 7277.57 [7123.7,7304.88] | 6.33733 [6.31843,6.41621] | 6.39354 [6.36142,6.45166] |
| negative-gate / 1 | 7184.39 [7032.85,7233.68] | 7059.28 [6837.29,7295.47] | 0.132248 [0.129023,0.148495] | 0.131941 [0.127095,0.145634] |
| negative-gate / 64 | 7101.86 [7022.05,7158.85] | 7202 [6865.69,7281.38] | 6.47422 [6.43773,6.53725] | 6.47878 [6.44186,6.53293] |
| negative-conv / 1 | 7324.19 [7131.18,7423.64] | 7309.67 [7232.94,7454.42] | 0.129326 [0.125332,0.144427] | 0.129267 [0.124607,0.152214] |
| negative-conv / 64 | 7278.8 [6975.74,7339.83] | 7305.77 [7250.26,7404.7] | 6.33786 [6.29254,6.40342] | 6.37347 [6.33013,6.48367] |
| upper-limit / 1 | 7383.87 [7244.02,7427] | 7125.15 [6891.3,7208.7] | 0.128497 [0.124713,0.129687] | 0.133889 [0.131053,0.143763] |
| upper-limit / 64 | 7340.37 [7209.55,7400.16] | 7024.43 [6853.86,7089.6] | 6.29718 [6.26169,6.91179] | 6.67511 [6.61435,6.79101] |
| lower-limit / 1 | 7364.44 [7332.25,7403.08] | 7022.03 [6837.23,7108.31] | 0.127813 [0.125795,0.149339] | 0.134897 [0.132518,0.151536] |
| lower-limit / 64 | 7278.85 [6795.73,7368.16] | 7034.49 [6957.6,7062.84] | 6.33262 [6.30818,6.37813] | 6.65783 [6.6454,6.7575] |
| zero-limit / 1 | 7227.41 [7189.91,7264.57] | 6984.6 [6878.07,7097] | 0.129844 [0.125904,0.15416] | 0.1331 [0.13071,0.141915] |
| zero-limit / 64 | 7301.33 [7225.35,7347.45] | 7010.19 [6923.4,7063.27] | 6.31413 [6.3026,6.33325] | 6.72794 [6.70988,6.81717] |
| open-upper-limit / 1 | 7381.31 [7313.62,7454.66] | 7008.54 [6805.88,7082.93] | 0.126426 [0.125227,0.129266] | 0.135565 [0.131903,0.171678] |
| open-upper-limit / 64 | 7367 [7301.07,7385.66] | 6910.58 [6831.54,6976.45] | 6.318 [6.30187,6.34282] | 6.80676 [6.71745,6.86145] |
| open-lower-limit / 1 | 7446.86 [7329.51,7483.41] | 7024.61 [6953.46,7153.97] | 0.129808 [0.125207,0.136153] | 0.133692 [0.129624,0.16888] |
| open-lower-limit / 64 | 7346.42 [7258.89,7351.74] | 7052.43 [6957.91,7092.87] | 6.3196 [6.28959,6.33797] | 6.66689 [6.60933,6.73161] |
| inverted-limits / 1 | 7210.03 [7183.94,7460.36] | 7044.13 [6865.65,7122.7] | 0.131816 [0.126975,0.150705] | 0.132849 [0.131493,0.153985] |
| inverted-limits / 64 | 7298.56 [7042.03,7349.51] | 6995.12 [6954.28,7057.27] | 6.30054 [6.25925,6.37121] | 6.6647 [6.63051,6.8515] |
| relu-convolution / 1 | 7386.97 [7252.15,7430.06] | 7578.77 [7322.85,7647.87] | 0.127143 [0.126031,0.12968] | 0.125709 [0.123326,0.16156] |
| relu-convolution / 64 | 7248.32 [7013.09,7346.18] | 7529.28 [7124.04,7576] | 6.32048 [6.29956,6.3547] | 6.17647 [6.1452,6.21852] |
| relu-bounded / 1 | 7259.6 [7197.95,7319.22] | 7147 [6909.71,7229.87] | 0.129678 [0.124619,0.153008] | 0.131512 [0.128648,0.168478] |
| relu-bounded / 64 | 7316.57 [7202.77,7400.86] | 7125.18 [6992.32,7207.51] | 6.2935 [6.27059,6.4218] | 6.52431 [6.50949,6.64102] |

| Contract | Logical weight bytes, old = current | State / history bytes, old = current | Artifact bytes, diagnostic old → admitted current | Compile seconds, diagnostic old → admitted current |
|---|---:|---:|---:|---:|
| default | 10672 | 512 / 384 | 94140 → 102204 | 0.28577 → 8.21468 |
| expanded-biased | 18112 | 1024 / 640 | 103806 → 111868 | 0.279678 → 5.5348 |
| zero-epsilon | 10672 | 512 / 384 | 94093 → 102208 | 0.279108 → 5.63181 |
| large-delta | 10672 | 512 / 384 | 94144 → 102204 | 0.281545 → 5.34897 |
| negative-gate | 11072 | 512 / 384 | 96742 → 104814 | 0.280311 → 5.44133 |
| negative-conv | 10672 | 512 / 384 | 94146 → 102204 | 0.278379 → 5.43043 |
| upper-limit | 10672 | 512 / 384 | 94144 → 102443 | 0.281139 → 5.42585 |
| lower-limit | 10672 | 512 / 384 | 94144 → 102472 | 0.279138 → 5.37716 |
| zero-limit | 10672 | 512 / 384 | 94143 → 102467 | 0.27989 → 5.4287 |
| open-upper-limit | 10672 | 512 / 384 | 94149 → 102433 | 0.281334 → 5.39994 |
| open-lower-limit | 10672 | 512 / 384 | 94149 → 102496 | 0.28113 → 5.41591 |
| inverted-limits | 10672 | 512 / 384 | 94148 → 102477 | 0.275055 → 5.47878 |
| relu-convolution | 10672 | 512 / 384 | 94149 → 102420 | 0.27971 → 5.48522 |
| relu-bounded | 10672 | 512 / 384 | 94099 → 102469 | 0.280962 → 5.4238 |

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
| NumPy CPU | Physical Windows and hosted Linux CPU; source compile/load/cached continuation, FP64/HF controls, hosted ordinary admission and API/CLI |
| Torch CPU | Actual Windows/Linux CPU tensors, HF/matrix comparisons, cached continuation, hosted admission and API/CLI |
| ONNX | No Mamba-2 ONNX execution claim |
| CUDA, ROCm, MPS, DirectML and other targets | No new device execution; existing matrix tiers unchanged |

Trained Mamba-2 and other model families, larger-model speed, packed quantized
execution, bounded serving memory, full pass composition, hardware validation,
archive hardening and blocking core/runtime types remain open. The
[hardware matrix](../hardware_validation_matrix.json), generated README table
and [pass audit](mission-pass-audit.md) retain their verification scopes.
