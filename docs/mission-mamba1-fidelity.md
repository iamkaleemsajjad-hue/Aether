# Mamba-1 checkpoint fidelity

At `ad77e08` on 2026-10-22, all nine untrained FP32 Mamba-1 source contracts
passed original-HF admission, CPU/Torch CPU, public generation/streaming and
strict CLI checks. [All 23 CI jobs passed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39747608124),
including the full dependency-enabled suite: **5421 passed, 51 skipped, zero
failures**. Security passed separately. Trained-family grades remain unchanged.
The failed local runs and two intermediate hosted integration failures below
are retained alongside the final results.

This correction follows the [engine-owned architecture decision](mission-architecture-note.md).
The actual CPU and Torch executors consume the corrected weights and configuration;
an IR rewrite alone cannot correct their output. The wider runtime mission remains
in progress.

## Reproduction and source math

Read the complete relevant ingestion, architecture, loader, Mamba, shared Torch,
hybrid/state and admission modules, their callers and related tests. Research
covered Gu and Dao's [Mamba paper](https://arxiv.org/html/2312.00752v2), including
its appendices, the v2.2.4 reference
[`Mamba.step`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py)
and
[`selective_scan_ref`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/selective_scan_interface.py),
and the complete installed Transformers 4.57.3 Mamba configuration/model code.

The checkpoint recurrence is `h[t] = exp(delta[t]*A)*h[t-1] + delta[t]*B[t]*x[t]`,
then `y[t] = C[t]*h[t] + D*x[t]`, SiLU gating, output projection and residual.
This implements paper equation (2), with the literal `delta*B` input injection
used by both reference implementations. Substituting equation (4)'s full ZOH
input discretization would change the checkpoint's function. The existing state
transpose, current-state readout, delta bias and D-skip order were correct.

The new independent FP64 oracle sums every causal source contribution using the
full sequence matrix and `numpy.convolve`. It does not call the runtime scan.
Complete deterministic, untrained FP32 checkpoints contain nonzero B/C, learned
norm scales, widths 1/3, expansion 2/4, optional projection biases and distinct
multilayer blocks. These test numerical fidelity, not corpus or task quality.

## Confirmed defects and root changes

* HF Mamba uses RMSNorm even with `layer_norm_epsilon` and an unused `rms_norm`
  flag. Architecture inference selected LayerNorm. Explicit `expand` now also
  determines Mamba-1's inner width when the raw config omits `intermediate_size`.
* Stored input/output projection biases now reach standalone and hybrid CPU/Torch
  execution. New optional dataclass fields are appended to preserve existing
  positional construction; actual hybrid artifact and oracle tests cover callers.
* Pure Mamba-1 loading preserves an explicit zero epsilon and rejects malformed
  learned norm shapes instead of padding or truncating trained values.
* Stable softplus keeps a finite delta bias of 1000 finite. CPU SiLU evaluates
  `x*sigmoid(x)` without clipping finite inputs; large negative inputs formerly
  created false nonzero values that subsequent finite weights could amplify.
* Both Mamba-1 engines expose the actual all-row/final-vector `forward_step`
  contract. Original-HF admission can dispatch to the real Torch Mamba executor.
  Recurrent references retain a separate 4096-position workload bound without
  inventing a positional-table limit; RAM/headroom guards are unchanged.
* Ingestion refuses unsupported Mamba-1 mixer activations at its common exit,
  after specialized/generic fallback and snapshot materialization. `silu`,
  `swish` and the absent-field default remain supported. SDK/CLI regressions
  check refusal before publication and preservation of an existing destination.

A possible cache-aliasing concern was rejected for the actual speculative path:
its `clone_cache` already clones the known Torch state/cache tensors. Generic
`deepcopy` behavior alone did not prove a consumer defect, so no cache-class
rewrite was made.

| Regression group | Before correction | After correction |
|---|---:|---:|
| Initial checkpoint/engine cases | 21 failed, 1 passed | 22 passed |
| Added finite-value/shape/activation cases | 8 failed, 2 passed | All 32 cumulative cases passed |
| Fallback/snapshot/CLI refusal cases against historical source | 4 failed | All 42 final Mamba-1 cases passed |
| Shared Mamba-2, hybrid, convolution, speculation and reasoning controls | Not a new baseline | 204 passed, including the earlier 32 Mamba-1 cases |

The matrix assertions use `rtol=3e-5`, `atol=3e-6`, cover full forward and owned
cached continuation, and preserve bounded convolution history. These tolerances
are test thresholds, not a reported measured KL or original-HF comparison.
Local raw records are `build/mission/mamba1-{fidelity,edges,refusal}-*`.

## Validation and measurement limits

The local original-HF attempt refused **before execution**: available RAM was
494,948,352 bytes, estimated worker reserve 536,925,852 bytes, and required
additional headroom 268,435,456 bytes. It supplies no original-HF result.
The [strict HF tool](../tools/validate_mamba2_hf.py) now accepts `--family mamba`
while retaining Mamba-2 as its backward-compatible default. It checks ordinary
compiler admission, reload, both public CPU backends, streaming and the CLI.
The blocking pinned hosted job exercises nine Mamba-1 contracts alongside the
existing Mamba-2 checks and complete CPU suite. Historical source is fixed at
`2f2a2c4659063b31797011ac03373a1c4e539173`; failed historical executions remain
failures, and nonfinite logits cannot become benchmark samples.

The final hosted original-HF and engine timing measurements appear below.
Inference peak RSS and trained-model quality remain unmeasured. No general
speed, resident-memory, no-regression or trained-family certification is claimed.

The complete local run returned **5386 passed, 42 skipped, 3 failed** in 972.92
seconds. It collected 38 of the new Mamba-1 cases; the last four refusal tests
were added after collection and passed in the separate final **47-case**
Mamba-1/hybrid integration run. The full run's failures were the previously
observed Unicode audit worker RAM admission (464,261,120 available versus
406,938,624 worker reserve plus 134,217,728 headroom bytes) and two MCP HTTP
handshake cases. No guard or assertion was relaxed. Raw line coverage fell from
45028/57244 (78.6597722%) to 45016/57277 (78.5935018%); branch coverage changed
from 10754/17228 to 10685/17242. This is not a green local suite or a satisfied
coverage/no-regression criterion. Raw records are `build/mission/mamba1-full*`.

Scoped static checks retain 516 Ruff and 184 mypy findings across the eight
changed production modules and three shared tests, with no added file/code or
file/message diagnostic. New fixtures/tests and the HF tool pass full Ruff and
mypy. This does not clear the repository-wide advisory backlog.

```powershell
python -m pytest tests/unit/test_mamba1_fidelity.py -o addopts='' -q
python tools/validate_mamba2_hf.py --family mamba --output hf-mamba1-results
# Optional: --baseline-src /path/to/2f2a2c4/src
```

## Passes, hardware and remaining scope

No optimizer pass changes in this area. The complete
[22-row disposition table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [measured per-pass audit](mission-pass-audit.md) retain their results and
unresolved benefit/composition/idempotency checks. Only P3 is default.

The matrix-oracle and actual Torch CPU checks executed on the physical Windows
x86 CPU. Hybrid bias tests establish their specific path, not whole-model HF
Jamba parity. CUDA/ROCm/MPS/ONNX and other targets acquire no validation tier;
the [generated hardware table](../README.md#supported-hardware-targets) and
[evidence matrix](../hardware_validation_matrix.json) retain their recorded
techniques, dates, versions and outstanding physical-hardware requirements.

Trained Mamba checkpoints, custom normalization overrides, non-SiLU execution,
long-context/large-model quality, accelerator parity, quantized execution,
controlled performance, bounded serving memory and the remaining family/pass
corrections remain unverified. Planning/dry-run does not execute ingestion or
certify the unsupported-activation check. Earlier trained-family divergence
records and family grades remain unchanged.

## Hosted integration finding: generation settings

The first hosted run at `48ce3b0` executed the original-HF default fixture on
an AMD EPYC 7763 CPU (four logical CPUs, Linux x86_64, Python 3.11.14,
NumPy 1.26.4, Torch 2.9.1+cpu, Transformers 4.57.3). Its independent default
comparison passed: 16/16 greedy IDs, worst-position KL 1.15727e-15, minimum
cosine 0.9999999999999917. Ordinary CPU/Torch numerical admission passed.
However, the [blocking job failed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39725488801)
at the public API check: both backends emitted eight tokens rather than stopping
at the sixth generated token, the model config's EOS ID 16. The remaining eight
Mamba-1 fixtures and complete HF-environment suite did not run. All other 22 CI
jobs and the separate security run passed. This is a failed integration result,
not a complete HF or whole-suite pass.

The root was shared packaging: it copied tokenizer files but dropped
`generation_config.json`, and never transferred a model-config-only stop ID.
The existing packaged tokenizer and both engines already consume stop IDs.
The correction packages the source generation file byte-for-byte, or emits its
declared top-level model EOS when that file is absent. This common boundary
covers both the framework-free fast-tokenizer copy and actual AutoTokenizer
materialization. Malformed JSON/non-object generation files refuse publication;
existing destination bytes survive. Source files and the graph/manifest
integrity chain are retained. Recompile affected artifacts to recover settings
that earlier packaging discarded.

Read the complete compiler and native backend, stop helpers, tokenizer/rotary
wiring tests and the public generation paths. The precedence follows the pinned
Transformers `PreTrainedModel.from_pretrained` generation-file selection and
`GenerationConfig.from_model_config`; it requires no new decoding algorithm.
The new ten-case API/stream/payload regression failed 10/10 before correction.
Two additional real slow-tokenizer and two malformed-file regressions failed
4/4 against the actual historical compiler. After correction, all **82 focused
cases passed**, covering those 14 cases, Mamba-1/Mamba-2 and existing rotary/stop
wiring. Compiler Ruff remains 26 findings and mypy 11, with no added diagnostics;
the new test and HF tool pass Ruff/mypy. Subsequent complete local and hosted
results appear below. Raw local records use
`build/mission/generation-settings-*`.

This corrects preserved source settings. It does not establish equivalence for
every tokenizer-specific stop convention, nested model generation config,
sampling default, or legacy artifact. Existing runtime merging of canonical
tokenizer delimiters requires separate conflicting-settings tests. No weights or
logit computation change in this packaging follow-up; it claims a correct stop
boundary for the tested sources, not a throughput or resident-memory gain.

The complete local packaging follow-up subsequently returned **5405 passed,
42 skipped, 2 failed** in 1631.96 seconds. The RAM-admission case passed without
a guard change; both MCP HTTP handshake failures remained. Coverage was
45107/57295 (**78.7276377%**) lines and 10790/17252 branches. This single run
includes all 42 Mamba-1 cases and all 14 packaging regressions; it is still not
a green local suite. The two HTTP failures also reproduced outside the sandbox.
Plain standard-library loopback HTTP succeeded independently, so a generic
network outage does not explain them. No MCP transport code was changed.

## Hosted integration finding: special-token reconstruction

At `1a21639`, the [next hosted run](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39733176852)
passed original-HF, CPU, Torch CPU, public API/streaming and strict CLI checks
for default, expanded/biased, zero-epsilon and large-delta Mamba-1 fixtures.
The fifth, negative-gate fixture passed numerical admission but exposed a second
public-text defect: both backends rendered eight literal `<unk>` markers while
the original HF tokenizer omitted these declared special tokens. Later fixtures
and the full HF-environment suite did not run after that failure.

`PackagedTokenizer` loaded raw `tokenizer.json` but did not restore AddedToken
declarations from the surrounding tokenizer config/special-token map. The
original [HF fast-tokenizer constructor](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/tokenization_utils_fast.py)
does restore them. The correction applies those declarations once during
framework-free loading, retains matching/whitespace/normalization flags,
checks declared token IDs and preserves already serialized flags. It is shared
by CPU, Torch and ONNX tokenizer loading; this is not accelerator execution
evidence. Registration uses batched calls. It changes actual token matching
and decoding rather than filtering generated strings after decoding.

The valid pre-change regression returned **20 failed**; an earlier test-setup
attempt used a nonexistent adapter convenience method and is excluded from that
count. Additional malformed-declaration/repeated-load controls returned two
failed and one passed against the historical loader. The corrected focused
selection passed **135 cases with 2 optional skips**, including all 23 new
cases, generation-settings regressions, prompt/stop wiring, framework-free and
ONNX backend controls. Actual HF tokenization is the oracle for token matching,
special-token removal and vocabulary/ID equality. Constructor tests forbid
Torch/Transformers imports during packaged loading. Repeated loading preserves
source bytes. Native-backend Ruff remains 61 findings and mypy 11, with no new
diagnostics; the new test and HF tool pass Ruff/mypy. Raw records are
`build/mission/tokenizer-special-*`. Subsequent complete local and hosted results
appear below; no separate tokenizer speed or resident-memory result is claimed.

## Final executed HF evidence and measurements

The final worker reported AMD EPYC 9V74, four logical CPUs, 16,760,344,576 host
RAM bytes, Linux x86_64 and Python 3.11.14. The pinned reference stack was NumPy
1.26.4, Torch 2.9.1+cpu, Transformers 4.57.3, SafeTensors 0.8.0, Tokenizers
0.22.2 and psutil 7.2.2. The [unaltered report](measurements/mamba1-fidelity-hf.json),
[hash/source verification](measurements/mamba1-fidelity-provenance.json) and
[dependency freeze](measurements/mamba1-fidelity-freeze.txt) retain complete raw
repeats, gate results and failed historical executions. Every recorded source
file and reference-array digest was verified; successful execution-code digests
and the fixture digest match the exact current/historical Git revisions.
Failed historical runs have no execution-code digest or invented numerical or
timing value. Their checkout/import-root guards and original errors are retained.

Each source has two fixed probes, four teacher-forced labels per probe and two
eight-ID greedy continuations. Both executors matched **144/144 original-HF IDs**
over nine contracts. The maximum position KL was **4.21244e-15**, minimum cosine
**0.9999999999999846**, and worst relative fixed-label PPL increase
**0.00000481011%**. These are small untrained fidelity probes, not corpus PPL or
nine trained-model evaluations; aliases/default variants reuse source weights.

| Source contract | Parameters | Worst KL, CPU/Torch | Greedy IDs per executor | Historical CPU execution |
|---|---:|---:|---:|---|
| Default | 2816 | 1.18433e-15 | 16/16 | Executed; failed quality |
| Expanded/biased | 5200 | 2.05005e-15 | 16/16 | Refused incompatible convolution geometry |
| Zero epsilon | 2816 | 4.21244e-15 | 16/16 | Executed; failed quality |
| Large delta | 2816 | 4.04379e-15 | 16/16 | Nonfinite logits; no benchmark accepted |
| Negative gate | 2896 | 1.15372e-15 | 16/16 | Executed; failed quality |
| Negative convolution | 2896 | 1.15372e-15 | 16/16 | Executed; failed quality |
| Swish alias | 2816 | 1.18433e-15 | 16/16 | Executed; failed quality |
| Absent activation field | 2816 | 1.18433e-15 | 16/16 | Executed; failed quality |
| Two distinct layers, expansion 3 | 7536 | 1.32894e-15 | 16/16 | Refused incompatible convolution geometry |

The same admitted artifact passed both public CPU backends and strict CLI for
every contract, including stop boundaries and special-token text removal. The
negative-gate fixture generates real UNK IDs that HF omits from displayed text;
usage still counts those generated IDs. Existing Mamba-2 default/expanded checks
also passed admission/API/CLI with verified source hashes; their
[control report](measurements/mamba2-control-during-mamba1.json) is retained.

| Default source, actual HF comparison | Before: `2f2a2c4` CPU diagnostic | After: CPU/Torch CPU admission |
|---|---:|---:|
| Greedy agreement | 8/16 (50%) | 16/16 per executor |
| Worst position KL | 0.0209825 | 1.18433e-15 |
| Minimum position cosine | 0.800110 | 0.9999999999999853 |
| Worst fixed-label PPL increase | 4.41205% | 0.000000665187% |
| Loaded logical weight bytes | 11264 | 11264 |
| Logical SSM/cache state bytes | 640 | 640 |
| Logical convolution history bytes | 256 | 256 |
| Complete artifact bytes | 94820 | 103017 |

State bytes include `MambaState.h` and `last_x`. These array geometries are not
peak RSS. Activation and runtime-overhead peaks remain unmeasured. Artifact
growth includes measured admission and generation settings; it is not a weight
increase. The dense loader's FP32 materialization remains unfinished memory work.

Timing measures the actual NumPy CPU engine, with BLAS/OpenMP threads set to one,
two warmups and seven repeats per probe, at one-/64-token prompt lengths and
eight generated IDs without EOS stopping. It excludes loading/tokenization and
is not complete API TTFT. Both implementations ran on this same hosted worker.

| Median [minimum, maximum] | Before | After |
|---|---:|---:|
| One-token prompt decode, IDs/s | 6785 [6563,7046] | 7314 [6813,7695] |
| 64-token prompt decode, IDs/s | 6837 [6554,6947] | 7362 [7047,7426] |
| One-token first output, ms | 0.14565 [0.13557,0.19945] | 0.13715 [0.13174,0.15210] |
| 64-token first output, ms | 7.0289 [7.0164,7.1653] | 6.4493 [6.3381,6.5014] |

The observed default decode medians increased about 7.8%; historical outputs were
incorrect and the host is shared. This does not establish a general speedup or
absence of regression. All other fixture timing repeats, including losses, remain
in the raw report. Default compilation took 6.1195 seconds with mandatory
original-HF/CPU/Torch admission versus 0.3284 seconds for historical diagnostic
compilation without that work. Those are different verification workloads.

The full hosted run returned **5421 passed, 51 skipped, zero failures** in
167.416 JUnit seconds. Coverage was **45162/57353 (78.7439192%)** lines and
10830/17290 branches, above the previous same-job 78.6772413% and its blocking
78.6% floor. Repository-wide style/type checks remain advisory. The 51 skips
include unavailable physical devices/Windows contracts, optional SDKs, offline
datasets/artifacts, network-gated and opt-in cases. Raw local/hosted case IDs
differ only in ten GGUF enum parameter display labels; the record retains both
labels and their codec values rather than claiming identical raw IDs.

The final local run returned **5427 passed, 42 skipped, 3 failed** in 1142.26
seconds: RAM admission and both previously observed MCP HTTP cases. Coverage
was 45152/57353 (**78.7264834%**) lines and 10822/17290 branches. This is
0.0011543 percentage points below the preceding local packaging run, above the
78.6597722% area baseline, and not a green local suite. The
[first local record](measurements/mamba1-local-validation.json),
[packaging follow-up](measurements/generation-settings-local-validation.json) and
[tokenizer follow-up](measurements/tokenizer-special-local-validation.json)
preserve each complete run separately. The
[first](measurements/mamba1-first-hf-integration.json) and
[second](measurements/mamba1-stop-hf-integration.json) failed hosted integration
reports are also retained without replacing them with the final green result.

The local MCP wire probe used the actual test handler: its emitted headers
contained `Content-Length`, while the client observed `Transfer-Encoding:
chunked` without matching chunk framing, for JSON and SSE on both tested paths.
The component rewriting the response is unidentified. These observations do not
justify accepting malformed HTTP in the product or weakening the tests. Raw
evidence is retained in the
[wire observation](measurements/mcp-local-framing-observation.json); no transport
fix is claimed.

| Backend scope | Verification tier and technique in this area |
|---|---|
| NumPy CPU | Native execution on Windows CPU and hosted Linux x86 CPU; full-matrix oracle, original-HF logits/greedy/NLL, actual artifacts and public API/CLI |
| Torch CPU | Real CPU tensors; original-HF admission and the same artifact/API/CLI contracts; no meta-device substitute |
| ONNX CPU | Existing generic exporter/backend controls executed; Mamba-1 ONNX parity remains unimplemented/unverified |
| CUDA, ROCm, MPS and other targets | Existing matrix tiers unchanged; this area adds no physical accelerator execution evidence |

Trained-model/corpus quality, quantized execution, custom normalization and
nested generation conventions, Mamba feature compositions (grammar, speculation,
adapters and serving caches), larger-model performance and bounded serving RSS
remain open. The generated hardware table and its matrix remain the source of
truth for all individual target tiers. The wider mission is unfinished.

## Unchanged optimizer dispositions

These changes repair checkpoint and public execution contracts. They do not
establish a benefit for any additional optimizer pass; the linked audit retains
its actual measurements and outstanding ordering/idempotency checks.

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
