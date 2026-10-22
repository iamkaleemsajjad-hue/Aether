# Preserve normalization epsilon across AEG loading

The shared loader now preserves an explicit `norm_eps: 0` for dense/MoE,
MLA, hybrid, Mamba-2, RWKV and encoder artifacts. Previously `value or default`
silently changed that constant to `1e-5`, or `1e-12` for encoders. Missing/null
values retain their established fallback. Mamba-1 uses the same reader with
unchanged behavior; seq2seq already preserved zero and is unchanged.

This repairs execution configuration. It adds no optimizer, graph interpreter,
artifact field, format version, weight conversion or numerical approximation.
The existing graph/manifest/payload integrity chain remains in use.

## Inspection and reproduction

Read the complete shared loader, its caller map, the Mamba-2 fixture and
normalization tests, the strict HF validation tool and CI workflow. Reviewed
the encoder, recurrent, hybrid, Mamba and ONNX regression paths and the existing
architecture, format, numerical-admission and model-fidelity reports. The
[architecture decision](mission-architecture-note.md) remains engine-owned
execution: the deserialized constant reaches real normalization kernels.

For RMSNorm, the consumed denominator is `sqrt(mean(x*x) + epsilon)`; LayerNorm
first subtracts the feature mean and uses population variance. The exact source
contract is visible in
[Transformers 4.57.3 LlamaRMSNorm](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/llama/modeling_llama.py)
and [Mamba-2 normalization](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/mamba2/modeling_mamba2.py).
Zero is therefore a numerical parameter, not a missing-value marker. No new
algorithm is introduced; reference equations and execution remain unchanged.

The regression creates real FP32 indexed weight blobs, manifests and hashed
minimal IR, reopens/authenticates them, and checks the loaded normalization
against independent FP64 arithmetic for nine dispatch cases. These are loader
fixtures, not complete source-model parity claims. Explicit `0.002`, null and
already-correct Mamba-1/seq2seq paths provide controls.

Separate tests compile complete local Llama and Mamba-2 SafeTensors checkpoints,
reload them, and execute full prefill, cached continuation and eight greedy IDs
against the original HF source model. Llama also runs actual ONNX Runtime CPU
operators. All fixtures are small and untrained. The Llama probe uses ordinary
projection scales and low embedding variance to distinguish substituted epsilon
from ordinary FP32 reduction-order differences; numerical tolerances were not
relaxed. Earlier probe attempts remain in local raw logs and are not substituted
for the final reproduction.

The retained pre-fix runtime is the exact 304-file Git archive of
`66e0a72ace69a371d21208c849efcf419bd289d4`, with its import root checked before
pytest. Against it, the final regression returned **12 failed, 19 passed**:
seven zero-epsilon loader cases and all five full-source/backend cases failed
numerical assertions. An initial missing fixture IR-version error was corrected
before that reproduction and is not counted as a product failure.

After the correction, focused validation returned **102 passed, 2 skipped**,
including all 31 new cases, existing Mamba-1/2, encoder, RWKV, hybrid and ONNX
controls. The two skips require an installed physical DirectML provider. These
checks ran on Windows x86 CPU with Python 3.10.11, NumPy 1.26.4, Torch 2.9.1,
Transformers 4.57.3, ONNX Runtime 1.23.2, SafeTensors 0.8.0 and Tokenizers 0.22.2.

## Measured validation scope

The strict [HF tool](../tools/validate_mamba2_hf.py) now includes a Mamba-2
zero-epsilon case alongside both existing controls. The blocking CI job compares
it to immediate pre-fix `66e0a72`; older normalization comparisons remain in
[their original report](measurements/mamba2-normalization-hf.json).

The first local strict run executed its original-HF reference, then refused the
current Torch admission worker: available RAM was **732,246,016 bytes**, versus
**536,923,100** reserved worker bytes plus **268,435,456** bytes of live headroom.
Its [complete partial report](measurements/normalization-epsilon-local-hf-attempt.json)
retains the failure and the executed diagnostic baseline. The committed copy
normalizes CRLF to LF without changing JSON values; the local validation record
retains both byte hashes. No current timing,
memory reduction, KL/PPL or public API/CLI success is assigned to that refusal.
The independent in-process pytest comparisons above are separately scoped
execution evidence; they do not bypass or satisfy subprocess RAM admission.

The [single complete local run](measurements/normalization-epsilon-local-validation.json)
returned **5459 passed, 42 skipped, two failures**, in 1174.83 seconds. All 31
final epsilon cases passed in that run. Both failures are the existing MCP HTTP
handshake cases, whose earlier
[wire observation](measurements/mcp-local-framing-observation.json) remains open.
The RAM audit case passed this time with its original guard; that is not an
environment fix. This complete local run is **not green**.

| Local run metric | Prior complete run | This complete run |
|---|---:|---:|
| Collected cases | 5472 | 5503 |
| Passed / skipped / failed | 5427 / 42 / 3 | 5459 / 42 / 2 |
| Covered / total lines | 45152 / 57353 | 45186 / 57356 |
| Line coverage | 78.7264834% | 78.7816445% |
| Covered / total branches | 10822 / 17290 | 10837 / 17290 |

The prior run is bound to `ad77e08`, whose runtime source is identical to
baseline `66e0a72`. Raw local records use `build/mission/epsilon-*`. Complete
suite, coverage and hosted benchmark results describe separate workloads;
passing focused cases do not establish controlled performance. The baseline
regression preceded a metadata-only fixture parameter-count annotation and
additional finiteness assertions; the complete current run used the final files.

Ruff remains **71 findings** and scoped loader mypy **39 errors**, with no added
file/code/message diagnostics. New test files and the validation tool pass scoped
mypy; new/changed tests and tool pass Ruff. The existing Mamba-2 fixture retains
its ten pre-existing mypy diagnostics. Repository-wide static debt remains open.

## Passes and hardware

No pass disposition changes or new pass-benefit claims follow from this fix.

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

| Backend scope | Verification tier and technique for this change |
|---|---|
| NumPy CPU | Physical Windows CPU execution; authenticated loader/FP64 checks and original-HF Llama/Mamba-2 prefill, cache and greedy comparisons |
| Torch CPU | Real CPU tensors and original-HF comparisons on the same compiled source contracts |
| ONNX CPU | Actual Llama prefill/cached-decode/generation operators against original HF; no recurrent ONNX claim |
| DirectML | Optional physical-provider checks skipped; historical matrix evidence unchanged |
| CUDA, ROCm, MPS and other targets | No physical execution in this change; existing matrix tiers unchanged |

Trained/corpus quality, other family-wide parity, accelerator execution,
larger-model performance, resident memory attribution/soak, pass composition,
archive/download hardening and blocking core/runtime type checks remain open.
The [hardware matrix](../hardware_validation_matrix.json), generated README
table and [pass audit](mission-pass-audit.md) retain their existing scopes.
