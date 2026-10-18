# Mission validation record

The architecture decision is recorded in [the pre-change note](mission-architecture-note.md).
The mission remains in progress. This record separates executed checks from
inspection and work that has not run.

## Initial source and environment

The initial runtime/compiler source was `7bed38c6722630fdfc7c16ddfe517a52b4369221`.
No production source changed during baseline execution on 2026-10-17. New audit
tool tests were added after that suite's collection and are recorded separately.

Host: Intel Core i5-1334U, 10 physical cores / 12 logical CPUs, 8,259,321,856
bytes RAM, Windows build 26200, AMD64. The isolated environment uses Python
3.10.11, NumPy 1.26.4, Torch 2.9.1+cpu, Transformers 4.57.3, ONNX 1.23.2 and
ONNX Runtime 1.23.2. `pip check` passed. Exact dependencies are retained in
`build/mission/clean-freeze.txt`; hardware/software details are in
`build/mission/clean-environment.json`.

The first environment inherited unrelated system-package conflicts. Its
interrupted suite/type-check attempts are not baseline results. A separate
clean environment was installed before the authoritative run.

## Executed baseline

| Check | Result | Evidence |
|---|---|---|
| Serial complete suite, coverage enabled | 4,923 passed; 41 skipped; 0 failures/errors; exit 0 | `build/mission/clean-baseline.log`, `.xml`, `.exit` |
| Suite wall time | 402.37 seconds | Same log |
| Line coverage | 43,996 / 56,271 = 78.19% | `build/mission/clean-baseline-coverage.xml` |
| Branch coverage | 10,441 / 16,850 = 61.96% | Same coverage XML |
| Combined pytest-cov display | 74% | Same suite log |
| Critical Ruff, all CI paths | Passed | `build/mission/critical-before.txt` |
| Full Ruff, source / source plus tests | 5,341 / 9,647 findings with Ruff 0.17.1 | `build/mission/clean-ruff-before.json` |
| Full-package mypy 2.4.0 | 1,796 errors in 160 files; 297 source files checked; exit 1 | `build/mission/mypy-source-baseline.log`, `.exit` |
| Four multiprocessing checks previously denied by the sandbox | 4 passed outside sandbox, exit 0 | `build/mission/ipc-corrected.log`, `.xml`, `.exit` |
| Audit helper contracts, first revision | 26 passed; 1 optional integration skip | `build/mission/tool-tests-initial.log` |
| Audit helper contracts, expanded revision | 31 passed; 1 optional integration skip | `build/mission/tool-tests-expanded.log` |
| Audit helper contracts, worker-identity sampler revision | 33 passed; 1 optional integration skip | `build/mission/harness-validation/sampler-pytest.log` |
| Audit helper mypy (third-party imports skipped) | Passed, 2 source files | `build/mission/tools-mypy-final.log` |

The complete suite ran with Torch and Transformers installed, offline model
access, one BLAS/OpenMP thread, separate cache/temp directories, and no pytest
workers. Network tests retained their explicit skip policy. Skips include
optional accelerate/cryptography/OpenTelemetry SDK dependencies, unavailable
datasets or hardware, deliberate graph-only synthetic checkpoints, and the four
subsequently rerun IPC checks. A passing suite does not certify skipped paths.

The supplied audit's 4,415-test/65%-coverage figures describe a different run;
these results are not a controlled before/after improvement in runtime behavior.
The supplied 982-error mypy count was not reproduced with this dependency/tool
set. The current full-package check fails; it is not a passing quality gate.
The helper-only check has narrower scope and does not clear production errors.

## Findings at the initial source, before corrections

- The legacy public `TreeSpeculativeEngine` builds candidates without a model.
  An executed reproduction with prefix `[1, 2, 3]` accepted `[3, 94, 2916]`
  with no target logits and recorded acceptance 1.0. Evidence:
  `build/mission/speculative-before.json`. Repository searches found no serving
  call to its builder; it remains publicly exported.
- A failed model-merging pass can be reported as failed while compilation
  returns an integrity-valid runnable artifact. An enabled grammar pass without
  its schema can also be skipped while compilation succeeds. These were
  independently reproduced before any compiler fix.
- Sub-2-bit compilation was independently reproduced without an evaluator,
  including a zero configured quality threshold. Its report remained
  `not_evaluated` / `uncertified`.
- The quality validator has no production caller. Empty or incomplete evidence
  can pass, and its engine comparison resends growing history with a populated
  cache. A real CPU-engine reproduction advanced cache lengths `[2, 5, 9]`
  instead of `[2, 3, 4]`; fixing the input protocol changed its generated IDs.
- Graph-only rewrites do not determine dense CPU/Torch forward schedules.
  Explicit plans and changed weights can have runtime consumers; each needs
  separate evidence. An unused reference IR interpreter also exists.
- Standalone watermark implementations exist. The unresolved integration claim
  concerns default artifact configuration being advertised as enabled without
  a demonstrated serving consumer.
- `aether hardware validate` already exists. Historical scoped DirectML records
  are separate from the current CPU environment and have not been rerun here.

## Measurements still required

The first native audit completed for the tiny fixture, all passes off versus
fusion alone, with two fresh builds each. Loader execution, numerical execution,
public `Runtime.generate`, and the original FP32 Hugging Face oracle succeeded.
On each variant, all 16 generated IDs matched the oracle; maximum KL was below
`5.5e-8`, minimum logit cosine exceeded `0.999995`, and relative fixed-label PPL
increases were below 0.016%. These are untrained synthetic-model results, not
corpus quality. Fusion logits matched the unoptimized AEG exactly. Timing was
not requested. Evidence: `build/mission/first-audit/report.json` and job logs.

The audit exited 1 because fresh native DLL builds differed despite the fixed
reproducible-build epoch. Only the DLL bytes and their dependent metadata and
manifest hashes differed; no fields were stripped. Applying fusion twice to
one graph also reset its accounting metadata, so it was not graph-idempotent.
These findings remain unfixed at this checkpoint.

The first harness version followed the Windows venv launcher for sampled
main-process RSS, while its phase boundaries came from the actual Python child.
Its peak-RSS results are therefore not accepted as validated measurements.
The corrected sampler now checks owned worker PID and creation time, measures
that worker separately from its process tree, and rejects stale/unowned markers.
A real allocation regression touches and frees 64 MiB in the Python child before
exit; the sampled peak exceeds final RSS by at least 48 MiB. The 33 helper tests
passed, including that regression. The original peak values remain invalid;
numerical/token comparisons are unaffected. Sampling can miss brief peaks, and
summed tree RSS can count shared pages more than once.

The legacy speculation regression file separately reproduced 35 failures and
5 passes against unchanged production source. Evidence:
`build/mission/legacy-before.log`, `.xml`, `.exit`.

The full pass matrix has not completed. No per-pass speedup, memory reduction,
pretrained-model quality, or engine optimization is claimed by the helper tests.
Working status requires an actual consumer and measured benefit or an executed
functional assertion. Raw artifacts must be compared without stripping fields.

The pending matrix includes all 22 passes, tiny and 27,271,680-parameter synthetic
models, and MoE/MTP/adapter fixtures. Synthetic numerical agreement is distinct
from language-quality evaluation. Repeated builds and repeated application to
one graph are separate checks. Current host memory pressure may prevent larger
workers; an admission estimate is not a measured footprint.

| Pass | Name | A/B evidence available at this checkpoint |
|---|---|---|
| 1 | Operator fusion | Identical logits to all-off AEG; artifact 210,737 to 205,111 bytes; no execution-schedule consumer or speed benefit established |
| 2 | Sensitivity analysis | Pending, including sensitivity/default-precision composition |
| 3 | Precision assignment | Pending |
| 4 | KV cache structuring | Pending |
| 5 | MoE routing | Pending MoE fixture |
| 6 | Parallelism discovery | Pending; multi-device execution requires separate evidence |
| 7 | Reasoning graph | Pending |
| 8 | Sparse attention | Pending; forced short-context diagnostic is not long-context validation |
| 9 | Pruning/sparsity | Pending |
| 10 | MTP head compilation | Pending MTP fixture |
| 11 | Grammar constraint compilation | Pending functional grammar A/B |
| 12 | Model merging | Pending numerical/functional A/B; failure admission issue reproduced |
| 13 | TTT fast weights | Pending persisted-slot consumption check |
| 14 | Semantic KV compression | Pending CPU behavior/quality/memory measurement |
| 15 | Cross-layer KV sharing | Pending CPU behavior/quality/memory measurement |
| 16 | Green energy compilation | Pending; no measured energy saving claimed |
| 17 | TEE wrapping | No executable TEE bundle in fixtures; hardware execution unverified |
| 18 | MDLM drafter | No trained drafter in fixtures; no fabricated substitute |
| 19 | Sub-2-bit quantization | Pending numerical A/B; missing mandatory evaluation reproduced |
| 20 | Video compression | Text fixtures cannot validate visual inference |
| 21 | Advanced PEFT | Pending adapter fixture and reference transformation |
| 22 | RLVR verifier | Inference generation cannot validate training/gradient execution |

CPU execution and real-process collectives have current physical-host evidence.
CUDA, ROCm, Metal, native vendor/NPU/FPGA paths, and hardware attestation require
their own execution evidence. None is promoted by CPU tests, mocks, package
installation, or target-profile presence.

## Correction: legacy speculation fails closed

Read scope included the complete legacy module, its public exports and runtime
caller, the executable speculative decoders and sampling/cache contracts, related
unit/integration tests, and runtime feature documentation. The runtime imports
the legacy class but does not call its builder in the serving path.

The unsupported builder now raises an explanatory error. Manual greedy
verification requires valid target logits, validates candidate IDs and
probabilities, and updates counters only after successful verification. Existing
imports and method signatures remain available. The real draft/target decoder
was preserved. See [migration and scope](legacy-speculation.md).

| Check | Before | After |
|---|---|---|
| New legacy regression file | 35 failed, 5 passed | All 40 passed in the focused selection |
| Focused runtime/speculation/Torch/persistence selection | Not run as one baseline selection | 131 passed, 0 failed |
| Complete suite with coverage | 4,923 passed, 41 skipped | 4,996 passed, 42 skipped, 0 failed; exit 0 |
| Line coverage | 78.19% | 78.22% |
| Branch coverage | 61.96% | 62.02% |
| Original AEG replay, maximum absolute logit error | Reference capture | 0 |
| Original AEG replay, greedy token IDs | 16 reference IDs | 16/16 identical; public `Runtime.generate` passed |
| Engine speed / memory benefit | Not measured | No benefit claimed; engine execution was unchanged |

Evidence: `build/mission/legacy-before.*`, `legacy-after.*`, `legacy-full.*`,
`legacy-full-coverage.xml`, `coverage-comparison.json`, and
`legacy-artifact-replay/report.json`. The complete after-suite took 646.78 seconds.
Suite wall times include new tests and uncontrolled host memory pressure; they
are not a decode throughput benchmark. The extra skip is the explicitly optional
heavy pass-audit integration test. All new helper and legacy regressions ran in
the complete suite.

Full Ruff passed for the corrected production module and new regression file;
the existing runtime test file retains exactly its prior 32 findings. Targeted
mypy passed for the corrected module with imported modules excluded. This does
not clear the 1,796-error full-package baseline. Native AEG reproducibility and
the remaining pass/correctness/performance work remain open.
