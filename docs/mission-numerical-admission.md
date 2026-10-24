# Measured numerical admission before publication and execution

Executed on 2026-10-20, physical Intel Core i5-1334U, Windows build 26200 AMD64,
8,259,321,856 bytes RAM; Python 3.10.11, NumPy 1.26.4, Torch 2.9.1+cpu,
Transformers 4.57.3, SafeTensors 0.8.0 and ONNX Runtime 1.23.2.
The overall runtime/compiler mission remains in progress.

## Contract and integration

Ordinary `Compiler.compile` now requires real original-HF numerical evidence.
The source config, tokenizer inputs and all SafeTensors shards are hashed. A
remote revision is resolved once to an immutable Hub commit, shared by detection,
ingestion and reference loading. An isolated worker loads that original source
as FP32 with local files only, eager attention, no remote code and no partial or
mismatched loading. It finishes before compiler ingestion starts; reference
weights do not remain resident alongside the compiler's graph.

After packaging into a private candidate, serial workers load and execute that
actual AEG through NumPy CPU and Torch CPU. Fixed teacher-forced labels determine
logits and NLL; separate append-only caches generate greedy continuations. Text
probes also compare the packaged and original tokenizers. The mathematical
oracle and primary/reference implementation citations remain in
[the validator report](mission-quality-validation.md).

Every probe and both executors must meet maximum per-position KL <= 0.01,
minimum cosine >= 0.999, greedy agreement >= 95%, and PPL increase <= the smaller
of 1% and the requested quality budget. Checks also retain the validator's MAE
bound. Empty, inconsistent, nonfinite or relaxed evidence cannot pass. Defaults
use three bounded text probes and 32 greedy tokens each. These regression probes
are not corpus perplexity, task accuracy or general model-quality certification.

The schema `aether-numerical-quality/1` report lives at
`quality/numerical_gate.json`. It binds source and sample identities to the
candidate manifest header and every declared payload except the report itself.
Unmanifested files and linked candidate payloads are refused. Workers compare
the loaded candidate identity with the parent's identity and check it again after
execution. Only passing evidence is persisted, hashed, revalidated and promoted.
The existing graph/manifest integrity chain remains in place. An optional
benchmark callback is supplementary and cannot substitute for this gate.

The loader, Runtime and direct native CPU/Torch/ONNX backend AEG admission paths
enforce the report before model weight loading; Runtime preflight precedes
provider selection. Cached diagnostic
handles cannot authorize later ordinary loads. Reports are bounded to 1 MiB;
hash authentication and parsing use the same bytes, for numerical evidence and
the older benchmark gate. Declared missing, malformed, tampered or rejected
reports remain refusals even in diagnostic mode. Merging removes stale numerical
evidence along with older weight-dependent evidence and requires reevaluation.

This is not a signer identity or a cryptographic measurement attestation.
Independent ONNX and accelerator numerical certification remains unfinished.
The report covers the measured base execution; request-time adaptation, task
deltas and adapters require their own expected-reference checks. Dynamic precision
requires explicit diagnostic execution and clears the loaded admission decision
when execution changes; strict automatic precision does not mutate admitted weights.
Compiler TTT, adapter-source/PEFT and source-merging admission fail closed when
that independent transformed reference is unavailable. Post-hoc changes,
including signing/package metadata changes, can invalidate the binding; a signed
and numerically admitted publication workflow is not yet implemented.

## Using and diagnosing the gate

Install the reference dependencies on the compilation machine:

```bash
pip install "aether-runtime[transformers-frontend]"
aether compile ./checkpoint --target cpu_generic --precision fp32 --quality-dataset probes.jsonl --output model.aeg
aether run model.aeg --prompt "Hello" --max-tokens 8 --temperature 0
```

Each JSONL record has a unique nonempty `id`, and either nonempty `prompt` and
`continuation` strings, or `prompt_ids` and `continuation_ids` integer arrays.
Do not mix the forms. Checks refuse truncation, out-of-vocabulary IDs, more than
128 probes or more than 4,096 fixed-label positions. `--revision` selects the
Hub revision to pin. Explicit FP32 preserves global weights and biases; BF16 and
lossy opt-ins must still pass the same gate.

An admitted CPU artifact can be executed without Torch or Transformers. That
was tested in a fresh process with imports of both frameworks explicitly denied,
followed by ordinary public `Runtime.generate` and four actual output tokens.
Fresh CLI execution of the same artifact also completed with diagnostic mode
disabled. Compilation requires the reference dependencies; framework-free
development compilation remains explicitly uncertified:

```bash
aether --diagnostic compile ./checkpoint --target cpu_generic --output development.aeg
aether --diagnostic run development.aeg --prompt "Hello"
```

The Python equivalents are `CompilerConfig(quality_mode="diagnostic")` and
`RuntimeConfig(allow_uncertified=True)`. Low-level loaders/backends accept the
explicit `allow_uncertified=True` decision. Diagnostic mode waives absent
evidence only; it does not turn a failed report into a success. Existing AEGs
remain readable/inspectable, with diagnostic execution available when evidence
is absent. No automatic certification upgrade is inferred from an old manifest.

## Actual quality, storage and admission costs

The successful real subprocess run used the untrained 26,784-parameter fixture,
4/64-token prefixes, four fixed labels and eight greedy tokens per prefix.
It executed original FP32 HF, CPU and Torch CPU, published the accepted AEG,
then generated through ordinary Runtime and CLI. Both executors matched all
16 tested greedy output IDs. Across both engines/probes, maximum KL was
4.0931255353e-16, minimum cosine 0.9999999999999749 and the largest positive
relative PPL change 1.149934199e-8. The artifact occupied 252,745 bytes.

The gate also correctly rejected the earlier purported FP32 candidate, whose
probe A matched only 6/8 greedy IDs. The underlying BF16 rounding was corrected
at the precision assignment/packer boundaries; [before/after data](mission-fp32-storage.md)
include its additional storage cost. Thresholds were not relaxed.

| Actual admission worker observation | HF | CPU | Torch CPU |
|---|---:|---:|---:|
| Wall time, seconds | 41.750 | 5.594 | 8.641 |
| Sampled worker peak RSS, bytes | 298,528,768 | 59,838,464 | 218,537,984 |
| Sampled owned-tree peak RSS, bytes | 302,997,504 | 116,137,984 | 223,031,296 |
| Minimum available host RAM, bytes | 375,062,528 | 779,825,152 | 630,231,040 |

These are diagnostic observations on a shared, memory-constrained host, not a
clean performance benchmark. Sampling may miss peaks and tree RSS can double
count shared pages. They do not break memory down into weights/KV/activations.
Mandatory comparison adds compilation cost; no serving speed/TTFT gain or
flat-RSS serving soak is claimed. Tiny functional execution does not certify
mid-size, trained, 8B or accelerator workloads.

Workers reserve estimated space before importing frameworks and retain default
256 MiB host headroom throughout execution. The parent proves PID/creation-time
ownership before acknowledging a worker and only terminates its owned process
tree on timeout, memory refusal or cancellation. Windows shutdown waits for
children to release inherited log handles before cleaning temporary storage.
Earlier RAM refusals and the pre-fix Windows cleanup error remain recorded.
The latest replay was refused before HF loading: 729,747,456 bytes available,
537,320,512 estimated worker bytes plus 268,435,456 required headroom. No artifact
was published. The estimate is a policy bound, not a measured worker footprint.

The final-code replay (`numerical-live-9`) completed all three isolated workers,
published the same-sized artifact and reproduced the same numerical metrics.
Fresh strict CLI execution and a fresh CPU process denying Torch/Transformers
imports both exited zero and generated four actual tokens. This replay was
instrumented with Coverage 7.16.2 subprocess collection: worker times were
54.547 / 2.953 / 17.187 seconds (HF / CPU / Torch CPU), sampled worker RSS peaks
306,274,304 / 66,134,016 / 226,299,904 bytes, and owned-tree peaks
310,771,712 / 111,210,496 / 230,825,984 bytes. These include instrumentation;
they are not before/after serving performance. All unsuccessful attempts remain
retained, including the later RAM-refused instrumented replay before this success.

## Regression evidence and scope

The read scope includes the complete new admission/worker/gate modules, compiler
and config, loader, direct backend/cache contracts, Runtime and CLI callers,
weight packer/optimizer, merge invalidation, hardware probe helper, affected
synthetic fixtures, tools and their tests. Fixture changes explicitly request
diagnostic execution; no global test waiver or manufactured passing record exists.
Hardware probes now state that their synthetic functional test is diagnostic
and has not validated original-HF quality.

Observed failing-before tests cover default admission, cached ONNX admission,
worker identity, undeclared runtime payloads, report replacement between hash
and parse, unbounded benchmark reads, stale numerical evidence after merging,
FP32 global/bias storage and owned Windows worker shutdown. Actual arithmetic
tests invoke original HF and both AEG worker bodies; they do not replace metric
calculations with passing DTOs. Separate real subprocess evidence establishes
the production worker path.

Focused results: 59 report/admission/FP32 cases passed; the subsequent expanded
merge/gate selection passed 53 cases. The six new Python files pass strict mypy
with followed imports silent. Complete product/tools coverage and touched-source
static deltas are recorded after the serial area run below.

Two existing MCP HTTP cases also fail in an independent stdlib-only POST
reproduction: the server sends Content-Length, while the client receives
Transfer-Encoding: chunked followed by an unchunked body. Raw socket POSTs show
the same altered framing; GETs retain correct framing. Disabling proxy use and
the approved outside-sandbox rerun did not resolve it. Product/test MCP code is
unchanged; no malformed-framing acceptance or assertion bypass was added.

Raw local records: `build/mission/numerical-live-5.log`, `numerical-live-6/`,
`numerical-live-6.log`, `numerical-live-7/`, `numerical-live-7.log`,
`numerical-live-comparison.json`, `numerical-reports-after.*`,
`numerical-invalidation-before.*`, `numerical-invalidation-after.*`,
`numerical-benchmark-bounds-before.*`, `numerical-report-race-before.*`,
`loopback-framing-post.json`, `urllib-framing-independent.log` and
`mcp-http-outside-sandbox.*`. Failed attempts remain failures.

## Complete area verification

The final product collection ran against unchanged source and returned
**5,223 passed, 41 skipped, two failed** in 1,501.74 seconds. Both failures are
the independently reproduced MCP HTTP framing cases described above. The earlier
run's additional lifecycle-fixture failure was corrected by explicitly choosing
diagnostic execution for its uncertified subprocess fixture; all 30 lifecycle
tests then passed, including that assertion in the final collection.

Two complete tool-suite attempts each returned 42 passed and one RAM-blocked
case. The first blocked Unicode/ASCII-parent compilation; its public
compile/load/generate smoke passed. The second passed that Unicode test but
blocked the public smoke's later fusion build: 512,770,048 available bytes
versus 403,510,272 estimated worker bytes plus 134,217,728 headroom. No assertion
or memory guard was relaxed. The third returned 41 passed and both smoke cases
RAM-blocked. Its Unicode worker had 537,862,144 available bytes versus
406,938,624 estimated bytes plus 134,217,728 headroom. This host has not produced
an all-green area run; the full-suite-green requirement remains open.

The final serial product/tools collections therefore contain **5,264 passed,
41 skipped and four failed**: two HTTP framing failures and two RAM refusals.
Combined line coverage increased from **44,497/56,579 (78.6457873%)** to
**45,001/57,206 (78.6648254%)**. Branch coverage increased from
10,586/16,976 (62.3586239%) to 10,734/17,206 (62.3852145%). Failed attempts are
included in the record rather than replaced with earlier passing case results.

The retained baseline contains 5,255 cases. The expanded collection contains
5,309 cases, adding 54 and retaining every prior identity. Assertion inspection
of 42 migrated existing test files found unchanged assertion syntax except the
cached-handle identity assertion, whose load now explicitly permits its
diagnostic fixture. The identity assertion itself is retained.

Touched-source strict mypy diagnostics decreased from 263 to 254, with no
file/error-code increase; 254 pre-existing findings remain. The six new files
have no diagnostics. The complete touched-file Ruff comparison decreased from
1,346 to 1,330 findings with no file/code increases. Critical Ruff
`E9,F63,F7,F82`, `git diff --check`, and generated hardware README drift checks
pass. These comparisons use the exact `e902da5` source baseline; they do not
clear the repository's wider lint/type backlog.

Final records are `build/mission/numerical-product-2.*`,
`numerical-tools-3.*`, `numerical-final-coverage.xml`,
`numerical-area-verification.json`, `numerical-assertions.json`,
`numerical-mypy-delta.json`, `numerical-mypy-features-before.*`,
`numerical-mypy-new-10.*`, `numerical-static-delta.json`,
`numerical-ruff-critical-final.*` and `numerical-live-9/`. Earlier complete
attempts and their failures remain under their original numbered names.

## Passes and hardware

The [22-row disposition table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [per-pass measured audit](mission-pass-audit.md) remain the baseline pass
findings. This area changes P3's explicitly requested FP32 preservation and adds
admission of measured results; it does not establish a benefit for the other
21 passes. Only P3 remains default. Unsafe/unmeasured opt-ins cannot inherit a
quality claim from an `applied` status.

Physical CPU execution is the current hardware evidence tier. The generated
README table and `hardware_validation_matrix.json` retain each backend's dated
technique/scope, including historical DirectML execution, reference/compile-time
checks and implementations awaiting hardware. No accelerator tier is promoted.
Original mission work on model-family math, ONNX parity, engine speed/memory,
KV eviction/COW/soak, larger trained models, archive hardening and blocking CI
remains unfinished.
