# Preserve published artifacts until compiler acceptance

The compiler builds and checks a private sibling candidate before replacing
the public AEG directory. Packaging failure, rejected evaluation and evaluator
exceptions preserve existing artifact bytes. Failed first compilation leaves
no public artifact. Mandatory original-source-FP32 Hugging Face admission remains
unfinished; this change supplies its publication boundary.

Read scope covered the complete compiler/configuration, evaluation pipeline,
artifact-admission policy, directory publication helper, compiler/discovery
tests, real SafeTensors roundtrip tests, AEG save/integrity/weight-store contracts,
merge publication callers, and architecture/format/quality reports. No model,
weight, engine or kernel algorithm changes in this area.

The original stage 4 deleted existing output and wrote directly to its public
path before validating or executing the optional evaluator. Eight actual
checkpoint regressions failed against dea65036a42f0d11d0a95471049de42facb64461:
rejection, evaluator exception and invalid callback with/without old output, plus
a genuinely missing down-projection tensor during packaging. All pass afterward.

Candidate construction, integrity checking, evaluation and promotion share a
nonblocking native writer lock. Windows locks byte zero with
[msvcrt.locking](https://docs.python.org/3.10/library/msvcrt.html#msvcrt.locking);
POSIX uses [fcntl.flock](https://docs.python.org/3.10/library/fcntl.html#fcntl.flock).
The Windows reference is
[CPython v3.10.11](https://github.com/python/cpython/blob/v3.10.11/PC/msvcrtmodule.c),
which calls the [CRT locking API](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking).
Explicit unlock addresses the same range. Unsupported native operations propagate;
no marker-file lease, retry loop or soft-lock fallback substitutes for exclusion.

A stable .aether-publication-<destination hash>.lock file stays beside the package.
Its existence does not mean its kernel lock is held. The opened descriptor must
match the named regular file. Descriptors close on acquire/unlock failures.
Real child-process tests verify exclusion, later reacquisition, and release after
an immediate process exit that runs no Python cleanup. Windows is physical-host
evidence; POSIX execution remains awaiting validation on that OS.

Private staging uses exclusive directory creation with up to eight filename
collision attempts; permission denial propagates immediately. Existing files,
links, filesystem roots and output directories that would replace/contain the
local source checkpoint are refused. Cleanup removes owned staging on ordinary
failure and logs retained files if deletion itself fails.

On acceptance the old directory moves to a private rollback location, then the
candidate moves to the public destination. Failed promotion restores the old
directory. This includes cancellation immediately after the first OS rename:
review found that a Python flag updated afterward could miss that completed move.
The added regression failed before this correction. Recovery now consults the
actual backup directory. If restoration itself fails, the error includes the
recovery path and both rename errors; cleanup retains the previous bytes.

Overwrite uses **two renames with rollback, not a crash-atomic exchange or a
reader lock**. Readers can briefly observe a missing destination. Process/machine
failure in that interval can require recovery from private previous.aeg.
Crash-recovery automation and fsync durability are not claimed. Programs mutating
the destination without its writer protocol are outside cooperating exclusion.
A new artifact needs one rename. A real Windows member handle that denies
deletion verifies bounded failure while retaining the old artifact.

Evaluators receive spec["aeg_path"] for actual candidate execution and
spec["publication_path"] for its intended destination. Callbacks that captured
old output must migrate to the candidate. Persisted benchmark reports record the
public destination, and the returned lazy weight index rebinds there. Positive
tests execute real CPU logits and greedy IDs before/after promotion and reload.
Callback-contract scores are controlled values, not measured MMLU or HF quality.

The rejected-artifact integration still verifies Runtime policy using an
independently persisted rejected control. Failed compilation now publishes no
rejected artifact. Default-cache compilation, discovery and Runtime listing retain
their canonical contract and exclude private stages and sidecar lock files.

## Executed correctness and memory checks

The original eight regressions changed from eight failures to eight passes.
The final two new publication test modules contain **26 passing cases**; the
earlier 44-case combined selection also passed before four additional native-lock
regressions. Complete collection results are in
[mission validation](mission-validation.md): **5214 passed, 41 skipped, zero failed**,
with every prior case retained and all 26 additions passing. Line/branch coverage
rose to **78.64579% / 62.35862%**. New source/test/harness full Ruff and
scoped checks pass; compiler/integration Ruff retain their existing 26/53
diagnostics, and compiler mypy retains three existing unrelated return-Any errors.

The first lock implementation used filelock 4.1. A real import profile found
1,363,968 bytes additional RSS and 1,303,710 traced Python allocation bytes,
including unused async, read/write and lease facilities. Native standard-library
locking reduced those observations to **8,192 RSS bytes and 28,399 Python bytes**.
A fresh-process allocation regression enforces a 256 KiB publication-import
budget. The direct dependency was removed; installed package metadata was rebuilt
offline using declared isolated build requirements, and pip check passes.
Before/after profiles and failed build/download attempts remain preserved.

## Serial compiler and runtime diagnostic

The [reusable audit](../tools/audit_compiler_publication.py) loads the exact prior
compiler from Git; other modules use the current checkout. It uses a real,
untrained 26,784-parameter two-layer Llama checkpoint, fixed source/epoch/thread
settings, three alternating cold-cache builds per revision, two warmups and seven
timed requests per prompt. A/B prompts contain 4/64 IDs and generate eight IDs.
This is a tiny Windows CPU diagnostic, not a trained-model or 27M/8B benchmark.

All **16 files** are byte-identical across all six builds, with no normalization:
**190,976 bytes per artifact**, including manifest, graph, weights and native
kernel. Both public Runtime calls succeed. Logits match exactly on all ten
teacher-forced positions: maximum absolute error and KL are **0**, minimum cosine
is **0.9999999999999998**, greedy agreement **16/16**, and fixed-label PPL change
**0%** on eight labels. This compares AEG to AEG; no HF oracle was executed.

The following values are median [min, max]; times/rates retain their observed
losses. No inference speedup is claimed.

| Measurement | Prior compiler artifact | Current compiler artifact |
|---|---|---|
| Cold compile + integrity, seconds | 2.5596 [2.4976, 3.2678] | 2.6803 [2.4373, 2.9893] |
| Prompt A decode tokens/s | 1411.5 [720.9, 1858.7] | 1307.7 [741.8, 1796.2] |
| Prompt B decode tokens/s | 1516.7 [1006.5, 1959.5] | 1404.2 [936.8, 2007.6] |
| Prompt A TTFT, ms | 1.0204 [0.6622, 2.3968] | 1.2395 [0.7863, 2.7310] |
| Prompt B TTFT, ms | 1.2552 [1.1125, 1.8208] | 1.9559 [1.2744, 2.1906] |
| Main compiler peak RSS, bytes | 57659392 [57303040, 57933824] | 57602048 [57155584, 57757696] |
| Compiler tree peak RSS, bytes | 144424960 [144179200, 144601088] | 144572416 [144412672, 145145856] |
| Load RSS delta, one worker, bytes | 4878336 | 4476928 |
| Generation phase peak RSS, one worker, bytes | 59047936 | 58642432 |
| Public Runtime peak RSS, one worker, bytes | 78413824 | 78417920 |

Timing medians were slower despite identical runtime code and artifact bytes.
Ranges overlap; the shared host's minimum available RAM fell to 444194816 bytes.
The unchanged 128 MiB diagnostic headroom and conservative action estimates stayed
active. These observations do not establish a causal engine speed regression,
a speedup, component-memory savings or a clean-host performance guarantee.
RSS samples can miss brief peaks, and tree RSS can double-count shared pages.
The compiler-import memory reduction is separately attributed by allocation
profiling. Larger models and clean-host performance validation remain open.

The successful report is build/mission/publication-audit-native/report.json.
Earlier two RAM-blocked attempts, an audit with a missing runtime pass-selection
field, and a successful general-lock prototype audit remain as separate evidence.
The harness field was corrected from the actual compiler configuration; runtime
worker behavior was not relaxed. No failed run is relabeled successful.
Raw profile/regression evidence uses publication-import-*, publication-cancel-*,
publication-native-focused.* and quality-publication-* under build/mission/.

The [22-pass dispositions](mission-default-policy.md),
[executed pass matrix](mission-pass-audit.md) and
[hardware evidence matrix](../hardware_validation_matrix.json) are unchanged.
Current evidence is physical Windows CPU/native locking, not accelerator or
other-OS validation. Mandatory numerical admission, model math, engine performance
and memory, broader hardware, robustness and blocking CI remain active work.
