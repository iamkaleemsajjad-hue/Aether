# Compiler pass audit

`pass_audit.py` is a diagnostic harness, with no changes to the compiler or
runtime. Run it from this checkout with an environment that has Aether's core
dependencies installed. Workers use this checkout's `src/`, and are always
serial. Do not run execution alongside another suite or benchmark. `--plan`
does not launch workers or create fixtures.

Workers pin Python file and pipe encoding to UTF-8, including on Windows and
when the parent requests another encoding. This makes the recorded UTF-8 logs
reliable; it does not fix unconfigured logging in other library applications.

```powershell
$python = 'build/mission/clean-env/Scripts/python.exe'
& $python -B tools/pass_audit.py --plan --fixtures tiny --passes 1
# Fast initial execution: baseline + fusion, each rebuilt twice; no timings.
& $python -B tools/pass_audit.py --fixtures tiny --passes 1 --output build/pass-audit/fusion
# All 22 single passes and representative combinations, all five fixtures.
# Large, and may be blocked by RAM admission. Use a new output directory.
& $python -B tools/pass_audit.py --fixtures all --passes all --hf-reference --output build/pass-audit/matrix
# Only when other suites/benchmarks have finished:
& $python -B tools/pass_audit.py --fixtures tiny --passes 3 14 15 19 --hf-reference --timing --repeats 7 --warmup 2 --output build/pass-audit/measured
```

The default fixture is `tiny`; `--fixtures tiny mid moe mtp adapter` or `all`
selects more. `--passes 1`, `--passes 1,3`, and `--passes 1 3` work. Every
selection includes an all-passes-off baseline and all 22 catalog rows; rows
outside the selection stay `not_run`. Combinations run only when their passes
are selected. `--no-combinations` narrows the run further. There is no all-on
configuration that hides unsupported passes.

Fixtures contain real deterministic, untrained FP32 SafeTensors checkpoints
and a packaged WordLevel tokenizer. Tiny is a two-layer Llama; mid has eight
layers, width 512 and **27,271,680 parameters**. MoE has four experts and a top-2
router. MTP has two real vocabulary projection heads. Adapter has real LoRA
A/B tensors. Each includes a matching source checkpoint with one changed Q
projection for model merging. The source fixture is generated twice and every
file is compared. No model downloads or trained heads are invented.

All **22 canonical enable flags are supplied explicitly** for every compile,
including the baseline. Pass 3 deliberately requests uniform Q4_K_M. Thus the
`(1,2,3)` combination exercises fusion plus sensitivity plus explicit precision.
The additional `(2,3)` case uses the real default sensitivity mode with no
explicit precision override; its conservative BF16 selection is observable in
the compiler configuration. Sparse
attention deliberately uses threshold **16**, instead of the default **32768**,
with a 64-token B prompt. Its default sparse windows may still behave densely;
this is not long-context validation. Cross-layer sharing uses threshold 0.1 to
produce groups even on a two-layer model. These overrides are in the report.

The report preserves the distinction between:

- compiler `applied/skipped/failed`, compile refusal, and memory `blocked`;
- real loader/generation and public `Runtime.generate` outcomes;
- an identified consumer versus observed size, latency and RSS differences;
- a pass benefit versus graph-only emission or an unattributed timing delta.

Compiler provenance supplies the status of every pass. Detailed emitted
explanations remain in each `worker.log`; the public compiler return value does
not expose every `PassReport`. TEE has no executable backend bundle; MDLM has
no trained drafter; text fixtures do not validate video. MTP/grammar public
Runtime results are recorded, but do not become `working` merely because
compilation/generation succeeded. Ordinary loader generation does not select
MTP heads, online TTT, reasoning, RLVR training, paged pools, or power control.
The adapter numerical worker explicitly selects its compiled adapter. Feature
coverage gaps stay visible; `working_claim` is never inferred from an opcode.

`versus_unoptimized_aeg` compares teacher-forced logits and greedy token IDs
against the all-off **AEG**, which packaging may still store at BF16. A and B
have fixed prompt IDs, and share a fixed teacher-forced continuation. Metrics
include KL(reference || candidate), cosine, maximum absolute logit error,
argmax agreement, greedy agreement and both token sequences. Continuation NLL
uses exactly the first four teacher-forced rows with the four fixed next-token
labels. The fifth logit row has no next-token label and does not contribute to
PPL. Relative PPL change is `expm1(candidate_mean_nll - reference_mean_nll)`;
overflow is explicit, with the full log difference preserved and no cap.
The numerical gate reports KL mean <=0.01, minimum cosine >=0.999, greedy
agreement >=0.95 and PPL regression <=1%. These synthetic measurements are
numerical regressions, not natural-language quality scores or compiler quality
certification; the comparison's AEG/HF reference label always applies.

`--hf-reference` optionally creates a **separate fresh** CPU HF Llama oracle
from the original checkpoint using strict state-dict loading and original FP32
weights, with no rounding to match the AEG. It supports `tiny`, `mid` and the
base model of `adapter`; MoE/MTP reference coverage is explicitly skipped.
`versus_original_fp32_hf` is only populated from that actual reference. Adapter
comparisons to the original base intentionally include the requested adapter's
effect. Compiler evaluation remains `uncertified`; no language-quality
evaluation callback or fabricated PPL gate is supplied.

Every build hashes **every file**, including the manifest, graph, weights,
tokenizer, native binaries, plans and provenance. Repeated builds use fixed
`SOURCE_DATE_EPOCH`, `reproducible_builds=True`, hash seed, fixture seed, and
thread counts. Diffs list added/removed/changed/identical files and SHA256s;
changed JSON leaves and potentially volatile timestamp/duration fields are
reported separately. No fields/files are stripped to obtain a passing hash.
Repeated-build determinism is distinct from applying a pass twice to one graph.
Each selected singleton also gets a separate idempotency worker: it ingests the
real checkpoint, applies that actual pipeline pass twice to the same graph,
and compares graph state, full tensor-byte hashes, metadata and every emitted
file. Changes or second-application failures are reported without repairing the
pass. This structural diagnostic establishes no runtime benefit. Use
`--no-idempotency` to explicitly skip it. Runtime file mutations are also
detected against the pre-runtime raw inventory.

Compilation, loader-only RSS, numerical execution, public text generation, HF,
and optional timing run in **different subprocesses**. This avoids compiler or
HF process high-water marks contaminating load/generation memory. Parent psutil
polls RSS every 5 ms and reports phase peaks, separate load resident/delta,
and process-tree RSS. Version-2 phase markers and final points contain the
actual worker PID and process creation time. The sampler verifies that identity
belongs to its launched process tree, then keeps that worker selected; a Windows
venv launcher or compiler child never substitutes for main-worker RSS.
`phase_peak_rss_bytes` measures that worker, while
`phase_tree_peak_rss_bytes` aggregates launcher, worker and descendants separately.
Unverified or reused identities remain explicit, rather than being inferred from
a boundary RSS value. These are actual sampled RSS values; brief peaks may be
missed and shared pages can be counted twice. They are not GPU memory, USS, or
separate weight/KV/activation attribution. The NumPy reference loader is the
default (`prepare_native_kernels=False`); `--native` opts into the normal native
provider. Public Runtime uses its normal CPU backend and is labeled separately.

Timing is opt-in. It reports raw samples, median, min/max range and population
standard deviation for each prompt's elapsed latency, tokens/s, decode tokens/s,
and TTFT (iterator start to first yielded token, including prefill). Iteration
is fully exhausted and every timed output is checked against the numerical
worker. Warmups and repeats create fresh request caches. Ratios and RSS deltas
against the baseline are observations, not automatic speed claims, particularly
for passes with no demonstrated runtime consumer.

Before fixture creation and **each** subprocess, admission checks actual
available RAM against a conservative action/fixture estimate plus 512 MiB of
headroom (`--min-free-mib`). Mid compilation reserves extra space for Python
lists in merging/sub-2-bit passes. A rejected job records `blocked` without
starting a worker. The parent also stops its worker if free headroom is
exhausted. Estimates are admission policy, never measured memory. This check
cannot guarantee allocation success; ordinary allocation errors remain failed
workers with logs. Existing outputs are never deleted or overwritten.

Execution returns nonzero for failed/refused/blocked jobs, repeat differences,
or fixture nondeterminism. Explicit unsupported skips are report data. Only
explicitly missing optional Torch/Transformers for the HF oracle can skip for
dependency absence; unexpected import failures in any worker are failed. Inspect
`report.json`, each job/config/admission and the worker logs; `complete` means
the requested audit finished, not that all passes work. The harness does not
create Git commits.

Focused tests (run only after coordinating with other test/benchmark work):

```powershell
& $python -B -m pytest tests/tools/test_pass_audit.py -q -o addopts='' -p no:cacheprovider
# Optional real compiler -> loader -> public Runtime smoke, still no timings:
$env:AETHER_RUN_PASS_AUDIT_INTEGRATION = '1'
& $python -B -m pytest tests/tools/test_pass_audit.py -q -o addopts='' -p no:cacheprovider
```

The real smoke uses an explicit 128 MiB diagnostic reserve and keeps admission
checks active. It tests compilation, loading, public generation and raw repeated
artifact equality without timing. A memory-blocked smoke fails with its report;
it is not treated as a passing execution check.

## Default-policy comparison

`audit_default_passes.py` compares actual production defaults against explicit
P1/P2/P4/P5/P6/P7/P8 removal. After that policy is adopted, those profiles are
intentionally identical. Add `--historical-baseline` to instead compare the
former seven-on flags with current production defaults. Both controller and
workers use this checkout's source; recorded pass labels must match the actual
config or compilation is refused. It does not change production defaults.

```powershell
# Serial only; leave the default 512 MiB free headroom on a suitable host.
& $python -B tools/audit_default_passes.py --historical-baseline --fixtures tiny moe --hf-reference --output build/pass-audit/default-policy
```

This comparison uses native CPU execution, three fresh cold-cache builds,
two warmups and seven repeats, eight generated IDs per prompt. It records
compile/integrity time, actual config, source hashes, phase RSS, every-file
diffs and numerical/token comparisons. Context 32768 is declared in the source;
the prompts remain 4/64 IDs, not a long-context performance test. The tiny HF
oracle is original FP32; MoE HF is explicitly unsupported. Mid can be selected,
but blocked jobs remain unsuccessful rather than inferred from tiny results.

Small HF workers now use a 512 MiB overhead estimate based on retained actual
small-oracle process-tree RSS below 300 MiB; mid retains 1 GiB. Independent
live free-memory checks and the configurable headroom remain active. These
estimates are admission policy, not measured component memory or guarantees.
See the [executed policy report](../docs/mission-default-policy.md), including
mixed TTFT observations and the limits of the small synthetic speed results.

## Quality-validator comparison

After an executed default-policy report contains original-FP32 HF and candidate
evidence, compare an explicit Git revision of the validator with current source:

```powershell
& $python -B tools/audit_quality_validator.py --report build/pass-audit/default-policy/report.json --baseline-revision 3ebcb26 --output build/pass-audit/quality-validator
```

This reuses hashed archived HF outputs, executes the actual NumPy AEG for cache
history checks, and records fixed-label metrics, fresh-prefix greedy-oracle
agreement, actual cache-array bytes, serial timing spread and worker/tree RSS.
It does not rerun HF or certify compiler quality. Baseline/current order
alternates across three blocks. Missing/changed evidence and blocked workers
remain unsuccessful. Existing outputs are not overwritten. The fixed tiny
quality action uses a 256 MiB overhead estimate, based on recorded tree RSS
below 64 MiB; the independent configurable headroom remains active. The default
headroom is 512 MiB. The executed constrained-host diagnostic used 128 MiB and
is labeled accordingly. See [quality validation](../docs/mission-quality-validation.md)
for the measured improvements, timing regressions and remaining production gate.
