# Aether Runtime implementation status

Reviewed 2026-10-24. The runtime/compiler mission is in progress.

The Mamba-2 activation profiling follow-up replaces masked piecewise sigmoid
evaluation with the equivalent stable log-sigmoid identity. A randomized
interleaved local comparison favored it; all 112 focused checks pass with no
new module lint/type diagnostics. Its complete local run returned **5,470
passed, 42 skipped, five failures**: the two existing HTTP cases, RAM admission
and two small exact-equality differences in unchanged Torch paths. The latter
three passed separately, without relaxing tests; their full-run failures remain
recorded. Local coverage was **78.7886185%**. Hosted validation and the coverage
non-reduction criterion remain open. See the
[profile, raw series and limits](docs/mission-mamba2-stability.md#profiling-follow-up-stable-log-sigmoid).

The [Mamba-2 finite-input correction](docs/mission-mamba2-stability.md) removes
CPU SiLU/softplus clipping that changed valid source equations. Five final
regressions fail on the exact prior runtime; focused validation passes 112
cases, including original-HF and independent matrix controls. The complete
local run returned **5,473 passed, 42 skipped, two existing MCP HTTP failures**,
with **78.8040446%** line coverage. All 14 new cases passed; the local suite
remains red. A strict local subprocess comparison was refused for live RAM,
so no speed or resident-memory improvement is assigned to that attempt.
[Hosted CI at `5dbe26d`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39903522524)
passed all 23 jobs; security passed separately. Its complete suite returned
**5,466 passed, 51 skipped, zero failures**. All six Mamba-2 contracts and nine
Mamba-1 controls passed original-HF/CPU/Torch CPU admission and actual API/CLI
checks. The three stress contracts recovered **16/16 greedy IDs per executor**
from **2–4/16**. Logical weight/cache bytes remain unchanged. Decode medians
were **0.29–5.61% lower**; reducing stable-activation overhead remains active
work. Hosted coverage was **78.7848675%**, slightly below the previous hosted
run despite more covered Mamba-2 lines. The linked record retains losses,
spreads and the open coverage criterion.

The [normalization-epsilon correction](docs/mission-normalization-epsilon.md)
preserves explicit zero in the shared loader. Its final regression failed 12
cases against the unchanged prior runtime; focused validation now passes 102
cases with two optional DirectML skips, including actual Llama CPU/Torch/ONNX
and Mamba-2 CPU/Torch comparisons against original HF. The first strict local
benchmark refused a Torch worker for insufficient live RAM; no benchmark
success or memory improvement is inferred from the focused checks. Its complete
local run returned **5459 passed, 42 skipped, two existing MCP HTTP failures**;
all 31 new cases passed and line coverage rose to **78.7816445%**. The local
suite remains red. [Hosted CI at `51c0394`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39874203000)
passed all 23 jobs; security passed separately. Its complete CPU suite returned
**5452 passed, 51 skipped, zero failures**, with **78.7921054%** line coverage.
Three Mamba-2 source contracts and all nine Mamba-1 controls passed original-HF,
CPU/Torch CPU, public API/streaming and CLI checks. The zero-epsilon correction
improves numerical fidelity with unchanged logical weight/cache bytes; observed
tiny-model decode medians were slightly lower and first-output latency higher.
The linked report retains raw
spreads and limits; broader mission work remains open.

The previous measured [CI at `ad77e08`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39747608124)
passed all 23 jobs and security passed separately. Nine untrained FP32 Mamba-1
contracts passed original-HF/CPU/Torch CPU admission, public generation/streaming
and CLI checks; both Mamba-2 controls passed too. The complete hosted suite
returned **5421 passed, 51 skipped, zero failures**, with **78.7439192%** line
coverage. The complete local run returned **5427 passed, 42 skipped, three
failures** (RAM admission and two HTTP-framing cases), with **78.7264834%** line
coverage. [Measurements, losses and limits](docs/mission-mamba1-fidelity.md) retain
all runs. These synthetic checks do not promote trained-family or accelerator
grades; full-package style/type checks remain advisory.

Supported CPU AEG compilation, integrity verification, reload and generation have
executed on an Intel Core i5-1334U running Windows. The last complete green
local run at `edc7d8f` passed **5,320 tests with 42 skips**, with Torch and
Transformers installed; line coverage was **78.7295039%**. A separate explicit
pass-audit integration run passed one case; it is not included in that full-run
count. The [convolution correction report](docs/mission-mamba-convolution.md)
records measured memory/numerical results and mixed timing observations.
[CI at that exact revision](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39437929183)
passed all 22 jobs. Hosted core tests do not generally install Torch/HF, and
repository-wide style/type checks remain advisory. The two formerly failing
local MCP HTTP cases passed without a product change; this does not establish
an environment fix. Mamba-2's distinct learned normalization and output bias
now have independent matrix and real CPU/Torch CPU regressions; its new strict
original-HF CI comparison passed on two untrained FP32 source fixtures, including
both public CPU backends and the CLI. All 23
[CI jobs at `1c6b718`](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39546525959)
passed. The complete local run returned 5,330 passed, 42 skipped and three
failures (the two local MCP HTTP cases and an unchanged timing guard); a focused
followup passed the timing guard but retained the HTTP failures. Raw local line
coverage was 78.6650129%, so full-local-green and the coverage criterion remain
open. See
[scope and execution evidence](docs/mission-mamba2-normalization.md).
The expanded hosted Linux job at `2bb08d2` subsequently passed the complete
suite with Torch, Transformers, server/format extras and CPU ONNX Runtime:
**5,324 passed, 51 skipped, zero failures**, with **78.6772413%** line coverage.
[All 23 CI jobs passed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39554606771).
That job now enforces a measured 78.6% line-coverage floor; differing platform
and optional-dependency skips remain explicit in the linked record.
These are functional checks, not production or
throughput certification. Exact environment, evidence, known failures and the
22-pass matrix are in [the validation record](docs/mission-validation.md).

| Area | Current evidence and remaining work |
|---|---|
| Portable AEG | Compile/save/load/generate and graph/manifest integrity checks execute on CPU. Compatibility depends on model operators and the selected backend contract. |
| Engine execution | CPU and Torch own handwritten forward passes. Graph fusion alone does not change their schedules; see [the architecture note](docs/mission-architecture-note.md). |
| Precision | Stored tensors can be transformed. The dense loader reconstructs FP32 projections; mmap and small artifact size do not prove quantized execution or low resident weight memory. |
| Passes 1–22 | Only P3 remains enabled by default; other passes require explicit selection. The [policy report](docs/mission-default-policy.md) records all 22 dispositions, actual consumers and unresolved checks. Full benefit, composition, idempotency and functional verification remain in progress. |
| Compiler failure admission | Reported optimizer failures now reject compilation before targeting or destination mutation. Original failure details survive; existing artifacts and diagnostic pipeline reports are preserved. This is separate from mandatory numerical quality gating. |
| Legacy tree speculation | Fabricated candidates removed; unsupported drafting fails closed. Manual verification requires target evidence. [Migration and tests](docs/legacy-speculation.md). |
| Executable speculation | Explicit draft/target execution and cache tests pass. Medusa/self-speculation require separate review; no blanket distribution or acceleration guarantee is claimed. |
| Quality gate | Default compilation measures original FP32 HF against actual CPU and Torch CPU AEG execution before publication. Loader/backend/cache admission requires bound evidence; diagnostics waive absence only. Real tiny public API/CLI execution passed. ONNX/accelerator checks, trained-model/corpus quality, transformed references and runtime adaptation remain unfinished; see [scope and costs](docs/mission-numerical-admission.md). |
| Model families | [The graded family registry](SUPPORTED_MODELS.md) distinguishes parity checks, execution-only coverage, known divergence and refused families. Nine untrained Mamba-1 contracts pass original-HF/CPU/Torch CPU, API and CLI checks; its trained-family grade remains unchanged. See [measurements](docs/mission-mamba1-fidelity.md). Trained Mamba/Mamba-2, RWKV-7 and Jamba still need broader validation/correction. |
| KV and batching | Real storage and runtime tests exist. Bounded memory, eviction, copy-on-write and long-running serving require per-path measurement; a pool implementation alone does not prove end-to-end behavior. |
| Hardware | [The maintained matrix](hardware_validation_matrix.json) separates recorded execution, compilation checks and incomplete integration. Historical Intel UHD DirectML evidence is scoped and dated; other accelerator paths need their own execution records. |
| Static checks | Critical Ruff passes at baseline. Full Ruff and full-package mypy fail (5,341 source lint findings and 1,796 type errors with the recorded tools). Narrow passing checks do not clear those errors. |
| Performance | Tiny/MoE synthetic default-policy A/Bs restored native decode dispatch with higher observed decode medians and exact A/B logits/tokens. TTFT results were mixed; no general latency or resident-memory improvement is certified. Trained/larger-model measurements remain open. |

To contribute execution evidence, use `aether hardware detect --json` followed by
`aether hardware validate TARGET --execute --json` with the matching backend
installed. Report actual placement and numerical parity; detection, emulation,
compile-only checks and production validation are different claims.

The previous September status report's blanket completion, expected speedup and
algorithm guarantees have been superseded by this evidence-based record. Its
history remains in Git.
