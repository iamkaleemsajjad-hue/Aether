# Aether Runtime implementation status

Reviewed 2026-10-20. The runtime/compiler mission is in progress.

Supported CPU AEG compilation, integrity verification, reload and generation have
executed on an Intel Core i5-1334U running Windows. The last complete green
collection passed 5,214 tests with 41 skips in two serial processes, with Torch
and Transformers installed; line coverage was 78.6457873%. The numerical-admission
area's complete verification is recorded in [its report](docs/mission-numerical-admission.md).
Independent stdlib-only local HTTP POSTs reproduce altered response framing
affecting two MCP tests. The latest serial product/tools runs returned 5,264
passed, 41 skipped and four failures (two HTTP framing cases and two RAM-refused
audit workers), retaining all prior cases and adding 54. Line coverage increased
to 78.6648254%. Full-suite-green remains open. These are
functional checks, not production or
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
| Model families | [The graded family registry](SUPPORTED_MODELS.md) distinguishes parity checks, execution-only coverage, known divergence and refused families. Mamba/Mamba-2/RWKV-7/Jamba require numerical correction. |
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
