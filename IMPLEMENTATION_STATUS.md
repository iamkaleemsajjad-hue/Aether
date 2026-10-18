# Aether Runtime implementation status

Reviewed 2026-10-18. The runtime/compiler mission is in progress.

Supported CPU AEG compilation, integrity verification, reload and generation have
executed on an Intel Core i5-1334U running Windows. The clean baseline passed
4,923 tests with 41 skips; after the speculation and package corrections, the
complete collection passed 5,059 tests with 41 skips in two serial processes.
Torch and Transformers were installed.
Line coverage was 78.26%. These are functional checks, not production or
throughput certification. Exact environment, evidence, known failures and the
22-pass matrix are in [the validation record](docs/mission-validation.md).

| Area | Current evidence and remaining work |
|---|---|
| Portable AEG | Compile/save/load/generate and graph/manifest integrity checks execute on CPU. Compatibility depends on model operators and the selected backend contract. |
| Engine execution | CPU and Torch own handwritten forward passes. Graph fusion alone does not change their schedules; see [the architecture note](docs/mission-architecture-note.md). |
| Precision | Stored tensors can be transformed. The dense loader reconstructs FP32 projections; mmap and small artifact size do not prove quantized execution or low resident weight memory. |
| Passes 1–22 | A repeatable audit harness and real synthetic fixtures exist. Full benefit, composition, idempotency and functional verification remain in progress. No blanket “validated” verdict applies. |
| Compiler failure admission | Reported optimizer failures now reject compilation before targeting or destination mutation. Original failure details survive; existing artifacts and diagnostic pipeline reports are preserved. This is separate from mandatory numerical quality gating. |
| Legacy tree speculation | Fabricated candidates removed; unsupported drafting fails closed. Manual verification requires target evidence. [Migration and tests](docs/legacy-speculation.md). |
| Executable speculation | Explicit draft/target execution and cache tests pass. Medusa/self-speculation require separate review; no blanket distribution or acceleration guarantee is claimed. |
| Quality gate | Validator input/protocol defects and missing compiler integration are reproduced. Automatic quality certification is unfinished. |
| Model families | [The graded family registry](SUPPORTED_MODELS.md) distinguishes parity checks, execution-only coverage, known divergence and refused families. Mamba/Mamba-2/RWKV-7/Jamba require numerical correction. |
| KV and batching | Real storage and runtime tests exist. Bounded memory, eviction, copy-on-write and long-running serving require per-path measurement; a pool implementation alone does not prove end-to-end behavior. |
| Hardware | [The maintained matrix](hardware_validation_matrix.json) separates recorded execution, compilation checks and incomplete integration. Historical Intel UHD DirectML evidence is scoped and dated; other accelerator paths need their own execution records. |
| Static checks | Critical Ruff passes at baseline. Full Ruff and full-package mypy fail (5,341 source lint findings and 1,796 type errors with the recorded tools). Narrow passing checks do not clear those errors. |
| Performance | Recorded benchmarks retain their model, backend and host scope. No decode speedup or memory improvement has yet been established by this mission. |

To contribute execution evidence, use `aether hardware detect --json` followed by
`aether hardware validate TARGET --execute --json` with the matching backend
installed. Report actual placement and numerical parity; detection, emulation,
compile-only checks and production validation are different claims.

The previous September status report's blanket completion, expected speedup and
algorithm guarantees have been superseded by this evidence-based record. Its
history remains in Git.
