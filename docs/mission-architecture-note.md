# Architecture note before mission changes

Source revision: `7bed38c` (2026-10-15). This note records code inspection,
not a claim of measured performance or completed hardware validation.

`Compiler.compile` detects the model and runs `_stage1_ingest`,
`_stage2_optimize`, `_stage3_target`, and `_stage4_package`. Ingestion attaches
checkpoint tensors to an `AEGGraph`; optimization can alter tensors, IR,
metadata, and files under a temporary staging directory. Targeting selects
profiles and execution contracts. Packaging converts the graph to AEG-IR,
writes indexed weights and tokenizer assets, copies staged pass output, and
saves the manifest. `AEGPackage.save/load/verify_integrity` maintain the
graph/manifest/payload integrity chain. The package can be a directory or
archive; changing packaging must preserve readable existing artifacts and
their integrity checks.

`load_engine_from_path` loads the package; `load_engine_from_package` verifies
integrity, reads the architecture and indexed tensors, validates dimensions,
and builds family-specific weight containers. Dense models become
`CPUExecutionEngine` instances. Backend adapters can wrap those weights in
`TorchAEGEngine`; the ONNX adapter has a separate semantic export contract.
Public `Runtime.generate` resolves and loads a compatible backend, builds a
generation request, and dispatches through the backend to the engine.
CPU `forward` iterates `weights.layers`; `generate_iter` prefills and repeatedly
samples actual logits and advances a KV cache. Torch similarly owns its forward
and generation loops. Neither is an AEG-IR interpreter. Explicit runtime plans
(for example semantic/cross-layer KV on CPU) are separate from graph execution.

**Decision B:** optimize measured engine/kernel paths and treat graph-only
rewrites as inspection/planning until a real consumer and measured benefit are
proved. A fused IR opcode alone establishes neither fused execution nor saved
launches. Do not replace the existing execution architecture without evidence
that doing so improves correctness, speed, memory, or quality.

Memory requires fresh measurement: the current dense loader calls
`_dequantized_by_key` and records quantization provenance separately from its
decoded FP32 projections. A mapped artifact is not evidence that inference
avoids resident decoded copies. Benchmark weights, KV, activations and overhead
separately, and record precision, host, versions, warmup and spread.

Read first: `README.md`, `docs/architecture.md`, `docs/aeg-format.md`,
`docs/optimizer-passes.md`, `IMPLEMENTATION_STATUS.md`, and
`hardware_validation_matrix.json`. Some documentation describes older pass
names or stronger contracts than the code. Supplied audit counts and verdicts
are hypotheses to reproduce on this revision. Historical DirectML execution
records are scoped evidence, not current execution or production certification.

Initial trace: `compiler/compiler.py` (`compile`, `_stage1_ingest`,
`_stage2_optimize`, `_stage3_target`, `_stage4_package`),
`compiler/stage1_ingestion/ingestion.py` (`ingest`),
`compiler/stage2_optimizer/optimizer.py` (`OptimizerPipeline.run`),
`core/aeg_format.py` (`AEGPackage`), `runtime/aeg_loader.py`,
`runtime/runtime.py` (`generate`), `runtime/cpu_engine.py` (`forward`,
`generate_iter`), and `runtime/torch_engine.py`.
