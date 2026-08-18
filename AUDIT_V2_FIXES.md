# Aether Runtime — Audit v2 Implementation Record

This file records changes made after the audit. The original `audit v2.md` was
not edited.

## Scope and integrity

- Requirements source used: `PRD.md`, `PRD_v2.md`, and `audit v2.md` only,
  plus the source and test tree.
- Preserved audit SHA-256:
  `9DD5F39F8DEBB6114C588D7B8DE0AE22D62833321A983BE0B104118F2FB3E5E8`
- No claim of universal production readiness is made for hardware that is not
  present in this Windows CPU environment.

## Implemented fixes

### Encoder ingestion and AEG execution

Fixed the real SafeTensors ingestion path for BERT-family encoder models:

- added encoder-specific graph construction for bidirectional attention,
  LayerNorm, GELU, positional/token-type embeddings, and pooling;
- corrected BERT/RoBERTa checkpoint aliases and graph-node aliases;
- made weight selection deterministic when a checkpoint has both `.weight` and
  `.bias` tensors under one normalized key;
- separated Q/K/V matrices from one-dimensional biases, including GQA-safe
  projection handling;
- persisted encoder tensors and parameter biases in the AEG quantized store;
- added encoder-specific weight-accounting invariants so decoder-only required
  tensors are not incorrectly demanded from an encoder artifact;
- added `EncoderExecutionEngine`, with real NumPy LayerNorm, bidirectional
  attention, GELU, residual blocks, pooler, mask validation, and finite output;
- made AEG loading dispatch by the manifest's `is_encoder` flag;
- connected packaged encoder AEGs to the PyTorch backend's `embed` and
  `rerank` methods, including tokenizers without a padding token.

Files:

- `src/aether/compiler/stage1_ingestion/ingestion.py`
- `src/aether/compiler/weight_quantizer.py`
- `src/aether/compiler/compiler.py`
- `src/aether/runtime/encoder_engine.py`
- `src/aether/runtime/aeg_loader.py`
- `src/aether/backends/torch_backend.py`
- `tests/integration/test_encoder_aeg_roundtrip.py`

### AEG-IR and API corrections already present in the working tree

The working tree also contains the earlier audit fixes that were preserved:

- accepted encoder operation types in AEG-IR lowering;
- exposed `AetherClient` and `AetherHub` at the top-level package;
- corrected the explicit CLI `compile`/`list` command registrations;
- improved CLI safety/provenance inspection;
- preserved manifest hashing while loading provenance metadata;
- connected compiler pass artifact staging and green-profile compilation;
- corrected GQA quantizer handling.

## Tests and executable evidence

### New test

`python -m pytest tests/integration/test_encoder_aeg_roundtrip.py -q --disable-warnings`

Result: **1 passed**.

This test creates a real SafeTensors BERT-style checkpoint and a real packaged
tokenizer, compiles it, reloads the AEG in a new loader path, executes pooled
embeddings, and executes the same artifact through `TorchBackend.embed`.

### Regression tests

`python -m pytest tests/integration/test_local_safetensors_aeg_roundtrip.py tests/unit/test_aeg_ir.py tests/unit/test_cli_contracts.py -q --disable-warnings --maxfail=1`

Result: **49 passed**.

`python -m pytest tests/test_passes_v2.py tests/unit/test_runtime_v4_extensions.py tests/unit/test_phase5_observability.py -q --disable-warnings --maxfail=1`

Result: **78 passed**.

Combined focused/regression evidence: **127 passed**.

### CLI and SDK smoke tests

`python -m aether compile --dry-run --target cpu_avx512 Qwen/Qwen3-0.6B`

Result: executed successfully and reported a feasible CPU target plan.

`python -c "from aether import AetherClient, AetherHub; ..."`

Result: both public imports succeeded.

### Real artifact execution path proven locally

For the new tiny encoder fixture:

```text
SafeTensors checkpoint
  -> architecture detection
  -> encoder AEG graph
  -> optimizer pipeline
  -> quantized AEG package
  -> integrity-checked reload
  -> EncoderExecutionEngine
  -> pooled embedding output
  -> TorchBackend public embedding API
```

The output was finite, had the expected `[1, 8]` shape, and was not a
hardcoded zero vector.

## Remaining audit status

This change set materially improves real local CPU functionality, but it does
not honestly make every PRD item 100% complete. The following remain
unverified or incomplete and must stay classified that way:

- NVIDIA/AMD/Apple/Intel NPU/Qualcomm/RISC-V/FPGA/TEE/CXL hardware execution
  was not physically available here;
- CUDA, ROCm, Metal, QNN, OpenVINO, vendor NPU, FPGA, and confidential-compute
  kernels were not validated on their devices;
- the PRD's benchmark and quality claims were not reproduced against all named
  baselines and real pretrained model families;
- distributed multi-node execution, fleet failover, and tenant isolation were
  not validated on a real cluster;
- external Hub authentication and remote upload/download were not validated;
- the complete REST/gRPC production deployment path was not exercised with a
  live authenticated service and real external clients;
- video/VLM, diffusion drafting, RLVR training, and hardware-backed TEE
  behavior remain environment-dependent or incomplete as documented by the
  audit.

## Honest completion statement

The repository is **not** entitled to a 100% production-ready claim from the
tests above. The local CPU SafeTensors-to-AEG-to-runtime path now has additional
real coverage, including BERT-style embeddings, while the full v3.1/v4/v5 PRD
surface still requires hardware, distributed, benchmark, and external-service
validation.
