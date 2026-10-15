# User feedback: fixes and validation

Validated on Windows on 12 October 2026. All three reported defects were real.
The fixes apply to artifact contracts and execution paths across models; they
contain no SmolLM2 or RTX 3050 special case.

## Project and execution contract

Aether is a model compiler and inference runtime. Its deployment unit is an
Aether Execution Graph (AEG), containing the graph, integrity-checked weights,
tokenizer, precision assignments, and execution contracts. Compilation performs
ingestion, graph optimization, hardware targeting, packaging, and reporting.
The runtime executes that artifact with its own CPU engine or a compatible
optional backend. A host fingerprint describes devices; an artifact describes
what can execute. Both are necessary inputs to dispatch.

The portable graph and weight contract is distinct from a vendor kernel binary.
CPU execution can use compatible native kernels or the NumPy reference path.
Portable CUDA, ROCm, and MPS execution requires the corresponding backend and
vendor runtime. Planning-only targets are not executable implementations.

## Reported defects

| Report | Finding | Fix |
|---|---|---|
| CPU AEG benchmark incorrectly demands CUDA PyTorch on GPU hosts | Confirmed: selection used the host's preferred target before respecting the artifact | Intersect emitted variants, portable contracts, host devices, and available backends; prefer an authorized usable accelerator and otherwise CPU; preserve explicit device/backend constraints |
| Quick Start precision option and Python backend import fail | Confirmed: the precision flag and convenience export were absent | Add validated `compile --precision`, carry it through assignment and weight serialization, export lazy `get_backend`, and replace stale examples with current APIs |
| `.[test]` cannot collect/run the documented suite | Confirmed: required pytest-timeout and FastAPI dependencies were missing | Include pytest-timeout, FastAPI, Uvicorn, and the plotting dependency used by benchmark tests; make genuinely optional-framework tests skip explicitly |

Dispatch records each loaded model's actual selected target. Generation and
benchmark metrics therefore do not label a CPU request as CUDA merely because
the host has a GPU. Tests cover simulated NVIDIA, AMD, Apple, and Intel hosts,
multiple artifact targets, different CPU variants, missing optional backends,
native accelerator plans, and explicit device constraints. An explicitly
requested unavailable accelerator fails instead of silently selecting CPU.

Explicit precision selects the existing uniform layer assignment and codec
path. It is persisted in configuration, precision maps, and weight indexes.
Embeddings and the output head retain BF16. Lossy precision remains an explicit
request requiring model-quality evaluation.

## Related fixes from the audit

- Backend discovery is lazy and synchronized. CPU startup does not initialize
  optional PyTorch, vLLM, or other engines. Concurrent lookup constructs a
  backend once. Public aliases and custom registration remain available.
- PyTorch availability checks physical device support independently of the
  previous model's execution mesh. Automatic device configuration is reset
  when loading another model.
- CLI runtime construction preserves environment backend, device, offline, and
  cache controls. Compilation and `pull` respect the selected cache directory.
- `run --stream` reaches incremental generation. `bench --json` and `--output`
  produce usable JSON. `--compare` executes a requested backend or reads a
  saved baseline and validates its throughput.
- Repeated benchmarking uses backend token counts and first-token timing.
  Reports identify the executing backend/target. Legacy callback estimates
  are labeled, unavailable per-token timing remains unavailable, and failed
  runs return a failure exit code. `info` and `graph` also fail correctly for
  missing artifacts.
- NumPy arrays exposing `.device` are no longer mistaken for PyTorch tensors
  in quantized linear dispatch. This was exposed by the fresh NumPy 2 install.
- MLX-format ingestion handles BF16 through the framework-free SafeTensors
  decoder. Corrupt shards fail instead of silently dropping weights.
- Embedded GGUF Unigram tokenizer export and SafeTensors LoRA adapter loading
  no longer require Transformers or PyTorch. Tokenizer IDs and chat templates
  are preserved; interoperability tests remain available with Transformers.
- DirectML DLL presence is no longer reported as an available executable GPU.
  Availability requires actual backend/device evidence.
- Missing logger/JSON names in native CPU fallback, GGUF export, and cascade
  routing were corrected after checking undefined names across the source.
- The comparison suite can resolve its retained optional engine adapters
  without changing its default engine set.

The primary guides and API reference now use existing configuration fields,
artifact imports, streaming APIs, CLI syntax, and typed gRPC service names.
Their local links and configuration keywords were checked. The CI core matrix
is configured to run the offline suite on Linux, macOS, and Windows with Python
3.10, 3.11, and 3.12; that hosted matrix was not executed from this workspace.

## Research basis

TVM separates computation from target-specific schedules and packages deployable
graphs, operators, and parameters. MLIR preserves semantics across progressive
lowering. The dispatch change applies these principles as an engineering
interpretation: portable graphs, emitted executable variants, physical devices,
and backend availability remain separate contracts. It does not infer a vendor
kernel from a hardware profile or change tensors to make dispatch succeed.
Primary sources: [TVM, OSDI 2018](https://www.usenix.org/system/files/osdi18-chen.pdf)
and [MLIR](https://arxiv.org/pdf/2002.11054).

## Verification

The results below record the original feedback phase. Later hardware and CLI
integration changes and their final-source results are recorded in the
[completion audit](cli-hardware-completion-audit.md).

| Environment/check | Result | Evidence |
|---|---|---|
| Windows, Python 3.10.11, NumPy 1.26.4, installed CPU PyTorch/Transformers/ONNX Runtime; full suite | **3,909 passed, 19 skipped, 0 failures, 0 collection errors** | [Log](../build/feedback-validation/full-validated.log), [JUnit](../build/feedback-validation/full-validated.xml), [environment](../build/feedback-validation/environment310.json) |
| Fresh Windows Python 3.12.15 environment installed with `pip install -e ".[test]"`; no PyTorch, Transformers, ONNX, or ONNX Runtime; full suite with coverage | **3,381 passed, 311 skipped, 0 failures, 0 collection errors** | [Log](../build/feedback-validation/python312-validated.log), [JUnit](../build/feedback-validation/python312-validated.xml), [environment](../build/feedback-validation/environment312.json) |
| Final Python 3.12 CLI, artifact-discovery, framework-free, and dispatch checks after the final inspection exit-code change | **133 passed, 0 failures**; includes all **51** feedback regression cases | [Log](../build/feedback-validation/final-cli.log), [JUnit](../build/feedback-validation/final-cli.xml) |
| Additional real OpenAI HTTP server test on Python 3.12 without optional frameworks | **1 passed**; compiles a local checkpoint, starts the CLI server, and completes a real TCP chat request | [Log](../build/feedback-validation/extra-http.log), [JUnit](../build/feedback-validation/extra-http.xml) |
| Clean Python 3.12 dependency consistency | `pip check`: no broken requirements | Captured installation in [dependency log](../build/feedback-validation/python312-dependencies.log) |
| Source-wide undefined names and syntax-related lint | Ruff `F821,F822,F823,E9`: passed | Checked across `src/aether` and the feedback tests |
| Documentation and patch integrity | Configuration-keyword/local-link audit and `git diff --check`: passed | Primary guides, README, and API reference |

The full-suite runs used two pytest workers and isolated workspace caches and
temporary directories. The Python 3.12 run retained the repository's default
coverage instrumentation. It recorded 62.47% line coverage and 42.69% branch
coverage across the source; see [coverage XML](../build/feedback-validation/python312-coverage.xml).
The existing Python 3.10 installation has older distribution metadata, but
the captured module path and source version verify that it executed this
working tree's Aether 1.3.0 code.

Integration coverage includes real SafeTensors compilation, artifact reload,
generation, quantized weight serialization, incremental output, benchmarks,
and REST serving. The new serving regression uses a small synthetic checkpoint
with real tensors and tokenization, independently of Transformers. Existing
optional-framework round trips and numerical comparisons also ran in the
Python 3.10 environment.

Optional-framework, network, and physical-device tests are skipped for their
actual missing prerequisites. Small fixture tensor sizes preserve the model
layouts under test. Subprocess tests have bounded waits, nonblocking log
capture, and cleanup so slow Windows startup cannot hang the test runner.
NumPy/Click deprecation warnings remain in the current suite; the result is
zero test failures, not a claim that the repository emits no warnings.

To repeat the focused checks after installing `.[test]`:

```bash
python -m pytest tests/unit/test_runtime_artifact_selection.py tests/unit/test_feedback_cli_contracts.py tests/unit/test_framework_free_feedback.py
```

The raw validation artifacts are local files under ignored
`build/feedback-validation/`. Pre-existing MacBook benchmark work was preserved.

## Evidence limits

This machine did not provide an RTX 3050, AMD accelerator, Apple Silicon, NPU,
or FPGA for physical execution. Cross-vendor dispatch was tested with detector
evidence and backend doubles; CPU artifact execution and HTTP serving were
tested directly. The reported 15.7 versus 12.4 tokens/second measurement was
not independently reproduced, and these fixes do not establish a universal
speedup, model-quality guarantee, or production certification of every hardware
profile. Experimental features still require their actual artifact and backend
contracts.
