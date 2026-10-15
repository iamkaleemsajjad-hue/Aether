# Hardware reporting feedback audit

This audit uses the local source tree, not the published PyPI package. Aether is
a model compiler and inference runtime: supported graphs and integrity-checked
weights are packaged as an AEG, with explicit executable and portable backend
contracts. A hardware profile is a compilation description, not proof that a
physical device or working inference backend exists.

## Findings and fixes

| Feedback | Finding | Result |
|---|---|---|
| RTX 4070 detected but not execution-tested | Compute capability 8.9 is consistent with Ada. Detection without execution evidence is expected; NVML-only availability and unchecked driver queries were bugs. | CUDA Driver API initialization, enumeration, device attributes and return codes are checked. NVML supplies telemetry independently. Driver release and supported CUDA API version are separate fields. |
| DirectML says ready after finding DLLs | DLL presence is insufficient. The earlier DLL-only success was already removed in current source, but registered-provider success and guessed precision/device identity remained misleading. | Provider/SDK presence stays separate from device readiness. An explicit ONNX DirectML adapter uses sequential execution, disables memory patterns and CPU fallback, and validates actual node attribution after a real workload. |
| No C++ compiler warning | A missing compiler is an expected optional dependency; incomplete Windows toolchain discovery and unsafe native-library reuse were real defects. | MSVC/clang-cl SDK environments, GCC/MinGW and Clang are checked. Errors remain visible. NumPy fallback is reported and actually tested. Libraries carry platform/ISA/build identities; Windows loads verified cache copies so loaded artifacts remain removable. |
| AVX-512 implies FP16/BF16 and CPU execution success | Genuine reporting bugs. CPU family/model names and AVX-512 foundation do not establish these separate extensions. Detection had set execution success without running inference. | SIMD comes from process-usable NumPy feature evidence, including OS register-state gating. FP16/BF16 require separate confirmed extensions. Detection sets no execution/production success. Storage conversion and FP32 accumulation are described separately. |
| Inconsistent capabilities | Genuine defects: guessed memory/precision/architecture, monitoring tools treated as compute runtimes, invalid wrapped JSON, bare command shadowing, failed JSON validation exiting zero. | Actual device identities and measurements, explicit unknown values/sources, checked HIP enumeration, queried OpenVINO capabilities, real Metal device probing, consistent CLI/status fields and failure exit codes. |
| Planning backends report inference readiness | Additional confirmed defects: SDK directories, package imports and engine metadata could mark incomplete adapters ready. | QNN, RISC-V NPU, OpenVINO generation, TensorRT-LLM and FPGA adapters reject inference readiness until execution is integrated. Existing graph compilation, IR emission and planning code remain available. |
| CPU AEG chooses CUDA on a GPU host | Genuine artifact-selection bug, fixed in the preceding feedback work and checked again here. Additional AMD target alias gaps were found. | Selection intersects emitted/portable AEG contracts, compatible host targets and ready backends. An authorized usable accelerator is preferred; compatible CPU execution remains available when GPU contracts or dependencies are absent. Canonical AMD HIP aliases work throughout artifact compatibility and explicit device selection. |

CUDA/ROCm/Metal adapters also used nonexistent delegate load/unload APIs. They
now preserve model IDs, call the real APIs, require matching frameworks, and
force the selected device. A CPU-only PyTorch wheel cannot satisfy CUDA or HIP
readiness. CUDA binary compatibility is checked within the same major compute
capability; a newer numeric architecture alone is insufficient. Ada FP8 and
Hopper's lack of INT4 Tensor Core inputs are handled separately.

Host-native CPU libraries require a compatible platform, conservative CPU
signature and OS-usable required instruction features. Missing, legacy or
incompatible identities cause recompilation or an explicit NumPy fallback
before loading. A compiler is optional because compatible packaged kernels or
the tested NumPy reference path can execute CPU artifacts.
When a concrete CPU identity cannot be established, compilation uses an
explicit architecture baseline and individually confirmed SIMD extensions
instead of host specialization. Disabled FP16/BF16 extensions are also false
in the detailed precision evidence, rather than being confirmed by key presence.

## Field meanings

| Field | Meaning |
|---|---|
| `hardware_present` | Device evidence; `null` if not confirmed. SDKs or DLLs alone do not confirm a device. |
| `runtime_available` | Vendor compute runtime can expose a device; `null` when only registration/SDK evidence exists. |
| `implemented` | An inference adapter exists for the stated scope. Compiler emitters and planning placeholders are separate. |
| `available` | Hardware plus usable runtime evidence. It does not imply that optional Aether execution dependencies are installed. |
| `backend_dependencies` / `backend_available` | Dependency presence versus checked execution adapter readiness. `null` means not checked/confirmed. |
| `execution_tested` / `execution_scope` | Successful real workload in this validation result, with a stated scope. Detection remains untested. |
| `production_validated` | Requires separate physical, multi-model, multi-workload evidence. This audit does not set it. |
| `memory_bytes` | Queried capacity. Zero is accompanied by `memory_bytes_known=false` if unknown; fingerprint JSON uses `null` for unknown GiB. `memory_kind` and `memory_source` explain shared/host/device capacity. |
| `driver_version` | Queried vendor driver release, or `unknown`. CUDA API compatibility version is a separate field, not a fabricated driver release. |
| `supports_*` | Confirmed device arithmetic/runtime format capability. False includes unconfirmed capability; evidence and Aether conversion/reference modes are separate. |

The checked-in hardware matrix is now explicitly a maintained catalog with
historical evidence, rather than a live host snapshot. It no longer calls the
CPU path production-ready while `production_validated` is false.

## Modified implementation and regression files

- Detection/evidence: `src/aether/backends/capabilities.py`,
  `hardware_detector.py`, new `cuda_detection.py`, `auxiliary_detection.py`,
  and `hardware_validation.py`.
- Reporting: `src/aether/cli.py`, `src/aether/runtime/hardware.py`.
- Execution adapters: `src/aether/backends/hardware_backends.py`,
  `native_cpu_backend.py`, `native_cuda_backend.py`, `torch_backend.py`,
  `onnx_backend.py`, and `trtllm_backend.py`.
- Native portability: `src/aether/kernels/native_cpu.py`, `native_cuda.py`,
  `src/aether/runtime/aeg_loader.py`, `src/aether/compiler/compiler.py`.
- Target contracts: `src/aether/core/types.py`, `constants.py`, `aeg_format.py`,
  `src/aether/compiler/stage3_targeting/hardware_profile.py`, `backend_selector.py`,
  `src/aether/runtime/runtime.py`.
- Regression tests: `tests/unit/test_cpu_host_portability.py`,
  `test_hardware_evidence.py`, `test_hardware_reporting_feedback.py`,
  `test_vendor_detection_evidence.py`, `test_vendor_backend_readiness.py`,
  `test_onnx_provider_contract.py`, `test_planning_backend_readiness.py`,
  `test_batched_inference.py`, `test_all_new_modules.py`,
  `test_runtime_artifact_selection.py`, `test_hardware_backends_complete.py`,
  `test_onnx_backend.py`, `tests/hardware/test_hardware_contract.py`, and
  `tests/integration/test_local_safetensors_aeg_roundtrip.py`.
- Packaging/docs: `pyproject.toml` (separate DirectML extra),
  `hardware_validation_matrix.json`, `README.md`, `docs/runtime.md`,
  `docs/api-reference.md`, and this report.

Pre-existing and earlier-feedback changes were preserved; their separate audit
is [user-feedback-validation.md](user-feedback-validation.md).

## Verification

The verification host is Windows, Intel Family 6 Model 186, approximately
7.69 GiB physical RAM and GCC/MinGW. The earlier detector omitted the physical
**Intel UHD Graphics** because no corresponding inference SDK was installed;
the continuation audit corrected that inventory defect. No NVIDIA CUDA, AMD HIP
or Apple Metal device is present. Python 3.10 has
NumPy 1.26.4, CPU-only PyTorch and ONNX Runtime. A separate Python 3.12
environment has the core package and `[test]` extra without PyTorch/ONNX extras.

Focused results obtained before the final full-suite run:

- CPU portability/runtime/selection: 125 passed.
- Windows packaged DLL cache-copy: 31 passed, including real native inference
  after deleting the source artifact directory.
- Reporting/vendor/selection final bundle: 92 passed.
- Vendor/DirectML readiness contracts: 168 passed, 4 skipped in Python 3.12.
- Real ONNX CPU execution and profiling: 24 passed, 1 physical DirectML skip.
- Initial root evidence tests: 26 passed, 1 physical CUDA skip.
- Final CPU/evidence/native-kernel bundle: 130 passed, 1 physical CUDA skip;
  a final-source focused check passed all 20 precision/identity cases, including
  real explicit-ISA native execution with no concrete CPU identity.
- Final planning adapter regressions and related modules: 246 passed,
  including 15 new regressions.
- Host-weight release: 56 batching tests passed. Eight additional real CPU
  comparisons retained exact logits and tokens after releasing host weights.
  The corrected test compares the same live engine before and after release;
  independent calibrated FP16 engines can select different valid kernels and
  differ by one rounding unit.
- Actual CPU CLI `--execute`: exit 0; native prefill/decode errors against the
  independent NumPy reference were `2.086162567138672e-7` and
  `1.1920928955078125e-7`. KV length progressed from 3 to 4.
- Actual CUDA CLI `--execute`: exit 1; execution false, because CUDA hardware
  is absent. Environment-only CPU validation exits 0 with execution false.
- Bare hardware, detection, capabilities, and backend inventory JSON parsed
  successfully. Syntax/undefined-name checks and `git diff --check` passed.

Earlier feedback-phase full-suite verification (before the continuation changes):

| Environment | Actual result | Evidence |
|---|---|---|
| Windows Python 3.10.11, NumPy 1.26.4, CPU PyTorch/Transformers/ONNX Runtime installed | **4,118 passed, 21 skipped; zero failures/errors; process exit 0** | [Log](../build/hardware-audit/final310/suite-source-final.log), [JUnit XML](../build/hardware-audit/final310/suite-source-final.xml) |
| Clean Windows Python 3.12.15, NumPy 2.5.4, editable `.[test]`, no PyTorch/ONNX extras | **3,589 passed, 314 skipped; zero failures/errors; process exit 0** | [Log](../build/hardware-audit/final312/suite-source-final.log), [JUnit XML](../build/hardware-audit/final312/suite-source-final.xml) |

The skips cover missing optional frameworks, disabled network tests and absent
physical devices. Deprecation warnings remain; passing tests do not mean that
the code emits no warnings. The runs recorded three warnings on Python 3.10
and 78 on Python 3.12. The final CLI checks also exited successfully, apart from
the intentionally failed absent-CUDA validation; [CLI results](../build/hardware-audit/reporting/cli-final.log)
and [real CPU workload](../build/hardware-audit/reporting/cpu-execution-verified.json)
retain the measured evidence. Clean Python 3.12 `pip check`, source-wide critical
Ruff checks, JSON catalog parsing and `git diff --check` passed.

Logs and JUnit XML live under `build/hardware-audit/` and
`build/hardware-validation/`; these generated files are not committed. Windows
sandbox temporary-directory ACLs prevented native/subprocess checks, so these
verification runs used approved execution outside that sandbox with workspace
caches and temporary directories.

Reproduction commands from the repository root:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q --no-cov
python -m aether hardware --json
python -m aether hardware detect --json --save host-hardware.json
python -m aether hardware capabilities --json
python -m aether hardware validate cpu_generic --execute --json
python -m aether hardware validate cuda_sm89 --backend pytorch --device-index 0 --execute --json
```

The last command requires a physical compatible GPU and a CUDA-enabled
PyTorch installation in the active environment. DirectML uses a separate
Windows environment installed from the same source with `.[directml]`, then:

```powershell
python -m aether hardware validate directml --device-index 0 --execute --json
```

## Remaining limits

The final continuation suites passed **4,379 tests / 22 skips** on Python 3.10
and **3,848 tests / 317 skips** on clean Python 3.12, with zero failures/errors
and exit 0. The isolated DirectML bundle passed **28 tests**, including real
generation and streaming; its local fault-handler workaround is documented in
the [completion audit](cli-hardware-completion-audit.md).

RTX 3050/4070, ROCm, Metal and OpenVINO GPU/NPU physical execution could
not be measured on this host. The continuation separately measured real Intel
UHD DirectML FP32 ONNX execution and Runtime generation with CPU fallback
disabled; see [completion audit](cli-hardware-completion-audit.md). Deterministic driver/provider mocks test error
handling and evidence rules; they are never recorded as physical execution.
MSVC/clang-cl discovery is covered by tests, but neither usable toolchain was
installed locally. Native CUDA retains its development/JIT code but its
memory/operator interface is incomplete and cannot perform full inference;
the CUDA AEG path requires a matching PyTorch backend. DirectML has raw ONNX
execution, not an AEG export pipeline. OpenVINO task-specific generation,
QNN, ANE, RISC-V NPU, TensorRT-LLM and FPGA Aether inference adapters remain
incomplete and report that status. No new performance improvement was measured.
No universal hardware, speed, quality or production certification is claimed.

## Primary research and specifications

The fixes apply the separation of portable semantics and target lowering
described in [TVM](https://www.usenix.org/system/files/osdi18-chen.pdf) and
[MLIR](https://arxiv.org/pdf/2002.11054). This is an engineering application of
those principles, not a claim that either paper proves Aether's implementation.

CUDA version, binary compatibility and format decisions follow the
[Driver API version contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html),
[Ada compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/),
and [compute capability/type tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
CPU gating follows [NumPy's runtime CPU feature implementation](https://github.com/numpy/numpy/blob/v2.4.6/numpy/_core/src/common/npy_cpu_features.c)
and the separate [Intel AVX-512 FP16 instruction specification](https://cdrdv2-public.intel.com/678970/intel-avx512-fp16.pdf).
Windows toolchain discovery follows [Microsoft vswhere](https://github.com/microsoft/vswhere/wiki/Find-VC)
and [Clang-cl documentation](https://clang.llvm.org/docs/UsersManual.html#clang-cl).
DirectML execution constraints follow [ONNX Runtime's provider documentation](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).
