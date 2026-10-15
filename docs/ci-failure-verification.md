# CI failure investigation — October 15, 2026

The failing [CI run 38684010722](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/38684010722)
tested commit `f49be63677531f69cd7a5eb1d7ca43cd7d41b403`. Its core test matrix
failed on eight of nine OS/Python combinations; Windows/Python 3.10 passed.
The workflow logs were inspected with GitHub CLI, without a browser.

## Confirmed causes and changes

| Cause | Fix | Regression coverage |
| --- | --- | --- |
| Loading a packaged native CPU library changed the compiler's process-wide provider. Linux and macOS retained the library's removable artifact path; a subsequent model removal left later kernel emission and compilation referencing a deleted `.so` or `.dylib`. Windows' existing DLL cache concealed the missing-path symptom. | Artifact loading owns independent mutable provider state. Verified packaged executable bytes use the content-addressed host cache on every OS. Loaded host handles and immutable toolchain settings can be shared; kernel execution adds no filesystem checks or compiler probes. | Storage-policy contracts for DLL/SO/dylib, plus real local compilation, generation, artifact removal, native argmax, kernel emission/loading, subsequent compilation and integrity verification. |
| The generic `metal` target was missing from three enum metadata tables. macOS discovery could select it, then `HardwareTarget.auto()`, `HardwareProfile.auto()` or `aether kernels` raised `KeyError`. | Display/backend metadata use canonical target definitions, generic Metal has vendor metadata, and automatic profiles preserve queried device evidence rather than substituting a catalog GPU. CLI inventory distinguishes profiles from installed execution readiness. | Every enum target, generic Intel/AMD Metal and Apple discovery contracts, one-probe selection, conservative fallback, serialization, CLI output and fingerprint identity. These are discovery contracts, not physical Mac execution tests. |
| The DirectML DLL-only test mocked Windows and ONNX Runtime but read independent OS inventory cached by previous tests. Its `hardware_present is None` assertion depended on the host. Backend unavailability was already reported correctly. | Isolate the external inventory probe in the fixture. Production detection remains unchanged. | Reproduced the failure after real inventory discovery, then covered nine controlled inventory/runtime-provider combinations. Registered providers and DLLs never count as successful execution. |
| Local integrated verification exposed a tiny KV snapshot read requesting the default 2 GiB maximum entry quota. It raised `MemoryError` under laptop memory pressure. The deduplication path also read existing blobs without a size bound. | Validate committed sizes, check the opened file's actual size, then read committed bytes plus a one-byte growth guard. Share this bounded read with deduplication. Keep quotas, checksums, framing and metadata checks intact. | Tiny reads retain the full quota but never request quota-sized allocation; truncated, appended, same-size corrupt, invalid-size, oversized and concurrently growing blobs are rejected. Real session restoration and persistence are exercised separately. |

Files changed: `src/aether/kernels/native_cpu.py`, `src/aether/runtime/aeg_loader.py`,
`src/aether/core/types.py`, `src/aether/compiler/stage3_targeting/hardware_profile.py`,
`src/aether/cli.py`, `src/aether/runtime/kv_storage.py`, their regression tests,
documentation, and `.github/workflows/ci.yml`.

The integrated run also caught an old adversarial test expecting native CPU
last while canonical policy places it first for five CPU targets. The updated
test verifies exact native-first selection without importing optional
frameworks, and separately verifies the ordered AVX2 native fallback.

The next GitHub matrix confirmed those original product failures were gone and
exposed one later CLI workflow assertion on POSIX: an encoded cache basename
started with `--` and the test passed it as an option. The workflow now places
command options before the standard `--` boundary and the literal cache alias
after it. Portable literal-name regressions retain unknown-option rejection;
cache naming and production option parsing remain compatible.

## Workflow corrections

- Python syntax/undefined-name checks and public API stub checks are blocking.
- Full Ruff, formatting and mypy remain explicitly advisory, with actual step
  outcomes in the job summary and downloadable diagnostics. The old green lint
  job suppressed 9,953 Ruff diagnostics, 393 formatting changes and 983 mypy
  errors; this change does not claim that repository-wide backlog is resolved.
- The v3.1 artifact job previously selected one network test that was skipped.
  It now compiles actual local safetensors weights and checks the same artifact
  contract and package integrity. The original network test is retained.
- v3.1 runtime tests and fatal Sphinx build errors are no longer suppressed.
  Documentation warnings remain allowed; the observed original build completed
  with 59 warnings.
- The core nine-job matrix retains its test selection, with offline model
  access, isolated runtime caches, one BLAS/OpenMP thread, a 180-second test
  timeout and a 30-minute job limit.

## Verification

Focused Windows verification after the fixes:

| Check | Python 3.10 | Clean Python 3.12 `[test]` |
| --- | --- | --- |
| Native CPU, artifact removal, emitter, compiler and CLI regression bundle | 280 passed, exit 0 | 278 passed, 2 optional dependency skips, exit 0 |
| Target metadata/profile/CLI bundle | 105 passed, 1 separately tested native case deselected, exit 0 | 105 passed, 1 separately tested native case deselected, exit 0 |
| Inventory-first DirectML reporting bundle | 22 passed, exit 0 | 22 passed, exit 0 |
| Adversarial backend selection, real native execution and target metadata | 67 passed, 1 network skip, exit 0 | 66 passed, 2 optional/network skips, exit 0 |

The DirectML-installed Python 3.12 environment additionally passed 10 reporting
contract tests, exit 0; these tests did not execute a GPU workload. Clean Python
3.12 passed the now-blocking local v3.1 artifact/runtime contracts: 9 passed,
exit 0. A Windows WMI first-chance exception was logged during that run despite
successful completion; the log is retained.

The SWIG shutdown warning in Python 3.10 is retained unsuppressed. A supervising
`subprocess.run` confirmed the actual child exit code is 0; an earlier
PowerShell wrapper status of 1 described native stderr handling, not a failing
Python/test process. Actionlint 1.7.7 validated the changed workflow with exit 0
(workflow syntax and expressions; external shellcheck/pyflakes disabled).

The KV storage/session/lifecycle/Hub bundle passed 121 tests on Python 3.10 and
104 tests with 17 optional Torch skips on clean Python 3.12, both exit 0. This
includes all 12 bounded-read regressions and the actual session restore that
previously raised `MemoryError`.

The complete core selection on Windows/clean Python 3.12 at runtime-fix commit
`7b784f6` passed **4,349 tests**, with 357 optional/environment skips and 19
marker deselections, in 763.32 seconds, native exit 0. The later CLI correction
changes test arguments and documentation, with its own focused and GitHub
matrix checks.

Reproduce the core CI selection after `python -m pip install -e ".[test]"`:

```console
python -m pytest tests -q -m "not slow and not network and not gpu and not cuda and not metal and not rocm" --no-cov --timeout=180 --junitxml=test-results-core.xml
python scripts/generate_stubs.py --check
ruff check src tests examples benchmark benchmarks scripts --select E9,F63,F7,F82
```

Targeted tests: `test_native_library_lifetime.py`,
`test_hardware_target_metadata.py`, `test_framework_free_feedback.py` and
`test_kv_storage_read_bounds.py`, plus
`test_v31_features.py::test_local_compiled_package_contains_v31_artifacts`, under
`tests/unit/`. The broader native bundle includes the CLI audit, kernel emitter,
hardening and native-kernel tests that failed in the original run.

Local evidence, including original GitHub logs, before-change reproductions,
test logs and JUnit XML, is saved under ignored `build/ci-checks/`. Complete
core-suite runs use `build/cif310/`, `build/cif312/` and `build/cif312-final/`.

The updated GitHub matrix provides Linux/macOS and Python 3.11 verification;
local Windows results alone cannot establish those outcomes. Hardware
discovery contracts do not certify physical CUDA, Metal,
DirectML or other accelerator execution, or universal model support. Existing
advisory policies in the separate Security, Integration and Pages deployment
workflows are outside these core CI failure fixes.
