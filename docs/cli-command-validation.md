# CLI command validation

This inventory describes the local repository source reviewed on 2026-10-13.
The Click tree contains **70 registered paths**, including command groups and
the `hw` alias. These are not 70 independent execution engines. All 70 `--help`
paths are exercised by the command-tree audit; working help alone does not
establish a working model, device, network service or experimental feature.

The 2026-10-15 local-source recheck again found exactly 70 registered paths and
all 70 help calls passed. Fresh CPU/inventory commands returned parseable JSON;
actual CPU AEG prefill/decode passed, while absent CUDA returned a nonzero failure
without execution evidence. This recheck does not extend the execution scope of
the individual commands below. Evidence is retained in
`build/runtime-completion/hardware-20261015`.

Commands accept the global cache option before the command:
`aether --cache-dir ./cache COMMAND`. `--json` output is unwrapped JSON on
stdout; check the exit status even when a failure report is parseable. Runtime
backend/device/offline environment controls are described in the
[API reference](api-reference.md).

## Evidence and limits

**CPU workflow** means a real local safetensors fixture was compiled to an AEG,
integrity-checked, loaded and used for actual CPU prefill/decode or generation.
The fixture establishes integration and numerical contracts, not conversational
quality on every model. **Local HTTP/stdio** means actual localhost services or
subprocesses, not remote production deployment. **Contract** means controlled
inputs, mocks or negative tests exercise the API boundary; it does not establish
physical execution on the mocked device. **Report/policy** means the handler
produces its stated local metadata or analytical result without implying model
execution. **Prerequisite** means full execution depends on the listed artifact,
implementation, compiler, dependency or service.

| # | Registered path after `aether` | Execution evidence beyond help | Required scope or remaining limit |
|---|---|---|---|
| 1 | `compile` | CPU workflow; serialized Q4_K_M/Q8_0/FP32 formats checked | Supported source graph and target; uniform precision does not certify quality |
| 2 | `pull` | Contract: forced `--compile-local` reaches overwrite compilation | Normal source acquisition may need network/cache; no external download required by audit |
| 3 | `run` | CPU workflow: blocking and streamed generation | Ready compatible backend; explicit device restrictions are honored |
| 4 | `serve` | Actual localhost OpenAI HTTP generation | `server` extra; backend task support; remote deployment and TLS configuration are separate |
| 5 | `bench` | CPU workflow; saved JSON and baseline/backend comparison contracts | Reports backend token counts and measured generation; model loading excluded |
| 6 | `info` | CPU workflow: real package metadata | Existing valid artifact/cache ID |
| 7 | `graph` | CPU workflow: persisted graph inspection | Existing artifact; graph presence is not execution proof |
| 8 | `list` | CPU workflow: canonical cached artifact discovery | Selected local cache only |
| 9 | `rm` | CPU workflow; canonical and legacy deletion tests | Existing cached artifact; missing model fails |
| 10 | `kernels` | Real local inventory/report handler | Listed plans and plugins are not all executable kernels |
| 11 | `logs` | No-log-directory report and invalid line-count contract | Existing configured log files for tail/follow output |
| 12 | `grammar` | Actual grammar mask/FSM contract; dry-run and missing-artifact guards | Full compiled generation requires applied grammar pass and supporting backend |
| 13 | `merge` | Actual AEG and source-checkpoint merges, stored-weight references and CPU execution | AEG graph/tokenizer/tensor compatibility; source ingestion/shape alignment; SLERP explicitly unsupported |
| 14 | `ttt-config` | Invalid/reserved-control guards and required-pass contract | Source compilation must apply TTT pass; explicit layers and TTL unsupported; no adaptation quality claim |
| 15 | `kv-compress` | Retention/group validation and required-pass contract | Eligible source graph and applied passes; explicit group count unsupported; no universal memory claim |
| 16 | `green-profile` | Profile persistence/API contract and invalid-threshold guard | Static/compiler scheduling metadata is not measured renewable power or guaranteed deferral |
| 17 | `tee` | Structural report/negative verification contract | Vendor cryptographic attestation unavailable; report verification stays false and exits nonzero |
| 18 | `mcp` | Actual localhost HTTP JSON/SSE CLI handshake/tool calls; real stdio API calls | Registered reachable server; add persists config only; WebSocket execution not established by alias tests |
| 19 | `version` | Actual version report | Local source version |
| 20 | `models` | Actual parseable catalog report | Catalog entries are supported profiles/metadata, not downloaded or executed models |
| 21 | `eval` | Dataset/evaluator wiring and failed-gate contract | Actual dataset and model execution required for quality evidence; mock scores are not model quality |
| 22 | `safety` | CPU workflow: full package integrity report | Integrity and safety metadata do not prove behavioral safety |
| 23 | `trace` | CPU workflow: real request; no-request state explicitly empty | Current CLI process; no unrelated service history |
| 24 | `reasoning` | Missing/disabled persisted reasoning graph guards | Requires actual enabled reasoning payload; no reasoning-model execution claim from this audit |
| 25 | `mla-stats` | CPU workflow: persisted MLA plan/report | Metadata inspection; MLA performance requires compatible graph/backend execution |
| 26 | `merge-info` | Actual merged AEG metadata report | Existing persisted merge record; not post-merge quality validation |
| 27 | `runtime` | Group help/dispatch | See `runtime reweight` |
| 28 | `kv-share` | Required-applied-pass contract | Eligible source graph and actual Pass 15; backend integration needed to measure sharing |
| 29 | `multi-agent` | Actual local KV coordinator initialization; count/API regression | Initialization only; no model generation or physical distributed transfer claim |
| 30 | `slo-status` | Actual scheduler queue/config report | Fresh current CLI process; does not query another server |
| 31 | `green-route` | Actual analytical selection and finite-input guards | Explicit static region profile, not live carbon telemetry |
| 32 | `slo-profile` | Group help/dispatch | See `slo-profile add` |
| 33 | `hub` | Group option/dispatch contract | Configured URL and scoped authorization; no external publication in audit |
| 34 | `train` | Group help/dispatch | See explicit training/verifier limits below |
| 35 | `kernel` | Group help/dispatch | Native executable generation differs from vendor planning |
| 36 | `kv` | Group help/dispatch | See local reports and analytical policy below |
| 37 | `status` | CPU workflow: actual artifact presence and current-process state | Does not fabricate loaded model or execution evidence |
| 38 | `quantize-report` | Report API and missing-artifact contract | Requires persisted quantization data; does not synthesize quality measurements |
| 39 | `cache` | Group help/dispatch | Selected local cache |
| 40 | `doctor` | Actual host report; compiler-failure warning contract | Dependency/environment checks are not backend execution tests |
| 41 | `hardware` | Bare actual fingerprint/report; JSON and identity contracts | Presence/runtime/backend/tested statuses remain distinct; unknown counts/memory remain unknown |
| 42 | `backend` | Group help/dispatch | See `backend list` |
| 43 | `inspect` | CPU workflow; cached IDs, paths, MTP and narrow-terminal JSON | Metadata inspection; MTP payload is not draft-head execution evidence |
| 44 | `sign` | Actual local package signature creation in CPU workflow | Local signing key; no externally verified publisher identity claim |
| 45 | `verify` | Actual local signature/integrity verification in CPU workflow | Trust depends on the signing key/certificate, separate from package integrity |
| 46 | `plan` | Actual manifest-only planning/infeasibility contracts | Estimates/microprobes and priors; no model weights executed by placement planning |
| 47 | `benchmark` | CPU workflow; repeated token-count/TTFT contracts | Per-token TBT unavailable when callbacks lack timestamps; estimates labeled |
| 48 | `hw` | Same actual fingerprint handler as `hardware` | Alias, not independent hardware validation |
| 49 | `runtime reweight` | Task specification/finite-value/runtime API contract | Requires persisted task vectors; changes only this process session, not the saved AEG |
| 50 | `slo-profile add` | Actual local profile persistence | Millisecond limits; policy config alone does not establish an SLO was met |
| 51 | `hub login` | Actual localhost token validation; rejected/malformed ACK preserves previous token | Explicit service authentication required; no fabricated offline login success |
| 52 | `hub search` | Actual localhost authorized search | Service/user visibility; empty remote result stays empty |
| 53 | `hub push` | Actual localhost upload plus server-restart persistence | Exact ID/hash/size ACK; valid bounded AEG archive; namespace write permission |
| 54 | `hub pull` | Actual localhost download/restart recovery/integrity | Content hash and read permission; no external Hub tested |
| 55 | `train grpo` | Controlled missing-training-callback failure | Functional gradient/optimizer callbacks required; CLI does not fabricate an update |
| 56 | `train verify` | Actual deterministic math reward and missing-truth/failing-answer tests | Math truth or code tests required; verifies response, not model training; code execution is explicit |
| 57 | `kernel generate` | Actual CPU shared-library generation/load; unavailable CUDA guard | Host C++ compiler for CPU; vendor path must have an implemented compiler/runtime integration |
| 58 | `kernel verify` | Actual CPU library load and exported-symbol verification | Hash/load/symbol scope; not a numerical test of every operation |
| 59 | `kv transfer-stats` | Actual current-process counters/report | No physical network transfer implied by zero counters |
| 60 | `kv nika-policy` | Actual analytical transfer/recompute policy | Input estimates; `physical_network_transfer: false` |
| 61 | `kv cxl-pool-status` | Actual pool/local-fallback report | CXL hardware and actual distributed pool operations require separate evidence |
| 62 | `cache stats` | Actual parseable local stats and cache API contract | Selected process/cache scope |
| 63 | `cache flush` | Local cache API/handler contract | Explicit local clearing; not another process's cache |
| 64 | `hardware detect` | Actual fast host detection; raw JSON/save-path/identity regressions | No heavyweight execution probe; DLL/provider presence never implies execution |
| 65 | `hardware capabilities` | Actual evidence/schema report plus unknown-value contracts | Native arithmetic and queried vendor formats are labeled separately from reference conversions |
| 66 | `hardware validate` | Actual CPU AEG prefill/decode; failed-device/backend contract tests | `--execute` needed; physical GPU evidence is device-specific; production validation remains false |
| 67 | `backend list` | Actual dependency/inventory report and planning-only failclosed tests | Unchecked readiness is unknown; installed SDK does not make a generation adapter available |
| 68 | `hw detect` | Same tested handler as `hardware detect` | Alias; same limits |
| 69 | `hw capabilities` | Same tested handler as `hardware capabilities` | Alias; same limits |
| 70 | `hw validate` | Same tested handler as `hardware validate` | Alias; same limits |

`grammar compile/list/test`, `tee attest/verify`, `mcp add/list/test`, and
`multi-agent test` are argument-dispatched forms within their registered
commands. They are checked separately in CLI tests but do not add Click paths
to the count above. Legacy native-kernel syntax and `websocket`/`ws` transport
aliases also do not add paths.

The subsequent CLI lifetime audit passed 182 tests on Python 3.10 and 155 on
clean Python 3.12, with 27 optional gRPC skips in the latter. A separate actual
localhost TLS/mTLS gRPC run passed eight tests. These overlapping bundles verify
command completion, interrupted streams, server startup failures and owned
worker/executor cleanup; they do not turn the 70 registered paths into 70
independent, universally available execution engines. Current full-suite
results are in the [runtime completion report](runtime-completion-verification.md).

## Reproduce the checks

Run from the repository source with a dedicated cache. A fresh base installation
with `.[test]` exercises the framework-independent contracts; physical execution
tests require the respective optional runtime and hardware. The full-suite
results and actual host limitations are recorded in the
[completion audit](cli-hardware-completion-audit.md),
[user feedback validation](user-feedback-validation.md) and
[hardware validation](hardware-feedback-validation.md).

```powershell
$env:AETHER_CACHE_DIR = Join-Path (Get-Location) 'build/cli-validation/cache'
$env:AETHER_PLACEMENT_LEDGER = Join-Path $env:AETHER_CACHE_DIR 'placement.json'
python -m pytest --no-cov -q tests/unit/test_cli_command_audit.py tests/unit/test_cli_all_commands.py tests/unit/test_feedback_cli_contracts.py
python -m pytest --no-cov -q tests/unit/test_hub_remote_contracts.py tests/unit/test_hub_complete.py
python -m pytest --no-cov -q tests/unit/test_mcp_transport_contracts.py
python -m pytest --no-cov -q tests/unit/test_merge_contract_audit.py tests/integration/test_aeg_merge_execution.py tests/integration/test_source_checkpoint_merge.py
python -m pytest --no-cov -q tests/unit/test_hardware_reporting_feedback.py tests/unit/test_hardware_evidence.py
```

The regression modules cover malformed remote acknowledgements, actual localhost
Hub restart recovery, archive path safety, coordinate-preserving merge math,
actual persisted AEG merge execution, every help path, argument failures, and
JSON output in a narrow terminal. GPU compatibility mocks are kept separate
from real backend workloads. Source-checkpoint regressions compare every
persisted tiny-model tensor against an independently interpolated checkpoint,
exercise matching partial tensors, and reject transposed tensors with identical
element counts. Detection, environment validation and an installed
SDK are never counted as a successful execution test.
