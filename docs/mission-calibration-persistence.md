# Calibration persistence: denied caches and bounded creation

The optional calibration ledger can now retain measurements in memory when
temporary-file creation is denied, without retrying that denial as a filename
collision. This corrects a first-forward stall observed on the physical Windows
CPU host on 2026-10-18 and corrected on 2026-10-19. It does not change model
weights or calibration equations.

Read scope covered the complete ledger and strategy calibrator, placement exports,
planner and census call paths, bootstrap, Torch construction/forward contracts,
planner and decode-strategy tests, and the placement/CLI documentation. The ledger
is shared by placement, hardware probes and runtime projection calibration; its
schema remains `aether_placement_ledger/1`.

The live `py-spy 0.4.1` stack stopped in CPython 3.10.11's
`tempfile._mkstemp_inner`, called by `CalibrationLedger.save` from the first Torch
forward. On Windows that helper retries `PermissionError` when the directory's
ACL probe still reports writable. Actual creation in this sandbox was denied in
the default user cache. The test was interrupted after the live stack was
captured; there is no completed baseline latency to report. Replacing
`NamedTemporaryFile` with `mkstemp` would use the same helper.

Saving now uses exclusive creation with a random filename in the destination
directory and mode `0600`, retrying only `FileExistsError`, at most eight times.
The descriptor closes before `os.replace`. Denied creation/replacement logs a
warning, preserves the previous on-disk ledger and leaves new measurements dirty
in memory. Serialization errors still propagate. Owned temporary files and raw
descriptors are cleaned up on failure; colliding files belonging to another writer
are preserved. This bounds creation attempts, not individual filesystem-call
latency or crash durability.

| Check | Before | After |
|---|---|---|
| Original five regression cases | 4 failed, 1 passed | All pass |
| Denied creation with Windows retry branch, test bound of four | Four attempted creations | One; existing bytes and in-memory notes retained |
| Exhausted collisions, test bound of 32 | 32 attempts | Eight; no existing file replaced |
| Failed replacement or JSON serialization | Partial temporary file remained | No owned temporary file remains; existing ledger preserved |
| File-wrapper failure | No prior measured regression result | Descriptor closed and owned file removed |
| Related planner, strategy and persistence tests | No combined before run | 175 passed in 15.42 seconds |
| Real Torch CPU with explicitly denied calibration writes | No completed before timing | Prefill/decode match the untuned reference within `rtol=1e-5`, `atol=1e-6`; one creation attempt per measured choice |
| Default-cache real-AEG comparison | Stalled in temporary creation | Completed after the ledger and separate forward-step shape corrections: one test passed in 4.63 seconds |

The real-artifact comparison initially completed after the ledger correction but
failed on Torch's separate `(1, vocab)` versus `(vocab,)` public protocol defect.
That failure is retained as evidence, not counted as a passing comparison.

Evidence is under `build/mission/ledger-before.*`, `ledger-after.*`,
`ledger-focused.*`, `quality-validator-live-stack.txt`,
`quality-validator-focused-termination.json`, `ledger-torch-default-cache.*` and
`ledger-torch-default-cache-fixed.*`. The original five-case before run returned
four failures in 0.71 seconds with deliberately capped standard-library retries;
it is not a measurement of the uncapped stall. Two additional regression cases
cover descriptor cleanup and actual Torch arithmetic.

This area establishes correctness and request progress for denied advisory cache
writes. It makes no decode-throughput, artifact-size or RSS improvement claim.
The full-suite result and remaining mission limits are recorded in
[mission validation](mission-validation.md). The
[22-row pass audit](mission-pass-audit.md) remains unchanged by this I/O correction.
Hardware evidence is physical CPU plus controlled I/O failures on Windows; no
accelerator validation tier is promoted.
