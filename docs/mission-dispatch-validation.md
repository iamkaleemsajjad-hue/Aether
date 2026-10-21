# Dispatch measurement validation contract

The complete CPU/HF CI job for `9167d1d` returned **5,332 passed, 51 skipped,
one failure**: two live CPU dispatch probes differed by `2.246305928260903x`.
The strict original-HF/API/CLI check and the other 22 CI jobs passed. The
coverage floor was skipped after the full-suite failure. This is separate
from the preceding [successful runtime run](mission-hybrid-convolution.md).

## Confirmed false positive

Read the complete placement-planner and dispatch-probe tests, census producer,
ledger reconciliation consumer and the
[planner design](architecture-execution-planner.html). The probe explicitly
estimates current issue cost, with bounded convergence/refusal. It does not
guarantee invariant timings under changing scheduler load or CPU clocks.
`take_census` supplies accepted measurements to `reconcile_dispatch`; that
consumer preserves small drift and replaces values differing by at least 2x.

The old live test required two independent measurements to differ by less than
2x. A controlled counterexample executes the actual probe twice with settled,
known per-operation clock inputs: `0.005/8` and `0.020/8` test seconds. Both
probes correctly return the supplied costs, whose ratio is 4x; the old assertion
rejects them. These are **synthetic clock inputs, not hardware timings**.
This proves the assertion demands a property the producer does not promise.
Caching an old result or increasing the ledger threshold would obscure real
changes in issue cost; neither is an appropriate correction.

The live CPU smoke check now requires finite nonnegative evidence (zero means
no settled measurement) and verifies that probing preserves Torch's thread
setting. Existing controlled tests still require a known positive result for
stable work, rejection of oscillating/invalid timing, bounded work, warmup,
adaptive block counts and refusal of an unobservable asynchronous backend.
Three added tests run the real producer with known clock costs, then consume
them through the real ledger: 1.1x drift retains the old value, while 4x and
0.25x changes replace it. Every stored result must match the independent cost
oracle and retain its measured provenance.

## Checks and limits

The focused placement/probe run returned **144 passed**, with no failures or
skips. The [validation record](measurements/dispatch-assertion-validation.json)
binds the failed hosted JUnit, counterexample, current production-source tree
and static-check counts. Touched-file Ruff findings remain **86 -> 86**;
mypy findings decrease **153 -> 143**, with no new file/message diagnostics.
The dispatch-probe file passes full Ruff. Runtime sources and the 2x ledger
threshold are unchanged. This is a test-contract correction, not a measured
engine speedup or a runtime calibration improvement; no new throughput, TTFT,
peak-memory, artifact-size or numerical benchmark is claimed.

Reproduce the controlled/real checks with test extras and Torch installed:

```powershell
python -m pytest tests/unit/test_dispatch_probe.py tests/unit/test_placement_planner.py -o addopts='' -q
```

Complete local and hosted runs are recorded separately; focused results are
not a substitute for their JUnit and coverage evidence. Earlier local
RAM-admission and MCP HTTP failures remain documented, without weakened guards
or assertions accepting malformed responses.

No optimizer changed. The [22-row pass dispositions](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [pass measurements](mission-pass-audit.md) retain their existing limitations.
The [hardware matrix](../hardware_validation_matrix.json) and
[generated hardware table](../README.md#supported-hardware-targets) retain their
per-backend techniques and tiers. The live smoke check runs on physical x86
CPU; controlled clocks do not validate any device's performance, and this
correction promotes no accelerator or model-family validation tier.
