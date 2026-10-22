# Mamba-1 checkpoint fidelity

This correction follows the [engine-owned architecture decision](mission-architecture-note.md).
The actual CPU and Torch executors consume the corrected weights and configuration;
an IR rewrite alone cannot correct their output. The wider runtime mission remains
in progress.

## Reproduction and source math

Read the complete relevant ingestion, architecture, loader, Mamba, shared Torch,
hybrid/state and admission modules, their callers and related tests. Research
covered Gu and Dao's [Mamba paper](https://arxiv.org/html/2312.00752v2), including
its appendices, the v2.2.4 reference
[`Mamba.step`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py)
and
[`selective_scan_ref`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/selective_scan_interface.py),
and the complete installed Transformers 4.57.3 Mamba configuration/model code.

The checkpoint recurrence is `h[t] = exp(delta[t]*A)*h[t-1] + delta[t]*B[t]*x[t]`,
then `y[t] = C[t]*h[t] + D*x[t]`, SiLU gating, output projection and residual.
This implements paper equation (2), with the literal `delta*B` input injection
used by both reference implementations. Substituting equation (4)'s full ZOH
input discretization would change the checkpoint's function. The existing state
transpose, current-state readout, delta bias and D-skip order were correct.

The new independent FP64 oracle sums every causal source contribution using the
full sequence matrix and `numpy.convolve`. It does not call the runtime scan.
Complete deterministic, untrained FP32 checkpoints contain nonzero B/C, learned
norm scales, widths 1/3, expansion 2/4, optional projection biases and distinct
multilayer blocks. These test numerical fidelity, not corpus or task quality.

## Confirmed defects and root changes

* HF Mamba uses RMSNorm even with `layer_norm_epsilon` and an unused `rms_norm`
  flag. Architecture inference selected LayerNorm. Explicit `expand` now also
  determines Mamba-1's inner width when the raw config omits `intermediate_size`.
* Stored input/output projection biases now reach standalone and hybrid CPU/Torch
  execution. New optional dataclass fields are appended to preserve existing
  positional construction; actual hybrid artifact and oracle tests cover callers.
* Pure Mamba-1 loading preserves an explicit zero epsilon and rejects malformed
  learned norm shapes instead of padding or truncating trained values.
* Stable softplus keeps a finite delta bias of 1000 finite. CPU SiLU evaluates
  `x*sigmoid(x)` without clipping finite inputs; large negative inputs formerly
  created false nonzero values that subsequent finite weights could amplify.
* Both Mamba-1 engines expose the actual all-row/final-vector `forward_step`
  contract. Original-HF admission can dispatch to the real Torch Mamba executor.
  Recurrent references retain a separate 4096-position workload bound without
  inventing a positional-table limit; RAM/headroom guards are unchanged.
* Ingestion refuses unsupported Mamba-1 mixer activations at its common exit,
  after specialized/generic fallback and snapshot materialization. `silu`,
  `swish` and the absent-field default remain supported. SDK/CLI regressions
  check refusal before publication and preservation of an existing destination.

A possible cache-aliasing concern was rejected for the actual speculative path:
its `clone_cache` already clones the known Torch state/cache tensors. Generic
`deepcopy` behavior alone did not prove a consumer defect, so no cache-class
rewrite was made.

| Regression group | Before correction | After correction |
|---|---:|---:|
| Initial checkpoint/engine cases | 21 failed, 1 passed | 22 passed |
| Added finite-value/shape/activation cases | 8 failed, 2 passed | All 32 cumulative cases passed |
| Fallback/snapshot/CLI refusal cases against historical source | 4 failed | All 42 final Mamba-1 cases passed |
| Shared Mamba-2, hybrid, convolution, speculation and reasoning controls | Not a new baseline | 204 passed, including the earlier 32 Mamba-1 cases |

The matrix assertions use `rtol=3e-5`, `atol=3e-6`, cover full forward and owned
cached continuation, and preserve bounded convolution history. These tolerances
are test thresholds, not a reported measured KL or original-HF comparison.
Local raw records are `build/mission/mamba1-{fidelity,edges,refusal}-*`.

## Validation and measurement limits

The local original-HF attempt refused **before execution**: available RAM was
494,948,352 bytes, estimated worker reserve 536,925,852 bytes, and required
additional headroom 268,435,456 bytes. It supplies no original-HF result.
The [strict HF tool](../tools/validate_mamba2_hf.py) now accepts `--family mamba`
while retaining Mamba-2 as its backward-compatible default. It checks ordinary
compiler admission, reload, both public CPU backends, streaming and the CLI.
The blocking pinned hosted job exercises nine Mamba-1 contracts alongside the
existing Mamba-2 checks and complete CPU suite. Historical source is fixed at
`2f2a2c4659063b31797011ac03373a1c4e539173`; failed historical executions remain
failures, and nonfinite logits cannot become benchmark samples.

Hosted Mamba-1 execution, speed/TTFT, peak RSS and original-HF KL/cosine/greedy/PPL
measurements are pending. No speed, memory, no-regression or trained-family
certification is claimed.

The complete local run returned **5386 passed, 42 skipped, 3 failed** in 972.92
seconds. It collected 38 of the new Mamba-1 cases; the last four refusal tests
were added after collection and passed in the separate final **47-case**
Mamba-1/hybrid integration run. The full run's failures were the previously
observed Unicode audit worker RAM admission (464,261,120 available versus
406,938,624 worker reserve plus 134,217,728 headroom bytes) and two MCP HTTP
handshake cases. No guard or assertion was relaxed. Raw line coverage fell from
45028/57244 (78.6597722%) to 45016/57277 (78.5935018%); branch coverage changed
from 10754/17228 to 10685/17242. This is not a green local suite or a satisfied
coverage/no-regression criterion. Raw records are `build/mission/mamba1-full*`.

Scoped static checks retain 516 Ruff and 184 mypy findings across the eight
changed production modules and three shared tests, with no added file/code or
file/message diagnostic. New fixtures/tests and the HF tool pass full Ruff and
mypy. This does not clear the repository-wide advisory backlog.

```powershell
python -m pytest tests/unit/test_mamba1_fidelity.py -o addopts='' -q
python tools/validate_mamba2_hf.py --family mamba --output hf-mamba1-results
# Optional: --baseline-src /path/to/2f2a2c4/src
```

## Passes, hardware and remaining scope

No optimizer pass changes in this area. The complete
[22-row disposition table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [measured per-pass audit](mission-pass-audit.md) retain their results and
unresolved benefit/composition/idempotency checks. Only P3 is default.

The matrix-oracle and actual Torch CPU checks executed on the physical Windows
x86 CPU. Hybrid bias tests establish their specific path, not whole-model HF
Jamba parity. CUDA/ROCm/MPS/ONNX and other targets acquire no validation tier;
the [generated hardware table](../README.md#supported-hardware-targets) and
[evidence matrix](../hardware_validation_matrix.json) retain their recorded
techniques, dates, versions and outstanding physical-hardware requirements.

Trained Mamba checkpoints, custom normalization overrides, non-SiLU execution,
long-context/large-model quality, accelerator parity, quantized execution,
controlled performance, bounded serving memory and the remaining family/pass
corrections remain unverified. Planning/dry-run does not execute ingestion or
certify the unsupported-activation check. Earlier trained-family divergence
records and family grades remain unchanged.
