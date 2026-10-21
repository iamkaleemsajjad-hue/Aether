# Mamba-2 learned-normalization contract

This area corrects checkpoint ingestion and the handwritten CPU/Torch CPU SSD
engines. The [architecture decision](mission-architecture-note.md) remains B:
engines consume stored weights and configuration; graph-only rewrites cannot
repair their numerical behavior. The mission remains in progress.

## Reproduction and source math

Read the ingestion/binding paths, architecture detector, quantizer, loader,
Mamba-2 engine, shared Torch state engines, admission workers, cache/speculation
callers and their related tests. Read Dao and Gu's
[Mamba-2 paper](https://arxiv.org/html/2405.21060v1), including appendices, and
the v2.2.4 reference implementation's
[`mamba2_simple.py`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2_simple.py),
[`ssd_minimal.py`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/ssd_minimal.py)
and full Mamba-2 module. Also read the installed Transformers 4.57.3
configuration and complete Mamba-2 implementation.

The source block uses `state = exp(dt * -exp(A_log)) * state + dt * x * B`,
`y = C * state + D * x`, then **learned RMSNorm after `y * SiLU(z)`** before
the output projection (paper section 7.1 and reference implementation).
The recurrence uses `dt * B`; replacing it with exact-ZOH discretization would
change this model. HF's outer and final norms are also RMSNorm even though
its epsilon is named `layer_norm_epsilon` and its config exposes an unused
`rms_norm` flag. The official implementation also supports other group-norm
contracts; this correction does not establish those variants' parity.

A complete, deterministic, untrained 2,668-parameter HF-format checkpoint
reproduced the defect on the old implementation:

* Both `backbone.layers.0.norm.weight` and
  `backbone.layers.0.mixer.norm.weight` normalized to `ssm_norm`. Binding chose
  the mixer vector and counted both original tensors as bound.
* The outer scale should have been 16 values from 0.5 to 1.5. Reload instead
  returned the first 16 values of the 32-element inner scale, 1.1 to 1.58387.
  The generic norm-vector resize hid the incompatible shape.
* Manifest inference selected LayerNorm. Neither engine consumed the learned
  post-gate scale; both also omitted stored output-projection bias.
* An explicit HF `expand=4` did not determine the expanded geometry.

The new regressions returned **10 failed / one passed before** production
changes. The first after-run returned **11/11 passed**, including actual
SafeTensors compile/save/load, both learned scale vectors, expansion 2/4,
optional input/output bias, cached continuation and real Torch CPU forwards.
The independent FP64 oracle constructs the full semiseparable matrix from
paper equations 2/3 and uses full convolution; it does not reuse the engine's
token recurrence. These are fidelity checks, not trained-model quality results.

## Root changes and compatibility

Ingestion now assigns `ssm_output_norm` its own stored slot, and both engines
apply that scale after gating. HF Mamba-2 gets its actual RMSNorm contract and
expanded head geometry. Output-projection bias reaches both implementations.
Loader checks require exact learned norm shapes rather than repeating or
truncating scales. Previously misbound artifacts must be recompiled: their
discarded original input scale cannot be reconstructed from the artifact.
Legacy diagnostic fixtures with valid shapes and no output-norm tensor retain
their prior optional contract; they acquire no numerical certification.

Both engines expose the existing `forward_step` row/final-vector protocol.
Original-HF admission can now exercise the real Mamba-2 Torch executor. The
reference accepts Mamba-2's absence of a positional-table context limit, while
separately bounding each reference workload to 4,096 positions. Other model
families retain their existing admission behavior. RAM guards are unchanged.

## Original-HF execution and measurements

The repeatable [strict HF tool](../tools/validate_mamba2_hf.py) writes complete
synthetic checkpoints, uses the original Transformers model as FP32 reference,
and requires ordinary compiler admission, reload, NumPy CPU, Torch CPU, public
generation/streaming and CLI execution. Its default and expanded/biased fixtures
use identical fixed probes for logits, fixed-label NLL and greedy continuation.
The optional historical checkout executes its actual old code in diagnostic
mode and retains failures; it never supplies a reference for current quality.

The local original-HF attempt was **refused before worker imports**: available
RAM was 687,149,056 bytes; the required reserve was 536,923,100 plus 268,435,456
bytes of headroom. This is not an HF parity result. The new blocking CI job
installs a pinned CPU reference stack and retains measurements, source/code
hashes and CLI logs as an artifact. Its outcome is pending until it executes.

Timing uses two warmups and seven measured repeats per probe, with BLAS/OpenMP
threads set to one. Array byte counts describe weights, SSM state and convolution
history, explicitly **not** peak RSS; activation and overhead peaks remain
unmeasured. No speedup or memory improvement is claimed before actual results.

```powershell
python -m pytest tests/unit/test_mamba2_normalization.py -o addopts='' -q
python tools/validate_mamba2_hf.py --output hf-parity-results
# Optional: pass --baseline-src /path/to/edc7d8f/src for the original implementation.
```

## Passes, hardware and remaining scope

No optimizer pass changes. All 22 dispositions and outstanding benefit,
composition and idempotency checks remain in the
[pass table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [measured pass audit](mission-pass-audit.md). Only P3 is default.

The matrix/oracle tests executed on a physical Windows x86 CPU, including real
Torch CPU. They do not promote any accelerator tier. The generated
[hardware table](../README.md#supported-hardware-targets) and
[evidence matrix](../hardware_validation_matrix.json) retain their recorded
dates, drivers, techniques and reference/compile-time/awaiting-hardware tiers.
No CUDA, ROCm, MPS, ONNX or other accelerator parity is established here.

Trained Mamba-2 quality, nondefault time-step limits/activations, extreme-value
numerics, other official normalization variants, Mamba-1/RWKV-7/MLA parity,
controlled performance, quantized execution, KV COW/eviction/soak, hardened
archives and blocking core/runtime type CI remain unfinished mission work.
