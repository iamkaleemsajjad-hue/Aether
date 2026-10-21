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

The first full-suite attempt also caught an overbroad rename introduced during
this change: hybrid `mamba.norm` is its input scale, whereas HF Mamba-2 uses
`mixer.norm` for its output scale. That attempt was interrupted, the shared
mapping was corrected, and an explicit naming regression failed before the
correction. All **15 normalization/hybrid cases** then passed in 13.52 seconds.

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
hashes and CLI logs as an artifact. It subsequently **executed and passed** on
2026-10-21 at revision `1c6b71833fe2897d880073d4dbe379d7442656b2`:
[all 23 CI jobs passed](https://github.com/iamkaleemsajjad-hue/Aether/actions/runs/39546525959),
including the new blocking HF job. Security also passed. Repository-wide
style/type checks remain advisory; success does not clear their backlog.

The worker reported AMD EPYC 9V45, four logical CPUs and 16,760,344,576 host
RAM bytes, Linux x86_64, Python 3.11.14, NumPy 1.26.4, Torch 2.9.1+cpu,
Transformers 4.57.3, SafeTensors 0.8.0 and Tokenizers 0.22.2. The
[unaltered measured report](measurements/mamba2-normalization-hf.json),
[Git/hash verification](measurements/mamba2-normalization-provenance.json) and
[complete dependency freeze](measurements/mamba2-normalization-freeze.txt)
retain the raw repeats, bound admission reports and historical failures.
Every recorded implementation-file hash and the fixture hash were checked
against the corresponding Git revision, including historical `edc7d8f`.

| Actual original-HF comparison | Before, default | After, default | After, expanded/biased |
|---|---:|---:|---:|
| Greedy agreement on two eight-token continuations | 4/16 (25%) | 16/16 per executor | 16/16 per executor |
| Worst-position KL | 0.163815 | 1.0123e-14 | 8.8418e-15 |
| Minimum position cosine | -0.096776 | 0.9999999999998925 | 0.9999999999999427 |
| Worst relative fixed-label PPL change | +34.2617% | +0.00000384% | -0.00000200% |
| Loaded logical weight-array bytes | 10,544 | 10,672 | 18,112 |
| Logical SSM state / convolution history bytes | 512 / 384 | 512 / 384 | 1,024 / 640 |
| Complete artifact bytes | 93,121 | 101,956 | 111,622 |

After values include both CPU and Torch CPU admission measurements; before
values are the historical CPU implementation. The historical expanded/biased
model compiled diagnostically but failed at actual forward execution:
`in_proj has 148 outputs; expected 84`. No numerical or speed value is assigned
to that failed execution. Stored weight arrays grew by the actual 128-byte
post-gate scale on the default fixture. Complete artifact growth also includes
the new measured gate and changed graph/manifest; it is not a weight-only cost.
There is no resident-memory improvement claim.

Timing uses two warmups and seven measured repeats per probe, with BLAS/OpenMP
threads set to one. Array byte counts describe weights, SSM state and convolution
history, explicitly **not** peak RSS; activation and overhead peaks remain
unmeasured. Measured engine-call timings on the default fixture are:

| Metric, median [minimum, maximum] | Before | After |
|---|---:|---:|
| One-token prompt decode, tokens/s | 15,350 [14,610,15,780] | 15,263 [13,987,15,591] |
| 64-token prompt decode, tokens/s | 14,277 [13,358,14,515] | 15,220 [14,807,15,618] |
| One-token first output, milliseconds | 0.06274 [0.06038,0.06434] | 0.06378 [0.06222,0.06560] |
| 64-token first output, milliseconds | 3.4582 [3.3835,3.4767] | 3.2390 [3.2114,3.3727] |

These are tiny synthetic models on a shared CI worker. Short-prompt decode
decreased about 0.57% and first-output latency increased about 1.66%; longer
prompt observations improved. The overlapping ranges and differing incorrect
historical outputs do not establish a general speedup or no-regression result.
These are engine timings, not complete API TTFT. Current default compilation
took 4.601 seconds including mandatory HF/CPU/Torch admission; historical
diagnostic compilation took 0.175 seconds without that admission. Those compile
times are different verification workloads, not an equivalent speed benchmark.

Both public backends generated and streamed the exact original-HF text and
token count from each same admitted artifact. The ordinary CLI streamed the
same response with exit 0, with no diagnostic waiver. This proves these two
untrained source contracts, not arbitrary trained Mamba-2 checkpoints.

## Local suite and static checks

The complete local run retained all previous cases and added 13. It returned
**5,330 passed, 42 skipped, three failures** in 1,015.24 seconds. All new
normalization cases, both existing SSD integration cases and the hybrid cases
passed. The failures were the two previously observed local MCP HTTP framing
cases and an unchanged placement dispatch timing guard (2.03x versus 2.0x).
A focused followup passed nine cases including the timing guard, but retained
both HTTP failures, including outside the sandbox. No assertion, threshold or
RAM guard was relaxed. This local run is **not green**, despite hosted CI being
green; the local HTTP environment and repeatability remain unresolved.
The HF job now also installs server/format extras and runs the complete suite
with real Torch/Transformers, retaining JUnit and coverage. That extended
step's first outcome is pending until the updated workflow executes.

Raw full-run coverage is **45,031/57,244 (78.6650129%)**, versus
**45,038/57,206 (78.7295039%)** before. Branch coverage is 10,751/17,228 versus
10,756/17,206. The reduction is retained honestly; this run does not satisfy
the mission's coverage/no-failures completion criterion. It must not be replaced
by the union of isolated reruns or inferred from the hosted pass status.

The six changed source files retain 317 Ruff and 102 mypy findings, with no
added file/code or file/message diagnostic against their before baseline.
New test/fixture/tool Ruff and the new tool's mypy checks pass. This scoped
comparison does not clear repository-wide lint/type debt. Local raw records
are `build/mission/mamba2-normalization-*`, `mamba2-norm-*`,
`mamba2-local-*` and `mamba2-hf-verification.json`.

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
