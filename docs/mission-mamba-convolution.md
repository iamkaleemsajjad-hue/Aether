# Mamba causal-convolution history correction

Executed on 2026-10-20, Intel Core i5-1334U, Windows AMD64, Python 3.10.11,
NumPy 1.26.4, Torch 2.9.1+cpu and pytest 9.1.1. This checkpoint establishes
one convolution/cache correction. Original-HF Mamba/Mamba-2 parity, learned
Mamba-2 output normalization, config propagation and model-family admission
remain separate unfinished work.

## Reproduction and root correction

Read the CPU Mamba and Mamba-2 engines, the shared Torch state-engine module,
their loader/backend dispatch, cache clone/handoff and worker geometry paths,
the existing SSD integration test and recurrent fixtures. The
[architecture decision](mission-architecture-note.md) remains engine-owned
execution. No graph rewrite can repair these handwritten convolution loops.

Each convolution step concatenated the previous `k-1` inputs and the current
input, then retained `window[..., -(k-1):]`. When `k=1`, Python interprets
`-0:` as the whole window. The history grows each token and old inputs enter
subsequent convolution sums. Both CPU engines and both Torch engines had this
defect; CPU cache handoff additionally rejected the resulting geometry.

All four consumers now retain `window[..., 1:]`. This drops the oldest column
and produces exactly `k-1` retained columns, including an empty width-1 history.
For larger widths it selects the same columns as before. Cache/weight schemas,
the AEG integrity chain and packaging are unchanged.

The independent oracle is a full-sequence `numpy.convolve` with reversed stored
filters, not another cache-slicing implementation. Per channel it evaluates
`u[t] = bias + sum_j weight[j] * input[t-k+1+j]`, with zero input before time 0.
Zero B/C projections isolate the convolution and learned D skip through actual
public engine forwards. Other parameters remain nontrivial. This deliberately
does not serve as an oracle for the whole SSD recurrence or original HF model.
The convolution/block contract was checked against
[Dao and Gu, section 7.1](https://arxiv.org/html/2405.21060v1#S7.SS1) and the
[v2.2.4 reference implementation](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2_simple.py).

Before the runtime fix, the new tests returned **18 failed / 34 passed**:
10 NumPy failures (including two handoff failures) and eight Torch CPU failures.
All width-2 and width-4 cases already passed. Afterward, **52/52 pass**, covering
both families and CPU implementations, optional convolution bias, two stored
filter layouts, one-shot forward, uneven prefill/decode partitions and CPU
export/import continuation. The existing real SafeTensors compile/save/load/
public-generate integration now also exercises convolution widths 1 and 3.
Its synthetic artifacts explicitly use diagnostic execution; they do not carry
original-HF numerical certification.

## Measured memory and numerics

The companion [microbenchmark](../tools/benchmark_mamba_convolution.py) uses the
same deterministic one-layer fixtures: inner size 8, vocabulary 17, 1,024 actual
input tokens, two warmups and seven timing repeats, serially. Quality compares
65 fixed input positions against the independent FP64 convolution oracle;
likelihood uses the same 64 next-token labels. These untrained fixtures do not
measure language-model corpus perplexity or autoregressive HF greedy parity.

| Width-1 result | Before | After |
|---|---:|---:|
| Mamba retained history, logical / backing bytes per layer, both engines | 32,768 / 32,768 | 0 / 32 |
| Mamba-2 retained history, logical / backing bytes per layer, both engines | 49,152 / 49,152 | 0 / 48 |
| Mamba teacher-forced top-1 agreement, both engines | 28/65 (43.08%) | 65/65 (100%) |
| Mamba-2 teacher-forced top-1 agreement, both engines | 27/65 (41.54%) | 65/65 (100%) |
| Mamba worst-position KL, CPU / Torch CPU | 0.527084 / 0.527084 | 6.208e-15 / 8.286e-15 |
| Mamba-2 worst-position KL, CPU / Torch CPU | 0.455382 / 0.455382 | 6.448e-15 / 8.077e-15 |
| Mamba minimum cosine, CPU / Torch CPU | -0.167295 / -0.167295 | >=0.999999999999986 |
| Mamba-2 minimum cosine, CPU / Torch CPU | -0.009221 / -0.009221 | >=0.999999999999987 |

The empty history remains a view backed by one input column: the backing bytes
are **not zero**. This removes growth of this component, without claiming zero
recurrent-state memory or a measured whole-process peak-RSS reduction. Width-4
logical/backing bytes remain 96/128 for Mamba and 144/192 for Mamba-2. All after
cases satisfy the convolution oracle's numerical thresholds; maximum positive
fixed-label relative PPL change is `9.037e-9` for width 1 (`7.715e-9` for width 4).
Before-fix fixed-label PPL was lower despite incorrect outputs; the KL, cosine
and agreement failures are why that statistic alone cannot establish quality.

## Timing observations and limits

Values below are median [minimum, maximum] decode tokens/s for 1,023 individual
cached forwards. They exclude loading, tokenization and output rendering. Each
process sets BLAS/OpenMP threads to one; Torch also uses one intra/inter-op
thread. No tests or other agent benchmarks ran concurrently, but the user's
host remained under load. Available RAM at process start was 728,387,584 bytes
before and 819,597,312 bytes after.

| Engine / family / convolution width | Before | After |
|---|---:|---:|
| NumPy / Mamba / 1 | 6718.0 [1533.8,7221.9] | 5148.7 [3865.2,7261.9] |
| NumPy / Mamba / 4 | 1278.1 [1181.5,1402.0] | 1038.3 [990.1,1084.0] |
| NumPy / Mamba-2 / 1 | 1228.7 [1167.7,2199.3] | 1201.6 [1056.1,1306.9] |
| NumPy / Mamba-2 / 4 | 1445.0 [1399.0,1963.5] | 1243.7 [1144.0,1281.7] |
| Torch CPU / Mamba / 1 | 460.2 [402.3,549.4] | 413.3 [394.3,511.5] |
| Torch CPU / Mamba / 4 | 460.2 [272.2,493.3] | 450.4 [420.0,471.9] |
| Torch CPU / Mamba-2 / 1 | 345.7 [225.4,395.7] | 364.1 [282.8,401.8] |
| Torch CPU / Mamba-2 / 4 | 394.9 [325.4,412.3] | 409.6 [382.5,467.0] |

Several medians worsened, including unchanged-computation width-4 controls.
These observations establish neither a speedup nor absence of speed regression.
A controlled performance recheck remains required. The verified benefit is
correct convolution outputs and bounded retained history. No trained-model,
large-model or production-throughput claim follows from these fixtures.

Median first-forward milliseconds (one-token prompt) before -> after are
0.162 -> 0.191 (NumPy Mamba width 1), 0.756 -> 0.818 (NumPy Mamba-2 width 1),
2.136 -> 2.314 (Torch Mamba width 1), and 3.394 -> 2.845 (Torch Mamba-2 width 1).
These are engine-call timings, not complete public-API TTFT measurements.
All raw repeats, first-forward ranges, numerical metrics, file hashes and
repeat-end sampled process RSS are preserved in the
[before report](measurements/mamba-convolution-before.json) and
[after report](measurements/mamba-convolution-after.json). Repeat-end RSS is
explicitly not a continuously sampled peak or per-component attribution.
No paired compile/artifact-size benchmark was run for this cache-only change;
the reports record that absence rather than inserting zero.

Reproduce with test extras and Torch installed:

```powershell
python -m pytest tests/unit/test_mamba_convolution_history.py -o addopts='' -q
python tools/benchmark_mamba_convolution.py --output report.json
```

For an older-runtime comparison, keep the new test and benchmark files while
using the three runtime files from the parent revision in a separate checkout.
Both measured reports bind the identical fixture and harness SHA-256 values;
runtime hashes distinguish the implementations even before the fix is committed.

## Verification status

The complete local suite returned **5,320 passed, 42 skipped, zero failures**
in 696.72 seconds, including all 54 convolution unit/integration cases. One
existing skip is the opt-in public compile/load/Runtime audit. Enabling
`AETHER_RUN_PASS_AUDIT_INTEGRATION=1` and running that case separately returned
**one passed** in 27.92 seconds; it was not silently counted in the full-run
pass total. The other skips retain the existing optional dependency, offline
dataset/network, physical device and unavailable Windows IPC limitations.
The two previously failing local MCP HTTP cases passed this time; their code
was unchanged, so this is not evidence of a product fix for that environment.

Line coverage increased from **45,001/57,206 (78.6648254%)** to
**45,038/57,206 (78.7295039%)**. Branch coverage increased from
**10,734/17,206 (62.3852145%)** to **10,756/17,206 (62.5130768%)**.
Both use Coverage 7.16.2. The separate opt-in followup does not contribute to
that coverage result.

Touched runtime Ruff findings remain 106 before/after, and the existing
integration test remains at six, without a file/code increase. Source mypy
remains at 84 pre-existing findings with no added file/message diagnostics;
the new benchmark has zero mypy findings. Critical Ruff, the new files' full
Ruff checks and `git diff --check` pass. These checks do not clear the wider
repository static-analysis backlog.

Raw local records are `build/mission/mamba-conv-before-{cpu,torch}.*`,
`mamba-conv-after.*`, `mamba-conv-full.*`, `mamba-conv-full-coverage.xml`,
`mamba-conv-public-audit.*`, `mamba-conv-{ruff,mypy}-*` and
`mamba-conv-verification.json`. The full-suite log and JUnit file are separate
from its explicit opt-in followup. No assertion or resource guard was waived.

## Passes, hardware and remaining mission work

No optimizer pass changed. The existing
[22-row pass disposition table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [per-pass measurements](mission-pass-audit.md) retain their limitations;
only P3 is default. This convolution fix cannot establish benefits for graph
rewrites or experimental passes.

The NumPy and Torch CPU checks ran on physical x86 CPU hardware. They verify
the scoped convolution oracle, not all Mamba semantics. The generated
[hardware table](../README.md#supported-hardware-targets)
and [hardware evidence matrix](../hardware_validation_matrix.json) retain the
existing per-backend dates/techniques and tiers, including historical DirectML
execution and other backends' reference, compile-time or awaiting-hardware
status. No CUDA, ROCm, MPS or other accelerator execution was performed here.

Original-HF Mamba/Mamba-2/RWKV-7/MLA parity, ONNX parity, the independent
numerical gate's specialized-family support, controlled performance validation,
trained-model benchmarks, KV COW/eviction/soak, archive hardening and blocking
core/runtime static CI remain unfinished parts of the original mission.
