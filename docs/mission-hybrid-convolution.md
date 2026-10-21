# Hybrid Torch causal-convolution correction

Executed on 2026-10-21 on the physical Intel Core i5-1334U Windows AMD64 host,
Python 3.10.11, NumPy 1.26.4, Torch 2.9.1+cpu, psutil 7.2.2 and pytest 9.1.1.
This is a convolution/cache correction, not trained Jamba or Mamba-1 admission.

## Reproduction and consumed correction

The [architecture decision](mission-architecture-note.md) remains engine-owned
execution. Inspection of the Mamba-1 ingestion/loader/CPU/Torch paths uncovered
a fifth copy of the causal-convolution loop in `TorchHybridAEGEngine._ssm_step`.
The [previous four-loop correction](mission-mamba-convolution.md) covered the
standalone Mamba/Mamba-2 engines; it did not cover this separate hybrid loop.
The CPU hybrid delegates its SSM step to the corrected standalone CPU engine.

Read the full shared Torch module, hybrid/SSM engines and cache-handoff modules,
their dispatch and worker/speculative callers, convolution, hybrid rotary,
handoff, speculative and real SafeTensors integration tests. Mamba-1 research
checked [Gu and Dao, section 3.4](https://arxiv.org/html/2312.00752v2) and
[the v2.2.4 `Mamba.step` reference](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py).
For channel `c`, the reference convolution is
`bias[c] + sum_j weight[c,j] * input[t-k+1+j,c]`, zero-padded before time zero.

The hybrid retained `window[..., -max(k-1, 0):]`. At width 1, `-0:` retains the
whole window, which grows every token; broadcasting the width-1 filter over
that history also sums previous inputs. The root correction drops the oldest
column with `window[..., 1:]`. This retains exactly `k-1` columns, including an
empty width-1 history; for wider kernels it selects the same columns as before.
Weights, artifact schemas, integrity checks and cache schemas are unchanged.

The new independent regression initially returned **2 failed / 6 passed**:
both Torch CPU width-1 cases failed; CPU and width-3 cases passed. Its FP64 oracle
uses full `numpy.convolve`, not another history update. A real SSM block has
zero B/C and a nonzero D skip; an attention/FFN block has zero output projections,
so it preserves the residual. The tests exercise whole forwards, uneven cached
partitions and independent cache clones, with and without convolution bias.
This isolation does not establish correctness of a trained hybrid's other math.

The real SafeTensors compile/save/load/public-generate integration now covers
widths 1 and 3, including CPU/Torch CPU and a two-device CPU dispatch mesh.
That mesh checks dispatch; it does not validate physical multi-GPU execution.
These synthetic integration artifacts explicitly use diagnostic mode and do
not bypass ordinary artifacts' mandatory original-HF numerical admission.
On the isolated parent source snapshot (`fd5aa179`), the width-1 artifact
integration fails CPU/Torch parity: 25/48 logit values differ beyond tolerance,
with maximum absolute error 0.00025474. This independently reproduces the
defect after real checkpoint ingestion and artifact loading.

## Measured memory and numerics

The [benchmark](../tools/benchmark_mamba_convolution.py) now accepts
`--families hybrid`, retaining the existing standalone families by default.
It executes actual forwards with two warmups, seven repeats, BLAS/OpenMP/Torch
threads fixed to one, 1,024 input tokens and an independent 65-position oracle.
It measures convolution history separately from the hybrid's ordinary growing
attention KV cache. Array/storage sizes are not a whole-process peak-RSS claim.

| Torch CPU width-1 result | Before | After |
|---|---:|---:|
| Retained history, logical / backing bytes after 1,024 inputs | 32,768 / 32,768 | 0 / 32 |
| Teacher-forced top-1 agreement | 28/65 (43.08%) | 65/65 (100%) |
| Worst-position KL | 0.5270837963 | 8.2857264e-15 |
| Minimum cosine | -0.1672951687 | 0.999999999999986 |
| Relative fixed-label PPL change | -0.0279592523 | 7.4516038e-10 |

The empty history still retains one 32-byte backing column. The two-layer
hybrid's width-4 history remains 192 logical / 224 backing bytes on both engines,
including its inactive SSM placeholder. The CPU width-1 control remains 0 / 32.
All after cases pass the oracle thresholds. Fixed-label PPL alone would have
missed the old wrong output; these 64 labels are not corpus perplexity.

## Timings and limits

Initial serial observations are median [minimum, maximum] tokens/s for 1,023
individual cached forwards, excluding load, tokenization and output rendering:

| Engine / width | Before | After |
|---|---:|---:|
| CPU / 1 | 4568.4 [4267.4,4666.7] | 4339.1 [4208.6,4468.5] |
| CPU / 4 | 4414.7 [4249.0,4598.4] | 2613.6 [2392.9,3102.8] |
| Torch CPU / 1 | 1849.1 [1759.2,2001.2] | 1347.9 [874.2,1415.3] |
| Torch CPU / 4 | 2019.3 [1850.6,2054.3] | 1297.3 [872.2,1384.6] |

Median one-token first-forward milliseconds before -> after are 0.258 -> 0.258
(CPU width 1), 0.253 -> 0.450 (CPU width 4), 0.723 -> 0.738 (Torch width 1),
and 0.512 -> 0.796 (Torch width 4). These are engine-call timings, not public
TTFT. The user's host remained under load. The large slowdown in unchanged
controls prevents attributing these timings to the slice change. These data
establish neither a speedup nor absence of speed regression. The measured
benefits are correct convolution and bounded SSM history. No paired artifact
size benchmark was run for this cache-only change; reports record its absence.

An additional Torch comparison pins the child process to logical CPU 2 and
uses 2,048 inputs, two warmups and seven repeats. It observes the following
median [minimum, maximum] tokens/s for 2,047 cached forwards:

| Width | Before | After |
|---|---:|---:|
| 1 | 415.8 [295.9,742.5] | 589.9 [460.6,673.3] |
| 4 | 364.7 [167.1,586.0] | 711.1 [545.4,763.7] |

The unchanged width-4 control improves more than width 1, so these numbers
also do not isolate a speed benefit. First-forward medians are 1.144 -> 1.693
ms (width 1) and 2.919 -> 0.916 ms (width 4). Width-1 history is 65,536 / 65,536
bytes before versus 0 / 32 after. A preliminary pinned after run returned
429.7 [151.5,637.3] / 370.9 [126.8,570.4] tokens/s at widths 1 / 4; its first
part overlapped source-snapshot copying. The subsequent before/after pair
ran serially without other agent tests/benchmarks; host background load still
varied. The preliminary run remains available and is not used to conceal a loss.

Raw [initial before](measurements/hybrid-convolution-before.json),
[initial after](measurements/hybrid-convolution-after.json),
[pinned before](measurements/hybrid-convolution-pinned-before-1.json),
[pinned after](measurements/hybrid-convolution-pinned-after-2.json),
[preliminary pinned after](measurements/hybrid-convolution-pinned-after-1.json)
and [provenance](measurements/hybrid-convolution-provenance.json) retain every
repeat, first-forward sample, numerical result, source hash and repeat-end
sampled RSS. Repeat-end RSS is not continuously sampled peak memory. Each A/B
pair uses identical measured fixtures/harness; only the runtime module hash
differs in the pinned comparison. Final test-only typing/format/assertion edits
preserve the fixture builder, verified by its AST hash in the provenance.

Reproduce on a host with test extras and Torch installed:

```powershell
python -m pytest tests/unit/test_hybrid_convolution_history.py tests/integration/test_jamba_hybrid_execution.py -o addopts='' -q
python tools/benchmark_mamba_convolution.py --families hybrid --output hybrid.json
python tools/benchmark_mamba_convolution.py --families hybrid --backends torch_cpu --tokens 2048 --cpu-affinity 2 --output hybrid-pinned.json
```

For a parent comparison, run the same harness and fixtures in a separate source
checkout of `fd5aa179`; ensure `PYTHONPATH` selects that checkout. Reports bind
runtime hashes as well as HEAD, so uncommitted fixes are distinguishable.

## Verification and remaining work

The first focused run returned **93 passed** across convolution, hybrid rotary,
typed handoff, speculative execution and real SafeTensors hybrid integration.
The expanded integration and complete-suite results will be recorded after
they finish; this paragraph is not a complete-suite claim.

The touched shared runtime retains 112 Ruff findings and 48 mypy findings
before/after, with identical code/message counters after ignoring line shifts.
The existing integration file retains seven Ruff and three mypy findings. The new test and
benchmark pass full Ruff and mypy; critical syntax/undefined-name checks and
`git diff --check` pass. No existing static-analysis backlog was cleared here.

No optimizer pass changed. The existing
[22-row pass disposition table](mission-default-policy.md#all-22-dispositions-and-remaining-work)
and [per-pass measurements](mission-pass-audit.md) retain their limitations;
only P3 is default. The [hardware evidence matrix](../hardware_validation_matrix.json)
and [generated table](../README.md#supported-hardware-targets) retain each
backend's dates/techniques/tier. These NumPy and Torch CPU convolution checks
ran on physical x86 CPU hardware. CUDA, ROCm, MPS and other accelerators did not
run here and gain no validation tier from this correction.

Original-HF Mamba-1 normalization, projection-bias and expansion contracts remain
under investigation. The [Mamba-2 HF fixtures](mission-mamba2-normalization.md)
retain their existing scope; trained-family admission, RWKV-7/MLA, ONNX parity,
the complete pass-benefit audit, controlled speed/memory work, paged-KV/COW/soak,
archive hardening and blocking core/runtime static checks remain mission work.
